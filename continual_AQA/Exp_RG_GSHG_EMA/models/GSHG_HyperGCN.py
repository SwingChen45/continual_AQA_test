# coding=utf-8
from builtins import print
import copy
import math
import os
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(pred, labels, type='new_mse', action_id=0):
    bias_ = 1e-6
    if type == 'new_mse':
        mean_pred = torch.mean(pred)
        mean_labels = torch.mean(labels)
        normalized_pred = (pred - mean_pred) / torch.sqrt(torch.var(pred) + bias_)
        normalized_labe = (labels - mean_labels) / torch.sqrt(torch.var(labels) + bias_)
        loss_new_mse = torch.mean((normalized_pred - normalized_labe) ** 2, dim=0)
        return loss_new_mse * 100.0
    elif type == 'pearson':
        mean_pred = torch.mean(pred)
        mean_labels = torch.mean(labels)
        loss_pearson = torch.tensor(1.0, device=pred.device) - torch.sum((pred - mean_pred) * (labels - mean_labels)) \
            / torch.sqrt(torch.sum((pred - mean_pred) ** 2) * torch.sum((labels - mean_labels) ** 2) + bias_)
        return loss_pearson * 100.0
    elif type == 'mse':
        return torch.mean((pred - labels) ** 2, dim=0)
    elif type == 'huber':
        crit = torch.nn.SmoothL1Loss()
        return crit(pred, labels)
    else:
        return None


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def _load_rg_adjacency(joint_num):
    a_file = './mat_a.npy'
    if not os.path.isfile(a_file):
        a_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mat_a.npy')
    graph = np.load(a_file).astype(np.float32)
    if graph.shape != (joint_num, joint_num):
        raise ValueError('Adjacency shape {} does not match joint_num={}'.format(graph.shape, joint_num))
    graph = graph + np.eye(joint_num, dtype=np.float32)
    graph[graph > 0] = 1.0
    return graph


def build_virtual_adjacency(joint_num, virtual_num, num_subset):
    graph = _load_rg_adjacency(joint_num)
    total_vertex = joint_num + virtual_num
    A = np.zeros((total_vertex, total_vertex), dtype=np.float32)
    A[:joint_num, :joint_num] = graph
    for i in range(virtual_num):
        v = joint_num + i
        A[v, v] = 1.0
        A[:joint_num, v] = 1.0
        A[v, :joint_num] = 1.0
    return np.repeat(A[np.newaxis, :, :], num_subset, axis=0)


def _row_normalize(mat):
    denom = mat.sum(axis=-1, keepdims=True) + 1e-8
    return mat / denom


def build_quality_base_graph(joint_num, virtual_num, num_subset):
    """Human-quality prior hyperedges for RG/OpenPose-18 joints."""
    total_vertex = joint_num + virtual_num

    # OpenPose-18 style layout: 0 nose, 1 neck, 2/5 shoulders, 3/6 elbows,
    # 4/7 wrists, 8/11 hips, 9/12 knees, 10/13 ankles, 14-17 face points.
    base_edges = [
        [0, 1, 2, 5, 8, 11],              # trunk stability
        [1, 5, 6, 7],                     # left upper-limb control
        [1, 2, 3, 4],                     # right upper-limb control
        [1, 11, 12, 13],                  # left lower-limb support
        [1, 8, 9, 10],                    # right lower-limb support
        [2, 5, 3, 6, 4, 7, 8, 11, 9, 12, 10, 13],  # left-right symmetry
        [2, 5, 8, 11, 9, 12, 10, 13],    # upper-lower coordination
        list(range(min(joint_num, 18))),  # whole-body coordination
    ]

    if virtual_num > 0:
        virtual_ids = list(range(joint_num, total_vertex))
        base_edges[0] = base_edges[0] + virtual_ids[:1]
        base_edges[-1] = base_edges[-1] + virtual_ids

    incidence = np.zeros((total_vertex, len(base_edges)), dtype=np.float32)
    for edge_id, edge in enumerate(base_edges):
        for joint_id in edge:
            if 0 <= joint_id < total_vertex:
                incidence[joint_id, edge_id] = 1.0

    edge_degree = incidence.sum(axis=0, keepdims=True) + 1e-8
    graph = incidence @ (incidence.T / edge_degree.T)
    graph = _row_normalize(graph + np.eye(total_vertex, dtype=np.float32))
    return np.repeat(graph[np.newaxis, :, :], num_subset, axis=0).astype(np.float32)


class GSHGHyperGC(nn.Module):
    """Hyper-GCN block with base + EMA-domain quality hypergraph modulation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        vertex_nums,
        virtual_num,
        task_num,
        A,
        num_subset=8,
        rel_reduction=4,
        gshg_beta=0.8,
        ema_rho=0.9,
        quality_dyn_weight=0.1,
        topk=9,
    ):
        super(GSHGHyperGC, self).__init__()
        if in_channels % num_subset != 0 or out_channels % num_subset != 0:
            raise ValueError('in/out channels must be divisible by num_subset')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vertex_nums = vertex_nums
        self.virtual_num = virtual_num
        self.total_vertex = vertex_nums + virtual_num
        self.task_num = task_num
        self.num_subset = num_subset
        self.hidden_channels = max(1, (in_channels // num_subset) // rel_reduction)
        self.mid_out_channels = out_channels // num_subset
        self.gshg_beta = gshg_beta
        self.ema_rho = ema_rho
        self.quality_dyn_weight = quality_dyn_weight
        self.topk = min(topk, self.total_vertex)

        self.to_V = nn.Conv1d(in_channels, num_subset * self.hidden_channels, kernel_size=1, groups=num_subset)
        self.to_W = nn.Sequential(
            nn.Conv1d(in_channels, num_subset * self.hidden_channels, kernel_size=1, groups=num_subset),
            nn.LeakyReLU(),
            nn.Conv1d(num_subset * self.hidden_channels, num_subset, kernel_size=1),
            nn.Tanh()
        )
        self.hyper_joint = nn.Parameter(torch.randn(virtual_num, in_channels))
        self.hyper_edge_weight_gen = nn.Parameter(torch.zeros(num_subset, self.total_vertex, self.total_vertex))
        self.hyper_edge_weight_spec = nn.Parameter(torch.zeros(task_num, num_subset, self.total_vertex, self.total_vertex))
        self.hyper_alpha = nn.Parameter(torch.ones(1))

        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=num_subset)
        self.register_buffer('PA', torch.from_numpy(A.astype(np.float32)))
        self.register_buffer(
            'quality_base_graph',
            torch.from_numpy(build_quality_base_graph(vertex_nums, virtual_num, num_subset))
        )
        self.register_buffer('quality_domain_graph', torch.zeros(num_subset, self.total_vertex, self.total_vertex))
        self.register_buffer('domain_initialized', torch.zeros(1))
        self.edge_importance = nn.Parameter(torch.ones(A.shape))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.GroupNorm(self._num_groups(out_channels), out_channels)
            )
        else:
            self.down = lambda x: x
        self.norm = nn.GroupNorm(self._num_groups(out_channels), out_channels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
    def _num_groups(self, channels):
        for groups in [32, 16, 8, 4, 2, 1]:
            if channels % groups == 0:
                return groups
        return 1

    def hyper_norm(self, H, W):
        w = torch.diag_embed(W)
        norm_w = torch.norm(H, 1, dim=-2, keepdim=True) + 1e-8
        w_ = w / norm_w
        H_w = H @ w
        norm_v = torch.norm(H_w, 1, dim=-1, keepdim=True) + 1e-8
        h_ = H_w / norm_v
        return h_ @ w_ @ H.transpose(-1, -2)

    def a_norm(self, A):
        d_r = torch.norm(A, 1, dim=-2, keepdim=True) + 1e-8
        return A / d_r

    def _weighted_quality_graph(self, quality_graph, weight):
        if quality_graph.dim() == 3:
            quality_graph = quality_graph.unsqueeze(0)
        weighted = quality_graph * F.softplus(weight).unsqueeze(0)
        return self.a_norm(weighted)

    def _append_virtual_joints(self, x):
        N, C, T, _ = x.size()
        h_x = self.hyper_joint.t().unsqueeze(0).unsqueeze(2)
        h_x = h_x.repeat(N, 1, T, 1)
        return torch.cat([x, h_x], dim=-1)

    def _dynamic_hypergraph(self, x):
        t_x = x.mean(2)
        v_x = self.to_V(t_x)
        dis_v_x = v_x.view(x.shape[0], self.num_subset, self.hidden_channels, self.total_vertex)
        dis_v_x = dis_v_x.permute(0, 1, 3, 2).contiguous()
        distance_x = torch.cdist(dis_v_x, dis_v_x)
        H = torch.zeros_like(distance_x)
        topk_v, topk_indices = torch.topk(distance_x, self.topk, largest=False)
        topk_v = self.softmax(-topk_v)
        H = torch.scatter(H, 3, topk_indices, topk_v)
        W_dyn = self.to_W(t_x)
        return H, W_dyn

    def _dynamic_quality_graph(self, x):
        H_dyn, W_dyn = self._dynamic_hypergraph(x)
        return self.a_norm(self.hyper_norm(H_dyn, F.softplus(W_dyn)))

    @torch.no_grad()
    def update_domain_graph(self, feat_joint, rho=None):
        x = feat_joint.permute(0, 3, 1, 2).contiguous()
        x = self._append_virtual_joints(x)
        domain_batch = self._dynamic_quality_graph(x).mean(dim=0)
        if self.domain_initialized.item() == 0:
            self.quality_domain_graph.copy_(domain_batch)
            self.domain_initialized.fill_(1.0)
        else:
            momentum = self.ema_rho if rho is None else rho
            self.quality_domain_graph.mul_(momentum).add_(domain_batch, alpha=1.0 - momentum)

    def forward(self, x):
        # x: [B, C, T, V]
        x_real = x
        x = self._append_virtual_joints(x)

        A = self.a_norm(self.edge_importance * self.PA)
        G_dyn = self._dynamic_quality_graph(x)
        quality_graph = self.a_norm(self.quality_base_graph + self.quality_domain_graph)
        if self.quality_dyn_weight > 0:
            quality_graph = self.a_norm(
                quality_graph.unsqueeze(0) + self.quality_dyn_weight * G_dyn
            )

        task_outputs = []
        d_x = self.conv_d(x)
        d_x = d_x.view(x.shape[0], self.num_subset, self.mid_out_channels, x.shape[2], self.total_vertex)
        alpha = self.relu(self.hyper_alpha)
        G_gen = self._weighted_quality_graph(quality_graph, self.hyper_edge_weight_gen)

        for task_id in range(self.task_num):
            G_spec = self._weighted_quality_graph(quality_graph, self.hyper_edge_weight_spec[task_id])
            G_task = self.gshg_beta * G_gen + (1.0 - self.gshg_beta) * G_spec
            A_task = A.unsqueeze(0) + alpha * G_task
            y = torch.einsum('nkuv,nkctv->nkctu', A_task, d_x).contiguous()
            y = y.view(x.shape[0], self.out_channels, x.shape[2], self.total_vertex)
            y = y[..., :self.vertex_nums]
            y = self.norm(y)
            y = y + self.down(x_real)
            y = self.relu(y)
            task_outputs.append(y)

        return torch.stack(task_outputs, dim=1)


class ASS_GSHG(nn.Module):
    def __init__(
        self,
        patch_size=256,
        seg_num=72,
        joint_num=18,
        out_dim=1,
        mode='',
        save_graph=False,
        feature_id_to_remove=None,
        task_list=None,
        G_E_graph=False,
        alpha=0.5,
        task_num=4,
        hyper_joints=3,
        num_subset=8,
        rel_reduction=4,
        gshg_beta=0.8,
        ema_rho=0.9,
        quality_dyn_weight=0.1,
    ):
        super(ASS_GSHG, self).__init__()
        self.patch_size = patch_size
        self.seg_num = seg_num
        self.joint_num = joint_num
        self.task_num = task_num
        self.save_graph = save_graph
        self.alpha = alpha
        self.g_e_graph = G_E_graph
        self.hidden_channels = 256

        A = build_virtual_adjacency(joint_num, hyper_joints, num_subset)
        self.gshg = GSHGHyperGC(
            in_channels=patch_size,
            out_channels=self.hidden_channels,
            vertex_nums=joint_num,
            virtual_num=hyper_joints,
            task_num=task_num,
            A=A,
            num_subset=num_subset,
            rel_reduction=rel_reduction,
            gshg_beta=gshg_beta,
            ema_rho=ema_rho,
            quality_dyn_weight=quality_dyn_weight,
        )
        self.temporal = nn.Sequential(
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.GroupNorm(self._num_groups(self.hidden_channels), self.hidden_channels),
            nn.ReLU(True),
        )
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden_channels, 512),
            nn.ReLU(True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def _num_groups(self, channels):
        for groups in [32, 16, 8, 4, 2, 1]:
            if channels % groups == 0:
                return groups
        return 1

    def forward(self, feat_joint):
        """
        feat_joint: [B, T, 18, 256]
        returns: [B, task_num, 512]
        """
        B, T, J, D = feat_joint.shape
        if J != self.joint_num:
            raise ValueError('Expected {} joints, got {}'.format(self.joint_num, J))
        if D != self.patch_size:
            raise ValueError('Expected joint feature dim {}, got {}'.format(self.patch_size, D))

        x = feat_joint.permute(0, 3, 1, 2).contiguous()
        task_maps = self.gshg(x)

        task_feats = []
        featmaps = []
        for task_id in range(self.task_num):
            y = self.temporal(task_maps[:, task_id])
            pooled = y.mean(dim=(2, 3))
            task_feats.append(self.regressor(pooled))
            featmaps.append(y.permute(0, 2, 3, 1).unsqueeze(3))

        fused_feat = torch.stack(task_feats, dim=1)
        featmap = torch.stack(featmaps, dim=-1)
        return fused_feat, [featmap]


def get_numpy_mse(pred, score):
    pred = np.array(pred)
    score = np.array(score)
    return np.sum((pred - score) ** 2) / pred.shape[0]


def get_numpy_spearman(pred, score):
    pred = np.array(pred)
    score = np.array(score)
    return stats.spearmanr(pred, score).correlation


def get_numpy_pearson(pred, score):
    pred = np.array(pred)
    score = np.array(score)
    return stats.pearsonr(pred, score)[0]


def run_jrg(model_, feat_joint, save_graph=False, seen_tasks=None, is_train=False, args=None):
    fused_feat, featmap_list = model_(feat_joint)
    fused_feat = fused_feat.transpose(0, 1)
    if not save_graph:
        fused_feat = fused_feat[0]
    return fused_feat, featmap_list


@torch.no_grad()
def update_domain_graph(model_, dataloader, device='cuda', rho=None, max_batches=None):
    net = model_.module if hasattr(model_, 'module') else model_
    was_training = net.training
    net.eval()
    for batch_idx, batch in enumerate(dataloader):
        feat_joint = batch[0].to(device)
        net.gshg.update_domain_graph(feat_joint, rho=rho)
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    if was_training:
        net.train()


def init_e_graph(model_, t, seen_tasks=None):
    if seen_tasks is None:
        seen_tasks = []
    net = model_.module if hasattr(model_, 'module') else model_
    if t == 0:
        for idx in seen_tasks[:1]:
            net.gshg.hyper_edge_weight_spec.data[idx].copy_(net.gshg.hyper_edge_weight_gen.data)
    else:
        net.gshg.hyper_edge_weight_spec.data[seen_tasks[-1]].copy_(
            copy.deepcopy(net.gshg.hyper_edge_weight_spec.data[seen_tasks[-2]])
        )
    return


if __name__ == '__main__':
    net = ASS_GSHG(patch_size=256, seg_num=72, joint_num=18, task_num=4, save_graph=True)
    feat_joint = torch.randn(2, 72, 18, 256)
    fused_feat, featmaps = net(feat_joint)
    print(fused_feat.shape)
    print(featmaps[0].shape)
