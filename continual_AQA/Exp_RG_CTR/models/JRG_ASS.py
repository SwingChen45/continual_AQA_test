# coding=utf-8
from builtins import print
import copy
import os
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn


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


def build_joint_graphs(joint_num, hop_num=4):
    a_file = './mat_a.npy'
    if not os.path.isfile(a_file):
        a_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mat_a.npy')
    graph = np.load(a_file).astype(float)
    if graph.shape != (joint_num, joint_num):
        raise ValueError('Adjacency shape {} does not match joint_num={}'.format(graph.shape, joint_num))
    graph = graph + np.identity(joint_num)
    graph = torch.from_numpy(graph).float()
    graphs = [graph]
    for _ in range(hop_num - 1):
        tmp = torch.matmul(graphs[-1], graph)
        tmp[tmp != 0] = 1
        graphs.append(tmp)
    for i in range(hop_num - 1, 0, -1):
        graphs[i] = graphs[i] * (1 - graphs[i - 1])
    return graphs


def conv_init(module):
    nn.init.kaiming_normal_(module.weight, mode='fan_out')
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)


def bn_init(module, scale=1):
    nn.init.constant_(module.weight, scale)
    nn.init.constant_(module.bias, 0)


class CTRGC(nn.Module):
    """CTR-GCN graph convolution with AGSG-provided task topology."""

    def __init__(self, in_channels, out_channels, hop_num=4, rel_reduction=8):
        super(CTRGC, self).__init__()
        rel_channels = max(8, in_channels // rel_reduction)
        self.hop_num = hop_num
        self.out_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(rel_channels, hop_num * out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        self.bn = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = lambda x: x
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x, graphs, gamma=0.1, dynamic=True):
        # x: [B,C,T,V], graphs: [A,H,V,V]
        B = x.shape[0]
        task_num, hop_num, V, _ = graphs.shape
        x1 = self.conv1(x).mean(dim=2)
        x2 = self.conv2(x).mean(dim=2)
        x3 = self.conv3(x)

        if dynamic:
            q = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
            q = self.conv4(q).view(B, hop_num, self.out_channels, V, V)
        else:
            q = x.new_zeros(B, hop_num, self.out_channels, V, V)

        outs = []
        for task_id in range(task_num):
            y = 0
            for hop_id in range(hop_num):
                r = graphs[task_id, hop_id].view(1, 1, V, V) + gamma * q[:, hop_id]
                y = y + torch.einsum('bcuv,bctv->bctu', r, x3)
            y = self.relu(self.bn(y) + self.down(x))
            outs.append(y)
        return torch.stack(outs, dim=1)  # [B,A,C,T,V]


class TaskWiseCTRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hop_num=4, rel_reduction=8, temporal_kernel=9):
        super(TaskWiseCTRBlock, self).__init__()
        self.gcn = CTRGC(in_channels, out_channels, hop_num=hop_num, rel_reduction=rel_reduction)
        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        for m in self.tcn.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, graphs, gamma=0.1, dynamic=True):
        y = self.gcn(x, graphs, gamma=gamma, dynamic=dynamic)
        outs = []
        for task_id in range(y.shape[1]):
            outs.append(self.relu(self.tcn(y[:, task_id]) + y[:, task_id]))
        return torch.stack(outs, dim=1)  # [B,A,C,T,V]


class ASS_JRG(nn.Module):
    """CTR-GCN-style extractor with AGSG general/specific topology.

    The class name is kept for compatibility with the original Continual-AQA
    training code. Internally this is no longer the JRG comm/diff framework.
    """

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
        ctr_dynamic=True,
        ctr_gamma=0.1,
        ctr_reduction=8,
    ):
        super(ASS_JRG, self).__init__()
        self.hop_num = 4
        self.task_num = task_num
        self.patch_size = patch_size
        self.seg_num = seg_num
        self.joint_num = joint_num
        self.mode = mode
        self.task_list = task_list or []
        self.alpha = alpha
        self.save_graph = save_graph
        self.out_dim = out_dim
        self.g_e_graph = G_E_graph
        self.ctr_dynamic = ctr_dynamic
        self.ctr_gamma = ctr_gamma

        self.register_buffer("joint_graphs", torch.stack(build_joint_graphs(joint_num), dim=0))

        self.register_parameter("spatial_mats", nn.Parameter(torch.zeros(self.task_num, self.hop_num, joint_num, joint_num)))
        self.register_parameter("temporal_mats", nn.Parameter(torch.zeros(self.task_num, self.hop_num, joint_num, joint_num)))
        self.register_parameter("general_spatial_mats", nn.Parameter(torch.randn(self.hop_num, joint_num, joint_num)))
        self.register_parameter("general_temporal_mats", nn.Parameter(torch.randn(self.hop_num, joint_num, joint_num)))

        self.data_bn = nn.BatchNorm1d(patch_size * joint_num)
        self.block1 = TaskWiseCTRBlock(patch_size, 256, hop_num=self.hop_num, rel_reduction=ctr_reduction)
        self.block2 = TaskWiseCTRBlock(256, 256, hop_num=self.hop_num, rel_reduction=ctr_reduction)
        self.dropout = nn.Dropout(0.5)
        self.proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
        )
        bn_init(self.data_bn, 1)

    def _build_graph_bank(self):
        if self.save_graph:
            spatial_graphs = torch.abs(self.spatial_mats * self.joint_graphs)
            if self.g_e_graph:
                general_spatial_graphs = torch.abs(self.general_spatial_mats * self.joint_graphs)
                spatial_graphs = (1 - self.alpha) * spatial_graphs + self.alpha * general_spatial_graphs.unsqueeze(0)
        else:
            spatial_graphs = torch.abs(self.general_spatial_mats * self.joint_graphs).unsqueeze(0)
        return spatial_graphs

    def forward(self, feat_joint):
        # feat_joint: [B,T,18,256]
        B, T, J, D = feat_joint.shape
        if J != self.joint_num:
            raise ValueError('Expected {} joints, got {}'.format(self.joint_num, J))
        if D != self.patch_size:
            raise ValueError('Expected joint feature dim {}, got {}'.format(self.patch_size, D))

        graphs = self._build_graph_bank()
        x = feat_joint.permute(0, 3, 1, 2).contiguous()  # [B,D,T,V]
        x = x.permute(0, 3, 1, 2).contiguous().view(B, J * D, T)
        x = self.data_bn(x)
        x = x.view(B, J, D, T).permute(0, 2, 3, 1).contiguous()

        y = self.block1(x, graphs, gamma=self.ctr_gamma, dynamic=self.ctr_dynamic)
        task_maps = []
        for task_id in range(y.shape[1]):
            task_maps.append(self.block2(y[:, task_id], graphs, gamma=self.ctr_gamma, dynamic=self.ctr_dynamic)[:, task_id])
        y = torch.stack(task_maps, dim=1)  # [B,A,256,T,V]

        pooled = y.mean(dim=(3, 4))
        fused_feat = self.proj(self.dropout(pooled))  # [B,A,512]
        featmap_list = [y.permute(0, 3, 4, 2, 1).unsqueeze(3).contiguous()]  # [B,T,V,1,C,A]
        return fused_feat, featmap_list


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


def init_e_graph(model_, t, seen_tasks=None):
    if seen_tasks is None:
        seen_tasks = []
    if t == 0:
        alpha = model_.module.alpha
        g_spatial_mat = copy.deepcopy(model_.module.general_spatial_mats.data)
        g_temporal_mat = copy.deepcopy(model_.module.general_temporal_mats.data)
        model_.module.spatial_mats.data[seen_tasks[0]] = (1 - alpha) * g_spatial_mat
        model_.module.temporal_mats.data[seen_tasks[0]] = (1 - alpha) * g_temporal_mat
    else:
        model_.module.spatial_mats.data[seen_tasks[-1]] += copy.deepcopy(model_.module.spatial_mats.data[seen_tasks[-2]])
        model_.module.temporal_mats.data[seen_tasks[-1]] += copy.deepcopy(model_.module.temporal_mats.data[seen_tasks[-2]])
    return


if __name__ == '__main__':
    net = ASS_JRG(patch_size=256, seg_num=72, joint_num=18, save_graph=True, G_E_graph=True)
    feat_joint = torch.randn(2, 72, 18, 256)
    fused_feat, featmaps = net(feat_joint)
    print(fused_feat.shape)
    print(featmaps[0].shape)
