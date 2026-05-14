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


class ASS_JRG(nn.Module):
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
    ):
        super(ASS_JRG, self).__init__()
        self.hop_num = 4
        self.task_num = task_num
        self.patch_size = patch_size
        self.seg_num = seg_num
        self.joint_num = joint_num
        self.module_num = 3 + self.hop_num * 3
        self.mode = mode
        self.task_list = task_list or []
        self.dropout_rate = 0.1
        self.hidden1 = 256
        self.alpha = alpha
        self.save_graph = save_graph
        self.out_dim = out_dim
        self.g_e_graph = G_E_graph

        self.register_buffer("joint_graphs", torch.stack(build_joint_graphs(joint_num), dim=0))

        self.register_parameter("spatial_mats", nn.Parameter(torch.zeros(self.task_num, self.hop_num, joint_num, joint_num)))
        self.register_parameter("temporal_mats", nn.Parameter(torch.zeros(self.task_num, self.hop_num, joint_num, joint_num)))
        self.register_parameter("general_spatial_mats", nn.Parameter(torch.randn(self.hop_num, joint_num, joint_num)))
        self.register_parameter("general_temporal_mats", nn.Parameter(torch.randn(self.hop_num, joint_num, joint_num)))
        self.register_parameter("spatial_JCWs", nn.Parameter(torch.randn(self.hop_num, joint_num, 1)))
        self.register_parameter("temporal_JCWs", nn.Parameter(torch.randn(self.hop_num, joint_num, 1)))

        self.encoders_whole = nn.ModuleList([self.build_encoder(self.patch_size)])
        self.encoders_diffwhole = nn.ModuleList([self.build_encoder(self.patch_size)])
        self.whole_fuse = nn.Sequential(
            nn.Linear(self.hidden1 // 2, self.hidden1 // 2),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True),
        )
        self.diffwhole_fuse = nn.Sequential(
            nn.Linear(self.hidden1 // 2, self.hidden1 // 2),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True),
        )

        self.encoders_comm0 = nn.ModuleList([self.build_encoder(self.patch_size)])
        self.encoders_comm1 = nn.ModuleList([self.build_encoder(self.patch_size) for _ in range(self.hop_num)])
        self.encoders_diff0 = nn.ModuleList([self.build_encoder(self.patch_size) for _ in range(self.hop_num)])
        self.encoders_diff1 = nn.ModuleList([self.build_encoder(self.patch_size) for _ in range(self.hop_num)])

        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.hidden1 // 2, 512),
            nn.ReLU(True)
        )
        self.last_fuse = nn.Linear(self.module_num, 1, bias=False)

    def build_encoder(self, input_size):
        return nn.Sequential(
            nn.Linear(input_size, self.hidden1 // 2),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True)
        )

    def _build_graph_bank(self):
        if self.save_graph:
            spatial_graphs = torch.abs(self.spatial_mats * self.joint_graphs)
            temporal_graphs = torch.abs(self.temporal_mats * self.joint_graphs)
            if self.g_e_graph:
                general_spatial_graphs = torch.abs(self.general_spatial_mats * self.joint_graphs)
                general_temporal_graphs = torch.abs(self.general_temporal_mats * self.joint_graphs)
                spatial_graphs = (1 - self.alpha) * spatial_graphs + self.alpha * general_spatial_graphs.unsqueeze(0)
                temporal_graphs = (1 - self.alpha) * temporal_graphs + self.alpha * general_temporal_graphs.unsqueeze(0)
        else:
            general_spatial_graphs = torch.abs(self.general_spatial_mats * self.joint_graphs)
            general_temporal_graphs = torch.abs(self.general_temporal_mats * self.joint_graphs)
            spatial_graphs = general_spatial_graphs.unsqueeze(0)
            temporal_graphs = general_temporal_graphs.unsqueeze(0)
        return spatial_graphs, temporal_graphs

    def _temporal_diff(self, x):
        x_next = torch.cat([x[:, 1:], x[:, -1:].clone()], dim=1)
        return torch.abs(x_next - x)

    def _repeat_whole(self, x, num_tasks):
        return x.unsqueeze(0).unsqueeze(3).expand(num_tasks, x.shape[0], x.shape[1], self.joint_num, x.shape[2])

    def _aggregate_pairwise_diff(self, feat_patch, graphs, jcws, temporal=False):
        # feat_patch: [B, J, T, D], graphs: [A, H, J, J]
        x = feat_patch.permute(0, 2, 3, 1).contiguous()  # [B, T, D, J]
        if temporal:
            x_next = torch.cat([x[:, 1:], x[:, -1:].clone()], dim=1)
            diff = x_next.unsqueeze(-1) - x.unsqueeze(-2)
        else:
            diff = x.unsqueeze(-1) - x.unsqueeze(-2)

        outs = []
        for a in range(graphs.shape[0]):
            hop_outs = []
            for h in range(graphs.shape[1]):
                weighted = diff * graphs[a, h].view(1, 1, 1, self.joint_num, self.joint_num)
                agg = torch.matmul(weighted, jcws[h]).squeeze(-1)  # [B, T, D, J]
                hop_outs.append(agg.permute(0, 1, 3, 2).contiguous())
            outs.append(torch.stack(hop_outs, dim=0))
        return torch.stack(outs, dim=0)  # [A, H, B, T, J, D]

    def forward(self, feat_joint):
        """
        feat_joint: [B, T, 18, 256]
        """
        B, T, J, D = feat_joint.shape
        if J != self.joint_num:
            raise ValueError('Expected {} joints, got {}'.format(self.joint_num, J))
        if D != self.patch_size:
            raise ValueError('Expected joint feature dim {}, got {}'.format(self.patch_size, D))

        spatial_graphs, temporal_graphs = self._build_graph_bank()
        num_tasks = spatial_graphs.shape[0]

        feat_patch = feat_joint.permute(0, 2, 1, 3).contiguous()  # [B,J,T,D]
        comm_H0 = feat_patch.permute(0, 2, 1, 3).contiguous()     # [B,T,J,D]
        comm_h1s = torch.einsum('ahij,btjd->ahbtid', spatial_graphs, comm_H0)
        diff_d0 = self._aggregate_pairwise_diff(feat_patch, spatial_graphs, self.spatial_JCWs, temporal=False)
        diff_d1 = self._aggregate_pairwise_diff(feat_patch, temporal_graphs, self.temporal_JCWs, temporal=True)

        joint_global = feat_joint.mean(dim=2)  # [B,T,D]
        whole_feat = self.whole_fuse(self.encoders_whole[0](joint_global))

        joint_global_diff = self._temporal_diff(joint_global)
        diffwhole_feat = self.diffwhole_fuse(self.encoders_diffwhole[0](joint_global_diff))

        whole_rep = self._repeat_whole(whole_feat, num_tasks)
        diffwhole_rep = self._repeat_whole(diffwhole_feat, num_tasks)
        comm0 = self.encoders_comm0[0](comm_H0).unsqueeze(0).expand(num_tasks, -1, -1, -1, -1)
        comm1 = [self.encoders_comm1[h](comm_h1s[:, h]) for h in range(self.hop_num)]
        diff0 = [self.encoders_diff0[h](diff_d0[:, h]) for h in range(self.hop_num)]
        diff1 = [self.encoders_diff1[h](diff_d1[:, h]) for h in range(self.hop_num)]

        all_feats = [whole_rep, diffwhole_rep, comm0] + comm1 + diff0 + diff1
        out = torch.stack(all_feats, dim=4)  # [A,B,T,J,M,D]
        pooled = out.mean(dim=(2, 3))        # [A,B,M,D]
        fused = self.last_fuse(pooled.permute(0, 1, 3, 2)).squeeze(-1)
        fused_feat = self.regressor(fused.permute(1, 0, 2))  # [B,A,512]
        featmap_list = [out.permute(1, 2, 3, 4, 5, 0).contiguous()]
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
    net = ASS_JRG(patch_size=3, seg_num=72, joint_num=18, save_graph=True, G_E_graph=True)
    feat_joint = torch.randn(2, 72, 18, 3)
    fused_feat, featmaps = net(feat_joint)
    print(fused_feat.shape)
    print(featmaps[0].shape)
