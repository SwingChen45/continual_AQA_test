# coding=utf-8
"""
RG-adapted ASS_JRG.

This version keeps the continual-learning graph-related interfaces used by the
original codebase:
    - save_graph / g_e_graph
    - spatial_mats / temporal_mats
    - general_spatial_mats / general_temporal_mats
    - spatial_JCWs / temporal_JCWs
    - init_e_graph()
    - run_jrg()

But the graph nodes are no longer physical joints from AQA-7 patch features.
Instead, RG's three high-level streams (vst / flow / stgcn) are projected into
latent graph tokens, and AGSG-style general/specific graphs are applied on these
latent tokens.
"""

from builtins import print
import copy
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from einops import rearrange


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
        loss_pearson = torch.tensor(1.0, device=pred.device) - torch.sum((pred - mean_pred) * (labels - mean_labels)) / (
            torch.sqrt(torch.sum((pred - mean_pred) ** 2) * torch.sum((labels - mean_labels) ** 2) + bias_)
        )
        return loss_pearson * 100.0
    elif type == 'mse':
        return torch.mean((pred - labels) ** 2, dim=0)
    elif type == 'huber':
        crit = torch.nn.SmoothL1Loss()
        return crit(pred, labels)
    else:
        return None


def build_joint_graphs(joint_num, hop_num=4):
    """
    Build a latent token graph instead of a physical human-joint graph.

    We deliberately keep the same function name and output format as the
    original project, because the rest of the continual-learning code expects
    joint_graphs with shape [hop_num, joint_num, joint_num].

    Base graph: a local chain + skip connections over latent tokens.
    This preserves a meaningful multi-hop structure without requiring physical
    skeleton joints.
    """
    graph = np.zeros((joint_num, joint_num), dtype=np.float32)

    for i in range(joint_num):
        graph[i, i] = 1.0
        if i - 1 >= 0:
            graph[i, i - 1] = 1.0
        if i + 1 < joint_num:
            graph[i, i + 1] = 1.0
        if i - 2 >= 0:
            graph[i, i - 2] = 1.0
        if i + 2 < joint_num:
            graph[i, i + 2] = 1.0

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
        whole_size=1024,
        patch_size=256,
        seg_num=72,
        joint_num=17,
        out_dim=1,
        mode='',
        save_graph=False,
        feature_id_to_remove=None,
        task_list=None,
        G_E_graph=False,
        alpha=0.5,
        task_num=4,
        token_dim=128,
    ):
        super(ASS_JRG, self).__init__()
        if feature_id_to_remove is None:
            feature_id_to_remove = []
        if task_list is None:
            task_list = []

        self.hop_num = 4
        self.task_num = task_num
        self.whole_size = whole_size   # vst / flow dim
        self.patch_size = patch_size   # stgcn dim
        self.seg_num = seg_num
        self.joint_num = joint_num     # latent token number
        self.mode = mode
        self.task_list = task_list
        self.dropout_rate = 0.1
        self.alpha = alpha
        self.save_graph = save_graph
        self.g_e_graph = G_E_graph
        self.out_dim = out_dim
        self.token_dim = token_dim
        self.module_num = 3 + self.hop_num * 3  # whole, diffwhole, comm0 + 3 groups of hop features

        self.register_buffer("joint_graphs", torch.stack(build_joint_graphs(joint_num), dim=0))

        # General-specific graph parameters kept for AGSG-style continual learning.
        self.register_parameter(
            "spatial_mats",
            nn.Parameter(torch.zeros(self.task_num, self.hop_num, joint_num, joint_num)),
        )
        self.register_parameter(
            "temporal_mats",
            nn.Parameter(torch.zeros(self.task_num, self.hop_num, joint_num, joint_num)),
        )
        self.register_parameter(
            "general_spatial_mats",
            nn.Parameter(torch.randn(self.hop_num, joint_num, joint_num)),
        )
        self.register_parameter(
            "general_temporal_mats",
            nn.Parameter(torch.randn(self.hop_num, joint_num, joint_num)),
        )
        self.register_parameter(
            "spatial_JCWs",
            nn.Parameter(torch.randn(self.hop_num, joint_num, 1)),
        )
        self.register_parameter(
            "temporal_JCWs",
            nn.Parameter(torch.randn(self.hop_num, joint_num, 1)),
        )

        # Per-stream encoders.
        self.encoders_whole = nn.ModuleList([
            self.build_encoder(self.whole_size),   # vst
            self.build_encoder(self.whole_size),   # flow
            self.build_encoder(self.patch_size),   # stgcn
        ])
        self.encoders_diffwhole = nn.ModuleList([
            self.build_encoder(self.whole_size),
            self.build_encoder(self.whole_size),
            self.build_encoder(self.patch_size),
        ])

        # Keep original attribute names so that optimizer code stays almost unchanged.
        self.encoders_comm0 = nn.ModuleList([self.build_encoder(self.token_dim)])
        self.encoders_comm1 = nn.ModuleList([self.build_encoder(self.token_dim) for _ in range(self.hop_num)])
        self.encoders_diff0 = nn.ModuleList([self.build_encoder(self.token_dim) for _ in range(self.hop_num)])
        self.encoders_diff1 = nn.ModuleList([self.build_encoder(self.token_dim) for _ in range(self.hop_num)])

        self.whole_fuse = nn.Sequential(
            nn.Linear(self.token_dim * 3, self.token_dim),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True),
        )
        self.diffwhole_fuse = nn.Sequential(
            nn.Linear(self.token_dim * 3, self.token_dim),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True),
        )

        # Build latent graph tokens from three-stream fused representation.
        self.token_projector = nn.Sequential(
            nn.Linear(self.token_dim * 3, self.joint_num * self.token_dim),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True),
        )
        self.token_norm = nn.LayerNorm(self.token_dim)

        # Final 128 -> 512 feature head. The score regressor remains external.
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.token_dim, 512),
            nn.ReLU(True),
        )
        self.last_fuse = nn.Linear(self.module_num, 1, bias=False)

    def build_encoder(self, input_size):
        return nn.Sequential(
            nn.Linear(input_size, self.token_dim),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True),
        )

    def _temporal_diff(self, x):
        # x: [B, T, D]
        x_next = torch.cat([x[:, 1:], x[:, -1:].clone()], dim=1)
        return torch.abs(x_next - x)

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

    def _graph_propagate(self, token_btjd, graphs):
        # token_btjd: [B, T, J, D], graphs: [A, H, J, J]
        return torch.einsum('ahij,btjd->ahbtid', graphs, token_btjd)

    def _aggregate_pairwise_diff(self, token_bjtd, graphs, jcws, temporal=False):
        # token_bjtd: [B, J, T, D]
        x = rearrange(token_bjtd, 'B J T D -> B T D J')  # [B, T, D, J]
        if temporal:
            x_next = torch.cat([x[:, 1:], x[:, -1:].clone()], dim=1)
            diff = x_next.unsqueeze(-1) - x.unsqueeze(-2)
        else:
            diff = x.unsqueeze(-1) - x.unsqueeze(-2)
        # diff: [B, T, D, J, J]
        A, H, J, _ = graphs.shape
        outputs = []
        for a in range(A):
            task_out = []
            for h in range(H):
                weighted = diff * graphs[a, h].view(1, 1, 1, J, J)
                agg = torch.matmul(weighted, jcws[h]).squeeze(-1)  # [B, T, D, J]
                agg = rearrange(agg, 'B T D J -> B T J D')
                task_out.append(agg)
            outputs.append(torch.stack(task_out, dim=0))  # [H, B, T, J, D]
        return torch.stack(outputs, dim=0)  # [A, H, B, T, J, D]

    def _repeat_whole_to_token_shape(self, feat_btd, num_tasks):
        # [B,T,D] -> [A,B,T,J,D]
        B, T, D = feat_btd.shape
        return feat_btd.unsqueeze(0).unsqueeze(3).expand(num_tasks, B, T, self.joint_num, D)

    def forward(self, feat_vst, feat_flow, feat_stgcn):
        """
        Inputs:
            feat_vst   : [B, T, 1024]
            feat_flow  : [B, T, 1024]
            feat_stgcn : [B, T,  256]
        Returns:
            features   : [B, A, 512] where A=task_num if save_graph else 1
            featmaps   : list for optional POD-style usage
        """
        B, T, _ = feat_vst.shape

        # Stream encodings.
        vst_whole = self.encoders_whole[0](feat_vst)
        flow_whole = self.encoders_whole[1](feat_flow)
        stgcn_whole = self.encoders_whole[2](feat_stgcn)
        whole_cat = torch.cat([vst_whole, flow_whole, stgcn_whole], dim=-1)
        whole_feat = self.whole_fuse(whole_cat)  # [B, T, token_dim]

        vst_diff = self.encoders_diffwhole[0](self._temporal_diff(feat_vst))
        flow_diff = self.encoders_diffwhole[1](self._temporal_diff(feat_flow))
        stgcn_diff = self.encoders_diffwhole[2](self._temporal_diff(feat_stgcn))
        diffwhole_cat = torch.cat([vst_diff, flow_diff, stgcn_diff], dim=-1)
        diffwhole_feat = self.diffwhole_fuse(diffwhole_cat)  # [B, T, token_dim]

        # Build latent graph tokens from the three-stream fused representation.
        token_seed = whole_cat
        latent_tokens = self.token_projector(token_seed).view(B, T, self.joint_num, self.token_dim)
        latent_tokens = self.token_norm(latent_tokens)
        token_bjtd = rearrange(latent_tokens, 'B T J D -> B J T D')
        token_btjd = rearrange(token_bjtd, 'B J T D -> B T J D')

        spatial_graphs, temporal_graphs = self._build_graph_bank()
        num_tasks = spatial_graphs.shape[0]

        # Graph propagation.
        comm_H0 = token_btjd                                          # [B, T, J, D]
        comm_h1s = self._graph_propagate(comm_H0, spatial_graphs)     # [A, H, B, T, J, D]
        diff_d0 = self._aggregate_pairwise_diff(token_bjtd, spatial_graphs, self.spatial_JCWs, temporal=False)
        diff_d1 = self._aggregate_pairwise_diff(token_bjtd, temporal_graphs, self.temporal_JCWs, temporal=True)

        # Encode blocks.
        whole_rep = self._repeat_whole_to_token_shape(whole_feat, num_tasks)
        diffwhole_rep = self._repeat_whole_to_token_shape(diffwhole_feat, num_tasks)

        comm0 = self.encoders_comm0[0](comm_H0).unsqueeze(0).expand(num_tasks, -1, -1, -1, -1)
        comm1 = [self.encoders_comm1[h](comm_h1s[:, h]) for h in range(self.hop_num)]
        diff0 = [self.encoders_diff0[h](diff_d0[:, h]) for h in range(self.hop_num)]
        diff1 = [self.encoders_diff1[h](diff_d1[:, h]) for h in range(self.hop_num)]

        all_feats = [whole_rep, diffwhole_rep, comm0] + comm1 + diff0 + diff1
        out = torch.stack(all_feats, dim=4)  # [A, B, T, J, M, D]

        pooled = out.mean(dim=(2, 3))  # [A, B, M, D]
        fused = self.last_fuse(pooled.permute(0, 1, 3, 2)).squeeze(-1)  # [A, B, D]
        feat_bank = self.regressor(fused.permute(1, 0, 2))  # [B, A, 512]

        # Optional POD-style feature maps.
        featmap_list = [out.permute(1, 2, 3, 4, 5, 0).contiguous()]  # [B, T, J, M, D, A]
        return feat_bank, featmap_list



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


def run_jrg(model_, feat_vst, feat_flow, feat_stgcn, save_graph=False, seen_tasks=None, is_train=False, args=None):
    if seen_tasks is None:
        seen_tasks = []
    fused_feat, featmap_list = model_(feat_vst, feat_flow, feat_stgcn)  # [B, A, 512]
    fused_feat = fused_feat.transpose(0, 1)                              # [A, B, 512]
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
    os_env = {}
    net = ASS_JRG(whole_size=1024, patch_size=256, seg_num=72, joint_num=17, out_dim=1, save_graph=True, G_E_graph=True, alpha=0.5, task_num=4)
    feat_vst = torch.randn(2, 72, 1024)
    feat_flow = torch.randn(2, 72, 1024)
    feat_stgcn = torch.randn(2, 72, 256)
    fused_feat, featmaps = net(feat_vst, feat_flow, feat_stgcn)
    print(fused_feat.shape)
    print(featmaps[0].shape)
