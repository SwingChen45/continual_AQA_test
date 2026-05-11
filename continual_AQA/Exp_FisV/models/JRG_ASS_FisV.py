# coding=utf-8
from builtins import print
import os
import copy
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from einops import rearrange, repeat, reduce


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
        loss_pearson = torch.tensor(1.0).cuda() - torch.sum((pred - mean_pred) * (labels - mean_labels)) / (
            torch.sqrt(torch.sum((pred - mean_pred) ** 2) * torch.sum((labels - mean_labels) ** 2) + bias_)
        )
        return loss_pearson * 100.0
    elif type == 'mse':
        loss_mse = torch.mean((pred - labels) ** 2, dim=0)
        return loss_mse
    elif type == 'huber':
        crit = torch.nn.SmoothL1Loss()
        return crit(pred, labels)
    else:
        return None


def build_joint_graphs(joint_num, hop_num=4):
    a_file = './mat_a.npy'
    if not os.path.isfile(a_file):
        a_file = '/home/administrator/exp--fs-aug/Continual-AQA-main/Exp_AQA7/mat_a.npy'

    if joint_num == 17:
        graph = np.load(a_file).astype(float) + np.identity(joint_num)
    else:
        # fallback small graph, but for our current Fis-V adaptation we still recommend joint_num=17
        graph = np.array([
            [1, 1, 1, 0],
            [1, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])

    graph = torch.from_numpy(graph).int()
    graphs = [graph]
    for _ in range(hop_num - 1):
        tmp = torch.matmul(graphs[-1], graph)
        tmp[tmp != 0] = 1
        graphs.append(tmp)
    for i in range(hop_num - 1, 0, -1):
        graphs[i] = graphs[i] * (1 - graphs[i - 1])
    return graphs


class ASS_JRG(nn.Module):
    """
    Two-path model:
    1) AQA-7 path (original): forward(feat_whole, feat_patch)
    2) Fis-V path (new):      forward(rgb_feat, flow_feat, skel_feat)

    For Fis-V:
    - RGB:  [B, T, 1024]
    - Flow: [B, T, 1024]
    - Skel: [B, T, 256]
    -> project each modality
    -> fuse sequence
    -> generate 17 latent graph tokens [B, J, T, C]
    -> reuse JRG graph reasoning trunk
    -> output 512-d feature so MLP heads stay unchanged
    """
    def __init__(
        self,
        whole_size=400,
        patch_size=400,
        seg_num=12,
        joint_num=17,
        out_dim=1,
        mode='',
        save_graph=False,
        feature_id_to_remove=None,
        task_list=None,
        G_E_graph=False,
        alpha=0.5,
        task_num=6,
        fisv_rgb_dim=1024,
        fisv_flow_dim=1024,
        fisv_skel_dim=256,
        fisv_hidden_dim=128,
    ):
        super(ASS_JRG, self).__init__()

        if feature_id_to_remove is None:
            feature_id_to_remove = []
        if task_list is None:
            task_list = []

        self.hop_num = 4
        self.task_num = task_num

        self.whole_size = whole_size
        self.patch_size = patch_size
        self.seg_num = seg_num
        self.joint_num = joint_num
        self.module_num = 12 - len(feature_id_to_remove)
        self.mode = mode
        self.task_list = task_list
        self.dropout_rate = 0.1
        self.hidden1 = 256
        self.hidden2 = 256
        self.hidden3 = 32
        self.alpha = alpha
        self.save_graph = save_graph
        self.fix_graph = True
        self.out_dim = out_dim

        # -------- Graph parameters (shared by both AQA-7 and Fis-V) --------
        self.register_buffer("joint_graphs", torch.stack(build_joint_graphs(joint_num), dim=0))

        self.register_parameter(
            "spatial_mats",
            nn.Parameter(torch.zeros(self.task_num, self.hop_num, joint_num, joint_num))
        )
        self.register_parameter(
            "temporal_mats",
            nn.Parameter(torch.zeros(self.task_num, self.hop_num, joint_num, joint_num))
        )

        self.g_e_graph = G_E_graph

        self.register_parameter(
            "general_spatial_mats",
            nn.Parameter(torch.randn(self.hop_num, joint_num, joint_num))
        )
        self.register_parameter(
            "general_temporal_mats",
            nn.Parameter(torch.randn(self.hop_num, joint_num, joint_num))
        )
        self.register_parameter(
            "spatial_JCWs",
            nn.Parameter(torch.randn(self.hop_num, joint_num, 1))
        )
        self.register_parameter(
            "temporal_JCWs",
            nn.Parameter(torch.randn(self.hop_num, joint_num, 1))
        )

        # -------- Original AQA-7 encoders --------
        self.encoders_whole = nn.ModuleList([self.build_encoder(self.whole_size) for _ in range(2)])
        self.encoders_diffwhole = nn.ModuleList([self.build_encoder(self.whole_size) for _ in range(2)])
        self.encoders_comm0 = nn.ModuleList([self.build_encoder(self.patch_size) for _ in range(2)])
        self.encoders_comm1 = nn.ModuleList([self.build_encoder(self.patch_size) for _ in range(self.hop_num * 2)])
        self.encoders_diff0 = nn.ModuleList([self.build_encoder(self.patch_size) for _ in range(self.hop_num * 2)])
        self.encoders_diff1 = nn.ModuleList([self.build_encoder(self.patch_size) for _ in range(self.hop_num * 2)])

        self.regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(True)
        )

        # -------- New Fis-V modality projections --------
        self.fisv_hidden_dim = fisv_hidden_dim

        self.rgb_proj = self.build_modality_proj(fisv_rgb_dim, fisv_hidden_dim)
        self.flow_proj = self.build_modality_proj(fisv_flow_dim, fisv_hidden_dim)
        self.skel_proj = self.build_modality_proj(fisv_skel_dim, fisv_hidden_dim)

        self.fisv_fuse = nn.Sequential(
            nn.Linear(fisv_hidden_dim * 3, fisv_hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True)
        )

        # generate latent graph tokens: [B, T, C] -> [B, T, J*C]
        self.token_generator = nn.Sequential(
            nn.Linear(fisv_hidden_dim, joint_num * fisv_hidden_dim),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True)
        )

        # Fis-V branch encoders after token construction
        self.fisv_whole_encoder = self.build_feature_encoder(fisv_hidden_dim, fisv_hidden_dim)
        self.fisv_diffwhole_encoder = self.build_feature_encoder(fisv_hidden_dim, fisv_hidden_dim)
        self.fisv_patch_encoder = self.build_feature_encoder(fisv_hidden_dim, fisv_hidden_dim)
        self.fisv_comm_encoder = self.build_feature_encoder(fisv_hidden_dim, fisv_hidden_dim)
        self.fisv_diff0_encoder = self.build_feature_encoder(fisv_hidden_dim, fisv_hidden_dim)
        self.fisv_diff1_encoder = self.build_feature_encoder(fisv_hidden_dim, fisv_hidden_dim)

        self.fisv_regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(fisv_hidden_dim, 512),
            nn.ReLU(True)
        )

        self.last_fuse = nn.Linear(self.module_num, 1, bias=False)

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------
    def build_encoder(self, input_size):
        # original AQA-7 encoder: split 800 into 400+400, so keep this unchanged
        return nn.Sequential(
            nn.Linear(input_size // 2, self.hidden1 // 2),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True)
        )

    def build_modality_proj(self, input_dim, output_dim):
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True)
        )

    def build_feature_encoder(self, input_dim, output_dim):
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.Dropout(self.dropout_rate),
            nn.ReLU(True)
        )

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    def _get_graphs(self):
        """
        Return:
        - save_graph=True  -> [A, H, J, J], where A=self.task_num
        - save_graph=False -> [1, H, J, J]
        """
        if self.save_graph:
            spatial_graphs = torch.abs(self.spatial_mats * self.joint_graphs)
            temporal_graphs = torch.abs(self.temporal_mats * self.joint_graphs)

            if self.g_e_graph:
                general_spatial_graphs = torch.abs(self.general_spatial_mats * self.joint_graphs)
                general_temporal_graphs = torch.abs(self.general_temporal_mats * self.joint_graphs)
                spatial_graphs = (1 - self.alpha) * spatial_graphs + self.alpha * general_spatial_graphs
                temporal_graphs = (1 - self.alpha) * temporal_graphs + self.alpha * general_temporal_graphs
        else:
            general_spatial_graphs = torch.abs(self.general_spatial_mats * self.joint_graphs)
            general_temporal_graphs = torch.abs(self.general_temporal_mats * self.joint_graphs)
            spatial_graphs = general_spatial_graphs.unsqueeze(0)
            temporal_graphs = general_temporal_graphs.unsqueeze(0)

        return spatial_graphs, temporal_graphs

    def _graph_reasoning_core(self, feat_whole, feat_patch):
        """
        Shared graph reasoning core for Fis-V after latent token construction.

        feat_whole: [B, T, C]
        feat_patch: [B, J, T, C]
        return:
            fused_feat: [B, A, 512]
        """
        B, J, T, D = feat_patch.shape
        spatial_graphs, temporal_graphs = self._get_graphs()

        comm_H0 = rearrange(feat_patch, 'B J T D -> B T J D')
        comm_h1s = rearrange(
            torch.matmul(rearrange(feat_patch, 'B J T D -> (B T D) J'), spatial_graphs),
            'A H (B T D) J -> A H B T J D',
            B=B, T=T, D=D
        )

        diff_mat_fp0 = rearrange(feat_patch, 'B J T D -> B T D J')
        diff_mat_fp1 = torch.cat([diff_mat_fp0[:, 1:], diff_mat_fp0[:, -1].unsqueeze(dim=1)], dim=1)

        diff_mat_f0 = diff_mat_fp0[..., None, :] - diff_mat_fp0[..., None]
        diff_mat_f1 = diff_mat_fp1[..., None, :] - diff_mat_fp0[..., None]

        diff_d0 = torch.matmul(
            diff_mat_f0.reshape(-1, J, J) * spatial_graphs.unsqueeze(dim=2),
            self.spatial_JCWs.unsqueeze(dim=1)
        )
        diff_d1 = torch.matmul(
            diff_mat_f1.reshape(-1, J, J) * temporal_graphs.unsqueeze(dim=2),
            self.temporal_JCWs.unsqueeze(dim=1)
        )

        diff_d0 = rearrange(diff_d0, 'A H (B T D) J 1 -> A H B T J D', B=B, T=T, D=D)
        diff_d1 = rearrange(diff_d1, 'A H (B T D) J 1 -> A H B T J D', B=B, T=T, D=D)

        feat_whole_shift = torch.cat(
            [feat_whole[:, 1:], feat_whole[:, -1].unsqueeze(dim=1)],
            dim=1
        )
        feat_diff = torch.abs(feat_whole_shift - feat_whole)

        # encode
        encoded_whole = self.fisv_whole_encoder(feat_whole)       # [B, T, C]
        encoded_diff = self.fisv_diffwhole_encoder(feat_diff)     # [B, T, C]
        encoded_comm0 = self.fisv_patch_encoder(comm_H0)          # [B, T, J, C]

        A = comm_h1s.shape[0]
        encoded_whole = repeat(encoded_whole, 'B T C -> A B T J C', A=A, J=J)
        encoded_diff = repeat(encoded_diff, 'B T C -> A B T J C', A=A, J=J)
        encoded_comm0 = repeat(encoded_comm0, 'B T J C -> A B T J C', A=A)

        encoded_comm1 = [self.fisv_comm_encoder(x) for x in comm_h1s.transpose(0, 1)]
        encoded_diff0 = [self.fisv_diff0_encoder(x) for x in diff_d0.transpose(0, 1)]
        encoded_diff1 = [self.fisv_diff1_encoder(x) for x in diff_d1.transpose(0, 1)]

        all_feats = [encoded_whole, encoded_diff, encoded_comm0] + encoded_comm1 + encoded_diff0 + encoded_diff1
        out = torch.stack(all_feats, dim=4)  # [A, B, T, J, M, C]

        fused_feat = reduce(out, 'A B T J M C -> A B C', 'mean')
        fused_feat = self.fisv_regressor(fused_feat).transpose(0, 1)  # [B, A, 512]
        return fused_feat, None

    # ------------------------------------------------------------------
    # Original AQA-7 path
    # ------------------------------------------------------------------
    def encode_feats(self, feat_whole, feat_diff, comm_H0, comm_h1s, diff_d0, diff_d1, rgb=True):
        assert self.patch_size == self.whole_size
        idx, begin, end = 0, 0, self.patch_size // 2
        if rgb[0] == 0:
            idx, begin, end = 1, self.patch_size // 2, self.patch_size

        _encoded_whole = self.encoders_whole[idx](feat_whole[..., begin:end])
        encoded_diff = self.encoders_diffwhole[idx](feat_diff[..., begin:end])
        encoded_comm0 = self.encoders_comm0[idx](comm_H0[..., begin:end])
        encoded_comm0 = repeat(encoded_comm0, 'B T J D -> A B T J D', J=self.joint_num, A=comm_h1s.shape[0])

        encoded_comm1 = [self.encoders_comm0[idx](x[..., begin:end]) for x in comm_h1s.transpose(0, 1)]
        encoded_diff0 = [self.encoders_diff0[idx](x[..., begin:end]) for x in diff_d0.transpose(0, 1)]
        encoded_diff1 = [self.encoders_diff1[idx](x[..., begin:end]) for x in diff_d1.transpose(0, 1)]
        encoded_whole = repeat(_encoded_whole, 'B T D -> A B T J D', J=self.joint_num, A=comm_h1s.shape[0])
        encoded_diff = repeat(encoded_diff, 'B T D -> A B T J D', J=self.joint_num, A=comm_h1s.shape[0])

        all_feats = [encoded_whole, encoded_diff, encoded_comm0] + encoded_comm1 + encoded_diff0 + encoded_diff1
        out = torch.stack(all_feats, dim=3)
        return out, _encoded_whole

    def _forward_aqa7(self, feat_whole, feat_patch):
        B, J, T, D = feat_patch.shape
        spatial_graphs, temporal_graphs = self._get_graphs()

        comm_H0 = rearrange(feat_patch, 'B J T D -> B T J D')
        comm_h1s = rearrange(
            torch.matmul(rearrange(feat_patch, 'B J T D -> (B T D) J'), spatial_graphs),
            'A H (B T D) J -> A H B T J D',
            B=B, D=D
        )

        diff_mat_fp0 = rearrange(feat_patch, 'B J T D -> B T D J')
        diff_mat_fp1 = torch.cat([diff_mat_fp0[:, 1:], diff_mat_fp0[:, -1].unsqueeze(dim=1)], dim=1)
        diff_mat_f0 = diff_mat_fp0[..., None, :] - diff_mat_fp0[..., None]
        diff_mat_f1 = diff_mat_fp1[..., None, :] - diff_mat_fp0[..., None]

        diff_d0 = torch.matmul(
            diff_mat_f0.reshape(-1, J, J) * spatial_graphs.unsqueeze(dim=2),
            self.spatial_JCWs.unsqueeze(dim=1)
        )
        diff_d1 = torch.matmul(
            diff_mat_f1.reshape(-1, J, J) * temporal_graphs.unsqueeze(dim=2),
            self.temporal_JCWs.unsqueeze(dim=1)
        )
        diff_d0 = rearrange(diff_d0, 'A H (B T D) J 1 -> A H B T J D', B=B, T=T, D=D)
        diff_d1 = rearrange(diff_d1, 'A H (B T D) J 1 -> A H B T J D', B=B, T=T, D=D)

        diff_mat_fp1 = torch.cat([feat_whole[:, 1:], feat_whole[:, -1].unsqueeze(dim=1)], dim=1)
        feat_diff = torch.abs(diff_mat_fp1 - feat_whole)

        bs = feat_whole.shape[0]
        rgb_t = torch.ones(bs, device=feat_whole.device)
        rgb_f = torch.zeros(bs, device=feat_whole.device)

        rgb_feat, _ = self.encode_feats(feat_whole, feat_diff, comm_H0, comm_h1s, diff_d0, diff_d1, rgb=rgb_t)
        flow_feat, _ = self.encode_feats(feat_whole, feat_diff, comm_H0, comm_h1s, diff_d0, diff_d1, rgb=rgb_f)

        fused_feat = torch.cat([rgb_feat + flow_feat, rgb_feat + flow_feat], dim=5)
        fused_feat = reduce(fused_feat, 'A B T J X D -> A B D', 'mean')
        fused_feat = self.regressor(fused_feat).transpose(0, 1)
        return fused_feat, None

    # ------------------------------------------------------------------
    # New Fis-V path
    # ------------------------------------------------------------------
    def _forward_fisv(self, rgb_feat, flow_feat, skel_feat):
        """
        rgb_feat : [B, T, 1024]
        flow_feat: [B, T, 1024]
        skel_feat: [B, T, 256]
        """
        # align temporal length inside model for extra robustness
        min_t = min(rgb_feat.shape[1], flow_feat.shape[1], skel_feat.shape[1])
        rgb_feat = rgb_feat[:, :min_t]
        flow_feat = flow_feat[:, :min_t]
        skel_feat = skel_feat[:, :min_t]

        rgb_h = self.rgb_proj(rgb_feat)      # [B, T, C]
        flow_h = self.flow_proj(flow_feat)   # [B, T, C]
        skel_h = self.skel_proj(skel_feat)   # [B, T, C]

        fused_seq = self.fisv_fuse(torch.cat([rgb_h, flow_h, skel_h], dim=-1))  # [B, T, C]

        # [B, T, J*C] -> [B, J, T, C]
        latent_tokens = self.token_generator(fused_seq)
        latent_tokens = rearrange(
            latent_tokens,
            'B T (J C) -> B J T C',
            J=self.joint_num,
            C=self.fisv_hidden_dim
        )

        return self._graph_reasoning_core(fused_seq, latent_tokens)

    # ------------------------------------------------------------------
    # Unified forward
    # ------------------------------------------------------------------
    def forward(self, *inputs):
        """
        AQA-7:
            model(feat_whole, feat_patch)

        Fis-V:
            model(rgb_feat, flow_feat, skel_feat)
        """
        if len(inputs) == 2:
            return self._forward_aqa7(inputs[0], inputs[1])
        elif len(inputs) == 3:
            return self._forward_fisv(inputs[0], inputs[1], inputs[2])
        else:
            raise ValueError(
                f"ASS_JRG.forward expects 2 inputs (AQA-7) or 3 inputs (Fis-V), got {len(inputs)}"
            )


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


def run_jrg(model_, *inputs, save_graph=False, seen_tasks=None, is_train=False, args=None):
    """
    Compatible with both:
    - run_jrg(model, feat_whole, feat_patch, ...)
    - run_jrg(model, rgb, flow, skel, ...)
    """
    if seen_tasks is None:
        seen_tasks = []

    fused_feat, featmap_list = model_(*inputs)
    fused_feat = fused_feat.transpose(0, 1)
    if not save_graph:
        fused_feat = fused_feat[0]
    return fused_feat, featmap_list


def init_e_graph(model_, t, seen_tasks=None):
    if seen_tasks is None:
        seen_tasks = []

    if t == 0:
        alpha = model_.module.alpha
        g_spatial_mat = copy.deepcopy(model_.module.general_spatial_mats)
        g_temporal_mat = copy.deepcopy(model_.module.general_temporal_mats)
        model_.module.spatial_mats[seen_tasks[0]] = (1 - alpha) * g_spatial_mat
        model_.module.temporal_mats[seen_tasks[0]] = (1 - alpha) * g_temporal_mat
    else:
        model_.module.spatial_mats[seen_tasks[-1]] += copy.deepcopy(model_.module.spatial_mats[seen_tasks[-2]])
        model_.module.temporal_mats[seen_tasks[-1]] += copy.deepcopy(model_.module.temporal_mats[seen_tasks[-2]])
    return


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    net = ASS_JRG(
        whole_size=800,
        patch_size=800,
        seg_num=12,
        joint_num=17,
        out_dim=1,
        save_graph=True,
        G_E_graph=True,
        alpha=0.5
    ).cuda()
    print(net)

    # Fis-V example
    rgb = torch.randn(2, 138, 1024).cuda()
    flow = torch.randn(2, 138, 1024).cuda()
    skel = torch.randn(2, 138, 256).cuda()

    fused_feat, featmap = net(rgb, flow, skel)
    print("Fis-V output shape:", fused_feat.shape)  # [B, A, 512]