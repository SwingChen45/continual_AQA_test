# coding=utf-8
from builtins import print
import os
import sys
import time
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from scipy import stats

import builder
import get_optim
import utils
import loss_fn

from models.JRG_ASS_FisV import run_jrg, init_e_graph


def get_size(obj, seen=None):
    """calculate the size of an object"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    if isinstance(obj, (list, tuple, dict, set)):
        size += sum(get_size(item, seen) for item in obj)
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    return size


# =========================================================
# Generic helpers for AQA-7 / Fis-V
# =========================================================
def is_fisv(args):
    return hasattr(args, "dataset_name") and args.dataset_name.lower() == "fisv"


def unpack_batch(batch, args, with_data_idx=False):
    """
    AQA-7:
        train/test batch: (feat_whole, feat_patch, scores, action_id)
        list-loader batch: (feat_whole, feat_patch, scores, action_id, data_idx)

    Fis-V:
        train/test batch: (rgb, flow, skel, scores, action_id)
        list-loader batch: (rgb, flow, skel, scores, action_id, data_idx)
    """
    if is_fisv(args):
        if with_data_idx:
            rgb, flow, skel, scores, action_id, data_idx = batch
            inputs = {"rgb": rgb, "flow": flow, "skel": skel}
            return inputs, scores, action_id, data_idx
        else:
            rgb, flow, skel, scores, action_id = batch
            inputs = {"rgb": rgb, "flow": flow, "skel": skel}
            return inputs, scores, action_id
    else:
        if with_data_idx:
            feat_whole, feat_patch, scores, action_id, data_idx = batch
            inputs = {"whole": feat_whole, "patch": feat_patch}
            return inputs, scores, action_id, data_idx
        else:
            feat_whole, feat_patch, scores, action_id = batch
            inputs = {"whole": feat_whole, "patch": feat_patch}
            return inputs, scores, action_id


def move_batch_to_cuda(inputs, scores, action_id):
    moved_inputs = {}
    for k, v in inputs.items():
        moved_inputs[k] = v.cuda()
    scores = scores.float().cuda()
    action_id = action_id.type(torch.int64).cuda()
    return moved_inputs, scores, action_id


def forward_model(model, inputs, args, seen_tasks=None):
    if seen_tasks is None:
        seen_tasks = []

    if is_fisv(args):
        feat, featmap_list = run_jrg(
            model,
            inputs["rgb"],
            inputs["flow"],
            inputs["skel"],
            save_graph=args.save_graph,
            seen_tasks=seen_tasks,
            args=args,
        )
    else:
        feat, featmap_list = run_jrg(
            model,
            inputs["whole"],
            inputs["patch"],
            save_graph=args.save_graph,
            seen_tasks=seen_tasks,
            args=args,
        )
    return feat, featmap_list


def select_task_feature(feat, action_id, args):
    """
    When save_graph=True, feat shape is [A, B, 512].
    Select the feature corresponding to current task/action_id.
    """
    if args.save_graph:
        feat = feat.transpose(0, 1).gather(
            1,
            action_id.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 512)
        ).squeeze(1)
    return feat


# =========================================================
# Exemplar helpers (self-contained; no dependency on old utils replay format)
# =========================================================
def reconstruct_exemplar_set(exemplar_set, args):
    if exemplar_set is None:
        return None

    if is_fisv(args):
        rgb_all = torch.Tensor([])
        flow_all = torch.Tensor([])
        skel_all = torch.Tensor([])
        score_all = torch.Tensor([])
        action_all = []

        for a in range(len(exemplar_set)):
            if len(exemplar_set[a]) == 0:
                continue
            for exemplar in exemplar_set[a]:
                rgb_all = torch.cat((rgb_all, torch.tensor(exemplar[0]).float().unsqueeze(0)), dim=0)
                flow_all = torch.cat((flow_all, torch.tensor(exemplar[1]).float().unsqueeze(0)), dim=0)
                skel_all = torch.cat((skel_all, torch.tensor(exemplar[2]).float().unsqueeze(0)), dim=0)
                score_all = torch.cat((score_all, torch.tensor(exemplar[3]).float().unsqueeze(0)), dim=0)
                action_all += [a]

        if len(action_all) == 0:
            return None

        return {
            "rgb": rgb_all,
            "flow": flow_all,
            "skel": skel_all,
            "scores": score_all,
            "action_ids": action_all,
        }

    else:
        whole_all = torch.Tensor([])
        patch_all = torch.Tensor([])
        score_all = torch.Tensor([])
        action_all = []

        for a in range(len(exemplar_set)):
            if len(exemplar_set[a]) == 0:
                continue
            for exemplar in exemplar_set[a]:
                whole_all = torch.cat((whole_all, torch.tensor(exemplar[0]).float().unsqueeze(0)), dim=0)
                patch_all = torch.cat((patch_all, torch.tensor(exemplar[1]).float().unsqueeze(0)), dim=0)
                score_all = torch.cat((score_all, torch.tensor(exemplar[2]).float().unsqueeze(0)), dim=0)
                action_all += [a]

        if len(action_all) == 0:
            return None

        return {
            "whole": whole_all,
            "patch": patch_all,
            "scores": score_all,
            "action_ids": action_all,
        }


def random_select_exemplar(combined_exemplar, b, count, args):
    np.random.seed(count)

    total_num = combined_exemplar["scores"].shape[0]
    select_id = np.random.randint(0, total_num, size=b)

    selected_inputs = {}
    if is_fisv(args):
        selected_inputs["rgb"] = combined_exemplar["rgb"][select_id]
        selected_inputs["flow"] = combined_exemplar["flow"][select_id]
        selected_inputs["skel"] = combined_exemplar["skel"][select_id]
    else:
        selected_inputs["whole"] = combined_exemplar["whole"][select_id]
        selected_inputs["patch"] = combined_exemplar["patch"][select_id]

    selected_score = combined_exemplar["scores"][select_id]
    selected_action = torch.tensor(combined_exemplar["action_ids"]).float()[select_id]
    return selected_inputs, selected_score, selected_action, select_id


def concat_current_and_exemplar(inputs, scores, action_id, ex_inputs, ex_scores, ex_action_id, args):
    merged_inputs = {}
    if is_fisv(args):
        merged_inputs["rgb"] = torch.cat([inputs["rgb"], ex_inputs["rgb"]], dim=0)
        merged_inputs["flow"] = torch.cat([inputs["flow"], ex_inputs["flow"]], dim=0)
        merged_inputs["skel"] = torch.cat([inputs["skel"], ex_inputs["skel"]], dim=0)
    else:
        merged_inputs["whole"] = torch.cat([inputs["whole"], ex_inputs["whole"]], dim=0)
        merged_inputs["patch"] = torch.cat([inputs["patch"], ex_inputs["patch"]], dim=0)

    scores = torch.cat([scores, ex_scores], dim=0)
    action_id = torch.cat([action_id, ex_action_id], dim=0)
    return merged_inputs, scores, action_id


def reduce_exemplar_sets(m, m_p, exemplar_set):
    new_set = [[] for _ in range(len(exemplar_set))]
    for a in range(len(exemplar_set)):
        if len(exemplar_set[a]) == 0:
            continue
        jump = int(m_p / m)
        l = [i for i in range(0, m_p, jump)]
        nl = l[0:int(m - 1)]
        nl.append(l[-1])
        new_set[a] = [exemplar_set[a][j] for j in range(len(exemplar_set[a])) if j in nl]
        exemplar_set[a] = new_set[a]


def get_features_score(jrg, rgs, loader, seen_tasks, args):
    combined_feature = torch.Tensor([])
    combined_score = torch.Tensor([])

    for batch_idx, batch in enumerate(loader):
        inputs, scores, action_id, data_idx = unpack_batch(batch, args, with_data_idx=True)
        inputs, scores, action_id = move_batch_to_cuda(inputs, scores, action_id)

        feat, _ = forward_model(jrg, inputs, args, seen_tasks=seen_tasks)
        feat = select_task_feature(feat, action_id, args)

        feat = feat.detach().cpu()
        combined_feature = torch.cat((combined_feature, feat), dim=0)
        combined_score = torch.cat((combined_score, scores.reshape(-1).cpu()), dim=0)

    feature_list = combined_feature.cpu()
    score_list = [score for score in combined_score]
    return feature_list, score_list


def herding(feature_list, score_list, m, index_list=None, selected_index=None):
    assert len(feature_list) >= m
    if index_list is None:
        index_list = [i for i in range(len(score_list))]
    if selected_index is None:
        selected_index = []

    feature_list = feature_list.cpu()
    center_feature = torch.mean(feature_list, 0)
    dis_list = [torch.norm((center_feature - feature_list[i])) for i in range(feature_list.shape[0])]
    min_idx = np.argmin(dis_list)

    selected_index.append(index_list[min_idx])
    new_feature_list = torch.cat([feature_list[:min_idx], feature_list[min_idx + 1:]], dim=0)
    new_score_list = score_list[:min_idx] + score_list[min_idx + 1:]
    del index_list[min_idx]

    if m > 1:
        herding(new_feature_list, new_score_list, m - 1, index_list, selected_index)
    return selected_index


def get_m_exemplar_simple(task, packed_sorted_list, m, args, feature_extractor, rgs, seen_tasks):
    data_size = len(packed_sorted_list)
    select_idx = []

    if args.replay_method == 'group_replay':
        if 'debug' in args.exp_name:
            print('group_replay')
        jump = int(data_size / m)
        idx_list_with_jump = [i for i in range(0, data_size, jump)]
        select_idx = idx_list_with_jump[0:int(m - 1)]
        select_idx.append(idx_list_with_jump[-1])

    elif args.replay_method == 'random':
        idx_list = np.arange(data_size)
        np.random.shuffle(idx_list)
        select_idx = idx_list[:int(m)].tolist()

    elif args.replay_method == 'herding':
        loader = builder.load_data_from_list(
            packed_sorted_list,
            task,
            dataset_name=args.dataset_name
        )
        features2herding, scores2herding = get_features_score(
            feature_extractor, rgs, loader, seen_tasks, args
        )
        select_idx = herding(features2herding, scores2herding, m)

    data2save = [packed_sorted_list[j] for j in range(len(packed_sorted_list)) if j in select_idx]
    return data2save


def after_train_update_exemplar(t, task, exemplar_set, packed_list, args, jrg, score_rgs, seen_tasks):
    m = args.memory_size / (t + 1)
    packed_sorted_list = sorted(packed_list, key=lambda x: x[-1])

    if t != 0:
        m_p = int(args.memory_size / t)
        reduce_exemplar_sets(m, m_p, exemplar_set)

    data2save = get_m_exemplar_simple(
        task=task,
        packed_sorted_list=packed_sorted_list,
        m=m,
        args=args,
        feature_extractor=jrg,
        rgs=score_rgs,
        seen_tasks=seen_tasks
    )
    exemplar_set[task] = data2save
    return exemplar_set


def pack_purely_task_data(purely_task_data, args):
    if is_fisv(args):
        train_rgb, train_flow, train_skel, train_scores = purely_task_data[:4]
        packed_list = []
        for i in range(len(train_scores)):
            packed_list.append([
                train_rgb[i],
                train_flow[i],
                train_skel[i],
                train_scores[i]
            ])
        return packed_list
    else:
        train_whole, train_patch, train_scores = purely_task_data[:3]
        packed_list = []
        for i in range(len(train_scores)):
            packed_list.append([
                train_whole[i],
                train_patch[i],
                train_scores[i]
            ])
        return packed_list


# =========================================================
# Training / Eval
# =========================================================
def train_epoch(
    t, task, epoch, jrg, jrg_pre, score_rgs, diff_rgs,
    dataloader, optimizer, mse, kl, ce, softmax, args,
    seen_tasks=None, combined_exemplar=None
):
    if seen_tasks is None:
        seen_tasks = []

    jrg.train()
    score_rgs.train()
    diff_rgs.train()
    jrg_pre.eval()
    torch.set_grad_enabled(True)

    total_loss = 0.0
    total_mse = 0.0
    total_distill = 0.0
    total_st_pod = 0.0
    total_diff = 0.0
    total_graph_distill = 0.0

    count = 0
    for batch_idx, batch in enumerate(dataloader):
        inputs, scores, action_id = unpack_batch(batch, args)
        batch_size = action_id.shape[0]

        # old data replay
        select_score = None
        if combined_exemplar is not None:
            ex_inputs, ex_scores, ex_action_id, select_id = random_select_exemplar(
                combined_exemplar, batch_size, count, args
            )
            inputs, scores, action_id = concat_current_and_exemplar(
                inputs, scores, action_id, ex_inputs, ex_scores, ex_action_id, args
            )
            select_score = ex_scores

        # cuda
        inputs, scores, action_id = move_batch_to_cuda(inputs, scores, action_id)

        # forward current / previous model
        feat, featmap_list = forward_model(jrg, inputs, args, seen_tasks=seen_tasks)
        feat_pre, featmap_pre_list = forward_model(jrg_pre, inputs, args, seen_tasks=seen_tasks)
        feat_pre = feat_pre.detach()

        if args.save_graph:
            feat_distill = feat
            feat_pre_distill = feat_pre
            feat = select_task_feature(feat, action_id, args)
            feat_pre = select_task_feature(feat_pre, action_id, args)

        pred_score = score_rgs(feat)

        # augmentation of selected previous data
        if combined_exemplar is not None:
            aug_inputs, aug_helper_score, aug_helper_action_id, aug_helper_select_id = random_select_exemplar(
                combined_exemplar, args.num_helpers, count, args
            )
            aug_inputs, aug_helper_score, aug_helper_action_id = move_batch_to_cuda(
                aug_inputs, aug_helper_score, aug_helper_action_id
            )

            aug_helper_feat, _ = forward_model(jrg_pre, aug_inputs, args, seen_tasks=seen_tasks)
            if args.save_graph:
                aug_helper_feat = select_task_feature(aug_helper_feat, aug_helper_action_id, args)
            aug_helper_feat = aug_helper_feat.detach()

            old_half = int(feat_pre.shape[0] / 2)
            aug_feat, aug_score = utils.feat_score_aug(
                feat_pre[old_half:].cpu(),
                aug_helper_feat.cpu(),
                select_score,
                aug_helper_score.cpu(),
                aug_scale=args.aug_scale,
                with_weight=args.aug_w_weight
            )

            aug_feat = aug_feat.cuda()
            aug_score = aug_score.cuda()
            score_diff = scores[old_half:] - aug_score
            combined_feature = torch.cat((feat[old_half:], aug_feat), dim=-1)

            if args.diff_loss:
                pred_diff = diff_rgs(combined_feature)
            elif args.aug_rgs:
                pred_aug_score = score_rgs(aug_feat)

        loss = 0.0
        distill_loss = 0.0
        st_pod_loss = 0.0
        diff_loss = 0.0
        graph_distill_loss = 0.0

        mse_loss = loss_fn.mse_(pred_score, scores, mse)

        if combined_exemplar is not None and args.diff_loss:
            diff_loss = loss_fn.mse_(pred_diff, score_diff, mse)
        if combined_exemplar is not None and args.aug_rgs:
            diff_loss = loss_fn.mse_(pred_aug_score, aug_score, mse)

        if t != 0:
            if args.save_graph:
                distill_loss = loss_fn.distill_save_graph_(
                    feat_distill, feat_pre_distill, mse, action_id, seen_tasks
                )
            else:
                distill_loss = loss_fn.distill_(feat, feat_pre, mse, softmax)

        if (t != 0) and args.pod_loss:
            st_pod_loss = loss_fn.st_pod_(
                featmap_list, featmap_pre_list, temporal_pool=True, norm=True
            )

        if (t != 0) and args.graph_distill:
            graph_distill_loss = loss_fn.ge_graph_distill_(
                model_=jrg, model_pre_=jrg_pre, seen_tasks=seen_tasks, mse=mse
            )

        loss = (
            args.lambda_distill * distill_loss
            + args.lambda_pod * st_pod_loss
            + mse_loss
            + args.lambda_diff * diff_loss
            + args.lambda_graph_distill * graph_distill_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse_loss.item()

        if (t != 0) and (args.approach != 'finetune'):
            total_distill += distill_loss.item()
        if (t != 0) and (args.approach == 'podnet'):
            total_st_pod += st_pod_loss.item()
        if (t != 0) and args.graph_distill and (args.lambda_graph_distill != 0):
            total_graph_distill += graph_distill_loss.item()
        if combined_exemplar is not None and args.diff_loss:
            total_diff += diff_loss.item()

        count += 1
    avg_loss = total_loss / max(count, 1)
    avg_mse = total_mse / max(count, 1)
    avg_distill = total_distill / max(count, 1)
    avg_diff = total_diff / max(count, 1)
    avg_graph = total_graph_distill / max(count, 1)
    avg_pod = total_st_pod / max(count, 1)

    print(
        f"  TrainLoss | total: {avg_loss:.6f} | "
        f"mse: {avg_mse:.6f} | "
        f"distill: {avg_distill:.6f} | "
        f"diff: {avg_diff:.6f} | "
        f"graph: {avg_graph:.6f} | "
        f"pod: {avg_pod:.6f}"
    )

    return jrg, score_rgs, diff_rgs, optimizer


# def test_net(t, jrg, score_rgs, dataloaders, rho_matrix, rl2_matrix, args, seen_tasks=None):
#     if seen_tasks is None:
#         seen_tasks = []

#     jrg.eval()
#     score_rgs.eval()
#     torch.set_grad_enabled(False)

#     for i, a_task in enumerate(seen_tasks):
#         dataloader = dataloaders[a_task]
#         true_scores = []
#         pred_scores = []

#         for batch_idx, batch in enumerate(dataloader):
#             inputs, scores, action_id = unpack_batch(batch, args)
#             inputs, scores, action_id = move_batch_to_cuda(inputs, scores, action_id)

#             feat, _ = forward_model(jrg, inputs, args, seen_tasks=seen_tasks)
#             feat = select_task_feature(feat, action_id, args)

#             pred_score = score_rgs(feat)

#             pred_scores.extend(pred_score.detach().data.cpu().reshape(-1).numpy())
#             true_scores.extend(scores.detach().data.cpu().reshape(-1).numpy())

#         pred_scores = np.array(pred_scores)
#         true_scores = np.array(true_scores)

#         rho, p = stats.spearmanr(pred_scores, true_scores)
#         RL2 = np.power(
#             (pred_scores - true_scores) / (true_scores.max() - true_scores.min()),
#             2
#         ).sum() / true_scores.shape[0]

#         rho_matrix[i][t] = rho
#         rl2_matrix[i][t] = RL2

#     return rho_matrix, rl2_matrix

def test_net(t, jrg, score_rgs, dataloaders, rho_matrix, mse_matrix, rl2_matrix, args, seen_tasks=[]):

    jrg.eval()
    score_rgs.eval()
    torch.set_grad_enabled(False)

    for i, a_task in enumerate(seen_tasks):
        dataloader = dataloaders[a_task]
        true_scores = []
        pred_scores = []

        for batch_idx, batch in enumerate(dataloader):
            inputs, scores, action_id = unpack_batch(batch, args)
            inputs, scores, action_id = move_batch_to_cuda(inputs, scores, action_id)

            feat, _ = forward_model(jrg, inputs, args, seen_tasks=seen_tasks)
            feat = select_task_feature(feat, action_id, args)

            pred_score = score_rgs(feat)

            pred_scores.extend(pred_score.detach().data.cpu().reshape(-1).numpy())
            true_scores.extend(scores.detach().data.cpu().reshape(-1).numpy())

        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)

        rho, _ = stats.spearmanr(pred_scores, true_scores)
        mse_value = np.mean((pred_scores - true_scores) ** 2)
        rl2_value = np.power(
            (pred_scores - true_scores) / (true_scores.max() - true_scores.min()),
            2
        ).sum() / true_scores.shape[0]

        rho_matrix[i][t] = rho
        mse_matrix[i][t] = mse_value
        rl2_matrix[i][t] = rl2_value

    return rho_matrix, mse_matrix, rl2_matrix


def eval_net(
    jrg, score_rgs, dataloader, best_jrg, best_rgs, rho_best, epoch_best,
    L2_min, RL2_min, args, epoch=0, step='', seen_tasks=None,
    rho_bank=None, local_best_jrg=None, local_best_rgs=None, local_best_result=None
):
    if seen_tasks is None:
        seen_tasks = []
    if rho_bank is None:
        rho_bank = []

    jrg.eval()
    score_rgs.eval()
    torch.set_grad_enabled(False)

    print(' {}: '.format(step), end='')
    true_scores = []
    pred_scores = []

    for batch_idx, batch in enumerate(dataloader):
        inputs, scores, action_id = unpack_batch(batch, args)
        inputs, scores, action_id = move_batch_to_cuda(inputs, scores, action_id)

        feat, _ = forward_model(jrg, inputs, args, seen_tasks=seen_tasks)
        feat = select_task_feature(feat, action_id, args)
        pred_score = score_rgs(feat)

        pred_scores.extend(pred_score.detach().data.cpu().reshape(-1).numpy())
        true_scores.extend(scores.detach().data.cpu().reshape(-1).numpy())

    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)

    rho, p = stats.spearmanr(pred_scores, true_scores)
    L2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
    RL2 = np.power(
        (pred_scores - true_scores) / (true_scores.max() - true_scores.min()),
        2
    ).sum() / true_scores.shape[0]
    eval_mse = np.mean((pred_scores - true_scores) ** 2)

    print(' correlation: %.6f, MSE: %.6f,L2: %.6f, RL2: %.6f' % (rho, eval_mse, L2, RL2), end='   ')

    if step == 'Test':
        if len(rho_bank) < 10:
            rho_bank.append(rho)
        else:
            del rho_bank[0]
            rho_bank.append(rho)

        if rho == max(rho_bank):
            local_best_jrg = utils.get_model(jrg)
            local_best_rgs = utils.get_model(score_rgs)
            local_best_result = (epoch, rho, L2, RL2)

        local_avg_rho = sum(rho_bank) / len(rho_bank)
        if local_avg_rho >= rho_best:
            rho_best = local_best_result[1]
            epoch_best = local_best_result[0]
            L2_min = local_best_result[2]
            RL2_min = local_best_result[3]
            best_jrg = local_best_jrg
            best_rgs = local_best_rgs
            print('*')

        print()
        print('Current best————Corr: %.6f , L2: %.6f , RL2: %.6f @ epoch %d \n'
              % (rho_best, L2_min, RL2_min, epoch_best))

    elif step == 'Train':
        print()
    elif step == 'Test_stage2':
        print()
        return rho, L2, RL2
    else:
        print('Wrong step name')
        return None

    return (
        best_jrg, best_rgs, rho_best, epoch_best,
        L2_min, RL2_min, rho_bank, local_best_jrg, local_best_rgs, local_best_result
    )


def run_exp(args, task_list, action2task, classes_name, exemplar_set):
    # get models
    jrg, score_rgs, diff_rgs = builder.build_moodel(args)
    jrg_pre, score_rgs_pre, _ = builder.build_moodel(args)

    mse = nn.MSELoss()
    kl = nn.KLDivLoss()

    # cuda
    jrg = jrg.cuda()
    jrg_pre = jrg_pre.cuda()
    score_rgs = score_rgs.cuda()
    diff_rgs = diff_rgs.cuda()
    mse = mse.cuda()
    kl = kl.cuda()
    softmax = nn.Softmax().cuda()
    ce = nn.CrossEntropyLoss().cuda()

    # DP
    jrg = nn.DataParallel(jrg)
    jrg_pre = nn.DataParallel(jrg_pre)
    score_rgs = nn.DataParallel(score_rgs)
    diff_rgs = nn.DataParallel(diff_rgs)

    best_jrg = utils.get_model(jrg)
    best_rgs = utils.get_model(score_rgs)
    init_diff_rgs = utils.get_model(diff_rgs)

    # load data
    loaders_test = []
    start_time = time.time()
    for i in range(len(task_list)):
        _, _, _, loader_test, _ = builder.load_data(
            data_root=args.data_root, set_id=i, args=args, exemplar_set=None
        )
        loaders_test.append(loader_test)
    print('dataset making time cost: ', time.time() - start_time)

    # rho_matrix = np.zeros((len(loaders_test), len(loaders_test)), dtype=np.float32)
    # rl2_matrix = np.zeros((len(loaders_test), len(loaders_test)), dtype=np.float32)
    rho_matrix = np.zeros((len(loaders_test), len(loaders_test)), dtype=np.float32)
    mse_matrix = np.zeros((len(loaders_test), len(loaders_test)), dtype=np.float32)
    rl2_matrix = np.zeros((len(loaders_test), len(loaders_test)), dtype=np.float32)

    seen_tasks = []
    for t, task in enumerate(task_list):
        print('*' * 60)
        print('Task {:2d} ({:s})'.format(t, classes_name[task]))
        print('*' * 60)
        seen_tasks.append(task)

        utils.freeze_model(jrg_pre)
        utils.freeze_model(jrg)

        if args.g_e_graph and (not args.graph_random_init):
            print('init graph')
            init_e_graph(jrg, t, seen_tasks=seen_tasks)

        utils.activate_model(jrg)
        utils.activate_model(score_rgs)

        if args.optim_mode == 'new_optim':
            print('use new_optim')
            optimizer = get_optim.get_optim(jrg, score_rgs, diff_rgs, args, optim_id=args.optim_id)
        else:
            optimizer = optim.SGD(
                [
                    {'params': filter(lambda p: p.requires_grad, jrg.parameters()), 'lr': args.base_lr * args.lr_factor},
                    {'params': score_rgs.parameters()},
                    {'params': diff_rgs.parameters()}
                ],
                lr=args.base_lr,
                weight_decay=args.weight_decay
            )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[500, 1000, 1500], gamma=0.5
        )

        dataset_train, dataset_test, loader_train, loader_test, purely_task_data = builder.load_data(
            data_root=args.data_root, set_id=task, args=args, exemplar_set=exemplar_set
        )

        epoch_best = 0
        rho_best = 0
        L2_min = 1000
        RL2_min = 1000

        rho_bank = []
        local_best_jrg, local_best_rgs, local_best_result = None, None, None

        if not args.dataset_mixup:
            combined_exemplar = reconstruct_exemplar_set(exemplar_set, args)
        else:
            combined_exemplar = None

        for epoch in range(0, args.num_epochs):
            print('Epoch: {}'.format(epoch))

            jrg, score_rgs, diff_rgs, optimizer = train_epoch(
                t, task, epoch, jrg, jrg_pre, score_rgs, diff_rgs,
                loader_train, optimizer, mse, kl, ce, softmax,
                args, seen_tasks, combined_exemplar
            )

            best_jrg, best_rgs, _, _, _, _, _, _, _, _ = eval_net(
                jrg, score_rgs, loader_train, best_jrg, best_rgs, rho_best,
                epoch_best, L2_min, RL2_min, args, epoch, 'Train',
                seen_tasks, None, None, None, None
            )

            best_jrg, best_rgs, rho_best, epoch_best, L2_min, RL2_min, rho_bank, \
                local_best_jrg, local_best_rgs, local_best_result = eval_net(
                    jrg, score_rgs, loader_test, best_jrg, best_rgs, rho_best,
                    epoch_best, L2_min, RL2_min, args, epoch, 'Test',
                    seen_tasks, rho_bank, local_best_jrg, local_best_rgs, local_best_result
                )

            if args.exp_name == 'debug':
                print('rho bank: ', rho_bank)
                print('local best result: ', local_best_result)

            if args.lr_decay:
                scheduler.step()
            print()

        utils.set_model_(jrg, best_jrg)
        utils.set_model_(jrg_pre, best_jrg)
        utils.set_model_(score_rgs, best_rgs)

        # update exemplar set
        if args.replay:
            packed_list = pack_purely_task_data(purely_task_data, args)
            exemplar_set = after_train_update_exemplar(
                t, task, exemplar_set, packed_list, args, jrg, score_rgs, seen_tasks
            )

            if args.exp_name == 'debug':
                with open(os.path.join('./ckpt', 'debug', 'exemplar_set.txt'), 'a') as f:
                    str_w = ''
                    for i in range(len(exemplar_set)):
                        str_w += (str(len(exemplar_set[i])) + ' ')
                    f.writelines(str_w + '\n')

                torch.save(
                    exemplar_set,
                    os.path.join('./ckpt', 'debug', 'exemplar_set_torch_{}.torch'.format(t))
                )

        # save ckpt
        if args.save_ckpt:
            ckpt_path = './ckpt/{}/'.format(args.exp_name) + \
                        str(t) + '_' + classes_name[task] + '_best@{}.pth'.format(epoch_best)
            utils.save_model(best_jrg, best_rgs, epoch_best, rho_best, L2_min, RL2_min, ckpt_path)

        # test matrix
        # rho_matrix, rl2_matrix = test_net(
        #     t, jrg, score_rgs, loaders_test, rho_matrix, rl2_matrix, args, seen_tasks
        # )
        # np.savetxt(os.path.join('./ckpt', args.exp_name, 'rho.txt'), rho_matrix, '%.4f')
        # np.savetxt(os.path.join('./ckpt', args.exp_name, 'rl2.txt'), rl2_matrix, '%.4f')
        rho_matrix, mse_matrix, rl2_matrix = test_net(
             t, jrg, score_rgs, loaders_test, rho_matrix, mse_matrix, rl2_matrix, args, seen_tasks
        )

        np.savetxt(os.path.join('./ckpt', args.exp_name, 'rho.txt'), rho_matrix, '%.4f')
        np.savetxt(os.path.join('./ckpt', args.exp_name, 'mse.txt'), mse_matrix, '%.4f')
        np.savetxt(os.path.join('./ckpt', args.exp_name, 'rl2.txt'), rl2_matrix, '%.4f')
        # all tasks finished -> print final summary
    final_srcc = float(np.mean(rho_matrix[:, -1]))
    final_mse = float(np.mean(mse_matrix[:, -1]))
    final_rl2 = float(np.mean(rl2_matrix[:, -1]))

    metric_name = args.score_type if args.dataset_name.lower() == 'fisv' else 'AQA-7'

    print("\n" + "=" * 70)
    print(f"FINAL {metric_name} RESULTS")
    print("-" * 70)
    print(f"Final Average SRCC : {final_srcc:.4f}")
    print(f"Final Average MSE  : {final_mse:.4f}")
    print(f"Final Average RL2  : {final_rl2:.4f}")
    print("-" * 70)
    print("Final column SRCC:", rho_matrix[:, -1])
    print("Final column MSE :", mse_matrix[:, -1])
    print("Final column RL2 :", rl2_matrix[:, -1])
    print("=" * 70 + "\n")


def main():
    args, task_list, action2task, classes_name = builder.build_exp()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    seed = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    exemplar_set = [[] for _ in range(len(task_list))]

    run_exp(args, task_list, action2task, classes_name, exemplar_set)


if __name__ == '__main__':
    start = time.time()
    global writer
    writer = None
    main()
    print('time cost: ', time.time() - start)