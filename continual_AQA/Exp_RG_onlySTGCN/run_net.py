from builtins import print
import sys
import time
import os
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import builder
import utils
import loss_fn
import get_optim

from models.JRG_ASS import run_jrg, init_e_graph
from scipy import stats


def setup_logger(log_file):
    logger = logging.getLogger("rg_run")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def log_print(logger, msg):
    logger.info(msg)


def train_epoch(
    t,
    task,
    epoch,
    jrg,
    jrg_pre,
    score_rgs,
    diff_rgs,
    dataloader,
    optimizer,
    mse,
    kl,
    ce,
    softmax,
    args,
    logger,
    seen_tasks=[],
    combined_exemplar=None
):
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

    for batch_idx, (feat_stgcn, scores, action_id) in enumerate(dataloader):
        batch_size = action_id.shape[0]

        if combined_exemplar is not None:
            select_stgcn, select_score, select_action_id, select_id = \
                utils.random_select_exemplar(combined_exemplar, batch_size, count)
            feat_stgcn = torch.cat([feat_stgcn, select_stgcn], dim=0)
            scores = torch.cat([scores, select_score], dim=0)
            action_id = torch.cat([action_id, select_action_id], dim=0)

        feat_stgcn = feat_stgcn.cuda()
        scores = scores.float().cuda()
        action_id = action_id.type(torch.int64).cuda()

        feat, featmap_list = run_jrg(
            jrg, feat_stgcn,
            save_graph=args.save_graph, seen_tasks=seen_tasks, args=args
        )
        feat_pre, featmap_pre_list = run_jrg(
            jrg_pre, feat_stgcn,
            save_graph=args.save_graph, seen_tasks=seen_tasks, args=args
        )
        feat_pre = feat_pre.detach()

        if args.save_graph:
            feat_distill = feat
            feat_pre_distill = feat_pre
            feat = feat.transpose(0, 1).gather(
                1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 512)
            ).squeeze(1)
            feat_pre = feat_pre.transpose(0, 1).gather(
                1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 512)
            ).squeeze(1)

        pred_score = score_rgs(feat)

        diff_loss = 0.0
        if combined_exemplar is not None:
            aug_helper_stgcn, aug_helper_score, aug_helper_action_id, aug_helper_select_id = \
                utils.random_select_exemplar(combined_exemplar, args.num_helpers, count)
            aug_helper_stgcn = aug_helper_stgcn.cuda()
            aug_helper_action_id = aug_helper_action_id.type(torch.int64).cuda()

            aug_helper_feat, _ = run_jrg(
                jrg_pre, aug_helper_stgcn,
                save_graph=args.save_graph, seen_tasks=seen_tasks, args=args
            )

            if args.save_graph:
                aug_helper_feat = aug_helper_feat.transpose(0, 1).gather(
                    1, aug_helper_action_id.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 512)
                ).squeeze(1)

            aug_helper_feat = aug_helper_feat.detach()

            current_old_half = int(feat_pre.shape[0] / 2)
            if current_old_half < feat_pre.shape[0]:
                aug_feat, aug_score = utils.feat_score_aug(
                    feat_pre[current_old_half:].cpu(),
                    aug_helper_feat.cpu(),
                    scores[current_old_half:].cpu(),
                    aug_helper_score,
                    aug_scale=args.aug_scale,
                    with_weight=args.aug_w_weight
                )
                aug_feat = aug_feat.cuda()
                aug_score = aug_score.cuda()

                if args.diff_loss:
                    score_diff = scores[current_old_half:] - aug_score
                    combined_feature = torch.cat((feat[current_old_half:], aug_feat), dim=-1)
                    pred_diff = diff_rgs(combined_feature)
                    diff_loss = loss_fn.mse_(pred_diff, score_diff, mse)
                elif args.aug_rgs:
                    pred_aug_score = score_rgs(aug_feat)
                    diff_loss = loss_fn.mse_(pred_aug_score, aug_score, mse)

        distill_loss = 0.0
        st_pod_loss = 0.0
        graph_distill_loss = 0.0

        mse_loss = loss_fn.mse_(pred_score, scores, mse)

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

        total_loss += float(loss.item())
        total_mse += float(mse_loss.item())
        total_distill += float(distill_loss if isinstance(distill_loss, float) else distill_loss.item())
        total_st_pod += float(st_pod_loss if isinstance(st_pod_loss, float) else st_pod_loss.item())
        total_diff += float(diff_loss if isinstance(diff_loss, float) else diff_loss.item())
        total_graph_distill += float(
            graph_distill_loss if isinstance(graph_distill_loss, float) else graph_distill_loss.item()
        )
        count += 1

    logs = {
        "loss": total_loss / max(count, 1),
        "mse": total_mse / max(count, 1),
        "distill": total_distill / max(count, 1),
        "pod": total_st_pod / max(count, 1),
        "diff": total_diff / max(count, 1),
        "graph_distill": total_graph_distill / max(count, 1),
    }
    return jrg, score_rgs, diff_rgs, optimizer, logs


def eval_epoch_loss(
    t,
    jrg,
    jrg_pre,
    score_rgs,
    diff_rgs,
    dataloader,
    mse,
    softmax,
    args,
    seen_tasks=[],
    logger=None
):
    jrg.eval()
    score_rgs.eval()
    diff_rgs.eval()
    jrg_pre.eval()
    torch.set_grad_enabled(False)

    total_loss = 0.0
    total_mse = 0.0
    total_distill = 0.0
    total_st_pod = 0.0
    total_diff = 0.0
    total_graph_distill = 0.0
    count = 0

    for batch_idx, (feat_stgcn, scores, action_id) in enumerate(dataloader):
        feat_stgcn = feat_stgcn.cuda()
        scores = scores.float().cuda()
        action_id = action_id.type(torch.int64).cuda()

        feat, featmap_list = run_jrg(
            jrg, feat_stgcn,
            save_graph=args.save_graph, seen_tasks=seen_tasks, args=args
        )
        feat_pre, featmap_pre_list = run_jrg(
            jrg_pre, feat_stgcn,
            save_graph=args.save_graph, seen_tasks=seen_tasks, args=args
        )
        feat_pre = feat_pre.detach()

        if args.save_graph:
            feat_distill = feat
            feat_pre_distill = feat_pre
            feat = feat.transpose(0, 1).gather(
                1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 512)
            ).squeeze(1)
            feat_pre = feat_pre.transpose(0, 1).gather(
                1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 512)
            ).squeeze(1)

        pred_score = score_rgs(feat)
        mse_loss = loss_fn.mse_(pred_score, scores, mse)

        distill_loss = 0.0
        st_pod_loss = 0.0
        diff_loss = 0.0
        graph_distill_loss = 0.0

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

        total_loss += float(loss.item())
        total_mse += float(mse_loss.item())
        total_distill += float(distill_loss if isinstance(distill_loss, float) else distill_loss.item())
        total_st_pod += float(st_pod_loss if isinstance(st_pod_loss, float) else st_pod_loss.item())
        total_diff += float(diff_loss if isinstance(diff_loss, float) else diff_loss)
        total_graph_distill += float(
            graph_distill_loss if isinstance(graph_distill_loss, float) else graph_distill_loss.item()
        )
        count += 1

    logs = {
        "loss": total_loss / max(count, 1),
        "mse": total_mse / max(count, 1),
        "distill": total_distill / max(count, 1),
        "pod": total_st_pod / max(count, 1),
        "diff": total_diff / max(count, 1),
        "graph_distill": total_graph_distill / max(count, 1),
    }
    return logs


def test_net(t, jrg, score_rgs, dataloaders, rho_matrix, rl2_matrix, args, seen_tasks=[]):
    jrg.eval()
    score_rgs.eval()
    torch.set_grad_enabled(False)

    for i, a_task in enumerate(seen_tasks):
        dataloader = dataloaders[a_task]
        true_scores = []
        pred_scores = []
        for batch_idx, (feat_stgcn, scores, action_id) in enumerate(dataloader):
            feat_stgcn = feat_stgcn.cuda()
            scores = scores.float().cuda()
            action_id = action_id.type(torch.int64).cuda()

            feat, _ = run_jrg(
                jrg, feat_stgcn,
                save_graph=args.save_graph, seen_tasks=seen_tasks, args=args
            )
            if args.save_graph:
                feat = feat.transpose(0, 1).gather(
                    1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 512)
                ).squeeze(1)
            pred_score = score_rgs(feat)

            pred_scores.extend(pred_score.detach().cpu().reshape(-1).numpy())
            true_scores.extend(scores.detach().cpu().reshape(-1).numpy())

        pred_scores = np.array(pred_scores)
        true_scores = np.array(true_scores)
        rho, _ = stats.spearmanr(pred_scores, true_scores)
        rl2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]
        rho_matrix[i][t] = rho
        rl2_matrix[i][t] = rl2

    return rho_matrix, rl2_matrix


def eval_net(
    jrg,
    score_rgs,
    dataloader,
    best_jrg,
    best_rgs,
    rho_best,
    epoch_best,
    l2_min,
    rl2_min,
    args,
    logger,
    epoch=0,
    step='',
    seen_tasks=[],
    rho_bank=[],
    local_best_jrg=None,
    local_best_rgs=None,
    local_best_result=None
):
    jrg.eval()
    score_rgs.eval()
    torch.set_grad_enabled(False)

    true_scores = []
    pred_scores = []

    for batch_idx, (feat_stgcn, scores, action_id) in enumerate(dataloader):
        feat_stgcn = feat_stgcn.cuda()
        scores = scores.float().cuda()
        action_id = action_id.type(torch.int64).cuda()

        feat, _ = run_jrg(
            jrg, feat_stgcn,
            save_graph=args.save_graph, seen_tasks=seen_tasks, args=args
        )
        if args.save_graph:
            feat = feat.transpose(0, 1).gather(
                1, action_id.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 512)
            ).squeeze(1)
        pred_score = score_rgs(feat)

        pred_scores.extend(pred_score.detach().cpu().reshape(-1).numpy())
        true_scores.extend(scores.detach().cpu().reshape(-1).numpy())

    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)
    rho, _ = stats.spearmanr(pred_scores, true_scores)
    l2 = np.power(pred_scores - true_scores, 2).sum() / true_scores.shape[0]
    rl2 = np.power((pred_scores - true_scores) / (true_scores.max() - true_scores.min()), 2).sum() / true_scores.shape[0]

    log_print(logger, f'{step}: correlation={rho:.6f}, L2={l2:.6f}, RL2={rl2:.6f}')

    if step == 'Test':
        if len(rho_bank) < 10:
            rho_bank.append(rho)
        else:
            del rho_bank[0]
            rho_bank.append(rho)

        if rho == max(rho_bank):
            local_best_jrg = utils.get_model(jrg)
            local_best_rgs = utils.get_model(score_rgs)
            local_best_result = (epoch, rho, l2, rl2)

        local_avg_rho = sum(rho_bank) / len(rho_bank)
        if local_avg_rho >= rho_best and local_best_result is not None:
            rho_best = local_best_result[1]
            epoch_best = local_best_result[0]
            l2_min = local_best_result[2]
            rl2_min = local_best_result[3]
            best_jrg = local_best_jrg
            best_rgs = local_best_rgs
            log_print(logger, 'New best checkpoint updated.')

        log_print(logger, f'Current best -> Corr={rho_best:.6f}, L2={l2_min:.6f}, RL2={rl2_min:.6f} @ epoch {epoch_best}')
    elif step == 'Train':
        pass
    elif step == 'Test_stage2':
        return rho, l2, rl2
    else:
        raise ValueError('Wrong step name')

    return best_jrg, best_rgs, rho_best, epoch_best, l2_min, rl2_min, rho_bank, local_best_jrg, local_best_rgs, local_best_result


def run_exp(args, task_list, action2task, classes_name, exemplar_set):
    jrg, score_rgs, diff_rgs = builder.build_moodel(args)
    jrg_pre, score_rgs_pre, _ = builder.build_moodel(args)

    mse = nn.MSELoss()
    kl = nn.KLDivLoss()

    jrg = jrg.cuda()
    jrg_pre = jrg_pre.cuda()
    score_rgs = score_rgs.cuda()
    diff_rgs = diff_rgs.cuda()
    mse = mse.cuda()
    kl = kl.cuda()
    softmax = nn.Softmax(dim=1).cuda()
    ce = nn.CrossEntropyLoss().cuda()

    jrg = nn.DataParallel(jrg)
    jrg_pre = nn.DataParallel(jrg_pre)
    score_rgs = nn.DataParallel(score_rgs)
    diff_rgs = nn.DataParallel(diff_rgs)

    best_jrg = utils.get_model(jrg)
    best_rgs = utils.get_model(score_rgs)

    log_dir = os.path.join('./ckpt', args.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, 'train.log'))

    loaders_test = []
    start_time = time.time()
    for i in range(len(task_list)):
        _, _, _, loader_test, _ = builder.load_data(
            data_root=args.data_root, set_id=i, args=args, exemplar_set=None
        )
        loaders_test.append(loader_test)
    log_print(logger, f'dataset making time cost: {time.time() - start_time:.2f}s')

    rho_matrix = np.zeros((len(loaders_test), len(loaders_test)), dtype=np.float32)
    rl2_matrix = np.zeros((len(loaders_test), len(loaders_test)), dtype=np.float32)
    seen_tasks = []

    for t, task in enumerate(task_list):
        log_print(logger, '*' * 70)
        log_print(logger, f'Task {t:2d} ({classes_name[task]})')
        log_print(logger, '*' * 70)
        seen_tasks.append(task)

        utils.freeze_model(jrg_pre)
        utils.freeze_model(jrg)

        if args.g_e_graph and (not args.graph_random_init):
            log_print(logger, 'init graph')
            init_e_graph(jrg, t, seen_tasks=seen_tasks)

        utils.activate_model(jrg)
        utils.activate_model(score_rgs)

        if args.optim_mode == 'new_optim':
            optimizer = get_optim.get_optim(jrg, score_rgs, diff_rgs, args, optim_id=args.optim_id)
        else:
            optimizer = optim.SGD([
                {'params': filter(lambda p: p.requires_grad, jrg.parameters()), 'lr': args.base_lr * args.lr_factor},
                {'params': score_rgs.parameters()},
                {'params': diff_rgs.parameters()}
            ], lr=args.base_lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.5)

        dataset_train, dataset_test, loader_train, loader_test, purely_task_data = builder.load_data(
            data_root=args.data_root, set_id=task, args=args, exemplar_set=exemplar_set
        )

        epoch_best = 0
        rho_best = 0
        l2_min = 1000
        rl2_min = 1000
        rho_bank = []
        local_best_jrg, local_best_rgs, local_best_result = None, None, None

        if not args.dataset_mixup:
            combined_exemplar = utils.reconstruct_exemplar_set(exemplar_set)
        else:
            combined_exemplar = None

        for epoch in range(0, args.num_epochs):
            log_print(logger, f'Epoch {epoch}')

            jrg, score_rgs, diff_rgs, optimizer, train_logs = train_epoch(
                t, task, epoch, jrg, jrg_pre, score_rgs, diff_rgs, loader_train,
                optimizer, mse, kl, ce, softmax, args, logger, seen_tasks, combined_exemplar
            )

            val_logs = eval_epoch_loss(
                t, jrg, jrg_pre, score_rgs, diff_rgs, loader_test,
                mse, softmax, args, seen_tasks, logger
            )

            log_print(
                logger,
                (
                    f'[Loss][Epoch {epoch}] '
                    f'Train total={train_logs["loss"]:.6f}, mse={train_logs["mse"]:.6f}, '
                    f'distill={train_logs["distill"]:.6f}, diff={train_logs["diff"]:.6f}, '
                    f'graph={train_logs["graph_distill"]:.6f}, pod={train_logs["pod"]:.6f} || '
                    f'Test total={val_logs["loss"]:.6f}, mse={val_logs["mse"]:.6f}, '
                    f'distill={val_logs["distill"]:.6f}, diff={val_logs["diff"]:.6f}, '
                    f'graph={val_logs["graph_distill"]:.6f}, pod={val_logs["pod"]:.6f}'
                )
            )

            best_jrg, best_rgs, _, _, _, _, _, _, _, _ = eval_net(
                jrg, score_rgs, loader_train, best_jrg, best_rgs, rho_best, epoch_best,
                l2_min, rl2_min, args, logger, epoch, 'Train', seen_tasks,
                None, None, None, None
            )

            best_jrg, best_rgs, rho_best, epoch_best, l2_min, rl2_min, rho_bank, local_best_jrg, local_best_rgs, local_best_result = eval_net(
                jrg, score_rgs, loader_test, best_jrg, best_rgs, rho_best, epoch_best,
                l2_min, rl2_min, args, logger, epoch, 'Test', seen_tasks,
                rho_bank, local_best_jrg, local_best_rgs, local_best_result
            )

            if args.lr_decay:
                scheduler.step()

        utils.set_model_(jrg, best_jrg)
        utils.set_model_(jrg_pre, best_jrg)
        utils.set_model_(score_rgs, best_rgs)

        if args.replay:
            packed_list = []
            for data_idx in range(len(purely_task_data[1])):
                packed_list.append([
                    purely_task_data[0][data_idx],
                    purely_task_data[1][data_idx],
                ])

            exemplar_set = utils.after_train(
                t, task, exemplar_set, packed_list, args, jrg, score_rgs, seen_tasks
            )

        if args.save_ckpt:
            ckpt_path = './ckpt/{}/'.format(args.exp_name) + str(t) + '_' + classes_name[task] + '_best@{}.pth'.format(epoch_best)
            utils.save_model(best_jrg, best_rgs, epoch_best, rho_best, l2_min, rl2_min, ckpt_path)

        rho_matrix, rl2_matrix = test_net(
            t, jrg, score_rgs, loaders_test, rho_matrix, rl2_matrix, args, seen_tasks
        )
        np.savetxt(os.path.join('./ckpt', args.exp_name, 'rho.txt'), rho_matrix, '%.4f')
        np.savetxt(os.path.join('./ckpt', args.exp_name, 'rl2.txt'), rl2_matrix, '%.4f')

        log_print(logger, 'Current rho matrix:')
        log_print(logger, str(rho_matrix))
        log_print(logger, 'Current rl2 matrix:')
        log_print(logger, str(rl2_matrix))

    log_print(logger, '=' * 70)
    log_print(logger, 'FINAL RESULTS')
    log_print(logger, '=' * 70)
    log_print(logger, 'rho matrix:')
    log_print(logger, str(rho_matrix))
    log_print(logger, 'rl2 matrix:')
    log_print(logger, str(rl2_matrix))

    final_avg_srcc = float(np.mean(rho_matrix[:, -1]))
    final_avg_rl2 = float(np.mean(rl2_matrix[:, -1]))
    log_print(logger, f'Final average SRCC over seen tasks: {final_avg_srcc:.6f}')
    log_print(logger, f'Final average RL2 over seen tasks: {final_avg_rl2:.6f}')


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

    exemplar_set = [[] for _ in range(args.task_num)]
    run_exp(args, task_list, action2task, classes_name, exemplar_set)


if __name__ == '__main__':
    start = time.time()
    main()
    print('time cost: ', time.time() - start)
