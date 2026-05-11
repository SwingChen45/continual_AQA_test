# coding=utf-8
import os
import pickle
import random
from pathlib import Path

import argparse
import numpy as np
import torch
import torch.utils.data as Data

from models.JRG_ASS_FisV import ASS_JRG
from models.MLP import MLP_block


# ========================================================
# Model builder
# ========================================================
def build_moodel(args):
    """
    Keep the original model builder for now.
    Later, when we add a Fis-V specific 3-modal model file,
    this function can branch on args.dataset_name == 'fisv'.
    """
    model_ = ASS_JRG(
        whole_size=args.whole_size,
        patch_size=args.patch_size,
        seg_num=args.seg_num,
        joint_num=args.joint_num,
        out_dim=1,
        save_graph=args.save_graph,
        mode=args.mode,
        G_E_graph=args.g_e_graph,
        alpha=args.alpha,
        task_num=args.num_tasks if args.dataset_name.lower() == 'fisv' else 6,
    )
    score_rgs = MLP_block(512, 1)
    diff_rgs = MLP_block(1024, 1)
    return model_, score_rgs, diff_rgs


def load_ckpt():
    return


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ========================================================
# Generic helpers
# ========================================================
def ensure_exists(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return str(path)


def load_pkl(path_str: str):
    path_str = ensure_exists(path_str)
    with open(path_str, "rb") as f:
        return pickle.load(f)


def normalize_scores_with_train_range(train_scores_full, scores_to_norm):
    train_scores_full = np.asarray(train_scores_full, dtype=np.float32)
    scores_to_norm = np.asarray(scores_to_norm, dtype=np.float32)

    train_min = float(train_scores_full.min())
    train_max = float(train_scores_full.max())

    if abs(train_max - train_min) < 1e-12:
        return scores_to_norm.copy()

    return (scores_to_norm - train_min) / (train_max - train_min) * 10.0


def crop_to_len(x: np.ndarray, target_len: int):
    """
    x shape: [N, T, D]
    Crop along temporal dimension.
    """
    if x.shape[1] == target_len:
        return x
    if x.shape[1] < target_len:
        raise ValueError(
            f"Target length {target_len} is larger than input temporal length {x.shape[1]}"
        )
    return x[:, :target_len, :]


# ========================================================
# AQA-7 original branch
# ========================================================
def load_data_aqa7(data_root, set_id, args, exemplar_set=None):
    feat_file = os.path.join(
        data_root,
        'AQA_pytorch_' + 'kinetics' + '_' + 'swind' + '_Set_' + str(set_id + 1) + '_Feats.npz'
    )

    all_dict = np.load(feat_file)
    print('AQA-7 load finished')

    train_whole = np.concatenate(
        (all_dict['train_rgb'][:, :, 0, :], all_dict['train_flow'][:, :, 0, :]),
        axis=2
    )
    train_patch = np.concatenate(
        (
            all_dict['train_rgb'][:, :, 1::, :].transpose((0, 2, 1, 3)),
            all_dict['train_flow'][:, :, 1::, :].transpose((0, 2, 1, 3))
        ),
        axis=3
    )
    test_whole = np.concatenate(
        (all_dict['test_rgb'][:, :, 0, :], all_dict['test_flow'][:, :, 0, :]),
        axis=2
    )
    test_patch = np.concatenate(
        (
            all_dict['test_rgb'][:, :, 1::, :].transpose((0, 2, 1, 3)),
            all_dict['test_flow'][:, :, 1::, :].transpose((0, 2, 1, 3))
        ),
        axis=3
    )

    train_scores_ = all_dict['train_label']
    train_scores = np.repeat(train_scores_, 2)
    test_scores = all_dict['test_label']

    # Score Normalize
    train_max = np.max(train_scores)
    train_min = np.min(train_scores)
    train_scores = (train_scores - train_min) / (train_max - train_min) * 10.0
    test_scores = (test_scores - train_min) / (train_max - train_min) * 10.0

    train_action_name = [set_id for _ in range(train_scores.shape[0])]
    test_action_name = [set_id for _ in range(test_scores.shape[0])]

    train_whole_with_memory = torch.tensor(train_whole).float()
    train_patch_with_memory = torch.tensor(train_patch).float()
    train_scores_with_memory = torch.tensor(train_scores).float()
    train_action_name_with_memory = train_action_name

    if (exemplar_set is not None) and (args.dataset_mixup):
        if 'debug' in args.exp_name:
            print('dataset_mixup')
        for a in range(len(exemplar_set)):
            if len(exemplar_set[a]) == 0:
                continue
            for exemplar in exemplar_set[a]:
                train_whole_with_memory = torch.cat(
                    (train_whole_with_memory, torch.tensor(exemplar[0]).float().unsqueeze(0)),
                    dim=0
                )
                train_patch_with_memory = torch.cat(
                    (train_patch_with_memory, torch.tensor(exemplar[1]).float().unsqueeze(0)),
                    dim=0
                )
                train_scores_with_memory = torch.cat(
                    (train_scores_with_memory, torch.tensor(exemplar[2]).float().unsqueeze(0)),
                    dim=0
                )
                train_action_name_with_memory += [a]

    g = torch.Generator()
    g.manual_seed(args.seed)

    dataset_train = Data.TensorDataset(
        train_whole_with_memory,
        train_patch_with_memory,
        train_scores_with_memory,
        torch.tensor(np.array(train_action_name_with_memory))
    )
    dataset_test = Data.TensorDataset(
        torch.tensor(test_whole).float(),
        torch.tensor(test_patch).float(),
        torch.tensor(test_scores).float(),
        torch.tensor(np.array(test_action_name))
    )

    loader_train = Data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        generator=g
    )
    loader_test = Data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    purely_task_data = (train_whole, train_patch, train_scores)
    print('AQA-7 pack finished')
    return dataset_train, dataset_test, loader_train, loader_test, purely_task_data


# ========================================================
# Fis-V helpers
# ========================================================
def resolve_fisv_paths(args):
    fisv_root = Path(args.fisv_root)

    score_type = args.score_type.lower()

    # 你的 split 文件放在这个子目录里
    split_root = fisv_root / "balance_5tasks"

    train_split = args.fisv_train_split
    test_split = args.fisv_test_split

    if train_split == "":
        train_split = split_root / f"fisv_{score_type}_train_split_5tasks.pkl"
    else:
        train_split = Path(train_split)

    if test_split == "":
        test_split = split_root / f"fisv_{score_type}_test_split_5tasks.pkl"
    else:
        test_split = Path(test_split)

    return {
        "rgb_train": str(fisv_root / "fisv_rgbvst_train.npy"),
        "rgb_test": str(fisv_root / "fisv_rgbvst_test.npy"),
        "flow_train": str(fisv_root / "fisv_flow_train.npy"),
        "flow_test": str(fisv_root / "fisv_flow_test.npy"),
        "skel_train": str(fisv_root / "FISV_stgcn_train.npy"),
        "skel_test": str(fisv_root / "FISV_stgcn_test.npy"),
        "train_split": str(train_split),
        "test_split": str(test_split),
    }


def validate_split_dict(split_dict, name="split"):
    required_keys = ["ids", "scores", "task_ids"]
    for k in required_keys:
        if k not in split_dict:
            raise KeyError(f"{name} missing required key: {k}")


def load_fisv_split(split_path):
    split_dict = load_pkl(split_path)
    if not isinstance(split_dict, dict):
        raise ValueError(f"Fis-V split file must be a dict, got {type(split_dict)} from {split_path}")
    validate_split_dict(split_dict, split_path)
    return split_dict


def select_indices_by_task(split_dict, set_id):
    task_ids = np.asarray(split_dict["task_ids"], dtype=np.int64)
    indices = np.where(task_ids == int(set_id))[0]
    return indices


def maybe_dataset_mixup_fisv(
    train_rgb_with_memory,
    train_flow_with_memory,
    train_skel_with_memory,
    train_scores_with_memory,
    train_action_name_with_memory,
    exemplar_set,
    args
):
    """
    For Fis-V, each exemplar is expected to be:
    [rgb_feat, flow_feat, skel_feat, score]
    """
    if (exemplar_set is None) or (not args.dataset_mixup):
        return (
            train_rgb_with_memory,
            train_flow_with_memory,
            train_skel_with_memory,
            train_scores_with_memory,
            train_action_name_with_memory,
        )

    if 'debug' in args.exp_name:
        print('Fis-V dataset_mixup')

    for a in range(len(exemplar_set)):
        if len(exemplar_set[a]) == 0:
            continue
        for exemplar in exemplar_set[a]:
            train_rgb_with_memory = torch.cat(
                (train_rgb_with_memory, torch.tensor(exemplar[0]).float().unsqueeze(0)),
                dim=0
            )
            train_flow_with_memory = torch.cat(
                (train_flow_with_memory, torch.tensor(exemplar[1]).float().unsqueeze(0)),
                dim=0
            )
            train_skel_with_memory = torch.cat(
                (train_skel_with_memory, torch.tensor(exemplar[2]).float().unsqueeze(0)),
                dim=0
            )
            train_scores_with_memory = torch.cat(
                (train_scores_with_memory, torch.tensor(exemplar[3]).float().unsqueeze(0)),
                dim=0
            )
            train_action_name_with_memory += [a]

    return (
        train_rgb_with_memory,
        train_flow_with_memory,
        train_skel_with_memory,
        train_scores_with_memory,
        train_action_name_with_memory,
    )


def load_data_fisv(data_root, set_id, args, exemplar_set=None):
    """
    Fis-V branch:
    - set_id means score-level task id: 0..4
    - returns 3-modal features: rgb / flow / skel
    """

    paths = resolve_fisv_paths(args)

    for _, p in paths.items():
        ensure_exists(p)

    train_split = load_fisv_split(paths["train_split"])
    test_split = load_fisv_split(paths["test_split"])

    rgb_train_all = np.load(paths["rgb_train"], mmap_mode="r")
    rgb_test_all = np.load(paths["rgb_test"], mmap_mode="r")
    flow_train_all = np.load(paths["flow_train"], mmap_mode="r")
    flow_test_all = np.load(paths["flow_test"], mmap_mode="r")
    skel_train_all = np.load(paths["skel_train"], mmap_mode="r")
    skel_test_all = np.load(paths["skel_test"], mmap_mode="r")

    # Select indices for the current task
    train_idx = select_indices_by_task(train_split, set_id)
    test_idx = select_indices_by_task(test_split, set_id)

    if len(train_idx) == 0:
        raise ValueError(f"No training samples found for Fis-V task {set_id}")
    if len(test_idx) == 0:
        raise ValueError(f"No testing samples found for Fis-V task {set_id}")

    # Read selected samples
    train_rgb = np.asarray(rgb_train_all[train_idx], dtype=np.float32)
    train_flow = np.asarray(flow_train_all[train_idx], dtype=np.float32)
    train_skel = np.asarray(skel_train_all[train_idx], dtype=np.float32)

    test_rgb = np.asarray(rgb_test_all[test_idx], dtype=np.float32)
    test_flow = np.asarray(flow_test_all[test_idx], dtype=np.float32)
    test_skel = np.asarray(skel_test_all[test_idx], dtype=np.float32)

    # Online temporal alignment:
    # based on current analysis, train is already 138 for all modalities;
    # test has rgb/skel=140 and flow=138, so crop to target_len=138 by default.
    target_len = args.fisv_seq_len
    train_rgb = crop_to_len(train_rgb, target_len)
    train_flow = crop_to_len(train_flow, target_len)
    train_skel = crop_to_len(train_skel, target_len)
    test_rgb = crop_to_len(test_rgb, target_len)
    test_flow = crop_to_len(test_flow, target_len)
    test_skel = crop_to_len(test_skel, target_len)

    # Scores
    train_scores_full = np.asarray(train_split["scores"], dtype=np.float32)
    test_scores_full = np.asarray(test_split["scores"], dtype=np.float32)

    train_scores = train_scores_full[train_idx]
    test_scores = test_scores_full[test_idx]

    # Normalize using full TRAIN split range of the selected score type
    train_scores = normalize_scores_with_train_range(train_scores_full, train_scores)
    test_scores = normalize_scores_with_train_range(train_scores_full, test_scores)

    train_action_name = [set_id for _ in range(len(train_scores))]
    test_action_name = [set_id for _ in range(len(test_scores))]

    train_rgb_with_memory = torch.tensor(train_rgb).float()
    train_flow_with_memory = torch.tensor(train_flow).float()
    train_skel_with_memory = torch.tensor(train_skel).float()
    train_scores_with_memory = torch.tensor(train_scores).float()
    train_action_name_with_memory = train_action_name

    (
        train_rgb_with_memory,
        train_flow_with_memory,
        train_skel_with_memory,
        train_scores_with_memory,
        train_action_name_with_memory,
    ) = maybe_dataset_mixup_fisv(
        train_rgb_with_memory,
        train_flow_with_memory,
        train_skel_with_memory,
        train_scores_with_memory,
        train_action_name_with_memory,
        exemplar_set,
        args
    )

    g = torch.Generator()
    g.manual_seed(args.seed)

    dataset_train = Data.TensorDataset(
        train_rgb_with_memory,
        train_flow_with_memory,
        train_skel_with_memory,
        train_scores_with_memory,
        torch.tensor(np.array(train_action_name_with_memory))
    )
    dataset_test = Data.TensorDataset(
        torch.tensor(test_rgb).float(),
        torch.tensor(test_flow).float(),
        torch.tensor(test_skel).float(),
        torch.tensor(test_scores).float(),
        torch.tensor(np.array(test_action_name))
    )

    loader_train = Data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        generator=g
    )
    loader_test = Data.DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    purely_task_data = (
        train_rgb,
        train_flow,
        train_skel,
        train_scores,
        np.asarray(train_split["ids"])[train_idx],
    )

    print(f'Fis-V pack finished | score_type={args.score_type} | task={set_id} | '
          f'train={len(train_idx)} | test={len(test_idx)} | seq_len={target_len}')
    return dataset_train, dataset_test, loader_train, loader_test, purely_task_data


# ========================================================
# Unified data entry
# ========================================================
def load_data(data_root, set_id, args, exemplar_set=None):
    if args.dataset_name.lower() == 'fisv':
        return load_data_fisv(data_root, set_id, args, exemplar_set)
    return load_data_aqa7(data_root, set_id, args, exemplar_set)


# ========================================================
# Fis-V helper loaders for exemplar/herding
# ========================================================
def load_data_from_list(data_list=None, cls_label=0, dataset_name='aqa7'):
    if dataset_name.lower() == 'fisv':
        train_rgb_with_memory = torch.Tensor([])
        train_flow_with_memory = torch.Tensor([])
        train_skel_with_memory = torch.Tensor([])
        train_scores_with_memory = torch.Tensor([])
        train_action_name_with_memory = []
        data_idx = []

        if data_list is not None:
            for idx, exemplar in enumerate(data_list):
                train_rgb_with_memory = torch.cat(
                    (train_rgb_with_memory, torch.tensor(exemplar[0]).float().unsqueeze(0)), dim=0
                )
                train_flow_with_memory = torch.cat(
                    (train_flow_with_memory, torch.tensor(exemplar[1]).float().unsqueeze(0)), dim=0
                )
                train_skel_with_memory = torch.cat(
                    (train_skel_with_memory, torch.tensor(exemplar[2]).float().unsqueeze(0)), dim=0
                )
                train_scores_with_memory = torch.cat(
                    (train_scores_with_memory, torch.tensor(exemplar[3]).float().unsqueeze(0)), dim=0
                )
                train_action_name_with_memory += [cls_label]
                data_idx += [idx]

        dataset_train = Data.TensorDataset(
            train_rgb_with_memory,
            train_flow_with_memory,
            train_skel_with_memory,
            train_scores_with_memory,
            torch.tensor(np.array(train_action_name_with_memory)),
            torch.tensor(np.array(data_idx))
        )
        loader_train = Data.DataLoader(dataset=dataset_train, batch_size=16, shuffle=False, num_workers=0)
        print('Fis-V load_data_from_list pack finished')
        return loader_train

    # original AQA-7 branch
    train_whole_with_memory = torch.Tensor([])
    train_patch_with_memory = torch.Tensor([])
    train_scores_with_memory = torch.Tensor([])
    train_action_name_with_memory = []
    data_idx = []

    if data_list is not None:
        for idx, exemplar in enumerate(data_list):
            train_whole_with_memory = torch.cat(
                (train_whole_with_memory, torch.tensor(exemplar[0]).float().unsqueeze(0)), dim=0
            )
            train_patch_with_memory = torch.cat(
                (train_patch_with_memory, torch.tensor(exemplar[1]).float().unsqueeze(0)), dim=0
            )
            train_scores_with_memory = torch.cat(
                (train_scores_with_memory, torch.tensor(exemplar[2]).float().unsqueeze(0)), dim=0
            )
            train_action_name_with_memory += [cls_label]
            data_idx += [idx]

    dataset_train = Data.TensorDataset(
        train_whole_with_memory,
        train_patch_with_memory,
        train_scores_with_memory,
        torch.tensor(np.array(train_action_name_with_memory)),
        torch.tensor(np.array(data_idx))
    )
    loader_train = Data.DataLoader(dataset=dataset_train, batch_size=16, shuffle=False, num_workers=0)
    print('AQA-7 load_data_from_list pack finished')
    return loader_train


def load_exemplar_data(exemplar_set=[], dataset_name='aqa7'):
    if dataset_name.lower() == 'fisv':
        train_rgb_with_memory = torch.Tensor([])
        train_flow_with_memory = torch.Tensor([])
        train_skel_with_memory = torch.Tensor([])
        train_scores_with_memory = torch.Tensor([])
        train_action_name_with_memory = []

        if exemplar_set is not None:
            for a in range(len(exemplar_set)):
                if len(exemplar_set[a]) == 0:
                    continue
                for exemplar in exemplar_set[a]:
                    train_rgb_with_memory = torch.cat(
                        (train_rgb_with_memory, torch.tensor(exemplar[0]).float().unsqueeze(0)), dim=0
                    )
                    train_flow_with_memory = torch.cat(
                        (train_flow_with_memory, torch.tensor(exemplar[1]).float().unsqueeze(0)), dim=0
                    )
                    train_skel_with_memory = torch.cat(
                        (train_skel_with_memory, torch.tensor(exemplar[2]).float().unsqueeze(0)), dim=0
                    )
                    train_scores_with_memory = torch.cat(
                        (train_scores_with_memory, torch.tensor(exemplar[3]).float().unsqueeze(0)), dim=0
                    )
                    train_action_name_with_memory += [a]

        dataset_train = Data.TensorDataset(
            train_rgb_with_memory,
            train_flow_with_memory,
            train_skel_with_memory,
            train_scores_with_memory,
            torch.tensor(np.array(train_action_name_with_memory))
        )
        loader_train = Data.DataLoader(
            dataset=dataset_train, batch_size=16, shuffle=True, num_workers=2, drop_last=True
        )
        print('Fis-V load_exemplar_data pack finished')
        return loader_train

    # original AQA-7 branch
    train_whole_with_memory = torch.Tensor([])
    train_patch_with_memory = torch.Tensor([])
    train_scores_with_memory = torch.Tensor([])
    train_action_name_with_memory = []

    if exemplar_set is not None:
        for a in range(len(exemplar_set)):
            if len(exemplar_set[a]) == 0:
                continue
            for exemplar in exemplar_set[a]:
                train_whole_with_memory = torch.cat(
                    (train_whole_with_memory, torch.tensor(exemplar[0]).float().unsqueeze(0)), dim=0
                )
                train_patch_with_memory = torch.cat(
                    (train_patch_with_memory, torch.tensor(exemplar[1]).float().unsqueeze(0)), dim=0
                )
                train_scores_with_memory = torch.cat(
                    (train_scores_with_memory, torch.tensor(exemplar[2]).float().unsqueeze(0)), dim=0
                )
                train_action_name_with_memory += [a]

    dataset_train = Data.TensorDataset(
        train_whole_with_memory,
        train_patch_with_memory,
        train_scores_with_memory,
        torch.tensor(np.array(train_action_name_with_memory))
    )
    loader_train = Data.DataLoader(
        dataset=dataset_train, batch_size=16, shuffle=True, num_workers=2, drop_last=True
    )
    print('AQA-7 load_exemplar_data pack finished')
    return loader_train


# ========================================================
# Parser
# ========================================================
def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='2', help='id of gpu device(s) to be used')
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # dataset switch
    parser.add_argument('--dataset_name', type=str, default='aqa7', choices=['aqa7', 'fisv'])
    parser.add_argument('--score_type', type=str, default='TES', choices=['TES', 'PCS'])
    parser.add_argument('--num_tasks', type=int, default=5, help='number of sequential tasks for Fis-V')

    # training switches
    parser.add_argument('--lr_decay', action='store_true', default=False, help='Learning rate decay')
    parser.add_argument('--graph_distill', action='store_true', default=False, help='graph distillation')
    parser.add_argument('--replay', action='store_true', default=False, help='replay')
    parser.add_argument('--pod_loss', action='store_true', default=False, help='pod loss')
    parser.add_argument('--diff_loss', action='store_true', default=False, help='difference loss')
    parser.add_argument('--aug_rgs', action='store_true', default=False, help='aug-rgs')
    parser.add_argument('--save_ckpt', action='store_true', default=False, help='save ckpt')
    parser.add_argument('--save_graph', action='store_true', default=False, help='save graph')
    parser.add_argument('--aug_w_weight', action='store_true', default=False, help='augmentation with weight')
    parser.add_argument('--g_e_graph', action='store_true', default=False, help='general expert graph')
    parser.add_argument('--graph_visualization', action='store_true', default=False, help='visualize graphs')
    parser.add_argument('--graph_random_init', action='store_true', default=False, help='randomly initialize graphs')

    parser.add_argument(
        '--dataset_mixup',
        action='store_true',
        default=False,
        help='If set True, previous data will mix with current data during training.'
    )

    parser.add_argument(
        '--approach',
        type=str,
        default='finetune',
        choices=['finetune', 'distill', 'lwf', 'group_replay', 'random_replay',
                 'herding_replay', 'podnet', 'aug-diff', 'e_graph', 'g_e_graph']
    )
    parser.add_argument(
        '--aug_approach',
        type=str,
        default='none',
        choices=['none', 'p-distill', 'aug-diff', 'aug-rgs']
    )
    parser.add_argument(
        '--aug_mode',
        type=str,
        default='fs_aug',
        choices=['f_aug', 's_aug', 'fs_aug'],
        help='augmentation setting'
    )
    parser.add_argument(
        '--fix_graph_mode',
        type=str,
        default='fix_old',
        choices=['fix_old', 'fix_new', 'no_fix', 'all_fix']
    )
    parser.add_argument(
        '--optim_mode',
        type=str,
        default='default',
        choices=['default', 'new_optim']
    )
    parser.add_argument(
        '--replay_method',
        type=str,
        default='random',
        choices=['random', 'herding', 'group_replay']
    )

    parser.add_argument('--visualization_schedule', type=int, default=10, help='visualization schedule')
    parser.add_argument('--num_helpers', type=int, default=3, help='number of helpers')
    parser.add_argument('--aug_scale', type=float, default=0.3, help='augmentation scale')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay')
    parser.add_argument('--base_lr', type=float, default=0.001, help='basic learning rate')
    parser.add_argument('--lr_factor', type=float, default=1, help='lr factor')
    parser.add_argument('--memory_size', type=int, default=60, help='memory size')
    parser.add_argument('--alpha', type=float, default=0.5, help='weight for GE graph')
    parser.add_argument('--lambda_distill', type=float, default=7, help='lambda distill')
    parser.add_argument('--lambda_diff', type=float, default=1, help='lambda diff')
    parser.add_argument('--lambda_graph_distill', type=float, default=7, help='lambda graph distill')
    parser.add_argument('--lambda_pod', type=float, default=1, help='lambda pod')
    parser.add_argument('--optim_id', type=int, default=1, help='optimizer id')

    parser.add_argument('--num_epochs', type=int, default=400, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=32, help='number of subprocesses for dataloader')
    parser.add_argument('--batch-size', type=int, default=32)

    # AQA-7 paths
    parser.add_argument('--data_root', type=str, default='/home/administrator/exp--fs-aug/Continual-AQA-main/DATA/AQA-7')

    # Fis-V paths
    parser.add_argument('--fisv_root', type=str, default='/home/administrator/exp--fs-aug/Continual-AQA-main/DATA/FisV')
    parser.add_argument('--fisv_train_split', type=str, default='')
    parser.add_argument('--fisv_test_split', type=str, default='')
    parser.add_argument('--fisv_seq_len', type=int, default=138, help='target temporal length for Fis-V')

    parser.add_argument('--ckpt_root', type=str, default='./ckpt/')
    parser.add_argument('--ckpt_path', type=str, default='')

    # model args
    parser.add_argument('--seg-num', type=int, default=12, help='number of video segments')
    parser.add_argument('--joint-num', type=int, default=17, help='number of human joints')
    parser.add_argument('--whole-size', type=int, default=800, help='I3D feat size')
    parser.add_argument('--patch-size', type=int, default=800, help='I3D feat size')
    parser.add_argument('--model_name', type=str, default='I3D_MLP', help='name of model')

    parser.add_argument(
        "--pretrained_i3d_weight",
        type=str,
        default='../pretrained_models/i3d_model_rgb.pth',
        help='pretrained i3d model'
    )
    parser.add_argument('-mode', type=str, default='single-head', choices=['single-head', 'multi-head'])
    return parser


def check_args_valid(args):
    if args.aug_approach != 'none':
        assert not args.dataset_mixup
    if args.aug_rgs:
        assert not args.diff_loss
    return


def build_exp():
    args = get_parser().parse_args()
    check_args_valid(args)
    print('gpu: ', args.gpu)

    if args.dataset_name.lower() == 'fisv':
        classes_name = [f'{args.score_type.lower()}_task_{i + 1}' for i in range(args.num_tasks)]
        task_list = [i for i in range(args.num_tasks)]
        action2task = [i for i in range(args.num_tasks)]
        print('Fis-V task list: ', task_list)
        print('Fis-V score type: ', args.score_type)
    else:
        classes_name = ['diving', 'gym_vault', 'ski_big_air',
                        'snowboard_big_air', 'sync_diving_3m', 'sync_diving_10m']
        seed = args.seed
        task_list_bank = [
            [5, 2, 1, 3, 0, 4],
            [2, 1, 4, 0, 3, 5],
            [4, 1, 3, 2, 5, 0],
            [3, 5, 4, 1, 0, 2]
        ]
        task_list = task_list_bank[seed]
        action2task = [0] * 6
        for i in range(6):
            action2task[task_list[i]] = i
        print('AQA-7 task list: ', task_list)

    if not os.path.isdir(os.path.join('./ckpt', args.exp_name)):
        os.makedirs(os.path.join('./ckpt', args.exp_name))

    args_dict = args.__dict__
    with open(os.path.join('./ckpt', args.exp_name, 'config.yaml'), 'w') as f:
        for each_arg, value in args_dict.items():
            f.writelines(each_arg + ' : ' + str(value) + '\n')

    return args, task_list, action2task, classes_name