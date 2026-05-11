import os
import pickle
import argparse
import random
import numpy as np
import torch
import torch.utils.data as Data

from models.GSHG_HyperGCN import ASS_GSHG
from models.MLP import MLP_block


ACTION_NAMES = ['ball', 'club', 'hoop', 'ribbon']
ACTION_ALIASES = {
    'ball': ['ball'],
    'club': ['club', 'clubs'],
    'hoop': ['hoop'],
    'ribbon': ['ribbon'],
}


def build_moodel(args):
    model_ = ASS_GSHG(
        patch_size=args.patch_size,
        seg_num=args.seg_num,
        joint_num=args.joint_num,
        out_dim=1,
        save_graph=args.save_graph,
        mode=args.mode,
        G_E_graph=args.g_e_graph,
        alpha=args.alpha,
        task_num=args.task_num,
        hyper_joints=args.hyper_joints,
        num_subset=args.num_subset,
        rel_reduction=args.rel_reduction,
        gshg_beta=args.gshg_beta,
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


def _resolve_action_name(set_id):
    if isinstance(set_id, str):
        return set_id.lower()
    return ACTION_NAMES[int(set_id)]


def _resolve_existing_path(path):
    if os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alt = os.path.join(repo_root, path)
    if os.path.exists(alt):
        return alt
    alt = os.path.join(repo_root, path.lstrip('./'))
    if os.path.exists(alt):
        return alt
    return path


def _find_file(data_root, keywords):
    keys = [k.lower() for k in keywords]
    hits = []
    for name in os.listdir(data_root):
        lower = name.lower()
        if all(k in lower for k in keys):
            hits.append(os.path.join(data_root, name))
    if not hits:
        raise FileNotFoundError('Cannot find file in {} with keywords {}'.format(data_root, keywords))
    return sorted(hits)[0]


def _load_label_pkl(file_path):
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    if not isinstance(obj, (tuple, list)) or len(obj) != 2:
        raise ValueError('Unexpected label pkl format: {}'.format(file_path))
    sample_ids = [str(x) for x in obj[0]]
    scores = np.asarray(obj[1], dtype=np.float32)
    return sample_ids, scores


def _load_rg_joint_split(joint_root, action_name, split='train'):
    joint_root = _resolve_existing_path(joint_root)
    joint_dir = os.path.join(joint_root, action_name)
    if not os.path.isdir(joint_dir):
        raise FileNotFoundError(joint_dir)

    joint_path = label_path = None
    for alias in ACTION_ALIASES[action_name]:
        try:
            if joint_path is None:
                joint_path = _find_file(joint_dir, [alias, 'stgcn', 'joint', split])
        except FileNotFoundError:
            pass
        try:
            if label_path is None:
                label_path = _find_file(joint_dir, [alias, 'label', split])
        except FileNotFoundError:
            pass

    if any(x is None for x in [joint_path, label_path]):
        raise FileNotFoundError(
            'Incomplete RG joint-only files for action={}, split={}: joint={}, label={}'.format(
                action_name, split, joint_path, label_path
            )
        )

    joint = np.load(joint_path).astype(np.float32)
    sample_ids, scores = _load_label_pkl(label_path)

    n = len(scores)
    if len(joint) != n:
        raise ValueError(
            'Size mismatch in {}-{}: joint={}, labels={}'.format(
                action_name, split, len(joint), n
            )
        )
    if joint.ndim != 4:
        raise ValueError('Expected joint feature [N,T,V,D], got {} from {}'.format(joint.shape, joint_path))

    return {
        'joint': joint,
        'scores': scores,
        'sample_ids': sample_ids,
    }


def load_data(data_root, set_id, args, exemplar_set=None):
    action_name = _resolve_action_name(set_id)
    train_pack = _load_rg_joint_split(args.joint_root, action_name, split='train')
    test_pack = _load_rg_joint_split(args.joint_root, action_name, split='test')

    train_joint = train_pack['joint']
    train_scores = train_pack['scores'].copy()
    test_joint = test_pack['joint']
    test_scores = test_pack['scores'].copy()

    train_max = np.max(train_scores)
    train_min = np.min(train_scores)
    train_scores = (train_scores - train_min) / (train_max - train_min + 1e-8) * 10.0
    test_scores = (test_scores - train_min) / (train_max - train_min + 1e-8) * 10.0

    train_joint_with_memory = torch.tensor(train_joint).float()
    train_scores_with_memory = torch.tensor(train_scores).float()
    train_action_name_with_memory = [set_id for _ in range(train_scores.shape[0])]

    if (exemplar_set is not None) and (args.dataset_mixup):
        for a in range(len(exemplar_set)):
            if len(exemplar_set[a]) == 0:
                continue
            for exemplar in exemplar_set[a]:
                train_joint_with_memory = torch.cat((train_joint_with_memory, torch.tensor(exemplar[0]).float().unsqueeze(0)), dim=0)
                train_scores_with_memory = torch.cat((train_scores_with_memory, torch.tensor(exemplar[1]).float().unsqueeze(0)), dim=0)
                train_action_name_with_memory += [a]

    g = torch.Generator()
    g.manual_seed(args.seed)
    dataset_train = Data.TensorDataset(
        train_joint_with_memory,
        train_scores_with_memory,
        torch.tensor(np.array(train_action_name_with_memory)),
    )
    dataset_test = Data.TensorDataset(
        torch.tensor(test_joint).float(),
        torch.tensor(test_scores).float(),
        torch.tensor(np.array([set_id for _ in range(test_scores.shape[0])])),
    )
    loader_train = Data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    loader_test = Data.DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=2)
    return dataset_train, dataset_test, loader_train, loader_test, (train_joint, train_scores)


def load_data_from_list(data_list=None, cls_label=0):
    train_joint = torch.Tensor([])
    train_scores = torch.Tensor([])
    train_action_name = []
    data_idx = []
    if data_list is not None:
        for idx, exemplar in enumerate(data_list):
            train_joint = torch.cat((train_joint, torch.tensor(exemplar[0]).float().unsqueeze(0)), dim=0)
            train_scores = torch.cat((train_scores, torch.tensor(exemplar[1]).float().unsqueeze(0)), dim=0)
            train_action_name += [cls_label]
            data_idx += [idx]

    dataset_train = Data.TensorDataset(
        train_joint,
        train_scores,
        torch.tensor(np.array(train_action_name)),
        torch.tensor(np.array(data_idx)),
    )
    loader_train = Data.DataLoader(dataset=dataset_train, batch_size=16, shuffle=False, num_workers=0)
    return loader_train


def load_exemplar_data(exemplar_set=None):
    train_joint = torch.Tensor([])
    train_scores = torch.Tensor([])
    train_action_name = []
    exemplar_set = exemplar_set or []
    for a in range(len(exemplar_set)):
        if len(exemplar_set[a]) == 0:
            continue
        for exemplar in exemplar_set[a]:
            train_joint = torch.cat((train_joint, torch.tensor(exemplar[0]).float().unsqueeze(0)), dim=0)
            train_scores = torch.cat((train_scores, torch.tensor(exemplar[1]).float().unsqueeze(0)), dim=0)
            train_action_name += [a]
    dataset_train = Data.TensorDataset(
        train_joint,
        train_scores,
        torch.tensor(np.array(train_action_name)),
    )
    loader_train = Data.DataLoader(dataset=dataset_train, batch_size=16, shuffle=True, num_workers=2, drop_last=True)
    return loader_train


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--exp_name', type=str, default='debug')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--lr_decay', action='store_true', default=False)
    parser.add_argument('--graph_distill', action='store_true', default=False)
    parser.add_argument('--replay', action='store_true', default=False)
    parser.add_argument('--pod_loss', action='store_true', default=False)
    parser.add_argument('--diff_loss', action='store_true', default=False)
    parser.add_argument('--aug_rgs', action='store_true', default=False)
    parser.add_argument('--save_ckpt', action='store_true', default=False)
    parser.add_argument('--save_graph', action='store_true', default=False)
    parser.add_argument('--aug_w_weight', action='store_true', default=False)
    parser.add_argument('--g_e_graph', action='store_true', default=False)
    parser.add_argument('--graph_visualization', action='store_true', default=False)
    parser.add_argument('--graph_random_init', action='store_true', default=False)
    parser.add_argument('--dataset_mixup', action='store_true', default=False)

    parser.add_argument('--approach', type=str, default='finetune',
                        choices=['finetune', 'distill', 'lwf', 'group_replay', 'random_replay', 'herding_replay', 'podnet', 'aug-diff', 'e_graph', 'g_e_graph'])
    parser.add_argument('--aug_approach', type=str, default='none',
                        choices=['none', 'p-distill', 'aug-diff', 'aug-rgs'])
    parser.add_argument('--aug_mode', type=str, default='fs_aug', choices=['f_aug', 's_aug', 'fs_aug'])
    parser.add_argument('--fix_graph_mode', type=str, default='fix_old', choices=['fix_old', 'fix_new', 'no_fix', 'all_fix'])
    parser.add_argument('--optim_mode', type=str, default='default', choices=['default', 'new_optim'])
    parser.add_argument('--replay_method', type=str, default='random', choices=['random', 'herding', 'group_replay'])

    parser.add_argument('--visualization_schedule', type=int, default=10)
    parser.add_argument('--num_helpers', type=int, default=3)
    parser.add_argument('--aug_scale', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--lr_factor', type=float, default=1)
    parser.add_argument('--memory_size', type=int, default=60)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lambda_distill', type=float, default=7)
    parser.add_argument('--lambda_diff', type=float, default=1)
    parser.add_argument('--lambda_graph_distill', type=float, default=7)
    parser.add_argument('--lambda_pod', type=float, default=1)
    parser.add_argument('--optim_id', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--joint_root', type=str, default='./DATA/RG_stgcn_joint_features')
    parser.add_argument('--ckpt_root', type=str, default='./ckpt/')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--seg-num', type=int, default=72)
    parser.add_argument('--joint-num', type=int, default=18)
    parser.add_argument('--patch-size', type=int, default=256)
    parser.add_argument('--task-num', type=int, default=4)
    parser.add_argument('--hyper-joints', type=int, default=3)
    parser.add_argument('--num-subset', type=int, default=8)
    parser.add_argument('--rel-reduction', type=int, default=4)
    parser.add_argument('--gshg-beta', type=float, default=0.8)
    parser.add_argument('--model_name', type=str, default='RG_GSHG_HYPERGCN_TEST')
    parser.add_argument('-mode', type=str, default='single-head', choices=['single-head', 'multi-head'])
    return parser


def check_args_valid(args):
    if args.aug_approach != 'none':
        assert (not args.dataset_mixup)
    if args.aug_rgs:
        assert (not args.diff_loss)
    return


def build_exp():
    args = get_parser().parse_args()
    check_args_valid(args)
    classes_name = ACTION_NAMES
    seed = args.seed % 4
    task_list_bank = [
        [0, 1, 2, 3],
        [1, 3, 0, 2],
        [2, 0, 3, 1],
        [3, 2, 1, 0],
    ]
    task_list = task_list_bank[seed]
    action2task = [0] * args.task_num
    for i in range(args.task_num):
        action2task[task_list[i]] = i
    if not os.path.isdir(os.path.join('./ckpt', args.exp_name)):
        os.makedirs(os.path.join('./ckpt', args.exp_name))
    with open(os.path.join('./ckpt', args.exp_name, 'config.yaml'), 'w') as f:
        for each_arg, value in args.__dict__.items():
            f.writelines(each_arg + ' : ' + str(value) + '\n')
    return args, task_list, action2task, classes_name
