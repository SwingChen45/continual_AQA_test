import torch.optim as optim


def _ctr_param_groups(model_, score_rgs, diff_rgs, args):
    m = model_.module
    graph_params = [
        m.general_spatial_mats,
        m.general_temporal_mats,
        m.spatial_mats,
        m.temporal_mats,
    ]
    graph_param_ids = {id(p) for p in graph_params}
    backbone_params = [p for p in model_.parameters() if p.requires_grad and id(p) not in graph_param_ids]
    return [
        {'params': graph_params, 'weight_decay': 0, 'lr': 0.01},
        {'params': backbone_params, 'weight_decay': args.weight_decay, 'lr': args.base_lr * args.lr_factor},
        {'params': score_rgs.parameters(), 'weight_decay': args.weight_decay, 'lr': args.base_lr},
        {'params': diff_rgs.parameters(), 'weight_decay': args.weight_decay, 'lr': args.base_lr},
    ]


def get_optim(model_, score_rgs, diff_rgs, args, optim_id=1):
    param_groups = _ctr_param_groups(model_, score_rgs, diff_rgs, args)

    if optim_id == 1:
        print('optim_1')
        return optim.Adam(param_groups, lr=args.base_lr)
    elif optim_id == 2:
        print('optim_2')
        return optim.SGD(param_groups, lr=args.base_lr, weight_decay=args.weight_decay)
    elif optim_id == 3:
        print('optim_3')
        return optim.Adam(param_groups, lr=args.base_lr, weight_decay=args.weight_decay)
    elif optim_id == 4:
        print('optim_4')
        return optim.SGD(param_groups, lr=args.base_lr)
    else:
        raise ValueError('Unknown optim_id: {}'.format(optim_id))
