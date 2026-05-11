import torch
import torch.optim as optim


def _maybe_add_param_group(param_groups, module_obj, attr_name, args, lr=None, weight_decay=None):
    """
    Add a parameter group if module_obj has this attribute.
    Works for both Parameter and nn.Module.
    """
    if not hasattr(module_obj, attr_name):
        return

    target = getattr(module_obj, attr_name)

    if isinstance(target, torch.nn.Parameter):
        params = [target]
    else:
        params = target.parameters()

    group = {"params": params}
    group["weight_decay"] = args.weight_decay if weight_decay is None else weight_decay
    if lr is not None:
        group["lr"] = lr
    param_groups.append(group)


def build_param_groups(model_, score_rgs, diff_rgs, args):
    """
    Build optimizer param groups that support both:
    - original AQA-7 JRG_ASS
    - adapted Fis-V JRG_ASS
    """
    m = model_.module if hasattr(model_, "module") else model_
    param_groups = []

    # ---- graph parameters: keep original higher lr setting ----
    _maybe_add_param_group(param_groups, m, "general_spatial_mats", args, lr=0.01, weight_decay=0.0)
    _maybe_add_param_group(param_groups, m, "general_temporal_mats", args, lr=0.01, weight_decay=0.0)
    _maybe_add_param_group(param_groups, m, "spatial_mats", args, lr=0.01, weight_decay=0.0)
    _maybe_add_param_group(param_groups, m, "temporal_mats", args, lr=0.01, weight_decay=0.0)
    _maybe_add_param_group(param_groups, m, "spatial_JCWs", args)
    _maybe_add_param_group(param_groups, m, "temporal_JCWs", args)

    # ---- original AQA-7 encoders ----
    _maybe_add_param_group(param_groups, m, "encoders_whole", args)
    _maybe_add_param_group(param_groups, m, "encoders_diffwhole", args)
    _maybe_add_param_group(param_groups, m, "encoders_comm0", args)
    _maybe_add_param_group(param_groups, m, "encoders_comm1", args)
    _maybe_add_param_group(param_groups, m, "encoders_diff0", args)
    _maybe_add_param_group(param_groups, m, "encoders_diff1", args)
    _maybe_add_param_group(param_groups, m, "regressor", args)
    _maybe_add_param_group(param_groups, m, "last_fuse", args)

    # ---- new Fis-V front-end modules ----
    _maybe_add_param_group(param_groups, m, "rgb_proj", args)
    _maybe_add_param_group(param_groups, m, "flow_proj", args)
    _maybe_add_param_group(param_groups, m, "skel_proj", args)
    _maybe_add_param_group(param_groups, m, "fisv_fuse", args)
    _maybe_add_param_group(param_groups, m, "token_generator", args)
    _maybe_add_param_group(param_groups, m, "fisv_whole_encoder", args)
    _maybe_add_param_group(param_groups, m, "fisv_diffwhole_encoder", args)
    _maybe_add_param_group(param_groups, m, "fisv_patch_encoder", args)
    _maybe_add_param_group(param_groups, m, "fisv_comm_encoder", args)
    _maybe_add_param_group(param_groups, m, "fisv_diff0_encoder", args)
    _maybe_add_param_group(param_groups, m, "fisv_diff1_encoder", args)
    _maybe_add_param_group(param_groups, m, "fisv_regressor", args)

    # ---- heads ----
    param_groups.append({"params": score_rgs.parameters(), "weight_decay": args.weight_decay})
    param_groups.append({"params": diff_rgs.parameters(), "weight_decay": args.weight_decay})

    return param_groups


def get_optim(model_, score_rgs, diff_rgs, args, optim_id=1):
    param_groups = build_param_groups(model_, score_rgs, diff_rgs, args)

    optimizer = optim.Adam(param_groups, lr=args.base_lr)

    optimizer2 = optim.SGD(
        [
            {'params': filter(lambda p: p.requires_grad, model_.parameters()), 'lr': args.base_lr * args.lr_factor},
            {'params': score_rgs.parameters()},
            {'params': diff_rgs.parameters()}
        ],
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )

    optimizer3 = optim.Adam(
        [
            {'params': filter(lambda p: p.requires_grad, model_.parameters()), 'lr': args.base_lr * args.lr_factor},
            {'params': score_rgs.parameters()},
            {'params': diff_rgs.parameters()}
        ],
        lr=args.base_lr,
        weight_decay=args.weight_decay
    )

    optimizer4 = optim.SGD(param_groups, lr=args.base_lr)

    if optim_id == 1:
        print('optim_1')
        return optimizer
    elif optim_id == 2:
        print('optim_2')
        return optimizer2
    elif optim_id == 3:
        print('optim_3')
        return optimizer3
    elif optim_id == 4:
        print('optim_4')
        return optimizer4
    else:
        raise ValueError(f"Unsupported optim_id: {optim_id}")