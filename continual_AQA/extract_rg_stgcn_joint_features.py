#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract per-joint ST-GCN features for RG *_average skeleton data.

The existing RG *_stgcn.npy files are global skeleton features shaped:
    [N, T, 256]

This script keeps the joint dimension before graph feature pooling and writes:
    [N, T, 18, 256]

These features are closer to the AQA-7 joint/patch-feature setup because every
OpenPose joint remains a graph node with its own learned feature vector.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


def add_stgcn_to_path(repo_root):
    stgcn_root = Path(repo_root) / "st-gcn-master"
    if not stgcn_root.is_dir():
        raise FileNotFoundError(f"Cannot find st-gcn-master at {stgcn_root}")
    sys.path.insert(0, str(stgcn_root))


def temporal_resample_stgcn_layout(x, seg_num):
    """Resample [N,C,T,V,M] to [N,C,seg_num,V,M]."""
    n, c, t, v, m = x.shape
    if t == seg_num:
        return x.astype(np.float32, copy=False)
    idx = np.linspace(0, t - 1, seg_num)
    left = np.floor(idx).astype(np.int64)
    right = np.ceil(idx).astype(np.int64)
    weight = (idx - left).astype(np.float32).reshape(1, 1, seg_num, 1, 1)
    return ((1.0 - weight) * x[:, :, left, :, :] + weight * x[:, :, right, :, :]).astype(np.float32)


def load_label(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, (tuple, list)) or len(obj) != 2:
        raise ValueError(f"Unsupported label format: {path}")
    ids = [str(x) for x in obj[0]]
    scores = np.asarray(obj[1], dtype=np.float32)
    return ids, scores


def load_stgcn_model(repo_root, weights, device):
    add_stgcn_to_path(repo_root)
    from net.st_gcn import Model

    model = Model(
        in_channels=3,
        num_class=400,
        edge_importance_weighting=True,
        graph_args={"layout": "openpose", "strategy": "spatial"},
    )

    if weights:
        state = torch.load(weights, map_location="cpu")
        model_state = model.state_dict()
        compatible = {}
        skipped = []
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                compatible[k] = v
            else:
                skipped.append(k)
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        print(f"Loaded {len(compatible)} tensors from {weights}")
        if skipped:
            print(f"Skipped incompatible tensors: {skipped}")
        if unexpected:
            print(f"Unexpected tensors: {unexpected}")
        if missing:
            print(f"Missing tensors: {missing}")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_backbone_joint_feature(model, x):
    """
    Run ST-GCN backbone and return [N,T,V,256].

    Input x is [N,C,T,V,M]. This mirrors net/st_gcn.py but stops before
    spatial-temporal global average pooling and fcn classification.
    """
    n, c, t, v, m = x.size()
    x = x.permute(0, 4, 3, 1, 2).contiguous()
    x = x.view(n * m, v * c, t)
    x = model.data_bn(x)
    x = x.view(n, m, v, c, t)
    x = x.permute(0, 1, 3, 4, 2).contiguous()
    x = x.view(n * m, c, t, v)

    for gcn, importance in zip(model.st_gcn_networks, model.edge_importance):
        x, _ = gcn(x, model.A * importance)

    _, channels, out_t, out_v = x.shape
    x = x.view(n, m, channels, out_t, out_v).mean(dim=1)
    x = x.permute(0, 2, 3, 1).contiguous()
    return x


def encode_split(model, data_path, label_path, output_data_path, output_label_path, meta_path, seg_num, batch_size, device):
    data = np.load(data_path).astype(np.float32)
    if data.ndim != 5:
        raise ValueError(f"Expected [N,C,T,V,M], got {data.shape}: {data_path}")
    if data.shape[1] != 3 or data.shape[3] != 18:
        raise ValueError(f"Expected C=3 and V=18, got {data.shape}: {data_path}")

    ids, scores = load_label(label_path)
    if len(ids) != data.shape[0]:
        raise ValueError(f"Data/label mismatch for {data_path}: {data.shape[0]} vs {len(ids)}")

    data = temporal_resample_stgcn_layout(data, seg_num=seg_num)
    outputs = []
    for start in range(0, data.shape[0], batch_size):
        batch = torch.from_numpy(data[start:start + batch_size]).to(device)
        feat = extract_backbone_joint_feature(model, batch)
        outputs.append(feat.cpu().numpy().astype(np.float32))
    outputs = np.concatenate(outputs, axis=0)

    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_data_path, outputs)
    with open(output_label_path, "wb") as f:
        pickle.dump((ids, scores.tolist()), f)

    meta = {
        "source_data": str(data_path),
        "source_label": str(label_path),
        "output_data": str(output_data_path),
        "output_label": str(output_label_path),
        "input_shape": list(np.load(data_path, mmap_mode="r").shape),
        "resampled_input_shape": list(data.shape),
        "output_shape": list(outputs.shape),
        "layout": "N,T,V,D",
        "feature_source": "ST-GCN backbone before global T/V pooling and fcn",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def find_average_dirs(data_root):
    return sorted(p for p in Path(data_root).glob("*/*_average") if p.is_dir())


def main():
    parser = argparse.ArgumentParser(description="Extract RG per-joint ST-GCN features.")
    parser.add_argument("--repo-root", default=".", help="Project root containing st-gcn-master.")
    parser.add_argument("--data-root", default="DATA/RG", help="RG root directory.")
    parser.add_argument("--output-root", default="DATA/RG_stgcn_joint_features", help="Output root directory.")
    parser.add_argument("--weights", default="st-gcn-master/models/st_gcn.kinetics.pt", help="ST-GCN backbone weights.")
    parser.add_argument("--seg-num", type=int, default=72, help="Number of temporal segments.")
    parser.add_argument("--batch-size", type=int, default=8, help="Extraction batch size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_stgcn_model(args.repo_root, args.weights, device)

    avg_dirs = find_average_dirs(args.data_root)
    if not avg_dirs:
        raise FileNotFoundError(f"No *_average directories found under {args.data_root}")

    all_meta = []
    for avg_dir in avg_dirs:
        action = avg_dir.parent.name
        prefix = action.upper()
        out_dir = Path(args.output_root) / action
        for split in ("train", "test"):
            data_candidates = sorted(avg_dir.glob(f"*data_{split}.npy"))
            label_candidates = sorted(avg_dir.glob(f"*label_{split}.pkl"))
            if not data_candidates or not label_candidates:
                raise FileNotFoundError(f"Missing {split} data/label in {avg_dir}")

            meta = encode_split(
                model=model,
                data_path=data_candidates[0],
                label_path=label_candidates[0],
                output_data_path=out_dir / f"{prefix}_stgcn_joint_{split}.npy",
                output_label_path=out_dir / f"{prefix}_label_{split}.pkl",
                meta_path=out_dir / f"{prefix}_stgcn_joint_{split}_meta.json",
                seg_num=args.seg_num,
                batch_size=args.batch_size,
                device=device,
            )
            all_meta.append(meta)
            print(f"{action:>6s} {split:>5s}: {meta['input_shape']} -> {meta['output_shape']}")

    summary_path = Path(args.output_root) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_meta, f, ensure_ascii=False, indent=2)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
