#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encode RG *_average OpenPose skeletons into joint-level features.

Input directories are expected to look like:
    DATA/RG/ball/ball_average/BALL_data_train.npy
    DATA/RG/ball/ball_average/BALL_label_train.pkl

The raw average skeleton tensor is in ST-GCN layout:
    [N, C, T, V, M]

This script writes AQA-style joint features:
    [N, seg_num, V, D]

The default feature encoder is deterministic and hand-crafted. It preserves
real joint nodes and adds temporal motion/bone cues. If --output-dim is larger
than the base feature size, a fixed random projection expands the features to
the requested dimension. This only aligns the feature shape with AQA-style code;
it does not make the features equivalent to AQA-7 visual patch features.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


OPENPOSE_18_PARENTS = np.array(
    [
        1,   # 0 nose -> neck
        -1,  # 1 neck/root
        1,   # 2 right shoulder
        2,   # 3 right elbow
        3,   # 4 right wrist
        1,   # 5 left shoulder
        5,   # 6 left elbow
        6,   # 7 left wrist
        1,   # 8 right hip
        8,   # 9 right knee
        9,   # 10 right ankle
        1,   # 11 left hip
        11,  # 12 left knee
        12,  # 13 left ankle
        0,   # 14 right eye
        0,   # 15 left eye
        14,  # 16 right ear
        15,  # 17 left ear
    ],
    dtype=np.int64,
)


def load_label(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, (tuple, list)) or len(obj) != 2:
        raise ValueError(f"Unsupported label format: {path}")
    ids = [str(x) for x in obj[0]]
    scores = np.asarray(obj[1], dtype=np.float32)
    if len(ids) != len(scores):
        raise ValueError(f"Label length mismatch in {path}")
    return ids, scores


def temporal_resample(x, seg_num):
    """Resample [T, V, C] to [seg_num, V, C] by linear indexing."""
    t = x.shape[0]
    if t == seg_num:
        return x.astype(np.float32, copy=False)
    if t <= 0:
        raise ValueError("Cannot resample an empty sequence.")
    idx = np.linspace(0, t - 1, seg_num)
    left = np.floor(idx).astype(np.int64)
    right = np.ceil(idx).astype(np.int64)
    weight = (idx - left).astype(np.float32).reshape(seg_num, 1, 1)
    return ((1.0 - weight) * x[left] + weight * x[right]).astype(np.float32)


def normalize_xyc(seq):
    """
    Normalize [T, V, 3] OpenPose coordinates per sample.

    Coordinates are centered on neck if available and scaled by the median
    shoulder distance. Confidence is kept as-is.
    """
    out = seq.astype(np.float32, copy=True)
    xy = out[..., :2]
    conf = out[..., 2:3]
    valid = conf > 0

    root = xy[:, 1:2, :]
    root_valid = valid[:, 1:2, :]
    if not np.any(root_valid):
        valid_xy = xy[valid.repeat(2, axis=2)].reshape(-1, 2)
        fallback = valid_xy.mean(axis=0, keepdims=True) if len(valid_xy) else np.zeros((1, 2), dtype=np.float32)
        root = np.broadcast_to(fallback.reshape(1, 1, 2), xy.shape[:1] + (1, 2)).copy()

    xy = xy - root

    shoulder = np.linalg.norm(out[:, 2, :2] - out[:, 5, :2], axis=1)
    shoulder = shoulder[shoulder > 1e-6]
    if len(shoulder) > 0:
        scale = float(np.median(shoulder))
    else:
        valid_xy = xy[valid.repeat(2, axis=2)].reshape(-1, 2)
        scale = float(np.std(valid_xy)) if len(valid_xy) else 1.0
    scale = max(scale, 1.0)

    out[..., :2] = xy / scale
    out[..., 2:3] = conf
    return out


def diff_along_time(x):
    first = np.zeros_like(x[:1])
    return np.concatenate([first, x[1:] - x[:-1]], axis=0)


def build_base_features(seq, parents):
    """
    Build [T, V, F] features from normalized [T, V, 3].

    Feature groups:
      normalized x/y/conf, velocity, acceleration, root-relative xy,
      parent-relative bone xy/length, velocity magnitude.
    """
    xyc = normalize_xyc(seq)
    vel = diff_along_time(xyc)
    acc = diff_along_time(vel)

    root_rel = xyc[..., :2] - xyc[:, 1:2, :2]

    bone_xy = np.zeros_like(root_rel)
    for j, p in enumerate(parents):
        if p >= 0:
            bone_xy[:, j, :] = xyc[:, j, :2] - xyc[:, p, :2]
    bone_len = np.linalg.norm(bone_xy, axis=-1, keepdims=True)
    speed = np.linalg.norm(vel[..., :2], axis=-1, keepdims=True)

    return np.concatenate(
        [
            xyc,
            vel,
            acc,
            root_rel,
            bone_xy,
            bone_len,
            speed,
        ],
        axis=-1,
    ).astype(np.float32)


def project_features(features, output_dim, seed):
    if output_dim <= 0 or output_dim == features.shape[-1]:
        return features
    rng = np.random.default_rng(seed)
    in_dim = features.shape[-1]
    weight = rng.normal(0.0, 1.0 / np.sqrt(in_dim), size=(in_dim, output_dim)).astype(np.float32)
    return np.tanh(features @ weight).astype(np.float32)


def encode_file(data_path, label_path, output_data_path, output_label_path, meta_path, seg_num, output_dim, seed):
    data = np.load(data_path)
    if data.ndim != 5:
        raise ValueError(f"Expected [N,C,T,V,M], got {data.shape}: {data_path}")
    n, c, _, v, m = data.shape
    if c < 3:
        raise ValueError(f"Expected at least 3 coordinate channels, got {c}: {data_path}")
    if m != 1:
        raise ValueError(f"Expected one person dimension M=1, got {m}: {data_path}")
    if v != len(OPENPOSE_18_PARENTS):
        raise ValueError(f"Expected OpenPose-18 joints, got V={v}: {data_path}")

    ids, scores = load_label(label_path)
    if len(ids) != n:
        raise ValueError(f"Data/label mismatch: data N={n}, labels={len(ids)} for {data_path}")

    encoded = []
    for i in range(n):
        seq = np.transpose(data[i, :3, :, :, 0], (1, 2, 0))  # [T,V,C]
        seq = temporal_resample(seq, seg_num=seg_num)
        feat = build_base_features(seq, OPENPOSE_18_PARENTS)
        feat = project_features(feat, output_dim=output_dim, seed=seed)
        encoded.append(feat)
    encoded = np.stack(encoded, axis=0).astype(np.float32)  # [N,T,V,D]

    output_data_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_data_path, encoded)
    with open(output_label_path, "wb") as f:
        pickle.dump((ids, scores.tolist()), f)

    meta = {
        "source_data": str(data_path),
        "source_label": str(label_path),
        "output_data": str(output_data_path),
        "output_label": str(output_label_path),
        "input_shape": list(data.shape),
        "output_shape": list(encoded.shape),
        "layout": "N,T,V,D",
        "seg_num": int(seg_num),
        "joint_num": int(v),
        "base_feature_dim": 15,
        "output_dim": int(encoded.shape[-1]),
        "projection_seed": int(seed) if output_dim > 0 and output_dim != 15 else None,
        "note": "Shape/node aligned with AQA-style joint features, not equivalent to AQA-7 visual patch features.",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def find_average_dirs(data_root):
    return sorted(p for p in Path(data_root).glob("*/*_average") if p.is_dir())


def main():
    parser = argparse.ArgumentParser(description="Encode RG *_average skeletons to joint-level features.")
    parser.add_argument("--data-root", default="DATA/RG", help="RG root directory.")
    parser.add_argument("--output-root", default="DATA/RG_joint_features", help="Output root directory.")
    parser.add_argument("--seg-num", type=int, default=72, help="Number of temporal segments.")
    parser.add_argument("--output-dim", type=int, default=400, help="Output joint feature dim. Use 15 to keep base features.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for fixed random projection.")
    args = parser.parse_args()

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

            out_data = out_dir / f"{prefix}_joint_{split}.npy"
            out_label = out_dir / f"{prefix}_label_{split}.pkl"
            out_meta = out_dir / f"{prefix}_joint_{split}_meta.json"

            meta = encode_file(
                data_path=data_candidates[0],
                label_path=label_candidates[0],
                output_data_path=out_data,
                output_label_path=out_label,
                meta_path=out_meta,
                seg_num=args.seg_num,
                output_dim=args.output_dim,
                seed=args.seed,
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
