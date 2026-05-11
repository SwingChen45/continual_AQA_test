# -*- coding: utf-8 -*-
"""
Analyze large Fis-V feature/label files safely.

Supported:
- .npy feature files (loaded with mmap_mode='r' to reduce RAM pressure)
- .pkl label files in format:
    (ids, scores)
  or dict-like with id/score fields

Example:
python analyze_fisv_files.py \
  --tes_train_label TES_label_train.pkl \
  --tes_test_label TES_label_test.pkl \
  --pcs_train_label PCS_label_train.pkl \
  --pcs_test_label PCS_label_test.pkl \
  --rgb_train fisv_rgbvst_train.npy \
  --rgb_test fisv_rgbvst_test.npy \
  --flow_train fisv_flow_train.npy \
  --flow_test fisv_flow_test.npy \
  --skel_train FISV_stgcn_train.npy \
  --skel_test FISV_stgcn_test.npy \
  --output_json fisv_analysis_report.json

  python analyze_fisv_files.py \
  --tes_train_label DATA/FisV/label/TES_label_train.pkl \
  --tes_test_label DATA/FisV/label/TES_label_test.pkl \
  --pcs_train_label DATA/FisV/label/PCS_label_train.pkl \
  --pcs_test_label DATA/FisV/label/PCS_label_test.pkl \
  --rgb_train DATA/FisV/fisv_rgbvst_train.npy \
  --rgb_test DATA/FisV/fisv_rgbvst_test.npy \
  --flow_train DATA/FisV/fisv_flow_train.npy \
  --flow_test DATA/FisV/fisv_flow_test.npy \
  --skel_train DATA/FisV/FISV_stgcn_train.npy \
  --skel_test DATA/FisV/FISV_stgcn_test.npy \
  --output_json DATA/FisV/fisv_full_analysis.json
"""

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def to_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, Path):
        return str(x)
    return x


def pretty_print(title: str, info: Dict[str, Any]) -> None:
    print("=" * 80)
    print(title)
    print("-" * 80)
    for k, v in info.items():
        print(f"{k}: {v}")
    print("=" * 80)


def load_label_pkl(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)

    ids = None
    scores = None

    if isinstance(data, (tuple, list)) and len(data) == 2:
        ids, scores = data[0], data[1]
    elif isinstance(data, dict):
        id_keys = ["ids", "video_ids", "sample_ids", "names", "filenames"]
        score_keys = ["scores", "labels", "label", "targets", "y"]

        for k in id_keys:
            if k in data:
                ids = data[k]
                break
        for k in score_keys:
            if k in data:
                scores = data[k]
                break

    if ids is None or scores is None:
        raise ValueError(
            f"Unsupported label pkl format in {path}. "
            f"Expected (ids, scores) or dict with id/score fields."
        )

    ids = [str(x) for x in ids]
    scores = np.asarray(scores, dtype=np.float64)

    if len(ids) != len(scores):
        raise ValueError(
            f"Length mismatch in {path}: len(ids)={len(ids)}, len(scores)={len(scores)}"
        )

    return ids, scores


def summarize_labels(path: str) -> Dict[str, Any]:
    ids, scores = load_label_pkl(path)
    unique_ids = len(set(ids))
    dup_count = len(ids) - unique_ids

    info = {
        "path": path,
        "num_samples": len(ids),
        "num_unique_ids": unique_ids,
        "num_duplicate_ids": dup_count,
        "id_examples_head": ids[:5],
        "id_examples_tail": ids[-5:],
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_q20": float(np.quantile(scores, 0.2)),
        "score_q40": float(np.quantile(scores, 0.4)),
        "score_q60": float(np.quantile(scores, 0.6)),
        "score_q80": float(np.quantile(scores, 0.8)),
    }
    return info


def analyze_npy(path: str, chunk_size: int = 16) -> Dict[str, Any]:
    arr = np.load(path, mmap_mode="r")

    info: Dict[str, Any] = {
        "path": path,
        "shape": tuple(arr.shape),
        "ndim": int(arr.ndim),
        "dtype": str(arr.dtype),
    }

    if arr.dtype == object:
        info["warning"] = "object dtype array; numeric stats skipped"
        return info

    # Try to summarize sample axis / time axis / feature axis
    if arr.ndim >= 1:
        info["num_samples_dim0"] = int(arr.shape[0])
    if arr.ndim >= 2:
        info["time_dim1"] = int(arr.shape[1])
    if arr.ndim >= 3:
        info["feat_dim2"] = int(arr.shape[2])

    # Global numeric stats in chunks to avoid loading whole file
    finite_count = 0
    nan_count = 0
    inf_count = 0

    global_min = math.inf
    global_max = -math.inf
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    if arr.ndim == 0:
        chunk_iter = [np.asarray(arr)]
    elif arr.ndim >= 1:
        n = arr.shape[0]
        chunk_iter = (np.asarray(arr[i:i + chunk_size]) for i in range(0, n, chunk_size))
    else:
        chunk_iter = [np.asarray(arr)]

    for chunk in chunk_iter:
        chunk = np.asarray(chunk)

        nan_mask = np.isnan(chunk)
        inf_mask = np.isinf(chunk)
        finite_mask = np.isfinite(chunk)

        nan_count += int(nan_mask.sum())
        inf_count += int(inf_mask.sum())
        finite_count += int(finite_mask.sum())

        if finite_mask.any():
            finite_vals = chunk[finite_mask]
            cmin = float(finite_vals.min())
            cmax = float(finite_vals.max())
            global_min = min(global_min, cmin)
            global_max = max(global_max, cmax)
            total_sum += float(finite_vals.sum())
            total_sq_sum += float((finite_vals ** 2).sum())
            total_count += int(finite_vals.size)

    mean_val = total_sum / total_count if total_count > 0 else None
    std_val = None
    if total_count > 0:
        var = max(total_sq_sum / total_count - mean_val ** 2, 0.0)
        std_val = math.sqrt(var)

    info.update(
        {
            "finite_count": finite_count,
            "nan_count": nan_count,
            "inf_count": inf_count,
            "global_min": None if total_count == 0 else global_min,
            "global_max": None if total_count == 0 else global_max,
            "global_mean": mean_val,
            "global_std": std_val,
        }
    )

    # Per-sample quick diagnostics for first few samples
    if arr.ndim >= 2 and arr.shape[0] > 0:
        preview = []
        num_preview = min(3, arr.shape[0])
        for i in range(num_preview):
            sample = np.asarray(arr[i])
            finite_mask = np.isfinite(sample)
            if finite_mask.any():
                vals = sample[finite_mask]
                preview.append(
                    {
                        "sample_index": i,
                        "sample_shape": tuple(sample.shape),
                        "sample_min": float(vals.min()),
                        "sample_max": float(vals.max()),
                        "sample_mean": float(vals.mean()),
                        "sample_std": float(vals.std()),
                    }
                )
            else:
                preview.append(
                    {
                        "sample_index": i,
                        "sample_shape": tuple(sample.shape),
                        "sample_min": None,
                        "sample_max": None,
                        "sample_mean": None,
                        "sample_std": None,
                    }
                )
        info["sample_preview"] = preview

    return info


def compare_feature_and_label_counts(
    feature_info: Optional[Dict[str, Any]],
    label_info: Optional[Dict[str, Any]],
    name: str,
) -> Dict[str, Any]:
    result = {"name": name}
    if feature_info is None:
        result["status"] = "feature_missing"
        return result
    if label_info is None:
        result["status"] = "label_missing"
        return result

    f_n = feature_info.get("num_samples_dim0", None)
    l_n = label_info.get("num_samples", None)
    result["feature_num_samples"] = f_n
    result["label_num_samples"] = l_n
    result["match"] = (f_n == l_n)
    result["status"] = "ok" if f_n == l_n else "mismatch"
    return result


def compare_time_dims(
    rgb_info: Optional[Dict[str, Any]],
    flow_info: Optional[Dict[str, Any]],
    skel_info: Optional[Dict[str, Any]],
    split_name: str,
) -> Dict[str, Any]:
    result = {"split": split_name}

    def get_t(info):
        if info is None:
            return None
        return info.get("time_dim1", None)

    t_rgb = get_t(rgb_info)
    t_flow = get_t(flow_info)
    t_skel = get_t(skel_info)

    result["rgb_time"] = t_rgb
    result["flow_time"] = t_flow
    result["skel_time"] = t_skel
    result["all_equal"] = (t_rgb == t_flow == t_skel)
    return result


def maybe_analyze_npy(path: Optional[str], chunk_size: int) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    return analyze_npy(path, chunk_size=chunk_size)


def maybe_analyze_label(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    return summarize_labels(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Fis-V feature and label files.")
    parser.add_argument("--tes_train_label", type=str, default="")
    parser.add_argument("--tes_test_label", type=str, default="")
    parser.add_argument("--pcs_train_label", type=str, default="")
    parser.add_argument("--pcs_test_label", type=str, default="")

    parser.add_argument("--rgb_train", type=str, default="")
    parser.add_argument("--rgb_test", type=str, default="")
    parser.add_argument("--flow_train", type=str, default="")
    parser.add_argument("--flow_test", type=str, default="")
    parser.add_argument("--skel_train", type=str, default="")
    parser.add_argument("--skel_test", type=str, default="")

    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--output_json", type=str, default="")

    args = parser.parse_args()

    report: Dict[str, Any] = {"labels": {}, "features": {}, "consistency_checks": {}}

    # Labels
    report["labels"]["tes_train"] = maybe_analyze_label(args.tes_train_label)
    report["labels"]["tes_test"] = maybe_analyze_label(args.tes_test_label)
    report["labels"]["pcs_train"] = maybe_analyze_label(args.pcs_train_label)
    report["labels"]["pcs_test"] = maybe_analyze_label(args.pcs_test_label)

    # Features
    report["features"]["rgb_train"] = maybe_analyze_npy(args.rgb_train, args.chunk_size)
    report["features"]["rgb_test"] = maybe_analyze_npy(args.rgb_test, args.chunk_size)
    report["features"]["flow_train"] = maybe_analyze_npy(args.flow_train, args.chunk_size)
    report["features"]["flow_test"] = maybe_analyze_npy(args.flow_test, args.chunk_size)
    report["features"]["skel_train"] = maybe_analyze_npy(args.skel_train, args.chunk_size)
    report["features"]["skel_test"] = maybe_analyze_npy(args.skel_test, args.chunk_size)

    # Count checks
    report["consistency_checks"]["tes_train_vs_rgb_train"] = compare_feature_and_label_counts(
        report["features"]["rgb_train"], report["labels"]["tes_train"], "tes_train_vs_rgb_train"
    )
    report["consistency_checks"]["tes_test_vs_rgb_test"] = compare_feature_and_label_counts(
        report["features"]["rgb_test"], report["labels"]["tes_test"], "tes_test_vs_rgb_test"
    )
    report["consistency_checks"]["pcs_train_vs_rgb_train"] = compare_feature_and_label_counts(
        report["features"]["rgb_train"], report["labels"]["pcs_train"], "pcs_train_vs_rgb_train"
    )
    report["consistency_checks"]["pcs_test_vs_rgb_test"] = compare_feature_and_label_counts(
        report["features"]["rgb_test"], report["labels"]["pcs_test"], "pcs_test_vs_rgb_test"
    )

    # Time-dim checks
    report["consistency_checks"]["train_time_dims"] = compare_time_dims(
        report["features"]["rgb_train"],
        report["features"]["flow_train"],
        report["features"]["skel_train"],
        "train",
    )
    report["consistency_checks"]["test_time_dims"] = compare_time_dims(
        report["features"]["rgb_test"],
        report["features"]["flow_test"],
        report["features"]["skel_test"],
        "test",
    )

    # Print summaries
    for split_name, info in report["labels"].items():
        if info is not None:
            pretty_print(f"LABEL SUMMARY: {split_name}", info)

    for split_name, info in report["features"].items():
        if info is not None:
            pretty_print(f"FEATURE SUMMARY: {split_name}", info)

    pretty_print("CONSISTENCY CHECKS", report["consistency_checks"])

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(report), f, ensure_ascii=False, indent=2)
        print(f"\nSaved analysis report to: {out_path}")


if __name__ == "__main__":
    main()