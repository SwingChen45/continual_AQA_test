# -*- coding: utf-8 -*-
"""
Generate fixed sequential-task splits for Fis-V labels.

Supported label pkl formats:
1) tuple/list of length 2:
   (ids, scores)
2) dict with fields like:
   {"ids": [...], "scores": [...]}
   or {"video_ids": [...], "labels": [...]}

Recommended usage:
- First, use TRAIN labels to compute boundaries and generate train split.
- Later, use the SAME boundaries to assign val/test samples.

Examples:
1) Generate TES train split with 5 quantile tasks
python split.py \
    --label_pkl DATA/FisV/label/TES_label_train.pkl \
    --output_pkl DATA/FisV/label_5tasks/fisv_tes_train_split_5tasks.pkl \
    --score_type TES \
    --num_tasks 5 \
    --method quantile

2) Generate PCS train split with 5 quantile tasks
python split.py \
    --label_pkl DATA/FisV/label/PCS_label_train.pkl \
    --output_pkl DATA/FisV/label_5tasks/fisv_pcs_train_split_5tasks.pkl \
    --score_type PCS \
    --num_tasks 5 \
    --method quantile

3) Use TRAIN boundaries to assign TEST labels
python split.py \
    --label_pkl  DATA/FisV/label/TES_label_test.pkl \
    --boundary_ref_pkl  DATA/FisV/label/TES_label_train.pkl \
    --output_pkl DATA/FisV/label_5tasks/fisv_tes_test_split_5tasks.pkl \
    --score_type TES \
    --num_tasks 5 \
    --method quantile
4) Use TRAIN boundaries to assign TEST labels
python split.py \
    --label_pkl  DATA/FisV/label/PCS_label_test.pkl \
    --boundary_ref_pkl  DATA/FisV/label/PCS_label_train.pkl \
    --output_pkl DATA/FisV/label_5tasks/fisv_pcs_test_split_5tasks.pkl \
    --score_type PCS \
    --num_tasks 5 \
    --method quantile
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def load_label_pkl(path: str) -> Tuple[List[str], np.ndarray]:
    """Load labels from a pkl file and return (ids, scores)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Label file not found: {p}")

    with open(p, "rb") as f:
        data = pickle.load(f)

    ids = None
    scores = None

    # Case 1: (ids, scores)
    if isinstance(data, (tuple, list)) and len(data) == 2:
        ids, scores = data[0], data[1]

    # Case 2: dict-like
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
                f"Unsupported dict format in {p}. "
                f"Need id-like and score-like keys."
            )
    else:
        raise ValueError(
            f"Unsupported pkl format in {p}. "
            f"Expected (ids, scores) or a dict."
        )

    ids = [str(x) for x in ids]
    scores = np.asarray(scores, dtype=np.float64)

    if len(ids) != len(scores):
        raise ValueError(
            f"Length mismatch in {p}: len(ids)={len(ids)}, len(scores)={len(scores)}"
        )

    if len(ids) == 0:
        raise ValueError(f"Empty label file: {p}")

    return ids, scores


def compute_boundaries(
    scores: np.ndarray,
    num_tasks: int,
    method: str = "quantile",
) -> np.ndarray:
    """Compute task boundaries."""
    if num_tasks < 2:
        raise ValueError("num_tasks must be >= 2")

    if method == "quantile":
        q = np.linspace(0, 1, num_tasks + 1)[1:-1]
        boundaries = np.quantile(scores, q)
    elif method == "equal_width":
        s_min, s_max = float(scores.min()), float(scores.max())
        boundaries = np.linspace(s_min, s_max, num_tasks + 1)[1:-1]
    else:
        raise ValueError(f"Unsupported method: {method}")

    return np.asarray(boundaries, dtype=np.float64)


def assign_tasks(scores: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """
    Assign each score to a task id in [0, num_tasks-1].

    Rule:
    - Task 0: [min, b0)
    - Task 1: [b0, b1)
    - ...
    - Last task: [b_last, max]
    """
    task_ids = np.digitize(scores, boundaries, right=False)
    return task_ids.astype(np.int64)


def summarize_split(
    ids: Sequence[str],
    scores: np.ndarray,
    task_ids: np.ndarray,
    boundaries: np.ndarray,
) -> List[Dict[str, Any]]:
    """Build summary for each task."""
    num_tasks = len(boundaries) + 1
    summary = []

    all_min = float(scores.min())
    all_max = float(scores.max())

    for t in range(num_tasks):
        idx = np.where(task_ids == t)[0]
        task_scores = scores[idx]

        if t == 0:
            left = all_min
            right = float(boundaries[0]) if len(boundaries) > 0 else all_max
            interval = f"[{left:.6f}, {right:.6f})" if num_tasks > 1 else f"[{left:.6f}, {right:.6f}]"
        elif t == num_tasks - 1:
            left = float(boundaries[-1])
            right = all_max
            interval = f"[{left:.6f}, {right:.6f}]"
        else:
            left = float(boundaries[t - 1])
            right = float(boundaries[t])
            interval = f"[{left:.6f}, {right:.6f})"

        item = {
            "task_id": int(t),
            "task_name": f"task_{t + 1}",
            "count": int(len(idx)),
            "interval": interval,
            "sample_indices": idx.tolist(),
            "sample_ids": [ids[i] for i in idx.tolist()],
            "score_min_in_task": float(task_scores.min()) if len(task_scores) > 0 else None,
            "score_max_in_task": float(task_scores.max()) if len(task_scores) > 0 else None,
            "score_mean_in_task": float(task_scores.mean()) if len(task_scores) > 0 else None,
        }
        summary.append(item)

    return summary


def to_jsonable(obj: Any) -> Any:
    """Convert numpy objects to json-safe objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def save_outputs(
    output_pkl: str,
    split_data: Dict[str, Any],
    output_json: str = "",
) -> None:
    """Save split file(s)."""
    output_pkl_path = Path(output_pkl)
    output_pkl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_pkl_path, "wb") as f:
        pickle.dump(split_data, f)

    if output_json:
        output_json_path = Path(output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(split_data), f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fis-V 5-task split file from label pkl.")
    parser.add_argument("--label_pkl", type=str, required=True, help="Label pkl to assign tasks for.")
    parser.add_argument(
        "--boundary_ref_pkl",
        type=str,
        default="",
        help="Optional label pkl used ONLY to compute boundaries. "
             "Useful for assigning test split with train boundaries.",
    )
    parser.add_argument("--output_pkl", type=str, required=True, help="Output split pkl.")
    parser.add_argument("--output_json", type=str, default="", help="Optional output json.")
    parser.add_argument("--score_type", type=str, default="UNKNOWN", help="TES / PCS / TOTAL / etc.")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of sequential tasks.")
    parser.add_argument(
        "--method",
        type=str,
        default="quantile",
        choices=["quantile", "equal_width"],
        help="Task boundary generation method.",
    )

    args = parser.parse_args()

    # Load assignment target
    ids, scores = load_label_pkl(args.label_pkl)

    # Load boundary reference
    if args.boundary_ref_pkl:
        _, ref_scores = load_label_pkl(args.boundary_ref_pkl)
        boundary_source = str(Path(args.boundary_ref_pkl).resolve())
    else:
        ref_scores = scores
        boundary_source = str(Path(args.label_pkl).resolve())

    boundaries = compute_boundaries(
        scores=ref_scores,
        num_tasks=args.num_tasks,
        method=args.method,
    )
    task_ids = assign_tasks(scores=scores, boundaries=boundaries)
    summary = summarize_split(ids=ids, scores=scores, task_ids=task_ids, boundaries=boundaries)

    split_data: Dict[str, Any] = {
        "meta": {
            "label_pkl": str(Path(args.label_pkl).resolve()),
            "boundary_ref_pkl": boundary_source,
            "score_type": args.score_type,
            "num_tasks": int(args.num_tasks),
            "method": args.method,
            "boundary_rule": "left-closed-right-open for intermediate bins; last bin right-closed",
        },
        "boundaries": boundaries,
        "ids": ids,
        "scores": scores,
        "task_ids": task_ids,
        "task_summary": summary,
    }

    save_outputs(
        output_pkl=args.output_pkl,
        split_data=split_data,
        output_json=args.output_json,
    )

    print("=" * 60)
    print("Split file saved.")
    print(f"Label file      : {args.label_pkl}")
    print(f"Boundary source : {boundary_source}")
    print(f"Score type      : {args.score_type}")
    print(f"Method          : {args.method}")
    print(f"Num tasks       : {args.num_tasks}")
    print(f"Boundaries      : {boundaries.tolist()}")
    print(f"Output pkl      : {args.output_pkl}")
    if args.output_json:
        print(f"Output json     : {args.output_json}")
    print("-" * 60)

    for item in summary:
        print(
            f"{item['task_name']:>8s} | "
            f"count={item['count']:>4d} | "
            f"interval={item['interval']} | "
            f"score_min={item['score_min_in_task']} | "
            f"score_max={item['score_max_in_task']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()