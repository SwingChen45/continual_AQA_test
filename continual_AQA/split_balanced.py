# -*- coding: utf-8 -*-
"""
Generate balanced/interleaved 5-task split for Fis-V.

python split_balanced.py \
  --label_pkl DATA/FisV/label/TES_label_train.pkl \
  --output_pkl DATA/FisV/balance_5tasks/fisv_tes_train_split_5tasks.pkl \
  --output_json DATA/FisV/balance_5tasks/fisv_tes_train_split_5tasks.json \
  --score_type TES \
  --num_tasks 5 \
  --seed 42

python split_balanced.py \
  --label_pkl DATA/FisV/label/TES_label_test.pkl \
  --output_pkl DATA/FisV/balance_5tasks/fisv_tes_test_split_5tasks.pkl \
  --output_json DATA/FisV/balance_5tasks/fisv_tes_test_split_5tasks.json \
  --score_type TES \
  --num_tasks 5 \
  --seed 42

python split_balanced.py \
  --label_pkl DATA/FisV/label/PCS_label_train.pkl \
  --output_pkl DATA/FisV/balance_5tasks/fisv_pcs_train_split_5tasks.pkl \
  --output_json DATA/FisV/balance_5tasks/fisv_pcs_train_split_5tasks.json \
  --score_type PCS \
  --num_tasks 5 \
  --seed 42

python split_balanced.py \
  --label_pkl DATA/FisV/label/PCS_label_test.pkl \
  --output_pkl DATA/FisV/balance_5tasks/fisv_pcs_test_split_5tasks.pkl \
  --output_json DATA/FisV/balance_5tasks/fisv_pcs_test_split_5tasks.json \
  --score_type PCS \
  --num_tasks 5 \
  --seed 42
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_label_pkl(path: str) -> Tuple[List[str], np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, (tuple, list)) and len(data) == 2:
        ids, scores = data
    elif isinstance(data, dict):
        id_keys = ["ids", "video_ids", "sample_ids", "names", "filenames"]
        score_keys = ["scores", "labels", "label", "targets", "y"]

        ids, scores = None, None
        for k in id_keys:
            if k in data:
                ids = data[k]
                break
        for k in score_keys:
            if k in data:
                scores = data[k]
                break
        if ids is None or scores is None:
            raise ValueError(f"Unsupported dict format in {path}")
    else:
        raise ValueError(f"Unsupported label format in {path}")

    ids = [str(x) for x in ids]
    scores = np.asarray(scores, dtype=np.float32)

    if len(ids) != len(scores):
        raise ValueError(f"Length mismatch in {path}: len(ids) != len(scores)")

    return ids, scores


def assign_tasks_balanced(scores: np.ndarray, num_tasks: int = 5, seed: int = 0) -> np.ndarray:
    """
    Balanced/interleaved rank-based split:
    - sort by score ascending
    - each consecutive block of `num_tasks` samples becomes a bucket
    - randomly permute task ids in that bucket
    """
    rng = np.random.RandomState(seed)

    n = len(scores)
    sorted_idx = np.argsort(scores, kind="stable")
    task_ids = np.full(n, fill_value=-1, dtype=np.int64)

    full_bucket_size = (n // num_tasks) * num_tasks

    # Full buckets
    for start in range(0, full_bucket_size, num_tasks):
        bucket = sorted_idx[start:start + num_tasks]
        permuted_tasks = rng.permutation(num_tasks)
        for sample_idx, task_id in zip(bucket, permuted_tasks):
            task_ids[sample_idx] = int(task_id)

    # Leftover samples
    leftover = sorted_idx[full_bucket_size:]
    if len(leftover) > 0:
        leftover_tasks = rng.choice(np.arange(num_tasks), size=len(leftover), replace=False)
        for sample_idx, task_id in zip(leftover, leftover_tasks):
            task_ids[sample_idx] = int(task_id)

    if np.any(task_ids < 0):
        raise RuntimeError("Some samples were not assigned a task.")

    return task_ids


def summarize_split(ids: List[str], scores: np.ndarray, task_ids: np.ndarray, num_tasks: int) -> List[Dict[str, Any]]:
    summary = []
    for t in range(num_tasks):
        idx = np.where(task_ids == t)[0]
        task_scores = scores[idx]

        item = {
            "task_id": int(t),
            "task_name": f"task_{t + 1}",
            "count": int(len(idx)),
            "score_min": float(task_scores.min()) if len(task_scores) > 0 else None,
            "score_max": float(task_scores.max()) if len(task_scores) > 0 else None,
            "score_mean": float(task_scores.mean()) if len(task_scores) > 0 else None,
            "score_std": float(task_scores.std()) if len(task_scores) > 0 else None,
            "sample_ids_head": [ids[i] for i in idx[:5]],
        }
        summary.append(item)
    return summary


def to_jsonable(obj: Any) -> Any:
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


def save_outputs(output_pkl: str, split_data: Dict[str, Any], output_json: str = ""):
    output_pkl = Path(output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_pkl, "wb") as f:
        pickle.dump(split_data, f)

    if output_json:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(split_data), f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate balanced/interleaved Fis-V split.")
    parser.add_argument("--label_pkl", type=str, required=True, help="Input label pkl, e.g. TES_label_train.pkl")
    parser.add_argument("--output_pkl", type=str, required=True, help="Output split pkl")
    parser.add_argument("--output_json", type=str, default="", help="Optional output json")
    parser.add_argument("--score_type", type=str, default="TES", help="TES / PCS")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of tasks")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bucket shuffling")

    args = parser.parse_args()

    ids, scores = load_label_pkl(args.label_pkl)
    task_ids = assign_tasks_balanced(scores, num_tasks=args.num_tasks, seed=args.seed)
    summary = summarize_split(ids, scores, task_ids, args.num_tasks)

    split_data = {
        "meta": {
            "label_pkl": str(Path(args.label_pkl).resolve()),
            "score_type": args.score_type,
            "num_tasks": int(args.num_tasks),
            "method": "balanced_rank_interleaving",
            "seed": int(args.seed),
            "description": (
                "Sort by score, form consecutive buckets of size num_tasks, "
                "shuffle task ids within each bucket so every task gets low/mid/high samples."
            ),
        },
        "ids": ids,
        "scores": scores,
        "task_ids": task_ids,
        "task_summary": summary,
    }

    save_outputs(args.output_pkl, split_data, args.output_json)

    print("=" * 60)
    print("Balanced split generated.")
    print(f"Label file : {args.label_pkl}")
    print(f"Score type : {args.score_type}")
    print(f"Num tasks  : {args.num_tasks}")
    print(f"Seed       : {args.seed}")
    print(f"Output pkl : {args.output_pkl}")
    if args.output_json:
        print(f"Output json: {args.output_json}")
    print("-" * 60)

    for item in summary:
        print(
            f"{item['task_name']:>8s} | "
            f"count={item['count']:>4d} | "
            f"score_min={item['score_min']:.4f} | "
            f"score_max={item['score_max']:.4f} | "
            f"score_mean={item['score_mean']:.4f} | "
            f"score_std={item['score_std']:.4f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()