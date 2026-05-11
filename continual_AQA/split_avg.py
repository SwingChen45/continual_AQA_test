# -*- coding: utf-8 -*-
"""
Generate Stratified Uniform sequential-task splits for Fis-V labels.
This ensures each task has an identical score distribution (Data-Incremental Learning).

Recommended usage:
1) Generate TES train split with 5 tasks
python split_stratified.py \
    --label_pkl DATA/FisV/label/TES_label_train.pkl \
    --output_pkl DATA/FisV/label_5tasks/fisv_tes_train_split_5tasks.pkl \
    --score_type TES \
    --num_tasks 5 \
    --seed 42

2) Generate TES test split with 5 tasks (Test sets are evaluated per task)
python split_stratified.py \
    --label_pkl DATA/FisV/label/TES_label_test.pkl \
    --output_pkl DATA/FisV/label_5tasks/fisv_tes_test_split_5tasks.pkl \
    --score_type TES \
    --num_tasks 5 \
    --seed 42
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
            raise ValueError(f"Unsupported dict format in {p}.")
    else:
        raise ValueError(f"Unsupported pkl format in {p}.")

    ids = [str(x) for x in ids]
    scores = np.asarray(scores, dtype=np.float64)

    if len(ids) != len(scores):
        raise ValueError("Length mismatch between ids and scores.")
    if len(ids) == 0:
        raise ValueError("Empty label file.")

    return ids, scores

def assign_tasks_stratified(scores: np.ndarray, num_tasks: int, seed: int = 42) -> np.ndarray:
    """
    Assign tasks by stratifying scores so each task gets an equal distribution.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(scores)
    
    # Get indices of scores sorted from lowest to highest
    sorted_indices = np.argsort(scores)
    task_ids = np.zeros(n_samples, dtype=np.int64)

    # Process in chunks of size 'num_tasks'
    for i in range(0, n_samples, num_tasks):
        chunk_indices = sorted_indices[i : i + num_tasks]
        
        # Create a list of available tasks for this chunk
        available_tasks = np.arange(num_tasks)
        rng.shuffle(available_tasks)
        
        # Assign each item in the chunk to a random unique task
        for idx, task_id in zip(chunk_indices, available_tasks):
            task_ids[idx] = task_id

    return task_ids

def summarize_split(
    ids: Sequence[str],
    scores: np.ndarray,
    task_ids: np.ndarray,
    num_tasks: int
) -> List[Dict[str, Any]]:
    """Build summary for each task to verify distribution."""
    summary = []

    for t in range(num_tasks):
        idx = np.where(task_ids == t)[0]
        task_scores = scores[idx]

        item = {
            "task_id": int(t),
            "task_name": f"task_{t + 1}",
            "count": int(len(idx)),
            "sample_indices": idx.tolist(),
            "sample_ids": [ids[i] for i in idx.tolist()],
            "score_min_in_task": float(task_scores.min()) if len(task_scores) > 0 else None,
            "score_max_in_task": float(task_scores.max()) if len(task_scores) > 0 else None,
            "score_mean_in_task": float(task_scores.mean()) if len(task_scores) > 0 else None,
            "score_std_in_task": float(task_scores.std()) if len(task_scores) > 0 else None,
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

def save_outputs(output_pkl: str, split_data: Dict[str, Any], output_json: str = "") -> None:
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
    parser = argparse.ArgumentParser(description="Generate Stratified Uniform 5-task split.")
    parser.add_argument("--label_pkl", type=str, required=True, help="Label pkl to assign tasks for.")
    parser.add_argument("--output_pkl", type=str, required=True, help="Output split pkl.")
    parser.add_argument("--output_json", type=str, default="", help="Optional output json.")
    parser.add_argument("--score_type", type=str, default="UNKNOWN", help="TES / PCS / TOTAL / etc.")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of sequential tasks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for consistent stratification.")

    args = parser.parse_args()

    ids, scores = load_label_pkl(args.label_pkl)

    task_ids = assign_tasks_stratified(scores=scores, num_tasks=args.num_tasks, seed=args.seed)
    summary = summarize_split(ids=ids, scores=scores, task_ids=task_ids, num_tasks=args.num_tasks)

    split_data: Dict[str, Any] = {
        "meta": {
            "label_pkl": str(Path(args.label_pkl).resolve()),
            "score_type": args.score_type,
            "num_tasks": int(args.num_tasks),
            "method": "stratified_uniform_random",
        },
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
    print("Stratified Split file saved.")
    print(f"Label file      : {args.label_pkl}")
    print(f"Score type      : {args.score_type}")
    print(f"Method          : Stratified Uniform Random")
    print(f"Num tasks       : {args.num_tasks}")
    print(f"Random Seed     : {args.seed}")
    print(f"Output pkl      : {args.output_pkl}")
    print("-" * 60)

    for item in summary:
        print(
            f"{item['task_name']:>8s} | "
            f"count={item['count']:>4d} | "
            f"mean={item['score_mean_in_task']:>6.2f} | "
            f"std={item['score_std_in_task']:>5.2f} | "
            f"min={item['score_min_in_task']:>5.2f} | "
            f"max={item['score_max_in_task']:>5.2f}"
        )
    print("=" * 60)

if __name__ == "__main__":
    main()