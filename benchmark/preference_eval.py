#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate preference satisfaction from pred_goal vs goal.

Usage:
  python benchmark/preference_eval.py --input benchmark/test_sample.json
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple


FACT_RE = re.compile(r"\(\s*([^\s()]+)\s+([^\s()]+)\s+([^\s()]+)\s*\)")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def parse_fact(fact: str) -> Tuple[str, str, str]:
    match = FACT_RE.search(fact)
    if not match:
        return "", "", ""
    pred, obj, receptacle = match.group(1), match.group(2), match.group(3)
    return pred, obj, receptacle


def build_goal_map(facts: List[str]) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    for fact in facts:
        if not isinstance(fact, str):
            continue
        pred, obj, receptacle = parse_fact(fact)
        if pred and obj and receptacle:
            mapping[obj] = (pred, receptacle)
    return mapping


class PreferenceMetric:
    """
    TaskMetric-like evaluator for static goal lists.
    Final score is the fraction of goal objects placed in the correct receptacle.
    """

    def evaluate(self, pred_goal: List[str], gt_goal: List[str]) -> float:
        metrics, _ = self.evaluate_with_details(pred_goal, gt_goal)
        return metrics["satisfaction"]

    def evaluate_with_details(
        self, pred_goal: List[str], gt_goal: List[str]
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        gt_map = build_goal_map(gt_goal)
        pred_map = build_goal_map(pred_goal)

        total = len(gt_map)
        matched = 0
        for obj, (gt_pred, gt_receptacle) in gt_map.items():
            pred_pair = pred_map.get(obj)
            if pred_pair is None:
                continue
            pred_pred, pred_receptacle = pred_pair
            if pred_pred == gt_pred and pred_receptacle == gt_receptacle:
                matched += 1

        if total == 0:
            satisfaction = 1.0
        else:
            satisfaction = matched / total

        precision = matched / max(len(pred_map), 1)

        metrics = {
            "satisfaction": satisfaction,
            "precision": precision,
        }
        counts = {
            "pred_total": len(pred_map),
            "gt_total": total,
            "matched_gt": matched,
        }
        return metrics, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="benchmark/test_sample.json",
        help="Path to input JSON containing goal and pred_goal.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path. If empty, print summary only.",
    )
    parser.add_argument("--goal_key", default="goal")
    parser.add_argument("--pred_goal_key", default="pred_goal")
    args = parser.parse_args()

    dataset = load_json(args.input)
    evaluator = PreferenceMetric()
    results = []
    satisfaction_scores = []
    precision_scores = []

    for sample in dataset:
        pred_goal = sample.get(args.pred_goal_key, [])
        gt_goal = sample.get(args.goal_key, [])
        if not isinstance(pred_goal, list):
            pred_goal = []
        if not isinstance(gt_goal, list):
            gt_goal = []

        metrics, counts = evaluator.evaluate_with_details(pred_goal, gt_goal)

        satisfaction_scores.append(metrics["satisfaction"])
        precision_scores.append(metrics["precision"])

        results.append(
            {
                "task": sample.get("task"),
                "pred_goal": pred_goal,
                "gt_goal": gt_goal,
                "metrics": metrics,
                "counts": counts,
            }
        )

    summary = {
        "avg_satisfaction": sum(satisfaction_scores) / max(len(satisfaction_scores), 1),
        "avg_precision": sum(precision_scores) / max(len(precision_scores), 1),
        "num_samples": len(results),
    }

    if args.output:
        save_json(args.output, {"summary": summary, "results": results})
        print(f"Saved results to {args.output}")
    else:
        print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
