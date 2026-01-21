#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate QA, predict goal, and evaluate preference satisfaction for a sample JSON.

Usage:
  python benchmark/questionner_test_sample.py --input benchmark/test_sample.json
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from oracle_answerer import OracleAnswerer
from organizer import Organizer
from preference_eval import PreferenceMetric
from questionner_baseline import QuestionnerBaseline


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="benchmark/test_sample.json",
        help="Path to input JSON (will be overwritten unless --output is set).",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output path. If empty, overwrite input file.",
    )
    parser.add_argument("--question_model", default="qwen2.5:7b")
    parser.add_argument("--oracle_model", default="qwen2.5:7b")
    parser.add_argument("--organizer_model", default="qwen2.5:7b")
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num_predict", type=int, default=256)
    parser.add_argument(
        "--summary_output",
        default="",
        help="Optional path to save evaluation summary JSON.",
    )
    args = parser.parse_args()

    output_path = args.output or args.input

    options = {
        "temperature": args.temperature,
        "num_predict": args.num_predict,
    }

    questioner = QuestionnerBaseline(args.question_model, options=options)
    oracle = OracleAnswerer(args.oracle_model, options=options)
    organizer = Organizer(args.organizer_model, options=options)
    evaluator = PreferenceMetric()

    dataset = load_json(args.input)
    satisfaction_scores = []
    precision_scores = []
    for sample in dataset:
        qa_history = []
        for _ in range(args.max_turns):
            question = questioner.ask({**sample, "qa_history": qa_history})

            answer = oracle.answer(sample, question, qa_history=qa_history)
            turn_id = len(qa_history) + 1
            qa_history.append({"turn_id": turn_id, "question": question, "answer": answer})

        sample["qa_history"] = qa_history
        pred_goal = organizer.predict_goal(sample)
        sample["pred_goal"] = pred_goal

        metrics, counts = evaluator.evaluate_with_details(
            pred_goal=pred_goal,
            gt_goal=sample.get("goal", []),
        )
        sample["preference_eval"] = {"metrics": metrics, "counts": counts}
        satisfaction_scores.append(metrics["satisfaction"])
        precision_scores.append(metrics["precision"])

    save_json(output_path, dataset)
    print(f"Saved updated QA, pred_goal, and eval to {output_path}")

    summary = {
        "avg_satisfaction": sum(satisfaction_scores) / max(len(satisfaction_scores), 1),
        "avg_precision": sum(precision_scores) / max(len(precision_scores), 1),
        "num_samples": len(dataset),
    }
    if args.summary_output:
        save_json(args.summary_output, summary)
        print(f"Saved summary to {args.summary_output}")
    else:
        print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
