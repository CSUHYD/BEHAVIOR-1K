#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predict goal facts from QA history.

Usage:
  python benchmark/organizer.py --config benchmark/configs/organizer_config.json
  python benchmark/organizer.py --input benchmark/test_sample.json --output out.json
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ollama_call import VLMAPI


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


def parse_json_maybe(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}


def parse_fact(fact: str) -> Tuple[str, str, str]:
    match = FACT_RE.search(fact or "")
    if not match:
        return "", "", ""
    pred, obj, receptacle = match.group(1), match.group(2), match.group(3)
    return pred, obj, receptacle


def build_fact_map(facts: List[str]) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}
    for fact in facts:
        if not isinstance(fact, str):
            continue
        pred, obj, receptacle = parse_fact(fact)
        if pred and obj and receptacle and obj not in mapping:
            mapping[obj] = (pred, receptacle)
    return mapping


def format_objects(objects: Dict[str, str]) -> str:
    return "\n".join([f"- {k}: {v}" for k, v in objects.items()])


def format_receptacles(receptacles: Dict[str, str]) -> str:
    return "\n".join([f"- {k}: {v}" for k, v in receptacles.items()])


def format_facts(facts: List[str]) -> str:
    return "\n".join([f"- {x}" for x in facts])


def format_qa_history(qa_history: List[Dict[str, Any]]) -> str:
    lines = []
    for item in qa_history:
        q = item.get("question", "")
        a = item.get("answer", "")
        tid = item.get("turn_id", "")
        lines.append(f"{tid}. Q: {q}\n   A: {a}")
    return "\n".join(lines) if lines else "(empty)"


def build_prompt(sample: Dict[str, Any]) -> Dict[str, str]:
    objects = format_objects(sample.get("objects", {}))
    receptacles = format_receptacles(sample.get("receptacles", {}))
    init = format_facts(sample.get("init", []))
    history = format_qa_history(sample.get("qa_history", []))

    systext = (
        "You are a household organizing assistant.\n"
        "Infer the user's desired final arrangement from the QA history.\n"
        "Return strict JSON with exactly one key: \"pred_goal\".\n"
        "Each item must be a fact like (inside obj receptacle) or (ontop obj receptacle).\n"
        "Use only the object/receptacle ids given. One placement per object.\n"
        "If unsure, keep the object in its current init location.\n"
        "No extra text, no markdown.\n"
    )
    usertext = (
        "Context:\n"
        f"Task: {sample.get('task', 'unknown')}\n\n"
        f"Objects:\n{objects}\n\n"
        f"Receptacles:\n{receptacles}\n\n"
        f"Init facts:\n{init}\n\n"
        f"QA history:\n{history}\n\n"
        "Predict the final placement for every object."
    )
    return {"systext": systext, "usertext": usertext}


def normalize_pred_goal(
    pred_goal: List[str],
    sample: Dict[str, Any],
) -> List[str]:
    objects = sample.get("objects", {})
    receptacles = sample.get("receptacles", {})
    allowed_objects = set(objects.keys())
    allowed_receptacles = set(receptacles.keys()) | allowed_objects
    init_map = build_fact_map(sample.get("init", []))

    pred_map: Dict[str, Tuple[str, str]] = {}
    for fact in pred_goal:
        if not isinstance(fact, str):
            continue
        pred, obj, receptacle = parse_fact(fact)
        if pred not in ("inside", "ontop"):
            continue
        if obj not in allowed_objects:
            continue
        if receptacle not in allowed_receptacles:
            continue
        if obj not in pred_map:
            pred_map[obj] = (pred, receptacle)

    results: List[str] = []
    for obj in objects.keys():
        if obj in pred_map:
            pred, receptacle = pred_map[obj]
        elif obj in init_map:
            pred, receptacle = init_map[obj]
        else:
            continue
        results.append(f"({pred} {obj} {receptacle})")
    return results


class Organizer:
    def __init__(
        self,
        model: str,
        options: Optional[Dict[str, Any]] = None,
        schema: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.api = VLMAPI(model)
        self.options = options or {}
        self.schema = schema or {
            "type": "object",
            "properties": {
                "pred_goal": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["pred_goal"],
        }

    def build_prompt(self, sample: Dict[str, Any]) -> Dict[str, str]:
        return build_prompt(sample)

    def parse_pred_goal(self, text: str) -> List[str]:
        parsed = parse_json_maybe(text)
        pred_goal = parsed.get("pred_goal", [])
        if not isinstance(pred_goal, list):
            return []
        return pred_goal

    def predict_goal(
        self,
        sample: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        prompt = self.build_prompt(sample)
        text = self.api.vlm_request_with_format(
            prompt["systext"],
            prompt["usertext"],
            format_schema=self.schema,
            options=self.options if options is None else options,
        )
        pred_goal = self.parse_pred_goal(text)
        return normalize_pred_goal(pred_goal, sample)


def predict_goal_for_sample(
    sample: Dict[str, Any],
    organizer: Organizer,
    options: Optional[Dict[str, Any]] = None,
) -> List[str]:
    return organizer.predict_goal(sample, options=options)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/organizer_config.json",
        help="Path to organizer config JSON.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input JSON (overrides config input_path).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output JSON (overrides config output_path).",
    )
    parser.add_argument(
        "--model",
        "--organizer_model",
        dest="model",
        default=None,
        help="Organizer model name (overrides config model).",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--num_predict", type=int, default=None)
    args = parser.parse_args()

    config = load_json(args.config) if args.config else {}
    input_path = args.input or config.get("input_path")
    output_path = args.output or config.get("output_path")
    if not output_path:
        output_path = input_path
    model = args.model or config.get("model", "qwen2.5:7b")
    options = {
        "temperature": float(
            config.get("temperature", 0.2)
            if args.temperature is None
            else args.temperature
        ),
        "num_predict": int(
            config.get("num_predict", 512)
            if args.num_predict is None
            else args.num_predict
        ),
    }

    if not input_path:
        raise ValueError("Missing input path. Set --input or provide input_path in config.")

    organizer = Organizer(model, options=options)
    dataset = load_json(input_path)

    for sample in dataset:
        pred_goal = predict_goal_for_sample(
            sample,
            organizer,
        )
        sample["pred_goal"] = pred_goal

    save_json(output_path, dataset)
    print(f"Saved organizer outputs to {output_path}")


if __name__ == "__main__":
    main()
