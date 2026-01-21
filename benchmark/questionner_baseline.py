#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline test: Qwen questioner + Oracle answerer (goal-visible).

Usage:
  python benchmark/questionner_baseline.py --config benchmark/configs/baseline_config.json
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

from ollama_call import VLMAPI
from oracle_answerer import OracleAnswerer


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
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


def format_objects(objects: Dict[str, str]) -> str:
    return "\n".join([f"- {k}: {v}" for k, v in objects.items()])


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


def build_question_prompt(sample: Dict[str, Any]) -> Dict[str, str]:
    objects = format_objects(sample["objects"])
    init = format_facts(sample["init"])
    history = format_qa_history(sample.get("qa_history", []))

    systext = (
        "You are a service robot that helps with personalized household tasks (e.g., organizing a fridge).\n"
        "Because user preferences vary, you must proactively ask questions to learn preferences.\n"
        "You can see the objects, initial state, and QA history, but you cannot see the hidden goal.\n"
        "Ask EXACTLY ONE concise, high-value question that reduces uncertainty about user preferences.\n"
        "Avoid repeating questions and avoid assuming the goal.\n"
        "Output must be strict JSON with exactly one key: \"question\".\n"
    )
    usertext = (
        "Context:\n"
        "We have limited question turns to learn the user's preferences for this household task.\n\n"
        f"Task: {sample.get('task', 'unknown')}\n\n"
        f"Objects:\n{objects}\n\n"
        f"Init facts:\n{init}\n\n"
        f"QA history:\n{history}\n\n"
        "Ask one question that best helps infer the user's preferences for arranging items."
    )
    return {"systext": systext, "usertext": usertext}


class QuestionnerBaseline:
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
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        }

    def build_prompt(self, sample: Dict[str, Any]) -> Dict[str, str]:
        return build_question_prompt(sample)

    def parse_question(self, text: str) -> str:
        parsed = parse_json_maybe(text)
        return parsed.get("question", "")

    def ask(self, sample: Dict[str, Any]) -> str:
        prompt = self.build_prompt(sample)
        text = self.api.vlm_request_with_format(
            prompt["systext"],
            prompt["usertext"],
            format_schema=self.schema,
            options=self.options,
        )
        question = self.parse_question(text).strip()
        return question or "Where should I put the remaining items?"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="benchmark/configs/baseline_config.json",
        help="Path to baseline config JSON.",
    )
    args = parser.parse_args()

    config = load_json(args.config)
    input_path = config["input_path"]
    output_path = config["output_path"]
    question_model = config["question_model"]
    oracle_model = config["oracle_model"]
    max_turns = int(config.get("max_turns", 10))
    options = {
        "temperature": float(config.get("temperature", 0.2)),
        "num_predict": int(config.get("num_predict", 256)),
    }

    questioner = QuestionnerBaseline(question_model, options=options)
    oracle = OracleAnswerer(oracle_model, options=options)
    dataset = load_json(input_path)
    results = []

    for sample in dataset:
        qa_history = list(sample.get("qa_history", []))
        generated = []

        for turn in range(max_turns):
            question = questioner.ask({**sample, "qa_history": qa_history})

            answer = oracle.answer(sample, question, qa_history=qa_history)

            turn_id = len(qa_history) + 1
            qa_item = {"turn_id": turn_id, "question": question, "answer": answer}
            qa_history.append(qa_item)
            generated.append(qa_item)

        results.append(
            {
                "task": sample.get("task"),
                "objects": sample.get("objects"),
                "init": sample.get("init"),
                "goal": sample.get("goal"),
                "qa_history": sample.get("qa_history"),
                "generated_qa": generated,
                "final_qa_history": qa_history,
            }
        )

    save_json(output_path, results)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
