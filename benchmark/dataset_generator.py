#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BDDL -> JSON dataset converter (persona + preference generated via VLMAPI.vlm_request)

Output schema (per item in JSON array):
{
  "problem": str,
  "persona": str,
  "persona_description": str,
  "objects": {obj_name: obj_type, ...},
  "init": ["(pred a b)", ...],
  "perference": ["English pref 1", ...],   # 3-5 items
  "goal": ["(pred a b)", ...]              # extracted binary goal facts
}

- Uses your provided VLMAPI interface directly.
- Generates good systext/usertext.
- Robustly parses model output (expects strict JSON, but has fallback extraction).

Usage:
  python bddl_to_dataset_vlm.py \
      --input_dir /path/to/bddl \
      --output_json /path/to/dataset.json \
      --model qwen2.5vl:32b \
      --max_files 0

Notes:
- This script assumes BDDL style similar to your examples.
- For complex goals (forall/exists), we extract inner binary literals as goal list.
"""

import os
import re
import json
import argparse
import random
from typing import Dict, List, Any, Optional, Tuple

# ---- Import your VLMAPI from your project ----
# If VLMAPI is in the same file, remove this import and paste VLMAPI above.
# from your_module_path import VLMAPI, PROMPT_CONFIG
#
# Here we assume the user already has VLMAPI + PROMPT_CONFIG in scope
# via their project; so we only import them.

from ollama_call import VLMAPI, PROMPT_CONFIG  # <-- CHANGE this import to your actual path

# -----------------------------
# Parsing utilities
# -----------------------------

OBJ_LINE_RE = re.compile(r"^\s*([^\s]+)\s*-\s*([^\s]+)\s*$")
FACT_RE = re.compile(r"\(\s*([^\s()]+)\s+([^\s()]+)\s+([^\s()]+)\s*\)")

def strip_comments(text: str) -> str:
    return re.sub(r";.*", "", text)

def find_block(text: str, key: str) -> str:
    """
    Extract a balanced parenthesis block starting at '(:key'
    """
    idx = text.find(f"(:{key}")
    if idx < 0:
        return ""
    start = idx
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return ""

def parse_problem(text: str) -> str:
    m = re.search(r"\(\s*define\s*\(\s*problem\s+([^\s()]+)\s*\)", text)
    return m.group(1) if m else "unknown_problem"

def parse_objects(objects_block: str) -> Dict[str, str]:
    if not objects_block:
        return {}
    inner = objects_block
    inner = inner[inner.find("(:objects") + len("(:objects"):]
    inner = inner[:-1]  # drop final ')'

    out: Dict[str, str] = {}
    for line in inner.splitlines():
        line = line.strip()
        if not line:
            continue

        # Case 1: "a - type"
        m = OBJ_LINE_RE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
            continue

        # Case 2: "a b c - type"
        if "-" in line:
            parts = [p.strip() for p in line.split("-")]
            if len(parts) == 2:
                names_part, typ = parts[0].strip(), parts[1].strip()
                for n in names_part.split():
                    out[n] = typ

    return out

def parse_facts(block: str) -> List[str]:
    if not block:
        return []
    facts: List[str] = []
    for m in FACT_RE.finditer(block):
        pred, a1, a2 = m.group(1), m.group(2), m.group(3)
        facts.append(f"({pred} {a1} {a2})")
    return facts

def unique_preserve(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def parse_goal_facts(goal_block: str) -> List[str]:
    # Extract binary literals inside goal. Works even if goal uses (and ...) or (forall ...)
    return unique_preserve(parse_facts(goal_block))


# -----------------------------
# Personas (you can expand)
# -----------------------------
PERSONA_POOL = [
    (
        "organized_scholar",
        "An academically minded and highly organized person who values systematic categorization and long-term order. "
        "This user prefers that reading and paper materials are carefully stored, the environment is uncluttered, "
        "and the living room supports focused activities such as reading or studying."
    ),
    (
        "minimalist_aesthetic",
        "A visually driven minimalist who prioritizes openness, simplicity, and clean sightlines. "
        "This user prefers items to be removed from the floor but dislikes overusing storage furniture, "
        "favoring accessible and visually balanced arrangements over strict categorization."
    ),
    (
        "practical_family_member",
        "A pragmatic and efficiency-oriented household member who values safety, functionality, and ease of maintenance. "
        "This user prefers quick and sensible tidying actions that keep frequently used items accessible while ensuring "
        "the living room remains safe and usable for everyday family activities."
    ),
]

def choose_persona(seed: int, idx: int) -> Tuple[str, str]:
    random.seed(seed + idx)
    return random.choice(PERSONA_POOL)


# -----------------------------
# VLM prompting (systext/usertext)
# -----------------------------

def build_systext_for_preferences() -> str:
    # You can also load from PROMPT_CONFIG if you want; but we craft a strong system prompt here.
    return (
        "You are a dataset generator for household robot rearrangement tasks.\n"
        "You will be given a BDDL problem context (objects, initial state, and a goal specification) and a user persona.\n"
        "Your job: write 3 to 5 sub-preferences in English that reflect what the user cares about.\n"
        "Constraints:\n"
        "- Preferences must be consistent with the GOAL and must not contradict INIT.\n"
        "- Preferences should be general and user-facing (not formal logic).\n"
        "- Do NOT restate the goal formulas verbatim.\n"
        "- Output MUST be strict JSON with exactly one key: \"perference\" mapping to an array of 3-5 strings.\n"
        "- No extra keys, no extra text, no markdown.\n"
    )

def build_usertext_for_preferences(problem: str,
                                  persona: str,
                                  persona_description: str,
                                  objects: Dict[str, str],
                                  init_facts: List[str],
                                  goal_facts: List[str]) -> str:
    obj_lines = "\n".join([f"- {k}: {v}" for k, v in objects.items()])
    init_lines = "\n".join([f"- {x}" for x in init_facts])
    goal_lines = "\n".join([f"- {x}" for x in goal_facts])

    return (
        f"Problem: {problem}\n\n"
        f"Persona: {persona}\n"
        f"Persona description: {persona_description}\n\n"
        f"Objects:\n{obj_lines}\n\n"
        f"Init facts:\n{init_lines}\n\n"
        f"Goal facts:\n{goal_lines}\n\n"
        "Task:\n"
        "Generate 3-5 concise English sub-preferences for this persona.\n"
        "Each preference should be a single sentence.\n"
        "Return ONLY strict JSON like:\n"
        "{\"perference\": [\"...\", \"...\", \"...\"]}\n"
    )

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Model may return extra text; try best-effort JSON extraction.
    """
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def validate_preferences(perf: Any) -> Optional[List[str]]:
    if not isinstance(perf, list):
        return None
    cleaned = []
    for x in perf:
        if isinstance(x, str):
            s = x.strip()
            if s:
                cleaned.append(s)
    cleaned = cleaned[:5]
    if len(cleaned) < 3:
        return None
    return cleaned

def heuristic_preferences(problem: str, goal_facts: List[str]) -> List[str]:
    # deterministic fallback if model fails
    prefs: List[str] = []
    if any("(inside" in g for g in goal_facts):
        prefs.append("Loose items should be stored neatly inside the designated storage furniture.")
    if any("(ontop" in g for g in goal_facts):
        prefs.append("Display or decorative items should be placed on appropriate furniture surfaces.")
    prefs.append("Items should not be left scattered on the floor after tidying.")
    while len(prefs) < 3:
        prefs.append("The final arrangement should look tidy and organized.")
    return prefs[:5]

def generate_preferences_with_vlm(llm: VLMAPI,
                                 problem: str,
                                 persona: str,
                                 persona_description: str,
                                 objects: Dict[str, str],
                                 init_facts: List[str],
                                 goal_facts: List[str],
                                 max_tokens: int = 600) -> List[str]:
    # Use VLMAPI.vlm_request (NOT format mode) per your instruction
    systext = build_systext_for_preferences()
    usertext = build_usertext_for_preferences(
        problem, persona, persona_description, objects, init_facts, goal_facts
    )

    # You can reuse PROMPT_CONFIG if desired, but we supply our own for dataset generation.
    resp = llm.vlm_request(
        systext=systext,
        usertext=usertext,
        max_tokens=max_tokens,
        retry_limit=3
    )

    parsed = extract_json_object(resp)
    if parsed and "perference" in parsed:
        prefs = validate_preferences(parsed["perference"])
        if prefs:
            return prefs

    return heuristic_preferences(problem, goal_facts)


# -----------------------------
# Conversion pipeline
# -----------------------------

def bddl_to_record(bddl_text: str,
                   llm: VLMAPI,
                   persona: str,
                   persona_description: str,
                   max_tokens: int) -> Dict[str, Any]:
    bddl_text = strip_comments(bddl_text)

    problem = parse_problem(bddl_text)
    objects = parse_objects(find_block(bddl_text, "objects"))
    init_facts = unique_preserve(parse_facts(find_block(bddl_text, "init")))
    goal_facts = parse_goal_facts(find_block(bddl_text, "goal"))

    perference = generate_preferences_with_vlm(
        llm=llm,
        problem=problem,
        persona=persona,
        persona_description=persona_description,
        objects=objects,
        init_facts=init_facts,
        goal_facts=goal_facts,
        max_tokens=max_tokens
    )

    return {
        "problem": problem,
        "persona": persona,
        "persona_description": persona_description,
        "objects": objects,
        "init": init_facts,
        "perference": perference,
        "goal": goal_facts
    }

def list_files(input_dir: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(root, fn))
    out.sort()
    return out

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def write_json(records: List[Dict[str, Any]], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {len(records)} records to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Directory with .bddl/.pddl files")
    ap.add_argument("--output_json", required=True, help="Output dataset JSON path")
    ap.add_argument("--exts", default=".bddl,.pddl", help="Comma separated extensions")
    ap.add_argument("--model", default="qwen2.5vl:32b", help="Model name for VLMAPI")
    ap.add_argument("--persona_seed", type=int, default=123, help="Seed for persona assignment")
    ap.add_argument("--max_files", type=int, default=0, help="Limit files (0 means all)")
    ap.add_argument("--max_tokens", type=int, default=600, help="Max tokens for preference generation")
    args = ap.parse_args()

    exts = tuple([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    files = list_files(args.input_dir, exts)
    if args.max_files and args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        raise FileNotFoundError(f"No files found under {args.input_dir} with exts={exts}")

    llm = VLMAPI(args.model)

    records: List[Dict[str, Any]] = []
    for i, path in enumerate(files):
        with open(path, "r", encoding="utf-8") as f:
            bddl_text = f.read()

        persona, persona_desc = choose_persona(args.persona_seed, i)
        rec = bddl_to_record(
            bddl_text=bddl_text,
            llm=llm,
            persona=persona,
            persona_description=persona_desc,
            max_tokens=args.max_tokens
        )
        records.append(rec)
        print(f"[{i+1}/{len(files)}] {os.path.basename(path)} -> {rec['problem']} ({rec['persona']}) prefs={len(rec['perference'])}")

    write_json(records, args.output_json)


if __name__ == "__main__":
    main()
