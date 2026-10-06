"""Load and validate the versioned, nonsensitive curated datasets."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NAMES = ("routing", "semantic_cache", "rag", "synthesis")


def validate(name, cases):
    if name not in NAMES or not isinstance(cases, list) or not cases:
        raise ValueError("Unknown or empty dataset")
    identities = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not re.fullmatch(r"[a-z0-9_-]+", case["id"]):
            raise ValueError("Each case requires an ID")
        if case["id"] in identities:
            raise ValueError("Duplicate case ID")
        identities.add(case["id"])
        fields = ("first", "second") if name == "semantic_cache" else ("query",)
        if any(not isinstance(case.get(key), str) or not case[key].strip() for key in fields):
            raise ValueError("Queries must be nonempty strings")
        if name == "routing" and case.get("expected_route") not in ("FAST", "QUALITY"):
            raise ValueError("Invalid expected route")
        if name == "semantic_cache" and type(case.get("expected_reuse")) is not bool:
            raise ValueError("Cache labels must be booleans")
        if name == "rag":
            source = case.get("expected_source", "")
            if Path(source).name != source or not (ROOT / "fixtures" / "rag" / source).is_file():
                raise ValueError("Expected RAG source must be a fixture filename")
            if not isinstance(case.get("expected_fact"), str) or not case["expected_fact"].strip():
                raise ValueError("Expected RAG fact is required")
            if case["expected_fact"] not in (ROOT / "fixtures" / "rag" / source).read_text(encoding="utf-8"):
                raise ValueError("Expected fact must occur in the source fixture")
        if name == "synthesis":
            groups = case.get("concept_groups")
            if not isinstance(groups, list) or not groups or any(
                not isinstance(group, list) or not group or any(
                    not isinstance(term, str) or not term.strip() for term in group
                ) for group in groups
            ):
                raise ValueError("Rubric requires nonempty concept groups")
            if case.get("classification") not in ("SIMPLE", "COMPLEX"):
                raise ValueError("Invalid synthesis classification")
            evidence(case["evidence"])
    if name == "synthesis" and (len(cases) != 2 or {case["classification"] for case in cases} != {"SIMPLE", "COMPLEX"}):
        raise ValueError("Synthesis requires exactly one SIMPLE and one COMPLEX prompt")
    return cases


def load(name):
    return validate(name, json.loads((ROOT / "datasets" / f"{name}.json").read_text(encoding="utf-8")))


def evidence(name):
    if name not in ("mcp", "hybrid", "rag_context"):
        raise ValueError("Unknown controlled evidence")
    data = json.loads((ROOT / "fixtures" / "web" / f"{name}.json").read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data or any(
        not all(isinstance(row.get(key), str) and row[key].strip() for key in ("title", "href", "body"))
        for row in data
    ):
        raise ValueError("Malformed controlled web evidence")
    return data
