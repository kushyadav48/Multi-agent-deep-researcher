from copy import deepcopy

import pytest

from evaluation.datasets import NAMES, ROOT, evidence, load, validate


def test_dataset_sizes_labels_sources_and_unique_ids():
    datasets = {name: load(name) for name in NAMES}
    assert {name: len(rows) for name, rows in datasets.items()} == dict(routing=36, semantic_cache=20, rag=10, synthesis=2)
    ids = [case["id"] for rows in datasets.values() for case in rows]
    assert len(ids) == len(set(ids))
    assert sum(case["expected_reuse"] for case in datasets["semantic_cache"]) == 10
    assert len(list((ROOT / "fixtures" / "rag").glob("*.md"))) == 6
    for name in ("mcp", "hybrid", "rag_context"):
        assert evidence(name)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("mutation", ("duplicate", "query", "id"))
def test_invalid_common_entries_rejected(name, mutation):
    rows = deepcopy(load(name))
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "id":
        rows[0]["id"] = "../escape"
    else:
        rows[0]["first" if name == "semantic_cache" else "query"] = " "
    with pytest.raises(ValueError):
        validate(name, rows)


@pytest.mark.parametrize("name,key,value", [
    ("routing", "expected_route", "AUTO"), ("semantic_cache", "expected_reuse", "positive"),
    ("rag", "expected_source", "../secret.md"), ("rag", "expected_fact", "nonexistent fact"),
    ("synthesis", "concept_groups", [[]]), ("synthesis", "classification", "UNKNOWN"),
    ("synthesis", "evidence", "unknown"),
])
def test_malformed_specific_fields_rejected(name, key, value):
    rows = load(name)
    rows[0][key] = value
    with pytest.raises(ValueError):
        validate(name, rows)


def test_synthesis_count_cannot_expand_the_generation_budget():
    rows = load("synthesis")
    rows.append(dict(rows[0], id="third"))
    with pytest.raises(ValueError):
        validate("synthesis", rows)
