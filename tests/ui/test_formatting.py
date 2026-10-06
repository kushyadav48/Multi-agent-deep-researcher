import pytest

from ui.formatting import duration, label, markdown_table, optional, rate, recent_rows, tokens


@pytest.mark.parametrize("value,expected", [
    (None, "Not executed"), (0, "0.0 ms"), (.25, "0.2 ms"),
    (999, "999.0 ms"), (1000, "1.00 s"), (1250, "1.25 s"),
])
def test_duration_preserves_missing_and_executed_zero(value, expected):
    assert duration(value) == expected


def test_optional_values_and_labels():
    assert duration(None, missing="Not available") == "Not available"
    assert label("SEMANTIC_HIT") == "SEMANTIC HIT"
    assert label("fast") == "FAST" and label(None) == "Not available"
    assert tokens(None) == "Not reported" and tokens(0) == "0" and tokens(1234) == "1,234"
    assert rate(None) == "Not available" and rate(.5) == "50.0%"
    assert optional(None) == "Not available" and optional(0) == "0"


def test_recent_rows_exclude_content_even_if_store_shape_expands():
    rows = recent_rows([dict(query_hash="abcdef0123456789", raw_query="SECRET",
                             final_answer="SECRET", agent_outputs="SECRET", rag_text="SECRET",
                             web_snippets="SECRET", total_tokens=None, total_ms=None)])
    assert rows[0]["Query hash"] == "abcdef0123"
    assert rows[0]["Tokens"] == "Not reported"
    assert rows[0]["Total Time"] == "Not available"
    assert "SECRET" not in str(rows)


def test_table_escapes_metadata_markup():
    assert markdown_table([]) == ""
    table = markdown_table([{"Status": "a|b\n[link](<url>)"}])
    assert "a\\|b \\[link\\](\\<url\\>)" in table
