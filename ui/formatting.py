"""Pure presentation helpers; no research execution or persistence."""


def duration(milliseconds: float | None, *, missing: str = "Not executed") -> str:
    if milliseconds is None:
        return missing
    if milliseconds < 1000:
        return f"{milliseconds:,.1f} ms"
    return f"{milliseconds / 1000:,.2f} s"


def label(value: str | None) -> str:
    return value.replace("_", " ").upper() if value else "Not available"


def optional(value, *, missing: str = "Not available") -> str:
    return missing if value is None else str(value)


def tokens(value: int | None) -> str:
    return "Not reported" if value is None else f"{value:,}"


def rate(value: float | None) -> str:
    return "Not available" if value is None else f"{value:.1%}"


def reason(value: str | None) -> str:
    return value.replace("_", " ") if value else "Not available"


def recent_rows(rows: list[dict]) -> list[dict]:
    """Allowlist operational fields before sending any data to the browser."""
    return [{
        "Timestamp (UTC)": row.get("started_at_utc"),
        "Query hash": (row.get("query_hash") or "")[:10],
        "Status": label(row.get("status")),
        "Route": label(row.get("selected_route")),
        "Cache Status": label(row.get("cache_status")),
        "RAG Used": "Yes" if row.get("rag_used") else "No",
        "DDGS Calls": row.get("ddgs_call_count"),
        "Web Results": row.get("web_result_count"),
        "Total Time": duration(row.get("total_ms"), missing="Not available"),
        "Tokens": tokens(row.get("total_tokens")),
    } for row in rows]


def markdown_table(rows: list[dict]) -> str:
    """Render compact operational tables without a dataframe dependency.

    Escape cells so stored metadata cannot inject Markdown structure or links.
    """
    if not rows:
        return ""

    def cell(value):
        text = optional(value).replace("\n", " ").replace("\r", " ")
        for character in ("\\", "|", "*", "_", "[", "]", "<", ">", "`"):
            text = text.replace(character, "\\" + character)
        return text

    columns = list(rows[0])
    lines = ["| " + " | ".join(cell(key) for key in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(row.get(key)) for key in columns) + " |" for row in rows)
    return "\n".join(lines)
