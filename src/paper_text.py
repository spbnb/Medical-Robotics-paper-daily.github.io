PLACEHOLDER_TEXTS = {
    "",
    ".",
    "..",
    "...",
    "\u2026",
    "n/a",
    "na",
    "none",
    "null",
    "-",
    "--",
}


def clean_display_text(value) -> str:
    """Return displayable text, treating placeholder-only values as empty."""
    if value is None:
        return ""

    text = str(value).strip()
    if text.casefold() in PLACEHOLDER_TEXTS:
        return ""

    punctuation_only = {".", "\u2026", "\u3002", "-", "_"}
    if text and all(char in punctuation_only for char in text):
        return ""

    return text


def normalize_paper_display_fields(paper: dict) -> dict:
    out = dict(paper)
    for key in ("summary", "summary_zh", "tldr", "tldr_zh"):
        out[key] = clean_display_text(out.get(key))
    return out


def first_meaningful_text(paper: dict, keys) -> str:
    for key in keys:
        text = clean_display_text(paper.get(key))
        if text:
            return text
    return ""
