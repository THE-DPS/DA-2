"""Conservative text normalization utilities."""
from __future__ import annotations

import html
import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def normalize_text(text: str) -> str:
    """Normalize text without deleting sentiment-bearing negations or punctuation.

    This function intentionally avoids stop-word removal and stemming. Those
    transformations can remove meaning and should be justified experimentally.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    text = unicodedata.normalize("NFC", text)
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("\x00", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text
