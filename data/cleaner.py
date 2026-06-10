"""
cleaner.py
----------
Text normalization for Amazon review bodies.

Responsibilities:
  - Strip HTML tags left over from the raw dataset
  - Collapse whitespace
  - Truncate excessively long reviews (> MAX_CHARS) to save memory
    downstream and prevent token overflow in the cross-encoder
  - Optionally lowercase (off by default — preserves brand names)

Usage:
    from data.cleaner import clean_reviews
    df = clean_reviews(df)
"""

import html
import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Reviews longer than this are truncated. 2 000 chars ≈ 400 tokens —
# well within the cross-encoder's 128-token budget for a snippet.
MAX_CHARS = 2_000

# Compiled regex patterns — compiled once at import time for speed
_HTML_TAG   = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """
    Clean a single review string.

    Steps:
      1. Unescape HTML entities (&amp; → &, &quot; → ", etc.)
      2. Strip HTML tags
      3. Collapse whitespace (tabs, newlines, multiple spaces → single space)
      4. Strip leading/trailing whitespace
      5. Truncate to MAX_CHARS
    """
    if not isinstance(text, str):
        return ""

    # 1. HTML entity decoding
    text = html.unescape(text)

    # 2. Remove HTML tags
    text = _HTML_TAG.sub(" ", text)

    # 3. Normalise whitespace
    text = _WHITESPACE.sub(" ", text)

    # 4. Strip
    text = text.strip()

    # 5. Truncate
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    return text


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text normalisation to the 'text' column in-place.

    Args:
        df: DataFrame produced by loader.load_reviews()

    Returns:
        Same DataFrame with 'text' column normalised.
        Also adds 'text_len' column (character count after cleaning)
        which is used as a signal in silver_labels.py.
    """
    if "text" not in df.columns:
        raise ValueError("DataFrame must have a 'text' column. Run loader.load_reviews() first.")

    logger.info("Cleaning review text...")

    before_empty = (df["text"] == "").sum()

    df = df.copy()
    df["text"] = df["text"].apply(_normalize_text)

    # Store character length — used as the engagement-depth signal
    # in silver label construction (clipped at MAX_CHARS = 1.0 after scaling)
    df["text_len"] = df["text"].str.len()

    after_empty = (df["text"] == "").sum()
    newly_empty = after_empty - before_empty
    if newly_empty > 0:
        logger.warning(f"  {newly_empty:,} reviews became empty after cleaning (were HTML-only)")

    logger.info(
        f"  Text cleaning complete. "
        f"Median length: {df['text_len'].median():.0f} chars | "
        f"Empty: {after_empty:,}"
    )
    return df
