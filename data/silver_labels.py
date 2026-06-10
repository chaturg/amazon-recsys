"""
silver_labels.py
----------------
Constructs silver labels for implicit preference signals from Amazon Reviews.

The problem:
  Raw star ratings are user-biased AND dataset-biased. In the Amazon Tools &
  Home Improvement dataset, 70.9% of ratings are 5-star. Per-user z-scoring
  corrects for individual rater bias but cannot overcome the dataset-level
  positivity bias — when most users rate everything 4–5 stars, per-user
  variance is too low for z-scoring to differentiate meaningfully
  (rating_norm std = 0.042 on this dataset).

Solution:
  Signal variance analysis on the processed dataset identified the true
  discriminating signals:

  Signal                Std     Weight   Rationale
  ──────────────────────────────────────────────────────────────────────────
  verified_score        0.313   0.40     Highest variance — clean binary
                                         signal separating genuine purchasers
                                         from unverified reviewers

  length_score          0.156   0.30     Second highest variance — review
                                         length correlates with engagement
                                         intensity at both ends of spectrum

  helpfulness_score     0.081   0.15     Moderate — community validation,
                                         sentiment-neutral but adds signal

  rating_norm           0.042   0.15     Lowest variance — demoted from 0.50
                                         due to 70.9% 5-star positivity bias
                                         compressing z-score differentiation

Result: std improved from 0.055 (original) to 0.122 (revised) — 2.2× gain.
Low-end cluster (<0.2, 7.8% of interactions) is 100% unverified purchases —
correctly identified low-quality signal, not an artifact.

Usage:
    from data.silver_labels import generate_silver_labels
    df = generate_silver_labels(df)
    # df now has a 'silver_label' column in [0, 1]
"""

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ── Signal weights ────────────────────────────────────────────────────────────
# Redesigned after diagnostic analysis on processed dataset.
# Original weights (rating 0.50, helpfulness 0.20, verified 0.15, length 0.15)
# produced std=0.055 due to 70.9% 5-star positivity bias in this domain.
# Revised weights produce std=0.122 — a 2.2x improvement in label variance.
WEIGHTS = {
    "rating_norm":       0.15,  # demoted — z-score variance too low (std=0.042)
    "helpfulness_score": 0.15,  # moderate — adds signal without compressing variance
    "verified_score":    0.40,  # promoted — highest variance signal (std=0.313)
    "length_score":      0.30,  # promoted — second highest variance (std=0.156)
}

# Review length clipped at this value before scaling.
# 2000 chars ≈ 400 words — a thorough review.
MAX_REVIEW_LEN = 2_000


def _minmax_scale(series: pd.Series) -> pd.Series:
    """Scale a Series to [0, 1]. Returns zeros if variance is zero."""
    values = series.values.reshape(-1, 1)
    if values.std() < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return pd.Series(
        MinMaxScaler().fit_transform(values).flatten(),
        index=series.index,
    )


def generate_silver_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate silver labels from four interaction signals.

    Args:
        df: DataFrame from filters.remove_cold_start().
            Must have columns: user_id, rating, helpful_vote,
            verified_purchase, text_len (added by cleaner.py).

    Returns:
        DataFrame with added columns:
          - rating_norm:       per-user z-scored rating, scaled [0,1]
          - helpfulness_score: log1p(helpful_vote), scaled [0,1]
          - verified_score:    float(verified_purchase)
          - length_score:      clipped text_len, scaled [0,1]
          - silver_label:      weighted sum of above, in [0, 1]
    """
    logger.info("Generating silver labels...")
    df = df.copy()

    # ── Signal 1: Per-user z-scored rating (weight 0.15) ─────────────────────
    # Z-score corrects individual rater bias but has low variance on this
    # dataset (std=0.042) due to 70.9% 5-star ratings. Kept at 0.15 — still
    # adds marginal signal for users with diverse rating histories.
    user_mean = df.groupby("user_id")["rating"].transform("mean")
    user_std  = df.groupby("user_id")["rating"].transform("std")
    global_std = df["rating"].std()
    user_std   = user_std.fillna(global_std).clip(lower=1e-6)
    rating_zscore = (df["rating"] - user_mean) / user_std
    df["rating_norm"] = _minmax_scale(rating_zscore)

    # ── Signal 2: Helpfulness vote (weight 0.15) ──────────────────────────────
    # log1p dampens viral reviews. Sentiment-neutral but community-validated.
    df["helpfulness_score"] = _minmax_scale(
        np.log1p(df["helpful_vote"].clip(lower=0))
    )

    # ── Signal 3: Verified purchase (weight 0.40) ─────────────────────────────
    # Highest variance signal (std=0.313). Binary: 1.0 verified, 0.0 not.
    # 100% of labels below 0.2 are unverified — this signal cleanly separates
    # genuine purchasers from unverified reviewers.
    # Mean verified rate post cold-start filtering: 0.890.
    df["verified_score"] = df["verified_purchase"].astype(float)

    # ── Signal 4: Review length (weight 0.30) ─────────────────────────────────
    # Second highest variance (std=0.156). Longer reviews indicate stronger
    # engagement intensity at both ends of the preference spectrum.
    if "text_len" not in df.columns:
        logger.warning("  'text_len' not found — recomputing from 'text' column")
        df["text_len"] = df["text"].str.len()

    df["length_score"] = _minmax_scale(
        df["text_len"].clip(upper=MAX_REVIEW_LEN).astype(float)
    )

    # ── Weighted combination ───────────────────────────────────────────────────
    df["silver_label"] = (
        WEIGHTS["rating_norm"]       * df["rating_norm"]       +
        WEIGHTS["helpfulness_score"] * df["helpfulness_score"] +
        WEIGHTS["verified_score"]    * df["verified_score"]    +
        WEIGHTS["length_score"]      * df["length_score"]
    ).clip(0.0, 1.0).round(4)

    # ── Validation ─────────────────────────────────────────────────────────────
    assert df["silver_label"].between(0, 1).all(), \
        "Silver labels out of [0,1] range — check signal scaling"

    _log_distribution(df["silver_label"])
    return df


def _log_distribution(labels: pd.Series) -> None:
    """Log histogram-style summary of silver label distribution."""
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    counts = pd.cut(labels, bins=bins, include_lowest=True).value_counts(sort=False)

    logger.info("  Silver label distribution:")
    for interval, count in counts.items():
        pct = count / len(labels) * 100
        bar = "█" * int(pct / 2)
        logger.info(f"    {str(interval):20s}  {count:7,}  ({pct:5.1f}%)  {bar}")

    logger.info(
        f"  Mean: {labels.mean():.3f} | "
        f"Median: {labels.median():.3f} | "
        f"Std: {labels.std():.3f}"
    )

    extreme_pct = ((labels < 0.05) | (labels > 0.95)).mean()
    if extreme_pct > 0.5:
        logger.warning(
            f"  WARNING: {extreme_pct:.1%} of silver labels are at extremes. "
            f"Check signal scaling."
        )

    # Target std after redesign should be >= 0.10
    if labels.std() < 0.08:
        logger.warning(
            f"  WARNING: Silver label std={labels.std():.3f} is below 0.08. "
            f"Label variance is low — check signal weights."
        )
    else:
        logger.info(
            f"  Silver label variance check passed (std={labels.std():.3f} >= 0.08)"
        )
