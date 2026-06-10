"""
filters.py
----------
Cold-start filtering for the Amazon Reviews dataset.

The problem: removing cold-start items creates cold-start users, and
vice versa. A single pass is not sufficient — items filtered out may
leave some users below the user threshold, requiring another user pass.

Solution: iterate (items first, then users) until the interaction
matrix is stable (no rows removed in a pass).

PRD spec:
  - min_user_reviews = 5   (users with fewer interactions removed)
  - min_item_reviews = 10  (items with fewer interactions removed)
  - max_passes       = 3   (empirically sufficient for this dataset)

Usage:
    from data.filters import remove_cold_start
    df = remove_cold_start(df)
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# PRD-specified thresholds
MIN_USER_REVIEWS = 5
MIN_ITEM_REVIEWS = 10
MAX_PASSES = 3


def remove_cold_start(
    df: pd.DataFrame,
    min_user_reviews: int = MIN_USER_REVIEWS,
    min_item_reviews: int = MIN_ITEM_REVIEWS,
    max_passes: int = MAX_PASSES,
) -> pd.DataFrame:
    """
    Iteratively remove cold-start users and items until stable.

    Args:
        df:               DataFrame from cleaner.clean_reviews()
        min_user_reviews: Minimum number of interactions a user must have
        min_item_reviews: Minimum number of interactions an item must have
        max_passes:       Maximum number of filter iterations

    Returns:
        Filtered DataFrame. Index is reset.

    Example log output:
        Pass 1: removed 12,431 items → 3,211 users
        Pass 2: removed 189 items → 44 users
        Pass 3: stable (0 removed)
    """
    logger.info(
        f"Cold-start filtering: "
        f"min_user={min_user_reviews}, min_item={min_item_reviews}, "
        f"max_passes={max_passes}"
    )

    initial_count = len(df)
    df = df.copy()

    for pass_num in range(1, max_passes + 1):
        before = len(df)

        # ── Step 1: Filter items ──────────────────────────────────────────────
        # Items first — they are typically more sparse than users
        item_counts = df["asin"].value_counts()
        valid_items = item_counts[item_counts >= min_item_reviews].index
        df = df[df["asin"].isin(valid_items)]
        items_removed = before - len(df)

        # ── Step 2: Filter users ──────────────────────────────────────────────
        after_items = len(df)
        user_counts = df["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_reviews].index
        df = df[df["user_id"].isin(valid_users)]
        users_removed = after_items - len(df)

        total_removed = before - len(df)

        logger.info(
            f"  Pass {pass_num}: "
            f"removed {items_removed:,} item interactions + "
            f"{users_removed:,} user interactions = "
            f"{total_removed:,} total | "
            f"remaining: {len(df):,}"
        )

        # Stable — no rows removed this pass
        if total_removed == 0:
            logger.info(f"  Stable after {pass_num} pass(es)")
            break
    else:
        # Ran all passes without stabilising — log a warning but continue
        logger.warning(
            f"  Did not stabilise after {max_passes} passes. "
            f"Consider increasing max_passes or reviewing thresholds."
        )

    df = df.reset_index(drop=True)

    total_removed = initial_count - len(df)
    logger.info(
        f"Cold-start filtering complete: "
        f"removed {total_removed:,} interactions "
        f"({total_removed / initial_count:.1%} of raw). "
        f"Final: {len(df):,} interactions | "
        f"{df['user_id'].nunique():,} users | "
        f"{df['asin'].nunique():,} items"
    )
    return df


def log_interaction_stats(df: pd.DataFrame, label: str = "") -> None:
    """
    Log descriptive statistics about the interaction distribution.
    Useful for checking that the filtered dataset looks reasonable.
    """
    user_counts = df.groupby("user_id").size()
    item_counts = df.groupby("asin").size()

    tag = f"[{label}] " if label else ""
    logger.info(
        f"{tag}Interaction stats:\n"
        f"  Users:  {df['user_id'].nunique():,}  "
        f"(median {user_counts.median():.0f} reviews/user, "
        f"max {user_counts.max()})\n"
        f"  Items:  {df['asin'].nunique():,}  "
        f"(median {item_counts.median():.0f} reviews/item, "
        f"max {item_counts.max()})\n"
        f"  Total interactions: {len(df):,}\n"
        f"  Sparsity: {1 - len(df) / (df['user_id'].nunique() * df['asin'].nunique()):.4%}"
    )
