"""
splitter.py
-----------
Temporal leave-one-out split per user.

PRD spec:
  For each user, sort interactions chronologically.
    - Last interaction    → test set
    - Second-to-last      → validation set
    - All prior           → training set

Why NOT a global random split:
  A random split leaks future interactions into training. If user A's
  January purchase is in the test set but their February purchase is in
  training, the model has seen "future" data during training — inflating
  every offline metric. Temporal leave-one-out guarantees causal ordering.

Why NOT a global time split:
  A global split (e.g., last 10% of all interactions as test) does not
  guarantee every user appears in all three splits. Users who are inactive
  in the test window are invisible at eval time — biasing metrics toward
  power users. Per-user leave-one-out guarantees every user with ≥ 3
  interactions is represented in train, val, and test.

Acceptance criterion (from PRD):
  - Every user in val and test also appears in train
  - No val or test interaction has a timestamp earlier than the
    most recent training interaction for that user

Usage:
    from data.splitter import time_split
    train, val, test = time_split(df)
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Split label constants
TRAIN = "train"
VAL   = "val"
TEST  = "test"


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Temporal leave-one-out split per user.

    Args:
        df: DataFrame from silver_labels.generate_silver_labels().
            Must have columns: user_id, asin, timestamp, silver_label.

    Returns:
        (train, val, test) as three separate DataFrames.
        Each has a 'split' column for reference.

    Notes:
        - Users with fewer than 3 interactions cannot produce all three
          splits. They are assigned all interactions to train.
        - The returned DataFrames are sorted by timestamp within each split.
    """
    logger.info("Computing temporal leave-one-out split per user...")

    df = df.copy()
    df = df.sort_values(["user_id", "timestamp"], ascending=True)

    split_labels = pd.Series(index=df.index, dtype=str)

    user_groups = df.groupby("user_id", sort=False)

    insufficient_users = 0
    for user_id, group in user_groups:
        idx = group.index.tolist()  # already sorted by timestamp ascending

        if len(idx) < 3:
            # Not enough interactions for a 3-way split — all go to train
            split_labels[idx] = TRAIN
            insufficient_users += 1
            continue

        # Last  → test, second-to-last → val, rest → train
        split_labels[idx[-1]]  = TEST
        split_labels[idx[-2]]  = VAL
        split_labels[idx[:-2]] = TRAIN

    if insufficient_users:
        logger.warning(
            f"  {insufficient_users:,} users had < 3 interactions — "
            f"assigned entirely to train"
        )

    df["split"] = split_labels

    train = df[df["split"] == TRAIN].copy()
    val   = df[df["split"] == VAL].copy()
    test  = df[df["split"] == TEST].copy()

    _validate_split(train, val, test)
    _log_split_stats(train, val, test)

    return train, val, test


def _validate_split(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
) -> None:
    """
    Assert that no future data leaks from val/test into train.

    Checks:
      1. Every user in val appears in train.
      2. Every user in test appears in train.
      3. For each user, their val interaction timestamp > their most recent
         train interaction timestamp.
      4. For each user, their test interaction timestamp > their val
         interaction timestamp.
    """
    train_users = set(train["user_id"])
    val_users   = set(val["user_id"])
    test_users  = set(test["user_id"])

    # Check 1 & 2: no unseen users in val or test
    val_only  = val_users  - train_users
    test_only = test_users - train_users
    if val_only:
        raise AssertionError(
            f"Temporal leak: {len(val_only)} users in val but not in train: "
            f"{list(val_only)[:5]}..."
        )
    if test_only:
        raise AssertionError(
            f"Temporal leak: {len(test_only)} users in test but not in train: "
            f"{list(test_only)[:5]}..."
        )

    # Check 3 & 4: timestamps are causally ordered per user
    train_max_ts = train.groupby("user_id")["timestamp"].max().rename("train_max")
    val_ts       = val.set_index("user_id")["timestamp"].rename("val_ts")
    test_ts      = test.set_index("user_id")["timestamp"].rename("test_ts")

    check = pd.concat([train_max_ts, val_ts, test_ts], axis=1).dropna()

    leaky_val = check[check["val_ts"] < check["train_max"]]
    if len(leaky_val):
        raise AssertionError(
            f"Temporal leak: {len(leaky_val)} users have val timestamp "
            f"≤ most recent train timestamp"
        )

    leaky_test = check[check["test_ts"] < check["val_ts"]]
    if len(leaky_test):
        raise AssertionError(
            f"Temporal leak: {len(leaky_test)} users have test timestamp "
            f"≤ val timestamp"
        )

    logger.info("  Split validation passed — no temporal leakage detected")


def _log_split_stats(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
) -> None:
    total = len(train) + len(val) + len(test)
    logger.info(
        f"  Split sizes:\n"
        f"    Train: {len(train):>8,}  ({len(train)/total:.1%})\n"
        f"    Val:   {len(val):>8,}  ({len(val)/total:.1%})\n"
        f"    Test:  {len(test):>8,}  ({len(test)/total:.1%})\n"
        f"    Total: {total:>8,}\n"
        f"  Users in train: {train['user_id'].nunique():,} | "
        f"val: {val['user_id'].nunique():,} | "
        f"test: {test['user_id'].nunique():,}"
    )
