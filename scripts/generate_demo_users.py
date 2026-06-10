"""
generate_demo_users.py
----------------------
Selects 10 representative users from the validation set and writes
their validation history as JSON files for the Gradio demo.

User selection criteria:
  u001–u006: Normal users (5–50 interactions)
  u007–u008: Sparse users (2–4 interactions) — demos adaptive alpha
  u009–u010: New users (0 interactions) — demos popularity fallback

Each JSON file contains:
  - user metadata (interaction count, verified ratio, etc.)
  - validation_history: list of items from training set
  - val_item: the held-out validation interaction
  - user_type: "normal", "sparse", or "new"

Usage:
    cd ~/cloudfiles/code/Users/casakaay/amazon-recsys
    python scripts/generate_demo_users.py
"""

import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE      = "processed"
DEMO_DIR  = "demo/users"
TITLES_PATH = "processed/item_titles.parquet"

random.seed(42)
np.random.seed(42)


def get_item_info(asin: str, titles_df: pd.DataFrame) -> dict:
    """Get item title and category from the product metadata."""
    row = titles_df[titles_df["asin"] == asin]
    if len(row) > 0:
        r = row.iloc[0]
        title    = r.get("title", "")
        category = r.get("categories", "Tools & Home Improvement")
        price    = r.get("price", None)
        return {
            "asin":     asin,
            "title":    title[:120] if isinstance(title, str) and len(title) > 5 else asin,
            "category": str(category)[:80],
            "price_usd": float(price) if isinstance(price, float) and price > 0 else None,
        }
    return {
        "asin":     asin,
        "title":    asin,
        "category": "Tools & Home Improvement",
        "price_usd": None,
    }


def build_user_json(
    uid:        str,
    user_id:    str,
    train_rows: pd.DataFrame,
    val_row:    pd.Series,
    titles_df:  pd.DataFrame,
    user_type:  str,
) -> dict:
    """Build the full user JSON for the demo."""

    # Build validation history from training interactions
    history = []
    for _, row in train_rows.sort_values("timestamp", ascending=False).head(15).iterrows():
        item_info = get_item_info(row["asin"], titles_df)

        # Get a review snippet if text is available
        text = str(row.get("text", ""))
        snippet = text[:120].strip() if len(text) > 20 else ""

        history.append({
            **item_info,
            "rating":           int(row["rating"]),
            "silver_label":     round(float(row["silver_label"]), 3),
            "verified_purchase":bool(row["verified_score"] > 0.5),
            "helpful_votes":    int(row.get("helpful_vote", 0)),
            "review_snippet":   snippet,
            "timestamp":        str(row["timestamp"]),
        })

    # Val item info
    val_item_info = get_item_info(val_row["asin"], titles_df)
    val_text = str(val_row.get("text", ""))
    val_snippet = val_text[:120].strip() if len(val_text) > 20 else ""

    return {
        "user_id":       user_id,
        "demo_id":       uid,
        "user_type":     user_type,
        "interaction_count": len(train_rows),
        "verified_ratio": round(float(train_rows["verified_score"].mean()), 3),
        "avg_rating":    round(float(train_rows["rating"].mean()), 2) if len(train_rows) > 0 else None,
        "validation_history": history,
        "val_item": {
            **val_item_info,
            "rating":           int(val_row["rating"]),
            "silver_label":     round(float(val_row["silver_label"]), 3),
            "verified_purchase":bool(val_row["verified_score"] > 0.5),
            "helpful_votes":    int(val_row.get("helpful_vote", 0)),
            "review_snippet":   val_snippet,
        },
    }


def select_demo_users(
    train_df:  pd.DataFrame,
    val_df:    pd.DataFrame,
    titles_df: pd.DataFrame,
) -> list:
    """Select 10 diverse demo users."""

    # Compute interaction counts per user
    user_counts = train_df.groupby("user_id").size().reset_index(name="count")

    # Val users only (must have val interaction)
    val_users = set(val_df["user_id"].unique())
    user_counts = user_counts[user_counts["user_id"].isin(val_users)]

    selected = []

    # ── Normal users (5–50 interactions) — 6 users ────────────────────────
    normal = user_counts[
        (user_counts["count"] >= 10) &
        (user_counts["count"] <= 50)
    ]

    # Try to select users with real titles in their history
    normal_with_titles = []
    for _, row in normal.sample(min(500, len(normal)), random_state=42).iterrows():
        uid = row["user_id"]
        user_asins = train_df[train_df["user_id"]==uid]["asin"].tolist()
        has_titles = any(
            isinstance(titles_df[titles_df["asin"]==a].iloc[0].get("title","") if len(titles_df[titles_df["asin"]==a])>0 else "", str)
            and len(str(titles_df[titles_df["asin"]==a].iloc[0].get("title","") if len(titles_df[titles_df["asin"]==a])>0 else "")) > 10
            for a in user_asins[:5]
        )
        if has_titles:
            normal_with_titles.append(uid)
        if len(normal_with_titles) >= 6:
            break

    # Fallback to random normal users
    if len(normal_with_titles) < 6:
        extra = normal.sample(
            min(6 - len(normal_with_titles), len(normal)), random_state=99
        )["user_id"].tolist()
        normal_with_titles.extend(extra)

    selected.extend([("normal", uid) for uid in normal_with_titles[:6]])

    # ── Sparse users (2–4 interactions) — 2 users ─────────────────────────
    sparse = user_counts[
        (user_counts["count"] >= 2) &
        (user_counts["count"] <= 4)
    ].sample(min(100, len(user_counts[
        (user_counts["count"] >= 2) & (user_counts["count"] <= 4)
    ])), random_state=42)

    sparse_selected = []
    for _, row in sparse.iterrows():
        uid = row["user_id"]
        sparse_selected.append(uid)
        if len(sparse_selected) >= 2:
            break

    selected.extend([("sparse", uid) for uid in sparse_selected[:2]])

    # ── New users (not in training set) — 2 users ─────────────────────────
    # These are val users with 0 training interactions
    val_only = val_df[~val_df["user_id"].isin(train_df["user_id"].unique())]
    if len(val_only) >= 2:
        new_users = val_only.sample(2, random_state=42)["user_id"].tolist()
    else:
        # Fallback: use users with only 1 interaction as "new"
        one_interaction = user_counts[user_counts["count"] == 1]
        new_users = one_interaction.sample(
            min(2, len(one_interaction)), random_state=42
        )["user_id"].tolist()

    selected.extend([("new", uid) for uid in new_users[:2]])

    logger.info(f"Selected {len(selected)} demo users:")
    for utype, uid in selected:
        count = user_counts[user_counts["user_id"]==uid]["count"].values
        n = count[0] if len(count) > 0 else 0
        logger.info(f"  {utype:8s} | interactions={n:3d} | {uid}")

    return selected


def generate_demo_users(
    train_path:  str = f"{BASE}/train.parquet",
    val_path:    str = f"{BASE}/val.parquet",
    titles_path: str = TITLES_PATH,
    output_dir:  str = DEMO_DIR,
) -> None:
    """Generate all 10 demo user JSON files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    train_df  = pd.read_parquet(train_path)
    val_df    = pd.read_parquet(val_path)
    titles_df = pd.read_parquet(titles_path) if Path(titles_path).exists() \
        else pd.DataFrame(columns=["asin","title","categories","price"])

    logger.info(f"  Train={len(train_df):,} | Val={len(val_df):,} | "
                f"Titles={len(titles_df):,}")

    # Set index for fast lookup
    val_idx = val_df.set_index("user_id")

    # Select 10 demo users
    selected = select_demo_users(train_df, val_df, titles_df)

    # Generate JSON for each
    for i, (user_type, user_id) in enumerate(selected):
        demo_id  = f"u{i+1:03d}"
        train_rows = train_df[train_df["user_id"] == user_id]
        val_row    = val_idx.loc[user_id] if user_id in val_idx.index else None

        if val_row is None:
            logger.warning(f"  {demo_id}: no val interaction found — skipping")
            continue

        user_data = build_user_json(
            uid       = demo_id,
            user_id   = user_id,
            train_rows= train_rows,
            val_row   = val_row,
            titles_df = titles_df,
            user_type = user_type,
        )

        out_path = Path(output_dir) / f"{demo_id}.json"
        with open(out_path, "w") as f:
            json.dump(user_data, f, indent=2, default=str)

        n_hist = len(user_data["validation_history"])
        logger.info(
            f"  {demo_id} ({user_type:6s}) | "
            f"interactions={user_data['interaction_count']:3d} | "
            f"history_items={n_hist} | "
            f"val_title={user_data['val_item']['title'][:50]}"
        )

    logger.info(f"\nAll demo users written to {output_dir}/")


if __name__ == "__main__":
    generate_demo_users()
