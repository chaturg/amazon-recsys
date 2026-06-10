"""
loader.py — chunked version with early filtering
"""
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)
REQUIRED_COLS = ["user_id","asin","rating","text","helpful_vote","verified_purchase","timestamp"]
COL_ALIASES = {
    "reviewerID":"user_id","reviewText":"text","overall":"rating",
    "unixReviewTime":"timestamp","reviewTime":"timestamp",
    "helpful":"helpful_vote","verified":"verified_purchase",
}
CHUNK_SIZE = 200_000

def load_reviews(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    logger.info(f"Loading reviews from {path} (chunked, {CHUNK_SIZE:,} rows/chunk)")
    chunks = []
    total_raw = 0
    with pd.read_json(path, lines=True, chunksize=CHUNK_SIZE) as reader:
        for i, chunk in enumerate(reader):
            total_raw += len(chunk)
            chunk = chunk.rename(columns=COL_ALIASES)
            available = [c for c in REQUIRED_COLS if c in chunk.columns]
            chunk = chunk[available].copy()
            if "helpful_vote"      not in chunk.columns: chunk["helpful_vote"]      = 0
            if "verified_purchase" not in chunk.columns: chunk["verified_purchase"] = False
            if "text"              not in chunk.columns: chunk["text"]              = ""
            chunk["rating"]            = pd.to_numeric(chunk["rating"], errors="coerce")
            chunk["helpful_vote"]      = pd.to_numeric(chunk["helpful_vote"], errors="coerce").fillna(0).astype(int)
            chunk["verified_purchase"] = chunk["verified_purchase"].astype(bool)
            chunk["text"]              = chunk["text"].fillna("").astype(str)
            if chunk["timestamp"].dtype == object:
                chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce").astype("int64") // 10**9
            else:
                chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
            chunk = chunk.dropna(subset=["user_id","asin","rating","timestamp"])
            chunks.append(chunk)
            if (i+1) % 10 == 0:
                logger.info(f"  Loaded {total_raw:,} rows so far ({i+1} chunks)...")
    logger.info(f"  All chunks loaded. Combining {total_raw:,} raw rows...")
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    import gc; gc.collect()

    # Early rough filter — drop users with only 1 interaction across full dataset
    # This cuts ~50% of rows before dedup and cold-start filtering
    logger.info("  Applying early user frequency filter (min 2 interactions)...")
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    before = len(df)
    df = df[df["user_id"].isin(valid_users)]
    logger.info(f"  Early filter removed {before - len(df):,} rows from single-interaction users")

    df = df.sort_values("timestamp", ascending=True)
    before = len(df)
    df = df.drop_duplicates(subset=["user_id","asin"], keep="last")
    logger.info(f"  Removed {before - len(df):,} duplicates")
    df = df.reset_index(drop=True)
    logger.info(f"  Final: {len(df):,} interactions | {df['user_id'].nunique():,} users | {df['asin'].nunique():,} items")
    return df
