"""
generate_ce_training_pairs.py
------------------------------
Generates cross-encoder training pairs from the Config 3 FAISS index.

For each user in the training set:
  1. Compute user embedding from aggregated features
  2. Search Config 3 FAISS index → top-100 candidates
  3. For each candidate, record:
     - User history summary (for cross-encoder left side)
     - Item title (for cross-encoder right side)
     - Relevance label: silver_label if user interacted, 0.0 otherwise

Output: processed/ce_training_pairs.parquet
  Columns: user_id, history_summary, item_title, item_category,
           asin, relevance_label, is_positive

This runs on CPU — no GPU needed. Takes ~30-60 min for 625k users.
Run on Azure CPU compute before cross-encoder fine-tuning on Kaggle.

Usage:
    python scripts/generate_ce_training_pairs.py

    # With custom sample size (for testing)
    python scripts/generate_ce_training_pairs.py --max_users 1000
"""

import os
import sys
import time
import pickle
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE          = "processed"
OUT_PATH      = "processed/ce_training_pairs.parquet"
HF_MODEL_REPO = "chaturg/amazon-recsys-cross-encoder"
FAISS_CONFIG  = "config3_full_system"   # use Config 3 FAISS for hardest negatives
TOP_K         = 100                     # candidates per user
MAX_PAIRS_PER_USER = 20                 # cap pairs per user to control dataset size
HISTORY_MAX_ITEMS  = 8                  # max items in history summary

USER_COLS = ["rating_norm_mean","rating_norm_std","helpfulness_mean",
             "verified_ratio","length_mean","interaction_count_norm","category_entropy"]
ITEM_COLS = ["avg_rating_norm","review_count_norm","avg_silver_label",
             "verified_ratio","avg_length_score"]


# ── Feature engineering (same as training) ────────────────────────────────────
def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("user_id").agg(
        rating_norm_mean =("rating_norm","mean"),
        rating_norm_std  =("rating_norm","std"),
        helpfulness_mean =("helpfulness_score","mean"),
        verified_ratio   =("verified_score","mean"),
        length_mean      =("length_score","mean"),
        interaction_count=("asin","count"),
    ).reset_index()
    agg["rating_norm_std"]        = agg["rating_norm_std"].fillna(0)
    agg["interaction_count_norm"] = agg["interaction_count"] / agg["interaction_count"].max()
    label_std = df.groupby("user_id")["silver_label"].std().fillna(0).rename("category_entropy")
    return agg.merge(label_std, on="user_id", how="left").fillna(0)[["user_id"]+USER_COLS]


def build_item_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("asin").agg(
        avg_rating_norm  =("rating_norm","mean"),
        review_count     =("user_id","count"),
        avg_silver_label =("silver_label","mean"),
        verified_ratio   =("verified_score","mean"),
        avg_length_score =("length_score","mean"),
    ).reset_index()
    agg["review_count_norm"] = agg["review_count"] / agg["review_count"].max()
    return agg[["asin"]+ITEM_COLS].fillna(0)


# ── Model definition (minimal — just for inference) ───────────────────────────
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, layers, dropout):
        super().__init__()
        net, d = [], in_dim
        for _ in range(layers-1):
            net += [nn.Linear(d,hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        net.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*net)
    def forward(self, x): return self.net(x)

class TwoTowerModel(nn.Module):
    def __init__(self, embed_dim=128, uid=7, iid=5, tower_width=256,
                 tower_depth=2, dropout=0.2):
        super().__init__()
        self.embed_dim  = embed_dim
        self.user_tower = MLP(uid, tower_width, embed_dim, tower_depth, dropout)
        self.item_tower = MLP(iid, tower_width, embed_dim, tower_depth, dropout)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim))

    def encode_user(self, uf, ni=None):
        e = self.user_tower(uf)
        if ni is not None:
            a = torch.clamp(ni.float()/5.0, 0, 1).unsqueeze(-1)
            e = a*e + (1-a)*e.mean(0, keepdim=True).expand_as(e)
        return F.normalize(e, p=2, dim=-1)

    def get_combined(self, ue):
        return F.normalize(self.projection(torch.cat([ue,ue],dim=-1)), p=2, dim=-1)


# ── Pull model and FAISS from HF ───────────────────────────────────────────────
def pull_artifacts(hf_token: str) -> tuple:
    """Download Config 3 model weights and FAISS index from HF."""
    import faiss

    logger.info(f"Pulling {FAISS_CONFIG} artifacts from HF...")

    # Pull model weights
    model_path = hf_hub_download(
        repo_id   = HF_MODEL_REPO,
        filename  = f"models/{FAISS_CONFIG}/model.pt",
        repo_type = "model",
        token     = hf_token,
        local_dir = "artifacts/",
    )
    logger.info(f"  ✓ Model: {model_path}")

    # Pull FAISS index
    faiss_path = hf_hub_download(
        repo_id   = HF_MODEL_REPO,
        filename  = f"faiss/{FAISS_CONFIG}.bin",
        repo_type = "model",
        token     = hf_token,
        local_dir = "artifacts/",
    )
    faiss_ids_path = hf_hub_download(
        repo_id   = HF_MODEL_REPO,
        filename  = f"faiss/{FAISS_CONFIG}.bin.ids.pkl",
        repo_type = "model",
        token     = hf_token,
        local_dir = "artifacts/",
    )
    logger.info(f"  ✓ FAISS: {faiss_path}")

    # Load FAISS index
    index = faiss.read_index(faiss_path)
    with open(faiss_ids_path, "rb") as f:
        item_ids = pickle.load(f)
    logger.info(f"  FAISS index: {index.ntotal:,} items")

    # Load model
    model = TwoTowerModel(embed_dim=128)
    state = torch.load(model_path, map_location="cpu")
    # Handle checkpoint wrapper if needed
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    logger.info(f"  Model loaded successfully")

    return model, index, item_ids


# ── History summary builder ────────────────────────────────────────────────────
def build_history_summary(user_id: str,
                           train_df: pd.DataFrame,
                           titles_df: pd.DataFrame) -> str:
    """
    Build a short text summary of a user's purchase history.
    Used as the left side of the cross-encoder input.
    Format: "bought {title1}, {title2}, {title3}"
    """
    user_items = train_df[train_df["user_id"] == user_id].sort_values(
        "timestamp", ascending=False
    ).head(HISTORY_MAX_ITEMS)

    if len(user_items) == 0:
        return "new user no purchase history"

    # Join with titles
    items_with_titles = user_items.merge(
        titles_df[["asin","title"]], on="asin", how="left"
    )

    titles = []
    for _, row in items_with_titles.iterrows():
        title = row.get("title", "")
        if isinstance(title, str) and len(title) > 5:
            titles.append(title[:60])
        else:
            titles.append(row["asin"])

    return "bought " + ", ".join(titles[:HISTORY_MAX_ITEMS])


# ── Main pair generation ───────────────────────────────────────────────────────
def generate_pairs(
    max_users: int = None,
    hf_token:  str = "",
) -> None:
    t0 = time.time()

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading data...")
    train_df  = pd.read_parquet(f"{BASE}/train.parquet")
    val_df    = pd.read_parquet(f"{BASE}/val.parquet")
    titles_df = pd.read_parquet(f"{BASE}/item_titles.parquet")
    all_df    = pd.concat([train_df, val_df], ignore_index=True)

    logger.info(f"  Train={len(train_df):,} | Val={len(val_df):,}")

    # ── Build features ─────────────────────────────────────────────────────
    logger.info("Building features...")
    user_agg  = build_user_features(train_df)
    item_agg  = build_item_features(all_df)
    user_feat_map = user_agg.set_index("user_id")
    item_feat_map = item_agg.set_index("asin")

    # ── Pull model and FAISS ───────────────────────────────────────────────
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN", "")
    model, faiss_index, item_ids = pull_artifacts(hf_token)

    # Map item_ids (ASINs) to their feature row index
    asin_to_idx = {asin: i for i, asin in enumerate(item_agg["asin"])}
    item_feats_np = item_agg[ITEM_COLS].values.astype(np.float32)

    # Build per-user silver label lookup for positive identification
    user_silver = train_df.set_index(["user_id","asin"])["silver_label"].to_dict()

    # ── Sample users ───────────────────────────────────────────────────────
    all_users = user_agg["user_id"].values
    if max_users:
        all_users = all_users[:max_users]
        logger.info(f"  Limited to {max_users:,} users for testing")
    else:
        logger.info(f"  Processing {len(all_users):,} users")

    # ── Generate pairs ─────────────────────────────────────────────────────
    records  = []
    BATCH    = 512
    n_users  = len(all_users)

    for batch_start in range(0, n_users, BATCH):
        batch_users = all_users[batch_start:batch_start + BATCH]

        # Build user embeddings for this batch
        user_vecs = []
        valid_users = []
        for uid in batch_users:
            if uid not in user_feat_map.index:
                continue
            feats = user_feat_map.loc[uid, USER_COLS].values.astype(np.float32)
            n_inter = float(user_feat_map.loc[uid, "interaction_count_norm"] * 625140)
            user_vecs.append((uid, feats, n_inter))
            valid_users.append(uid)

        if not user_vecs:
            continue

        # Encode users in sub-batch
        feat_tensor  = torch.tensor(np.array([v[1] for v in user_vecs]))
        n_inter_tensor = torch.tensor([v[2] for v in user_vecs])

        with torch.no_grad():
            user_embs = model.encode_user(feat_tensor, n_inter_tensor)
            combined  = model.get_combined(user_embs)
            query_np  = combined.numpy().astype(np.float32)

        # FAISS search — top-100 per user
        distances, indices = faiss_index.search(query_np, TOP_K)

        # Build pairs for each user
        for i, uid in enumerate(valid_users):
            history_summary = build_history_summary(uid, train_df, titles_df)
            candidate_asins = item_ids[indices[i]]

            pairs_this_user = 0
            for rank, cand_asin in enumerate(candidate_asins):
                if pairs_this_user >= MAX_PAIRS_PER_USER:
                    break

                # Get item title
                title_row = titles_df[titles_df["asin"] == cand_asin]
                if len(title_row) > 0 and isinstance(title_row.iloc[0]["title"], str):
                    item_title    = title_row.iloc[0]["title"][:100]
                    item_category = title_row.iloc[0].get("categories", "Tools & Home Improvement")
                else:
                    item_title    = cand_asin
                    item_category = "Tools & Home Improvement"

                # Relevance label
                key = (uid, cand_asin)
                if key in user_silver:
                    relevance  = float(user_silver[key])
                    is_positive = True
                else:
                    relevance   = 0.0
                    is_positive = False

                records.append({
                    "user_id":          uid,
                    "history_summary":  history_summary,
                    "item_title":       item_title,
                    "item_category":    str(item_category),
                    "asin":             cand_asin,
                    "relevance_label":  relevance,
                    "is_positive":      is_positive,
                    "faiss_rank":       rank,
                })
                pairs_this_user += 1

        # Progress log
        processed = min(batch_start + BATCH, n_users)
        if batch_start % (BATCH * 10) == 0:
            elapsed = time.time() - t0
            rate    = processed / elapsed if elapsed > 0 else 1
            eta_min = (n_users - processed) / rate / 60
            pos_rate = sum(1 for r in records if r["is_positive"]) / len(records) if records else 0
            logger.info(
                f"  {processed:,}/{n_users:,} users | "
                f"{len(records):,} pairs | "
                f"positive_rate={pos_rate:.2%} | "
                f"ETA={eta_min:.0f}min"
            )

    # ── Save ───────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(records)
    df_out.to_parquet(OUT_PATH, index=False)

    elapsed = time.time() - t0
    n_pos   = df_out["is_positive"].sum()
    n_neg   = (~df_out["is_positive"]).sum()

    logger.info(f"\n{'='*60}")
    logger.info(f"Pair generation complete in {elapsed/60:.1f} min")
    logger.info(f"  Total pairs:    {len(df_out):,}")
    logger.info(f"  Positive pairs: {n_pos:,} ({n_pos/len(df_out):.2%})")
    logger.info(f"  Negative pairs: {n_neg:,} ({n_neg/len(df_out):.2%})")
    logger.info(f"  Output:         {OUT_PATH}")
    logger.info(f"{'='*60}")

    # Show 3 sample pairs
    logger.info("\nSample pairs:")
    for _, row in df_out[df_out["is_positive"]].head(3).iterrows():
        logger.info(f"  User history: {row['history_summary'][:80]}")
        logger.info(f"  Item:         {row['item_title'][:80]}")
        logger.info(f"  Relevance:    {row['relevance_label']:.3f} (positive={row['is_positive']})")
        logger.info("")


def push_pairs_to_hf(hf_token: str) -> None:
    """Push generated pairs to HF dataset repo."""
    from huggingface_hub import HfApi
    api = HfApi()
    logger.info("Pushing training pairs to HF...")
    api.upload_file(
        path_or_fileobj = OUT_PATH,
        path_in_repo    = "processed/ce_training_pairs.parquet",
        repo_id         = "chaturg/amazon-recsys-dataset",
        repo_type       = "dataset",
        token           = hf_token,
        commit_message  = "Add cross-encoder training pairs",
    )
    logger.info("  ✓ Pushed to chaturg/amazon-recsys-dataset/processed/ce_training_pairs.parquet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cross-encoder training pairs from FAISS top-100"
    )
    parser.add_argument("--max_users",  type=int, default=None,
                        help="Limit users for testing (omit for full run)")
    parser.add_argument("--hf_token",   type=str,
                        default=os.environ.get("HF_TOKEN",""),
                        help="HF read token (defaults to HF_TOKEN env var)")
    parser.add_argument("--push_to_hf", action="store_true",
                        help="Push output to HF after generation")
    args = parser.parse_args()

    generate_pairs(max_users=args.max_users, hf_token=args.hf_token)

    if args.push_to_hf:
        push_pairs_to_hf(args.hf_token)
