"""
run_ce_pairs_fast.py
Fast version of CE pair generation — uses dict lookups instead of
DataFrame row searches. Reduces runtime from ~94hrs to ~2hrs.
"""
import os, time, pickle, logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import faiss

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BASE          = "processed"
OUT_PATH      = "processed/ce_training_pairs.parquet"
HF_MODEL_REPO = "chaturg/amazon-recsys-cross-encoder"
FAISS_CONFIG  = "config3_full_system"
TOP_K         = 100
MAX_NEG_PER_USER = 19

USER_COLS = ["rating_norm_mean","rating_norm_std","helpfulness_mean",
             "verified_ratio","length_mean","interaction_count_norm","category_entropy"]
ITEM_COLS = ["avg_rating_norm","review_count_norm","avg_silver_label",
             "verified_ratio","avg_length_score"]

class MLP(nn.Module):
    def __init__(self, in_dim, h, out_dim, layers, dropout):
        super().__init__()
        net, d = [], in_dim
        for _ in range(layers-1):
            net += [nn.Linear(d,h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        net.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*net)
    def forward(self, x): return self.net(x)

class TwoTowerModel(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim  = embed_dim
        self.user_tower = MLP(7, 256, embed_dim, 2, 0.2)
        self.item_tower = MLP(5, 256, embed_dim, 2, 0.2)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim))
    def encode_user(self, uf, ni=None):
        e = self.user_tower(uf)
        if ni is not None:
            a = torch.clamp(ni.float()/5.0,0,1).unsqueeze(-1)
            e = a*e + (1-a)*e.mean(0,keepdim=True).expand_as(e)
        return F.normalize(e, p=2, dim=-1)
    def get_combined(self, ue):
        return F.normalize(self.projection(torch.cat([ue,ue],dim=-1)), p=2, dim=-1)

def build_user_features(df):
    agg = df.groupby("user_id").agg(
        rating_norm_mean=("rating_norm","mean"), rating_norm_std=("rating_norm","std"),
        helpfulness_mean=("helpfulness_score","mean"), verified_ratio=("verified_score","mean"),
        length_mean=("length_score","mean"), interaction_count=("asin","count"),
    ).reset_index()
    agg["rating_norm_std"]        = agg["rating_norm_std"].fillna(0)
    agg["interaction_count_norm"] = agg["interaction_count"] / agg["interaction_count"].max()
    label_std = df.groupby("user_id")["silver_label"].std().fillna(0).rename("category_entropy")
    return agg.merge(label_std, on="user_id", how="left").fillna(0)[["user_id"]+USER_COLS]

def build_item_features(df):
    agg = df.groupby("asin").agg(
        avg_rating_norm=("rating_norm","mean"), review_count=("user_id","count"),
        avg_silver_label=("silver_label","mean"), verified_ratio=("verified_score","mean"),
        avg_length_score=("length_score","mean"),
    ).reset_index()
    agg["review_count_norm"] = agg["review_count"] / agg["review_count"].max()
    return agg[["asin"]+ITEM_COLS].fillna(0)

def run(max_users=None):
    t0 = time.time()

    # Load data
    logger.info("Loading data...")
    train_df  = pd.read_parquet(f"{BASE}/train.parquet")
    val_df    = pd.read_parquet(f"{BASE}/val.parquet")
    titles_df = pd.read_parquet(f"{BASE}/item_titles.parquet")
    all_df    = pd.concat([train_df, val_df], ignore_index=True)
    logger.info(f"  Train={len(train_df):,} Val={len(val_df):,}")

    # ── Pre-build fast lookup dicts — O(1) access ─────────────────────────
    logger.info("Building lookup dicts...")

    # Title lookup: asin → (title, category)
    title_lookup = {}
    for _, r in titles_df.iterrows():
        t = r.get("title","")
        c = str(r.get("categories","Tools & Home Improvement"))
        if isinstance(t, str) and len(t) > 5:
            title_lookup[r["asin"]] = (t[:100], c)

    # Val positive lookup: user_id → (asin, silver_label)
    val_positive = val_df.set_index("user_id")[["asin","silver_label"]].to_dict("index")

    # Train silver label set: (user_id, asin) → silver_label
    train_silver_set = set(zip(train_df["user_id"], train_df["asin"]))

    # User history: user_id → history summary string
    logger.info("Building user history summaries...")
    user_history_items = train_df.sort_values("timestamp", ascending=False)\
        .groupby("user_id")["asin"].apply(lambda x: list(x)[:8]).to_dict()

    def history_summary(uid):
        items = user_history_items.get(uid, [])
        if not items: return "no purchase history"
        titles = []
        for asin in items:
            t, _ = title_lookup.get(asin, (asin, ""))
            titles.append(t[:50])
        return "bought " + ", ".join(titles)

    # Build user history summaries once per user — not per pair
    logger.info("Pre-computing history summaries...")
    all_users_list = list(train_df["user_id"].unique())
    if max_users:
        all_users_list = all_users_list[:max_users]
    user_hist_cache = {uid: history_summary(uid) for uid in all_users_list}
    logger.info(f"  {len(user_hist_cache):,} history summaries built")

    # Build features
    logger.info("Building features...")
    user_agg      = build_user_features(train_df)
    item_agg      = build_item_features(all_df)
    user_feat_map = user_agg.set_index("user_id")

    # Pull FAISS and model
    hf_token = os.environ.get("HF_TOKEN","")
    logger.info("Pulling FAISS and model from HF...")
    model_path = hf_hub_download(HF_MODEL_REPO, f"models/{FAISS_CONFIG}/model.pt",
        repo_type="model", token=hf_token, local_dir="artifacts/")
    faiss_path = hf_hub_download(HF_MODEL_REPO, f"faiss/{FAISS_CONFIG}.bin",
        repo_type="model", token=hf_token, local_dir="artifacts/")
    ids_path   = hf_hub_download(HF_MODEL_REPO, f"faiss/{FAISS_CONFIG}.bin.ids.pkl",
        repo_type="model", token=hf_token, local_dir="artifacts/")

    faiss_index = faiss.read_index(faiss_path)
    with open(ids_path,"rb") as f: item_ids = pickle.load(f)

    model = TwoTowerModel(128)
    state = torch.load(model_path, map_location="cpu")
    if "model" in state: state = state["model"]
    model.load_state_dict(state); model.eval()
    logger.info(f"  FAISS: {faiss_index.ntotal:,} | Model loaded")

    all_users = np.array(all_users_list)
    records   = []
    BATCH     = 512

    for batch_start in range(0, len(all_users), BATCH):
        batch_users = all_users[batch_start:batch_start+BATCH]

        user_vecs, valid_uids = [], []
        for uid in batch_users:
            if uid not in user_feat_map.index: continue
            feats   = user_feat_map.loc[uid, USER_COLS].values.astype(np.float32)
            n_inter = float(user_feat_map.loc[uid,"interaction_count_norm"]*625140)
            user_vecs.append((uid, feats, n_inter))
            valid_uids.append(uid)

        if not user_vecs: continue

        feat_t = torch.tensor(np.array([v[1] for v in user_vecs]))
        ni_t   = torch.tensor([v[2] for v in user_vecs])
        with torch.no_grad():
            ue = model.encode_user(feat_t, ni_t)
            qv = model.get_combined(ue).numpy().astype(np.float32)

        _, indices = faiss_index.search(qv, TOP_K)

        for i, uid in enumerate(valid_uids):
            hist = user_hist_cache.get(uid, "no history")

            # Guaranteed positive
            pos_asin = None
            if uid in val_positive:
                pos_asin  = val_positive[uid]["asin"]
                pos_label = float(val_positive[uid]["silver_label"])
                title, cat = title_lookup.get(pos_asin, (pos_asin, "Tools & Home Improvement"))
                records.append({
                    "user_id":         uid,
                    "history_summary": hist,
                    "item_title":      title,
                    "item_category":   cat,
                    "asin":            pos_asin,
                    "relevance_label": pos_label,
                    "is_positive":     True,
                    "source":          "val_positive",
                })

            # Hard negatives from FAISS
            neg_count = 0
            for cand in item_ids[indices[i]].tolist():
                if neg_count >= MAX_NEG_PER_USER: break
                if cand == pos_asin: continue
                if (uid, cand) in train_silver_set: continue

                title, cat = title_lookup.get(cand, (cand, "Tools & Home Improvement"))
                records.append({
                    "user_id":         uid,
                    "history_summary": hist,
                    "item_title":      title,
                    "item_category":   cat,
                    "asin":            cand,
                    "relevance_label": 0.0,
                    "is_positive":     False,
                    "source":          "faiss_negative",
                })
                neg_count += 1

        if batch_start % (BATCH*10) == 0 and records:
            pos     = sum(1 for r in records if r["is_positive"])
            elapsed = time.time()-t0
            rate    = (batch_start+len(batch_users))/elapsed if elapsed>0 else 1
            eta     = (len(all_users)-batch_start-len(batch_users))/rate/60
            logger.info(f"  {batch_start+len(batch_users):,}/{len(all_users):,} | "
                        f"{len(records):,} pairs | "
                        f"pos={pos/len(records)*100:.1f}% | "
                        f"ETA={eta:.0f}min")

    # Save
    df_out = pd.DataFrame(records)
    df_out.to_parquet(OUT_PATH, index=False)
    pos = df_out["is_positive"].sum()

    logger.info(f"\n{'='*60}")
    logger.info(f"Complete in {(time.time()-t0)/60:.1f} min")
    logger.info(f"  Total pairs:    {len(df_out):,}")
    logger.info(f"  Positive pairs: {pos:,} ({pos/len(df_out):.1%})")
    logger.info(f"  Negative pairs: {len(df_out)-pos:,}")
    logger.info(f"  Output: {OUT_PATH}")
    logger.info(f"{'='*60}")

    # Push to HF
    hf_token = os.environ.get("HF_TOKEN","")
    if hf_token:
        from huggingface_hub import HfApi
        api = HfApi()
        logger.info("Pushing to HF...")
        api.upload_file(
            path_or_fileobj = OUT_PATH,
            path_in_repo    = "processed/ce_training_pairs.parquet",
            repo_id         = "chaturg/amazon-recsys-dataset",
            repo_type       = "dataset",
            token           = hf_token,
            commit_message  = "Add cross-encoder training pairs (5% positive rate)",
        )
        logger.info("  ✓ Pushed to HF")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--max_users", type=int, default=None)
    args = p.parse_args()
    run(max_users=args.max_users)
