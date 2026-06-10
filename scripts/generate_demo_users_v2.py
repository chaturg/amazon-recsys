"""
generate_demo_users_v2.py
-------------------------
Finds the 10 best demo users by iterating through 500 random users
and scoring each on:
  - Val item appears in FAISS top-100 (required for normal users)
  - Val item rank in top-100 (lower = better)
  - Title coverage in top-10 recommendations (more real titles = better)
  - History title coverage (richer history = better)

Selection:
  u001–u006: Normal users  (val item in top-100, ≥7 real titles in top-10)
  u007–u008: Sparse users  (2–4 interactions, val item in top-100 if possible)
  u009–u010: New users     (synthetic — 0 interactions, popularity fallback)

Each JSON includes:
  - validation_history: purchase history items with titles
  - val_item: ground truth held-out item
  - val_item_faiss_rank: rank of val item in FAISS top-100 (null if not found)
  - top10_recommendations: pre-computed top-10 for display
  - recall_hit: bool — did val item appear in top-100?

Usage:
    cd ~/cloudfiles/code/Users/casakaay/amazon-recsys
    python scripts/generate_demo_users_v2.py
"""

import json
import logging
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE          = "processed"
DEMO_DIR      = "demo/users"
HF_MODEL_REPO = "chaturg/amazon-recsys-cross-encoder"
CANDIDATE_POOL = 500   # users to evaluate
TITLE_MIN      = 7     # min real titles in top-10
HIST_TITLE_MIN = 4     # min real titles in history
MAX_RANK       = 75    # val item must appear within rank 75

USER_COLS = ["rating_norm_mean","rating_norm_std","helpfulness_mean",
             "verified_ratio","length_mean","interaction_count_norm","category_entropy"]
ITEM_COLS = ["avg_rating_norm","review_count_norm","avg_silver_label",
             "verified_ratio","avg_length_score"]

random.seed(42)
np.random.seed(42)


# ── Minimal model ─────────────────────────────────────────────────────────────
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
            a = torch.clamp(ni.float()/5.0, 0, 1).unsqueeze(-1)
            e = a*e + (1-a)*e.mean(0, keepdim=True).expand_as(e)
        return F.normalize(e, p=2, dim=-1)
    def get_combined(self, ue):
        return F.normalize(self.projection(torch.cat([ue,ue],dim=-1)), p=2, dim=-1)


# ── Feature engineering ───────────────────────────────────────────────────────
def build_user_features(df):
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

def build_item_features(df):
    agg = df.groupby("asin").agg(
        avg_rating_norm  =("rating_norm","mean"),
        review_count     =("user_id","count"),
        avg_silver_label =("silver_label","mean"),
        verified_ratio   =("verified_score","mean"),
        avg_length_score =("length_score","mean"),
    ).reset_index()
    agg["review_count_norm"] = agg["review_count"] / agg["review_count"].max()
    return agg[["asin"]+ITEM_COLS].fillna(0)


# ── Item info helpers ─────────────────────────────────────────────────────────
def build_title_lookup(titles_df):
    lookup = {}
    for _, r in titles_df.iterrows():
        t = r.get("title","")
        c = str(r.get("categories","Tools & Home Improvement"))
        p = r.get("price", None)
        if isinstance(t, str) and len(t) > 5:
            lookup[r["asin"]] = {
                "title":    t[:120],
                "category": c[:80],
                "price":    float(p) if isinstance(p, float) and p > 0 else None,
            }
    return lookup


def has_real_title(asin, title_lookup):
    return asin in title_lookup


def get_item_display(asin, title_lookup, train_df=None, rating=None, silver_label=None,
                     verified=None, helpful_votes=None, text=None):
    """Build display dict for one item."""
    info = title_lookup.get(asin, {})
    title    = info.get("title", asin)
    category = info.get("category", "Tools & Home Improvement")
    price    = info.get("price", None)

    snippet = ""
    if text and len(str(text)) > 20:
        snippet = str(text)[:120].strip()

    return {
        "asin":             asin,
        "title":            title,
        "category":         category,
        "price_usd":        price,
        "rating":           int(rating) if rating is not None else None,
        "silver_label":     round(float(silver_label), 3) if silver_label is not None else None,
        "verified_purchase":bool(verified > 0.5) if verified is not None else None,
        "helpful_votes":    int(helpful_votes) if helpful_votes is not None else 0,
        "review_snippet":   snippet,
        "has_real_title":   asin in title_lookup,
    }


# ── FAISS search ──────────────────────────────────────────────────────────────
def encode_user(uid, user_feat_map, model):
    if uid not in user_feat_map.index:
        return None
    feats   = user_feat_map.loc[uid, USER_COLS].values.astype(np.float32)
    n_inter = float(user_feat_map.loc[uid, "interaction_count_norm"] * 625140)
    feat_t  = torch.tensor(feats).unsqueeze(0)
    ni_t    = torch.tensor([n_inter])
    with torch.no_grad():
        ue = model.encode_user(feat_t, ni_t)
        qv = model.get_combined(ue)
    return qv.numpy()[0]


def search_faiss(query_vec, faiss_index, item_ids, k=100):
    qv = query_vec.reshape(1,-1).astype(np.float32)
    distances, indices = faiss_index.search(qv, k)
    return distances[0], item_ids[indices[0]]


# ── User scoring ──────────────────────────────────────────────────────────────
def score_user(uid, val_asin, top100_asins, top10_asins, title_lookup,
               train_rows, n_inter):
    """
    Score a user for demo quality. Higher = better demo candidate.

    Returns (score, val_rank, title_count, hist_title_count)
    """
    # Val item rank in top-100
    val_rank = None
    for i, asin in enumerate(top100_asins):
        if asin == val_asin:
            val_rank = i + 1  # 1-indexed
            break

    # Title coverage in top-10
    title_count = sum(1 for a in top10_asins if has_real_title(a, title_lookup))

    # History title coverage
    hist_asins       = train_rows["asin"].tolist()
    hist_title_count = sum(1 for a in hist_asins[:10] if has_real_title(a, title_lookup))

    # Score
    rank_score  = (100 - val_rank) * 0.5   if val_rank else 0
    title_score = title_count * 30
    hist_score  = hist_title_count * 20

    score = rank_score + title_score + hist_score
    return score, val_rank, title_count, hist_title_count


# ── Main ──────────────────────────────────────────────────────────────────────
def generate_demo_users(hf_token=""):
    import faiss

    Path(DEMO_DIR).mkdir(parents=True, exist_ok=True)
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN", "")

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading data...")
    train_df  = pd.read_parquet(f"{BASE}/train.parquet")
    val_df    = pd.read_parquet(f"{BASE}/val.parquet")
    titles_df = pd.read_parquet(f"{BASE}/item_titles.parquet")
    all_df    = pd.concat([train_df, val_df], ignore_index=True)

    title_lookup  = build_title_lookup(titles_df)
    val_idx       = val_df.set_index("user_id")
    user_agg      = build_user_features(train_df)
    item_agg      = build_item_features(all_df)
    user_feat_map = user_agg.set_index("user_id")
    user_counts   = train_df.groupby("user_id").size()

    logger.info(f"  Title lookup: {len(title_lookup):,} items with real titles")

    # ── Pull model + FAISS ─────────────────────────────────────────────────
    logger.info("Pulling model and FAISS from HF...")
    model_path = hf_hub_download(HF_MODEL_REPO,
        "models/config3_full_system/model.pt",
        repo_type="model", token=hf_token, local_dir="artifacts/")
    faiss_path = hf_hub_download(HF_MODEL_REPO,
        "faiss/config3_full_system.bin",
        repo_type="model", token=hf_token, local_dir="artifacts/")
    ids_path   = hf_hub_download(HF_MODEL_REPO,
        "faiss/config3_full_system.bin.ids.pkl",
        repo_type="model", token=hf_token, local_dir="artifacts/")

    faiss_index = faiss.read_index(faiss_path)
    with open(ids_path, "rb") as f:
        item_ids = pickle.load(f)

    model = TwoTowerModel(128)
    state = torch.load(model_path, map_location="cpu")
    if "model" in state: state = state["model"]
    model.load_state_dict(state)
    model.eval()
    logger.info(f"  FAISS: {faiss_index.ntotal:,} items | Model loaded")

    # ── Popularity items for new user fallback ─────────────────────────────
    popularity_items = (
        item_agg.sort_values("avg_silver_label", ascending=False)
        .head(200)["asin"].tolist()
    )

    # ── Find best normal users ─────────────────────────────────────────────
    logger.info(f"Evaluating {CANDIDATE_POOL} users to find best 6 normal...")

    # Pool: users with 10-50 interactions who are in val set
    val_users = set(val_df["user_id"].unique())
    normal_pool = user_counts[
        (user_counts >= 10) & (user_counts <= 50)
    ].index.tolist()
    normal_pool = [u for u in normal_pool if u in val_users]
    random.shuffle(normal_pool)
    normal_pool = normal_pool[:CANDIDATE_POOL]

    candidates = []
    t0 = time.time()

    for i, uid in enumerate(normal_pool):
        if uid not in val_idx.index:
            continue

        val_row  = val_idx.loc[uid]
        val_asin = val_row["asin"]

        # Encode user
        qv = encode_user(uid, user_feat_map, model)
        if qv is None:
            continue

        # FAISS search
        _, top100 = search_faiss(qv, faiss_index, item_ids, k=100)
        top10     = top100[:10].tolist()
        top100_list = top100.tolist()

        train_rows = train_df[train_df["user_id"] == uid]
        n_inter    = len(train_rows)

        score, val_rank, title_count, hist_title_count = score_user(
            uid, val_asin, top100_list, top10,
            title_lookup, train_rows, n_inter
        )

        # Filter: must meet minimum criteria
        if title_count < TITLE_MIN:
            continue
        if val_rank is None or val_rank > MAX_RANK:
            continue
        if hist_title_count < HIST_TITLE_MIN:
            continue

        candidates.append({
            "uid":              uid,
            "score":            score,
            "val_rank":         val_rank,
            "title_count":      title_count,
            "hist_title_count": hist_title_count,
            "n_inter":          n_inter,
            "top100":           top100_list,
            "top10":            top10,
            "val_asin":         val_asin,
            "val_row":          val_row,
            "train_rows":       train_rows,
        })

        if (i+1) % 50 == 0:
            logger.info(
                f"  Evaluated {i+1}/{len(normal_pool)} | "
                f"qualified={len(candidates)} | "
                f"elapsed={time.time()-t0:.0f}s"
            )

        if len(candidates) >= 30:  # enough candidates to pick from
            break

    if len(candidates) < 6:
        logger.warning(f"  Only {len(candidates)} qualified users found — relaxing criteria")
        # Relax: remove val_rank requirement
        candidates2 = []
        for uid in normal_pool[:CANDIDATE_POOL]:
            if uid not in val_idx.index: continue
            val_row  = val_idx.loc[uid]
            val_asin = val_row["asin"]
            qv = encode_user(uid, user_feat_map, model)
            if qv is None: continue
            _, top100 = search_faiss(qv, faiss_index, item_ids, k=100)
            top10 = top100[:10].tolist()
            top100_list = top100.tolist()
            train_rows = train_df[train_df["user_id"] == uid]
            score, val_rank, title_count, hist_title_count = score_user(
                uid, val_asin, top100_list, top10,
                title_lookup, train_rows, len(train_rows)
            )
            if title_count >= 5:
                candidates2.append({
                    "uid": uid, "score": score, "val_rank": val_rank,
                    "title_count": title_count, "hist_title_count": hist_title_count,
                    "n_inter": len(train_rows), "top100": top100_list,
                    "top10": top10, "val_asin": val_asin,
                    "val_row": val_row, "train_rows": train_rows,
                })
            if len(candidates2) >= 20: break
        candidates = candidates2

    # Sort by score, take top 6
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best_normal = candidates[:6]

    logger.info(f"\nTop 6 normal users:")
    for c in best_normal:
        logger.info(
            f"  {c['uid']} | score={c['score']:.0f} | "
            f"val_rank={c['val_rank']} | "
            f"titles={c['title_count']}/10 | "
            f"hist_titles={c['hist_title_count']} | "
            f"interactions={c['n_inter']}"
        )

    # ── Find best sparse users ─────────────────────────────────────────────
    logger.info("\nFinding sparse users...")
    sparse_pool = user_counts[
        (user_counts >= 2) & (user_counts <= 4)
    ].index.tolist()
    sparse_pool = [u for u in sparse_pool if u in val_users]
    random.shuffle(sparse_pool)

    sparse_candidates = []
    for uid in sparse_pool[:200]:
        if uid not in val_idx.index: continue
        val_row  = val_idx.loc[uid]
        val_asin = val_row["asin"]
        qv = encode_user(uid, user_feat_map, model)
        if qv is None: continue
        _, top100 = search_faiss(qv, faiss_index, item_ids, k=100)
        top10 = top100[:10].tolist()
        top100_list = top100.tolist()
        train_rows = train_df[train_df["user_id"] == uid]

        val_rank = next((i+1 for i, a in enumerate(top100_list) if a == val_asin), None)
        title_count = sum(1 for a in top10 if has_real_title(a, title_lookup))

        score = (100 - val_rank) * 0.3 if val_rank else 0
        score += title_count * 25

        sparse_candidates.append({
            "uid": uid, "score": score, "val_rank": val_rank,
            "title_count": title_count, "n_inter": len(train_rows),
            "top100": top100_list, "top10": top10,
            "val_asin": val_asin, "val_row": val_row,
            "train_rows": train_rows,
        })
        if len(sparse_candidates) >= 20: break

    sparse_candidates.sort(key=lambda x: x["score"], reverse=True)
    best_sparse = sparse_candidates[:2]
    logger.info(f"Best sparse users: {[c['uid'] for c in best_sparse]}")

    # ── Write user JSON files ──────────────────────────────────────────────
    logger.info("\nWriting user JSON files...")
    all_selected = (
        [("normal", c) for c in best_normal] +
        [("sparse", c) for c in best_sparse]
    )

    written = []
    for i, (user_type, c) in enumerate(all_selected):
        demo_id    = f"u{i+1:03d}"
        uid        = c["uid"]
        train_rows = c["train_rows"]
        val_row    = c["val_row"]
        val_asin   = c["val_asin"]

        # Build history
        history = []
        for _, row in train_rows.sort_values("timestamp", ascending=False).head(12).iterrows():
            history.append(get_item_display(
                asin          = row["asin"],
                title_lookup  = title_lookup,
                rating        = row["rating"],
                silver_label  = row["silver_label"],
                verified      = row["verified_score"],
                helpful_votes = row.get("helpful_vote", 0),
                text          = row.get("text",""),
            ))

        # Build top-10 recommendations with rank info
        recommendations = []
        for rank, asin in enumerate(c["top10"], start=1):
            rec = get_item_display(asin=asin, title_lookup=title_lookup)
            rec["faiss_rank"] = rank
            recommendations.append(rec)

        # Val item display
        val_item = get_item_display(
            asin          = val_asin,
            title_lookup  = title_lookup,
            rating        = val_row["rating"],
            silver_label  = val_row["silver_label"],
            verified      = val_row["verified_score"],
            helpful_votes = val_row.get("helpful_vote", 0),
            text          = val_row.get("text",""),
        )

        user_data = {
            "user_id":                uid,
            "demo_id":                demo_id,
            "user_type":              user_type,
            "interaction_count":      len(train_rows),
            "verified_ratio":         round(float(train_rows["verified_score"].mean()), 3),
            "avg_rating":             round(float(train_rows["rating"].mean()), 2),
            "validation_history":     history,
            "val_item":               val_item,
            "val_item_faiss_rank":    c["val_rank"],
            "recall_hit":             c["val_rank"] is not None,
            "top10_recommendations":  recommendations,
            "demo_score":             round(c["score"], 1),
            "title_coverage":         f"{c['title_count']}/10",
        }

        out_path = Path(DEMO_DIR) / f"{demo_id}.json"
        with open(out_path, "w") as f:
            json.dump(user_data, f, indent=2, default=str)

        written.append(demo_id)
        logger.info(
            f"  {demo_id} ({user_type:6s}) | "
            f"interactions={len(train_rows):3d} | "
            f"val_rank={c['val_rank']} | "
            f"titles={c['title_count']}/10 | "
            f"val_title={val_item['title'][:50]}"
        )

    # ── Add 2 synthetic new users ──────────────────────────────────────────
    for i, (demo_id, category) in enumerate([("u009", "Power Tools"),
                                              ("u010", "Lighting & Electrical")]):
        # Get top-10 popular items as recommendations
        pop_recs = []
        for rank, asin in enumerate(popularity_items[:10], start=1):
            rec = get_item_display(asin=asin, title_lookup=title_lookup)
            rec["faiss_rank"] = None
            pop_recs.append(rec)

        user_data = {
            "user_id":               f"synthetic_{demo_id}",
            "demo_id":               demo_id,
            "user_type":             "new",
            "interaction_count":     0,
            "verified_ratio":        None,
            "avg_rating":            None,
            "validation_history":    [],
            "val_item":              None,
            "val_item_faiss_rank":   None,
            "recall_hit":            None,
            "top10_recommendations": pop_recs,
            "demo_score":            None,
            "title_coverage":        None,
        }
        out_path = Path(DEMO_DIR) / f"{demo_id}.json"
        with open(out_path, "w") as f:
            json.dump(user_data, f, indent=2, default=str)
        written.append(demo_id)
        logger.info(f"  {demo_id} (new    ) | synthetic — popularity fallback")

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Done. Written {len(written)} demo users to {DEMO_DIR}/")

    # Verify title coverage
    logger.info("\nTitle coverage check:")
    for p in sorted(Path(DEMO_DIR).glob("*.json")):
        d = json.loads(p.read_text())
        recs = d.get("top10_recommendations", [])
        real = sum(1 for r in recs if r.get("has_real_title"))
        vrank = d.get("val_item_faiss_rank")
        logger.info(
            f"  {p.name}: {real}/10 real titles | "
            f"val_rank={vrank} | "
            f"type={d['user_type']}"
        )
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    generate_demo_users()
