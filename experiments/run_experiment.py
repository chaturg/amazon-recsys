"""
run_experiment.py
-----------------
Runs a complete evaluation for one config and appends results to CSV.

For each config:
  1. Load trained model and FAISS index from HF or local
  2. Generate top-K recommendations for all val users
  3. Compute all 6 metrics
  4. Append row to results/eval_table.csv

Run configs in order: als → config1 → config2 → config3

Usage:
    # Evaluate all configs
    python experiments/run_experiment.py --config all

    # Evaluate single config
    python experiments/run_experiment.py --config config2

    # Pull artifacts from HF before eval
    python experiments/run_experiment.py --config config3 --pull_from_hf
"""

import argparse
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from eval.metrics import evaluate_rankings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASE          = "processed"
RESULTS_PATH  = "results/eval_table.csv"
HF_MODEL_REPO = "chaturg/amazon-recsys-cross-encoder"
TOP_K         = 10
RECALL_K      = 100
CATALOG_SIZE  = 157_462

USER_COLS = ["rating_norm_mean","rating_norm_std","helpfulness_mean",
             "verified_ratio","length_mean","interaction_count_norm","category_entropy"]
ITEM_COLS = ["avg_rating_norm","review_count_norm","avg_silver_label",
             "verified_ratio","avg_length_score"]


# ── Minimal model definition for inference ────────────────────────────────────
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
    def __init__(self, embed_dim=128, uid=7, iid=5):
        super().__init__()
        self.embed_dim  = embed_dim
        self.user_tower = MLP(uid, 256, embed_dim, 2, 0.2)
        self.item_tower = MLP(iid, 256, embed_dim, 2, 0.2)
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


# ── Feature engineering ────────────────────────────────────────────────────────
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


# ── Artifact loading ──────────────────────────────────────────────────────────
def load_artifacts(config_name: str, embed_dim: int, hf_token: str = "") -> tuple:
    """Load model and FAISS index — local first, HF fallback."""
    import faiss

    model_path = Path(f"artifacts/models/{config_name}/model.pt")
    faiss_path = Path(f"artifacts/faiss/{config_name}.bin")
    ids_path   = Path(f"artifacts/faiss/{config_name}.bin.ids.pkl")

    # Pull from HF if not local
    if not model_path.exists() or not faiss_path.exists():
        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            from huggingface_hub import hf_hub_download
            logger.info(f"  Pulling {config_name} from HF...")
            for fname in [f"models/{config_name}/model.pt",
                          f"faiss/{config_name}.bin",
                          f"faiss/{config_name}.bin.ids.pkl"]:
                hf_hub_download(HF_MODEL_REPO, fname, repo_type="model",
                                token=hf_token, local_dir="artifacts/")

    # Load FAISS
    faiss_index = faiss.read_index(str(faiss_path))
    with open(ids_path, "rb") as f:
        item_ids = pickle.load(f)

    # Load model
    model = TwoTowerModel(embed_dim=embed_dim)
    state = torch.load(model_path, map_location="cpu")
    if "model" in state: state = state["model"]
    model.load_state_dict(state)
    model.eval()

    logger.info(f"  Loaded {config_name}: {faiss_index.ntotal:,} items")
    return model, faiss_index, item_ids


# ── Cross-encoder re-ranking ───────────────────────────────────────────────────
def rerank_with_cross_encoder(
    candidates_per_user: list,
    histories:           list,
    titles_lookup:       dict,
    top_k:               int = TOP_K,
) -> list:
    """Re-rank FAISS candidates using the fine-tuned cross-encoder."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        logger.warning("transformers not installed — skipping cross-encoder reranking")
        return [c[:top_k] for c in candidates_per_user]

    ce_path = "artifacts/cross_encoder"
    if not Path(ce_path).exists():
        logger.warning(f"Cross-encoder not found at {ce_path} — skipping reranking")
        return [c[:top_k] for c in candidates_per_user]

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(ce_path)
    ce_model  = AutoModelForSequenceClassification.from_pretrained(
        ce_path, num_labels=1).to(device)
    ce_model.eval()

    reranked = []
    for cands, hist in zip(candidates_per_user, histories):
        if not cands:
            reranked.append([])
            continue

        item_texts = [
            f"{titles_lookup.get(a, (a,''))[0]} {titles_lookup.get(a, ('',''))[1]}"
            for a in cands
        ]
        hists = [hist] * len(cands)

        enc = tokenizer(hists, item_texts, max_length=128,
                        padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            out    = ce_model(input_ids=enc["input_ids"].to(device),
                              attention_mask=enc["attention_mask"].to(device))
            scores = torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy()

        ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
        reranked.append([a for a, _ in ranked[:top_k]])

    return reranked


# ── Main experiment runner ────────────────────────────────────────────────────
def run_experiment(
    config_name:    str,
    embed_dim:      int   = 128,
    use_ce:         bool  = False,
    results_path:   str   = RESULTS_PATH,
    hf_token:       str   = "",
) -> dict:
    """
    Run end-to-end evaluation for one two-tower config.

    Args:
        config_name:  One of config1_baseline, config2_better_retrieval,
                      config3_full_system
        embed_dim:    Embedding dimension (64 for config1, 128 for config2/3)
        use_ce:       Apply cross-encoder re-ranking (True for config3)
        results_path: CSV to append results to
        hf_token:     HF token for pulling artifacts

    Returns:
        dict with all metrics
    """
    t0 = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment: {config_name}")
    logger.info(f"  embed_dim={embed_dim} | cross_encoder={use_ce}")
    logger.info(f"{'='*60}")

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading data...")
    train_df  = pd.read_parquet(f"{BASE}/train.parquet")
    val_df    = pd.read_parquet(f"{BASE}/val.parquet")
    titles_df = pd.read_parquet(f"{BASE}/item_titles.parquet") \
        if Path(f"{BASE}/item_titles.parquet").exists() else None
    all_df    = pd.concat([train_df, val_df], ignore_index=True)

    # Build features
    user_agg      = build_user_features(train_df)
    item_agg      = build_item_features(all_df)
    user_feat_map = user_agg.set_index("user_id")

    # Build title lookup for cross-encoder
    titles_lookup = {}
    if titles_df is not None:
        for _, r in titles_df.iterrows():
            t = r.get("title","")
            c = str(r.get("categories",""))
            titles_lookup[r["asin"]] = (t[:100] if isinstance(t,str) and len(t)>5 else r["asin"], c)

    # Val ground truth
    val_gt = val_df.set_index("user_id")[["asin","silver_label"]].to_dict("index")

    # ── Load model + FAISS ─────────────────────────────────────────────────
    model, faiss_index, item_ids = load_artifacts(config_name, embed_dim, hf_token)

    # ── Generate recommendations ───────────────────────────────────────────
    logger.info("Generating recommendations...")
    user_ids_eval   = []
    recommendations = []
    ground_truths   = []
    histories       = []
    silver_labels   = []

    BATCH = 512
    all_users = user_agg["user_id"].values

    for batch_start in range(0, len(all_users), BATCH):
        batch_users = all_users[batch_start:batch_start+BATCH]

        user_vecs, valid_uids = [], []
        for uid in batch_users:
            if uid not in user_feat_map.index or uid not in val_gt:
                continue
            feats   = user_feat_map.loc[uid, USER_COLS].values.astype(np.float32)
            n_inter = float(user_feat_map.loc[uid,"interaction_count_norm"]*625140)
            user_vecs.append((uid, feats, n_inter))
            valid_uids.append(uid)

        if not user_vecs:
            continue

        feat_t = torch.tensor(np.array([v[1] for v in user_vecs]))
        ni_t   = torch.tensor([v[2] for v in user_vecs])
        with torch.no_grad():
            ue = model.encode_user(feat_t, ni_t)
            qv = model.get_combined(ue).numpy().astype(np.float32)

        _, indices = faiss_index.search(qv, RECALL_K)

        for i, uid in enumerate(valid_uids):
            gt_asin  = val_gt[uid]["asin"]
            gt_label = val_gt[uid]["silver_label"]
            cand_asins = item_ids[indices[i]].tolist()

            user_ids_eval.append(uid)
            recommendations.append(cand_asins)
            ground_truths.append([gt_asin])
            silver_labels.append({gt_asin: float(gt_label)})

            # History for cross-encoder
            if use_ce:
                u_items = train_df[train_df["user_id"]==uid].sort_values(
                    "timestamp", ascending=False).head(8)["asin"].tolist()
                hist_titles = [titles_lookup.get(a,(a,""))[0][:50] for a in u_items]
                histories.append("bought " + ", ".join(hist_titles))

        if batch_start % (BATCH*20) == 0:
            logger.info(f"  {batch_start+len(batch_users):,}/{len(all_users):,} users processed...")

    logger.info(f"  {len(user_ids_eval):,} users with valid val interactions")

    # ── Optional cross-encoder re-ranking ─────────────────────────────────
    if use_ce and histories:
        logger.info("Applying cross-encoder re-ranking...")
        recommendations = rerank_with_cross_encoder(
            recommendations, histories, titles_lookup, top_k=TOP_K)

    # ── Evaluate ───────────────────────────────────────────────────────────
    logger.info("Computing metrics...")
    results = evaluate_rankings(
        user_ids     = user_ids_eval,
        recommended  = recommendations,
        ground_truth = ground_truths,
        catalog_size = CATALOG_SIZE,
        silver_labels= silver_labels,
        k            = TOP_K,
        recall_k     = RECALL_K,
    )

    elapsed = time.time() - t0
    results.update({
        "config":      config_name,
        "description": f"Config {config_name} — embed_dim={embed_dim} ce={use_ce}",
        "source":      "real",
        "runtime_min": round(elapsed/60, 1),
        "recall_synthetic":   None,  # computed separately in eval pipeline
        "recall_title_proxy": None,
    })

    # ── Save ───────────────────────────────────────────────────────────────
    _append_to_csv(results, results_path)
    logger.info(f"\n{config_name} complete in {elapsed/60:.1f} min")
    return results


def _append_to_csv(results: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([results])
    if Path(path).exists():
        df_existing = pd.read_csv(path)
        df_existing = df_existing[df_existing["config"] != results["config"]]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(path, index=False)
    logger.info(f"  Results saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RecSys evaluation")
    parser.add_argument("--config",
        choices=["all","als","config1","config2","config3"], required=True)
    parser.add_argument("--pull_from_hf", action="store_true")
    parser.add_argument("--results_path", default=RESULTS_PATH)
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN","") if args.pull_from_hf else ""

    configs_to_run = {
        "all":     ["als","config1","config2","config3"],
        "als":     ["als"],
        "config1": ["config1"],
        "config2": ["config2"],
        "config3": ["config3"],
    }[args.config]

    for cfg in configs_to_run:
        if cfg == "als":
            from experiments.als_baseline import run_als_baseline
            run_als_baseline(results_path=args.results_path)
        elif cfg == "config1":
            run_experiment("config1_baseline", embed_dim=64,
                           use_ce=False, results_path=args.results_path,
                           hf_token=hf_token)
        elif cfg == "config2":
            run_experiment("config2_better_retrieval", embed_dim=128,
                           use_ce=False, results_path=args.results_path,
                           hf_token=hf_token)
        elif cfg == "config3":
            run_experiment("config3_full_system", embed_dim=128,
                           use_ce=True, results_path=args.results_path,
                           hf_token=hf_token)
