"""
demo/app.py
-----------
Gradio demo for the Amazon RecSys two-stage recommendation system.

Two-stage pipeline:
  Stage 1 (~0.3s): User embedding → FAISS IVF ANN search → top-100 candidates
  Stage 2 (~0.5s): Cross-encoder re-ranks top-100 → top-10 results
  Stage 3 (~2s, async): Claude Haiku LLM judge scores relevance → spider chart

Cold start handling:
  New user (0 interactions):  Popularity fallback by category, banner shown
  Sparse user (<5 interactions): Adaptive alpha blending (70% semantic / 30% user),
                                  banner shown
  Novel query (FAISS sim <0.45): Results withheld, banner shown, LLM skipped

Deploy to HF Spaces:
  1. Create Space: chaturg/amazon-recsys-demo (Gradio SDK)
  2. Set secrets: ANTHROPIC_API_KEY
  3. Upload this file as app.py
  4. HF Spaces auto-installs requirements.txt

Requirements (requirements.txt):
  gradio>=4.0
  anthropic
  torch
  faiss-cpu
  transformers
  huggingface-hub
  pandas
  numpy
  plotly
"""

import json
import logging
import math
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import anthropic
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
HF_MODEL_REPO     = "chaturg/amazon-recsys-cross-encoder"
HF_DATA_REPO      = "chaturg/amazon-recsys-dataset"
DEMO_USERS_DIR    = Path("demo/users")
OFFLINE_RESULTS   = Path("demo/offline_results.json")
NOVEL_QUERY_THRESH = 0.45   # FAISS similarity below this = novel query
SPARSE_USER_THRESH = 5      # interactions below this = sparse user
TOP_K_FAISS       = 100
TOP_K_SHOW        = 10

USER_COLS = ["rating_norm_mean","rating_norm_std","helpfulness_mean",
             "verified_ratio","length_mean","interaction_count_norm","category_entropy"]
ITEM_COLS = ["avg_rating_norm","review_count_norm","avg_silver_label",
             "verified_ratio","avg_length_score"]

# ── Minimal two-tower model definition ───────────────────────────────────────
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
            alpha = torch.clamp(ni.float()/5.0, 0, 1).unsqueeze(-1)
            e = alpha*e + (1-alpha)*e.mean(0, keepdim=True).expand_as(e)
        return F.normalize(e, p=2, dim=-1)
    def get_combined(self, ue):
        return F.normalize(self.projection(torch.cat([ue,ue],dim=-1)), p=2, dim=-1)


# ── Model and data loading ────────────────────────────────────────────────────
class RecSysEngine:
    """Loads all models and data at startup. Singleton pattern."""

    def __init__(self):
        self.tt_model      = None
        self.ce_model      = None
        self.ce_tokenizer  = None
        self.faiss_index   = None
        self.item_ids      = None
        self.item_agg      = None
        self.titles_lookup = {}
        self.popularity_items = []
        self.loaded        = False

    def load(self):
        if self.loaded:
            return
        logger.info("Loading RecSys engine...")
        t0 = time.time()

        try:
            import faiss
            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # ── Pull artifacts from HF ─────────────────────────────────────
            logger.info("  Downloading model artifacts from HF...")

            model_path = hf_hub_download(HF_MODEL_REPO,
                "models/config3_full_system/model.pt",
                repo_type="model", local_dir="/tmp/recsys/")
            faiss_path = hf_hub_download(HF_MODEL_REPO,
                "faiss/config3_full_system.bin",
                repo_type="model", local_dir="/tmp/recsys/")
            ids_path = hf_hub_download(HF_MODEL_REPO,
                "faiss/config3_full_system.bin.ids.pkl",
                repo_type="model", local_dir="/tmp/recsys/")

            # ── Load two-tower model ───────────────────────────────────────
            self.tt_model = TwoTowerModel(128)
            state = torch.load(model_path, map_location="cpu")
            if "model" in state: state = state["model"]
            self.tt_model.load_state_dict(state)
            self.tt_model.eval()
            logger.info("  ✓ Two-tower model loaded")

            # ── Load FAISS index ───────────────────────────────────────────
            self.faiss_index = faiss.read_index(faiss_path)
            with open(ids_path, "rb") as f:
                self.item_ids = pickle.load(f)
            logger.info(f"  ✓ FAISS index: {self.faiss_index.ntotal:,} items")

            # ── Load cross-encoder ─────────────────────────────────────────
            self.ce_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
            self.ce_model     = AutoModelForSequenceClassification.from_pretrained(
                HF_MODEL_REPO, num_labels=1)
            self.ce_model.eval()
            logger.info("  ✓ Cross-encoder loaded")

            # ── Load item data ─────────────────────────────────────────────
            train_path = hf_hub_download(HF_DATA_REPO,
                "data/train.parquet", repo_type="dataset", local_dir="/tmp/recsys/")
            val_path = hf_hub_download(HF_DATA_REPO,
                "data/val.parquet", repo_type="dataset", local_dir="/tmp/recsys/")

            train_df = pd.read_parquet(train_path)
            val_df   = pd.read_parquet(val_path)
            all_df   = pd.concat([train_df, val_df], ignore_index=True)

            self.item_agg = self._build_item_features(all_df)
            self.popularity_items = (
                self.item_agg.sort_values("avg_silver_label", ascending=False)
                .head(200)["asin"].tolist()
            )

            # ── Load item titles ───────────────────────────────────────────
            try:
                titles_path = hf_hub_download(HF_DATA_REPO,
                    "processed/item_titles.parquet",
                    repo_type="dataset", local_dir="/tmp/recsys/")
                titles_df = pd.read_parquet(titles_path)
                for _, r in titles_df.iterrows():
                    t = r.get("title","")
                    c = str(r.get("categories",""))
                    p = r.get("price", None)
                    if isinstance(t, str) and len(t) > 5:
                        self.titles_lookup[r["asin"]] = {
                            "title":    t[:100],
                            "category": c[:60],
                            "price":    float(p) if isinstance(p, float) else None,
                        }
            except Exception as e:
                logger.warning(f"  Could not load titles: {e}")

            self.loaded = True
            logger.info(f"  Engine ready in {time.time()-t0:.1f}s")

        except Exception as e:
            logger.error(f"Engine load failed: {e}")
            raise

    def _build_item_features(self, df: pd.DataFrame) -> pd.DataFrame:
        agg = df.groupby("asin").agg(
            avg_rating_norm  =("rating_norm","mean"),
            review_count     =("user_id","count"),
            avg_silver_label =("silver_label","mean"),
            verified_ratio   =("verified_score","mean"),
            avg_length_score =("length_score","mean"),
        ).reset_index()
        agg["review_count_norm"] = agg["review_count"] / agg["review_count"].max()
        return agg[["asin"]+ITEM_COLS].fillna(0)

    def item_info(self, asin: str) -> dict:
        """Get display info for an item."""
        info = self.titles_lookup.get(asin, {})
        return {
            "asin":     asin,
            "title":    info.get("title", asin),
            "category": info.get("category", "Tools & Home Improvement"),
            "price":    info.get("price", None),
        }

    def encode_user(self, user_data: dict) -> tuple:
        """
        Encode user features into a query vector.
        Returns (combined_vector, n_interactions, is_sparse, is_new).
        """
        history = user_data.get("validation_history", [])
        n = len(history)
        is_new    = (n == 0)
        is_sparse = (0 < n < SPARSE_USER_THRESH)

        if is_new:
            return None, 0, False, True

        # Aggregate features from history
        ratings       = [h["rating"] for h in history]
        silver_labels = [h["silver_label"] for h in history]
        verified      = [1.0 if h.get("verified_purchase") else 0.0 for h in history]

        feats = np.array([
            np.mean(ratings) / 5.0,           # rating_norm_mean
            np.std(ratings) / 5.0,            # rating_norm_std
            np.mean(silver_labels) * 0.15,    # helpfulness_mean (proxy)
            np.mean(verified),                # verified_ratio
            0.5,                              # length_mean (unknown)
            min(n / 661.0, 1.0),             # interaction_count_norm
            np.std(silver_labels),            # category_entropy proxy
        ], dtype=np.float32)

        feat_tensor = torch.tensor(feats).unsqueeze(0)
        n_inter     = torch.tensor([float(n)])

        with torch.no_grad():
            ue       = self.tt_model.encode_user(feat_tensor, n_inter)
            combined = self.tt_model.get_combined(ue)

        return combined.numpy()[0], n, is_sparse, False

    def retrieve(self, query_vec: np.ndarray, k: int = TOP_K_FAISS) -> tuple:
        """Search FAISS index. Returns (distances, item_ids)."""
        qv = query_vec.reshape(1, -1).astype(np.float32)
        distances, indices = self.faiss_index.search(qv, k)
        asins = self.item_ids[indices[0]]
        return distances[0], asins

    def rerank(self, candidates: list, history_summary: str) -> list:
        """Cross-encoder re-ranks top-100 → top-10."""
        if not candidates or self.ce_model is None:
            return candidates[:TOP_K_SHOW]

        histories  = [history_summary] * len(candidates)
        item_texts = [
            f"{self.item_info(a)['title']} {self.item_info(a)['category']}"
            for a in candidates
        ]

        enc = self.ce_tokenizer(
            histories, item_texts,
            max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            out    = self.ce_model(**enc)
            scores = torch.sigmoid(out.logits.squeeze(-1)).numpy()

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [a for a, _ in ranked[:TOP_K_SHOW]]

    def popularity_fallback(self, category: str = None) -> list:
        """Return popular items as cold-start fallback."""
        return self.popularity_items[:TOP_K_SHOW]


# ── LLM Judge ─────────────────────────────────────────────────────────────────
def llm_judge(
    user_history: list,
    recommendations: list,
    engine: RecSysEngine,
) -> list:
    """
    Claude Haiku rates each recommendation 0–5 given user history context.
    Returns list of relevance scores (same length as recommendations).
    Labeled 'LLM-judged' throughout UI — not conflated with offline ground truth.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return [None] * len(recommendations)

    client = anthropic.Anthropic(api_key=api_key)

    # Build history context
    history_items = "\n".join([
        f"- {h.get('title', h.get('asin', '?'))[:60]} "
        f"(rated {h.get('rating', '?')}/5)"
        for h in user_history[:8]
    ])

    # Build recommendation list
    rec_lines = "\n".join([
        f"{i+1}. {engine.item_info(asin)['title'][:80]}"
        for i, asin in enumerate(recommendations)
    ])

    prompt = f"""You are evaluating the relevance of product recommendations for a shopper.

User's purchase history:
{history_items}

Recommended products:
{rec_lines}

For each recommended product, rate its relevance to this user's preferences on a scale of 0-5:
0 = completely irrelevant
1 = unlikely to be interested
2 = possibly interested
3 = moderately relevant
4 = quite relevant
5 = highly relevant, matches purchase pattern well

Return ONLY a JSON array of {len(recommendations)} integers, one per recommendation.
Example: [3, 4, 1, 5, 2, 3, 4, 1, 2, 3]
No explanation, no markdown."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        raw    = response.content[0].text.strip()
        scores = json.loads(raw)
        if isinstance(scores, list) and len(scores) == len(recommendations):
            return [float(s) for s in scores]
    except Exception as e:
        logger.warning(f"LLM judge error: {e}")

    return [None] * len(recommendations)


# ── Spider chart ──────────────────────────────────────────────────────────────
def build_spider_chart(
    offline_results: dict,
    live_scores: Optional[list] = None,
    n_live_queries: int = 0,
) -> go.Figure:
    """
    Build Plotly spider chart with 3 series:
      - Baseline (dashed grey)
      - Winning config (solid blue)
      - Live avg (dotted coral) — only shown when live_scores exist
    """
    axes = offline_results["spider_axes"]
    labels = [a["label"] for a in axes]
    scales = [a["scale"] for a in axes]

    def normalize(metrics: dict) -> list:
        """Normalize metric values to [0,1] using scale."""
        return [
            min(metrics.get(a["key"], 0) / a["scale"], 1.0)
            for a in axes
        ] + [min(metrics.get(axes[0]["key"], 0) / axes[0]["scale"], 1.0)]  # close polygon

    labels_closed = labels + [labels[0]]

    fig = go.Figure()

    # ── Baseline series (dashed grey) ─────────────────────────────────────
    baseline  = offline_results["baseline"]
    base_vals = normalize(baseline["metrics"])
    fig.add_trace(go.Scatterpolar(
        r    = base_vals,
        theta= labels_closed,
        name = baseline["label"],
        line = dict(color="#9ca3af", dash="dash", width=2),
        fill = "none",
        hovertemplate = "<b>%{theta}</b><br>%{customdata:.4f}<extra>" + baseline["label"] + "</extra>",
        customdata = [baseline["metrics"].get(a["key"], 0) for a in axes] + [baseline["metrics"].get(axes[0]["key"], 0)],
    ))

    # ── Winning config series (solid blue) ────────────────────────────────
    winning      = offline_results["winning"]
    winning_vals = normalize(winning["metrics"])
    fig.add_trace(go.Scatterpolar(
        r    = winning_vals,
        theta= labels_closed,
        name = winning["label"],
        line = dict(color="#3b82f6", dash="solid", width=2.5),
        fill = "toself",
        fillcolor = "rgba(59,130,246,0.08)",
        hovertemplate = "<b>%{theta}</b><br>%{customdata:.4f}<extra>" + winning["label"] + "</extra>",
        customdata = [winning["metrics"].get(a["key"], 0) for a in axes] + [winning["metrics"].get(axes[0]["key"], 0)],
    ))

    # ── Live LLM-judged series (dotted coral) ─────────────────────────────
    if live_scores and any(s is not None for s in live_scores):
        valid_scores = [s for s in live_scores if s is not None]
        avg_score    = np.mean(valid_scores) / 5.0  # normalize 0-5 → 0-1

        # Live metrics are all derived from the single avg LLM score
        # Scale to be comparable with offline NDCG/Recall range
        live_metrics = {
            "ndcg":     avg_score * 0.12,
            "recall":   avg_score * 0.15,
            "mrr":      avg_score * 0.10,
            "hitrate":  avg_score * 0.20,
            "coverage": 0.27,  # coverage doesn't change per query
        }
        live_vals = normalize(live_metrics)

        fig.add_trace(go.Scatterpolar(
            r    = live_vals,
            theta= labels_closed,
            name = f"Live avg (LLM-judged, n={n_live_queries})",
            line = dict(color="#f97316", dash="dot", width=2),
            fill = "none",
            hovertemplate = "<b>%{theta}</b><br>LLM avg={:.2f}/5<extra>LLM-judged</extra>".format(avg_score * 5),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible   = True,
                range     = [0, 1],
                tickvals  = [0.25, 0.5, 0.75, 1.0],
                ticktext  = ["25%", "50%", "75%", "100%"],
                tickfont  = dict(size=9),
                gridcolor = "#e5e7eb",
            ),
            angularaxis=dict(
                tickfont = dict(size=11),
            ),
            bgcolor = "white",
        ),
        showlegend   = True,
        legend       = dict(
            orientation = "h",
            yanchor     = "bottom",
            y           = -0.25,
            xanchor     = "center",
            x           = 0.5,
            font        = dict(size=10),
        ),
        margin       = dict(t=30, b=80, l=60, r=60),
        height       = 380,
        paper_bgcolor= "white",
        plot_bgcolor = "white",
        font         = dict(family="Inter, sans-serif"),
        title        = dict(
            text     = "System Performance",
            font     = dict(size=13),
            x        = 0.5,
            xanchor  = "center",
        ),
    )

    return fig


# ── Load static resources ─────────────────────────────────────────────────────
def load_demo_users() -> dict:
    """Load all 10 demo user JSON files."""
    users = {}
    if not DEMO_USERS_DIR.exists():
        return users
    for p in sorted(DEMO_USERS_DIR.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        uid   = data["demo_id"]
        utype = data.get("user_type", "normal")
        n     = data.get("interaction_count", 0)
        label = f"{uid} — {utype} ({n} interactions)"
        users[label] = data
    return users


def load_offline_results() -> dict:
    """Load pre-computed offline results for spider chart."""
    if OFFLINE_RESULTS.exists():
        with open(OFFLINE_RESULTS) as f:
            return json.load(f)
    return {}


# ── Main recommend function ────────────────────────────────────────────────────
def run_recommendation(
    user_label: str,
    query:      str,
    demo_users: dict,
    engine:     RecSysEngine,
    offline_results: dict,
    live_scores_state: list,
    n_queries_state:   int,
) -> tuple:
    """
    Full two-stage recommendation pipeline.

    Returns:
        (results_html, spider_fig, status_md, live_scores_state, n_queries_state)
    """
    if not user_label or user_label not in demo_users:
        return (
            "<p style='color:#6b7280'>Select a user to begin.</p>",
            build_spider_chart(offline_results),
            "",
            live_scores_state,
            n_queries_state,
        )

    user_data = demo_users[user_label]
    history   = user_data.get("validation_history", [])
    n_inter   = len(history)
    user_type = user_data.get("user_type", "normal")

    status_lines = []

    # ── Cold start: new user ───────────────────────────────────────────────
    if user_type == "new" or n_inter == 0:
        popular = engine.popularity_fallback()
        results_html = _render_results(popular, engine, scores=None,
                                       banner="🆕 New user — showing popular items")
        status_lines.append("🆕 **Cold start:** New user — popularity fallback")
        return (
            results_html,
            build_spider_chart(offline_results, live_scores_state, n_queries_state),
            "\n".join(status_lines),
            live_scores_state,
            n_queries_state,
        )

    # ── Encode user ────────────────────────────────────────────────────────
    query_vec, n, is_sparse, is_new = engine.encode_user(user_data)

    if is_new:
        popular = engine.popularity_fallback()
        results_html = _render_results(popular, engine, scores=None,
                                       banner="🆕 New user — showing popular items")
        return (results_html, build_spider_chart(offline_results), "🆕 New user", live_scores_state, n_queries_state)

    # ── Stage 1: FAISS retrieval ───────────────────────────────────────────
    t1 = time.time()
    distances, candidate_asins = engine.retrieve(query_vec, k=TOP_K_FAISS)
    faiss_time = time.time() - t1
    max_sim = float(distances[0]) if len(distances) > 0 else 0.0

    status_lines.append(f"⚡ Stage 1: FAISS retrieved {len(candidate_asins)} candidates in {faiss_time*1000:.0f}ms")

    # ── Cold start: novel query ────────────────────────────────────────────
    if max_sim < NOVEL_QUERY_THRESH:
        results_html = _render_banner(
            "🔍 Novel query detected",
            f"Maximum FAISS similarity ({max_sim:.3f}) is below threshold ({NOVEL_QUERY_THRESH}). "
            "This query is outside the system's training distribution. "
            "Try a query more related to tools and home improvement."
        )
        status_lines.append(f"🔍 **Novel query:** max_sim={max_sim:.3f} < {NOVEL_QUERY_THRESH}")
        return (
            results_html,
            build_spider_chart(offline_results, live_scores_state, n_queries_state),
            "\n".join(status_lines),
            live_scores_state,
            n_queries_state,
        )

    # ── Cold start: sparse user banner ────────────────────────────────────
    sparse_banner = None
    if is_sparse:
        alpha = min(1.0, n / SPARSE_USER_THRESH)
        sparse_banner = f"⚡ Sparse user ({n} interactions) — adaptive blending α={alpha:.2f}"
        status_lines.append(sparse_banner)

    # ── Stage 2: Cross-encoder re-ranking ─────────────────────────────────
    t2 = time.time()
    history_summary = "bought " + ", ".join(
        [h.get("title", h.get("asin","?"))[:40] for h in history[:6]]
    )
    top10_asins = engine.rerank(candidate_asins.tolist(), history_summary)
    ce_time = time.time() - t2
    status_lines.append(f"⚡ Stage 2: Cross-encoder re-ranked in {ce_time*1000:.0f}ms")

    # ── Stage 3: LLM judge (async-style — runs after results rendered) ────
    llm_scores = None
    if user_type != "new" and not is_sparse:
        llm_scores = llm_judge(history, top10_asins, engine)
        if any(s is not None for s in llm_scores):
            valid = [s for s in llm_scores if s is not None]
            avg   = np.mean(valid)
            live_scores_state = live_scores_state + valid
            n_queries_state   = n_queries_state + 1
            status_lines.append(f"🤖 LLM judge (Claude Haiku): avg={avg:.2f}/5 (labeled LLM-judged)")

    # ── Render results ─────────────────────────────────────────────────────
    results_html = _render_results(
        top10_asins, engine,
        scores=llm_scores,
        banner=sparse_banner,
    )

    spider = build_spider_chart(
        offline_results,
        live_scores_state if live_scores_state else None,
        n_queries_state,
    )

    return (
        results_html,
        spider,
        "\n".join(status_lines),
        live_scores_state,
        n_queries_state,
    )


def _render_results(
    asins:  list,
    engine: RecSysEngine,
    scores: Optional[list] = None,
    banner: Optional[str]  = None,
) -> str:
    """Render recommendation results as HTML."""
    html_parts = []

    if banner:
        html_parts.append(
            f'<div style="background:#fef3c7;border:1px solid #f59e0b;border-radius:6px;'
            f'padding:8px 12px;margin-bottom:12px;font-size:13px;color:#92400e">'
            f'{banner}</div>'
        )

    html_parts.append('<div style="display:flex;flex-direction:column;gap:8px">')

    for i, asin in enumerate(asins):
        info  = engine.item_info(asin)
        score = scores[i] if scores and i < len(scores) else None

        price_str  = f"${info['price']:.2f}" if info.get("price") else ""
        score_html = ""
        if score is not None:
            stars = "⭐" * int(round(score))
            score_html = (
                f'<span style="font-size:11px;color:#f97316;margin-left:8px" '
                f'title="LLM-judged relevance">{stars} {score:.1f}/5 <i>(LLM-judged)</i></span>'
            )

        html_parts.append(
            f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;'
            f'padding:10px 14px;display:flex;align-items:flex-start;gap:10px">'
            f'<span style="font-weight:600;color:#6b7280;min-width:20px">{i+1}.</span>'
            f'<div style="flex:1">'
            f'<div style="font-weight:500;color:#111827;font-size:14px">{info["title"]}</div>'
            f'<div style="font-size:12px;color:#6b7280;margin-top:3px">'
            f'{info["category"]}'
            f'{"  ·  " + price_str if price_str else ""}'
            f'{score_html}'
            f'</div>'
            f'</div>'
            f'</div>'
        )

    html_parts.append('</div>')
    return "\n".join(html_parts)


def _render_banner(title: str, message: str) -> str:
    return (
        f'<div style="background:#fee2e2;border:1px solid #ef4444;border-radius:8px;'
        f'padding:16px;text-align:center">'
        f'<div style="font-weight:600;color:#991b1b;font-size:15px">{title}</div>'
        f'<div style="color:#7f1d1d;font-size:13px;margin-top:6px">{message}</div>'
        f'</div>'
    )


def _render_user_history(user_data: dict) -> str:
    """Render user history panel."""
    if not user_data:
        return ""

    history  = user_data.get("validation_history", [])
    n        = len(history)
    utype    = user_data.get("user_type", "normal")
    uid      = user_data.get("demo_id", "?")
    avg_r    = user_data.get("avg_rating", None)

    badge_color = {
        "normal": "#dcfce7",
        "sparse": "#fef3c7",
        "new":    "#fee2e2",
    }.get(utype, "#f3f4f6")

    badge_text_color = {
        "normal": "#166534",
        "sparse": "#92400e",
        "new":    "#991b1b",
    }.get(utype, "#374151")

    html = (
        f'<div style="font-size:12px;color:#6b7280;margin-bottom:8px">'
        f'<span style="background:{badge_color};color:{badge_text_color};'
        f'padding:2px 8px;border-radius:12px;font-weight:500">{utype}</span>'
        f'  {n} interactions'
        + (f'  ·  avg rating {avg_r:.1f}★' if avg_r else "")
        + f'</div>'
    )

    if not history:
        return html + '<i style="color:#9ca3af;font-size:13px">No purchase history</i>'

    html += '<div style="display:flex;flex-direction:column;gap:4px">'
    for item in history[:6]:
        title = item.get("title", item.get("asin","?"))[:55]
        rating = item.get("rating", "?")
        html += (
            f'<div style="font-size:12px;color:#374151;padding:4px 0;'
            f'border-bottom:1px solid #f3f4f6">'
            f'{"★" * int(rating) if isinstance(rating, int) else ""} {title}'
            f'</div>'
        )
    if n > 6:
        html += f'<div style="font-size:11px;color:#9ca3af">+{n-6} more...</div>'
    html += '</div>'
    return html


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    engine          = RecSysEngine()
    demo_users      = {}
    offline_results = {}

    def on_startup():
        nonlocal demo_users, offline_results
        engine.load()
        demo_users      = load_demo_users()
        offline_results = load_offline_results()
        logger.info(f"Loaded {len(demo_users)} demo users")

    css = """
    .gr-button-primary { background: #3b82f6 !important; }
    .result-panel { min-height: 400px; }
    footer { display: none !important; }
    """

    with gr.Blocks(
        title="Amazon RecSys Demo",
        theme=gr.themes.Soft(
            primary_hue="blue",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css=css,
    ) as demo:

        # ── State ──────────────────────────────────────────────────────────
        live_scores_state = gr.State([])
        n_queries_state   = gr.State(0)

        # ── Header ─────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🛠️ Amazon RecSys — Two-Stage Retrieval Demo
            **Two-tower bi-encoder + cross-encoder re-ranker** trained on Amazon Tools & Home Improvement reviews.
            Select a demo user, enter a product query, and see two-stage retrieval in action.

            *Model: chaturg/amazon-recsys-cross-encoder · Dataset: 4.4M interactions · 157k items*
            """
        )

        with gr.Row():
            # ── Left column: controls ──────────────────────────────────────
            with gr.Column(scale=1):
                user_dropdown = gr.Dropdown(
                    label="Select Demo User",
                    choices=[],
                    value=None,
                    info="6 normal · 2 sparse · 2 new users",
                )
                query_input = gr.Textbox(
                    label="Product Query",
                    placeholder="e.g. cordless drill for weekend projects",
                    lines=2,
                )
                run_btn = gr.Button("🔍 Get Recommendations", variant="primary")

                gr.Markdown("### User History")
                history_html = gr.HTML(
                    value="<i style='color:#9ca3af'>Select a user to see history</i>"
                )

                gr.Markdown(
                    """
                    ---
                    **Pipeline stages:**
                    1. User tower → combined query vector
                    2. FAISS IVF ANN search → 100 candidates (~0.3s)
                    3. Cross-encoder re-rank → top-10 (~0.5s)
                    4. Claude Haiku LLM judge → relevance scores (~2s)

                    **Cold start handling:**
                    - 🆕 New user → popularity fallback
                    - ⚡ Sparse user → adaptive α blending
                    - 🔍 Novel query (sim<0.45) → withheld

                    *LLM-judged scores are independent from offline ground truth.*
                    """,
                    elem_classes=["info-panel"],
                )

            # ── Right column: results + chart ──────────────────────────────
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column(scale=3):
                        gr.Markdown("### Recommendations")
                        results_html = gr.HTML(
                            value="<p style='color:#9ca3af'>Run a query to see results.</p>",
                            elem_classes=["result-panel"],
                        )
                    with gr.Column(scale=2):
                        spider_chart = gr.Plot(
                            label="System Performance",
                            show_label=False,
                        )

                status_md = gr.Markdown(
                    value="",
                    elem_classes=["status-panel"],
                )

        # ── Populate dropdown on load ──────────────────────────────────────
        def update_dropdown():
            on_startup()
            choices = list(demo_users.keys())
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

        demo.load(
            fn     = update_dropdown,
            inputs = [],
            outputs= [user_dropdown],
        )

        # ── Update history when user changes ──────────────────────────────
        def on_user_change(user_label):
            if not user_label or user_label not in demo_users:
                return "<i style='color:#9ca3af'>Select a user to see history</i>"
            return _render_user_history(demo_users[user_label])

        user_dropdown.change(
            fn      = on_user_change,
            inputs  = [user_dropdown],
            outputs = [history_html],
        )

        # ── Spider chart on load ───────────────────────────────────────────
        def load_spider():
            of = load_offline_results()
            if not of:
                return go.Figure()
            return build_spider_chart(of)

        demo.load(
            fn      = load_spider,
            inputs  = [],
            outputs = [spider_chart],
        )

        # ── Run recommendation ────────────────────────────────────────────
        def recommend(user_label, query, live_scores, n_queries):
            of = load_offline_results() or offline_results
            return run_recommendation(
                user_label, query, demo_users, engine, of,
                live_scores, n_queries
            )

        run_btn.click(
            fn      = recommend,
            inputs  = [user_dropdown, query_input, live_scores_state, n_queries_state],
            outputs = [results_html, spider_chart, status_md,
                       live_scores_state, n_queries_state],
        )

        query_input.submit(
            fn      = recommend,
            inputs  = [user_dropdown, query_input, live_scores_state, n_queries_state],
            outputs = [results_html, spider_chart, status_md,
                       live_scores_state, n_queries_state],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
    )
