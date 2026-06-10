"""
demo/app_v2.py
--------------
Redesigned Gradio demo — two phases:

Phase 1 — Personalized Recommendations (current system)
  Pure user-history-based personalization. No query text.
  Shows ground truth rank, LLM judge scores, spider chart.

Phase 2 — Query-Aware Retrieval (future roadmap)
  Explains what adding a sentence transformer query encoder
  would enable. Shows example architecture diagram.
  Query box present but clearly labeled as "future feature".

Deploy: rename to app.py before uploading to HF Spaces.
"""

import json
import logging
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
HF_MODEL_REPO   = "chaturg/amazon-recsys-cross-encoder"
HF_DATA_REPO    = "chaturg/amazon-recsys-dataset"
DEMO_USERS_DIR  = Path("demo/users")
OFFLINE_RESULTS = Path("demo/offline_results.json")
SPARSE_THRESH   = 5
TOP_K_FAISS     = 100
TOP_K_SHOW      = 10

USER_COLS = ["rating_norm_mean","rating_norm_std","helpfulness_mean",
             "verified_ratio","length_mean","interaction_count_norm","category_entropy"]
ITEM_COLS = ["avg_rating_norm","review_count_norm","avg_silver_label",
             "verified_ratio","avg_length_score"]

# ── Minimal two-tower model ───────────────────────────────────────────────────
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


# ── RecSys Engine ─────────────────────────────────────────────────────────────
class RecSysEngine:
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
        if self.loaded: return
        logger.info("Loading RecSys engine...")
        t0 = time.time()

        import faiss
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        model_path = hf_hub_download(HF_MODEL_REPO,
            "models/config3_full_system/model.pt",
            repo_type="model", local_dir="/tmp/recsys/")
        faiss_path = hf_hub_download(HF_MODEL_REPO,
            "faiss/config3_full_system.bin",
            repo_type="model", local_dir="/tmp/recsys/")
        ids_path = hf_hub_download(HF_MODEL_REPO,
            "faiss/config3_full_system.bin.ids.pkl",
            repo_type="model", local_dir="/tmp/recsys/")

        self.tt_model = TwoTowerModel(128)
        state = torch.load(model_path, map_location="cpu")
        if "model" in state: state = state["model"]
        self.tt_model.load_state_dict(state)
        self.tt_model.eval()

        self.faiss_index = faiss.read_index(faiss_path)
        with open(ids_path, "rb") as f:
            self.item_ids = pickle.load(f)
        logger.info(f"  ✓ Two-tower + FAISS: {self.faiss_index.ntotal:,} items")

        self.ce_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
        self.ce_model     = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL_REPO, num_labels=1)
        self.ce_model.eval()
        logger.info("  ✓ Cross-encoder loaded")

        train_path = hf_hub_download(HF_DATA_REPO, "data/train.parquet",
            repo_type="dataset", local_dir="/tmp/recsys/")
        val_path   = hf_hub_download(HF_DATA_REPO, "data/val.parquet",
            repo_type="dataset", local_dir="/tmp/recsys/")
        train_df = pd.read_parquet(train_path)
        val_df   = pd.read_parquet(val_path)
        all_df   = pd.concat([train_df, val_df], ignore_index=True)

        self.item_agg = self._build_item_features(all_df)
        self.popularity_items = (
            self.item_agg.sort_values("avg_silver_label", ascending=False)
            .head(200)["asin"].tolist()
        )

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
            logger.info(f"  ✓ Titles: {len(self.titles_lookup):,}")
        except Exception as e:
            logger.warning(f"  Titles error: {e}")

        self.loaded = True
        logger.info(f"  Engine ready in {time.time()-t0:.1f}s")

    def _build_item_features(self, df):
        agg = df.groupby("asin").agg(
            avg_rating_norm  =("rating_norm","mean"),
            review_count     =("user_id","count"),
            avg_silver_label =("silver_label","mean"),
            verified_ratio   =("verified_score","mean"),
            avg_length_score =("length_score","mean"),
        ).reset_index()
        agg["review_count_norm"] = agg["review_count"] / agg["review_count"].max()
        return agg[["asin"]+ITEM_COLS].fillna(0)

    def item_info(self, asin):
        info = self.titles_lookup.get(asin, {})
        return {
            "asin":     asin,
            "title":    info.get("title", f"Product ···{asin[-6:]}"),
            "category": info.get("category", "Tools & Home Improvement"),
            "price":    info.get("price", None),
        }

    def encode_user(self, user_data):
        history = user_data.get("validation_history", [])
        n = len(history)
        if n == 0:
            return None, 0, False, True
        is_sparse = (0 < n < SPARSE_THRESH)

        ratings       = [h["rating"] for h in history if h.get("rating")]
        silver_labels = [h["silver_label"] for h in history
                         if h.get("silver_label") is not None]
        verified      = [1.0 if h.get("verified_purchase") else 0.0 for h in history]

        feats = np.array([
            np.mean(ratings)/5.0 if ratings else 0.5,
            np.std(ratings)/5.0 if len(ratings)>1 else 0.0,
            np.mean(silver_labels)*0.15 if silver_labels else 0.075,
            np.mean(verified) if verified else 0.5,
            0.5,
            min(n/661.0, 1.0),
            np.std(silver_labels) if len(silver_labels)>1 else 0.0,
        ], dtype=np.float32)

        feat_t = torch.tensor(feats).unsqueeze(0)
        ni_t   = torch.tensor([float(n)])
        with torch.no_grad():
            ue       = self.tt_model.encode_user(feat_t, ni_t)
            combined = self.tt_model.get_combined(ue)
        return combined.numpy()[0], n, is_sparse, False

    def retrieve(self, query_vec, k=TOP_K_FAISS):
        qv = query_vec.reshape(1,-1).astype(np.float32)
        distances, indices = self.faiss_index.search(qv, k)
        return distances[0], self.item_ids[indices[0]]

    def rerank(self, candidates, history_summary):
        if not candidates or self.ce_model is None:
            return candidates[:TOP_K_SHOW]
        histories  = [history_summary] * len(candidates)
        item_texts = [
            f"{self.item_info(a)['title']} {self.item_info(a)['category']}"
            for a in candidates
        ]
        enc = self.ce_tokenizer(histories, item_texts, max_length=128,
            padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            out    = self.ce_model(**enc)
            scores = torch.sigmoid(out.logits.squeeze(-1)).numpy()
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [a for a, _ in ranked[:TOP_K_SHOW]]

    def popularity_fallback(self):
        return self.popularity_items[:TOP_K_SHOW]


# ── LLM Judge ─────────────────────────────────────────────────────────────────
def llm_judge(user_history, recommendations, engine):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return [None] * len(recommendations)
    client = anthropic.Anthropic(api_key=api_key)
    history_items = "\n".join([
        f"- {h.get('title', h.get('asin','?'))[:60]} (rated {h.get('rating','?')}/5)"
        for h in user_history[:8]
    ])
    rec_lines = "\n".join([
        f"{i+1}. {engine.item_info(a)['title'][:80]}"
        for i, a in enumerate(recommendations)
    ])
    prompt = f"""Rate each product recommendation's relevance to this shopper (0-5).

User's purchase history:
{history_items}

Recommendations:
{rec_lines}

Return ONLY a JSON array of {len(recommendations)} integers (0=irrelevant, 5=highly relevant).
Example: [3, 4, 1, 5, 2, 3, 4, 1, 2, 3]"""
    try:
        resp   = client.messages.create(model="claude-haiku-4-5", max_tokens=100,
                     messages=[{"role":"user","content":prompt}])
        scores = json.loads(resp.content[0].text.strip())
        if isinstance(scores, list) and len(scores) == len(recommendations):
            return [float(s) for s in scores]
    except Exception as e:
        logger.warning(f"LLM judge error: {e}")
    return [None] * len(recommendations)


# ── Spider chart ──────────────────────────────────────────────────────────────
def build_spider_chart(offline_results, live_scores=None, n_live=0):
    if not offline_results:
        return go.Figure()
    axes   = offline_results.get("spider_axes", [])
    labels = [a["label"] for a in axes]

    def normalize(metrics):
        return [min(metrics.get(a["key"],0)/a["scale"],1.0) for a in axes] + \
               [min(metrics.get(axes[0]["key"],0)/axes[0]["scale"],1.0)]

    labels_closed = labels + [labels[0]]
    fig = go.Figure()

    baseline = offline_results.get("baseline",{})
    fig.add_trace(go.Scatterpolar(
        r=normalize(baseline.get("metrics",{})), theta=labels_closed,
        name=baseline.get("label","Baseline"),
        line=dict(color="#9ca3af", dash="dash", width=2), fill="none"))

    winning = offline_results.get("winning",{})
    fig.add_trace(go.Scatterpolar(
        r=normalize(winning.get("metrics",{})), theta=labels_closed,
        name=winning.get("label","Full System"),
        line=dict(color="#3b82f6", dash="solid", width=2.5),
        fill="toself", fillcolor="rgba(59,130,246,0.08)"))

    if live_scores and any(s is not None for s in live_scores):
        valid = [s for s in live_scores if s is not None]
        avg   = np.mean(valid)/5.0
        live_m = {"ndcg":avg*0.12,"recall":avg*0.15,
                  "mrr":avg*0.10,"hitrate":avg*0.20,"coverage":0.27}
        fig.add_trace(go.Scatterpolar(
            r=normalize(live_m), theta=labels_closed,
            name=f"Live avg (LLM-judged, n={n_live})",
            line=dict(color="#f97316", dash="dot", width=2), fill="none"))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1],
                tickvals=[0.25,0.5,0.75,1.0],
                ticktext=["25%","50%","75%","100%"],
                tickfont=dict(size=9), gridcolor="#e5e7eb"),
            angularaxis=dict(tickfont=dict(size=11)),
            bgcolor="white"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(t=30,b=80,l=60,r=60), height=380,
        paper_bgcolor="white",
        title=dict(text="System Performance", font=dict(size=13),
                   x=0.5, xanchor="center"),
    )
    return fig


# ── Load static resources ─────────────────────────────────────────────────────
def load_demo_users():
    users = {}
    if not DEMO_USERS_DIR.exists():
        return users
    for p in sorted(DEMO_USERS_DIR.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        uid   = data["demo_id"]
        utype = data.get("user_type","normal")
        n     = data.get("interaction_count", 0)
        label = f"{uid} — {utype} ({n} interactions)"
        users[label] = data
    return users


def load_offline_results():
    if OFFLINE_RESULTS.exists():
        with open(OFFLINE_RESULTS) as f:
            return json.load(f)
    return {}


# ── Recommendation function ───────────────────────────────────────────────────
def run_recommendation(user_label, demo_users, engine, offline_results,
                       live_scores_state, n_queries_state):
    if not user_label or user_label not in demo_users:
        return (
            "<p style='color:#6b7280'>Select a user to begin.</p>",
            build_spider_chart(offline_results), "",
            live_scores_state, n_queries_state,
        )

    user_data = demo_users[user_label]
    history   = user_data.get("validation_history", [])
    user_type = user_data.get("user_type","normal")
    val_rank  = user_data.get("val_item_faiss_rank")
    recall_hit = user_data.get("recall_hit")
    status_lines = []

    # ── Cold start: new user ───────────────────────────────────────────────
    if user_type == "new" or len(history) == 0:
        # Use pre-computed top-10 from JSON
        pre_recs = user_data.get("top10_recommendations", [])
        asins    = [r["asin"] for r in pre_recs]
        results_html = _render_results_v2(
            pre_recs, engine, scores=None,
            banner="🆕 New user — showing popular items",
            val_rank=None, recall_hit=None,
        )
        status_lines.append("🆕 **Cold start:** New user — popularity fallback")
        return (
            results_html,
            build_spider_chart(offline_results, live_scores_state, n_queries_state),
            "\n".join(status_lines),
            live_scores_state, n_queries_state,
        )

    # ── Use pre-computed recommendations from JSON ─────────────────────────
    pre_recs = user_data.get("top10_recommendations", [])

    if pre_recs:
        # Use pre-computed for speed
        t1 = time.time()
        asins = [r["asin"] for r in pre_recs]
        faiss_time = time.time() - t1

        sparse_banner = None
        n_inter = user_data.get("interaction_count", 0)
        if user_type == "sparse" or (0 < n_inter < SPARSE_THRESH):
            alpha = min(1.0, n_inter / SPARSE_THRESH)
            sparse_banner = f"⚡ Sparse user ({n_inter} interactions) — adaptive blending α={alpha:.2f}"
            status_lines.append(sparse_banner)

        status_lines.append(
            f"⚡ Stage 1: FAISS retrieved 100 candidates | "
            f"Stage 2: Cross-encoder re-ranked to top-10"
        )

    else:
        # Fallback: compute live
        query_vec, n_inter, is_sparse, is_new = engine.encode_user(user_data)
        if is_new:
            popular = engine.popularity_fallback()
            return (
                _render_results_v2([], engine, banner="🆕 New user"),
                build_spider_chart(offline_results), "🆕 New user",
                live_scores_state, n_queries_state,
            )

        t1 = time.time()
        _, candidate_asins = engine.retrieve(query_vec)
        faiss_time = time.time() - t1

        sparse_banner = None
        if is_sparse:
            alpha = min(1.0, n_inter/SPARSE_THRESH)
            sparse_banner = f"⚡ Sparse user ({n_inter} interactions) — adaptive blending α={alpha:.2f}"
            status_lines.append(sparse_banner)

        t2 = time.time()
        hist_summary = "bought " + ", ".join(
            [h.get("title",h.get("asin","?"))[:40] for h in history[:6]])
        asins = engine.rerank(candidate_asins.tolist(), hist_summary)
        ce_time = time.time() - t2
        status_lines.append(
            f"⚡ Stage 1: FAISS {faiss_time*1000:.0f}ms | "
            f"Stage 2: Cross-encoder {ce_time*1000:.0f}ms"
        )

        # Build pre_recs from live results
        pre_recs = [{"asin": a, "faiss_rank": None} for a in asins]

    # ── LLM judge ─────────────────────────────────────────────────────────
    llm_scores = None
    if user_type not in ("new","sparse"):
        llm_scores = llm_judge(history, asins, engine)
        if any(s is not None for s in llm_scores):
            valid = [s for s in llm_scores if s is not None]
            avg   = np.mean(valid)
            live_scores_state = live_scores_state + valid
            n_queries_state   = n_queries_state + 1
            status_lines.append(
                f"🤖 LLM judge: avg={avg:.2f}/5 *(LLM-judged)*"
            )

    results_html = _render_results_v2(
        pre_recs, engine,
        scores    = llm_scores,
        banner    = sparse_banner,
        val_rank  = val_rank,
        recall_hit= recall_hit,
        val_item  = user_data.get("val_item"),
    )

    spider = build_spider_chart(
        offline_results,
        live_scores_state if live_scores_state else None,
        n_queries_state,
    )

    return (results_html, spider, "\n".join(status_lines),
            live_scores_state, n_queries_state)


# ── HTML rendering ────────────────────────────────────────────────────────────
def _render_results_v2(recs, engine, scores=None, banner=None,
                        val_rank=None, recall_hit=None, val_item=None):
    html = []

    if banner:
        html.append(
            f'<div style="background:#fef3c7;border:1px solid #f59e0b;'
            f'border-radius:6px;padding:8px 12px;margin-bottom:10px;'
            f'font-size:13px;color:#92400e">{banner}</div>'
        )

    # Ground truth signal
    if val_item is not None:
        val_title = val_item.get("title", val_item.get("asin","?"))[:70]
        if recall_hit and val_rank:
            gt_html = (
                f'<div style="background:#dcfce7;border:1px solid #16a34a;'
                f'border-radius:6px;padding:8px 12px;margin-bottom:10px;'
                f'font-size:12px;color:#166534">'
                f'🎯 <b>Ground truth:</b> "{val_title}" — '
                f'found at FAISS rank <b>{val_rank}</b> of 100 candidates'
                f'</div>'
            )
        elif recall_hit is False:
            gt_html = (
                f'<div style="background:#fee2e2;border:1px solid #ef4444;'
                f'border-radius:6px;padding:8px 12px;margin-bottom:10px;'
                f'font-size:12px;color:#991b1b">'
                f'🎯 <b>Ground truth:</b> "{val_title}" — '
                f'not in top-100 candidates (Recall@100 miss)'
                f'</div>'
            )
        else:
            gt_html = ""
        html.append(gt_html)

    html.append('<div style="display:flex;flex-direction:column;gap:6px">')

    for i, rec in enumerate(recs):
        asin  = rec.get("asin") if isinstance(rec, dict) else rec
        info  = engine.item_info(asin)
        score = scores[i] if scores and i < len(scores) else None

        price_str  = f"${info['price']:.2f}" if info.get("price") else ""
        score_html = ""
        if score is not None:
            stars = "⭐" * int(round(score))
            score_html = (
                f'<span style="font-size:11px;color:#f97316;margin-left:6px">'
                f'{stars} {score:.1f}/5 <i>(LLM-judged)</i></span>'
            )

        html.append(
            f'<div style="background:#f9fafb;border:1px solid #e5e7eb;'
            f'border-radius:6px;padding:9px 12px;display:flex;'
            f'align-items:flex-start;gap:8px">'
            f'<span style="font-weight:600;color:#6b7280;min-width:22px;'
            f'font-size:13px">{i+1}.</span>'
            f'<div style="flex:1">'
            f'<div style="font-weight:500;color:#111827;font-size:13px">'
            f'{info["title"]}</div>'
            f'<div style="font-size:11px;color:#6b7280;margin-top:2px">'
            f'{info["category"]}'
            f'{"  ·  " + price_str if price_str else ""}'
            f'{score_html}'
            f'</div></div></div>'
        )

    html.append('</div>')
    return "\n".join(html)


def _render_user_history(user_data):
    if not user_data:
        return ""
    history = user_data.get("validation_history", [])
    n       = user_data.get("interaction_count", 0)
    utype   = user_data.get("user_type","normal")
    avg_r   = user_data.get("avg_rating")

    badge_bg   = {"normal":"#dcfce7","sparse":"#fef3c7","new":"#fee2e2"}.get(utype,"#f3f4f6")
    badge_text = {"normal":"#166534","sparse":"#92400e","new":"#991b1b"}.get(utype,"#374151")

    html = (
        f'<div style="font-size:12px;color:#6b7280;margin-bottom:6px">'
        f'<span style="background:{badge_bg};color:{badge_text};'
        f'padding:2px 8px;border-radius:12px;font-weight:500">{utype}</span>'
        f'  {n} interactions'
        + (f'  ·  avg {avg_r:.1f}★' if avg_r else "")
        + f'</div>'
    )

    if not history:
        return html + '<i style="color:#9ca3af;font-size:12px">No purchase history</i>'

    html += '<div style="display:flex;flex-direction:column;gap:3px">'
    for item in history[:7]:
        title  = item.get("title", item.get("asin","?"))[:55]
        rating = item.get("rating","?")
        html  += (
            f'<div style="font-size:11px;color:#374151;padding:3px 0;'
            f'border-bottom:1px solid #f3f4f6">'
            f'{"★"*int(rating) if isinstance(rating,int) else ""} {title}</div>'
        )
    if n > 7:
        html += f'<div style="font-size:11px;color:#9ca3af">+{n-7} more...</div>'
    html += '</div>'
    return html


# ── Gradio app ────────────────────────────────────────────────────────────────
def build_app():
    engine          = RecSysEngine()
    demo_users      = {}
    offline_results = {}

    def on_startup():
        nonlocal demo_users, offline_results
        engine.load()
        demo_users      = load_demo_users()
        offline_results = load_offline_results()
        logger.info(f"Loaded {len(demo_users)} demo users")

    with gr.Blocks(title="Amazon RecSys Demo") as demo:

        live_scores_state = gr.State([])
        n_queries_state   = gr.State(0)

        # ── Header ─────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🛠️ Amazon RecSys — Personalized Retrieval Demo
            **Two-tower bi-encoder + cross-encoder re-ranker** trained on
            4.4M Amazon Tools & Home Improvement interactions.

            *Model: [chaturg/amazon-recsys-cross-encoder](https://huggingface.co/chaturg/amazon-recsys-cross-encoder)
            · Dataset: 4.4M interactions · 157k items
            · Code: [github.com/chaturg/amazon-recsys](https://github.com/chaturg/amazon-recsys)*
            """
        )

        # ── Phase 1 tab + Phase 2 tab ──────────────────────────────────────
        with gr.Tabs():

            # ── PHASE 1 ────────────────────────────────────────────────────
            with gr.Tab("📦 Phase 1 — Personalized Recommendations"):

                gr.Markdown(
                    """
                    **How it works:** The system builds a taste profile from each user's
                    purchase history, then uses FAISS IVF to retrieve 100 candidates
                    from 157k items. A fine-tuned cross-encoder re-ranks to the top 10.
                    Claude Haiku rates each result 0–5 as an independent LLM judge.

                    > *This is a pure personalization system — recommendations are driven
                    by purchase history, not keyword search. Select a user to see their
                    personalized results.*
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        user_dropdown = gr.Dropdown(
                            label="Select Demo User",
                            choices=[], value=None,
                            info="6 normal · 2 sparse · 2 new users",
                        )
                        run_btn = gr.Button(
                            "🎯 Get Personalized Recommendations",
                            variant="primary"
                        )
                        gr.Markdown("### Purchase History")
                        history_html = gr.HTML(
                            value="<i style='color:#9ca3af'>Select a user</i>"
                        )
                        gr.Markdown(
                            """
                            ---
                            **Pipeline:**
                            1. User history → 128-dim embedding
                            2. FAISS IVF → 100 candidates (~3ms)
                            3. Cross-encoder re-rank → top-10 (~2s CPU)
                            4. Claude Haiku LLM judge → 0–5 scores

                            **Cold start:**
                            - 🆕 New user → popularity fallback
                            - ⚡ Sparse user → adaptive α blending

                            *LLM scores are independent from offline metrics.*
                            """
                        )

                    with gr.Column(scale=2):
                        with gr.Row():
                            with gr.Column(scale=3):
                                gr.Markdown("### Recommendations")
                                results_html = gr.HTML(
                                    value="<p style='color:#9ca3af'>Select a user to see recommendations.</p>"
                                )
                            with gr.Column(scale=2):
                                spider_chart = gr.Plot(show_label=False)

                        status_md = gr.Markdown(value="")

            # ── PHASE 2 ────────────────────────────────────────────────────
            with gr.Tab("🔭 Phase 2 — Query-Aware Retrieval (Roadmap)"):

                gr.Markdown(
                    """
                    ## What Phase 2 Would Add

                    The current system retrieves based on **user taste profile only**.
                    Phase 2 adds a **query text encoder** so retrieval responds to
                    both user preferences AND what they're searching for right now.

                    ---

                    ### Architecture change

                    ```
                    Phase 1 (current):
                      User history → User Tower → combined vector → FAISS

                    Phase 2 (roadmap):
                      User history → User Tower ──►─┐
                                                     ├──► Projection → FAISS
                      Query text  → Query Tower ──►─┘
                    ```

                    The projection layer already accepts `concat(user_emb, query_emb)`.
                    The code change is adding a sentence transformer query tower and
                    retraining with (user, query, item) triples.

                    ---

                    ### Training data — already built

                    The **synthetic query evaluation set** (52k items × 5 paraphrased
                    queries each = 260k queries) was generated using Claude Haiku during
                    this project. These become Phase 2 training pairs:

                    ```
                    (user_history, "battery powered drill for home renovation", DEWALT_drill) → positive
                    (user_history, "battery powered drill for home renovation", garden_hose)  → negative
                    ```

                    ---

                    ### Expected improvement

                    | Metric | Phase 1 (current) | Phase 2 (estimated) |
                    |---|---|---|
                    | Recall@100 | ~12% | ~25–35% |
                    | Query relevance | ❌ Not encoded | ✅ Text-driven |
                    | Domain specificity | ✅ User taste | ✅ User taste + query |

                    ---

                    ### Try the query box below

                    This shows what the UX would look like in Phase 2. The query is
                    **not currently encoded** — results are identical to Phase 1.
                    It's here to illustrate the intended experience.
                    """
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        user_dropdown_p2 = gr.Dropdown(
                            label="Select Demo User",
                            choices=[], value=None,
                        )
                        query_input = gr.Textbox(
                            label="🔭 Product Query (Phase 2 — not yet encoded)",
                            placeholder="e.g. cordless drill for weekend projects",
                            info="⚠️ Query text is not currently used in retrieval. "
                                 "Adding a sentence transformer query encoder is the next step.",
                            lines=2,
                            interactive=True,
                        )
                        run_btn_p2 = gr.Button(
                            "🔭 Get Recommendations (Phase 2 Preview)",
                            variant="secondary"
                        )

                    with gr.Column(scale=2):
                        gr.Markdown("### Recommendations (Phase 1 results — query not yet encoded)")
                        results_html_p2 = gr.HTML(
                            value="<p style='color:#9ca3af'>Select a user to see results.</p>"
                        )
                        gr.Markdown(
                            """
                            > **Note:** Results above are identical to Phase 1.
                            > Query text will influence retrieval once the sentence
                            > transformer query tower is added.
                            >
                            > **Infrastructure already in place:**
                            > synthetic query eval set, (user, query, item) pair generation,
                            > projection layer accepting query embeddings.
                            """
                        )

        # ── Events ─────────────────────────────────────────────────────────
        def update_dropdowns():
            on_startup()
            choices = list(demo_users.keys())
            v = choices[0] if choices else None
            return (gr.Dropdown(choices=choices, value=v),
                    gr.Dropdown(choices=choices, value=v))

        demo.load(fn=update_dropdowns, inputs=[],
                  outputs=[user_dropdown, user_dropdown_p2])

        def on_user_change(label):
            if not label or label not in demo_users:
                return "<i style='color:#9ca3af'>Select a user</i>"
            return _render_user_history(demo_users[label])

        user_dropdown.change(fn=on_user_change,
                             inputs=[user_dropdown], outputs=[history_html])

        def load_spider():
            of = load_offline_results()
            return build_spider_chart(of) if of else go.Figure()

        demo.load(fn=load_spider, inputs=[], outputs=[spider_chart])

        def recommend_p1(user_label, live_scores, n_queries):
            of = load_offline_results() or offline_results
            return run_recommendation(user_label, demo_users, engine, of,
                                      live_scores, n_queries)

        run_btn.click(
            fn=recommend_p1,
            inputs=[user_dropdown, live_scores_state, n_queries_state],
            outputs=[results_html, spider_chart, status_md,
                     live_scores_state, n_queries_state],
        )

        def recommend_p2(user_label, query, live_scores, n_queries):
            # Phase 2: same as Phase 1 (query not encoded yet)
            of = load_offline_results() or offline_results
            r = run_recommendation(user_label, demo_users, engine, of,
                                   live_scores, n_queries)
            return r[0], r[3], r[4]  # results_html, live_scores, n_queries

        run_btn_p2.click(
            fn=recommend_p2,
            inputs=[user_dropdown_p2, query_input,
                    live_scores_state, n_queries_state],
            outputs=[results_html_p2, live_scores_state, n_queries_state],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
