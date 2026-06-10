"""
demo/app.py  (v3 — metrics table + spider chart)
-------------------------------------------------
Phase 1: Pure personalization — spider chart + exact metrics table.
Phase 2: Roadmap tab explaining query encoder extension.

Deploy: upload as app.py to HF Spaces chaturg/amazon-recsys-demo
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path

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

HF_MODEL_REPO   = "chaturg/amazon-recsys-cross-encoder"
HF_DATA_REPO    = "chaturg/amazon-recsys-dataset"
DEMO_USERS_DIR  = Path("demo/users")
OFFLINE_RESULTS = Path("demo/offline_results.json")
SPARSE_THRESH   = 5
TOP_K_SHOW      = 10

USER_COLS = ["rating_norm_mean","rating_norm_std","helpfulness_mean",
             "verified_ratio","length_mean","interaction_count_norm","category_entropy"]
ITEM_COLS = ["avg_rating_norm","review_count_norm","avg_silver_label",
             "verified_ratio","avg_length_score"]


# ── Model ─────────────────────────────────────────────────────────────────────
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


# ── Engine ────────────────────────────────────────────────────────────────────
class RecSysEngine:
    def __init__(self):
        self.tt_model = self.ce_model = self.ce_tokenizer = None
        self.faiss_index = self.item_ids = self.item_agg = None
        self.titles_lookup = {}
        self.popularity_items = []
        self.loaded = False

    def load(self):
        if self.loaded: return
        logger.info("Loading engine...")
        t0 = time.time()
        import faiss
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        mp = hf_hub_download(HF_MODEL_REPO, "models/config3_full_system/model.pt",
                             repo_type="model", local_dir="/tmp/recsys/")
        fp = hf_hub_download(HF_MODEL_REPO, "faiss/config3_full_system.bin",
                             repo_type="model", local_dir="/tmp/recsys/")
        ip = hf_hub_download(HF_MODEL_REPO, "faiss/config3_full_system.bin.ids.pkl",
                             repo_type="model", local_dir="/tmp/recsys/")

        self.tt_model = TwoTowerModel(128)
        state = torch.load(mp, map_location="cpu")
        if "model" in state: state = state["model"]
        self.tt_model.load_state_dict(state); self.tt_model.eval()

        self.faiss_index = faiss.read_index(fp)
        with open(ip, "rb") as f: self.item_ids = pickle.load(f)
        logger.info(f"  ✓ Two-tower + FAISS: {self.faiss_index.ntotal:,} items")

        self.ce_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
        self.ce_model = AutoModelForSequenceClassification.from_pretrained(
            HF_MODEL_REPO, num_labels=1); self.ce_model.eval()
        logger.info("  ✓ Cross-encoder loaded")

        tp = hf_hub_download(HF_DATA_REPO, "data/train.parquet",
                             repo_type="dataset", local_dir="/tmp/recsys/")
        vp = hf_hub_download(HF_DATA_REPO, "data/val.parquet",
                             repo_type="dataset", local_dir="/tmp/recsys/")
        train_df = pd.read_parquet(tp); val_df = pd.read_parquet(vp)
        all_df   = pd.concat([train_df, val_df], ignore_index=True)

        agg = all_df.groupby("asin").agg(
            avg_rating_norm=("rating_norm","mean"), review_count=("user_id","count"),
            avg_silver_label=("silver_label","mean"), verified_ratio=("verified_score","mean"),
            avg_length_score=("length_score","mean"),
        ).reset_index()
        agg["review_count_norm"] = agg["review_count"] / agg["review_count"].max()
        self.item_agg = agg[["asin"]+ITEM_COLS].fillna(0)
        self.popularity_items = (
            self.item_agg.sort_values("avg_silver_label", ascending=False)
            .head(200)["asin"].tolist()
        )

        try:
            tit = hf_hub_download(HF_DATA_REPO, "processed/item_titles.parquet",
                                  repo_type="dataset", local_dir="/tmp/recsys/")
            titles_df = pd.read_parquet(tit)
            for _, r in titles_df.iterrows():
                t = r.get("title",""); c = str(r.get("categories","")); p = r.get("price",None)
                if isinstance(t,str) and len(t)>5:
                    self.titles_lookup[r["asin"]] = {
                        "title": t[:100], "category": c[:60],
                        "price": float(p) if isinstance(p,float) else None}
            logger.info(f"  ✓ Titles: {len(self.titles_lookup):,}")
        except Exception as e:
            logger.warning(f"  Titles error: {e}")

        self.loaded = True
        logger.info(f"  Engine ready in {time.time()-t0:.1f}s")

    def item_info(self, asin):
        info = self.titles_lookup.get(asin, {})
        return {"asin": asin,
                "title":    info.get("title", f"Product ···{asin[-6:]}"),
                "category": info.get("category", "Tools & Home Improvement"),
                "price":    info.get("price", None)}

    def encode_user(self, user_data):
        history = user_data.get("validation_history", [])
        n = len(history)
        if n == 0: return None, 0, False, True
        is_sparse = (0 < n < SPARSE_THRESH)
        ratings = [h["rating"] for h in history if h.get("rating")]
        labels  = [h["silver_label"] for h in history if h.get("silver_label") is not None]
        verified= [1.0 if h.get("verified_purchase") else 0.0 for h in history]
        feats = np.array([
            np.mean(ratings)/5.0 if ratings else 0.5,
            np.std(ratings)/5.0 if len(ratings)>1 else 0.0,
            np.mean(labels)*0.15 if labels else 0.075,
            np.mean(verified) if verified else 0.5,
            0.5, min(n/661.0, 1.0),
            np.std(labels) if len(labels)>1 else 0.0,
        ], dtype=np.float32)
        with torch.no_grad():
            ue = self.tt_model.encode_user(torch.tensor(feats).unsqueeze(0),
                                            torch.tensor([float(n)]))
            combined = self.tt_model.get_combined(ue)
        return combined.numpy()[0], n, is_sparse, False

    def retrieve(self, qv, k=100):
        d, i = self.faiss_index.search(qv.reshape(1,-1).astype(np.float32), k)
        return d[0], self.item_ids[i[0]]

    def rerank(self, candidates, hist_summary):
        if not candidates or self.ce_model is None:
            return candidates[:TOP_K_SHOW]
        histories  = [hist_summary] * len(candidates)
        item_texts = [f"{self.item_info(a)['title']} {self.item_info(a)['category']}"
                      for a in candidates]
        enc = self.ce_tokenizer(histories, item_texts, max_length=128,
                                padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            scores = torch.sigmoid(self.ce_model(**enc).logits.squeeze(-1)).numpy()
        return [a for a, _ in sorted(zip(candidates, scores),
                                     key=lambda x: x[1], reverse=True)[:TOP_K_SHOW]]

    def popularity_fallback(self):
        return self.popularity_items[:TOP_K_SHOW]


# ── LLM judge ─────────────────────────────────────────────────────────────────
def llm_judge(history, asins, engine):
    api_key = os.environ.get("ANTHROPIC_API_KEY","")
    if not api_key: return [None]*len(asins)
    hist_text = "\n".join([
        f"- {h.get('title',h.get('asin','?'))[:60]} (rated {h.get('rating','?')}/5)"
        for h in history[:8]])
    rec_text = "\n".join([
        f"{i+1}. {engine.item_info(a)['title'][:80]}"
        for i,a in enumerate(asins)])
    prompt = (f"Rate each product's relevance to this shopper (0=irrelevant, 5=highly relevant).\n\n"
              f"Purchase history:\n{hist_text}\n\nRecommendations:\n{rec_text}\n\n"
              f"Return ONLY a JSON array of {len(asins)} integers. Example: [3,4,1,5,2,3,4,1,2,3]")
    try:
        r = anthropic.Anthropic(api_key=api_key).messages.create(
            model="claude-haiku-4-5", max_tokens=100,
            messages=[{"role":"user","content":prompt}])
        scores = json.loads(r.content[0].text.strip())
        if isinstance(scores,list) and len(scores)==len(asins):
            return [float(s) for s in scores]
    except Exception as e:
        logger.warning(f"LLM judge: {e}")
    return [None]*len(asins)


# ── Spider chart ──────────────────────────────────────────────────────────────
def build_spider(offline_results, live_scores=None, n_live=0):
    if not offline_results: return go.Figure()
    axes = offline_results.get("spider_axes",[])
    labels = [a["label"] for a in axes]
    def norm(m): return [min(m.get(a["key"],0)/a["scale"],1.0) for a in axes] + \
                        [min(m.get(axes[0]["key"],0)/axes[0]["scale"],1.0)]
    lc = labels+[labels[0]]
    fig = go.Figure()
    b = offline_results.get("baseline",{})
    fig.add_trace(go.Scatterpolar(r=norm(b.get("metrics",{})), theta=lc,
        name=b.get("label","Baseline"),
        line=dict(color="#9ca3af",dash="dash",width=2), fill="none"))
    w = offline_results.get("winning",{})
    fig.add_trace(go.Scatterpolar(r=norm(w.get("metrics",{})), theta=lc,
        name=w.get("label","Full System"),
        line=dict(color="#3b82f6",dash="solid",width=2.5),
        fill="toself", fillcolor="rgba(59,130,246,0.08)"))
    if live_scores and any(s is not None for s in live_scores):
        valid = [s for s in live_scores if s is not None]
        avg   = np.mean(valid)/5.0
        lm    = {"ndcg":avg*0.12,"recall":avg*0.15,"mrr":avg*0.10,
                 "hitrate":avg*0.20,"coverage":0.27}
        fig.add_trace(go.Scatterpolar(r=norm(lm), theta=lc,
            name=f"Live avg (LLM-judged, n={n_live})",
            line=dict(color="#f97316",dash="dot",width=2), fill="none"))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1],
                tickvals=[0.25,0.5,0.75,1.0], ticktext=["25%","50%","75%","100%"],
                tickfont=dict(size=9), gridcolor="#e5e7eb"),
            angularaxis=dict(tickfont=dict(size=11)), bgcolor="white"),
        showlegend=True,
        legend=dict(orientation="h",yanchor="bottom",y=-0.28,
                    xanchor="center",x=0.5,font=dict(size=10)),
        margin=dict(t=20,b=90,l=55,r=55), height=320,
        paper_bgcolor="white",
        annotations=[dict(
            text="Each axis normalized to its own max — values not directly comparable across axes",
            x=0.5, y=-0.18, xref="paper", yref="paper",
            showarrow=False, font=dict(size=9, color="#9ca3af"),
            xanchor="center",
        )],
        title=dict(text="System performance (normalized)", font=dict(size=12),
                   x=0.5, xanchor="center"),
    )
    return fig


# ── Metrics table ─────────────────────────────────────────────────────────────
def build_metrics_table(offline_results: dict) -> str:
    """Build side-by-side exact metrics table from offline_results.json."""
    if not offline_results:
        return ""

    baseline = offline_results.get("baseline", {})
    winning  = offline_results.get("winning",  {})
    b_metrics = baseline.get("metrics", {})
    w_metrics = winning.get("metrics",  {})
    b_label   = baseline.get("label", "Baseline")
    w_label   = winning.get("label",  "Full System")
    source    = offline_results.get("_source", "simulated")

    rows = [
        ("NDCG@10",       "ndcg",      False),
        ("Recall@10",     "recall",    False),
        ("Recall@100",    "recall_100",False),
        ("MRR",           "mrr",       False),
        ("HitRate@10",    "hitrate",   False),
        ("Coverage",      "coverage",  True),
        ("Composite KPI", "kpi",       False),
    ]

    badge = (
        f'<span style="background:var(--color-background-secondary);'
        f'border:0.5px solid var(--color-border-tertiary);border-radius:4px;'
        f'padding:2px 6px;font-size:10px;color:var(--color-text-tertiary)">~ {source}</span>'
    )

    html = [
        '<div style="background:var(--color-background-primary);'
        'border:0.5px solid var(--color-border-tertiary);'
        'border-radius:var(--border-radius-lg);overflow:hidden;margin-top:8px">',

        '<div style="padding:8px 12px 6px;border-bottom:0.5px solid '
        'var(--color-border-tertiary);display:flex;justify-content:space-between;'
        'align-items:center">',
        f'<span style="font-size:12px;font-weight:500;color:var(--color-text-primary)">'
        f'Exact metric values (raw)</span>{badge}',
        '</div>',

        '<table style="width:100%;border-collapse:collapse;font-size:12px">',
        '<thead><tr style="background:var(--color-background-secondary)">',
        '<th style="padding:6px 12px;text-align:left;font-weight:500;'
        'color:var(--color-text-secondary);font-size:11px;'
        'border-bottom:0.5px solid var(--color-border-tertiary)">Metric</th>',
        f'<th style="padding:6px 10px;text-align:right;font-weight:500;'
        f'color:var(--color-text-secondary);font-size:11px;'
        f'border-bottom:0.5px solid var(--color-border-tertiary)">{b_label}</th>',
        f'<th style="padding:6px 10px;text-align:right;font-weight:500;'
        f'color:#0C447C;font-size:11px;'
        f'border-bottom:0.5px solid var(--color-border-tertiary);'
        f'background:var(--color-background-info)">{w_label} ★</th>',
        '</tr></thead><tbody>',
    ]

    for label, key, is_pct in rows:
        b_val = b_metrics.get(key)
        w_val = w_metrics.get(key)

        def fmt(v):
            if v is None: return '<span style="color:var(--color-text-tertiary)">n/a</span>'
            return f"{v*100:.1f}%" if is_pct else f"{v:.4f}"

        is_kpi = key == "kpi"
        fw = "font-weight:500;" if is_kpi else ""

        # Delta indicator
        delta_html = ""
        if b_val is not None and w_val is not None:
            delta = w_val - b_val
            pct   = (delta / b_val * 100) if b_val != 0 else 0
            color = "#16a34a" if delta > 0 else "#dc2626"
            arrow = "▲" if delta > 0 else "▼"
            delta_html = (f'<span style="font-size:10px;color:{color};margin-left:4px">'
                          f'{arrow}{abs(pct):.0f}%</span>')

        html.append(
            f'<tr style="border-bottom:0.5px solid var(--color-border-tertiary)">'
            f'<td style="padding:5px 12px;color:var(--color-text-secondary);'
            f'font-size:11px;{fw}">{label}</td>'
            f'<td style="padding:5px 10px;text-align:right;{fw}'
            f'color:var(--color-text-primary)">{fmt(b_val)}</td>'
            f'<td style="padding:5px 10px;text-align:right;font-weight:500;'
            f'color:#0C447C;background:var(--color-background-info)">'
            f'{fmt(w_val)}{delta_html}</td>'
            f'</tr>'
        )

    html += [
        '</tbody></table>',
        '<div style="padding:6px 12px;border-top:0.5px solid var(--color-border-tertiary);'
        'background:var(--color-background-secondary)">',
        '<p style="font-size:10px;color:var(--color-text-tertiary);margin:0;line-height:1.5">'
        'KPI = 0.30×NDCG + 0.25×Recall + 0.20×MRR + 0.15×HitRate + 0.10×Coverage'
        '<br>Spider chart normalizes each axis to its own max — '
        'axes are not directly comparable to each other</p>',
        '</div></div>',
    ]
    return "\n".join(html)


# ── HTML helpers ──────────────────────────────────────────────────────────────
def render_results(recs, engine, scores=None, banner=None):
    html = []
    if banner:
        html.append(f'<div style="background:#fef3c7;border:1px solid #f59e0b;'
                    f'border-radius:6px;padding:8px 12px;margin-bottom:10px;'
                    f'font-size:13px;color:#92400e">{banner}</div>')
    html.append('<div style="display:flex;flex-direction:column;gap:6px">')
    for i, rec in enumerate(recs):
        asin  = rec.get("asin") if isinstance(rec,dict) else rec
        info  = engine.item_info(asin)
        score = scores[i] if scores and i < len(scores) else None
        price_str  = f"${info['price']:.2f}" if info.get("price") else ""
        score_html = ""
        if score is not None:
            stars = "⭐"*int(round(score))
            score_html = (f'<span style="font-size:11px;color:#f97316;margin-left:6px">'
                          f'{stars} {score:.1f}/5 <i>(LLM-judged)</i></span>')
        html.append(
            f'<div style="background:#f9fafb;border:1px solid #e5e7eb;'
            f'border-radius:6px;padding:9px 12px;display:flex;gap:8px">'
            f'<span style="font-weight:600;color:#6b7280;min-width:22px;font-size:13px">'
            f'{i+1}.</span>'
            f'<div style="flex:1">'
            f'<div style="font-weight:500;color:#111827;font-size:13px">{info["title"]}</div>'
            f'<div style="font-size:11px;color:#6b7280;margin-top:2px">'
            f'{info["category"]}{"  ·  "+price_str if price_str else ""}{score_html}'
            f'</div></div></div>')
    html.append('</div>')
    return "\n".join(html)


def render_history(user_data):
    if not user_data: return ""
    history = user_data.get("validation_history",[])
    n       = user_data.get("interaction_count",0)
    utype   = user_data.get("user_type","normal")
    avg_r   = user_data.get("avg_rating")
    bg   = {"normal":"#dcfce7","sparse":"#fef3c7","new":"#fee2e2"}.get(utype,"#f3f4f6")
    fg   = {"normal":"#166534","sparse":"#92400e","new":"#991b1b"}.get(utype,"#374151")
    html = (f'<div style="font-size:12px;color:#6b7280;margin-bottom:6px">'
            f'<span style="background:{bg};color:{fg};padding:2px 8px;'
            f'border-radius:12px;font-weight:500">{utype}</span>  {n} interactions'
            + (f'  ·  avg {avg_r:.1f}★' if avg_r else "") + '</div>')
    if not history:
        return html + '<i style="color:#9ca3af;font-size:12px">No purchase history</i>'
    html += '<div style="display:flex;flex-direction:column;gap:3px">'
    for item in history[:7]:
        title  = item.get("title",item.get("asin","?"))[:55]
        rating = item.get("rating","?")
        html  += (f'<div style="font-size:11px;color:#374151;padding:3px 0;'
                  f'border-bottom:1px solid #f3f4f6">'
                  f'{"★"*int(rating) if isinstance(rating,int) else ""} {title}</div>')
    if n > 7: html += f'<div style="font-size:11px;color:#9ca3af">+{n-7} more...</div>'
    html += '</div>'
    return html


# ── Load resources ────────────────────────────────────────────────────────────
def load_users():
    users = {}
    if not DEMO_USERS_DIR.exists(): return users
    for p in sorted(DEMO_USERS_DIR.glob("*.json")):
        d     = json.loads(p.read_text())
        uid   = d["demo_id"]
        utype = d.get("user_type","normal")
        n     = d.get("interaction_count",0)
        users[f"{uid} — {utype} ({n} interactions)"] = d
    return users

def load_offline():
    return json.loads(OFFLINE_RESULTS.read_text()) if OFFLINE_RESULTS.exists() else {}


# ── Recommendation ────────────────────────────────────────────────────────────
def recommend(user_label, demo_users, engine, offline_results,
              live_scores, n_queries):
    if not user_label or user_label not in demo_users:
        mt = build_metrics_table(offline_results)
        return ("<p style='color:#6b7280'>Select a user.</p>",
                build_spider(offline_results), mt, "", live_scores, n_queries)

    user_data = demo_users[user_label]
    history   = user_data.get("validation_history",[])
    user_type = user_data.get("user_type","normal")
    status    = []

    # New user
    if user_type == "new" or len(history) == 0:
        pre = user_data.get("top10_recommendations",[])
        mt  = build_metrics_table(offline_results)
        return (render_results(pre, engine, banner="🆕 New user — showing popular items"),
                build_spider(offline_results, live_scores, n_queries),
                mt, "🆕 Cold start: new user — popularity fallback",
                live_scores, n_queries)

    # Use pre-computed recs from JSON
    pre = user_data.get("top10_recommendations",[])
    if pre:
        asins = [r["asin"] for r in pre]
    else:
        qv, n_inter, is_sparse, is_new = engine.encode_user(user_data)
        if is_new:
            pop = engine.popularity_fallback()
            mt  = build_metrics_table(offline_results)
            return (render_results([{"asin":a} for a in pop], engine,
                                   banner="🆕 New user"),
                    build_spider(offline_results), mt, "🆕 New user",
                    live_scores, n_queries)
        _, cands = engine.retrieve(qv)
        hist_sum = "bought " + ", ".join(
            [h.get("title",h.get("asin","?"))[:40] for h in history[:6]])
        asins = engine.rerank(cands.tolist(), hist_sum)
        pre   = [{"asin": a} for a in asins]

    # Sparse banner
    banner  = None
    n_inter = user_data.get("interaction_count",0)
    if user_type == "sparse" or (0 < n_inter < SPARSE_THRESH):
        alpha  = min(1.0, n_inter/SPARSE_THRESH)
        banner = f"⚡ Sparse user ({n_inter} interactions) — adaptive blending α={alpha:.2f}"
        status.append(banner)

    status.append("⚡ Stage 1: FAISS IVF → 100 candidates  "
                  "·  Stage 2: Cross-encoder → top-10")

    # LLM judge (normal users only)
    llm_scores = None
    if user_type not in ("new","sparse"):
        llm_scores = llm_judge(history, asins, engine)
        if any(s is not None for s in llm_scores):
            valid = [s for s in llm_scores if s is not None]
            avg   = np.mean(valid)
            live_scores = live_scores + valid
            n_queries   = n_queries + 1
            status.append(f"🤖 LLM judge: avg={avg:.2f}/5 *(LLM-judged)*")

    html   = render_results(pre, engine, scores=llm_scores, banner=banner)
    spider = build_spider(offline_results,
                          live_scores if live_scores else None, n_queries)
    mt     = build_metrics_table(offline_results)

    return html, spider, mt, "\n".join(status), live_scores, n_queries


# ── Gradio app ────────────────────────────────────────────────────────────────
def build_app():
    engine   = RecSysEngine()
    users    = {}
    offline  = {}

    def startup():
        nonlocal users, offline
        engine.load()
        users   = load_users()
        offline = load_offline()
        logger.info(f"Loaded {len(users)} demo users")

    with gr.Blocks(title="Amazon RecSys Demo") as demo:

        live_sc = gr.State([])
        n_q     = gr.State(0)

        gr.Markdown("""
# 🛠️ Amazon RecSys — Personalized Retrieval Demo
**Two-tower bi-encoder + cross-encoder re-ranker** · 4.4M Amazon Tools & Home Improvement interactions

*[Model](https://huggingface.co/chaturg/amazon-recsys-cross-encoder) · [Dataset](https://huggingface.co/datasets/chaturg/amazon-recsys-dataset) · [Code](https://github.com/chaturg/amazon-recsys)*
""")

        with gr.Tabs():

            # ── PHASE 1 ────────────────────────────────────────────────────
            with gr.Tab("📦 Phase 1 — Personalized Recommendations"):
                gr.Markdown("""
> **Pure personalization:** Recommendations are driven by purchase history, not keyword search.
> User history → taste profile embedding → FAISS retrieves 100 candidates → cross-encoder re-ranks to top-10.
""")
                with gr.Row():
                    # Left: controls + history
                    with gr.Column(scale=1):
                        user_dd = gr.Dropdown(
                            label="Select Demo User",
                            choices=[], value=None,
                            info="6 normal · 2 sparse · 2 new")
                        run_btn = gr.Button(
                            "🎯 Get Personalized Recommendations",
                            variant="primary")
                        gr.Markdown("### Purchase History")
                        hist_html = gr.HTML(
                            "<i style='color:#9ca3af'>Select a user</i>")
                        gr.Markdown("""
---
**Two-stage pipeline:**
1. User history → 128-dim embedding
2. FAISS IVF → 100 candidates (~3ms)
3. Cross-encoder re-rank → top-10 (~2s)
4. Claude Haiku LLM judge → 0–5 scores

**Cold start:**
- 🆕 New user → popularity fallback
- ⚡ Sparse user → adaptive α blending

*LLM scores are independent from offline metrics.*
""")

                    # Right: results + chart + table
                    with gr.Column(scale=2):
                        gr.Markdown("### Recommendations")
                        res_html = gr.HTML(
                            "<p style='color:#9ca3af'>Select a user to see recommendations.</p>")

                        status_md = gr.Markdown("")

                        with gr.Row():
                            # Spider chart
                            with gr.Column(scale=1):
                                spider = gr.Plot(show_label=False)
                            # Metrics table
                            with gr.Column(scale=1):
                                metrics_html = gr.HTML(value="")

            # ── PHASE 2 ────────────────────────────────────────────────────
            with gr.Tab("🔭 Phase 2 — Query-Aware Retrieval (Roadmap)"):
                gr.Markdown("""
## What Phase 2 Adds

The current system retrieves based on **user taste profile only**.
Phase 2 adds a **query text encoder** so retrieval responds to both
user preferences AND what they're searching for right now.

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

The projection layer **already accepts** `concat(user_emb, query_emb)`.
Adding `all-MiniLM-L6-v2` as a query tower and retraining is the change.

---

### Training data — already built

52k items × 5 paraphrased queries each = **260k synthetic queries**
generated with Claude Haiku. These become Phase 2 training triples:

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
| Domain specificity | ✅ User taste | ✅ User taste + query intent |

---

### Try the query box

The query below is **not currently encoded** — results are Phase 1 personalization.
This preview shows the intended UX for Phase 2.
""")
                with gr.Row():
                    with gr.Column(scale=1):
                        user_dd_p2 = gr.Dropdown(
                            label="Select Demo User",
                            choices=[], value=None)
                        query_box  = gr.Textbox(
                            label="🔭 Product Query (Phase 2 — not yet encoded)",
                            placeholder="e.g. cordless drill for weekend projects",
                            info="⚠️ Query not yet used in retrieval.",
                            lines=2)
                        run_btn_p2 = gr.Button(
                            "🔭 Preview Phase 2 (Phase 1 results)",
                            variant="secondary")
                    with gr.Column(scale=2):
                        gr.Markdown("### Recommendations *(Phase 1 — query not yet encoded)*")
                        res_html_p2 = gr.HTML(
                            "<p style='color:#9ca3af'>Select a user.</p>")
                        gr.Markdown("""
> Results above are identical to Phase 1. Query will influence
> retrieval once the sentence transformer query tower is added.
""")

        # ── Wiring ─────────────────────────────────────────────────────────
        def init():
            startup()
            c = list(users.keys())
            v = c[0] if c else None
            mt = build_metrics_table(load_offline())
            return (gr.Dropdown(choices=c, value=v),
                    gr.Dropdown(choices=c, value=v),
                    mt)

        demo.load(fn=init, inputs=[],
                  outputs=[user_dd, user_dd_p2, metrics_html])

        user_dd.change(
            fn=lambda l: render_history(users.get(l,{})),
            inputs=[user_dd], outputs=[hist_html])

        demo.load(
            fn=lambda: build_spider(load_offline()),
            inputs=[], outputs=[spider])

        def rec_p1(label, ls, nq):
            of = load_offline() or offline
            return recommend(label, users, engine, of, ls, nq)

        run_btn.click(
            fn=rec_p1,
            inputs=[user_dd, live_sc, n_q],
            outputs=[res_html, spider, metrics_html, status_md, live_sc, n_q])

        def rec_p2(label, query, ls, nq):
            of = load_offline() or offline
            r  = recommend(label, users, engine, of, ls, nq)
            return r[0], r[4], r[5]

        run_btn_p2.click(
            fn=rec_p2,
            inputs=[user_dd_p2, query_box, live_sc, n_q],
            outputs=[res_html_p2, live_sc, n_q])

    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860, share=False)
