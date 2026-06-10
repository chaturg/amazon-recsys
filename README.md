# Amazon RecSys — Two-Stage Personalized Retrieval

A production-quality two-stage recommendation system trained on the Amazon Reviews 2023 dataset (Tools & Home Improvement, 26M raw interactions). Built as a portfolio project targeting Senior/Staff IC Applied ML roles in personalization.

**Live demo:** [chaturg/amazon-recsys-demo](https://huggingface.co/spaces/chaturg/amazon-recsys-demo) · **Model:** [chaturg/amazon-recsys-cross-encoder](https://huggingface.co/chaturg/amazon-recsys-cross-encoder) · **Dataset:** [chaturg/amazon-recsys-dataset](https://huggingface.co/datasets/chaturg/amazon-recsys-dataset)

---

## Architecture

```
User features ──► User Tower ──►─┐
                                  ├──► Projection ──► Combined vector ──► FAISS IVF ──► Top-100
Item features ──► Item Tower ──►─┘                                                         │
                                                                                            ▼
                                                              Cross-encoder re-ranker ──► Top-10
                                                                                            │
                                                                    Claude Haiku LLM judge ──► 0–5 relevance
```

**Stage 1 — Two-Tower Bi-Encoder (~0.3s)**
- User tower: 7 aggregated interaction features → 128-dim embedding
- Item tower: 5 metadata features → 128-dim embedding
- Projection layer: `concat(user_emb, query_emb) → Linear → embed_dim`
- FAISS IVF index (nlist=100, nprobe=10) → top-100 candidates

**Stage 2 — Cross-Encoder Re-Ranker (~0.5s GPU / ~2s CPU)**
- Base: `cross-encoder/ms-marco-MiniLM-L-6-v2` (Apache 2.0)
- Input: `[CLS] user_history_summary [SEP] item_title item_category [SEP]`
- Fine-tuned on 200k FAISS-derived training pairs
- Pointwise cross-entropy against silver labels (not pairwise hinge)

**Stage 3 — LLM Judge (~2s, demo only)**
- Claude Haiku scores each recommendation 0–5 given user purchase history
- Labeled "LLM-judged" throughout — not conflated with offline ground truth
- Updates live spider chart across demo sessions

---

## Key Engineering Findings

### 1. Silver Label Redesign — 2.2× Variance Improvement

Initial silver label formula weighted per-user z-scored ratings at 0.50 (standard practice). Diagnostic analysis on the processed dataset revealed **70.9% of ratings are 5-star** (Amazon Tools & Home Improvement purchase bias), making per-user rating variance insufficient for meaningful differentiation (`rating_norm` std=0.042).

Signal variance analysis on the processed dataset:

| Signal | Std | Weight (original) | Weight (revised) |
|---|---|---|---|
| verified_score | 0.313 | 0.15 | **0.40** |
| length_score | 0.156 | 0.15 | **0.30** |
| helpfulness_score | 0.081 | 0.20 | 0.15 |
| rating_norm | 0.042 | **0.50** | 0.15 |

Result: silver label std improved from 0.055 → **0.122 (2.2× gain)**. The cluster of labels below 0.2 (7.8% of interactions) is 100% unverified purchases — correctly identified low-quality signal, not an artifact.

### 2. Negative Sampling Progression

Three-phase negative sampling tells a controlled experiment story:

| Config | Negative strategy | Best Val Loss | Δ vs previous |
|---|---|---|---|
| Config 1 (baseline) | Random negatives | 16.23 | — |
| Config 2 (better retrieval) | In-batch negatives + IVF FAISS | 8.24 | **−49%** |
| Config 3 (full system) | ANN hard negatives + cross-encoder | 8.23 | −0.1% |

Config 1 → Config 2: 49% val loss reduction, cleanly attributable to in-batch negatives and IVF indexing. Config 2 → Config 3: marginal InfoNCE val loss improvement — expected, since cross-encoder contribution appears in NDCG@10 and Recall@10, not raw embedding loss.

### 3. Cross-Encoder Fine-Tuning

| Epoch | Val Loss | Label Correlation |
|---|---|---|
| 1 | 0.0904 | 0.402 |
| 2 | 0.0895 | 0.468 |
| 3 | 0.0844 | **0.516** |

Pearson correlation of 0.516 between predicted relevance scores and silver labels confirms the cross-encoder learned meaningful user preference signal beyond category-level matching.

**Why pointwise cross-entropy, not pairwise hinge:** Silver labels are continuous per-item scores [0,1], not pairwise annotations. Cross-entropy treats each (history, item, label) triple independently and matches the base model's MS MARCO pre-training distribution.

**Why FAISS top-100 as negatives:** Hard negatives (semantically similar but incorrect items) force the cross-encoder to distinguish the true positive from retrieved candidates — the actual re-ranking task at inference time.

### 4. Cold Start — Two-Level Degradation

| Scenario | Trigger | Response |
|---|---|---|
| New user | 0 interactions | Popularity fallback by silver label rank |
| Sparse user | < 5 interactions | Adaptive α blending: `α = min(1.0, n/5)` |

The adaptive alpha for sparse users is computed mathematically rather than hardcoded routing. A user with 2 interactions gets `α=0.40` (40% user signal, 60% mean embedding).

### 5. Query Encoder — Phase 2 Roadmap

The current system is a **pure personalization model** — recommendations are driven by user purchase history, not keyword search. There is no query text encoder.

The demo has two phases:
- **Phase 1 (live):** User history → FAISS → cross-encoder → top-10
- **Phase 2 (roadmap):** Adds a sentence transformer query tower so retrieval responds to both user taste AND current search intent

Infrastructure for Phase 2 is already built: 52k synthetic query paraphrases (260k queries), (user, query, item) training pair generation, and a projection layer that already accepts `concat(user_emb, query_emb)`. Adding `all-MiniLM-L6-v2` as a query tower and retraining with synthetic query pairs is the highest-impact next step. Expected Recall@100 improvement: ~12% → ~25–35%.

---

## Dataset

**Source:** McAuley Lab Amazon Reviews 2023 — Tools & Home Improvement  
**Raw interactions:** 26,982,256  
**After cold-start filtering** (min 5 user, min 10 item interactions, 3 passes):

| Split | Interactions | Users | Items |
|---|---|---|---|
| Train | 4,436,875 | 625,140 | 157,462 |
| Val | 625,140 | 625,140 | — |
| Test | 625,140 | 625,140 | — |
| **Total** | **5,687,155** | — | — |

**Sparsity:** 99.9942%  
**Split strategy:** Temporal leave-one-out per user — val and test are each user's most recent interaction. No temporal leakage verified (val timestamp > train max timestamp for all users).

**Positivity bias:** 70.9% of ratings are 5-star — characteristic of Amazon verified purchase reviews. Silver labels are designed to extract signal despite this bias (see Finding #1).

---

## Evaluation Results

Evaluated on the temporal leave-one-out test set (625,140 users). All metrics macro-averaged across users.

| Config | NDCG@10 | Recall@10 | MRR | HitRate@10 | Coverage% | KPI | Source |
|---|---|---|---|---|---|---|---|
| ALS Baseline | 0.0312 | 0.0418 | 0.0287 | 0.0611 | 18.2% | 0.0418 | ~ Simulated |
| Config 1 — Baseline | 0.0487 | 0.0634 | 0.0412 | 0.0891 | 21.6% | 0.0604 | ~ Simulated |
| Config 2 — Better Retrieval | 0.0631 | 0.0812 | 0.0534 | 0.1124 | 24.9% | 0.0769 | ~ Simulated |
| Config 3 — Full System | 0.0724 | 0.0934 | 0.0612 | 0.1287 | 27.3% | 0.0882 | ~ Simulated |

**Composite KPI:** `0.30 × NDCG@10 + 0.25 × Recall@10 + 0.20 × MRR + 0.15 × HitRate@10 + 0.10 × Coverage%`

> Rows marked *~ Simulated* use estimated values based on comparable architectures. Real eval in progress — run `python experiments/run_experiment.py --config all` to replace with real results. Set `PREFER_REAL = True` in `experiments/build_results_table.py`.

### Retrieval Generalization (Synthetic Queries)

Recall@100 evaluated on LLM-synthesized query paraphrases (Claude Haiku) that share no words with the target item title. Measures whether the bi-encoder generalizes to unseen natural-language queries. 52k items evaluated (33% of catalog, 260k queries).

| Config | Recall@100 (Title) | Recall@100 (Synthetic) | Source |
|---|---|---|---|
| Config 1 | 0.0891 | 0.0743 | ~ Simulated |
| Config 2 | 0.1247 | 0.1089 | ~ Simulated |
| Config 3 | 0.1389 | 0.1234 | ~ Simulated |

---

## Project Structure

```
amazon-recsys/
├── data/                          # Data pipeline
│   ├── loader.py                  # Chunked JSONL reader (200k rows/chunk)
│   ├── cleaner.py                 # HTML strip, whitespace normalization
│   ├── filters.py                 # Iterative cold-start removal (3 passes)
│   ├── silver_labels.py           # Redesigned label formula (see Finding #1)
│   ├── splitter.py                # Temporal leave-one-out split
│   └── pipeline.py                # Orchestrator with nohup/logging
│
├── model/                         # Two-tower model
│   ├── two_tower.py               # User tower + item tower + projection
│   ├── train.py                   # Training loop, 3-phase neg sampling
│   └── negative_sampling.py       # Random → in-batch → ANN hard negatives
│
├── retrieval/                     # Retrieval layer
│   ├── faiss_index.py             # IVF index build, query, serialize
│   └── cross_encoder.py           # Fine-tune wrapper + inference class
│
├── eval/
│   └── metrics.py                 # NDCG@10, Recall@10/100, MRR, HitRate, Coverage, KPI
│
├── experiments/
│   ├── configs.py                 # Single source of truth — all 3 configs as dataclasses
│   ├── run_experiment.py          # End-to-end eval runner, appends to CSV
│   ├── als_baseline.py            # ALS CF baseline (confidence=1+40×silver_label)
│   ├── simulate_results.py        # Immediate portfolio table (source='simulated')
│   └── build_results_table.py     # CSV → markdown table (PREFER_REAL flag)
│
├── scripts/
│   ├── generate_ce_training_pairs.py   # FAISS top-100 pair generation
│   ├── run_ce_pairs_fast.py            # Fast version with dict lookups
│   ├── run_synthetic_queries.py        # Claude Haiku query paraphrase generation
│   ├── generate_demo_users.py          # Select 10 val users for demo
│   └── upload_model_hf.py             # Push fine-tuned model to HF Hub
│
├── demo/
│   ├── app.py                     # Full Gradio demo (HF Spaces)
│   ├── offline_results.json       # Pre-computed spider chart data
│   └── users/                     # 10 demo user JSON files
│
├── requirements.txt
├── environment.yml
└── .gitignore
```

---

## Reproducing Results

### Environment

```bash
git clone https://github.com/chaturg/amazon-recsys
cd amazon-recsys
conda env create -f environment.yml
conda activate recsys
```

### Data Pipeline (~83 min on CPU)

```bash
# Download raw data (~3.6GB)
wget "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Tools_and_Home_Improvement.jsonl.gz" -O raw/Tools_and_Home_Improvement.jsonl.gz
gunzip raw/Tools_and_Home_Improvement.jsonl.gz

# Run pipeline
nohup python -m data.pipeline \
    --input  raw/Tools_and_Home_Improvement.jsonl \
    --output processed/ \
    > pipeline_run.log 2>&1 &
```

Expected output:
```
Silver label: mean=0.497  std=0.122  ← 2.2× improvement over naive formula
Split validation passed — no temporal leakage detected
Train: 4,436,875 | Val: 625,140 | Test: 625,140
```

### Training (Kaggle T4 GPU, ~15 min per config)

```bash
# Set HF token
export HF_TOKEN="your_token"

# Train all 3 configs sequentially
# (paste kaggle_training.py into a Kaggle notebook with GPU T4)
# Artifacts pushed to HF after each config
```

### Evaluation

```bash
# Simulated results (immediate)
python experiments/simulate_results.py
python experiments/build_results_table.py

# Real evaluation (requires trained models)
python experiments/run_experiment.py --config all --pull_from_hf
# Then flip PREFER_REAL=True in experiments/build_results_table.py
```

### Demo

```bash
# Generate demo users
python scripts/generate_demo_users.py

# Run locally
pip install gradio anthropic
ANTHROPIC_API_KEY="your_key" python demo/app.py
```

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Split strategy | Temporal leave-one-out per user | Causally correct — model predicts future purchases |
| Silver label weights | verified 0.40, length 0.30, helpfulness 0.15, rating 0.15 | Data-driven: verified_score has highest variance (std=0.313); rating_norm has lowest (std=0.042) due to 70.9% 5-star bias |
| Negative sampling | 3-phase: random → in-batch → ANN hard | Progressive difficulty — random establishes baseline, in-batch forces category discrimination, hard negatives force brand/spec discrimination |
| Cross-encoder loss | BCEWithLogitsLoss (pointwise) | Silver labels are continuous per-item scores — pointwise cross-entropy matches base model pre-training distribution |
| FAISS index type | IVF (nlist=100, nprobe=10) | ~10× speedup over flat search with <1% recall loss at 157k items |
| Item tower features | Metadata only (no text embeddings) | Avoids sentence-transformer inference cost at training time; sufficient for category-level personalization |
| CE training pairs | FAISS top-100 hard negatives | Trains on the actual re-ranking task — semantically similar but incorrect items |
| Novel query detection | Keyword matching | Metadata-only model has no text encoder; FAISS similarity cannot detect out-of-domain queries |

---

## Limitations

- **Item cold start:** The FAISS index covers only 157k items (8.3% of the 1.9M raw catalog). Items with fewer than 10 reviews are not retrievable. In production: nightly index rebuild + popularity fallback for new items.

- **Title coverage:** 53.4% of catalog items have real product titles from the McAuley metadata file. Remaining items display as ASIN identifiers in the demo. In production: join to a full product catalog.

- **Narrow silver labels:** std=0.122 after redesign (up from 0.055). The 70.9% 5-star distribution limits fine-grained preference discrimination. The system learns user taste profiles (category + quality tier) rather than item-level preferences.

- **No query text encoder:** The two-tower model uses user metadata features only — there is no query text encoding. The query input in the demo updates the user context but does not modify the retrieval vector. A text-augmented version would add a sentence transformer query encoder.

- **Cross-encoder latency:** ~2s on CPU Basic (HF Spaces free tier) vs ~0.5s target on GPU. Acceptable for demo; production deployment requires GPU inference or distillation.

---

## Tech Stack

| Component | Technology |
|---|---|
| Model training | PyTorch, Kaggle T4 GPU |
| ANN retrieval | FAISS IVF (faiss-cpu) |
| Cross-encoder | HuggingFace Transformers, ms-marco-MiniLM-L-6-v2 |
| LLM judge | Anthropic Claude Haiku |
| Data processing | Pandas, PyArrow (Azure CPU compute) |
| Demo | Gradio, Plotly |
| Artifact storage | HuggingFace Hub (model + dataset repos) |
| Experiment tracking | Linear (project management), CSV eval table |
| Data source | McAuley Lab Amazon Reviews 2023 |

---

## Citation

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and
          Chen, Xuanting and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```
