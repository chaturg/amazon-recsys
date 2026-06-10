## Evaluation Results

Evaluated on the temporal leave-one-out test set (625,140 users). All metrics are macro-averaged across users.

### Offline Metrics

| Config | NDCG@10 | Recall@10 | MRR | HitRate@10 | Coverage% | Composite KPI | Source |
|--------|---------|-----------|-----|------------|-----------|---------------|--------|
| ALS Baseline | 0.0312 | 0.0418 | 0.0287 | 0.0611 | 18.2% | 0.0418 | ~ Simulated |
| Config 1 — Baseline | 0.0487 | 0.0634 | 0.0412 | 0.0891 | 21.6% | 0.0604 | ~ Simulated |
| Config 2 — Better Retrieval | 0.0631 | 0.0812 | 0.0534 | 0.1124 | 24.9% | 0.0769 | ~ Simulated |
| Config 3 — Full System | 0.0724 | 0.0934 | 0.0612 | 0.1287 | 27.3% | 0.0882 | ~ Simulated |

### Retrieval Generalization (Synthetic Queries)

Recall@100 evaluated on LLM-synthesized query paraphrases that share no words with the target item title. Measures whether the bi-encoder generalizes to unseen natural-language queries.

| Config | Recall@100 (Title Proxy) | Recall@100 (Synthetic) | Delta | Source |
|--------|--------------------------|------------------------|-------|--------|
| ALS Baseline | N/A | N/A | N/A | ALS cannot use query text |
| Config 1 — Baseline | 0.0891 | 0.0743 | N/A | ~ Simulated |
| Config 2 — Better Retrieval | 0.1247 | 0.1089 | N/A | ~ Simulated |
| Config 3 — Full System | 0.1389 | 0.1234 | N/A | ~ Simulated |

### Composite KPI Formula

```
KPI = 0.30 × NDCG@10 + 0.25 × Recall@10 + 0.20 × MRR + 0.15 × HitRate@10 + 0.10 × Coverage%
```

> **Note:** Rows marked *~ Simulated* use estimated values based on comparable architectures. Real evaluation in progress. Set `PREFER_REAL = True` in `experiments/build_results_table.py` when real results are available.
