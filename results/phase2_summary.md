# Phase 2: TF-IDF Feature Representation — Empirical Results & Analysis

## 1. Phase 2 Baseline

| Metric | Value |
|--------|-------|
| Preprocessing | Phase 1 best: Contractions + Preserved Negations + Negation Marking |
| Classifier | Logistic Regression (maxIter=20) |
| Feature Config | Unigram+Bigram TF-IDF, vocab=20,000, minDF=2 |
| Accuracy | **0.7813** |
| F1 | **0.7814** |
| ROC-AUC | **0.8504** |
| Training Time | 5.33s |
| Train / Test size | 69,391 / 17,068 |

---

## 2. All TF-IDF Experiments

| Experiment | Feature Rep | Vocab Size | minDF | N-gram | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time | Pred Time |
|:----------|:-----------|:----------|:-----|:------|:-------:|:--------:|:-----:|:--:|:------:|:---------:|:--------:|
| **EXP 1 Baseline (20k, minDF=2)** | Unigram+Bigram TF-IDF | uni=20000 bi=20000 | 2.0 | 1+2 | 0.7813 | 0.7817 | 0.7813 | 0.7814 | 0.8504 | 5.3300s | 0.4400s |
| **EXP 2 50k vocab minDF=2** | Unigram+Bigram TF-IDF | uni=50000 bi=50000 | 2.0 | 1+2 | 0.7813 | 0.7817 | 0.7813 | 0.7814 | 0.8504 | 2.5300s | 0.2100s |
| **EXP 3 50k vocab minDF=5** | Unigram+Bigram TF-IDF | uni=50000 bi=50000 | 5.0 | 1+2 | 0.7896 | 0.7902 | 0.7896 | 0.7897 | 0.8734 | 1.3700s | 0.1500s |
| **EXP 4 100k vocab minDF=5** | Unigram+Bigram TF-IDF | uni=100000 bi=100000 | 5.0 | 1+2 | 0.7896 | 0.7902 | 0.7896 | 0.7897 | 0.8734 | 1.3200s | 0.1500s |
| **EXP 5 Trigram (50k/30k/20k)** | Unigram+Bigram+Trigram TF-IDF | uni=50000 bi=30000 tri=20000 | 5.0 | 1+2+3 | 0.7893 | 0.7900 | 0.7893 | 0.7894 | 0.8736 | 1.6400s | 0.2000s |

---

## 3. Sublinear TF Feasibility

Spark MLlib 3.5 `IDF` does **not** expose a `sublinear_tf` parameter.  
`CountVectorizer` outputs raw integer term counts; there is no built-in log-scaled TF transformer in the Spark ML API.

Implementing sublinear TF would require either:
- A Python UDF operating over `SparseVector` objects (O(nnz) Python serialisation per row → destroys parallelism)
- A Scala Spark extension not available in the current environment

**Decision: Sublinear TF was skipped.** It would compromise the distributed scalability of the pipeline without a guaranteed accuracy benefit, contrary to Phase 2's constraints.

---

## 4. Trigram Feasibility

✅ Trigram features were computationally feasible. (Acc=0.7893, F1=0.7894)

---

## 5. Best Feature Configuration

| Field | Value |
|-------|-------|
| **Experiment** | EXP 3 — 50k vocab, minDF=5 |
| **Feature Representation** | Unigram+Bigram TF-IDF |
| **Vocabulary Size** | uni=50000 bi=50000 |
| **minDF** | 5 |
| **N-gram Configuration** | 1+2 |
| **Accuracy** | **0.7896** |
| **F1 Score** | **0.7897** |
| **ROC-AUC** | **0.8734** |
| **Training Time** | 1.37s |

---

## 6. Phase 1 vs Phase 2 Comparison

| Metric | Phase 1 Best | Phase 2 Best | Δ |
|--------|:----------:|:----------:|:--:|
| **Accuracy** | 0.7813 | 0.7896 | **+0.0083** |
| **F1 Score** | 0.7814 | 0.7897 | **+0.0083** |
| **ROC-AUC** | 0.8504 | 0.8734 | +0.0230 |
| **Training Time** | 3.87s | 1.37s | -2.50s |

---

## 7. Why the Best Configuration Performed Best

A larger vocabulary (uni=50000 bi=50000) with minDF=5 retains frequent sentiment-relevant terms while filtering out very rare noisy tokens. More unigram and bigram features give Logistic Regression more signal without adding noise.

---

## 8. Runtime & Memory Notes

- All experiments ran on 16 GB RAM (MacBook Air, local[*] Spark).
- Larger vocabulary sizes increase the dimensionality of feature vectors, which causes a slight increase in training time.
- Memory remained within 8 GB Spark driver limit for all feasible experiments.
- Trigram feature vectors, when summed with unigram + bigram, can become very large-dimensional — note result above.

---

## 9. Model Preservation

Best Phase 2 model saved to `saved_model_phase2/` and `pipeline/saved_model_phase2/`.

The existing production model in `pipeline/saved_model/` and all `results/results.json` data are completely untouched.
