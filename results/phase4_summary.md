# Phase 4: Classifier Comparison & Hyperparameter Tuning

## 1. Locked Phase 2 Configuration (Applied to All Experiments)

| Parameter | Value |
|-----------|-------|
| Preprocessing | Phase 1 best — Contractions + Preserved Negations + Negation Marking |
| Feature representation | Unigram + Bigram TF-IDF |
| Vocabulary size | 50,000 (unigrams) + 50,000 (bigrams) |
| minDF | 5.0 |
| N-gram range | 1 + 2 |
| Train / Test size | 80,190 / 19,810 |
| Random seed | 42 |

---

## 2. Phase 2 Best (Reference Baseline)

| Metric | Value |
|--------|-------|
| Classifier | Logistic Regression (maxIter=20) |
| Accuracy | **0.7896** |
| F1 Score | **0.7897** |
| ROC-AUC | **0.8734** |
| Training Time | 1.37s |

---

## 3. Experiment Results

| Experiment | Classifier | Hyperparameters | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time | Pred Time |
|:----------|:----------|:---------------|:-------:|:--------:|:-----:|:--:|:------:|:---------:|:--------:|
| **3-A — Phase 4 Baseline** | LR | maxIter=20 | 0.8671 | 0.8671 | 0.8671 | 0.8671 | 0.9362 | 17.53s | 1.27s |
| **3-B — Naive Bayes** | NaiveBayes | smoothing=1.0 | 0.8812 | 0.8815 | 0.8812 | 0.8812 | 0.9353 | 8.95s | 0.89s |
| **3-C — LR Tuned** | LR (TVS) | regParam=0.5, elasticNetParam=0.0, maxIter=200 | 0.9055 | 0.9055 | 0.9055 | 0.9054 | 0.9636 | 16.93s | 1.06s |
| **3-D — LinearSVC Tuned** | LinearSVC (TVS) | regParam=0.01, maxIter=200 | 0.8912 | 0.8912 | 0.8912 | 0.8912 | 0.9542 (margin) | 205.24s | 1.17s |



> **ROC-AUC for LinearSVC** is computed using the raw signed decision-function margin (`rawPrediction`), not calibrated probabilities. It is an approximate ranking-based AUC — flagged as `(margin)`.

---

## 4. Hyperparameter Tuning Details

### Logistic Regression (TrainValidationSplit)
- Grid: regParam ∈ {0.001, 0.01, 0.05, 0.1, 0.5} × elasticNetParam ∈ {0.0, 0.5} = **10 combinations**
- Metric: Accuracy on 20% validation split of training data
- Tuning time: 264.92s
- Best params: `regParam=0.5, elasticNetParam=0.0, maxIter=200`
- Final retrain: maxIter=200 on full training set

### LinearSVC (TrainValidationSplit)
- Grid: regParam ∈ {0.001, 0.01, 0.05, 0.1, 0.5} = **5 combinations**
- Metric: Accuracy on 20% validation split of training data
- Tuning time: 583.44s
- Best params: `regParam=0.01, maxIter=200`
- Final retrain: maxIter=200 on full training set

### No-leakage Contract
Training data → 80/20 TVS validation split → select best params → retrain on 100% training set → evaluate ONCE on untouched test set.

---

## 5. Class Balance Analysis

### Class Balance Check
- Positive: 41,059 (51.2%)
- Negative: 39,131 (48.8%)
- Imbalance: 2.4% — **≤ 5%, classes are balanced. Weighted experiments skipped.**

---

## 6. Best Classifier

| Field | Value |
|-------|-------|
| **Best Experiment** | 3-C LR Tuned |
| **Best Params** | `regParam=0.5, elasticNetParam=0.0, maxIter=200` |
| **Accuracy** | **0.9055** (90.55%) |
| **Precision** | 0.9055 |
| **Recall** | 0.9055 |
| **F1 Score** | **0.9054** |
| **ROC-AUC** | 0.9636 |
| **Training Time** | 16.93s |
| **Prediction Time** | 1.06s |
| **Confusion Matrix** | TP=9,368  TN=8,569  FP=1,033  FN=840 |

---

## 7. Phase 2 vs Phase 4 Comparison

| Metric | Phase 2 Best | Phase 4 Best | Δ |
|--------|:----------:|:----------:|:--:|
| **Accuracy** | 0.7896 | 0.9055 | **+0.1159** |
| **F1 Score** | 0.7897 | 0.9054 | **+0.1157** |
| **ROC-AUC** | 0.8734 | 0.9636 | +0.0902 |
| **Training Time** | 1.37s | 16.93s | +15.56s |
| **Prediction Time** | 0.15s | 1.06s | +0.91s |

---

## 8. Error Analysis — Best Phase 4 Classifier

### Summary
| Metric | Value |
|--------|-------|
| Total test samples | 19,810 |
| Correct predictions | 17,937 |
| Incorrect predictions | 1,873 |
| Overall error rate | 9.45% |
| Reviews with negation | 62 (0.3% of test) |
| Negation errors | 8 (0.43% of all errors) |
| Negation error rate | 12.90% |

### Error Rate by Review Length
| Length Bucket | Total | Errors | Error Rate |
|:-------------|:-----:|:------:|:----------:|
| Long (>50) | 13,613 | 1,352 | 9.9% |
| Medium (15-50) | 6,193 | 521 | 8.4% |
| Short (<15) | 4 | 0 | 0.0% |

### Sample Misclassified Examples
| Type | Negation | Length | Prediction | Review (truncated) |
|:-----|:--------:|:------:|:----------:|:------------------|
| FP | ✗ | 58 | Negative → Positive | Rita Hayworth was dubbed in almost all of her films by Jo Ann Greer and Anita El... |
| FP | ✗ | 146 | Negative → Positive | Perhaps I will be accused of owning tickets to "Short Attention-Span Theatre," b... |
| FP | ✗ | 136 | Negative → Positive | I was excited to tackle this book, but my excitement faded quickly. The prose ar... |
| FP | ✗ | 190 | Negative → Positive | I picked up this book for English class and after reading it I can respect how m... |
| FP | ✗ | 81 | Negative → Positive | Ironside themes do not a great album make. Even worse are the puerile child sing... |

---

## 9. 90% Accuracy Target

The 90% accuracy target was **achieved**.
Best measured test accuracy = **90.55%** (honest, untouched test set, single evaluation).

---

## 10. Model Preservation

Best Phase 4 model saved to `saved_model_phase4/` and `pipeline/saved_model_phase4/`.

Existing production model (`pipeline/saved_model/`), ABSA, complaint mining, Flask API — completely untouched.
