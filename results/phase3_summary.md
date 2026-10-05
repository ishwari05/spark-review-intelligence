# Phase 3: Classifier Comparison & Hyperparameter Tuning

## 1. Locked Phase 2 Configuration (Applied to All Experiments)

| Parameter | Value |
|-----------|-------|
| Preprocessing | Phase 1 best — Contractions + Preserved Negations + Negation Marking |
| Feature representation | Unigram + Bigram TF-IDF |
| Vocabulary size | 50,000 (unigrams) + 50,000 (bigrams) |
| minDF | 5.0 |
| N-gram range | 1 + 2 |
| Train / Test size | 69,391 / 17,068 |
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
| **3-A — Phase 3 Baseline** | LR | maxIter=20 | 0.7896 | 0.7902 | 0.7896 | 0.7897 | 0.8734 | 5.8s | 0.37s |
| **3-B — Naive Bayes** | NaiveBayes | smoothing=1.0 | 0.7836 | 0.7844 | 0.7836 | 0.7831 | 0.8698 | 1.0s | 0.3s |
| **3-C — LR Tuned** | LR (TVS) | regParam=0.05, elasticNetParam=0.0, maxIter=200 | 0.7980 | 0.7983 | 0.7980 | 0.7980 | 0.8825 | 1.5s | 0.14s |
| **3-D — LinearSVC Tuned** | LinearSVC (TVS) | regParam=0.01, maxIter=200 | 0.7891 | 0.7908 | 0.7891 | 0.7883 | 0.8768 (margin) | 13.42s | 0.16s |



> **ROC-AUC for LinearSVC** is computed using the raw signed decision-function margin (`rawPrediction`), not calibrated probabilities. It is an approximate ranking-based AUC — flagged as `(margin)`.

---

## 4. Hyperparameter Tuning Details

### Logistic Regression (TrainValidationSplit)
- Grid: regParam ∈ {0.001, 0.01, 0.05, 0.1, 0.5} × elasticNetParam ∈ {0.0, 0.5} = **10 combinations**
- Metric: Accuracy on 20% validation split of training data
- Tuning time: 14.05s
- Best params: `regParam=0.05, elasticNetParam=0.0, maxIter=200`
- Final retrain: maxIter=200 on full training set

### LinearSVC (TrainValidationSplit)
- Grid: regParam ∈ {0.001, 0.01, 0.05, 0.1, 0.5} = **5 combinations**
- Metric: Accuracy on 20% validation split of training data
- Tuning time: 31.67s
- Best params: `regParam=0.01, maxIter=200`
- Final retrain: maxIter=200 on full training set

### No-leakage Contract
Training data → 80/20 TVS validation split → select best params → retrain on 100% training set → evaluate ONCE on untouched test set.

---

## 5. Class Balance Analysis

### Class Balance Check
- Positive: 35,602 (51.3%)
- Negative: 33,789 (48.7%)
- Imbalance: 2.6% — **≤ 5%, classes are balanced. Weighted experiments skipped.**

---

## 6. Best Classifier

| Field | Value |
|-------|-------|
| **Best Experiment** | 3-C LR Tuned |
| **Best Params** | `regParam=0.05, elasticNetParam=0.0, maxIter=200` |
| **Accuracy** | **0.7980** (79.80%) |
| **Precision** | 0.7983 |
| **Recall** | 0.7980 |
| **F1 Score** | **0.7980** |
| **ROC-AUC** | 0.8825 |
| **Training Time** | 1.50s |
| **Prediction Time** | 0.14s |
| **Confusion Matrix** | TP=6,997  TN=6,623  FP=1,620  FN=1,828 |

---

## 7. Phase 2 vs Phase 3 Comparison

| Metric | Phase 2 Best | Phase 3 Best | Δ |
|--------|:----------:|:----------:|:--:|
| **Accuracy** | 0.7896 | 0.7980 | **+0.0084** |
| **F1 Score** | 0.7897 | 0.7980 | **+0.0083** |
| **ROC-AUC** | 0.8734 | 0.8825 | +0.0091 |
| **Training Time** | 1.37s | 1.50s | +0.13s |
| **Prediction Time** | 0.15s | 0.14s | -0.01s |

---

## 8. Error Analysis — Best Phase 3 Classifier

### Summary
| Metric | Value |
|--------|-------|
| Total test samples | 17,068 |
| Correct predictions | 13,620 |
| Incorrect predictions | 3,448 |
| Overall error rate | 20.20% |
| Reviews with negation | 47 (0.3% of test) |
| Negation errors | 6 (0.17% of all errors) |
| Negation error rate | 12.77% |

### Error Rate by Review Length
| Length Bucket | Total | Errors | Error Rate |
|:-------------|:-----:|:------:|:----------:|
| Medium (15-50) | 55 | 6 | 10.9% |
| Short (<15) | 17,013 | 3,442 | 20.2% |

### Sample Misclassified Examples
| Type | Negation | Length | Prediction | Review (truncated) |
|:-----|:--------:|:------:|:----------:|:------------------|
| FP | ✗ | 4 | Negative → Positive | They keep bringing it!... |
| FN | ✗ | 3 | Positive → Negative | Girl powerIS DEFINED... |
| FN | ✗ | 3 | Positive → Negative | The Far Horizon... |
| FP | ✗ | 8 | Negative → Positive | 'Skulls' hits a nail, dead in the head.... |
| FP | ✗ | 7 | Negative → Positive | Starts Great, but not typically good Cussler ending... |

---

## 9. 90% Accuracy Target

The 90% accuracy target was **NOT achieved**.
Best measured test accuracy = **79.80%** (honest, untouched test set, single evaluation).

---

## 10. Model Preservation

Best Phase 3 model saved to `saved_model_phase3/` and `pipeline/saved_model_phase3/`.

Existing production model (`pipeline/saved_model/`), ABSA, complaint mining, Flask API — completely untouched.
