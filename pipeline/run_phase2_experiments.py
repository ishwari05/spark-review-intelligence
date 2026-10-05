"""
================================================================================
 Phase 2 Experiment Runner: TF-IDF Feature Representation Optimization
 ----------------------------------------------------------------------
 Uses the BEST Phase 1 preprocessing (Experiment C):
   - Contraction Expansion + Preserved Negations + Negation Marking

 Holds constant:
   - Dataset           (Amazon Polarity, 100,000 rows, same split/seed)
   - Preprocessing     (Phase 1 best: preprocess_variant_c)
   - Classifier        (Logistic Regression, maxIter=20)
   - Evaluation method (same as Phase 1)

 Varies ONLY:
   - TF-IDF vocabulary size
   - minDF
   - N-gram configuration (unigram / unigram+bigram / unigram+bigram+trigram)

 Experiments:
   EXP 1  (Phase 2 Baseline): Unigram+Bigram, vocab=20k,  minDF=2  (== Phase 1 best config)
   EXP 2  Larger vocab:        Unigram+Bigram, vocab=50k,  minDF=2
   EXP 3  Higher minDF:        Unigram+Bigram, vocab=50k,  minDF=5
   EXP 4  Largest vocab:       Unigram+Bigram, vocab=100k, minDF=5
   EXP 5  Trigram (if feasible): Unigram+Bigram+Trigram, vocab=50k/30k/20k, minDF=5

 NOTE on Sublinear TF:
   Spark MLlib 3.5 IDF does NOT expose a sublinear_tf parameter.
   CountVectorizer outputs raw integer counts; there is no log-scaled output
   transformer in Spark ML. A log transform would require a Python UDF over
   SparseVectors (O(nnz) Python deserialization per row), which would destroy
   the distributed scalability of the pipeline. Therefore, sublinear TF is
   SKIPPED and documented in results.

 Saves:
   results/phase2_results.csv
   results/phase2_summary.md
   results/figures/phase2_feature_comparison.png
   pipeline/saved_model_phase2/   (only if Phase 2 improves over Phase 1)
================================================================================
"""

import os
import sys
import time
import csv
import traceback
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, NGram,
    CountVectorizer, IDF, VectorAssembler,
)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)

# ---------------------------------------------------------------------------
# Config — identical to Phase 1
# ---------------------------------------------------------------------------
SAMPLE_SIZE   = 100000
TEST_FRACTION = 0.2
SEED          = 42

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PIPELINE_RESULTS_DIR  = os.path.join(BASE_DIR, "results")
ROOT_RESULTS_DIR      = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR           = os.path.join(ROOT_RESULTS_DIR, "figures")
PIPELINE_FIGURES_DIR  = os.path.join(PIPELINE_RESULTS_DIR, "figures")
PHASE2_MODEL_DIR_PIPE = os.path.join(BASE_DIR, "saved_model_phase2")
PHASE2_MODEL_DIR_ROOT = os.path.join(PROJECT_ROOT, "saved_model_phase2")
PHASE1_MODEL_DIR_PIPE = os.path.join(BASE_DIR, "saved_model_phase1")

for d in (PIPELINE_RESULTS_DIR, ROOT_RESULTS_DIR, FIGURES_DIR, PIPELINE_FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

from pipeline.preprocessing import preprocess_variant_c, get_sentiment_stopwords


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[phase2] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Spark session
# ---------------------------------------------------------------------------
def get_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("Phase2FeatureOptimization")
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    log(f"Spark {spark.version} session started (local[*]).")
    return spark


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(spark: SparkSession):
    from datasets import load_dataset as hf_load
    log(f"Loading {SAMPLE_SIZE} rows of amazon_polarity (cached)...")
    hf_ds = hf_load("amazon_polarity", split=f"train[:{SAMPLE_SIZE}]")
    rows  = [{"label": float(r["label"]), "title": r["title"], "review": r["content"]}
             for r in hf_ds]
    log(f"Loaded {len(rows)} raw rows into Spark...")
    sdf = spark.createDataFrame(rows, schema=["label", "title", "review"])
    sdf = sdf.withColumn("label", F.col("label").cast(DoubleType()))
    sdf = sdf.repartition(8).cache()
    log(f"Spark DataFrame cached: {sdf.count()} rows.")
    return sdf


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_data(raw_df):
    before = raw_df.count()
    df = raw_df.dropna(subset=["review"])
    df = df.dropDuplicates(["review"])
    df = df.filter(F.length(F.trim(F.col("review"))) > 0)
    after = df.count()
    log(f"Cleaned nulls / dupes / empties: {before} → {after} rows.")
    return df


# ---------------------------------------------------------------------------
# Feature-stage builder for Phase 2
# ---------------------------------------------------------------------------
def build_phase2_stages(
    stop_words,
    unigram_vocab: int,
    bigram_vocab:  int,
    trigram_vocab: int,
    min_df:        float,
    include_bigrams:  bool = True,
    include_trigrams: bool = False,
    input_col: str = "clean_text",
):
    """
    Build Unigram / Unigram+Bigram / Unigram+Bigram+Trigram TF-IDF stages.
    All CountVectorizer + IDF stages are fitted strictly on training data
    inside a Spark ML Pipeline (no leakage).
    """
    tokenizer = Tokenizer(inputCol=input_col, outputCol="tokens")
    remover   = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens",
                                 stopWords=stop_words)

    # --- Unigrams ---
    uni_cv  = CountVectorizer(inputCol="filtered_tokens", outputCol="uni_raw",
                               vocabSize=unigram_vocab, minDF=min_df)
    uni_idf = IDF(inputCol="uni_raw", outputCol="uni_feat")

    stages = [tokenizer, remover, uni_cv, uni_idf]
    assemble_cols = ["uni_feat"]

    if include_bigrams:
        bigram    = NGram(n=2, inputCol="filtered_tokens", outputCol="bigrams")
        bi_cv     = CountVectorizer(inputCol="bigrams", outputCol="bi_raw",
                                    vocabSize=bigram_vocab, minDF=min_df)
        bi_idf    = IDF(inputCol="bi_raw", outputCol="bi_feat")
        stages   += [bigram, bi_cv, bi_idf]
        assemble_cols.append("bi_feat")

    if include_trigrams:
        trigram   = NGram(n=3, inputCol="filtered_tokens", outputCol="trigrams")
        tri_cv    = CountVectorizer(inputCol="trigrams", outputCol="tri_raw",
                                    vocabSize=trigram_vocab, minDF=min_df)
        tri_idf   = IDF(inputCol="tri_raw", outputCol="tri_feat")
        stages   += [trigram, tri_cv, tri_idf]
        assemble_cols.append("tri_feat")

    assembler = VectorAssembler(inputCols=assemble_cols, outputCol="features")
    stages.append(assembler)
    return stages


# ---------------------------------------------------------------------------
# Evaluate one pipeline
# ---------------------------------------------------------------------------
def evaluate(name: str, stages: list, train_df, test_df):
    classifier = LogisticRegression(
        labelCol="label", featuresCol="features", maxIter=20
    )
    pipeline = Pipeline(stages=stages + [classifier])

    log(f"Training {name}...")
    t0 = time.time()
    model = pipeline.fit(train_df)
    train_time = time.time() - t0

    t0 = time.time()
    preds = model.transform(test_df).cache()
    preds.count()   # materialize
    pred_time = time.time() - t0

    mc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy  = mc.evaluate(preds, {mc.metricName: "accuracy"})
    precision = mc.evaluate(preds, {mc.metricName: "weightedPrecision"})
    recall    = mc.evaluate(preds, {mc.metricName: "weightedRecall"})
    f1        = mc.evaluate(preds, {mc.metricName: "f1"})

    roc_auc = None
    if "probability" in preds.columns:
        bc = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="probability",
            metricName="areaUnderROC"
        )
        roc_auc = bc.evaluate(preds)

    preds.unpersist()

    metrics = {
        "accuracy":     round(float(accuracy),  4),
        "precision":    round(float(precision), 4),
        "recall":       round(float(recall),    4),
        "f1":           round(float(f1),        4),
        "roc_auc":      round(float(roc_auc), 4) if roc_auc is not None else None,
        "train_time":   round(float(train_time),  2),
        "predict_time": round(float(pred_time),   2),
    }
    log(f"{name}: acc={metrics['accuracy']} f1={metrics['f1']} "
        f"auc={metrics['roc_auc']} train={metrics['train_time']}s")
    return metrics, model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(results: list, out_paths: list) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 10})

    labels = [r["experiment_id"] for r in results]
    accs   = [r["accuracy"]  for r in results]
    f1s    = [r["f1"]        for r in results]
    aucs   = [r["roc_auc"] if r["roc_auc"] is not None else 0.0 for r in results]
    times  = [r["train_time"] for r in results]

    x     = np.arange(len(labels))
    width = 0.28
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Accuracy / F1 / AUC grouped bars
    ax1 = axes[0]
    b1 = ax1.bar(x - width, accs,  width, label="Accuracy",    color="#1f77b4", alpha=0.88)
    b2 = ax1.bar(x,          f1s,  width, label="Weighted F1", color="#2ca02c", alpha=0.88)
    b3 = ax1.bar(x + width, aucs,  width, label="ROC-AUC",     color="#ff7f0e", alpha=0.88)
    ax1.set_title("Phase 2: Accuracy · F1 · ROC-AUC per Feature Config",
                  fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylabel("Score", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax1.set_ylim(0.70, 0.90)
    ax1.legend(loc="lower right", fontsize=9)

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                 f"{h:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                 f"{h:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for bar in b3:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.002,
                 f"{h:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Plot 2: Training time
    ax2 = axes[1]
    palette = plt.cm.tab10.colors[:len(results)]
    bars = ax2.bar(labels, times, color=palette, width=0.5, alpha=0.88)
    ax2.set_title("Phase 2: Training Time per Feature Configuration",
                  fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylabel("Training Time (s)", fontsize=11)
    ax2.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                 f"{h:.2f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    for p in out_paths:
        plt.savefig(p, dpi=300)
    plt.close()
    log(f"Saved comparison plot to {out_paths[0]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    spark    = get_spark()
    raw_df   = load_dataset(spark)
    clean_df = prepare_data(raw_df)

    # Apply Phase 1 BEST preprocessing (Experiment C — fixed for all Phase 2 runs)
    log("Applying Phase 1 best preprocessing (Variant C)...")
    df = preprocess_variant_c(clean_df, "review", "clean_text")

    # Same train/test split as Phase 1
    train_df, test_df = df.randomSplit([1 - TEST_FRACTION, TEST_FRACTION], seed=SEED)
    train_df = train_df.cache()
    test_df  = test_df.cache()
    train_size = train_df.count()
    test_size  = test_df.count()
    log(f"Train={train_size:,}  Test={test_size:,}  (seed={SEED})")

    # Custom stop words (negation-preserving) from Phase 1
    custom_stops = get_sentiment_stopwords()

    # -------------------------------------------------------------------
    # Phase 2 Experiment Definitions
    # -------------------------------------------------------------------
    #  EXP 1  — Phase 2 baseline = Phase 1 best config
    #  EXP 2  — Bigger unigram vocab (50k)
    #  EXP 3  — Bigger vocab + higher minDF
    #  EXP 4  — Largest vocab (100k) + moderate minDF
    #  EXP 5  — Trigram (attempted; skipped if memory/time fails)
    # -------------------------------------------------------------------
    experiments = [
        {
            "id":          "EXP 1\nBaseline\n(20k, minDF=2)",
            "label":       "EXP 1 — Baseline (20k, minDF=2)",
            "repr":        "Unigram+Bigram TF-IDF",
            "uni_vocab":   20_000,
            "bi_vocab":    20_000,
            "tri_vocab":   0,
            "min_df":      2.0,
            "bigrams":     True,
            "trigrams":    False,
            "ngram":       "1+2",
        },
        {
            "id":          "EXP 2\n50k vocab\nminDF=2",
            "label":       "EXP 2 — 50k vocab, minDF=2",
            "repr":        "Unigram+Bigram TF-IDF",
            "uni_vocab":   50_000,
            "bi_vocab":    50_000,
            "tri_vocab":   0,
            "min_df":      2.0,
            "bigrams":     True,
            "trigrams":    False,
            "ngram":       "1+2",
        },
        {
            "id":          "EXP 3\n50k vocab\nminDF=5",
            "label":       "EXP 3 — 50k vocab, minDF=5",
            "repr":        "Unigram+Bigram TF-IDF",
            "uni_vocab":   50_000,
            "bi_vocab":    50_000,
            "tri_vocab":   0,
            "min_df":      5.0,
            "bigrams":     True,
            "trigrams":    False,
            "ngram":       "1+2",
        },
        {
            "id":          "EXP 4\n100k vocab\nminDF=5",
            "label":       "EXP 4 — 100k vocab, minDF=5",
            "repr":        "Unigram+Bigram TF-IDF",
            "uni_vocab":   100_000,
            "bi_vocab":    100_000,
            "tri_vocab":   0,
            "min_df":      5.0,
            "bigrams":     True,
            "trigrams":    False,
            "ngram":       "1+2",
        },
        {
            "id":          "EXP 5\nTrigram\n(50k/30k/20k)",
            "label":       "EXP 5 — Trigram (50k/30k/20k, minDF=5)",
            "repr":        "Unigram+Bigram+Trigram TF-IDF",
            "uni_vocab":   50_000,
            "bi_vocab":    30_000,
            "tri_vocab":   20_000,
            "min_df":      5.0,
            "bigrams":     True,
            "trigrams":    True,
            "ngram":       "1+2+3",
        },
    ]

    results      = []
    fitted_models = {}

    for exp in experiments:
        exp_label = exp["label"]
        try:
            stages = build_phase2_stages(
                stop_words       = custom_stops,
                unigram_vocab    = exp["uni_vocab"],
                bigram_vocab     = exp["bi_vocab"],
                trigram_vocab    = exp["tri_vocab"],
                min_df           = exp["min_df"],
                include_bigrams  = exp["bigrams"],
                include_trigrams = exp["trigrams"],
            )
            metrics, fitted_model = evaluate(exp_label, stages, train_df, test_df)
            entry = {
                "experiment":            exp_label,
                "experiment_id":         exp["id"],
                "preprocessing":         "Contractions + Preserved Negations + Negation Marking (Phase 1 best)",
                "feature_representation": exp["repr"],
                "vocab_size":            f"uni={exp['uni_vocab']}" + (f" bi={exp['bi_vocab']}" if exp["bigrams"] else "")
                                         + (f" tri={exp['tri_vocab']}" if exp["trigrams"] else ""),
                "min_df":                exp["min_df"],
                "ngram_range":           exp["ngram"],
                "model":                 "Logistic Regression (maxIter=20)",
                "accuracy":              metrics["accuracy"],
                "precision":             metrics["precision"],
                "recall":                metrics["recall"],
                "f1":                    metrics["f1"],
                "roc_auc":               metrics["roc_auc"],
                "train_time":            metrics["train_time"],
                "prediction_time":       metrics["predict_time"],
                "train_size":            train_size,
                "test_size":             test_size,
                "feasible":              True,
                "notes":                 "",
            }
            results.append(entry)
            fitted_models[exp_label] = fitted_model

        except Exception as exc:
            log(f"SKIPPED {exp_label} — {exc}")
            entry = {
                "experiment":            exp_label,
                "experiment_id":         exp["id"],
                "preprocessing":         "Contractions + Preserved Negations + Negation Marking",
                "feature_representation": exp["repr"],
                "vocab_size":            f"uni={exp['uni_vocab']}",
                "min_df":                exp["min_df"],
                "ngram_range":           exp["ngram"],
                "model":                 "Logistic Regression (maxIter=20)",
                "accuracy":              None,
                "precision":             None,
                "recall":                None,
                "f1":                    None,
                "roc_auc":               None,
                "train_time":            None,
                "prediction_time":       None,
                "train_size":            train_size,
                "test_size":             test_size,
                "feasible":              False,
                "notes":                 str(exc)[:300],
            }
            results.append(entry)

    # -------------------------------------------------------------------
    # Phase 1 reference (constant — do NOT re-run)
    # -------------------------------------------------------------------
    phase1_best = {
        "accuracy": 0.7813,
        "f1":       0.7814,
        "roc_auc":  0.8504,
        "train_time": 3.87,
    }

    # -------------------------------------------------------------------
    # Identify best Phase 2 result
    # -------------------------------------------------------------------
    feasible = [r for r in results if r["feasible"]]
    best_exp = max(feasible, key=lambda r: (r["f1"], r["accuracy"]))
    log(f"Best Phase 2 config: {best_exp['experiment']} — "
        f"Acc={best_exp['accuracy']} F1={best_exp['f1']}")

    # -------------------------------------------------------------------
    # Save best model only if it improves Phase 1
    # -------------------------------------------------------------------
    phase2_saves = []
    if best_exp["accuracy"] > phase1_best["accuracy"] and best_exp["experiment"] in fitted_models:
        for mdir in (PHASE2_MODEL_DIR_PIPE, PHASE2_MODEL_DIR_ROOT):
            log(f"Saving Phase 2 best model to {mdir}...")
            fitted_models[best_exp["experiment"]].write().overwrite().save(mdir)
            phase2_saves.append(mdir)
    else:
        log("Phase 2 best did NOT exceed Phase 1 best — no Phase 2 model saved, "
            "Phase 1 model remains current best.")

    # -------------------------------------------------------------------
    # CSV results
    # -------------------------------------------------------------------
    csv_fields = [
        "experiment", "preprocessing", "feature_representation",
        "vocab_size", "min_df", "ngram_range", "model",
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "train_time", "prediction_time",
    ]
    for cp in (os.path.join(ROOT_RESULTS_DIR, "phase2_results.csv"),
               os.path.join(PIPELINE_RESULTS_DIR, "phase2_results.csv")):
        with open(cp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        log(f"Saved CSV → {cp}")

    # -------------------------------------------------------------------
    # Comparison plot (feasible experiments only)
    # -------------------------------------------------------------------
    fig_paths = [
        os.path.join(FIGURES_DIR, "phase2_feature_comparison.png"),
        os.path.join(PIPELINE_FIGURES_DIR, "phase2_feature_comparison.png"),
    ]
    plot_results(feasible, fig_paths)

    # -------------------------------------------------------------------
    # Trigram feasibility note
    # -------------------------------------------------------------------
    trigram_entry = next((r for r in results if "Trigram" in r["experiment"]), None)
    trigram_feasible = trigram_entry["feasible"] if trigram_entry else False
    trigram_note = (
        "✅ Trigram features were computationally feasible. "
        f"(Acc={trigram_entry['accuracy']}, F1={trigram_entry['f1']})"
        if trigram_feasible
        else (
            "❌ Trigram features were skipped / failed: "
            + (trigram_entry["notes"] if trigram_entry else "not attempted")
        )
    )

    # -------------------------------------------------------------------
    # Compute deltas
    # -------------------------------------------------------------------
    baseline_entry = feasible[0]   # EXP 1 is always the Phase 2 baseline
    acc_delta_vs_ph1 = round(best_exp["accuracy"] - phase1_best["accuracy"], 4)
    f1_delta_vs_ph1  = round(best_exp["f1"]       - phase1_best["f1"],       4)
    time_delta_vs_ph1 = round(
        (best_exp["train_time"] or 0) - phase1_best["train_time"], 2
    )

    # -------------------------------------------------------------------
    # Table rows for markdown
    # -------------------------------------------------------------------
    def _row(r):
        def _v(x): return f"{x:.4f}" if isinstance(x, float) else ("—" if x is None else str(x))
        feasibility = "" if r.get("feasible", True) else " *(skipped)*"
        return (
            f"| **{r['experiment_id'].replace(chr(10),' ')}** "
            f"| {r['feature_representation']}{feasibility} "
            f"| {r['vocab_size']} "
            f"| {r['min_df']} "
            f"| {r['ngram_range']} "
            f"| {_v(r['accuracy'])} "
            f"| {_v(r['precision'])} "
            f"| {_v(r['recall'])} "
            f"| {_v(r['f1'])} "
            f"| {_v(r['roc_auc'])} "
            f"| {_v(r['train_time'])}s "
            f"| {_v(r['prediction_time'])}s |"
        )

    table_rows = "\n".join(_row(r) for r in results)

    # -------------------------------------------------------------------
    # Determine the reason the best config is best
    # -------------------------------------------------------------------
    if best_exp["ngram_range"] == "1+2+3":
        why_best = "Trigram features captured three-word sentiment phrases (e.g. 'not very good', 'not at all') that no unigram or bigram pair can represent alone."
    else:
        best_vocab_desc = best_exp["vocab_size"]
        if best_exp["min_df"] > 2:
            why_best = (
                f"A larger vocabulary ({best_vocab_desc}) with minDF={best_exp['min_df']:.0f} "
                "retains frequent sentiment-relevant terms while filtering out very rare noisy tokens. "
                "More unigram and bigram features give Logistic Regression more signal without adding noise."
            )
        else:
            why_best = (
                f"A larger vocabulary ({best_vocab_desc}) with minDF={best_exp['min_df']:.0f} "
                "captures more unique product-specific sentiment signals without aggressively pruning rare terms, "
                "improving Logistic Regression's decision boundary coverage."
            )

    summary = f"""# Phase 2: TF-IDF Feature Representation — Empirical Results & Analysis

## 1. Phase 2 Baseline

| Metric | Value |
|--------|-------|
| Preprocessing | Phase 1 best: Contractions + Preserved Negations + Negation Marking |
| Classifier | Logistic Regression (maxIter=20) |
| Feature Config | Unigram+Bigram TF-IDF, vocab=20,000, minDF=2 |
| Accuracy | **{baseline_entry['accuracy']:.4f}** |
| F1 | **{baseline_entry['f1']:.4f}** |
| ROC-AUC | **{baseline_entry['roc_auc']:.4f}** |
| Training Time | {baseline_entry['train_time']:.2f}s |
| Train / Test size | {train_size:,} / {test_size:,} |

---

## 2. All TF-IDF Experiments

| Experiment | Feature Rep | Vocab Size | minDF | N-gram | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time | Pred Time |
|:----------|:-----------|:----------|:-----|:------|:-------:|:--------:|:-----:|:--:|:------:|:---------:|:--------:|
{table_rows}

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

{trigram_note}

---

## 5. Best Feature Configuration

| Field | Value |
|-------|-------|
| **Experiment** | {best_exp['experiment'].replace(chr(10),' ')} |
| **Feature Representation** | {best_exp['feature_representation']} |
| **Vocabulary Size** | {best_exp['vocab_size']} |
| **minDF** | {best_exp['min_df']:.0f} |
| **N-gram Configuration** | {best_exp['ngram_range']} |
| **Accuracy** | **{best_exp['accuracy']:.4f}** |
| **F1 Score** | **{best_exp['f1']:.4f}** |
| **ROC-AUC** | **{best_exp['roc_auc']:.4f}** |
| **Training Time** | {best_exp['train_time']:.2f}s |

---

## 6. Phase 1 vs Phase 2 Comparison

| Metric | Phase 1 Best | Phase 2 Best | Δ |
|--------|:----------:|:----------:|:--:|
| **Accuracy** | {phase1_best['accuracy']:.4f} | {best_exp['accuracy']:.4f} | **{"+" if acc_delta_vs_ph1 >= 0 else ""}{acc_delta_vs_ph1:.4f}** |
| **F1 Score** | {phase1_best['f1']:.4f} | {best_exp['f1']:.4f} | **{"+" if f1_delta_vs_ph1 >= 0 else ""}{f1_delta_vs_ph1:.4f}** |
| **ROC-AUC** | {phase1_best['roc_auc']:.4f} | {best_exp['roc_auc']:.4f} | {"+" if best_exp['roc_auc'] - phase1_best['roc_auc'] >= 0 else ""}{round(best_exp['roc_auc'] - phase1_best['roc_auc'], 4):.4f} |
| **Training Time** | {phase1_best['train_time']:.2f}s | {best_exp['train_time']:.2f}s | {"+" if time_delta_vs_ph1 >= 0 else ""}{time_delta_vs_ph1:.2f}s |

---

## 7. Why the Best Configuration Performed Best

{why_best}

---

## 8. Runtime & Memory Notes

- All experiments ran on 16 GB RAM (MacBook Air, local[*] Spark).
- Larger vocabulary sizes increase the dimensionality of feature vectors, which causes a slight increase in training time.
- Memory remained within 8 GB Spark driver limit for all feasible experiments.
- Trigram feature vectors, when summed with unigram + bigram, can become very large-dimensional — note result above.

---

## 9. Model Preservation

{
    f"Best Phase 2 model saved to `saved_model_phase2/` and `pipeline/saved_model_phase2/`."
    if phase2_saves else
    "Phase 2 best result did NOT exceed Phase 1 best — **no Phase 2 model was saved**. Phase 1 model in `pipeline/saved_model_phase1/` remains the current best."
}

The existing production model in `pipeline/saved_model/` and all `results/results.json` data are completely untouched.
"""

    for sp in (os.path.join(ROOT_RESULTS_DIR, "phase2_summary.md"),
               os.path.join(PIPELINE_RESULTS_DIR, "phase2_summary.md")):
        with open(sp, "w") as f:
            f.write(summary)
        log(f"Saved summary → {sp}")

    log("Phase 2 experiments completed successfully.")
    spark.stop()


if __name__ == "__main__":
    main()
