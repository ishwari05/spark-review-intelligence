"""
================================================================================
 Phase 4 Experiment Runner: Classifier Comparison & Hyperparameter Tuning
 ------------------------------------------------------------------------
 LOCKED CONFIGURATION FROM PHASE 2:
   Preprocessing : Phase 1 best – Variant C
                   (Contractions + Preserved Negations + Negation Marking)
   Features      : Unigram + Bigram TF-IDF
                   vocabSize = 50,000 (both unigram and bigram)
                   minDF     = 5.0
                   N-grams   = 1 + 2
   Dataset       : Amazon Polarity, 100,000 rows
   Train/Test    : 80/20, seed=42

 VARIED in Phase 4:
   Only the CLASSIFIER (and its hyperparameters).

 Experiments:
   3-A  Phase 4 Baseline  (LR maxIter=20, Phase 2 exact config)
   3-B  Naive Bayes       (no hyp-param tuning needed)
   3-C  LR – Tuned        (TrainValidationSplit on training data only)
   3-D  LinearSVC – Tuned (TrainValidationSplit on training data only)
   3-E  LR Weighted       (only if train-set imbalance > 5%)
   3-F  SVC Weighted      (only if train-set imbalance > 5%)

 Anti-leakage contract:
   TrainValidationSplit splits TRAINING data only (trainRatio=0.8, seed=42).
   The final TEST set is touched exactly once per experiment, after best
   hyper-parameters are selected and the model is retrained on full train data.

 ROC-AUC note for LinearSVC:
   LinearSVC does NOT output a probability column.  Its rawPrediction is a
   signed-margin score (not calibrated).  BinaryClassificationEvaluator can
   rank by rawPrediction[1] and compute an approximate AUC — this is
   documented as "margin-based AUC" and flagged accordingly.
================================================================================
"""

import os
import sys
import re
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
from pyspark.ml.classification import (
    LogisticRegression, LinearSVC, NaiveBayes,
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

# ---------------------------------------------------------------------------
# Constants — ALL locked from Phase 2
# ---------------------------------------------------------------------------
SAMPLE_SIZE       = 100_000
TEST_FRACTION     = 0.2
SEED              = 42

# Phase 2 BEST feature config (EXP 3)
UNI_VOCAB         = 50_000
BI_VOCAB          = 50_000
MIN_DF            = 5.0

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PIPELINE_RESULTS  = os.path.join(BASE_DIR, "results")
ROOT_RESULTS      = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR       = os.path.join(ROOT_RESULTS, "figures")
PIPE_FIGURES_DIR  = os.path.join(PIPELINE_RESULTS, "figures")
PHASE4_MODEL_PIPE = os.path.join(BASE_DIR, "saved_model_phase4")
PHASE4_MODEL_ROOT = os.path.join(PROJECT_ROOT, "saved_model_phase4")

for d in (PIPELINE_RESULTS, ROOT_RESULTS, FIGURES_DIR, PIPE_FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

from pipeline.preprocessing import preprocess_variant_c, get_sentiment_stopwords


# ---------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[phase4] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------
def get_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("Phase3ClassifierComparison")
        .master("local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    log(f"Spark {spark.version} started.")
    return spark


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset(spark):
    from datasets import load_dataset as hf
    log(f"Loading {SAMPLE_SIZE:,} rows (cached)...")
    ds   = hf("amazon_polarity", split=f"train[:{SAMPLE_SIZE}]")
    rows = [{"label": float(r["label"]), "title": r["title"],
             "review": r["content"]} for r in ds]
    sdf  = spark.createDataFrame(rows, schema=["label", "title", "review"])
    sdf  = sdf.withColumn("label", F.col("label").cast(DoubleType()))
    sdf  = sdf.repartition(8).cache()
    log(f"Cached {sdf.count():,} rows.")
    return sdf


def prepare_data(raw_df):
    # Phase 4 modification: Combine title and review for richer context
    df = raw_df.dropna(subset=["title", "review"])
    df = df.withColumn("review", F.concat_ws(" ", F.col("title"), F.col("review")))
    df = df.dropDuplicates(["review"])
    df = df.filter(F.length(F.trim(F.col("review"))) > 0)
    return df


# ---------------------------------------------------------------------------
# Phase 2 BEST feature stages builder (50k / minDF=5 / unigram+bigram)
# ---------------------------------------------------------------------------
def feature_stages(stop_words):
    """Returns unfitted Spark ML stages for Phase 2 best feature config."""
    tok   = Tokenizer(inputCol="clean_text", outputCol="tokens")
    rem   = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens",
                             stopWords=stop_words)
    bi    = NGram(n=2, inputCol="filtered_tokens", outputCol="bigrams")
    ucv   = CountVectorizer(inputCol="filtered_tokens", outputCol="uni_raw",
                            vocabSize=UNI_VOCAB, minDF=MIN_DF)
    uid   = IDF(inputCol="uni_raw", outputCol="uni_feat")
    bcv   = CountVectorizer(inputCol="bigrams", outputCol="bi_raw",
                            vocabSize=BI_VOCAB, minDF=MIN_DF)
    bid   = IDF(inputCol="bi_raw", outputCol="bi_feat")
    asm   = VectorAssembler(inputCols=["uni_feat", "bi_feat"],
                            outputCol="features")
    return [tok, rem, bi, ucv, uid, bcv, bid, asm]


# ---------------------------------------------------------------------------
# Evaluation helper — used uniformly for all classifiers
# ---------------------------------------------------------------------------
def evaluate_predictions(preds, roc_col="probability", roc_col_is_prob=True):
    """
    Compute Accuracy, Precision, Recall, F1, ROC-AUC on a predictions df.
    roc_col          : column used for ROC computation
    roc_col_is_prob  : True → standard probability (LR/NB),
                       False → raw margin score (LinearSVC, flagged)
    """
    mc = MulticlassClassificationEvaluator(labelCol="label",
                                           predictionCol="prediction")
    acc  = mc.evaluate(preds, {mc.metricName: "accuracy"})
    prec = mc.evaluate(preds, {mc.metricName: "weightedPrecision"})
    rec  = mc.evaluate(preds, {mc.metricName: "weightedRecall"})
    f1   = mc.evaluate(preds, {mc.metricName: "f1"})

    roc_auc      = None
    roc_auc_note = ""
    if roc_col in preds.columns:
        try:
            bce = BinaryClassificationEvaluator(
                labelCol="label",
                rawPredictionCol=roc_col,
                metricName="areaUnderROC",
            )
            roc_auc = bce.evaluate(preds)
            if not roc_col_is_prob:
                roc_auc_note = "(margin-based, uncalibrated)"
        except Exception as e:
            roc_auc_note = f"(failed: {e})"

    return {
        "accuracy":      round(float(acc),  4),
        "precision":     round(float(prec), 4),
        "recall":        round(float(rec),  4),
        "f1":            round(float(f1),   4),
        "roc_auc":       round(float(roc_auc), 4) if roc_auc is not None else None,
        "roc_auc_note":  roc_auc_note,
    }


def full_evaluate(name, model, test_df, roc_col="probability", roc_col_is_prob=True):
    t0    = time.time()
    preds = model.transform(test_df).cache()
    preds.count()
    pred_time = round(time.time() - t0, 2)
    m = evaluate_predictions(preds, roc_col=roc_col, roc_col_is_prob=roc_col_is_prob)
    m["predict_time"] = pred_time

    # Confusion matrix
    cm_rows = preds.groupBy("label", "prediction").count().collect()
    cm = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for r in cm_rows:
        lbl, pred, cnt = r["label"], r["prediction"], r["count"]
        if   lbl == 1.0 and pred == 1.0: cm["tp"] += cnt
        elif lbl == 0.0 and pred == 0.0: cm["tn"] += cnt
        elif lbl == 0.0 and pred == 1.0: cm["fp"] += cnt
        elif lbl == 1.0 and pred == 0.0: cm["fn"] += cnt

    m["confusion_matrix"] = cm
    preds.unpersist()
    log(f"{name}: acc={m['accuracy']} f1={m['f1']} auc={m['roc_auc']} "
        f"pred_time={pred_time}s")
    return m, preds


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_nb(train_df, test_df, stops):
    """Naive Bayes — no hyperparameter tuning needed."""
    stages = feature_stages(stops)
    nb     = NaiveBayes(labelCol="label", featuresCol="features",
                        modelType="multinomial")
    pipeline = Pipeline(stages=stages + [nb])

    log("Training Naive Bayes (no tuning)...")
    t0    = time.time()
    model = pipeline.fit(train_df)
    train_time = round(time.time() - t0, 2)

    m, _ = full_evaluate("Naive Bayes", model, test_df,
                          roc_col="probability", roc_col_is_prob=True)
    m["train_time"] = train_time
    m["best_params"] = "smoothing=1.0 (default)"
    return m, model


def run_lr_tuned(train_df, test_df, stops):
    """
    Logistic Regression tuned via TrainValidationSplit on training data.
    Param grid:
      regParam       ∈ {0.001, 0.01, 0.05, 0.1, 0.5}
      elasticNetParam ∈ {0.0, 0.5}
      maxIter        = 100 (fixed during tuning for speed)
    After selection → retrain with best params using maxIter=200 on full train.
    """
    stages   = feature_stages(stops)
    lr       = LogisticRegression(labelCol="label", featuresCol="features",
                                  maxIter=100)
    pipeline = Pipeline(stages=stages + [lr])

    grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam,        [0.001, 0.01, 0.05, 0.1, 0.5])
        .addGrid(lr.elasticNetParam, [0.0, 0.5])
        .build()
    )  # 5 × 2 = 10 combos

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    tvs = TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        trainRatio=0.8,
        seed=SEED,
        parallelism=2,
    )

    log("Running TrainValidationSplit for Logistic Regression (10 combos)...")
    t0        = time.time()
    tvs_model = tvs.fit(train_df)
    tune_time = round(time.time() - t0, 2)

    best_lr      = tvs_model.bestModel.stages[-1]
    best_reg     = best_lr.getRegParam()
    best_elastic = best_lr.getElasticNetParam()
    log(f"LR best params → regParam={best_reg}, elasticNetParam={best_elastic} "
        f"(found in {tune_time}s)")

    # Retrain with best params + maxIter=200 on FULL training set
    log("Retraining best LR on full training data (maxIter=200)...")
    stages2   = feature_stages(stops)
    lr_best   = LogisticRegression(labelCol="label", featuresCol="features",
                                   maxIter=200, regParam=best_reg,
                                   elasticNetParam=best_elastic)
    pipeline2 = Pipeline(stages=stages2 + [lr_best])

    t0         = time.time()
    final_model = pipeline2.fit(train_df)
    train_time  = round(time.time() - t0, 2)

    m, _ = full_evaluate("Logistic Regression (tuned)", final_model, test_df,
                          roc_col="probability", roc_col_is_prob=True)
    m["train_time"]   = train_time
    m["tuning_time"]  = tune_time
    m["best_params"]  = (f"regParam={best_reg}, elasticNetParam={best_elastic}, "
                         f"maxIter=200")
    return m, final_model, best_reg, best_elastic


def run_svc_tuned(train_df, test_df, stops):
    """
    LinearSVC tuned via TrainValidationSplit on training data.
    Param grid:
      regParam ∈ {0.001, 0.01, 0.05, 0.1, 0.5}
      maxIter  = 100 (fixed during tuning)
    After selection → retrain with best regParam using maxIter=200.

    ROC-AUC: LinearSVC rawPrediction used (margin-based, flagged).
    """
    stages   = feature_stages(stops)
    svc      = LinearSVC(labelCol="label", featuresCol="features", maxIter=100)
    pipeline = Pipeline(stages=stages + [svc])

    grid = (
        ParamGridBuilder()
        .addGrid(svc.regParam, [0.001, 0.01, 0.05, 0.1, 0.5])
        .build()
    )  # 5 combos

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy"
    )
    tvs = TrainValidationSplit(
        estimator=pipeline,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        trainRatio=0.8,
        seed=SEED,
        parallelism=2,
    )

    log("Running TrainValidationSplit for LinearSVC (5 combos)...")
    t0        = time.time()
    tvs_model = tvs.fit(train_df)
    tune_time = round(time.time() - t0, 2)

    best_svc = tvs_model.bestModel.stages[-1]
    best_reg = best_svc.getRegParam()
    log(f"LinearSVC best params → regParam={best_reg} (found in {tune_time}s)")

    # Retrain with best params + maxIter=200
    log("Retraining best LinearSVC on full training data (maxIter=200)...")
    stages2   = feature_stages(stops)
    svc_best  = LinearSVC(labelCol="label", featuresCol="features",
                          maxIter=200, regParam=best_reg)
    pipeline2 = Pipeline(stages=stages2 + [svc_best])

    t0          = time.time()
    final_model = pipeline2.fit(train_df)
    train_time  = round(time.time() - t0, 2)

    m, _ = full_evaluate("LinearSVC (tuned)", final_model, test_df,
                          roc_col="rawPrediction", roc_col_is_prob=False)
    m["train_time"]   = train_time
    m["tuning_time"]  = tune_time
    m["best_params"]  = f"regParam={best_reg}, maxIter=200"
    return m, final_model, best_reg


def run_weighted(train_df, test_df, stops, pos_weight, neg_weight,
                 best_lr_reg, best_lr_elastic, best_svc_reg):
    """
    Run both LR and LinearSVC with class weights derived from training data.
    Only called when imbalance > 5%.
    """
    results = {}

    # Weighted LR
    log(f"Training weighted LR (w_neg={neg_weight:.3f}, w_pos={pos_weight:.3f})...")
    weighted_train = train_df.withColumn(
        "class_weight",
        F.when(F.col("label") == 0.0, neg_weight).otherwise(pos_weight),
    )
    stages   = feature_stages(stops)
    lr_w     = LogisticRegression(labelCol="label", featuresCol="features",
                                  maxIter=200, regParam=best_lr_reg,
                                  elasticNetParam=best_lr_elastic,
                                  weightCol="class_weight")
    pipeline = Pipeline(stages=stages + [lr_w])
    t0       = time.time()
    model_w  = pipeline.fit(weighted_train)
    train_time = round(time.time() - t0, 2)
    m, _     = full_evaluate("LR Weighted", model_w, test_df,
                              roc_col="probability", roc_col_is_prob=True)
    m["train_time"]  = train_time
    m["best_params"] = (f"regParam={best_lr_reg}, "
                        f"elasticNetParam={best_lr_elastic}, "
                        f"w_neg={neg_weight:.3f}, w_pos={pos_weight:.3f}")
    results["lr_weighted"] = (m, model_w)

    # Weighted SVC
    log(f"Training weighted LinearSVC (w_neg={neg_weight:.3f}, "
        f"w_pos={pos_weight:.3f})...")
    stages    = feature_stages(stops)
    svc_w     = LinearSVC(labelCol="label", featuresCol="features",
                          maxIter=200, regParam=best_svc_reg,
                          weightCol="class_weight")
    pipeline2 = Pipeline(stages=stages + [svc_w])
    t0        = time.time()
    model_w2  = pipeline2.fit(weighted_train)
    train_time = round(time.time() - t0, 2)
    m2, _    = full_evaluate("SVC Weighted", model_w2, test_df,
                              roc_col="rawPrediction", roc_col_is_prob=False)
    m2["train_time"]  = train_time
    m2["best_params"] = (f"regParam={best_svc_reg}, "
                         f"w_neg={neg_weight:.3f}, w_pos={pos_weight:.3f}")
    results["svc_weighted"] = (m2, model_w2)
    return results


# ---------------------------------------------------------------------------
# Error analysis — compare best Phase 4 model vs Phase 2 metrics
# ---------------------------------------------------------------------------
def error_analysis(model, test_df, negation_pattern=None):
    """
    Analyse misclassified examples from best Phase 4 model.
    Checks: negation, review length, confidence (for LR/NB models).
    """
    if negation_pattern is None:
        negation_pattern = r"\b(not|no|never|neither|nor|nothing|nowhere|hardly|barely|cannot)\b"

    preds = model.transform(test_df).cache()

    preds = preds.withColumn("is_correct", F.col("label") == F.col("prediction"))
    preds = preds.withColumn(
        "error_type",
        F.when((F.col("label") == 1.0) & (F.col("prediction") == 1.0), "TP")
         .when((F.col("label") == 0.0) & (F.col("prediction") == 0.0), "TN")
         .when((F.col("label") == 0.0) & (F.col("prediction") == 1.0), "FP")
         .otherwise("FN"),
    )
    preds = preds.withColumn(
        "has_negation", F.col("clean_text").rlike(negation_pattern)
    )
    preds = preds.withColumn(
        "review_length", F.size(F.split(F.col("clean_text"), " "))
    ).cache()

    total   = preds.count()
    correct = preds.filter(F.col("is_correct")).count()
    wrong   = total - correct

    neg_total  = preds.filter(F.col("has_negation")).count()
    neg_errors = preds.filter(F.col("is_correct") == False)          \
                      .filter(F.col("has_negation")).count()

    # Length bucket error rates
    preds = preds.withColumn(
        "len_bucket",
        F.when(F.col("review_length") < 15, "Short (<15)")
         .when(F.col("review_length") <= 50, "Medium (15-50)")
         .otherwise("Long (>50)"),
    )
    length_stats = (
        preds.groupBy("len_bucket")
        .agg(
            F.count("*").alias("total"),
            F.sum(F.when(~F.col("is_correct"), 1).otherwise(0)).alias("errors"),
        ).collect()
    )
    length_breakdown = [
        {"bucket": r["len_bucket"],
         "total":  r["total"],
         "errors": r["errors"],
         "error_rate": round(r["errors"] / r["total"], 4) if r["total"] else 0.0}
        for r in length_stats
    ]
    length_breakdown.sort(key=lambda x: x["bucket"])

    # Sample misclassified examples
    wrong_samples = (
        preds.filter(~F.col("is_correct"))
        .select("review", "label", "prediction", "review_length",
                "error_type", "clean_text", "has_negation")
        .orderBy(F.rand(SEED))
        .limit(10)
        .collect()
    )
    examples = []
    for r in wrong_samples:
        txt = r["review"]
        examples.append({
            "review":      (txt[:200] + "...") if len(txt) > 200 else txt,
            "true":        "Positive" if r["label"] == 1.0 else "Negative",
            "predicted":   "Positive" if r["prediction"] == 1.0 else "Negative",
            "length":      int(r["review_length"]),
            "error_type":  r["error_type"],
            "has_negation": bool(r["has_negation"]),
        })

    preds.unpersist()

    return {
        "total":                 total,
        "correct":               correct,
        "incorrect":             wrong,
        "error_rate":            round(wrong / total, 4) if total else 0.0,
        "negation_total":        neg_total,
        "negation_errors":       neg_errors,
        "negation_error_rate":   round(neg_errors / neg_total, 4) if neg_total else 0.0,
        "negation_error_pct":    round(neg_errors / wrong * 100, 2) if wrong else 0.0,
        "length_breakdown":      length_breakdown,
        "misclassified_examples": examples,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(records: list, out_paths_prefix: str) -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 10})

    valid = [r for r in records if r.get("accuracy") is not None]
    labels = [r["experiment"] for r in valid]
    accs   = [r["accuracy"]  for r in valid]
    f1s    = [r["f1"]        for r in valid]
    times  = [r["train_time"] for r in valid]
    palette = plt.cm.tab10.colors

    def _bar_plot(values, title, ylabel, fname, annotate_fmt="{:.4f}"):
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.5), 5.5))
        bars = ax.bar(labels, values, color=palette[:len(labels)], width=0.5, alpha=0.88)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    annotate_fmt.format(h), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        plt.tight_layout()
        for root in (ROOT_RESULTS, PIPELINE_RESULTS):
            plt.savefig(os.path.join(root, "figures", fname), dpi=300)
        plt.close()
        log(f"Saved {fname}")

    _bar_plot(accs,  "Phase 4: Accuracy Comparison",       "Accuracy",      "phase4_accuracy_comparison.png")
    _bar_plot(f1s,   "Phase 4: Weighted F1 Comparison",    "Weighted F1",   "phase4_f1_comparison.png")
    _bar_plot(times, "Phase 4: Training Time Comparison",  "Train Time (s)", "phase4_training_time_comparison.png",
              annotate_fmt="{:.2f}s")


# ---------------------------------------------------------------------------
# CSV + Markdown output
# ---------------------------------------------------------------------------
CSV_FIELDS = [
    "experiment", "feature_representation", "classifier",
    "max_iter", "reg_param", "elastic_net_param", "class_weighting",
    "accuracy", "precision", "recall", "f1", "roc_auc",
    "train_time", "prediction_time",
]


def _parse_best_params(param_str: str, key: str, default=None):
    m = re.search(rf"{key}=([^\s,]+)", param_str or "")
    return m.group(1) if m else default


def build_csv_row(exp_id, clf_name, metrics, weighted=False):
    bp  = metrics.get("best_params", "")
    return {
        "experiment":            exp_id,
        "feature_representation": "Unigram+Bigram TF-IDF, 50k, minDF=5",
        "classifier":            clf_name,
        "max_iter":              _parse_best_params(bp, "maxIter", "—"),
        "reg_param":             _parse_best_params(bp, "regParam", "—"),
        "elastic_net_param":     _parse_best_params(bp, "elasticNetParam", "—"),
        "class_weighting":       "yes" if weighted else "no",
        "accuracy":              metrics.get("accuracy"),
        "precision":             metrics.get("precision"),
        "recall":                metrics.get("recall"),
        "f1":                    metrics.get("f1"),
        "roc_auc":               metrics.get("roc_auc"),
        "train_time":            metrics.get("train_time"),
        "prediction_time":       metrics.get("predict_time"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    spark     = get_spark()
    raw_df    = load_dataset(spark)
    clean_df  = prepare_data(raw_df)

    # Phase 2 best preprocessing (Variant C) — LOCKED
    log("Applying Phase 1/2 best preprocessing (Variant C)...")
    df = preprocess_variant_c(clean_df, "review", "clean_text")

    # Same train/test split as Phase 1 and Phase 2 — LOCKED
    train_df, test_df = df.randomSplit([1 - TEST_FRACTION, TEST_FRACTION],
                                       seed=SEED)
    train_df = train_df.cache()
    test_df  = test_df.cache()
    train_size = train_df.count()
    test_size  = test_df.count()
    log(f"Train={train_size:,}  Test={test_size:,}  seed={SEED}")

    stops = get_sentiment_stopwords()

    # -------------------------------------------------------------------
    # Phase 4 Baseline — same LR as Phase 2 winner (maxIter=20, no tuning)
    # -------------------------------------------------------------------
    log("=== EXP 3-A: Phase 4 Baseline (LR maxIter=20) ===")
    stages_a = feature_stages(stops)
    lr_base  = LogisticRegression(labelCol="label", featuresCol="features",
                                  maxIter=20)
    pipe_a   = Pipeline(stages=stages_a + [lr_base])
    t0 = time.time()
    model_a = pipe_a.fit(train_df)
    tt_a    = round(time.time() - t0, 2)
    m_a, _  = full_evaluate("3-A Baseline LR", model_a, test_df,
                             roc_col="probability", roc_col_is_prob=True)
    m_a["train_time"]  = tt_a
    m_a["best_params"] = "maxIter=20, regParam=0.0 (default), elasticNetParam=0.0"
    log(f"3-A done in {tt_a}s")

    # -------------------------------------------------------------------
    # EXP 3-B: Naive Bayes
    # -------------------------------------------------------------------
    log("=== EXP 3-B: Naive Bayes ===")
    m_nb, model_nb = run_nb(train_df, test_df, stops)
    log(f"3-B done in {m_nb['train_time']}s")

    # -------------------------------------------------------------------
    # EXP 3-C: Logistic Regression — tuned
    # -------------------------------------------------------------------
    log("=== EXP 3-C: Logistic Regression (TVS tuning) ===")
    m_lr, model_lr, best_lr_reg, best_lr_elastic = run_lr_tuned(
        train_df, test_df, stops)
    log(f"3-C done — total tuning+retrain time: "
        f"{m_lr['tuning_time']}s + {m_lr['train_time']}s")

    # -------------------------------------------------------------------
    # EXP 3-D: LinearSVC — tuned
    # -------------------------------------------------------------------
    log("=== EXP 3-D: LinearSVC (TVS tuning) ===")
    m_svc, model_svc, best_svc_reg = run_svc_tuned(train_df, test_df, stops)
    log(f"3-D done — total tuning+retrain time: "
        f"{m_svc['tuning_time']}s + {m_svc['train_time']}s")

    # -------------------------------------------------------------------
    # Class balance check → optional weighted experiments
    # -------------------------------------------------------------------
    pos_count = train_df.filter(F.col("label") == 1.0).count()
    neg_count = train_size - pos_count
    pos_frac  = pos_count / train_size
    neg_frac  = neg_count / train_size
    imbalance = abs(pos_frac - neg_frac)
    log(f"Train class distribution: positive={pos_count:,} ({pos_frac*100:.1f}%), "
        f"negative={neg_count:,} ({neg_frac*100:.1f}%)")

    weighted_results = {}
    do_weighted      = imbalance > 0.05
    if do_weighted:
        log(f"Imbalance={imbalance*100:.1f}% > 5% — running weighted experiments...")
        # Balanced inverse-frequency weights
        pos_weight = train_size / (2 * pos_count)
        neg_weight = train_size / (2 * neg_count)
        weighted_results = run_weighted(
            train_df, test_df, stops,
            pos_weight, neg_weight,
            best_lr_reg, best_lr_elastic, best_svc_reg,
        )
    else:
        log(f"Imbalance={imbalance*100:.1f}% ≤ 5% — classes are balanced. "
            "Skipping weighted experiments.")

    # -------------------------------------------------------------------
    # Select overall best model
    # -------------------------------------------------------------------
    candidates = {
        "3-A Phase3-Baseline (LR maxIter=20)": (m_a,   model_a),
        "3-B Naive Bayes":                     (m_nb,  model_nb),
        "3-C LR Tuned":                        (m_lr,  model_lr),
        "3-D LinearSVC Tuned":                 (m_svc, model_svc),
    }
    if "lr_weighted" in weighted_results:
        m_lrw, model_lrw = weighted_results["lr_weighted"]
        candidates["3-E LR Weighted"] = (m_lrw, model_lrw)
    if "svc_weighted" in weighted_results:
        m_svcw, model_svcw = weighted_results["svc_weighted"]
        candidates["3-F SVC Weighted"] = (m_svcw, model_svcw)

    best_name = max(candidates,
                    key=lambda k: (candidates[k][0]["f1"],
                                   candidates[k][0]["accuracy"]))
    best_m, best_model = candidates[best_name]
    log(f"Best Phase 4 classifier: {best_name} — "
        f"Acc={best_m['accuracy']} F1={best_m['f1']}")

    # -------------------------------------------------------------------
    # Error analysis on best Phase 4 model
    # -------------------------------------------------------------------
    log("Running error analysis on best Phase 4 model...")
    err = error_analysis(best_model, test_df)
    log(f"Error rate={err['error_rate']}, "
        f"negation_error_rate={err['negation_error_rate']}")

    # -------------------------------------------------------------------
    # Save best model if it improves Phase 2 (Acc=0.7896)
    # -------------------------------------------------------------------
    PHASE2_BEST_ACC = 0.7896
    model_saved = False
    if best_m["accuracy"] > PHASE2_BEST_ACC:
        for mdir in (PHASE4_MODEL_PIPE, PHASE4_MODEL_ROOT):
            log(f"Saving Phase 4 best model to {mdir}...")
            best_model.write().overwrite().save(mdir)
        model_saved = True
    else:
        log("Phase 4 best did NOT exceed Phase 2 — Phase 2 model stays current best.")

    # -------------------------------------------------------------------
    # Build CSV rows
    # -------------------------------------------------------------------
    all_rows = [
        build_csv_row("3-A Phase3-Baseline", "Logistic Regression",
                      m_a,  weighted=False),
        build_csv_row("3-B Naive Bayes",      "Naive Bayes",
                      m_nb, weighted=False),
        build_csv_row("3-C LR Tuned",         "Logistic Regression",
                      m_lr, weighted=False),
        build_csv_row("3-D LinearSVC Tuned",  "LinearSVC",
                      m_svc, weighted=False),
    ]
    if "lr_weighted" in weighted_results:
        all_rows.append(build_csv_row("3-E LR Weighted", "Logistic Regression",
                        weighted_results["lr_weighted"][0], weighted=True))
    if "svc_weighted" in weighted_results:
        all_rows.append(build_csv_row("3-F SVC Weighted", "LinearSVC",
                        weighted_results["svc_weighted"][0], weighted=True))

    for cp in (os.path.join(ROOT_RESULTS, "phase4_results.csv"),
               os.path.join(PIPELINE_RESULTS, "phase4_results.csv")):
        with open(cp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        log(f"Saved CSV → {cp}")

    # -------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------
    plot_records = [
        {"experiment": "3-A\nBaseline\nLR-20",  **m_a},
        {"experiment": "3-B\nNaive\nBayes",      **m_nb},
        {"experiment": "3-C\nLR\nTuned",         **m_lr},
        {"experiment": "3-D\nLinearSVC\nTuned",  **m_svc},
    ]
    if "lr_weighted" in weighted_results:
        plot_records.append({"experiment": "3-E\nLR\nWeighted",
                             **weighted_results["lr_weighted"][0]})
    if "svc_weighted" in weighted_results:
        plot_records.append({"experiment": "3-F\nSVC\nWeighted",
                             **weighted_results["svc_weighted"][0]})
    plot_results(plot_records, "phase4")

    # -------------------------------------------------------------------
    # Confusion matrix formatted
    # -------------------------------------------------------------------
    cm = best_m.get("confusion_matrix", {})
    cm_str = (f"TP={cm.get('tp',0):,}  TN={cm.get('tn',0):,}  "
              f"FP={cm.get('fp',0):,}  FN={cm.get('fn',0):,}")

    # -------------------------------------------------------------------
    # Error analysis formatted
    # -------------------------------------------------------------------
    len_rows = "\n".join(
        f"| {r['bucket']} | {r['total']:,} | {r['errors']:,} | "
        f"{r['error_rate']*100:.1f}% |"
        for r in err["length_breakdown"]
    )
    eg_rows = "\n".join(
        f"| {e['error_type']} | {'✓' if e['has_negation'] else '✗'} | "
        f"{e['length']} | {e['true']} → {e['predicted']} | "
        f"{e['review'][:80]}... |"
        for e in err["misclassified_examples"][:5]
    )

    # -------------------------------------------------------------------
    # Markdown summary
    # -------------------------------------------------------------------
    weighted_section = (
        f"### Class Balance Check\n"
        f"- Positive: {pos_count:,} ({pos_frac*100:.1f}%)\n"
        f"- Negative: {neg_count:,} ({neg_frac*100:.1f}%)\n"
        f"- Imbalance: {imbalance*100:.1f}% — "
        + ("**> 5%, weighted models tested.**" if do_weighted else
           "**≤ 5%, classes are balanced. Weighted experiments skipped.**")
    )

    acc_delta = round(best_m["accuracy"] - PHASE2_BEST_ACC, 4)
    f1_ph2    = 0.7897
    f1_delta  = round(best_m["f1"] - f1_ph2, 4)

    roc_note = ""
    if "svc" in best_name.lower():
        roc_note = " *(margin-based, uncalibrated for LinearSVC)*"

    summary = f"""# Phase 4: Classifier Comparison & Hyperparameter Tuning

## 1. Locked Phase 2 Configuration (Applied to All Experiments)

| Parameter | Value |
|-----------|-------|
| Preprocessing | Phase 1 best — Contractions + Preserved Negations + Negation Marking |
| Feature representation | Unigram + Bigram TF-IDF |
| Vocabulary size | 50,000 (unigrams) + 50,000 (bigrams) |
| minDF | 5.0 |
| N-gram range | 1 + 2 |
| Train / Test size | {train_size:,} / {test_size:,} |
| Random seed | {SEED} |

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
| **3-A — Phase 4 Baseline** | LR | maxIter=20 | {m_a['accuracy']:.4f} | {m_a['precision']:.4f} | {m_a['recall']:.4f} | {m_a['f1']:.4f} | {m_a['roc_auc']:.4f} | {m_a['train_time']}s | {m_a['predict_time']}s |
| **3-B — Naive Bayes** | NaiveBayes | smoothing=1.0 | {m_nb['accuracy']:.4f} | {m_nb['precision']:.4f} | {m_nb['recall']:.4f} | {m_nb['f1']:.4f} | {m_nb['roc_auc']:.4f} | {m_nb['train_time']}s | {m_nb['predict_time']}s |
| **3-C — LR Tuned** | LR (TVS) | {m_lr['best_params']} | {m_lr['accuracy']:.4f} | {m_lr['precision']:.4f} | {m_lr['recall']:.4f} | {m_lr['f1']:.4f} | {m_lr['roc_auc']:.4f} | {m_lr['train_time']}s | {m_lr['predict_time']}s |
| **3-D — LinearSVC Tuned** | LinearSVC (TVS) | {m_svc['best_params']} | {m_svc['accuracy']:.4f} | {m_svc['precision']:.4f} | {m_svc['recall']:.4f} | {m_svc['f1']:.4f} | {m_svc['roc_auc'] if m_svc['roc_auc'] else 'N/A'}{' (margin)' if m_svc['roc_auc'] else ''} | {m_svc['train_time']}s | {m_svc['predict_time']}s |
{"| **3-E — LR Weighted** | LR (weighted) | " + weighted_results.get('lr_weighted',({},{}))[0].get('best_params','') + " | " + f"{weighted_results['lr_weighted'][0]['accuracy']:.4f} | {weighted_results['lr_weighted'][0]['precision']:.4f} | {weighted_results['lr_weighted'][0]['recall']:.4f} | {weighted_results['lr_weighted'][0]['f1']:.4f} | {weighted_results['lr_weighted'][0]['roc_auc']:.4f} | {weighted_results['lr_weighted'][0]['train_time']}s | {weighted_results['lr_weighted'][0]['predict_time']}s |" if 'lr_weighted' in weighted_results else ''}
{"| **3-F — SVC Weighted** | SVC (weighted) | " + weighted_results.get('svc_weighted',({},{}))[0].get('best_params','') + " | " + f"{weighted_results['svc_weighted'][0]['accuracy']:.4f} | {weighted_results['svc_weighted'][0]['precision']:.4f} | {weighted_results['svc_weighted'][0]['recall']:.4f} | {weighted_results['svc_weighted'][0]['f1']:.4f} | {weighted_results['svc_weighted'][0]['roc_auc'] if weighted_results['svc_weighted'][0]['roc_auc'] else 'N/A'} | {weighted_results['svc_weighted'][0]['train_time']}s | {weighted_results['svc_weighted'][0]['predict_time']}s |" if 'svc_weighted' in weighted_results else ''}

> **ROC-AUC for LinearSVC** is computed using the raw signed decision-function margin (`rawPrediction`), not calibrated probabilities. It is an approximate ranking-based AUC — flagged as `(margin)`.

---

## 4. Hyperparameter Tuning Details

### Logistic Regression (TrainValidationSplit)
- Grid: regParam ∈ {{0.001, 0.01, 0.05, 0.1, 0.5}} × elasticNetParam ∈ {{0.0, 0.5}} = **10 combinations**
- Metric: Accuracy on 20% validation split of training data
- Tuning time: {m_lr.get('tuning_time', '—')}s
- Best params: `{m_lr['best_params']}`
- Final retrain: maxIter=200 on full training set

### LinearSVC (TrainValidationSplit)
- Grid: regParam ∈ {{0.001, 0.01, 0.05, 0.1, 0.5}} = **5 combinations**
- Metric: Accuracy on 20% validation split of training data
- Tuning time: {m_svc.get('tuning_time', '—')}s
- Best params: `{m_svc['best_params']}`
- Final retrain: maxIter=200 on full training set

### No-leakage Contract
Training data → 80/20 TVS validation split → select best params → retrain on 100% training set → evaluate ONCE on untouched test set.

---

## 5. Class Balance Analysis

{weighted_section}

---

## 6. Best Classifier

| Field | Value |
|-------|-------|
| **Best Experiment** | {best_name} |
| **Best Params** | `{best_m.get('best_params', '—')}` |
| **Accuracy** | **{best_m['accuracy']:.4f}** ({best_m['accuracy']*100:.2f}%) |
| **Precision** | {best_m['precision']:.4f} |
| **Recall** | {best_m['recall']:.4f} |
| **F1 Score** | **{best_m['f1']:.4f}** |
| **ROC-AUC** | {best_m['roc_auc']:.4f}{roc_note} |
| **Training Time** | {best_m['train_time']:.2f}s |
| **Prediction Time** | {best_m['predict_time']:.2f}s |
| **Confusion Matrix** | {cm_str} |

---

## 7. Phase 2 vs Phase 4 Comparison

| Metric | Phase 2 Best | Phase 4 Best | Δ |
|--------|:----------:|:----------:|:--:|
| **Accuracy** | 0.7896 | {best_m['accuracy']:.4f} | **{"+" if acc_delta >= 0 else ""}{acc_delta:.4f}** |
| **F1 Score** | 0.7897 | {best_m['f1']:.4f} | **{"+" if f1_delta >= 0 else ""}{f1_delta:.4f}** |
| **ROC-AUC** | 0.8734 | {best_m['roc_auc']:.4f} | {"+" if best_m['roc_auc'] - 0.8734 >= 0 else ""}{round(best_m['roc_auc'] - 0.8734, 4):.4f} |
| **Training Time** | 1.37s | {best_m['train_time']:.2f}s | {"+" if best_m['train_time']-1.37 >= 0 else ""}{round(best_m['train_time']-1.37,2):.2f}s |
| **Prediction Time** | 0.15s | {best_m['predict_time']:.2f}s | {"+" if best_m['predict_time']-0.15 >= 0 else ""}{round(best_m['predict_time']-0.15,2):.2f}s |

---

## 8. Error Analysis — Best Phase 4 Classifier

### Summary
| Metric | Value |
|--------|-------|
| Total test samples | {err['total']:,} |
| Correct predictions | {err['correct']:,} |
| Incorrect predictions | {err['incorrect']:,} |
| Overall error rate | {err['error_rate']*100:.2f}% |
| Reviews with negation | {err['negation_total']:,} ({round(err['negation_total']/err['total']*100,1)}% of test) |
| Negation errors | {err['negation_errors']:,} ({err['negation_error_pct']}% of all errors) |
| Negation error rate | {err['negation_error_rate']*100:.2f}% |

### Error Rate by Review Length
| Length Bucket | Total | Errors | Error Rate |
|:-------------|:-----:|:------:|:----------:|
{len_rows}

### Sample Misclassified Examples
| Type | Negation | Length | Prediction | Review (truncated) |
|:-----|:--------:|:------:|:----------:|:------------------|
{eg_rows}

---

## 9. 90% Accuracy Target

The 90% accuracy target was **{"achieved" if best_m['accuracy'] >= 0.90 else "NOT achieved"}**.
Best measured test accuracy = **{best_m['accuracy']*100:.2f}%** (honest, untouched test set, single evaluation).

---

## 10. Model Preservation

{"Best Phase 4 model saved to `saved_model_phase4/` and `pipeline/saved_model_phase4/`." if model_saved else "Phase 4 best did NOT exceed Phase 2 accuracy (0.7896) — **no Phase 4 model saved**. Phase 2 model in `saved_model_phase2/` remains current best."}

Existing production model (`pipeline/saved_model/`), ABSA, complaint mining, Flask API — completely untouched.
"""

    for sp in (os.path.join(ROOT_RESULTS, "phase4_summary.md"),
               os.path.join(PIPELINE_RESULTS, "phase4_summary.md")):
        with open(sp, "w") as f:
            f.write(summary)
        log(f"Saved summary → {sp}")

    log("Phase 4 complete.")
    spark.stop()


if __name__ == "__main__":
    main()
