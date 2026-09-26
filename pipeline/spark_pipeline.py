"""
================================================================================
 Scalable Product Review Intelligence Using Apache Spark and Machine Learning
 ---------------------------------------------------------------------------
 Big Data Analytics Mini Project

 This script is the "Big Data" half of the project. It:
   1. Sets up a SparkSession (4GB driver memory, 8 shuffle partitions)
   2. Loads a large-scale product review dataset (Amazon Polarity, via
      HuggingFace `datasets`) into a distributed Spark DataFrame
   3. Cleans + preprocesses text using Spark ML transformers
   4. Builds TF-IDF features (Unigram & Bigram pipelines) via Spark ML Pipelines
   5. Trains THREE Spark MLlib classifiers: Naive Bayes, Logistic Regression,
      Random Forest
   6. Evaluates every model with empirical metrics (Accuracy, Precision, Recall,
      F1, ROC-AUC, Training Time, Prediction Time)
   7. Conducts comprehensive Error Analysis & Business Insights
   8. Generates presentation-quality figures saved to `results/figures/`
   9. Saves the winning fitted PipelineModel to disk for live serving by api.py
================================================================================
"""

import json
import os
import re
import time
from collections import Counter
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    NGram,
    CountVectorizer,
    IDF,
    VectorAssembler,
)
from pyspark.ml.classification import (
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
SAMPLE_SIZE = 200000          # 200k subset size — balanced for Colab & Local
TEST_FRACTION = 0.2
SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def log(msg):
    print(f"[pipeline] {msg}", flush=True)


# --------------------------------------------------------------------------
# 1. Spark Setup
# --------------------------------------------------------------------------
def get_spark():
    spark = (
        SparkSession.builder.appName("ProductReviewIntelligence")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    log(f"Spark {spark.version} session started (local[*]).")
    return spark


# --------------------------------------------------------------------------
# 2. Dataset Acquisition + Distributed Loading
# --------------------------------------------------------------------------
def load_dataset(spark):
    """
    Dataset: Amazon Polarity (McAuley/He, Stanford SNAP) — 3.6M binary-labeled
    Amazon product reviews (label: 0=negative, 1=positive), title + review text.
    """
    from datasets import load_dataset as hf_load_dataset

    log(f"Downloading {SAMPLE_SIZE} rows of amazon_polarity (streaming)...")
    hf_ds = hf_load_dataset("amazon_polarity", split="train", streaming=True)
    rows = []
    for i, row in enumerate(hf_ds):
        if i >= SAMPLE_SIZE:
            break
        rows.append(
            {
                "label": float(row["label"]),  # 1 = positive, 0 = negative
                "title": row["title"],
                "review": row["content"],
            }
        )
    log(f"Pulled {len(rows)} rows. Handing off to Spark...")

    pdf_schema = ["label", "title", "review"]
    spark_df = spark.createDataFrame(rows, schema=pdf_schema)
    spark_df = spark_df.withColumn("label", F.col("label").cast(DoubleType()))
    spark_df = spark_df.repartition(8).cache()
    log(f"Spark DataFrame created and cached: {spark_df.count()} rows, "
        f"{len(spark_df.columns)} columns.")
    return spark_df


# --------------------------------------------------------------------------
# 3. Cleaning + Exploratory Data Analysis (EDA)
# --------------------------------------------------------------------------
def clean_and_explore(spark_df):
    before = spark_df.count()
    df = spark_df.dropna(subset=["review"])
    df = df.dropDuplicates(["review"])
    df = df.filter(F.length(F.trim(F.col("review"))) > 0)
    after = df.count()
    log(f"Cleaning: {before} -> {after} rows (removed nulls/dupes/empties).")

    # Lowercase + strip punctuation/special chars
    df = df.withColumn(
        "clean_text",
        F.regexp_replace(F.lower(F.col("review")), r"[^a-z0-9\s]", " "),
    )
    df = df.withColumn("clean_text", F.regexp_replace("clean_text", r"\s+", " "))
    df = df.withColumn("review_length", F.size(F.split(F.col("clean_text"), " ")))

    class_counts = df.groupBy("label").count().orderBy("label").collect()
    class_distribution = {
        ("positive" if r["label"] == 1.0 else "negative"): r["count"]
        for r in class_counts
    }

    length_stats = df.select(
        F.min("review_length").alias("min"),
        F.max("review_length").alias("max"),
        F.avg("review_length").alias("avg"),
    ).first()

    # Histogram buckets for review length
    bucket_col = F.when(F.col("review_length") >= 200, 200).otherwise(
        (F.floor(F.col("review_length") / 20) * 20)
    )
    length_hist = (
        df.withColumn("bucket", bucket_col)
        .groupBy("bucket")
        .count()
        .orderBy("bucket")
        .collect()
    )
    length_distribution = [{"bucket": int(r["bucket"]), "count": r["count"]} for r in length_hist]

    # Driver-side top-word frequency for word distribution
    sample_texts = [
        r["clean_text"]
        for r in df.select("clean_text").sample(0.05, seed=SEED).limit(20000).collect()
    ]
    stop = set(StopWordsRemover().getStopWords())
    counter = Counter()
    for t in sample_texts:
        for w in t.split():
            if len(w) > 2 and w not in stop:
                counter[w] += 1
    top_words = [{"word": w, "count": c} for w, c in counter.most_common(25)]

    eda = {
        "total_rows": after,
        "rows_removed_in_cleaning": before - after,
        "class_distribution": class_distribution,
        "review_length": {
            "min": length_stats["min"],
            "max": length_stats["max"],
            "avg": round(length_stats["avg"], 2),
            "histogram": length_distribution,
        },
        "top_words": top_words,
    }
    return df, eda


# --------------------------------------------------------------------------
# 4. Feature Pipelines: Unigrams & Unigram+Bigrams
# --------------------------------------------------------------------------
def build_unigram_feature_stages():
    """Unigram-only TF-IDF feature pipeline (Optimal for Naive Bayes)."""
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    unigram_count_vec = CountVectorizer(
        inputCol="filtered_tokens", outputCol="raw_features", vocabSize=20000, minDF=2.0
    )
    unigram_idf = IDF(inputCol="raw_features", outputCol="features")
    return [tokenizer, remover, unigram_count_vec, unigram_idf]


def build_feature_stages():
    """Unigram + Bigram TF-IDF feature pipeline."""
    tokenizer = Tokenizer(inputCol="clean_text", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    ngram = NGram(n=2, inputCol="filtered_tokens", outputCol="bigrams")
    unigram_count_vec = CountVectorizer(
        inputCol="filtered_tokens", outputCol="unigram_raw_features", vocabSize=20000, minDF=2.0
    )
    unigram_idf = IDF(inputCol="unigram_raw_features", outputCol="unigram_features")
    bigram_count_vec = CountVectorizer(
        inputCol="bigrams", outputCol="bigram_raw_features", vocabSize=20000, minDF=2.0
    )
    bigram_idf = IDF(inputCol="bigram_raw_features", outputCol="bigram_features")
    assembler = VectorAssembler(
        inputCols=["unigram_features", "bigram_features"], outputCol="features"
    )
    return [
        tokenizer,
        remover,
        ngram,
        unigram_count_vec,
        unigram_idf,
        bigram_count_vec,
        bigram_idf,
        assembler,
    ]


def run_scaling_demo(df):
    """Measure feature engineering plus Logistic Regression scaling at two dataset sizes."""
    scaling_results = []
    classifier = LogisticRegression(labelCol="label", featuresCol="features", maxIter=20)
    pipeline = Pipeline(stages=build_feature_stages() + [classifier])

    for requested_size in [10000, SAMPLE_SIZE]:
        subset = df.limit(requested_size).cache()
        actual_size = subset.count()
        log(f"Scaling demo: fitting Logistic Regression on {actual_size} rows...")
        started = time.time()
        pipeline.fit(subset)
        elapsed = round(time.time() - started, 2)
        scaling_results.append({"rows": int(actual_size), "train_time_sec": elapsed})
        log(f"Scaling demo: rows={actual_size} train={elapsed}s")
        subset.unpersist()

    return scaling_results


# --------------------------------------------------------------------------
# 5. Train + Evaluate One Model
# --------------------------------------------------------------------------
def train_and_evaluate(name, classifier, feature_rep, feature_stages, train_df, test_df):
    stages = feature_stages + [classifier]
    pipeline = Pipeline(stages=stages)

    log(f"Training {name} ({feature_rep})...")
    t0 = time.time()
    model = pipeline.fit(train_df)
    train_time = time.time() - t0

    t0 = time.time()
    preds = model.transform(test_df).cache()
    preds.count()  # materialize action to time prediction honestly
    predict_time = time.time() - t0

    mc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = mc_eval.evaluate(preds, {mc_eval.metricName: "accuracy"})
    precision = mc_eval.evaluate(preds, {mc_eval.metricName: "weightedPrecision"})
    recall = mc_eval.evaluate(preds, {mc_eval.metricName: "weightedRecall"})
    f1 = mc_eval.evaluate(preds, {mc_eval.metricName: "f1"})

    roc_auc = None
    roc_points = []
    if "probability" in preds.columns:
        bin_eval = BinaryClassificationEvaluator(
            labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC"
        )
        roc_auc = bin_eval.evaluate(preds)

        # Subsample ROC points for JSON payload
        score_labels = sorted(
            (float(row["probability"][1]), float(row["label"]))
            for row in preds.select("probability", "label").collect()
        )[::-1]
        positives = sum(label == 1.0 for _, label in score_labels)
        negatives = len(score_labels) - positives
        true_positives = false_positives = 0
        roc_curve = [(0.0, 0.0)]
        index = 0
        while index < len(score_labels):
            score = score_labels[index][0]
            while index < len(score_labels) and score_labels[index][0] == score:
                if score_labels[index][1] == 1.0:
                    true_positives += 1
                else:
                    false_positives += 1
                index += 1
            roc_curve.append(
                (
                    false_positives / negatives if negatives else 0.0,
                    true_positives / positives if positives else 0.0,
                )
            )
        step = max(1, len(roc_curve) // 60)
        roc_points = [{"fpr": float(p[0]), "tpr": float(p[1])} for p in roc_curve[::step]]

    # Confusion matrix
    cm_rows = preds.groupBy("label", "prediction").count().collect()
    cm = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for r in cm_rows:
        lbl, pred, cnt = r["label"], r["prediction"], r["count"]
        if lbl == 1.0 and pred == 1.0:
            cm["tp"] += cnt
        elif lbl == 0.0 and pred == 0.0:
            cm["tn"] += cnt
        elif lbl == 0.0 and pred == 1.0:
            cm["fp"] += cnt
        elif lbl == 1.0 and pred == 0.0:
            cm["fn"] += cnt

    metrics = {
        "name": name,
        "feature_representation": feature_rep,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "train_time_sec": round(train_time, 2),
        "predict_time_sec": round(predict_time, 2),
        "confusion_matrix": cm,
        "roc_curve": roc_points,
    }
    log(f"{name}: acc={metrics['accuracy']} f1={metrics['f1']} "
        f"auc={metrics['roc_auc']} train={metrics['train_time_sec']}s")

    preds.unpersist()
    return metrics, model


# --------------------------------------------------------------------------
# 6. Extract Top Indicative Terms (Positive & Negative)
# --------------------------------------------------------------------------
def extract_important_words(fitted_model, top_n=15):
    """
    Extract top positive and negative terms from the fitted model.
    Supports Naive Bayes (log-odds ratio) and Logistic Regression (linear coefficients).
    """
    try:
        from pyspark.ml.feature import CountVectorizerModel
        from pyspark.ml.classification import NaiveBayesModel, LogisticRegressionModel

        cv_models = [s for s in fitted_model.stages if isinstance(s, CountVectorizerModel)]

        # If Naive Bayes Model
        nb_models = [s for s in fitted_model.stages if isinstance(s, NaiveBayesModel)]
        if nb_models and cv_models:
            nb_model = nb_models[0]
            vocab = cv_models[0].vocabulary
            theta = nb_model.theta.toArray()  # theta[0]=neg log probs, theta[1]=pos log probs
            log_odds = theta[1] - theta[0]
            top_pos_idx = np.argsort(log_odds)[-top_n:][::-1]
            top_neg_idx = np.argsort(log_odds)[:top_n]
            top_positive_words = [
                {"word": vocab[i], "score": round(float(log_odds[i]), 4)} for i in top_pos_idx
            ]
            top_negative_words = [
                {"word": vocab[i], "score": round(float(abs(log_odds[i])), 4)} for i in top_neg_idx
            ]
            return top_positive_words, top_negative_words

        # If Logistic Regression Model
        lr_models = [s for s in fitted_model.stages if isinstance(s, LogisticRegressionModel)]
        if lr_models and cv_models:
            lr_model = lr_models[0]
            full_vocab = []
            for cv_m in cv_models:
                full_vocab.extend(cv_m.vocabulary)
            weights = lr_model.coefficients.toArray()
            top_pos_idx = np.argsort(weights)[-top_n:][::-1]
            top_neg_idx = np.argsort(weights)[:top_n]
            top_positive_words = [
                {"word": full_vocab[i], "score": round(float(weights[i]), 4)} for i in top_pos_idx
            ]
            top_negative_words = [
                {"word": full_vocab[i], "score": round(float(abs(weights[i])), 4)} for i in top_neg_idx
            ]
            return top_positive_words, top_negative_words

        return [], []
    except Exception as e:
        log(f"Warning: Could not extract word weights: {e}")
        return [], []


# --------------------------------------------------------------------------
# 7. Comprehensive Error Analysis on Test Set Predictions
# --------------------------------------------------------------------------
def conduct_error_analysis(model, test_df, n_examples=8):
    """
    Performs empirical error analysis on the test set predictions:
    Calculates total, correct, incorrect, error rate, FP, FN, negation breakdown,
    length bucket error rates, prediction confidence, and extracts annotated examples.
    """
    preds = model.transform(test_df).cache()

    get_prob_udf = F.udf(lambda v, p: float(v[int(p)]), DoubleType())
    preds_analysis = preds.withColumn(
        "confidence", get_prob_udf(F.col("probability"), F.col("prediction"))
    )
    preds_analysis = preds_analysis.withColumn(
        "error_type",
        F.when((F.col("label") == 1.0) & (F.col("prediction") == 1.0), "TP")
        .when((F.col("label") == 0.0) & (F.col("prediction") == 0.0), "TN")
        .when((F.col("label") == 0.0) & (F.col("prediction") == 1.0), "FP")
        .otherwise("FN"),
    )
    preds_analysis = preds_analysis.withColumn(
        "is_correct", F.col("label") == F.col("prediction")
    ).cache()

    total_samples = preds_analysis.count()
    correct_count = preds_analysis.filter(F.col("is_correct") == True).count()
    incorrect_count = preds_analysis.filter(F.col("is_correct") == False).count()
    fp_count = preds_analysis.filter(F.col("error_type") == "FP").count()
    fn_count = preds_analysis.filter(F.col("error_type") == "FN").count()
    error_rate = round(incorrect_count / total_samples, 4) if total_samples > 0 else 0.0

    # Negation pattern analysis
    negation_pattern = r"\b(not|no|never|hardly|barely|scarcely|neither|nor|n't|without|none|nothing)\b"
    preds_analysis = preds_analysis.withColumn(
        "has_negation", F.col("clean_text").rlike(negation_pattern)
    )

    total_with_negation = preds_analysis.filter(F.col("has_negation") == True).count()
    errors_with_negation = preds_analysis.filter(
        (F.col("is_correct") == False) & (F.col("has_negation") == True)
    ).count()
    negation_error_rate = (
        round(errors_with_negation / total_with_negation, 4)
        if total_with_negation > 0
        else 0.0
    )
    overall_negation_pct = round(total_with_negation / total_samples * 100, 2)
    error_negation_pct = round(errors_with_negation / incorrect_count * 100, 2)

    # Length bucket error breakdown
    preds_analysis = preds_analysis.withColumn(
        "length_bucket",
        F.when(F.col("review_length") < 15, "Short (<15 words)")
        .when(F.col("review_length") <= 50, "Medium (15-50 words)")
        .otherwise("Long (>50 words)"),
    )

    length_stats = (
        preds_analysis.groupBy("length_bucket")
        .agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("is_correct") == False, 1).otherwise(0)).alias("errors"),
            F.avg("confidence").alias("avg_confidence"),
        )
        .collect()
    )

    length_error_breakdown = []
    for row in length_stats:
        tot = row["total"]
        err = row["errors"]
        length_error_breakdown.append(
            {
                "bucket": row["length_bucket"],
                "total": tot,
                "errors": err,
                "error_rate": round(err / tot, 4) if tot > 0 else 0.0,
                "avg_confidence": round(row["avg_confidence"], 4),
            }
        )

    # Confidence comparison: Correct vs Misclassified
    avg_conf_correct = (
        preds_analysis.filter(F.col("is_correct") == True)
        .select(F.avg("confidence"))
        .first()[0]
    )
    avg_conf_wrong = (
        preds_analysis.filter(F.col("is_correct") == False)
        .select(F.avg("confidence"))
        .first()[0]
    )

    # Misclassified examples
    wrong_rows = (
        preds_analysis.filter(F.col("is_correct") == False)
        .select(
            "review",
            "label",
            "prediction",
            "review_length",
            "confidence",
            "error_type",
            "clean_text",
        )
        .orderBy(F.rand(SEED))
        .limit(n_examples)
        .collect()
    )

    examples = []
    for r in wrong_rows:
        txt = r["review"]
        has_neg = bool(re.search(negation_pattern, r["clean_text"]))
        apparent_cause = (
            "Negation / Polarity Inversion"
            if has_neg
            else ("Short / Sparse Context" if r["review_length"] < 15 else "Mixed Sentiment / Sarcasm")
        )
        examples.append(
            {
                "review": (txt[:250] + "...") if len(txt) > 250 else txt,
                "true_sentiment": "Positive" if r["label"] == 1.0 else "Negative",
                "predicted_sentiment": "Positive" if r["prediction"] == 1.0 else "Negative",
                "confidence": round(float(r["confidence"]), 4),
                "length_words": int(r["review_length"]),
                "error_type": r["error_type"],
                "apparent_cause": apparent_cause,
            }
        )

    preds_analysis.unpersist()
    preds.unpersist()

    return {
        "summary": {
            "total_test_samples": total_samples,
            "correct_predictions": correct_count,
            "incorrect_predictions": incorrect_count,
            "error_rate": error_rate,
            "false_positives": fp_count,
            "false_negatives": fn_count,
            "avg_confidence_correct": round(float(avg_conf_correct), 4),
            "avg_confidence_incorrect": round(float(avg_conf_wrong), 4),
        },
        "negation_analysis": {
            "total_reviews_with_negation": total_with_negation,
            "overall_negation_pct": overall_negation_pct,
            "error_reviews_with_negation": errors_with_negation,
            "error_negation_pct": error_negation_pct,
            "negation_error_rate": negation_error_rate,
        },
        "length_error_breakdown": length_error_breakdown,
        "misclassified_examples": examples,
    }


# --------------------------------------------------------------------------
# 8. Business Insights Generation
# --------------------------------------------------------------------------
def generate_business_insights(df, best_metrics, top_pos, top_neg):
    total = df.count()
    neg = df.filter(F.col("label") == 0.0).count()
    pos = total - neg
    neg_rate = round(neg / total * 100, 2)
    pos_rate = round(pos / total * 100, 2)

    avg_len_neg = df.filter(F.col("label") == 0.0).select(F.avg("review_length")).first()[0]
    avg_len_pos = df.filter(F.col("label") == 1.0).select(F.avg("review_length")).first()[0]
    avg_len_overall = df.select(F.avg("review_length")).first()[0]

    return {
        "overall_sentiment": {
            "total_reviews_analyzed": total,
            "positive_reviews_count": pos,
            "negative_reviews_count": neg,
            "percent_positive": pos_rate,
            "percent_negative": neg_rate,
        },
        "rating_vs_sentiment": {
            "dataset_structure": "Amazon Polarity (Stanford SNAP) provides binary polarity labels (0=Negative [1-2 Stars], 1=Positive [4-5 Stars]). Neutral 3-star reviews are excluded to maximize signal-to-noise ratio.",
            "average_rating_note": "Direct 1-5 numeric rating field is abstracted as binary polarity ground truth in this benchmark.",
        },
        "review_characteristics": {
            "avg_words_overall": round(float(avg_len_overall), 2),
            "avg_words_positive_review": round(float(avg_len_pos), 2),
            "avg_words_negative_review": round(float(avg_len_neg), 2),
            "length_difference_words": round(float(avg_len_neg - avg_len_pos), 2),
        },
        "important_terms": {
            "top_positive_terms": top_pos,
            "top_negative_terms": top_neg,
        },
        "model_summary": {
            "best_model_name": best_metrics["name"],
            "best_model_accuracy": best_metrics["accuracy"],
            "best_model_f1": best_metrics["f1"],
            "best_model_roc_auc": best_metrics["roc_auc"],
        },
    }


# --------------------------------------------------------------------------
# 9. Presentation-Quality Visualizations
# --------------------------------------------------------------------------
def generate_presentation_figures(all_metrics, best_metrics, eda, top_pos, top_neg):
    """
    Generates 9 presentation-ready charts and saves them to `results/figures/`.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})

    model_names = [f"{m['name']}\n({m['feature_representation']})" for m in all_metrics]

    # 1. Model Accuracy Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        model_names, [m["accuracy"] for m in all_metrics], color=["#2b5c8f", "#4682b4", "#90afc7"], width=0.55
    )
    plt.title("Model Accuracy Comparison", fontsize=14, pad=15, fontweight="bold")
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0, 1.0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_accuracy_comparison.png"), dpi=300)
    plt.close()

    # 2. Model F1 Score Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        model_names, [m["f1"] for m in all_metrics], color=["#1b7837", "#7fbf7b", "#d9f0d3"], width=0.55
    )
    plt.title("Model Weighted F1 Score Comparison", fontsize=14, pad=15, fontweight="bold")
    plt.ylabel("F1 Score", fontsize=12)
    plt.ylim(0, 1.0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_f1_comparison.png"), dpi=300)
    plt.close()

    # 3. Model ROC-AUC Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        model_names, [m["roc_auc"] for m in all_metrics], color=["#762a83", "#af8dc3", "#e7d4e8"], width=0.55
    )
    plt.title("Model ROC-AUC Comparison", fontsize=14, pad=15, fontweight="bold")
    plt.ylabel("ROC-AUC", fontsize=12)
    plt.ylim(0, 1.0)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_roc_auc_comparison.png"), dpi=300)
    plt.close()

    # 4. Training Time Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        model_names, [m["train_time_sec"] for m in all_metrics], color=["#d73027", "#fc8d59", "#fee08b"], width=0.55
    )
    plt.title("Model Training Time Comparison (Seconds - Log Scale)", fontsize=14, pad=15, fontweight="bold")
    plt.ylabel("Training Time (s) [Log Scale]", fontsize=12)
    plt.yscale("log")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval * 1.15, f"{yval:.2f}s", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_train_time_comparison.png"), dpi=300)
    plt.close()

    # 5. Overall Sentiment Distribution
    pos_cnt = eda["class_distribution"]["positive"]
    neg_cnt = eda["class_distribution"]["negative"]
    plt.figure(figsize=(7, 7))
    plt.pie(
        [pos_cnt, neg_cnt],
        labels=[f"Positive ({pos_cnt:,})", f"Negative ({neg_cnt:,})"],
        autopct="%1.1f%%",
        colors=["#2ca02c", "#d62728"],
        startangle=140,
        explode=(0.04, 0),
        textprops={"fontsize": 12, "fontweight": "bold"},
    )
    plt.title(f"Amazon Product Reviews Sentiment Distribution (N={eda['total_rows']:,})", fontsize=14, pad=20, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sentiment_distribution.png"), dpi=300)
    plt.close()

    # 6. Confusion Matrix for Best Model (Naive Bayes)
    cm = best_metrics["confusion_matrix"]
    cm_matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(
        cm_matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted Negative", "Predicted Positive"],
        yticklabels=["Actual Negative", "Actual Positive"],
        annot_kws={"size": 14, "fontweight": "bold"},
    )
    plt.title(f"Confusion Matrix: {best_metrics['name']} (Best Model)", fontsize=14, pad=15, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrix_best_model.png"), dpi=300)
    plt.close()

    # 7. ROC Curve for Final Model
    fpr_list = [p["fpr"] for p in best_metrics["roc_curve"]]
    tpr_list = [p["tpr"] for p in best_metrics["roc_curve"]]
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_list, tpr_list, color="#1f77b4", lw=2.5, label=f"{best_metrics['name']} (AUC = {best_metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, pad=15, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curve_best_model.png"), dpi=300)
    plt.close()

    # 8. Top Positive Terms
    if top_pos:
        plt.figure(figsize=(9, 6))
        words_pos = [w["word"] for w in top_pos][::-1]
        scores_pos = [w["score"] for w in top_pos][::-1]
        plt.barh(words_pos, scores_pos, color="#2ca02c", height=0.65)
        plt.xlabel("Log-Odds Ratio (Positive Association)", fontsize=12)
        plt.title(f"Top {len(top_pos)} Indicative Positive Review Terms ({best_metrics['name']})", fontsize=14, pad=15, fontweight="bold")
        for i, v in enumerate(scores_pos):
            plt.text(v + 0.05, i, f"+{v:.2f}", va="center", fontweight="bold", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "top_positive_terms.png"), dpi=300)
        plt.close()

    # 9. Top Negative Terms
    if top_neg:
        plt.figure(figsize=(9, 6))
        words_neg = [w["word"] for w in top_neg][::-1]
        scores_neg = [w["score"] for w in top_neg][::-1]
        plt.barh(words_neg, scores_neg, color="#d62728", height=0.65)
        plt.xlabel("Log-Odds Magnitude (Negative Association)", fontsize=12)
        plt.title(f"Top {len(top_neg)} Indicative Negative Review Terms ({best_metrics['name']})", fontsize=14, pad=15, fontweight="bold")
        for i, v in enumerate(scores_neg):
            plt.text(v + 0.05, i, f"-{v:.2f}", va="center", fontweight="bold", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "top_negative_terms.png"), dpi=300)
        plt.close()

    log(f"All 9 presentation figures saved to {FIGURES_DIR}/")


# --------------------------------------------------------------------------
# Main Orchestration
# --------------------------------------------------------------------------
def main():
    spark = get_spark()
    raw_df = load_dataset(spark)
    df, eda = clean_and_explore(raw_df)
    num_partitions = raw_df.rdd.getNumPartitions()

    # Scaling proof using raw cleaned text
    scaling_source = raw_df.withColumn(
        "clean_text", F.regexp_replace(F.lower(F.col("review")), r"[^a-z0-9\s]", " ")
    )
    scaling_source = scaling_source.withColumn(
        "clean_text", F.regexp_replace("clean_text", r"\s+", " ")
    )
    scaling_demo = run_scaling_demo(scaling_source)

    # Train / Test split
    train_df, test_df = df.randomSplit([1 - TEST_FRACTION, TEST_FRACTION], seed=SEED)
    train_df, test_df = train_df.cache(), test_df.cache()
    log(f"Train rows: {train_df.count()}, Test rows: {test_df.count()}")

    # Evaluated Model Architectures
    model_configs = [
        {
            "name": "Naive Bayes",
            "feature_rep": "Unigram TF-IDF",
            "classifier": NaiveBayes(labelCol="label", featuresCol="features", modelType="multinomial"),
            "feature_stages": build_unigram_feature_stages(),
        },
        {
            "name": "Logistic Regression",
            "feature_rep": "Unigram + Bigram TF-IDF",
            "classifier": LogisticRegression(labelCol="label", featuresCol="features", maxIter=20),
            "feature_stages": build_feature_stages(),
        },
        {
            "name": "Random Forest",
            "feature_rep": "Unigram + Bigram TF-IDF",
            "classifier": RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=60, maxDepth=12),
            "feature_stages": build_feature_stages(),
        },
    ]

    all_metrics = []
    fitted_models = {}
    for cfg in model_configs:
        metrics, model = train_and_evaluate(
            cfg["name"], cfg["classifier"], cfg["feature_rep"], cfg["feature_stages"], train_df, test_df
        )
        all_metrics.append(metrics)
        fitted_models[cfg["name"]] = model

    # Select Best Model strictly on F1 & Accuracy
    best = max(all_metrics, key=lambda m: m["f1"])
    best_model = fitted_models[best["name"]]
    log(f"Final Selected Model by F1/ROC-AUC: {best['name']} ({best['feature_representation']})")

    # 1. Extract Top Indicative Words
    top_pos, top_neg = extract_important_words(best_model, top_n=15)

    # 2. Comprehensive Error Analysis
    error_report = conduct_error_analysis(best_model, test_df, n_examples=8)

    # 3. Business Insights
    insights = generate_business_insights(df, best, top_pos, top_neg)

    # 4. Aspect-Based Sentiment Analysis (ABSA)
    from backend.services.absa import run_spark_absa, generate_absa_figures
    log("Running Aspect-Based Sentiment Analysis (ABSA) on Spark DataFrame...")
    absa_df, absa_summary = run_spark_absa(spark, df, best_model)
    log(f"ABSA completed: {absa_summary['total_aspect_mentions']:,} total aspect mentions found.")
    generate_absa_figures(absa_summary["aspects"], FIGURES_DIR)

    # Save aspect sentiment sample CSV
    csv_path = os.path.join(RESULTS_DIR, "aspect_sentiment.csv")
    sample_pd = absa_df.limit(2000).toPandas()
    sample_pd.to_csv(csv_path, index=False)
    log(f"Aspect sentiment sample CSV saved to {csv_path}")

    # 5. Topic & Customer Complaint Mining (Negative Reviews)
    from backend.services.complaint_mining import (
        mine_negative_complaints_spark,
        run_spark_lda_topics,
        generate_complaint_figures,
    )
    log("Running Topic & Complaint Mining on negative reviews...")
    neg_reviews_df = df.filter(F.col("label") == 0.0)
    total_neg_count = neg_reviews_df.count()
    complaints_data = mine_negative_complaints_spark(spark, neg_reviews_df, total_neg_count=total_neg_count, top_n=25)
    log(f"Complaint Mining completed: {len(complaints_data['top_complaints'])} quality complaint phrases extracted.")

    log("Running Spark MLlib LDA Topic Discovery (k=5)...")
    lda_topics = run_spark_lda_topics(spark, neg_reviews_df, k=5, max_iter=15)
    generate_complaint_figures(complaints_data, lda_topics, FIGURES_DIR)

    # Save complaints and topics CSVs
    import pandas as pd
    pd.DataFrame(complaints_data["top_complaints"]).to_csv(os.path.join(RESULTS_DIR, "complaints.csv"), index=False)
    pd.DataFrame([
        {
            "topic_id": t["topic_id"],
            "label": t["label"],
            "top_terms": t["top_words_str"],
            "interpretation": t["statistical_interpretation"],
        }
        for t in lda_topics
    ]).to_csv(os.path.join(RESULTS_DIR, "topics.csv"), index=False)
    log("Saved complaints.csv and topics.csv.")

    # 6. Generate Baseline Presentation Figures
    generate_presentation_figures(all_metrics, best, eda, top_pos, top_neg)

    # 7. Save Winning Model for Live Inference (api.py)
    best_model.write().overwrite().save(MODEL_DIR)
    with open(os.path.join(RESULTS_DIR, "best_model_name.txt"), "w") as f:
        f.write(best["name"])

    # 8. Save Consolidated Results JSON
    output = {
        "eda": eda,
        "models": all_metrics,
        "best_model": best["name"],
        "error_analysis": error_report,
        "business_insights": insights,
        "absa": absa_summary,
        "complaint_mining": complaints_data,
        "lda_topics": lda_topics,
        "dataset_info": {
            "source": "Amazon Polarity (HuggingFace datasets)",
            "sample_size": eda["total_rows"],
            "num_partitions": num_partitions,
            "columns": ["label", "title", "review"],
            "target_variable": "label (0=negative, 1=positive)",
        },
        "scaling_demo": scaling_demo,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(output, f, indent=2)

    log(f"Done. All metrics, error analysis, and ABSA saved to {RESULTS_DIR}/results.json")
    log(f"Best model saved to {MODEL_DIR}/ for live API serving.")
    spark.stop()


if __name__ == "__main__":
    main()
