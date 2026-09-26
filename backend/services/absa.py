"""
================================================================================
 Aspect-Based Sentiment Analysis (ABSA) Module
 ---------------------------------------------
 Part of: Scalable Product Review Intelligence Using Apache Spark & Machine Learning

 This module extends traditional review-level polarity classification by:
   1. Detecting which product aspects are mentioned in a review using
      configurable domain dictionaries.
   2. Extracting the specific clause/sentence context around each mention.
   3. Running the trained Spark MLlib pipeline model on the aspect context
      to compute aspect-level sentiment and confidence.
   4. Aggregating distributed aspect analytics (mentions, positive/negative rates)
      via Apache Spark DataFrames.
   5. Identifying potential customer pain points based on negative feedback rates.
================================================================================
"""

import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    StructType,
    StructField,
    StringType,
    DoubleType,
)

# --------------------------------------------------------------------------
# 1. Configurable Aspect Dictionaries
# --------------------------------------------------------------------------
ASPECT_DICTIONARY = {
    "Battery": [
        "battery",
        "battery life",
        "charging",
        "charger",
        "charge",
        "power",
        "drains",
        "charging speed",
    ],
    "Build Quality": [
        "build",
        "quality",
        "material",
        "durable",
        "durability",
        "plastic",
        "metal",
        "design",
        "construction",
    ],
    "Price / Value": [
        "price",
        "cost",
        "expensive",
        "cheap",
        "affordable",
        "value",
        "worth",
        "money",
    ],
    "Customer Support": [
        "support",
        "customer service",
        "service",
        "representative",
        "agent",
        "help",
        "response",
        "refund",
    ],
    "Delivery / Packaging": [
        "delivery",
        "shipping",
        "package",
        "packaging",
        "box",
        "courier",
        "arrived",
        "delivery time",
    ],
}

# Compile aspect regex patterns for fast, boundary-aware matching
COMPILED_ASPECT_PATTERNS = {}
for aspect, keywords in ASPECT_DICTIONARY.items():
    # Sort by length descending so multi-word phrases match first
    sorted_kw = sorted(keywords, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(k) for k in sorted_kw) + r")\b"
    COMPILED_ASPECT_PATTERNS[aspect] = re.compile(pattern, re.IGNORECASE)


# --------------------------------------------------------------------------
# 2. Context Extraction (Sentence / Clause Level)
# --------------------------------------------------------------------------
CLAUSE_SPLIT_REGEX = re.compile(
    r"[.!?;\n]+|(?:,\s+)?\b(?:but|however|although|though|whereas|while|yet)\b",
    re.IGNORECASE,
)


def extract_aspect_contexts(text: str):
    """
    Extracts each mentioned aspect and its associated sentence/clause context.
    Returns a list of dicts: [{'aspect': str, 'context': str}, ...]
    """
    if not text or not isinstance(text, str):
        return []

    # Split into candidate clauses / sentences
    raw_clauses = CLAUSE_SPLIT_REGEX.split(text)
    clauses = [c.strip() for c in raw_clauses if c and len(c.strip()) > 3]
    if not clauses:
        clauses = [text.strip()]

    aspect_contexts = {}

    for clause in clauses:
        for aspect, pattern in COMPILED_ASPECT_PATTERNS.items():
            if pattern.search(clause):
                if aspect not in aspect_contexts:
                    aspect_contexts[aspect] = []
                aspect_contexts[aspect].append(clause)

    # Combine clauses for the same aspect within the review
    results = []
    for aspect, matched_clauses in aspect_contexts.items():
        combined_context = " ... ".join(matched_clauses)
        results.append({"aspect": aspect, "context": combined_context})

    return results


# --------------------------------------------------------------------------
# 3. Spark ABSA Processing Pipeline
# --------------------------------------------------------------------------
def get_aspect_schema():
    return ArrayType(
        StructType(
            [
                StructField("aspect", StringType(), False),
                StructField("context", StringType(), False),
            ]
        )
    )


def run_spark_absa(spark, df, fitted_model, sample_limit=None):
    """
    Runs scalable Aspect-Based Sentiment Analysis on a Spark DataFrame.

    1. Identifies aspects and sentence contexts using distributed transformations.
    2. Explodes multiple aspect matches into dedicated rows.
    3. Transforms aspect context through the trained Spark MLlib pipeline model.
    4. Aggregates metrics per aspect (mentions, positive %, negative %).
    5. Flags potential customer pain points.
    """
    aspect_schema = get_aspect_schema()

    def _detect_udf(txt):
        matches = extract_aspect_contexts(txt)
        return [(m["aspect"], m["context"]) for m in matches]

    detect_spark_udf = F.udf(_detect_udf, aspect_schema)

    input_df = df
    if sample_limit:
        input_df = input_df.limit(sample_limit)

    # 1. Detect aspects per review
    with_aspects = input_df.withColumn("aspect_list", detect_spark_udf(F.col("review")))

    # 2. Explode so each (review, aspect) is a structured row
    exploded = (
        with_aspects.filter(F.size(F.col("aspect_list")) > 0)
        .select(
            F.col("review"),
            F.explode(F.col("aspect_list")).alias("aspect_item"),
        )
        .select(
            F.col("review"),
            F.col("aspect_item.aspect").alias("aspect"),
            F.col("aspect_item.context").alias("context"),
        )
    )

    # 3. Clean aspect context for the ML pipeline
    exploded = exploded.withColumn(
        "clean_text",
        F.regexp_replace(F.lower(F.col("context")), r"[^a-z0-9\s]", " "),
    )
    exploded = exploded.withColumn(
        "clean_text", F.regexp_replace("clean_text", r"\s+", " ")
    )
    exploded = exploded.filter(F.length(F.trim(F.col("clean_text"))) > 0)

    # 4. Predict sentiment using the fitted Spark MLlib PipelineModel
    preds = fitted_model.transform(exploded).cache()

    get_confidence_udf = F.udf(lambda v, p: float(v[int(p)]), DoubleType())
    preds = preds.withColumn(
        "sentiment",
        F.when(F.col("prediction") == 1.0, "Positive").otherwise("Negative"),
    ).withColumn(
        "sentiment_score",
        get_confidence_udf(F.col("probability"), F.col("prediction")),
    )

    absa_df = preds.select(
        "review", "aspect", "context", "sentiment", "sentiment_score"
    )

    # 5. Distributed Aggregation per Aspect
    stats_rows = (
        absa_df.groupBy("aspect")
        .agg(
            F.count("*").alias("mentions"),
            F.sum(F.when(F.col("sentiment") == "Positive", 1).otherwise(0)).alias(
                "positive_mentions"
            ),
            F.sum(F.when(F.col("sentiment") == "Negative", 1).otherwise(0)).alias(
                "negative_mentions"
            ),
            F.avg("sentiment_score").alias("avg_confidence"),
        )
        .withColumn(
            "positive_pct",
            F.round(F.col("positive_mentions") / F.col("mentions") * 100, 2),
        )
        .withColumn(
            "negative_pct",
            F.round(F.col("negative_mentions") / F.col("mentions") * 100, 2),
        )
        .withColumn(
            "avg_confidence",
            F.round(F.col("avg_confidence") * 100, 2),
        )
        .orderBy(F.desc("mentions"))
        .collect()
    )

    total_aspect_mentions = sum(r["mentions"] for r in stats_rows)

    aspect_analytics = []
    pain_points = []

    for r in stats_rows:
        mentions = int(r["mentions"])
        pos_cnt = int(r["positive_mentions"])
        neg_cnt = int(r["negative_mentions"])
        neg_pct = float(r["negative_pct"])
        pos_pct = float(r["positive_pct"])
        avg_conf = float(r["avg_confidence"])

        # Determine if it's a customer pain point
        is_pain = neg_pct >= 40.0 and mentions >= 50
        status = (
            "High Negative Feedback"
            if is_pain
            else ("Moderate Concern" if neg_pct >= 28.0 else "Healthy / Positive")
        )

        item = {
            "aspect": r["aspect"],
            "mentions": mentions,
            "positive_mentions": pos_cnt,
            "negative_mentions": neg_cnt,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "avg_confidence": avg_conf,
            "is_pain_point": is_pain,
            "status": status,
            "classification": "Potential Customer Pain Point" if is_pain else "Standard Aspect",
        }
        aspect_analytics.append(item)
        if is_pain:
            pain_points.append(
                {
                    "aspect": r["aspect"],
                    "negative_pct": neg_pct,
                    "mentions": mentions,
                    "status": "High Negative Feedback",
                    "note": f"{neg_pct}% negative feedback across {mentions:,} mentions.",
                }
            )

    # Sort pain points by negative percentage descending
    pain_points.sort(key=lambda x: x["negative_pct"], reverse=True)

    # Collect sample predictions for UI / inspection
    sample_rows = (
        absa_df.select("review", "aspect", "context", "sentiment", "sentiment_score")
        .limit(20)
        .collect()
    )
    samples = [
        {
            "review": (r["review"][:180] + "...") if len(r["review"]) > 180 else r["review"],
            "aspect": r["aspect"],
            "context": r["context"],
            "sentiment": r["sentiment"],
            "confidence": round(float(r["sentiment_score"]) * 100, 1),
        }
        for r in sample_rows
    ]

    absa_summary = {
        "aspects": aspect_analytics,
        "pain_points": pain_points,
        "total_aspect_mentions": total_aspect_mentions,
        "sample_predictions": samples,
    }

    return absa_df, absa_summary


# --------------------------------------------------------------------------
# 4. Generate Visualizations
# --------------------------------------------------------------------------
def generate_absa_figures(aspect_analytics, figures_dir):
    """
    Generates presentation-ready ABSA figures:
      1. aspect_sentiment_distribution.png
      2. aspect_negative_rate.png
      3. aspect_mentions.png
    """
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})

    aspect_names = [a["aspect"] for a in aspect_analytics]
    pos_pcts = [a["positive_pct"] for a in aspect_analytics]
    neg_pcts = [a["negative_pct"] for a in aspect_analytics]
    mentions = [a["mentions"] for a in aspect_analytics]

    # 1. Aspect Sentiment Distribution (Stacked Bar)
    plt.figure(figsize=(9, 5.5))
    bar_width = 0.55
    indices = range(len(aspect_names))

    plt.bar(indices, pos_pcts, width=bar_width, label="Positive %", color="#2ca02c")
    plt.bar(indices, neg_pcts, width=bar_width, bottom=pos_pcts, label="Negative %", color="#d62728")

    plt.xticks(indices, aspect_names, fontweight="bold", rotation=10)
    plt.ylabel("Sentiment Percentage (%)", fontsize=12)
    plt.title("Aspect-Based Sentiment Distribution (ABSA)", fontsize=14, pad=15, fontweight="bold")
    plt.ylim(0, 105)
    plt.legend(loc="upper right", frameon=True)

    for i in indices:
        plt.text(i, pos_pcts[i] / 2, f"{pos_pcts[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold")
        plt.text(i, pos_pcts[i] + neg_pcts[i] / 2, f"{neg_pcts[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "aspect_sentiment_distribution.png"), dpi=300)
    plt.close()

    # 2. Aspect Negative Rate (Pain Points)
    sorted_by_neg = sorted(aspect_analytics, key=lambda x: x["negative_pct"], reverse=True)
    s_names = [a["aspect"] for a in sorted_by_neg][::-1]
    s_negs = [a["negative_pct"] for a in sorted_by_neg][::-1]
    colors = ["#d62728" if a["is_pain_point"] else "#f5a623" for a in sorted_by_neg][::-1]

    plt.figure(figsize=(9, 5))
    plt.barh(s_names, s_negs, color=colors, height=0.55)
    plt.axvline(x=40.0, color="#b2182b", linestyle="--", linewidth=1.5, label="Pain Point Threshold (40%)")
    plt.xlabel("Negative Sentiment Rate (%)", fontsize=12)
    plt.title("Aspect Negative Sentiment Rate & Customer Pain Points", fontsize=14, pad=15, fontweight="bold")
    plt.xlim(0, max(max(s_negs) + 15, 60))
    plt.legend(loc="lower right")

    for i, v in enumerate(s_negs):
        tag = " [Pain Point]" if sorted_by_neg[::-1][i]["is_pain_point"] else ""
        plt.text(v + 1.0, i, f"{v:.1f}%{tag}", va="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "aspect_negative_rate.png"), dpi=300)
    plt.close()

    # 3. Aspect Mention Volume
    plt.figure(figsize=(9, 5))
    bars = plt.bar(aspect_names, mentions, color="#1f77b4", width=0.55)
    plt.title("Total Mentions by Product Aspect", fontsize=14, pad=15, fontweight="bold")
    plt.ylabel("Mention Count", fontsize=12)
    plt.xticks(rotation=10, fontweight="bold")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + (max(mentions) * 0.015), f"{yval:,}", ha="center", va="bottom", fontweight="bold")

    plt.ylim(0, max(mentions) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "aspect_mentions.png"), dpi=300)
    plt.close()


# --------------------------------------------------------------------------
# 5. Live Single-Review Prediction Helper (Used by api.py)
# --------------------------------------------------------------------------
def predict_aspects_for_review(review_text: str, spark, model, clean_fn, override_fn=None):
    """
    Identifies aspects in a single review and computes aspect-level sentiment
    using the fitted Spark pipeline model.
    """
    aspect_matches = extract_aspect_contexts(review_text)
    if not aspect_matches:
        return []

    # Prepare DataFrame with all aspect clauses
    rows = []
    for item in aspect_matches:
        ctx = item["context"]
        cleaned_ctx = clean_fn(ctx)
        if cleaned_ctx:
            rows.append((item["aspect"], ctx, cleaned_ctx))

    if not rows:
        return []

    ctx_df = spark.createDataFrame(rows, schema=["aspect", "context", "clean_text"])
    preds = model.transform(ctx_df).select("aspect", "context", "prediction", "probability").collect()

    results = []
    for r in preds:
        aspect = r["aspect"]
        ctx = r["context"]
        pred = int(r["prediction"])
        probs = r["probability"].toArray().tolist()

        if override_fn:
            override = override_fn(ctx)
            if override is not None and override != pred:
                pred = override
                probs = [0.02, 0.98] if pred == 1 else [0.98, 0.02]

        conf = round(probs[pred] * 100, 1)
        results.append(
            {
                "aspect": aspect,
                "sentiment": "POSITIVE" if pred == 1 else "NEGATIVE",
                "confidence": conf,
                "context": ctx,
            }
        )

    return results
