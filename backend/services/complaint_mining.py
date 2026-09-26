"""
================================================================================
 Topic & Customer Complaint Mining Module
 ----------------------------------------
 Part of: Scalable Product Review Intelligence Using Apache Spark & Machine Learning

 This module analyzes customer dissatisfaction in negative reviews:
   1. Filters negative reviews (from existing sentiment classification).
   2. Performs distributed Tokenization, custom StopWords removal, and
      N-Gram feature extraction (Bigrams & Trigrams) via Spark ML.
   3. Applies complaint-quality filtering using domain complaint indicators.
   4. Ranks recurring customer complaints and computes frequency and review percentages.
   5. Fits an unsupervised Spark MLlib Latent Dirichlet Allocation (LDA) model
      to discover statistical latent complaint themes across negative reviews.
   6. Bridges mined complaints with the existing Aspect-Based Sentiment Analysis (ABSA).
   7. Generates publication-ready figures and exportable CSVs.
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
    StringType,
)
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    NGram,
    CountVectorizer,
)
from pyspark.ml.clustering import LDA

# --------------------------------------------------------------------------
# 1. Complaint Indicators & Noise Stopwords
# --------------------------------------------------------------------------
# Words signaling dissatisfaction, broken functionality, or bad experiences
COMPLAINT_INDICATORS = {
    "poor", "bad", "broken", "defective", "issue", "problem", "stopped",
    "failed", "damaged", "terrible", "disappointing", "useless", "waste",
    "expensive", "slow", "late", "missing", "never", "doesnt", "didnt",
    "cheap", "fail", "return", "refund", "crack", "cracked", "worst",
    "horrible", "junk", "quit", "died", "defect", "fault", "faulty",
    "error", "warning", "wrong", "leak", "leaking", "broke", "garbage",
    "awful", "dreadful", "ripoff", "unusable", "headache", "died", "crap",
    "rubbish", "horrid", "worthless", "pathetic", "nightmare"
}

# Domain-specific conversational fillers to suppress meaningless n-grams
DOMAIN_STOPWORDS = {
    "product", "item", "amazon", "one", "would", "get", "like", "also",
    "even", "bought", "buy", "really", "ordered", "order", "received",
    "use", "using", "used", "see", "say", "know", "much", "time", "day",
    "days", "back", "still", "well", "way", "give", "make", "think",
    "could", "got", "first", "two", "take", "go", "going", "many", "good",
    "great", "look", "looks", "looking", "put", "another", "find", "made",
    "ever", "thing", "things", "something", "anything", "nothing", "everything",
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "with", "about", "against", "between", "into", "through", "during", "before"
}

# Mapping keywords to ABSA product aspects
ASPECT_KEYWORD_MAP = {
    "Battery": ["battery", "charge", "charger", "charging", "power", "drain", "drains"],
    "Build Quality": ["build", "quality", "material", "plastic", "metal", "durable", "broken", "cracked", "screen", "hinge", "case"],
    "Price / Value": ["price", "cost", "money", "worth", "expensive", "cheap", "waste", "value", "dollar", "bucks"],
    "Customer Support": ["service", "support", "representative", "response", "agent", "refund", "return", "warranty", "help", "email"],
    "Delivery / Packaging": ["delivery", "shipping", "package", "packaging", "box", "arrived", "courier", "shipped", "mail", "late"],
}


def map_complaint_to_aspect(phrase: str) -> str:
    """Classifies an n-gram complaint into an ABSA aspect if keywords match."""
    p_lower = phrase.lower()
    for aspect, keywords in ASPECT_KEYWORD_MAP.items():
        if any(kw in p_lower for kw in keywords):
            return aspect
    return "General Dissatisfaction"


# --------------------------------------------------------------------------
# 2. Distributed Spark N-Gram Complaint Mining
# --------------------------------------------------------------------------
def mine_negative_complaints_spark(spark, negative_df, total_neg_count=None, top_n=25):
    """
    Extracts, ranks, and categorizes recurring customer complaints from negative reviews
    using native Apache Spark DataFrame transformations.
    """
    if total_neg_count is None or total_neg_count == 0:
        total_neg_count = negative_df.count()

    if total_neg_count == 0:
        return {"top_complaints": [], "top_bigrams": [], "top_trigrams": [], "aspect_complaints": {}}

    # 1. Clean and normalize text (strip punctuation and symbols)
    cleaned_df = negative_df.withColumn(
        "clean_norm",
        F.regexp_replace(F.lower(F.col("clean_text")), r"[^a-z0-9\s]", " "),
    ).withColumn("clean_norm", F.regexp_replace("clean_norm", r"\s+", " "))

    tokenizer = Tokenizer(inputCol="clean_norm", outputCol="raw_tokens")
    tokenized_df = tokenizer.transform(cleaned_df)

    # 2. Filter default and domain stop words
    remover = StopWordsRemover(inputCol="raw_tokens", outputCol="filtered_tokens")
    all_stop = list(set(remover.getStopWords()).union(DOMAIN_STOPWORDS))
    remover.setStopWords(all_stop)
    tokens_clean_df = remover.transform(tokenized_df)

    # Filter out single/double character tokens (like '2', '-', 's')
    filter_short_udf = F.udf(lambda tokens: [t for t in tokens if len(t) > 2 and not t.isdigit()], ArrayType(StringType()))
    tokens_clean_df = tokens_clean_df.withColumn("filtered_tokens", filter_short_udf(F.col("filtered_tokens"))).cache()

    # 3. Generate Bigrams and Trigrams via Spark NGram
    bigram_gen = NGram(n=2, inputCol="filtered_tokens", outputCol="bigrams")
    trigram_gen = NGram(n=3, inputCol="filtered_tokens", outputCol="trigrams")

    bigrams_df = bigram_gen.transform(tokens_clean_df)
    ngrams_df = trigram_gen.transform(bigrams_df)

    # 4. Explode & aggregate Bigrams
    exploded_bi = (
        ngrams_df.select(F.explode(F.col("bigrams")).alias("phrase"))
        .filter(F.length(F.col("phrase")) > 4)
        .groupBy("phrase")
        .count()
        .orderBy(F.desc("count"))
    )

    # 5. Explode & aggregate Trigrams
    exploded_tri = (
        ngrams_df.select(F.explode(F.col("trigrams")).alias("phrase"))
        .filter(F.length(F.col("phrase")) > 6)
        .groupBy("phrase")
        .count()
        .orderBy(F.desc("count"))
    )

    raw_bi = exploded_bi.limit(200).collect()
    raw_tri = exploded_tri.limit(200).collect()

    # 6. Apply Complaint Relevance Filter
    def has_complaint_signal(phrase):
        words = phrase.lower().split()
        return any(w in COMPLAINT_INDICATORS for w in words)

    filtered_complaints = []
    top_bigrams = []
    top_trigrams = []

    # Process Bigrams
    for r in raw_bi:
        phrase = r["phrase"]
        cnt = int(r["count"])
        pct = round(cnt / total_neg_count * 100, 2)
        asp = map_complaint_to_aspect(phrase)
        item = {
            "phrase": phrase,
            "n_gram_type": "Bigram",
            "frequency": cnt,
            "percentage": pct,
            "aspect": asp,
        }
        top_bigrams.append(item)
        if has_complaint_signal(phrase):
            filtered_complaints.append(item)

    # Process Trigrams
    for r in raw_tri:
        phrase = r["phrase"]
        cnt = int(r["count"])
        pct = round(cnt / total_neg_count * 100, 2)
        asp = map_complaint_to_aspect(phrase)
        item = {
            "phrase": phrase,
            "n_gram_type": "Trigram",
            "frequency": cnt,
            "percentage": pct,
            "aspect": asp,
        }
        top_trigrams.append(item)
        if has_complaint_signal(phrase):
            filtered_complaints.append(item)

    # Sort all complaint-indicative n-grams by frequency
    filtered_complaints.sort(key=lambda x: x["frequency"], reverse=True)
    top_complaints = filtered_complaints[:top_n]
    top_bi_result = top_bigrams[:top_n]
    top_tri_result = top_trigrams[:top_n]

    # 7. Aspect -> Complaints Mapping
    aspect_complaints = {
        "Battery": [],
        "Build Quality": [],
        "Price / Value": [],
        "Customer Support": [],
        "Delivery / Packaging": [],
        "General Dissatisfaction": [],
    }

    for c in filtered_complaints:
        asp = c["aspect"]
        if asp in aspect_complaints and len(aspect_complaints[asp]) < 6:
            aspect_complaints[asp].append(c)

    tokens_clean_df.unpersist()

    return {
        "total_negative_reviews": total_neg_count,
        "top_complaints": top_complaints,
        "top_bigrams": top_bi_result,
        "top_trigrams": top_tri_result,
        "aspect_complaints": aspect_complaints,
    }


# --------------------------------------------------------------------------
# 3. Unsupervised Spark MLlib LDA Topic Modeling
# --------------------------------------------------------------------------
def run_spark_lda_topics(spark, negative_df, k=5, max_iter=15, vocab_size=3000):
    """
    Fits a Latent Dirichlet Allocation (LDA) model using Spark MLlib
    to discover statistical recurring themes across negative customer feedback.
    """
    # 1. Tokenize & Clean
    cleaned_df = negative_df.withColumn(
        "clean_norm",
        F.regexp_replace(F.lower(F.col("clean_text")), r"[^a-z0-9\s]", " "),
    ).withColumn("clean_norm", F.regexp_replace("clean_norm", r"\s+", " "))

    tokenizer = Tokenizer(inputCol="clean_norm", outputCol="raw_tokens")
    tokenized_df = tokenizer.transform(cleaned_df)

    remover = StopWordsRemover(inputCol="raw_tokens", outputCol="filtered_tokens")
    all_stop = list(set(remover.getStopWords()).union(DOMAIN_STOPWORDS))
    remover.setStopWords(all_stop)
    clean_tokens_df = remover.transform(tokenized_df)

    filter_short_udf = F.udf(lambda tokens: [t for t in tokens if len(t) > 2 and not t.isdigit()], ArrayType(StringType()))
    clean_tokens_df = clean_tokens_df.withColumn("filtered_tokens", filter_short_udf(F.col("filtered_tokens")))

    # 2. Count Vectorizer (Word Frequencies for LDA)
    cv = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="features",
        vocabSize=vocab_size,
        minDF=3.0,
    )
    cv_model = cv.fit(clean_tokens_df)
    lda_data = cv_model.transform(clean_tokens_df).select("features").cache()

    if lda_data.count() == 0:
        return []

    # 3. Fit Spark MLlib LDA Model
    lda = LDA(k=k, maxIter=max_iter, seed=42, featuresCol="features")
    lda_model = lda.fit(lda_data)

    vocab = cv_model.vocabulary
    topics_df = lda_model.describeTopics(maxTermsPerTopic=8)
    topic_rows = topics_df.collect()

    # Common semantic interpretation labels for academic presentation
    interpretation_hints = [
        "Product Durability & Build Issues",
        "Cost, Pricing & Value Assessment",
        "Delivery, Packaging & Shipping",
        "Battery, Power & Charging Functionality",
        "Customer Service, Returns & Support",
        "General Performance & Usability",
    ]

    lda_results = []
    for r in topic_rows:
        t_id = int(r["topic"])
        term_indices = r["termIndices"]
        term_weights = r["termWeights"]

        terms = [vocab[idx] for idx in term_indices if idx < len(vocab)]
        weights = [round(float(w), 4) for w in term_weights]

        top_words_with_weights = [
            {"term": t, "weight": w} for t, w in zip(terms, weights)
        ]

        hint = interpretation_hints[t_id % len(interpretation_hints)]

        lda_results.append(
            {
                "topic_id": t_id + 1,
                "label": f"Topic {t_id + 1}",
                "top_words": terms,
                "top_words_str": ", ".join(terms),
                "terms_with_weights": top_words_with_weights,
                "statistical_interpretation": hint,
                "note": "Statistical latent distribution; human interpretation suggested.",
            }
        )

    lda_data.unpersist()
    return lda_results


# --------------------------------------------------------------------------
# 4. Generate Publication-Quality Figures
# --------------------------------------------------------------------------
def generate_complaint_figures(complaints_data, lda_data, figures_dir):
    """
    Generates presentation-ready figures:
      1. top_complaints.png
      2. top_bigrams.png
      3. top_trigrams.png
      4. topic_distribution.png
    """
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 11})

    # 1. Top Complaints
    top_c = complaints_data.get("top_complaints", [])[:12]
    if top_c:
        phrases = [c["phrase"] for c in top_c][::-1]
        freqs = [c["frequency"] for c in top_c][::-1]
        pcts = [c["percentage"] for c in top_c][::-1]

        plt.figure(figsize=(9.5, 6))
        bars = plt.barh(phrases, freqs, color="#d62728", height=0.6)
        plt.xlabel("Occurrences in Negative Reviews", fontsize=12)
        plt.title("Top Recurring Customer Complaints (Filtered N-Grams)", fontsize=14, pad=15, fontweight="bold")

        for i, (bar, p) in enumerate(zip(bars, pcts)):
            w = bar.get_width()
            plt.text(w + (max(freqs) * 0.015), i, f"{w:,} ({p}%)", va="center", fontweight="bold", fontsize=10)

        plt.xlim(0, max(freqs) * 1.25)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "top_complaints.png"), dpi=300)
        plt.close()

    # 2. Top Bigrams
    top_bi = complaints_data.get("top_bigrams", [])[:10]
    if top_bi:
        bi_phrases = [c["phrase"] for c in top_bi][::-1]
        bi_freqs = [c["frequency"] for c in top_bi][::-1]

        plt.figure(figsize=(9, 5.5))
        bars = plt.barh(bi_phrases, bi_freqs, color="#e6550d", height=0.6)
        plt.xlabel("Frequency", fontsize=12)
        plt.title("Top 10 Negative Bigrams (2-Word Phrases)", fontsize=14, pad=15, fontweight="bold")

        for i, bar in enumerate(bars):
            w = bar.get_width()
            plt.text(w + (max(bi_freqs) * 0.015), i, f"{w:,}", va="center", fontweight="bold", fontsize=10)

        plt.xlim(0, max(bi_freqs) * 1.2)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "top_bigrams.png"), dpi=300)
        plt.close()

    # 3. Top Trigrams
    top_tri = complaints_data.get("top_trigrams", [])[:10]
    if top_tri:
        tri_phrases = [c["phrase"] for c in top_tri][::-1]
        tri_freqs = [c["frequency"] for c in top_tri][::-1]

        plt.figure(figsize=(9.5, 5.5))
        bars = plt.barh(tri_phrases, tri_freqs, color="#756bb1", height=0.6)
        plt.xlabel("Frequency", fontsize=12)
        plt.title("Top 10 Negative Trigrams (3-Word Phrases)", fontsize=14, pad=15, fontweight="bold")

        for i, bar in enumerate(bars):
            w = bar.get_width()
            plt.text(w + (max(tri_freqs) * 0.015), i, f"{w:,}", va="center", fontweight="bold", fontsize=10)

        plt.xlim(0, max(tri_freqs) * 1.2)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "top_trigrams.png"), dpi=300)
        plt.close()

    # 4. LDA Topic Distribution
    if lda_data:
        plt.figure(figsize=(10, 6))
        topic_labels = [f"Topic {t['topic_id']}\n({t['top_words'][0]}, {t['top_words'][1]}...)" for t in lda_data]
        top_weights_sum = [sum(w["weight"] for w in t["terms_with_weights"][:4]) for t in lda_data]

        bars = plt.bar(topic_labels, top_weights_sum, color="#3182bd", width=0.55)
        plt.ylabel("Cumulative Top-Terms Weight", fontsize=12)
        plt.title("Spark MLlib LDA Discovered Complaint Themes (k=5)", fontsize=14, pad=15, fontweight="bold")

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.005, f"{yval:.3f}", ha="center", va="bottom", fontweight="bold")

        plt.ylim(0, max(top_weights_sum) * 1.25)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "topic_distribution.png"), dpi=300)
        plt.close()


# --------------------------------------------------------------------------
# 5. Live Single-Review Complaint Matcher (Used by api.py)
# --------------------------------------------------------------------------
def extract_complaints_from_text(text: str, known_complaints_list=None):
    """
    Scans an individual review for complaint phrases and indicators.
    Returns matched complaints formatted with their associated aspect.
    """
    if not text or not isinstance(text, str):
        return []

    text_lower = text.lower()
    matches = []
    seen = set()

    # Check against known high-frequency mined complaints if provided
    if known_complaints_list:
        for c in known_complaints_list:
            phrase = c["phrase"]
            if phrase in text_lower and phrase not in seen:
                seen.add(phrase)
                matches.append(
                    {
                        "phrase": phrase,
                        "n_gram_type": c.get("n_gram_type", "Mined Phrase"),
                        "aspect": c.get("aspect", map_complaint_to_aspect(phrase)),
                    }
                )

    # Also detect direct complaint signal patterns (e.g., "stopped working", "waste money")
    heuristic_patterns = [
        ("stopped working", "Battery"),
        ("drains quickly", "Battery"),
        ("wont charge", "Battery"),
        ("poor quality", "Build Quality"),
        ("cheap plastic", "Build Quality"),
        ("fell apart", "Build Quality"),
        ("broke after", "Build Quality"),
        ("waste of money", "Price / Value"),
        ("not worth", "Price / Value"),
        ("over priced", "Price / Value"),
        ("customer service", "Customer Support"),
        ("never responded", "Customer Support"),
        ("no refund", "Customer Support"),
        ("damaged box", "Delivery / Packaging"),
        ("late delivery", "Delivery / Packaging"),
        ("arrived broken", "Delivery / Packaging"),
    ]

    for phrase, aspect in heuristic_patterns:
        if phrase in text_lower and phrase not in seen:
            seen.add(phrase)
            matches.append(
                {
                    "phrase": phrase,
                    "n_gram_type": "Trigram" if len(phrase.split()) == 3 else "Bigram",
                    "aspect": aspect,
                }
            )

    return matches[:5]
