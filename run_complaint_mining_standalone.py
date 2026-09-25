"""
Helper script to execute Topic & Complaint Mining on Amazon Polarity negative reviews,
generate figures, export CSVs, and update results.json.
"""
import json
import os
import pandas as pd
from pyspark.sql import SparkSession
from datasets import load_dataset as hf_load_dataset

from complaint_mining import (
    mine_negative_complaints_spark,
    run_spark_lda_topics,
    generate_complaint_figures,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")

print("[complaints-runner] Starting Spark session...")
spark = (
    SparkSession.builder.appName("ComplaintMiningExecution")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

print("[complaints-runner] Streaming sample from amazon_polarity...")
SAMPLE_REVIEWS = 30000
hf_ds = hf_load_dataset("amazon_polarity", split="train", streaming=True)
rows = []
for i, item in enumerate(hf_ds):
    if i >= SAMPLE_REVIEWS:
        break
    # Amazon polarity label: 0.0 is Negative, 1.0 is Positive
    if float(item["label"]) == 0.0:
        rows.append((item["content"], item["content"].lower()))

print(f"[complaints-runner] Loaded {len(rows)} negative reviews. Creating Spark DataFrame...")
schema = ["review", "clean_text"]
neg_df = spark.createDataFrame(rows, schema=schema).repartition(8).cache()

print("[complaints-runner] Mining recurring n-gram complaints with Spark MLlib...")
complaints_data = mine_negative_complaints_spark(spark, neg_df, total_neg_count=len(rows), top_n=25)
print(f"[complaints-runner] Extracted {len(complaints_data['top_complaints'])} quality complaint phrases.")
for c in complaints_data["top_complaints"][:5]:
    print(f"  -> [{c['n_gram_type']}] \"{c['phrase']}\" ({c['frequency']} times, {c['percentage']}%) -> {c['aspect']}")

print("[complaints-runner] Running Spark MLlib LDA Topic Discovery (k=5)...")
lda_topics = run_spark_lda_topics(spark, neg_df, k=5, max_iter=15)
for t in lda_topics:
    print(f"  -> {t['label']} ({t['statistical_interpretation']}): {t['top_words_str']}")

print("[complaints-runner] Generating presentation figures in results/figures/...")
generate_complaint_figures(complaints_data, lda_topics, FIGURES_DIR)

# Export results/complaints.csv
complaints_csv = os.path.join(RESULTS_DIR, "complaints.csv")
df_comp_out = pd.DataFrame(complaints_data["top_complaints"])
df_comp_out.to_csv(complaints_csv, index=False)
print(f"[complaints-runner] Saved complaints to {complaints_csv}")

# Export results/topics.csv
topics_csv = os.path.join(RESULTS_DIR, "topics.csv")
df_top_out = pd.DataFrame([
    {
        "topic_id": t["topic_id"],
        "label": t["label"],
        "top_terms": t["top_words_str"],
        "interpretation": t["statistical_interpretation"],
    }
    for t in lda_topics
])
df_top_out.to_csv(topics_csv, index=False)
print(f"[complaints-runner] Saved topics to {topics_csv}")

# Update results.json
if os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON, "r") as f:
        data = json.load(f)
    data["complaint_mining"] = complaints_data
    data["lda_topics"] = lda_topics
    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[complaints-runner] Updated {RESULTS_JSON} with Complaint & Topic Mining analytics!")

print("[complaints-runner] Complaint mining execution finished successfully!")
spark.stop()
