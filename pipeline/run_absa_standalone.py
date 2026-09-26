"""
Helper script to execute ABSA on a streaming Amazon Polarity sample using the
saved Spark PipelineModel and update results.json, aspect_sentiment.csv, and figures.
"""
import json
import os
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from datasets import load_dataset as hf_load_dataset

from absa import run_spark_absa, generate_absa_figures

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")

print("[absa-runner] Starting Spark session...")
spark = (
    SparkSession.builder.appName("ABSAExecution")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

print(f"[absa-runner] Loading saved model from {MODEL_DIR}...")
model = PipelineModel.load(MODEL_DIR)

print("[absa-runner] Streaming sample from amazon_polarity...")
SAMPLE_REVIEWS = 30000
hf_ds = hf_load_dataset("amazon_polarity", split="train", streaming=True)
rows = []
for i, item in enumerate(hf_ds):
    if i >= SAMPLE_REVIEWS:
        break
    rows.append((item["content"],))

print(f"[absa-runner] Loaded {len(rows)} reviews. Creating Spark DataFrame...")
df = spark.createDataFrame(rows, schema=["review"]).repartition(8).cache()

print("[absa-runner] Running distributed ABSA across Spark partitions...")
absa_df, absa_summary = run_spark_absa(spark, df, model)
print(f"[absa-runner] Total aspect mentions detected: {absa_summary['total_aspect_mentions']}")
for a in absa_summary["aspects"]:
    print(f"  -> {a['aspect']}: {a['mentions']} mentions, {a['positive_pct']}% Pos, {a['negative_pct']}% Neg | {a['status']}")

print("[absa-runner] Generating ABSA figures...")
generate_absa_figures(absa_summary["aspects"], FIGURES_DIR)

csv_path = os.path.join(RESULTS_DIR, "aspect_sentiment.csv")
print(f"[absa-runner] Saving aspect sentiment sample CSV to {csv_path}...")
sample_pd = absa_df.limit(2000).toPandas()
sample_pd.to_csv(csv_path, index=False)

# Update results.json with the real ABSA data
if os.path.exists(RESULTS_JSON):
    with open(RESULTS_JSON, "r") as f:
        data = json.load(f)
    data["absa"] = absa_summary
    with open(RESULTS_JSON, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[absa-runner] Updated {RESULTS_JSON} with ABSA analytics!")

# Also write to root results.json if it exists
root_results = os.path.join(BASE_DIR, "results.json")
if os.path.exists(root_results):
    with open(root_results, "r") as f:
        data = json.load(f)
    data["absa"] = absa_summary
    with open(root_results, "w") as f:
        json.dump(data, f, indent=2)

print("[absa-runner] ABSA execution finished successfully!")
spark.stop()
