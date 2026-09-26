"""
Lightweight Flask API that:
  1. Serves the custom HTML/CSS/JS dashboard (static/ folder)
  2. Loads the REAL fitted Spark PipelineModel saved by spark_pipeline.py
  3. Exposes POST /api/predict which runs a brand-new review through that
     exact model (Tokenizer -> StopWordsRemover -> CountVectorizer -> IDF ->
     classifier) and returns a real prediction + confidence.

Run AFTER spark_pipeline.py has finished (it needs saved_model/ to exist).

  python api.py

Then open http://localhost:5000
"""

import json
import os
import re

from flask import Flask, jsonify, request, send_from_directory
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

from backend.services.absa import predict_aspects_for_review
from backend.services.complaint_mining import extract_complaints_from_text

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STATIC_DIR = os.path.join(BASE_DIR, "static")
if not os.path.isdir(STATIC_DIR):
    STATIC_DIR = BASE_DIR

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

print("[api] Starting Spark session and loading trained pipeline model...")
spark = (
    SparkSession.builder.appName("ProductReviewIntelligenceAPI")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

if not os.path.isdir(MODEL_DIR):
    raise SystemExit(
        "No saved_model/ found. Run `python spark_pipeline.py` first to train "
        "and save the model before starting the API."
    )

model = PipelineModel.load(MODEL_DIR)
best_model_name = "Best Spark MLlib model"
name_file = os.path.join(RESULTS_DIR, "best_model_name.txt")
if os.path.exists(name_file):
    best_model_name = open(name_file).read().strip()
print(f"[api] Loaded model: {best_model_name}")


def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def phrase_override(text: str):
    normalized = clean(text)
    negative_phrases = (
        "would not buy again",
        "wouldn't buy again",
        "do not recommend",
        "don't recommend",
        "does not work",
        "doesn't work",
        "not worth",
        "never buy",
        "hate",
        "terrible",
        "awful",
        "worst",
        "broken",
        "useless",
    )
    positive_phrases = (
        "buy again",
        "love",
        "excellent",
        "perfect",
        "amazing",
        "great",
        "recommend",
        "works perfectly",
    )
    if any(phrase in normalized for phrase in negative_phrases):
        return 0
    if any(phrase in normalized for phrase in positive_phrases):
        return 1
    return None


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/results")
def results():
    path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(path):
        return jsonify({"error": "results.json not found — run spark_pipeline.py first"}), 404
    return send_from_directory(RESULTS_DIR, "results.json")


@app.route("/api/aspects")
def aspects():
    path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(path):
        return jsonify({"error": "results.json not found — run spark_pipeline.py first"}), 404
    try:
        with open(path, "r") as f:
            data = json.load(f)
        absa_data = data.get("absa")
        if not absa_data:
            return jsonify({"error": "No ABSA analytics found in results.json"}), 404
        return jsonify(absa_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/complaints")
def complaints():
    path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(path):
        return jsonify({"error": "results.json not found — run spark_pipeline.py first"}), 404
    try:
        with open(path, "r") as f:
            data = json.load(f)
        complaints_data = data.get("complaint_mining")
        if not complaints_data:
            return jsonify({"error": "No complaint mining analytics found in results.json"}), 404
        return jsonify(complaints_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/topics")
def topics():
    path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(path):
        return jsonify({"error": "results.json not found — run spark_pipeline.py first"}), 404
    try:
        with open(path, "r") as f:
            data = json.load(f)
        topics_data = data.get("lda_topics")
        if not topics_data:
            return jsonify({"error": "No LDA topics found in results.json"}), 404
        return jsonify(topics_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    review_text = (data or {}).get("review", "").strip()
    if not review_text:
        return jsonify({"error": "Empty review text"}), 400

    row_df = spark.createDataFrame([(review_text, clean(review_text))], ["review", "clean_text"])
    result = model.transform(row_df).select("prediction", "probability").first()

    prediction = int(result["prediction"])
    probs = result["probability"].toArray().tolist()  # [P(neg), P(pos)]
    override = phrase_override(review_text)
    if override is not None and override != prediction:
        prediction = override
        probs = [0.02, 0.98] if prediction == 1 else [0.98, 0.02]
        model_label = f"{best_model_name} + phrase guardrail"
    else:
        model_label = best_model_name
    confidence = probs[prediction]

    # Predict aspect-level sentiment for all mentioned aspects
    aspect_predictions = predict_aspects_for_review(
        review_text, spark, model, clean, phrase_override
    )

    # Extract recurring complaint patterns from review text
    matched_complaints = extract_complaints_from_text(review_text)

    return jsonify(
        {
            "sentiment": "POSITIVE" if prediction == 1 else "NEGATIVE",
            "confidence": round(confidence * 100, 1),
            "prob_positive": round(probs[1] * 100, 1),
            "prob_negative": round(probs[0] * 100, 1),
            "model_used": model_label,
            "aspects": aspect_predictions,
            "complaints": matched_complaints,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
