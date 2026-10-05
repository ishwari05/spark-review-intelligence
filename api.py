"""
================================================================================
ReviewIQ Multi-Tenant SaaS Backend API
--------------------------------------------------------------------------------
Architecture:
  - Supabase Auth + JWT Verification
  - Multi-Tenant Row Level Security & Organization Data Isolation
  - E-Commerce Platform vs. Seller Roles & Permissions Matrix
  - Apache Spark MLlib Phase 4 Tuned Logistic Regression (90.55% Acc)
  - Aspect-Based Sentiment Analysis (ABSA) & Recurring Complaint Mining
  - Production-grade Error Handling (401, 403, 404, 409, 500)
================================================================================
"""

import json
import os
import re

from flask import Flask, jsonify, request, send_from_directory
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

from backend.auth import extract_token_from_header, require_auth, resolve_auth_user
from backend.routes.auth_routes import auth_bp
from backend.routes.platform_routes import platform_bp
from backend.routes.product_routes import product_bp
from backend.routes.review_routes import review_bp
from backend.routes.seller_routes import seller_bp
from backend.routes.settings_routes import settings_bp
from backend.services.absa import predict_aspects_for_review
from backend.services.complaint_mining import extract_complaints_from_text
from pipeline.preprocessing import preprocess_variant_c

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STATIC_DIR = os.path.join(BASE_DIR, "frontend")

# Phase 4 model — Tuned Logistic Regression (90.55% accuracy)
MODEL_DIR = os.path.join(BASE_DIR, "saved_model_phase4")
if not os.path.isdir(MODEL_DIR):
    MODEL_DIR = os.path.join(BASE_DIR, "pipeline", "saved_model_phase4")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

# Register Modular Multi-Tenant SaaS Blueprints
app.register_blueprint(auth_bp)
app.register_blueprint(platform_bp)
app.register_blueprint(seller_bp)
app.register_blueprint(product_bp)
app.register_blueprint(review_bp)
app.register_blueprint(settings_bp)

print("[api] Starting Spark session and loading Phase 4 pipeline model...")
spark = (
    SparkSession.builder.appName("ReviewIQMultiTenantAPI")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

if not os.path.isdir(MODEL_DIR):
    raise SystemExit(
        "Phase 4 model not found. Run `python pipeline/run_phase4_experiments.py` "
        "first to train and save saved_model_phase4/ before starting the API."
    )

model = PipelineModel.load(MODEL_DIR)
best_model_name = "Tuned Logistic Regression (Phase 4)"
name_file = os.path.join(RESULTS_DIR, "best_model_name.txt")
if os.path.exists(name_file):
    raw = open(name_file).read().strip()
    if raw:
        best_model_name = raw
print(f"[api] Loaded model: {best_model_name}  (from {MODEL_DIR})")


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


# ------------------------------------------------------------------------------
# FRONTEND SPA SERVING
# ------------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


# Support direct URLs for routing
@app.route("/login")
@app.route("/signup")
@app.route("/platform/dashboard")
@app.route("/seller/dashboard")
def spa_routes():
    return send_from_directory(STATIC_DIR, "index.html")


# ------------------------------------------------------------------------------
# MACHINE LEARNING & PREDICTION ENDPOINTS (PRESERVED & PROTECTED)
# ------------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
@require_auth
def predict():
    """
    Executes live sentiment inference, ABSA aspect extraction, and recurring
    complaint mining using the Phase 4 Spark MLlib pipeline.
    Requires authentication.
    """
    data = request.get_json(force=True) or {}
    review_text = (data.get("review") or "").strip()
    if not review_text:
        return jsonify({"error": "Empty review text."}), 400

    row_df = spark.createDataFrame([(review_text,)], ["review"])
    row_df = preprocess_variant_c(row_df, text_col="review", output_col="clean_text")
    result = model.transform(row_df).select("prediction", "probability").first()

    prediction = int(result["prediction"])
    probs = result["probability"].toArray().tolist()
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

    return jsonify({
        "sentiment": "POSITIVE" if prediction == 1 else "NEGATIVE",
        "confidence": round(confidence * 100, 1),
        "prob_positive": round(probs[1] * 100, 1),
        "prob_negative": round(probs[0] * 100, 1),
        "model_used": model_label,
        "aspects": aspect_predictions,
        "complaints": matched_complaints,
    })


@app.route("/api/results")
def results():
    """Returns baseline Phase 1-4 experimental and model evaluation metrics."""
    path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(path):
        return jsonify({"error": "results.json not found"}), 404
    return send_from_directory(RESULTS_DIR, "results.json")


@app.route("/api/aspects")
def aspects():
    """Returns aspect intelligence data."""
    # Check if authenticated user exists to scope aspects to tenant
    token = extract_token_from_header()
    user = resolve_auth_user(token) if token else None
    if user:
        from backend.repository import get_aspect_results
        aspect_list = get_aspect_results(user.accessible_org_ids)
        if aspect_list:
            return jsonify({"aspects": aspect_list})

    path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(path):
        return jsonify({"error": "results.json not found"}), 404
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return jsonify(data.get("absa", {}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/complaints")
def complaints():
    """Returns complaint mining data."""
    token = extract_token_from_header()
    user = resolve_auth_user(token) if token else None
    if user:
        from backend.repository import get_issue_results
        issue_list = get_issue_results(user.accessible_org_ids)
        if issue_list:
            return jsonify({"top_complaints": issue_list})

    path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(path):
        return jsonify({"error": "results.json not found"}), 404
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return jsonify(data.get("complaint_mining", {}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/topics")
def topics():
    path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(path):
        return jsonify({"error": "results.json not found"}), 404
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return jsonify(data.get("lda_topics", {}))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard")
def dashboard():
    """
    Context-aware dashboard metrics endpoint.
    If authenticated, returns tenant-specific dashboard metrics.
    If unauthenticated or fallback, returns baseline Amazon Product Reviews metrics.
    """
    token = extract_token_from_header()
    user = resolve_auth_user(token) if token else None

    if user:
        if user.org_type == "platform":
            from backend.repository import get_platform_overview_metrics
            p_data = get_platform_overview_metrics(user.org_id)
            return jsonify({
                "role": "platform",
                "organization_name": user.org_name,
                "marketplace": p_data,
            })
        else:
            from backend.repository import get_seller_overview_metrics
            s_data = get_seller_overview_metrics(user.org_id)
            return jsonify({
                "role": "seller",
                "organization_name": user.org_name,
                "seller": s_data,
            })

    # Baseline unauthenticated results fallback
    path = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(path):
        return jsonify({"error": "results.json not found"}), 404
    try:
        with open(path, "r") as f:
            data = json.load(f)

        eda = data.get("eda", {})
        total_rows = eda.get("total_rows", 168676)
        class_dist = eda.get("class_distribution", {})
        pos_count = class_dist.get("positive", 86008)
        neg_count = class_dist.get("negative", 82668)
        total_valid = pos_count + neg_count if (pos_count + neg_count) > 0 else total_rows
        pos_pct = round((pos_count / total_valid) * 100, 1)
        neg_pct = round((neg_count / total_valid) * 100, 1)

        absa = data.get("absa", {})
        aspects_list = absa.get("aspects", [])

        total_mentions = 0
        weighted_pos = 0.0
        for asp in aspects_list:
            m = asp.get("mentions", 0)
            p_pct = asp.get("positive_pct", 50.0)
            total_mentions += m
            weighted_pos += m * p_pct

        health_score = round(weighted_pos / total_mentions) if total_mentions > 0 else round(pos_pct)

        return jsonify({
            "product_name": "Amazon Product Reviews",
            "total_reviews": total_rows,
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_percent": pos_pct,
            "negative_percent": neg_pct,
            "product_health": {
                "score": health_score,
                "status": "Healthy" if health_score >= 75 else "Moderate",
                "badge": "success" if health_score >= 75 else "warning",
                "total_aspect_mentions": total_mentions,
            },
            "aspects": aspects_list,
            "model_summary": {
                "name": data.get("best_model", best_model_name),
                "accuracy": 90.55,
                "f1": 90.54,
                "roc_auc": 0.9636,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------------------
# ERROR HANDLERS (PRODUCTION GRADE)
# ------------------------------------------------------------------------------

@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad Request: Please check your parameters."}), 400


@app.errorhandler(401)
def unauthorized(error):
    return jsonify({"error": "Unauthorized: Authentication required."}), 401


@app.errorhandler(403)
def forbidden(error):
    return jsonify({"error": "Forbidden: You don't have permission to access this resource."}), 403


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Resource not found."}), 404


@app.errorhandler(409)
def conflict(error):
    return jsonify({"error": "Conflict: Resource already exists."}), 409


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal Server Error: Something went wrong."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
