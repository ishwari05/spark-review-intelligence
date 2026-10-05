"""
ReviewIQ Backend API Entrypoint Forwarder
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api import app, spark, model, best_model_name

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
