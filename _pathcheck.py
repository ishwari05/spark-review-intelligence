"""Temporary: verify where api.py / spark_pipeline.py look for artifacts vs where they live."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
BE = os.path.join(ROOT, "backend")
PL = os.path.join(ROOT, "pipeline")

checks = [
    ("api.py  MODEL_DIR   (backend/saved_model)", os.path.join(BE, "saved_model")),
    ("api.py  RESULTS_DIR (backend/results)", os.path.join(BE, "results")),
    ("api.py  STATIC_DIR  (backend/static)", os.path.join(BE, "static")),
    ("ACTUAL  backend/model/saved_model", os.path.join(BE, "model", "saved_model")),
    ("ACTUAL  ./results", os.path.join(ROOT, "results")),
    ("ACTUAL  ./frontend/index.html", os.path.join(ROOT, "frontend", "index.html")),
    ("pipeline RESULTS_DIR (pipeline/results)", os.path.join(PL, "results")),
    ("pipeline MODEL_DIR   (pipeline/saved_model)", os.path.join(PL, "saved_model")),
]
for label, path in checks:
    print(f"{'EXISTS ' if os.path.exists(path) else 'MISSING'}  {label:45s} -> {path}")

print()
for d in ("backend", "backend/services", "pipeline", "test"):
    p = os.path.join(ROOT, d, "__init__.py")
    print(f"{'YES' if os.path.isfile(p) else 'NO ':4s} __init__.py in {d}")
