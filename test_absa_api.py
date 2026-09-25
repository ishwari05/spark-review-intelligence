"""
Unit test for API and ABSA functionality
"""
import json
from api import app

client = app.test_client()

print("--- Testing GET /api/results ---")
res = client.get("/api/results")
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
data = res.get_json()
assert "absa" in data, "absa key missing in /api/results"
print("GET /api/results PASSED!")

print("--- Testing GET /api/aspects ---")
res = client.get("/api/aspects")
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
absa_data = res.get_json()
assert "aspects" in absa_data, "aspects missing in /api/aspects"
assert "pain_points" in absa_data, "pain_points missing in /api/aspects"
print(f"GET /api/aspects PASSED! Found {len(absa_data['aspects'])} aspects.")

print("--- Testing POST /api/predict (Mixed Aspects) ---")
test_review = "The battery life is terrible but the build quality is excellent."
res = client.post("/api/predict", json={"review": test_review})
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
pred_data = res.get_json()
print("Prediction response:")
print(json.dumps(pred_data, indent=2))
assert "aspects" in pred_data, "aspects missing in predict response"
aspects_dict = {a["aspect"]: a["sentiment"] for a in pred_data["aspects"]}
print("Detected aspects:", aspects_dict)
assert "Battery" in aspects_dict, "Battery aspect not detected"
assert "Build Quality" in aspects_dict, "Build Quality aspect not detected"
assert aspects_dict["Battery"] == "NEGATIVE", f"Battery sentiment expected NEGATIVE, got {aspects_dict['Battery']}"
assert aspects_dict["Build Quality"] == "POSITIVE", f"Build Quality sentiment expected POSITIVE, got {aspects_dict['Build Quality']}"
print("Mixed Aspect prediction PASSED!")

print("--- Testing POST /api/predict (No Aspects) ---")
general_review = "I really enjoyed reading this book."
res = client.post("/api/predict", json={"review": general_review})
assert res.status_code == 200, f"Expected 200, got {res.status_code}"
gen_data = res.get_json()
assert "aspects" in gen_data
assert len(gen_data["aspects"]) == 0, f"Expected 0 aspects, got {len(gen_data['aspects'])}"
print("No aspects prediction PASSED!")

print("\nALL API AND ABSA TESTS PASSED SUCCESSFULLY! 🎉")
