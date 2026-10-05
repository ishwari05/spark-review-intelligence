"""
================================================================================
ReviewIQ Multi-Tenant SaaS & Data Isolation Test Suite
--------------------------------------------------------------------------------
Validates:
  1. Authentication Flow (Signup, Login, Current User Context, Invalid Auth)
  2. Platform Capabilities (Marketplace Overview, Seller List, Cross-Seller Analytics)
  3. Seller Capabilities (Private Dashboard, Own Products, Own Reviews)
  4. CRITICAL DATA ISOLATION & PERMISSION MATRIX:
     - Seller A requests Seller B's product -> 403 or 404
     - Seller A requests reviews -> Only Seller A reviews returned
     - Seller A attempts spoofing organization_id = Seller B -> Request rejected/enforced
     - Platform A requests Seller A & Seller B -> Success
     - Seller attempts to access Platform endpoints -> 403 Forbidden
     - Viewer role attempts product creation or analysis -> 403 Forbidden
     - Service role secret key is never leaked via /api/config
================================================================================
"""

import json
import unittest

from api import app
from backend.auth import generate_dev_token
from backend.repository import (
    create_organization,
    create_profile,
    add_org_member,
    create_product,
    create_review,
    link_platform_seller,
)


class TestReviewIQMultiTenancy(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

        # Set up known test tokens:
        # Platform A Admin:
        cls.platform_token = "demo-platform-token"
        cls.platform_headers = {
            "Authorization": f"Bearer {cls.platform_token}",
            "Content-Type": "application/json",
        }

        # Seller A Owner:
        cls.seller_a_token = "demo-seller-aura-token"
        cls.seller_a_headers = {
            "Authorization": f"Bearer {cls.seller_a_token}",
            "Content-Type": "application/json",
        }

        # Seller A Viewer:
        cls.seller_a_viewer_token = "demo-seller-aura-viewer-token"
        cls.seller_a_viewer_headers = {
            "Authorization": f"Bearer {cls.seller_a_viewer_token}",
            "Content-Type": "application/json",
        }

        # Seller B Owner:
        cls.seller_b_token = "demo-seller-lumina-token"
        cls.seller_b_headers = {
            "Authorization": f"Bearer {cls.seller_b_token}",
            "Content-Type": "application/json",
        }

        # Known IDs from seed
        cls.platform_id = "00000000-0000-0000-0000-000000000001"
        cls.seller_a_id = "00000000-0000-0000-0000-000000000002"
        cls.seller_b_id = "00000000-0000-0000-0000-000000000003"

        cls.product_a_id = "10000000-0000-0000-0000-000000000001"  # Aura Headphones (Seller A)
        cls.product_b_id = "10000000-0000-0000-0000-000000000003"  # Lumina Lamp (Seller B)

    # --------------------------------------------------------------------------
    # 1. AUTHENTICATION TESTS
    # --------------------------------------------------------------------------

    def test_01_unauthenticated_requests_return_401(self):
        """Unauthenticated requests to protected endpoints must return 401 Unauthorized."""
        res = self.client.get("/api/products")
        self.assertEqual(res.status_code, 401)
        self.assertIn("error", res.json)

        res2 = self.client.get("/api/reviews")
        self.assertEqual(res2.status_code, 401)

        res3 = self.client.get("/api/platform/overview")
        self.assertEqual(res3.status_code, 401)

    def test_02_login_and_user_context(self):
        """Valid login returns JWT and full profile + organization context."""
        res = self.client.get("/api/auth/me", headers=self.seller_a_headers)
        self.assertEqual(res.status_code, 200)
        data = res.json
        self.assertTrue(data["authenticated"])
        self.assertEqual(data["user"]["organization_type"], "seller")
        self.assertEqual(data["user"]["organization_id"], self.seller_a_id)

    def test_03_signup_flow(self):
        """Signup creates organization, profile, owner role, and generates token."""
        import uuid
        unique_email = f"founder_{uuid.uuid4().hex[:8]}@brandtest.io"
        signup_payload = {
            "full_name": "Test Seller Founder",
            "email": unique_email,
            "password": "strongPassword123",
            "organization_name": "Nova Tech Brand",
            "organization_type": "seller",
        }
        res = self.client.post("/api/auth/signup", data=json.dumps(signup_payload), content_type="application/json")
        self.assertEqual(res.status_code, 201)
        data = res.json
        self.assertIn("token", data)
        self.assertEqual(data["user"]["organization_type"], "seller")
        self.assertEqual(data["user"]["role"], "owner")

    def test_04_public_config_never_exposes_service_role_key(self):
        """Service role key must NEVER be leaked to frontend."""
        res = self.client.get("/api/config")
        self.assertEqual(res.status_code, 200)
        data = res.json
        self.assertNotIn("service_role", json.dumps(data).lower())
        self.assertNotIn("supabase_service_role_key", data)

    # --------------------------------------------------------------------------
    # 2. PLATFORM MARKETPLACE INTELLIGENCE TESTS
    # --------------------------------------------------------------------------

    def test_05_platform_overview_metrics(self):
        """Platform user receives aggregated marketplace intelligence."""
        res = self.client.get("/api/platform/overview", headers=self.platform_headers)
        self.assertEqual(res.status_code, 200)
        data = res.json
        self.assertIn("total_sellers", data)
        self.assertIn("total_products", data)
        self.assertIn("total_reviews", data)
        self.assertIn("marketplace_health", data)
        self.assertGreaterEqual(data["total_sellers"], 2)
        self.assertGreaterEqual(data["total_products"], 4)

    def test_06_platform_can_access_seller_a_and_seller_b(self):
        """Platform A can view intelligence for both managed Seller A and Seller B."""
        res_a = self.client.get(f"/api/platform/sellers/{self.seller_a_id}", headers=self.platform_headers)
        self.assertEqual(res_a.status_code, 200)
        self.assertEqual(res_a.json["seller"]["name"], "Aura Sound Technologies")

        res_b = self.client.get(f"/api/platform/sellers/{self.seller_b_id}", headers=self.platform_headers)
        self.assertEqual(res_b.status_code, 200)
        self.assertEqual(res_b.json["seller"]["name"], "Lumina Smart Devices")

    def test_07_platform_can_view_all_marketplace_products(self):
        """Platform user lists products across all managed sellers."""
        res = self.client.get("/api/products", headers=self.platform_headers)
        self.assertEqual(res.status_code, 200)
        product_names = [p["name"] for p in res.json["products"]]
        self.assertIn("Aura ANC Wireless Headphones", product_names)
        self.assertIn("Lumina Smart Ambient Lamp", product_names)

    # --------------------------------------------------------------------------
    # 3. SELLER PRODUCT INTELLIGENCE & DATA ISOLATION TESTS (MANDATORY)
    # --------------------------------------------------------------------------

    def test_08_seller_dashboard_overview(self):
        """Seller user receives private brand product intelligence."""
        res = self.client.get("/api/seller/overview", headers=self.seller_a_headers)
        self.assertEqual(res.status_code, 200)
        data = res.json
        self.assertEqual(data["seller_name"], "Aura Sound Technologies")
        self.assertIn("metrics", data)
        self.assertIn("health_score", data["metrics"])
        self.assertIn("product_performance", data["metrics"])

    def test_09_seller_cannot_access_platform_overview(self):
        """Seller user attempting to access Platform Overview is rejected with 403 Forbidden."""
        res = self.client.get("/api/platform/overview", headers=self.seller_a_headers)
        self.assertEqual(res.status_code, 403)
        self.assertIn("Forbidden", res.json["error"])

    def test_10_seller_a_only_sees_own_products(self):
        """Seller A querying products must NOT receive Seller B's products."""
        res = self.client.get("/api/products", headers=self.seller_a_headers)
        self.assertEqual(res.status_code, 200)
        products = res.json["products"]
        for p in products:
            self.assertEqual(p["organization_id"], self.seller_a_id)
            self.assertNotEqual(p["name"], "Lumina Smart Ambient Lamp")

    def test_11_seller_a_requests_product_b_is_forbidden(self):
        """
        CRITICAL MANDATORY TEST:
        Seller A requests Product B (belonging to Seller B).
        Expected: 403 Forbidden or 404 Not Found.
        """
        res = self.client.get(f"/api/products/{self.product_b_id}", headers=self.seller_a_headers)
        self.assertIn(res.status_code, (403, 404))

    def test_12_seller_a_requests_reviews_only_sees_own_reviews(self):
        """Seller A querying reviews must never receive reviews for Seller B's products."""
        res = self.client.get("/api/reviews", headers=self.seller_a_headers)
        self.assertEqual(res.status_code, 200)
        for r in res.json["reviews"]:
            self.assertEqual(r["organization_id"], self.seller_a_id)

    def test_13_seller_cannot_modify_organization_id_on_create(self):
        """
        CRITICAL SECURITY TEST:
        Seller A attempts to specify organization_id = Seller B in product creation.
        Expected: Backend enforces Seller A's organization_id derived from the session.
        """
        payload = {
            "name": "Spoofed Product Attempt",
            "sku": "SPOOF-01",
            "organization_id": self.seller_b_id,  # Attempting to insert into Seller B
        }
        res = self.client.post("/api/products", data=json.dumps(payload), headers=self.seller_a_headers)
        self.assertEqual(res.status_code, 201)
        created_prod = res.json["product"]
        # Must be assigned to Seller A, NOT Seller B!
        self.assertEqual(created_prod["organization_id"], self.seller_a_id)

    def test_14_viewer_role_cannot_create_product(self):
        """Viewer role must NOT be allowed to create products (403 Forbidden)."""
        payload = {
            "name": "Unauthorized Product by Viewer",
            "sku": "VIEWER-01",
        }
        res = self.client.post("/api/products", data=json.dumps(payload), headers=self.seller_a_viewer_headers)
        self.assertEqual(res.status_code, 403)
        self.assertIn("Forbidden", res.json["error"])

    def test_15_viewer_role_cannot_import_reviews(self):
        """Viewer role must NOT be allowed to import reviews or run batch analyses."""
        payload = {
            "product_id": self.product_a_id,
            "reviews": [{"title": "Test", "content": "Sample review text", "rating": 5}],
        }
        res = self.client.post("/api/reviews/import", data=json.dumps(payload), headers=self.seller_a_viewer_headers)
        self.assertEqual(res.status_code, 403)

    def test_16_live_predict_inference_pipeline(self):
        """Spark ML prediction pipeline executes correctly for authenticated users."""
        payload = {"review": "I love this product! It is amazing, excellent, and works perfectly."}
        res = self.client.post("/api/predict", data=json.dumps(payload), headers=self.seller_a_headers)
        self.assertEqual(res.status_code, 200)
        data = res.json
        self.assertEqual(data["sentiment"], "POSITIVE")
        self.assertGreaterEqual(data["confidence"], 50.0)
        self.assertIn("aspects", data)
        self.assertIn("complaints", data)


if __name__ == "__main__":
    unittest.main()
