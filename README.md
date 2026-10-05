# ReviewIQ

ReviewIQ is a review intelligence platform for e-commerce platforms and seller organizations. It combines a Flask backend, a Supabase/PostgreSQL-ready data layer, and a Spark ML pipeline to analyze customer reviews, surface aspect-level sentiment, and expose tenant-aware dashboards for platform and seller users.

This repository contains the currently implemented application code rather than a purely experimental notebook workflow. The live app entry point is `api.py`, the main business logic sits under `backend/`, the ML pipeline and saved artifacts live under `pipeline/`, and the data model is defined in `supabase/migrations/`.

## Current implementation status

The current codebase includes:

- A Flask API with protected multi-tenant routes for authentication, products, reviews, platform dashboards, seller dashboards, and settings.
- JWT-based authentication and role enforcement in `backend/auth.py`.
- A database layer that supports Supabase PostgreSQL and a local/dev fallback path.
- Organization and product-level access control for `platform` and `seller` accounts.
- A live Spark pipeline model loaded from `saved_model_phase4/`.
- Aspect-based sentiment analysis and complaint mining services.
- A static frontend dashboard shell served by the Flask app.

This README describes what is implemented in the repository today. It does not describe a future roadmap or aspirational architecture unless a corresponding implementation already exists in the codebase.

---

## Architecture and repository layout

- `api.py` — Flask application entry point and live endpoints.
- `backend/` — auth, repository/data access, and route modules.
- `backend/services/absa.py` — aspect-based sentiment extraction.
- `backend/services/complaint_mining.py` — negative phrase and complaint mining logic.
- `pipeline/` — Spark preprocessing and experiment runners, plus saved model artifacts.
- `saved_model_phase4/` — current production model bundle used by the API.
- `supabase/migrations/` — actual schema and row-level security definitions.
- `frontend/` — static dashboard UI shell and client-side logic.
- `test/` — Python unittest coverage for auth, tenancy, and live prediction flows.

---

## Product capabilities

### 1. Multi-tenant review intelligence

The app models two organization types:

- `platform` organizations: marketplace operators that can view aggregated intelligence across managed sellers.
- `seller` organizations: brands that can view only their own products and reviews.

The backend enforces this through server-side organization resolution and permission checks in `backend/auth.py` and the repository layer. The database schema also defines `platform_sellers` mapping records and `organization_members` role assignments.

### 2. Authentication and role enforcement

The current auth layer includes:

- JWT verification for bearer tokens.
- Local demo tokens for testing and demonstration flows.
- Role-based access control for `owner`, `admin`, `analyst`, and `viewer`.
- Server-side enforcement that prevents frontend-supplied organization IDs from being trusted.

The auth system explicitly distinguishes platform and seller authorization and rejects unauthorized cross-tenant access with 401 or 403 responses.

### 3. Product and review management

The repository includes data modeling for:

- organizations
- platform-to-seller relationships
- products
- datasets
- reviews
- analyses
- aspect results
- issue results
- user preferences
- audit logs

The actual schema is defined in `supabase/migrations/001_initial_schema.sql` and the row-level security policies are in `supabase/migrations/002_rls_policies.sql`.

### 4. Sentiment analysis and ABSA

The app loads a Spark-based sentiment pipeline from `saved_model_phase4/` and serves live review inference. The repository also includes aspect extraction via `backend/services/absa.py`, which identifies aspect phrases such as:

- Battery
- Build Quality
- Price / Value
- Customer Support
- Delivery / Packaging

### 5. Complaint mining and issue discovery

A separate complaint-mining service identifies negative phrases and organizes them into issue themes. The repository contains complaint mining code in `backend/services/complaint_mining.py` and result tables such as `issue_results` in the database schema.

### 6. Dashboard and reporting

The frontend shell is a static dashboard interface served by Flask. It is built around the actual backend metrics and analytics routes rather than a separate JavaScript application framework. The current project includes dashboard views for overview metrics, seller/platform intelligence, and review-level analysis.

---

## Verified model status

The live model used by the application is a saved Spark `PipelineModel` from `saved_model_phase4/`. It is a tuned logistic regression trained with the repository’s Phase 4 configuration.

Verified repository metrics from `results/phase4_summary.md`:

- Model: Tuned Logistic Regression
- Accuracy: 0.9055 (90.55%)
- F1: 0.9054
- ROC-AUC: 0.9636
- Dataset size: 100,000 rows (Amazon Polarity sample)
- Feature representation: unigram + bigram TF-IDF
- Train/test split: 80/20 with seed 42

The Phase 4 summary also records that the tuned model was selected after validation and then evaluated once on the untouched test set.

---

## Data and persistence model

The current repository is wired for a multi-tenant SaaS setup backed by Supabase PostgreSQL and ready for server-side auth. The core schema includes the following tables in the migration set:

- `profiles`
- `organizations`
- `organization_members`
- `platform_sellers`
- `organization_invitations`
- `products`
- `datasets`
- `reviews`
- `analyses`
- `aspect_results`
- `issue_results`
- `user_preferences`
- `audit_logs`

This is the actual implementation in the repo. The project also contains a SQLite/dev fallback path in the backend data layer, but the formal schema defined for the product is the Supabase/PostgreSQL design above.

---

## API surface

The current backend registers a set of authenticated endpoints. The main route groups are:

### Authentication and session

- `GET /api/config`
- `POST /api/auth/signup`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `POST /api/auth/logout`

### Platform dashboard

- `GET /api/platform/overview`
- `GET /api/platform/sellers`
- `POST /api/platform/sellers`
- `GET /api/platform/sellers/<seller_id>`
- `GET /api/platform/analytics`

### Seller dashboard

- `GET /api/seller/overview`
- `GET /api/seller/analytics`

### Product operations

- `GET /api/products`
- `POST /api/products`
- `GET /api/products/<product_id>`

### Review operations

- `GET /api/reviews`
- `POST /api/reviews/upload`
- `POST /api/reviews/import`

### Settings and admin flows

- `GET /api/settings/team`
- `GET /api/settings/invitations`
- `POST /api/settings/invite`
- `GET /api/settings/audit`

### Live model inference and analytics

- `POST /api/predict`
- `GET /api/results`
- `GET /api/aspects`
- `GET /api/complaints`
- `GET /api/topics`
- `GET /api/dashboard`

In addition to these, the app serves the dashboard shell and auth pages from `api.py` and the static frontend files.

---

## Frontend and dashboard

The frontend is a static HTML/CSS/JavaScript dashboard served by Flask. The key app files are:

- `frontend/index.html`
- `frontend/style.css`
- `frontend/script.js`

This is a browser-based dashboard shell and client logic for the ReviewIQ app; it is not a separate React or Node service.

---

## Environment and local setup

### Prerequisites

- Python 3.9+
- Java installed for Spark-based local execution
- A configured `.env` file (optional for local demo mode, required for Supabase-backed flows)

### Install

```bash
git clone <repository-url>
cd <repository-folder>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Run the app

```bash
source .venv/bin/activate
python api.py
```

The app runs on the default Flask port and can be opened in a browser at the local endpoint created by the app (typically `http://localhost:5000`).

The repository also includes demo auth tokens in `backend/auth.py`, including platform and seller test tokens such as:

- `demo-platform-token`
- `demo-seller-aura-token`
- `demo-seller-aura-viewer-token`
- `demo-seller-lumina-token`

These are used for local testing and demonstrations.

---

## Testing

The repository includes a Python unittest suite under `test/` covering:

- unauthenticated request enforcement
- login and profile creation
- signup flow behavior
- organization access rules
- platform vs. seller isolation
- viewer role restrictions
- live prediction pipeline execution

The project is executed with:

```bash
source .venv/bin/activate
python -m unittest discover -s test -v
```

This repository currently contains a test suite with multiple checks in the current workspace. The actual execution in this environment discovered 17 tests and reported 1 failure and 1 error, so the suite is not currently described as fully passing in this README.

---

## Project notes and boundaries

The repository includes both application code and historical experimental artifacts:

- `pipeline/` contains the experiment scripts and result files used to compare different feature and model configurations.
- `saved_model_phase1/`, `saved_model_phase2/`, `saved_model_phase3/`, and `saved_model_phase4/` are model artifacts retained in the repo.
- The active app is wired to `saved_model_phase4/` via `api.py`.

This is important because the repo contains both a product/workflow layer and a research experiment layer. The product app is the API + backend + data model; the pipeline directories store the model development history.

---

## Summary

ReviewIQ is a working multi-tenant review analytics application with a live Spark ML inference path, an authenticated API, a product schema for multi-organization data, and a dashboard-oriented frontend. The current repository state is clearly far more than a one-off sentiment classifier: it implements a review intelligence platform with tenant separation, role enforcement, persisted analytics, and model-backed insight generation.
