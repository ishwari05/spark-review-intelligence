"""
================================================================================
ReviewIQ Multi-Tenant Database Layer
--------------------------------------------------------------------------------
Provides unified database persistence supporting:
  1. Live Supabase PostgreSQL (via Supabase client or psycopg2 connection URI)
  2. Local SQLite persistence for development, automated offline testing, and initial setup
================================================================================
"""

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SQLITE_DB_PATH = os.path.join(BASE_DIR, "reviewiq_local.db")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# Check if real Supabase credentials are configured
IS_SUPABASE_CONFIGURED = bool(
    SUPABASE_URL
    and not SUPABASE_URL.startswith("https://your-project")
    and (SUPABASE_SERVICE_ROLE_KEY and not SUPABASE_SERVICE_ROLE_KEY.startswith("eyJhbGciOi..."))
)

supabase_client = None
if IS_SUPABASE_CONFIGURED:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print(f"[db] Connected to Supabase at {SUPABASE_URL}")
    except Exception as e:
        print(f"[db] Warning: Failed to initialize Supabase client: {e}. Falling back to local store.")
        IS_SUPABASE_CONFIGURED = False


def get_sqlite_conn() -> sqlite3.Connection:
    """Returns a SQLite connection with row_factory set to sqlite3.Row."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_database():
    """Initializes local SQLite schema matching the Supabase PostgreSQL schema."""
    conn = get_sqlite_conn()
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS profiles (
        id TEXT PRIMARY KEY,
        auth_user_id TEXT UNIQUE,
        full_name TEXT NOT NULL,
        email TEXT NOT NULL,
        avatar_url TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS organizations (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        organization_type TEXT NOT NULL CHECK (organization_type IN ('platform', 'seller')),
        slug TEXT UNIQUE,
        logo_url TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS organization_members (
        id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        user_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
        role TEXT NOT NULL CHECK (role IN ('owner', 'admin', 'analyst', 'viewer')),
        created_at TEXT NOT NULL,
        UNIQUE (organization_id, user_id)
    );

    CREATE TABLE IF NOT EXISTS platform_sellers (
        id TEXT PRIMARY KEY,
        platform_organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        seller_organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended', 'pending')),
        created_at TEXT NOT NULL,
        UNIQUE (platform_organization_id, seller_organization_id)
    );

    CREATE TABLE IF NOT EXISTS organization_invitations (
        id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        email TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'analyst' CHECK (role IN ('owner', 'admin', 'analyst', 'viewer')),
        token TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'expired', 'revoked')),
        invited_by TEXT REFERENCES profiles(id),
        expires_at TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS products (
        id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        sku TEXT,
        category TEXT,
        description TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        file_name TEXT,
        storage_path TEXT,
        row_count INTEGER DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'ready' CHECK (status IN ('uploading', 'ready', 'analyzing', 'error')),
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS reviews (
        id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        product_id TEXT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
        dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
        title TEXT,
        content TEXT NOT NULL,
        rating INTEGER CHECK (rating >= 1 AND rating <= 5),
        sentiment TEXT CHECK (sentiment IN ('POSITIVE', 'NEGATIVE', 'NEUTRAL')),
        sentiment_score REAL,
        review_date TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS analyses (
        id TEXT PRIMARY KEY,
        organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        product_id TEXT REFERENCES products(id) ON DELETE CASCADE,
        dataset_id TEXT REFERENCES datasets(id) ON DELETE SET NULL,
        status TEXT NOT NULL DEFAULT 'completed' CHECK (status IN ('queued', 'running', 'completed', 'failed')),
        total_reviews INTEGER NOT NULL DEFAULT 0,
        positive_reviews INTEGER NOT NULL DEFAULT 0,
        negative_reviews INTEGER NOT NULL DEFAULT 0,
        health_score INTEGER NOT NULL DEFAULT 0,
        accuracy REAL DEFAULT 0.9055,
        f1_score REAL DEFAULT 0.9054,
        roc_auc REAL DEFAULT 0.9636,
        model_name TEXT DEFAULT 'Tuned Logistic Regression (Phase 4)',
        created_at TEXT NOT NULL,
        completed_at TEXT
    );

    CREATE TABLE IF NOT EXISTS aspect_results (
        id TEXT PRIMARY KEY,
        analysis_id TEXT REFERENCES analyses(id) ON DELETE CASCADE,
        organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        product_id TEXT REFERENCES products(id) ON DELETE CASCADE,
        aspect TEXT NOT NULL,
        mentions INTEGER NOT NULL DEFAULT 0,
        positive_count INTEGER NOT NULL DEFAULT 0,
        negative_count INTEGER NOT NULL DEFAULT 0,
        positive_pct REAL NOT NULL DEFAULT 0.0,
        negative_pct REAL NOT NULL DEFAULT 0.0,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS issue_results (
        id TEXT PRIMARY KEY,
        analysis_id TEXT REFERENCES analyses(id) ON DELETE CASCADE,
        organization_id TEXT NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
        product_id TEXT REFERENCES products(id) ON DELETE CASCADE,
        phrase TEXT NOT NULL,
        aspect TEXT,
        frequency INTEGER NOT NULL DEFAULT 0,
        percentage REAL NOT NULL DEFAULT 0.0,
        priority TEXT NOT NULL DEFAULT 'MEDIUM' CHECK (priority IN ('HIGH', 'MEDIUM', 'LOW')),
        negative_pct REAL DEFAULT 80.0,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS user_preferences (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES profiles(id) ON DELETE CASCADE UNIQUE,
        theme TEXT DEFAULT 'dark',
        notifications INTEGER DEFAULT 1,
        default_aspect TEXT DEFAULT 'Price / Value',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS audit_logs (
        id TEXT PRIMARY KEY,
        organization_id TEXT REFERENCES organizations(id) ON DELETE CASCADE,
        user_id TEXT REFERENCES profiles(id) ON DELETE SET NULL,
        action TEXT NOT NULL,
        resource_type TEXT NOT NULL,
        resource_id TEXT,
        details TEXT,
        ip_address TEXT,
        created_at TEXT NOT NULL
    );
    """)
    conn.commit()
    conn.close()


# Ensure database is initialized on import
init_database()
