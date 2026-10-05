"""
================================================================================
ReviewIQ Multi-Tenant Repository
--------------------------------------------------------------------------------
High-performance Data Access Layer with multi-tenant filtering,
relationship loading, and atomic updates.
================================================================================
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from backend.db import get_sqlite_conn, IS_SUPABASE_CONFIGURED, supabase_client


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------------------
# PROFILES & USERS
# ------------------------------------------------------------------------------

def create_profile(
    email: str,
    full_name: str,
    auth_user_id: Optional[str] = None,
    profile_id: Optional[str] = None
) -> Dict[str, Any]:
    pid = profile_id or str(uuid.uuid4())
    aid = auth_user_id or pid
    now = now_iso()

    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO profiles (id, auth_user_id, full_name, email, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            full_name = excluded.full_name,
            email = excluded.email,
            updated_at = excluded.updated_at
        """,
        (pid, aid, full_name, email, now, now),
    )
    conn.commit()
    conn.close()

    # Also sync to Supabase if configured
    if IS_SUPABASE_CONFIGURED and supabase_client:
        try:
            supabase_client.table("profiles").upsert({
                "id": pid,
                "auth_user_id": aid,
                "full_name": full_name,
                "email": email,
                "created_at": now,
                "updated_at": now
            }).execute()
        except Exception as e:
            print(f"[repo] Supabase sync profile error: {e}")

    return {
        "id": pid,
        "auth_user_id": aid,
        "full_name": full_name,
        "email": email,
        "created_at": now,
        "updated_at": now,
    }


def get_profile(profile_id: str) -> Optional[Dict[str, Any]]:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM profiles WHERE id = ?", (profile_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_profile_by_auth_id(auth_user_id: str) -> Optional[Dict[str, Any]]:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM profiles WHERE auth_user_id = ?", (auth_user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_profile_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM profiles WHERE LOWER(email) = LOWER(?)", (email,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


# ------------------------------------------------------------------------------
# ORGANIZATIONS & MEMBERS
# ------------------------------------------------------------------------------

def create_organization(
    name: str,
    org_type: str,
    slug: Optional[str] = None,
    org_id: Optional[str] = None
) -> Dict[str, Any]:
    oid = org_id or str(uuid.uuid4())
    org_slug = slug or name.lower().replace(" ", "-").replace("/", "-")
    now = now_iso()

    conn = get_sqlite_conn()
    cur = conn.cursor()

    # Check if slug exists to avoid collisions
    cur.execute("SELECT id FROM organizations WHERE slug = ?", (org_slug,))
    if cur.fetchone():
        org_slug = f"{org_slug}-{uuid.uuid4().hex[:6]}"

    cur.execute(
        """
        INSERT INTO organizations (id, name, organization_type, slug, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            organization_type = excluded.organization_type,
            updated_at = excluded.updated_at
        """,
        (oid, name, org_type, org_slug, now, now),
    )
    conn.commit()
    conn.close()

    if IS_SUPABASE_CONFIGURED and supabase_client:
        try:
            supabase_client.table("organizations").upsert({
                "id": oid,
                "name": name,
                "organization_type": org_type,
                "slug": org_slug,
                "created_at": now,
                "updated_at": now
            }).execute()
        except Exception as e:
            print(f"[repo] Supabase sync organization error: {e}")

    return {
        "id": oid,
        "name": name,
        "organization_type": org_type,
        "slug": org_slug,
        "created_at": now,
        "updated_at": now,
    }


def get_organization(org_id: str) -> Optional[Dict[str, Any]]:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM organizations WHERE id = ?", (org_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def add_org_member(org_id: str, user_id: str, role: str) -> Dict[str, Any]:
    mid = str(uuid.uuid4())
    now = now_iso()
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO organization_members (id, organization_id, user_id, role, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(organization_id, user_id) DO UPDATE SET
            role = excluded.role
        """,
        (mid, org_id, user_id, role, now),
    )
    conn.commit()
    conn.close()

    if IS_SUPABASE_CONFIGURED and supabase_client:
        try:
            supabase_client.table("organization_members").upsert({
                "id": mid,
                "organization_id": org_id,
                "user_id": user_id,
                "role": role,
                "created_at": now
            }).execute()
        except Exception as e:
            print(f"[repo] Supabase sync member error: {e}")

    return {"id": mid, "organization_id": org_id, "user_id": user_id, "role": role}


def get_user_memberships(user_id: str) -> List[Dict[str, Any]]:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT om.id, om.organization_id, om.role, o.name as organization_name,
               o.organization_type, o.slug
        FROM organization_members om
        JOIN organizations o ON o.id = om.organization_id
        WHERE om.user_id = ?
        """,
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def link_platform_seller(
    platform_org_id: str,
    seller_org_id: str,
    status: str = "active"
) -> Dict[str, Any]:
    lid = str(uuid.uuid4())
    now = now_iso()
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO platform_sellers (id, platform_organization_id, seller_organization_id, status, created_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(platform_organization_id, seller_organization_id) DO UPDATE SET
            status = excluded.status
        """,
        (lid, platform_org_id, seller_org_id, status, now),
    )
    conn.commit()
    conn.close()

    if IS_SUPABASE_CONFIGURED and supabase_client:
        try:
            supabase_client.table("platform_sellers").upsert({
                "id": lid,
                "platform_organization_id": platform_org_id,
                "seller_organization_id": seller_org_id,
                "status": status,
                "created_at": now
            }).execute()
        except Exception as e:
            print(f"[repo] Supabase sync platform_sellers error: {e}")

    return {
        "id": lid,
        "platform_organization_id": platform_org_id,
        "seller_organization_id": seller_org_id,
        "status": status,
    }


def list_sellers_for_platform(platform_org_id: str) -> List[Dict[str, Any]]:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT o.id, o.name, o.organization_type, o.slug, o.created_at, ps.status
        FROM platform_sellers ps
        JOIN organizations o ON o.id = ps.seller_organization_id
        WHERE ps.platform_organization_id = ?
        ORDER BY o.name ASC
        """,
        (platform_org_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_accessible_org_ids(user_id: str) -> List[str]:
    """
    Returns list of organization IDs accessible by this user.
    - If user belongs to a Seller org: returns only that Seller org ID.
    - If user belongs to a Platform org: returns the Platform org ID + all managed Seller org IDs.
    """
    memberships = get_user_memberships(user_id)
    accessible = set()
    for m in memberships:
        org_id = m["organization_id"]
        accessible.add(org_id)
        if m["organization_type"] == "platform":
            managed = list_sellers_for_platform(org_id)
            for s in managed:
                accessible.add(s["id"])
    return list(accessible)


# ------------------------------------------------------------------------------
# INVITATIONS
# ------------------------------------------------------------------------------

def create_invitation(
    org_id: str,
    email: str,
    role: str = "analyst",
    invited_by: Optional[str] = None
) -> Dict[str, Any]:
    iid = str(uuid.uuid4())
    token = str(uuid.uuid4()).replace("-", "")
    now = now_iso()
    # 7-day expiration
    expires_at = datetime.fromtimestamp(
        datetime.now(timezone.utc).timestamp() + 7 * 86400, timezone.utc
    ).isoformat()

    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO organization_invitations (id, organization_id, email, role, token, status, invited_by, expires_at, created_at)
        VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
        """,
        (iid, org_id, email, role, token, invited_by, expires_at, now),
    )
    conn.commit()
    conn.close()

    return {
        "id": iid,
        "organization_id": org_id,
        "email": email,
        "role": role,
        "token": token,
        "status": "pending",
        "expires_at": expires_at,
        "created_at": now,
    }


def list_invitations(org_id: str) -> List[Dict[str, Any]]:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM organization_invitations WHERE organization_id = ? ORDER BY created_at DESC",
        (org_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------------------
# PRODUCTS
# ------------------------------------------------------------------------------

def create_product(
    org_id: str,
    name: str,
    sku: Optional[str] = None,
    category: Optional[str] = None,
    description: Optional[str] = None,
    product_id: Optional[str] = None
) -> Dict[str, Any]:
    pid = product_id or str(uuid.uuid4())
    now = now_iso()
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO products (id, organization_id, name, sku, category, description, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            sku = excluded.sku,
            category = excluded.category,
            description = excluded.description,
            updated_at = excluded.updated_at
        """,
        (pid, org_id, name, sku or "", category or "General", description or "", now, now),
    )
    conn.commit()
    conn.close()

    if IS_SUPABASE_CONFIGURED and supabase_client:
        try:
            supabase_client.table("products").upsert({
                "id": pid,
                "organization_id": org_id,
                "name": name,
                "sku": sku or "",
                "category": category or "General",
                "description": description or "",
                "created_at": now,
                "updated_at": now
            }).execute()
        except Exception as e:
            print(f"[repo] Supabase sync product error: {e}")

    return {
        "id": pid,
        "organization_id": org_id,
        "name": name,
        "sku": sku,
        "category": category,
        "description": description,
        "created_at": now,
        "updated_at": now,
    }


def get_product(product_id: str) -> Optional[Dict[str, Any]]:
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT p.*, o.name as seller_name, o.organization_type
        FROM products p
        JOIN organizations o ON o.id = p.organization_id
        WHERE p.id = ?
        """,
        (product_id,),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def list_products(
    accessible_org_ids: List[str],
    org_id_filter: Optional[str] = None,
    category_filter: Optional[str] = None,
    search: Optional[str] = None
) -> List[Dict[str, Any]]:
    if not accessible_org_ids:
        return []

    placeholders = ",".join(["?"] * len(accessible_org_ids))
    params: List[Any] = list(accessible_org_ids)

    query = f"""
        SELECT p.*, o.name as seller_name,
               COUNT(r.id) as review_count,
               COALESCE(AVG(CASE WHEN r.sentiment = 'POSITIVE' THEN 100.0 ELSE 0.0 END), 0.0) as positive_pct,
               COALESCE(MAX(a.health_score), 80) as health_score
        FROM products p
        JOIN organizations o ON o.id = p.organization_id
        LEFT JOIN reviews r ON r.product_id = p.id
        LEFT JOIN analyses a ON a.product_id = p.id AND a.status = 'completed'
        WHERE p.organization_id IN ({placeholders})
    """

    if org_id_filter:
        query += " AND p.organization_id = ?"
        params.append(org_id_filter)

    if category_filter and category_filter != "All":
        query += " AND p.category = ?"
        params.append(category_filter)

    if search:
        query += " AND (LOWER(p.name) LIKE ? OR LOWER(p.sku) LIKE ? OR LOWER(p.description) LIKE ?)"
        term = f"%{search.lower()}%"
        params.extend([term, term, term])

    query += " GROUP BY p.id ORDER BY p.created_at DESC"

    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    results = []
    for r in rows:
        d = dict(r)
        d["review_count"] = int(d.get("review_count") or 0)
        d["positive_pct"] = round(float(d.get("positive_pct") or 0.0), 1)
        d["negative_pct"] = round(100.0 - d["positive_pct"], 1) if d["review_count"] > 0 else 0.0
        d["health_score"] = int(d.get("health_score") or 80)
        results.append(d)
    return results


# ------------------------------------------------------------------------------
# DATASETS
# ------------------------------------------------------------------------------

def create_dataset(
    org_id: str,
    name: str,
    file_name: str,
    storage_path: Optional[str] = None,
    row_count: int = 0,
    status: str = "ready",
    dataset_id: Optional[str] = None
) -> Dict[str, Any]:
    did = dataset_id or str(uuid.uuid4())
    now = now_iso()
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO datasets (id, organization_id, name, file_name, storage_path, row_count, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            row_count = excluded.row_count,
            status = excluded.status
        """,
        (did, org_id, name, file_name, storage_path or "", row_count, status, now),
    )
    conn.commit()
    conn.close()

    return {
        "id": did,
        "organization_id": org_id,
        "name": name,
        "file_name": file_name,
        "storage_path": storage_path,
        "row_count": row_count,
        "status": status,
        "created_at": now,
    }


def list_datasets(accessible_org_ids: List[str]) -> List[Dict[str, Any]]:
    if not accessible_org_ids:
        return []
    placeholders = ",".join(["?"] * len(accessible_org_ids))
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT d.*, o.name as seller_name
        FROM datasets d
        JOIN organizations o ON o.id = d.organization_id
        WHERE d.organization_id IN ({placeholders})
        ORDER BY d.created_at DESC
        """,
        accessible_org_ids,
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------------------
# REVIEWS
# ------------------------------------------------------------------------------

def create_review(
    org_id: str,
    product_id: str,
    content: str,
    title: Optional[str] = None,
    rating: Optional[int] = None,
    sentiment: Optional[str] = None,
    sentiment_score: Optional[float] = None,
    dataset_id: Optional[str] = None,
    review_date: Optional[str] = None,
    review_id: Optional[str] = None
) -> Dict[str, Any]:
    rid = review_id or str(uuid.uuid4())
    now = now_iso()
    rdate = review_date or now
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO reviews (id, organization_id, product_id, dataset_id, title, content, rating, sentiment, sentiment_score, review_date, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (rid, org_id, product_id, dataset_id, title or "", content, rating, sentiment or "POSITIVE", sentiment_score or 0.95, rdate, now),
    )
    conn.commit()
    conn.close()

    return {
        "id": rid,
        "organization_id": org_id,
        "product_id": product_id,
        "title": title,
        "content": content,
        "rating": rating,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "review_date": rdate,
    }


def bulk_create_reviews(reviews_data: List[Dict[str, Any]]) -> int:
    if not reviews_data:
        return 0
    now = now_iso()
    records = []
    for r in reviews_data:
        records.append((
            r.get("id") or str(uuid.uuid4()),
            r["organization_id"],
            r["product_id"],
            r.get("dataset_id"),
            r.get("title") or "",
            r["content"],
            r.get("rating"),
            r.get("sentiment") or "POSITIVE",
            float(r.get("sentiment_score") or 0.95),
            r.get("review_date") or now,
            now,
        ))

    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO reviews (id, organization_id, product_id, dataset_id, title, content, rating, sentiment, sentiment_score, review_date, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    conn.commit()
    count = cur.rowcount
    conn.close()
    return count


def list_reviews(
    accessible_org_ids: List[str],
    product_id: Optional[str] = None,
    org_id_filter: Optional[str] = None,
    aspect: Optional[str] = None,
    sentiment: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "default",
    page: int = 1,
    limit: int = 20
) -> Dict[str, Any]:
    if not accessible_org_ids:
        return {"total": 0, "page": page, "limit": limit, "total_pages": 1, "reviews": []}

    placeholders = ",".join(["?"] * len(accessible_org_ids))
    params: List[Any] = list(accessible_org_ids)

    where_clauses = [f"r.organization_id IN ({placeholders})"]

    if org_id_filter:
        where_clauses.append("r.organization_id = ?")
        params.append(org_id_filter)

    if product_id:
        where_clauses.append("r.product_id = ?")
        params.append(product_id)

    if sentiment and sentiment != "All":
        where_clauses.append("LOWER(r.sentiment) = LOWER(?)")
        params.append(sentiment)

    if search:
        where_clauses.append("(LOWER(r.content) LIKE ? OR LOWER(r.title) LIKE ?)")
        term = f"%{search.lower()}%"
        params.extend([term, term])

    where_sql = " AND ".join(where_clauses)

    conn = get_sqlite_conn()
    cur = conn.cursor()

    count_query = f"SELECT COUNT(*) FROM reviews r WHERE {where_sql}"
    cur.execute(count_query, params)
    total = cur.fetchone()[0]

    order_by = "r.review_date DESC"
    if sort_by == "confidence_desc":
        order_by = "r.sentiment_score DESC"
    elif sort_by == "longest":
        order_by = "LENGTH(r.content) DESC"
    elif sort_by == "shortest":
        order_by = "LENGTH(r.content) ASC"

    offset = (page - 1) * limit
    data_query = f"""
        SELECT r.*, p.name as product_name, p.sku as product_sku, o.name as seller_name
        FROM reviews r
        JOIN products p ON p.id = r.product_id
        JOIN organizations o ON o.id = r.organization_id
        WHERE {where_sql}
        ORDER BY {order_by}
        LIMIT ? OFFSET ?
    """
    cur.execute(data_query, params + [limit, offset])
    rows = cur.fetchall()
    conn.close()

    formatted_reviews = []
    for r in rows:
        d = dict(r)
        formatted_reviews.append({
            "id": d["id"],
            "product_id": d["product_id"],
            "product_name": d.get("product_name"),
            "product_sku": d.get("product_sku"),
            "seller_name": d.get("seller_name"),
            "organization_id": d["organization_id"],
            "title": d.get("title") or "",
            "review": d["content"],
            "rating": d.get("rating"),
            "sentiment": d.get("sentiment") or "POSITIVE",
            "sentiment_score": float(d.get("sentiment_score") or 0.95),
            "review_date": d.get("review_date"),
        })

    total_pages = max(1, (total + limit - 1) // limit)
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": total_pages,
        "reviews": formatted_reviews,
    }


# ------------------------------------------------------------------------------
# ANALYSES, ASPECTS, ISSUES
# ------------------------------------------------------------------------------

def create_analysis(
    org_id: str,
    product_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    total_reviews: int = 0,
    positive_reviews: int = 0,
    negative_reviews: int = 0,
    health_score: int = 80,
    accuracy: float = 0.9055,
    f1_score: float = 0.9054,
    roc_auc: float = 0.9636,
    model_name: str = "Tuned Logistic Regression (Phase 4)",
    analysis_id: Optional[str] = None
) -> Dict[str, Any]:
    aid = analysis_id or str(uuid.uuid4())
    now = now_iso()
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO analyses (id, organization_id, product_id, dataset_id, status, total_reviews, positive_reviews, negative_reviews, health_score, accuracy, f1_score, roc_auc, model_name, created_at, completed_at)
        VALUES (?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            total_reviews = excluded.total_reviews,
            positive_reviews = excluded.positive_reviews,
            negative_reviews = excluded.negative_reviews,
            health_score = excluded.health_score,
            completed_at = excluded.completed_at
        """,
        (aid, org_id, product_id, dataset_id, total_reviews, positive_reviews, negative_reviews, health_score, accuracy, f1_score, roc_auc, model_name, now, now),
    )
    conn.commit()
    conn.close()

    return {
        "id": aid,
        "organization_id": org_id,
        "product_id": product_id,
        "dataset_id": dataset_id,
        "total_reviews": total_reviews,
        "positive_reviews": positive_reviews,
        "negative_reviews": negative_reviews,
        "health_score": health_score,
        "accuracy": accuracy,
        "f1_score": f1_score,
        "roc_auc": roc_auc,
        "model_name": model_name,
    }


def save_aspect_results(
    analysis_id: str,
    org_id: str,
    product_id: Optional[str],
    aspect_list: List[Dict[str, Any]]
):
    now = now_iso()
    records = []
    for a in aspect_list:
        records.append((
            str(uuid.uuid4()),
            analysis_id,
            org_id,
            product_id,
            a["aspect"],
            int(a.get("mentions") or 0),
            int(a.get("positive_count") or 0),
            int(a.get("negative_count") or 0),
            float(a.get("positive_pct") or 0.0),
            float(a.get("negative_pct") or 0.0),
            now,
        ))

    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO aspect_results (id, analysis_id, organization_id, product_id, aspect, mentions, positive_count, negative_count, positive_pct, negative_pct, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    conn.commit()
    conn.close()


def get_aspect_results(
    accessible_org_ids: List[str],
    product_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    if not accessible_org_ids:
        return []
    placeholders = ",".join(["?"] * len(accessible_org_ids))
    params: List[Any] = list(accessible_org_ids)

    query = f"""
        SELECT aspect,
               SUM(mentions) as mentions,
               SUM(positive_count) as positive_count,
               SUM(negative_count) as negative_count
        FROM aspect_results
        WHERE organization_id IN ({placeholders})
    """
    if product_id:
        query += " AND product_id = ?"
        params.append(product_id)

    query += " GROUP BY aspect ORDER BY mentions DESC"

    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    results = []
    for r in rows:
        m = int(r["mentions"] or 0)
        pos = int(r["positive_count"] or 0)
        neg = int(r["negative_count"] or 0)
        pos_pct = round((pos / m * 100.0), 1) if m > 0 else 50.0
        neg_pct = round(100.0 - pos_pct, 1)
        results.append({
            "aspect": r["aspect"],
            "mentions": m,
            "positive_count": pos,
            "negative_count": neg,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
        })
    return results


def save_issue_results(
    analysis_id: str,
    org_id: str,
    product_id: Optional[str],
    issues_list: List[Dict[str, Any]]
):
    now = now_iso()
    records = []
    for iss in issues_list:
        records.append((
            str(uuid.uuid4()),
            analysis_id,
            org_id,
            product_id,
            iss.get("phrase") or "",
            iss.get("aspect") or "General Dissatisfaction",
            int(iss.get("frequency") or 0),
            float(iss.get("percentage") or 0.0),
            iss.get("priority") or "MEDIUM",
            float(iss.get("negative_pct") or 80.0),
            now,
        ))

    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.executemany(
        """
        INSERT INTO issue_results (id, analysis_id, organization_id, product_id, phrase, aspect, frequency, percentage, priority, negative_pct, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    conn.commit()
    conn.close()


def get_issue_results(
    accessible_org_ids: List[str],
    product_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    if not accessible_org_ids:
        return []
    placeholders = ",".join(["?"] * len(accessible_org_ids))
    params: List[Any] = list(accessible_org_ids)

    query = f"""
        SELECT phrase, aspect,
               SUM(frequency) as frequency,
               AVG(percentage) as percentage,
               MAX(priority) as priority,
               AVG(negative_pct) as negative_pct
        FROM issue_results
        WHERE organization_id IN ({placeholders})
    """
    if product_id:
        query += " AND product_id = ?"
        params.append(product_id)

    query += " GROUP BY phrase, aspect ORDER BY frequency DESC LIMIT 15"

    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "phrase": r["phrase"],
            "aspect": r["aspect"],
            "frequency": int(r["frequency"] or 0),
            "percentage": round(float(r["percentage"] or 0.0), 2),
            "priority": r["priority"] or "MEDIUM",
            "negative_pct": round(float(r["negative_pct"] or 80.0), 1),
        })
    return results


# ------------------------------------------------------------------------------
# MULTI-TENANT KPI AGGREGATORS (PLATFORM & SELLER)
# ------------------------------------------------------------------------------

def get_platform_overview_metrics(platform_org_id: str) -> Dict[str, Any]:
    """Calculates live aggregated marketplace intelligence across all managed sellers."""
    managed_sellers = list_sellers_for_platform(platform_org_id)
    seller_ids = [s["id"] for s in managed_sellers]

    if not seller_ids:
        return {
            "total_sellers": 0,
            "total_products": 0,
            "total_reviews": 0,
            "overall_positive_pct": 0.0,
            "overall_negative_pct": 0.0,
            "marketplace_health": 80,
            "sellers": [],
        }

    placeholders = ",".join(["?"] * len(seller_ids))
    conn = get_sqlite_conn()
    cur = conn.cursor()

    # Aggregate total products
    cur.execute(f"SELECT COUNT(*) FROM products WHERE organization_id IN ({placeholders})", seller_ids)
    total_products = cur.fetchone()[0]

    # Aggregate total reviews and sentiment
    cur.execute(
        f"""
        SELECT COUNT(*) as total_revs,
               SUM(CASE WHEN sentiment = 'POSITIVE' THEN 1 ELSE 0 END) as pos_count,
               SUM(CASE WHEN sentiment = 'NEGATIVE' THEN 1 ELSE 0 END) as neg_count
        FROM reviews
        WHERE organization_id IN ({placeholders})
        """,
        seller_ids,
    )
    rev_row = cur.fetchone()
    total_reviews = int(rev_row["total_revs"] or 0)
    pos_count = int(rev_row["pos_count"] or 0)
    neg_count = int(rev_row["neg_count"] or 0)

    if total_reviews > 0:
        pos_pct = round((pos_count / total_reviews) * 100.0, 1)
        neg_pct = round((neg_count / total_reviews) * 100.0, 1)
    else:
        pos_pct = 85.0
        neg_pct = 15.0

    # Per-Seller breakdown
    seller_breakdown = []
    weighted_health = 0.0
    total_weight = 0

    for s in managed_sellers:
        sid = s["id"]
        # Product count
        cur.execute("SELECT COUNT(*) FROM products WHERE organization_id = ?", (sid,))
        prod_count = cur.fetchone()[0]

        # Review stats
        cur.execute(
            """
            SELECT COUNT(*) as count,
                   SUM(CASE WHEN sentiment = 'POSITIVE' THEN 1 ELSE 0 END) as pos
            FROM reviews
            WHERE organization_id = ?
            """,
            (sid,),
        )
        s_rev = cur.fetchone()
        s_count = int(s_rev["count"] or 0)
        s_pos = int(s_rev["pos"] or 0)

        s_pos_pct = round((s_pos / s_count * 100.0), 1) if s_count > 0 else 82.0
        s_neg_pct = round(100.0 - s_pos_pct, 1) if s_count > 0 else 18.0

        # Health score from latest analysis or derived from pos_pct
        cur.execute(
            "SELECT health_score FROM analyses WHERE organization_id = ? ORDER BY created_at DESC LIMIT 1",
            (sid,),
        )
        a_row = cur.fetchone()
        s_health = a_row["health_score"] if a_row else round(s_pos_pct * 0.95)

        weighted_health += s_health * max(1, s_count)
        total_weight += max(1, s_count)

        seller_breakdown.append({
            "id": s["id"],
            "name": s["name"],
            "slug": s.get("slug"),
            "status": s.get("status", "active"),
            "products_count": prod_count,
            "reviews_count": s_count,
            "positive_pct": s_pos_pct,
            "negative_pct": s_neg_pct,
            "health_score": s_health,
        })

    conn.close()

    marketplace_health = round(weighted_health / total_weight) if total_weight > 0 else 80

    return {
        "total_sellers": len(managed_sellers),
        "total_products": total_products,
        "total_reviews": total_reviews,
        "overall_positive_pct": pos_pct,
        "overall_negative_pct": neg_pct,
        "marketplace_health": marketplace_health,
        "sellers": seller_breakdown,
    }


def get_seller_overview_metrics(seller_org_id: str) -> Dict[str, Any]:
    """Calculates private product intelligence for a single seller organization."""
    conn = get_sqlite_conn()
    cur = conn.cursor()

    # Product count
    cur.execute("SELECT COUNT(*) FROM products WHERE organization_id = ?", (seller_org_id,))
    total_products = cur.fetchone()[0]

    # Review count and sentiment
    cur.execute(
        """
        SELECT COUNT(*) as count,
               SUM(CASE WHEN sentiment = 'POSITIVE' THEN 1 ELSE 0 END) as pos,
               SUM(CASE WHEN sentiment = 'NEGATIVE' THEN 1 ELSE 0 END) as neg
        FROM reviews
        WHERE organization_id = ?
        """,
        (seller_org_id,),
    )
    rev_row = cur.fetchone()
    total_reviews = int(rev_row["count"] or 0)
    pos_count = int(rev_row["pos"] or 0)
    neg_count = int(rev_row["neg"] or 0)

    if total_reviews > 0:
        pos_pct = round((pos_count / total_reviews) * 100.0, 1)
        neg_pct = round((neg_count / total_reviews) * 100.0, 1)
    else:
        pos_pct = 84.0
        neg_pct = 16.0

    # Seller health score
    cur.execute(
        "SELECT health_score FROM analyses WHERE organization_id = ? ORDER BY created_at DESC LIMIT 1",
        (seller_org_id,),
    )
    a_row = cur.fetchone()
    health_score = a_row["health_score"] if a_row else round(pos_pct)

    # Product performance breakdown
    cur.execute(
        """
        SELECT p.id, p.name, p.sku, p.category,
               COUNT(r.id) as review_count,
               SUM(CASE WHEN r.sentiment = 'POSITIVE' THEN 1 ELSE 0 END) as pos_count
        FROM products p
        LEFT JOIN reviews r ON r.product_id = p.id
        WHERE p.organization_id = ?
        GROUP BY p.id
        ORDER BY review_count DESC
        """,
        (seller_org_id,),
    )
    prod_rows = cur.fetchall()
    products_perf = []
    for p in prod_rows:
        rc = int(p["review_count"] or 0)
        pc = int(p["pos_count"] or 0)
        p_pos = round((pc / rc * 100.0), 1) if rc > 0 else 84.0
        p_neg = round(100.0 - p_pos, 1) if rc > 0 else 16.0
        products_perf.append({
            "id": p["id"],
            "name": p["name"],
            "sku": p["sku"],
            "category": p["category"],
            "reviews_count": rc,
            "positive_pct": p_pos,
            "negative_pct": p_neg,
            "health_score": round(p_pos),
        })

    # Recent reviews (last 5)
    cur.execute(
        """
        SELECT r.*, p.name as product_name
        FROM reviews r
        JOIN products p ON p.id = r.product_id
        WHERE r.organization_id = ?
        ORDER BY r.review_date DESC
        LIMIT 5
        """,
        (seller_org_id,),
    )
    recent_reviews = [dict(r) for r in cur.fetchall()]

    conn.close()

    # Top issues for this seller
    top_issues = get_issue_results([seller_org_id])[:6]

    return {
        "total_products": total_products,
        "total_reviews": total_reviews,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "health_score": health_score,
        "product_performance": products_perf,
        "top_issues": top_issues,
        "recent_reviews": recent_reviews,
    }


# ------------------------------------------------------------------------------
# AUDIT LOGS & USER PREFERENCES
# ------------------------------------------------------------------------------

def log_audit(
    org_id: str,
    action: str,
    resource_type: str,
    user_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None
):
    lid = str(uuid.uuid4())
    now = now_iso()
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO audit_logs (id, organization_id, user_id, action, resource_type, resource_id, details, ip_address, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (lid, org_id, user_id, action, resource_type, resource_id, json.dumps(details or {}), ip_address, now),
    )
    conn.commit()
    conn.close()


def get_audit_logs(accessible_org_ids: List[str], limit: int = 50) -> List[Dict[str, Any]]:
    if not accessible_org_ids:
        return []
    placeholders = ",".join(["?"] * len(accessible_org_ids))
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT a.*, p.full_name as user_name, p.email as user_email, o.name as org_name
        FROM audit_logs a
        LEFT JOIN profiles p ON p.id = a.user_id
        LEFT JOIN organizations o ON o.id = a.organization_id
        WHERE a.organization_id IN ({placeholders})
        ORDER BY a.created_at DESC
        LIMIT ?
        """,
        accessible_org_ids + [limit],
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------------------
# INITIAL SEEDING HELPER (DEVELOPMENT ONLY)
# ------------------------------------------------------------------------------

def seed_default_data_if_empty():
    """Seeds 1 platform and 2 sellers with products, reviews, and analyses if empty."""
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM organizations")
    count = cur.fetchone()[0]
    conn.close()

    if count > 0:
        return

    print("[repo] Database is empty. Seeding initial DEVELOPMENT DATA (1 Platform, 2 Sellers)...")
    
    # 1. Platform
    platform_org = create_organization(
        "Apex Global Marketplace",
        "platform",
        slug="apex-marketplace",
        org_id="00000000-0000-0000-0000-000000000001",
    )
    # Platform Admin Profile
    platform_admin = create_profile(
        email="platform_admin@reviewiq.io",
        full_name="Alex Platform Admin",
        auth_user_id="auth-platform-admin-01",
        profile_id="p0000000-0000-0000-0000-000000000001",
    )
    add_org_member(platform_org["id"], platform_admin["id"], "owner")

    # 2. Seller A: Aura Sound
    seller_aura = create_organization(
        "Aura Sound Technologies",
        "seller",
        slug="aura-sound",
        org_id="00000000-0000-0000-0000-000000000002",
    )
    aura_owner = create_profile(
        email="sarah@aurasound.com",
        full_name="Sarah Chen",
        auth_user_id="auth-seller-aura-01",
        profile_id="p0000000-0000-0000-0000-000000000002",
    )
    add_org_member(seller_aura["id"], aura_owner["id"], "owner")

    # Seller A Viewer (for permission matrix testing)
    aura_viewer = create_profile(
        email="viewer@aurasound.com",
        full_name="Mark Viewer",
        auth_user_id="auth-seller-aura-viewer",
        profile_id="p0000000-0000-0000-0000-000000000004",
    )
    add_org_member(seller_aura["id"], aura_viewer["id"], "viewer")

    # 3. Seller B: Lumina Smart Devices
    seller_lumina = create_organization(
        "Lumina Smart Devices",
        "seller",
        slug="lumina-devices",
        org_id="00000000-0000-0000-0000-000000000003",
    )
    lumina_owner = create_profile(
        email="david@luminadevices.com",
        full_name="David Rossi",
        auth_user_id="auth-seller-lumina-01",
        profile_id="p0000000-0000-0000-0000-000000000003",
    )
    add_org_member(seller_lumina["id"], lumina_owner["id"], "owner")

    # 4. Link Platform -> Managed Sellers
    link_platform_seller(platform_org["id"], seller_aura["id"])
    link_platform_seller(platform_org["id"], seller_lumina["id"])

    # 5. Products for Seller A
    p1 = create_product(
        seller_aura["id"],
        "Aura ANC Wireless Headphones",
        sku="AURA-HP-01",
        category="Audio & Headphones",
        description="Flagship noise-canceling over-ear wireless headphones with 40mm drivers.",
        product_id="10000000-0000-0000-0000-000000000001",
    )
    p2 = create_product(
        seller_aura["id"],
        "Aura Pulse Earbuds Pro",
        sku="AURA-EB-02",
        category="Audio & Headphones",
        description="True wireless sports earbuds with IPX7 water resistance and deep bass.",
        product_id="10000000-0000-0000-0000-000000000002",
    )

    # Products for Seller B
    p3 = create_product(
        seller_lumina["id"],
        "Lumina Smart Ambient Lamp",
        sku="LUM-LAMP-01",
        category="Smart Home",
        description="RGB smart table lamp compatible with Alexa and Google Assistant.",
        product_id="10000000-0000-0000-0000-000000000003",
    )
    p4 = create_product(
        seller_lumina["id"],
        "Lumina Wi-Fi Air Purifier",
        sku="LUM-AIR-02",
        category="Home Appliances",
        description="HEPA H13 air purifier with real-time AQI monitoring and app control.",
        product_id="10000000-0000-0000-0000-000000000004",
    )

    # 6. Datasets
    d1 = create_dataset(
        seller_aura["id"],
        "Aura Headphones Q3 Reviews",
        "aura_q3_reviews.csv",
        row_count=8421,
        dataset_id="20000000-0000-0000-0000-000000000001",
    )
    d2 = create_dataset(
        seller_lumina["id"],
        "Lumina Smart Home Reviews",
        "lumina_annual_reviews.csv",
        row_count=6120,
        dataset_id="20000000-0000-0000-0000-000000000002",
    )

    # 7. Reviews for Seller A
    reviews_a = [
        {
            "id": "40000000-0000-0000-0000-000000000001",
            "organization_id": seller_aura["id"],
            "product_id": p1["id"],
            "dataset_id": d1["id"],
            "title": "Phenomenal audio experience!",
            "content": "The noise cancellation is stellar on daily commute flights. Battery lasts over 30 hours without recharging.",
            "rating": 5,
            "sentiment": "POSITIVE",
            "sentiment_score": 0.98,
        },
        {
            "id": "40000000-0000-0000-0000-000000000002",
            "organization_id": seller_aura["id"],
            "product_id": p1["id"],
            "dataset_id": d1["id"],
            "title": "Great value and comfort",
            "content": "Earcups are super plush. For the price, you cannot beat the rich soundstage.",
            "rating": 5,
            "sentiment": "POSITIVE",
            "sentiment_score": 0.94,
        },
        {
            "id": "40000000-0000-0000-0000-000000000003",
            "organization_id": seller_aura["id"],
            "product_id": p1["id"],
            "dataset_id": d1["id"],
            "title": "Disappointed with plastic hinge",
            "content": "The headband plastic feels fragile and cracked after 3 months of normal use.",
            "rating": 2,
            "sentiment": "NEGATIVE",
            "sentiment_score": 0.89,
        },
        {
            "id": "40000000-0000-0000-0000-000000000004",
            "organization_id": seller_aura["id"],
            "product_id": p1["id"],
            "dataset_id": d1["id"],
            "title": "Charging cable missing in box",
            "content": "Sound is good but customer service took two weeks to send a replacement charging cable.",
            "rating": 3,
            "sentiment": "NEGATIVE",
            "sentiment_score": 0.78,
        },
    ]
    bulk_create_reviews(reviews_a)

    # Reviews for Seller B
    reviews_b = [
        {
            "id": "40000000-0000-0000-0000-000000000005",
            "organization_id": seller_lumina["id"],
            "product_id": p3["id"],
            "dataset_id": d2["id"],
            "title": "Beautiful colors and transitions",
            "content": "Syncs smoothly with my smart home setup. Soft bedtime warm light is perfect.",
            "rating": 5,
            "sentiment": "POSITIVE",
            "sentiment_score": 0.96,
        },
        {
            "id": "40000000-0000-0000-0000-000000000006",
            "organization_id": seller_lumina["id"],
            "product_id": p3["id"],
            "dataset_id": d2["id"],
            "title": "Wi-Fi connection drops often",
            "content": "The app keeps losing connection to the lamp every few days. Pairing setup is frustrating.",
            "rating": 2,
            "sentiment": "NEGATIVE",
            "sentiment_score": 0.88,
        },
    ]
    bulk_create_reviews(reviews_b)

    # 8. Analyses
    a1 = create_analysis(
        seller_aura["id"],
        p1["id"],
        d1["id"],
        total_reviews=8421,
        positive_reviews=7074,
        negative_reviews=1347,
        health_score=84,
        analysis_id="30000000-0000-0000-0000-000000000001",
    )
    save_aspect_results(a1["id"], seller_aura["id"], p1["id"], [
        {"aspect": "Price / Value", "mentions": 3120, "positive_count": 2808, "negative_count": 312, "positive_pct": 90.0, "negative_pct": 10.0},
        {"aspect": "Battery", "mentions": 2410, "positive_count": 2120, "negative_count": 290, "positive_pct": 88.0, "negative_pct": 12.0},
        {"aspect": "Build Quality", "mentions": 1890, "positive_count": 1361, "negative_count": 529, "positive_pct": 72.0, "negative_pct": 28.0},
        {"aspect": "Customer Support", "mentions": 1001, "positive_count": 786, "negative_count": 215, "positive_pct": 78.5, "negative_pct": 21.5},
    ])
    save_issue_results(a1["id"], seller_aura["id"], p1["id"], [
        {"phrase": "plastic hinge crack", "aspect": "Build Quality", "frequency": 88, "percentage": 1.04, "priority": "HIGH", "negative_pct": 89.2},
        {"phrase": "bluetooth disconnects", "aspect": "Connectivity", "frequency": 64, "percentage": 0.76, "priority": "MEDIUM", "negative_pct": 82.5},
        {"phrase": "slow charging speed", "aspect": "Battery", "frequency": 42, "percentage": 0.50, "priority": "MEDIUM", "negative_pct": 74.0},
    ])

    a2 = create_analysis(
        seller_lumina["id"],
        p3["id"],
        d2["id"],
        total_reviews=6120,
        positive_reviews=4406,
        negative_reviews=1714,
        health_score=72,
        analysis_id="30000000-0000-0000-0000-000000000002",
    )
    save_aspect_results(a2["id"], seller_lumina["id"], p3["id"], [
        {"aspect": "Design & Aesthetics", "mentions": 2200, "positive_count": 1980, "negative_count": 220, "positive_pct": 90.0, "negative_pct": 10.0},
        {"aspect": "App & Connectivity", "mentions": 1840, "positive_count": 1104, "negative_count": 736, "positive_pct": 60.0, "negative_pct": 40.0},
        {"aspect": "Price / Value", "mentions": 1400, "positive_count": 1050, "negative_count": 350, "positive_pct": 75.0, "negative_pct": 25.0},
    ])
    save_issue_results(a2["id"], seller_lumina["id"], p3["id"], [
        {"phrase": "wifi drops connection", "aspect": "Connectivity", "frequency": 112, "percentage": 1.83, "priority": "HIGH", "negative_pct": 92.0},
        {"phrase": "app pairing failed", "aspect": "Software", "frequency": 75, "percentage": 1.22, "priority": "HIGH", "negative_pct": 88.5},
    ])

    log_audit(platform_org["id"], "SYSTEM_INIT", "ORGANIZATION", user_id=platform_admin["id"], details={"note": "Development multi-tenant seed completed"})
    print("[repo] Seed completed successfully.")


# Ensure seed runs if SQLite is fresh
seed_default_data_if_empty()
