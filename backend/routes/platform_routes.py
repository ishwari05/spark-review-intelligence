"""
================================================================================
ReviewIQ Platform / Marketplace Intelligence Routes
--------------------------------------------------------------------------------
Provides:
  - Marketplace Health & Overview KPIs
  - Seller Management (list, inspect, add/invite seller)
  - Cross-seller comparative analytics
  - Strict platform-only access enforcement
================================================================================
"""

from flask import Blueprint, jsonify, request

from backend.auth import require_auth, require_platform, require_role
from backend.repository import (
    create_invitation,
    create_organization,
    create_profile,
    get_aspect_results,
    get_issue_results,
    get_organization,
    get_platform_overview_metrics,
    get_seller_overview_metrics,
    link_platform_seller,
    list_products,
    list_reviews,
    list_sellers_for_platform,
    log_audit,
)

platform_bp = Blueprint("platform_bp", __name__)


@platform_bp.route("/api/platform/overview", methods=["GET"])
@require_auth
@require_platform
def platform_overview():
    """
    Returns live marketplace health overview, aggregates, and seller performance.
    Values are derived from real database rows and Spark ML analysis runs.
    """
    user = getattr(request, "user")
    data = get_platform_overview_metrics(user.org_id)
    return jsonify(data)


@platform_bp.route("/api/platform/sellers", methods=["GET"])
@require_auth
@require_platform
def get_sellers():
    """Returns all sellers managed by this platform organization."""
    user = getattr(request, "user")
    overview = get_platform_overview_metrics(user.org_id)
    return jsonify({
        "total": overview["total_sellers"],
        "sellers": overview["sellers"],
    })


@platform_bp.route("/api/platform/sellers", methods=["POST"])
@require_auth
@require_platform
@require_role(["owner", "admin"])
def add_seller():
    """
    Adds or invites a new seller organization to this marketplace platform.
    Fields: company_name, contact_email, role (default 'owner').
    """
    user = getattr(request, "user")
    data = request.get_json(force=True) or {}
    company_name = (data.get("company_name") or data.get("name") or "").strip()
    contact_email = (data.get("contact_email") or data.get("email") or "").strip().lower()

    if not company_name:
        return jsonify({"error": "Company Name is required."}), 400
    if not contact_email or "@" not in contact_email:
        return jsonify({"error": "A valid contact email is required."}), 400

    # 1. Create Seller Organization
    seller_org = create_organization(name=company_name, org_type="seller")

    # 2. Link Platform -> Seller in platform_sellers
    link_platform_seller(user.org_id, seller_org["id"], status="active")

    # 3. Create Invitation for seller contact
    invitation = create_invitation(
        org_id=seller_org["id"],
        email=contact_email,
        role="owner",
        invited_by=user.profile_id,
    )

    log_audit(
        org_id=user.org_id,
        action="ADD_SELLER",
        resource_type="SELLER_ORGANIZATION",
        user_id=user.profile_id,
        resource_id=seller_org["id"],
        details={"seller_name": company_name, "contact_email": contact_email},
    )

    return jsonify({
        "message": f"Seller '{company_name}' successfully added and linked to platform.",
        "seller": {
            "id": seller_org["id"],
            "name": seller_org["name"],
            "organization_type": "seller",
            "contact_email": contact_email,
        },
        "invitation": invitation,
    }), 201


@platform_bp.route("/api/platform/sellers/<seller_id>", methods=["GET"])
@require_auth
@require_platform
def get_seller_detail(seller_id: str):
    """
    Returns deep-dive intelligence for a specific seller managed by this platform.
    CRITICAL: Validates that seller_id belongs to the platform's managed sellers.
    """
    user = getattr(request, "user")
    if seller_id not in user.accessible_org_ids:
        return jsonify({"error": "Forbidden: You do not have permission to access this seller's data."}), 403

    seller = get_organization(seller_id)
    if not seller:
        return jsonify({"error": "Seller not found."}), 404

    seller_metrics = get_seller_overview_metrics(seller_id)
    products = list_products([seller_id])
    aspects = get_aspect_results([seller_id])
    issues = get_issue_results([seller_id])
    reviews = list_reviews([seller_id], limit=10)

    return jsonify({
        "seller": seller,
        "metrics": seller_metrics,
        "products": products,
        "aspects": aspects,
        "top_issues": issues,
        "recent_reviews": reviews["reviews"],
    })


@platform_bp.route("/api/platform/analytics", methods=["GET"])
@require_auth
@require_platform
def platform_cross_seller_analytics():
    """
    Returns cross-seller comparative analytics for side-by-side benchmarking.
    """
    user = getattr(request, "user")
    managed_sellers = list_sellers_for_platform(user.org_id)
    seller_ids = [s["id"] for s in managed_sellers]

    comparison = []
    for s in managed_sellers:
        sid = s["id"]
        metrics = get_seller_overview_metrics(sid)
        aspects = get_aspect_results([sid])
        comparison.append({
            "seller_id": sid,
            "seller_name": s["name"],
            "total_products": metrics["total_products"],
            "total_reviews": metrics["total_reviews"],
            "positive_pct": metrics["positive_pct"],
            "negative_pct": metrics["negative_pct"],
            "health_score": metrics["health_score"],
            "top_aspect": aspects[0]["aspect"] if aspects else "N/A",
            "top_issue": metrics["top_issues"][0]["phrase"] if metrics["top_issues"] else "None",
        })

    # Marketplace-wide aspects and issues
    market_aspects = get_aspect_results(seller_ids)
    market_issues = get_issue_results(seller_ids)

    return jsonify({
        "comparison": comparison,
        "marketplace_aspects": market_aspects,
        "marketplace_issues": market_issues,
    })
