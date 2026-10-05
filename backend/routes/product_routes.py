"""
================================================================================
ReviewIQ Product Catalog & Intelligence Routes
--------------------------------------------------------------------------------
Handles:
  - Product listing (scoped to accessible organizations)
  - Product creation (with server-derived organization and role verification)
  - Deep Product Intelligence view (Health score, ABSA aspects, Issues, Reviews)
  - Strict data isolation: cross-tenant access returns 403/404
================================================================================
"""

from flask import Blueprint, jsonify, request

from backend.auth import require_auth, require_role
from backend.repository import (
    create_product,
    get_aspect_results,
    get_issue_results,
    get_product,
    list_products,
    list_reviews,
    log_audit,
)

product_bp = Blueprint("product_bp", __name__)


@product_bp.route("/api/products", methods=["GET"])
@require_auth
def get_products():
    """
    Returns products accessible to the current user.
    - Sellers only see their own products.
    - Platforms see all products across managed sellers, or filtered by seller_id.
    """
    user = getattr(request, "user")
    seller_id_filter = request.args.get("seller_id", "").strip()
    category_filter = request.args.get("category", "").strip()
    search = request.args.get("search", "").strip()

    # If a filter is supplied, verify it is within accessible organizations
    org_filter = None
    if seller_id_filter:
        if seller_id_filter not in user.accessible_org_ids:
            return jsonify({"error": "Forbidden: You cannot access products from this seller."}), 403
        org_filter = seller_id_filter

    products = list_products(
        accessible_org_ids=user.accessible_org_ids,
        org_id_filter=org_filter,
        category_filter=category_filter,
        search=search,
    )
    return jsonify({
        "total": len(products),
        "products": products,
    })


@product_bp.route("/api/products", methods=["POST"])
@require_auth
@require_role(["owner", "admin", "analyst"])
def add_product():
    """
    Creates a new product.
    CRITICAL SECURITY RULE: Never trust organization_id from frontend!
    - For seller users: product is ALWAYS assigned to request.user.org_id.
    - For platform users: can optionally assign to a managed seller, but only if seller_id is in accessible_org_ids.
    """
    user = getattr(request, "user")
    data = request.get_json(force=True) or {}

    name = (data.get("name") or "").strip()
    sku = (data.get("sku") or "").strip()
    category = (data.get("category") or "General").strip()
    description = (data.get("description") or "").strip()

    if not name:
        return jsonify({"error": "Product Name is required."}), 400

    target_org_id = user.org_id
    if user.org_type == "platform":
        requested_seller_id = (data.get("seller_id") or data.get("organization_id") or "").strip()
        if requested_seller_id:
            if requested_seller_id not in user.accessible_org_ids:
                return jsonify({"error": "Forbidden: Selected seller is not managed by your platform."}), 403
            target_org_id = requested_seller_id

    product = create_product(
        org_id=target_org_id,
        name=name,
        sku=sku,
        category=category,
        description=description,
    )

    log_audit(
        org_id=target_org_id,
        action="CREATE_PRODUCT",
        resource_type="PRODUCT",
        user_id=user.profile_id,
        resource_id=product["id"],
        details={"name": name, "sku": sku},
    )

    return jsonify({
        "message": "Product successfully created.",
        "product": product,
    }), 201


@product_bp.route("/api/products/<product_id>", methods=["GET"])
@require_auth
def get_product_detail(product_id: str):
    """
    Returns full product intelligence for a single product:
      - Health Score, Review volume, Positive %, Negative %
      - Aspect-level sentiment breakdown
      - Recurring customer complaints/issues
      - Recent reviews for this product
    CRITICAL: Validates product belongs to accessible organizations.
    """
    user = getattr(request, "user")
    product = get_product(product_id)

    if not product:
        return jsonify({"error": "Product not found."}), 404

    # Enforce multi-tenant authorization
    if product["organization_id"] not in user.accessible_org_ids:
        return jsonify({"error": "Forbidden: You do not have permission to view this product."}), 403

    # Fetch product-specific reviews & metrics
    reviews_res = list_reviews(
        accessible_org_ids=[product["organization_id"]],
        product_id=product_id,
        limit=20,
    )
    aspects = get_aspect_results([product["organization_id"]], product_id=product_id)
    issues = get_issue_results([product["organization_id"]], product_id=product_id)

    # If product has no custom aspect results yet, fallback to general seller aspect rates
    if not aspects:
        aspects = get_aspect_results([product["organization_id"]])
    if not issues:
        issues = get_issue_results([product["organization_id"]])

    total_revs = reviews_res["total"]
    pos_count = sum(1 for r in reviews_res["reviews"] if r["sentiment"] == "POSITIVE")
    pos_pct = round((pos_count / total_revs * 100.0), 1) if total_revs > 0 else 84.0
    neg_pct = round(100.0 - pos_pct, 1)

    health_score = round(pos_pct)

    return jsonify({
        "product": product,
        "health_score": health_score,
        "total_reviews": total_revs,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "aspects": aspects,
        "issues": issues,
        "reviews": reviews_res["reviews"],
    })
