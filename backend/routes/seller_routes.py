"""
================================================================================
ReviewIQ Seller / Brand Intelligence Routes
--------------------------------------------------------------------------------
Provides:
  - Seller Product Intelligence Overview
  - Private KPI Cards & Product Performance
  - Top Issues & Customer Feedback for Seller's Products
  - Strict seller isolation: Never exposes data to other sellers
================================================================================
"""

from flask import Blueprint, jsonify, request

from backend.auth import require_auth, require_seller
from backend.repository import (
    get_aspect_results,
    get_issue_results,
    get_seller_overview_metrics,
    list_products,
    list_reviews,
)

seller_bp = Blueprint("seller_bp", __name__)


@seller_bp.route("/api/seller/overview", methods=["GET"])
@require_auth
@require_seller
def seller_overview():
    """
    Returns private product intelligence for the authenticated seller organization.
    Never exposes any other seller's data.
    """
    user = getattr(request, "user")
    data = get_seller_overview_metrics(user.org_id)
    return jsonify({
        "seller_name": user.org_name,
        "user_name": user.full_name,
        "metrics": data,
    })


@seller_bp.route("/api/seller/analytics", methods=["GET"])
@require_auth
@require_seller
def seller_analytics():
    """Returns aggregated aspect and issue intelligence for this seller only."""
    user = getattr(request, "user")
    aspects = get_aspect_results([user.org_id])
    issues = get_issue_results([user.org_id])
    return jsonify({
        "aspects": aspects,
        "issues": issues,
    })
