"""
================================================================================
ReviewIQ Authentication & User Routes
--------------------------------------------------------------------------------
Handles:
  - Signup (Platform vs Seller)
  - Login (Supabase Auth / Local Dev Auth)
  - Current User Profile & Organization Context (/api/auth/me)
  - Safe public config (/api/config)
================================================================================
"""

import os
from flask import Blueprint, jsonify, request

from backend.auth import (
    generate_dev_token,
    register_new_account,
    require_auth,
    resolve_auth_user,
)
from backend.db import IS_SUPABASE_CONFIGURED, SUPABASE_ANON_KEY, SUPABASE_URL, supabase_client
from backend.repository import (
    get_profile_by_email,
    get_user_memberships,
    log_audit,
)

auth_bp = Blueprint("auth_bp", __name__)


@auth_bp.route("/api/config", methods=["GET"])
def get_public_config():
    """
    Returns public client configuration.
    CRITICAL SECURITY RULE: NEVER expose SUPABASE_SERVICE_ROLE_KEY!
    """
    return jsonify({
        "supabase_url": SUPABASE_URL if IS_SUPABASE_CONFIGURED else "",
        "supabase_anon_key": SUPABASE_ANON_KEY if IS_SUPABASE_CONFIGURED else "",
        "is_supabase_configured": IS_SUPABASE_CONFIGURED,
        "app_name": "ReviewIQ",
        "version": "2.0.0-multi-tenant",
    })


@auth_bp.route("/api/auth/signup", methods=["POST"])
def signup():
    """
    Registers a new Platform or Seller organization and owner account.
    Validates work email, full name, organization name, and organization type.
    """
    data = request.get_json(force=True) or {}
    full_name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()
    org_name = (data.get("organization_name") or "").strip()
    org_type = (data.get("organization_type") or "seller").strip().lower()

    if not full_name:
        return jsonify({"error": "Full Name is required."}), 400
    if not email or "@" not in email:
        return jsonify({"error": "A valid work email is required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters long."}), 400
    if not org_name:
        return jsonify({"error": "Company / Organization Name is required."}), 400
    if org_type not in ("platform", "seller"):
        return jsonify({"error": "Invalid account type. Choose 'platform' or 'seller'."}), 400

    try:
        result = register_new_account(
            email=email,
            password=password,
            full_name=full_name,
            org_name=org_name,
            org_type=org_type,
        )
        return jsonify({
            "message": "Account created successfully.",
            "token": result["token"],
            "user": result["user"],
        }), 201
    except ValueError as e:
        return jsonify({"error": str(e)}), 409
    except Exception as e:
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500


@auth_bp.route("/api/auth/login", methods=["POST"])
def login():
    """
    Authenticates a user via Supabase Auth (if live) or local verified store.
    Returns access token, user profile, and organization context.
    """
    data = request.get_json(force=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email:
        return jsonify({"error": "Email is required."}), 400
    if not password:
        return jsonify({"error": "Password is required."}), 400

    token = None
    auth_user_id = None

    # 1. Try Supabase Auth if configured
    if IS_SUPABASE_CONFIGURED and supabase_client:
        try:
            auth_res = supabase_client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            if auth_res and auth_res.session:
                token = auth_res.session.access_token
                auth_user_id = str(auth_res.user.id)
        except Exception as e:
            # Check if invalid credentials or unconfirmed
            err_msg = str(e).lower()
            if "invalid login credentials" in err_msg or "invalid credentials" in err_msg:
                return jsonify({"error": "Invalid email or password."}), 401

    # 2. Local verified store fallback
    if not token:
        profile = get_profile_by_email(email)
        if not profile:
            return jsonify({"error": "Invalid email or password."}), 401

        # Generate JWT token for this user
        token = generate_dev_token(
            auth_user_id=profile.get("auth_user_id") or profile["id"],
            email=profile["email"],
        )

    # Resolve full user context
    user_context = resolve_auth_user(token)
    if not user_context:
        return jsonify({"error": "User profile or organization not found for this account."}), 404

    log_audit(
        org_id=user_context.org_id,
        action="LOGIN",
        resource_type="USER",
        user_id=user_context.profile_id,
        ip_address=request.remote_addr,
    )

    return jsonify({
        "message": "Login successful.",
        "token": token,
        "user": user_context.to_dict(),
    })


@auth_bp.route("/api/auth/me", methods=["GET"])
@require_auth
def get_current_user():
    """
    Returns the authenticated user's profile, organization, role, and
    permissions matrix. Used by the frontend on application initialization.
    """
    user = getattr(request, "user")
    return jsonify({
        "authenticated": True,
        "user": user.to_dict(),
        "permissions": {
            "is_platform": user.org_type == "platform",
            "is_seller": user.org_type == "seller",
            "can_upload": user.can_modify(),
            "can_analyze": user.can_modify(),
            "can_manage_team": user.is_admin(),
            "role": user.role,
        }
    })


@auth_bp.route("/api/auth/logout", methods=["POST"])
def logout():
    """Logs out user and logs audit action."""
    # Supabase logout if configured
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if token and IS_SUPABASE_CONFIGURED and supabase_client:
        try:
            supabase_client.auth.sign_out()
        except Exception:
            pass

    return jsonify({"message": "Successfully logged out."})
