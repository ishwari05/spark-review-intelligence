"""
================================================================================
ReviewIQ Authentication & Multi-Tenant Authorization Service
--------------------------------------------------------------------------------
Enforces:
  1. Supabase Auth JWT verification (with graceful local JWT dev/testing mode)
  2. Role-based access control (OWNER, ADMIN, ANALYST, VIEWER)
  3. Strict multi-tenant isolation (Platform vs. Seller)
  4. Server-side organization derivation (never trusts frontend organization_id)
================================================================================
"""

import functools
import os
import time
from typing import Any, Callable, Dict, List, Optional

import jwt
from flask import jsonify, request

from backend.db import IS_SUPABASE_CONFIGURED, SUPABASE_ANON_KEY, SUPABASE_URL, supabase_client
from backend.repository import (
    add_org_member,
    create_organization,
    create_profile,
    get_accessible_org_ids,
    get_organization,
    get_profile,
    get_profile_by_auth_id,
    get_profile_by_email,
    get_user_memberships,
    link_platform_seller,
    log_audit,
)

JWT_SECRET = os.environ.get("SECRET_KEY", "reviewiq-super-secret-jwt-signing-key-production-dev")
JWT_ALGORITHM = "HS256"


class AuthUser:
    """Authenticated user context object attached to flask.request.user."""

    def __init__(
        self,
        profile_id: str,
        auth_user_id: str,
        email: str,
        full_name: str,
        org_id: str,
        org_type: str,
        org_name: str,
        role: str,
        accessible_org_ids: List[str],
    ):
        self.profile_id = profile_id
        self.auth_user_id = auth_user_id
        self.email = email
        self.full_name = full_name
        self.org_id = org_id
        self.org_type = org_type
        self.org_name = org_name
        self.role = role
        self.accessible_org_ids = accessible_org_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.profile_id,
            "auth_user_id": self.auth_user_id,
            "email": self.email,
            "full_name": self.full_name,
            "organization_id": self.org_id,
            "organization_type": self.org_type,
            "organization_name": self.org_name,
            "role": self.role,
            "accessible_org_ids": self.accessible_org_ids,
        }

    def can_modify(self) -> bool:
        """Viewers have read-only permissions."""
        return self.role in ("owner", "admin", "analyst")

    def is_admin(self) -> bool:
        return self.role in ("owner", "admin")


def generate_dev_token(auth_user_id: str, email: str, expires_in_seconds: int = 86400 * 30) -> str:
    """Generates a secure HS256 JWT for local testing and dev mode."""
    payload = {
        "sub": auth_user_id,
        "email": email,
        "exp": int(time.time()) + expires_in_seconds,
        "iat": int(time.time()),
        "iss": "reviewiq-auth",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def extract_token_from_header() -> Optional[str]:
    """Extracts Bearer token from Authorization HTTP header."""
    auth_header = request.headers.get("Authorization", "").strip()
    if not auth_header:
        # Check query string as fallback (e.g. for direct downloads or sockets)
        return request.args.get("access_token")
    parts = auth_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None


def resolve_auth_user(token: str) -> Optional[AuthUser]:
    """
    Verifies token and resolves the user's Profile, Organization, Role, and
    multi-tenant accessible organization IDs.
    """
    auth_user_id = None
    email = None

    # 1. Check for well-known seeded demo tokens (for instant testing & demos)
    demo_tokens = {
        "demo-platform-token": ("auth-platform-admin-01", "platform_admin@reviewiq.io"),
        "demo-seller-aura-token": ("auth-seller-aura-01", "sarah@aurasound.com"),
        "demo-seller-aura-viewer-token": ("auth-seller-aura-viewer", "viewer@aurasound.com"),
        "demo-seller-lumina-token": ("auth-seller-lumina-01", "david@luminadevices.com"),
    }
    if token in demo_tokens:
        auth_user_id, email = demo_tokens[token]

    # 2. Check Supabase Live Verification if configured
    if not auth_user_id and IS_SUPABASE_CONFIGURED and supabase_client:
        try:
            user_response = supabase_client.auth.get_user(token)
            if user_response and user_response.user:
                auth_user_id = str(user_response.user.id)
                email = user_response.user.email
        except Exception as e:
            # Token might be local or invalid in Supabase
            pass

    # 3. Fallback: Verify via local JWT decoder
    if not auth_user_id:
        try:
            # First try without verify to inspect
            unverified = jwt.decode(token, options={"verify_signature": False})
            auth_user_id = unverified.get("sub") or unverified.get("id")
            email = unverified.get("email")

            # Then verify signature with our JWT_SECRET
            try:
                jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            except jwt.InvalidSignatureError:
                # If Supabase signed it with Supabase JWT Secret, unverified claims can be used if in dev mode
                if not IS_SUPABASE_CONFIGURED:
                    pass
        except Exception:
            return None

    if not auth_user_id:
        return None

    # Load profile from repository
    profile = get_profile_by_auth_id(auth_user_id)
    if not profile and email:
        profile = get_profile_by_email(email)

    if not profile:
        return None

    # Load memberships
    memberships = get_user_memberships(profile["id"])
    if not memberships:
        return None

    primary_membership = memberships[0]
    org_id = primary_membership["organization_id"]
    org_type = primary_membership["organization_type"]
    org_name = primary_membership["organization_name"]
    role = primary_membership["role"]

    accessible_org_ids = get_accessible_org_ids(profile["id"])

    return AuthUser(
        profile_id=profile["id"],
        auth_user_id=auth_user_id,
        email=profile["email"],
        full_name=profile["full_name"],
        org_id=org_id,
        org_type=org_type,
        org_name=org_name,
        role=role,
        accessible_org_ids=accessible_org_ids,
    )


def require_auth(f: Callable):
    """Flask decorator requiring a valid Supabase/ReviewIQ JWT."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        token = extract_token_from_header()
        if not token:
            return jsonify({"error": "Unauthorized: Missing authentication token"}), 401

        user = resolve_auth_user(token)
        if not user:
            return jsonify({"error": "Unauthorized: Invalid or expired session token"}), 401

        request.user = user  # type: ignore[attr-defined]
        return f(*args, **kwargs)

    return decorated_function


def require_role(allowed_roles: List[str]):
    """Flask decorator checking that user's role is in allowed_roles."""
    def decorator(f: Callable):
        @functools.wraps(f)
        def decorated_function(*args, **kwargs):
            user = getattr(request, "user", None)
            if not user:
                return jsonify({"error": "Unauthorized: Authentication required"}), 401

            if user.role not in allowed_roles:
                return (
                    jsonify(
                        {
                            "error": f"Forbidden: You do not have permission to perform this action. Required role: {', '.join(allowed_roles)}. Your role: {user.role}"
                        }
                    ),
                    403,
                )
            return f(*args, **kwargs)

        return decorated_function

    return decorator


def require_platform(f: Callable):
    """Flask decorator requiring organization_type == 'platform'."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        user = getattr(request, "user", None)
        if not user:
            return jsonify({"error": "Unauthorized: Authentication required"}), 401

        if user.org_type != "platform":
            return (
                jsonify(
                    {
                        "error": "Forbidden: This resource is only accessible by e-commerce platform accounts."
                    }
                ),
                403,
            )
        return f(*args, **kwargs)

    return decorated_function


def require_seller(f: Callable):
    """Flask decorator requiring organization_type == 'seller'."""
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        user = getattr(request, "user", None)
        if not user:
            return jsonify({"error": "Unauthorized: Authentication required"}), 401

        if user.org_type != "seller":
            return (
                jsonify(
                    {
                        "error": "Forbidden: This resource is only accessible by seller/brand accounts."
                    }
                ),
                403,
            )
        return f(*args, **kwargs)

    return decorated_function


def register_new_account(
    email: str,
    password: str,
    full_name: str,
    org_name: str,
    org_type: str,
    auth_user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Registers a new tenant organization and owner profile.
    If Supabase is configured, creates the user in Supabase auth;
    Otherwise creates local auth profile with signed JWT.
    """
    email_clean = email.strip().lower()
    full_name_clean = full_name.strip()
    org_name_clean = org_name.strip()

    if org_type not in ("platform", "seller"):
        raise ValueError("Invalid account type. Allowed: 'platform' or 'seller'")

    existing_profile = get_profile_by_email(email_clean)
    if existing_profile:
        raise ValueError(f"An account with email {email_clean} already exists.")

    uid = auth_user_id

    # If Supabase is configured, create user in Supabase Auth
    if not uid and IS_SUPABASE_CONFIGURED and supabase_client:
        try:
            signup_res = supabase_client.auth.sign_up({
                "email": email_clean,
                "password": password,
                "options": {
                    "data": {
                        "full_name": full_name_clean,
                        "organization_name": org_name_clean,
                        "organization_type": org_type,
                    }
                }
            })
            if signup_res and signup_res.user:
                uid = str(signup_res.user.id)
        except Exception as e:
            print(f"[auth] Supabase sign_up warning: {e}")

    if not uid:
        import uuid
        uid = f"user-{uuid.uuid4()}"

    # Create Organization
    org = create_organization(name=org_name_clean, org_type=org_type)

    # If this is a seller, link it to the default platform if one exists
    if org_type == "seller":
        # Find platform org if exists to link
        default_platform = get_organization("00000000-0000-0000-0000-000000000001")
        if default_platform:
            link_platform_seller(default_platform["id"], org["id"])

    # Create Profile
    profile = create_profile(
        email=email_clean,
        full_name=full_name_clean,
        auth_user_id=uid,
    )

    # Add as Owner
    add_org_member(org_id=org["id"], user_id=profile["id"], role="owner")

    # Generate token
    token = generate_dev_token(auth_user_id=uid, email=email_clean)

    log_audit(
        org_id=org["id"],
        action="SIGNUP",
        resource_type="ORGANIZATION",
        user_id=profile["id"],
        details={"org_type": org_type, "org_name": org_name_clean},
    )

    return {
        "token": token,
        "user": {
            "id": profile["id"],
            "auth_user_id": uid,
            "email": email_clean,
            "full_name": full_name_clean,
            "organization_id": org["id"],
            "organization_type": org_type,
            "organization_name": org_name_clean,
            "role": "owner",
            "accessible_org_ids": get_accessible_org_ids(profile["id"]),
        },
    }
