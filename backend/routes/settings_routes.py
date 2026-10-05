"""
================================================================================
ReviewIQ Organization & Account Settings Routes
--------------------------------------------------------------------------------
Handles:
  - Team member management
  - Role assignments (OWNER, ADMIN, ANALYST, VIEWER)
  - Team invitations
  - Security & audit logs
================================================================================
"""

from flask import Blueprint, jsonify, request

from backend.auth import require_auth, require_role
from backend.db import get_sqlite_conn, IS_SUPABASE_CONFIGURED, supabase_client
from backend.repository import (
    create_invitation,
    get_audit_logs,
    get_organization,
    list_invitations,
    log_audit,
)

settings_bp = Blueprint("settings_bp", __name__)


@settings_bp.route("/api/settings/team", methods=["GET"])
@require_auth
def get_team_members():
    """Returns all members of the user's organization."""
    user = getattr(request, "user")
    conn = get_sqlite_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT om.id, om.role, om.created_at, p.id as profile_id, p.full_name, p.email, p.avatar_url
        FROM organization_members om
        JOIN profiles p ON p.id = om.user_id
        WHERE om.organization_id = ?
        ORDER BY om.created_at ASC
        """,
        (user.org_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return jsonify({
        "organization_id": user.org_id,
        "organization_name": user.org_name,
        "organization_type": user.org_type,
        "members": [dict(r) for r in rows],
    })


@settings_bp.route("/api/settings/invitations", methods=["GET"])
@require_auth
def get_invitations():
    """Returns invitations for the organization."""
    user = getattr(request, "user")
    invites = list_invitations(user.org_id)
    return jsonify({
        "invitations": invites,
    })


@settings_bp.route("/api/settings/invite", methods=["POST"])
@require_auth
@require_role(["owner", "admin"])
def invite_team_member():
    """
    Invites a new team member with a designated role (admin, analyst, viewer).
    """
    user = getattr(request, "user")
    data = request.get_json(force=True) or {}
    email = (data.get("email") or "").strip().lower()
    role = (data.get("role") or "analyst").strip().lower()

    if not email or "@" not in email:
        return jsonify({"error": "A valid email is required."}), 400
    if role not in ("owner", "admin", "analyst", "viewer"):
        return jsonify({"error": "Role must be one of: admin, analyst, viewer."}), 400

    invitation = create_invitation(
        org_id=user.org_id,
        email=email,
        role=role,
        invited_by=user.profile_id,
    )

    log_audit(
        org_id=user.org_id,
        action="INVITE_USER",
        resource_type="INVITATION",
        user_id=user.profile_id,
        details={"invited_email": email, "role": role},
    )

    return jsonify({
        "message": f"Invitation successfully sent to {email}.",
        "invitation": invitation,
    }), 201


@settings_bp.route("/api/settings/audit", methods=["GET"])
@require_auth
def get_org_audit_logs():
    """Returns audit log trail for the user's accessible organizations."""
    user = getattr(request, "user")
    logs = get_audit_logs(user.accessible_org_ids, limit=40)
    return jsonify({
        "audit_logs": logs,
    })
