-- ==============================================================================
-- ReviewIQ SaaS Database Migration 002: Row Level Security (RLS) Policies
-- ==============================================================================
-- Enforces true multi-tenant data isolation:
--   - Sellers can ONLY view & modify data for their own organization.
--   - Platforms can view their own data AND data of managed sellers.
--   - Viewers have read-only access (no uploads, no analysis, no product creation).
--   - Protects against malicious organization_id spoofing.
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- HELPER FUNCTIONS (SECURITY DEFINER avoids infinite recursion)
-- ------------------------------------------------------------------------------

-- Get current profile ID for the authenticated Supabase user
CREATE OR REPLACE FUNCTION public.get_current_profile_id()
RETURNS UUID AS $$
    SELECT id FROM public.profiles WHERE auth_user_id = auth.uid() LIMIT 1;
$$ LANGUAGE sql STABLE SECURITY DEFINER;

-- Get list of organization IDs the current user directly belongs to
CREATE OR REPLACE FUNCTION public.get_user_org_ids()
RETURNS TABLE(org_id UUID) AS $$
    SELECT organization_id 
    FROM public.organization_members 
    WHERE user_id = public.get_current_profile_id();
$$ LANGUAGE sql STABLE SECURITY DEFINER;

-- Get current user's role in a specific organization
CREATE OR REPLACE FUNCTION public.get_user_org_role(target_org_id UUID)
RETURNS TEXT AS $$
    SELECT role 
    FROM public.organization_members 
    WHERE organization_id = target_org_id 
      AND user_id = public.get_current_profile_id() 
    LIMIT 1;
$$ LANGUAGE sql STABLE SECURITY DEFINER;

-- Get all accessible organization IDs for the current user:
-- If user is a member of a Platform organization, includes managed seller organizations.
-- If user is a member of a Seller organization, includes ONLY their own organization.
CREATE OR REPLACE FUNCTION public.get_accessible_org_ids()
RETURNS TABLE(org_id UUID) AS $$
    -- Direct organization memberships
    SELECT om.organization_id 
    FROM public.organization_members om
    WHERE om.user_id = public.get_current_profile_id()
    
    UNION
    
    -- Managed sellers if user belongs to a platform organization
    SELECT ps.seller_organization_id
    FROM public.platform_sellers ps
    JOIN public.organizations o ON o.id = ps.platform_organization_id
    JOIN public.organization_members om ON om.organization_id = o.id
    WHERE om.user_id = public.get_current_profile_id()
      AND o.organization_type = 'platform';
$$ LANGUAGE sql STABLE SECURITY DEFINER;

-- ------------------------------------------------------------------------------
-- ENABLE ROW LEVEL SECURITY ON ALL TABLES
-- ------------------------------------------------------------------------------
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.organizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.organization_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.platform_sellers ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.organization_invitations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.products ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reviews ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.aspect_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.issue_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.user_preferences ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;

-- ------------------------------------------------------------------------------
-- 1. PROFILES POLICIES
-- ------------------------------------------------------------------------------
CREATE POLICY "Users can view profiles in their accessible organizations or own profile"
    ON public.profiles
    FOR SELECT
    USING (
        auth_user_id = auth.uid() OR
        id IN (
            SELECT om.user_id 
            FROM public.organization_members om 
            WHERE om.organization_id IN (SELECT org_id FROM public.get_accessible_org_ids())
        )
    );

CREATE POLICY "Users can update their own profile"
    ON public.profiles
    FOR UPDATE
    USING (auth_user_id = auth.uid());

CREATE POLICY "Service role or signup trigger can insert profiles"
    ON public.profiles
    FOR INSERT
    WITH CHECK (auth_user_id = auth.uid() OR auth.role() = 'service_role');

-- ------------------------------------------------------------------------------
-- 2. ORGANIZATIONS POLICIES
-- ------------------------------------------------------------------------------
CREATE POLICY "Users can view accessible organizations"
    ON public.organizations
    FOR SELECT
    USING (id IN (SELECT org_id FROM public.get_accessible_org_ids()));

CREATE POLICY "Owners and admins can update their organization"
    ON public.organizations
    FOR UPDATE
    USING (
        id IN (SELECT org_id FROM public.get_user_org_ids()) AND
        public.get_user_org_role(id) IN ('owner', 'admin')
    );

CREATE POLICY "Authenticated users or service role can create organizations"
    ON public.organizations
    FOR INSERT
    WITH CHECK (auth.role() IN ('authenticated', 'service_role'));

-- ------------------------------------------------------------------------------
-- 3. ORGANIZATION MEMBERS POLICIES
-- ------------------------------------------------------------------------------
CREATE POLICY "Members can view membership in accessible organizations"
    ON public.organization_members
    FOR SELECT
    USING (organization_id IN (SELECT org_id FROM public.get_accessible_org_ids()));

CREATE POLICY "Owners and admins can manage organization members"
    ON public.organization_members
    FOR ALL
    USING (
        organization_id IN (SELECT org_id FROM public.get_user_org_ids()) AND
        public.get_user_org_role(organization_id) IN ('owner', 'admin')
    );

-- ------------------------------------------------------------------------------
-- 4. PLATFORM SELLERS POLICIES
-- ------------------------------------------------------------------------------
CREATE POLICY "Platform and seller members can view link"
    ON public.platform_sellers
    FOR SELECT
    USING (
        platform_organization_id IN (SELECT org_id FROM public.get_user_org_ids()) OR
        seller_organization_id IN (SELECT org_id FROM public.get_user_org_ids())
    );

CREATE POLICY "Platform owners/admins can manage platform sellers"
    ON public.platform_sellers
    FOR ALL
    USING (
        platform_organization_id IN (SELECT org_id FROM public.get_user_org_ids()) AND
        public.get_user_org_role(platform_organization_id) IN ('owner', 'admin')
    );

-- ------------------------------------------------------------------------------
-- 5. ORGANIZATION INVITATIONS POLICIES
-- ------------------------------------------------------------------------------
CREATE POLICY "Admins can view and manage organization invitations"
    ON public.organization_invitations
    FOR ALL
    USING (
        organization_id IN (SELECT org_id FROM public.get_user_org_ids()) AND
        public.get_user_org_role(organization_id) IN ('owner', 'admin')
    );

-- ------------------------------------------------------------------------------
-- 6. PRODUCTS POLICIES
-- ------------------------------------------------------------------------------
CREATE POLICY "Users can view products in accessible organizations"
    ON public.products
    FOR SELECT
    USING (organization_id IN (SELECT org_id FROM public.get_accessible_org_ids()));

CREATE POLICY "Owners, admins, and analysts can create products in their own organization"
    ON public.products
    FOR INSERT
    WITH CHECK (
        organization_id IN (SELECT org_id FROM public.get_user_org_ids()) AND
        public.get_user_org_role(organization_id) IN ('owner', 'admin', 'analyst')
    );

CREATE POLICY "Owners and admins can update products in their organization"
    ON public.products
    FOR UPDATE
    USING (
        organization_id IN (SELECT org_id FROM public.get_user_org_ids()) AND
        public.get_user_org_role(organization_id) IN ('owner', 'admin')
    );

CREATE POLICY "Owners and admins can delete products in their organization"
    ON public.products
    FOR DELETE
    USING (
        organization_id IN (SELECT org_id FROM public.get_user_org_ids()) AND
        public.get_user_org_role(organization_id) IN ('owner', 'admin')
    );

-- ------------------------------------------------------------------------------
-- 7. DATASETS POLICIES
-- ------------------------------------------------------------------------------
CREATE POLICY "Users can view datasets in accessible organizations"
    ON public.datasets
    FOR SELECT
    USING (organization_id IN (SELECT org_id FROM public.get_accessible_org_ids()));

CREATE POLICY "Owners, admins, and analysts can upload datasets"
    ON public.datasets
    FOR INSERT
    WITH CHECK (
        organization_id IN (SELECT org_id FROM public.get_user_org_ids()) AND
        public.get_user_org_role(organization_id) IN ('owner', 'admin', 'analyst')
    );

-- ------------------------------------------------------------------------------
-- 8. REVIEWS POLICIES
-- ------------------------------------------------------------------------------
CREATE POLICY "Users can view reviews in accessible organizations"
    ON public.reviews
    FOR SELECT
    USING (organization_id IN (SELECT org_id FROM public.get_accessible_org_ids()));

CREATE POLICY "Analysts, admins, and owners can import reviews"
    ON public.reviews
    FOR INSERT
    WITH CHECK (
        organization_id IN (SELECT org_id FROM public.get_user_org_ids()) AND
        public.get_user_org_role(organization_id) IN ('owner', 'admin', 'analyst')
    );

-- ------------------------------------------------------------------------------
-- 9. ANALYSES POLICIES
-- ------------------------------------------------------------------------------
CREATE POLICY "Users can view analyses in accessible organizations"
    ON public.analyses
    FOR SELECT
    USING (organization_id IN (SELECT org_id FROM public.get_accessible_org_ids()));

CREATE POLICY "Analysts, admins, and owners can run analyses"
    ON public.analyses
    FOR INSERT
    WITH CHECK (
        organization_id IN (SELECT org_id FROM public.get_user_org_ids()) AND
        public.get_user_org_role(organization_id) IN ('owner', 'admin', 'analyst')
    );

-- ------------------------------------------------------------------------------
-- 10. ASPECT & ISSUE RESULTS POLICIES
-- ------------------------------------------------------------------------------
CREATE POLICY "Users can view aspect results in accessible organizations"
    ON public.aspect_results
    FOR SELECT
    USING (organization_id IN (SELECT org_id FROM public.get_accessible_org_ids()));

CREATE POLICY "Users can view issue results in accessible organizations"
    ON public.issue_results
    FOR SELECT
    USING (organization_id IN (SELECT org_id FROM public.get_accessible_org_ids()));

-- ------------------------------------------------------------------------------
-- 11. USER PREFERENCES POLICIES
-- ------------------------------------------------------------------------------
CREATE POLICY "Users can manage their own preferences"
    ON public.user_preferences
    FOR ALL
    USING (user_id = public.get_current_profile_id());

-- ------------------------------------------------------------------------------
-- 12. AUDIT LOGS POLICIES
-- ------------------------------------------------------------------------------
CREATE POLICY "Members can view audit logs for their accessible organizations"
    ON public.audit_logs
    FOR SELECT
    USING (organization_id IN (SELECT org_id FROM public.get_accessible_org_ids()));

CREATE POLICY "Authenticated users can create audit log entries"
    ON public.audit_logs
    FOR INSERT
    WITH CHECK (auth.role() IN ('authenticated', 'service_role'));
