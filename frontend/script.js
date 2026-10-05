/**
 * ReviewIQ Multi-Tenant SaaS Frontend Application Controller
 * Handles Supabase/ReviewIQ authentication, role-based view routing,
 * real-time ML prediction, and multi-tenant data access.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global Application State
  const state = {
    user: null,
    permissions: null,
    token: localStorage.getItem("reviewiq_token"),
    platformData: null,
    sellerData: null,
    products: [],
    sellers: [],
    aspects: [],
    issues: [],
    reviewsQuery: {
      page: 1,
      limit: 20,
      seller_id: "",
      product_id: "",
      sentiment: "All",
      aspect: "All",
      search: "",
      sort: "default",
    },
    reviewsPagination: {
      total: 0,
      totalPages: 1,
    },
    charts: {},
    activeView: "overview",
  };

  // =========================================================================
  // 1. Authenticated API Fetch Helper
  // =========================================================================
  async function apiFetch(url, options = {}) {
    const token = localStorage.getItem("reviewiq_token");
    const headers = options.headers || {};
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }
    if (!(options.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
    }

    try {
      const res = await fetch(url, { ...options, headers });
      if (res.status === 401) {
        // Token expired or invalid
        localStorage.removeItem("reviewiq_token");
        state.token = null;
        state.user = null;
        showAuthModal();
        throw new Error("Unauthorized");
      }
      return res;
    } catch (err) {
      throw err;
    }
  }

  // =========================================================================
  // 2. Auth State & Session Management
  // =========================================================================
  const authOverlay = document.getElementById("auth-overlay");
  const tabLoginBtn = document.getElementById("tab-login-btn");
  const tabSignupBtn = document.getElementById("tab-signup-btn");
  const formLogin = document.getElementById("form-login");
  const formSignup = document.getElementById("form-signup");
  const authErrorMsg = document.getElementById("auth-error-msg");

  function showAuthModal(defaultTab = "login") {
    if (!authOverlay) return;
    authOverlay.classList.remove("hidden");
    if (defaultTab === "signup") {
      switchAuthTab("signup");
    } else {
      switchAuthTab("login");
    }
  }

  function hideAuthModal() {
    if (authOverlay) authOverlay.classList.add("hidden");
  }

  function switchAuthTab(tab) {
    authErrorMsg.style.display = "none";
    if (tab === "signup") {
      tabSignupBtn.classList.add("active");
      tabLoginBtn.classList.remove("active");
      formSignup.style.display = "flex";
      formLogin.style.display = "none";
      document.getElementById("auth-subtitle").textContent = "Create your ReviewIQ SaaS workspace";
    } else {
      tabLoginBtn.classList.add("active");
      tabSignupBtn.classList.remove("active");
      formLogin.style.display = "flex";
      formSignup.style.display = "none";
      document.getElementById("auth-subtitle").textContent = "Turn customer reviews into product intelligence.";
    }
  }

  if (tabLoginBtn && tabSignupBtn) {
    tabLoginBtn.addEventListener("click", () => switchAuthTab("login"));
    tabSignupBtn.addEventListener("click", () => switchAuthTab("signup"));
  }

  // Radio selection visual card handler
  const roleCardSeller = document.getElementById("card-role-seller");
  const roleCardPlatform = document.getElementById("card-role-platform");
  if (roleCardSeller && roleCardPlatform) {
    roleCardSeller.addEventListener("click", () => {
      roleCardSeller.classList.add("selected");
      roleCardPlatform.classList.remove("selected");
      roleCardSeller.querySelector("input").checked = true;
    });
    roleCardPlatform.addEventListener("click", () => {
      roleCardPlatform.classList.add("selected");
      roleCardSeller.classList.remove("selected");
      roleCardPlatform.querySelector("input").checked = true;
    });
  }

  // Login Form Submission
  if (formLogin) {
    formLogin.addEventListener("submit", async (e) => {
      e.preventDefault();
      authErrorMsg.style.display = "none";
      const email = document.getElementById("login-email").value.trim();
      const password = document.getElementById("login-password").value.trim();
      const submitBtn = document.getElementById("btn-login-submit");

      submitBtn.disabled = true;
      submitBtn.textContent = "Signing In...";

      try {
        const res = await fetch("/api/auth/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Login failed");
        }

        localStorage.setItem("reviewiq_token", data.token);
        state.token = data.token;
        state.user = data.user;
        await checkAuthAndInit();
      } catch (err) {
        authErrorMsg.textContent = err.message;
        authErrorMsg.style.display = "block";
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Sign In";
      }
    });
  }

  // Signup Form Submission
  if (formSignup) {
    formSignup.addEventListener("submit", async (e) => {
      e.preventDefault();
      authErrorMsg.style.display = "none";

      const full_name = document.getElementById("signup-name").value.trim();
      const email = document.getElementById("signup-email").value.trim();
      const password = document.getElementById("signup-password").value.trim();
      const organization_name = document.getElementById("signup-company").value.trim();
      const orgTypeInput = document.querySelector('input[name="signup_org_type"]:checked');
      const organization_type = orgTypeInput ? orgTypeInput.value : "seller";
      const submitBtn = document.getElementById("btn-signup-submit");

      submitBtn.disabled = true;
      submitBtn.textContent = "Creating Account...";

      try {
        const res = await fetch("/api/auth/signup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            full_name,
            email,
            password,
            organization_name,
            organization_type,
          }),
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || "Signup failed");
        }

        localStorage.setItem("reviewiq_token", data.token);
        state.token = data.token;
        state.user = data.user;
        await checkAuthAndInit();
      } catch (err) {
        authErrorMsg.textContent = err.message;
        authErrorMsg.style.display = "block";
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Create Account";
      }
    });
  }

  // 1-Click Demo Login Handler
  window.demoLogin = async (demoToken) => {
    localStorage.setItem("reviewiq_token", demoToken);
    state.token = demoToken;
    await checkAuthAndInit();
  };

  // Sign out
  window.logout = async () => {
    try {
      await apiFetch("/api/auth/logout", { method: "POST" });
    } catch (_) {}
    localStorage.removeItem("reviewiq_token");
    state.token = null;
    state.user = null;
    showAuthModal();
  };

  // Verify Auth on Startup
  async function checkAuthAndInit() {
    const token = localStorage.getItem("reviewiq_token");
    if (!token) {
      showAuthModal();
      return;
    }

    try {
      const res = await apiFetch("/api/auth/me");
      if (!res.ok) {
        throw new Error("Session invalid");
      }
      const data = await res.json();
      state.user = data.user;
      state.permissions = data.permissions;

      hideAuthModal();
      updateUserInterfaceForRole();
      await loadInitialData();

      // Navigate to current hash or default
      const hash = window.location.hash.replace("#", "") || "overview";
      navigateTo(hash);
    } catch (err) {
      console.warn("[ReviewIQ Auth] Auth check failed:", err);
      showAuthModal();
    }
  }

  // =========================================================================
  // 3. UI Role Customization (Platform vs Seller)
  // =========================================================================
  function updateUserInterfaceForRole() {
    if (!state.user) return;
    const u = state.user;
    const isPlatform = u.organization_type === "platform";

    // Update Topbar
    const topbarOrgName = document.getElementById("topbar-org-name");
    const topbarOrgPill = document.getElementById("topbar-org-type-pill");
    const topbarTenantBadge = document.getElementById("topbar-tenant-badge");
    const topbarUserName = document.getElementById("topbar-user-name");
    const topbarUserRole = document.getElementById("topbar-user-role");
    const avatarLetter = document.getElementById("user-avatar-letter");
    const menuUserName = document.getElementById("menu-user-fullname");
    const menuUserEmail = document.getElementById("menu-user-email");
    const sidebarOrgName = document.getElementById("sidebar-org-name");
    const sidebarTenantType = document.getElementById("sidebar-tenant-type");
    const footerOrgLabel = document.getElementById("footer-org-label");

    if (topbarOrgName) topbarOrgName.textContent = u.organization_name;
    if (topbarOrgPill) {
      topbarOrgPill.textContent = isPlatform ? "Platform" : "Seller";
      topbarOrgPill.className = `badge ${isPlatform ? "primary" : "success"}`;
    }
    if (topbarTenantBadge) {
      topbarTenantBadge.className = `topbar-tenant-badge ${isPlatform ? "platform" : "seller"}`;
    }
    if (topbarUserName) topbarUserName.textContent = u.full_name;
    if (topbarUserRole) topbarUserRole.textContent = u.role.toUpperCase();
    if (avatarLetter) avatarLetter.textContent = (u.full_name || "U")[0].toUpperCase();
    if (menuUserName) menuUserName.textContent = u.full_name;
    if (menuUserEmail) menuUserEmail.textContent = u.email;
    if (sidebarOrgName) sidebarOrgName.textContent = u.organization_name;
    if (sidebarTenantType) sidebarTenantType.textContent = isPlatform ? "Platform" : "Seller";
    if (footerOrgLabel) footerOrgLabel.textContent = u.organization_name;

    // Toggle Platform-Only sidebar items
    const platformItems = document.querySelectorAll(".nav-platform-only");
    platformItems.forEach((el) => {
      el.style.display = isPlatform ? "flex" : "none";
    });

    // Update Nav labels
    const navOverviewText = document.getElementById("nav-text-overview");
    const navProductsText = document.getElementById("nav-text-products");
    if (navOverviewText) navOverviewText.textContent = isPlatform ? "Marketplace Overview" : "Overview";
    if (navProductsText) navProductsText.textContent = isPlatform ? "Marketplace Products" : "My Products";

    // Products View Titles
    const prodTitle = document.getElementById("products-view-title");
    const prodSub = document.getElementById("products-view-sub");
    if (prodTitle) prodTitle.textContent = isPlatform ? "Marketplace Products" : "My Products";
    if (prodSub) {
      prodSub.textContent = isPlatform
        ? "Explore catalog items across all managed sellers."
        : "Manage your brand's products, SKU performance, and reviews.";
    }

    // Role check for Add Product button
    const btnAddProd = document.getElementById("btn-add-product-main");
    if (btnAddProd) {
      if (u.role === "viewer") {
        btnAddProd.style.display = "none";
      } else {
        btnAddProd.style.display = "inline-flex";
      }
    }

    // Review Explorer seller filter
    const revSellerFilter = document.getElementById("review-filter-seller");
    if (revSellerFilter) {
      revSellerFilter.style.display = isPlatform ? "block" : "none";
    }
  }

  // =========================================================================
  // 4. Router & Navigation
  // =========================================================================
  const navItems = document.querySelectorAll(".nav-item");
  const viewSections = document.querySelectorAll(".view-section");
  const topbarTitle = document.getElementById("topbar-title");
  const mobileToggle = document.getElementById("mobile-toggle");
  const sidebar = document.getElementById("app-sidebar");

  function navigateTo(viewId) {
    if (!state.user) {
      showAuthModal();
      return;
    }

    // Determine target view section
    let targetSectionId = `view-${viewId}`;
    if (viewId === "overview") {
      targetSectionId = state.user.organization_type === "platform" ? "view-platform-overview" : "view-seller-overview";
    } else if (viewId === "sellers") {
      targetSectionId = "view-platform-sellers";
    } else if (viewId === "analytics") {
      targetSectionId = "view-platform-analytics";
    }

    // Update URL hash
    if (window.location.hash !== `#${viewId}`) {
      window.history.pushState(null, "", `#${viewId}`);
    }
    state.activeView = viewId;

    // Toggle active view section
    viewSections.forEach((sec) => {
      sec.classList.remove("active");
      if (sec.id === targetSectionId) {
        sec.classList.add("active");
      }
    });

    // Toggle active nav link
    navItems.forEach((item) => {
      item.classList.remove("active");
      if (item.getAttribute("data-view") === viewId) {
        item.classList.add("active");
      }
    });

    // Update topbar title
    const viewTitleMap = {
      overview: state.user.organization_type === "platform" ? "Marketplace Overview" : "Your Product Intelligence",
      sellers: "Seller Management",
      products: state.user.organization_type === "platform" ? "Marketplace Products" : "My Products",
      sentiment: "Sentiment Intelligence",
      aspects: "Aspect Intelligence",
      issues: "Customer Issues & Complaints",
      reviews: "Review Explorer",
      analytics: "Cross-Seller Analytics",
      "live-analyzer": "Live Review Analyzer",
      analyze: "Analyze Reviews & Datasets",
      settings: "Organization & Team Settings",
    };
    if (topbarTitle) topbarTitle.textContent = viewTitleMap[viewId] || "Intelligence";

    if (sidebar) sidebar.classList.remove("open");

    // Load data for specific view
    if (viewId === "overview") {
      if (state.user.organization_type === "platform") loadPlatformData();
      else loadSellerData();
    } else if (viewId === "sellers") {
      loadPlatformSellers();
    } else if (viewId === "products") {
      loadProducts();
    } else if (viewId === "reviews") {
      loadReviews();
    } else if (viewId === "analytics") {
      loadCrossSellerAnalytics();
    } else if (viewId === "sentiment" || viewId === "aspects") {
      renderSentimentCharts();
    } else if (viewId === "issues") {
      loadIssuesTable();
    } else if (viewId === "settings") {
      loadSettingsView();
    }

    window.dispatchEvent(new Event("resize"));
  }

  window.addEventListener("hashchange", () => {
    const hash = window.location.hash.replace("#", "") || "overview";
    navigateTo(hash);
  });

  navItems.forEach((item) => {
    item.addEventListener("click", (e) => {
      e.preventDefault();
      const viewId = item.getAttribute("data-view");
      navigateTo(viewId);
    });
  });

  if (mobileToggle && sidebar) {
    mobileToggle.addEventListener("click", () => sidebar.classList.toggle("open"));
  }

  // User avatar dropdown toggle
  const userAvatarBtn = document.getElementById("user-avatar-btn");
  const userDropdown = document.getElementById("user-dropdown");
  if (userAvatarBtn && userDropdown) {
    userAvatarBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      userDropdown.classList.toggle("open");
    });
    document.addEventListener("click", () => userDropdown.classList.remove("open"));
  }

  // =========================================================================
  // 5. Data Loading Functions
  // =========================================================================

  async function loadInitialData() {
    if (state.user.organization_type === "platform") {
      await loadPlatformData();
    } else {
      await loadSellerData();
    }
    await loadProducts();
    await loadAspectsAndIssues();
  }

  // --- Platform Data ---
  async function loadPlatformData() {
    try {
      const res = await apiFetch("/api/platform/overview");
      if (!res.ok) return;
      const data = await res.json();
      state.platformData = data;

      // Update KPI Cards
      document.getElementById("platform-kpi-sellers").textContent = data.total_sellers;
      document.getElementById("platform-kpi-products").textContent = data.total_products;
      document.getElementById("platform-kpi-reviews").textContent = Number(data.total_reviews).toLocaleString();
      document.getElementById("platform-kpi-sentiment").textContent = `${data.overall_positive_pct}%`;
      document.getElementById("platform-kpi-neg").textContent = `${data.overall_negative_pct}%`;
      document.getElementById("platform-kpi-health").textContent = `${data.marketplace_health} / 100`;

      // Update sidebar counter
      const sidebarCnt = document.getElementById("sidebar-review-count");
      if (sidebarCnt) sidebarCnt.textContent = `${(data.total_reviews / 1000).toFixed(1)}K`;

      // Render Sellers Table
      const tbody = document.getElementById("platform-sellers-tbody");
      if (tbody && data.sellers) {
        tbody.innerHTML = data.sellers
          .map((s) => {
            return `
            <tr>
              <td><strong>${s.name}</strong></td>
              <td>${s.products_count}</td>
              <td>${Number(s.reviews_count).toLocaleString()}</td>
              <td style="color:var(--status-pos-text); font-weight:600;">${s.positive_pct}%</td>
              <td style="color:var(--status-neg-text); font-weight:600;">${s.negative_pct}%</td>
              <td>
                <span class="badge ${s.health_score >= 75 ? 'success' : s.health_score >= 50 ? 'warning' : 'danger'}">
                  ${s.health_score} / 100
                </span>
              </td>
              <td><span class="badge ${s.status === 'active' ? 'success' : 'warning'}">${s.status}</span></td>
              <td>
                <button class="btn btn-secondary btn-sm" onclick="window.inspectSeller('${s.id}')">Inspect Seller</button>
              </td>
            </tr>
          `;
          })
          .join("");
      }

      // Populate review seller filter
      const revSellerFilter = document.getElementById("review-filter-seller");
      if (revSellerFilter && data.sellers) {
        revSellerFilter.innerHTML = '<option value="">All Managed Sellers</option>' +
          data.sellers.map((s) => `<option value="${s.id}">${s.name}</option>`).join("");
      }
    } catch (err) {
      console.error("[ReviewIQ] Error loading platform overview:", err);
    }
  }

  // --- Seller Data ---
  async function loadSellerData() {
    try {
      const res = await apiFetch("/api/seller/overview");
      if (!res.ok) return;
      const data = await res.json();
      state.sellerData = data;
      const m = data.metrics;

      // Greeting
      const greeting = document.getElementById("seller-greeting-title");
      if (greeting) greeting.textContent = `Good morning, ${state.user.full_name} — Your Product Intelligence`;

      // KPIs
      document.getElementById("seller-kpi-products").textContent = m.total_products;
      document.getElementById("seller-kpi-reviews").textContent = Number(m.total_reviews).toLocaleString();
      document.getElementById("seller-kpi-pos").textContent = `${m.positive_pct}%`;
      document.getElementById("seller-kpi-neg").textContent = `${m.negative_pct}%`;
      document.getElementById("seller-kpi-health").textContent = `${m.health_score} / 100`;

      // Update sidebar counter
      const sidebarCnt = document.getElementById("sidebar-review-count");
      if (sidebarCnt) sidebarCnt.textContent = `${(m.total_reviews / 1000).toFixed(1)}K`;

      // Product performance breakdown table
      const tbody = document.getElementById("seller-products-perf-tbody");
      if (tbody && m.product_performance) {
        tbody.innerHTML = m.product_performance
          .map((p) => {
            return `
            <tr>
              <td><strong>${p.name}</strong></td>
              <td style="font-family:var(--font-mono); font-size:12px;">${p.sku || '--'}</td>
              <td><span class="badge-pill">${p.category}</span></td>
              <td>${Number(p.reviews_count).toLocaleString()}</td>
              <td style="color:var(--status-pos-text); font-weight:600;">${p.positive_pct}%</td>
              <td style="color:var(--status-neg-text); font-weight:600;">${p.negative_pct}%</td>
              <td>
                <span class="badge ${p.health_score >= 75 ? 'success' : p.health_score >= 50 ? 'warning' : 'danger'}">
                  ${p.health_score} / 100
                </span>
              </td>
              <td>
                <button class="btn btn-secondary btn-sm" onclick="window.openProductDetailModal('${p.id}')">Inspect</button>
              </td>
            </tr>
          `;
          })
          .join("");
      }

      // Top recurring issues list
      const issuesContainer = document.getElementById("seller-top-issues-list");
      if (issuesContainer) {
        if (!m.top_issues || m.top_issues.length === 0) {
          issuesContainer.innerHTML = '<div style="font-size:13px; color:var(--text-muted); padding:10px;">No recurring complaint signals detected.</div>';
        } else {
          issuesContainer.innerHTML = m.top_issues
            .map((iss) => {
              return `
              <div style="padding:10px 14px; background:#f8fafc; border:1px solid var(--border-light); border-radius:var(--radius-md); display:flex; justify-content:space-between; align-items:center;">
                <div>
                  <div style="font-weight:700; font-size:13px; text-transform:capitalize;">${iss.phrase}</div>
                  <div style="font-size:11px; color:var(--text-muted);">${iss.aspect || 'General'} · ${iss.frequency} mentions</div>
                </div>
                <span class="badge ${iss.priority === 'HIGH' ? 'danger' : 'warning'}">${iss.priority}</span>
              </div>
            `;
            })
            .join("");
        }
      }

      // Recent customer feedback list
      const feedbackContainer = document.getElementById("seller-recent-feedback-list");
      if (feedbackContainer) {
        if (!m.recent_reviews || m.recent_reviews.length === 0) {
          feedbackContainer.innerHTML = '<div style="font-size:13px; color:var(--text-muted); padding:10px;">No recent customer feedback imported yet.</div>';
        } else {
          feedbackContainer.innerHTML = m.recent_reviews
            .map((r) => {
              const isPos = r.sentiment === "POSITIVE";
              return `
              <div style="padding:10px 14px; background:#f8fafc; border:1px solid var(--border-light); border-radius:var(--radius-md);">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                  <span style="font-size:11px; font-weight:600; color:var(--text-muted);">${r.product_name || 'Product'}</span>
                  <span class="badge ${isPos ? 'success' : 'danger'}">${r.sentiment}</span>
                </div>
                <div style="font-size:12px; font-weight:600; color:var(--text-main); margin-bottom:2px;">${r.title || ''}</div>
                <div style="font-size:12px; color:var(--text-muted); line-height:1.4;">${r.content}</div>
              </div>
            `;
            })
            .join("");
        }
      }
    } catch (err) {
      console.error("[ReviewIQ] Error loading seller overview:", err);
    }
  }

  // --- Platform Sellers Directory ---
  async function loadPlatformSellers() {
    try {
      const res = await apiFetch("/api/platform/sellers");
      if (!res.ok) return;
      const data = await res.json();
      state.sellers = data.sellers || [];

      document.getElementById("sellers-count-badge").textContent = `${data.total} Sellers`;

      const tbody = document.getElementById("sellers-directory-tbody");
      if (tbody) {
        tbody.innerHTML = state.sellers
          .map((s) => {
            return `
            <tr>
              <td><strong>${s.name}</strong></td>
              <td>${s.products_count}</td>
              <td>${Number(s.reviews_count).toLocaleString()}</td>
              <td style="color:var(--status-pos-text); font-weight:600;">${s.positive_pct}%</td>
              <td style="color:var(--status-neg-text); font-weight:600;">${s.negative_pct}%</td>
              <td>
                <span class="badge ${s.health_score >= 75 ? 'success' : s.health_score >= 50 ? 'warning' : 'danger'}">
                  ${s.health_score} / 100
                </span>
              </td>
              <td><span class="badge ${s.status === 'active' ? 'success' : 'warning'}">${s.status}</span></td>
              <td>
                <button class="btn btn-secondary btn-sm" onclick="window.inspectSeller('${s.id}')">Inspect</button>
              </td>
            </tr>
          `;
          })
          .join("");
      }
    } catch (err) {
      console.error("[ReviewIQ] Error loading sellers directory:", err);
    }
  }

  // --- Products Catalog ---
  async function loadProducts() {
    try {
      const res = await apiFetch("/api/products");
      if (!res.ok) return;
      const data = await res.json();
      state.products = data.products || [];

      const grid = document.getElementById("products-cards-grid");
      if (grid) {
        if (state.products.length === 0) {
          grid.innerHTML = '<div style="grid-column:1/-1; padding:30px; text-align:center; color:var(--text-muted);">No products found. Click "Add Product" to add your first product.</div>';
          return;
        }

        grid.innerHTML = state.products
          .map((p) => {
            const healthBadge = p.health_score >= 75 ? "success" : p.health_score >= 50 ? "warning" : "danger";
            return `
            <div class="product-card" onclick="window.openProductDetailModal('${p.id}')">
              <div>
                <div class="product-card-header">
                  <span class="badge-pill">${p.category || 'General'}</span>
                  <span class="badge ${healthBadge}">Health: ${p.health_score}</span>
                </div>
                <div class="product-card-title">${p.name}</div>
                <div class="product-card-sku">SKU: ${p.sku || 'N/A'} · ${p.seller_name ? p.seller_name : ''}</div>
              </div>
              <div class="product-stats-row">
                <div class="stat-item">
                  <span class="stat-label">Reviews</span>
                  <span class="stat-val">${Number(p.review_count).toLocaleString()}</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Positive</span>
                  <span class="stat-val" style="color:var(--status-pos);">${p.positive_pct}%</span>
                </div>
                <div class="stat-item">
                  <span class="stat-label">Negative</span>
                  <span class="stat-val" style="color:var(--status-neg);">${p.negative_pct}%</span>
                </div>
              </div>
            </div>
          `;
          })
          .join("");
      }

      // Populate upload target product selector
      const uploadProdSelect = document.getElementById("upload-target-product");
      if (uploadProdSelect) {
        uploadProdSelect.innerHTML = state.products
          .map((p) => `<option value="${p.id}">${p.name} (${p.sku || 'No SKU'})</option>`)
          .join("");
      }
    } catch (err) {
      console.error("[ReviewIQ] Error loading products:", err);
    }
  }

  // --- Cross-Seller Comparison ---
  async function loadCrossSellerAnalytics() {
    try {
      const res = await apiFetch("/api/platform/analytics");
      if (!res.ok) return;
      const data = await res.json();
      const tbody = document.getElementById("tbody-cross-seller");
      if (tbody && data.comparison) {
        tbody.innerHTML = data.comparison
          .map((c) => {
            return `
            <tr>
              <td><strong>${c.seller_name}</strong></td>
              <td>${c.total_products}</td>
              <td>${Number(c.total_reviews).toLocaleString()}</td>
              <td style="color:var(--status-pos-text); font-weight:600;">${c.positive_pct}%</td>
              <td style="color:var(--status-neg-text); font-weight:600;">${c.negative_pct}%</td>
              <td>
                <span class="badge ${c.health_score >= 75 ? 'success' : 'warning'}">
                  ${c.health_score} / 100
                </span>
              </td>
              <td><span class="badge-pill">${c.top_aspect}</span></td>
              <td><span class="badge danger" style="text-transform:capitalize;">${c.top_issue}</span></td>
            </tr>
          `;
          })
          .join("");
      }
    } catch (err) {
      console.error("[ReviewIQ] Error loading cross-seller analytics:", err);
    }
  }

  // --- Aspects & Complaints ---
  async function loadAspectsAndIssues() {
    try {
      const [aspRes, issRes] = await Promise.all([
        apiFetch("/api/aspects"),
        apiFetch("/api/complaints"),
      ]);
      if (aspRes.ok) {
        const aspData = await aspRes.json();
        state.aspects = aspData.aspects || [];
      }
      if (issRes.ok) {
        const issData = await issRes.json();
        state.issues = issData.top_complaints || [];
      }
    } catch (err) {
      console.error("[ReviewIQ] Error loading aspects and issues:", err);
    }
  }

  function loadIssuesTable() {
    const tbody = document.getElementById("issues-main-tbody");
    if (tbody && state.issues) {
      tbody.innerHTML = state.issues
        .map((iss) => {
          return `
          <tr>
            <td><strong style="text-transform:capitalize;">${iss.phrase}</strong></td>
            <td><span class="badge-pill">${iss.aspect || 'General'}</span></td>
            <td>${iss.frequency} mentions</td>
            <td>${iss.percentage}%</td>
            <td><span class="badge ${iss.priority === 'HIGH' ? 'danger' : 'warning'}">${iss.priority}</span></td>
            <td style="color:var(--status-neg-text); font-weight:600;">${iss.negative_pct || 80}% negative</td>
          </tr>
        `;
        })
        .join("");
    }
  }

  // --- Review Explorer ---
  async function loadReviews() {
    try {
      const q = state.reviewsQuery;
      let url = `/api/reviews?page=${q.page}&limit=${q.limit}&sort=${q.sort}`;
      if (q.seller_id) url += `&seller_id=${encodeURIComponent(q.seller_id)}`;
      if (q.product_id) url += `&product_id=${encodeURIComponent(q.product_id)}`;
      if (q.sentiment && q.sentiment !== "All") url += `&sentiment=${encodeURIComponent(q.sentiment)}`;
      if (q.search) url += `&search=${encodeURIComponent(q.search)}`;

      const res = await apiFetch(url);
      if (!res.ok) return;
      const data = await res.json();
      state.reviewsPagination.total = data.total;
      state.reviewsPagination.totalPages = data.total_pages;

      const tbody = document.getElementById("reviews-tbody");
      if (tbody) {
        if (!data.reviews || data.reviews.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:20px; color:var(--text-muted);">No reviews match your query.</td></tr>';
        } else {
          tbody.innerHTML = data.reviews
            .map((r) => {
              const isPos = r.sentiment === "POSITIVE";
              return `
              <tr>
                <td><strong>${r.product_name || 'Product'}</strong></td>
                <td><span style="color:#eab308;">★</span> ${r.rating || 5}</td>
                <td><span class="badge ${isPos ? 'success' : 'danger'}">${r.sentiment}</span></td>
                <td>${Math.round(r.sentiment_score * 100)}%</td>
                <td style="max-width:380px; white-space:normal; line-height:1.4;">
                  ${r.title ? `<strong>${r.title}</strong> — ` : ''}${r.review}
                </td>
                <td style="font-size:11px; color:var(--text-muted);">${(r.review_date || '').split('T')[0]}</td>
              </tr>
            `;
            })
            .join("");
        }
      }

      const pagText = document.getElementById("reviews-pagination-text");
      if (pagText) {
        const start = (data.page - 1) * data.limit + 1;
        const end = Math.min(data.total, data.page * data.limit);
        pagText.textContent = `Showing ${data.total > 0 ? start : 0} to ${end} of ${data.total} reviews`;
      }
    } catch (err) {
      console.error("[ReviewIQ] Error loading reviews:", err);
    }
  }

  window.filterReviews = () => {
    state.reviewsQuery.page = 1;
    const sellerFilter = document.getElementById("review-filter-seller");
    const sentFilter = document.getElementById("review-filter-sentiment");
    const searchInput = document.getElementById("review-search-input");

    if (sellerFilter) state.reviewsQuery.seller_id = sellerFilter.value;
    if (sentFilter) state.reviewsQuery.sentiment = sentFilter.value;
    if (searchInput) state.reviewsQuery.search = searchInput.value.trim();
    loadReviews();
  };

  window.changeReviewPage = (delta) => {
    const next = state.reviewsQuery.page + delta;
    if (next >= 1 && next <= state.reviewsPagination.totalPages) {
      state.reviewsQuery.page = next;
      loadReviews();
    }
  };

  // --- Settings View ---
  async function loadSettingsView() {
    if (!state.user) return;
    document.getElementById("settings-profile-name").value = state.user.full_name;
    document.getElementById("settings-profile-email").value = state.user.email;
    document.getElementById("settings-org-name").value = state.user.organization_name;
    document.getElementById("settings-org-type").value = state.user.organization_type.toUpperCase();

    try {
      const [teamRes, auditRes] = await Promise.all([
        apiFetch("/api/settings/team"),
        apiFetch("/api/settings/audit"),
      ]);

      if (teamRes.ok) {
        const tdata = await teamRes.json();
        const tbody = document.getElementById("settings-team-tbody");
        if (tbody && tdata.members) {
          tbody.innerHTML = tdata.members
            .map((m) => `
              <tr>
                <td><strong>${m.full_name}</strong></td>
                <td>${m.email}</td>
                <td><span class="badge primary">${m.role.toUpperCase()}</span></td>
                <td style="font-size:12px; color:var(--text-muted);">${(m.created_at || '').split('T')[0]}</td>
              </tr>
            `)
            .join("");
        }
      }

      if (auditRes.ok) {
        const adata = await auditRes.json();
        const tbody = document.getElementById("settings-audit-tbody");
        if (tbody && adata.audit_logs) {
          tbody.innerHTML = adata.audit_logs
            .map((l) => `
              <tr>
                <td><span class="badge ${l.action === 'SIGNUP' || l.action === 'LOGIN' ? 'success' : 'primary'}">${l.action}</span></td>
                <td>${l.resource_type}</td>
                <td>${l.user_email || 'System'}</td>
                <td style="font-size:11px; color:var(--text-muted);">${(l.created_at || '').replace('T', ' ').slice(0, 19)}</td>
              </tr>
            `)
            .join("");
        }
      }
    } catch (err) {
      console.error("[ReviewIQ] Error loading settings data:", err);
    }
  }

  // --- Charts Rendering ---
  function renderSentimentCharts() {
    // 1. Sentiment Pie Chart
    const pieCanvas = document.getElementById("chart-sentiment-pie");
    if (pieCanvas) {
      if (state.charts.pie) state.charts.pie.destroy();
      const posPct = state.sellerData?.metrics?.positive_pct || state.platformData?.overall_positive_pct || 84;
      const negPct = roundTo(100 - posPct, 1);

      state.charts.pie = new Chart(pieCanvas, {
        type: "doughnut",
        data: {
          labels: ["Positive Feedback", "Negative Feedback"],
          datasets: [{
            data: [posPct, negPct],
            backgroundColor: ["#10b981", "#ef4444"],
            borderWidth: 0,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: "bottom" },
          },
        },
      });
    }

    // 2. Aspect Bar Chart
    const barCanvas = document.getElementById("chart-aspects-bar");
    if (barCanvas && state.aspects.length > 0) {
      if (state.charts.aspects) state.charts.aspects.destroy();
      const labels = state.aspects.map((a) => a.aspect);
      const posData = state.aspects.map((a) => a.positive_pct);
      const negData = state.aspects.map((a) => a.negative_pct);

      state.charts.aspects = new Chart(barCanvas, {
        type: "bar",
        data: {
          labels,
          datasets: [
            { label: "Positive %", data: posData, backgroundColor: "#10b981" },
            { label: "Negative %", data: negData, backgroundColor: "#ef4444" },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { stacked: true },
            y: { stacked: true, max: 100 },
          },
        },
      });
    }
  }

  function roundTo(num, decimals) {
    return Number(Math.round(num + "e" + decimals) + "e-" + decimals);
  }

  // =========================================================================
  // 6. Live Analyzer Inference
  // =========================================================================
  window.fillSampleLiveReview = (idx) => {
    const input = document.getElementById("live-review-input");
    if (idx === 1) {
      input.value = "The battery life is phenomenal on flights, lasting over 30 hours without recharging. Earcups are plush and comfortable.";
    } else if (idx === 2) {
      input.value = "Headband plastic feels fragile and cracked after 3 months of normal use. The hinge is very cheap.";
    } else if (idx === 3) {
      input.value = "Sound is okay, but customer support took three weeks to reply and refused to send a replacement charging cable.";
    }
  };

  window.runLivePredict = async () => {
    const input = document.getElementById("live-review-input");
    const text = input.value.trim();
    if (!text) {
      alert("Please enter a review text to analyze.");
      return;
    }

    const btn = document.getElementById("btn-run-live-predict");
    btn.disabled = true;
    btn.textContent = "Analyzing...";

    try {
      const res = await apiFetch("/api/predict", {
        method: "POST",
        body: JSON.stringify({ review: text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Inference failed");

      // Show result box
      const box = document.getElementById("live-result-box");
      box.style.display = "block";

      const predSent = document.getElementById("live-pred-sentiment");
      predSent.textContent = data.sentiment;
      predSent.style.color = data.sentiment === "POSITIVE" ? "var(--status-pos)" : "var(--status-neg)";

      document.getElementById("live-pred-confidence").textContent = `Confidence: ${data.confidence}% (Pos: ${data.prob_positive}%, Neg: ${data.prob_negative}%)`;
      document.getElementById("live-pred-model").textContent = data.model_used;

      // Aspect breakdown
      const aspectsList = document.getElementById("live-pred-aspects-list");
      if (aspectsList) {
        if (!data.aspects || data.aspects.length === 0) {
          aspectsList.innerHTML = '<div style="font-size:12px; color:var(--text-muted);">No domain aspects explicitly detected in this sample.</div>';
        } else {
          aspectsList.innerHTML = data.aspects
            .map((a) => `
              <div style="padding:8px 12px; background:#ffffff; border:1px solid var(--border-light); border-radius:6px; display:flex; justify-content:space-between; align-items:center;">
                <div>
                  <strong>${a.aspect}</strong>
                  <span style="font-size:11px; color:var(--text-muted); margin-left:6px;">"${a.context}"</span>
                </div>
                <span class="badge ${a.sentiment === 'Positive' ? 'success' : 'danger'}">${a.sentiment} (${a.confidence}%)</span>
              </div>
            `)
            .join("");
        }
      }

      // Complaints list
      const compList = document.getElementById("live-pred-complaints-list");
      if (compList) {
        if (!data.complaints || data.complaints.length === 0) {
          compList.innerHTML = '<span class="badge success">No recurring complaint patterns matched</span>';
        } else {
          compList.innerHTML = data.complaints
            .map((c) => `<span class="badge danger" style="text-transform:capitalize;">${c.phrase} (${c.aspect})</span>`)
            .join("");
        }
      }
    } catch (err) {
      alert("Prediction error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Run Spark Inference";
    }
  };

  // =========================================================================
  // 7. Modals (Product Detail, Add Product, Add Seller, Invite Member)
  // =========================================================================

  // --- Product Detail Modal ---
  window.openProductDetailModal = async (productId) => {
    try {
      const res = await apiFetch(`/api/products/${productId}`);
      if (!res.ok) {
        const err = await res.json();
        alert(err.error || "Cannot view this product");
        return;
      }
      const data = await res.json();
      const p = data.product;

      document.getElementById("pmodal-title").textContent = p.name;
      document.getElementById("pmodal-meta").textContent = `SKU: ${p.sku || 'N/A'} · Category: ${p.category} · Seller: ${p.seller_name || ''}`;

      document.getElementById("pmodal-kpi-health").textContent = `${data.health_score} / 100`;
      document.getElementById("pmodal-kpi-reviews").textContent = Number(data.total_reviews).toLocaleString();
      document.getElementById("pmodal-kpi-pos").textContent = `${data.positive_pct}%`;
      document.getElementById("pmodal-kpi-neg").textContent = `${data.negative_pct}%`;

      // Aspects
      const aspList = document.getElementById("pmodal-aspects-list");
      if (aspList) {
        aspList.innerHTML = data.aspects
          .map((a) => `
            <div style="padding:8px 12px; background:#f8fafc; border:1px solid var(--border-light); border-radius:6px; display:flex; justify-content:space-between; align-items:center;">
              <div>
                <strong>${a.aspect}</strong>
                <span style="font-size:11px; color:var(--text-muted); margin-left:8px;">${a.mentions} mentions</span>
              </div>
              <div>
                <span style="color:var(--status-pos); font-weight:600; font-size:12px;">${a.positive_pct}% Pos</span>
                <span style="color:var(--status-neg); font-weight:600; font-size:12px; margin-left:8px;">${a.negative_pct}% Neg</span>
              </div>
            </div>
          `)
          .join("");
      }

      // Complaints
      const issList = document.getElementById("pmodal-issues-list");
      if (issList) {
        if (!data.issues || data.issues.length === 0) {
          issList.innerHTML = '<div style="font-size:12px; color:var(--text-muted);">No major complaint patterns detected.</div>';
        } else {
          issList.innerHTML = data.issues
            .map((iss) => `
              <div style="padding:8px 12px; background:#f8fafc; border:1px solid var(--border-light); border-radius:6px; display:flex; justify-content:space-between; align-items:center;">
                <span style="font-weight:600; text-transform:capitalize;">${iss.phrase}</span>
                <span class="badge ${iss.priority === 'HIGH' ? 'danger' : 'warning'}">${iss.priority}</span>
              </div>
            `)
            .join("");
        }
      }

      // Reviews
      const revList = document.getElementById("pmodal-reviews-list");
      if (revList) {
        if (!data.reviews || data.reviews.length === 0) {
          revList.innerHTML = '<div style="font-size:12px; color:var(--text-muted);">No reviews for this product yet.</div>';
        } else {
          revList.innerHTML = data.reviews
            .map((r) => `
              <div style="padding:10px 12px; background:#f8fafc; border:1px solid var(--border-light); border-radius:6px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
                  <span style="font-size:12px; font-weight:700;">${r.title || 'Review'}</span>
                  <span class="badge ${r.sentiment === 'POSITIVE' ? 'success' : 'danger'}">${r.sentiment}</span>
                </div>
                <div style="font-size:12px; color:var(--text-muted); line-height:1.4;">${r.review}</div>
              </div>
            `)
            .join("");
        }
      }

      document.getElementById("modal-product-detail").classList.remove("hidden");
    } catch (err) {
      alert("Error loading product: " + err.message);
    }
  };

  window.closeProductModal = () => {
    document.getElementById("modal-product-detail").classList.add("hidden");
  };

  // --- Add Product Modal ---
  window.openAddProductModal = () => {
    if (state.user.role === "viewer") {
      alert("Forbidden: Viewer role has read-only access.");
      return;
    }
    document.getElementById("modal-add-product").classList.remove("hidden");
  };

  window.closeAddProductModal = () => {
    document.getElementById("modal-add-product").classList.add("hidden");
  };

  window.submitAddProduct = async (e) => {
    e.preventDefault();
    const name = document.getElementById("new-prod-name").value.trim();
    const sku = document.getElementById("new-prod-sku").value.trim();
    const category = document.getElementById("new-prod-category").value.trim();
    const description = document.getElementById("new-prod-desc").value.trim();

    try {
      const res = await apiFetch("/api/products", {
        method: "POST",
        body: JSON.stringify({ name, sku, category, description }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to create product");

      alert("Product created successfully!");
      window.closeAddProductModal();
      await loadProducts();
      if (state.user.organization_type === "seller") loadSellerData();
      else loadPlatformData();
    } catch (err) {
      alert("Error: " + err.message);
    }
  };

  // --- Add / Invite Seller Modal (Platform Only) ---
  window.openAddSellerModal = () => {
    document.getElementById("modal-add-seller").classList.remove("hidden");
  };

  window.closeAddSellerModal = () => {
    document.getElementById("modal-add-seller").classList.add("hidden");
  };

  window.submitAddSeller = async (e) => {
    e.preventDefault();
    const company_name = document.getElementById("seller-company-name").value.trim();
    const contact_email = document.getElementById("seller-contact-email").value.trim();

    try {
      const res = await apiFetch("/api/platform/sellers", {
        method: "POST",
        body: JSON.stringify({ company_name, contact_email }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to add seller");

      alert(data.message);
      window.closeAddSellerModal();
      await loadPlatformData();
      await loadPlatformSellers();
    } catch (err) {
      alert("Error: " + err.message);
    }
  };

  window.inspectSeller = (sellerId) => {
    // Navigate to Review Explorer or filter products for this seller
    state.reviewsQuery.seller_id = sellerId;
    const revSellerFilter = document.getElementById("review-filter-seller");
    if (revSellerFilter) revSellerFilter.value = sellerId;
    navigateTo("reviews");
  };

  // --- Invite Member Modal ---
  window.openInviteMemberModal = () => {
    document.getElementById("modal-invite-member").classList.remove("hidden");
  };

  window.closeInviteMemberModal = () => {
    document.getElementById("modal-invite-member").classList.add("hidden");
  };

  window.submitInviteMember = async (e) => {
    e.preventDefault();
    const email = document.getElementById("invite-member-email").value.trim();
    const role = document.getElementById("invite-member-role").value;

    try {
      const res = await apiFetch("/api/settings/invite", {
        method: "POST",
        body: JSON.stringify({ email, role }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to invite member");

      alert(data.message);
      window.closeInviteMemberModal();
      loadSettingsView();
    } catch (err) {
      alert("Error: " + err.message);
    }
  };

  // --- Upload & Import Reviews ---
  let validatedUploadData = null;

  window.handleUploadValidate = async () => {
    const fileInput = document.getElementById("upload-file-input");
    if (!fileInput.files || fileInput.files.length === 0) {
      alert("Please select a CSV or Excel file to upload.");
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
      const res = await apiFetch("/api/reviews/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Validation failed");

      validatedUploadData = data;
      showUploadValidationResults(data);
    } catch (err) {
      alert("Upload error: " + err.message);
    }
  };

  window.handleUploadSample = async () => {
    try {
      const res = await apiFetch("/api/reviews/upload", {
        method: "POST",
        body: JSON.stringify({ sample: true }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Validation failed");

      validatedUploadData = data;
      showUploadValidationResults(data);
    } catch (err) {
      alert("Upload error: " + err.message);
    }
  };

  function showUploadValidationResults(data) {
    const box = document.getElementById("upload-validation-box");
    const stats = document.getElementById("upload-validation-stats");
    box.style.display = "block";

    stats.innerHTML = `
      <div><strong>File Name:</strong> ${data.filename}</div>
      <div><strong>Total Rows Detected:</strong> ${Number(data.row_count).toLocaleString()}</div>
      <div><strong>Valid Review Rows:</strong> ${Number(data.valid_rows || data.row_count).toLocaleString()}</div>
      <div><strong>Detected Text Column:</strong> <code>${data.detected_text_column || 'content'}</code></div>
      <div><strong>Columns Identified:</strong> ${data.columns.join(", ")}</div>
    `;
  }

  window.runImportAndAnalysis = async () => {
    if (state.user.role === "viewer") {
      alert("Forbidden: Viewer role does not have permission to run analysis.");
      return;
    }

    const prodSelect = document.getElementById("upload-target-product");
    const productId = prodSelect ? prodSelect.value : null;

    if (!productId) {
      alert("Please select or create a target product first.");
      return;
    }

    const btn = document.getElementById("btn-run-pipeline-import");
    btn.disabled = true;
    btn.textContent = "Processing with Apache Spark MLlib...";

    try {
      const res = await apiFetch("/api/reviews/import", {
        method: "POST",
        body: JSON.stringify({
          product_id: productId,
          dataset_name: validatedUploadData ? validatedUploadData.filename : "Imported Reviews Batch",
          reviews: validatedUploadData ? validatedUploadData.preview : [],
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Analysis failed");

      alert(`Spark ML pipeline completed!\n\nImported ${data.analysis.total_reviews} reviews.\nProduct Health: ${data.health_score}/100\nModel Accuracy: 90.55%`);
      document.getElementById("upload-validation-box").style.display = "none";

      await loadInitialData();
      navigateTo("overview");
    } catch (err) {
      alert("Pipeline error: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "Import & Run Spark Pipeline";
    }
  };

  // Run Startup Initialization
  checkAuthAndInit();
});
