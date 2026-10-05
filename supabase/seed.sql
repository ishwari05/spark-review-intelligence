-- ==============================================================================
-- ReviewIQ SaaS Database Seed: DEVELOPMENT DATA ONLY
-- ==============================================================================
-- CAUTION: This seed file is strictly for DEVELOPMENT and DEMONSTRATION purposes.
-- DO NOT RUN IN A LIVE PRODUCTION ENVIRONMENT.
--
-- To remove all seed data run:
--   DELETE FROM public.organizations WHERE id IN (
--     '00000000-0000-0000-0000-000000000001',
--     '00000000-0000-0000-0000-000000000002',
--     '00000000-0000-0000-0000-000000000003'
--   );
-- ==============================================================================

-- 1. ORGANIZATIONS
-- 1 Platform: "Apex Global Marketplace"
INSERT INTO public.organizations (id, name, organization_type, slug)
VALUES 
    ('00000000-0000-0000-0000-000000000001', 'Apex Global Marketplace', 'platform', 'apex-marketplace')
ON CONFLICT (id) DO NOTHING;

-- 2 Sellers: "Aura Sound Technologies" and "Lumina Smart Devices"
INSERT INTO public.organizations (id, name, organization_type, slug)
VALUES 
    ('00000000-0000-0000-0000-000000000002', 'Aura Sound Technologies', 'seller', 'aura-sound'),
    ('00000000-0000-0000-0000-000000000003', 'Lumina Smart Devices', 'seller', 'lumina-devices')
ON CONFLICT (id) DO NOTHING;

-- 2. PLATFORM MANAGED SELLERS
-- Link Apex Marketplace -> Aura Sound & Lumina Devices
INSERT INTO public.platform_sellers (platform_organization_id, seller_organization_id, status)
VALUES
    ('00000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002', 'active'),
    ('00000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000003', 'active')
ON CONFLICT DO NOTHING;

-- 3. PRODUCTS
-- Aura Sound Products
INSERT INTO public.products (id, organization_id, name, sku, category, description)
VALUES
    ('10000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002', 'Aura ANC Wireless Headphones', 'AURA-HP-01', 'Audio & Headphones', 'Flagship noise-canceling over-ear wireless headphones with 40mm drivers.'),
    ('10000000-0000-0000-0000-000000000002', '00000000-0000-0000-0000-000000000002', 'Aura Pulse Earbuds Pro', 'AURA-EB-02', 'Audio & Headphones', 'True wireless sports earbuds with IPX7 water resistance and deep bass.')
ON CONFLICT (id) DO NOTHING;

-- Lumina Smart Devices Products
INSERT INTO public.products (id, organization_id, name, sku, category, description)
VALUES
    ('10000000-0000-0000-0000-000000000003', '00000000-0000-0000-0000-000000000003', 'Lumina Smart Ambient Lamp', 'LUM-LAMP-01', 'Smart Home', 'RGB smart table lamp compatible with Alexa and Google Assistant.'),
    ('10000000-0000-0000-0000-000000000004', '00000000-0000-0000-0000-000000000003', 'Lumina Wi-Fi Air Purifier', 'LUM-AIR-02', 'Home Appliances', 'HEPA H13 air purifier with real-time AQI monitoring and app control.')
ON CONFLICT (id) DO NOTHING;

-- 4. DATASETS
INSERT INTO public.datasets (id, organization_id, name, file_name, row_count, status)
VALUES
    ('20000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002', 'Aura Headphones Q3 Reviews', 'aura_q3_reviews.csv', 8421, 'ready'),
    ('20000000-0000-0000-0000-000000000002', '00000000-0000-0000-0000-000000000003', 'Lumina Smart Home Reviews', 'lumina_annual_reviews.csv', 6120, 'ready')
ON CONFLICT (id) DO NOTHING;

-- 5. ANALYSES
INSERT INTO public.analyses (id, organization_id, product_id, dataset_id, status, total_reviews, positive_reviews, negative_reviews, health_score, accuracy, f1_score, roc_auc, completed_at)
VALUES
    ('30000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002', '10000000-0000-0000-0000-000000000001', '20000000-0000-0000-0000-000000000001', 'completed', 8421, 7074, 1347, 84, 0.9055, 0.9054, 0.9636, NOW()),
    ('30000000-0000-0000-0000-000000000002', '00000000-0000-0000-0000-000000000003', '10000000-0000-0000-0000-000000000003', '20000000-0000-0000-0000-000000000002', 'completed', 6120, 4406, 1714, 72, 0.9055, 0.9054, 0.9636, NOW())
ON CONFLICT (id) DO NOTHING;

-- 6. REVIEWS
-- Sample Reviews for Aura ANC Wireless Headphones (Seller A)
INSERT INTO public.reviews (id, organization_id, product_id, dataset_id, title, content, rating, sentiment, sentiment_score, review_date)
VALUES
    ('40000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002', '10000000-0000-0000-0000-000000000001', '20000000-0000-0000-0000-000000000001', 'Phenomenal audio experience!', 'The noise cancellation is stellar on daily commute flights. Battery lasts over 30 hours without recharging.', 5, 'POSITIVE', 0.98, NOW() - INTERVAL '2 days'),
    ('40000000-0000-0000-0000-000000000002', '00000000-0000-0000-0000-000000000002', '10000000-0000-0000-0000-000000000001', '20000000-0000-0000-0000-000000000001', 'Great value and comfort', 'Earcups are super plush. For the price, you cannot beat the rich soundstage.', 5, 'POSITIVE', 0.94, NOW() - INTERVAL '4 days'),
    ('40000000-0000-0000-0000-000000000003', '00000000-0000-0000-0000-000000000002', '10000000-0000-0000-0000-000000000001', '20000000-0000-0000-0000-000000000001', 'Disappointed with plastic hinge', 'The headband plastic feels fragile and cracked after 3 months of normal use.', 2, 'NEGATIVE', 0.89, NOW() - INTERVAL '6 days'),
    ('40000000-0000-0000-0000-000000000004', '00000000-0000-0000-0000-000000000002', '10000000-0000-0000-0000-000000000001', '20000000-0000-0000-0000-000000000001', 'Charging cable missing in box', 'Sound is good but customer service took two weeks to send a replacement charging cable.', 3, 'NEGATIVE', 0.78, NOW() - INTERVAL '8 days')
ON CONFLICT (id) DO NOTHING;

-- Sample Reviews for Lumina Smart Ambient Lamp (Seller B)
INSERT INTO public.reviews (id, organization_id, product_id, dataset_id, title, content, rating, sentiment, sentiment_score, review_date)
VALUES
    ('40000000-0000-0000-0000-000000000005', '00000000-0000-0000-0000-000000000003', '10000000-0000-0000-0000-000000000003', '20000000-0000-0000-0000-000000000002', 'Beautiful colors and transitions', 'Syncs smoothly with my smart home setup. Soft bedtime warm light is perfect.', 5, 'POSITIVE', 0.96, NOW() - INTERVAL '1 day'),
    ('40000000-0000-0000-0000-000000000006', '00000000-0000-0000-0000-000000000003', '20000000-0000-0000-0000-000000000002', 'Wi-Fi connection drops often', 'The app keeps losing connection to the lamp every few days. Pairing setup is frustrating.', 2, 'NEGATIVE', 0.88, NOW() - INTERVAL '3 days')
ON CONFLICT (id) DO NOTHING;

-- 7. ASPECT RESULTS
INSERT INTO public.aspect_results (analysis_id, organization_id, product_id, aspect, mentions, positive_count, negative_count, positive_pct, negative_pct)
VALUES
    ('30000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002', '10000000-0000-0000-0000-000000000001', 'Battery', 2410, 2120, 290, 88.0, 12.0),
    ('30000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002', '10000000-0000-0000-0000-000000000001', 'Price / Value', 3120, 2808, 312, 90.0, 10.0),
    ('30000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002', '10000000-0000-0000-0000-000000000001', 'Build Quality', 1890, 1361, 529, 72.0, 28.0),
    ('30000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002', '10000000-0000-0000-0000-000000000001', 'Customer Support', 1001, 786, 215, 78.5, 21.5)
ON CONFLICT DO NOTHING;

-- 8. ISSUE RESULTS
INSERT INTO public.issue_results (analysis_id, organization_id, product_id, phrase, aspect, frequency, percentage, priority, negative_pct)
VALUES
    ('30000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002', '10000000-0000-0000-0000-000000000001', 'plastic hinge crack', 'Build Quality', 88, 1.04, 'HIGH', 89.2),
    ('30000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002', '10000000-0000-0000-0000-000000000001', 'bluetooth disconnects', 'Connectivity', 64, 0.76, 'MEDIUM', 82.5),
    ('30000000-0000-0000-0000-000000000001', '00000000-0000-0000-0000-000000000002', '10000000-0000-0000-0000-000000000001', 'slow charging speed', 'Battery', 42, 0.50, 'MEDIUM', 74.0)
ON CONFLICT DO NOTHING;
