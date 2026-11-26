-- 1. Data quality: products whose category tree has no proper top level (broken trees)
SELECT COUNT(*) AS broken_tree_rows
FROM products p LEFT JOIN taxonomy_map m ON p.l1 = m.l1 AND (m.l2 IS NULL OR p.l2 = m.l2)
WHERE m.client_category IS NULL;

-- 2. Ground-truth client labels via the mapping table (clothing mapped on level 2, the rest on level 1)
CREATE TABLE labels AS
SELECT p.pid, p.product_name, p.brand, m.client_category
FROM products p JOIN taxonomy_map m ON p.l1 = m.l1 AND (m.l2 IS NULL OR p.l2 = m.l2);

-- 3. Missing brands by client category
SELECT client_category, COUNT(*) AS n, ROUND(100.0 * SUM(brand IS NULL) / COUNT(*), 1) AS pct_brand_missing
FROM labels GROUP BY client_category ORDER BY pct_brand_missing DESC;

-- 4. Accuracy and review load per category on the test set
SELECT label AS category, COUNT(*) AS n,
       ROUND(100.0 * AVG(pred = label), 1) AS accuracy_pct,
       ROUND(100.0 * AVG(conf < 0.90), 1) AS review_pct_at_0_90
FROM predictions GROUP BY label ORDER BY accuracy_pct;

-- 5. Today's manual-review queue: least confident first
SELECT pid, product_name, pred AS suggested_category, ROUND(conf, 2) AS confidence
FROM predictions WHERE conf < 0.90 ORDER BY conf LIMIT 20;

-- 6. Most common confusions (window function: share of each category's errors)
SELECT label, pred, n, ROUND(100.0 * n / SUM(n) OVER (PARTITION BY label), 1) AS pct_of_category_errors
FROM (SELECT label, pred, COUNT(*) AS n FROM predictions WHERE pred <> label GROUP BY label, pred)
ORDER BY n DESC LIMIT 10;
