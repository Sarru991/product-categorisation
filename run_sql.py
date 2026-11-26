"""Loads the data into SQLite and runs queries.sql, so the labelling and QA steps are reproducible in SQL."""
import sqlite3, pandas as pd, taxonomy as T
con = sqlite3.connect(":memory:")
d = pd.read_csv("data.csv")
t = d.product_category_tree.map(T.parse_tree)
d["l1"], d["l2"] = t.str[0], t.str[1]
d[["pid", "product_name", "brand", "l1", "l2"]].to_sql("products", con, index=False)
rows = [(k, None, v) for k, v in T.L1_MAP.items()] + [("Clothing", k, v) for k, v in T.CLOTHING_L2.items()]
pd.DataFrame(rows, columns=["l1", "l2", "client_category"]).to_sql("taxonomy_map", con, index=False)
pd.read_csv("test_predictions.csv").to_sql("predictions", con, index=False)
for i, q in enumerate([q.strip() for q in open("queries.sql").read().split(";") if q.strip()], 1):
    if q.upper().lstrip("-0123456789. ").startswith("CREATE") or "CREATE TABLE" in q:
        con.execute(q); n = con.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        print(f"\n[{i}] labels table created: {n} rows (pandas pipeline: 19,664)")
    else:
        print(f"\n[{i}] {q.splitlines()[0]}"); print(pd.read_sql(q, con).head(12).to_string(index=False))
