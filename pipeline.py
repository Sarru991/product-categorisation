"""Product categorisation: rules + text classifier + manual-review queue, audited like a backtest.
Inputs to every method: product name, brand, description. The category tree is only used as ground truth."""
import re, json, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union
from sklearn.metrics import classification_report, accuracy_score
import taxonomy as T

RNG = 42
d = pd.read_csv("data.csv")
report = {}

# ---------- 1. data checks ----------
d["label"] = d.product_category_tree.map(T.client_label)
l1 = d.product_category_tree.map(lambda s: T.parse_tree(s)[0])
report["rows"] = len(d)
report["unlabeled_broken_tree"] = int(d.label.isna().sum())
report["broken_tree_examples"] = l1[d.label.isna()].value_counts().head(5).to_dict()
report["brand_missing_pct"] = float(d.brand.isna().mean())
report["exact_duplicate_names"] = int(d.product_name.duplicated().sum())
d = d[d.label.notna()].copy()
d["text"] = (d.product_name.fillna("") + " | " + d.brand.fillna("") + " | " + d.description.fillna("").str[:600])
d["group"] = d.product_name.str.lower().str.replace(r"\(.*?\)", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
report["labeled_rows"] = len(d); report["unique_name_groups"] = int(d.group.nunique())

def model():
    vec = make_union(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                     TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=150000))
    return vec, LogisticRegression(C=10, max_iter=2000)

def fit_predict(tr, te):
    vec, clf = model()
    clf.fit(vec.fit_transform(tr.text), tr.label)
    P = clf.predict_proba(vec.transform(te.text))
    return clf.classes_[P.argmax(1)], P.max(1)

# ---------- 2. leakage check: random split vs grouped split ----------
tr_r, te_r = train_test_split(d, test_size=0.2, random_state=RNG, stratify=d.label)
pr, _ = fit_predict(tr_r, te_r)
report["acc_random_split_LEAKY"] = accuracy_score(te_r.label, pr)
report["random_split_test_rows_with_same_name_in_train"] = float(te_r.group.isin(set(tr_r.group)).mean())

# grouped: no product name appears on both sides
g1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG)
i_rest, i_test = next(g1.split(d, groups=d.group)); rest, test = d.iloc[i_rest], d.iloc[i_test]
g2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RNG)
i_tr, i_val = next(g2.split(rest, groups=rest.group)); train, val = rest.iloc[i_tr], rest.iloc[i_val]
assert not set(test.group) & set(rest.group) and not set(val.group) & set(train.group)
report["split_sizes"] = {"train": len(train), "val": len(val), "test": len(test)}

# ---------- 3. rules alone ----------
def rules_eval(df):
    r = df.product_name.map(T.apply_rules)
    cat = r.str[0]; fired = cat.notna()
    by_rule = {}
    for name in [n for n, _, _ in T.RULES]:
        m = r.str[1] == name
        if m.sum(): by_rule[name] = {"fired": int(m.sum()), "precision": float((cat[m] == df.label[m]).mean())}
    return cat, fired, {"coverage": float(fired.mean()), "precision": float((cat[fired] == df.label[fired]).mean()),
                        "conflicts": int((r.str[1] == "conflict").sum()), "by_rule": by_rule}
_, _, report["rules_train_val"] = rules_eval(rest)          # where the rules were sanity-checked
rcat_test, rfired_test, report["rules_test"] = rules_eval(test)

# ---------- 4. model alone (train -> val for threshold, then train+val -> test) ----------
pv, cv = fit_predict(train, val)
pt, ct = fit_predict(rest, test)
report["model_acc_val"] = accuracy_score(val.label, pv); report["model_acc_test"] = accuracy_score(test.label, pt)

# ---------- 5. hybrid + review queue; threshold chosen on VAL only ----------
def hybrid(df, pred, conf, rcat):
    final = np.where(rcat.notna(), rcat, pred)
    source = np.where(rcat.notna(), "rule", "model")
    c = np.where(rcat.notna(), 1.0, conf)
    return final, source, c
rcat_val, _, _ = rules_eval(val)
fv, sv, cfv = hybrid(val, pv, cv, rcat_val)
TARGET = 0.98
thr = 0.0
for t in np.arange(0.30, 1.0, 0.01):
    auto = cfv >= t
    if auto.any() and (fv[auto] == val.label.values[auto]).mean() >= TARGET: thr = round(float(t), 2); break
report["review_threshold_from_val"] = thr

ft, st, cft = hybrid(test, pt, ct, rcat_test)
auto = cft >= thr
report["hybrid_test"] = {"overall_acc_no_review": float((ft == test.label.values).mean()),
                         "auto_tagged_share": float(auto.mean()), "auto_tagged_acc": float((ft[auto] == test.label.values[auto]).mean()),
                         "sent_to_review_share": float((~auto).mean()),
                         "rule_share": float((st == "rule").mean())}
curve = []
for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
    a = cft >= t
    curve.append({"threshold": t, "review_share": float((~a).mean()), "auto_acc": float((ft[a] == test.label.values[a]).mean())})
report["test_tradeoff_curve"] = curve

# ---------- 6. per-category and errors ----------
cr = classification_report(test.label, ft, output_dict=True, zero_division=0)
report["per_category_test"] = {k: {m: round(v[m], 3) for m in ["precision", "recall", "f1-score", "support"]} for k, v in cr.items() if k not in ("accuracy", "macro avg", "weighted avg")}
err = test.assign(pred=ft, conf=cft, source=st)
err = err[err.pred != err.label]
report["top_confusions"] = {f"{a} -> {b}": int(n) for (a, b), n in err.groupby(["label", "pred"]).size().sort_values(ascending=False).head(8).items()}
report["confident_errors_examples"] = err.sort_values("conf", ascending=False).head(12)[["product_name", "label", "pred", "conf", "source"]].to_dict("records")
# (v1 predictions are not saved; see v2 below)
json.dump(report, open("report.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in report.items() if k not in ("confident_errors_examples", "per_category_test")}, indent=1, default=float))

# ---------- 7. v2 design, decided from TRAIN/VAL evidence only ----------
# Findings on train+val (not test): the stationery (47%), kids apparel (75%) and furnishing (81%) rules are
# unreliable, and rules given confidence 1.0 can never be sent to review. Fix, fixed before re-scoring test:
#  (a) keep only rules with >= 95% precision on train+val;  (b) when a rule and the model DISAGREE -> review;
#  (c) otherwise the model's confidence decides, with the threshold chosen on val.
keep = {n for n, v in report["rules_train_val"]["by_rule"].items() if v["precision"] >= 0.95}
report["v2_rules_kept"] = sorted(keep); report["v2_rules_dropped"] = sorted(set(report["rules_train_val"]["by_rule"]) - keep)
def v2(df, pred, conf):
    r = df.product_name.map(T.apply_rules)
    rc = np.where(r.str[1].isin(keep), r.str[0], None)
    agree = (rc != None) & (rc == pred)
    disagree = (rc != None) & (rc != pred)
    c = np.where(agree, 1.0, np.where(disagree, -1.0, conf))   # -1 = always review
    return pred, c, agree, disagree
_, cv2, _, _ = v2(val, pv, cv)
thr2 = None
for t in np.arange(0.30, 1.0, 0.01):
    a = cv2 >= t
    if a.any() and (pv[a] == val.label.values[a]).mean() >= TARGET: thr2 = round(float(t), 2); break
report["v2_threshold_from_val"] = thr2
f2, ct2, ag, dg = v2(test, pt, ct)
a2 = ct2 >= thr2
report["v2_test"] = {"auto_tagged_share": float(a2.mean()), "auto_tagged_acc": float((f2[a2] == test.label.values[a2]).mean()),
                     "sent_to_review_share": float((~a2).mean()), "rule_model_agree_share": float(ag.mean()),
                     "rule_model_disagree_share": float(dg.mean()),
                     "acc_when_agree": float((f2[ag] == test.label.values[ag]).mean()),
                     "model_right_when_disagree": float((f2[dg] == test.label.values[dg]).mean())}
curve2 = []
for t in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
    a = ct2 >= t
    curve2.append({"threshold": t, "review_share": float((~a).mean()), "auto_acc": float((f2[a] == test.label.values[a]).mean())})
report["v2_curve"] = curve2
mc = [{"threshold": t, "review_share": float((ct < t).mean()), "auto_acc": float((pt[ct >= t] == test.label.values[ct >= t]).mean())} for t in [0.0,0.5,0.7,0.8,0.9,0.95,0.99]]
report["model_only_curve"] = mc
cr2 = classification_report(test.label[a2], f2[a2], output_dict=True, zero_division=0)
report["v2_per_category_auto"] = {k: {m: round(v[m], 3) for m in ["precision", "recall", "support"]} for k, v in cr2.items() if k not in ("accuracy", "macro avg", "weighted avg")}
json.dump(report, open("report.json", "w"), indent=1, default=float)
print(json.dumps({k: report[k] for k in ["v2_rules_dropped", "v2_threshold_from_val", "v2_test", "v2_curve", "model_only_curve"]}, indent=1))
print(json.dumps(report["v2_per_category_auto"], indent=0))
# final predictions file = v2 design (used by run_sql.py and the review queue)
test.assign(pred=f2, conf=np.clip(ct2, 0, 1), source=np.where(ag, "rule+model agree", np.where(dg, "rule vs model disagree", "model")),
            auto=a2)[["pid", "product_name", "brand", "label", "pred", "conf", "source", "auto"]].to_csv("test_predictions.csv", index=False)
