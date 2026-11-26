import json, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = json.load(open("report.json")); p = pd.read_csv("test_predictions.csv")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
v = [100*R["acc_random_split_LEAKY"], 100*R["model_acc_test"]]
ax[0].bar(["Random split\n(44% of test names\nalso in train)", "Grouped split\n(no shared names)"], v, color=["#d62728", "#1f77b4"])
for i, x in enumerate(v): ax[0].text(i, x+0.1, f"{x:.1f}%", ha="center")
ax[0].set(ylim=(95, 99.5), ylabel="Test accuracy (%)", title="Leakage check: duplicate listings inflate accuracy")
for key, lab, c in [("test_tradeoff_curve", "v1: rules override model", "#d62728"), ("model_only_curve", "Model only", "#999"), ("v2_curve", "v2: rule + model must agree", "#1f77b4")]:
    cv = R[key]; ax[1].plot([100*x["review_share"] for x in cv], [100*x["auto_acc"] for x in cv], "o-", color=c, label=lab)
ax[1].axhline(98, ls="--", color="k", lw=.8); ax[1].text(30, 98.1, "98% target", fontsize=8)
ax[1].set(xlabel="Share of products sent to manual review (%)", ylabel="Accuracy of auto-tagged products (%)", title="Accuracy vs manual review load (test set)", xlim=(-1, 46)); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
acc = (100*(p.pred == p.label)).groupby(p.label).mean().sort_values()
ax[2].barh(acc.index, acc.values, color=["#d62728" if x < 95 else "#1f77b4" for x in acc.values])
for i, x in enumerate(acc.values): ax[2].text(x+0.3, i, f"{x:.0f}%", va="center", fontsize=8)
ax[2].set(xlim=(50, 103), xlabel="Accuracy before review (%)", title="Per-category accuracy (v2, test set)")
plt.tight_layout(); plt.savefig("chart.png", dpi=130)
