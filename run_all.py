"""Run the whole pipeline: data -> routing -> features -> models ->
charts + metrics JSON. Charts regenerate with this script (PNGs are
gitignored and excluded from any GitHub push).
Usage: python run_all.py   (from the project root)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.data_gen import generate, save_synthetic
from src.evaluate import accuracy, auc, lift_vs_naive, mape, precision_recall_at_k
from src.features import FEATURES, build_card_choice_frame, build_churn_frame, build_yield_frame
from src.models import card_choice_models, churn_models, rule_churn_predict, yield_models
from src.routing import route_transactions, routing_summary

ROOT = Path(__file__).resolve().parent
CHARTS = ROOT / "reports" / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
NAVY, GREEN, GOLD, RED = "#1A3A6B", "#2E7D5B", "#8A6D00", "#B03A2E"


def main():
    txns, users = generate()
    save_synthetic(txns, users, ROOT / "data" / "synthetic")
    routed = route_transactions(txns, users)
    routed.to_csv(ROOT / "data" / "processed" / "routed_transactions.csv", index=False)
    by_cat, total = routing_summary(routed)

    # ---- Churn: features months 1-11, label = silent in month 12 ----
    churn = build_churn_frame(routed)
    Xc, yc = churn[FEATURES], churn["churn"]
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.25, random_state=42,
                                          stratify=yc)
    churn_res = {}
    rule_pred = rule_churn_predict(Xte)
    churn_res["rule"] = {"auc": auc(yte, rule_pred), "accuracy": accuracy(yte, rule_pred),
                               "precision_at_20": precision_recall_at_k(yte, rule_pred)[0],
                               "recall_at_20": precision_recall_at_k(yte, rule_pred)[1]}
    models = churn_models()
    for name, m in models.items():
        m.fit(Xtr, ytr)
        score = m.predict_proba(Xte)[:, 1]
        pred = m.predict(Xte)
        p20, r20 = precision_recall_at_k(yte, score)
        churn_res[name] = {"auc": auc(yte, score), "accuracy": accuracy(yte, pred),
                           "precision_at_20": p20, "recall_at_20": r20}
    rf_churn = models["random_forest"]
    feat_imp = sorted(zip(FEATURES, rf_churn.feature_importances_),
                      key=lambda t: -t[1])

    # ---- Yield: next-month points; time split (train target_month<=8) ----
    yld = build_yield_frame(routed)
    Xy, yy = yld[FEATURES], yld["next_month_points"]
    tr_mask = yld["target_month"] <= 8
    y_models = yield_models()
    yield_res = {"mean_baseline": {"mape": mape(yy[~tr_mask], np.full((~tr_mask).sum(), yy[tr_mask].mean())),
                                   "r2": 0.0}}
    for name, m in y_models.items():
        m.fit(Xy[tr_mask], yy[tr_mask])
        pred = m.predict(Xy[~tr_mask])
        yield_res[name] = {"mape": mape(yy[~tr_mask], pred),
                           "r2": float(r2_score(yy[~tr_mask], pred))}
    rf_yield_pred = y_models["random_forest"].predict(Xy[~tr_mask])

    # ---- Card choice: agreement with the rule (sample for speed) ----
    Xcc, ycc, cc_df = build_card_choice_frame(routed, users)
    sample = np.random.default_rng(42).choice(len(Xcc), size=min(40000, len(Xcc)), replace=False)
    Xs, ys = Xcc.iloc[sample], ycc.iloc[sample]
    Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.25, random_state=42,
                                          stratify=ys)
    choice_res = {}
    for name, m in card_choice_models().items():
        m.fit(Xtr, ytr)
        choice_res[name] = {"agreement": accuracy(yte, m.predict(Xte))}

    metrics = {
        "n_users": int(len(users)),
        "n_txns": int(len(txns)),
        "n_months": 12,
        "churn_rate": float(churn["churn"].mean()),
        "routing_total": total,
        "routing_by_category": by_cat.to_dict(orient="records"),
        "churn": churn_res,
        "churn_feature_importance": [{"feature": f, "importance": round(float(v), 4)}
                                     for f, v in feat_imp],
        "yield": yield_res,
        "card_choice": choice_res,
        "synthetic_data": True,
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # ---- Charts ----
    # 1. Rewards yield by category: actual vs optimal
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(by_cat))
    ax.bar(x - 0.2, by_cat["yield_actual"], 0.4, label="Actual (user's card)", color=NAVY)
    ax.bar(x + 0.2, by_cat["yield_optimal"], 0.4, label="Optimal routing", color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(by_cat["category"], rotation=25)
    ax.set_ylabel("Points per $100 spent")
    ax.set_title("Rewards yield by category — actual vs optimal routing (synthetic data)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(CHARTS / "yield_by_category.png"); plt.close(fig)

    # 2. Routing lift by category
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(by_cat["category"], by_cat["lift_pct"], color=GOLD)
    ax.set_ylabel("Extra points from optimal routing (%)")
    ax.set_title("Where the points are left on the table — routing lift by category")
    plt.xticks(rotation=25)
    fig.tight_layout(); fig.savefig(CHARTS / "routing_lift.png"); plt.close(fig)

    # 3. Churn AUC by model
    fig, ax = plt.subplots(figsize=(9, 4))
    names = list(churn_res); vals = [churn_res[n]["auc"] for n in names]
    ax.bar(names, vals, color=[RED, NAVY, GREEN])
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("Out-of-sample AUC"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Churn prediction — AUC by model (hold-out, months 1-11 -> month 12)")
    fig.tight_layout(); fig.savefig(CHARTS / "churn_auc.png"); plt.close(fig)

    # 4. Churn feature importance (top 8)
    top = feat_imp[:8]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh([f for f, _ in reversed(top)], [v for _, v in reversed(top)], color=NAVY)
    ax.set_title("What predicts churn — random forest feature importance")
    fig.tight_layout(); fig.savefig(CHARTS / "churn_feature_importance.png"); plt.close(fig)

    # 5. Monthly points: actual vs optimal
    monthly = routed.groupby("month").agg(actual=("points_earned", "sum"),
                                          optimal=("optimal_points", "sum"))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(monthly.index, monthly["actual"], marker="o", label="Actual", color=NAVY)
    ax.plot(monthly.index, monthly["optimal"], marker="o", label="Optimal routing", color=GREEN)
    ax.set_xlabel("Month"); ax.set_ylabel("Points earned")
    ax.set_title("Monthly points — actual vs optimal routing (synthetic data)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(CHARTS / "monthly_points.png"); plt.close(fig)

    # 6. Churn rate by archetype
    merged = churn.merge(users[["user_id", "archetype"]], on="user_id")
    cr = merged.groupby("archetype")["churn"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(cr.index, cr.values, color=RED)
    ax.set_ylabel("Churn rate (%)")
    ax.set_title("Churn rate by spend archetype (month-12 inactivity)")
    plt.xticks(rotation=20)
    fig.tight_layout(); fig.savefig(CHARTS / "churn_by_archetype.png"); plt.close(fig)

    # 7. Yield forecast: actual vs predicted (RF, hold-out months 9-11)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    yt, yp = yy[~tr_mask].values, rf_yield_pred
    idx = np.random.default_rng(0).choice(len(yt), size=min(1500, len(yt)), replace=False)
    ax.scatter(yt[idx], yp[idx], s=6, alpha=0.4, color=NAVY)
    lim = max(yt.max(), yp.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=RED, linestyle="--", linewidth=1)
    ax.set_xlabel("Actual next-month points"); ax.set_ylabel("Predicted next-month points")
    ax.set_title("Yield forecast — actual vs predicted (random forest, hold-out)")
    fig.tight_layout(); fig.savefig(CHARTS / "yield_actual_vs_pred.png"); plt.close(fig)

    print(f"users={len(users)} txns={len(txns)} churn_rate={churn['churn'].mean():.3f}")
    print(f"routing lift: {total['lift_pct']:.1f}%  agree_rate: {total['agree_rate']:.3f}")
    print("churn AUC:", {k: round(v['auc'], 3) for k, v in churn_res.items()})
    print("yield MAPE:", {k: round(v['mape'], 1) for k, v in yield_res.items()})
    print("card-choice agreement:", {k: round(v['agreement'], 3) for k, v in choice_res.items()})
    print(f"Charts written to {CHARTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
