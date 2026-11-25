"""Test suite. Prints a runtime 'N passed' line — quote that line,
never a count of check() call sites (checks inside loops run many times).
Run: python -m tests.test_all  (from the project root)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_gen import CARDS, CATEGORIES, ROTATING, earn_rate, generate
from src.evaluate import accuracy, auc, lift_vs_naive, mape, precision_recall_at_k
from src.features import FEATURES, build_card_choice_frame, build_churn_frame, build_yield_frame
from src.models import card_choice_models, churn_models, rule_churn_predict, yield_models
from src.routing import best_card, route_transactions, routing_summary

PASSED = 0
FAILED = 0


def check(name, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        print(f"FAIL: {name}")


def main():
    global PASSED, FAILED
    txns, users = generate(n_users=1000)  # 1000 keeps the suite fast; run_all.py uses the full 2000
    # --- Data generation ---
    check("1000 users", len(users) == 1000)
    check("5 cards in catalogue", len(CARDS) == 5)
    check("7 categories", set(txns["category"].unique()) <= set(CATEGORIES))
    check("12 months present", set(txns["month"].unique()) <= set(range(1, 13)))
    check("amounts positive", (txns["amount"] > 0).all())
    check("points non-negative", (txns["points_earned"] >= 0).all())
    held = {r.user_id: set(r.held_cards.split(";")) for r in users.itertuples()}
    check("card_used is held by the user",
          all(r.card_used in held[r.user_id] for r in txns.itertuples()))
    check("each user holds 2-4 cards",
          users["held_cards"].str.split(";").str.len().between(2, 4).all())
    check("seed reproducible",
          generate(n_users=1000)[0].equals(txns) and generate(n_users=1000)[1].equals(users))
    check("different seed differs", not generate(seed=7, n_users=1000)[0].equals(txns))
    # Points equal amount x earn rate for a spot-checked transaction.
    r0 = txns.iloc[0]
    check("points = amount x earn rate",
          abs(r0["points_earned"] - round(r0["amount"] * earn_rate(r0["card_used"], r0["category"], r0["date"]), 2)) < 0.01)
    check("rotator 5% in its quarter",
          earn_rate("Rotator", ROTATING[1], pd.Timestamp("2026-01-15")) == 0.05)
    check("rotator 1% outside its quarter",
          earn_rate("Rotator", "Travel", pd.Timestamp("2026-01-15")) == 0.01)

    # --- Routing ---
    routed = route_transactions(txns, users)
    check("optimal points >= actual points",
          (routed["optimal_points"] >= routed["points_earned"] - 1e-9).all())
    check("points left on table >= 0", (routed["points_left_on_table"] >= 0).all())
    check("router agrees is boolean", set(routed["router_agrees"].unique()) <= {True, False})
    check("router agrees sometimes but not always",
          0.0 < routed["router_agrees"].mean() < 1.0)
    check("best card for Dining in Q1 is Rotator (5%)",
          best_card(["Apex Flat", "Bistro", "Rotator"], "Dining", pd.Timestamp("2026-01-10"))[0] == "Rotator")
    check("best card respects the cap once spent",
          best_card(["Bistro", "Rotator"], "Dining", pd.Timestamp("2026-01-10"),
                    rotator_spent_qtr=2000.0)[0] == "Bistro")
    by_cat, total = routing_summary(routed)
    check("routing lift positive", total["lift_pct"] > 0)
    check("routing lift below 100% (no magic)", total["lift_pct"] < 100)
    check("one row per category", len(by_cat) == 7)
    for _, row in by_cat.iterrows():
        check(f"{row['category']}: optimal >= actual",
              row["optimal_points"] >= row["actual_points"] - 1e-9)
        check(f"{row['category']}: yield in (0, 6]%",
              0 < row["yield_optimal"] <= 6)

    # --- Features ---
    churn = build_churn_frame(routed)
    check("churn frame one row per user", len(churn) == 1000)
    check("churn is 0/1", set(churn["churn"].unique()) <= {0, 1})
    check("churn rate between 5% and 25%", 0.05 < churn["churn"].mean() < 0.25)
    check("no NaN in churn features", churn[FEATURES].notna().all().all())
    check("feature columns present", all(c in churn.columns for c in FEATURES))
    yld = build_yield_frame(routed)
    check("yield frame has 11 months of rows", yld["target_month"].nunique() == 11)
    check("next-month points finite", np.isfinite(yld["next_month_points"]).all())
    check("next-month points non-negative", (yld["next_month_points"] >= 0).all())

    # --- Churn models (quick fit on a split) ---
    Xc, yc = churn[FEATURES], churn["churn"]
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.25, random_state=42, stratify=yc)
    rule_pred = rule_churn_predict(Xte)
    check("rule predicts 0/1", set(np.unique(rule_pred)) <= {0, 1})
    check("rule AUC in (0.5, 1)", 0.5 < auc(yte, rule_pred) < 1.0)
    for name, m in churn_models().items():
        m.fit(Xtr, ytr)
        score = m.predict_proba(Xte)[:, 1]
        check(f"churn {name}: AUC in (0.5, 1)", 0.5 < auc(yte, score) < 1.0)
        check(f"churn {name}: accuracy in (0.5, 1)", 0.5 < accuracy(yte, m.predict(Xte)) < 1.0)
        p20, r20 = precision_recall_at_k(yte, score)
        check(f"churn {name}: precision@20 in [0,1]", 0 <= p20 <= 1)
        check(f"churn {name}: recall@20 in [0,1]", 0 <= r20 <= 1)

    # --- Yield models ---
    Xy, yy = yld[FEATURES], yld["next_month_points"]
    tr = yld["target_month"] <= 8
    for name, m in yield_models().items():
        m.fit(Xy[tr], yy[tr])
        pred = m.predict(Xy[~tr])
        check(f"yield {name}: MAPE finite and < 200%", np.isfinite(mape(yy[~tr], pred)) and mape(yy[~tr], pred) < 200)
        check(f"yield {name}: predictions finite", np.isfinite(pred).all())

    # --- Card-choice agreement with the rule ---
    Xcc, ycc, _ = build_card_choice_frame(routed, users)
    rng = np.random.default_rng(42)
    s = rng.choice(len(Xcc), size=min(20000, len(Xcc)), replace=False)
    Xtr, Xte, ytr, yte = train_test_split(Xcc.iloc[s], ycc.iloc[s], test_size=0.25,
                                          random_state=42, stratify=ycc.iloc[s])
    for name, m in card_choice_models().items():
        m.fit(Xtr, ytr)
        agree = accuracy(yte, m.predict(Xte))
        check(f"card choice {name}: agreement in (0.3, 0.99)", 0.3 < agree < 0.99)

    # --- Metric helpers ---
    check("auc single class returns 0.5", auc(np.ones(10), np.random.rand(10)) == 0.5)
    check("auc perfect ranking is 1.0",
          abs(auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) - 1.0) < 1e-9)
    check("mape of perfect forecast is 0",
          mape(np.array([100.0, 200.0]), np.array([100.0, 200.0])) == 0.0)
    check("lift vs naive is 25% on 100->125",
          abs(lift_vs_naive(100, 125) - 25.0) < 1e-9)
    p, r = precision_recall_at_k(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.9, 0.8]), k=0.5)
    check("precision@50% is 1.0 on toy example", abs(p - 1.0) < 1e-9)
    check("recall@50% is 1.0 on toy example", abs(r - 1.0) < 1e-9)

    print(f"{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
