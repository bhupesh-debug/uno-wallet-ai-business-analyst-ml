"""User-month feature builder.

Features for month m are computed ONLY from transactions in months
<= m (no lookahead). Targets:
  - churn: 1 if the user has zero transactions in month 12
    (final-month inactivity), features from months 1-11.
  - next_month_points: points earned in month m+1, features from months <= m.
"""
import numpy as np
import pandas as pd

from src.data_gen import CATEGORIES

FEATURES = [
    "total_spend", "txn_count", "avg_txn", "active_days", "distinct_cats",
    "points_earned", "points_per_dollar", "router_lift_avail", "dining_share",
    "groceries_share", "travel_share", "online_share", "max_gap_days",
    "spend_trend_3m", "months_active",
    "spend_last_3m", "txns_last_3m", "points_last_3m",
]


def _month_frame(txns, upto_month):
    sub = txns[txns["month"] <= upto_month]
    rows = []
    for uid, g in sub.groupby("user_id"):
        g = g.sort_values("date")
        shares = {c: (g["category"] == c).mean() for c in CATEGORIES}
        gaps = g["date"].diff().dt.days.dropna()
        # Trend compares MONTHLY averages (last 3 months vs all earlier
        # months); comparing raw 3-month vs 8-month totals would read
        # about -0.6 for every stable user.
        recent = g[g["month"] > upto_month - 3]
        prior = g[g["month"] <= upto_month - 3]
        recent_m = recent["amount"].sum() / 3.0
        prior_m = prior["amount"].sum() / max(upto_month - 3, 1)
        rows.append({
            "user_id": uid,
            "total_spend": g["amount"].sum(),
            "txn_count": len(g),
            "avg_txn": g["amount"].mean(),
            "active_days": g["date"].dt.date.nunique(),
            "distinct_cats": g["category"].nunique(),
            "points_earned": g["points_earned"].sum(),
            "points_per_dollar": g["points_earned"].sum() / max(g["amount"].sum(), 1),
            "router_lift_avail": g["points_left_on_table"].sum() / max(len(g), 1),
            "dining_share": shares["Dining"],
            "groceries_share": shares["Groceries"],
            "travel_share": shares["Travel"],
            "online_share": shares["Online"],
            "max_gap_days": float(gaps.max()) if len(gaps) else 0.0,
            "spend_trend_3m": (recent_m - prior_m) / max(prior_m, 1.0),
            "months_active": g["month"].nunique(),
            "spend_last_3m": recent["amount"].sum(),
            "txns_last_3m": len(recent),
            "points_last_3m": recent["points_earned"].sum(),
        })
    return pd.DataFrame(rows)


def build_churn_frame(routed):
    """One row per user: features from months 1-11, churn = silent in month 12."""
    feat = _month_frame(routed, 11)
    m12 = routed[routed["month"] == 12].groupby("user_id").size()
    feat["churn"] = (~feat["user_id"].isin(m12.index)).astype(int)
    return feat


def build_yield_frame(routed):
    """One row per user-month (months 1-11): features to month m, target = points in m+1."""
    frames = []
    for m in range(1, 12):
        feat = _month_frame(routed, m)
        nxt = routed[routed["month"] == m + 1].groupby("user_id")["points_earned"].sum()
        feat["target_month"] = m + 1
        feat["next_month_points"] = feat["user_id"].map(nxt).fillna(0.0)
        frames.append(feat)
    return pd.concat(frames, ignore_index=True)


def build_card_choice_frame(routed, users):
    """One row per transaction: predict the optimal card.

    Features are transaction context only (category, quarter, amount
    bucket, user archetype, number of cards held). The rule's answer is
    the label — the classifier is measured on agreement with the rule.
    """
    held_n = {r.user_id: len(r.held_cards.split(";")) for r in users.itertuples()}
    arch = {r.user_id: r.archetype for r in users.itertuples()}
    df = routed.copy()
    df["quarter"] = (df["date"].dt.month - 1) // 3 + 1
    df["amount_bucket"] = pd.cut(df["amount"], [0, 25, 75, 200, 1500],
                                 labels=["small", "medium", "large", "xlarge"]).astype(str)
    df["n_cards"] = df["user_id"].map(held_n)
    df["archetype"] = df["user_id"].map(arch)
    X = pd.get_dummies(df[["category", "quarter", "amount_bucket", "n_cards", "archetype"]],
                       columns=["category", "amount_bucket", "archetype"], drop_first=False)
    return X, df["optimal_card"], df
