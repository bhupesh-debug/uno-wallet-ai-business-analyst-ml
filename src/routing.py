"""Card-routing engine: the interpretable rule baseline.

For a transaction, the optimal card is the held card with the highest
earn rate on that date — the rotating 5% applies only until the
quarterly $1,500 cap is used. The rule is optimal for points by
construction; any ML model is measured against it, not the other way
round. Points are the only objective here (fees are flagged separately
in the report, not folded into points).
"""
import numpy as np
import pandas as pd

from src.data_gen import CARDS, ROTATING, ROTATOR_CAP, earn_rate


def best_card(held, category, date, rotator_spent_qtr=0.0):
    """Highest-earn held card for one transaction; ties -> first held."""
    best, best_rate = held[0], -1.0
    for card in held:
        rate = earn_rate(card, category, date)
        if card == "Rotator" and ROTATING[(date.month - 1) // 3 + 1] == category \
                and rotator_spent_qtr >= ROTATOR_CAP:
            rate = 0.01
        if rate > best_rate:
            best, best_rate = card, rate
    return best, best_rate


def route_transactions(txns, users):
    """Add optimal card + optimal points per transaction (cap-aware).

    Vectorised. The Rotator at 5% always beats every other card on the
    rotating category (next best is 4%), so the quarterly cap is spent
    exactly by rotating-category spend, in date order; once cumulative
    rotating-category spend for the user-quarter reaches $1,500, the
    Rotator falls to 1% and the best remaining held card wins. Ties are
    broken by catalogue order; points are unaffected by tie-breaking.
    """
    txns = txns.sort_values(["user_id", "date"]).copy()
    txns["quarter"] = (txns["date"].dt.month - 1) // 3 + 1
    txns["rot_cat"] = txns["quarter"].map(ROTATING)
    txns["is_rot_cat"] = txns["category"] == txns["rot_cat"]
    # Cumulative rotating-category spend before this transaction.
    rot_spend = txns.assign(
        _rot_amt=np.where(txns["is_rot_cat"], txns["amount"], 0.0)
    ).groupby(["user_id", "quarter"])["_rot_amt"].cumsum() - np.where(
        txns["is_rot_cat"], txns["amount"], 0.0)
    txns["rot_cap_left"] = rot_spend < ROTATOR_CAP

    held_map = {r.user_id: set(r.held_cards.split(";")) for r in users.itertuples()}
    held_mask = np.column_stack([
        txns["user_id"].map(lambda u: c in held_map[u]).values for c in CARDS
    ])
    rates = np.column_stack([
        np.where(
            (txns["category"] == txns["rot_cat"]) & (txns["rot_cap_left"]) & (c == "Rotator"),
            0.05,
            txns["category"].map(CARDS[c]),
        ) if c != "Rotator" else np.where(
            (txns["category"] == txns["rot_cat"]) & (txns["rot_cap_left"]),
            0.05, 0.01)
        for c in CARDS
    ])
    rates = np.where(held_mask, rates, -1.0)
    idx = rates.argmax(axis=1)
    card_list = list(CARDS)
    txns["optimal_card"] = [card_list[i] for i in idx]
    # Use Python round() (not pandas/numpy .round): numpy scales first, so a
    # true half-cent product such as 94.5 x 0.01 can round down where Python
    # rounds up. data_gen uses Python round(), so routing must too, or
    # optimal points can come out a cent below actual points.
    opt_rates = rates[np.arange(len(txns)), idx]
    txns["optimal_points"] = [round(float(a) * float(r), 2)
                              for a, r in zip(txns["amount"].values, opt_rates)]
    txns["points_left_on_table"] = (txns["optimal_points"] - txns["points_earned"]).clip(lower=0)
    txns["router_agrees"] = txns["card_used"] == txns["optimal_card"]
    return txns.drop(columns=["quarter", "rot_cat", "is_rot_cat", "rot_cap_left"])


def routing_summary(routed):
    """Aggregate routing lift: actual vs optimal points, by category."""
    g = routed.groupby("category").agg(
        actual_points=("points_earned", "sum"),
        optimal_points=("optimal_points", "sum"),
        txns=("amount", "size"),
        spend=("amount", "sum"),
        agree_rate=("router_agrees", "mean"),
    ).reset_index()
    g["lift_pct"] = (g["optimal_points"] / g["actual_points"] - 1) * 100
    g["yield_actual"] = g["actual_points"] / g["spend"] * 100
    g["yield_optimal"] = g["optimal_points"] / g["spend"] * 100
    total = {
        "actual_points": float(routed["points_earned"].sum()),
        "optimal_points": float(routed["optimal_points"].sum()),
        "lift_pct": float(routed["optimal_points"].sum() / routed["points_earned"].sum() * 100 - 100),
        "agree_rate": float(routed["router_agrees"].mean()),
        "total_spend": float(routed["amount"].sum()),
        "n_txns": int(len(routed)),
    }
    return g, total
