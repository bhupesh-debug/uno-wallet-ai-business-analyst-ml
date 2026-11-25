"""Synthetic data generator — AI Rewards Optimization & Spend Analytics.

Everything here is SYNTHETIC. No real Uno Wallet data, no real card
products, no real users. The five cards are fictional, with earn rates
chosen to make the routing problem non-trivial (a rotating 5% category
with a quarterly cap, plus category specialists and a flat-rate card).
Seeded and reproducible.
"""
import numpy as np
import pandas as pd

# Fictional cards. Earn rates are points per $1 spent, by category.
CARDS = {
    "Apex Flat":   {"Dining": 0.02, "Groceries": 0.02, "Travel": 0.02, "Gas": 0.02,
                    "Online": 0.02, "Streaming": 0.02, "Other": 0.02, "fee": 0},
    "Bistro":      {"Dining": 0.04, "Groceries": 0.02, "Travel": 0.01, "Gas": 0.01,
                    "Online": 0.01, "Streaming": 0.01, "Other": 0.01, "fee": 0},
    "Voyager":     {"Dining": 0.01, "Groceries": 0.01, "Travel": 0.03, "Gas": 0.02,
                    "Online": 0.01, "Streaming": 0.01, "Other": 0.01, "fee": 95},
    "Everyday":    {"Dining": 0.01, "Groceries": 0.03, "Travel": 0.01, "Gas": 0.02,
                    "Online": 0.02, "Streaming": 0.01, "Other": 0.01, "fee": 0},
    # Rotator pays 5% on the rotating category (cap $1,500/quarter), 1% else.
    "Rotator":     {"Dining": 0.01, "Groceries": 0.01, "Travel": 0.01, "Gas": 0.01,
                    "Online": 0.01, "Streaming": 0.01, "Other": 0.01, "fee": 0},
}
# Rotating 5% category by calendar quarter (1-4).
ROTATING = {1: "Dining", 2: "Travel", 3: "Groceries", 4: "Online"}
ROTATOR_CAP = 1500.0  # $ spend per quarter at the 5% rate

CATEGORIES = ["Dining", "Groceries", "Travel", "Gas", "Online", "Streaming", "Other"]

# Spend profile per archetype: category share + typical monthly txn counts.
ARCHETYPES = {
    "Foodie":         {"share": {"Dining": .38, "Groceries": .22, "Travel": .06, "Gas": .06,
                                 "Online": .12, "Streaming": .06, "Other": .10}, "txns": 16},
    "Family":         {"share": {"Dining": .12, "Groceries": .38, "Travel": .08, "Gas": .10,
                                 "Online": .14, "Streaming": .06, "Other": .12}, "txns": 18},
    "Traveller":      {"share": {"Dining": .18, "Groceries": .12, "Travel": .38, "Gas": .08,
                                 "Online": .10, "Streaming": .04, "Other": .10}, "txns": 14},
    "Online Shopper": {"share": {"Dining": .10, "Groceries": .14, "Travel": .06, "Gas": .04,
                                 "Online": .42, "Streaming": .10, "Other": .14}, "txns": 15},
    "Balanced":       {"share": {"Dining": .16, "Groceries": .20, "Travel": .10, "Gas": .08,
                                 "Online": .18, "Streaming": .08, "Other": .20}, "txns": 12},
}
# Mean transaction amount ($) per category.
AMOUNT_MEAN = {"Dining": 42, "Groceries": 85, "Travel": 210, "Gas": 45,
               "Online": 68, "Streaming": 15, "Other": 55}

N_USERS = 2000
N_MONTHS = 12
START = pd.Timestamp("2025-12-01")


def earn_rate(card, category, date):
    """Points per $1 for a card/category on a date (rotating rule applied)."""
    if card == "Rotator" and ROTATING[(date.month - 1) // 3 + 1] == category:
        return 0.05
    return CARDS[card][category]


def generate(seed=42, n_users=N_USERS):
    rng = np.random.default_rng(seed)
    archetypes = list(ARCHETYPES)
    user_arch = rng.choice(archetypes, size=n_users)
    # Each user holds 2-4 of the 5 cards; a "default" card they reach for most.
    rows, users = [], []
    for u in range(n_users):
        arch = user_arch[u]
        n_cards = int(rng.integers(2, 5))
        held = list(rng.choice(list(CARDS), size=n_cards, replace=False))
        default = str(rng.choice(held))
        # Churn-prone users: ~12% disengage gradually (months 9-11) then
        # go quiet in the final 30 days. Disengagement is the signal the
        # churn models learn; abrupt silence alone would be unpredictable.
        churn_prone = bool(rng.random() < 0.12)
        users.append({"user_id": u, "archetype": arch, "held_cards": ";".join(held),
                      "default_card": default, "churn_prone": churn_prone})
        base_txns = ARCHETYPES[arch]["txns"]
        rotator_spent = {}  # (quarter) -> $ spent at the 5% Rotator rate
        for m in range(N_MONTHS):
            # Final month: churn-prone users stop transacting after day 335.
            if m == N_MONTHS - 1 and churn_prone:
                continue
            # Disengagement ramp for churn-prone users in months 9, 10, 11.
            scale = 1.0
            if churn_prone and m >= 8:
                scale = {8: 0.6, 9: 0.4, 10: 0.25}[m]
            n = int(rng.poisson(base_txns * scale))
            for j in range(n):
                cat = str(rng.choice(CATEGORIES,
                                     p=[ARCHETYPES[arch]["share"][c] for c in CATEGORIES]))
                amount = round(float(np.clip(rng.lognormal(np.log(AMOUNT_MEAN[cat]), 0.55), 3, 1500)), 2)
                # Days run sequentially within the month so generation order
                # matches date order — the Rotator cap is consumed in date
                # order, exactly as the router in src/routing.py assumes.
                day = START + pd.Timedelta(days=int(m * 30 + min(j, 29)))
                # Actual card used: default 60% of the time, else random held card.
                card = default if rng.random() < 0.60 else str(rng.choice(held))
                rate = earn_rate(card, cat, day)
                q = (day.month - 1) // 3 + 1
                # The 5% Rotator rate applies to the first $1,500 of
                # rotating-category spend per user per quarter (spend on any
                # card counts toward the window — the same rule the router
                # in src/routing.py uses, so optimal >= actual by construction).
                if ROTATING[q] == cat:
                    if card == "Rotator" and rotator_spent.get(q, 0.0) >= ROTATOR_CAP:
                        rate = 0.01  # quarterly 5% cap exhausted
                    rotator_spent[q] = rotator_spent.get(q, 0.0) + amount
                rows.append({"user_id": u, "date": day, "month": m + 1,
                             "category": cat, "amount": amount,
                             "card_used": card,
                             "points_earned": round(amount * rate, 2)})
    txns = pd.DataFrame(rows)
    users = pd.DataFrame(users)
    return txns, users


def save_synthetic(txns, users, out_dir):
    out_dir = str(out_dir)
    import os
    os.makedirs(out_dir, exist_ok=True)
    txns.to_csv(f"{out_dir}/synthetic_transactions.csv", index=False)
    users.to_csv(f"{out_dir}/synthetic_users.csv", index=False)
