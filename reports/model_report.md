# Model Report — AI Rewards Optimization & Spend Analytics

**Role:** AI Business Analyst, Uno Wallet (YC P26) · **Author:** Bhupesh Chandra Dimri · **Date:** Nov 2026
**Data:** synthetic only — 2,000 users, 349,182 transactions, $32.06M spend, 12 months.

## 1. Problems and results

| Problem | Models compared | Winner | Metric |
|---|---|---|---|
| Card routing (P1) | Earn-rate rule vs logistic / RF classifiers | Rule (optimal by construction) | +51.4% points vs actual; 35.7% user agreement |
| Churn, silent in month 12 (P2) | Threshold rule / logistic / RF | Logistic regression | AUC 0.995; RF 0.992; rule 0.869 |
| Next-month yield (P3) | Mean / linear / RF | Linear regression | MAPE 74.7%, R² 0.20 (RF 84.4%, R² 0.11) |
| Card choice vs rule (P1 check) | Logistic / RF | Random forest | 59.3% agreement (logistic 54.7%) |

## 2. Churn detail (hold-out, 25%)

| Model | AUC | Accuracy | Precision@20% | Recall@20% |
|---|---|---|---|---|
| Threshold rule (gap > 30d or trend < −40%) | 0.869 | 0.948 | 0.46 | 0.82 |
| Logistic regression | 0.995 | 0.986 | 0.55 | 0.98 |
| Random forest | 0.992 | 0.980 | 0.54 | 0.96 |

Top churn signals (RF importance): transactions in last 3 months 0.355,
3-month spend trend 0.229, spend in last 3 months 0.186, points in last 3 months
0.148, max gap days 0.039. Recency is the whole story.

## 3. Routing lift by category

| Category | Spend | Yield actual → optimal | Lift |
|---|---|---|---|
| Dining | $3.30M | 1.98% → 3.40% | +71.7% |
| Travel | $11.43M | 1.75% → 2.76% | +57.6% |
| Groceries | $7.50M | 1.98% → 2.90% | +46.4% |
| Online | $5.25M | 1.58% → 2.29% | +44.8% |
| Other | $2.88M | 1.20% → 1.61% | +34.0% |
| Streaming | $0.41M | 1.20% → 1.61% | +34.0% |
| Gas | $1.30M | 1.60% → 1.97% | +23.0% |

Blended yield: 1.74% actual → 2.63% optimal. Total: 557,906 → 844,707 points.

## 4. What the numbers mean (and what they don't)

- The 51.4% lift is large because simulated users use one default card 60% of
  the time. A real optimiser-user sees less. The lift is a product opportunity
  (prompts, cap reminders), not a model accuracy claim.
- Churn AUC 0.995 is high because the synthetic disengagement ramp is
  pronounced. Treat the signal ranking as transferable, not the AUC.
- Yield MAPE 74.7% is poor in absolute terms; monthly points are Poisson-noisy.
  The forecast is usable for planning ranges, not for point promises.
- The card-choice classifiers cannot see held cards or cap state, so 59%
  agreement is expected — it is evidence the rule should stay the router.

## 5. Bugs found by the test suite (fixed, regression-tested)

1. `spend_trend_3m` compared 3 months of spend against 8 months without
   normalising — it read ≈ −0.6 for every stable user and the churn rule's
   accuracy was 12%. Fixed to monthly averages; rule AUC is now 0.869.
2. Points were computed from unrounded amounts while the stored amount was
   rounded, and pandas `.round()` disagrees with Python `round()` on half-cent
   values — optimal points came out a cent below actual points. Fixed by
   rounding once in the generator and using Python `round()` in the router.

## 6. How to reproduce

```bash
python run_all.py            # regenerates metrics.json + 7 charts
python -m tests.test_all     # 66 passed, 0 failed
python build_docs.py         # report + deck PDFs
```
