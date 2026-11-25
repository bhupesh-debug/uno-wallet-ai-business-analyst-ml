"""Models: interpretable baselines first, ML second.

Churn (classification): rule baseline (max gap > 45 days in months 1-11)
vs logistic regression vs random forest.
Yield (regression): mean baseline vs linear regression vs random forest.
Card choice (multi-class): rule IS the label, so the classifiers are
measured on agreement with the rule — logistic regression vs random
forest. No model is expected to beat the rule on routing; that is the
honest finding the report leads with.
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def churn_models():
    return {
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=1000, C=1.0)),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                                min_samples_leaf=20, random_state=42,
                                                n_jobs=-1),
    }


def rule_churn_predict(feat):
    """Rule baseline: flag churn if the user's longest gap exceeded 30 days
    OR spend fell more than 40% in the last 3 months vs the period before.
    This is the kind of threshold rule a product analyst would write first.
    """
    return ((feat["max_gap_days"] > 30) | (feat["spend_trend_3m"] < -0.4)).astype(int).values


def yield_models():
    return {
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "random_forest": RandomForestRegressor(n_estimators=300, max_depth=8,
                                               min_samples_leaf=20, random_state=42,
                                               n_jobs=-1),
    }


def card_choice_models():
    return {
        "logistic": make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=1000, C=1.0)),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=10,
                                                min_samples_leaf=10, random_state=42,
                                                n_jobs=-1),
    }
