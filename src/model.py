"""Model definitions shared by training and the app.

Two feature sets are compared in train.py:
- MATCH_STATE: only what's happening in the chase (runs, balls, wickets, rates, momentum)
- + CONTEXT:   batting team, bowling team and city on top of that

Adding team/city made predictions on new seasons clearly worse: IPL squads are
reshuffled at every auction, so "Mumbai Indians" in 2015 tells you little about
"Mumbai Indians" in 2025. The shipped model uses match state only.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

MATCH_STATE = ["target", "runs_left", "balls_left", "wickets_left", "crr", "rrr",
               "runs_last_5", "wickets_last_5"]
CONTEXT = ["batting_team", "bowling_team", "city"]
FEATURES = MATCH_STATE  # what the final model and the app use


def _clip_rates(X):
    # Needing 50 off the last ball gives a required rate of 300/over -- that's
    # just "impossible". Capping at 36 (six sixes an over) stops these extremes
    # from dominating the linear model.
    X = np.asarray(X, dtype=float).copy()
    for col in ("crr", "rrr"):
        i = MATCH_STATE.index(col)
        X[:, i] = np.clip(X[:, i], 0, 36)
    return X


def _numeric():
    return Pipeline([("clip", FunctionTransformer(_clip_rates)), ("scale", StandardScaler())])


def logistic_regression(with_context: bool = False) -> Pipeline:
    parts = [("num", _numeric(), MATCH_STATE)]
    if with_context:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=200), CONTEXT))
    return Pipeline([("pre", ColumnTransformer(parts)), ("model", LogisticRegression(max_iter=2000))])


def gradient_boosting() -> Pipeline:
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, min_samples_leaf=200, l2_regularization=1.0, random_state=42
    )
    return Pipeline([("pre", ColumnTransformer([("num", "passthrough", MATCH_STATE)])), ("model", model)])
