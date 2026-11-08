"""Train and evaluate the IPL win-probability model.

Split by SEASON, not randomly: train on 2008-2023, test on 2024-2026.
A random split would put balls from the same match in both train and test
(near-identical rows), and wouldn't tell us how the model copes with future
seasons -- which matters, because scoring has jumped since 2024.

Two models are compared on probability quality (log loss, Brier score) as
well as accuracy, since the app shows a probability, not just a winner.

Run:  python src/train.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from model import CONTEXT, MATCH_STATE, gradient_boosting, logistic_regression

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "chase_states.csv"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"
TEST_FROM = 2024


def evaluate(y, p) -> dict:
    return {
        "log_loss": round(log_loss(y, p), 4),
        "brier": round(brier_score_loss(y, p), 4),
        "accuracy": round(accuracy_score(y, p >= 0.5), 4),
        "roc_auc": round(roc_auc_score(y, p), 4),
    }


def phase(balls_left: pd.Series) -> pd.Series:
    overs_done = (120 - balls_left) / 6
    return pd.cut(overs_done, [-0.1, 6, 15, 20], labels=["Powerplay (0-6)", "Middle (6-15)", "Death (15-20)"])


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    train, test = df[df.season < TEST_FROM], df[df.season >= TEST_FROM]
    print(f"Train: {train.match_id.nunique()} matches ({train.season.min()}-{train.season.max()}), {len(train):,} balls")
    print(f"Test:  {test.match_id.nunique()} matches ({test.season.min()}-{test.season.max()}), {len(test):,} balls\n")

    avg_target = df.groupby("match_id").agg(season=("season", "first"), target=("target", "first"))
    print("Average target by era:", avg_target.groupby(avg_target.season >= TEST_FROM).target.mean().round(1).to_dict(), "\n")

    candidates = {
        "Logistic Regression": logistic_regression(),
        "Logistic Regression + team & city": logistic_regression(with_context=True),
        "Gradient Boosting": gradient_boosting(),
    }
    cols = MATCH_STATE + CONTEXT
    results, probs = {}, {}
    baseline = np.full(len(test), train.groupby("match_id").won.first().mean())
    results["Baseline (chasing-team win rate)"] = evaluate(test.won, baseline)
    for name, model in candidates.items():
        model.fit(train[cols], train.won)
        probs[name] = model.predict_proba(test[cols])[:, 1]
        results[name] = evaluate(test.won, probs[name])

    table = pd.DataFrame(results).T
    print(table.to_string(), "\n")
    table.to_csv(ROOT / "reports" / "model_comparison.csv")

    best_name = min(candidates, key=lambda n: results[n]["log_loss"])
    best = candidates[best_name]
    p = probs[best_name]
    print(f"Best model by log loss: {best_name}")

    by_phase = (
        test.assign(correct=((p >= 0.5) == test.won.astype(bool)), phase=phase(test.balls_left))
        .groupby("phase", observed=True).correct.mean().round(4)
    )
    print("\nAccuracy by phase of the chase:\n" + by_phase.to_string())

    metrics = {"model": best_name, "train_seasons": f"{train.season.min()}-{train.season.max()}",
               "test_seasons": f"{test.season.min()}-{test.season.max()}",
               "test_matches": int(test.match_id.nunique()), **results[best_name],
               "accuracy_by_phase": by_phase.to_dict()}
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plot_calibration(test.won, probs)
    plot_phase(by_phase)
    for match_id, title in pick_example_matches(test, p):
        plot_match(test, p, match_id, title)

    # Refit on every season before saving, so the shipped model knows the newest scoring era.
    final = candidates[best_name]
    final.fit(df[cols], df.won)
    joblib.dump(final, MODELS / "win_predictor.joblib")
    print(f"\nSaved model -> {MODELS / 'win_predictor.joblib'}")


# ---------------------------------------------------------------- plotting
def plot_calibration(y, probs):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
    for name, p in probs.items():
        frac, mean = calibration_curve(y, p, n_bins=10)
        ax.plot(mean, frac, "o-", label=name)
    ax.set_xlabel("Predicted win probability")
    ax.set_ylabel("Actual win rate")
    ax.set_title(f"Calibration on {TEST_FROM}+ seasons")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png", dpi=150)
    plt.close(fig)


def plot_phase(by_phase):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.bar(by_phase.index.astype(str), by_phase.values * 100, color=["#8DA0CB", "#66C2A5", "#FC8D62"])
    for i, v in enumerate(by_phase.values):
        ax.text(i, v * 100 + 1, f"{v:.0%}", ha="center")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Prediction accuracy gets better as the chase unfolds")
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_by_phase.png", dpi=150)
    plt.close(fig)


def pick_example_matches(test, p):
    """The test match with the biggest swing in win probability = most dramatic finish."""
    t = test.assign(p=p)
    swing = t.groupby("match_id").p.agg(lambda s: s.max() - s.min())
    final_margin = t.groupby("match_id").runs_left.last()
    close = swing[final_margin.reindex(swing.index) <= 15].sort_values(ascending=False)
    yield close.index[0], "Most dramatic chase in the test seasons"


def plot_match(test, p, match_id, title):
    m = test.assign(p=p)[test.match_id == match_id]
    overs = (120 - m.balls_left) / 6
    bat, bowl = m.batting_team.iloc[0], m.bowling_team.iloc[0]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(overs, m.p * 100, 50, where=m.p >= 0.5, color="#66C2A5", alpha=0.4, interpolate=True)
    ax.fill_between(overs, m.p * 100, 50, where=m.p < 0.5, color="#FC8D62", alpha=0.4, interpolate=True)
    ax.plot(overs, m.p * 100, color="black", lw=1.5)
    wk = m[m.wickets_left.diff() < 0]
    ax.scatter((120 - wk.balls_left) / 6, wk.p * 100, color="red", zorder=3, label="Wicket")
    ax.axhline(50, color="grey", lw=0.8)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 20)
    ax.set_xlabel("Overs")
    ax.set_ylabel(f"{bat} win probability (%)")
    winner = bat if m.won.iloc[0] else bowl
    ax.set_title(f"{bat} chasing {int(m.target.iloc[0])} vs {bowl}, {m.match_date.iloc[0]} — {winner} won\n{title}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "win_probability_example.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
