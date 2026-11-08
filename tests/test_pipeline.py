import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import MATCH_STATE  # noqa: E402
from prepare_data import innings_summary, rain_affected  # noqa: E402


def balls(match_id, innings, n_legal, runs_per_ball=1, wickets=0, wides=0):
    rows = []
    for i in range(n_legal):
        rows.append({"match_id": match_id, "innings": innings, "runs": runs_per_ball,
                     "is_wicket": int(i < wickets), "is_legal": True})
    for _ in range(wides):
        rows.append({"match_id": match_id, "innings": innings, "runs": 1, "is_wicket": 0, "is_legal": False})
    return rows


def test_wides_do_not_count_as_legal_balls():
    df = pd.DataFrame(balls(1, 1, 120, wides=5) + balls(1, 2, 120))
    s = innings_summary(df)
    assert s.loc[1, ("legal_balls", 1)] == 120
    assert s.loc[1, ("total", 1)] == 125


def test_rain_affected_matches_are_detected():
    normal = balls(1, 1, 120) + balls(1, 2, 110, runs_per_ball=2)        # chased it down early: fine
    all_out = balls(2, 1, 80, wickets=10) + balls(2, 2, 120)              # bowled out early: fine
    short_first = balls(3, 1, 90, wickets=3) + balls(3, 2, 90)            # reduced overs from the start
    stopped_chase = balls(4, 1, 120) + balls(4, 2, 60, wickets=2)         # rain stopped the chase
    s = innings_summary(pd.DataFrame(normal + all_out + short_first + stopped_chase))
    assert set(rain_affected(s)) == {3, 4}


@pytest.fixture(scope="module")
def model():
    path = ROOT / "models" / "win_predictor.joblib"
    if not path.exists():
        pytest.skip("run src/train.py first")
    import joblib
    return joblib.load(path)


def state(**kw):
    base = dict(target=180, runs_left=80, balls_left=48, wickets_left=6, crr=6.25,
                rrr=10.0, runs_last_5=40, wickets_last_5=1)
    base.update(kw)
    return pd.DataFrame([base])[MATCH_STATE]


def prob(model, **kw):
    return model.predict_proba(state(**kw))[0, 1]


def test_more_wickets_in_hand_means_higher_win_probability(model):
    assert prob(model, wickets_left=8) > prob(model, wickets_left=6) > prob(model, wickets_left=2)


def test_fewer_runs_needed_means_higher_win_probability(model):
    assert prob(model, runs_left=30, rrr=3.75) > prob(model, runs_left=80) > prob(model, runs_left=120, rrr=15)


def test_easy_and_impossible_chases(model):
    assert prob(model, runs_left=2, balls_left=30, rrr=0.4, wickets_left=8) > 0.95
    assert prob(model, runs_left=60, balls_left=6, rrr=60, wickets_left=2) < 0.05
