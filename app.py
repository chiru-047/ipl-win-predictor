"""IPL Win Predictor — Streamlit app.

Run:  streamlit run app.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from model import MATCH_STATE  # noqa: E402  (needed so joblib can load the pipeline)

TEAMS = [
    "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans", "Kolkata Knight Riders",
    "Lucknow Super Giants", "Mumbai Indians", "Punjab Kings", "Rajasthan Royals",
    "Royal Challengers Bengaluru", "Sunrisers Hyderabad",
]


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "win_predictor.joblib")


@st.cache_data
def load_recent():
    return pd.read_csv(ROOT / "data" / "sample" / "recent_chases.csv")


def match_state(target, score, overs, balls, wickets_down, runs_last_5, wickets_last_5):
    balls_bowled = overs * 6 + balls
    balls_left = 120 - balls_bowled
    runs_left = target - score
    return pd.DataFrame([{
        "target": target,
        "runs_left": runs_left,
        "balls_left": balls_left,
        "wickets_left": 10 - wickets_down,
        "crr": score * 6 / balls_bowled if balls_bowled else 0.0,
        "rrr": runs_left * 6 / max(balls_left, 1),
        "runs_last_5": runs_last_5,
        "wickets_last_5": wickets_last_5,
    }])[MATCH_STATE]


st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="centered")
st.title("🏏 IPL Win Predictor")
st.caption("Live win probability for the chasing team, trained on every IPL chase from 2008 to 2026.")
model = load_model()

tab_live, tab_replay = st.tabs(["Live chase", "Replay a recent match"])

with tab_live:
    c1, c2 = st.columns(2)
    batting = c1.selectbox("Chasing team", TEAMS, index=5)
    bowling = c2.selectbox("Defending team", [t for t in TEAMS if t != batting], index=1)

    target = st.number_input("Target", min_value=50, max_value=300, value=190)
    c1, c2, c3 = st.columns(3)
    score = c1.number_input("Current score", min_value=0, max_value=299, value=95)
    wickets_down = c2.number_input("Wickets down", min_value=0, max_value=9, value=3)
    overs = c3.number_input("Overs completed", min_value=0, max_value=19, value=10)
    balls = st.slider("Balls into the current over", 0, 5, 0)

    with st.expander("Momentum (last 5 overs)"):
        m1, m2 = st.columns(2)
        runs_last_5 = m1.number_input("Runs in last 5 overs", min_value=0, max_value=150, value=45)
        wickets_last_5 = m2.number_input("Wickets in last 5 overs", min_value=0, max_value=10, value=1)

    balls_bowled = overs * 6 + balls
    if score >= target:
        st.success(f"{batting} have already won!")
    elif balls_bowled == 0 and score > 0:
        st.warning("Score must be 0 before the first ball.")
    else:
        X = match_state(target, score, overs, balls, wickets_down, runs_last_5, wickets_last_5)
        p = float(model.predict_proba(X)[0, 1])
        need, left = int(X.runs_left[0]), int(X.balls_left[0])
        st.markdown(f"**{batting} need {need} runs from {left} balls** "
                    f"(required rate {X.rrr[0]:.2f}, current rate {X.crr[0]:.2f})")
        c1, c2 = st.columns(2)
        c1.metric(batting, f"{p:.0%}")
        c2.metric(bowling, f"{1 - p:.0%}")
        st.progress(p)

with tab_replay:
    recent = load_recent()
    matches = (recent.groupby("match_id")
               .agg(date=("match_date", "first"), bat=("batting_team", "first"),
                    bowl=("bowling_team", "first"), target=("target", "first"), won=("won", "first"))
               .sort_values("date", ascending=False))
    labels = {mid: f"{r.date} — {r.bat} chasing {int(r.target)} vs {r.bowl}" for mid, r in matches.iterrows()}
    match_id = st.selectbox("Match (2024 onwards)", list(labels), format_func=labels.get)

    m = recent[recent.match_id == match_id].copy()
    m["p"] = model.predict_proba(m[MATCH_STATE])[:, 1]
    m["over"] = (120 - m.balls_left) / 6
    row = matches.loc[match_id]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.fill_between(m.over, m.p * 100, 50, where=m.p >= 0.5, color="#66C2A5", alpha=0.4, interpolate=True)
    ax.fill_between(m.over, m.p * 100, 50, where=m.p < 0.5, color="#FC8D62", alpha=0.4, interpolate=True)
    ax.plot(m.over, m.p * 100, color="black", lw=1.5)
    wk = m[m.wickets_left.diff() < 0]
    ax.scatter(wk.over, wk.p * 100, color="red", zorder=3, label="Wicket")
    ax.axhline(50, color="grey", lw=0.8)
    ax.set(xlim=(0, 20), ylim=(0, 100), xlabel="Overs", ylabel=f"{row.bat} win %")
    ax.legend(loc="upper left")
    st.pyplot(fig)
    st.write(f"**Result:** {row.bat if row.won else row.bowl} won.")
