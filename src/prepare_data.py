"""Turn raw ball-by-ball data into one row per legal delivery of every run chase.

Each row describes the state of the chase *after* that ball:
    batting_team, bowling_team, city, target, runs_left, balls_left,
    wickets_left, current/required run rate, runs & wickets in the last 5 overs
and the label `won` = 1 if the chasing team went on to win.

Cleaning decisions (each one is a real problem in the raw data):
- Franchise renames are merged (Delhi Daredevils -> Delhi Capitals, etc.).
- Ties (decided by super over) and no-results are dropped.
- Rain-affected (DLS / reduced-overs) matches are dropped, because their real
  target isn't "first-innings score + 1" and would teach the model nonsense.
- Wides and no-balls don't use up one of the 120 legal balls.

Run:  python src/prepare_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "chase_states.csv"
# Small slice of recent seasons that IS committed, so the app's "replay a match" tab works out of the box.
SAMPLE = ROOT / "data" / "sample" / "recent_chases.csv"
SAMPLE_FROM = 2024

TEAM_RENAMES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}
CITY_FIXES = {"Bangalore": "Bengaluru", "Mohali": "Chandigarh", "New Chandigarh": "Chandigarh"}
VENUE_CITY = {"Dubai International Cricket Stadium": "Dubai", "Sharjah Cricket Stadium": "Sharjah"}

TOTAL_BALLS = 120
MOMENTUM_BALLS = 30  # "last 5 overs"


def load_matches() -> pd.DataFrame:
    m = pd.read_csv(RAW / "match_info.csv")
    m = m.rename(columns={"match_number": "match_id"})
    for col in ["team1", "team2", "winner", "toss_winner"]:
        m[col] = m[col].replace(TEAM_RENAMES)
    m["city"] = m["city"].fillna(m["venue"].map(VENUE_CITY)).replace(CITY_FIXES).fillna("Other")
    m["season"] = pd.to_datetime(m["match_date"]).dt.year
    return m[m["result"] == "Win"]


def load_balls() -> pd.DataFrame:
    b = pd.read_csv(RAW / "ball_by_ball.csv")
    b = b.rename(columns={"ID": "match_id", "Innings": "innings", "TotalRun": "runs",
                          "IsWicketDelivery": "is_wicket", "BattingTeam": "batting_team"})
    b["batting_team"] = b["batting_team"].replace(TEAM_RENAMES)
    extra = b["ExtraType"].fillna("")
    b["is_legal"] = ~extra.str.contains("wides|noballs")
    return b[b["innings"].isin([1, 2])]


def innings_summary(balls: pd.DataFrame) -> pd.DataFrame:
    s = balls.groupby(["match_id", "innings"]).agg(
        total=("runs", "sum"), wickets=("is_wicket", "sum"), legal_balls=("is_legal", "sum")
    )
    return s.unstack("innings")


def rain_affected(summary: pd.DataFrame) -> pd.Index:
    """Matches where the 120-ball / first-innings+1 assumption breaks."""
    t1, w1, b1 = summary[("total", 1)], summary[("wickets", 1)], summary[("legal_balls", 1)]
    t2, w2, b2 = summary[("total", 2)], summary[("wickets", 2)], summary[("legal_balls", 2)]
    target = t1 + 1
    first_cut_short = (b1 < TOTAL_BALLS) & (w1 < 10)
    chase_stopped = (b2 < TOTAL_BALLS) & (w2 < 10) & (t2 < target)
    return summary.index[first_cut_short | chase_stopped]


def build_chase_states(balls: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    summary = innings_summary(balls).dropna()
    bad = set(rain_affected(summary))
    good = matches[matches["match_id"].isin(summary.index) & ~matches["match_id"].isin(bad)]

    first_total = summary[("total", 1)].rename("first_innings_total")
    chase = balls[(balls["innings"] == 2) & balls["match_id"].isin(good["match_id"])].copy()
    chase = chase.merge(first_total, left_on="match_id", right_index=True)
    chase = chase.merge(good[["match_id", "team1", "team2", "winner", "city", "season", "match_date"]], on="match_id")

    # Ball order inside each match (data is already in delivery order).
    chase = chase.sort_values(["match_id", "Overs", "BallNumber"], kind="stable")
    g = chase.groupby("match_id")
    chase["score"] = g["runs"].cumsum()
    chase["wickets_down"] = g["is_wicket"].cumsum()
    chase["balls_bowled"] = g["is_legal"].cumsum()

    # Momentum: runs and wickets in the last 30 deliveries (rolling window per match).
    chase["runs_last_5"] = g["runs"].transform(lambda s: s.rolling(MOMENTUM_BALLS, min_periods=1).sum())
    chase["wickets_last_5"] = g["is_wicket"].transform(lambda s: s.rolling(MOMENTUM_BALLS, min_periods=1).sum())

    chase["bowling_team"] = np.where(chase["batting_team"] == chase["team1"], chase["team2"], chase["team1"])
    chase["target"] = chase["first_innings_total"] + 1
    chase["runs_left"] = chase["target"] - chase["score"]
    chase["balls_left"] = TOTAL_BALLS - chase["balls_bowled"]
    chase["wickets_left"] = 10 - chase["wickets_down"]
    overs_done = chase["balls_bowled"] / 6
    chase["crr"] = np.where(overs_done > 0, chase["score"] / overs_done.replace(0, np.nan), 0.0)
    chase["rrr"] = chase["runs_left"] * 6 / chase["balls_left"].clip(lower=1)
    chase["won"] = (chase["batting_team"] == chase["winner"]).astype(int)

    # Keep only legal deliveries before the result is already decided.
    live = chase["is_legal"] & (chase["runs_left"] > 0) & (chase["wickets_left"] > 0) & (chase["balls_left"] > 0)
    cols = ["match_id", "season", "match_date", "batting_team", "bowling_team", "city", "target",
            "score", "runs_left", "balls_left", "wickets_left", "crr", "rrr",
            "runs_last_5", "wickets_last_5", "won"]
    out = chase.loc[live, cols].reset_index(drop=True)
    out[["crr", "rrr"]] = out[["crr", "rrr"]].round(3)
    return out, len(matches), len(bad)


def main():
    matches, balls = load_matches(), load_balls()
    states, n_matches, n_rain = build_chase_states(balls, matches)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    states.to_csv(OUT, index=False)
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    states[states.season >= SAMPLE_FROM].to_csv(SAMPLE, index=False)
    print(f"Decided matches: {n_matches} | rain-affected dropped: {n_rain}")
    print(f"Matches kept: {states.match_id.nunique()} | ball states: {len(states):,}")
    print(f"Chasing team win rate: {states.groupby('match_id').won.first().mean():.1%}")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
