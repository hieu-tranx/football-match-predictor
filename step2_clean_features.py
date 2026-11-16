"""STEP 2 - Clean the data and build features.

2a. Build our OWN expected-goals (xG) model.
    xG = probability that a shot becomes a goal. We learn it from ~10k shots
    using where the shot was taken (distance + angle to goal) and how
    (header? penalty? free kick?). Model: logistic regression.

2b. "Seconds before shot" = how long the team had the ball before shooting.
    Short = direct / counter-attacking team, long = slow build-up.

2c. Squash everything into ONE row per team per match (team_matches.csv),
    which is the table step 3 learns from.

Run:  .venv/bin/python step2_clean_features.py
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

import config

D = config.DATA_DIR

# StatsBomb pitch is 120 x 80. The goal is at x=120, between posts y=36 and y=44.
GOAL_X, POST_LEFT, POST_RIGHT = 120, 36, 44


def shot_features(shots):
    dx = GOAL_X - shots["x"]
    dy_center = shots["y"] - 40
    feats = pd.DataFrame(index=shots.index)
    feats["distance"] = np.hypot(dx, dy_center)
    # Angle the goal mouth "opens up" from the shot location (bigger = easier).
    a1 = np.arctan2(POST_LEFT - shots["y"], dx)
    a2 = np.arctan2(POST_RIGHT - shots["y"], dx)
    feats["angle"] = np.abs(a2 - a1)
    feats["is_header"] = (shots["body_part"] == "Head").astype(int)
    feats["is_penalty"] = (shots["shot_type"] == "Penalty").astype(int)
    feats["is_free_kick"] = (shots["shot_type"] == "Free Kick").astype(int)
    return feats


def build_xg(shots):
    X, y = shot_features(shots), shots["goal"]
    model = LogisticRegression(max_iter=1000)
    # cross_val_predict = every shot is scored by a model that never saw it,
    # so the xG values are honest (no peeking at the answer).
    shots["xg"] = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]

    model.fit(X, y)
    print("xG model coefficients (positive = more likely goal):")
    for name, coef in zip(X.columns, model.coef_[0]):
        print(f"  {name:13s} {coef:+.3f}")
    corr = shots[["xg", "statsbomb_xg"]].corr().iloc[0, 1]
    print(f"Our xG vs StatsBomb's professional xG: correlation {corr:.2f}")
    print(f"Total goals {shots['goal'].sum()}  vs  total xG {shots['xg'].sum():.0f}\n")
    return shots


def add_seconds_before_shot(shots, possessions):
    shots = shots.merge(possessions[["match_id", "possession", "start_sec"]],
                        on=["match_id", "possession"], how="left")
    shots["secs_before_shot"] = (shots["sec"] - shots["start_sec"]).clip(lower=0)
    return shots


def team_match_table(matches, shots, cards):
    """One row per (match, team): goals, xG, shots, cards - for AND against."""
    per_team = shots.groupby(["match_id", "team"]).agg(
        xg=("xg", "sum"), shots=("xg", "size"),
        secs_before_shot=("secs_before_shot", "mean")).reset_index()
    card_points = cards.assign(pts=cards["card"].map({"Yellow Card": 1, "Second Yellow": 2, "Red Card": 3}))
    per_team = per_team.merge(card_points.groupby(["match_id", "team"])["pts"].sum()
                              .rename("card_points").reset_index(), on=["match_id", "team"], how="left")

    rows = []
    for _, m in matches.iterrows():
        for side, team, opp, gf, ga in [("home", m.home_team, m.away_team, m.home_goals, m.away_goals),
                                        ("away", m.away_team, m.home_team, m.away_goals, m.home_goals)]:
            rows.append({"match_id": m.match_id, "date": m.date, "week": m.week, "team": team,
                         "opponent": opp, "is_home": int(side == "home"),
                         "goals_for": gf, "goals_against": ga})
    tm = pd.DataFrame(rows)

    # Attach our own stats, then the opponent's stats as "against".
    tm = tm.merge(per_team, on=["match_id", "team"], how="left")
    opp = per_team.rename(columns={"team": "opponent", "xg": "xg_against", "shots": "shots_against"})
    tm = tm.merge(opp[["match_id", "opponent", "xg_against", "shots_against"]],
                  on=["match_id", "opponent"], how="left")
    tm = tm.rename(columns={"xg": "xg_for", "shots": "shots_for"})
    # A team with zero shots has no row in `per_team` -> fill those gaps with 0.
    return tm.fillna({"xg_for": 0, "shots_for": 0, "xg_against": 0, "shots_against": 0, "card_points": 0})


def main():
    matches = pd.read_csv(D / "matches.csv")
    shots = pd.read_csv(D / "shots.csv")
    possessions = pd.read_csv(D / "possessions.csv")
    cards = pd.read_csv(D / "cards.csv")

    shots = build_xg(shots)
    shots = add_seconds_before_shot(shots, possessions)
    shots.to_csv(D / "shots_with_xg.csv", index=False)

    tm = team_match_table(matches, shots, cards)
    tm.to_csv(D / "team_matches.csv", index=False)
    print(f"Saved team_matches.csv: {len(tm)} rows (= {len(matches)} matches x 2 teams)")
    print(tm.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
