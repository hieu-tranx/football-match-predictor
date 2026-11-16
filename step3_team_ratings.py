"""STEP 3 - Turn match stats into an ATTACK and DEFENCE rating per team.

  attack  1.30 = creates 30% more than an average team
  defence 0.70 = concedes 30% LESS than an average team (lower is better)

Three ideas make it club-specific:
  * Home advantage - home and away teams get separate league averages.
  * Strength of schedule - scoring 2 xG vs the champions is worth more than
    vs the bottom side, so we divide by the opponent's rating and repeat
    until the numbers settle (the "competition difficulty" idea from the video,
    but inside one league).
  * Form - recent matches weigh more (FORM_HALF_LIFE in config.py).

Only matchweeks <= TRAIN_UNTIL_WEEK are used, so step 5 can test on the rest.

Run:  .venv/bin/python step3_team_ratings.py
"""
import json

import pandas as pd

import config

D = config.DATA_DIR


def compute_ratings(tm, iterations=30):
    tm = tm.copy()
    # Blend xG (what SHOULD have happened) with goals (what DID happen).
    w = config.XG_WEIGHT
    tm["att_value"] = w * tm["xg_for"] + (1 - w) * tm["goals_for"]
    tm["def_value"] = w * tm["xg_against"] + (1 - w) * tm["goals_against"]

    # Form weight: the latest match = 1.0, FORM_HALF_LIFE games earlier = 0.5, ...
    tm = tm.sort_values("date")
    games_ago = tm.groupby("team").cumcount(ascending=False)
    tm["weight"] = 0.5 ** (games_ago / config.FORM_HALF_LIFE)

    # League-average output for a home team and for an away team.
    avg = {"home": tm.loc[tm.is_home == 1, "att_value"].mean(),
           "away": tm.loc[tm.is_home == 0, "att_value"].mean()}
    tm["my_avg"] = tm["is_home"].map({1: avg["home"], 0: avg["away"]})    # what I'd score if average
    tm["opp_avg"] = tm["is_home"].map({1: avg["away"], 0: avg["home"]})   # what they'd score if average

    teams = sorted(tm["team"].unique())
    attack = pd.Series(1.0, index=teams)
    defence = pd.Series(1.0, index=teams)

    for _ in range(iterations):
        # attack = what I produced / what an average team would vs these same defences
        exp_for = tm["my_avg"] * tm["opponent"].map(defence)
        attack = ((tm["weight"] * tm["att_value"]).groupby(tm["team"]).sum()
                  / (tm["weight"] * exp_for).groupby(tm["team"]).sum())
        exp_against = tm["opp_avg"] * tm["opponent"].map(attack)
        defence = ((tm["weight"] * tm["def_value"]).groupby(tm["team"]).sum()
                   / (tm["weight"] * exp_against).groupby(tm["team"]).sum())
        # Keep the league average at exactly 1.0 so ratings stay readable.
        attack /= attack.mean()
        defence /= defence.mean()

    ratings = pd.DataFrame({"attack": attack, "defence": defence})
    ratings["strength"] = ratings["attack"] / ratings["defence"]

    # Style stats - not used in the maths, but useful to read and compare.
    style = tm.groupby("team").agg(games=("match_id", "size"),
                                   secs_before_shot=("secs_before_shot", "mean"),
                                   shots_pg=("shots_for", "mean"),
                                   cards_pg=("card_points", "mean"))
    ratings = ratings.join(style).sort_values("strength", ascending=False)
    return ratings, avg


def main():
    tm = pd.read_csv(D / "team_matches.csv")
    train = tm[tm["week"] <= config.TRAIN_UNTIL_WEEK]
    ratings, avg = compute_ratings(train)

    ratings.to_csv(D / "ratings.csv", index_label="team")
    (D / "league_averages.json").write_text(json.dumps(avg, indent=2))

    print(f"Trained on weeks 1-{config.TRAIN_UNTIL_WEEK} ({len(train) // 2} matches)")
    print(f"League average output: home {avg['home']:.2f}, away {avg['away']:.2f}  "
          f"-> home advantage {avg['home'] / avg['away'] - 1:+.0%}\n")
    print(ratings.round(2).to_string())


if __name__ == "__main__":
    main()
