"""STEP 4 - Simulate a match with a Poisson model.

1. Expected goals for this fixture:
     home_xg = league_home_avg * home_attack * away_defence
     away_xg = league_away_avg * away_attack * home_defence
2. Goals in football are rare, random events -> they follow a Poisson
   distribution. We draw a random scoreline from it.
3. Teams don't play the same every week, so each simulated match first gives
   each team a random "good day / bad day" factor. Weaker teams get MORE
   randomness (bigger upsets), like in the video.
4. Repeat N_SIMULATIONS times and count: % home win / draw / away win,
   and the most common scorelines.

Run:  .venv/bin/python step4_simulate.py "Leicester City" "Arsenal"
"""
import difflib
import json
import sys
from collections import Counter

import numpy as np
import pandas as pd

import config

D = config.DATA_DIR


def load_model():
    ratings = pd.read_csv(D / "ratings.csv", index_col="team")
    avg = json.loads((D / "league_averages.json").read_text())
    # Strength percentile: 0 = weakest team, 1 = strongest.
    pct = ratings["strength"].rank(pct=True)
    ratings["noise"] = config.BASE_NOISE + config.EXTRA_NOISE_WEAK * (1 - pct)
    return ratings, avg


def simulate(home, away, ratings, avg, n=config.N_SIMULATIONS, seed=0):
    rng = np.random.default_rng(seed)
    h, a = ratings.loc[home], ratings.loc[away]
    home_xg = avg["home"] * h.attack * a.defence
    away_xg = avg["away"] * a.attack * h.defence

    # Lognormal "form on the day" factor with average 1.0 (the -s^2/2 keeps the mean unchanged).
    home_day = np.exp(rng.normal(-h.noise ** 2 / 2, h.noise, n))
    away_day = np.exp(rng.normal(-a.noise ** 2 / 2, a.noise, n))
    home_goals = rng.poisson(home_xg * home_day)
    away_goals = rng.poisson(away_xg * away_day)

    return {
        "home_xg": home_xg, "away_xg": away_xg,
        "home_win": np.mean(home_goals > away_goals),
        "draw": np.mean(home_goals == away_goals),
        "away_win": np.mean(home_goals < away_goals),
        "scorelines": Counter(zip(home_goals, away_goals)).most_common(5),
    }


def find_team(name, teams):
    # 1) every typed word starts a word in the team name: "man city" -> Manchester City
    words = name.lower().split()
    hits = [t for t in teams if all(any(tw.startswith(w) for tw in t.lower().split()) for w in words)]
    if len(hits) == 1:
        return hits[0]
    # 2) otherwise fall back to closest spelling
    match = difflib.get_close_matches(name.title(), teams, n=1, cutoff=0.3)
    if not match:
        sys.exit(f"Unknown team '{name}'. Options: {', '.join(teams)}")
    return match[0]


def main():
    ratings, avg = load_model()
    teams = list(ratings.index)
    if len(sys.argv) != 3:
        sys.exit('Usage: step4_simulate.py "Home Team" "Away Team"\nTeams: ' + ", ".join(teams))
    home, away = find_team(sys.argv[1], teams), find_team(sys.argv[2], teams)

    r = simulate(home, away, ratings, avg)
    print(f"\n{home} (home) vs {away}  -  {config.N_SIMULATIONS:,} simulations")
    print(f"Expected goals: {r['home_xg']:.2f} - {r['away_xg']:.2f}\n")
    print(f"  {home:22s} win  {r['home_win']:6.1%}")
    print(f"  {'Draw':22s}      {r['draw']:6.1%}")
    print(f"  {away:22s} win  {r['away_win']:6.1%}\n")
    print("Most likely scorelines:")
    for (hg, ag), count in r["scorelines"]:
        print(f"  {hg}-{ag}  {count / config.N_SIMULATIONS:5.1%}")


if __name__ == "__main__":
    main()
