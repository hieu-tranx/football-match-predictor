"""STEP 5 - Is the model any good? Test it on matches it has never seen.

Ratings come from weeks 1..TRAIN_UNTIL_WEEK (step 3). Here we predict every
later match and compare with what really happened, against a "dumb" baseline
that just uses how often home wins / draws / away wins happen in general.

Scores:
  accuracy  - % of matches where the most likely outcome actually happened
  log loss  - punishes being confidently wrong. LOWER is better.
  Brier     - squared error of the probabilities. LOWER is better.

Run:  .venv/bin/python step5_evaluate.py
"""
import numpy as np
import pandas as pd

import config
from step4_simulate import load_model, simulate

D = config.DATA_DIR
OUTCOMES = ["home_win", "draw", "away_win"]


def score(probs, actual):
    """probs: (n, 3) array, actual: (n,) array of 0/1/2."""
    p = np.clip(probs, 1e-9, 1)
    onehot = np.eye(3)[actual]
    return {"accuracy": np.mean(p.argmax(1) == actual),
            "log loss": -np.mean(np.log(p[np.arange(len(actual)), actual])),
            "Brier": np.mean(((p - onehot) ** 2).sum(1))}


def main():
    ratings, avg = load_model()
    matches = pd.read_csv(D / "matches.csv")
    train = matches[matches["week"] <= config.TRAIN_UNTIL_WEEK]
    test = matches[matches["week"] > config.TRAIN_UNTIL_WEEK].copy()

    # What actually happened: 0 = home win, 1 = draw, 2 = away win
    def outcome(df):
        return np.select([df.home_goals > df.away_goals, df.home_goals == df.away_goals], [0, 1], 2)
    test["actual"] = outcome(test)

    preds = [simulate(m.home_team, m.away_team, ratings, avg, n=5000) for m in test.itertuples()]
    model_p = np.array([[p[o] for o in OUTCOMES] for p in preds])
    base_rates = np.bincount(outcome(train), minlength=3) / len(train)
    base_p = np.tile(base_rates, (len(test), 1))

    print(f"Testing on weeks {config.TRAIN_UNTIL_WEEK + 1}-{test.week.max()} ({len(test)} matches)\n")
    results = pd.DataFrame({"our model": score(model_p, test.actual),
                            "baseline": score(base_p, test.actual)})
    print(results.round(3).to_string(), "\n")

    test[["P(home)", "P(draw)", "P(away)"]] = model_p.round(2)
    test["result"] = test.home_goals.astype(str) + "-" + test.away_goals.astype(str)
    print("Sample predictions:")
    print(test[["week", "home_team", "away_team", "P(home)", "P(draw)", "P(away)", "result"]]
          .head(12).to_string(index=False))
    test.to_csv(D / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
