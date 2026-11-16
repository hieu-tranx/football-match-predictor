"""STEP 1 - Gather data.

StatsBomb gives one JSON file per match with ~3,500 "events" (every pass, shot,
tackle...). A full season of raw files is >1 GB, so we download each match and
keep ONLY the rows we need for the model:

  matches.csv      one row per match (teams, date, final score)
  shots.csv        every shot with its x/y location  -> used to build xG
  possessions.csv  when each possession started       -> "seconds before a shot"
  cards.csv        yellow / red cards                 -> discipline metric

Run:  .venv/bin/python step1_gather_data.py
"""
import warnings
warnings.filterwarnings("ignore")  # hides a harmless SSL warning on macOS Python (must run before `import requests`)

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

import config


def get_json(path):
    r = requests.get(f"{config.BASE_URL}/{path}", timeout=60)
    r.raise_for_status()
    return r.json()


def load_matches():
    raw = get_json(f"matches/{config.COMPETITION_ID}/{config.SEASON_ID}.json")
    rows = [{
        "match_id": m["match_id"],
        "date": m["match_date"],
        "week": m["match_week"],
        "home_team": m["home_team"]["home_team_name"],
        "away_team": m["away_team"]["away_team_name"],
        "home_goals": m["home_score"],
        "away_goals": m["away_score"],
    } for m in raw]
    return pd.DataFrame(rows).sort_values(["date", "match_id"])


def slim_events(match_id):
    """Download one match and return (shots, possessions, cards) as lists of dicts."""
    events = get_json(f"events/{match_id}.json")
    shots, possessions, cards = [], [], []
    seen_possessions = set()

    for e in events:
        if e["period"] > 4:          # period 5 = penalty shoot-out, not part of the match
            continue
        secs = e["minute"] * 60 + e["second"]

        # First event of each possession = the moment that team won the ball.
        if e["possession"] not in seen_possessions:
            seen_possessions.add(e["possession"])
            possessions.append({"match_id": match_id, "possession": e["possession"],
                                "team": e["possession_team"]["name"], "start_sec": secs})

        if e["type"]["name"] == "Shot":
            s = e["shot"]
            shots.append({
                "match_id": match_id, "team": e["team"]["name"], "possession": e["possession"],
                "sec": secs, "x": e["location"][0], "y": e["location"][1],
                "body_part": s["body_part"]["name"], "shot_type": s["type"]["name"],
                "goal": int(s["outcome"]["name"] == "Goal"),
                "statsbomb_xg": s.get("statsbomb_xg"),  # kept only to compare with OUR xG later
            })

        # Cards can live on a foul or on "bad behaviour" (e.g. dissent).
        for key in ("foul_committed", "bad_behaviour"):
            card = e.get(key, {}).get("card")
            if card:
                cards.append({"match_id": match_id, "team": e["team"]["name"], "card": card["name"]})

    return shots, possessions, cards


def main():
    matches = load_matches()
    print(f"{len(matches)} matches found. Downloading events (takes a few minutes)...")

    shots, possessions, cards = [], [], []
    # 8 downloads at once instead of one by one - much faster.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, (s, p, c) in enumerate(pool.map(slim_events, matches["match_id"]), 1):
            shots += s; possessions += p; cards += c
            if i % 50 == 0:
                print(f"  {i}/{len(matches)}")

    matches.to_csv(config.DATA_DIR / "matches.csv", index=False)
    pd.DataFrame(shots).to_csv(config.DATA_DIR / "shots.csv", index=False)
    pd.DataFrame(possessions).to_csv(config.DATA_DIR / "possessions.csv", index=False)
    pd.DataFrame(cards).to_csv(config.DATA_DIR / "cards.csv", index=False)
    print(f"Saved {len(shots)} shots, {len(possessions)} possessions, {len(cards)} cards to data/")


if __name__ == "__main__":
    main()
