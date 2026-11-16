"""Shared settings for every step. Change things here, not inside the step files."""
from pathlib import Path

# --- Which league to use -------------------------------------------------
# StatsBomb open data has these COMPLETE club seasons (all 2015/16):
#   Premier League (2, 27)   La Liga (11, 27)   Serie A (12, 27)   Ligue 1 (7, 27)
COMPETITION_ID = 2
SEASON_ID = 27

# --- Train / test split --------------------------------------------------
# Ratings are built from matchweeks 1..TRAIN_UNTIL_WEEK.
# Step 5 then predicts the remaining weeks the model has never seen.
TRAIN_UNTIL_WEEK = 30

# --- Model knobs ---------------------------------------------------------
XG_WEIGHT = 0.7          # rating blend: 70% expected goals, 30% real goals
FORM_HALF_LIFE = 10      # a match 10 games ago counts half as much as the latest one
N_SIMULATIONS = 20_000   # how many times we replay each fixture
BASE_NOISE = 0.10        # match-to-match randomness for the strongest teams
EXTRA_NOISE_WEAK = 0.15  # extra randomness added for the weakest teams (upsets)

# --- Paths ---------------------------------------------------------------
BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
