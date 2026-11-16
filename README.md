# projectsMatchesReader — a football club match predictor

A learning project that rebuilds the method from the JomoData TikTok
(gather data → clean → attack/defence ratings → Poisson simulation) for
**club football**, using free [StatsBomb open data](https://github.com/statsbomb/open-data)
(Premier League 2015/16 by default).

Each step is one small script. Run them in order, read the code, then try the
exercises at the bottom.

## Setup (once)

```bash
cd projectsMatchesReader
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## The pipeline

| Step | Script | What it does | Output |
|---|---|---|---|
| 1 | `step1_gather_data.py` | Downloads every match and keeps only shots, possessions and cards | `data/matches.csv`, `shots.csv`, `possessions.csv`, `cards.csv` |
| 2 | `step2_clean_features.py` | Trains our own xG model, adds "seconds before shot", builds one row per team per match | `data/team_matches.csv` |
| 3 | `step3_team_ratings.py` | Attack and defence rating per team (home advantage, opponent strength, form) | `data/ratings.csv` |
| 4 | `step4_simulate.py` | Simulates a fixture 20,000 times with Poisson goals | printed probabilities |
| 5 | `step5_evaluate.py` | Tests predictions on matchweeks the model never saw | printed scores |

```bash
.venv/bin/python step1_gather_data.py          # ~1 minute
.venv/bin/python step2_clean_features.py
.venv/bin/python step3_team_ratings.py
.venv/bin/python step4_simulate.py "Leicester" "Arsenal"
.venv/bin/python step5_evaluate.py
```

All settings (league, train/test split, model knobs) are in `config.py`.

---

## Step 1 — Gather data

**Why event data?** Basic stats like "10 shots" hide the important part: a
shot from 6 yards is very different from one from 30 yards. Event data gives
the **x/y location** of every action, so we can measure chance *quality*.

**Why filter while downloading?** One match file has about 3,500 events (≈3.5 MB),
so a whole season would be over 1 GB. We only need shots, the start time of each
possession and cards, which together come to about 3 MB.

Pitch coordinates: StatsBomb uses a 120 × 80 pitch, and the attacking goal is at `x = 120`.

## Step 2 — Clean and build features

**2a. Our own xG model.** xG = the probability that a shot is a goal. We fit a
**logistic regression** (a model that outputs a probability between 0 and 1) on:

- `distance` to the centre of the goal
- `angle`: how wide the goal looks from the shot location
- `is_header`, `is_penalty`, `is_free_kick`

We use `cross_val_predict` so that each shot is scored by a model that was
trained *without* that shot. Otherwise the model would be marking its own homework.

Sanity checks the script prints:
- the coefficients have sensible signs (distance −, angle +, header −)
- correlation with StatsBomb's professional xG is **≈ 0.83**

**2b. Seconds before shot.** This is the time from winning the ball to shooting.
A low value means a direct, counter-attacking team. In 2015/16, Crystal Palace
and Leicester were the most direct teams (≈14–15 s), while Manchester United was
the slowest (≈24 s).

**2c. Team-match table.** We reshape everything into 760 rows (380 matches × 2
teams) with goals, xG and shots, both **for** and **against**. Each match
appears twice, once from each team's point of view, which makes step 3 easy.

## Step 3 — Attack and defence ratings

For every team:

```
attack  = what they produced      / what an average team would produce vs the same opponents
defence = what they conceded      / what an average team would concede vs the same opponents
```

1.00 means league average. For attack, higher is better. For defence, **lower is better**.

Four ideas go into the ratings:

1. **Blend xG and goals** (`XG_WEIGHT = 0.7`). xG is more stable, while goals
   include finishing skill.
2. **Home advantage.** Home and away teams get separate league averages. In
   2015/16, home teams produced about 26% more than away teams.
3. **Strength of schedule.** Ratings depend on the opponents' ratings, so we
   repeat the calculation 30 times until the numbers stop changing. This
   replaces the "FIFA ranking / competition difficulty" idea from the video,
   because all teams play in the same league.
4. **Form.** Recent games weigh more. A game played `FORM_HALF_LIFE` matches ago
   counts half as much as the latest one.

Only weeks 1–30 are used, so step 5 has untouched games to test on.

## Step 4 — Poisson simulation

```
home_xg = league_home_avg × home_attack × away_defence
away_xg = league_away_avg × away_attack × home_defence
```

Goals are rare, independent-ish events, so the number of goals in a match follows a
**Poisson distribution** with that average. For each of the 20,000 simulations:

1. Give each team a random "form on the day" multiplier (average 1.0). Weaker
   teams get a wider spread (`EXTRA_NOISE_WEAK`), which makes upsets more
   likely, as described in the video.
2. Draw home and away goals from the Poisson distribution.

Then we count the results: % home win / draw / away win and the most common scorelines.

## Step 5 — Evaluate (the step people skip!)

We predict weeks 31–38 and compare against a **baseline** that always predicts
the league's usual home/draw/away rates:

| metric | our model | baseline |
|---|---|---|
| accuracy | 0.475 | 0.438 |
| log loss (lower = better) | 1.012 | 1.084 |
| Brier (lower = better) | 0.604 | 0.656 |

The model beats the baseline on every metric. Bookmakers manage about 50–53%
accuracy, so this is a reasonable start. With only 80 test matches, though,
the numbers are noisy.

**Be honest about limits.** The model doesn't know about injuries, lineups,
transfers or motivation. It's a tool for insight, not a guarantee, as the
video also says.

---

## Exercises to learn more

1. **Change one knob at a time** in `config.py` (`XG_WEIGHT`, `FORM_HALF_LIFE`,
   `EXTRA_NOISE_WEAK`), rerun steps 3 and 5, and see whether the log loss improves.
2. **Another league.** Set `COMPETITION_ID = 11` (La Liga) or `12` (Serie A), then rerun every step.
3. **Better xG.** Add features in `shot_features()`, such as `shot_type == "Corner"`
   or the play pattern (counter-attack?). Check whether the correlation with StatsBomb's xG rises.
4. **Use the style stats.** `secs_before_shot` and `cards_pg` are calculated
   but not used. Can you work them into the ratings, and does step 5 get better?
5. **Draws.** Plain Poisson tends to predict too few draws. Look up the
   *Dixon–Coles* adjustment and try adding it.

Data: StatsBomb Open Data. Please credit StatsBomb if you publish anything built on it.
