import pandas as pd
import numpy as np
import requests
import time
from pybaseball import statcast

print("Step 1: Extracting Raw Statcast Data...")
# Using the 2024 season (Pitch Clock Era) to avoid OOM errors
df = statcast(start_dt="2024-03-28", end_dt="2024-07-11")

print("Step 2: Engineering Features in Memory...")
# Sort chronologically
df = df.sort_values(
    by=["game_date", "game_pk", "at_bat_number", "pitch_number"]
).reset_index(drop=True)

# Build the game state features
df["on_1b"] = df["on_1b"].notna().astype(int)
df["on_2b"] = df["on_2b"].notna().astype(int)
df["on_3b"] = df["on_3b"].notna().astype(int)
df["total_runs"] = df["bat_score"] + df["fld_score"]
df["run_diff"] = abs(df["bat_score"] - df["fld_score"])
df["is_tied"] = np.where(df["run_diff"] == 0, 1, 0)

df["total_pitch_count"] = df.groupby("game_pk").cumcount() + 1
df["total_pa"] = df["at_bat_number"]

# Build the lead feature
df["is_home_leading"] = np.where(
    (df["inning_topbot"] == "Top") & (df["fld_score"] > df["bat_score"]),
    1,
    np.where(
        (df["inning_topbot"] == "Bot") & (df["bat_score"] > df["fld_score"]), 1, 0
    ),
)

# Build the personnel features
# Build the personnel features
# Build the personnel features
df["home_pitcher"] = np.where(df["inning_topbot"] == "Top", df["pitcher"], np.nan)
df["away_pitcher"] = np.where(df["inning_topbot"] == "Bot", df["pitcher"], np.nan)
df["home_pitcher_ffill"] = df.groupby("game_pk")["home_pitcher"].ffill()
df["away_pitcher_ffill"] = df.groupby("game_pk")["away_pitcher"].ffill()
df["home_pitchers_used"] = df.groupby("game_pk")["home_pitcher_ffill"].transform(
    lambda x: pd.factorize(x)[0] + 1
)
df["away_pitchers_used"] = df.groupby("game_pk")["away_pitcher_ffill"].transform(
    lambda x: pd.factorize(x)[0] + 1
)

df["home_starting_pitcher"] = np.where(df["home_pitchers_used"] <= 1, 1, 0)
df["away_starting_pitcher"] = np.where(df["away_pitchers_used"] <= 1, 1, 0)

# DROP THE BLOAT: Only keep what we absolutely need moving forward
core_features = [
    "game_pk",
    "inning",
    "outs_when_up",
    "run_diff",
    "is_home_leading",
    "is_tied",
    "on_1b",
    "on_2b",
    "on_3b",
    "total_runs",
    "home_pitchers_used",
    "away_pitchers_used",
    "home_starting_pitcher",
    "away_starting_pitcher",
    "total_pitch_count",
    "total_pa",
]
df = df[core_features]

print("Step 3: Fetching API Metadata...")
unique_games = df["game_pk"].unique()
game_metadata = []

# Fetch broadcasts for the 2024 season
schedule_url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=2024-03-28&endDate=2024-07-11&hydrate=broadcasts"
schedule_resp = requests.get(schedule_url).json()
game_info_map = {}

if "dates" in schedule_resp:
    for date_obj in schedule_resp["dates"]:
        for game_obj in date_obj.get("games", []):
            g_id = game_obj["gamePk"]
            is_national = 0
            broadcasts = game_obj.get("broadcasts", [])
            for b in broadcasts:
                if b.get("isNational", False) and b.get("type", "") == "TV":
                    is_national = 1
                    break
            is_night = 1 if game_obj.get("dayNight", "") == "night" else 0
            game_info_map[g_id] = {
                "is_national_tv": is_national,
                "is_night_game": is_night,
            }

for count, game_id in enumerate(unique_games):
    try:
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_id}/feed/live"
        response = requests.get(url).json()
        game_info = response["gameData"]["gameInfo"]

        attendance = game_info.get("attendance", 0)
        temp = response["gameData"]["weather"].get("temp", 70)
        is_dome = (
            1 if response["gameData"]["weather"].get("condition", "") == "Dome" else 0
        )
        duration = game_info.get("gameDurationMinutes", 160)

        home_div = (
            response["gameData"]
            .get("teams", {})
            .get("home", {})
            .get("division", {})
            .get("id")
        )
        away_div = (
            response["gameData"]
            .get("teams", {})
            .get("away", {})
            .get("division", {})
            .get("id")
        )
        is_rivalry = 1 if home_div == away_div and home_div is not None else 0

        info = game_info_map.get(game_id, {"is_national_tv": 0, "is_night_game": 0})

        game_metadata.append(
            {
                "game_pk": game_id,
                "attendance": attendance,
                "temp": int(temp),
                "is_dome": is_dome,
                "is_national_tv": info["is_national_tv"],
                "is_night_game": info["is_night_game"],
                "is_rivalry": is_rivalry,
                "final_game_minutes": int(duration),
            }
        )
    except Exception:
        pass  # Skip missing games

    # Rate limiting to respect the MLB API
    time.sleep(0.1)

metadata_df = pd.DataFrame(game_metadata)

print("Step 4: Merging and Pickling...")
# Merge the API data into our lean Statcast dataframe
master_df = pd.merge(df, metadata_df, on="game_pk", how="left")
master_df = master_df.dropna(subset=["final_game_minutes"])

# Drop the game_pk identifier (XGBoost doesn't need it)
master_df = master_df.drop(columns=["game_pk"])

# Save exactly what the model needs to a highly compressed pickle file
master_df.to_pickle("mlb_training_data_clean.pkl")
print("SUCCESS! File saved as 'mlb_training_data_clean.pkl'.")
