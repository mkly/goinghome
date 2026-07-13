from zoneinfo import ZoneInfo
from streamlit_javascript import st_javascript
import pickle
from datetime import datetime, timedelta
import streamlit as st
import requests
import pandas as pd

pd.options.mode.string_storage = "python"
try:
    pd.options.future.infer_string = False
except Exception:
    pass

# ==========================================
# 1. PAGE CONFIG & MODEL CACHING
# ==========================================
st.set_page_config(
    page_title="MLB Duration Predictor", page_icon="⚾", layout="centered"
)


@st.cache_resource
def load_model():
    """Loads the XGBoost model."""
    try:
        with open("xgb_live_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


model = load_model()

# ==========================================
# 2. MLB API HELPER FUNCTIONS
# ==========================================


def get_todays_games():
    """Fetches all live MLB games happening today."""
    today = datetime.today().strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=broadcasts"

    response = requests.get(url).json()

    if response["totalGames"] == 0:
        return []

    games = response["dates"][0]["games"]
    live_games = []

    for game in games:
        # Check if the game is currently live
        if game["status"]["abstractGameState"] == "Live":
            away_team = game["teams"]["away"]["team"]["name"]
            home_team = game["teams"]["home"]["team"]["name"]

            is_national = 0
            for b in game.get("broadcasts", []):
                if b.get("isNational", False) and b.get("type", "") == "TV":
                    is_national = 1
                    break

            is_night = 1 if game.get("dayNight", "") == "night" else 0

            live_games.append(
                {
                    "id": game["gamePk"],
                    "matchup": f"{away_team} @ {home_team}",
                    "start_time": game["gameDate"],
                    "is_national_tv": is_national,
                    "is_night_game": is_night,
                }
            )

    return live_games


def get_live_game_state(game_pk, is_national_tv=0, is_night_game=0):
    """Pulls the pitch-by-pitch state required for our XGBoost model."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    response = requests.get(url).json()

    live_data = response.get("liveData", {})
    linescore = live_data.get("linescore", {})
    boxscore = live_data.get("boxscore", {})
    weather_info = response.get("gameData", {}).get("weather", {})

    # Extract Score & Inning
    inning = linescore.get("currentInning", 1)
    outs = linescore.get("outs", 0)
    home_score = linescore.get("teams", {}).get("home", {}).get("runs", 0)
    away_score = linescore.get("teams", {}).get("away", {}).get("runs", 0)
    run_diff = abs(home_score - away_score)
    is_home_leading = 1 if home_score > away_score else 0
    total_runs = home_score + away_score

    # Extract Base Runners
    offense = linescore.get("offense", {})
    on_1b = 1 if "first" in offense else 0
    on_2b = 1 if "second" in offense else 0
    on_3b = 1 if "third" in offense else 0

    home_pitchers_used = len(
        boxscore.get("teams", {}).get("home", {}).get("pitchers", [])
    )
    away_pitchers_used = len(
        boxscore.get("teams", {}).get("away", {}).get("pitchers", [])
    )

    home_starting_pitcher = 1 if home_pitchers_used <= 1 else 0
    away_starting_pitcher = 1 if away_pitchers_used <= 1 else 0

    is_dome = 1 if weather_info.get("condition", "") == "Dome" else 0

    # New Context Features
    home_div = (
        response.get("gameData", {})
        .get("teams", {})
        .get("home", {})
        .get("division", {})
        .get("id")
    )
    away_div = (
        response.get("gameData", {})
        .get("teams", {})
        .get("away", {})
        .get("division", {})
        .get("id")
    )
    is_rivalry = 1 if home_div == away_div and home_div is not None else 0

    home_pitches = (
        boxscore.get("teams", {})
        .get("home", {})
        .get("teamStats", {})
        .get("pitching", {})
        .get("numberOfPitches", 0)
    )
    away_pitches = (
        boxscore.get("teams", {})
        .get("away", {})
        .get("teamStats", {})
        .get("pitching", {})
        .get("numberOfPitches", 0)
    )
    total_pitch_count = int(home_pitches) + int(away_pitches)

    home_pa = (
        boxscore.get("teams", {})
        .get("home", {})
        .get("teamStats", {})
        .get("batting", {})
        .get("plateAppearances", 0)
    )
    away_pa = (
        boxscore.get("teams", {})
        .get("away", {})
        .get("teamStats", {})
        .get("batting", {})
        .get("plateAppearances", 0)
    )
    total_pa = int(home_pa) + int(away_pa)

    is_tied = 1 if run_diff == 0 else 0

    state = {
        "inning": int(inning),
        "outs_when_up": int(outs),
        "run_diff": int(run_diff),
        "is_home_leading": int(is_home_leading),
        "is_tied": int(is_tied),
        "on_1b": int(on_1b),
        "on_2b": int(on_2b),
        "on_3b": int(on_3b),
        "total_runs": int(total_runs),
        "home_pitchers_used": int(home_pitchers_used),
        "away_pitchers_used": int(away_pitchers_used),
        "home_starting_pitcher": int(home_starting_pitcher),
        "away_starting_pitcher": int(away_starting_pitcher),
        "total_pitch_count": int(total_pitch_count),
        "total_pa": int(total_pa),
        "is_dome": int(is_dome),
        "is_national_tv": int(is_national_tv),
        "is_night_game": int(is_night_game),
        "is_rivalry": int(is_rivalry),
    }

    is_home_pitching = linescore.get("inningHalf", "Top") == "Top"
    inning_half = "Top" if is_home_pitching else "Bot"
    summary = (
        f"{inning_half} {inning} | {outs} Outs | Score: {away_score} - {home_score}"
    )

    return state, summary


# ==========================================
# 3. STREAMLIT UI BUILDER
# ==========================================
st.title("⚾ MLB Live Duration Predictor")

if model is None:
    st.error("⚠️ 'xgb_live_model.pkl' not found. Please run the training script first.")
    st.stop()


# Automatically detect the user's timezone from their browser using JS
client_timezone = st_javascript("Intl.DateTimeFormat().resolvedOptions().timeZone")
if isinstance(client_timezone, str):
    try:
        user_tz = ZoneInfo(client_timezone)
    except Exception:
        user_tz = None
else:
    user_tz = None

st.write("Select a live game to see a prediction of how much time is left.")

# Fetch games
with st.spinner("Scanning MLB for live games..."):
    live_games = get_todays_games()

if not live_games:
    st.info("No MLB games are currently live.")
else:
    # Create Dropdown Dictionary
    game_options = {game["matchup"]: game for game in live_games}
    selected_matchup = st.selectbox(
        "Select a Matchup:",
        options=list(game_options.keys()),
        index=None,
        placeholder="Choose a live game to predict...",
    )

    if selected_matchup:
        selected_game = game_options[selected_matchup]

        with st.spinner(f"Pulling live data for {selected_matchup}..."):
            state_dict, summary = get_live_game_state(
                selected_game["id"],
                selected_game.get("is_national_tv", 0),
                selected_game.get("is_night_game", 0),
            )

            # Make the Prediction
            # Match the saved model's schema. ``reindex`` also supplies a neutral
            # value for features that may be absent when the deployed model and
            # live-data code were produced by different releases.
            live_df = pd.DataFrame([state_dict])
            if hasattr(model, "feature_names_in_"):
                live_df = live_df.reindex(columns=model.feature_names_in_, fill_value=0)

            predicted_total_mins = model.predict(live_df)[0]

            # Calculate Time Remaining
            from datetime import timezone

            start_time_utc = datetime.strptime(
                selected_game["start_time"], "%Y-%m-%dT%H:%M:%SZ"
            ).replace(tzinfo=timezone.utc)
            current_time_utc = datetime.now(timezone.utc)

            # Prevent negative elapsed time if the game hasn't reached its scheduled start yet
            minutes_elapsed = max(
                0, (current_time_utc - start_time_utc).total_seconds() / 60
            )
            mins_remaining = float(predicted_total_mins) - minutes_elapsed

        # --- DISPLAY RESULTS ---
        st.divider()
        st.subheader(f"Current State: {summary}")

        # Big Prediction Metrics
        st.markdown("### 🔮 Model Forecast")
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            if mins_remaining > 0:
                if user_tz:
                    expected_end_time = datetime.now(user_tz) + timedelta(
                        minutes=mins_remaining
                    )
                else:
                    expected_end_time = datetime.now().astimezone() + timedelta(
                        minutes=mins_remaining
                    )

                end_time_str = expected_end_time.strftime("%I:%M %p %Z").lstrip("0")
                st.metric(label="Expected End Time", value=end_time_str)
            else:
                st.metric(label="Expected End Time", value="Any minute now!")

        with res_col2:
            st.metric(
                label="Projected Total Duration",
                value=f"{predicted_total_mins:.1f} mins",
            )

        with res_col3:
            if mins_remaining > 0:
                st.metric(
                    label="Estimated Time Remaining", value=f"{mins_remaining:.1f} mins"
                )
            else:
                st.metric(label="Status", value="Game is wrapping up!")

        st.divider()

        # Show all model input features
        st.write("**Model Input Features:**")
        cols = st.columns(4)
        for i, (key, value) in enumerate(state_dict.items()):
            cols[i % 4].metric(key, value)
