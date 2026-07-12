import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import pickle

# ==========================================
# 1. PAGE CONFIG & MODEL CACHING
# ==========================================
st.set_page_config(page_title="MLB Duration Predictor",
                   page_icon="⚾", layout="centered")


@st.cache_resource
def load_model():
    """Loads the XGBoost model once and caches it in memory."""
    try:
        with open('xgb_live_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


model = load_model()

# ==========================================
# 2. MLB API HELPER FUNCTIONS
# ==========================================


def get_todays_games():
    """Fetches all live MLB games happening today."""
    today = datetime.today().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"

    response = requests.get(url).json()
    if response['totalGames'] == 0:
        return []

    games = response['dates'][0]['games']
    live_games = []

    for game in games:
        # Check if the game is currently live
        if game['status']['abstractGameState'] == 'Live':
            away_team = game['teams']['away']['team']['name']
            home_team = game['teams']['home']['team']['name']
            live_games.append({
                'id': game['gamePk'],
                'matchup': f"{away_team} @ {home_team}",
                'start_time': game['gameDate']
            })

    return live_games


def get_live_game_state(game_pk):
    """Pulls the pitch-by-pitch state required for our XGBoost model."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    response = requests.get(url).json()

    live_data = response.get('liveData', {})
    linescore = live_data.get('linescore', {})
    boxscore = live_data.get('boxscore', {})
    game_info = response.get('gameData', {}).get('gameInfo', {})
    weather_info = response.get('gameData', {}).get('weather', {})

    # Extract Score & Inning
    inning = linescore.get('currentInning', 1)
    outs = linescore.get('outs', 0)
    home_score = linescore.get('teams', {}).get('home', {}).get('runs', 0)
    away_score = linescore.get('teams', {}).get('away', {}).get('runs', 0)
    run_diff = abs(home_score - away_score)
    is_home_leading = 1 if home_score > away_score else 0
    total_runs = home_score + away_score

    # Extract Base Runners
    offense = linescore.get('offense', {})
    on_1b = 1 if 'first' in offense else 0
    on_2b = 1 if 'second' in offense else 0
    on_3b = 1 if 'third' in offense else 0

    # Extract Pitcher Status
    is_home_pitching = linescore.get('inningHalf', 'Top') == 'Top'
    pitching_team_key = 'home' if is_home_pitching else 'away'
    pitchers_used = len(boxscore.get('teams', {}).get(
        pitching_team_key, {}).get('pitchers', []))
    is_starter_pitching = 1 if pitchers_used <= 1 else 0

    # Extract Environment
    attendance = game_info.get('attendance', 25000)
    temp = weather_info.get('temp', 70)
    is_dome = 1 if weather_info.get('condition', '') == 'Dome' else 0

    state = {
        'inning': inning, 'outs_when_up': outs, 'run_diff': run_diff,
        'is_home_leading': is_home_leading, 'on_1b': on_1b, 'on_2b': on_2b,
        'on_3b': on_3b, 'total_runs': total_runs, 'pitchers_used': pitchers_used,
        'is_starter_pitching': is_starter_pitching, 'attendance': attendance,
        'temp': int(temp), 'is_dome': is_dome
    }

    inning_half = "Top" if is_home_pitching else "Bot"
    summary = f"{inning_half} {inning} | {outs} Outs | Score: {away_score} - {home_score}"

    return state, summary


# ==========================================
# 3. STREAMLIT UI BUILDER
# ==========================================
st.title("⚾ MLB Live Duration Predictor")

if model is None:
    st.error(
        "⚠️ 'xgb_live_model.pkl' not found. Please run the training script first.")
    st.stop()

st.write("Select a live game to calculate exactly how much time is left based on the current pitch data.")

# Fetch games
with st.spinner("Scanning MLB for live games..."):
    live_games = get_todays_games()

if not live_games:
    st.info("No MLB games are currently live. Try again later!")
else:
    # Create Dropdown Dictionary
    game_options = {game['matchup']: game for game in live_games}
    selected_matchup = st.selectbox(
        "Select a Matchup:", list(game_options.keys()))
    selected_game = game_options[selected_matchup]

    if st.button("🚀 Predict Live State", type="primary", use_container_width=True):
        with st.spinner(f"Pulling live data for {selected_matchup}..."):
            state_dict, summary = get_live_game_state(selected_game['id'])

            # Make the Prediction
            live_df = pd.DataFrame([state_dict])
            predicted_total_mins = model.predict(live_df)[0]

            # Calculate Time Remaining
            start_time_utc = datetime.strptime(
                selected_game['start_time'], "%Y-%m-%dT%H:%M:%SZ")
            current_time_utc = datetime.utcnow()
            
            # Prevent negative elapsed time if the game hasn't reached its scheduled start yet
            minutes_elapsed = max(0, (current_time_utc - start_time_utc).total_seconds() / 60)
            mins_remaining = float(predicted_total_mins) - minutes_elapsed

            # --- DISPLAY RESULTS ---
            st.divider()
            st.subheader(f"Current State: {summary}")

            # Show all model input features
            st.write("**Model Input Features:**")
            cols = st.columns(4)
            for i, (key, value) in enumerate(state_dict.items()):
                cols[i % 4].metric(key, value)

            st.divider()

            # Big Prediction Metrics
            st.markdown("### 🔮 Model Forecast")
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                st.metric(
                    label="Projected Total Duration",
                    value=f"{predicted_total_mins:.1f} mins"
                )

            with res_col2:
                if mins_remaining > 0:
                    st.metric(
                        label="Estimated Time Remaining",
                        value=f"{mins_remaining:.1f} mins"
                    )
                else:
                    st.metric(
                        label="Status",
                        value="Game is wrapping up!"
                    )
