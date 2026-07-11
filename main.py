import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import pickle

# ==========================================
# 1. LOAD THE TRAINED XGBOOST MODEL
# ==========================================
try:
    # We assume you have trained and saved the XGBoost model as a pickle file previously
    # If not, the script will mock the prediction so the CLI still runs
    with open('xgb_live_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.\n")
except FileNotFoundError:
    print("WARNING: 'xgb_live_model.pkl' not found. Using a mock predictor for demonstration.\n")
    model = None

# ==========================================
# 2. MLB API HELPER FUNCTIONS
# ==========================================


def get_todays_games():
    """Fetches all MLB games happening today and filters for live ones."""
    today = datetime.today().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"

    response = requests.get(url).json()

    if response['totalGames'] == 0:
        return []

    games = response['dates'][0]['games']
    live_games = []

    for game in games:
        # Check if the game is currently "In Progress"
        if game['status']['abstractGameState'] == 'Live':
            away_team = game['teams']['away']['team']['name']
            home_team = game['teams']['home']['team']['name']
            game_pk = game['gamePk']
            live_games.append({
                'id': game_pk,
                'matchup': f"{away_team} @ {home_team}",
                'start_time': game['gameDate']
            })

    return live_games


def get_live_game_state(game_pk):
    """Pulls the exact pitch-by-pitch state required for our XGBoost model."""
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

    # Extract Pitcher Status (Are we in the bullpen?)
    is_home_pitching = linescore.get('inningHalf', 'Top') == 'Top'
    pitching_team_key = 'home' if is_home_pitching else 'away'
    pitchers_used = len(boxscore.get('teams', {}).get(
        pitching_team_key, {}).get('pitchers', []))
    is_starter_pitching = 1 if pitchers_used <= 1 else 0

    # Extract Environment
    attendance = game_info.get('attendance', 25000)
    temp = weather_info.get('temp', 70)
    is_dome = 1 if weather_info.get('condition', '') == 'Dome' else 0

    # Format exactly to our XGBoost features
    state = {
        'inning': inning,
        'outs_when_up': outs,
        'run_diff': run_diff,
        'is_home_leading': is_home_leading,
        'on_1b': on_1b,
        'on_2b': on_2b,
        'on_3b': on_3b,
        'total_runs': total_runs,
        'pitchers_used': pitchers_used,
        'is_starter_pitching': is_starter_pitching,
        'attendance': attendance,
        'temp': int(temp),
        'is_dome': is_dome
    }

    # Also return human-readable summary
    summary = f"Inning {inning} ({'Top' if is_home_pitching else 'Bot'}) | {outs} Outs | Score: {away_score}-{home_score}"
    return state, summary

# ==========================================
# 3. THE CLI APPLICATION
# ==========================================


def main():
    print("=====================================")
    print("⚾ MLB LIVE GAME DURATION PREDICTOR ⚾")
    print("=====================================\n")

    print("Fetching live games from MLB API...\n")
    live_games = get_todays_games()

    if not live_games:
        print("No MLB games are currently live.")
        return

    # Build the CLI Menu
    print("Live Games Available:")
    for idx, game in enumerate(live_games):
        print(f"[{idx + 1}] {game['matchup']}")

    print("\n[0] Exit")

    while True:
        choice = input("\nSelect a game number to predict (or 0 to quit): ")

        try:
            choice = int(choice)
            if choice == 0:
                print("Exiting...")
                break
            if choice < 1 or choice > len(live_games):
                print("Invalid selection. Try again.")
                continue

            selected_game = live_games[choice - 1]
            print(f"\nPulling live data for {selected_game['matchup']}...")

            # Fetch the state
            state_dict, summary = get_live_game_state(selected_game['id'])
            print(f"Current State: {summary}")

            # Make the prediction
            if model:
                # Convert dict to single-row dataframe for XGBoost
                live_df = pd.DataFrame([state_dict])
                predicted_total_mins = model.predict(live_df)[0]
            else:
                # Mock math if no model is loaded: baseline 155 mins + extra for runs/pitchers
                predicted_total_mins = 155 + \
                    (state_dict['total_runs'] * 3) + \
                    (state_dict['pitchers_used'] * 2)

            # Calculate actual end time based on the game's start time
            # MLB API returns start_time in UTC (e.g., '2026-07-11T20:10:00Z')
            start_time_utc = datetime.strptime(
                selected_game['start_time'], "%Y-%m-%dT%H:%M:%SZ")
            predicted_end_time_utc = start_time_utc + \
                timedelta(minutes=float(predicted_total_mins))

            # Convert to local time for output (assuming user wants local formatting)
            # A crude offset for display purposes; a real app would use pytz or tzlocal
            # Here we print the time remaining
            current_time_utc = datetime.utcnow()
            mins_remaining = (predicted_end_time_utc -
                              current_time_utc).total_seconds() / 60

            print("\n-------------------------------------")
            print(
                f"⏱️  Predicted Total Duration: {predicted_total_mins:.1f} minutes")
            if mins_remaining > 0:
                print(
                    f"⏳ Estimated Time Remaining: {mins_remaining:.1f} minutes")
            else:
                print("⏳ The model predicts this game should be ending momentarily!")
            print("-------------------------------------\n")

        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"An error occurred pulling the live data: {e}")


if __name__ == "__main__":
    main()
