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
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=broadcasts"

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
            
            is_national = 0
            for b in game.get('broadcasts', []):
                if b.get('isNational', False) and b.get('type', '') == 'TV':
                    is_national = 1
                    break
                    
            is_night = 1 if game.get('dayNight', '') == 'night' else 0
                    
            live_games.append({
                'id': game_pk,
                'matchup': f"{away_team} @ {home_team}",
                'start_time': game['gameDate'],
                'is_national_tv': is_national,
                'is_night_game': is_night
            })

    return live_games


def get_live_game_state(game_pk, is_national_tv=0, is_night_game=0):
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

    home_pitchers_used = len(boxscore.get('teams', {}).get('home', {}).get('pitchers', []))
    away_pitchers_used = len(boxscore.get('teams', {}).get('away', {}).get('pitchers', []))
    
    is_home_pitching = linescore.get('inningHalf', 'Top') == 'Top'
    
    home_starting_pitcher = 1 if home_pitchers_used <= 1 else 0
    away_starting_pitcher = 1 if away_pitchers_used <= 1 else 0

    is_dome = 1 if weather_info.get('condition', '') == 'Dome' else 0

    # New Context Features
    home_div = response.get('gameData', {}).get('teams', {}).get('home', {}).get('division', {}).get('id')
    away_div = response.get('gameData', {}).get('teams', {}).get('away', {}).get('division', {}).get('id')
    is_rivalry = 1 if home_div == away_div and home_div is not None else 0

    home_pitches = boxscore.get('teams', {}).get('home', {}).get('teamStats', {}).get('pitching', {}).get('numberOfPitches', 0)
    away_pitches = boxscore.get('teams', {}).get('away', {}).get('teamStats', {}).get('pitching', {}).get('numberOfPitches', 0)
    total_pitch_count = int(home_pitches) + int(away_pitches)

    home_pa = boxscore.get('teams', {}).get('home', {}).get('teamStats', {}).get('batting', {}).get('plateAppearances', 0)
    away_pa = boxscore.get('teams', {}).get('away', {}).get('teamStats', {}).get('batting', {}).get('plateAppearances', 0)
    total_pa = int(home_pa) + int(away_pa)

    is_tied = 1 if run_diff == 0 else 0

    # Format exactly to our XGBoost features
    state = {
        'inning': int(inning), 'outs_when_up': int(outs), 'run_diff': int(run_diff),
        'is_home_leading': int(is_home_leading), 'is_tied': int(is_tied), 'on_1b': int(on_1b), 'on_2b': int(on_2b),
        'on_3b': int(on_3b), 'total_runs': int(total_runs), 
        'home_pitchers_used': int(home_pitchers_used), 'away_pitchers_used': int(away_pitchers_used),
        'home_starting_pitcher': int(home_starting_pitcher), 'away_starting_pitcher': int(away_starting_pitcher),
        'total_pitch_count': int(total_pitch_count),
        'total_pa': int(total_pa), 'is_dome': int(is_dome), 'is_national_tv': int(is_national_tv),
        'is_night_game': int(is_night_game), 'is_rivalry': int(is_rivalry)
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
            state_dict, summary = get_live_game_state(selected_game['id'], selected_game.get('is_national_tv', 0), selected_game.get('is_night_game', 0))
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

            # Calculate elapsed time from the game's scheduled start time
            # MLB API returns start_time in UTC (e.g., '2026-07-11T20:10:00Z')
            start_time_utc = datetime.strptime(
                selected_game['start_time'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            current_time_utc = datetime.now(timezone.utc)

            # Prevent negative elapsed time if the game hasn't reached its scheduled start yet
            minutes_elapsed = max(0, (current_time_utc - start_time_utc).total_seconds() / 60)
            mins_remaining = float(predicted_total_mins) - minutes_elapsed

            print("\n-------------------------------------")
            print(
                f"⏱️  Predicted Total Duration: {predicted_total_mins:.1f} minutes")
            if mins_remaining > 0:
                print(
                    f"⏳ Estimated Time Remaining: {mins_remaining:.1f} minutes")
                
                expected_end_time = datetime.now().astimezone() + timedelta(minutes=mins_remaining)
                end_time_str = expected_end_time.strftime("%I:%M %p %Z").lstrip("0")
                print(f"⏰ Expected End Time: {end_time_str}")
            else:
                print("⏳ The model predicts this game should be ending momentarily!")
            print("-------------------------------------\n")

        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"An error occurred pulling the live data: {e}")


if __name__ == "__main__":
    main()
