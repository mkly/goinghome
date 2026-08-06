import base64
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from huggingface_hub import hf_hub_download
from xgboost import XGBRegressor

pd.options.mode.string_storage = "python"
try:
    pd.options.future.infer_string = False
except Exception:
    pass


# ==========================================
# 1. PAGE CONFIG & MODEL CACHING
# ==========================================
st.set_page_config(
    page_title="Going Home | MLB Duration Predictor", layout="centered")

APP_DIR = Path(__file__).resolve().parent
HF_REPO_ID = "mkly/mlb-game-duration-xgboost"
MLB_REQUEST_TIMEOUT = 10


@st.cache_resource
def load_model():
    """Load the XGBoost model from Hugging Face Hub, with a local fallback."""
    try:
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename="xgb_live_model.json")
        model = XGBRegressor()
        model.load_model(model_path)
        return model
    except Exception:
        try:
            model = XGBRegressor()
            model.load_model(APP_DIR / "xgb_live_model.json")
            return model
        except Exception:
            return None


def render_page_design():
    """Inject the local visual theme and decorative stadium scene."""
    assets_dir = APP_DIR / "assets"
    css = (assets_dir / "theme.css").read_text(encoding="utf-8")
    field_uri = base64.b64encode(
        (assets_dir / "citi-field-night.webp").read_bytes()
    ).decode()
    baseball_uri = base64.b64encode(
        (assets_dir / "baseball.webp").read_bytes()).decode()
    css = css.replace("{{FIELD_URI}}", field_uri).replace(
        "{{BASEBALL_URI}}", baseball_uri
    )
    st.html(
        f"""
        <style>{css}</style>
        """
    )


def fetch_mlb_json(url):
    """Fetch MLB data with a bounded timeout and normal HTTP error handling."""
    response = requests.get(url, timeout=MLB_REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


model = load_model()


# ==========================================
# 2. MLB API HELPER FUNCTIONS
# ==========================================
@st.cache_data(ttl=30, show_spinner=False)
def get_todays_games():
    """Fetch live MLB games happening today."""
    today = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={today}&hydrate=broadcasts"
    )
    payload = fetch_mlb_json(url)

    if payload.get("totalGames", 0) == 0 or not payload.get("dates"):
        return []

    live_games = []
    for game in payload["dates"][0].get("games", []):
        if game.get("status", {}).get("abstractGameState") != "Live":
            continue

        away_team = game["teams"]["away"]["team"]["name"]
        home_team = game["teams"]["home"]["team"]["name"]
        is_national = any(
            broadcast.get("isNational", False)
            and broadcast.get("type", "") == "TV"
            for broadcast in game.get("broadcasts", [])
        )

        live_games.append(
            {
                "id": game["gamePk"],
                "matchup": f"{away_team} @ {home_team}",
                "start_time": game["gameDate"],
                "is_national_tv": int(is_national),
                "is_night_game": int(game.get("dayNight", "") == "night"),
            }
        )

    return live_games


def get_live_game_state(game_pk, is_national_tv=0, is_night_game=0):
    """Pull the pitch-by-pitch state required by the XGBoost model."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    payload = fetch_mlb_json(url)

    live_data = payload.get("liveData", {})
    linescore = live_data.get("linescore", {})
    boxscore = live_data.get("boxscore", {})
    game_data = payload.get("gameData", {})
    weather_info = game_data.get("weather", {})

    inning = linescore.get("currentInning", 1)
    outs = linescore.get("outs", 0)
    home_score = linescore.get("teams", {}).get("home", {}).get("runs", 0)
    away_score = linescore.get("teams", {}).get("away", {}).get("runs", 0)
    run_diff = abs(home_score - away_score)
    is_home_leading = int(home_score > away_score)
    total_runs = home_score + away_score

    offense = linescore.get("offense", {})
    on_1b = int("first" in offense)
    on_2b = int("second" in offense)
    on_3b = int("third" in offense)

    home_boxscore = boxscore.get("teams", {}).get("home", {})
    away_boxscore = boxscore.get("teams", {}).get("away", {})
    home_pitchers_used = len(home_boxscore.get("pitchers", []))
    away_pitchers_used = len(away_boxscore.get("pitchers", []))

    home_division = (
        game_data.get("teams", {}).get(
            "home", {}).get("division", {}).get("id")
    )
    away_division = (
        game_data.get("teams", {}).get(
            "away", {}).get("division", {}).get("id")
    )

    home_pitches = (
        home_boxscore.get("teamStats", {})
        .get("pitching", {})
        .get("numberOfPitches", 0)
    )
    away_pitches = (
        away_boxscore.get("teamStats", {})
        .get("pitching", {})
        .get("numberOfPitches", 0)
    )
    home_plate_appearances = (
        home_boxscore.get("teamStats", {})
        .get("batting", {})
        .get("plateAppearances", 0)
    )
    away_plate_appearances = (
        away_boxscore.get("teamStats", {})
        .get("batting", {})
        .get("plateAppearances", 0)
    )

    state = {
        "inning": int(inning),
        "outs_when_up": int(outs),
        "run_diff": int(run_diff),
        "is_home_leading": is_home_leading,
        "is_tied": int(run_diff == 0),
        "on_1b": on_1b,
        "on_2b": on_2b,
        "on_3b": on_3b,
        "total_runs": int(total_runs),
        "home_pitchers_used": int(home_pitchers_used),
        "away_pitchers_used": int(away_pitchers_used),
        "home_starting_pitcher": int(home_pitchers_used <= 1),
        "away_starting_pitcher": int(away_pitchers_used <= 1),
        "total_pitch_count": int(home_pitches) + int(away_pitches),
        "total_pa": int(home_plate_appearances) + int(away_plate_appearances),
        "is_dome": int(weather_info.get("condition", "") == "Dome"),
        "is_national_tv": int(is_national_tv),
        "is_night_game": int(is_night_game),
        "is_rivalry": int(
            home_division == away_division and home_division is not None
        ),
    }

    inning_half = "Top" if linescore.get(
        "inningHalf", "Top") == "Top" else "Bottom"
    summary = (
        f"{inning_half} {inning} | {outs} outs | "
        f"Away {away_score} - Home {home_score}"
    )
    return state, summary


def get_user_timezone():
    """Resolve the user's browser timezone from Streamlit's native context."""
    client_timezone = st.context.timezone
    if not isinstance(client_timezone, str):
        return None

    try:
        return ZoneInfo(client_timezone)
    except Exception:
        return None


def show_empty_state():
    st.markdown(
        """
        <div class="empty-state" role="status">
            <span class="empty-state__indicator"></span>
            <div>
                <strong>No live games right now</strong>
                <p>Check back when today's schedule is underway.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_details(state):
    """Render model inputs as a readable, responsive definition list."""
    feature_items = []
    for key, value in state.items():
        label = escape(key.replace("_", " ").title())
        feature_items.append(
            '<div class="feature-grid__item">'
            f"<dt>{label}</dt>"
            f"<dd>{escape(str(value))}</dd>"
            "</div>"
        )

    st.markdown(
        f'<dl class="feature-grid">{"".join(feature_items)}</dl>',
        unsafe_allow_html=True,
    )


# ==========================================
# 3. STREAMLIT UI
# ==========================================
render_page_design()
user_tz = get_user_timezone()

st.markdown(
    """
    <section class="hero">
        <p class="hero__brand">Going Home</p>
        <h1>MLB Live Duration Predictor</h1>
        <p class="hero__lede">
            Choose a game in progress to estimate when the game will end.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error("The prediction model could not be loaded. Please try again shortly.")
    st.stop()

try:
    with st.spinner("Checking today's MLB schedule..."):
        live_games = get_todays_games()
except (requests.RequestException, ValueError, KeyError, TypeError):
    st.error("Live game data is temporarily unavailable. Please try again shortly.")
    st.stop()

if not live_games:
    show_empty_state()
else:
    game_options = {game["matchup"]: game for game in live_games}
    selected_matchup = st.selectbox(
        "Live matchup",
        options=list(game_options.keys()),
        index=None,
        placeholder="Choose a game",
    )

    if selected_matchup:
        selected_game = game_options[selected_matchup]

        try:
            with st.spinner(f"Refreshing {selected_matchup}..."):
                state_dict, summary = get_live_game_state(
                    selected_game["id"],
                    selected_game.get("is_national_tv", 0),
                    selected_game.get("is_night_game", 0),
                )

                live_df = pd.DataFrame([state_dict])
                if hasattr(model, "feature_names_in_"):
                    live_df = live_df.reindex(
                        columns=model.feature_names_in_, fill_value=0
                    )

                predicted_total_mins = float(model.predict(live_df)[0])
        except (requests.RequestException, ValueError, KeyError, TypeError):
            st.error(
                "This game's live data could not be refreshed. Please try again shortly."
            )
            st.stop()

        start_time_utc = datetime.strptime(
            selected_game["start_time"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        minutes_elapsed = max(
            0,
            (datetime.now(timezone.utc) - start_time_utc).total_seconds() / 60,
        )
        mins_remaining = predicted_total_mins - minutes_elapsed

        st.divider()
        st.markdown(
            '<p class="section-kicker">Current game state</p>',
            unsafe_allow_html=True,
        )
        st.subheader(summary)

        st.markdown(
            '<p class="section-kicker section-kicker--forecast">Model forecast</p>',
            unsafe_allow_html=True,
        )
        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            if mins_remaining > 0:
                display_timezone = user_tz or datetime.now().astimezone().tzinfo
                expected_end_time = datetime.now(display_timezone) + timedelta(
                    minutes=mins_remaining
                )
                end_time_str = expected_end_time.strftime(
                    "%I:%M %p %Z").lstrip("0")
                st.metric("Expected end time", end_time_str)
            else:
                st.metric("Expected end time", "Any minute now")

        with result_col2:
            st.metric("Projected duration", f"{predicted_total_mins:.1f} min")

        with result_col3:
            if mins_remaining > 0:
                st.metric("Time remaining", f"{mins_remaining:.1f} min")
            else:
                st.metric("Status", "Wrapping up")

        with st.expander("Model details"):
            render_model_details(state_dict)
