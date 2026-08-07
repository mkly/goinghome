import base64
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
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

# The tumbling baseball in the corner. Streamlit serves the frontend directory
# itself, so three.js and the .glb are plain relative files the browser caches
# normally. theme.css lifts the iframe out of the page flow and parks it.
_baseball = components.declare_component(
    "baseball_3d", path=str(APP_DIR / "frontend"))


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
    css = css.replace("{{FIELD_URI}}", field_uri)
    st.html(
        f"""
        <style>{css}</style>
        """
    )
    # Keyed so Streamlit reuses the same iframe across reruns instead of
    # remounting it - the ball keeps spinning while the live game data refreshes.
    #
    # `spin` is read by the component: whenever the value changes it flicks the
    # ball, so pulling up a new game sets it going. This runs before the
    # selectbox is even created, but the widget's key is already in session state
    # by then - Streamlit populates it from the click that triggered the rerun -
    # so the new matchup is readable here rather than a rerun late.
    _baseball(key="baseball_3d", spin=st.session_state.get("matchup") or "")


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

    home_team = game_data.get("teams", {}).get("home", {})
    away_team = game_data.get("teams", {}).get("away", {})

    def abbreviate(team):
        """Broadcast-style short code, falling back to the club name."""
        return team.get("abbreviation") or team.get("teamName", "")[:3].upper()

    scoreboard = {
        "inning": int(inning),
        "is_top": bool(linescore.get("isTopInning", True)),
        "outs": int(outs),
        "on_1b": on_1b,
        "on_2b": on_2b,
        "on_3b": on_3b,
        "away_abbr": abbreviate(away_team) or "AWAY",
        "home_abbr": abbreviate(home_team) or "HOME",
        "away_name": away_team.get("name", "Away"),
        "home_name": home_team.get("name", "Home"),
        "away_runs": int(away_score),
        "home_runs": int(home_score),
    }
    return state, scoreboard


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


def render_scorebug(sb):
    """Render the live game state the way a broadcast score bug would."""
    # Drawn in this order so the diamond reads top, left, right on screen.
    bases = "".join(
        '<span class="scorebug__base scorebug__base--{0}{1}"></span>'.format(
            corner, " is-on" if sb[key] else ""
        )
        for corner, key in (("second", "on_2b"), ("third", "on_3b"), ("first", "on_1b"))
    )
    outs = "".join(
        '<span class="scorebug__out{0}"></span>'.format(
            " is-out" if recorded < sb["outs"] else ""
        )
        for recorded in range(3)
    )

    # The arrow points up in the top of the inning, down in the bottom, and the
    # side that is up to bat is the one carrying the runs that can still change.
    half = "Top" if sb["is_top"] else "Bottom"
    arrow = "up" if sb["is_top"] else "down"
    away_batting = " is-batting" if sb["is_top"] else ""
    home_batting = "" if sb["is_top"] else " is-batting"

    runners = [
        corner
        for corner, key in (("first", "on_1b"), ("second", "on_2b"), ("third", "on_3b"))
        if sb[key]
    ]
    # The graphic carries all of this visually, so it is one image to a screen
    # reader with the whole state spelled out rather than a pile of empty spans.
    label = (
        f'{half} of the {sb["inning"]}, {sb["outs"]} out, '
        f'{"runners on " + ", ".join(runners) if runners else "bases empty"}. '
        f'{sb["away_name"]} {sb["away_runs"]}, {sb["home_name"]} {sb["home_runs"]}.'
    )

    st.markdown(
        f"""
        <div class="scorebug" role="img" aria-label="{escape(label)}">
            <div class="scorebug__teams">
                <div class="scorebug__row{away_batting}">
                    <span class="scorebug__abbr">{escape(sb["away_abbr"])}</span>
                    <span class="scorebug__runs">{sb["away_runs"]}</span>
                </div>
                <div class="scorebug__row{home_batting}">
                    <span class="scorebug__abbr">{escape(sb["home_abbr"])}</span>
                    <span class="scorebug__runs">{sb["home_runs"]}</span>
                </div>
            </div>
            <div class="scorebug__inning">
                <span class="scorebug__arrow scorebug__arrow--{arrow}"></span>
                <span class="scorebug__frame">{sb["inning"]}</span>
            </div>
            <div class="scorebug__diamond">{bases}</div>
            <div class="scorebug__outs">
                <span class="scorebug__outs-label">Out</span>
                <span class="scorebug__outs-dots">{outs}</span>
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
        # Kept for screen readers but hidden - the placeholder already says what
        # the control is for.
        "Live matchup",
        options=list(game_options.keys()),
        index=None,
        placeholder="Choose a game",
        label_visibility="collapsed",
        # Named so the baseball component can read the selection at the top of
        # the script, before this widget exists.
        key="matchup",
    )

    if selected_matchup:
        selected_game = game_options[selected_matchup]

        try:
            with st.spinner(f"Refreshing {selected_matchup}..."):
                state_dict, scoreboard = get_live_game_state(
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

        # The bug and the three metrics below it are self-describing, so they run
        # without section headings above them, and the spacing in theme.css is
        # what marks them as a group.
        render_scorebug(scoreboard)

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
