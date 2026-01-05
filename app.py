import streamlit as st
import pandas as pd
from datetime import date

# NBA API (safe usage)
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import scoreboardv2, leaguedashteamstats

st.set_page_config(page_title="BetLab NBA", layout="centered")

# -----------------------------
# UTILITIES
# -----------------------------

DEF_TIERS = {
    "Elite (Top 5)": (1, 5),
    "Above Avg (6–10)": (6, 10),
    "Average (11–20)": (11, 20),
    "Below Avg (21–25)": (21, 25),
    "Poor (26–30)": (26, 30),
}

PACE_TIERS = ["Slow", "Average", "Fast"]

def defense_rank_to_tier(rank):
    for tier, (low, high) in DEF_TIERS.items():
        if low <= rank <= high:
            return tier
    return "Average (11–20)"

def pace_value_to_tier(pace):
    if pace >= 102:
        return "Fast"
    if pace <= 97:
        return "Slow"
    return "Average"

# -----------------------------
# DATA FETCH (CACHED)
# -----------------------------

@st.cache_data(ttl=86400)
def get_team_defense_and_pace():
    df = leaguedashteamstats.LeagueDashTeamStats(
        season="2025-26",
        measure_type_detailed_defense="Defense"
    ).get_data_frames()[0]

    df = df[["TEAM_ID", "TEAM_NAME", "DEF_RATING", "PACE"]]
    df["DEF_RANK"] = df["DEF_RATING"].rank(method="min")
    return df

@st.cache_data(ttl=300)
def get_today_matchups():
    board = scoreboardv2.ScoreboardV2(
        game_date=date.today().strftime("%Y-%m-%d")
    ).get_data_frames()[0]

    matchups = {}
    for _, row in board.iterrows():
        matchups[row["HOME_TEAM_ID"]] = row["VISITOR_TEAM_ID"]
        matchups[row["VISITOR_TEAM_ID"]] = row["HOME_TEAM_ID"]
    return matchups

# -----------------------------
# UI
# -----------------------------

st.title("🏀 BetLab — NBA Only (Fast)")
st.caption("Auto-Fill syncs opponent, defense tier & pace. All fields overridable.")

player_name = st.text_input("Player (full name)", "Stephen Curry")

col1, col2 = st.columns(2)
market = col1.selectbox("Market", ["PTS", "REB", "AST", "PTS+REB", "PTS+AST"])
prop_line = col2.number_input("Prop line", value=18.5, step=0.5)

pick_side = st.selectbox("Pick side", ["Over", "Under"])
odds = st.number_input("Odds (American)", value=-110)

st.markdown("### ⚡ Auto-Fill")
auto_fill = st.button("Auto-Fill (Today)")

# DEFAULTS
opponent = "—"
def_tier = "Average (11–20)"
pace_tier = "Average"

# -----------------------------
# AUTO-FILL LOGIC
# -----------------------------

if auto_fill:
    try:
        player = players.find_players_by_full_name(player_name)[0]
        team_id = player["TEAM_ID"]

        team_map = {t["id"]: t["full_name"] for t in teams.get_teams()}
        opponent_map = get_today_matchups()

        if team_id not in opponent_map:
            st.warning("Player does not play today.")
        else:
            opp_id = opponent_map[team_id]
            opponent = team_map[opp_id]

            df = get_team_defense_and_pace()
            opp_row = df[df["TEAM_ID"] == opp_id].iloc[0]

            def_tier = defense_rank_to_tier(int(opp_row["DEF_RANK"]))
            pace_tier = pace_value_to_tier(float(opp_row["PACE"]))

            st.success(f"Auto-filled vs {opponent}")

    except Exception as e:
        st.error("Auto-fill failed. Player name may be incorrect.")

# -----------------------------
# CONTEXT FILTERS (OVERRIDABLE)
# -----------------------------

st.markdown("### 🛡 Opponent Context")

col3, col4 = st.columns(2)
defense_vs_position = col3.selectbox(
    "Overall Defense Tier", list(DEF_TIERS.keys()),
    index=list(DEF_TIERS.keys()).index(def_tier)
)

opponent_pace = col4.selectbox(
    "Opponent Pace", PACE_TIERS,
    index=PACE_TIERS.index(pace_tier)
)

st.markdown("### ⏱ Minutes & Usage")

expected_minutes = st.number_input("Expected minutes", value=34, min_value=10, max_value=45)
usage_bump = st.selectbox("Usage change (injuries/role)", ["None", "Small", "Moderate", "Large"])
risk_level = st.selectbox("Overall risk tolerance", ["Low", "Medium", "High"])

# -----------------------------
# SIMPLE MODEL OUTPUT
# -----------------------------

st.markdown("### 📊 Model Read")

score = 0

if defense_vs_position in ["Below Avg (21–25)", "Poor (26–30)"]:
    score += 1
if opponent_pace == "Fast":
    score += 1
if expected_minutes >= 34:
    score += 1
if usage_bump in ["Moderate", "Large"]:
    score += 1
if odds > -115:
    score += 1

confidence = min(90, 50 + score * 8)

st.metric("Estimated Hit Rate", f"{confidence}%")

if confidence >= 65:
    st.success("✅ APPROVED PLAY")
else:
    st.warning("⚠️ Not a top-tier edge")

st.caption("This tool is decision support — not a guarantee. Aim for discipline over volume.")