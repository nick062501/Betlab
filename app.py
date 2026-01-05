import math
import numpy as np
import pandas as pd
import streamlit as st

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as nba_players
import nfl_data_py as nfl
st.warning("DEBUG: App reloaded")
st.write("If you see this, this file is running")
# -----------------------------
# Defense vs Position modifiers
# -----------------------------
DEFENSE_MODIFIERS = {
    "Elite": 0.92,      # top 5 defense vs position
    "Above Avg": 0.96,  # top 6–10
    "Average": 1.00,    # 11–20
    "Below Avg": 1.05,  # 21–25
    "Poor": 1.10        # 26–30
}
# -----------------------------
# Math helpers (NO scipy)
# -----------------------------
def normal_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def normal_over_under(mean, stdev, line):
    stdev = max(1.0, stdev)
    z = (line - mean) / stdev
    p_under = normal_cdf(z)
    p_over = 1 - p_under
    return p_under, p_over

def american_to_prob(odds):
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

st.set_page_config(page_title="BetLab", layout="wide")
st.title("📊 BetLab — NBA & NFL Props")

tab1, tab2 = st.tabs(["🏀 NBA", "🏈 NFL"])

# ================= NBA =================
with tab1:
    st.subheader("NBA Prop Analyzer")

    player = st.text_input("Player", "Stephen Curry")
    season = st.text_input("Season", "2025-26")
    market = st.selectbox("Market", ["PTS", "REB", "AST", "PRA"])
line = st.number_input("Line", 18.5, step=0.5)
side = st.selectbox("Pick", ["Over", "Under"])

defense_tier = st.selectbox(
    "Opponent Defense vs Position",
    ["Elite", "Above Avg", "Average", "Below Avg", "Poor"]
)
    if st.button("Analyze NBA"):
        pid = nba_players.find_players_by_full_name(player)
        if not pid:
            st.error("Player not found")
        else:
            pid = pid[0]["id"]
            df = playergamelog.PlayerGameLog(player_id=pid, season=season).get_data_frames()[0]

            series = (
                df["PTS"] if market == "PTS" else
                df["REB"] if market == "REB" else
                df["AST"] if market == "AST" else
                df["PTS"] + df["REB"] + df["AST"]
            )

            base_mean = series.head(10).mean()
std = max(series.head(10).std(), 3.0)

def_mod = DEFENSE_MODIFIERS[defense_tier]
mean = base_mean * def_mod
            pu, po = normal_over_under(mean, std, line)
            prob = pu if side == "Under" else po

            st.metric("Projected", round(mean, 2))
            st.metric(f"% {side}", f"{prob*100:.1f}%")
            st.dataframe(df[["GAME_DATE","PTS","REB","AST"]].head(10))

# ================= NFL =================
with tab2:
    st.subheader("NFL Prop Analyzer")

    season = st.number_input("Season", 2025)
    player = st.text_input("Player Name", "Josh Allen")
    market = st.selectbox("Market", ["Pass Yds", "Rush Yds", "Rec Yds"])
    line = st.number_input("Line", 250.5)
    side = st.selectbox("Pick Side", ["Over", "Under"], key="nfl")

    if st.button("Analyze NFL"):
        df = nfl.import_weekly_data([season])
        df = df[df["player_name"].str.contains(player, case=False, na=False)]

        series = (
            df["passing_yards"].fillna(0) if market == "Pass Yds" else
            df["rushing_yards"].fillna(0) if market == "Rush Yds" else
            df["receiving_yards"].fillna(0)
        )

        mean = series.tail(8).mean()
        std = max(series.tail(8).std(), 10)

        pu, po = normal_over_under(mean, std, line)
        prob = pu if side == "Under" else po

        st.metric("Projected", round(mean,2))
        st.metric(f"% {side}", f"{prob*100:.1f}%")
