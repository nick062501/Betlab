import math
import numpy as np
import pandas as pd
import streamlit as st

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as nba_players
import nfl_data_py as nfl

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
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(page_title="BetLab", layout="wide")
st.title("📊 BetLab — NBA + NFL Prop Analyzer")

tab1, tab2 = st.tabs(["🏀 NBA Props", "🏈 NFL Props"])

# =============================
# NBA TAB
# =============================
with tab1:
    st.subheader("NBA Player Props")

    c1, c2, c3 = st.columns(3)

    with c1:
        player_name = st.text_input("Player (full name)", "Stephen Curry")
        season = st.text_input("Season (e.g. 2025-26)", "2025-26")
        market = st.selectbox("Market", ["PTS", "REB", "AST", "PRA"])

    with c2:
        prop_line = st.number_input("Prop Line", value=18.5, step=0.5)
        side = st.selectbox("Pick Side", ["Over", "Under"])
        games_used = st.slider("Recent games used", 5, 25, 10)

    with c3:
        odds_text = st.text_input("Odds (optional)", "-110")

    if st.button("Analyze NBA Prop"):
        player = nba_players.find_players_by_full_name(player_name)
        if not player:
            st.error("Player not found. Use full name.")
        else:
            pid = player[0]["id"]
            logs = playergamelog.PlayerGameLog(player_id=pid, season=season).get_data_frames()[0]

            if market == "PTS":
                series = logs["PTS"]
            elif market == "REB":
                series = logs["REB"]
            elif market == "AST":
                series = logs["AST"]
            else:
                series = logs["PTS"] + logs["REB"] + logs["AST"]

            recent = series.head(games_used)
            mean = round(recent.mean(), 2)
            stdev = round(recent.std() if recent.std() > 0 else 3.0, 2)

            p_under, p_over = normal_over_under(mean, stdev, prop_line)
            model_prob = p_under if side == "Under" else p_over

            st.metric("Projected", mean)
            st.metric("Std Dev", stdev)
            st.metric(f"% {side}", f"{model_prob*100:.1f}%")

            if odds_text:
                odds = int(odds_text)
                implied = american_to_prob(odds)
                edge = (model_prob - implied) * 100
                st.metric("Edge vs Odds", f"{edge:+.1f}%")

            st.dataframe(logs[["GAME_DATE", "MATCHUP", "PTS", "REB", "AST"]].head(10))

# =============================
# NFL TAB
# =============================
with tab2:
    st.subheader("NFL Player Props")

    c1, c2, c3 = st.columns(3)

    with c1:
        season = st.number_input("Season", value=2025)
        player_name = st.text_input("Player Name", "Josh Allen")
        market = st.selectbox("Market", ["Pass Yds", "Rush Yds", "Rec Yds"])

    with c2:
        prop_line = st.number_input("Prop Line", value=250.5)
        side = st.selectbox("Pick Side", ["Over", "Under"], key="nflside")
        games_used = st.slider("Recent games used", 5, 17, 8)

    with c3:
        odds_text = st.text_input("Odds", "-110", key="nflodds")

    if st.button("Analyze NFL Prop"):
        df = nfl.import_weekly_data([season])
        df = df[df["player_name"].str.contains(player_name, case=False, na=False)]

        if df.empty:
            st.error("Player not found.")
        else:
            if market == "Pass Yds":
                series = df["passing_yards"].fillna(0)
            elif market == "Rush Yds":
                series = df["rushing_yards"].fillna(0)
            else:
                series = df["receiving_yards"].fillna(0)

            recent = series.tail(games_used)
            mean = round(recent.mean(), 2)
            stdev = round(recent.std() if recent.std() > 0 else 10.0, 2)

            p_under, p_over = normal_over_under(mean, stdev, prop_line)
            model_prob = p_under if side == "Under" else p_over

            st.metric("Projected", mean)
            st.metric("Std Dev", stdev)
            st.metric(f"% {side}", f"{model_prob*100:.1f}%")

            odds = int(odds_text)
            implied = american_to_prob(odds)
            edge = (model_prob - implied) * 100
            st.metric("Edge vs Odds", f"{edge:+.1f}%")

            st.dataframe(df[["week", "opponent_team", "passing_yards", "rushing_yards", "receiving_yards"]].tail(10))
