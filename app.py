import math
import time
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm

# NBA
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as nba_players

# NFL
import nfl_data_py as nfl


st.set_page_config(page_title="BetLab (NBA+NFL)", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def american_to_implied_prob(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def normal_over_under(mean: float, stdev: float, line: float):
    stdev = max(0.75, float(stdev))
    z = (line - mean) / stdev
    p_under = float(norm.cdf(z))
    p_over = 1.0 - p_under
    return p_under, p_over, stdev

def safe_std(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(vals) < 6:
        return 5.0
    return float(np.std(vals, ddof=1))

def rolling_last_n(df: pd.DataFrame, n: int):
    return df.head(n).copy()

# -----------------------------
# NBA data
# -----------------------------
@st.cache_data(ttl=60*30)
def nba_find_player_id(name: str):
    allp = nba_players.find_players_by_full_name(name)
    if not allp:
        return None
    # take best match
    return allp[0]["id"]

@st.cache_data(ttl=60*30)
def nba_player_gamelogs(player_id: int, season: str):
    gl = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gl.get_data_frames()[0]
    # Most recent first already
    return df

def nba_prop_series(df, market: str):
    # df columns include: PTS, REB, AST, STL, BLK, TOV, FG3M...
    if market == "Points":
        return df["PTS"].astype(float)
    if market == "Rebounds":
        return df["REB"].astype(float)
    if market == "Assists":
        return df["AST"].astype(float)
    if market == "P+R":
        return (df["PTS"] + df["REB"]).astype(float)
    if market == "P+A":
        return (df["PTS"] + df["AST"]).astype(float)
    if market == "R+A":
        return (df["REB"] + df["AST"]).astype(float)
    if market == "PRA":
        return (df["PTS"] + df["REB"] + df["AST"]).astype(float)
    if market == "3PM":
        return df["FG3M"].astype(float)
    return df["PTS"].astype(float)

# -----------------------------
# NFL data
# -----------------------------
@st.cache_data(ttl=60*60*6)
def nfl_weekly(season: int):
    # weekly has player_name, recent stats, opponent_team, etc.
    df = nfl.import_weekly_data([season])
    return df

def nfl_player_rows(df, name: str):
    # loose match
    name_l = name.lower().strip()
    hits = df[df["player_name"].astype(str).str.lower().str.contains(name_l, na=False)]
    return hits

def nfl_prop_series(df, market: str):
    # common columns:
    # receiving_yards, receptions, rushing_yards, passing_yards, passing_tds, rushing_tds, receiving_tds
    m = market
    if m == "Pass Yds":
        return df["passing_yards"].fillna(0).astype(float)
    if m == "Rush Yds":
        return df["rushing_yards"].fillna(0).astype(float)
    if m == "Rec Yds":
        return df["receiving_yards"].fillna(0).astype(float)
    if m == "Receptions":
        return df["receptions"].fillna(0).astype(float)
    if m == "Pass TD":
        return df["passing_tds"].fillna(0).astype(float)
    if m == "Rush TD":
        return df["rushing_tds"].fillna(0).astype(float)
    if m == "Rec TD":
        return df["receiving_tds"].fillna(0).astype(float)
    return df["receiving_yards"].fillna(0).astype(float)

# -----------------------------
# UI
# -----------------------------
st.title("BetLab — NBA + NFL (Free Data)")

tab1, tab2 = st.tabs(["NBA Props", "NFL Props"])

with tab1:
    st.subheader("NBA Props → projection + % chance")
    colA, colB, colC = st.columns([1.2, 1, 1])

    with colA:
        player_name = st.text_input("Player full name (e.g., Stephen Curry)", "")
        season = st.text_input("Season (NBA format, e.g., 2025-26)", "2025-26")
        market = st.selectbox(
            "Market",
            ["Points","Rebounds","Assists","P+R","P+A","R+A","PRA","3PM"]
        )

    with colB:
        line = st.number_input("Line", value=18.5, step=0.5)
        side = st.selectbox("Pick", ["Under", "Over"])
        last_n = st.slider("Use last N games", 5, 25, 10)

    with colC:
        odds_in = st.text_input("Optional: American odds (e.g., -110 or +120)", "")
        st.caption("Odds are optional — model works without them.")

    if st.button("Analyze NBA Prop", type="primary"):
        pid = nba_find_player_id(player_name) if player_name else None
        if not pid:
            st.error("Couldn’t find that NBA player name. Try full name spelling.")
        else:
            df = nba_player_gamelogs(pid, season)
            if df.empty:
                st.error("No game logs found for that season.")
            else:
                series = nba_prop_series(df, market)
                recent = series.head(last_n)
                season_mean = float(series.mean())
                recent_mean = float(recent.mean())
                # weighted blend (season + lastN)
                mean = 0.65 * season_mean + 0.35 * recent_mean

                stdev = safe_std(series.head(25).tolist())  # stability on recent 25
                p_under, p_over, stdev = normal_over_under(mean, stdev, float(line))

                pick_p = p_under if side == "Under" else p_over

                # Confidence: lower variance & more stable recent results
                hit_rate = float((series.head(25) < line).mean()) if side=="Under" else float((series.head(25) > line).mean())
                conf = max(0.05, min(0.95, 0.55*(1 - stdev/25) + 0.45*hit_rate))

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Projected", f"{mean:.2f}")
                c2.metric("StDev (est.)", f"{stdev:.2f}")
                c3.metric(f"P({side})", f"{pick_p*100:.1f}%")
                c4.metric("Confidence", f"{conf*100:.0f}%")

                # Odds comparison (optional)
                if odds_in.strip():
                    try:
                        odds_val = int(odds_in.strip())
                        implied = american_to_implied_prob(odds_val)
                        edge = pick_p - implied
                        st.write(f"**Implied prob from odds:** {implied*100:.1f}%")
                        st.write(f"**Model edge:** {edge*100:+.1f}% (positive = model likes it)")
                    except:
                        st.warning("Couldn’t parse odds — use format like -110 or +120")

                # Show recent games
                show = df[["GAME_DATE","MATCHUP","MIN","PTS","REB","AST","FG3M"]].head(10)
                st.write("Recent games:")
                st.dataframe(show, use_container_width=True)

with tab2:
    st.subheader("NFL Props → projection + % chance")
    colA, colB, colC = st.columns([1.2, 1, 1])

    with colA:
        season = st.number_input("Season (NFL year, e.g., 2025)", value=2025, step=1)
        player_name = st.text_input("Player name (e.g., Josh Allen)", key="nflname")
        market = st.selectbox(
            "Market",
            ["Pass Yds","Rush Yds","Rec Yds","Receptions","Pass TD","Rush TD","Rec TD"]
        )

    with colB:
        line = st.number_input("Line", value=249.5, step=0.5, key="nflline")
        side = st.selectbox("Pick", ["Under", "Over"], key="nflside")
        last_n = st.slider("Use last N games", 3, 17, 8, key="nfln")

    with colC:
        odds_in = st.text_input("Optional: American odds (e.g., -110 or +120)", key="nflodds")
        st.caption("Uses weekly player data. If name matches multiple players, it uses top rows.")

    if st.button("Analyze NFL Prop", type="primary"):
        df = nfl_weekly(int(season))
        hits = nfl_player_rows(df, player_name) if player_name else pd.DataFrame()
        if hits.empty:
            st.error("No matching NFL player rows found. Try full name spelling.")
        else:
            # Sort most recent weeks first
            hits = hits.sort_values(["week"], ascending=False)
            series = nfl_prop_series(hits, market)
            recent = series.head(last_n)

            season_mean = float(series.mean())
            recent_mean = float(recent.mean())
            mean = 0.70 * season_mean + 0.30 * recent_mean

            stdev = safe_std(series.head(12).tolist())
            p_under, p_over, stdev = normal_over_under(mean, stdev, float(line))
            pick_p = p_under if side == "Under" else p_over

            hit_rate = float((series.head(12) < line).mean()) if side=="Under" else float((series.head(12) > line).mean())
            conf = max(0.05, min(0.95, 0.55*(1 - stdev/120) + 0.45*hit_rate))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Projected", f"{mean:.2f}")
            c2.metric("StDev (est.)", f"{stdev:.2f}")
            c3.metric(f"P({side})", f"{pick_p*100:.1f}%")
            c4.metric("Confidence", f"{conf*100:.0f}%")

            if odds_in.strip():
                try:
                    odds_val = int(odds_in.strip())
                    implied = american_to_implied_prob(odds_val)
                    edge = pick_p - implied
                    st.write(f"**Implied prob from odds:** {implied*100:.1f}%")
                    st.write(f"**Model edge:** {edge*100:+.1f}% (positive = model likes it)")
                except:
                    st.warning("Couldn’t parse odds — use format like -110 or +120")

            show_cols = [c for c in ["week","recent_team","opponent_team",
                                     "passing_yards","rushing_yards","receiving_yards",
                                     "receptions","passing_tds","rushing_tds","receiving_tds"] if c in hits.columns]
            st.write("Recent games:")
            st.dataframe(hits[show_cols].head(12), use_container_width=True)

st.caption("Note: This is a free-data model. It’s best used to estimate probability & compare to the lines/odds you see on FanDuel.")
