import math
from datetime import date
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import (
    commonplayerinfo,
    playergamelog,
    leaguedashteamstats,
    scoreboardv2,
)

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="NBA Prop Research Tool", page_icon="🏀", layout="wide")
st.title("🏀 NBA Prop Research Tool")
st.caption("Auto-fill matchup context (opponent, DEF rank/tier, pace rank/tier) + player trend research. NBA-only.")

# -----------------------------
# Helpers (no SciPy)
# -----------------------------
DEF_TIER_ORDER = ["Elite (Top 5)", "Strong (6–10)", "Average (11–20)", "Weak (21–25)", "Bottom (26–30)"]
PACE_TIER_ORDER = ["Fast", "Average", "Slow"]
MARKETS = ["PTS", "REB", "AST", "3PM", "PR", "PA", "RA", "PRA"]

def normal_cdf(x: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

def prob_over(line: float, mu: float, sigma: float) -> float:
    return 1.0 - normal_cdf(line, mu, sigma)

def tier_from_def_rank(rank: Optional[int]) -> str:
    if rank is None:
        return "Average (11–20)"
    if rank <= 5:
        return "Elite (Top 5)"
    if rank <= 10:
        return "Strong (6–10)"
    if rank <= 20:
        return "Average (11–20)"
    if rank <= 25:
        return "Weak (21–25)"
    return "Bottom (26–30)"

def tier_from_pace_rank(rank: Optional[int]) -> str:
    # pace rank 1 = fastest
    if rank is None:
        return "Average"
    if rank <= 10:
        return "Fast"
    if rank <= 20:
        return "Average"
    return "Slow"

def find_player_id(name: str) -> Optional[int]:
    name = (name or "").strip().lower()
    if not name:
        return None

    allp = nba_players.get_players()
    # exact
    for p in allp:
        if p.get("full_name", "").lower() == name:
            return int(p["id"])

    # fallback contains
    matches = [p for p in allp if name in p.get("full_name", "").lower()]
    if matches:
        return int(matches[0]["id"])
    return None

@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def teams_maps():
    t = nba_teams.get_teams()
    id_to_name = {int(x["id"]): x["full_name"] for x in t}
    id_to_abbr = {int(x["id"]): x["abbreviation"] for x in t}
    abbr_to_id = {x["abbreviation"]: int(x["id"]) for x in t}
    return id_to_name, id_to_abbr, abbr_to_id

@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_team_context(season: str) -> pd.DataFrame:
    """
    Returns TEAM_ID indexed DF with DEF_RATING, DEF_RANK, PACE, PACE_RANK.
    """
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
        timeout=20,
    ).get_data_frames()[0]

    needed = ["TEAM_ID", "TEAM_NAME", "DEF_RATING", "PACE"]
    for c in needed:
        if c not in df.columns:
            return pd.DataFrame()

    out = df[needed].copy()
    out["TEAM_ID"] = out["TEAM_ID"].astype(int)
    out["DEF_RANK"] = out["DEF_RATING"].rank(method="min", ascending=True).astype(int)   # 1 best defense
    out["PACE_RANK"] = out["PACE"].rank(method="min", ascending=False).astype(int)      # 1 fastest pace
    return out.set_index("TEAM_ID")

@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_player_team_id(player_id: int) -> Optional[int]:
    """
    Uses CommonPlayerInfo to get current TEAM_ID.
    """
    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=20).get_data_frames()[0]
    if info is None or info.empty or "TEAM_ID" not in info.columns:
        return None
    try:
        team_id = int(info.iloc[0]["TEAM_ID"])
        return team_id if team_id != 0 else None
    except Exception:
        return None

@st.cache_data(ttl=10 * 60, show_spinner=False)
def get_scoreboard_games(game_date_mmddyyyy: str) -> pd.DataFrame:
    """
    ScoreboardV2 expects MM/DD/YYYY. Uses GameHeader (DF0).
    """
    games = scoreboardv2.ScoreboardV2(game_date=game_date_mmddyyyy, timeout=20).get_data_frames()[0]
    return games

@st.cache_data(ttl=20 * 60, show_spinner=False)
def get_player_logs(player_id: int, season: str) -> pd.DataFrame:
    df = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season",
        timeout=20,
    ).get_data_frames()[0].copy()
    if df.empty:
        return df
    # normalize numeric columns used
    for col in ["MIN", "PTS", "REB", "AST", "FG3M"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # sort newest first
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df = df.sort_values("GAME_DATE", ascending=False)
    return df

def market_series(df: pd.DataFrame, market: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if market == "PTS":
        return df["PTS"].dropna()
    if market == "REB":
        return df["REB"].dropna()
    if market == "AST":
        return df["AST"].dropna()
    if market == "3PM":
        return df["FG3M"].dropna()

    pts = df["PTS"].fillna(0)
    reb = df["REB"].fillna(0)
    ast = df["AST"].fillna(0)

    if market == "PR":
        return (pts + reb).astype(float)
    if market == "PA":
        return (pts + ast).astype(float)
    if market == "RA":
        return (reb + ast).astype(float)
    if market == "PRA":
        return (pts + reb + ast).astype(float)

    return pd.Series(dtype=float)

# -----------------------------
# Session state defaults (for autofill sync)
# -----------------------------
def ss_init(key, val):
    if key not in st.session_state:
        st.session_state[key] = val

ss_init("autofill_status", "")
ss_init("auto_team_id", None)
ss_init("auto_opp_id", None)
ss_init("auto_def_rank", None)
ss_init("auto_pace_rank", None)

# These are the UI-controlled fields we want autofill to update:
ss_init("team_override", "(auto)")
ss_init("opp_override", "(auto)")
ss_init("overall_def_tier", "Average (11–20)")
ss_init("opp_pace_tier", "Average")

# -----------------------------
# Top controls
# -----------------------------
id_to_name, id_to_abbr, _abbr_to_id = teams_maps()
team_names = sorted(list(id_to_name.values()))

colA, colB, colC = st.columns([2, 1, 1])
with colA:
    player_name = st.text_input("Player (full name)", value="Stephen Curry", key="player_name")
with colB:
    season = st.selectbox("Season", options=["2025-26", "2024-25", "2023-24"], index=0, key="season")
with colC:
    game_date = st.date_input("Game date", value=date.today(), key="game_date")

game_date_mmddyyyy = game_date.strftime("%m/%d/%Y")

# -----------------------------
# Auto-Fill section
# -----------------------------
st.subheader("⚡ Auto-Fill matchup context")
st.caption("Click Auto-Fill to populate opponent, opponent DEF rank/tier, and opponent pace rank/tier. You can override any field afterward.")

auto_btn = st.button("⚡ Auto-Fill", key="autofill_btn")

if auto_btn:
    pid = find_player_id(player_name)
    if pid is None:
        st.error("Could not find that player. Try exact full name (e.g., 'Jayson Tatum').")
    else:
        try:
            team_id = get_player_team_id(pid)
            if team_id is None:
                st.error("Could not detect player team (TEAM_ID).")
            else:
                games = get_scoreboard_games(game_date_mmddyyyy)
                if games is None or games.empty:
                    st.error(f"No scoreboard games returned for {game_date_mmddyyyy}.")
                else:
                    row = games[(games["HOME_TEAM_ID"] == team_id) | (games["VISITOR_TEAM_ID"] == team_id)]
                    if row.empty:
                        st.warning("That player’s team is not on the slate for the selected date.")
                    else:
                        r = row.iloc[0]
                        home_id = int(r["HOME_TEAM_ID"])
                        away_id = int(r["VISITOR_TEAM_ID"])
                        opp_id = away_id if home_id == team_id else home_id

                        # Pull opponent context
                        ctx = get_team_context(season)
                        def_rank = None
                        pace_rank = None
                        if ctx is not None and not ctx.empty and opp_id in ctx.index:
                            def_rank = int(ctx.loc[opp_id]["DEF_RANK"])
                            pace_rank = int(ctx.loc[opp_id]["PACE_RANK"])

                        # Update session state for UI widgets (this is what makes autofill "stick")
                        st.session_state["auto_team_id"] = team_id
                        st.session_state["auto_opp_id"] = opp_id
                        st.session_state["auto_def_rank"] = def_rank
                        st.session_state["auto_pace_rank"] = pace_rank

                        # Set overrides to specific teams (so the visible dropdown updates)
                        st.session_state["team_override"] = id_to_name.get(team_id, "(auto)")
                        st.session_state["opp_override"] = id_to_name.get(opp_id, "(auto)")

                        # Set tiers (so the visible dropdown updates)
                        st.session_state["overall_def_tier"] = tier_from_def_rank(def_rank)
                        st.session_state["opp_pace_tier"] = tier_from_pace_rank(pace_rank)

                        st.session_state["autofill_status"] = (
                            f"Auto-filled {id_to_name.get(team_id,'(team)')} vs "
                            f"{id_to_name.get(opp_id,'(opp)')} on {game_date_mmddyyyy} "
                            f"| Opp DEF rank: {def_rank if def_rank is not None else '—'} "
                            f"| Opp Pace rank: {pace_rank if pace_rank is not None else '—'}"
                        )

                        st.success("Auto-Fill complete.")
                        st.rerun()
        except Exception as e:
            st.error(f"Auto-Fill failed: {type(e).__name__}: {e}")

with st.expander("Auto-Fill status"):
    st.write(st.session_state.get("autofill_status", ""))
    if st.session_state.get("auto_def_rank") is not None:
        st.write(f"Opponent DEF rank: **{st.session_state['auto_def_rank']}/30**")
    if st.session_state.get("auto_pace_rank") is not None:
        st.write(f"Opponent Pace rank: **{st.session_state['auto_pace_rank']}/30**")

# -----------------------------
# Research / analysis section
# -----------------------------
st.subheader("📊 Research view")

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("### Inputs (overrideable)")
    market = st.selectbox("Market", MARKETS, index=0, key="market")
    line = st.number_input("Line (optional)", value=18.5, step=0.5, key="line")
    side = st.selectbox("Side", ["Over", "Under"], index=0, key="side")

    team_override = st.selectbox("Player team", ["(auto)"] + team_names, index=0,
                                 key="team_override")
    opp_override = st.selectbox("Opponent", ["(auto)"] + team_names, index=0,
                                key="opp_override")

    overall_def_tier = st.selectbox("Opponent DEF tier (overall)", DEF_TIER_ORDER,
                                    index=DEF_TIER_ORDER.index(st.session_state["overall_def_tier"])
                                    if st.session_state["overall_def_tier"] in DEF_TIER_ORDER else 2,
                                    key="overall_def_tier")

    opp_pace_tier = st.selectbox("Opponent pace tier", PACE_TIER_ORDER,
                                 index=PACE_TIER_ORDER.index(st.session_state["opp_pace_tier"])
                                 if st.session_state["opp_pace_tier"] in PACE_TIER_ORDER else 1,
                                 key="opp_pace_tier")

    st.markdown("### Trend settings")
    last_n = st.slider("Recent games (N)", min_value=5, max_value=30, value=15, step=1, key="last_n")
    w_recent = st.slider("Weight recent vs season (%)", min_value=0, max_value=100, value=70, step=5, key="w_recent")

    st.markdown("### Optional context knobs (manual)")
    expected_minutes = st.number_input("Expected minutes", min_value=0, max_value=48, value=34, step=1, key="exp_min")
    minutes_vol = st.selectbox("Minutes volatility", ["Low", "Medium", "High"], index=1, key="min_vol")
    usage_bump = st.selectbox("Usage bump", ["None", "Small", "Medium", "Large"], index=0, key="usage")
    b2b = st.selectbox("Back-to-back?", ["No", "Yes"], index=0, key="b2b")

    analyze_btn = st.button("Analyze", key="analyze_btn")

with right:
    st.markdown("### Output")

    if analyze_btn:
        pid = find_player_id(player_name)
        if pid is None:
            st.error("Player not found.")
        else:
            try:
                logs = get_player_logs(pid, season)
                if logs.empty:
                    st.error("No game logs returned for that player/season.")
                else:
                    s = market_series(logs, market).dropna()
                    if s.empty:
                        st.error("Could not compute that market from logs.")
                    else:
                        season_mean = float(s.mean())
                        season_std = float(s.std(ddof=1)) if len(s) >= 2 else 3.0

                        recent = s.head(min(last_n, len(s)))
                        recent_mean = float(recent.mean())
                        recent_std = float(recent.std(ddof=1)) if len(recent) >= 2 else season_std

                        w = float(w_recent) / 100.0
                        base_mu = (1 - w) * season_mean + w * recent_mean
                        base_sigma = max(1.0, (1 - w) * season_std + w * recent_std)

                        # Simple context adjustments (light touch)
                        # DEF tier moves mean slightly
                        def_adj_map = {
                            "Elite (Top 5)": -0.7,
                            "Strong (6–10)": -0.35,
                            "Average (11–20)": 0.0,
                            "Weak (21–25)": 0.35,
                            "Bottom (26–30)": 0.7
                        }
                        pace_adj_map = {"Fast": 0.45, "Average": 0.0, "Slow": -0.45}
                        usage_adj_map = {"None": 0.0, "Small": 0.3, "Medium": 0.7, "Large": 1.2}
                        vol_sigma_map = {"Low": 0.95, "Medium": 1.05, "High": 1.15}

                        mu = base_mu
                        mu += def_adj_map.get(overall_def_tier, 0.0)
                        mu += pace_adj_map.get(opp_pace_tier, 0.0)
                        mu += (expected_minutes - 34.0) * 0.08
                        mu += usage_adj_map.get(usage_bump, 0.0)
                        if b2b == "Yes":
                            mu -= 0.25

                        sigma = base_sigma * vol_sigma_map.get(minutes_vol, 1.05)

                        p_over = prob_over(float(line), mu, sigma)
                        p_pick = p_over if side == "Over" else (1.0 - p_over)

                        # Display summary
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Season avg", f"{season_mean:.2f}")
                        c2.metric(f"Last {len(recent)} avg", f"{recent_mean:.2f}")
                        c3.metric("Model mean", f"{mu:.2f}")

                        c4, c5, c6 = st.columns(3)
                        c4.metric("Season std", f"{season_std:.2f}")
                        c5.metric(f"Last {len(recent)} std", f"{recent_std:.2f}")
                        c6.metric("Model std", f"{sigma:.2f}")

                        st.markdown("### Probability (research)")
                        st.metric(f"P({side} {line:.1f})", f"{p_pick*100:.1f}%")

                        st.markdown("### Recent results (last 10)")
                        last10 = s.head(10).tolist()

                        def fmt_num(x):
                            try:
                                x = float(x)
                                return str(int(x)) if x.is_integer() else f"{x:.1f}"
                            except Exception:
                                return "—"

                        st.write(", ".join(fmt_num(x) for x in last10))

                        st.markdown("### Raw last 10 rows")
                        show_cols = ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST", "FG3M"]
                        existing = [c for c in show_cols if c in logs.columns]
                        st.dataframe(logs[existing].head(10), use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Analyze failed: {type(e).__name__}: {e}")