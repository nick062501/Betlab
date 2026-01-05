import math
import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from nba_api.stats.endpoints import (
    scoreboardv2,
    commonallplayers,
    commonplayerinfo,
    playergamelog,
    leaguedashteamstats,
)

# -----------------------------
# Helpers (NO SciPy)
# -----------------------------
def normal_cdf(x: float, mu: float, sigma: float) -> float:
    """CDF of Normal(mu, sigma) using erf (no scipy)."""
    if sigma <= 1e-9:
        return 1.0 if x >= mu else 0.0
    z = (x - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))

def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds == 0:
        return 0.0
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="BetLab — NBA Only (Fast)", layout="centered")

st.title("🏀 BetLab — NBA Only (Fast)")
st.caption("NBA-only build for speed + stability. Auto-Fill syncs opponent + DEF tier + Pace tier. Everything remains overrideable.")

# -----------------------------
# Caching NBA API calls
# -----------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def get_all_players_df() -> pd.DataFrame:
    df = commonallplayers.CommonAllPlayers(is_only_current_season=1).get_data_frames()[0]
    # Columns: PERSON_ID, DISPLAY_FIRST_LAST, ...
    df["NAME_LOWER"] = df["DISPLAY_FIRST_LAST"].str.lower()
    return df

@st.cache_data(ttl=60 * 15, show_spinner=False)
def get_player_info(player_id: int) -> dict:
    df = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return row  # includes TEAM_ID, TEAM_NAME, TEAM_ABBREVIATION, etc.

@st.cache_data(ttl=60 * 10, show_spinner=False)
def get_today_scoreboard(date_str: str) -> pd.DataFrame:
    # date_str format: MM/DD/YYYY
    sb = scoreboardv2.ScoreboardV2(game_date=date_str)
    games = sb.get_data_frames()[0]  # GameHeader
    return games

@st.cache_data(ttl=60 * 60, show_spinner=False)
def get_team_stats(season: str) -> pd.DataFrame:
    # Team stats includes: DEF_RATING, PACE, NET_RATING, etc.
    ds = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season"
    )
    df = ds.get_data_frames()[0]
    # Rank DEF_RATING (lower is better defense)
    df["DEF_RANK"] = df["DEF_RATING"].rank(method="min", ascending=True).astype(int)
    # Rank PACE (higher is faster)
    df["PACE_RANK"] = df["PACE"].rank(method="min", ascending=False).astype(int)
    return df

@st.cache_data(ttl=60 * 15, show_spinner=False)
def get_player_gamelogs(player_id: int, season: str) -> pd.DataFrame:
    gl = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gl.get_data_frames()[0]
    # df includes: MIN, PTS, REB, AST, FG3M, STL, BLK, TOV, MATCHUP, GAME_DATE
    # Ensure numeric
    for c in ["MIN","PTS","REB","AST","FG3M","STL","BLK","TOV"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# -----------------------------
# Market math
# -----------------------------
MARKETS = {
    "PTS": lambda r: r["PTS"],
    "REB": lambda r: r["REB"],
    "AST": lambda r: r["AST"],
    "3PM": lambda r: r["FG3M"],
    "STL": lambda r: r["STL"],
    "BLK": lambda r: r["BLK"],
    "TOV": lambda r: r["TOV"],
    "PR":  lambda r: r["PTS"] + r["REB"],
    "PA":  lambda r: r["PTS"] + r["AST"],
    "RA":  lambda r: r["REB"] + r["AST"],
    "PRA": lambda r: r["PTS"] + r["REB"] + r["AST"],
}

# How strongly defense/pace should affect each market (0 to 1)
DEF_COEF = {
    "PTS": 0.55, "3PM": 0.45, "AST": 0.30, "REB": 0.15,
    "STL": 0.15, "BLK": 0.10, "TOV": 0.10,
    "PR": 0.45, "PA": 0.45, "RA": 0.20, "PRA": 0.42
}
PACE_COEF = {
    "PTS": 0.55, "3PM": 0.50, "AST": 0.40, "REB": 0.25,
    "STL": 0.15, "BLK": 0.10, "TOV": 0.20,
    "PR": 0.45, "PA": 0.45, "RA": 0.28, "PRA": 0.45
}

def tier_from_rank(rank: int) -> str:
    # 30 teams
    if rank <= 5: return "Elite (Top 5)"
    if rank <= 10: return "Strong (6–10)"
    if rank <= 20: return "Average (11–20)"
    if rank <= 25: return "Weak (21–25)"
    return "Very Weak (26–30)"

def pace_tier_from_rank(rank: int) -> str:
    # rank 1 = fastest
    if rank <= 5: return "Very Fast (Top 5)"
    if rank <= 10: return "Fast (6–10)"
    if rank <= 20: return "Medium (11–20)"
    if rank <= 25: return "Slow (21–25)"
    return "Very Slow (26–30)"

def blowout_risk_from_net(team_net: float, opp_net: float) -> str:
    diff = abs(team_net - opp_net)
    if diff >= 8: return "High"
    if diff >= 5: return "Medium"
    return "Low"

# -----------------------------
# Session State defaults
# -----------------------------
def ss_init(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

ss_init("daily_card", [])  # list of dict results

# -----------------------------
# UI
# -----------------------------
tab_nba, tab_card, tab_settings = st.tabs(["🏀 NBA", "🧾 Daily Card", "⚙️ Settings"])

with tab_settings:
    st.subheader("Settings (speed/stability)")
    st.write("These are safe defaults to keep Streamlit fast.")
    ss_init("api_timeout_note", True)
    st.caption("Tip: If Streamlit Cloud feels slow, avoid hammering Auto-Fill repeatedly — it caches results for you.")

with tab_card:
    st.subheader("Daily Card")
    if not st.session_state["daily_card"]:
        st.info("No saved plays yet. Analyze a prop and check 'Add result to Daily Card' before clicking Analyze.")
    else:
        df = pd.DataFrame(st.session_state["daily_card"])
        st.dataframe(df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear Daily Card", key="clear_card_btn"):
                st.session_state["daily_card"] = []
                st.rerun()
        with c2:
            st.download_button(
                "Download Daily Card (CSV)",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="betlab_daily_card.csv",
                mime="text/csv",
                key="dl_card_csv"
            )

with tab_nba:
    st.subheader("NBA Prop Analyzer (Auto-Fill syncs defense/pace into the dropdowns)")

    # Core inputs
    colA, colB = st.columns(2)
    with colA:
        player_name = st.text_input("Player (full name)", value="Stephen Curry", key="player_name_in")
        season = st.selectbox("Season (NBA API format)", ["2025-26", "2024-25", "2023-24"], index=0, key="season_in")
    with colB:
        market = st.selectbox("Market", list(MARKETS.keys()), index=0, key="market_in")
        pick_side = st.selectbox("Pick side", ["Over", "Under"], index=0, key="pick_side_in")

    prop_line = st.number_input("Prop line", value=18.5, step=0.5, key="prop_line_in")
    odds_str = st.text_input("Odds (optional, American)", value="-110", key="odds_in")

    # Recent/weight controls
    games_used = st.slider("Games used (recent)", min_value=5, max_value=30, value=15, step=1, key="recent_n_in")
    weight_recent = st.slider("Weight: recent vs season (0=season, 100=recent)", min_value=0, max_value=100, value=70, step=5, key="weight_recent_in")
    show_only_approved = st.checkbox("Show only Approved plays", value=False, key="approved_only_in")

    st.divider()

    # Opponent context (AUTO-FILL will set these, but you can override)
    st.markdown("### Opponent context (auto-fill + override)")
    ss_init("opp_team_id", None)
    ss_init("opp_team_name", "")
    ss_init("your_team_id", None)
    ss_init("your_team_name", "")
    ss_init("def_tier", "Average (11–20)")
    ss_init("pace_tier", "Medium (11–20)")
    ss_init("blowout_risk", "Low")

    def_tier_options = ["Elite (Top 5)", "Strong (6–10)", "Average (11–20)", "Weak (21–25)", "Very Weak (26–30)"]
    pace_tier_options = ["Very Fast (Top 5)", "Fast (6–10)", "Medium (11–20)", "Slow (21–25)", "Very Slow (26–30)"]
    blowout_options = ["Low", "Medium", "High"]

    c1, c2 = st.columns(2)
    with c1:
        def_tier = st.selectbox("Overall defense (tier)", def_tier_options, index=def_tier_options.index(st.session_state["def_tier"]), key="def_tier_sb")
        pace_tier = st.selectbox("Opponent pace (tier)", pace_tier_options, index=pace_tier_options.index(st.session_state["pace_tier"]), key="pace_tier_sb")
    with c2:
        blowout_risk = st.selectbox("Blowout risk", blowout_options, index=blowout_options.index(st.session_state["blowout_risk"]), key="blowout_sb")
        back_to_back = st.selectbox("Back-to-back?", ["No", "Yes"], index=0, key="b2b_sb")

    st.divider()

    # Minutes & usage
    st.markdown("### Minutes & usage (auto-fill + override)")
    ss_init("expected_minutes", 34)
    ss_init("minutes_volatility", "Low")
    ss_init("usage_bump", "None")
    ss_init("key_out_count", 0)
    ss_init("risk_level", "Low")

    c3, c4 = st.columns(2)
    with c3:
        expected_minutes = st.number_input("Expected minutes", min_value=0, max_value=48, value=int(st.session_state["expected_minutes"]), step=1, key="exp_min_in")
        minutes_volatility = st.selectbox("Minutes volatility", ["Low", "Medium", "High"], index=["Low","Medium","High"].index(st.session_state["minutes_volatility"]), key="min_vol_sb")
    with c4:
        usage_bump = st.selectbox("Usage bump (injuries/role)", ["None", "Small", "Medium", "Large"], index=["None","Small","Medium","Large"].index(st.session_state["usage_bump"]), key="usage_sb")
        key_out_count = st.slider("Key teammates OUT (count)", min_value=0, max_value=5, value=int(st.session_state["key_out_count"]), step=1, key="out_cnt_sl")

    risk_level = st.selectbox("Overall risk level", ["Low", "Medium", "High"], index=["Low","Medium","High"].index(st.session_state["risk_level"]), key="risk_sb")

    add_to_card = st.checkbox("Add result to Daily Card after analyze", value=False, key="add_card_cb")

    st.divider()

    # -----------------------------
    # AUTO-FILL Button
    # -----------------------------
    def run_autofill():
        name = (player_name or "").strip().lower()
        if not name:
            st.warning("Type a player name first.")
            return

        players = get_all_players_df()
        # Exact match first; fallback to contains
        exact = players[players["NAME_LOWER"] == name]
        if exact.empty:
            contains = players[players["NAME_LOWER"].str.contains(name, na=False)]
            if contains.empty:
                st.error("Could not find that player name. Try full name (e.g., 'Stephen Curry').")
                return
            pid = int(contains.iloc[0]["PERSON_ID"])
        else:
            pid = int(exact.iloc[0]["PERSON_ID"])

        info = get_player_info(pid)
        if not info or not info.get("TEAM_ID"):
            st.error("Could not load player team info.")
            return

        team_id = int(info["TEAM_ID"])
        team_name = str(info.get("TEAM_NAME", "")).strip()
        st.session_state["your_team_id"] = team_id
        st.session_state["your_team_name"] = team_name

        # Today date (US Eastern-ish; Streamlit cloud is UTC, but scoreboard uses date string)
        # We'll use local date to you; if it misses due to timezone, you can manually override.
        today = dt.datetime.now().date()
        date_str = today.strftime("%m/%d/%Y")

        games = get_today_scoreboard(date_str)
        if games.empty:
            st.warning(f"No games found for {date_str}. (If games are late, try tomorrow's date manually in code later.)")
            return

        # Find player's game
        # Scoreboard GameHeader uses HOME_TEAM_ID / VISITOR_TEAM_ID
        match = games[(games["HOME_TEAM_ID"] == team_id) | (games["VISITOR_TEAM_ID"] == team_id)]
        if match.empty:
            st.warning("Auto-Fill couldn't find this player's team on today's slate.")
            return

        g = match.iloc[0]
        home_id = int(g["HOME_TEAM_ID"])
        vis_id = int(g["VISITOR_TEAM_ID"])

        opp_id = vis_id if home_id == team_id else home_id
        st.session_state["opp_team_id"] = opp_id

        # Pull team stats (for def/pace tiers)
        ts = get_team_stats(season)
        your_row = ts[ts["TEAM_ID"] == team_id]
        opp_row = ts[ts["TEAM_ID"] == opp_id]

        if not your_row.empty:
            st.session_state["your_team_name"] = str(your_row.iloc[0]["TEAM_NAME"])
        if not opp_row.empty:
            st.session_state["opp_team_name"] = str(opp_row.iloc[0]["TEAM_NAME"])

            opp_def_rank = int(opp_row.iloc[0]["DEF_RANK"])
            opp_pace_rank = int(opp_row.iloc[0]["PACE_RANK"])
            st.session_state["def_tier"] = tier_from_rank(opp_def_rank)
            st.session_state["pace_tier"] = pace_tier_from_rank(opp_pace_rank)

            # Blowout risk from NET_RATING gap
            if not your_row.empty:
                team_net = float(your_row.iloc[0]["NET_RATING"])
                opp_net = float(opp_row.iloc[0]["NET_RATING"])
                st.session_state["blowout_risk"] = blowout_risk_from_net(team_net, opp_net)

        # Auto-fill expected minutes from recent logs
        logs = get_player_gamelogs(pid, season)
        if not logs.empty and "MIN" in logs.columns:
            recent_min = logs["MIN"].head(10).dropna()
            if len(recent_min) > 0:
                st.session_state["expected_minutes"] = int(round(float(recent_min.mean())))

        st.success(
            f"Auto-Fill set: {st.session_state.get('your_team_name','')} vs "
            f"{st.session_state.get('opp_team_name','(opponent)')} | "
            f"DEF: {st.session_state.get('def_tier')} | Pace: {st.session_state.get('pace_tier')}"
        )

    if st.button("⚡ Auto-Fill (Today)", key="autofill_btn"):
        with st.spinner("Auto-filling opponent + defense/pace tiers + minutes…"):
            run_autofill()
        st.rerun()

    # -----------------------------
    # Analyze
    # -----------------------------
    def analyze_prop():
        name = (player_name or "").strip().lower()
        players = get_all_players_df()

        exact = players[players["NAME_LOWER"] == name]
        if exact.empty:
            contains = players[players["NAME_LOWER"].str.contains(name, na=False)]
            if contains.empty:
                return None, "Player not found. Use full name."
            player_id = int(contains.iloc[0]["PERSON_ID"])
            display_name = str(contains.iloc[0]["DISPLAY_FIRST_LAST"])
        else:
            player_id = int(exact.iloc[0]["PERSON_ID"])
            display_name = str(exact.iloc[0]["DISPLAY_FIRST_LAST"])

        logs = get_player_gamelogs(player_id, season)
        if logs.empty:
            return None, "No game logs returned for that season."

        # Build stat series for selected market
        if market not in MARKETS:
            return None, "Invalid market."

        vals = logs.apply(MARKETS[market], axis=1).astype(float)
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()

        if len(vals) < 8:
            return None, "Not enough games to model (need ~8+)."

        season_mean = float(vals.mean())
        season_std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        recent_vals = vals.head(games_used)
        recent_mean = float(recent_vals.mean())
        recent_std = float(recent_vals.std(ddof=1)) if len(recent_vals) > 1 else season_std

        w = weight_recent / 100.0
        base_mu = (1 - w) * season_mean + w * recent_mean
        base_sigma = max(0.75, (1 - w) * season_std + w * recent_std)  # floor for stability

        # Minutes scale
        mins = logs["MIN"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        base_min = float(mins.mean()) if len(mins) else float(expected_minutes)
        min_scale = (float(expected_minutes) / base_min) if base_min > 1 else 1.0

        # Defense & pace adjustments (using tiers as proxies)
        # Convert tier to numeric multiplier around 1.0
        def_mult_map = {
            "Elite (Top 5)": 0.92,
            "Strong (6–10)": 0.96,
            "Average (11–20)": 1.00,
            "Weak (21–25)": 1.04,
            "Very Weak (26–30)": 1.08
        }
        pace_mult_map = {
            "Very Fast (Top 5)": 1.06,
            "Fast (6–10)": 1.03,
            "Medium (11–20)": 1.00,
            "Slow (21–25)": 0.97,
            "Very Slow (26–30)": 0.94
        }

        dcoef = DEF_COEF.get(market, 0.35)
        pcoef = PACE_COEF.get(market, 0.35)

        def_mult = 1.0 + (def_mult_map.get(def_tier, 1.0) - 1.0) * dcoef
        pace_mult = 1.0 + (pace_mult_map.get(pace_tier, 1.0) - 1.0) * pcoef

        # Usage bump (manual)
        usage_map = {"None": 1.00, "Small": 1.03, "Medium": 1.06, "Large": 1.10}
        usage_mult = usage_map.get(usage_bump, 1.00)

        # Back-to-back penalty (light)
        b2b_mult = 0.98 if back_to_back == "Yes" else 1.00

        # Blowout risk reduces minutes/ceiling a bit
        blow_map = {"Low": 1.00, "Medium": 0.985, "High": 0.97}
        blow_mult = blow_map.get(blowout_risk, 1.00)

        mu = base_mu * min_scale * def_mult * pace_mult * usage_mult * b2b_mult * blow_mult

        # Sigma scaled softly with minutes volatility / blowout
        vol_map = {"Low": 1.00, "Medium": 1.08, "High": 1.18}
        sigma = base_sigma * vol_map.get(minutes_volatility, 1.00) * (1.05 if blowout_risk == "High" else 1.00)

        line = float(prop_line)

        # Probability
        if pick_side == "Over":
            prob = 1.0 - normal_cdf(line, mu, sigma)
        else:
            prob = normal_cdf(line, mu, sigma)

        prob = max(0.0, min(1.0, prob))

        # Odds & “approved” logic
        try:
            odds_int = int(str(odds_str).strip())
        except Exception:
            odds_int = None

        implied = american_to_implied_prob(odds_int) if odds_int is not None else None
        edge = (prob - implied) if implied is not None else None

        # Conservative approval threshold if odds missing
        # For -110-ish, breakeven ~52.4%. We'll require 56% unless user wants looser.
        approved = (prob >= 0.56) if implied is None else (edge >= 0.03)

        # Build output dict
        out = {
            "Player": display_name,
            "Season": season,
            "Market": market,
            "Pick": pick_side,
            "Line": line,
            "Prob%": round(prob * 100, 1),
            "Approved": "✅" if approved else "—",
            "SeasonAvg": round(season_mean, 2),
            "RecentAvg": round(recent_mean, 2),
            "ModelMean": round(mu, 2),
            "ModelStd": round(sigma, 2),
            "ExpMin": int(expected_minutes),
            "DEF Tier": def_tier,
            "Pace Tier": pace_tier,
            "B2B": back_to_back,
            "Blowout": blowout_risk,
            "Odds": odds_int if odds_int is not None else "",
            "Implied%": round(implied * 100, 1) if implied is not None else "",
            "Edge%": round(edge * 100, 1) if edge is not None else "",
        }
        return out, None

    if st.button("Analyze NBA", key="analyze_btn"):
        with st.spinner("Crunching season + recent + variance + context…"):
            result, err = analyze_prop()

        if err:
            st.error(err)
        else:
            # If "approved only" is checked and it's not approved, hide it
            if show_only_approved and result["Approved"] != "✅":
                st.warning("Not Approved under your filter. (Toggle off 'Show only Approved plays' to view.)")
            else:
                st.success("Analysis complete ✅")
                st.dataframe(pd.DataFrame([result]), use_container_width=True)

                st.markdown("#### Quick read")
                st.write(
                    f"**{result['Player']}** — **{result['Pick']} {result['Market']} {result['Line']}** "
                    f"→ Model prob **{result['Prob%']}%** | Model mean **{result['ModelMean']}** "
                    f"(Season {result['SeasonAvg']} / Recent {result['RecentAvg']})"
                )

                if add_to_card:
                    st.session_state["daily_card"].append(result)
                    st.toast("Added to Daily Card ✅", icon="🧾")