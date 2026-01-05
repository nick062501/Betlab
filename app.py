import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import streamlit as st

# nba_api (works on Streamlit Cloud if in requirements.txt)
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams
from nba_api.stats.endpoints import (
    playergamelog,
    commonplayerinfo,
    scoreboardv2,
    leaguedashteamstats,
)

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="BetLab — NBA & NFL Props (v2)", page_icon="📊", layout="centered")

# -----------------------------
# Helpers: keys / formatting
# -----------------------------
def K(prefix: str, name: str) -> str:
    return f"{prefix}__{name}"

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def american_to_implied_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None:
        return None
    try:
        odds = float(odds)
    except Exception:
        return None
    if odds == 0:
        return None
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)

def normal_cdf(z: float) -> float:
    # Standard normal CDF via erf (no SciPy needed)
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def prob_over_line(mu: float, sigma: float, line: float) -> float:
    # P(X > line) where X ~ Normal(mu, sigma)
    sigma = max(0.01, sigma)
    z = (line - mu) / sigma
    return 1.0 - normal_cdf(z)

def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

# -----------------------------
# NBA: team & matchup autofill
# -----------------------------
@st.cache_data(ttl=60 * 60)
def nba_id_to_name_map() -> Dict[int, str]:
    return {t["id"]: t["full_name"] for t in nba_teams.get_teams()}

@st.cache_data(ttl=60 * 60)
def nba_name_to_id_map() -> Dict[str, int]:
    return {t["full_name"]: t["id"] for t in nba_teams.get_teams()}

@st.cache_data(ttl=60 * 60)
def nba_get_player_id(full_name: str) -> Optional[int]:
    matches = nba_players.find_players_by_full_name(full_name.strip())
    if not matches:
        return None
    # best match first
    return matches[0]["id"]

@st.cache_data(ttl=60 * 60)
def nba_get_player_team_id(player_id: int) -> Optional[int]:
    df = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    if df.empty:
        return None
    tid = df.loc[0, "TEAM_ID"]
    if pd.isna(tid):
        return None
    return int(tid)

@st.cache_data(ttl=60 * 10)
def nba_today_scoreboard_df() -> pd.DataFrame:
    # Using local "today" — good enough for 99% usage.
    # If you want, you can add a date override in UI later.
    game_date = datetime.now().strftime("%m/%d/%Y")
    games = scoreboardv2.ScoreboardV2(game_date=game_date).get_data_frames()[0]
    return games

def nba_find_today_opponent_team_id(player_team_id: int) -> Optional[int]:
    games = nba_today_scoreboard_df()
    if games.empty:
        return None
    row = games[(games["HOME_TEAM_ID"] == player_team_id) | (games["VISITOR_TEAM_ID"] == player_team_id)]
    if row.empty:
        return None
    r = row.iloc[0]
    if int(r["HOME_TEAM_ID"]) == int(player_team_id):
        return int(r["VISITOR_TEAM_ID"])
    return int(r["HOME_TEAM_ID"])

# -----------------------------
# NBA: Team defense + pace (auto)
# -----------------------------
@st.cache_data(ttl=60 * 60)
def nba_team_defense_pace(season: str) -> pd.DataFrame:
    """
    Returns team stats with DEF_RATING and PACE and their ranks.
    Uses leaguedashteamstats (works reliably on nba_api).
    """
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    ).get_data_frames()[0]

    # Columns typically include: TEAM_ID, TEAM_NAME, DEF_RATING, PACE
    if "DEF_RATING" not in df.columns or "PACE" not in df.columns:
        return pd.DataFrame()

    out = df[["TEAM_ID", "TEAM_NAME", "DEF_RATING", "PACE"]].copy()

    # Lower DEF_RATING = better defense
    out["DEF_RANK"] = out["DEF_RATING"].rank(method="min", ascending=True).astype(int)
    # Higher pace = faster
    out["PACE_RANK"] = out["PACE"].rank(method="min", ascending=False).astype(int)
    out = out.sort_values("DEF_RANK")
    return out

def categorize_rank(rank: Optional[int], total: int = 30) -> str:
    if rank is None:
        return "Unknown"
    if rank <= 5:
        return "Elite (Top 5)"
    if rank <= 10:
        return "Strong (6-10)"
    if rank <= 20:
        return "Average (11-20)"
    return "Weak (21-30)"

# -----------------------------
# NBA: Player game logs + stats
# -----------------------------
NBA_MARKETS = {
    "PTS": ("PTS",),
    "REB": ("REB",),
    "AST": ("AST",),
    "3PM": ("FG3M",),
    "PRA": ("PTS", "REB", "AST"),
    "PR": ("PTS", "REB"),
    "PA": ("PTS", "AST"),
}

def compute_market_series(df: pd.DataFrame, market: str) -> pd.Series:
    cols = NBA_MARKETS.get(market, ("PTS",))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return pd.Series(dtype=float)
    s = df[list(cols)].sum(axis=1)
    return pd.to_numeric(s, errors="coerce").dropna()

@st.cache_data(ttl=60 * 20)
def nba_get_player_gamelog(player_id: int, season: str) -> pd.DataFrame:
    df = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    # Newest first typically
    return df

def hit_rate(series: pd.Series, line: float, side: str) -> float:
    if series.empty:
        return 0.0
    if side.lower() == "over":
        return float((series > line).mean())
    return float((series < line).mean())

# -----------------------------
# Heuristic adjustments (simple + controllable)
# -----------------------------
def defense_category_multiplier(cat: str, side: str) -> float:
    # cat describes opponent defense quality against your prop
    # Over is harder vs elite defense; Under is easier vs elite defense
    cat = cat.lower()
    side = side.lower()
    if side == "over":
        if "elite" in cat:
            return 0.94
        if "strong" in cat:
            return 0.97
        if "average" in cat:
            return 1.00
        if "weak" in cat:
            return 1.04
        return 1.00
    else:
        # for under picks, invert slightly
        if "elite" in cat:
            return 1.04
        if "strong" in cat:
            return 1.02
        if "average" in cat:
            return 1.00
        if "weak" in cat:
            return 0.96
        return 1.00

def pace_multiplier(pace_choice: str) -> float:
    p = pace_choice.lower()
    if "fast" in p:
        return 1.03
    if "slow" in p:
        return 0.97
    return 1.00

def blowout_minutes_multiplier(blowout_risk: str) -> float:
    r = blowout_risk.lower()
    if "high" in r:
        return 0.94
    if "medium" in r:
        return 0.97
    return 1.00

def b2b_multiplier(is_b2b: str) -> float:
    t = is_b2b.lower()
    if t.startswith("yes"):
        return 0.98
    return 1.00

def usage_bump_multiplier(bump: str) -> float:
    b = bump.lower()
    if "major" in b:
        return 1.07
    if "some" in b:
        return 1.03
    return 1.00

def minutes_sigma_multiplier(vol: str) -> float:
    v = vol.lower()
    if "high" in v:
        return 1.18
    if "medium" in v:
        return 1.08
    return 1.00

# -----------------------------
# Daily Card storage
# -----------------------------
if "daily_card" not in st.session_state:
    st.session_state.daily_card = []

def add_to_daily_card(entry: Dict[str, Any]):
    st.session_state.daily_card.append(entry)

# -----------------------------
# UI Header
# -----------------------------
st.title("📊 BetLab — NBA & NFL Props (v2)")
st.caption("Decision support tool. Betting involves risk. Focus on disciplined staking and only 2–3 high-quality bets/day.")

tabs = st.tabs(["🏀 NBA", "🏈 NFL", "⚡ Live (2H)", "🧾 Daily Card", "⚙️ Settings"])

# =========================================================
# NBA TAB
# =========================================================
with tabs[0]:
    prefix = "nba"

    st.subheader("NBA Prop Analyzer (Auto Opponent + Defense + Minutes/Usage + Approved Filter)")

    colA, colB = st.columns(2)
    with colA:
        player_name = st.text_input("Player (full name)", value="Stephen Curry", key=K(prefix, "player_name"))
        season = st.text_input("Season (NBA API format)", value="2025-26", key=K(prefix, "season"))
        market = st.selectbox("Market", list(NBA_MARKETS.keys()), index=0, key=K(prefix, "market"))

    with colB:
        line = st.number_input("Prop line", value=18.5, step=0.5, key=K(prefix, "line"))
        side = st.selectbox("Pick side", ["Over", "Under"], key=K(prefix, "side"))
        odds_str = st.text_input("Odds (optional, American)", value="-110", key=K(prefix, "odds_str"))
        odds = safe_float(odds_str, default=None)

    st.divider()

    # Controls
    games_used = st.slider("Games used (recent)", min_value=5, max_value=30, value=21, key=K(prefix, "games_used"))
    weight_recent = st.slider("Weight: recent vs season (0=season, 100=recent)", 0, 100, 70, key=K(prefix, "weight_recent"))
    show_only_approved = st.checkbox("Show only Approved plays", value=False, key=K(prefix, "show_only_approved"))

    st.markdown("### Opponent context (auto + manual override)")

    team_names = sorted(nba_name_to_id_map().keys())
    id_to_name = nba_id_to_name_map()

    auto_today = st.checkbox("Auto-fill opponent/team from TODAY's game (if found)", value=True, key=K(prefix, "auto_today"))

    # Optional manual overrides
    player_team_override = st.selectbox(
        "Player team (override, optional)",
        ["(auto)"] + team_names,
        key=K(prefix, "player_team_override"),
    )
    opp_team_override = st.selectbox(
        "Opponent team (override, optional)",
        ["(auto)"] + team_names,
        key=K(prefix, "opp_team_override"),
    )

    defense_vs_pos = st.selectbox(
        "Defense vs position (manual)",
        ["Elite (Top 5)", "Strong (6-10)", "Average (11-20)", "Weak (21-30)"],
        index=2,
        key=K(prefix, "def_vs_pos"),
    )
    overall_def = st.selectbox(
        "Overall defense (manual)",
        ["Elite (Top 5)", "Strong (6-10)", "Average (11-20)", "Weak (21-30)"],
        index=2,
        key=K(prefix, "overall_def"),
    )
    opp_pace_manual = st.selectbox(
        "Opponent pace (manual)",
        ["Fast", "Average", "Slow"],
        index=1,
        key=K(prefix, "opp_pace_manual"),
    )

    b2b = st.selectbox("Back-to-back?", ["No", "Yes (player team)"], index=0, key=K(prefix, "b2b"))
    blowout_risk = st.selectbox("Blowout risk", ["Low", "Medium", "High"], index=0, key=K(prefix, "blowout_risk"))

    st.markdown("### Minutes & usage intelligence")
    expected_minutes = st.number_input("Expected minutes", min_value=0, max_value=48, value=34, step=1, key=K(prefix, "exp_min"))
    minutes_vol = st.selectbox("Minutes volatility", ["Low", "Medium", "High"], index=0, key=K(prefix, "min_vol"))
    usage_bump = st.selectbox("Usage bump (injuries / role)", ["None", "Some", "Major"], index=0, key=K(prefix, "usage_bump"))

    st.markdown("### Injury / availability notes (practical)")
    teammates_out = st.slider("Key teammates OUT (count)", 0, 5, 0, key=K(prefix, "tm_out"))
    injury_notes = st.text_area("Notes (optional)", value="", height=80, key=K(prefix, "inj_notes"))
    injury_minutes_adjust = st.slider(
        "Minutes impact from injuries (manual)",
        min_value=-8, max_value=8, value=0, step=1,
        help="If you expect the player to play more/less because of injuries, adjust minutes here.",
        key=K(prefix, "inj_min_adj"),
    )

    add_to_card = st.checkbox("Add result to Daily Card after analyze", value=False, key=K(prefix, "add_to_card"))

    st.divider()

    analyze = st.button("Analyze NBA", key=K(prefix, "analyze_btn"))

    if analyze:
        with st.spinner("Pulling NBA data..."):
            pid = nba_get_player_id(player_name)
            if pid is None:
                st.error("Could not find that player name. Try full name spelling.")
                st.stop()

            # Autofill team/opponent
            player_team_id = nba_get_player_team_id(pid)
            auto_player_team_name = id_to_name.get(player_team_id) if player_team_id else None

            opp_team_id = None
            auto_opp_team_name = None
            if auto_today and player_team_id:
                opp_team_id = nba_find_today_opponent_team_id(player_team_id)
                auto_opp_team_name = id_to_name.get(opp_team_id) if opp_team_id else None

            # Apply overrides if chosen
            player_team_name = auto_player_team_name
            if player_team_override != "(auto)":
                player_team_name = player_team_override

            opp_team_name = auto_opp_team_name
            if opp_team_override != "(auto)":
                opp_team_name = opp_team_override

            # Auto ranks from NBA team stats if we can identify opponent
            team_stats = nba_team_defense_pace(season)
            opp_def_rank = None
            opp_pace_rank = None
            opp_def_rating = None
            opp_pace = None
            if not team_stats.empty and opp_team_name:
                row = team_stats[team_stats["TEAM_NAME"] == opp_team_name]
                if not row.empty:
                    r = row.iloc[0]
                    opp_def_rank = int(r["DEF_RANK"])
                    opp_pace_rank = int(r["PACE_RANK"])
                    opp_def_rating = float(r["DEF_RATING"])
                    opp_pace = float(r["PACE"])

            # Pull player logs
            logs = nba_get_player_gamelog(pid, season)
            if logs.empty:
                st.error("No game logs found for that player/season yet.")
                st.stop()

            series = compute_market_series(logs, market)
            if series.empty:
                st.error("Could not compute that market from game logs.")
                st.stop()

            # Recent / season windows
            recent = series.head(games_used)  # logs are newest-first
            season_all = series

            mu_recent = float(recent.mean())
            sd_recent = float(recent.std(ddof=1)) if len(recent) > 1 else float(recent.std(ddof=0))
            mu_season = float(season_all.mean())
            sd_season = float(season_all.std(ddof=1)) if len(season_all) > 1 else float(season_all.std(ddof=0))

            w = weight_recent / 100.0
            mu_blend = (w * mu_recent) + ((1 - w) * mu_season)

            # Base sigma blend (keep it honest)
            sd_blend = (w * sd_recent) + ((1 - w) * sd_season)
            sd_blend = max(sd_blend, 1.0)

            # Adjustments
            # Defense context: combine manual "vs position" and "overall"
            mult_def = defense_category_multiplier(defense_vs_pos, side) * defense_category_multiplier(overall_def, side)

            # Pace: prefer auto rank if available; else manual
            pace_choice = opp_pace_manual
            if opp_pace_rank is not None:
                if opp_pace_rank <= 10:
                    pace_choice = "Fast"
                elif opp_pace_rank <= 20:
                    pace_choice = "Average"
                else:
                    pace_choice = "Slow"
            mult_pace = pace_multiplier(pace_choice)

            mult_b2b = b2b_multiplier(b2b)
            mult_blowout = blowout_minutes_multiplier(blowout_risk)
            mult_usage = usage_bump_multiplier(usage_bump)

            # Minutes logic (simple)
            minutes_effect = (expected_minutes + injury_minutes_adjust) / 34.0
            minutes_effect = clamp(minutes_effect, 0.75, 1.25)

            # Teammates out nudges usage slightly (user controls via usage_bump too)
            tm_out_effect = 1.0 + (0.01 * teammates_out)  # small
            tm_out_effect = clamp(tm_out_effect, 1.0, 1.05)

            mu_adj = mu_blend * mult_def * mult_pace * mult_b2b * mult_usage * minutes_effect * tm_out_effect

            # Volatility adjustments (sigma up/down)
            sd_adj = sd_blend * minutes_sigma_multiplier(minutes_vol)

            # Probabilities
            p_over = prob_over_line(mu_adj, sd_adj, line)
            p_pick = p_over if side.lower() == "over" else (1.0 - p_over)

            # Empirical hit rates
            hr_recent = hit_rate(recent, line, side)
            hr_season = hit_rate(season_all, line, side)

            implied = american_to_implied_prob(odds) if odds is not None else None
            edge = (p_pick - implied) if (implied is not None) else None

            # Approval logic (conservative)
            approved = True
            reasons = []
            if len(recent) < 8:
                approved = False
                reasons.append("Too few recent games in sample.")
            if p_pick < 0.53:
                approved = False
                reasons.append("Model probability < 53%.")
            if implied is not None and edge is not None and edge < 0.02:
                approved = False
                reasons.append("Edge vs implied < 2%.")
            if blowout_risk.lower() == "high" and side.lower() == "over":
                approved = False
                reasons.append("High blowout risk hurts overs.")
            if minutes_vol.lower() == "high" and side.lower() == "over":
                reasons.append("High minutes volatility adds risk (watch).")

            # Show results
            st.markdown("## Results")
            top = st.columns(2)
            with top[0]:
                st.write(f"**Player:** {player_name}  |  **Season:** {season}")
                st.write(f"**Pick:** {side} {line} **{market}**")
                st.write(f"**Auto team:** {auto_player_team_name or 'Unknown'}")
                st.write(f"**Auto opponent today:** {auto_opp_team_name or 'Not found today'}")
                if player_team_name:
                    st.write(f"**Using player team:** {player_team_name}")
                if opp_team_name:
                    st.write(f"**Using opponent:** {opp_team_name}")

            with top[1]:
                if opp_def_rank is not None:
                    st.write(f"**Opponent DEF Rank (auto):** {opp_def_rank}/30 ({categorize_rank(opp_def_rank)})")
                    st.write(f"**Opponent DEF Rating:** {opp_def_rating:.2f}")
                else:
                    st.write("**Opponent DEF Rank (auto):** Not available")
                if opp_pace_rank is not None:
                    st.write(f"**Opponent Pace Rank (auto):** {opp_pace_rank}/30 ({pace_choice})")
                    st.write(f"**Opponent Pace:** {opp_pace:.2f}")
                else:
                    st.write(f"**Opponent Pace (manual):** {pace_choice}")

            st.divider()

            st.markdown("### Probability & edge")
            c1, c2, c3 = st.columns(3)
            c1.metric("Model Prob (your side)", f"{p_pick*100:.1f}%")
            c2.metric("Recent hit rate", f"{hr_recent*100:.1f}%")
            c3.metric("Season hit rate", f"{hr_season*100:.1f}%")

            if implied is not None:
                st.write(f"**Implied probability from odds ({odds_str}):** {implied*100:.1f}%")
            if edge is not None:
                st.write(f"**Edge (Model − Implied):** {edge*100:.1f}%")

            st.divider()

            st.markdown("### Expected output (blended + adjusted)")
            st.write(f"**Base mean (blend):** {mu_blend:.2f}")
            st.write(f"**Adjusted mean (def/pace/min/usage):** {mu_adj:.2f}")
            st.write(f"**Adjusted sigma (volatility):** {sd_adj:.2f}")

            st.divider()

            st.markdown("### Filters / context used")
            st.write(f"- Defense vs position (manual): **{defense_vs_pos}**")
            st.write(f"- Overall defense (manual): **{overall_def}**")
            st.write(f"- Pace used: **{pace_choice}**")
            st.write(f"- Back-to-back: **{b2b}**")
            st.write(f"- Blowout risk: **{blowout_risk}**")
            st.write(f"- Expected minutes: **{expected_minutes}** (inj adj {injury_minutes_adjust:+d})")
            st.write(f"- Minutes volatility: **{minutes_vol}**")
            st.write(f"- Usage bump: **{usage_bump}**")
            st.write(f"- Teammates out: **{teammates_out}**")
            if injury_notes.strip():
                st.write(f"- Injury notes: {injury_notes}")

            st.divider()

            st.markdown("### Approved?")
            if approved:
                st.success("✅ Approved play (based on your filters)")
            else:
                st.warning("⚠️ Not approved (based on your filters)")
                if reasons:
                    for r in reasons:
                        st.write(f"- {r}")

            if show_only_approved and not approved:
                st.info("You have 'Show only Approved plays' enabled — this play is not approved.")
                st.stop()

            if add_to_card:
                add_to_daily_card({
                    "League": "NBA",
                    "Player": player_name,
                    "Market": market,
                    "Side": side,
                    "Line": line,
                    "Odds": odds_str,
                    "Prob": round(p_pick * 100, 1),
                    "RecentHR": round(hr_recent * 100, 1),
                    "SeasonHR": round(hr_season * 100, 1),
                    "Opponent": opp_team_name or auto_opp_team_name or "",
                    "Notes": injury_notes[:140] if injury_notes else "",
                    "Approved": approved
                })
                st.success("Added to Daily Card ✅")

# =========================================================
# NFL TAB (lightweight but powerful, since auto schedule/roster varies)
# =========================================================
with tabs[1]:
    prefix = "nfl"
    st.subheader("NFL Prop Analyzer (Fast + Manual — built to avoid broken data feeds)")

    st.caption("NFL auto rosters/injury feeds are messy without paid providers. This is the best reliable setup: you enter line/averages and context, and it produces probabilities + approval.")

    col1, col2 = st.columns(2)
    with col1:
        nfl_player = st.text_input("Player", value="(type player)", key=K(prefix, "player"))
        nfl_market = st.selectbox("Market", ["Pass Yds", "Rush Yds", "Rec Yds", "Receptions", "Anytime TD (heuristic)"], key=K(prefix, "market"))
        nfl_side = st.selectbox("Pick side", ["Over", "Under"], key=K(prefix, "side"))
        nfl_line = st.number_input("Line", value=49.5, step=0.5, key=K(prefix, "line"))

    with col2:
        nfl_odds_str = st.text_input("Odds (optional, American)", value="-110", key=K(prefix, "odds_str"))
        nfl_odds = safe_float(nfl_odds_str, default=None)
        nfl_recent_n = st.slider("Games used (recent)", 3, 10, 6, key=K(prefix, "recent_n"))
        nfl_weight_recent = st.slider("Weight: recent vs season", 0, 100, 60, key=K(prefix, "w_recent"))

    st.markdown("### Inputs (paste your numbers)")
    c3, c4, c5 = st.columns(3)
    with c3:
        season_avg = st.number_input("Season avg", value=55.0, step=0.5, key=K(prefix, "season_avg"))
        season_sd = st.number_input("Season stdev (rough)", value=18.0, step=0.5, key=K(prefix, "season_sd"))
    with c4:
        recent_avg = st.number_input(f"Last {nfl_recent_n} avg", value=58.0, step=0.5, key=K(prefix, "recent_avg"))
        recent_sd = st.number_input(f"Last {nfl_recent_n} stdev (rough)", value=16.0, step=0.5, key=K(prefix, "recent_sd"))
    with c5:
        opp_def = st.selectbox("Opponent defense strength vs this market", ["Elite (Top 5)", "Strong (6-10)", "Average (11-20)", "Weak (21-30)"], index=2, key=K(prefix, "opp_def"))
        game_script = st.selectbox("Game script", ["Neutral", "Likely lead (more run)", "Likely trail (more pass)"], key=K(prefix, "script"))
        injury_role = st.selectbox("Role/Injury impact", ["None", "Some boost", "Major boost", "Some downgrade"], key=K(prefix, "inj_role"))

    st.markdown("### Optional notes")
    nfl_notes = st.text_area("Notes", value="", height=80, key=K(prefix, "notes"))

    if st.button("Analyze NFL", key=K(prefix, "analyze")):
        w = nfl_weight_recent / 100.0
        mu = (w * recent_avg) + ((1 - w) * season_avg)
        sd = (w * recent_sd) + ((1 - w) * season_sd)
        sd = max(sd, 1.0)

        # Adjustments
        mult_def = defense_category_multiplier(opp_def, nfl_side)
        mult_script = 1.0
        if "lead" in game_script.lower() and "pass" in nfl_market.lower():
            mult_script = 0.96
        if "trail" in game_script.lower() and "pass" in nfl_market.lower():
            mult_script = 1.04
        if "lead" in game_script.lower() and ("rush" in nfl_market.lower() or "td" in nfl_market.lower()):
            mult_script = 1.03

        mult_inj = 1.0
        if "some boost" in injury_role.lower():
            mult_inj = 1.03
        elif "major boost" in injury_role.lower():
            mult_inj = 1.07
        elif "downgrade" in injury_role.lower():
            mult_inj = 0.95

        mu_adj = mu * mult_def * mult_script * mult_inj

        p_over = prob_over_line(mu_adj, sd, nfl_line)
        p_pick = p_over if nfl_side.lower() == "over" else (1 - p_over)

        implied = american_to_implied_prob(nfl_odds) if nfl_odds is not None else None
        edge = (p_pick - implied) if implied is not None else None

        approved = True
        reasons = []
        if p_pick < 0.53:
            approved = False
            reasons.append("Model probability < 53%")
        if edge is not None and edge < 0.02:
            approved = False
            reasons.append("Edge vs implied < 2%")
        if sd > 28 and nfl_side.lower() == "over":
            reasons.append("High volatility risk (watch)")

        st.markdown("## Results")
        st.write(f"**{nfl_player} — {nfl_side} {nfl_line} {nfl_market}**")
        st.write(f"**Adj mean:** {mu_adj:.2f}  |  **Sigma:** {sd:.2f}")
        st.write(f"**Model prob:** {p_pick*100:.1f}%")
        if implied is not None:
            st.write(f"**Implied prob ({nfl_odds_str}):** {implied*100:.1f}%")
        if edge is not None:
            st.write(f"**Edge:** {edge*100:.1f}%")

        if approved:
            st.success("✅ Approved")
        else:
            st.warning("⚠️ Not approved")
            for r in reasons:
                st.write(f"- {r}")

        if nfl_notes.strip():
            st.write(f"**Notes:** {nfl_notes}")

        if st.checkbox("Add this to Daily Card", value=False, key=K(prefix, "add_card_after")):
            add_to_daily_card({
                "League": "NFL",
                "Player": nfl_player,
                "Market": nfl_market,
                "Side": nfl_side,
                "Line": nfl_line,
                "Odds": nfl_odds_str,
                "Prob": round(p_pick * 100, 1),
                "RecentHR": "",
                "SeasonHR": "",
                "Opponent": "",
                "Notes": nfl_notes[:140] if nfl_notes else "",
                "Approved": approved
            })
            st.success("Added to Daily Card ✅")

# =========================================================
# LIVE (2H) TAB
# =========================================================
with tabs[2]:
    prefix = "live"
    st.subheader("Live (2H) — Quick Decision Helper")

    st.caption("This doesn’t scrape live scores. You paste halftime stats/score and it tells you what’s needed to hit props.")

    col1, col2 = st.columns(2)
    with col1:
        live_player = st.text_input("Player", value="(player)", key=K(prefix, "player"))
        live_market = st.selectbox("Market", ["PTS", "REB", "AST", "3PM", "PRA", "PR", "PA"], key=K(prefix, "market"))
        live_line = st.number_input("Full game line", value=24.5, step=0.5, key=K(prefix, "line"))
        live_side = st.selectbox("Pick side", ["Over", "Under"], key=K(prefix, "side"))
    with col2:
        halftime_value = st.number_input("Player stat at halftime", value=12.0, step=0.5, key=K(prefix, "ht_val"))
        minutes_played = st.number_input("Minutes played at halftime", value=17.0, step=1.0, key=K(prefix, "ht_min"))
        exp_total_minutes = st.number_input("Expected total minutes", value=34.0, step=1.0, key=K(prefix, "exp_total_min"))
        live_odds = st.text_input("Live odds (optional)", value="", key=K(prefix, "odds"))

    if st.button("Compute 2H needs", key=K(prefix, "compute")):
        remaining = max(0.0, live_line - halftime_value)
        if live_side.lower() == "under":
            # Under: how much room left
            remaining = max(0.0, live_line - halftime_value)

        minutes_left = max(0.0, exp_total_minutes - minutes_played)

        st.markdown("## 2H Snapshot")
        if live_side.lower() == "over":
            st.write(f"Needs **{remaining:.1f}** more in 2H to clear **{live_line}**.")
        else:
            st.write(f"Has **{(live_line - halftime_value):.1f}** 'room' left to stay under **{live_line}**.")

        if minutes_left > 0:
            needed_per_min = remaining / minutes_left if live_side.lower() == "over" else 0.0
            st.write(f"Estimated minutes left: **{minutes_left:.1f}**")
            if live_side.lower() == "over":
                st.write(f"Needs about **{needed_per_min:.3f} per minute** for the rest of the game.")
        st.info("Tip: if foul trouble / blowout risk changes minutes, adjust expected total minutes and rerun.")

# =========================================================
# DAILY CARD TAB
# =========================================================
with tabs[3]:
    st.subheader("Daily Card")
    if not st.session_state.daily_card:
        st.info("No picks saved yet. Use 'Add to Daily Card' in NBA/NFL tabs.")
    else:
        df = pd.DataFrame(st.session_state.daily_card)
        st.dataframe(df, use_container_width=True)

        st.markdown("### Quick view (Approved first)")
        try:
            df2 = df.copy()
            df2["ApprovedSort"] = df2["Approved"].astype(int)
            df2 = df2.sort_values(["ApprovedSort", "Prob"], ascending=False).drop(columns=["ApprovedSort"])
            st.dataframe(df2, use_container_width=True)
        except Exception:
            pass

        if st.button("Clear Daily Card", key="clear_daily_card"):
            st.session_state.daily_card = []
            st.success("Cleared ✅")

# =========================================================
# SETTINGS TAB
# =========================================================
with tabs[4]:
    st.subheader("Settings / Notes")
    st.write("✅ This app is designed to run reliably on Streamlit Cloud (no SciPy, no fragile scraping).")
    st.write("If NBA auto opponent doesn’t find a game, it’s usually because:")
    st.write("- no NBA games today, or")
    st.write("- the game date is in a different timezone window (late night).")
    st.write("")
    st.write("If you want, I can add a manual 'date override' for the NBA scoreboard so you can set the date in-app.")