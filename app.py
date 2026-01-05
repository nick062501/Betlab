import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

from nba_api.stats.endpoints import (
    playergamelog,
    leaguedashteamstats,
    commonteamroster,
)
from nba_api.stats.static import players as nba_players
from nba_api.stats.static import teams as nba_teams

import nfl_data_py as nfl


# =========================
# Math helpers (NO scipy)
# =========================
def normal_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def normal_over_under(mean: float, stdev: float, line: float) -> Tuple[float, float]:
    stdev = float(max(1.0, stdev))
    z = (line - mean) / stdev
    p_under = float(normal_cdf(z))
    p_over = 1.0 - p_under
    return p_under, p_over


def american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def expected_value(prob_win: float, odds: int) -> float:
    """
    EV per $1 staked.
    Positive odds: win profit = odds/100
    Negative odds: win profit = 100/abs(odds)
    """
    if odds > 0:
        win_profit = odds / 100.0
    else:
        win_profit = 100.0 / abs(odds)
    loss = 1.0
    return prob_win * win_profit - (1 - prob_win) * loss


def kelly_fraction(prob_win: float, odds: int) -> float:
    """
    Kelly bet fraction for decimal b:
    f* = (bp - q) / b, where b = net odds (profit per $1)
    """
    if odds > 0:
        b = odds / 100.0
    else:
        b = 100.0 / abs(odds)
    p = prob_win
    q = 1 - p
    f = (b * p - q) / b
    return float(max(0.0, f))


# =========================
# Modifiers
# =========================
NBA_DEF_POS_MOD = {
    "Elite (Top 5)": 0.92,
    "Above Avg (6–10)": 0.96,
    "Average (11–20)": 1.00,
    "Below Avg (21–25)": 1.05,
    "Poor (26–30)": 1.10,
}

NBA_DEF_OVERALL_MOD = {
    "Elite (Top 5)": 0.96,
    "Above Avg (6–10)": 0.98,
    "Average (11–20)": 1.00,
    "Below Avg (21–25)": 1.02,
    "Poor (26–30)": 1.04,
}

NBA_PACE_MOD = {"Fast": 1.03, "Average": 1.00, "Slow": 0.97}
NBA_B2B_MOD = {"No": 1.00, "Yes (tired legs)": 0.98}
NBA_BLOWOUT_RISK_MOD = {"Low": 1.00, "Medium": 0.98, "High": 0.95}
MINUTES_VOLATILITY_STDEV_MOD = {"Low": 0.95, "Medium": 1.05, "High": 1.15}
USAGE_BUMP_MOD = {"None": 1.00, "Small (+2–4%)": 1.03, "Medium (+5–8%)": 1.06, "Large (+9%+)": 1.10}

NFL_DEF_MOD = {
    "Elite (Top 8)": 0.95,
    "Above Avg": 0.98,
    "Average": 1.00,
    "Below Avg": 1.03,
    "Poor (Bottom 8)": 1.06,
}
NFL_GAME_SCRIPT_MOD = {"Neutral": 1.00, "Likely leading (more rush)": 0.96, "Likely trailing (more pass)": 1.04}


# =========================
# Scoring / discipline
# =========================
def confidence_grade(prob: float, edge_pct: float, risk_level: str) -> Tuple[str, int]:
    base = (prob - 0.50) * 160.0
    edge_boost = edge_pct * 2.0
    risk_penalty = {"Low": 0.0, "Medium": 6.0, "High": 14.0}.get(risk_level, 6.0)
    score = int(clamp(50 + base + edge_boost - risk_penalty, 0, 100))
    if score >= 80:
        return "A", score
    if score >= 70:
        return "B", score
    if score >= 60:
        return "C", score
    return "D", score


def approved(prob: float, edge_pct: float, risk_level: str) -> bool:
    min_prob = 0.58
    min_edge = 4.0
    if risk_level == "High":
        min_prob = 0.60
        min_edge = 5.0
    return (prob >= min_prob) and (edge_pct >= min_edge)


# =========================
# Data pull
# =========================
@st.cache_data(ttl=60 * 30)
def nba_get_game_log(player_id: int, season: str) -> pd.DataFrame:
    return playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]


@st.cache_data(ttl=60 * 60)
def nba_team_context(season: str) -> pd.DataFrame:
    """
    Returns team context table with DEF_RATING and PACE ranks.
    If endpoint fails (rate limiting), return empty dataframe.
    """
    try:
        df = leaguedashteamstats.LeagueDashTeamStats(season=season).get_data_frames()[0]
        # Lower DEF_RATING = better defense
        df = df.copy()
        df["DEF_RANK"] = df["DEF_RATING"].rank(method="min", ascending=True).astype(int)
        df["PACE_RANK"] = df["PACE"].rank(method="min", ascending=False).astype(int)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 60)
def nba_roster(team_id: int, season: str) -> pd.DataFrame:
    try:
        return commonteamroster.CommonTeamRoster(team_id=team_id, season=season).get_data_frames()[0]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60 * 60)
def nfl_weekly(season: int) -> pd.DataFrame:
    return nfl.import_weekly_data([season])


def nba_series_from_market(df: pd.DataFrame, market: str) -> pd.Series:
    if market == "PTS":
        return df["PTS"]
    if market == "REB":
        return df["REB"]
    if market == "AST":
        return df["AST"]
    if market == "PRA":
        return df["PTS"] + df["REB"] + df["AST"]
    if market == "PR":
        return df["PTS"] + df["REB"]
    if market == "PA":
        return df["PTS"] + df["AST"]
    if market == "RA":
        return df["REB"] + df["AST"]
    raise ValueError("Unknown market")


def tier_from_rank(rank: int) -> str:
    if rank <= 5:
        return "Elite (Top 5)"
    if rank <= 10:
        return "Above Avg (6–10)"
    if rank <= 20:
        return "Average (11–20)"
    if rank <= 25:
        return "Below Avg (21–25)"
    return "Poor (26–30)"


def pace_bucket_from_rank(rank: int) -> str:
    # Pace_RANK: 1 = fastest
    if rank <= 10:
        return "Fast"
    if rank <= 20:
        return "Average"
    return "Slow"


# =========================
# App state
# =========================
st.set_page_config(page_title="BetLab v3", layout="wide")
st.title("📊 BetLab — NBA & NFL Props (v3)")
st.caption("Decision support. Your edge comes from disciplined filtering, not more bets.")

if "daily_card" not in st.session_state:
    st.session_state.daily_card = []

if "bankroll" not in st.session_state:
    st.session_state.bankroll = 150.0

if "kelly_cap" not in st.session_state:
    st.session_state.kelly_cap = 0.05  # cap at 5% bankroll by default


tab_nba, tab_nfl, tab_live, tab_slip, tab_card, tab_bank = st.tabs(
    ["🏀 NBA", "🏈 NFL", "⚡ Live (2H)", "🧾 Slip Analyzer", "🗂️ Daily Card", "💰 Bankroll"]
)


# =========================
# NBA TAB
# =========================
with tab_nba:
    st.subheader("NBA Prop Analyzer (Auto Opponent Context + Value + Approved Filter)")

    teams = nba_teams.get_teams()
    team_names = sorted([t["full_name"] for t in teams])
    team_name_to_id = {t["full_name"]: t["id"] for t in teams}

    col1, col2, col3 = st.columns(3)

    with col1:
        player_name = st.text_input("Player (full name)", value="Stephen Curry")
        season = st.text_input("Season (NBA API format)", value="2025-26")
        market = st.selectbox("Market", ["PTS", "REB", "AST", "PRA", "PR", "PA", "RA"])
        line = st.number_input("Prop line", value=18.5, step=0.5)
        pick_side = st.selectbox("Pick side", ["Over", "Under"])
        odds_str = st.text_input("Odds (American)", value="-110")

    with col2:
        opp_team = st.selectbox("Opponent team (auto context)", team_names)
        use_last_n = st.slider("Recent games window", 5, 25, 10)
        season_weight = st.slider("Weight: recent vs season (0=season, 100=recent)", 0, 100, 70)
        show_only_approved = st.checkbox("Show only Approved plays", value=False)

    with col3:
        st.markdown("**Context overrides (optional)**")
        b2b = st.selectbox("Back-to-back?", list(NBA_B2B_MOD.keys()))
        blowout = st.selectbox("Blowout risk", list(NBA_BLOWOUT_RISK_MOD.keys()))
        minutes_vol = st.selectbox("Minutes volatility", list(MINUTES_VOLATILITY_STDEV_MOD.keys()))
        usage_bump = st.selectbox("Usage bump (injuries/role)", list(USAGE_BUMP_MOD.keys()))
        teammate_outs = st.slider("Key teammates OUT (count)", 0, 4, 0)
        risk_level = st.selectbox("Overall risk level", ["Low", "Medium", "High"])

    # Auto opponent context
    ctx = nba_team_context(season)
    auto_def_overall = "Average (11–20)"
    auto_pace = "Average"
    if not ctx.empty:
        row = ctx[ctx["TEAM_NAME"].str.contains(opp_team.split()[-1], case=False, na=False)]
        # Fallback: match by exact TEAM_NAME if possible
        exact = ctx[ctx["TEAM_NAME"].str.lower() == opp_team.lower()]
        if not exact.empty:
            row = exact
        if not row.empty:
            r = row.iloc[0]
            auto_def_overall = tier_from_rank(int(r["DEF_RANK"]))
            auto_pace = pace_bucket_from_rank(int(r["PACE_RANK"]))

    st.markdown("---")
    st.markdown("**Opponent context (auto)**")
    cA, cB, cC = st.columns(3)
    with cA:
        def_overall = st.selectbox("Overall defense", list(NBA_DEF_OVERALL_MOD.keys()), index=list(NBA_DEF_OVERALL_MOD.keys()).index(auto_def_overall))
    with cB:
        pace = st.selectbox("Opponent pace", list(NBA_PACE_MOD.keys()), index=list(NBA_PACE_MOD.keys()).index(auto_pace))
    with cC:
        def_vs_pos = st.selectbox("Defense vs position (manual for now)", list(NBA_DEF_POS_MOD.keys()), index=list(NBA_DEF_POS_MOD.keys()).index("Average (11–20)"))

    st.markdown("---")
    st.markdown("**Minutes & usage intelligence**")
    auto_minutes = st.checkbox("Auto-fill expected minutes from recent games", value=True)
    expected_minutes = st.number_input("Expected minutes", value=34, min_value=0, max_value=48, step=1)
    add_to_card = st.checkbox("Add result to Daily Card after analyze", value=False)

    if st.button("Analyze NBA"):
        found = nba_players.find_players_by_full_name(player_name)
        if not found:
            st.error("Player not found. Use full name (example: 'Jalen Brunson').")
        else:
            player_id = found[0]["id"]
            df = nba_get_game_log(player_id, season)

            # Auto minutes
            if auto_minutes and "MIN" in df.columns:
                min_recent = pd.to_numeric(df["MIN"].head(use_last_n), errors="coerce").dropna()
                if len(min_recent):
                    expected_minutes = int(round(min_recent.mean()))

            s = nba_series_from_market(df, market).astype(float)

            recent = s.head(use_last_n)
            season_all = s

            w = season_weight / 100.0
            base_mean = float(w * recent.mean() + (1 - w) * season_all.mean())
            base_std = float(recent.std(ddof=1)) if recent.std(ddof=1) > 0 else float(season_all.std(ddof=1))
            if not np.isfinite(base_std) or base_std <= 0:
                base_std = 3.0

            # Minutes adjust relative to player recent mins
            if "MIN" in df.columns:
                min_recent = pd.to_numeric(df["MIN"].head(use_last_n), errors="coerce").dropna()
                recent_min_avg = float(min_recent.mean()) if len(min_recent) else float(expected_minutes)
            else:
                recent_min_avg = float(expected_minutes)

            minutes_mod = clamp(expected_minutes / max(1.0, recent_min_avg), 0.70, 1.25)

            mean_mod = (
                NBA_DEF_POS_MOD[def_vs_pos]
                * NBA_DEF_OVERALL_MOD[def_overall]
                * NBA_PACE_MOD[pace]
                * NBA_B2B_MOD[b2b]
                * NBA_BLOWOUT_RISK_MOD[blowout]
                * USAGE_BUMP_MOD[usage_bump]
                * minutes_mod
            )

            mean_mod *= (1.0 + clamp(0.03 * teammate_outs, 0.0, 0.10))

            adj_mean = base_mean * mean_mod
            adj_std = base_std * MINUTES_VOLATILITY_STDEV_MOD[minutes_vol]
            if risk_level == "High":
                adj_std *= 1.05

            p_under, p_over = normal_over_under(adj_mean, adj_std, line)
            prob = p_under if pick_side == "Under" else p_over

            # Odds-based edge/EV
            implied = None
            edge_pct = 0.0
            ev = None
            kelly = None
            try:
                odds = int(odds_str.strip())
                implied = american_to_prob(odds)
                edge_pct = (prob - implied) * 100.0
                ev = expected_value(prob, odds)
                kelly = kelly_fraction(prob, odds)
            except Exception:
                odds = None

            grade, score = confidence_grade(prob, edge_pct, risk_level)
            is_approved = approved(prob, edge_pct, risk_level) if odds is not None else (prob >= 0.60)

            if show_only_approved and not is_approved:
                st.warning("Not an Approved play under your filter.")
            else:
                a, b, c, d = st.columns(4)
                a.metric("Adj Projection", f"{adj_mean:.2f}")
                b.metric(f"% {pick_side}", f"{prob*100:.1f}%")
                c.metric("Confidence", f"{grade} ({score}/100)")
                d.metric("Approved?", "✅ YES" if is_approved else "❌ NO")

                details = {
                    "Base mean": round(base_mean, 2),
                    "Base stdev": round(base_std, 2),
                    "Adj stdev": round(adj_std, 2),
                    "Expected minutes": expected_minutes,
                    "Minutes mod": round(minutes_mod, 3),
                    "Total mean mod": round(mean_mod, 3),
                    "Opponent (auto) defense": def_overall,
                    "Opponent (auto) pace": pace,
                    "Defense vs position": def_vs_pos,
                }

                if implied is not None:
                    details["Odds"] = odds
                    details["Implied %"] = round(implied * 100.0, 1)
                    details["Edge %"] = round(edge_pct, 1)
                    details["EV per $1"] = round(ev, 3) if ev is not None else None
                    details["Kelly fraction"] = round(kelly, 3) if kelly is not None else None

                    # stake suggestion
                    cap = float(st.session_state.kelly_cap)
                    br = float(st.session_state.bankroll)
                    stake = br * min(kelly if kelly is not None else 0.0, cap)
                    details["Suggested stake (capped Kelly)"] = round(stake, 2)

                st.json(details)

                show_cols = [c for c in ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST"] if c in df.columns]
                st.caption("Recent log (most recent first)")
                st.dataframe(df[show_cols].head(12), use_container_width=True)

                if add_to_card:
                    st.session_state.daily_card.append({
                        "Sport": "NBA",
                        "Player": player_name,
                        "Market": market,
                        "Line": line,
                        "Side": pick_side,
                        "Odds": odds_str,
                        "Prob%": round(prob * 100.0, 1),
                        "Edge%": round(edge_pct, 1) if implied is not None else None,
                        "EV/$": round(ev, 3) if ev is not None else None,
                        "Confidence": f"{grade} ({score})",
                        "Approved": bool(is_approved),
                        "Opponent": opp_team,
                    })
                    st.success("Added to Daily Card ✅")


# =========================
# NFL TAB
# =========================
with tab_nfl:
    st.subheader("NFL Prop Analyzer (Value + Approved Filter)")

    col1, col2, col3 = st.columns(3)
    with col1:
        nfl_season = st.number_input("Season", value=2025, min_value=1999, max_value=2100, step=1)
        nfl_player = st.text_input("Player name", value="Josh Allen")
        nfl_market = st.selectbox("Market", ["Pass Yds", "Rush Yds", "Rec Yds"])
        nfl_line = st.number_input("Prop line", value=250.5, step=0.5)
        nfl_side = st.selectbox("Pick side", ["Over", "Under"])
        nfl_odds_str = st.text_input("Odds (American)", value="-110", key="nfl_odds")

    with col2:
        nfl_last_n = st.slider("Recent games window", 4, 17, 8)
        nfl_def = st.selectbox("Opponent defense tier (manual)", list(NFL_DEF_MOD.keys()))
        nfl_script = st.selectbox("Game script", list(NFL_GAME_SCRIPT_MOD.keys()))
        nfl_risk = st.selectbox("Risk level", ["Low", "Medium", "High"])

    with col3:
        nfl_show_only_approved = st.checkbox("Show only Approved plays", value=False, key="nfl_only_approved")
        nfl_add_to_card = st.checkbox("Add result to Daily Card after analyze", value=False, key="nfl_add_to_card")

    if st.button("Analyze NFL"):
        df = nfl_weekly(int(nfl_season))
        dfp = df[df["player_name"].str.contains(nfl_player, case=False, na=False)].copy()

        if dfp.empty:
            st.error("Player not found in weekly data.")
        else:
            if nfl_market == "Pass Yds":
                series = dfp["passing_yards"].fillna(0).astype(float)
            elif nfl_market == "Rush Yds":
                series = dfp["rushing_yards"].fillna(0).astype(float)
            else:
                series = dfp["receiving_yards"].fillna(0).astype(float)

            recent = series.tail(nfl_last_n)
            base_mean = float(recent.mean())
            base_std = float(recent.std(ddof=1)) if recent.std(ddof=1) > 0 else float(series.std(ddof=1))
            if not np.isfinite(base_std) or base_std <= 0:
                base_std = 15.0

            mean_mod = NFL_DEF_MOD[nfl_def] * NFL_GAME_SCRIPT_MOD[nfl_script]
            adj_mean = base_mean * mean_mod
            adj_std = base_std * (1.05 if nfl_risk == "High" else 1.00)

            p_under, p_over = normal_over_under(adj_mean, adj_std, float(nfl_line))
            prob = p_under if nfl_side == "Under" else p_over

            implied = None
            edge_pct = 0.0
            ev = None
            kelly = None
            try:
                odds = int(nfl_odds_str.strip())
                implied = american_to_prob(odds)
                edge_pct = (prob - implied) * 100.0
                ev = expected_value(prob, odds)
                kelly = kelly_fraction(prob, odds)
            except Exception:
                odds = None

            grade, score = confidence_grade(prob, edge_pct, nfl_risk)
            is_approved = approved(prob, edge_pct, nfl_risk) if odds is not None else (prob >= 0.60)

            if nfl_show_only_approved and not is_approved:
                st.warning("Not an Approved play under your filter.")
            else:
                a, b, c, d = st.columns(4)
                a.metric("Adj Projection", f"{adj_mean:.1f}")
                b.metric(f"% {nfl_side}", f"{prob*100:.1f}%")
                c.metric("Confidence", f"{grade} ({score}/100)")
                d.metric("Approved?", "✅ YES" if is_approved else "❌ NO")

                details = {
                    "Base mean": round(base_mean, 1),
                    "Base stdev": round(base_std, 1),
                    "Mean mod": round(mean_mod, 3),
                    "Adj stdev": round(adj_std, 1),
                }
                if implied is not None:
                    details["Odds"] = odds
                    details["Implied %"] = round(implied * 100.0, 1)
                    details["Edge %"] = round(edge_pct, 1)
                    details["EV per $1"] = round(ev, 3) if ev is not None else None
                    details["Kelly fraction"] = round(kelly, 3) if kelly is not None else None
                    cap = float(st.session_state.kelly_cap)
                    br = float(st.session_state.bankroll)
                    stake = br * min(kelly if kelly is not None else 0.0, cap)
                    details["Suggested stake (capped Kelly)"] = round(stake, 2)

                st.json(details)

                show_cols = [c for c in ["week", "opponent_team", "passing_yards", "rushing_yards", "receiving_yards"] if c in dfp.columns]
                st.dataframe(dfp[show_cols].tail(12), use_container_width=True)

                if nfl_add_to_card:
                    st.session_state.daily_card.append({
                        "Sport": "NFL",
                        "Player": nfl_player,
                        "Market": nfl_market,
                        "Line": nfl_line,
                        "Side": nfl_side,
                        "Odds": nfl_odds_str,
                        "Prob%": round(prob * 100.0, 1),
                        "Edge%": round(edge_pct, 1) if implied is not None else None,
                        "EV/$": round(ev, 3) if ev is not None else None,
                        "Confidence": f"{grade} ({score})",
                        "Approved": bool(is_approved),
                    })
                    st.success("Added to Daily Card ✅")


# =========================
# LIVE TAB (2H)
# =========================
with tab_live:
    st.subheader("Live Betting Mode (2H projection + value)")
    st.caption("Use halftime stats + regression + context. This is for LIVE props/2H lines.")

    col1, col2, col3 = st.columns(3)
    with col1:
        current_stat = st.number_input("Current stat (at halftime/now)", value=12.0, step=1.0)
        minutes_played = st.number_input("Minutes played so far", value=18.0, step=1.0)
        live_line = st.number_input("2H line", value=10.5, step=0.5)
        live_side = st.selectbox("Pick (2H)", ["Over", "Under"])
    with col2:
        live_odds_str = st.text_input("Odds (American)", value="-110", key="live_odds")
        regression = st.slider("Regression toward normal (0–50%)", 0, 50, 20)
        live_pace = st.selectbox("Pace", ["Fast", "Average", "Slow"])
    with col3:
        live_blowout = st.selectbox("Blowout risk", ["Low", "Medium", "High"])
        foul_risk = st.selectbox("Foul trouble / minutes risk", ["No", "Some risk"])
        live_risk = st.selectbox("Risk level", ["Low", "Medium", "High"], key="live_risk")

    if st.button("Analyze Live (2H)"):
        mins = max(1.0, float(minutes_played))
        rate = float(current_stat) / mins
        remaining = max(1.0, 24.0 - mins)
        raw_2h = rate * remaining

        reg = regression / 100.0
        reg_target = raw_2h * 0.90
        proj_2h = (1 - reg) * raw_2h + reg * reg_target

        pace_mod = NBA_PACE_MOD[live_pace]
        blow_mod = NBA_BLOWOUT_RISK_MOD[live_blowout]
        foul_mod = 0.94 if foul_risk == "Some risk" else 1.00
        proj_2h *= pace_mod * blow_mod * foul_mod

        stdev = max(2.0, proj_2h * 0.35)
        if live_risk == "High":
            stdev *= 1.10

        p_under, p_over = normal_over_under(proj_2h, stdev, float(live_line))
        prob = p_under if live_side == "Under" else p_over

        implied = None
        edge_pct = 0.0
        ev = None
        kelly = None
        try:
            odds = int(live_odds_str.strip())
            implied = american_to_prob(odds)
            edge_pct = (prob - implied) * 100.0
            ev = expected_value(prob, odds)
            kelly = kelly_fraction(prob, odds)
        except Exception:
            odds = None

        grade, score = confidence_grade(prob, edge_pct, live_risk)
        is_approved = approved(prob, edge_pct, live_risk) if odds is not None else (prob >= 0.60)

        a, b, c, d = st.columns(4)
        a.metric("2H Projection", f"{proj_2h:.2f}")
        b.metric(f"% {live_side}", f"{prob*100:.1f}%")
        c.metric("Confidence", f"{grade} ({score}/100)")
        d.metric("Approved?", "✅ YES" if is_approved else "❌ NO")

        out = {
            "Rate so far (/min)": round(rate, 3),
            "Remaining minutes": round(remaining, 1),
            "Model stdev": round(stdev, 2),
        }
        if implied is not None:
            out["Odds"] = odds
            out["Implied %"] = round(implied * 100.0, 1)
            out["Edge %"] = round(edge_pct, 1)
            out["EV per $1"] = round(ev, 3) if ev is not None else None
            out["Kelly fraction"] = round(kelly, 3) if kelly is not None else None
            cap = float(st.session_state.kelly_cap)
            br = float(st.session_state.bankroll)
            stake = br * min(kelly if kelly is not None else 0.0, cap)
            out["Suggested stake (capped Kelly)"] = round(stake, 2)

        st.json(out)

        if st.button("Add Live pick to Daily Card"):
            st.session_state.daily_card.append({
                "Sport": "LIVE",
                "Player": "Live",
                "Market": "2H",
                "Line": live_line,
                "Side": live_side,
                "Odds": live_odds_str,
                "Prob%": round(prob * 100.0, 1),
                "Edge%": round(edge_pct, 1) if implied is not None else None,
                "EV/$": round(ev, 3) if ev is not None else None,
                "Confidence": f"{grade} ({score})",
                "Approved": bool(is_approved),
            })
            st.success("Added ✅")


# =========================
# SLIP ANALYZER (batch)
# =========================
with tab_slip:
    st.subheader("Slip Analyzer (paste a list of bets → get best 2–3)")
    st.caption(
        "Paste CSV rows: Sport,NBA_Player,Season,Market,Line,Side,Odds,Opponent(optional)\n"
        "Example: NBA,Stephen Curry,2025-26,PTS,27.5,Over,-110,Boston Celtics"
    )

    default = "NBA,Stephen Curry,2025-26,PTS,18.5,Over,-110,Los Angeles Lakers"
    text = st.text_area("Paste slip rows (one per line)", value=default, height=140)

    only_approved = st.checkbox("Show only Approved", value=True)
    max_results = st.slider("How many results to show", 3, 20, 8)

    if st.button("Analyze Slip"):
        rows = [r.strip() for r in text.splitlines() if r.strip()]
        parsed = []
        for r in rows:
            parts = [p.strip() for p in r.split(",")]
            if len(parts) < 7:
                continue
            sport = parts[0].upper()
            player = parts[1]
            season = parts[2]
            market = parts[3]
            line = float(parts[4])
            side = parts[5]
            odds = int(parts[6])
            opp = parts[7] if len(parts) >= 8 else None
            parsed.append((sport, player, season, market, line, side, odds, opp))

        results = []
        for sport, player, season, market, line, side, odds, opp in parsed:
            try:
                if sport == "NBA":
                    found = nba_players.find_players_by_full_name(player)
                    if not found:
                        continue
                    pid = found[0]["id"]
                    df = nba_get_game_log(pid, season)
                    s = nba_series_from_market(df, market).astype(float)
                    recent = s.head(10)
                    base_mean = float(recent.mean())
                    base_std = float(recent.std(ddof=1)) if recent.std(ddof=1) > 0 else float(s.std(ddof=1))
                    if not np.isfinite(base_std) or base_std <= 0:
                        base_std = 3.0

                    # basic opponent auto context if provided
                    def_overall = "Average (11–20)"
                    pace = "Average"
                    if opp:
                        ctx = nba_team_context(season)
                        if not ctx.empty:
                            exact = ctx[ctx["TEAM_NAME"].str.lower() == opp.lower()]
                            row = exact if not exact.empty else ctx[ctx["TEAM_NAME"].str.contains(opp.split()[-1], case=False, na=False)]
                            if not row.empty:
                                r0 = row.iloc[0]
                                def_overall = tier_from_rank(int(r0["DEF_RANK"]))
                                pace = pace_bucket_from_rank(int(r0["PACE_RANK"]))

                    adj_mean = base_mean * NBA_DEF_OVERALL_MOD[def_overall] * NBA_PACE_MOD[pace]
                    adj_std = base_std

                    p_under, p_over = normal_over_under(adj_mean, adj_std, line)
                    prob = p_under if side.lower() == "under" else p_over

                else:
                    # NFL: lightweight batch (manual tiers not included in batch mode)
                    df = nfl_weekly(int(season)) if season.isdigit() else nfl_weekly(2025)
                    dfp = df[df["player_name"].str.contains(player, case=False, na=False)].copy()
                    if dfp.empty:
                        continue
                    if market.lower().startswith("pass"):
                        series = dfp["passing_yards"].fillna(0).astype(float)
                    elif market.lower().startswith("rush"):
                        series = dfp["rushing_yards"].fillna(0).astype(float)
                    else:
                        series = dfp["receiving_yards"].fillna(0).astype(float)
                    recent = series.tail(8)
                    base_mean = float(recent.mean())
                    base_std = float(recent.std(ddof=1)) if recent.std(ddof=1) > 0 else float(series.std(ddof=1))
                    if not np.isfinite(base_std) or base_std <= 0:
                        base_std = 15.0
                    p_under, p_over = normal_over_under(base_mean, base_std, line)
                    prob = p_under if side.lower() == "under" else p_over

                implied = american_to_prob(odds)
                edge_pct = (prob - implied) * 100.0
                ev = expected_value(prob, odds)
                kelly = kelly_fraction(prob, odds)

                grade, score = confidence_grade(prob, edge_pct, "Medium")
                is_ok = approved(prob, edge_pct, "Medium")

                results.append({
                    "Sport": sport,
                    "Player": player,
                    "Market": market,
                    "Line": line,
                    "Side": side,
                    "Odds": odds,
                    "Prob%": round(prob * 100.0, 1),
                    "Implied%": round(implied * 100.0, 1),
                    "Edge%": round(edge_pct, 1),
                    "EV/$": round(ev, 3),
                    "Kelly": round(kelly, 3),
                    "Confidence": f"{grade} ({score})",
                    "Approved": is_ok,
                    "Opponent": opp,
                })
            except Exception:
                continue

        if not results:
            st.warning("No valid rows analyzed. Check formatting.")
        else:
            df_res = pd.DataFrame(results).sort_values(by=["Approved", "EV/$", "Prob%"], ascending=[False, False, False])
            if only_approved:
                df_res = df_res[df_res["Approved"] == True]
            st.dataframe(df_res.head(max_results), use_container_width=True)

            st.markdown("**One-tap add top 3 to Daily Card**")
            if st.button("Add top picks to Daily Card"):
                add = df_res.head(3).to_dict("records")
                st.session_state.daily_card.extend(add)
                st.success("Added ✅")


# =========================
# DAILY CARD
# =========================
with tab_card:
    st.subheader("Daily Card (max 3 recommended)")
    if not st.session_state.daily_card:
        st.info("No picks saved yet.")
    else:
        df_card = pd.DataFrame(st.session_state.daily_card)

        # Prefer Approved
        if "Approved" in df_card.columns:
            df_card = df_card.sort_values(by=["Approved", "EV/$", "Prob%"], ascending=[False, False, False])

        st.dataframe(df_card, use_container_width=True)

        # Recommend top 3 approved
        st.markdown("---")
        st.markdown("### Recommended Card (Top 3 Approved)")
        if "Approved" in df_card.columns:
            top3 = df_card[df_card["Approved"] == True].head(3)
        else:
            top3 = df_card.head(3)
        st.dataframe(top3, use_container_width=True)

        # Download CSV
        csv = df_card.to_csv(index=False).encode("utf-8")
        st.download_button("Download Card CSV", csv, "daily_card.csv", "text/csv")

        # Clear
        if st.button("Clear Daily Card"):
            st.session_state.daily_card = []
            st.success("Cleared ✅")


# =========================
# BANKROLL TAB
# =========================
with tab_bank:
    st.subheader("Bankroll & Stake Rules (keeps you from blowing up)")

    st.session_state.bankroll = st.number_input("Bankroll ($)", value=float(st.session_state.bankroll), step=5.0)
    st.session_state.kelly_cap = st.slider("Kelly cap (max % bankroll per bet)", 0.01, 0.15, float(st.session_state.kelly_cap), step=0.01)

    st.markdown("**Simple default plan (recommended):**")
    st.write("- Singles only unless you have 2 Approved legs.")
    st.write("- 1–2 units per bet. (Unit = ~1–2% bankroll)")
    st.write("- Avoid High-risk plays unless edge is huge.")

    unit_pct = st.slider("Unit size (% bankroll)", 0.5, 3.0, 1.5, step=0.25)
    unit = float(st.session_state.bankroll) * (unit_pct / 100.0)
    st.metric("Your Unit Size ($)", f"{unit:.2f}")

    st.markdown("---")
    st.markdown("### Quick checklist before you bet")
    st.write("✅ Approved = YES")
    st.write("✅ Odds not worse than -125 (unless huge edge)")
    st.write("✅ Minutes volatility not High")
    st.write("✅ Blowout risk not High")
    st.write("✅ You’re not chasing losses")