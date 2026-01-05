import math
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players as nba_players
import nfl_data_py as nfl


# =========================
# Core math helpers (NO scipy)
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


# =========================
# Modifiers & scoring
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

NBA_PACE_MOD = {
    "Fast": 1.03,
    "Average": 1.00,
    "Slow": 0.97,
}

NBA_B2B_MOD = {
    "No": 1.00,
    "Yes (tired legs)": 0.98,
}

NBA_BLOWOUT_RISK_MOD = {
    "Low": 1.00,
    "Medium": 0.98,
    "High": 0.95,
}

MINUTES_VOLATILITY_STDEV_MOD = {
    "Low": 0.95,
    "Medium": 1.05,
    "High": 1.15,
}

USAGE_BUMP_MOD = {
    "None": 1.00,
    "Small (+2–4% usage)": 1.03,
    "Medium (+5–8% usage)": 1.06,
    "Large (+9%+ usage)": 1.10,
}


NFL_DEF_MOD = {
    "Elite (Top 8)": 0.95,
    "Above Avg": 0.98,
    "Average": 1.00,
    "Below Avg": 1.03,
    "Poor (Bottom 8)": 1.06,
}

NFL_GAME_SCRIPT_MOD = {
    "Neutral": 1.00,
    "Likely leading (more rush)": 0.96,
    "Likely trailing (more pass)": 1.04,
}


def confidence_grade(prob: float, edge_pct: float, risk_level: str) -> Tuple[str, int]:
    """
    Returns (grade, score 0–100) based on win prob, edge vs implied, and risk.
    """
    base = (prob - 0.50) * 160.0  # 0.56 -> 9.6, 0.60 -> 16
    edge_boost = edge_pct * 2.0   # +5% edge -> +10
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
    # Your “2–3 bets/day” filter: only show plays that are meaningfully above coinflip.
    min_prob = 0.58
    min_edge = 4.0
    if risk_level == "High":
        min_prob = 0.60
        min_edge = 5.0
    return (prob >= min_prob) and (edge_pct >= min_edge)


# =========================
# Data pulling
# =========================
@st.cache_data(ttl=60 * 30)
def nba_get_game_log(player_id: int, season: str) -> pd.DataFrame:
    df = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
    return df


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


# =========================
# UI setup
# =========================
st.set_page_config(page_title="BetLab", layout="wide")
st.title("📊 BetLab — NBA & NFL Props (v2)")

st.caption(
    "This tool is for decision support. Sports betting involves risk. "
    "Aim for disciplined staking and only 2–3 high-quality bets/day."
)

if "daily_card" not in st.session_state:
    st.session_state.daily_card = []


tab_nba, tab_nfl, tab_live, tab_card = st.tabs(["🏀 NBA", "🏈 NFL", "⚡ Live (2H)", "🧾 Daily Card"])


# =========================
# NBA TAB
# =========================
with tab_nba:
    st.subheader("NBA Prop Analyzer (Defense + Minutes/Usage + Approved Filter)")

    col1, col2, col3 = st.columns(3)

    with col1:
        player_name = st.text_input("Player (full name)", value="Stephen Curry")
        season = st.text_input("Season (NBA API format)", value="2025-26")
        market = st.selectbox("Market", ["PTS", "REB", "AST", "PRA", "PR", "PA", "RA"])
        line = st.number_input("Prop line", value=18.5, step=0.5)
        pick_side = st.selectbox("Pick side", ["Over", "Under"])

    with col2:
        odds_str = st.text_input("Odds (optional, American)", value="-110")
        use_last_n = st.slider("Games used (recent)", 5, 25, 10)
        season_weight = st.slider("Weight: recent vs season (0=season, 100=recent)", 0, 100, 70)
        show_only_approved = st.checkbox("Show only Approved plays", value=False)

    with col3:
        st.markdown("**Opponent context (manual, fast)**")
        def_vs_pos = st.selectbox("Defense vs position", list(NBA_DEF_POS_MOD.keys()))
        def_overall = st.selectbox("Overall defense", list(NBA_DEF_OVERALL_MOD.keys()))
        pace = st.selectbox("Opponent pace", list(NBA_PACE_MOD.keys()))
        b2b = st.selectbox("Back-to-back?", list(NBA_B2B_MOD.keys()))
        blowout = st.selectbox("Blowout risk", list(NBA_BLOWOUT_RISK_MOD.keys()))

    st.markdown("---")
    st.markdown("**Minutes & usage intelligence**")
    c4, c5, c6 = st.columns(3)
    with c4:
        expected_minutes = st.number_input("Expected minutes", value=34, min_value=0, max_value=48, step=1)
        minutes_vol = st.selectbox("Minutes volatility", list(MINUTES_VOLATILITY_STDEV_MOD.keys()))
    with c5:
        usage_bump = st.selectbox("Usage bump (injuries / role)", list(USAGE_BUMP_MOD.keys()))
        teammate_outs = st.slider("Key teammates OUT (count)", 0, 4, 0)
    with c6:
        risk_level = st.selectbox("Overall risk level", ["Low", "Medium", "High"])
        add_to_card = st.checkbox("Add result to Daily Card after analyze", value=False)

    if st.button("Analyze NBA"):
        found = nba_players.find_players_by_full_name(player_name)
        if not found:
            st.error("Player not found. Use full name (e.g., 'Jalen Brunson').")
        else:
            player_id = found[0]["id"]
            df = nba_get_game_log(player_id, season)

            s = nba_series_from_market(df, market).astype(float)

            # Recent + season blend
            recent = s.head(use_last_n)
            season_all = s

            w = season_weight / 100.0
            base_mean = float(w * recent.mean() + (1 - w) * season_all.mean())

            # stdev from recent window (more relevant) with fallback
            base_std = float(recent.std(ddof=1)) if recent.std(ddof=1) > 0 else float(season_all.std(ddof=1))
            if not np.isfinite(base_std) or base_std <= 0:
                base_std = 3.0

            # Minutes adjust relative to player's recent minutes
            if "MIN" in df.columns:
                min_recent = pd.to_numeric(df["MIN"].head(use_last_n), errors="coerce").dropna()
                recent_min_avg = float(min_recent.mean()) if len(min_recent) else float(expected_minutes)
            else:
                recent_min_avg = float(expected_minutes)

            minutes_mod = clamp(expected_minutes / max(1.0, recent_min_avg), 0.70, 1.25)

            # Context modifiers
            mean_mod = (
                NBA_DEF_POS_MOD[def_vs_pos]
                * NBA_DEF_OVERALL_MOD[def_overall]
                * NBA_PACE_MOD[pace]
                * NBA_B2B_MOD[b2b]
                * NBA_BLOWOUT_RISK_MOD[blowout]
                * USAGE_BUMP_MOD[usage_bump]
                * minutes_mod
            )

            # Teammates out: modest bump (often real), capped
            mean_mod *= (1.0 + clamp(0.03 * teammate_outs, 0.0, 0.10))

            adj_mean = base_mean * mean_mod

            # Volatility affects stdev (higher minutes volatility -> higher variance)
            adj_std = base_std * MINUTES_VOLATILITY_STDEV_MOD[minutes_vol]
            # Risk level adds a touch more variance if High
            if risk_level == "High":
                adj_std *= 1.05

            p_under, p_over = normal_over_under(adj_mean, adj_std, line)
            prob = p_under if pick_side == "Under" else p_over

            # Odds / edge
            edge_pct = 0.0
            implied = None
            try:
                odds = int(odds_str.strip())
                implied = american_to_prob(odds)
                edge_pct = (prob - implied) * 100.0
            except Exception:
                odds = None

            grade, score = confidence_grade(prob, edge_pct, risk_level)
            is_approved = approved(prob, edge_pct, risk_level) if odds is not None else (prob >= 0.60)

            # If user only wants approved plays
            if show_only_approved and not is_approved:
                st.warning("Not an Approved play under your filter. Toggle off 'Show only Approved' to view details.")
            else:
                left, mid, right = st.columns(3)
                left.metric("Adjusted Projection", f"{adj_mean:.2f}")
                mid.metric(f"% {pick_side}", f"{prob*100:.1f}%")
                right.metric("Confidence", f"{grade} ({score}/100)")

                st.write("**Details**")
                details = {
                    "Base mean": round(base_mean, 2),
                    "Base stdev": round(base_std, 2),
                    "Adj stdev": round(adj_std, 2),
                    "Minutes mod": round(minutes_mod, 3),
                    "Mean mod (total)": round(mean_mod, 3),
                    "Approved": bool(is_approved),
                }
                if implied is not None:
                    details["Implied %"] = round(implied * 100.0, 1)
                    details["Edge %"] = round(edge_pct, 1)
                    details["Odds"] = odds
                st.json(details)

                st.caption("Recent game log (most recent first)")
                show_cols = [c for c in ["GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST"] if c in df.columns]
                st.dataframe(df[show_cols].head(12), use_container_width=True)

                if add_to_card:
                    card_item = {
                        "Sport": "NBA",
                        "Player": player_name,
                        "Market": market,
                        "Line": line,
                        "Side": pick_side,
                        "Prob%": round(prob * 100.0, 1),
                        "Confidence": f"{grade} ({score})",
                        "Edge%": round(edge_pct, 1) if implied is not None else None,
                        "Approved": bool(is_approved),
                    }
                    st.session_state.daily_card.append(card_item)
                    st.success("Added to Daily Card ✅")


# =========================
# NFL TAB
# =========================
with tab_nfl:
    st.subheader("NFL Prop Analyzer (Context + Approved Filter)")

    col1, col2, col3 = st.columns(3)

    with col1:
        nfl_season = st.number_input("Season", value=2025, min_value=1999, max_value=2100, step=1)
        nfl_player = st.text_input("Player name", value="Josh Allen")
        nfl_market = st.selectbox("Market", ["Pass Yds", "Rush Yds", "Rec Yds"])
        nfl_line = st.number_input("Prop line", value=250.5, step=0.5)
        nfl_side = st.selectbox("Pick side", ["Over", "Under"], key="nfl_side")

    with col2:
        nfl_odds_str = st.text_input("Odds (optional, American)", value="-110", key="nfl_odds")
        nfl_last_n = st.slider("Games used (recent)", 4, 17, 8)
        nfl_show_only_approved = st.checkbox("Show only Approved plays", value=False, key="nfl_only_approved")

    with col3:
        st.markdown("**Opponent context (manual)**")
        nfl_def = st.selectbox("Opponent defense tier", list(NFL_DEF_MOD.keys()))
        nfl_script = st.selectbox("Game script", list(NFL_GAME_SCRIPT_MOD.keys()))
        nfl_risk = st.selectbox("Risk level", ["Low", "Medium", "High"], key="nfl_risk")

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

            edge_pct = 0.0
            implied = None
            try:
                odds = int(nfl_odds_str.strip())
                implied = american_to_prob(odds)
                edge_pct = (prob - implied) * 100.0
            except Exception:
                odds = None

            grade, score = confidence_grade(prob, edge_pct, nfl_risk)
            is_approved = approved(prob, edge_pct, nfl_risk) if odds is not None else (prob >= 0.60)

            if nfl_show_only_approved and not is_approved:
                st.warning("Not an Approved play under your filter.")
            else:
                a, b, c = st.columns(3)
                a.metric("Adjusted Projection", f"{adj_mean:.1f}")
                b.metric(f"% {nfl_side}", f"{prob*100:.1f}%")
                c.metric("Confidence", f"{grade} ({score}/100)")

                details = {
                    "Base mean": round(base_mean, 1),
                    "Base stdev": round(base_std, 1),
                    "Mean mod": round(mean_mod, 3),
                    "Adj stdev": round(adj_std, 1),
                    "Approved": bool(is_approved),
                }
                if implied is not None:
                    details["Implied %"] = round(implied * 100.0, 1)
                    details["Edge %"] = round(edge_pct, 1)
                    details["Odds"] = odds
                st.json(details)

                show_cols = [c for c in ["week", "opponent_team", "passing_yards", "rushing_yards", "receiving_yards"] if c in dfp.columns]
                st.dataframe(dfp[show_cols].tail(12), use_container_width=True)

                if st.button("Add this NFL pick to Daily Card"):
                    st.session_state.daily_card.append({
                        "Sport": "NFL",
                        "Player": nfl_player,
                        "Market": nfl_market,
                        "Line": nfl_line,
                        "Side": nfl_side,
                        "Prob%": round(prob * 100.0, 1),
                        "Confidence": f"{grade} ({score})",
                        "Edge%": round(edge_pct, 1) if implied is not None else None,
                        "Approved": bool(is_approved),
                    })
                    st.success("Added to Daily Card ✅")


# =========================
# LIVE TAB (2H / live)
# =========================
with tab_live:
    st.subheader("Live Betting Mode (2H projection + quick props support)")

    st.caption(
        "This is a simple 2H model: it uses first-half production rate + regression + context modifiers. "
        "Best used for live overreactions."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        live_player = st.text_input("Player", value="Stephen Curry", key="live_player")
        live_market = st.selectbox("Market", ["PTS", "REB", "AST", "PRA"], key="live_market")
        current_stat = st.number_input("Current stat at halftime (or now)", value=12.0, step=1.0, key="current_stat")
        minutes_played = st.number_input("Minutes played so far", value=18.0, step=1.0, key="minutes_played")

    with col2:
        live_line_2h = st.number_input("2H prop line (for 2H only)", value=10.5, step=0.5, key="live_line")
        live_pick = st.selectbox("Pick (2H)", ["Over", "Under"], key="live_pick")
        live_odds = st.text_input("Odds (optional)", value="-110", key="live_odds")
        regression = st.slider("Regression toward average (0–50%)", 0, 50, 20, key="regression")

    with col3:
        st.markdown("**Context (manual)**")
        live_pace = st.selectbox("Pace", ["Fast", "Average", "Slow"], key="live_pace")
        live_blowout = st.selectbox("Blowout risk", ["Low", "Medium", "High"], key="live_blowout")
        live_foul_trouble = st.selectbox("Foul trouble / minutes risk", ["No", "Some risk"], key="live_foul")
        live_risk = st.selectbox("Risk level", ["Low", "Medium", "High"], key="live_risk")

    if st.button("Analyze Live (2H)"):
        # Rate so far
        mins = max(1.0, float(minutes_played))
        rate = float(current_stat) / mins  # per-minute rate so far

        # Project remaining minutes (assume 24 min in half, scale)
        remaining = max(1.0, 24.0 - mins)
        raw_2h = rate * remaining

        # Regression (hot/cold halves normalize)
        reg = regression / 100.0
        reg_target = raw_2h * 0.90  # slight natural cooling
        proj_2h = (1 - reg) * raw_2h + reg * reg_target

        # Context mods
        pace_mod = NBA_PACE_MOD[live_pace]
        blow_mod = NBA_BLOWOUT_RISK_MOD[live_blowout]
        foul_mod = 0.94 if live_foul_trouble == "Some risk" else 1.00

        proj_2h *= pace_mod * blow_mod * foul_mod

        # Variance: bigger in live
        stdev = max(2.0, proj_2h * 0.35)
        if live_risk == "High":
            stdev *= 1.10

        p_under, p_over = normal_over_under(proj_2h, stdev, float(live_line_2h))
        prob = p_under if live_pick == "Under" else p_over

        edge_pct = 0.0
        implied = None
        try:
            odds = int(live_odds.strip())
            implied = american_to_prob(odds)
            edge_pct = (prob - implied) * 100.0
        except Exception:
            odds = None

        grade, score = confidence_grade(prob, edge_pct, live_risk)
        is_approved = approved(prob, edge_pct, live_risk) if implied is not None else (prob >= 0.60)

        a, b, c = st.columns(3)
        a.metric("2H Projection", f"{proj_2h:.2f}")
        b.metric(f"% {live_pick}", f"{prob*100:.1f}%")
        c.metric("Confidence", f"{grade} ({score}/100)")

        out = {
            "Rate so far (per min)": round(rate, 3),
            "Projected remaining mins": round(remaining, 1),
            "Context mods": {"pace": live_pace, "blowout": live_blowout, "foul": live_foul_trouble},
            "Model stdev": round(stdev, 2),
            "Approved": bool(is_approved),
        }
        if implied is not None:
            out["Implied %"] = round(implied * 100.0, 1)
            out["Edge %"] = round(edge_pct, 1)
            out["Odds"] = odds
        st.json(out)

        if st.button("Add live pick to Daily Card"):
            st.session_state.daily_card.append({
                "Sport": "LIVE",
                "Player": live_player,
                "Market": f"{live_market} (2H)",
                "Line": live_line_2h,
                "Side": live_pick,
                "Prob%": round(prob * 100.0, 1),
                "Confidence": f"{grade} ({score})",
                "Edge%": round(edge_pct, 1) if implied is not None else None,
                "Approved": bool(is_approved),
            })
            st.success("Added to Daily Card ✅")


# =========================
# DAILY CARD TAB
# =========================
with tab_card:
    st.subheader("Daily Card (Aim: 2–3 Approved Bets)")

    if not st.session_state.daily_card:
        st.info("No picks saved yet. Analyze a bet and add it to your card.")
    else:
        df_card = pd.DataFrame(st.session_state.daily_card)

        # Sort by Approved then confidence
        if "Approved" in df_card.columns:
            df_card = df_card.sort_values(by=["Approved", "Prob%"], ascending=[False, False])

        st.dataframe(df_card, use_container_width=True)

        approved_only = st.checkbox("Show Approved only (recommended)", value=True)
        if approved_only and "Approved" in df_card.columns:
            st.dataframe(df_card[df_card["Approved"] == True], use_container_width=True)

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Quick discipline rules:**")
            st.write("- If it’s not Approved, don’t force it.")
            st.write("- Prefer -115 to +130 singles for steady growth.")
            st.write("- One 2-leg parlay max, only if both legs are Approved.")
        with c2:
            if st.button("Clear Daily Card"):
                st.session_state.daily_card = []
                st.success("Cleared ✅")