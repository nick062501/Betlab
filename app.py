import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

try:
    from nba_api.stats.endpoints import (
        leaguedashteamstats,
        playergamelog,
        scoreboardv2,
    )
    from nba_api.stats.static import players as nba_players
    from nba_api.stats.static import teams as nba_teams
    NBA_API_AVAILABLE = True
except Exception:
    NBA_API_AVAILABLE = False


# ----------------------------
# Streamlit config
# ----------------------------
st.set_page_config(page_title="BetLab — NBA Only", page_icon="🏀", layout="centered")
st.title("🏀 BetLab — NBA Only (Fast)")
st.caption("NBA-only build for speed + stability. Auto-Fill syncs Opponent + DEF rank tier + Pace tier into the filters.")


# ----------------------------
# Keys / math helpers
# ----------------------------
def k(prefix: str, name: str) -> str:
    return f"{prefix}__{name}"

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def prob_over(line: float, mean: float, sd: float) -> float:
    sd = max(sd, 1e-6)
    z = (line - mean) / sd
    return 1.0 - normal_cdf(z)

def american_to_implied_prob(odds: Optional[int]) -> Optional[float]:
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)

def safe_call(fn, timeout_sec: float = 10.0, *args, **kwargs):
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            res = fut.result(timeout=timeout_sec)
            return True, res, ""
        except FuturesTimeoutError:
            return False, None, f"Timed out after {timeout_sec:.0f}s"
        except Exception as e:
            return False, None, f"{type(e).__name__}: {e}"


# ----------------------------
# Cached NBA lookups
# ----------------------------
@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def nba_players_df() -> pd.DataFrame:
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    return pd.DataFrame(nba_players.get_players())

@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def nba_teams_df() -> pd.DataFrame:
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    return pd.DataFrame(nba_teams.get_teams())

@st.cache_data(ttl=60 * 60, show_spinner=False)
def nba_team_context_df(season: str) -> pd.DataFrame:
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    ).get_data_frames()[0]
    if "DEF_RATING" not in df.columns or "PACE" not in df.columns:
        return pd.DataFrame()

    out = df[["TEAM_ID", "TEAM_NAME", "DEF_RATING", "PACE"]].copy()
    out["DEF_RANK"] = out["DEF_RATING"].rank(method="min", ascending=True).astype(int)      # 1 = best defense
    out["PACE_RANK"] = out["PACE"].rank(method="min", ascending=False).astype(int)         # 1 = fastest
    return out

@st.cache_data(ttl=60 * 20, show_spinner=False)
def nba_gamelog_df(player_id: int, season: str) -> pd.DataFrame:
    return playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]

@st.cache_data(ttl=60 * 10, show_spinner=False)
def nba_today_scoreboard_df() -> pd.DataFrame:
    game_date = datetime.now().strftime("%m/%d/%Y")
    return scoreboardv2.ScoreboardV2(game_date=game_date).get_data_frames()[0]


# ----------------------------
# Autofill helpers
# ----------------------------
DEF_TIER = ["Elite (Top 5)", "Good (6-10)", "Average (11-20)", "Weak (21-25)", "Bad (26-30)"]

def tier_from_rank(rank: Optional[int]) -> Optional[str]:
    if rank is None:
        return None
    if rank <= 5:
        return "Elite (Top 5)"
    if rank <= 10:
        return "Good (6-10)"
    if rank <= 20:
        return "Average (11-20)"
    if rank <= 25:
        return "Weak (21-25)"
    return "Bad (26-30)"

def pace_from_rank(rank: Optional[int]) -> Optional[str]:
    if rank is None:
        return None
    if rank <= 10:
        return "Fast"
    if rank <= 20:
        return "Average"
    return "Slow"

def find_player_id(full_name: str) -> Optional[int]:
    df = nba_players_df()
    if df.empty:
        return None
    name = full_name.strip().lower()
    exact = df[df["full_name"].str.lower() == name]
    if not exact.empty:
        return int(exact.iloc[0]["id"])
    parts = name.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        fuzzy = df[df["full_name"].str.lower().str.contains(first) & df["full_name"].str.lower().str.contains(last)]
        if not fuzzy.empty:
            return int(fuzzy.iloc[0]["id"])
    return None

def infer_team_id_from_gamelog(glog: pd.DataFrame) -> Optional[int]:
    for col in ["TEAM_ID", "Team_ID", "team_id"]:
        if col in glog.columns:
            try:
                return int(glog.iloc[0][col])
            except Exception:
                pass
    if "MATCHUP" not in glog.columns:
        return None
    abbr = str(glog.iloc[0]["MATCHUP"]).split()[0].strip()
    tdf = nba_teams_df()
    if tdf.empty:
        return None
    row = tdf[tdf["abbreviation"] == abbr]
    if row.empty:
        return None
    return int(row.iloc[0]["id"])

def team_id_to_name(team_id: Optional[int]) -> Optional[str]:
    if team_id is None:
        return None
    tdf = nba_teams_df()
    if tdf.empty:
        return None
    row = tdf[tdf["id"] == team_id]
    if row.empty:
        return None
    return str(row.iloc[0]["full_name"])

def find_today_opponent_team_id(team_id: int) -> Optional[int]:
    ok, games, _err = safe_call(nba_today_scoreboard_df, 6.0)
    if (not ok) or games is None or games.empty:
        return None
    row = games[(games["HOME_TEAM_ID"] == team_id) | (games["VISITOR_TEAM_ID"] == team_id)]
    if row.empty:
        return None
    r = row.iloc[0]
    return int(r["VISITOR_TEAM_ID"]) if int(r["HOME_TEAM_ID"]) == int(team_id) else int(r["HOME_TEAM_ID"])


# ----------------------------
# Markets
# ----------------------------
MARKETS = ["PTS", "REB", "AST", "3PM", "PRA", "PR", "PA", "RA"]

def market_series(df: pd.DataFrame, market: str) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if market == "PTS":
        return pd.to_numeric(df["PTS"], errors="coerce").dropna()
    if market == "REB":
        return pd.to_numeric(df["REB"], errors="coerce").dropna()
    if market == "AST":
        return pd.to_numeric(df["AST"], errors="coerce").dropna()
    if market == "3PM":
        col = "FG3M" if "FG3M" in df.columns else None
        if not col:
            return pd.Series(dtype=float)
        return pd.to_numeric(df[col], errors="coerce").dropna()

    need = {
        "PRA": ["PTS", "REB", "AST"],
        "PR": ["PTS", "REB"],
        "PA": ["PTS", "AST"],
        "RA": ["REB", "AST"],
    }.get(market)
    if not need:
        return pd.Series(dtype=float)

    for c in need:
        if c not in df.columns:
            return pd.Series(dtype=float)

    s = df[need].sum(axis=1)
    return pd.to_numeric(s, errors="coerce").dropna()


# ----------------------------
# Context multipliers
# ----------------------------
def defense_mult(tier: str, side: str) -> float:
    if side == "Over":
        return {
            "Elite (Top 5)": 0.94, "Good (6-10)": 0.97, "Average (11-20)": 1.00,
            "Weak (21-25)": 1.03, "Bad (26-30)": 1.06
        }.get(tier, 1.00)
    else:
        return {
            "Elite (Top 5)": 1.05, "Good (6-10)": 1.02, "Average (11-20)": 1.00,
            "Weak (21-25)": 0.98, "Bad (26-30)": 0.95
        }.get(tier, 1.00)

def pace_mult(pace: str) -> float:
    return {"Fast": 1.03, "Average": 1.00, "Slow": 0.97}.get(pace, 1.00)

def b2b_mult(b2b: str) -> float:
    return 0.98 if b2b == "Yes" else 1.00

def blowout_mult(risk: str) -> float:
    return {"Low": 1.00, "Medium": 0.98, "High": 0.95}.get(risk, 1.00)

def usage_mult(bump: str) -> float:
    return {"None": 1.00, "Small": 1.03, "Medium": 1.06, "Large": 1.10}.get(bump, 1.00)

def sigma_mult(vol: str) -> float:
    return {"Low": 0.95, "Medium": 1.05, "High": 1.15}.get(vol, 1.05)


# ----------------------------
# Daily Card
# ----------------------------
if "daily_card" not in st.session_state:
    st.session_state.daily_card = []


# ----------------------------
# Tabs
# ----------------------------
tab_nba, tab_card, tab_settings = st.tabs(["🏀 NBA", "🧾 Daily Card", "⚙️ Settings"])

with tab_settings:
    st.subheader("Speed / Reliability")
    api_timeout = st.slider("NBA API timeout (seconds)", 4, 20, 10, key=k("settings", "api_timeout"))
    st.caption("Auto-Fill + Analyze are click-only (no background API calls).")


with tab_nba:
    st.subheader("NBA Prop Analyzer (Auto-Fill syncs defense/pace into the dropdowns)")
    if not NBA_API_AVAILABLE:
        st.error("nba_api not found. Add 'nba_api' to requirements.txt and redeploy.")
        st.stop()

    teams = nba_teams_df()
    team_names = sorted(teams["full_name"].tolist()) if not teams.empty else []

    # --- Auto-Fill button (outside form, so it runs instantly) ---
    colA, colB = st.columns([1, 3])
with colA:
    auto_fill_clicked = st.button("⚡ Auto-Fill (Today)", key=k("nba", "autofill_btn"))
with colB:
    st.caption("Auto-fills opponent + sets the dropdowns (overall defense + pace + opponent/team fields). Still overrideable.")

if auto_fill_clicked:
    timeout = float(st.session_state.get(k("settings", "api_timeout"), 10))

    # IMPORTANT: Use whatever is currently in the text box (if it exists),
    # otherwise fall back to what’s in session_state.
    player_name_for_fill = st.session_state.get(k("nba", "player_name"), "").strip()
    season_for_fill = st.session_state.get(k("nba", "season"), "2025-26").strip()

    with st.spinner("Auto-filling filters..."):
        pid = find_player_id(player_name_for_fill)
        if pid is None:
            st.error("Player not found for auto-fill. Use full name (e.g. 'Stephen Curry').")
        else:
            ok, glog, err = safe_call(nba_gamelog_df, timeout, pid, season_for_fill)
            if (not ok) or glog is None or glog.empty:
                st.error(f"Could not pull player logs. {err}")
            else:
                if "GAME_DATE" in glog.columns:
                    glog = glog.sort_values("GAME_DATE", ascending=False)

                team_id = infer_team_id_from_gamelog(glog)
                team_name = team_id_to_name(team_id)

                opp_id = find_today_opponent_team_id(team_id) if team_id else None
                opp_name = team_id_to_name(opp_id) if opp_id else None

                # Save autofill info (for display)
                st.session_state[k("nba", "autofill_team_name")] = team_name
                st.session_state[k("nba", "autofill_opp_name")] = opp_name

                # ✅ SET THE ACTUAL DROPDOWNS so the UI shows it:
                # Set team override dropdown to team name if found
                if team_name and (team_name in team_names):
                    st.session_state[k("nba", "team_override")] = team_name

                # Set opponent override dropdown to opponent name if found
                if opp_name and (opp_name in team_names):
                    st.session_state[k("nba", "opp_override")] = opp_name

                # Pull opponent rank/tier and write into defense/pace widgets
                if opp_name:
                    ok2, tctx, err2 = safe_call(nba_team_context_df, timeout, season_for_fill)
                    if ok2 and tctx is not None and (not tctx.empty):
                        row = tctx[tctx["TEAM_NAME"] == opp_name]
                        if not row.empty:
                            def_rank = int(row.iloc[0]["DEF_RANK"])
                            pace_rank = int(row.iloc[0]["PACE_RANK"])
                            def_tier = tier_from_rank(def_rank)
                            pace_tier = pace_from_rank(pace_rank)
                            st.session_state[k("nba", "autofill_def_rank")] = def_rank
                            st.session_state[k("nba", "autofill_pace_rank")] = pace_rank

                            # ✅ These are the REAL widget keys in your form:
                            # overall_def selectbox key = k("nba","overall_def")
                            # dvp selectbox key = k("nba","dvp")
                            # pace selectbox key = k("nba","pace")
                            if def_tier in DEF_TIER:
                                st.session_state[k("nba", "overall_def")] = def_tier
                                st.session_state[k("nba", "dvp")] = def_tier  # placeholder = same tier
                            if pace_tier in ["Fast", "Average", "Slow"]:
                                st.session_state[k("nba", "pace")] = pace_tier
                    else:
                        st.warning(f"Could not pull defense/pace ranks right now. {err2}")

    # ✅ Force a rerun so widgets repaint with the new state
    st.rerun()
                    st.success(f"Auto-Fill ✅ Team: {team_name or 'Unknown'} | Opp: {opp_name or 'Not found today'}")

    # Show current autofill state
    auto_team = st.session_state.get(k("nba", "autofill_team_name"))
    auto_opp = st.session_state.get(k("nba", "autofill_opp_name"))
    auto_def_rank = st.session_state.get(k("nba", "autofill_def_rank"))
    auto_pace_rank = st.session_state.get(k("nba", "autofill_pace_rank"))

    with st.expander("Auto-Fill status"):
        st.write(f"- Auto team: **{auto_team or '—'}**")
        st.write(f"- Auto opponent today: **{auto_opp or '—'}**")
        if auto_def_rank is not None:
            st.write(f"- Opp DEF rank: **{auto_def_rank}/30**")
        if auto_pace_rank is not None:
            st.write(f"- Opp Pace rank: **{auto_pace_rank}/30**")

    # --- Form (widgets will now show auto-selected values because we wrote session_state above) ---
    with st.form(key=k("nba", "form")):
        col1, col2 = st.columns([2, 1])
        with col1:
            player_name = st.text_input("Player (full name)", value="Stephen Curry", key=k("nba", "player_name"))
        with col2:
            season = st.text_input("Season", value="2025-26", key=k("nba", "season"))

        market = st.selectbox("Market", MARKETS, key=k("nba", "market"))
        line = st.number_input("Line", value=18.5, step=0.5, key=k("nba", "line"))
        side = st.selectbox("Pick side", ["Over", "Under"], key=k("nba", "side"))
        odds_str = st.text_input("Odds (American, optional)", value="-110", key=k("nba", "odds"))

        st.markdown("### Manual overrides (optional)")
        player_team_override = st.selectbox("Player team (override)", ["(auto)"] + team_names, key=k("nba", "team_override"))
        opp_team_override = st.selectbox("Opponent team (override)", ["(auto)"] + team_names, key=k("nba", "opp_override"))

        st.markdown("### Defense / Pace (auto-synced on Auto-Fill)")
        dvp = st.selectbox("Defense vs position", DEF_TIER, key=k("nba", "dvp"))
        overall_def = st.selectbox("Overall defense", DEF_TIER, key=k("nba", "overall_def"))
        pace = st.selectbox("Opponent pace", ["Fast", "Average", "Slow"], key=k("nba", "pace"))

        st.markdown("### Minutes / Usage / Risk")
        expected_minutes = st.number_input("Expected minutes", 0, 48, 34, 1, key=k("nba", "mins"))
        minutes_vol = st.selectbox("Minutes volatility", ["Low", "Medium", "High"], index=1, key=k("nba", "mins_vol"))
        usage_bump = st.selectbox("Usage bump", ["None", "Small", "Medium", "Large"], index=0, key=k("nba", "usage"))
        teammates_out = st.slider("Key teammates OUT (count)", 0, 6, 0, key=k("nba", "tm_out"))
        b2b = st.selectbox("Back-to-back?", ["No", "Yes"], index=0, key=k("nba", "b2b"))
        blowout = st.selectbox("Blowout risk", ["Low", "Medium", "High"], index=0, key=k("nba", "blowout"))

        injury_notes = st.text_area("Injury notes (optional)", value="", height=90, key=k("nba", "inj_notes"))

        games_recent = st.slider("Recent games used", 5, 30, 15, key=k("nba", "recent_n"))
        weight_recent = st.slider("Weight recent vs season", 0, 100, 70, key=k("nba", "w_recent"))
        approved_only = st.checkbox("Show only Approved", value=False, key=k("nba", "approved_only"))
        add_to_card = st.checkbox("Add to Daily Card", value=True, key=k("nba", "add_card"))

        analyze_submit = st.form_submit_button("Analyze")

    if analyze_submit:
        timeout = float(st.session_state.get(k("settings", "api_timeout"), 10))

        odds_val = None
        try:
            odds_val = int(str(odds_str).strip())
        except Exception:
            odds_val = None

        # Resolve opponent/team used
        used_opp = auto_opp if auto_opp else None
        if opp_team_override != "(auto)":
            used_opp = opp_team_override

        pid = find_player_id(player_name)
        if pid is None:
            st.error("Player not found. Use full name.")
        else:
            with st.spinner("Analyzing (safe timeout)…"):
                ok, glog, err = safe_call(nba_gamelog_df, timeout, pid, season)
            if not ok or glog is None or glog.empty:
                st.error(f"Could not fetch logs. {err}")
            else:
                if "GAME_DATE" in glog.columns:
                    glog = glog.sort_values("GAME_DATE", ascending=False)

                s = market_series(glog, market)
                if s.empty:
                    st.error("Could not compute market series from logs.")
                else:
                    recent_n = min(int(games_recent), len(s))
                    recent = s.iloc[:recent_n].astype(float).to_numpy()
                    season_all = s.astype(float).to_numpy()

                    recent_mean = float(np.mean(recent))
                    season_mean = float(np.mean(season_all))

                    w = float(weight_recent) / 100.0
                    base_mean = w * recent_mean + (1 - w) * season_mean

                    sd = float(np.std(recent, ddof=1)) if len(recent) >= 3 else float(np.std(season_all, ddof=1)) if len(season_all) >= 3 else 3.0
                    sd = max(sd, 1.0)

                    adj_mean = base_mean
                    adj_mean *= defense_mult(dvp, side)
                    adj_mean *= defense_mult(overall_def, side)
                    adj_mean *= pace_mult(pace)
                    adj_mean *= b2b_mult("Yes" if b2b == "Yes" else "No")
                    adj_mean *= blowout_mult(blowout)
                    adj_mean *= usage_mult(usage_bump)
                    adj_mean *= clamp(1.0 + 0.02 * min(teammates_out, 5), 1.0, 1.10)
                    adj_mean *= clamp(expected_minutes / 34.0, 0.75, 1.25)

                    sd_adj = sd * sigma_mult(minutes_vol)

                    p_over = prob_over(float(line), adj_mean, sd_adj)
                    p_pick = p_over if side == "Over" else (1 - p_over)

                    implied = american_to_implied_prob(odds_val)
                    edge = (p_pick - implied) if implied is not None else None

                    approved = True
                    reasons = []
                    if p_pick < 0.56:
                        approved = False
                        reasons.append("Model win% < 56%")
                    if edge is not None and edge < 0.02:
                        approved = False
                        reasons.append("Edge vs implied < 2%")
                    if blowout == "High" and side == "Over":
                        approved = False
                        reasons.append("High blowout risk hurts overs.")
                    if minutes_vol == "High":
                        reasons.append("High minutes volatility (risk).")

                    if approved_only and not approved:
                        st.info("Not Approved (and Approved-only filter is on).")
                    else:
                        st.markdown("## Result")
                        badge = "✅ Approved" if approved else "⚠️ Not Approved"
                        st.subheader(f"{player_name} — {market} {side} {line:.1f}  •  {badge}")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Model win%", f"{p_pick*100:.1f}%")
                        c2.metric("Adj mean", f"{adj_mean:.2f}")
                        c3.metric("Adj SD", f"{sd_adj:.2f}")

                        st.write(f"**Opponent used:** {used_opp or '—'}")
                        st.write(f"**Overall DEF used:** {overall_def}  |  **Pace used:** {pace}")

                        if implied is not None:
                            st.write(f"**Implied win% ({odds_val}):** {implied*100:.1f}%")
                        if edge is not None:
                            st.write(f"**Edge:** {edge*100:.1f}%")

                        st.markdown("### Trend")
                        st.write(f"- Last {recent_n} avg: **{recent_mean:.2f}**")
                        st.write(f"- Season avg: **{season_mean:.2f}**")
                        st.write(f"- Base mean blend: **{base_mean:.2f}**")

                        st.markdown("### Last 10 (most recent first)")
                        last10 = s.iloc[:10].tolist()
                        st.write(", ".join([str(int(x)) if float(x).is_integer() else f"{x:.1f}" for x in last10]))

                        if reasons:
                            st.markdown("### Flags")
                            for r in reasons:
                                st.write(f"- {r}")

                        if injury_notes.strip():
                            st.markdown("### Injury notes")
                            st.write(injury_notes.strip())

                        if add_to_card:
                            st.session_state.daily_card.append({
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "player": player_name,
                                "market": market,
                                "side": side,
                                "line": float(line),
                                "odds": odds_val,
                                "win%": round(p_pick*100, 1),
                                "adj_mean": round(adj_mean, 2),
                                "approved": approved,
                                "opp": used_opp or "",
                            })
                            st.success("Added to Daily Card ✅")


with tab_card:
    st.subheader("Daily Card")
    if not st.session_state.daily_card:
        st.info("No saved plays yet.")
    else:
        df = pd.DataFrame(st.session_state.daily_card)
        st.dataframe(df, use_container_width=True)

    if st.button("Clear Card", key=k("card", "clear")):
        st.session_state.daily_card = []
        st.success("Cleared ✅")