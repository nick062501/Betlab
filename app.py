import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# nba_api is optional at runtime (but recommended). We guard calls with try/except.
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

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError


# ----------------------------
# App config
# ----------------------------
st.set_page_config(page_title="BetLab — NBA & NFL Props (v2)", page_icon="📊", layout="centered")


# ----------------------------
# Helpers (keys, timeouts, math)
# ----------------------------
def k(prefix: str, name: str) -> str:
    """Unique streamlit key generator to avoid DuplicateElementId issues."""
    return f"{prefix}__{name}"


def american_to_implied_prob(odds: Optional[int]) -> Optional[float]:
    if odds is None:
        return None
    try:
        o = int(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    return (-o) / ((-o) + 100.0)


def normal_cdf(x: float) -> float:
    """Standard normal CDF without scipy."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def prob_over(line: float, mean: float, sd: float) -> float:
    sd = max(sd, 1e-6)
    # P(X > line) where X~N(mean, sd)
    z = (line - mean) / sd
    return 1.0 - normal_cdf(z)


def safe_call(fn, timeout_sec: float = 8.0, *args, **kwargs):
    """
    Runs a function in a separate thread with a hard timeout.
    If it times out or errors, returns (False, None, "error/timeout message").
    """
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            result = fut.result(timeout=timeout_sec)
            return True, result, ""
        except FuturesTimeoutError:
            return False, None, f"Timed out after {timeout_sec:.0f}s"
        except Exception as e:
            return False, None, f"{type(e).__name__}: {e}"


# ----------------------------
# NBA API wrappers (cached + safe)
# ----------------------------
@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def nba_player_lookup_df() -> pd.DataFrame:
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    plist = nba_players.get_players()
    df = pd.DataFrame(plist)
    # Keep common name fields
    return df[["id", "full_name", "first_name", "last_name", "is_active"]].copy()


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def nba_teams_df() -> pd.DataFrame:
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    tlist = nba_teams.get_teams()
    df = pd.DataFrame(tlist)
    return df[["id", "full_name", "abbreviation", "nickname", "city", "state"]].copy()


@st.cache_data(ttl=60 * 60, show_spinner=False)
def nba_team_defense_pace(season: str) -> pd.DataFrame:
    """
    Team DEF_RATING (lower=better) and PACE (higher=faster).
    Cached. Wrapped by safe_call at runtime.
    """
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()

    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Base",
    ).get_data_frames()[0]

    if "DEF_RATING" not in df.columns or "PACE" not in df.columns:
        return pd.DataFrame()

    out = df[["TEAM_ID", "TEAM_NAME", "DEF_RATING", "PACE"]].copy()
    out["DEF_RANK"] = out["DEF_RATING"].rank(method="min", ascending=True).astype(int)
    out["PACE_RANK"] = out["PACE"].rank(method="min", ascending=False).astype(int)
    return out


@st.cache_data(ttl=60 * 10, show_spinner=False)
def nba_today_scoreboard_df() -> pd.DataFrame:
    """
    Safe-ish scoreboard. Cached short. Wrapped by safe_call at runtime.
    """
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    game_date = datetime.now().strftime("%m/%d/%Y")
    return scoreboardv2.ScoreboardV2(game_date=game_date).get_data_frames()[0]


@st.cache_data(ttl=60 * 30, show_spinner=False)
def nba_player_gamelog(player_id: int, season: str, season_type: str) -> pd.DataFrame:
    """
    Player game log. Cached. Wrapped by safe_call at runtime.
    """
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    df = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
    ).get_data_frames()[0]
    return df


def find_player_id_by_name(full_name: str) -> Optional[int]:
    df = nba_player_lookup_df()
    if df.empty:
        return None
    m = df[df["full_name"].str.lower() == full_name.strip().lower()]
    if m.empty:
        # light fuzzy: contains both first and last
        parts = full_name.strip().lower().split()
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            m = df[
                df["full_name"].str.lower().str.contains(first)
                & df["full_name"].str.lower().str.contains(last)
            ]
    if m.empty:
        return None
    return int(m.iloc[0]["id"])


def extract_team_id_from_gamelog(df: pd.DataFrame) -> Optional[int]:
    # PlayerGameLog includes "Team_ID" sometimes; if not, infer from "MATCHUP" via abbreviation map.
    if df is None or df.empty:
        return None
    for col in ["Team_ID", "TEAM_ID", "team_id"]:
        if col in df.columns:
            try:
                return int(df.iloc[0][col])
            except Exception:
                pass

    # Fallback: parse MATCHUP abbreviation
    if "MATCHUP" not in df.columns:
        return None
    matchup = str(df.iloc[0]["MATCHUP"])
    # Example: "GSW vs LAL" or "GSW @ LAL"
    abbr = matchup.split()[0].strip()
    tdf = nba_teams_df()
    if tdf.empty:
        return None
    row = tdf[tdf["abbreviation"] == abbr]
    if row.empty:
        return None
    return int(row.iloc[0]["id"])


def nba_find_today_opponent_team_id(player_team_id: int) -> Optional[int]:
    ok, games, _err = safe_call(nba_today_scoreboard_df, 6.0)
    if (not ok) or games is None or games.empty:
        return None
    row = games[(games["HOME_TEAM_ID"] == player_team_id) | (games["VISITOR_TEAM_ID"] == player_team_id)]
    if row.empty:
        return None
    r = row.iloc[0]
    return int(r["VISITOR_TEAM_ID"]) if int(r["HOME_TEAM_ID"]) == int(player_team_id) else int(r["HOME_TEAM_ID"])


# ----------------------------
# Model logic (simple, fast, practical)
# ----------------------------
@dataclass
class NBAContext:
    season: str
    market: str
    line: float
    pick_over: bool
    odds: Optional[int]
    games_recent: int
    weight_recent: float  # 0..1

    # opponent context
    defense_vs_pos: str
    overall_def: str
    pace: str
    b2b: str
    blowout: str

    # minutes / usage
    expected_minutes: int
    minutes_vol: str
    usage_bump: str
    teammates_out: int

    # injuries text
    injury_notes: str

    approved_only: bool
    add_to_card: bool


def adjust_mean_for_context(base_mean: float, ctx: NBAContext) -> float:
    """
    Heuristic adjustments that are:
    - transparent
    - fast
    - user-controllable
    """
    mean = base_mean

    # Defense vs position
    dvp_adj = {
        "Elite (Top 5)": -0.08,
        "Good (6-10)": -0.05,
        "Average (11-20)": 0.00,
        "Weak (21-25)": +0.05,
        "Bad (26-30)": +0.08,
    }.get(ctx.defense_vs_pos, 0.0)

    # Overall defense
    od_adj = {
        "Elite (Top 5)": -0.06,
        "Good (6-10)": -0.04,
        "Average (11-20)": 0.00,
        "Weak (21-25)": +0.04,
        "Bad (26-30)": +0.06,
    }.get(ctx.overall_def, 0.0)

    # Pace
    pace_adj = {
        "Fast": +0.05,
        "Average": 0.00,
        "Slow": -0.05,
    }.get(ctx.pace, 0.0)

    # Back-to-back
    b2b_adj = {"Yes": -0.03, "No": 0.0}.get(ctx.b2b, 0.0)

    # Blowout risk reduces ceiling + minutes
    blow_adj = {"Low": 0.0, "Medium": -0.02, "High": -0.05}.get(ctx.blowout, 0.0)

    # Usage bump
    usage_adj = {
        "None": 0.0,
        "Small": +0.03,
        "Medium": +0.06,
        "Large": +0.10,
    }.get(ctx.usage_bump, 0.0)

    # Teammates out
    team_out_adj = min(ctx.teammates_out, 5) * 0.01

    # Apply multiplicative adjustments to mean
    mult = 1.0 + dvp_adj + od_adj + pace_adj + b2b_adj + blow_adj + usage_adj + team_out_adj
    mean *= mult

    # Minutes scaling (relative to a baseline 34)
    mean *= (ctx.expected_minutes / 34.0)

    return mean


def estimate_sd_from_logs(values: np.ndarray) -> float:
    if len(values) < 3:
        return float(np.std(values)) if len(values) > 0 else 1.0
    sd = float(np.std(values, ddof=1))
    return max(sd, 1.0)


def analyze_nba_prop(player_name: str, ctx: NBAContext, season_type: str = "Regular Season") -> Dict[str, Any]:
    """
    Returns analysis dict. All API calls are safe_call guarded.
    """
    result: Dict[str, Any] = {"ok": True, "errors": []}

    if not NBA_API_AVAILABLE:
        result["ok"] = False
        result["errors"].append("nba_api not available. Add nba_api to requirements.txt.")
        return result

    pid = find_player_id_by_name(player_name)
    if pid is None:
        result["ok"] = False
        result["errors"].append("Player not found. Try full name exactly (e.g., 'Stephen Curry').")
        return result

    # Fetch game logs safely (this is the most valuable part)
    ok, glog, err = safe_call(nba_player_gamelog, 10.0, pid, ctx.season, season_type)
    if not ok or glog is None or glog.empty:
        result["ok"] = False
        result["errors"].append(f"Could not fetch game logs. {err}")
        return result

    # Make sure newest first
    if "GAME_DATE" in glog.columns:
        glog = glog.sort_values("GAME_DATE", ascending=False)

    market_map = {
        "PTS": "PTS",
        "REB": "REB",
        "AST": "AST",
        "PRA": None,  # computed
        "PR": None,
        "PA": None,
        "RA": None,
        "3PM": "FG3M" if "FG3M" in glog.columns else None,
    }

    market = ctx.market
    if market in ["PRA", "PR", "PA", "RA"]:
        # build composite
        need = {
            "PRA": ["PTS", "REB", "AST"],
            "PR": ["PTS", "REB"],
            "PA": ["PTS", "AST"],
            "RA": ["REB", "AST"],
        }[market]
        for c in need:
            if c not in glog.columns:
                result["ok"] = False
                result["errors"].append(f"Missing stat column {c} in logs.")
                return result
        vals = glog[need].sum(axis=1).to_numpy(dtype=float)
        stat_label = "+".join(need)
    else:
        col = market_map.get(market, None)
        if col is None or col not in glog.columns:
            result["ok"] = False
            result["errors"].append(f"Market '{market}' not supported from logs.")
            return result
        vals = glog[col].to_numpy(dtype=float)
        stat_label = col

    # Recent + season blends
    recent_n = max(3, min(int(ctx.games_recent), len(vals)))
    recent_vals = vals[:recent_n]
    season_vals = vals

    recent_mean = float(np.mean(recent_vals)) if len(recent_vals) else float(np.mean(season_vals))
    season_mean = float(np.mean(season_vals))

    w = float(np.clip(ctx.weight_recent, 0.0, 1.0))
    base_mean = w * recent_mean + (1.0 - w) * season_mean

    # SD from recent first, fallback to season
    sd = estimate_sd_from_logs(recent_vals) if len(recent_vals) >= 3 else estimate_sd_from_logs(season_vals)

    # Context adjustment
    adj_mean = adjust_mean_for_context(base_mean, ctx)

    p_over = prob_over(ctx.line, adj_mean, sd)
    p_pick = p_over if ctx.pick_over else (1.0 - p_over)

    implied = american_to_implied_prob(ctx.odds) if ctx.odds is not None else None
    edge = (p_pick - implied) if (implied is not None) else None

    # Approval logic (simple + transparent)
    approval = True
    reasons = []
    if ctx.minutes_vol == "High":
        approval = False
        reasons.append("Minutes volatility is High.")
    if ctx.blowout == "High":
        reasons.append("Blowout risk High (minutes risk).")
    if ctx.expected_minutes < 28:
        reasons.append("Expected minutes < 28 (low volume).")
    if edge is not None and edge < 0.02:
        reasons.append("Edge vs implied is small (<2%).")
    if edge is not None and edge < 0:
        approval = False
        reasons.append("Negative edge vs implied.")

    result.update(
        {
            "player_id": pid,
            "stat_label": stat_label,
            "recent_n": recent_n,
            "recent_mean": recent_mean,
            "season_mean": season_mean,
            "base_mean": base_mean,
            "adj_mean": adj_mean,
            "sd": sd,
            "p_over": p_over,
            "p_pick": p_pick,
            "implied": implied,
            "edge": edge,
            "approved": approval,
            "reasons": reasons,
            "last_10": list(map(float, vals[:10])),
        }
    )
    return result


# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📊 BetLab — NBA & NFL Props (v2)")
st.caption("Decision support tool. Betting involves risk. Use disciplined staking and focus on 2–3 high-quality bets/day.")

tabs = st.tabs(["🏀 NBA", "🏈 NFL", "⚡ Live (2H)", "🧾 Daily Card", "⚙️ Settings"])

if "daily_card" not in st.session_state:
    st.session_state.daily_card = []  # list of dicts


# ----------------------------
# Settings
# ----------------------------
with tabs[4]:
    st.subheader("Settings")

    fast_boot = st.checkbox(
        "Fast Boot Mode (recommended on Streamlit Cloud)",
        value=True,
        help="Prevents slow API calls during page load. Analysis still works when you click Analyze.",
        key=k("settings", "fast_boot"),
    )

    api_timeout = st.slider(
        "API timeout (seconds)",
        min_value=4,
        max_value=20,
        value=10,
        help="Hard timeout for nba_api calls. Lower = faster fail instead of hanging.",
        key=k("settings", "api_timeout"),
    )

    st.info(
        "If the app ever loads slowly, keep Fast Boot ON and reduce API timeout to ~8–10s."
    )

    st.write("---")
    st.write("If you deploy on Streamlit Cloud, make sure your `requirements.txt` includes:")
    st.code("streamlit\npandas\nnumpy\nnba_api\nrequests\n", language="text")


# ----------------------------
# NBA TAB
# ----------------------------
with tabs[0]:
    prefix = "nba"

    st.subheader("NBA Prop Analyzer (Defense + Minutes/Usage + Approved Filter)")
    if not NBA_API_AVAILABLE:
        st.warning("nba_api not detected in this environment. Add it to requirements.txt on Streamlit Cloud.")

    # Form prevents reruns while you change controls
    with st.form(key=k(prefix, "form")):
        col1, col2 = st.columns([2, 1])

        with col1:
            player_name = st.text_input(
                "Player (full name)",
                value="Stephen Curry",
                key=k(prefix, "player_name"),
            )

        with col2:
            season = st.text_input(
                "Season (NBA API format)",
                value="2025-26",
                help="Example: 2025-26",
                key=k(prefix, "season"),
            )

        market = st.selectbox(
            "Market",
            ["PTS", "REB", "AST", "3PM", "PRA", "PR", "PA", "RA"],
            key=k(prefix, "market"),
        )

        line = st.number_input("Prop line", value=18.5, step=0.5, key=k(prefix, "line"))

        pick_side = st.selectbox("Pick side", ["Over", "Under"], key=k(prefix, "pick_side"))
        odds_in = st.text_input("Odds (optional, American)", value="-110", key=k(prefix, "odds"))

        st.markdown("#### Recent form")
        games_recent = st.slider("Games used (recent)", 3, 25, 21, key=k(prefix, "games_recent"))
        weight_recent_pct = st.slider("Weight: recent vs season (0=season, 100=recent)", 0, 100, 70, key=k(prefix, "weight_recent"))

        st.markdown("#### Opponent context (fast)")
        cA, cB = st.columns(2)
        with cA:
            defense_vs_pos = st.selectbox(
                "Defense vs position",
                ["Elite (Top 5)", "Good (6-10)", "Average (11-20)", "Weak (21-25)", "Bad (26-30)"],
                index=2,
                key=k(prefix, "dvp"),
            )
            pace = st.selectbox("Opponent pace", ["Fast", "Average", "Slow"], index=1, key=k(prefix, "pace"))

        with cB:
            overall_def = st.selectbox(
                "Overall defense",
                ["Elite (Top 5)", "Good (6-10)", "Average (11-20)", "Weak (21-25)", "Bad (26-30)"],
                index=2,
                key=k(prefix, "overall_def"),
            )
            b2b = st.selectbox("Back-to-back?", ["No", "Yes"], index=0, key=k(prefix, "b2b"))

        blowout = st.selectbox("Blowout risk", ["Low", "Medium", "High"], index=0, key=k(prefix, "blowout"))

        st.markdown("#### Minutes & usage intelligence")
        expected_minutes = st.number_input("Expected minutes", min_value=0, max_value=48, value=34, step=1, key=k(prefix, "mins"))
        minutes_vol = st.selectbox("Minutes volatility", ["Low", "Medium", "High"], index=0, key=k(prefix, "mins_vol"))
        usage_bump = st.selectbox("Usage bump (injuries / role)", ["None", "Small", "Medium", "Large"], index=0, key=k(prefix, "usage"))
        teammates_out = st.slider("Key teammates OUT (count)", 0, 6, 0, key=k(prefix, "tm_out"))

        st.markdown("#### Injury report (notes)")
        injury_notes = st.text_area(
            "Paste injury notes here (optional). This won’t fetch automatically (keeps app fast).",
            value="",
            height=90,
            key=k(prefix, "inj_notes"),
        )

        st.markdown("#### Output controls")
        approved_only = st.checkbox("Show only Approved plays", value=False, key=k(prefix, "approved_only"))
        add_to_card = st.checkbox("Add result to Daily Card after analyze", value=True, key=k(prefix, "add_to_card"))

        submit = st.form_submit_button("Analyze NBA")

    # Analyze only on submit (keeps it fast)
    if submit:
        # Parse odds
        odds_val: Optional[int] = None
        try:
            odds_val = int(str(odds_in).strip())
        except Exception:
            odds_val = None

        ctx = NBAContext(
            season=season.strip(),
            market=market,
            line=float(line),
            pick_over=(pick_side == "Over"),
            odds=odds_val,
            games_recent=int(games_recent),
            weight_recent=float(weight_recent_pct) / 100.0,
            defense_vs_pos=defense_vs_pos,
            overall_def=overall_def,
            pace=pace,
            b2b=b2b,
            blowout=blowout,
            expected_minutes=int(expected_minutes),
            minutes_vol=minutes_vol,
            usage_bump=usage_bump,
            teammates_out=int(teammates_out),
            injury_notes=injury_notes.strip(),
            approved_only=approved_only,
            add_to_card=add_to_card,
        )

        with st.spinner("Running analysis (safe timeout)…"):
            out = analyze_nba_prop(player_name, ctx, season_type="Regular Season")

        if not out.get("ok", False):
            st.error("Could not analyze.")
            for e in out.get("errors", []):
                st.write(f"- {e}")
        else:
            approved = out["approved"]
            if approved_only and not approved:
                st.info("This play is not Approved, and you enabled Approved-only filter.")
            else:
                # Display core decision block
                st.markdown("### Result")
                badge = "✅ Approved" if approved else "⚠️ Not Approved"
                st.subheader(f"{player_name} — {market} {pick_side} {line:.1f}  •  {badge}")

                c1, c2, c3 = st.columns(3)
                c1.metric("Adj mean", f"{out['adj_mean']:.2f}")
                c2.metric("Std dev", f"{out['sd']:.2f}")
                c3.metric("Win % (model)", f"{out['p_pick']*100:.1f}%")

                if out["implied"] is not None:
                    st.caption(f"Implied win % from odds: {out['implied']*100:.1f}%")
                if out["edge"] is not None:
                    st.caption(f"Edge vs implied: {out['edge']*100:.1f}%")

                st.markdown("**Recent vs Season**")
                st.write(
                    f"- Last {out['recent_n']} avg: **{out['recent_mean']:.2f}**\n"
                    f"- Season avg: **{out['season_mean']:.2f}**\n"
                    f"- Blended base: **{out['base_mean']:.2f}** → Context-adjusted: **{out['adj_mean']:.2f}**"
                )

                st.markdown("**Last 10 (most recent first)**")
                st.write(", ".join([str(int(x)) if float(x).is_integer() else f"{x:.1f}" for x in out["last_10"]]))

                if out["reasons"]:
                    st.markdown("**Notes / Flags**")
                    for r in out["reasons"]:
                        st.write(f"- {r}")

                if ctx.injury_notes:
                    st.markdown("**Injury notes you pasted**")
                    st.write(ctx.injury_notes)

                # Add to Daily Card
                if ctx.add_to_card:
                    st.session_state.daily_card.append(
                        {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "sport": "NBA",
                            "player": player_name,
                            "market": market,
                            "side": pick_side,
                            "line": float(line),
                            "odds": odds_val,
                            "win_pct": float(out["p_pick"]) * 100.0,
                            "adj_mean": float(out["adj_mean"]),
                            "sd": float(out["sd"]),
                            "approved": bool(out["approved"]),
                        }
                    )
                    st.success("Added to Daily Card ✅")


# ----------------------------
# NFL TAB (skeleton + fast, extendable)
# ----------------------------
with tabs[1]:
    st.subheader("NFL Prop Analyzer (fast + manual opponent context)")
    st.info("NFL auto data pulls can be added later. For speed/reliability, this tab is manual-first (still very useful).")

    prefix = "nfl"
    with st.form(key=k(prefix, "form")):
        player = st.text_input("Player", value="", key=k(prefix, "player"))
        market = st.selectbox("Market", ["REC YDS", "RUSH YDS", "PASS YDS", "RECEPTIONS", "RUSH+REC YDS", "TD"], key=k(prefix, "market"))
        line = st.number_input("Prop line", value=0.0, step=0.5, key=k(prefix, "line"))
        pick = st.selectbox("Pick side", ["Over", "Under"], key=k(prefix, "pick"))
        odds = st.text_input("Odds (optional, American)", value="", key=k(prefix, "odds"))

        st.markdown("#### Opponent context (manual, fast)")
        redzone_def = st.selectbox("Opponent red zone defense", ["Elite", "Good", "Average", "Bad"], index=2, key=k(prefix, "rz"))
        matchup = st.selectbox("Matchup strength vs role", ["Great", "Good", "Average", "Tough"], index=2, key=k(prefix, "matchup"))
        game_script = st.selectbox("Game script", ["Likely leading", "Neutral", "Likely trailing"], index=1, key=k(prefix, "script"))
        injury_notes = st.text_area("Injury notes (paste)", value="", height=80, key=k(prefix, "inj"))

        submit_nfl = st.form_submit_button("Analyze NFL")

    if submit_nfl:
        # lightweight heuristic output (no external calls)
        try:
            o = int(odds) if str(odds).strip() else None
        except Exception:
            o = None

        base = 0.52
        # quick heuristics
        adj = 0.0
        adj += {"Elite": -0.05, "Good": -0.02, "Average": 0.0, "Bad": 0.03}.get(redzone_def, 0.0)
        adj += {"Great": 0.05, "Good": 0.02, "Average": 0.0, "Tough": -0.03}.get(matchup, 0.0)
        adj += {"Likely leading": 0.02, "Neutral": 0.0, "Likely trailing": -0.02}.get(game_script, 0.0)

        p_over = float(np.clip(base + adj, 0.05, 0.95))
        p_pick = p_over if pick == "Over" else (1 - p_over)

        implied = american_to_implied_prob(o) if o is not None else None
        edge = (p_pick - implied) if implied is not None else None

        st.markdown("### Result")
        st.subheader(f"{player} — {market} {pick} {line:.1f}")
        st.metric("Win % (heuristic)", f"{p_pick*100:.1f}%")
        if implied is not None:
            st.caption(f"Implied win % from odds: {implied*100:.1f}%")
        if edge is not None:
            st.caption(f"Edge vs implied: {edge*100:.1f}%")
        if injury_notes.strip():
            st.markdown("**Injury notes**")
            st.write(injury_notes.strip())


# ----------------------------
# Live (2H) TAB (fast)
# ----------------------------
with tabs[2]:
    st.subheader("Live (2H) Quick Check")
    st.caption("Fast calculator for halftime situations. Paste current stat + line, get quick probability estimate.")

    prefix = "live"
    with st.form(key=k(prefix, "form")):
        market = st.selectbox("Market", ["PTS", "REB", "AST", "PRA"], key=k(prefix, "market"))
        current = st.number_input("Current stat (right now)", value=0.0, step=1.0, key=k(prefix, "current"))
        line = st.number_input("Full game line", value=0.0, step=0.5, key=k(prefix, "line"))
        minutes_left = st.number_input("Estimated minutes left", value=24.0, step=1.0, key=k(prefix, "mins_left"))
        pick = st.selectbox("Pick side", ["Over", "Under"], key=k(prefix, "pick"))
        odds = st.text_input("Odds (optional)", value="", key=k(prefix, "odds"))
        vol = st.selectbox("Volatility", ["Low", "Medium", "High"], index=1, key=k(prefix, "vol"))
        submit_live = st.form_submit_button("Estimate")

    if submit_live:
        try:
            o = int(odds) if str(odds).strip() else None
        except Exception:
            o = None

        # crude pace projection: assume remaining production proportional to minutes left
        # add volatility SD based on remaining time
        remaining = max(line - current, 0.0)
        # mean remaining = (current / minutes played) * minutes_left; approximate minutes played = 48 - minutes_left
        mins_played = max(48.0 - float(minutes_left), 1.0)
        rate = float(current) / mins_played
        mean_final = float(current) + rate * float(minutes_left)

        base_sd = {"Low": 6.0, "Medium": 9.0, "High": 12.0}.get(vol, 9.0)
        # scale sd by remaining time
        sd = base_sd * math.sqrt(float(minutes_left) / 24.0)

        p_over = prob_over(float(line), mean_final, sd)
        p_pick = p_over if pick == "Over" else (1 - p_over)

        implied = american_to_implied_prob(o) if o is not None else None
        edge = (p_pick - implied) if implied is not None else None

        st.markdown("### Live estimate")
        st.write(f"Projected final (rough): **{mean_final:.1f}**  |  SD: **{sd:.1f}**")
        st.metric("Win %", f"{p_pick*100:.1f}%")
        if implied is not None:
            st.caption(f"Implied win %: {implied*100:.1f}%")
        if edge is not None:
            st.caption(f"Edge vs implied: {edge*100:.1f}%")


# ----------------------------
# Daily Card TAB
# ----------------------------
with tabs[3]:
    st.subheader("Daily Card")
    if not st.session_state.daily_card:
        st.info("No plays saved yet. Analyze a play and check 'Add result to Daily Card'.")
    else:
        df = pd.DataFrame(st.session_state.daily_card)
        st.dataframe(df, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear Daily Card", key="clear_daily"):
                st.session_state.daily_card = []
                st.success("Cleared.")
        with c2:
            st.caption("Tip: keep it to 2–3 highest-confidence plays/day.")