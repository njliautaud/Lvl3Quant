#!/usr/bin/env python3
"""
Iron Condor Income V1 — Regime-Gated + Corrected Accounting
============================================================
Symmetric premium selling strategy: short put spread + short call spread.
Profits when underlying stays between the short strikes (range-bound).

KEY DIFFERENCE FROM JADE LIZARD:
  - SYMMETRIC risk: equal-width spreads on both sides
  - Capped max loss on BOTH sides (vs Jade Lizard's naked put risk)
  - Lower net credit (buying 2 protective legs vs 1)
  - Should have better regime balance — profits from range, not direction

STRUCTURE (per position):
  - Sell OTM put at target delta → buy put 5% further OTM (bull put spread)
  - Sell OTM call at target delta → buy call 5% further OTM (bear call spread)
  - Net credit = (short_put - long_put) + (short_call - long_call) - BA costs
  - Max loss = max(put_spread_width, call_spread_width) * 100 * contracts - credit

ACCOUNTING: Corrected (no double-count). Credit NOT added at open, only realized
at close/expiry. Decaying premium multiplier applied.

SWEEP (11 configs):
  1. Baseline: 1x BS, 15 pos, 21d DTE, no gates
  2. VIX gate only
  3. SPY trend filter only
  4. VIX + SPY combined
  5. Shorter DTE (7d) + VIX gate
  6. Longer DTE (35d) + VIX gate
  7. Wide strikes (0.15 delta) + VIX gate
  8. Narrow strikes (0.25 delta) + VIX gate
  9. Conservative 5 positions + VIX gate
 10. 1.5x premium + VIX+SPY
 11. Full gate + emergency close, 10 pos
"""

import sys, json, time, math, warnings, os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import timedelta

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUTPUT = ROOT / "output" / "income_research"
OUTPUT.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100_000

TICKERS = [
    'AAPL','ABBV','ABNB','ADBE','AMD','AMZN','ARM','AXP','BA','BAC',
    'BLK','BRK-B','C','CAT','CL','COIN','COST','CRM','CRWD','CVX',
    'DDOG','DE','DIS','F','GE','GM','GOOGL','GS','HD','HOOD',
    'INTC','JNJ','JPM','KO','LLY','LOW','MA','MCD','META','MRNA',
    'MS','MSFT','NFLX','NOW','NVDA','ORCL','OXY','PANW','PEP','PFE',
    'PG','PLTR','PYPL','RTX','SBUX','SCHW','SHOP','SLB','SMCI','T',
    'TGT','TMUS','TSLA','UBER','UNH','V','VZ','WFC','WMT','XOM',
]


# ═══════════════════════════════════════════════════════════════════
# BS Primitives
# ═══════════════════════════════════════════════════════════════════

def _Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _ndtri(p):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def bs_price(S, K, T, sigma, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if kind == "put":
        return K*math.exp(-r*T)*_Phi(-d2) - S*math.exp(-q*T)*_Phi(-d1)
    return S*math.exp(-q*T)*_Phi(d1) - K*math.exp(-r*T)*_Phi(d2)

def strike_from_delta(S, T, sigma, target_delta, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0:
        return S
    target = abs(target_delta)
    p = target if kind == "call" else (1 - target)
    p = min(max(p, 1e-6), 1 - 1e-6)
    d1 = _ndtri(p)
    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r - q + 0.5*sigma**2)*T))
    return K

def decaying_premium_mult(base_mult, T_remaining, T_original):
    if T_original <= 0:
        return 1.0
    frac_remaining = max(0, min(1, T_remaining / T_original))
    return 1.0 + (base_mult - 1.0) * frac_remaining


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def generate_iv_features(prices_df, tickers):
    frames = []
    for tk in tickers:
        tk_px = prices_df[prices_df["ticker"] == tk].sort_values("date").copy()
        if len(tk_px) < 60:
            continue
        tk_px["log_ret"] = np.log1p(tk_px["close"].pct_change())
        tk_px["rv_20"] = tk_px["log_ret"].rolling(20).std() * np.sqrt(252)
        tk_px["rv_60"] = tk_px["log_ret"].rolling(60).std() * np.sqrt(252)
        tk_px["sigma"] = tk_px["rv_20"] * 1.15
        tk_px["iv_high_252"] = tk_px["sigma"].rolling(252).max()
        tk_px["iv_low_252"] = tk_px["sigma"].rolling(252).min()
        iv_range = tk_px["iv_high_252"] - tk_px["iv_low_252"]
        tk_px["iv_rank"] = np.where(iv_range > 0.001,
            (tk_px["sigma"] - tk_px["iv_low_252"]) / iv_range, 0.5)
        tk_px["sigma_rv"] = tk_px["rv_20"]
        tk_px["ticker"] = tk
        cols = ["date", "ticker", "sigma_rv", "sigma", "iv_rank"]
        frame = tk_px.dropna(subset=["sigma"])[cols]
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def load_all_data():
    print("Loading data...")
    prices = pd.read_parquet(CACHE / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    for extra in ["prices_expanded.parquet", "prices_v3_expansion.parquet"]:
        try:
            pexp = pd.read_parquet(CACHE / extra)
            if "Open" in pexp.columns:
                pexp = pexp.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                             "Close": "close", "Volume": "volume"})
            pexp["date"] = pd.to_datetime(pexp["date"]).dt.tz_localize(None)
            new_tks = set(pexp["ticker"].unique()) - set(prices["ticker"].unique())
            if new_tks:
                pexp = pexp[pexp["ticker"].isin(new_tks)]
                prices = pd.concat([prices, pexp], ignore_index=True)
        except:
            pass

    prices = prices[prices["date"] >= "2019-01-01"].copy()
    prices = prices.drop_duplicates(subset=["ticker", "date"], keep="first")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    if "ret" not in prices.columns:
        prices["ret"] = prices.groupby("ticker")["close"].pct_change()
    if "log_ret" not in prices.columns:
        prices["log_ret"] = np.log1p(prices["ret"])

    iv = pd.read_parquet(CACHE / "iv_features_modeled.parquet")
    iv["date"] = pd.to_datetime(iv["date"]).dt.tz_localize(None)
    iv = iv[iv["date"] >= "2019-01-01"]

    all_needed = set(TICKERS)
    iv_tickers = set(iv["ticker"].unique())
    need_iv = (all_needed & set(prices["ticker"].unique())) - iv_tickers
    if need_iv:
        iv_new = generate_iv_features(prices, sorted(need_iv))
        if not iv_new.empty:
            iv = pd.concat([iv, iv_new], ignore_index=True)

    macro = pd.read_parquet(CACHE / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
    macro = macro[macro["date"] >= "2019-01-01"]

    vix_df = pd.read_parquet(CACHE / "vix_history.parquet")
    vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.tz_localize(None)

    spy = pd.read_parquet(CACHE / "spy_prices.parquet")
    spy["date"] = pd.to_datetime(spy["date"]).dt.tz_localize(None)
    spy = spy.sort_values("date")
    spy["spy_ret"] = spy["close"].pct_change()
    spy["spy_sma20"] = spy["close"].rolling(20).mean()

    try:
        earnings = pd.read_parquet(CACHE / "earnings_dates.parquet")
        earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"]).dt.tz_localize(None)
    except:
        earnings = pd.DataFrame(columns=["ticker", "earnings_date"])

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique()) & all_needed
    avail_list = sorted([t for t in TICKERS if t in available])

    print(f"  {len(avail_list)} tickers available")
    return prices, iv, macro, vix_df, spy, earnings, avail_list


def build_earnings_lookup(earnings_df):
    lookup = {}
    for ticker in earnings_df["ticker"].unique():
        dates = earnings_df[earnings_df["ticker"] == ticker]["earnings_date"].sort_values().values
        if len(dates) > 0:
            lookup[ticker] = dates
    return lookup


def has_earnings_within(ticker, open_date, dte_target, earnings_lookup, buffer_days=7):
    if ticker not in earnings_lookup:
        return False
    earn_dates = earnings_lookup[ticker]
    hold_start = np.datetime64(open_date) - np.timedelta64(buffer_days, 'D')
    hold_end = np.datetime64(open_date) + np.timedelta64(dte_target + buffer_days, 'D')
    return ((earn_dates >= hold_start) & (earn_dates <= hold_end)).any()


def ba_cost_per_leg(premium, ba_frac):
    return premium * (ba_frac / 2.0) * 100


# ═══════════════════════════════════════════════════════════════════
# REGIME GATES
# ═══════════════════════════════════════════════════════════════════

def vix_tier_sizing(vix_val):
    """BPS Conservative proven tiers. Returns sizing multiplier 0.0-1.0."""
    if np.isnan(vix_val):
        return 0.5  # conservative default
    if vix_val > 30:
        return 0.0   # HARD STOP — no new positions
    elif vix_val > 25:
        return 0.25
    elif vix_val > 20:
        return 0.50
    elif vix_val > 15:
        return 0.80
    else:
        return 1.0


def spy_trend_filter(spy_close, spy_sma20):
    """Only open new positions when SPY > 20-day SMA (uptrend)."""
    if spy_close is None or spy_sma20 is None or np.isnan(spy_close) or np.isnan(spy_sma20):
        return False
    return spy_close > spy_sma20


def emergency_close_check(spy_ret):
    """Close all positions if SPY drops >2% in a day."""
    if spy_ret is None or np.isnan(spy_ret):
        return False
    return spy_ret < -0.02


# ═══════════════════════════════════════════════════════════════════
# IRON CONDOR V1 — CORRECTED + REGIME-GATED
# ═══════════════════════════════════════════════════════════════════

def run_iron_condor(prices, iv, macro, vix_df, spy_df, earnings_lookup, ticker_list,
                    ba_frac=0.10, dte_target=21, put_delta=-0.20, call_delta=0.20,
                    long_put_offset=0.05, long_call_offset=0.05,
                    profit_take=0.50, stop_loss_mult=2.0,
                    max_concurrent=15, per_name_pct=0.04,
                    earnings_buffer=7, premium_mult=1.0,
                    # Regime gate settings
                    use_vix_gate=False, use_spy_trend=False,
                    use_emergency_close=False,
                    gate_label=""):

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique()) & set(ticker_list)
    prices_df = prices[prices["ticker"].isin(available)].copy()
    iv_df = iv[iv["ticker"].isin(available)].copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()

    macro_copy = macro.copy()
    macro_copy["date"] = pd.to_datetime(macro_copy["date"])
    macro_by_date = macro_copy.set_index("date").to_dict("index")

    spy_copy = spy_df.copy()
    spy_copy["date"] = pd.to_datetime(spy_copy["date"])
    spy_by_date = spy_copy.set_index("date").to_dict("index")

    all_dates = sorted(prices_df["date"].unique())

    cash = float(STARTING_CAPITAL)
    positions = {}
    equity_curve = []
    detailed_trades = []
    last_close = {}
    pos_counter = 0
    stats = {"n_opened": 0, "n_closed": 0, "n_profit_take": 0, "n_stop_loss": 0,
             "n_expiry_win": 0, "n_expiry_loss": 0,
             "n_put_side_loss": 0, "n_call_side_loss": 0,
             "n_earnings_blocked": 0, "n_vix_blocked": 0, "n_spy_blocked": 0,
             "n_emergency_closed": 0, "total_ba_cost": 0.0,
             "total_premium_collected": 0.0}

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        m_data = macro_by_date.get(dt, {})
        spy_data = spy_by_date.get(dt, {})

        vix = m_data.get("vix", float("nan")) if isinstance(m_data, dict) else float("nan")
        try:
            vix_val = float(vix)
        except:
            vix_val = float("nan")

        spy_close = spy_data.get("close", None) if isinstance(spy_data, dict) else None
        spy_sma20 = spy_data.get("spy_sma20", None) if isinstance(spy_data, dict) else None
        spy_ret = spy_data.get("spy_ret", None) if isinstance(spy_data, dict) else None

        # ── Emergency close check ──
        if use_emergency_close and emergency_close_check(spy_ret) and len(positions) > 0:
            for pos_id, pos in list(positions.items()):
                tk = pos["ticker"]
                S = date_px.get(tk)
                if S is None or np.isnan(S):
                    continue
                T_days = (pos["expiry"] - dt).days
                T = max(T_days, 0) / 365.0
                T_orig = pos["dte_target"] / 365.0
                sigma_cur = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
                curr_mult = decaying_premium_mult(premium_mult, T, T_orig)

                # Cost to close all 4 legs
                sp_prem = bs_price(S, pos["short_put_strike"], T, sigma_cur, kind="put") * curr_mult
                lp_prem = bs_price(S, pos["long_put_strike"], T, sigma_cur, kind="put") * curr_mult
                sc_prem = bs_price(S, pos["short_call_strike"], T, sigma_cur, kind="call") * curr_mult
                lc_prem = bs_price(S, pos["long_call_strike"], T, sigma_cur, kind="call") * curr_mult

                # To close: buy back shorts, sell longs
                cost_to_close = (sp_prem - lp_prem + sc_prem - lc_prem) * 100 * pos["contracts"]
                ba_close = (ba_cost_per_leg(sp_prem, ba_frac) +
                            ba_cost_per_leg(lp_prem, ba_frac) +
                            ba_cost_per_leg(sc_prem, ba_frac) +
                            ba_cost_per_leg(lc_prem, ba_frac)) * pos["contracts"]

                pnl = pos["net_credit_dollar"] - cost_to_close - ba_close
                cash += pnl
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": pnl, "close_reason": "emergency_close",
                    "hold_days": (dt - pos["open_date"]).days,
                })
                stats["n_emergency_closed"] += 1
                stats["n_closed"] += 1
                last_close[tk] = dt

            positions.clear()

        # ── Manage existing positions ──
        to_remove = []
        for pos_id, pos in list(positions.items()):
            tk = pos["ticker"]
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            T_orig = pos["dte_target"] / 365.0
            sigma_cur = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            if T_days <= 0:
                # ── Expiry settlement ──
                # P&L starts with full credit received
                pnl = pos["net_credit_dollar"]

                # Put side loss: stock below short put
                if S < pos["short_put_strike"]:
                    if S <= pos["long_put_strike"]:
                        # Max loss on put side (long put caps it)
                        put_loss = (pos["short_put_strike"] - pos["long_put_strike"]) * 100 * pos["contracts"]
                    else:
                        # Partial loss: between long and short put
                        put_loss = (pos["short_put_strike"] - S) * 100 * pos["contracts"]
                    pnl -= put_loss
                    stats["n_put_side_loss"] += 1

                # Call side loss: stock above short call
                if S > pos["short_call_strike"]:
                    if S >= pos["long_call_strike"]:
                        # Max loss on call side (long call caps it)
                        call_loss = (pos["long_call_strike"] - pos["short_call_strike"]) * 100 * pos["contracts"]
                    else:
                        # Partial loss: between short and long call
                        call_loss = (S - pos["short_call_strike"]) * 100 * pos["contracts"]
                    pnl -= call_loss
                    stats["n_call_side_loss"] += 1

                if pnl >= 0:
                    stats["n_expiry_win"] += 1
                else:
                    stats["n_expiry_loss"] += 1

                cash += pnl
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": pnl, "close_reason": "expiry",
                    "hold_days": (dt - pos["open_date"]).days,
                })
                to_remove.append(pos_id)
                last_close[tk] = dt
                stats["n_closed"] += 1
                continue

            # ── Profit-take / stop-loss (mark-to-market) ──
            curr_mult = decaying_premium_mult(premium_mult, T, T_orig)
            sp_prem_now = bs_price(S, pos["short_put_strike"], T, sigma_cur, kind="put") * curr_mult
            lp_prem_now = bs_price(S, pos["long_put_strike"], T, sigma_cur, kind="put") * curr_mult
            sc_prem_now = bs_price(S, pos["short_call_strike"], T, sigma_cur, kind="call") * curr_mult
            lc_prem_now = bs_price(S, pos["long_call_strike"], T, sigma_cur, kind="call") * curr_mult

            # Cost to close: buy back shorts (pay sp + sc), sell longs (receive lp + lc)
            cost_to_close = (sp_prem_now - lp_prem_now + sc_prem_now - lc_prem_now) * 100 * pos["contracts"]
            ba_close = (ba_cost_per_leg(sp_prem_now, ba_frac) +
                        ba_cost_per_leg(lp_prem_now, ba_frac) +
                        ba_cost_per_leg(sc_prem_now, ba_frac) +
                        ba_cost_per_leg(lc_prem_now, ba_frac)) * pos["contracts"]

            unrealized_pnl = pos["net_credit_dollar"] - cost_to_close - ba_close

            if unrealized_pnl >= profit_take * pos["net_credit_dollar"]:
                cash += unrealized_pnl
                stats["total_ba_cost"] += ba_close
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": unrealized_pnl, "close_reason": "profit_take",
                    "hold_days": (dt - pos["open_date"]).days,
                })
                to_remove.append(pos_id)
                last_close[tk] = dt
                stats["n_profit_take"] += 1
                stats["n_closed"] += 1
                continue

            if unrealized_pnl < -(stop_loss_mult * pos["net_credit_dollar"]):
                cash += unrealized_pnl
                stats["total_ba_cost"] += ba_close
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": unrealized_pnl, "close_reason": "stop_loss",
                    "hold_days": (dt - pos["open_date"]).days,
                })
                to_remove.append(pos_id)
                last_close[tk] = dt
                stats["n_stop_loss"] += 1
                stats["n_closed"] += 1
                continue

        for pid in to_remove:
            if pid in positions:
                del positions[pid]

        # ── Regime gates for new opens ──
        vix_mult = 1.0
        can_open = True

        if use_vix_gate:
            vix_mult = vix_tier_sizing(vix_val)
            if vix_mult <= 0:
                stats["n_vix_blocked"] += 1
                can_open = False

        if use_spy_trend and can_open:
            if not spy_trend_filter(spy_close, spy_sma20):
                stats["n_spy_blocked"] += 1
                can_open = False

        # ── Open new positions ──
        if can_open and len(positions) < max_concurrent:
            ticker_pos_count = defaultdict(int)
            for pid, pos in positions.items():
                ticker_pos_count[pos["ticker"]] += 1

            candidates = []
            for tk in ticker_list:
                if tk not in available:
                    continue
                if ticker_pos_count.get(tk, 0) >= 1:
                    continue
                S = date_px.get(tk)
                sigma = date_sigma.get(tk)
                if S is None or sigma is None or np.isnan(S) or np.isnan(sigma):
                    continue
                if sigma < 0.10:
                    continue
                if has_earnings_within(tk, dt, dte_target, earnings_lookup, earnings_buffer):
                    stats["n_earnings_blocked"] += 1
                    continue
                candidates.append((tk, S, sigma))

            # Sort by highest IV — more premium to collect
            candidates.sort(key=lambda x: x[2], reverse=True)

            for tk, S, sigma in candidates:
                if len(positions) >= max_concurrent:
                    break

                T = dte_target / 365.0

                # ── PUT SPREAD (bull put spread): sell OTM put, buy further OTM put ──
                K_short_put = strike_from_delta(S, T, sigma, put_delta, kind="put")
                K_long_put = K_short_put * (1 - long_put_offset)  # 5% further OTM (lower)

                # ── CALL SPREAD (bear call spread): sell OTM call, buy further OTM call ──
                K_short_call = strike_from_delta(S, T, sigma, call_delta, kind="call")
                K_long_call = K_short_call * (1 + long_call_offset)  # 5% further OTM (higher)

                # Price all 4 legs
                sp_prem = bs_price(S, K_short_put, T, sigma, kind="put") * premium_mult
                lp_prem = bs_price(S, K_long_put, T, sigma, kind="put") * premium_mult
                sc_prem = bs_price(S, K_short_call, T, sigma, kind="call") * premium_mult
                lc_prem = bs_price(S, K_long_call, T, sigma, kind="call") * premium_mult

                # Net credit = (short_put - long_put) + (short_call - long_call)
                net_credit_per_share = (sp_prem - lp_prem) + (sc_prem - lc_prem)
                if net_credit_per_share <= 0.05:
                    continue

                # BA cost on all 4 legs at open
                ba_open = (ba_cost_per_leg(sp_prem, ba_frac) +
                           ba_cost_per_leg(lp_prem, ba_frac) +
                           ba_cost_per_leg(sc_prem, ba_frac) +
                           ba_cost_per_leg(lc_prem, ba_frac))

                # Margin = max of the two spread widths (only one side can lose)
                put_spread_width = (K_short_put - K_long_put) * 100
                call_spread_width = (K_long_call - K_short_call) * 100
                margin_needed = max(put_spread_width, call_spread_width)

                if margin_needed <= 0:
                    continue

                # Apply VIX sizing
                sizing_base = min(cash, STARTING_CAPITAL * 1.5)
                max_alloc = sizing_base * per_name_pct * vix_mult
                contracts = max(1, int(max_alloc / margin_needed))
                if contracts * margin_needed > sizing_base * 0.15:
                    contracts = max(1, int(sizing_base * 0.15 / margin_needed))

                net_credit_dollar = net_credit_per_share * 100 * contracts
                total_ba_open = ba_open * contracts
                net_credit_dollar -= total_ba_open
                stats["total_ba_cost"] += total_ba_open

                if net_credit_dollar <= 0:
                    continue

                expiry = dt + pd.Timedelta(days=dte_target)
                pos_counter += 1
                pos_id = f"{tk}_{pos_counter}"
                positions[pos_id] = {
                    "ticker": tk, "open_date": dt, "expiry": expiry,
                    "dte_target": dte_target, "contracts": contracts,
                    "short_put_strike": K_short_put, "long_put_strike": K_long_put,
                    "short_call_strike": K_short_call, "long_call_strike": K_long_call,
                    "net_credit_dollar": net_credit_dollar,
                    "open_sigma": sigma, "open_price": S,
                }
                # NO cash += credit at open (corrected accounting — no double-count)
                stats["n_opened"] += 1
                stats["total_premium_collected"] += net_credit_dollar

        # ── Equity curve (mark-to-market) ──
        mtm = 0
        for pos_id, pos in positions.items():
            tk = pos["ticker"]
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            T_orig = pos["dte_target"] / 365.0
            sigma_cur = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            curr_mult = decaying_premium_mult(premium_mult, T, T_orig)

            sp_p = bs_price(S, pos["short_put_strike"], T, sigma_cur, kind="put") * curr_mult
            lp_p = bs_price(S, pos["long_put_strike"], T, sigma_cur, kind="put") * curr_mult
            sc_p = bs_price(S, pos["short_call_strike"], T, sigma_cur, kind="call") * curr_mult
            lc_p = bs_price(S, pos["long_call_strike"], T, sigma_cur, kind="call") * curr_mult
            cost_to_close = (sp_p - lp_p + sc_p - lc_p) * 100 * pos["contracts"]
            mtm += pos["net_credit_dollar"] - cost_to_close

        equity_curve.append({"date": dt, "equity": cash + mtm, "cash": cash, "n_pos": len(positions)})

    return equity_curve, detailed_trades, stats


# ═══════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(eq, label):
    df = pd.DataFrame(eq)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["r"] = df["equity"].pct_change().dropna()
    df = df.dropna(subset=["r"])
    if len(df) < 30:
        return {"label": label, "error": "too few"}
    rets = df["r"].values
    mu = np.mean(rets)
    sigma = np.std(rets, ddof=1)
    sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0
    ds = rets[rets < 0]
    ds_std = np.std(ds, ddof=1) if len(ds) > 1 else 1e-6
    sortino = (mu / ds_std) * np.sqrt(252) if ds_std > 0 else 0
    yrs = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
    tot_ret = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1
    cagr = (1 + tot_ret) ** (1/yrs) - 1 if yrs > 0 else 0
    dd = (df["equity"] - df["equity"].cummax()) / df["equity"].cummax()
    g = rets[rets > 0].sum()
    l = abs(rets[rets < 0].sum())
    pf = g / l if l > 0 else float("inf")
    return {
        "label": label, "sharpe": round(sharpe, 2), "sortino": round(sortino, 2),
        "cagr_pct": round(cagr * 100, 1), "max_dd_pct": round(dd.min() * 100, 1),
        "pf": round(pf, 2), "wr_pct": round(np.mean(rets > 0) * 100, 1),
        "final": round(df["equity"].iloc[-1], 0),
    }


def regime_test(eq, spy_df, label):
    df = pd.DataFrame(eq)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["r"] = df["equity"].pct_change()
    df = df.dropna(subset=["r"])
    spy = spy_df[["date", "spy_ret"]].dropna().copy()
    spy["date"] = pd.to_datetime(spy["date"])
    m = df.merge(spy, on="date", how="inner")
    if len(m) < 60:
        return {"label": label, "error": "insufficient"}
    m["regime"] = np.where(m["spy_ret"] > 0.001, "green",
                  np.where(m["spy_ret"] < -0.001, "red", "flat"))
    res = {}
    for reg in ["green", "red", "flat"]:
        sub = m[m["regime"] == reg]["r"]
        if len(sub) < 10:
            res[reg] = 0.0
            continue
        mu = sub.mean(); std = sub.std(ddof=1)
        res[reg] = round((mu / std) * np.sqrt(252), 2) if std > 0 else 0
    mx = max(abs(res["green"]), abs(res["red"]))
    gap = abs(res["green"] - res["red"]) / mx if mx > 0 else 0
    return {"label": label, "green": res["green"], "red": res["red"], "flat": res["flat"],
            "gap": round(gap, 3), "pass": gap < 0.50}


def trade_stats(trades, label):
    if not trades:
        return {"label": label, "n": 0}
    df = pd.DataFrame(trades)
    n = len(df); w = (df["pnl"] > 0).sum()
    return {
        "label": label, "n": n, "wr": round(w/n*100, 1),
        "avg_pnl": round(df["pnl"].mean(), 1),
        "total_pnl": round(df["pnl"].sum(), 0),
        "avg_hold": round(df["hold_days"].mean(), 1),
        "reasons": df["close_reason"].value_counts().to_dict(),
    }


def permutation_test(trades, n_perms=2000):
    if not trades or len(trades) < 10:
        return 1.0
    pnls = np.array([t["pnl"] for t in trades])
    actual_t = np.mean(pnls) / (np.std(pnls, ddof=1) / np.sqrt(len(pnls)))
    count = 0
    for _ in range(n_perms):
        s = pnls * np.random.choice([-1, 1], size=len(pnls))
        pt = np.mean(s) / (np.std(s, ddof=1) / np.sqrt(len(s)))
        if pt >= actual_t:
            count += 1
    return round(count / n_perms, 4)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    t0 = time.time()

    prices, iv, macro, vix_df, spy, earnings, avail = load_all_data()
    earnings_lookup = build_earnings_lookup(earnings)

    print(f"\n{'#'*80}")
    print(f"# IRON CONDOR V1 — REGIME-GATED SWEEP")
    print(f"{'#'*80}\n")

    configs = [
        # ── 1. BASELINE (no gates) ──
        {"label": "IC_BASELINE_1x_15p_21d",
         "premium_mult": 1.0, "max_concurrent": 15, "dte_target": 21,
         "put_delta": -0.20, "call_delta": 0.20,
         "use_vix_gate": False, "use_spy_trend": False, "use_emergency_close": False},

        # ── 2. VIX gate only ──
        {"label": "IC_VIX_GATE_1x_15p_21d",
         "premium_mult": 1.0, "max_concurrent": 15, "dte_target": 21,
         "put_delta": -0.20, "call_delta": 0.20,
         "use_vix_gate": True, "use_spy_trend": False, "use_emergency_close": False},

        # ── 3. SPY trend filter only ──
        {"label": "IC_SPY_TREND_1x_15p_21d",
         "premium_mult": 1.0, "max_concurrent": 15, "dte_target": 21,
         "put_delta": -0.20, "call_delta": 0.20,
         "use_vix_gate": False, "use_spy_trend": True, "use_emergency_close": False},

        # ── 4. VIX + SPY combined ──
        {"label": "IC_VIX+SPY_1x_15p_21d",
         "premium_mult": 1.0, "max_concurrent": 15, "dte_target": 21,
         "put_delta": -0.20, "call_delta": 0.20,
         "use_vix_gate": True, "use_spy_trend": True, "use_emergency_close": False},

        # ── 5. Shorter DTE (7d) + VIX gate ──
        {"label": "IC_VIX_1x_15p_7d",
         "premium_mult": 1.0, "max_concurrent": 15, "dte_target": 7,
         "put_delta": -0.20, "call_delta": 0.20,
         "use_vix_gate": True, "use_spy_trend": False, "use_emergency_close": False},

        # ── 6. Longer DTE (35d) + VIX gate ──
        {"label": "IC_VIX_1x_15p_35d",
         "premium_mult": 1.0, "max_concurrent": 15, "dte_target": 35,
         "put_delta": -0.20, "call_delta": 0.20,
         "use_vix_gate": True, "use_spy_trend": False, "use_emergency_close": False},

        # ── 7. Wide strikes (0.15 delta) + VIX gate ──
        {"label": "IC_WIDE_0.15d_VIX_15p_21d",
         "premium_mult": 1.0, "max_concurrent": 15, "dte_target": 21,
         "put_delta": -0.15, "call_delta": 0.15,
         "use_vix_gate": True, "use_spy_trend": False, "use_emergency_close": False},

        # ── 8. Narrow strikes (0.25 delta) + VIX gate ──
        {"label": "IC_NARROW_0.25d_VIX_15p_21d",
         "premium_mult": 1.0, "max_concurrent": 15, "dte_target": 21,
         "put_delta": -0.25, "call_delta": 0.25,
         "use_vix_gate": True, "use_spy_trend": False, "use_emergency_close": False},

        # ── 9. Conservative 5 positions + VIX gate ──
        {"label": "IC_VIX_1x_5p_21d",
         "premium_mult": 1.0, "max_concurrent": 5, "dte_target": 21,
         "put_delta": -0.20, "call_delta": 0.20,
         "use_vix_gate": True, "use_spy_trend": False, "use_emergency_close": False},

        # ── 10. 1.5x premium + VIX+SPY ──
        {"label": "IC_VIX+SPY_1.5x_15p_21d",
         "premium_mult": 1.5, "max_concurrent": 15, "dte_target": 21,
         "put_delta": -0.20, "call_delta": 0.20,
         "use_vix_gate": True, "use_spy_trend": True, "use_emergency_close": False},

        # ── 11. Full gate + emergency close, 10 pos ──
        {"label": "IC_FULL_GATE_VIX+SPY+EM_10p_21d",
         "premium_mult": 1.0, "max_concurrent": 10, "dte_target": 21,
         "put_delta": -0.20, "call_delta": 0.20,
         "use_vix_gate": True, "use_spy_trend": True, "use_emergency_close": True},
    ]

    results = {}
    for cfg in configs:
        label = cfg["label"]
        print(f"  Running {label}...")
        eq, trades, stats = run_iron_condor(
            prices, iv, macro, vix_df, spy, earnings_lookup, avail,
            ba_frac=0.10,
            put_delta=cfg["put_delta"], call_delta=cfg["call_delta"],
            long_put_offset=0.05, long_call_offset=0.05,
            profit_take=0.50, stop_loss_mult=2.0,
            per_name_pct=0.04, earnings_buffer=7,
            premium_mult=cfg["premium_mult"],
            max_concurrent=cfg["max_concurrent"],
            dte_target=cfg["dte_target"],
            use_vix_gate=cfg["use_vix_gate"],
            use_spy_trend=cfg["use_spy_trend"],
            use_emergency_close=cfg["use_emergency_close"],
        )

        m = compute_metrics(eq, label)
        r = regime_test(eq, spy, label)
        t = trade_stats(trades, label)
        p = permutation_test(trades)

        results[label] = {"cfg": cfg, "metrics": m, "regime": r, "trades": t, "perm_p": p, "stats": stats}

        if "error" not in m:
            print(f"    Sharpe={m['sharpe']:>5} CAGR={m['cagr_pct']:>5.1f}% "
                  f"DD={m['max_dd_pct']:>6.1f}% R1={'PASS' if r.get('pass') else 'FAIL'} "
                  f"(G={r.get('green','?')} R={r.get('red','?')} gap={r.get('gap','?')}) "
                  f"perm={p}")

    # ═══════════════════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*140}")
    print("IRON CONDOR V1 — REGIME-GATED RESULTS")
    print(f"{'='*140}")
    print(f"{'Config':<38} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>7} "
          f"{'PF':>6} {'Grn':>5} {'Red':>5} {'Gap':>6} {'R1':>5} {'Trd':>5} {'TrdWR':>6} {'p-val':>7}")
    print("-" * 140)

    for label in [c["label"] for c in configs]:
        res = results[label]
        m = res["metrics"]
        r = res["regime"]
        t = res["trades"]
        p = res["perm_p"]
        if "error" in m:
            print(f"  {label}: ERROR")
            continue
        print(f"  {label:<36} {m['sharpe']:>7} {m['sortino']:>8} {m['cagr_pct']:>6.1f}% "
              f"{m['max_dd_pct']:>6.1f}% {m['pf']:>6.2f} {r.get('green',0):>5} {r.get('red',0):>5} "
              f"{r.get('gap',0):>6.3f} {'PASS' if r.get('pass') else 'FAIL':>5} "
              f"{t.get('n',0):>5} {t.get('wr',0):>5.1f}% {p:>6.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Trade stats detail
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*100}")
    print("TRADE STATISTICS DETAIL")
    print(f"{'='*100}")
    for label in [c["label"] for c in configs]:
        res = results[label]
        t = res["trades"]
        s = res["stats"]
        print(f"\n  {label}:")
        print(f"    Opened={s['n_opened']}  Closed={s['n_closed']}  "
              f"ProfitTake={s['n_profit_take']}  StopLoss={s['n_stop_loss']}  "
              f"ExpiryWin={s['n_expiry_win']}  ExpiryLoss={s['n_expiry_loss']}")
        print(f"    PutSideLoss={s['n_put_side_loss']}  CallSideLoss={s['n_call_side_loss']}  "
              f"EarningsBlocked={s['n_earnings_blocked']}  EmergencyClose={s['n_emergency_closed']}")
        if t.get("n", 0) > 0:
            print(f"    AvgPnL=${t['avg_pnl']:.0f}  TotalPnL=${t['total_pnl']:,.0f}  "
                  f"AvgHold={t['avg_hold']:.1f}d  WR={t['wr']:.1f}%")
            if "reasons" in t:
                print(f"    Close reasons: {t['reasons']}")

    # ═══════════════════════════════════════════════════════════════
    # Income projection
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("INCOME PROJECTION ($100K)")
    print(f"{'='*80}")
    for label in [c["label"] for c in configs]:
        m = results[label]["metrics"]
        r = results[label]["regime"]
        if "error" in m:
            continue
        r1_tag = "PASS" if r.get("pass") else "FAIL"
        print(f"  [{r1_tag}] {label:<36} ${m['cagr_pct']/100*100000:>8,.0f}/yr ({m['cagr_pct']:.1f}% CAGR) "
              f"  MaxDD={m['max_dd_pct']:.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════
    def clean(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (pd.Timestamp, np.bool_)): return str(obj)
        if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [clean(x) for x in obj]
        return obj

    out = OUTPUT / "iron_condor_v1_results.json"
    with open(out, "w") as f:
        json.dump(clean(results), f, indent=2, default=str)

    print(f"\nSaved to {out}")
    print(f"Runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
