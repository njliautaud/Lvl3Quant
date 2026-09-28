#!/usr/bin/env python3
"""
Jade Lizard V2 — LEAKAGE AUDIT & CORRECTED BACKTEST
=====================================================
HC #743: User flagged 88.8% CAGR / Sharpe 6.5 / 1% DD as too good to be true.

LEAKAGE FOUND:
  1. DOUBLE-COUNT PREMIUM (CRITICAL): Credit added to cash at open (line 495),
     then AGAIN at close via unrealized_pnl (line 366/380) and at expiry (line 340).
     Every winning trade gets 2x actual premium.
  2. PREMIUM MULTIPLIER ON MTM: 2.2x applied to cost-to-close uniformly.
     In reality, BS-to-market gap shrinks near expiry. Use decaying multiplier.

FIX:
  - Remove cash += credit at open. Track credit as obligation.
  - At close: cash change = credit_received - buyback_cost - ba_costs
  - Premium multiplier decays toward 1.0 as T→0 (linear interpolation)
"""

import sys, json, time, math, warnings, os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats
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
# Black-Scholes Primitives (unchanged)
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
    """
    Premium multiplier decays linearly from base_mult at open toward 1.0 at expiry.
    Rationale: BS-to-market gap is largest at entry and shrinks as options near expiry
    (near-zero BS ≈ near-zero market price, multiplier doesn't apply).
    """
    if T_original <= 0:
        return 1.0
    frac_remaining = max(0, min(1, T_remaining / T_original))
    return 1.0 + (base_mult - 1.0) * frac_remaining


# ═══════════════════════════════════════════════════════════════════
# Data Loading (same as V2)
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
        tk_px["iv_rv_ratio"] = np.where(tk_px["rv_20"] > 0.001, tk_px["sigma"] / tk_px["rv_20"], 1.15)
        tk_px["term_ratio"] = np.where(tk_px["rv_20"] > 0.001,
            tk_px["rv_60"].fillna(tk_px["rv_20"]) / tk_px["rv_20"], 1.0)
        tk_px["r_1m"] = tk_px["close"].pct_change(21)
        tk_px["pricing_source"] = "modeled_rv"
        tk_px["ticker"] = tk
        cols = ["date", "ticker", "sigma_rv", "iv_rv_ratio", "term_ratio",
                "sigma", "iv_rank", "r_1m", "pricing_source"]
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
    if "rv_20" not in prices.columns:
        prices["rv_20"] = prices.groupby("ticker")["log_ret"].transform(
            lambda x: x.rolling(20).std() * np.sqrt(252))

    iv = pd.read_parquet(CACHE / "iv_features_modeled.parquet")
    iv["date"] = pd.to_datetime(iv["date"]).dt.tz_localize(None)
    iv = iv[iv["date"] >= "2019-01-01"]

    all_needed = set(TICKERS)
    iv_tickers = set(iv["ticker"].unique())
    need_iv = (all_needed & set(prices["ticker"].unique())) - iv_tickers
    if need_iv:
        print(f"  Generating modeled IV for {len(need_iv)} tickers...")
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

    try:
        earnings = pd.read_parquet(CACHE / "earnings_dates.parquet")
        earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"]).dt.tz_localize(None)
    except:
        earnings = pd.DataFrame(columns=["ticker", "earnings_date"])

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique()) & all_needed
    avail_list = sorted([t for t in TICKERS if t in available])

    print(f"  Prices: {prices.shape[0]} rows, {prices['ticker'].nunique()} tickers")
    print(f"  IV: {iv.shape[0]} rows, {iv['ticker'].nunique()} tickers")
    print(f"  Available tickers: {len(avail_list)}")

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
    mask = (earn_dates >= hold_start) & (earn_dates <= hold_end)
    return mask.any()


# ═══════════════════════════════════════════════════════════════════
# Cost Model
# ═══════════════════════════════════════════════════════════════════

def ba_cost_per_leg(premium, ba_frac):
    return premium * (ba_frac / 2.0) * 100


def vix_size_multiplier(vix_val, strategy="jade_lizard"):
    if np.isnan(vix_val):
        return 0.75
    if strategy == "jade_lizard":
        if vix_val > 35:
            return 0.25
        elif vix_val > 30:
            return 0.50
        elif vix_val > 25:
            return 0.75
        else:
            return 1.0
    return 1.0


# ═══════════════════════════════════════════════════════════════════
# JADE LIZARD V2 — CORRECTED (no double-count, decaying premium mult)
# ═══════════════════════════════════════════════════════════════════

def run_jade_lizard_v2_corrected(prices, iv, macro, vix_df, earnings_lookup, ticker_list,
                                  ba_frac=0.10, dte_target=35, put_delta=-0.25, call_delta=0.25,
                                  long_call_offset=0.05, profit_take=0.50, stop_loss_mult=2.0,
                                  max_concurrent=25, per_name_pct=0.04, vix_hard_cutoff=35,
                                  earnings_buffer=7, premium_mult=1.0,
                                  allow_reentry=True, reentry_cooldown=0,
                                  use_decaying_mult=True):
    """
    CORRECTED Jade Lizard V2.

    FIXES vs original:
      1. NO cash += credit at open. Credit is tracked as obligation.
         At close: cash += credit - buyback_cost - ba_costs (single count).
      2. Premium multiplier decays toward 1.0 as T→0 (optional).
    """
    print(f"\n{'='*70}")
    print(f"JADE LIZARD V2 CORRECTED (DTE={dte_target}, BA={ba_frac*100:.0f}%, "
          f"prem_mult={premium_mult}x{'_decay' if use_decaying_mult else ''}, max_pos={max_concurrent})")
    print(f"{'='*70}")

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

    all_dates = sorted(prices_df["date"].unique())

    cash = float(STARTING_CAPITAL)
    positions = {}
    equity_curve = []
    detailed_trades = []
    last_close = {}
    pos_counter = 0
    stats = {"n_opened": 0, "n_closed": 0, "n_profit_take": 0, "n_stop_loss": 0,
             "n_expiry_win": 0, "n_expiry_loss": 0, "n_assigned_put": 0,
             "n_earnings_blocked": 0, "n_vix_blocked": 0, "total_ba_cost": 0.0,
             "total_premium_collected": 0.0, "total_premium_paid": 0.0}

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        m_data = macro_by_date.get(dt, {})
        vix = m_data.get("vix", float("nan")) if isinstance(m_data, dict) else float("nan")
        try:
            vix_val = float(vix)
        except:
            vix_val = float("nan")

        # ── Mark-to-market and manage existing positions ──
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

            # Get current premium multiplier (decaying or fixed)
            if use_decaying_mult:
                curr_mult = decaying_premium_mult(premium_mult, T, T_orig)
            else:
                curr_mult = premium_mult

            if T_days <= 0:
                # ── Expiry settlement ──
                put_itm = S < pos["put_strike"]
                call_itm = S > pos["short_call_strike"]

                # FIX: P&L = credit received - assignment losses (single count)
                # Credit was NOT added to cash at open, so add it now minus losses
                pnl = pos["net_credit_dollar"]

                if put_itm:
                    assign_loss = (pos["put_strike"] - S) * 100 * pos["contracts"]
                    pnl -= assign_loss
                    stats["n_assigned_put"] += 1

                if call_itm:
                    if S >= pos["long_call_strike"]:
                        call_loss = (pos["long_call_strike"] - pos["short_call_strike"]) * 100 * pos["contracts"]
                    else:
                        call_loss = (S - pos["short_call_strike"]) * 100 * pos["contracts"]
                    pnl -= call_loss

                if pnl >= 0:
                    stats["n_expiry_win"] += 1
                else:
                    stats["n_expiry_loss"] += 1

                # FIX: This is the ONLY time credit enters cash (at settlement)
                cash += pnl
                stats["total_premium_paid"] += max(-pnl, 0)
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": pnl, "close_reason": "expiry",
                    "put_itm": put_itm, "call_itm": call_itm,
                    "hold_days": (dt - pos["open_date"]).days,
                })
                to_remove.append(pos_id)
                last_close[tk] = dt
                stats["n_closed"] += 1
                continue

            # ── Mid-life: profit-take / stop-loss ──
            put_prem_now = bs_price(S, pos["put_strike"], T, sigma_cur, kind="put") * curr_mult
            short_call_prem_now = bs_price(S, pos["short_call_strike"], T, sigma_cur, kind="call") * curr_mult
            long_call_prem_now = bs_price(S, pos["long_call_strike"], T, sigma_cur, kind="call") * curr_mult
            current_cost_to_close = (put_prem_now + short_call_prem_now - long_call_prem_now) * 100 * pos["contracts"]
            ba_close = (ba_cost_per_leg(put_prem_now, ba_frac) +
                        ba_cost_per_leg(short_call_prem_now, ba_frac) +
                        ba_cost_per_leg(long_call_prem_now, ba_frac)) * pos["contracts"]

            # FIX: unrealized P&L = credit - cost_to_close - ba
            unrealized_pnl = pos["net_credit_dollar"] - current_cost_to_close - ba_close

            if unrealized_pnl >= profit_take * pos["net_credit_dollar"]:
                # FIX: Cash gets credit - buyback cost (single count)
                # credit was NOT in cash, so: cash += credit - buyback - ba
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

        # ── Open new positions ──
        if len(positions) < max_concurrent and not np.isnan(vix_val):
            if vix_val >= vix_hard_cutoff:
                stats["n_vix_blocked"] += 1
            else:
                size_mult = vix_size_multiplier(vix_val, "jade_lizard")
                if size_mult > 0:
                    ticker_pos_count = defaultdict(int)
                    for pid, pos in positions.items():
                        ticker_pos_count[pos["ticker"]] += 1

                    candidates = []
                    for tk in ticker_list:
                        if tk not in available:
                            continue
                        if ticker_pos_count.get(tk, 0) >= 1:
                            continue
                        if reentry_cooldown > 0 and tk in last_close:
                            days_since = (dt - last_close[tk]).days
                            if days_since < reentry_cooldown:
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

                    candidates.sort(key=lambda x: x[2], reverse=True)

                    for tk, S, sigma in candidates:
                        if len(positions) >= max_concurrent:
                            break

                        T = dte_target / 365.0

                        K_put = strike_from_delta(S, T, sigma, put_delta, kind="put")
                        K_short_call = strike_from_delta(S, T, sigma, call_delta, kind="call")
                        K_long_call = K_short_call * (1 + long_call_offset)

                        # BS premiums × premium_mult at entry (full multiplier)
                        put_prem = bs_price(S, K_put, T, sigma, kind="put") * premium_mult
                        short_call_prem = bs_price(S, K_short_call, T, sigma, kind="call") * premium_mult
                        long_call_prem = bs_price(S, K_long_call, T, sigma, kind="call") * premium_mult

                        net_credit_per_share = put_prem + short_call_prem - long_call_prem
                        if net_credit_per_share <= 0.10:
                            continue

                        ba_open = (ba_cost_per_leg(put_prem, ba_frac) +
                                   ba_cost_per_leg(short_call_prem, ba_frac) +
                                   ba_cost_per_leg(long_call_prem, ba_frac))

                        put_margin = K_put * 100
                        call_spread_width = (K_long_call - K_short_call) * 100
                        margin_needed = max(put_margin, call_spread_width)

                        sizing_base = min(cash, STARTING_CAPITAL * 1.5)
                        max_alloc = sizing_base * per_name_pct * size_mult
                        contracts = max(1, int(max_alloc / margin_needed))
                        if contracts * margin_needed > sizing_base * 0.15:
                            contracts = max(1, int(sizing_base * 0.15 / margin_needed))

                        net_credit_dollar = net_credit_per_share * 100 * contracts
                        total_ba_open = ba_open * contracts
                        net_credit_dollar -= total_ba_open
                        stats["total_ba_cost"] += total_ba_open

                        if net_credit_dollar <= 0:
                            continue

                        call_spread_risk = call_spread_width * contracts
                        jade_condition = net_credit_dollar > call_spread_risk

                        expiry = dt + pd.Timedelta(days=dte_target)
                        pos_counter += 1
                        pos_id = f"{tk}_{pos_counter}"
                        positions[pos_id] = {
                            "ticker": tk,
                            "open_date": dt,
                            "expiry": expiry,
                            "dte_target": dte_target,  # store original DTE for decay calc
                            "contracts": contracts,
                            "put_strike": K_put,
                            "short_call_strike": K_short_call,
                            "long_call_strike": K_long_call,
                            "net_credit_dollar": net_credit_dollar,
                            "open_sigma": sigma,
                            "jade_condition": jade_condition,
                            "open_price": S,
                        }
                        # FIX: DO NOT add credit to cash at open
                        # cash += net_credit_dollar  ← REMOVED (was the double-count bug)
                        stats["n_opened"] += 1
                        stats["total_premium_collected"] += net_credit_dollar

        # ── Equity curve ──
        # FIX: Equity = cash + sum(credit - cost_to_close) for open positions
        # Since credit is NOT in cash, we need: cash + sum(credit) - sum(cost_to_close)
        mtm = 0
        for pos_id, pos in positions.items():
            tk = pos["ticker"]
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                # If no price, assume break-even (credit - credit = 0)
                mtm += 0
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            T_orig = pos["dte_target"] / 365.0
            sigma_cur = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            if use_decaying_mult:
                curr_mult = decaying_premium_mult(premium_mult, T, T_orig)
            else:
                curr_mult = premium_mult

            put_prem = bs_price(S, pos["put_strike"], T, sigma_cur, kind="put") * curr_mult
            sc_prem = bs_price(S, pos["short_call_strike"], T, sigma_cur, kind="call") * curr_mult
            lc_prem = bs_price(S, pos["long_call_strike"], T, sigma_cur, kind="call") * curr_mult
            cost_to_close = (put_prem + sc_prem - lc_prem) * 100 * pos["contracts"]

            # Position value = credit received - current cost to close
            pos_value = pos["net_credit_dollar"] - cost_to_close
            mtm += pos_value

        total_equity = cash + mtm
        equity_curve.append({"date": dt, "equity": total_equity, "cash": cash, "n_pos": len(positions)})

    return equity_curve, detailed_trades, stats


# ═══════════════════════════════════════════════════════════════════
# Analysis functions (unchanged)
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(equity_curve, label="Strategy"):
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["daily_ret"] = df["equity"].pct_change()
    df = df.dropna(subset=["daily_ret"])

    if len(df) < 30:
        return {"label": label, "error": "too few data points"}

    rets = df["daily_ret"].values
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)

    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-6
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    total_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    years = total_days / 365.25
    total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0

    cum_max = df["equity"].cummax()
    drawdown = (df["equity"] - cum_max) / cum_max
    max_dd = drawdown.min()

    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")
    wr = np.mean(rets > 0)

    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["daily_ret"].sum()
    avg_monthly_ret = monthly.mean() * 100
    monthly_wr = (monthly > 0).mean() * 100

    df["week"] = df["date"].dt.to_period("W")
    weekly = df.groupby("week")["daily_ret"].sum()
    avg_weekly_ret = weekly.mean() * 100

    return {
        "label": label,
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "cagr_pct": round(cagr * 100, 1),
        "max_dd_pct": round(max_dd * 100, 1),
        "total_return_pct": round(total_return * 100, 1),
        "pf": round(pf, 2),
        "daily_wr_pct": round(wr * 100, 1),
        "avg_monthly_ret_pct": round(avg_monthly_ret, 2),
        "monthly_wr_pct": round(monthly_wr, 1),
        "avg_weekly_ret_pct": round(avg_weekly_ret, 3),
        "n_days": len(df),
        "start": str(df["date"].iloc[0].date()),
        "end": str(df["date"].iloc[-1].date()),
        "final_equity": round(df["equity"].iloc[-1], 2),
    }


def regime_test(equity_curve, spy_df, label="Strategy"):
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["daily_ret"] = df["equity"].pct_change()
    df = df.dropna(subset=["daily_ret"])

    spy = spy_df.copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy["spy_ret"] = spy["close"].pct_change()
    spy = spy[["date", "spy_ret"]].dropna()

    merged = df.merge(spy, on="date", how="inner")
    if len(merged) < 60:
        return {"label": label, "error": "insufficient data"}

    merged["regime"] = np.where(merged["spy_ret"] > 0.001, "green",
                       np.where(merged["spy_ret"] < -0.001, "red", "flat"))

    results = {}
    for regime in ["green", "red", "flat"]:
        subset = merged[merged["regime"] == regime]["daily_ret"]
        if len(subset) < 10:
            results[regime] = {"sharpe": 0, "n_days": len(subset)}
            continue
        mean_r = subset.mean()
        std_r = subset.std(ddof=1)
        sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0
        results[regime] = {"sharpe": round(sharpe, 2), "n_days": len(subset)}

    s_green = results.get("green", {}).get("sharpe", 0)
    s_red = results.get("red", {}).get("sharpe", 0)
    max_abs = max(abs(s_green), abs(s_red))
    r1_gap = abs(s_green - s_red) / max_abs if max_abs > 0 else 0

    return {
        "label": label,
        "green_sharpe": s_green,
        "red_sharpe": s_red,
        "flat_sharpe": results.get("flat", {}).get("sharpe", 0),
        "r1_gap": round(r1_gap, 3),
        "r1_pass": r1_gap < 0.50,
    }


def trade_analysis(trades, label="Strategy"):
    if not trades:
        return {"label": label, "error": "no trades"}
    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df["pnl"] > 0).sum()
    losses = (df["pnl"] <= 0).sum()
    wr = wins / n
    avg_win = df[df["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
    avg_loss = df[df["pnl"] <= 0]["pnl"].mean() if losses > 0 else 0
    pf = abs(df[df["pnl"] > 0]["pnl"].sum() / df[df["pnl"] <= 0]["pnl"].sum()) if losses > 0 and df[df["pnl"] <= 0]["pnl"].sum() != 0 else float("inf")
    close_reasons = df["close_reason"].value_counts().to_dict()
    avg_hold = df["hold_days"].mean() if "hold_days" in df.columns else 0
    trades_per_year = n / max((df["close_date"].max() - df["open_date"].min()).days / 365.25, 0.1) if n > 0 else 0

    return {
        "label": label,
        "n_trades": n,
        "trade_wr_pct": round(wr * 100, 1),
        "profit_factor": round(pf, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_pnl": round(df["pnl"].sum(), 2),
        "avg_hold_days": round(avg_hold, 1),
        "trades_per_year": round(trades_per_year, 0),
        "close_reasons": close_reasons,
    }


def permutation_test(trades, n_perms=2000, label="Strategy"):
    if not trades or len(trades) < 10:
        return {"label": label, "error": "insufficient trades"}
    pnls = np.array([t["pnl"] for t in trades])
    actual_mean = np.mean(pnls)
    actual_t = actual_mean / (np.std(pnls, ddof=1) / np.sqrt(len(pnls)))
    perm_ts = []
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(pnls))
        shuffled = pnls * signs
        perm_mean = np.mean(shuffled)
        perm_t = perm_mean / (np.std(shuffled, ddof=1) / np.sqrt(len(shuffled)))
        perm_ts.append(perm_t)
    p_value = np.mean(np.array(perm_ts) >= actual_t)
    return {
        "label": label,
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "n_trades": len(pnls),
        "avg_pnl": round(actual_mean, 2),
    }


def sub_period_test(equity_curve, label="Strategy", n_periods=3):
    """Split equity curve into n_periods and compute Sharpe for each. CV < 0.50 = PASS."""
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["daily_ret"] = df["equity"].pct_change()
    df = df.dropna(subset=["daily_ret"])

    if len(df) < n_periods * 30:
        return {"label": label, "error": "too few data points for sub-period"}

    chunk_size = len(df) // n_periods
    sharpes = []
    for i in range(n_periods):
        start = i * chunk_size
        end = start + chunk_size if i < n_periods - 1 else len(df)
        chunk = df.iloc[start:end]
        rets = chunk["daily_ret"].values
        mean_r = np.mean(rets)
        std_r = np.std(rets, ddof=1)
        s = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0
        sharpes.append(round(s, 2))

    cv = np.std(sharpes) / abs(np.mean(sharpes)) if np.mean(sharpes) != 0 else 999
    return {
        "label": label,
        "period_sharpes": sharpes,
        "cv": round(cv, 3),
        "pass": cv < 0.50,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN — Compare ORIGINAL (buggy) vs CORRECTED
# ═══════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    t0 = time.time()

    prices, iv, macro, vix_df, spy, earnings, avail_tickers = load_all_data()
    earnings_lookup = build_earnings_lookup(earnings)

    print(f"\nData loaded in {time.time()-t0:.1f}s")
    print(f"\n{'#'*80}")
    print(f"# JADE LIZARD V2 — LEAKAGE AUDIT (HC #743)")
    print(f"# Comparing ORIGINAL (double-count) vs CORRECTED (single-count)")
    print(f"{'#'*80}\n")

    configs = [
        # The "star" config that showed 88.8% CAGR
        {"label": "CORRECTED_2.2x_25pos_7d_decay",
         "premium_mult": 2.2, "max_concurrent": 25, "dte_target": 7,
         "use_decaying_mult": True},

        # Same but without premium decay (fixed 2.2x throughout)
        {"label": "CORRECTED_2.2x_25pos_7d_fixed",
         "premium_mult": 2.2, "max_concurrent": 25, "dte_target": 7,
         "use_decaying_mult": False},

        # Conservative (1x BS, no inflated premiums)
        {"label": "CORRECTED_1x_25pos_7d",
         "premium_mult": 1.0, "max_concurrent": 25, "dte_target": 7,
         "use_decaying_mult": False},

        # Full V2 with longer hold
        {"label": "CORRECTED_2.2x_25pos_21d_decay",
         "premium_mult": 2.2, "max_concurrent": 25, "dte_target": 21,
         "use_decaying_mult": True},

        {"label": "CORRECTED_2.2x_25pos_35d_decay",
         "premium_mult": 2.2, "max_concurrent": 25, "dte_target": 35,
         "use_decaying_mult": True},

        # Baseline: V1 equivalent
        {"label": "CORRECTED_1x_5pos_35d",
         "premium_mult": 1.0, "max_concurrent": 5, "dte_target": 35,
         "use_decaying_mult": False},

        # Conservative 1.5x
        {"label": "CORRECTED_1.5x_25pos_21d_decay",
         "premium_mult": 1.5, "max_concurrent": 25, "dte_target": 21,
         "use_decaying_mult": True},

        # 1x at scale (pure BS, max positions)
        {"label": "CORRECTED_1x_25pos_21d",
         "premium_mult": 1.0, "max_concurrent": 25, "dte_target": 21,
         "use_decaying_mult": False},
    ]

    all_results = {}

    for cfg in configs:
        label = cfg["label"]
        pm = cfg["premium_mult"]
        mc = cfg["max_concurrent"]
        dte = cfg["dte_target"]
        decay = cfg.get("use_decaying_mult", True)

        eq, trades, stats = run_jade_lizard_v2_corrected(
            prices, iv, macro, vix_df, earnings_lookup, avail_tickers,
            premium_mult=pm, max_concurrent=mc, dte_target=dte,
            ba_frac=0.10, put_delta=-0.25, call_delta=0.25,
            long_call_offset=0.05, profit_take=0.50, stop_loss_mult=2.0,
            per_name_pct=0.04, earnings_buffer=7,
            use_decaying_mult=decay,
        )

        metrics = compute_metrics(eq, label)
        regime = regime_test(eq, spy, label)
        trd = trade_analysis(trades, label)
        perm = permutation_test(trades, n_perms=2000, label=label)
        sub = sub_period_test(eq, label)

        all_results[label] = {
            "config": {"premium_mult": pm, "max_concurrent": mc, "dte_target": dte,
                       "decaying_mult": decay},
            "metrics": metrics,
            "regime": regime,
            "trade_stats": trd,
            "permutation": perm,
            "sub_period": sub,
            "engine_stats": {k: v for k, v in stats.items() if not isinstance(v, (dict, list))},
        }

        m = metrics
        if "error" not in m:
            print(f"\n  {label}:")
            print(f"    Sharpe={m['sharpe']} Sortino={m['sortino']} "
                  f"CAGR={m['cagr_pct']}% MaxDD={m['max_dd_pct']}%")
            print(f"    PF={m['pf']} WR={m['daily_wr_pct']}% "
                  f"MonthlyRet={m['avg_monthly_ret_pct']}%")
            print(f"    Trades={trd.get('n_trades','?')} TradeWR={trd.get('trade_wr_pct','?')}% "
                  f"AvgHold={trd.get('avg_hold_days','?')}d")
            print(f"    R1_gap={regime.get('r1_gap','?')} R1={'PASS' if regime.get('r1_pass') else 'FAIL'} "
                  f"Perm_p={perm.get('p_value','?')}")
            print(f"    SubPeriod: {sub.get('period_sharpes','?')} CV={sub.get('cv','?')} "
                  f"{'PASS' if sub.get('pass') else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════════
    # LEAKAGE AUDIT REPORT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*120}")
    print("JADE LIZARD V2 — LEAKAGE AUDIT RESULTS (HC #743)")
    print(f"{'='*120}")
    print(f"\nLEAKAGE FOUND:")
    print(f"  1. DOUBLE-COUNT PREMIUM (CRITICAL): Credit added to cash at open AND again at close/expiry.")
    print(f"     Every winning trade got 2x actual premium. This is the primary inflation source.")
    print(f"  2. CONSTANT PREMIUM MULTIPLIER: 2.2x applied to cost-to-close uniformly.")
    print(f"     In reality, BS-to-market gap shrinks near expiry. Now using decaying multiplier.")
    print(f"  3. OTHER CONCERNS (not fixed, noted):")
    print(f"     - IV approximation (RV_20 * 1.15, not real IV)")
    print(f"     - Perfect fills (no partial fills, no market impact)")
    print(f"     - Fixed bid-ask % (real spreads vary by name, time, moneyness)")
    print(f"\nCORRECTED RESULTS:")

    print(f"\n{'Config':<40} {'Prem':>5} {'Pos':>4} {'DTE':>4} {'Sharpe':>7} {'Sortino':>8} "
          f"{'CAGR%':>7} {'MaxDD%':>7} {'PF':>6} {'TrdWR%':>7} {'R1':>5} {'p-val':>7} {'SubP':>5}")
    print("-" * 120)

    for label in [c["label"] for c in configs]:
        r = all_results[label]
        c = r["config"]
        m = r["metrics"]
        rg = r["regime"]
        t = r["trade_stats"]
        p = r["permutation"]
        sp = r["sub_period"]
        if "error" in m:
            print(f"  {label}: ERROR: {m['error']}")
            continue
        print(f"  {label:<38} {c['premium_mult']:>5.1f} {c['max_concurrent']:>4} {c['dte_target']:>4} "
              f"{m['sharpe']:>7} {m['sortino']:>8} {m['cagr_pct']:>6.1f}% {m['max_dd_pct']:>6.1f}% "
              f"{m['pf']:>6.2f} {t.get('trade_wr_pct',0):>6.1f}% "
              f"{'PASS' if rg.get('r1_pass') else 'FAIL':>5} {p.get('p_value',1):>6.4f} "
              f"{'PASS' if sp.get('pass') else 'FAIL':>5}")

    # Income projection
    print(f"\n\n{'='*80}")
    print("HONEST INCOME PROJECTION ($100K portfolio)")
    print(f"{'='*80}")
    for label in [c["label"] for c in configs]:
        m = all_results[label]["metrics"]
        if "error" in m:
            continue
        cagr = m["cagr_pct"]
        monthly = m["avg_monthly_ret_pct"]
        print(f"  {label:<40} Monthly: ${monthly/100*100000:>8,.0f}  "
              f"Annual: ${cagr/100*100000:>9,.0f}  ({cagr:.1f}% CAGR)")

    # Save
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        return obj

    def clean_dict(d):
        if isinstance(d, dict): return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list): return [clean_dict(x) for x in d]
        else: return make_serializable(d)

    output_file = OUTPUT / "jade_lizard_v2_leakage_audit.json"
    with open(output_file, "w") as f:
        json.dump(clean_dict(all_results), f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
