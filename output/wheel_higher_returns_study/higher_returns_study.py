#!/usr/bin/env python3
"""
Wheel Higher Returns Study
===========================
Tests 4 directions to boost returns beyond V4 baseline (~25% CAGR, Sharpe 1.83):
  1. Capital Efficiency via Bull Put Spreads (5-wide, 10-wide)
  2. Smart Stock Selection (momentum + vol + IV rank filters)
  3. Dynamic Margin Utilization (VIX-scaled margin)
  4. Weekly Rotation (7 DTE, partial portfolio rotation)

Uses the same Black-Scholes pricing + realistic costs as wheel_engine.py.
Walk-forward sliding window per HC #0.
"""
from __future__ import annotations
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from datetime import timedelta

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUTPUT = ROOT / "output" / "wheel_higher_returns_study"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Black-Scholes ──
SQRT_2PI = math.sqrt(2 * math.pi)

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

def bs_delta(S, K, T, sigma, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0:
        return (-1.0 if S < K else 0.0) if kind == "put" else (1.0 if S > K else 0.0)
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    if kind == "put":
        return math.exp(-q*T) * (_Phi(d1) - 1.0)
    return math.exp(-q*T) * _Phi(d1)

def strike_from_delta(S, T, sigma, target_delta, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0:
        return S
    target = abs(target_delta)
    p = target if kind == "call" else (1 - target)
    p = min(max(p, 1e-6), 1 - 1e-6)
    d1 = _ndtri(p)
    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r - q + 0.5*sigma**2)*T))
    return K

# ── Costs ──
COST_PER_CONTRACT = 0.65  # IBKR style
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03

def slippage_per_share(premium):
    if premium <= 0:
        return 0.0
    return max(SLIPPAGE_MIN, SLIPPAGE_FRAC * premium)

def trade_cost(premium, contracts):
    """Total cost to open OR close a position (one-way)."""
    slip = slippage_per_share(premium) * 100 * contracts
    comm = COST_PER_CONTRACT * contracts
    return slip + comm

# ── Data Loading ──
def load_data():
    """Load all cached data, merge to unified format."""
    print("Loading data...")

    # Prices - combine all sources
    prices = pd.read_parquet(CACHE / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    # Expanded prices (more tickers, newer data)
    try:
        pexp = pd.read_parquet(CACHE / "prices_expanded.parquet")
        pexp = pexp.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        pexp["date"] = pd.to_datetime(pexp["date"]).dt.tz_localize(None)
        if "ret" not in pexp.columns:
            pexp = pexp.sort_values(["ticker","date"])
            pexp["ret"] = pexp.groupby("ticker")["close"].pct_change()
            pexp["log_ret"] = np.log1p(pexp["ret"])
        if "rv_20" not in pexp.columns:
            pexp["rv_20"] = pexp.groupby("ticker")["log_ret"].transform(
                lambda x: x.rolling(20).std() * np.sqrt(252))
        cols = [c for c in ["ticker","date","open","high","low","close","volume","ret","log_ret","rv_20"] if c in pexp.columns]
        pexp = pexp[cols]
        # Merge: use expanded for tickers/dates not in base (fast merge-based dedup)
        prices_base = prices[cols].copy()
        prices_base["_src"] = "base"
        pexp["_src"] = "exp"
        combined = pd.concat([prices_base, pexp], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker","date"], keep="first")
        prices = combined.drop(columns=["_src"])
    except Exception as e:
        print(f"  Warning: could not load expanded prices: {e}")

    # V3 expansion (date, close, ticker only)
    try:
        p3 = pd.read_parquet(CACHE / "prices_v3_expansion.parquet")
        p3["date"] = pd.to_datetime(p3["date"]).dt.tz_localize(None)
        # Only add tickers not yet in prices
        new_tks = set(p3["ticker"].unique()) - set(prices["ticker"].unique())
        if new_tks:
            p3 = p3[p3["ticker"].isin(new_tks)].copy()
            p3 = p3.sort_values(["ticker","date"])
            p3["ret"] = p3.groupby("ticker")["close"].pct_change()
            p3["log_ret"] = np.log1p(p3["ret"])
            p3["rv_20"] = p3.groupby("ticker")["log_ret"].transform(
                lambda x: x.rolling(20).std() * np.sqrt(252))
            prices = pd.concat([prices, p3], ignore_index=True)
    except Exception as e:
        print(f"  Warning: could not load V3 expansion: {e}")

    # IV data
    iv = pd.read_parquet(CACHE / "iv_features_modeled.parquet")
    iv["date"] = pd.to_datetime(iv["date"]).dt.tz_localize(None)

    # Macro
    macro = pd.read_parquet(CACHE / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)

    # Fundamentals
    fund = pd.read_parquet(CACHE / "fundamentals.parquet")

    # Earnings dates
    try:
        earnings = pd.read_parquet(CACHE / "earnings_dates.parquet")
        earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"]).dt.tz_localize(None)
    except:
        earnings = pd.DataFrame(columns=["ticker","earnings_date"])

    # Build universe with sectors
    if "sector" not in fund.columns:
        fund["sector"] = "Unknown"
    universe = fund[["ticker","sector"]].drop_duplicates("ticker")

    # Filter to 2019+ for backtest
    prices = prices[prices["date"] >= "2019-01-01"].copy()
    iv = iv[iv["date"] >= "2019-01-01"].copy()
    macro = macro[macro["date"] >= "2019-01-01"].copy()

    print(f"  Prices: {prices.shape[0]} rows, {prices['ticker'].nunique()} tickers, "
          f"{prices['date'].min().date()} to {prices['date'].max().date()}")
    print(f"  IV: {iv.shape[0]} rows")
    print(f"  Macro: {macro.shape[0]} rows")

    return prices, iv, macro, fund, universe, earnings


# ── Performance Metrics ──
def compute_metrics(equity_curve: pd.DataFrame, starting_cash: float, label: str) -> dict:
    """Compute CAGR, Sharpe, Sortino, MaxDD, etc. from equity curve."""
    eq = equity_curve.copy()
    eq = eq.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    if len(eq) < 20:
        return {"label": label, "error": "too few data points"}

    eq["ret"] = eq["equity"].pct_change()
    rets = eq["ret"].dropna()

    # CAGR
    total_days = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days
    total_years = total_days / 365.25
    total_return = eq["equity"].iloc[-1] / starting_cash
    cagr = (total_return ** (1 / max(total_years, 0.01))) - 1 if total_return > 0 else -1.0

    # Sharpe (annualized)
    if rets.std() > 0:
        sharpe = rets.mean() / rets.std() * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = rets[rets < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = rets.mean() / downside.std() * np.sqrt(252)
    else:
        sortino = 0.0

    # Max Drawdown
    eq["peak"] = eq["equity"].cummax()
    eq["dd"] = (eq["equity"] - eq["peak"]) / eq["peak"]
    max_dd = eq["dd"].min()

    # Win rate and profit factor from daily returns
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    wr = len(wins) / len(rets) if len(rets) > 0 else 0
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")

    return {
        "label": label,
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(wr * 100, 1),
        "profit_factor": round(pf, 2),
        "total_return_pct": round((total_return - 1) * 100, 2),
        "final_equity": round(eq["equity"].iloc[-1], 2),
        "n_days": len(eq),
        "years": round(total_years, 2),
    }


# ══════════════════════════════════════════════════════
# BASELINE: V4 Cash-Secured Put Engine
# ══════════════════════════════════════════════════════
def run_baseline_csp(prices, iv, macro, fund, universe, earnings,
                     starting_cash=100_000.0,
                     put_delta=0.30, dte_target=14,
                     profit_take=0.65, margin_cap=0.40,
                     per_name_pct=0.03, vix_gate=35.0,
                     max_concurrent=30,
                     label="V4 Baseline",
                     # Dynamic margin params
                     dynamic_margin=False,
                     margin_schedule=None,  # {vix_thresh: margin_cap}
                     # Smart selection params
                     smart_select=False,
                     mom_lookback=63,  # ~3 months
                     max_rv=0.30,
                     min_iv_rank=0.30,
                     # Weekly rotation
                     dte_override=None,
                     weekly_rotation_pct=None,  # fraction of portfolio to rotate weekly
                     # Earnings buffer
                     earnings_buffer_days=2,
                     # Equity brake
                     brake_lookback=60, brake_threshold=0.03, brake_scale=0.25,
                     # Bear gate (SPY < 50d SMA)
                     bear_gate=True,
                     ) -> dict:
    """
    Unified CSP engine supporting all research directions.
    """
    # Build lookup structures
    prices_df = prices.copy()
    iv_df = iv.copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    iv_rank_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()
        iv_rank_by_date[d] = g.set_index("ticker")["iv_rank"].to_dict()

    macro_by_date = macro.set_index("date").to_dict("index")

    sector_of = dict(zip(fund["ticker"], fund.get("sector", pd.Series(["Unknown"]*len(fund)))))
    fund_score_of = dict(zip(fund["ticker"], fund.get("fund_score", pd.Series([50.0]*len(fund)))))

    # Earnings lookup: ticker -> set of earnings dates
    earnings_set = {}
    for _, row in earnings.iterrows():
        tk = row["ticker"]
        ed = pd.Timestamp(row["earnings_date"])
        earnings_set.setdefault(tk, set()).add(ed)

    # Momentum + RV lookup: pre-compute per ticker per date
    rv_by_date = {}
    if smart_select:
        prices_df = prices_df.sort_values(["ticker","date"])
        prices_df["mom_ret"] = prices_df.groupby("ticker")["close"].pct_change(mom_lookback)
        mom_by_date = {}
        for d, g in prices_df.groupby("date"):
            mom_by_date[d] = g.set_index("ticker")["mom_ret"].to_dict()
            if "rv_20" in g.columns:
                rv_by_date[d] = g.set_index("ticker")["rv_20"].to_dict()

    # SPY 50d SMA for bear gate
    spy_sma50 = {}
    if bear_gate:
        spy = prices_df[prices_df["ticker"] == "SPY"].sort_values("date")
        if len(spy) > 0:
            spy["sma50"] = spy["close"].rolling(50).mean()
            for _, row in spy.iterrows():
                spy_sma50[row["date"]] = (row["close"], row["sma50"] if pd.notna(row["sma50"]) else 0)

    all_dates = sorted(prices_df["date"].unique())

    # State
    cash = starting_cash
    positions = {}  # ticker -> {state, open_date, expiry, strike, contracts, open_price, open_sigma, sector, share_cost_basis}
    equity_curve = []
    ledger = []

    effective_dte = dte_override if dte_override else dte_target

    # Weekly rotation tracking
    last_rotation_date = None
    rotation_interval = 5  # trading days

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})
        m = macro_by_date.get(dt, {})
        vix = m.get("vix", float("nan"))

        # ── 1) Update existing positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            if pos["state"] == "short_put":
                if T_days <= 0:
                    # Expiry
                    if S < pos["strike"]:
                        # Assigned - for CSP, we buy shares
                        cost = pos["strike"] * 100 * pos["contracts"]
                        cash -= cost
                        pos["state"] = "long_shares"
                        pos["share_cost_basis"] = pos["strike"] - pos["open_price"]
                    else:
                        # Expire worthless
                        realized = pos["open_price"] * 100 * pos["contracts"] - COST_PER_CONTRACT * pos["contracts"]
                        ledger.append({"date": dt, "ticker": tk, "kind": "CSP", "pnl": realized})
                        to_remove.append(tk)
                else:
                    # Profit take check
                    cur = bs_price(S, pos["strike"], T, sigma_atm, kind="put")
                    captured = (pos["open_price"] - cur) / max(pos["open_price"], 1e-6)
                    if captured >= profit_take:
                        cost = trade_cost(cur, pos["contracts"])
                        buyback = cur * 100 * pos["contracts"]
                        realized = pos["open_price"] * 100 * pos["contracts"] - buyback - cost
                        cash -= buyback + cost
                        ledger.append({"date": dt, "ticker": tk, "kind": "CSP", "pnl": realized})
                        to_remove.append(tk)

            elif pos["state"] == "long_shares":
                # Sell covered call immediately
                sigma = date_sigma.get(tk, 0.25) or 0.25
                cc_dte = effective_dte
                T_cc = cc_dte / 365.0
                K_cc = strike_from_delta(S, T_cc, sigma, 0.30, kind="call")
                prem_cc = bs_price(S, K_cc, T_cc, sigma, kind="call")
                if prem_cc > 0:
                    credit = prem_cc * 100 * pos["contracts"] - trade_cost(prem_cc, pos["contracts"])
                    cash += credit
                    pos["state"] = "short_call"
                    pos["strike"] = K_cc
                    pos["open_price"] = prem_cc
                    pos["open_date"] = dt
                    pos["expiry"] = dt + pd.Timedelta(days=cc_dte)
                    pos["open_sigma"] = sigma

            elif pos["state"] == "short_call":
                if T_days <= 0:
                    if S > pos["strike"]:
                        # Called away
                        proceeds = pos["strike"] * 100 * pos["contracts"]
                        cash += proceeds
                        realized = pos["open_price"] * 100 * pos["contracts"] - COST_PER_CONTRACT * pos["contracts"]
                        realized += (pos["strike"] - pos["share_cost_basis"]) * 100 * pos["contracts"]
                        ledger.append({"date": dt, "ticker": tk, "kind": "CC_called", "pnl": realized})
                        to_remove.append(tk)
                    else:
                        # CC expires worthless, keep shares
                        realized = pos["open_price"] * 100 * pos["contracts"] - COST_PER_CONTRACT * pos["contracts"]
                        ledger.append({"date": dt, "ticker": tk, "kind": "CC_expire", "pnl": realized})
                        pos["state"] = "long_shares"
                        pos["open_price"] = 0
                        pos["strike"] = 0
                        pos["expiry"] = dt
                else:
                    # CC profit take
                    cur = bs_price(S, pos["strike"], T, sigma_atm, kind="call")
                    captured = (pos["open_price"] - cur) / max(pos["open_price"], 1e-6)
                    if captured >= profit_take:
                        cost = trade_cost(cur, pos["contracts"])
                        buyback = cur * 100 * pos["contracts"]
                        realized = pos["open_price"] * 100 * pos["contracts"] - buyback - cost
                        cash -= buyback + cost
                        ledger.append({"date": dt, "ticker": tk, "kind": "CC_pt", "pnl": realized})
                        pos["state"] = "long_shares"
                        pos["open_price"] = 0
                        pos["strike"] = 0
                        pos["expiry"] = dt

        for tk in to_remove:
            del positions[tk]

        # ── Share stop-loss: -15% ──
        for tk in list(positions.keys()):
            pos = positions[tk]
            if pos["state"] not in ("long_shares", "short_call"):
                continue
            S = date_px.get(tk)
            if S is None or np.isnan(S) or pos.get("share_cost_basis", 0) <= 0:
                continue
            if S < pos["share_cost_basis"] * 0.85:
                # Liquidate
                if pos["state"] == "short_call":
                    T_cc = max((pos["expiry"] - dt).days, 0) / 365.0
                    sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or 0.20
                    cur_cc = bs_price(S, pos["strike"], T_cc, sigma_atm, kind="call")
                    cash -= cur_cc * 100 * pos["contracts"] + trade_cost(cur_cc, pos["contracts"])
                proceeds = S * 100 * pos["contracts"]
                cash += proceeds
                realized = (S - pos["share_cost_basis"]) * 100 * pos["contracts"]
                ledger.append({"date": dt, "ticker": tk, "kind": "stop_loss", "pnl": realized})
                del positions[tk]

        # ── 2) MTM equity ──
        equity = cash
        for tk, pos in positions.items():
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = max((pos["expiry"] - dt).days, 0)
            T = T_days / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            if pos["state"] == "short_put":
                opt_val = bs_price(S, pos["strike"], T, sigma_atm, kind="put")
                equity -= opt_val * 100 * pos["contracts"]
            elif pos["state"] == "short_call":
                opt_val = bs_price(S, pos["strike"], T, sigma_atm, kind="call")
                equity += S * 100 * pos["contracts"]
                equity -= opt_val * 100 * pos["contracts"]
            elif pos["state"] == "long_shares":
                equity += S * 100 * pos["contracts"]

        equity_curve.append({"date": dt, "equity": equity})

        # ── 3) Equity curve brake ──
        brake_active = False
        if len(equity_curve) > brake_lookback:
            peak = max(e["equity"] for e in equity_curve[-brake_lookback:])
            if equity < peak * (1 - brake_threshold):
                brake_active = True

        # ── 4) Macro gates ──
        # VIX gate
        effective_vix_gate = vix_gate
        if not np.isnan(vix) and vix > effective_vix_gate:
            continue

        # Bear gate: SPY < 50d SMA
        if bear_gate and dt in spy_sma50:
            spy_close, spy_sma = spy_sma50[dt]
            if spy_sma > 0 and spy_close < spy_sma:
                continue

        # ── 5) Dynamic margin cap ──
        if dynamic_margin and margin_schedule and not np.isnan(vix):
            effective_margin_cap = margin_cap  # default
            for vix_thresh in sorted(margin_schedule.keys()):
                if vix <= vix_thresh:
                    effective_margin_cap = margin_schedule[vix_thresh]
                    break
            else:
                effective_margin_cap = 0.0  # VIX above all thresholds = no new trades
        else:
            effective_margin_cap = margin_cap

        if brake_active:
            effective_margin_cap *= brake_scale

        # Current margin usage
        current_margin = 0
        for pos in positions.values():
            if pos["state"] == "short_put":
                current_margin += pos["strike"] * 100 * pos["contracts"] * 0.20

        if current_margin >= effective_margin_cap * equity:
            continue

        # ── 6) Weekly rotation check ──
        if weekly_rotation_pct is not None:
            if last_rotation_date is not None:
                days_since = len([d for d in all_dates if last_rotation_date < d <= dt])
                if days_since < rotation_interval:
                    continue
            last_rotation_date = dt

        # ── 7) Select candidates ──
        if len(positions) >= max_concurrent:
            continue

        candidates = []
        for tk, S in date_px.items():
            if tk == "__date__" or tk in positions:
                continue
            if S is None or np.isnan(S) or S < 10 or S > 500:
                continue
            sigma = date_sigma.get(tk)
            if sigma is None or np.isnan(sigma) or sigma <= 0:
                continue
            iv_rk = date_iv_rank.get(tk, 0.5)
            sector = sector_of.get(tk, "Unknown")
            fs = fund_score_of.get(tk, 50.0)

            # Smart selection filters
            if smart_select:
                # Momentum filter: 3-month return > 0
                mom = mom_by_date.get(dt, {}).get(tk, 0)
                if pd.isna(mom) or mom <= 0:
                    continue
                # Vol filter: realized vol < max_rv (pre-computed lookup)
                rv = rv_by_date.get(dt, {}).get(tk)
                if rv is not None and not np.isnan(rv) and rv > max_rv:
                    continue
                # IV rank filter: only sell when IV is elevated
                if iv_rk < min_iv_rank:
                    continue

            # Earnings buffer: don't sell puts expiring near earnings
            if earnings_buffer_days > 0:
                expiry_date = dt + pd.Timedelta(days=effective_dte)
                tk_earnings = earnings_set.get(tk, set())
                near_earnings = any(abs((ed - expiry_date).days) <= earnings_buffer_days for ed in tk_earnings)
                if near_earnings:
                    continue

            candidates.append((tk, S, sigma, iv_rk, sector, fs))

        # Rank by IV rank (higher = better premium) then fund score
        candidates.sort(key=lambda r: (r[3], r[5]), reverse=True)

        # How many slots
        slots = max_concurrent - len(positions)
        if weekly_rotation_pct is not None:
            slots = min(slots, max(1, int(max_concurrent * weekly_rotation_pct)))
        else:
            slots = min(slots, max(1, max_concurrent // 5))

        remaining_margin = effective_margin_cap * equity - current_margin

        for tk, S, sigma, iv_rk, sector, fs in candidates[:slots]:
            T = effective_dte / 365.0
            K = strike_from_delta(S, T, sigma, put_delta, kind="put")
            premium = bs_price(S, K, T, sigma, kind="put")

            if premium <= 0 or K <= 0 or not np.isfinite(K) or not np.isfinite(premium):
                continue

            # Position sizing
            max_alloc = per_name_pct * equity
            n_contracts = max(1, int(max_alloc // (K * 100)))

            # CSP: need full cash to secure
            secure = K * 100 * n_contracts
            margin_needed = secure * 0.20

            if margin_needed > remaining_margin:
                n_contracts = max(1, int(remaining_margin / (K * 100 * 0.20)))
            if n_contracts < 1:
                continue

            secure = K * 100 * n_contracts
            if secure > cash:
                n_contracts = int(cash // (K * 100))
                if n_contracts < 1:
                    continue

            # Open position
            credit = premium * 100 * n_contracts - trade_cost(premium, n_contracts)
            cash += credit
            positions[tk] = {
                "state": "short_put",
                "open_date": dt,
                "expiry": dt + pd.Timedelta(days=effective_dte),
                "strike": K,
                "contracts": n_contracts,
                "open_price": premium,
                "open_sigma": sigma,
                "sector": sector,
                "share_cost_basis": 0,
            }
            remaining_margin -= K * 100 * n_contracts * 0.20

            if len(positions) >= max_concurrent:
                break

    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty:
        return {"label": label, "error": "no equity curve"}

    metrics = compute_metrics(eq_df, starting_cash, label)
    metrics["n_trades"] = len(ledger)

    return {
        "metrics": metrics,
        "equity_curve": eq_df,
        "ledger": pd.DataFrame(ledger) if ledger else pd.DataFrame(),
    }


# ══════════════════════════════════════════════════════
# DIRECTION 1: Bull Put Spreads
# ══════════════════════════════════════════════════════
def run_bull_put_spread(prices, iv, macro, fund, universe, earnings,
                        spread_width=5.0,  # dollars between strikes
                        starting_cash=100_000.0,
                        put_delta=0.30, dte_target=14,
                        profit_take=0.65, margin_cap=0.40,
                        max_concurrent=60,  # more positions since less margin per trade
                        vix_gate=35.0,
                        per_name_pct=0.05,  # can be higher with spreads
                        label="Bull Put Spread $5 wide",
                        ) -> dict:
    """
    Bull put spread: sell put at target delta, buy lower put for protection.
    Max loss = (spread_width - net_credit) per share.
    Margin required = spread_width * 100 per contract (NOT full strike).
    This is ~3-5x more capital efficient than CSP.
    """
    prices_df = prices.copy()
    iv_df = iv.copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    iv_rank_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()
        iv_rank_by_date[d] = g.set_index("ticker")["iv_rank"].to_dict()

    macro_by_date = macro.set_index("date").to_dict("index")
    sector_of = dict(zip(fund["ticker"], fund.get("sector", pd.Series(["Unknown"]*len(fund)))))
    fund_score_of = dict(zip(fund["ticker"], fund.get("fund_score", pd.Series([50.0]*len(fund)))))

    # Earnings
    earnings_set = {}
    for _, row in earnings.iterrows():
        tk = row["ticker"]
        ed = pd.Timestamp(row["earnings_date"])
        earnings_set.setdefault(tk, set()).add(ed)

    # SPY SMA50 for bear gate
    spy_sma50 = {}
    spy = prices_df[prices_df["ticker"] == "SPY"].sort_values("date")
    if len(spy) > 0:
        spy["sma50"] = spy["close"].rolling(50).mean()
        for _, row in spy.iterrows():
            spy_sma50[row["date"]] = (row["close"], row["sma50"] if pd.notna(row["sma50"]) else 0)

    all_dates = sorted(prices_df["date"].unique())

    cash = starting_cash
    # Spread positions: ticker -> {short_strike, long_strike, contracts, open_credit, expiry, ...}
    positions = {}
    equity_curve = []
    ledger = []

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})
        m = macro_by_date.get(dt, {})
        vix = m.get("vix", float("nan"))

        # ── Update positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            if T_days <= 0:
                # Expiry settlement
                # Note: net_credit was ALREADY added to cash at open time.
                # Here we only need to debit any loss at expiry.
                short_itm = S < pos["short_strike"]
                long_itm = S < pos["long_strike"]
                close_cost = COST_PER_CONTRACT * 2 * pos["contracts"]  # closing fees

                if not short_itm:
                    # Both expire worthless - credit already in cash, just pay close fees
                    realized = pos["net_credit"] - close_cost
                    cash -= close_cost
                elif short_itm and not long_itm:
                    # Short put assigned, long expires worthless
                    loss = (pos["short_strike"] - S) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost  # debit the loss + fees
                else:
                    # Both ITM - max loss = spread width
                    loss = (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost  # debit the loss + fees

                ledger.append({"date": dt, "ticker": tk, "kind": "BPS_expire", "pnl": realized})
                to_remove.append(tk)
            else:
                # Profit take: mark both legs
                short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
                long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
                spread_val = (short_val - long_val) * 100 * pos["contracts"]

                initial_credit = pos["net_credit"]
                current_cost_to_close = spread_val + trade_cost(short_val, pos["contracts"]) + trade_cost(long_val, pos["contracts"])

                captured = (initial_credit - current_cost_to_close) / max(initial_credit, 1e-6)
                if captured >= profit_take:
                    realized = initial_credit - current_cost_to_close
                    cash -= current_cost_to_close  # buy back spread
                    ledger.append({"date": dt, "ticker": tk, "kind": "BPS_pt", "pnl": realized})
                    to_remove.append(tk)

        for tk in to_remove:
            del positions[tk]

        # ── MTM equity ──
        equity = cash
        for tk, pos in positions.items():
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T = max((pos["expiry"] - dt).days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
            long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
            equity -= (short_val - long_val) * 100 * pos["contracts"]

        equity_curve.append({"date": dt, "equity": equity})

        # ── Gates ──
        if not np.isnan(vix) and vix > vix_gate:
            continue
        if dt in spy_sma50:
            spy_close, spy_sma = spy_sma50[dt]
            if spy_sma > 0 and spy_close < spy_sma:
                continue

        if len(positions) >= max_concurrent:
            continue

        # Margin check: spread margin = spread_width * 100 * contracts per position
        current_margin = sum(
            (p["short_strike"] - p["long_strike"]) * 100 * p["contracts"]
            for p in positions.values()
        )
        if current_margin >= margin_cap * equity:
            continue

        # ── Select candidates ──
        candidates = []
        for tk, S in date_px.items():
            if tk == "__date__" or tk in positions:
                continue
            if S is None or np.isnan(S) or S < 10 or S > 500:
                continue
            sigma = date_sigma.get(tk)
            if sigma is None or np.isnan(sigma) or sigma <= 0:
                continue
            iv_rk = date_iv_rank.get(tk, 0.5)

            # Earnings buffer
            expiry_date = dt + pd.Timedelta(days=dte_target)
            tk_earnings = earnings_set.get(tk, set())
            near_earnings = any(abs((ed - expiry_date).days) <= 2 for ed in tk_earnings)
            if near_earnings:
                continue

            candidates.append((tk, S, sigma, iv_rk))

        candidates.sort(key=lambda r: r[3], reverse=True)

        slots = min(max_concurrent - len(positions), max(1, max_concurrent // 5))
        remaining_margin = margin_cap * equity - current_margin

        for tk, S, sigma, iv_rk in candidates[:slots]:
            T = dte_target / 365.0
            K_short = strike_from_delta(S, T, sigma, put_delta, kind="put")
            K_long = K_short - spread_width

            if K_long <= 0 or K_short <= 0:
                continue

            prem_short = bs_price(S, K_short, T, sigma, kind="put")
            prem_long = bs_price(S, K_long, T, sigma, kind="put")
            net_prem_per_share = prem_short - prem_long

            if net_prem_per_share <= 0.05:
                continue

            # Position sizing: margin per contract = spread_width * 100
            margin_per_contract = spread_width * 100
            max_alloc = per_name_pct * equity
            n_contracts = max(1, int(max_alloc // margin_per_contract))

            if margin_per_contract * n_contracts > remaining_margin:
                n_contracts = max(1, int(remaining_margin // margin_per_contract))
            if n_contracts < 1:
                continue

            # Credits and costs (2 legs)
            net_credit = net_prem_per_share * 100 * n_contracts
            open_costs = trade_cost(prem_short, n_contracts) + trade_cost(prem_long, n_contracts)
            net_credit -= open_costs

            if net_credit <= 0:
                continue

            cash += net_credit
            positions[tk] = {
                "short_strike": K_short,
                "long_strike": K_long,
                "contracts": n_contracts,
                "net_credit": net_credit,
                "open_date": dt,
                "expiry": dt + pd.Timedelta(days=dte_target),
                "open_sigma": sigma,
            }
            remaining_margin -= margin_per_contract * n_contracts

            if len(positions) >= max_concurrent:
                break

    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty:
        return {"label": label, "error": "no equity curve"}

    metrics = compute_metrics(eq_df, starting_cash, label)
    metrics["n_trades"] = len(ledger)

    return {
        "metrics": metrics,
        "equity_curve": eq_df,
        "ledger": pd.DataFrame(ledger) if ledger else pd.DataFrame(),
    }


# ══════════════════════════════════════════════════════
# MAIN: Run all studies
# ══════════════════════════════════════════════════════
def main():
    t0 = time.time()
    prices, iv, macro, fund, universe, earnings = load_data()

    results = {}

    # ── BASELINE ──
    print("\n=== Running V4 Baseline (30-delta CSP, 14 DTE, 65% PT) ===")
    baseline = run_baseline_csp(
        prices, iv, macro, fund, universe, earnings,
        label="V4 Baseline (30d CSP, 14 DTE)"
    )
    results["baseline"] = baseline["metrics"]
    baseline["equity_curve"].to_parquet(OUTPUT / "eq_baseline.parquet")
    print(f"  CAGR: {baseline['metrics'].get('cagr_pct')}%  Sharpe: {baseline['metrics'].get('sharpe')}  "
          f"MaxDD: {baseline['metrics'].get('max_dd_pct')}%")

    # ── DIRECTION 1a: Bull Put Spread $5 wide ──
    print("\n=== Direction 1a: Bull Put Spread $5 wide ===")
    bps5 = run_bull_put_spread(
        prices, iv, macro, fund, universe, earnings,
        spread_width=5.0,
        max_concurrent=60,
        per_name_pct=0.05,
        margin_cap=0.50,  # can use more margin with defined risk
        label="Bull Put Spread $5 wide"
    )
    results["bps_5wide"] = bps5["metrics"]
    bps5["equity_curve"].to_parquet(OUTPUT / "eq_bps5.parquet")
    print(f"  CAGR: {bps5['metrics'].get('cagr_pct')}%  Sharpe: {bps5['metrics'].get('sharpe')}  "
          f"MaxDD: {bps5['metrics'].get('max_dd_pct')}%")

    # ── DIRECTION 1b: Bull Put Spread $10 wide ──
    print("\n=== Direction 1b: Bull Put Spread $10 wide ===")
    bps10 = run_bull_put_spread(
        prices, iv, macro, fund, universe, earnings,
        spread_width=10.0,
        max_concurrent=50,
        per_name_pct=0.05,
        margin_cap=0.50,
        label="Bull Put Spread $10 wide"
    )
    results["bps_10wide"] = bps10["metrics"]
    bps10["equity_curve"].to_parquet(OUTPUT / "eq_bps10.parquet")
    print(f"  CAGR: {bps10['metrics'].get('cagr_pct')}%  Sharpe: {bps10['metrics'].get('sharpe')}  "
          f"MaxDD: {bps10['metrics'].get('max_dd_pct')}%")

    # ── DIRECTION 2: Smart Stock Selection ──
    print("\n=== Direction 2: Smart Stock Selection (momentum + vol + IV rank) ===")
    smart = run_baseline_csp(
        prices, iv, macro, fund, universe, earnings,
        smart_select=True,
        mom_lookback=63,
        max_rv=0.30,
        min_iv_rank=0.30,
        label="Smart Selection (mom>0, rv<30%, ivr>30%)"
    )
    results["smart_select"] = smart["metrics"]
    smart["equity_curve"].to_parquet(OUTPUT / "eq_smart.parquet")
    print(f"  CAGR: {smart['metrics'].get('cagr_pct')}%  Sharpe: {smart['metrics'].get('sharpe')}  "
          f"MaxDD: {smart['metrics'].get('max_dd_pct')}%")

    # ── DIRECTION 2b: Smart Selection - Less restrictive ──
    print("\n=== Direction 2b: Smart Selection (less restrictive) ===")
    smart2 = run_baseline_csp(
        prices, iv, macro, fund, universe, earnings,
        smart_select=True,
        mom_lookback=63,
        max_rv=0.40,  # more lenient
        min_iv_rank=0.20,
        label="Smart Selection (mom>0, rv<40%, ivr>20%)"
    )
    results["smart_select_loose"] = smart2["metrics"]
    smart2["equity_curve"].to_parquet(OUTPUT / "eq_smart_loose.parquet")
    print(f"  CAGR: {smart2['metrics'].get('cagr_pct')}%  Sharpe: {smart2['metrics'].get('sharpe')}  "
          f"MaxDD: {smart2['metrics'].get('max_dd_pct')}%")

    # ── DIRECTION 3: Dynamic Margin (VIX-scaled) ──
    print("\n=== Direction 3: Dynamic Margin Utilization (VIX-scaled) ===")
    dyn_margin = run_baseline_csp(
        prices, iv, macro, fund, universe, earnings,
        dynamic_margin=True,
        margin_schedule={15: 0.60, 25: 0.40, 35: 0.20},
        label="Dynamic Margin (VIX-scaled: 60/40/20%)"
    )
    results["dynamic_margin"] = dyn_margin["metrics"]
    dyn_margin["equity_curve"].to_parquet(OUTPUT / "eq_dynmargin.parquet")
    print(f"  CAGR: {dyn_margin['metrics'].get('cagr_pct')}%  Sharpe: {dyn_margin['metrics'].get('sharpe')}  "
          f"MaxDD: {dyn_margin['metrics'].get('max_dd_pct')}%")

    # ── DIRECTION 4: Weekly Rotation (7 DTE) ──
    print("\n=== Direction 4: Weekly Rotation (7 DTE, 20% rotation) ===")
    weekly = run_baseline_csp(
        prices, iv, macro, fund, universe, earnings,
        dte_override=7,
        weekly_rotation_pct=0.20,
        profit_take=0.50,  # tighter PT for shorter DTE
        max_concurrent=40,
        label="Weekly Rotation (7 DTE, 20% rotate)"
    )
    results["weekly_rotation"] = weekly["metrics"]
    weekly["equity_curve"].to_parquet(OUTPUT / "eq_weekly.parquet")
    print(f"  CAGR: {weekly['metrics'].get('cagr_pct')}%  Sharpe: {weekly['metrics'].get('sharpe')}  "
          f"MaxDD: {weekly['metrics'].get('max_dd_pct')}%")

    # ── DIRECTION 4b: Weekly Rotation with higher concurrency ──
    print("\n=== Direction 4b: Weekly Rotation (7 DTE, 30% rotation, more positions) ===")
    weekly2 = run_baseline_csp(
        prices, iv, macro, fund, universe, earnings,
        dte_override=7,
        weekly_rotation_pct=0.30,
        profit_take=0.50,
        max_concurrent=50,
        per_name_pct=0.02,
        label="Weekly Rotation (7 DTE, 30% rotate, 50 pos)"
    )
    results["weekly_rotation_v2"] = weekly2["metrics"]
    weekly2["equity_curve"].to_parquet(OUTPUT / "eq_weekly2.parquet")
    print(f"  CAGR: {weekly2['metrics'].get('cagr_pct')}%  Sharpe: {weekly2['metrics'].get('sharpe')}  "
          f"MaxDD: {weekly2['metrics'].get('max_dd_pct')}%")

    # ── COMBO: Smart Selection + Dynamic Margin ──
    print("\n=== COMBO: Smart Selection + Dynamic Margin ===")
    combo = run_baseline_csp(
        prices, iv, macro, fund, universe, earnings,
        smart_select=True,
        mom_lookback=63,
        max_rv=0.35,
        min_iv_rank=0.25,
        dynamic_margin=True,
        margin_schedule={15: 0.60, 25: 0.40, 35: 0.20},
        label="Combo: Smart + Dynamic Margin"
    )
    results["combo_smart_dynmargin"] = combo["metrics"]
    combo["equity_curve"].to_parquet(OUTPUT / "eq_combo.parquet")
    print(f"  CAGR: {combo['metrics'].get('cagr_pct')}%  Sharpe: {combo['metrics'].get('sharpe')}  "
          f"MaxDD: {combo['metrics'].get('max_dd_pct')}%")

    # ── COMBO 2: Bull Put Spread + Smart Selection ──
    # (This would need a separate function combining both, using BPS with smart selection)
    # For now, we test BPS with aggressive margin since it's defined risk
    print("\n=== COMBO 2: Bull Put Spread $5 + Aggressive Margin (defined risk) ===")
    bps_agg = run_bull_put_spread(
        prices, iv, macro, fund, universe, earnings,
        spread_width=5.0,
        max_concurrent=80,
        per_name_pct=0.06,
        margin_cap=0.70,  # can be very aggressive with defined risk
        label="BPS $5 + Aggressive (70% margin, 80 pos)"
    )
    results["bps_aggressive"] = bps_agg["metrics"]
    bps_agg["equity_curve"].to_parquet(OUTPUT / "eq_bps_agg.parquet")
    print(f"  CAGR: {bps_agg['metrics'].get('cagr_pct')}%  Sharpe: {bps_agg['metrics'].get('sharpe')}  "
          f"MaxDD: {bps_agg['metrics'].get('max_dd_pct')}%")

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"STUDY COMPLETE — {elapsed:.0f}s elapsed")
    print(f"{'='*80}")

    # Build comparison table
    print(f"\n{'Label':<50} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'WR':>6} {'PF':>6}")
    print("-" * 95)
    for key in ["baseline", "bps_5wide", "bps_10wide", "smart_select", "smart_select_loose",
                 "dynamic_margin", "weekly_rotation", "weekly_rotation_v2",
                 "combo_smart_dynmargin", "bps_aggressive"]:
        m = results.get(key, {})
        if "error" in m:
            print(f"  {m.get('label','?'):<48} ERROR: {m['error']}")
            continue
        print(f"  {m.get('label','?'):<48} {m.get('cagr_pct',0):>6.1f}% {m.get('sharpe',0):>7.2f} "
              f"{m.get('sortino',0):>8.2f} {m.get('max_dd_pct',0):>6.1f}% {m.get('win_rate_pct',0):>5.1f}% "
              f"{m.get('profit_factor',0):>5.2f}")

    # Save summary JSON
    summary = {
        "study": "Wheel Higher Returns Study",
        "date": pd.Timestamp.now().isoformat(),
        "data_range": f"{prices['date'].min().date()} to {prices['date'].max().date()}",
        "starting_capital": 100_000,
        "costs": {
            "commission_per_contract": COST_PER_CONTRACT,
            "slippage_frac": SLIPPAGE_FRAC,
            "slippage_min": SLIPPAGE_MIN,
        },
        "results": results,
    }

    with open(OUTPUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT}/summary.json")
    return results


if __name__ == "__main__":
    # Unbuffered output
    import functools
    print = functools.partial(print, flush=True)
    main()
