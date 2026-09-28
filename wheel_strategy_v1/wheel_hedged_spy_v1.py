#!/usr/bin/env python3
"""
wheel_hedged_spy_v1.py - Tier2_Balanced_Scalp on SPY with three hedge-overlay
variants, evaluated against HC #428 deploy gates.

Builds on wheel_expansion_qqq_iwm_v1.py mechanics; adds an overlay book that
holds SPY puts and/or VIX calls in parallel with the wheel.

Variants (per task spec):
  1) BASELINE       - SPY wheel only (no hedge)
  2) STATIC_PUT     - Always hold 1 long SPY put, ~90 DTE, delta ~-0.10,
                      rolled when DTE < 30.
  3) VIX_CALL_COND  - When VIX > 20, buy 1 long ~30 DTE +0.20 delta VIX call,
                      roll monthly, close when VIX drops below 16.
  4) PUTSPREAD_COLL - When VIX > 20 OR SPY 50d MA < 200d MA, overlay a 90 DTE
                      put-spread (-10 long, -5 short delta).

For each variant we compute the HC #428 gate set on the combined book.

Outputs:
  - MLflow experiment "wheel_hedged_spy_v1" (4 runs)
  - /home/jupiter/Lvl3Quant/research/findings/wheel_hedged_spy_v1.md
  - /home/jupiter/Lvl3Quant/output/wheel_hedged_spy_v1/{equity,ledger}_{variant}.parquet

Pricing caveats (honest):
  - SPY options: BS with modeled ATM sigma. No skew. The -0.10 delta long put
    will therefore be UNDERPRICED vs real chain (skew premium not modeled),
    so hedge cost reported here is a LOWER BOUND on the real drag.
  - VIX options: BS on VIX with a constant vol-of-vol of 1.10 (Goldman 2020-24
    avg VVIX/VIX-implied vol approximation), r=0. This is rough but VIX option
    pricing is path-dependent and a proper LMM model is out of scope.

Run:
  conda activate ray311 && python wheel_hedged_spy_v1.py
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------ paths -------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "wheel_hedged_spy_v1"
REPORT_DIR = ROOT / "research" / "findings"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------ wheel config ------------------------------
TIER2_BALANCED_SCALP = {
    "tier_name": "Tier2_Balanced_Scalp",
    "put_delta_target": 0.22,
    "call_delta_target": 0.22,
    "dte_min": 30,
    "dte_max": 45,
    "dte_target": 37,
    "profit_take_pct": 0.50,
    "roll_dte_trigger": 10,
    "vix_max_gate": 32.0,
    "leverage": 1.0,
}

STARTING_CASH = 50_000.0  # SPY at $237-680 requires >=$22K collateral per contract;
                          # $20K spec is infeasible (0 contracts). $50K matches small-
                          # account scale while permitting 1-2 SPY contracts.
RISK_FREE = 0.04
TRADING_DAYS = 252

SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN_PER_SHARE = 0.03
COST_PER_CONTRACT = 0.65

START_DATE = pd.Timestamp("2018-01-01")
END_DATE = pd.Timestamp("2025-12-31")

# Hedge configs
HEDGE_STATIC_PUT = {
    "name": "STATIC_PUT",
    "type": "static_put",
    "target_dte": 90,
    "roll_dte": 30,
    "delta_target": 0.10,   # |delta| of -0.10 put
}
HEDGE_VIX_CALL = {
    "name": "VIX_CALL_COND",
    "type": "vix_call",
    "vix_on": 20.0,         # buy when VIX above
    "vix_off": 16.0,        # close when VIX drops below
    "target_dte": 30,
    "delta_target": 0.20,
    "roll_monthly": True,
    "vix_vol": 1.10,        # vol-of-VIX for BS pricing
}
HEDGE_PUTSPREAD_COLLAR = {
    "name": "PUTSPREAD_COLLAR",
    "type": "putspread_collar",
    "vix_on": 20.0,
    "trend_filter": True,   # also activate when 50d MA < 200d MA
    "target_dte": 90,
    "roll_dte": 30,
    "long_delta": 0.10,
    "short_delta": 0.05,
}

# ------------------------------ BS helpers --------------------------------
SQRT_2PI = math.sqrt(2.0 * math.pi)


def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if kind == "put":
            return max(K - S, 0.0)
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
    return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)


def strike_from_delta(S, T, sigma, target_delta, kind="put", r=RISK_FREE):
    """Solve K such that |delta| = target_delta using Acklam inverse normal."""
    if T <= 0 or sigma <= 0:
        return S
    p = (1.0 - abs(target_delta)) if kind == "put" else abs(target_delta)
    p = min(max(p, 1e-9), 1 - 1e-9)
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d_ = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
          3.754408661907416]
    pl, pu = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d_[0]*q+d_[1])*q+d_[2])*q+d_[3])*q+1)
    elif p <= pu:
        q = p - 0.5
        rr = q*q
        z = (((((a[0]*rr+a[1])*rr+a[2])*rr+a[3])*rr+a[4])*rr+a[5])*q / (((((b[0]*rr+b[1])*rr+b[2])*rr+b[3])*rr+b[4])*rr+1)
    else:
        q = math.sqrt(-2 * math.log(1-p))
        z = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d_[0]*q+d_[1])*q+d_[2])*q+d_[3])*q+1)
    d1 = z
    K = S * math.exp((r + 0.5 * sigma * sigma) * T - d1 * sigma * math.sqrt(T))
    K = round(K, 2)
    return max(0.01, K)


def slippage_per_share(premium):
    if premium is None or premium <= 0:
        return 0.0
    return max(SLIPPAGE_MIN_PER_SHARE, SLIPPAGE_FRAC * premium)


# ----------------------------- state objs ---------------------------------
@dataclass
class Position:
    underlying: str
    side: str            # 'short_put', 'long_shares', 'short_call'
    strike: float
    expiry: pd.Timestamp
    open_date: pd.Timestamp
    open_price: float
    contracts: int
    open_sigma: float
    max_profit: float = 0.0


@dataclass
class HedgePosition:
    """A long-only hedge leg. We just pay premium and mark-to-market."""
    kind: str            # 'long_spy_put', 'long_vix_call', 'long_spy_put_l', 'short_spy_put_s'
    strike: float
    expiry: pd.Timestamp
    open_date: pd.Timestamp
    open_price: float    # premium paid per share (or per VIX point)
    contracts: int
    side_sign: int       # +1 long, -1 short (for spread short leg)


@dataclass
class State:
    cash: float
    positions: list = field(default_factory=list)
    hedges: list = field(default_factory=list)
    ledger: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    # hedge accounting
    hedge_cash_paid: float = 0.0
    hedge_cash_received: float = 0.0


# ----------------------------- data ---------------------------------------
def load_data():
    sec = pd.read_parquet(CACHE / "sector_etfs.parquet")
    spy = sec[sec["ticker"] == "SPY"][["date", "close"]].copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")

    iv = pd.read_parquet(CACHE / "iv_features.parquet")
    iv = iv[iv["ticker"] == "SPY"][["date", "sigma", "iv_rank"]].copy()
    iv["date"] = pd.to_datetime(iv["date"])
    iv = iv.sort_values("date").set_index("date")

    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.sort_values("date").set_index("date")

    df = spy.join(iv, how="inner").join(macro, how="left")
    df["vix"] = df["vix"].ffill()
    df = df.loc[START_DATE:END_DATE].dropna(subset=["close", "sigma"])
    df["sigma"] = df["sigma"].clip(lower=0.05, upper=1.5)
    df["spy_ret"] = df["close"].pct_change()
    # moving averages for trend filter
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    return df


# ------------------------- wheel core (SPY) -------------------------------
def find_expiry(open_date, cfg):
    best, best_dist = None, 10_000
    for d_off in range(cfg["dte_min"], cfg["dte_max"] + 1):
        cand = open_date + pd.Timedelta(days=d_off)
        shift = (4 - cand.weekday()) % 7
        cand_fri = cand + pd.Timedelta(days=shift)
        dte = (cand_fri - open_date).days
        if dte < cfg["dte_min"] or dte > cfg["dte_max"]:
            continue
        dist = abs(dte - cfg["dte_target"])
        if dist < best_dist:
            best, best_dist = cand_fri, dist
    return best


def expiry_dte(open_date, target_dte):
    cand = open_date + pd.Timedelta(days=target_dte)
    shift = (4 - cand.weekday()) % 7
    return cand + pd.Timedelta(days=shift)


def process_wheel_positions(state, today, S, sigma, cfg):
    new_positions = []
    profit_take = cfg["profit_take_pct"]
    roll_dte = cfg["roll_dte_trigger"]
    for pos in state.positions:
        T = max((pos.expiry - today).days, 0) / 365.0
        if pos.side == "short_put":
            opt = bs_price(S, pos.strike, T, sigma, kind="put")
            pnl_per_share = pos.open_price - opt
            profit_frac = pnl_per_share / pos.open_price if pos.open_price > 0 else 0.0
            close_for_profit = profit_frac >= profit_take
            close_for_roll = (pos.expiry - today).days <= roll_dte
            is_expiry = today >= pos.expiry
            if close_for_profit or close_for_roll or is_expiry:
                if is_expiry and S < pos.strike:
                    cost = pos.strike * 100 * pos.contracts
                    state.cash -= cost
                    state.cash -= COST_PER_CONTRACT * pos.contracts
                    state.ledger.append({
                        "open_date": pos.open_date, "close_date": today,
                        "kind": "csp_assigned", "strike": pos.strike,
                        "S_close": S, "premium_open": pos.open_price,
                        "premium_close": 0.0,
                        "realized_pnl": pos.open_price * 100 * pos.contracts
                                        - COST_PER_CONTRACT * pos.contracts,
                    })
                    new_positions.append(Position(
                        underlying="SPY", side="long_shares",
                        strike=pos.strike - pos.open_price,
                        expiry=today, open_date=today,
                        open_price=pos.strike - pos.open_price,
                        contracts=pos.contracts, open_sigma=sigma, max_profit=0.0,
                    ))
                else:
                    slip = slippage_per_share(opt)
                    cost = (opt + slip) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    realized = pos.open_price * 100 * pos.contracts - cost
                    state.cash += realized
                    state.ledger.append({
                        "open_date": pos.open_date, "close_date": today,
                        "kind": "csp_closed", "strike": pos.strike,
                        "S_close": S, "premium_open": pos.open_price,
                        "premium_close": opt,
                        "realized_pnl": realized,
                    })
            else:
                new_positions.append(pos)
        elif pos.side == "long_shares":
            new_positions.append(pos)
        elif pos.side == "short_call":
            opt = bs_price(S, pos.strike, T, sigma, kind="call")
            pnl_per_share = pos.open_price - opt
            profit_frac = pnl_per_share / pos.open_price if pos.open_price > 0 else 0.0
            close_for_profit = profit_frac >= profit_take
            close_for_roll = (pos.expiry - today).days <= roll_dte
            is_expiry = today >= pos.expiry
            if close_for_profit or close_for_roll or is_expiry:
                if is_expiry and S > pos.strike:
                    proceeds = pos.strike * 100 * pos.contracts
                    share_basis = pos.max_profit if pos.max_profit > 0 else pos.strike
                    share_pnl = (pos.strike - share_basis) * 100 * pos.contracts
                    premium_kept = pos.open_price * 100 * pos.contracts
                    state.cash += proceeds + premium_kept - COST_PER_CONTRACT * pos.contracts
                    state.ledger.append({
                        "open_date": pos.open_date, "close_date": today,
                        "kind": "cc_called_away", "strike": pos.strike,
                        "S_close": S, "premium_open": pos.open_price,
                        "premium_close": 0.0,
                        "realized_pnl": premium_kept + share_pnl
                                        - COST_PER_CONTRACT * pos.contracts,
                    })
                else:
                    slip = slippage_per_share(opt)
                    cost = (opt + slip) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    realized = pos.open_price * 100 * pos.contracts - cost
                    state.cash += realized
                    state.ledger.append({
                        "open_date": pos.open_date, "close_date": today,
                        "kind": "cc_closed", "strike": pos.strike,
                        "S_close": S, "premium_open": pos.open_price,
                        "premium_close": opt,
                        "realized_pnl": realized,
                    })
                    share_basis = pos.max_profit if pos.max_profit > 0 else pos.strike
                    new_positions.append(Position(
                        underlying="SPY", side="long_shares",
                        strike=share_basis, expiry=today, open_date=today,
                        open_price=share_basis, contracts=pos.contracts,
                        open_sigma=sigma, max_profit=share_basis,
                    ))
    state.positions = new_positions


def open_new_wheel_legs(state, today, S, sigma, vix, cfg):
    put_delta_t = cfg["put_delta_target"]
    call_delta_t = cfg["call_delta_target"]
    vix_gate = cfg["vix_max_gate"]
    lev = cfg["leverage"]

    has_option = any(p.side in ("short_put", "short_call") for p in state.positions)
    has_shares = any(p.side == "long_shares" for p in state.positions)

    if has_shares and not any(p.side == "short_call" for p in state.positions):
        updated = []
        for p in state.positions:
            if p.side == "long_shares" and vix <= vix_gate:
                expiry = find_expiry(today, cfg)
                if expiry is None:
                    updated.append(p); continue
                T = (expiry - today).days / 365.0
                K_call = strike_from_delta(S, T, sigma, call_delta_t, kind="call")
                premium = bs_price(S, K_call, T, sigma, kind="call")
                slip = slippage_per_share(premium)
                if premium - slip <= 0:
                    updated.append(p); continue
                credit = (premium - slip) * 100 * p.contracts - COST_PER_CONTRACT * p.contracts
                state.cash += credit
                updated.append(Position(
                    underlying=p.underlying, side="short_call",
                    strike=K_call, expiry=expiry, open_date=today,
                    open_price=premium - slip, contracts=p.contracts,
                    open_sigma=sigma, max_profit=p.open_price,
                ))
            else:
                updated.append(p)
        state.positions = updated

    if not has_option and not has_shares and vix <= vix_gate:
        expiry = find_expiry(today, cfg)
        if expiry is not None:
            T = (expiry - today).days / 365.0
            K_put = strike_from_delta(S, T, sigma, put_delta_t, kind="put")
            premium = bs_price(S, K_put, T, sigma, kind="put")
            slip = slippage_per_share(premium)
            collateral_budget = state.cash * lev * 0.95
            contracts = int(collateral_budget // (K_put * 100))
            if contracts >= 1 and (premium - slip) > 0.05:
                credit = (premium - slip) * 100 * contracts - COST_PER_CONTRACT * contracts
                state.cash += credit
                state.positions.append(Position(
                    underlying="SPY", side="short_put",
                    strike=K_put, expiry=expiry, open_date=today,
                    open_price=premium - slip, contracts=contracts,
                    open_sigma=sigma, max_profit=(premium - slip) * 100 * contracts,
                ))


# -------------------------- hedge overlay logic ---------------------------
def hedge_mtm(state, today, S, sigma, vix, vix_vol=1.10):
    """Mark-to-market all hedge legs and return total hedge MTM equity contribution."""
    mtm = 0.0
    for h in state.hedges:
        T = max((h.expiry - today).days, 0) / 365.0
        if h.kind == "long_spy_put" or h.kind == "long_spy_put_l" or h.kind == "short_spy_put_s":
            opt = bs_price(S, h.strike, T, sigma, kind="put")
            mtm += h.side_sign * (opt - h.open_price) * 100 * h.contracts
        elif h.kind == "long_vix_call":
            opt = bs_price(vix, h.strike, T, vix_vol, kind="call")
            mtm += h.side_sign * (opt - h.open_price) * 100 * h.contracts
    return mtm


def hedge_close_position(state, today, S, sigma, vix, h, reason, vix_vol=1.10):
    T = max((h.expiry - today).days, 0) / 365.0
    if h.kind == "long_spy_put" or h.kind == "long_spy_put_l" or h.kind == "short_spy_put_s":
        opt = bs_price(S, h.strike, T, sigma, kind="put")
        underlying_label = "SPY"
    else:
        opt = bs_price(vix, h.strike, T, vix_vol, kind="call")
        underlying_label = "VIX"
    slip = slippage_per_share(opt) if opt > 0 else 0.0
    # Long sell at bid (opt - slip), short buy back at ask (opt + slip)
    if h.side_sign == 1:
        proceeds = (opt - slip) * 100 * h.contracts - COST_PER_CONTRACT * h.contracts
        state.cash += proceeds
        state.hedge_cash_received += proceeds
        realized = (opt - h.open_price) * 100 * h.contracts - slip * 100 * h.contracts - COST_PER_CONTRACT * h.contracts
    else:
        cost = (opt + slip) * 100 * h.contracts + COST_PER_CONTRACT * h.contracts
        state.cash -= cost
        state.hedge_cash_paid += cost
        realized = (h.open_price - opt) * 100 * h.contracts - slip * 100 * h.contracts - COST_PER_CONTRACT * h.contracts
    state.ledger.append({
        "open_date": h.open_date, "close_date": today,
        "kind": f"hedge_{h.kind}_close_{reason}",
        "strike": h.strike, "S_close": S if underlying_label == "SPY" else vix,
        "premium_open": h.open_price, "premium_close": opt,
        "realized_pnl": realized,
    })


def overlay_static_put(state, today, S, sigma, vix, hcfg):
    """Always hold 1 long SPY put, ~90 DTE, |delta|=0.10, roll when DTE<30."""
    target_dte = hcfg["target_dte"]
    roll_dte = hcfg["roll_dte"]
    delta_t = hcfg["delta_target"]

    # remove any that need rolling
    keep = []
    for h in state.hedges:
        if h.kind != "long_spy_put":
            keep.append(h); continue
        days_left = (h.expiry - today).days
        if days_left <= roll_dte:
            hedge_close_position(state, today, S, sigma, vix, h, "roll")
        else:
            keep.append(h)
    state.hedges = keep

    has_put = any(h.kind == "long_spy_put" for h in state.hedges)
    if not has_put:
        expiry = expiry_dte(today, target_dte)
        T = (expiry - today).days / 365.0
        K = strike_from_delta(S, T, sigma, delta_t, kind="put")
        premium = bs_price(S, K, T, sigma, kind="put")
        slip = slippage_per_share(premium)
        cost = (premium + slip) * 100 * 1 + COST_PER_CONTRACT * 1
        if state.cash > cost and premium > 0.01:
            state.cash -= cost
            state.hedge_cash_paid += cost
            state.hedges.append(HedgePosition(
                kind="long_spy_put", strike=K, expiry=expiry,
                open_date=today, open_price=premium + slip,
                contracts=1, side_sign=1,
            ))


def overlay_vix_call(state, today, S, sigma, vix, hcfg):
    """When VIX>20: buy +0.20 delta VIX call, 30 DTE. Roll monthly.
    Close when VIX<16."""
    target_dte = hcfg["target_dte"]
    delta_t = hcfg["delta_target"]
    vix_on = hcfg["vix_on"]
    vix_off = hcfg["vix_off"]
    vix_vol = hcfg["vix_vol"]

    # check existing positions
    keep = []
    for h in state.hedges:
        if h.kind != "long_vix_call":
            keep.append(h); continue
        days_left = (h.expiry - today).days
        if vix < vix_off:
            hedge_close_position(state, today, S, sigma, vix, h, "vix_off", vix_vol=vix_vol)
        elif days_left <= 5:
            hedge_close_position(state, today, S, sigma, vix, h, "roll", vix_vol=vix_vol)
        else:
            keep.append(h)
    state.hedges = keep

    has_call = any(h.kind == "long_vix_call" for h in state.hedges)
    if vix >= vix_on and not has_call:
        expiry = expiry_dte(today, target_dte)
        T = (expiry - today).days / 365.0
        K = strike_from_delta(vix, T, vix_vol, delta_t, kind="call")
        premium = bs_price(vix, K, T, vix_vol, kind="call")
        slip = max(0.05, 0.05 * premium)  # VIX options slip ~5%
        cost = (premium + slip) * 100 * 1 + COST_PER_CONTRACT * 1
        if state.cash > cost and premium > 0.05:
            state.cash -= cost
            state.hedge_cash_paid += cost
            state.hedges.append(HedgePosition(
                kind="long_vix_call", strike=K, expiry=expiry,
                open_date=today, open_price=premium + slip,
                contracts=1, side_sign=1,
            ))


def overlay_putspread_collar(state, today, S, sigma, vix, hcfg, ma50, ma200):
    """Put-spread when VIX>20 OR (50d MA < 200d MA). Long -10 delta put,
    short -5 delta put. 90 DTE, roll when DTE<30."""
    target_dte = hcfg["target_dte"]
    roll_dte = hcfg["roll_dte"]
    long_d = hcfg["long_delta"]
    short_d = hcfg["short_delta"]
    vix_on = hcfg["vix_on"]

    trend_signal = (ma50 is not None and ma200 is not None
                    and not (np.isnan(ma50) or np.isnan(ma200))
                    and ma50 < ma200)
    activate = (vix >= vix_on) or trend_signal

    # close legs needing roll, or close both if regime is no longer active
    keep = []
    for h in state.hedges:
        if h.kind not in ("long_spy_put_l", "short_spy_put_s"):
            keep.append(h); continue
        days_left = (h.expiry - today).days
        if days_left <= roll_dte:
            hedge_close_position(state, today, S, sigma, vix, h, "roll")
        elif not activate:
            hedge_close_position(state, today, S, sigma, vix, h, "regime_off")
        else:
            keep.append(h)
    state.hedges = keep

    has_long = any(h.kind == "long_spy_put_l" for h in state.hedges)
    has_short = any(h.kind == "short_spy_put_s" for h in state.hedges)
    if activate and not (has_long and has_short):
        expiry = expiry_dte(today, target_dte)
        T = (expiry - today).days / 365.0
        K_long = strike_from_delta(S, T, sigma, long_d, kind="put")
        K_short = strike_from_delta(S, T, sigma, short_d, kind="put")
        if K_short >= K_long:
            return  # degenerate, skip
        p_long = bs_price(S, K_long, T, sigma, kind="put")
        p_short = bs_price(S, K_short, T, sigma, kind="put")
        slip_long = slippage_per_share(p_long)
        slip_short = slippage_per_share(p_short)
        # long pays ask, short collects bid
        long_cost = (p_long + slip_long) * 100 + COST_PER_CONTRACT
        short_credit = (p_short - slip_short) * 100 - COST_PER_CONTRACT
        net = long_cost - short_credit
        if state.cash > long_cost and p_long > 0.01:
            state.cash -= long_cost
            state.cash += short_credit
            state.hedge_cash_paid += long_cost
            state.hedge_cash_received += short_credit
            state.hedges.append(HedgePosition(
                kind="long_spy_put_l", strike=K_long, expiry=expiry,
                open_date=today, open_price=p_long + slip_long,
                contracts=1, side_sign=1,
            ))
            state.hedges.append(HedgePosition(
                kind="short_spy_put_s", strike=K_short, expiry=expiry,
                open_date=today, open_price=p_short - slip_short,
                contracts=1, side_sign=-1,
            ))


# ------------------------- main backtest loop -----------------------------
def run_variant(variant_name, hcfg, df, cfg, starting_cash):
    state = State(cash=starting_cash)
    dates = df.index.to_list()

    for today in dates:
        row = df.loc[today]
        S = float(row["close"])
        sigma = float(row["sigma"])
        vix = float(row["vix"]) if pd.notna(row["vix"]) else 0.0
        ma50 = float(row["ma50"]) if pd.notna(row["ma50"]) else None
        ma200 = float(row["ma200"]) if pd.notna(row["ma200"]) else None

        # 1. Wheel mechanics first
        process_wheel_positions(state, today, S, sigma, cfg)
        open_new_wheel_legs(state, today, S, sigma, vix, cfg)

        # 2. Hedge overlay
        if hcfg is not None:
            if hcfg["type"] == "static_put":
                overlay_static_put(state, today, S, sigma, vix, hcfg)
            elif hcfg["type"] == "vix_call":
                overlay_vix_call(state, today, S, sigma, vix, hcfg)
            elif hcfg["type"] == "putspread_collar":
                overlay_putspread_collar(state, today, S, sigma, vix, hcfg, ma50, ma200)

        # 3. MTM total equity (wheel + hedge book)
        equity = state.cash
        for p in state.positions:
            T = max((p.expiry - today).days, 0) / 365.0
            if p.side == "short_put":
                opt = bs_price(S, p.strike, T, sigma, kind="put")
                equity += (p.open_price - opt) * 100 * p.contracts
            elif p.side == "long_shares":
                share_basis = p.open_price
                equity += (S - share_basis) * 100 * p.contracts
            elif p.side == "short_call":
                opt = bs_price(S, p.strike, T, sigma, kind="call")
                share_basis = p.max_profit
                equity += (S - share_basis) * 100 * p.contracts
                equity += (p.open_price - opt) * 100 * p.contracts
        # hedge MTM: cash already debited at open, so we add back the current value
        # of each hedge position (open premium has been paid out of cash).
        vix_vol_param = HEDGE_VIX_CALL["vix_vol"]
        for h in state.hedges:
            T = max((h.expiry - today).days, 0) / 365.0
            if h.kind in ("long_spy_put", "long_spy_put_l", "short_spy_put_s"):
                opt = bs_price(S, h.strike, T, sigma, kind="put")
            else:  # vix call
                opt = bs_price(vix, h.strike, T, vix_vol_param, kind="call")
            if h.side_sign == 1:
                # we paid premium, currently worth opt*100
                equity += opt * 100 * h.contracts
            else:
                # we received premium (cash already credited), liability is opt*100
                equity -= opt * 100 * h.contracts

        state.equity_curve.append({
            "date": today, "equity": equity, "S": S, "vix": vix,
            "spy_ret": float(row["spy_ret"]) if pd.notna(row["spy_ret"]) else np.nan,
            "ma50": ma50, "ma200": ma200,
        })

    eq_df = pd.DataFrame(state.equity_curve).set_index("date")
    eq_df["ret"] = eq_df["equity"].pct_change()
    ledger_df = pd.DataFrame(state.ledger)
    return eq_df, ledger_df, state


# ------------------------------ metrics -----------------------------------
def compute_metrics(eq_df):
    rets = eq_df["ret"].dropna()
    if len(rets) < 2:
        return {}
    years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
    cagr = (eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    mu, sd = rets.mean(), rets.std()
    downside = rets[rets < 0].std()
    sharpe = (mu / sd) * math.sqrt(TRADING_DAYS) if sd > 0 else float("nan")
    sortino = (mu / downside) * math.sqrt(TRADING_DAYS) if downside and downside > 0 else float("nan")
    peak = eq_df["equity"].cummax()
    dd = eq_df["equity"] / peak - 1.0
    max_dd = float(dd.min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    wr = float((rets > 0).mean())
    pos = rets[rets > 0].sum()
    neg = abs(rets[rets < 0].sum())
    pf = float(pos / neg) if neg > 0 else float("nan")
    return {
        "n_days": int(len(rets)),
        "years": float(years),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
        "win_rate": wr,
        "profit_factor": pf,
        "total_return_pct": float((eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0] - 1.0) * 100.0),
        "final_equity": float(eq_df["equity"].iloc[-1]),
    }


def regime_gate(eq_df):
    sub = eq_df.dropna(subset=["spy_ret", "ret"])
    sigma = sub["spy_ret"].std()
    thresh = 0.5 * sigma
    green = sub[sub["spy_ret"] >= thresh]["ret"]
    red = sub[sub["spy_ret"] <= -thresh]["ret"]
    flat = sub[(sub["spy_ret"] > -thresh) & (sub["spy_ret"] < thresh)]["ret"]

    def ann_sharpe(s):
        if len(s) < 2 or s.std() == 0:
            return float("nan")
        return (s.mean() / s.std()) * math.sqrt(TRADING_DAYS)

    sh_g, sh_r, sh_f = ann_sharpe(green), ann_sharpe(red), ann_sharpe(flat)
    denom = max(abs(sh_g) if np.isfinite(sh_g) else 0,
                abs(sh_r) if np.isfinite(sh_r) else 0, 1e-9)
    gap = abs((sh_g if np.isfinite(sh_g) else 0.0) - (sh_r if np.isfinite(sh_r) else 0.0)) / denom
    return {
        "n_green": int(len(green)),
        "n_red": int(len(red)),
        "n_flat": int(len(flat)),
        "sharpe_green": float(sh_g) if np.isfinite(sh_g) else None,
        "sharpe_red": float(sh_r) if np.isfinite(sh_r) else None,
        "sharpe_flat": float(sh_f) if np.isfinite(sh_f) else None,
        "regime_gap": float(gap),
        "hc428_r1_pass": bool(gap <= 0.50),
    }


def day_concentration(ledger_df):
    if ledger_df is None or ledger_df.empty or "realized_pnl" not in ledger_df.columns:
        return float("nan")
    pos = ledger_df[ledger_df["realized_pnl"] > 0]["realized_pnl"]
    if len(pos) == 0:
        return float("nan")
    return float(pos.max() / pos.sum())


def tail_event_stats(eq_df, label, start, end):
    sub = eq_df.loc[start:end].copy()
    if len(sub) < 2:
        return {"label": label, "n_days": 0}
    peak = sub["equity"].cummax()
    dd = sub["equity"] / peak - 1.0
    max_dd = float(dd.min())
    trough = dd.idxmin()
    pre = eq_df.loc[:sub.index[0]]
    pre_peak = float(pre["equity"].max()) if len(pre) else float(sub["equity"].iloc[0])
    after = eq_df.loc[trough:]
    rec = after[after["equity"] >= pre_peak]
    rec_days = int((rec.index[0] - trough).days) if len(rec) else None
    weekly = sub["equity"].resample("W").last().pct_change()
    worst_week = float(weekly.min()) if len(weekly) else float("nan")
    return {
        "label": label,
        "window": f"{start} to {end}",
        "n_days": int(len(sub)),
        "max_dd_pct": max_dd * 100,
        "vix_peak": float(sub["vix"].max()) if "vix" in sub.columns else None,
        "recover_days": rec_days,
        "worst_week_pct": worst_week * 100 if np.isfinite(worst_week) else None,
        "cum_ret_pct": float((sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1.0) * 100.0),
    }


# ------------------------------ mlflow ------------------------------------
def log_to_mlflow(variant, metrics, gate, day_conc, tail, eq_df, ledger_df, cfg, hcfg, state):
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("wheel_hedged_spy_v1")
        with mlflow.start_run(run_name=variant):
            mlflow.log_params(cfg)
            if hcfg is not None:
                mlflow.log_params({f"hedge_{k}": v for k, v in hcfg.items()})
            mlflow.log_param("variant", variant)
            mlflow.log_param("start", str(eq_df.index[0].date()))
            mlflow.log_param("end", str(eq_df.index[-1].date()))
            mlflow.log_param("starting_cash", STARTING_CASH)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(k, v)
            for k, v in gate.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(f"regime_{k}", v)
            mlflow.log_metric("day_concentration",
                              day_conc if np.isfinite(day_conc) else -1.0)
            mlflow.log_metric("hc428_r1_pass", 1.0 if gate.get("hc428_r1_pass") else 0.0)
            mlflow.log_metric("hedge_cash_paid", state.hedge_cash_paid)
            mlflow.log_metric("hedge_cash_received", state.hedge_cash_received)
            mlflow.log_metric("hedge_net_drag_dollars",
                              state.hedge_cash_paid - state.hedge_cash_received)
            for t in tail:
                if "max_dd_pct" in t:
                    safe = t["label"].replace(" ", "_")
                    mlflow.log_metric(f"tail_{safe}_max_dd_pct", t["max_dd_pct"])
                    if t.get("recover_days") is not None:
                        mlflow.log_metric(f"tail_{safe}_recover_days", t["recover_days"])
                    if t.get("worst_week_pct") is not None:
                        mlflow.log_metric(f"tail_{safe}_worst_week_pct", t["worst_week_pct"])
            csv_path = OUT_DIR / f"equity_{variant}.csv"
            eq_df.to_csv(csv_path)
            mlflow.log_artifact(str(csv_path))
            return mlflow.active_run().info.run_id
    except Exception as e:
        print(f"[mlflow] skipped ({e})")
        return None


# ------------------------------- main -------------------------------------
def main():
    print("[load] SPY/IV/VIX data ...")
    df = load_data()
    print(f"[load] {len(df)} trading days {df.index[0].date()} -> {df.index[-1].date()}")

    variants = [
        ("BASELINE", None),
        ("STATIC_PUT", HEDGE_STATIC_PUT),
        ("VIX_CALL_COND", HEDGE_VIX_CALL),
        ("PUTSPREAD_COLLAR", HEDGE_PUTSPREAD_COLLAR),
    ]

    results = {}
    baseline_cagr = None

    for vname, hcfg in variants:
        print(f"\n[{vname}] running backtest ...")
        eq_df, ledger_df, state = run_variant(vname, hcfg, df, TIER2_BALANCED_SCALP, STARTING_CASH)
        metrics = compute_metrics(eq_df)
        gate = regime_gate(eq_df)
        day_conc = day_concentration(ledger_df)
        tail = [
            tail_event_stats(eq_df, "COVID_2020", "2020-02-15", "2020-05-31"),
            tail_event_stats(eq_df, "2022_bear", "2022-01-01", "2022-12-31"),
            tail_event_stats(eq_df, "Aug_2024_carry", "2024-07-15", "2024-09-15"),
        ]
        gates_pass = {
            "sharpe_ge_1.0": metrics["sharpe"] >= 1.0 if np.isfinite(metrics.get("sharpe", float("nan"))) else False,
            "calmar_ge_1.5": metrics["calmar"] >= 1.5 if np.isfinite(metrics.get("calmar", float("nan"))) else False,
            "regime_gap_le_0.50": gate["hc428_r1_pass"],
            "day_conc_le_0.70": (day_conc <= 0.70) if np.isfinite(day_conc) else True,
            "n_days_ge_40": metrics.get("n_days", 0) >= 40,
        }
        deploy_ready = all(gates_pass.values())

        eq_df.to_parquet(OUT_DIR / f"equity_{vname}.parquet")
        if not ledger_df.empty:
            ledger_df.to_parquet(OUT_DIR / f"ledger_{vname}.parquet")
        with open(OUT_DIR / f"results_{vname}.json", "w") as f:
            json.dump({"variant": vname, "metrics": metrics, "gate": gate,
                       "day_conc": day_conc, "tail": tail,
                       "hedge_cash_paid": state.hedge_cash_paid,
                       "hedge_cash_received": state.hedge_cash_received,
                       "hedge_net_drag_dollars": state.hedge_cash_paid - state.hedge_cash_received,
                       "gates_pass": gates_pass, "deploy_ready": deploy_ready},
                      f, indent=2, default=str)

        run_id = log_to_mlflow(vname, metrics, gate, day_conc, tail, eq_df, ledger_df,
                               TIER2_BALANCED_SCALP, hcfg, state)
        results[vname] = {
            "metrics": metrics, "gate": gate, "day_conc": day_conc,
            "tail": tail, "gates_pass": gates_pass, "deploy_ready": deploy_ready,
            "hedge_drag_dollars": state.hedge_cash_paid - state.hedge_cash_received,
            "mlflow_run_id": run_id,
        }
        if vname == "BASELINE":
            baseline_cagr = metrics["cagr"]
        print(f"[{vname}] Sharpe={metrics['sharpe']:.2f} CAGR={metrics['cagr']*100:.1f}% "
              f"MaxDD={metrics['max_dd']*100:.1f}% Calmar={metrics['calmar']:.2f} "
              f"regime_gap={gate['regime_gap']:.2f} deploy={deploy_ready}")

    # ---------- markdown report ----------
    L = []
    L.append("# Wheel + Hedge Overlay v1 - SPY Tier2 Balanced Scalp\n")
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Window: {START_DATE.date()} to {END_DATE.date()}")
    L.append(f"Starting cash: ${STARTING_CASH:,.0f}, leverage 1.0x, single underlying SPY")
    L.append("Wheel cfg: put_delta 0.22, call_delta 0.22, DTE 30-45, profit-take 50%, "
             "roll DTE<=10, VIX gate 32.0\n")
    L.append("Pricing: Black-Scholes with modeled ATM sigma for SPY; BS on VIX with "
             "vol-of-vol=1.10 for VIX calls. No skew model -> hedge premium estimates "
             "are LOWER BOUNDS; real OTM put cost would be 30-60% higher in the chain.")
    L.append("Slippage: max(2.5% premium, $0.03/share) for SPY; 5% for VIX. "
             "Commission: $0.65/contract/leg.\n")
    L.append("Hedge variants:")
    L.append("- **STATIC_PUT**: always-on long SPY put, 90 DTE, |delta|=0.10, roll DTE<30")
    L.append("- **VIX_CALL_COND**: long VIX call (delta +0.20, 30 DTE) when VIX>20, "
             "close when VIX<16")
    L.append("- **PUTSPREAD_COLLAR**: long-short SPY put-spread (long -0.10, short -0.05) "
             "when VIX>20 OR 50d MA < 200d MA, 90 DTE, roll DTE<30\n")

    L.append("## Headline Metrics\n")
    L.append("| Variant | CAGR | Sharpe | Sortino | MaxDD | Calmar | WR | PF | Hedge $ drag | Final $ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for v, r in results.items():
        m = r["metrics"]
        hd = r["hedge_drag_dollars"]
        L.append(f"| {v} | {m['cagr']*100:.2f}% | {m['sharpe']:.2f} | {m['sortino']:.2f} | "
                 f"{m['max_dd']*100:.1f}% | {m['calmar']:.2f} | {m['win_rate']*100:.1f}% | "
                 f"{m['profit_factor']:.2f} | ${hd:,.0f} | ${m['final_equity']:,.0f} |")

    L.append("\n## CAGR cost of the hedge (vs BASELINE)\n")
    L.append("| Variant | CAGR | Delta vs baseline | Hedge cost % CAGR |")
    L.append("|---|---|---|---|")
    base_c = results["BASELINE"]["metrics"]["cagr"]
    for v, r in results.items():
        c = r["metrics"]["cagr"]
        delta = c - base_c
        L.append(f"| {v} | {c*100:.2f}% | {delta*100:+.2f}pp | {-delta*100:+.2f}pp |")

    L.append("\n## Regime Gate (HC #428 R1)\n")
    L.append("| Variant | n_green | n_red | n_flat | Sh green | Sh red | Sh flat | gap | pass |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for v, r in results.items():
        g = r["gate"]
        def f2(x): return f"{x:.2f}" if (x is not None and np.isfinite(x)) else "nan"
        L.append(f"| {v} | {g['n_green']} | {g['n_red']} | {g['n_flat']} | "
                 f"{f2(g['sharpe_green'])} | {f2(g['sharpe_red'])} | {f2(g['sharpe_flat'])} | "
                 f"{g['regime_gap']:.2f} | {'PASS' if g['hc428_r1_pass'] else 'FAIL'} |")

    L.append("\n## Deploy Gates\n")
    L.append("| Variant | Sharpe>=1.0 | Calmar>=1.5 | Regime gap<=0.50 | Day conc<=0.70 | n_days>=40 | DEPLOY |")
    L.append("|---|---|---|---|---|---|---|")
    for v, r in results.items():
        gp = r["gates_pass"]
        x = lambda b: "PASS" if b else "FAIL"
        L.append(f"| {v} | {x(gp['sharpe_ge_1.0'])} | {x(gp['calmar_ge_1.5'])} | "
                 f"{x(gp['regime_gap_le_0.50'])} | {x(gp['day_conc_le_0.70'])} | "
                 f"{x(gp['n_days_ge_40'])} | **{'YES' if r['deploy_ready'] else 'NO'}** |")

    L.append("\n## Tail Event Stress\n")
    L.append("| Variant | Event | MaxDD | VIX peak | Recover days | Worst week | Cum ret |")
    L.append("|---|---|---|---|---|---|---|")
    for v, r in results.items():
        for t in r["tail"]:
            if t.get("n_days", 0) == 0:
                continue
            ww = f"{t['worst_week_pct']:.1f}%" if t.get("worst_week_pct") is not None else "n/a"
            rd = t.get("recover_days") if t.get("recover_days") is not None else "not_recovered"
            L.append(f"| {v} | {t['label']} | {t['max_dd_pct']:.1f}% | "
                     f"{(t.get('vix_peak') or 0):.1f} | {rd} | {ww} | {t['cum_ret_pct']:.1f}% |")

    L.append("\n## Recommendation\n")
    passing = [v for v, r in results.items() if r["deploy_ready"] and v != "BASELINE"]
    if passing:
        L.append(f"- **{', '.join(passing)} PASSES all HC #428 deploy gates.** Wire into "
                 f"wheel_paper_engine.py behind `hedge_overlay` flag (DEFAULT OFF).")
    else:
        L.append("- **No hedge variant clears all HC #428 deploy gates** in this backtest.")
        # diagnose why baseline failed (if it did)
        base_failed = [k for k, v in results["BASELINE"]["gates_pass"].items() if not v]
        if base_failed:
            L.append(f"- BASELINE itself fails: {', '.join(base_failed)}. The hedge can only "
                     f"narrow the regime gap; it cannot create alpha. If baseline fails "
                     f"Sharpe/Calmar, a hedge that costs CAGR will make those metrics worse.")
        # show direction of regime gap change
        base_gap = results["BASELINE"]["gate"]["regime_gap"]
        for v in ("STATIC_PUT", "VIX_CALL_COND", "PUTSPREAD_COLLAR"):
            g = results[v]["gate"]["regime_gap"]
            sign = "narrowed" if g < base_gap else "widened"
            L.append(f"- {v}: regime gap {sign} from {base_gap:.2f} -> {g:.2f}.")

    L.append("\n## Honest caveats\n")
    L.append("- BS with no skew model UNDER-prices OTM SPY puts. Real chain premium for "
             "a -0.10 delta 90 DTE SPY put runs 30-60% above modeled here due to crash "
             "skew. Hedge drag in production will exceed these numbers.")
    L.append("- VIX option pricing uses a constant vol-of-vol=1.10; the true VIX surface "
             "is mean-reverting and has its own term structure. Treat VIX_CALL_COND P&L "
             "with extra skepticism vs the put hedges.")
    L.append("- The wheel itself is short vol and short put gamma. A long-put hedge that "
             "PASSES the regime-gap gate while still earning >=1.0 Sharpe is mathematically "
             "asking the wheel to earn more carry than the hedge bleeds, in all regimes - "
             "a high bar that the underlying SPY mid-DTE 22-delta wheel rarely clears.")
    L.append("- This run uses the SAME 2018-2025 window the unhedged SPY backtest used. "
             "Sub-agent aac533ea documented the BASELINE regime gap is structurally large; "
             "the question this report answers is HOW MUCH does each hedge variant flatten "
             "that gap, and AT WHAT COST in CAGR/Sharpe.")

    report_path = REPORT_DIR / "wheel_hedged_spy_v1.md"
    report_path.write_text("\n".join(L))
    print(f"\n[done] report -> {report_path}")
    print(f"[done] artifacts -> {OUT_DIR}")
    return results


if __name__ == "__main__":
    main()
