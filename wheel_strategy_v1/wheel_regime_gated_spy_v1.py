#!/usr/bin/env python3
"""
wheel_regime_gated_spy_v1.py - Tier2_Balanced_Scalp on SPY with five entry-gate
variants + an unchanged baseline. Per HC #583 R3 (2026-06-09).

Builds on wheel_hedged_spy_v1.py: reuses the exact wheel mechanics, but instead
of layering a hedge overlay we gate NEW short-put entries by a regime filter.
Existing positions still manage/exit normally regardless of gate state.

Variants:
  0) BASELINE             - unchanged Tier2 wheel
  1) TREND_ONLY           - SPY > 50d MA
  2) TREND_VOL            - SPY > 50d MA AND VIX < 20
  3) TIGHT_TREND_VOL      - SPY > 20d MA AND SPY > 50d MA AND VIX < 18
  4) RV_GATE              - 20d realized vol < 15% AND SPY > 50d MA
  5) DD_FROM_HIGH         - SPY within 5% of trailing-30d high AND VIX < 25

Each variant logs the % of trading days the gate is OPEN ("in-market %"),
per-day Sharpe stratified by SPY green/red/flat days, and HC #428 R1 gap.

Outputs:
  - MLflow experiment "wheel_regime_gated_spy_v1"
  - /home/jupiter/Lvl3Quant/research/findings/wheel_regime_gated_spy_v1.md
  - /home/jupiter/Lvl3Quant/output/wheel_regime_gated_spy_v1/{equity,ledger}_{variant}.parquet
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
OUT_DIR = ROOT / "output" / "wheel_regime_gated_spy_v1"
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

STARTING_CASH = 50_000.0
RISK_FREE = 0.04
TRADING_DAYS = 252

SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN_PER_SHARE = 0.03
COST_PER_CONTRACT = 0.65

START_DATE = pd.Timestamp("2018-01-01")
END_DATE = pd.Timestamp("2025-12-31")

# Gate variants
GATE_VARIANTS = [
    {"name": "BASELINE", "gate": None},
    {"name": "TREND_ONLY",      "gate": {"sma_window": 50, "vix_max": None, "realized_vol_max": None, "dd_from_high_max": None, "sma_short": None}},
    {"name": "TREND_VOL",       "gate": {"sma_window": 50, "vix_max": 20.0, "realized_vol_max": None, "dd_from_high_max": None, "sma_short": None}},
    {"name": "TIGHT_TREND_VOL", "gate": {"sma_window": 50, "vix_max": 18.0, "realized_vol_max": None, "dd_from_high_max": None, "sma_short": 20}},
    {"name": "RV_GATE",         "gate": {"sma_window": 50, "vix_max": None, "realized_vol_max": 0.15, "dd_from_high_max": None, "sma_short": None}},
    {"name": "DD_FROM_HIGH",    "gate": {"sma_window": None, "vix_max": 25.0, "realized_vol_max": None, "dd_from_high_max": 0.05, "sma_short": None, "high_lookback": 30}},
]


# ------------------------------ BS helpers --------------------------------
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
    side: str
    strike: float
    expiry: pd.Timestamp
    open_date: pd.Timestamp
    open_price: float
    contracts: int
    open_sigma: float
    max_profit: float = 0.0


@dataclass
class State:
    cash: float
    positions: list = field(default_factory=list)
    ledger: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    gate_open_days: int = 0
    gate_total_days: int = 0
    entries_attempted: int = 0
    entries_taken: int = 0
    entries_blocked: int = 0


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
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    # realized vol (20d annualized)
    df["rv20"] = df["spy_ret"].rolling(20).std() * math.sqrt(TRADING_DAYS)
    # rolling 30d high
    df["high30"] = df["close"].rolling(30).max()
    df["dd_from_high30"] = (df["high30"] - df["close"]) / df["high30"]
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


def evaluate_gate(gate, row):
    """Evaluate the regime gate against one daily row. Returns (open: bool, reason: str)."""
    if gate is None:
        return True, "no_gate"
    S = float(row["close"])
    vix = float(row["vix"]) if pd.notna(row["vix"]) else 0.0

    # SMA long check
    sma_w = gate.get("sma_window")
    if sma_w is not None:
        ma_col = f"ma{sma_w}"
        ma = row.get(ma_col)
        if ma is None or pd.isna(ma):
            return False, "sma_unavailable"
        if S <= ma:
            return False, f"S<={ma_col}"

    # SMA short check
    sma_s = gate.get("sma_short")
    if sma_s is not None:
        ma_col = f"ma{sma_s}"
        ma = row.get(ma_col)
        if ma is None or pd.isna(ma):
            return False, "sma_short_unavailable"
        if S <= ma:
            return False, f"S<={ma_col}"

    # VIX cap
    vmax = gate.get("vix_max")
    if vmax is not None and vix >= vmax:
        return False, f"vix>={vmax}"

    # Realized vol cap
    rvmax = gate.get("realized_vol_max")
    if rvmax is not None:
        rv = row.get("rv20")
        if rv is None or pd.isna(rv):
            return False, "rv_unavailable"
        if rv >= rvmax:
            return False, f"rv>={rvmax}"

    # Drawdown-from-high cap
    ddmax = gate.get("dd_from_high_max")
    if ddmax is not None:
        dd = row.get("dd_from_high30")
        if dd is None or pd.isna(dd):
            return False, "dd_unavailable"
        if dd >= ddmax:
            return False, f"dd>={ddmax}"

    return True, "open"


def open_new_wheel_legs(state, today, S, sigma, vix, cfg, gate_open):
    put_delta_t = cfg["put_delta_target"]
    call_delta_t = cfg["call_delta_target"]
    vix_gate = cfg["vix_max_gate"]
    lev = cfg["leverage"]

    has_option = any(p.side in ("short_put", "short_call") for p in state.positions)
    has_shares = any(p.side == "long_shares" for p in state.positions)

    # Covered calls on assigned shares: managed regardless of gate (existing positions
    # manage normally per spec).
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

    # NEW short-put entries: gated.
    if not has_option and not has_shares and vix <= vix_gate:
        state.entries_attempted += 1
        if not gate_open:
            state.entries_blocked += 1
            return
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
                state.entries_taken += 1


# ------------------------- main backtest loop -----------------------------
def run_variant(variant_name, gate, df, cfg, starting_cash):
    state = State(cash=starting_cash)
    dates = df.index.to_list()

    for today in dates:
        row = df.loc[today]
        S = float(row["close"])
        sigma = float(row["sigma"])
        vix = float(row["vix"]) if pd.notna(row["vix"]) else 0.0

        # 1. Manage existing positions FIRST (always, regardless of gate)
        process_wheel_positions(state, today, S, sigma, cfg)

        # 2. Evaluate gate for new entries
        gate_open, gate_reason = evaluate_gate(gate, row)
        state.gate_total_days += 1
        if gate_open:
            state.gate_open_days += 1

        # 3. Maybe open new wheel legs
        open_new_wheel_legs(state, today, S, sigma, vix, cfg, gate_open)

        # 4. MTM total equity
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

        state.equity_curve.append({
            "date": today, "equity": equity, "S": S, "vix": vix,
            "spy_ret": float(row["spy_ret"]) if pd.notna(row["spy_ret"]) else np.nan,
            "gate_open": bool(gate_open),
            "gate_reason": gate_reason,
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


def regime_gate_metrics(eq_df):
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
def log_to_mlflow(variant, metrics, gate_metrics, day_conc, tail, eq_df, cfg, gate_cfg, in_mkt_pct, entries_taken, entries_blocked):
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("wheel_regime_gated_spy_v1")
        with mlflow.start_run(run_name=variant):
            mlflow.log_params(cfg)
            if gate_cfg is not None:
                mlflow.log_params({f"gate_{k}": v for k, v in gate_cfg.items()})
            mlflow.log_param("variant", variant)
            mlflow.log_param("start", str(eq_df.index[0].date()))
            mlflow.log_param("end", str(eq_df.index[-1].date()))
            mlflow.log_param("starting_cash", STARTING_CASH)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(k, v)
            for k, v in gate_metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(f"regime_{k}", v)
            mlflow.log_metric("day_concentration",
                              day_conc if np.isfinite(day_conc) else -1.0)
            mlflow.log_metric("hc428_r1_pass", 1.0 if gate_metrics.get("hc428_r1_pass") else 0.0)
            mlflow.log_metric("in_market_pct", in_mkt_pct)
            mlflow.log_metric("entries_taken", entries_taken)
            mlflow.log_metric("entries_blocked", entries_blocked)
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

    results = {}
    for v in GATE_VARIANTS:
        vname = v["name"]
        gate_cfg = v["gate"]
        print(f"\n[{vname}] running backtest ...")
        eq_df, ledger_df, state = run_variant(vname, gate_cfg, df, TIER2_BALANCED_SCALP, STARTING_CASH)
        metrics = compute_metrics(eq_df)
        rgate = regime_gate_metrics(eq_df)
        day_conc = day_concentration(ledger_df)
        tail = [
            tail_event_stats(eq_df, "COVID_2020", "2020-02-15", "2020-05-31"),
            tail_event_stats(eq_df, "2022_bear", "2022-01-01", "2022-12-31"),
            tail_event_stats(eq_df, "Aug_2024_carry", "2024-07-15", "2024-09-15"),
        ]
        in_mkt_pct = (state.gate_open_days / state.gate_total_days * 100.0) if state.gate_total_days else 0.0

        gates_pass = {
            "sharpe_ge_1.0": metrics["sharpe"] >= 1.0 if np.isfinite(metrics.get("sharpe", float("nan"))) else False,
            "calmar_ge_1.5": metrics["calmar"] >= 1.5 if np.isfinite(metrics.get("calmar", float("nan"))) else False,
            "regime_gap_le_0.50": rgate["hc428_r1_pass"],
            "day_conc_le_0.70": (day_conc <= 0.70) if np.isfinite(day_conc) else True,
            "in_market_ge_40": in_mkt_pct >= 40.0,
        }
        deploy_ready = all(gates_pass.values())

        eq_df.to_parquet(OUT_DIR / f"equity_{vname}.parquet")
        if not ledger_df.empty:
            ledger_df.to_parquet(OUT_DIR / f"ledger_{vname}.parquet")
        with open(OUT_DIR / f"results_{vname}.json", "w") as f:
            json.dump({"variant": vname, "metrics": metrics, "gate": rgate,
                       "day_conc": day_conc, "tail": tail,
                       "in_market_pct": in_mkt_pct,
                       "entries_taken": state.entries_taken,
                       "entries_blocked": state.entries_blocked,
                       "gate_cfg": gate_cfg,
                       "gates_pass": gates_pass, "deploy_ready": deploy_ready},
                      f, indent=2, default=str)

        run_id = log_to_mlflow(vname, metrics, rgate, day_conc, tail, eq_df,
                               TIER2_BALANCED_SCALP, gate_cfg, in_mkt_pct,
                               state.entries_taken, state.entries_blocked)
        results[vname] = {
            "metrics": metrics, "gate": rgate, "day_conc": day_conc,
            "tail": tail, "gates_pass": gates_pass, "deploy_ready": deploy_ready,
            "in_market_pct": in_mkt_pct, "gate_cfg": gate_cfg,
            "entries_taken": state.entries_taken,
            "entries_blocked": state.entries_blocked,
            "mlflow_run_id": run_id,
        }
        print(f"[{vname}] Sharpe={metrics['sharpe']:.2f} CAGR={metrics['cagr']*100:.1f}% "
              f"MaxDD={metrics['max_dd']*100:.1f}% Calmar={metrics['calmar']:.2f} "
              f"regime_gap={rgate['regime_gap']:.2f} in_mkt={in_mkt_pct:.0f}% "
              f"deploy={deploy_ready}")

    # ---------- markdown report ----------
    L = []
    L.append("# Wheel + Regime-Gated Entries v1 - SPY Tier2 Balanced Scalp\n")
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Window: {START_DATE.date()} to {END_DATE.date()}")
    L.append(f"Starting cash: ${STARTING_CASH:,.0f}, leverage 1.0x, SPY only")
    L.append("Wheel cfg: put_delta 0.22, call_delta 0.22, DTE 30-45, profit-take 50%, "
             "roll DTE<=10, VIX gate 32.0\n")
    L.append("All variants gate NEW short-put ENTRIES only. Existing positions manage/"
             "exit normally regardless of gate state.\n")
    L.append("Entry-gate variants:")
    L.append("- **TREND_ONLY**: SPY > 50d MA")
    L.append("- **TREND_VOL**: SPY > 50d MA AND VIX < 20")
    L.append("- **TIGHT_TREND_VOL**: SPY > 20d MA AND SPY > 50d MA AND VIX < 18")
    L.append("- **RV_GATE**: 20d realized vol < 15% AND SPY > 50d MA")
    L.append("- **DD_FROM_HIGH**: SPY within 5% of trailing-30d high AND VIX < 25\n")

    L.append("## Headline Metrics\n")
    L.append("| Variant | In-market | CAGR | Sharpe | Sortino | MaxDD | Calmar | WR | PF | Final $ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for v, r in results.items():
        m = r["metrics"]
        L.append(f"| {v} | {r['in_market_pct']:.0f}% | {m['cagr']*100:.2f}% | {m['sharpe']:.2f} | "
                 f"{m['sortino']:.2f} | {m['max_dd']*100:.1f}% | {m['calmar']:.2f} | "
                 f"{m['win_rate']*100:.1f}% | {m['profit_factor']:.2f} | "
                 f"${m['final_equity']:,.0f} |")

    L.append("\n## CAGR vs BASELINE\n")
    L.append("| Variant | CAGR | Delta vs baseline | Entries taken | Entries blocked |")
    L.append("|---|---|---|---|---|")
    base_c = results["BASELINE"]["metrics"]["cagr"]
    for v, r in results.items():
        c = r["metrics"]["cagr"]
        delta = c - base_c
        L.append(f"| {v} | {c*100:.2f}% | {delta*100:+.2f}pp | "
                 f"{r['entries_taken']} | {r['entries_blocked']} |")

    L.append("\n## Regime Gate (HC #428 R1) — green/red day Sharpe gap\n")
    L.append("| Variant | n_green | n_red | n_flat | Sh green | Sh red | Sh flat | gap | pass |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for v, r in results.items():
        g = r["gate"]
        def f2(x): return f"{x:.2f}" if (x is not None and np.isfinite(x)) else "nan"
        L.append(f"| {v} | {g['n_green']} | {g['n_red']} | {g['n_flat']} | "
                 f"{f2(g['sharpe_green'])} | {f2(g['sharpe_red'])} | {f2(g['sharpe_flat'])} | "
                 f"{g['regime_gap']:.2f} | {'PASS' if g['hc428_r1_pass'] else 'FAIL'} |")

    L.append("\n## Deploy Gates\n")
    L.append("| Variant | Sharpe>=1.0 | Calmar>=1.5 | Regime gap<=0.50 | Day conc<=0.70 | In-mkt>=40% | DEPLOY |")
    L.append("|---|---|---|---|---|---|---|")
    for v, r in results.items():
        gp = r["gates_pass"]
        x = lambda b: "PASS" if b else "FAIL"
        L.append(f"| {v} | {x(gp['sharpe_ge_1.0'])} | {x(gp['calmar_ge_1.5'])} | "
                 f"{x(gp['regime_gap_le_0.50'])} | {x(gp['day_conc_le_0.70'])} | "
                 f"{x(gp['in_market_ge_40'])} | **{'YES' if r['deploy_ready'] else 'NO'}** |")

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
        # pick the best by Sharpe among passing
        best = max(passing, key=lambda v: results[v]["metrics"]["sharpe"])
        L.append(f"- **PASS**: {', '.join(passing)} clear all HC #428 deploy gates.")
        L.append(f"- **Recommended winner**: {best} (highest Sharpe among passers).")
        L.append(f"- Wire `regime_gate` block matching {best}'s config into wheel_paper_engine.py.")
        gc = results[best]["gate_cfg"]
        L.append(f"- Config: `{json.dumps(gc)}`")
    else:
        L.append("- **No variant passes all deploy gates.** Recommendation: keep "
                 "`entries_paused: true` in wheel_paper_engine.py and do NOT add regime_gate "
                 "logic. Investigate whether the wheel's directional short-vol exposure can "
                 "be made regime-agnostic at all, or pivot to a different income strategy.")
        # diagnose closest passer
        L.append("\n### Why each variant failed\n")
        for v, r in results.items():
            if v == "BASELINE": continue
            failed = [k for k, val in r["gates_pass"].items() if not val]
            L.append(f"- **{v}** (in-mkt {r['in_market_pct']:.0f}%, "
                     f"Sharpe {r['metrics']['sharpe']:.2f}, "
                     f"Calmar {r['metrics']['calmar']:.2f}, "
                     f"regime gap {r['gate']['regime_gap']:.2f}): "
                     f"failed [{', '.join(failed)}]")

    L.append("\n## Honest caveats\n")
    L.append("- BS with modeled ATM sigma, no skew. Same pricing limitations as the "
             "hedge-overlay study; gate decisions are based on freely-available SPY/VIX "
             "data so this transfers cleanly to live paper trading via yfinance.")
    L.append("- Gates use spot prices and indicators known at the open of each trading day. "
             "Realized-vol and 30d-high are computed on prior close, so no look-ahead.")
    L.append("- A regime gate that flattens green/red Sharpe gap by REDUCING TIME IN MARKET "
             "must still pass the 40% in-market floor to be deployable. A gate that's open "
             "20% of the time produces a 'great Sharpe' on a tiny sample - not robust.")
    L.append("- Even if a variant passes the gates, BS without skew under-prices OTM puts; "
             "real chain premiums will be ~10-20% higher, slightly reducing realized P&L.")

    report_path = REPORT_DIR / "wheel_regime_gated_spy_v1.md"
    report_path.write_text("\n".join(L))
    print(f"\n[done] report -> {report_path}")
    print(f"[done] artifacts -> {OUT_DIR}")
    return results


if __name__ == "__main__":
    main()
