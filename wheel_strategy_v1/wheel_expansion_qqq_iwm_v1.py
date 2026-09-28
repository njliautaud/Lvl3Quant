#!/usr/bin/env python3
"""
wheel_expansion_qqq_iwm_v1.py — Tier2_Balanced_Scalp on QQQ and IWM (single underlying each).

Mirrors the live wheel_paper_engine.py ruleset (put_delta=0.22, DTE 30-45,
profit_take 50%, roll DTE<10, VIX gate 32.0, leverage 1.0) and the v1 wheel
engine mechanics (CSP -> assignment -> CC -> back to cash), using the cached
sector_etfs.parquet prices and iv_features.parquet sigma series.

Applies HC #428 R1 gates:
  - regime stratification by SPY close-to-close ret
  - regime-gap test: |Sharpe_green - Sharpe_red| / max(|.|) <= 0.50
  - day-concentration cap (top trade P&L / total) <= 0.70
  - Sharpe >= 1.0, Calmar >= 1.5

Outputs:
  - /home/jupiter/Lvl3Quant/research/findings/wheel_expansion_qqq_iwm_v1.md
  - per-ticker equity parquet + ledger parquet under
    /home/jupiter/Lvl3Quant/output/wheel_expansion_qqq_iwm_v1/
  - MLflow runs under experiment 'wheel_expansion_qqq_iwm'
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------- paths -----------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "wheel_expansion_qqq_iwm_v1"
REPORT_DIR = ROOT / "research" / "findings"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------- config ----------------------------------
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

STARTING_CASH = 20_000.0       # per task — $20k account for the % vs $ framing
RISK_FREE = 0.04               # flat r for BS pricing
TRADING_DAYS = 252

# slippage params, matching wheel_engine defaults
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN_PER_SHARE = 0.03
COST_PER_CONTRACT = 0.65       # broker + reg fees per contract per leg

START_DATE = pd.Timestamp("2018-01-01")
END_DATE   = pd.Timestamp("2025-12-31")

# ------------------------------- BS helpers ------------------------------
SQRT_2PI = math.sqrt(2.0 * math.pi)


def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def bs_price(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE, kind: str = "put") -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if kind == "put":
            return max(K - S, 0.0)
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
    return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)


def bs_delta(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE, kind: str = "put") -> float:
    if T <= 0 or sigma <= 0:
        if kind == "put":
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    if kind == "put":
        return _Phi(d1) - 1.0
    return _Phi(d1)


def strike_from_delta(S: float, T: float, sigma: float, target_delta: float, kind: str = "put") -> float:
    """Solve for strike that gives |delta| = target_delta. Uses analytic inversion."""
    if T <= 0 or sigma <= 0:
        return S
    # For put: delta = N(d1)-1, so N(d1) = 1-|target_delta|. Solve d1.
    # For call: delta = N(d1), N(d1) = target_delta.
    p = (1.0 - abs(target_delta)) if kind == "put" else abs(target_delta)
    # Inverse normal — Acklam's approximation (fast & accurate enough)
    p = min(max(p, 1e-9), 1 - 1e-9)
    # rational approximation
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d_ = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
          3.754408661907416]
    pl = 0.02425
    pu = 1 - pl
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
    # d1 = (ln(S/K) + (r + sigma^2/2) T) / (sigma sqrt T)
    # => K = S * exp((r + sigma^2/2) T - d1 sigma sqrt T)
    K = S * math.exp((RISK_FREE + 0.5 * sigma * sigma) * T - d1 * sigma * math.sqrt(T))
    # Round to penny (ETF options) but enforce sensible bounds
    K = round(K, 2)
    return max(0.01, K)


def slippage_per_share(premium: float) -> float:
    if premium is None or premium <= 0:
        return 0.0
    return max(SLIPPAGE_MIN_PER_SHARE, SLIPPAGE_FRAC * premium)


# ------------------------------- state -----------------------------------
@dataclass
class Position:
    underlying: str
    side: str                # 'short_put' | 'long_shares' | 'short_call'
    strike: float
    expiry: pd.Timestamp
    open_date: pd.Timestamp
    open_price: float        # premium per share (credit received) for options; cost basis for shares
    contracts: int           # 1 contract = 100 shares
    open_sigma: float
    max_profit: float = 0.0  # premium received at open for short options


@dataclass
class State:
    cash: float
    positions: list = field(default_factory=list)
    ledger: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


# ------------------------------- data ------------------------------------
def load_data(ticker: str):
    """Returns (prices, iv, macro, spy_ret)."""
    sec = pd.read_parquet(CACHE / "sector_etfs.parquet")
    px = sec[sec["ticker"] == ticker][["date", "close"]].copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values("date").set_index("date")

    iv = pd.read_parquet(CACHE / "iv_features.parquet")
    iv = iv[iv["ticker"] == ticker][["date", "sigma", "iv_rank"]].copy()
    iv["date"] = pd.to_datetime(iv["date"])
    iv = iv.sort_values("date").set_index("date")

    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.sort_values("date").set_index("date")

    spy = sec[sec["ticker"] == "SPY"][["date", "close"]].copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")
    spy["spy_ret"] = spy["close"].pct_change()

    df = px.join(iv, how="inner").join(macro, how="left").join(spy[["spy_ret"]], how="left")
    df["vix"] = df["vix"].ffill()
    df = df.loc[START_DATE:END_DATE].dropna(subset=["close", "sigma"])
    df["sigma"] = df["sigma"].clip(lower=0.05, upper=1.5)
    return df


# ------------------------------- backtest ---------------------------------
def run_single_underlying(ticker: str, df: pd.DataFrame, cfg: dict, starting_cash: float):
    state = State(cash=starting_cash)
    dte_target = cfg["dte_target"]
    profit_take = cfg["profit_take_pct"]
    roll_dte = cfg["roll_dte_trigger"]
    vix_gate = cfg["vix_max_gate"]
    put_delta_t = cfg["put_delta_target"]
    call_delta_t = cfg["call_delta_target"]
    lev = cfg["leverage"]

    dates = df.index.to_list()
    # quick day lookup
    date_set = set(dates)

    # find the expiry on-or-after target_date that is closest to target_dte
    def find_expiry(open_date: pd.Timestamp) -> pd.Timestamp:
        target = open_date + pd.Timedelta(days=dte_target)
        # Use 3rd Friday of next or following month as proxy expiry
        # Simpler: snap to nearest calendar day in window [dte_min, dte_max]
        # Use a Friday in the window so expiry-day mechanics are realistic
        best = None
        best_dist = 10_000
        for d_off in range(cfg["dte_min"], cfg["dte_max"] + 1):
            cand = open_date + pd.Timedelta(days=d_off)
            # snap to Friday: weekday 4
            shift = (4 - cand.weekday()) % 7
            cand_fri = cand + pd.Timedelta(days=shift)
            dte = (cand_fri - open_date).days
            if dte < cfg["dte_min"] or dte > cfg["dte_max"]:
                continue
            dist = abs(dte - dte_target)
            if dist < best_dist:
                best = cand_fri
                best_dist = dist
        return best

    for i, today in enumerate(dates):
        row = df.loc[today]
        S = float(row["close"])
        sigma = float(row["sigma"])
        vix = float(row["vix"]) if pd.notna(row["vix"]) else 0.0

        # ----- process existing positions -----
        new_positions = []
        for pos in state.positions:
            T = max((pos.expiry - today).days, 0) / 365.0
            # Mark-to-market option value
            if pos.side == "short_put":
                opt = bs_price(S, pos.strike, T, sigma, kind="put")
                pnl_per_share = pos.open_price - opt
                profit_frac = pnl_per_share / pos.open_price if pos.open_price > 0 else 0.0
                close_for_profit = profit_frac >= profit_take
                close_for_roll = (pos.expiry - today).days <= roll_dte
                # Expiry day handling
                is_expiry = today >= pos.expiry
                if close_for_profit or close_for_roll or is_expiry:
                    if is_expiry and S < pos.strike:
                        # assignment — buy 100*contracts shares at strike
                        cost = pos.strike * 100 * pos.contracts
                        state.cash -= cost
                        state.cash -= COST_PER_CONTRACT * pos.contracts
                        state.ledger.append({
                            "underlying": pos.underlying, "open_date": pos.open_date,
                            "close_date": today, "kind": "csp_assigned",
                            "strike": pos.strike, "S_open": None,
                            "S_close": S, "premium_open": pos.open_price,
                            "premium_close": 0.0,
                            "realized_pnl_per_share": pos.open_price,
                            "realized_pnl": pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts,
                        })
                        # Convert into long_shares position with cost basis = strike - premium_received
                        new_positions.append(Position(
                            underlying=pos.underlying, side="long_shares",
                            strike=pos.strike - pos.open_price,  # net basis
                            expiry=today,
                            open_date=today,
                            open_price=pos.strike - pos.open_price,
                            contracts=pos.contracts,
                            open_sigma=sigma,
                            max_profit=0.0,
                        ))
                    else:
                        # close out option at current price + slippage to cross
                        slip = slippage_per_share(opt)
                        cost = (opt + slip) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                        realized = pos.open_price * 100 * pos.contracts - cost
                        state.cash += realized
                        # Note: premium was already credited at open
                        state.ledger.append({
                            "underlying": pos.underlying, "open_date": pos.open_date,
                            "close_date": today, "kind": "csp_closed",
                            "strike": pos.strike, "S_open": None,
                            "S_close": S, "premium_open": pos.open_price,
                            "premium_close": opt,
                            "realized_pnl_per_share": pos.open_price - opt - slip,
                            "realized_pnl": realized,
                        })
                else:
                    new_positions.append(pos)

            elif pos.side == "long_shares":
                # Immediately roll into short_call next day in the loop; just keep
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
                        # called away — sell shares at strike
                        proceeds = pos.strike * 100 * pos.contracts
                        cost_basis = pos.open_price  # carried from share state? Need share basis. Use strike attr.
                        # We stored share basis on the long_shares conversion above; here pos has no basis.
                        # For simplicity, carry share basis on the short_call as well via pos.max_profit.
                        share_basis = pos.max_profit if pos.max_profit > 0 else pos.strike
                        share_pnl = (pos.strike - share_basis) * 100 * pos.contracts
                        premium_kept = pos.open_price * 100 * pos.contracts
                        state.cash += proceeds + premium_kept - COST_PER_CONTRACT * pos.contracts
                        # remove the shares value (they're now sold)
                        # We don't track shares as cash; assignment already debited at strike before, so
                        # the share basis at assignment time was already paid. We add back proceeds.
                        state.ledger.append({
                            "underlying": pos.underlying, "open_date": pos.open_date,
                            "close_date": today, "kind": "cc_called_away",
                            "strike": pos.strike, "S_open": None,
                            "S_close": S, "premium_open": pos.open_price,
                            "premium_close": 0.0,
                            "realized_pnl_per_share": pos.open_price + (pos.strike - share_basis),
                            "realized_pnl": premium_kept + share_pnl - COST_PER_CONTRACT * pos.contracts,
                        })
                        # do NOT carry forward shares
                    else:
                        slip = slippage_per_share(opt)
                        cost = (opt + slip) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                        realized = pos.open_price * 100 * pos.contracts - cost
                        state.cash += realized
                        state.ledger.append({
                            "underlying": pos.underlying, "open_date": pos.open_date,
                            "close_date": today, "kind": "cc_closed",
                            "strike": pos.strike, "S_open": None,
                            "S_close": S, "premium_open": pos.open_price,
                            "premium_close": opt,
                            "realized_pnl_per_share": pos.open_price - opt - slip,
                            "realized_pnl": realized,
                        })
                        # Convert back to long_shares — still hold shares
                        share_basis = pos.max_profit if pos.max_profit > 0 else pos.strike
                        new_positions.append(Position(
                            underlying=pos.underlying, side="long_shares",
                            strike=share_basis, expiry=today, open_date=today,
                            open_price=share_basis, contracts=pos.contracts,
                            open_sigma=sigma, max_profit=share_basis,
                        ))

        state.positions = new_positions

        # ----- decide if we should open a new position -----
        # Open short_put when in cash & VIX OK
        has_option = any(p.side in ("short_put", "short_call") for p in state.positions)
        has_shares = any(p.side == "long_shares" for p in state.positions)

        # convert long_shares to short_call (if any shares uncovered)
        if has_shares and not any(p.side == "short_call" for p in state.positions):
            updated = []
            for p in state.positions:
                if p.side == "long_shares" and vix <= vix_gate:
                    expiry = find_expiry(today)
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
                        open_sigma=sigma,
                        max_profit=p.open_price,  # carry share basis on max_profit
                    ))
                else:
                    updated.append(p)
            state.positions = updated

        # open short_put when in cash
        if not has_option and not has_shares and vix <= vix_gate:
            expiry = find_expiry(today)
            if expiry is not None:
                T = (expiry - today).days / 365.0
                K_put = strike_from_delta(S, T, sigma, put_delta_t, kind="put")
                premium = bs_price(S, K_put, T, sigma, kind="put")
                slip = slippage_per_share(premium)
                # Sizing: we need collateral = strike * 100 * contracts.
                # Use leverage * cash for collateral budget.
                # Use 80% of available cash to leave a buffer.
                collateral_budget = state.cash * lev * 0.95
                contracts = int(collateral_budget // (K_put * 100))
                if contracts >= 1 and (premium - slip) > 0.05:
                    credit = (premium - slip) * 100 * contracts - COST_PER_CONTRACT * contracts
                    state.cash += credit
                    state.positions.append(Position(
                        underlying=ticker, side="short_put",
                        strike=K_put, expiry=expiry, open_date=today,
                        open_price=premium - slip, contracts=contracts,
                        open_sigma=sigma, max_profit=(premium - slip) * 100 * contracts,
                    ))

        # ----- mark-to-market equity -----
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
        state.equity_curve.append({"date": today, "equity": equity, "S": S, "vix": vix,
                                   "spy_ret": float(row["spy_ret"]) if pd.notna(row["spy_ret"]) else np.nan})

    eq_df = pd.DataFrame(state.equity_curve).set_index("date")
    eq_df["ret"] = eq_df["equity"].pct_change()
    ledger_df = pd.DataFrame(state.ledger)
    return eq_df, ledger_df


# ------------------------------- metrics ---------------------------------
def compute_metrics(eq_df: pd.DataFrame) -> dict:
    rets = eq_df["ret"].dropna()
    if len(rets) < 2:
        return {}
    years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
    cagr = (eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    mu = rets.mean()
    sd = rets.std()
    downside = rets[rets < 0].std()
    sharpe = (mu / sd) * math.sqrt(TRADING_DAYS) if sd > 0 else float("nan")
    sortino = (mu / downside) * math.sqrt(TRADING_DAYS) if downside and downside > 0 else float("nan")
    # max DD
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


def regime_gate(eq_df: pd.DataFrame) -> dict:
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
                abs(sh_r) if np.isfinite(sh_r) else 0,
                1e-9)
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


def day_concentration(ledger_df: pd.DataFrame) -> float:
    if ledger_df is None or ledger_df.empty or "realized_pnl" not in ledger_df.columns:
        return float("nan")
    pos = ledger_df[ledger_df["realized_pnl"] > 0]["realized_pnl"]
    if len(pos) == 0:
        return float("nan")
    return float(pos.max() / pos.sum())


def tail_event_stats(eq_df: pd.DataFrame, label: str, start: str, end: str) -> dict:
    sub = eq_df.loc[start:end].copy()
    if len(sub) < 2:
        return {"label": label, "n_days": 0}
    peak = sub["equity"].cummax()
    dd = sub["equity"] / peak - 1.0
    max_dd = float(dd.min())
    trough = dd.idxmin()
    # recovery: first day post-trough where equity >= pre-event peak (peak before window)
    pre = eq_df.loc[:sub.index[0]]
    pre_peak = float(pre["equity"].max()) if len(pre) else float(sub["equity"].iloc[0])
    after = eq_df.loc[trough:]
    rec = after[after["equity"] >= pre_peak]
    rec_days = int((rec.index[0] - trough).days) if len(rec) else None
    return {
        "label": label,
        "window": f"{start} to {end}",
        "n_days": int(len(sub)),
        "max_dd_pct": max_dd * 100,
        "vix_peak": float(sub["vix"].max()) if "vix" in sub.columns else None,
        "recover_days": rec_days,
        "cum_ret_pct": float((sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1.0) * 100.0),
    }


# ------------------------------- mlflow ----------------------------------
def log_to_mlflow(ticker: str, metrics: dict, gate: dict, day_conc: float, tail: list,
                  eq_df: pd.DataFrame, cfg: dict):
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("wheel_expansion_qqq_iwm")
        with mlflow.start_run(run_name=f"tier2_balanced_scalp_{ticker}"):
            mlflow.log_params(cfg)
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("start", str(eq_df.index[0].date()))
            mlflow.log_param("end", str(eq_df.index[-1].date()))
            mlflow.log_param("starting_cash", STARTING_CASH)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(k, v)
            for k, v in gate.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(f"regime_{k}", v)
            mlflow.log_metric("day_concentration", day_conc if np.isfinite(day_conc) else -1.0)
            mlflow.log_metric("hc428_r1_pass", 1.0 if gate.get("hc428_r1_pass") else 0.0)
            for t in tail:
                if "max_dd_pct" in t:
                    safe = t["label"].replace(" ", "_")
                    mlflow.log_metric(f"tail_{safe}_max_dd_pct", t["max_dd_pct"])
                    if t.get("recover_days") is not None:
                        mlflow.log_metric(f"tail_{safe}_recover_days", t["recover_days"])
            # equity csv artifact
            csv_path = OUT_DIR / f"equity_{ticker}.csv"
            eq_df.to_csv(csv_path)
            mlflow.log_artifact(str(csv_path))
            return mlflow.active_run().info.run_id
    except Exception as e:
        print(f"[mlflow] skipped ({e})")
        return None


# ------------------------------- main ------------------------------------
def main():
    results = {}
    for ticker in ("QQQ", "IWM"):
        print(f"\n[{ticker}] loading data...")
        df = load_data(ticker)
        print(f"[{ticker}] {len(df)} trading days {df.index[0].date()} -> {df.index[-1].date()}")
        eq_df, ledger_df = run_single_underlying(ticker, df, TIER2_BALANCED_SCALP, STARTING_CASH)
        metrics = compute_metrics(eq_df)
        gate = regime_gate(eq_df)
        day_conc = day_concentration(ledger_df)
        tail = [
            tail_event_stats(eq_df, "COVID_2020", "2020-02-15", "2020-05-31"),
            tail_event_stats(eq_df, "2022_bear", "2022-01-01", "2022-12-31"),
            tail_event_stats(eq_df, "Aug_2024_carry", "2024-07-15", "2024-09-15"),
        ]

        # Hard gates per task
        gates_pass = {
            "sharpe_ge_1.0": metrics["sharpe"] >= 1.0 if np.isfinite(metrics.get("sharpe", float("nan"))) else False,
            "calmar_ge_1.5": metrics["calmar"] >= 1.5 if np.isfinite(metrics.get("calmar", float("nan"))) else False,
            "regime_gap_le_0.50": gate["hc428_r1_pass"],
            "day_conc_le_0.70": (day_conc <= 0.70) if np.isfinite(day_conc) else False,
            "n_days_ge_40": metrics.get("n_days", 0) >= 40,
        }
        deploy_ready = all(gates_pass.values())

        # save artifacts
        eq_df.to_parquet(OUT_DIR / f"equity_{ticker}.parquet")
        if not ledger_df.empty:
            ledger_df.to_parquet(OUT_DIR / f"ledger_{ticker}.parquet")
        with open(OUT_DIR / f"results_{ticker}.json", "w") as f:
            json.dump({"ticker": ticker, "metrics": metrics, "gate": gate,
                       "day_conc": day_conc, "tail": tail,
                       "gates_pass": gates_pass, "deploy_ready": deploy_ready},
                      f, indent=2, default=str)

        run_id = log_to_mlflow(ticker, metrics, gate, day_conc, tail, eq_df, TIER2_BALANCED_SCALP)
        results[ticker] = {
            "metrics": metrics, "gate": gate, "day_conc": day_conc,
            "tail": tail, "gates_pass": gates_pass, "deploy_ready": deploy_ready,
            "mlflow_run_id": run_id,
        }
        print(f"[{ticker}] Sharpe={metrics['sharpe']:.2f} CAGR={metrics['cagr']*100:.1f}% "
              f"MaxDD={metrics['max_dd']*100:.1f}% Calmar={metrics['calmar']:.2f} "
              f"regime_gap={gate['regime_gap']:.2f} deploy_ready={deploy_ready}")

    # ---------- markdown report ----------
    L = []
    L.append("# Wheel Expansion — QQQ & IWM, Tier2 Balanced Scalp\n")
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Window: {START_DATE.date()} to {END_DATE.date()}")
    L.append(f"Starting cash: ${STARTING_CASH:,.0f}, leverage 1.0x, single-underlying\n")
    L.append("Config: put_delta 0.22, call_delta 0.22, DTE 30-45, profit-take 50%, "
             "roll DTE<10, VIX gate 32.0\n")
    L.append("Pricing: Black-Scholes with modeled ATM sigma (iv_features.parquet). ")
    L.append("Slippage: max(2.5% of premium, $0.03/share/leg). Commission: $0.65/contract/leg.\n")
    L.append("Regime classification: SPY close-to-close, threshold ±0.5σ.\n")
    L.append("Note on chain data: live-market option chains for 2018-2025 on QQQ/IWM ")
    L.append("are not in the local cache. This run uses delta-replicated synthetic ")
    L.append("chains (BS at the modeled ATM sigma) per the task fallback. Headline ")
    L.append("metrics here represent a first-look estimate, not vendor-priced fills.\n")

    L.append("\n## Headline Metrics\n")
    L.append("| Ticker | CAGR | Sharpe | Sortino | MaxDD | Calmar | WR | PF | Total Ret | Final $ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for tk, r in results.items():
        m = r["metrics"]
        L.append(f"| {tk} | {m['cagr']*100:.1f}% | {m['sharpe']:.2f} | {m['sortino']:.2f} | "
                 f"{m['max_dd']*100:.1f}% | {m['calmar']:.2f} | {m['win_rate']*100:.1f}% | "
                 f"{m['profit_factor']:.2f} | {m['total_return_pct']:.1f}% | ${m['final_equity']:,.0f} |")

    L.append("\n## Regime Gate (HC #428 R1)\n")
    L.append("| Ticker | n_green | n_red | n_flat | Sh green | Sh red | Sh flat | gap | pass |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for tk, r in results.items():
        g = r["gate"]
        L.append(f"| {tk} | {g['n_green']} | {g['n_red']} | {g['n_flat']} | "
                 f"{(g['sharpe_green'] or float('nan')):.2f} | "
                 f"{(g['sharpe_red'] or float('nan')):.2f} | "
                 f"{(g['sharpe_flat'] or float('nan')):.2f} | "
                 f"{g['regime_gap']:.2f} | {'PASS' if g['hc428_r1_pass'] else 'FAIL'} |")

    L.append("\n## Deploy Gates\n")
    L.append("| Ticker | Sharpe≥1.0 | Calmar≥1.5 | Regime gap≤0.50 | Day conc≤0.70 | n_days≥40 | DEPLOY READY |")
    L.append("|---|---|---|---|---|---|---|")
    for tk, r in results.items():
        gp = r["gates_pass"]
        def x(b): return "PASS" if b else "FAIL"
        L.append(f"| {tk} | {x(gp['sharpe_ge_1.0'])} | {x(gp['calmar_ge_1.5'])} | "
                 f"{x(gp['regime_gap_le_0.50'])} | {x(gp['day_conc_le_0.70'])} | "
                 f"{x(gp['n_days_ge_40'])} | **{'YES' if r['deploy_ready'] else 'NO'}** |")

    L.append("\n## Tail Event Stress\n")
    L.append("| Ticker | Event | MaxDD | VIX peak | Recover days | Cum ret |")
    L.append("|---|---|---|---|---|---|")
    for tk, r in results.items():
        for t in r["tail"]:
            if t.get("n_days", 0) == 0:
                continue
            L.append(f"| {tk} | {t['label']} | {t['max_dd_pct']:.1f}% | "
                     f"{(t.get('vix_peak') or 0):.1f} | "
                     f"{t.get('recover_days') if t.get('recover_days') is not None else 'not_recovered'} | "
                     f"{t['cum_ret_pct']:.1f}% |")

    L.append("\n## Recommendation\n")
    any_pass = any(r["deploy_ready"] for r in results.values())
    for tk, r in results.items():
        if r["deploy_ready"]:
            L.append(f"- **{tk}: DEPLOY-READY** — passes all gates. Wire into multi-underlying engine.")
        else:
            failed = [k for k, v in r["gates_pass"].items() if not v]
            L.append(f"- **{tk}: SHELVE** — fails: {', '.join(failed)}.")
    if not any_pass:
        L.append("\nNeither ticker passes all deploy gates with synthetic-chain pricing. "
                 "Before final reject, re-run with vendor option chains "
                 "(Polygon/CBOE) so spreads/IV skew aren't approximated.")
    else:
        L.append("\nMulti-underlying support should be added with a `underlying_basket` "
                 "config flag, position-sizing split evenly, and a one-position-per-underlying "
                 "rule. Leave it OFF in the live paper daemon until the user toggles it on.")

    L.append("\n## Caveats\n")
    L.append("- Pricing is BS with modeled ATM sigma — no real bid/ask, no skew. ")
    L.append("Real fills will differ; expect ~5-15% premium haircut on QQQ/IWM at 30-45 DTE 22Δ.")
    L.append("- The original SPY t=7.45 alpha figure was from a MULTI-NAME tier ladder backtest "
             "(~329 names, full IV-rank gating). Comparing single-underlying ETF wheels to that ")
    L.append("number is apples-to-oranges; this report's purpose is a relative QQQ-vs-IWM-vs-SPY check.")
    L.append("- VIX gate at 32 means the strategy STOPS opening new positions during high-vol — ")
    L.append("this is what produces the regime gap (most red days have elevated VIX).")

    report_path = REPORT_DIR / "wheel_expansion_qqq_iwm_v1.md"
    report_path.write_text("\n".join(L))
    print(f"\n[done] report -> {report_path}")
    print(f"[done] artifacts -> {OUT_DIR}")
    return results


if __name__ == "__main__":
    main()
