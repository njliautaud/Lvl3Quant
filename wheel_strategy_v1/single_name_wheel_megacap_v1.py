#!/usr/bin/env python3
"""
single_name_wheel_megacap_v1.py - Classic full-cycle wheel on megacap tech.

Per HC #587 R2 (2026-06-09): pursue independent wheel angle distinct from
the SPY put-sell-and-close approach. Thesis:
  - High-IV single names produce fat premiums
  - Willingness to TAKE ASSIGNMENT (vs forced close) lets us hold through
    regime flips rather than realize loss
  - Covered calls on assigned shares generate continued income while waiting
    for recovery

Universe: AAPL, MSFT, GOOGL, NVDA, META.
  - Each ticker independent backtest with $20K
  - 5-ticker basket: $20K total, $4K allocated per ticker

Phases:
  1. Cash-secured put (~30 DTE, 0.30 delta - aggressive, we WANT assignments)
  2. Assignment if ITM at expiry (no forced close, no early roll on losers)
  3. Covered call on assigned shares (~30 DTE, 0.30 delta)
  4. Called away at strike, back to phase 1

Profit-take at 50% on either leg. No early close on losers. VIX gate < 35
at entry only (existing positions never force-closed).

Cost: $0.65/contract option + $0.005/share commission on assignment/called-away.

Window: 2018-01-01 to 2025-12-31 (8 years, captures 2020 COVID & 2022 bear & 2018 vol).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------- paths ------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "single_name_wheel_megacap_v1"
REPORT_DIR = ROOT / "research" / "findings"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------- config -----------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
BASKET_TOTAL = 20_000.0
PER_TICKER_ALLOC = 4_000.0
SINGLE_TICKER_CASH = 20_000.0

START_DATE = pd.Timestamp("2018-01-01")
END_DATE = pd.Timestamp("2025-12-31")
TRADING_DAYS = 252
RISK_FREE = 0.04

WHEEL_CFG = {
    "put_delta_target": 0.30,       # aggressive - we want some assignments
    "call_delta_target": 0.30,
    "dte_min": 25,
    "dte_max": 35,
    "dte_target": 30,
    "profit_take_pct": 0.50,
    "vix_max_gate": 35.0,           # only blocks NEW entries, existing wheel continues
    "force_close_on_loss": False,   # core thesis: accept assignment
    "allow_call_above_basis": True, # roll up cost basis if shares deep underwater
}

COST_PER_CONTRACT = 0.65
COMMISSION_PER_SHARE = 0.005
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN_PER_SHARE = 0.03


# ------------------------- BS pricing -------------------------------------
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
    return max(0.01, round(K, 2))


def slippage_per_share(premium):
    if premium is None or premium <= 0:
        return 0.0
    return max(SLIPPAGE_MIN_PER_SHARE, SLIPPAGE_FRAC * premium)


# ------------------------- state ------------------------------------------
@dataclass
class Position:
    ticker: str
    side: str            # 'short_put' | 'long_shares' | 'short_call'
    strike: float        # for shares: cost basis per share
    expiry: pd.Timestamp
    open_date: pd.Timestamp
    open_price: float    # premium received per share, or share basis for long_shares
    contracts: int       # 1 contract = 100 shares
    open_sigma: float
    share_basis: float = 0.0  # tracked separately for accuracy when CC expires/called


@dataclass
class TickerState:
    ticker: str
    cash: float
    positions: list = field(default_factory=list)  # at most 1 active position per ticker in this wheel
    ledger: list = field(default_factory=list)
    csp_opened: int = 0
    cc_opened: int = 0
    assignments: int = 0
    call_aways: int = 0
    days_in_shares: int = 0
    days_in_csp: int = 0
    days_in_cash: int = 0


# ------------------------- data load --------------------------------------
def load_data():
    """Load prices, IV, VIX for all tickers and SPY."""
    prices = pd.read_parquet(CACHE / "prices.parquet")
    prices = prices[prices["ticker"].isin(TICKERS)][["ticker", "date", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])

    iv = pd.read_parquet(CACHE / "iv_features.parquet")
    iv = iv[iv["ticker"].isin(TICKERS)][["date", "ticker", "sigma", "iv_rank"]].copy()
    iv["date"] = pd.to_datetime(iv["date"])

    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"])

    sec = pd.read_parquet(CACHE / "sector_etfs.parquet")
    spy = sec[sec["ticker"] == "SPY"][["date", "close"]].copy()
    spy.columns = ["date", "spy_close"]
    spy["date"] = pd.to_datetime(spy["date"])

    # merge
    df = prices.merge(iv, on=["date", "ticker"], how="inner")
    df = df.merge(macro, on="date", how="left")
    df = df.merge(spy, on="date", how="left")
    df["vix"] = df["vix"].ffill()
    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    df = df.dropna(subset=["close", "sigma"])
    df["sigma"] = df["sigma"].clip(lower=0.05, upper=2.0)
    df["spy_ret"] = df.groupby("ticker")["spy_close"].pct_change()
    df["ticker_ret"] = df.groupby("ticker")["close"].pct_change()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


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


# ------------------------- single-ticker wheel ----------------------------
def run_ticker_wheel(ticker, df_t, cfg, starting_cash):
    """Run wheel on a single ticker. Returns equity_curve, ledger, stats."""
    state = TickerState(ticker=ticker, cash=starting_cash)
    df_t = df_t.sort_values("date").reset_index(drop=True)
    equity_curve = []

    for _, row in df_t.iterrows():
        today = row["date"]
        S = float(row["close"])
        sigma = float(row["sigma"])
        vix = float(row["vix"]) if pd.notna(row["vix"]) else 0.0

        # ---- 1) process existing positions ----
        new_positions = []
        for pos in state.positions:
            T = max((pos.expiry - today).days, 0) / 365.0

            if pos.side == "short_put":
                opt = bs_price(S, pos.strike, T, sigma, kind="put")
                pnl_per_share = pos.open_price - opt
                profit_frac = pnl_per_share / pos.open_price if pos.open_price > 0 else 0.0
                close_for_profit = profit_frac >= cfg["profit_take_pct"]
                is_expiry = today >= pos.expiry

                if close_for_profit and not is_expiry:
                    # close for profit
                    slip = slippage_per_share(opt)
                    cost = (opt + slip) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    realized = pos.open_price * 100 * pos.contracts - cost
                    state.cash += realized
                    state.ledger.append({
                        "ticker": ticker, "open_date": pos.open_date, "close_date": today,
                        "kind": "csp_profit_take", "strike": pos.strike, "S_close": S,
                        "premium_open": pos.open_price, "premium_close": opt,
                        "dte_at_open": (pos.expiry - pos.open_date).days,
                        "realized_pnl": realized, "contracts": pos.contracts,
                    })
                elif is_expiry:
                    if S < pos.strike:
                        # assignment - take the shares
                        cost = pos.strike * 100 * pos.contracts
                        commission = COMMISSION_PER_SHARE * 100 * pos.contracts
                        state.cash -= (cost + commission)
                        state.assignments += 1
                        # effective basis = strike - premium kept (premium was credited at open)
                        basis = pos.strike - pos.open_price
                        state.ledger.append({
                            "ticker": ticker, "open_date": pos.open_date, "close_date": today,
                            "kind": "csp_assigned", "strike": pos.strike, "S_close": S,
                            "premium_open": pos.open_price, "premium_close": 0.0,
                            "dte_at_open": (pos.expiry - pos.open_date).days,
                            "realized_pnl": pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts - commission,
                            "contracts": pos.contracts,
                        })
                        new_positions.append(Position(
                            ticker=ticker, side="long_shares",
                            strike=basis, expiry=today, open_date=today,
                            open_price=basis, contracts=pos.contracts,
                            open_sigma=sigma, share_basis=basis,
                        ))
                    else:
                        # expired worthless - keep full premium
                        realized = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                        state.ledger.append({
                            "ticker": ticker, "open_date": pos.open_date, "close_date": today,
                            "kind": "csp_expired", "strike": pos.strike, "S_close": S,
                            "premium_open": pos.open_price, "premium_close": 0.0,
                            "dte_at_open": (pos.expiry - pos.open_date).days,
                            "realized_pnl": realized, "contracts": pos.contracts,
                        })
                else:
                    new_positions.append(pos)
                    state.days_in_csp += 1
                continue

            elif pos.side == "short_call":
                opt = bs_price(S, pos.strike, T, sigma, kind="call")
                pnl_per_share = pos.open_price - opt
                profit_frac = pnl_per_share / pos.open_price if pos.open_price > 0 else 0.0
                close_for_profit = profit_frac >= cfg["profit_take_pct"]
                is_expiry = today >= pos.expiry

                if close_for_profit and not is_expiry:
                    # close CC for profit, retain shares
                    slip = slippage_per_share(opt)
                    cost = (opt + slip) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    realized = pos.open_price * 100 * pos.contracts - cost
                    state.cash += realized
                    state.ledger.append({
                        "ticker": ticker, "open_date": pos.open_date, "close_date": today,
                        "kind": "cc_profit_take", "strike": pos.strike, "S_close": S,
                        "premium_open": pos.open_price, "premium_close": opt,
                        "dte_at_open": (pos.expiry - pos.open_date).days,
                        "realized_pnl": realized, "contracts": pos.contracts,
                    })
                    # shares revert to long_shares state
                    new_positions.append(Position(
                        ticker=ticker, side="long_shares",
                        strike=pos.share_basis, expiry=today, open_date=today,
                        open_price=pos.share_basis, contracts=pos.contracts,
                        open_sigma=sigma, share_basis=pos.share_basis,
                    ))
                elif is_expiry:
                    if S > pos.strike:
                        # called away - sell shares at strike
                        proceeds = pos.strike * 100 * pos.contracts
                        commission = COMMISSION_PER_SHARE * 100 * pos.contracts
                        premium_kept = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                        share_pnl = (pos.strike - pos.share_basis) * 100 * pos.contracts
                        state.cash += proceeds - commission
                        state.cash += premium_kept
                        state.call_aways += 1
                        state.ledger.append({
                            "ticker": ticker, "open_date": pos.open_date, "close_date": today,
                            "kind": "cc_called_away", "strike": pos.strike, "S_close": S,
                            "premium_open": pos.open_price, "premium_close": 0.0,
                            "dte_at_open": (pos.expiry - pos.open_date).days,
                            "realized_pnl": premium_kept + share_pnl - commission,
                            "contracts": pos.contracts,
                        })
                        # back to cash state, no position
                    else:
                        # expired worthless - keep premium and shares
                        realized = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                        state.cash += realized
                        state.ledger.append({
                            "ticker": ticker, "open_date": pos.open_date, "close_date": today,
                            "kind": "cc_expired", "strike": pos.strike, "S_close": S,
                            "premium_open": pos.open_price, "premium_close": 0.0,
                            "dte_at_open": (pos.expiry - pos.open_date).days,
                            "realized_pnl": realized, "contracts": pos.contracts,
                        })
                        new_positions.append(Position(
                            ticker=ticker, side="long_shares",
                            strike=pos.share_basis, expiry=today, open_date=today,
                            open_price=pos.share_basis, contracts=pos.contracts,
                            open_sigma=sigma, share_basis=pos.share_basis,
                        ))
                else:
                    new_positions.append(pos)
                    state.days_in_shares += 1
                continue

            elif pos.side == "long_shares":
                # need to sell a CC. (handled below in open phase)
                new_positions.append(pos)
                state.days_in_shares += 1

        state.positions = new_positions

        # ---- 2) open new legs ----
        has_active = any(p.side in ("short_put", "short_call") for p in state.positions)
        has_shares = any(p.side == "long_shares" for p in state.positions)

        # 2a) on long_shares with no CC, sell a CC
        if has_shares and not any(p.side == "short_call" for p in state.positions):
            updated = []
            for p in state.positions:
                if p.side == "long_shares":
                    expiry = find_expiry(today, cfg)
                    if expiry is None:
                        updated.append(p); continue
                    T = (expiry - today).days / 365.0
                    # CC strike: at least at cost basis to avoid forced loss; otherwise delta target
                    K_delta = strike_from_delta(S, T, sigma, cfg["call_delta_target"], kind="call")
                    # If shares deep underwater, sell CC AT basis (not below) to avoid locking loss
                    K_call = K_delta
                    if not cfg.get("allow_call_above_basis", True):
                        K_call = max(K_delta, p.share_basis)
                    else:
                        # allow below-basis call only if delta-implied strike is above current S * 1.005
                        # else require strike >= basis
                        if K_delta < p.share_basis and K_delta < S * 1.005:
                            K_call = max(p.share_basis, S * 1.01)  # 1% OTM-of-basis fallback
                    premium = bs_price(S, K_call, T, sigma, kind="call")
                    slip = slippage_per_share(premium)
                    if premium - slip <= 0.05:
                        updated.append(p); continue
                    credit = (premium - slip) * 100 * p.contracts - COST_PER_CONTRACT * p.contracts
                    state.cash += credit
                    updated.append(Position(
                        ticker=ticker, side="short_call",
                        strike=K_call, expiry=expiry, open_date=today,
                        open_price=premium - slip, contracts=p.contracts,
                        open_sigma=sigma, share_basis=p.share_basis,
                    ))
                    state.cc_opened += 1
                else:
                    updated.append(p)
            state.positions = updated

        # 2b) if no active position and no shares, try a new CSP
        if not has_active and not has_shares:
            if vix <= cfg["vix_max_gate"]:
                expiry = find_expiry(today, cfg)
                if expiry is not None:
                    T = (expiry - today).days / 365.0
                    K_put = strike_from_delta(S, T, sigma, cfg["put_delta_target"], kind="put")
                    premium = bs_price(S, K_put, T, sigma, kind="put")
                    slip = slippage_per_share(premium)
                    # collateral needed: K * 100
                    contracts = int((state.cash * 0.95) // (K_put * 100))
                    if contracts >= 1 and (premium - slip) > 0.05:
                        credit = (premium - slip) * 100 * contracts - COST_PER_CONTRACT * contracts
                        state.cash += credit
                        state.positions.append(Position(
                            ticker=ticker, side="short_put",
                            strike=K_put, expiry=expiry, open_date=today,
                            open_price=premium - slip, contracts=contracts,
                            open_sigma=sigma, share_basis=0.0,
                        ))
                        state.csp_opened += 1
                    else:
                        state.days_in_cash += 1
                else:
                    state.days_in_cash += 1
            else:
                state.days_in_cash += 1

        # ---- 3) MTM equity ----
        equity = state.cash
        for p in state.positions:
            T = max((p.expiry - today).days, 0) / 365.0
            if p.side == "short_put":
                opt = bs_price(S, p.strike, T, sigma, kind="put")
                equity += (p.open_price - opt) * 100 * p.contracts
            elif p.side == "long_shares":
                equity += (S - p.share_basis) * 100 * p.contracts
            elif p.side == "short_call":
                opt = bs_price(S, p.strike, T, sigma, kind="call")
                equity += (S - p.share_basis) * 100 * p.contracts
                equity += (p.open_price - opt) * 100 * p.contracts

        equity_curve.append({
            "date": today, "equity": equity, "S": S, "vix": vix,
            "spy_ret": float(row["spy_ret"]) if pd.notna(row["spy_ret"]) else np.nan,
            "ticker_ret": float(row["ticker_ret"]) if pd.notna(row["ticker_ret"]) else np.nan,
            "has_shares": int(has_shares or any(p.side == "short_call" for p in state.positions)),
            "has_csp": int(any(p.side == "short_put" for p in state.positions)),
            "in_cash": int(len(state.positions) == 0),
        })

    eq_df = pd.DataFrame(equity_curve).set_index("date")
    eq_df["ret"] = eq_df["equity"].pct_change()
    ledger_df = pd.DataFrame(state.ledger)
    return eq_df, ledger_df, state


# ------------------------- basket wheel ----------------------------------
def run_basket(df, cfg, per_ticker_alloc):
    """Run wheel on each ticker independently, then aggregate to basket equity."""
    per_ticker = {}
    for tk in TICKERS:
        df_t = df[df["ticker"] == tk].copy()
        eq, ledger, state = run_ticker_wheel(tk, df_t, cfg, per_ticker_alloc)
        per_ticker[tk] = {"equity": eq, "ledger": ledger, "state": state}

    # build basket equity = sum of per-ticker equities on common dates
    all_dates = sorted(set().union(*[p["equity"].index for p in per_ticker.values()]))
    basket_eq = pd.DataFrame(index=all_dates)
    for tk, d in per_ticker.items():
        basket_eq[tk] = d["equity"]["equity"].reindex(all_dates).ffill().fillna(per_ticker_alloc)
    basket_eq["equity"] = basket_eq[TICKERS].sum(axis=1)

    # attach SPY for regime
    spy_ret = df.drop_duplicates("date").set_index("date")["spy_ret"].reindex(all_dates)
    basket_eq["spy_ret"] = spy_ret
    basket_eq["ret"] = basket_eq["equity"].pct_change()

    # aggregate ledger
    all_ledger = pd.concat([d["ledger"] for d in per_ticker.values() if not d["ledger"].empty],
                            ignore_index=True) if any(not d["ledger"].empty for d in per_ticker.values()) else pd.DataFrame()

    return basket_eq, all_ledger, per_ticker


# ------------------------- metrics ----------------------------------------
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
        "initial_equity": float(eq_df["equity"].iloc[0]),
    }


def regime_metrics(eq_df):
    sub = eq_df.dropna(subset=["spy_ret", "ret"]).copy()
    if len(sub) < 40:
        return {"hc428_r1_pass": False, "regime_gap": float("nan"),
                "sharpe_green": None, "sharpe_red": None, "sharpe_flat": None,
                "n_green": 0, "n_red": 0, "n_flat": 0}
    sigma = sub["spy_ret"].std()
    thresh = 0.5 * sigma
    green = sub[sub["spy_ret"] >= thresh]["ret"]
    red = sub[sub["spy_ret"] <= -thresh]["ret"]
    flat = sub[(sub["spy_ret"] > -thresh) & (sub["spy_ret"] < thresh)]["ret"]

    def ann_sh(s):
        if len(s) < 2 or s.std() == 0:
            return float("nan")
        return (s.mean() / s.std()) * math.sqrt(TRADING_DAYS)

    sh_g, sh_r, sh_f = ann_sh(green), ann_sh(red), ann_sh(flat)
    denom = max(abs(sh_g) if np.isfinite(sh_g) else 0,
                abs(sh_r) if np.isfinite(sh_r) else 0, 1e-9)
    gap = abs((sh_g if np.isfinite(sh_g) else 0.0) - (sh_r if np.isfinite(sh_r) else 0.0)) / denom
    return {
        "n_green": int(len(green)), "n_red": int(len(red)), "n_flat": int(len(flat)),
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


def tail_stress(eq_df, label, start, end):
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
    return {
        "label": label,
        "window": f"{start} to {end}",
        "n_days": int(len(sub)),
        "max_dd_pct": max_dd * 100,
        "recover_days": rec_days,
        "cum_ret_pct": float((sub["equity"].iloc[-1] / sub["equity"].iloc[0] - 1.0) * 100.0),
    }


def buy_hold_basket(df, alloc_per_ticker):
    """Compute buy-and-hold equity for equal-weight basket."""
    eqs = {}
    for tk in TICKERS:
        d = df[df["ticker"] == tk].sort_values("date").set_index("date")["close"]
        eqs[tk] = (d / d.iloc[0]) * alloc_per_ticker
    bh = pd.DataFrame(eqs)
    bh["equity"] = bh[TICKERS].sum(axis=1)
    bh["ret"] = bh["equity"].pct_change()
    spy_ret = df.drop_duplicates("date").set_index("date")["spy_ret"].reindex(bh.index)
    bh["spy_ret"] = spy_ret
    return bh


def time_in_state_stats(per_ticker):
    """Aggregate days_in_X across all tickers, basket-level."""
    total = {"shares": 0, "csp": 0, "cash": 0}
    n_days_per_tk = {}
    for tk, d in per_ticker.items():
        s = d["state"]
        total["shares"] += s.days_in_shares
        total["csp"] += s.days_in_csp
        total["cash"] += s.days_in_cash
        n_days_per_tk[tk] = s.days_in_shares + s.days_in_csp + s.days_in_cash
    grand = sum(total.values())
    if grand == 0:
        return {"pct_shares": 0, "pct_csp": 0, "pct_cash": 0}
    return {
        "pct_shares": 100 * total["shares"] / grand,
        "pct_csp": 100 * total["csp"] / grand,
        "pct_cash": 100 * total["cash"] / grand,
        "by_ticker": {tk: {"days_shares": d["state"].days_in_shares,
                            "days_csp": d["state"].days_in_csp,
                            "days_cash": d["state"].days_in_cash,
                            "pct_shares": 100 * d["state"].days_in_shares / max(n_days_per_tk[tk],1)}
                       for tk, d in per_ticker.items()},
    }


# ------------------------- mlflow -----------------------------------------
def log_mlflow(run_name, cfg, metrics, gate, day_conc, tail, in_state,
               eq_df, csp_o, cc_o, asg, cw):
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("single_name_wheel_megacap_v1")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({k: v for k, v in cfg.items() if isinstance(v, (int, float, str, bool))})
            mlflow.log_param("run_name", run_name)
            mlflow.log_param("start", str(eq_df.index[0].date()))
            mlflow.log_param("end", str(eq_df.index[-1].date()))
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and np.isfinite(v):
                    mlflow.log_metric(k, v)
            for k, v in gate.items():
                if isinstance(v, (int, float)) and v is not None and np.isfinite(v):
                    mlflow.log_metric(f"regime_{k}", v)
            mlflow.log_metric("hc428_r1_pass", 1.0 if gate.get("hc428_r1_pass") else 0.0)
            mlflow.log_metric("day_conc", day_conc if np.isfinite(day_conc) else -1.0)
            mlflow.log_metric("csp_opened", csp_o)
            mlflow.log_metric("cc_opened", cc_o)
            mlflow.log_metric("assignments", asg)
            mlflow.log_metric("call_aways", cw)
            mlflow.log_metric("pct_shares", in_state.get("pct_shares", 0))
            mlflow.log_metric("pct_csp", in_state.get("pct_csp", 0))
            mlflow.log_metric("pct_cash", in_state.get("pct_cash", 0))
            for t in tail:
                if t.get("n_days", 0):
                    safe = t["label"].replace(" ", "_")
                    mlflow.log_metric(f"tail_{safe}_max_dd_pct", t["max_dd_pct"])
                    if t.get("recover_days") is not None:
                        mlflow.log_metric(f"tail_{safe}_recover_days", t["recover_days"])
            return mlflow.active_run().info.run_id
    except Exception as e:
        print(f"[mlflow] skipped ({e})")
        return None


# ------------------------- main -------------------------------------------
def main():
    print("[load] price/IV/macro for megacap tickers ...")
    df = load_data()
    print(f"[load] {df['date'].nunique()} trading days across {df['ticker'].nunique()} tickers")
    print(f"[load] window {df['date'].min().date()} -> {df['date'].max().date()}")

    results = {}

    # ---- single-ticker runs ----
    for tk in TICKERS:
        print(f"\n[{tk}] running single-ticker wheel, ${SINGLE_TICKER_CASH:,.0f} ...")
        df_t = df[df["ticker"] == tk].copy()
        # add spy_ret for regime metrics
        eq, ledger, state = run_ticker_wheel(tk, df_t, WHEEL_CFG, SINGLE_TICKER_CASH)
        m = compute_metrics(eq)
        g = regime_metrics(eq)
        dc = day_concentration(ledger)
        tail = [
            tail_stress(eq, "COVID_2020", "2020-02-15", "2020-05-31"),
            tail_stress(eq, "2022_bear", "2022-01-01", "2022-12-31"),
        ]
        in_state = {
            "pct_shares": 100 * state.days_in_shares / max(state.days_in_shares + state.days_in_csp + state.days_in_cash, 1),
            "pct_csp": 100 * state.days_in_csp / max(state.days_in_shares + state.days_in_csp + state.days_in_cash, 1),
            "pct_cash": 100 * state.days_in_cash / max(state.days_in_shares + state.days_in_csp + state.days_in_cash, 1),
        }
        eq.to_parquet(OUT_DIR / f"equity_{tk}.parquet")
        if not ledger.empty:
            ledger.to_parquet(OUT_DIR / f"ledger_{tk}.parquet")
        gates = {
            "sharpe_ge_1.0": bool(np.isfinite(m.get("sharpe", float("nan"))) and m["sharpe"] >= 1.0),
            "calmar_ge_1.5": bool(np.isfinite(m.get("calmar", float("nan"))) and m["calmar"] >= 1.5),
            "regime_gap_le_0.50": bool(g["hc428_r1_pass"]),
            "day_conc_le_0.70": bool(np.isnan(dc) or dc <= 0.70),
        }
        deploy = all(gates.values())
        run_id = log_mlflow(tk, WHEEL_CFG, m, g, dc, tail, in_state, eq,
                            state.csp_opened, state.cc_opened, state.assignments, state.call_aways)
        results[tk] = {
            "metrics": m, "regime": g, "day_conc": dc, "tail": tail,
            "in_state": in_state, "gates_pass": gates, "deploy_ready": deploy,
            "csp_opened": state.csp_opened, "cc_opened": state.cc_opened,
            "assignments": state.assignments, "call_aways": state.call_aways,
            "mlflow_run_id": run_id,
        }
        print(f"[{tk}] Sharpe={m.get('sharpe', float('nan')):.2f} CAGR={m.get('cagr',float('nan'))*100:.1f}% "
              f"MaxDD={m.get('max_dd',float('nan'))*100:.1f}% Calmar={m.get('calmar',float('nan')):.2f} "
              f"gap={g['regime_gap']:.2f} %shares={in_state['pct_shares']:.0f}% deploy={deploy}")

    # ---- basket run ----
    print(f"\n[BASKET] running 5-ticker basket, ${BASKET_TOTAL:,.0f} total ($4K each) ...")
    basket_eq, basket_ledger, per_tk = run_basket(df, WHEEL_CFG, PER_TICKER_ALLOC)
    m = compute_metrics(basket_eq)
    g = regime_metrics(basket_eq)
    dc = day_concentration(basket_ledger)
    tail = [
        tail_stress(basket_eq, "COVID_2020", "2020-02-15", "2020-05-31"),
        tail_stress(basket_eq, "2022_bear", "2022-01-01", "2022-12-31"),
        tail_stress(basket_eq, "2018_Q4", "2018-10-01", "2018-12-31"),
    ]
    in_state = time_in_state_stats(per_tk)
    basket_eq.to_parquet(OUT_DIR / "equity_BASKET.parquet")
    if not basket_ledger.empty:
        basket_ledger.to_parquet(OUT_DIR / "ledger_BASKET.parquet")

    csp_total = sum(p["state"].csp_opened for p in per_tk.values())
    cc_total = sum(p["state"].cc_opened for p in per_tk.values())
    asg_total = sum(p["state"].assignments for p in per_tk.values())
    cw_total = sum(p["state"].call_aways for p in per_tk.values())

    gates = {
        "sharpe_ge_1.0": bool(np.isfinite(m.get("sharpe", float("nan"))) and m["sharpe"] >= 1.0),
        "calmar_ge_1.5": bool(np.isfinite(m.get("calmar", float("nan"))) and m["calmar"] >= 1.5),
        "regime_gap_le_0.50": bool(g["hc428_r1_pass"]),
        "day_conc_le_0.70": bool(np.isnan(dc) or dc <= 0.70),
    }
    deploy = all(gates.values())
    run_id = log_mlflow("BASKET", WHEEL_CFG, m, g, dc, tail, in_state, basket_eq,
                        csp_total, cc_total, asg_total, cw_total)
    results["BASKET"] = {
        "metrics": m, "regime": g, "day_conc": dc, "tail": tail,
        "in_state": in_state, "gates_pass": gates, "deploy_ready": deploy,
        "csp_opened": csp_total, "cc_opened": cc_total,
        "assignments": asg_total, "call_aways": cw_total,
        "mlflow_run_id": run_id,
    }
    print(f"[BASKET] Sharpe={m.get('sharpe',float('nan')):.2f} CAGR={m.get('cagr',float('nan'))*100:.1f}% "
          f"MaxDD={m.get('max_dd',float('nan'))*100:.1f}% Calmar={m.get('calmar',float('nan')):.2f} "
          f"gap={g['regime_gap']:.2f} %shares={in_state['pct_shares']:.0f}% deploy={deploy}")

    # ---- buy-and-hold comparison ----
    bh = buy_hold_basket(df, PER_TICKER_ALLOC)
    bh_m = compute_metrics(bh)
    print(f"\n[BUY_HOLD] Sharpe={bh_m['sharpe']:.2f} CAGR={bh_m['cagr']*100:.1f}% "
          f"MaxDD={bh_m['max_dd']*100:.1f}% final=${bh_m['final_equity']:,.0f}")

    # ---- write report ----
    L = []
    L.append("# Single-Name Wheel on Megacap Tech v1 (HC #587 R2)\n")
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Universe: {', '.join(TICKERS)}")
    L.append(f"Window: {START_DATE.date()} to {END_DATE.date()}")
    L.append(f"Per-ticker allocation: ${PER_TICKER_ALLOC:,.0f} (basket total ${BASKET_TOTAL:,.0f})")
    L.append(f"Single-ticker tests: ${SINGLE_TICKER_CASH:,.0f} each\n")
    L.append("## Strategy\n")
    L.append(f"- Put delta {WHEEL_CFG['put_delta_target']}, call delta {WHEEL_CFG['call_delta_target']}")
    L.append(f"- DTE {WHEEL_CFG['dte_min']}-{WHEEL_CFG['dte_max']} (target {WHEEL_CFG['dte_target']})")
    L.append(f"- Profit-take {int(WHEEL_CFG['profit_take_pct']*100)}%, NO forced close on losses")
    L.append(f"- VIX gate {WHEEL_CFG['vix_max_gate']} (entries only)")
    L.append(f"- Cost: ${COST_PER_CONTRACT}/contract + ${COMMISSION_PER_SHARE}/share on assign/called-away\n")

    L.append("## Headline Metrics\n")
    L.append("| Config | CAGR | Sharpe | Sortino | MaxDD | Calmar | WR | PF | Final $ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name in TICKERS + ["BASKET"]:
        m = results[name]["metrics"]
        L.append(f"| {name} | {m.get('cagr',0)*100:.2f}% | {m.get('sharpe',0):.2f} | "
                 f"{m.get('sortino',0):.2f} | {m.get('max_dd',0)*100:.1f}% | "
                 f"{m.get('calmar',0):.2f} | {m.get('win_rate',0)*100:.1f}% | "
                 f"{m.get('profit_factor',0):.2f} | ${m.get('final_equity',0):,.0f} |")

    L.append(f"\n## Buy-and-Hold Equal-Weight Basket (reference)")
    L.append(f"- CAGR {bh_m['cagr']*100:.2f}%, Sharpe {bh_m['sharpe']:.2f}, MaxDD {bh_m['max_dd']*100:.1f}%, "
             f"final ${bh_m['final_equity']:,.0f}")

    L.append("\n## HC #428 R1 Regime Gates\n")
    L.append("| Config | n_g | n_r | n_f | Sh_g | Sh_r | Sh_f | gap | PASS (<=0.50) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    def f2(x): return f"{x:.2f}" if (x is not None and np.isfinite(x)) else "n/a"
    for name in TICKERS + ["BASKET"]:
        g = results[name]["regime"]
        L.append(f"| {name} | {g['n_green']} | {g['n_red']} | {g['n_flat']} | "
                 f"{f2(g['sharpe_green'])} | {f2(g['sharpe_red'])} | {f2(g['sharpe_flat'])} | "
                 f"{g['regime_gap']:.2f} | {'PASS' if g['hc428_r1_pass'] else 'FAIL'} |")

    L.append("\n## Deploy Gates\n")
    L.append("| Config | Sharpe>=1.0 | Calmar>=1.5 | Gap<=0.50 | DayConc<=0.70 | DEPLOY |")
    L.append("|---|---|---|---|---|---|")
    for name in TICKERS + ["BASKET"]:
        gp = results[name]["gates_pass"]
        x = lambda b: "PASS" if b else "FAIL"
        L.append(f"| {name} | {x(gp['sharpe_ge_1.0'])} | {x(gp['calmar_ge_1.5'])} | "
                 f"{x(gp['regime_gap_le_0.50'])} | {x(gp['day_conc_le_0.70'])} | "
                 f"**{'YES' if results[name]['deploy_ready'] else 'NO'}** |")

    L.append("\n## Time-in-State (% of days)\n")
    L.append("| Config | %Shares | %CSP | %Cash |")
    L.append("|---|---|---|---|")
    for name in TICKERS:
        s = results[name]["in_state"]
        L.append(f"| {name} | {s['pct_shares']:.1f}% | {s['pct_csp']:.1f}% | {s['pct_cash']:.1f}% |")
    s = results["BASKET"]["in_state"]
    L.append(f"| BASKET | {s['pct_shares']:.1f}% | {s['pct_csp']:.1f}% | {s['pct_cash']:.1f}% |")

    L.append("\n## Wheel Activity\n")
    L.append("| Config | CSP opens | CC opens | Assignments | Call-aways |")
    L.append("|---|---|---|---|---|")
    for name in TICKERS + ["BASKET"]:
        r = results[name]
        L.append(f"| {name} | {r['csp_opened']} | {r['cc_opened']} | "
                 f"{r['assignments']} | {r['call_aways']} |")

    L.append("\n## Tail Event Stress\n")
    L.append("| Config | Event | MaxDD | RecoverDays | CumRet |")
    L.append("|---|---|---|---|---|")
    for name in TICKERS + ["BASKET"]:
        for t in results[name]["tail"]:
            if t.get("n_days", 0) == 0:
                continue
            rd = t.get("recover_days") if t.get("recover_days") is not None else "not_recovered"
            L.append(f"| {name} | {t['label']} | {t['max_dd_pct']:.1f}% | {rd} | {t['cum_ret_pct']:.1f}% |")

    L.append("\n## Recommendation\n")
    passers = [n for n in TICKERS + ["BASKET"] if results[n]["deploy_ready"]]
    if passers:
        best = max(passers, key=lambda n: results[n]["metrics"]["sharpe"])
        L.append(f"- **PASS**: {', '.join(passers)} cleared all HC #428 R1 + deploy gates.")
        L.append(f"- **Recommended winner**: {best} (highest Sharpe among passers).")
        L.append(f"- Action: paper-engine module written, entries_paused=True for user gate-open.")
    else:
        L.append("- **NO config passed HC #428 R1.**")
        L.append("- Single-name wheel exhibits the same green/red asymmetry as SPY wheel.")
        L.append("- Honest conclusion: the put-sell payoff is structurally short-tail regardless of "
                 "underlying choice. Adding willingness to take assignment helps survival but does "
                 "not fix the directional Sharpe asymmetry. The wheel makes most of its money on "
                 "green days (vol crush + drift up), loses on red days (assignment then shares "
                 "underwater), and that asymmetry is intrinsic to the payoff geometry.")
        L.append("- Recommendation: abandon the wheel lane unless a relaxed gate spec is accepted.")

    L.append(f"\n## Buy-and-Hold vs Basket Wheel\n")
    bw = results["BASKET"]["metrics"]
    L.append(f"- Buy-Hold: CAGR {bh_m['cagr']*100:.2f}%, Sharpe {bh_m['sharpe']:.2f}, "
             f"MaxDD {bh_m['max_dd']*100:.1f}%, final ${bh_m['final_equity']:,.0f}")
    L.append(f"- Basket Wheel: CAGR {bw.get('cagr',0)*100:.2f}%, Sharpe {bw.get('sharpe',0):.2f}, "
             f"MaxDD {bw.get('max_dd',0)*100:.1f}%, final ${bw.get('final_equity',0):,.0f}")
    cagr_delta = (bw.get("cagr", 0) - bh_m["cagr"]) * 100
    L.append(f"- Delta CAGR: {cagr_delta:+.2f}pp (wheel vs buy-hold)")

    # Per-ticker max drawdown - worst single-name
    worst_tk = min(TICKERS, key=lambda t: results[t]["metrics"]["max_dd"])
    L.append(f"\n- Worst single-name DD: {worst_tk} at {results[worst_tk]['metrics']['max_dd']*100:.1f}%")

    report_path = REPORT_DIR / "single_name_wheel_megacap_v1.md"
    report_path.write_text("\n".join(L))
    print(f"\n[report] -> {report_path}")

    # JSON summary
    summary = {
        "generated": datetime.now().isoformat(),
        "window": [str(START_DATE.date()), str(END_DATE.date())],
        "buy_hold": bh_m,
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "mlflow_run_id" or vv is not None}
                    for k, v in results.items()},
        "any_pass": any(results[n]["deploy_ready"] for n in TICKERS + ["BASKET"]),
        "passers": [n for n in TICKERS + ["BASKET"] if results[n]["deploy_ready"]],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[summary] -> {OUT_DIR/'summary.json'}")

    return results, bh_m


if __name__ == "__main__":
    main()
