#!/usr/bin/env python3
"""
iron_condor_real_chains.py — Iron Condor backtest using REAL Dolt option chain data.

PURE REAL PRICING: Both entry AND exit use actual bid/ask from DoltHub.
No Black-Scholes anywhere in the P&L calculation.

DESIGN:
  The Dolt data has observation dates where real chains exist. Each observation
  date has 2-3 expirations with real bid/ask for ~20-30 strikes.

  Many expirations appear on 2+ observation dates, which means we can:
    - OPEN on observation date 1 (sell at bid, buy at ask)
    - CLOSE on observation date 2+ (buy back at ask, sell at bid)
  This gives us a pure real-chain round trip.

  For expirations that appear on only 1 observation date, we skip them
  (can't get honest close pricing).

Strategy:
  - SELL short put at ~20-delta
  - BUY long put at ~10-delta (wing)
  - SELL short call at ~20-delta
  - BUY long call at ~10-delta (wing)
  - Target DTE ~30 at open
  - Close on next chain observation that has the same expiry
  - Profit take at 50% of max profit
  - Close at DTE <= 7 if chain data available
  - At expiry: settle at intrinsic (exact — this is what happens in reality)
  - VIX gate: don't open if VIX > 28
  - Commission: $0 (Robinhood, HC #694)
  - Sizing: FIXED 1 contract per trade (no compounding distortion)
    Also runs at 50% equity for compound comparison
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
CHAINS_FILE = CACHE / "options_real" / "chains" / "SPY.parquet"
OUT_DIR = ROOT / "output" / "ic_real_chain_backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CASH = 20_000.0
TRADING_DAYS = 252

VIX_GATE_OPEN = 28.0
DTE_OPEN_MIN = 20
DTE_OPEN_MAX = 50
DTE_OPEN_TARGET = 30
DTE_CLOSE_TRIGGER = 7
PROFIT_TAKE_FRAC = 0.50

SHORT_DELTA = 0.20
LONG_DELTA = 0.10
DELTA_TOLERANCE = 0.10


def load_spy_daily() -> pd.DataFrame:
    sec = pd.read_parquet(CACHE / "sector_etfs.parquet")
    spy = sec[sec["ticker"] == "SPY"][["date", "close"]].copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")
    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"])
    macro = macro.sort_values("date").set_index("date")
    df = spy.join(macro, how="left")
    df["vix"] = df["vix"].ffill()
    df["spy_ret"] = df["close"].pct_change()
    return df


def load_chains() -> pd.DataFrame:
    if not CHAINS_FILE.exists():
        raise SystemExit(f"SPY chain file missing: {CHAINS_FILE}")
    df = pd.read_parquet(CHAINS_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    for c in ["strike", "bid", "ask", "mid", "delta", "gamma"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["bid"].notna() & df["ask"].notna()].copy()
    return df


def find_best_strike(chain_slice: pd.DataFrame, target_delta: float,
                     option_type: str) -> Optional[pd.Series]:
    sub = chain_slice[chain_slice["type"] == option_type].copy()
    if sub.empty:
        return None
    sub["delta_abs"] = sub["delta"].abs()
    sub["delta_diff"] = (sub["delta_abs"] - target_delta).abs()
    valid = sub[sub["delta_diff"] <= DELTA_TOLERANCE]
    if valid.empty:
        return None
    return valid.loc[valid["delta_diff"].idxmin()]


def find_strike_by_value(chain_slice: pd.DataFrame, strike: float,
                         option_type: str) -> Optional[pd.Series]:
    sub = chain_slice[chain_slice["type"] == option_type].copy()
    if sub.empty:
        return None
    exact = sub[sub["strike"] == strike]
    if not exact.empty:
        return exact.iloc[0]
    sub["strike_diff"] = (sub["strike"] - strike).abs()
    nearest = sub[sub["strike_diff"] <= 3.0]
    if nearest.empty:
        return None
    return nearest.loc[nearest["strike_diff"].idxmin()]


@dataclass
class ICTrade:
    """Complete round-trip iron condor trade with real pricing."""
    open_date: pd.Timestamp
    close_date: pd.Timestamp
    expiry: pd.Timestamp
    short_put_K: float
    long_put_K: float
    short_call_K: float
    long_call_K: float
    # Entry (per share)
    open_sp_bid: float
    open_lp_ask: float
    open_sc_bid: float
    open_lc_ask: float
    open_net_credit: float
    # Exit (per share)
    close_sp_ask: float
    close_lp_bid: float
    close_sc_ask: float
    close_lc_bid: float
    close_cost: float
    # PnL (per share, 1 contract)
    pnl_per_share: float
    pnl_per_contract: float  # = pnl_per_share * 100
    # Metadata
    close_reason: str
    entry_deltas: Dict
    close_pricing: str  # "real_chain" or "intrinsic"

    @property
    def put_width(self):
        return self.short_put_K - self.long_put_K

    @property
    def call_width(self):
        return self.long_call_K - self.short_call_K

    @property
    def max_risk_per_share(self):
        return max(self.put_width, self.call_width) - self.open_net_credit

    @property
    def return_on_risk(self):
        if self.max_risk_per_share > 0:
            return self.pnl_per_share / self.max_risk_per_share
        return 0.0


def try_build_ic(exp_chain: pd.DataFrame, S: float) -> Optional[Dict]:
    """Try to build IC legs from chain. Returns leg info or None."""
    sp = find_best_strike(exp_chain, SHORT_DELTA, "p")
    lp = find_best_strike(exp_chain, LONG_DELTA, "p")
    sc = find_best_strike(exp_chain, SHORT_DELTA, "c")
    lc = find_best_strike(exp_chain, LONG_DELTA, "c")
    if any(x is None for x in [sp, lp, sc, lc]):
        return None
    if lp["strike"] >= sp["strike"] or lc["strike"] <= sc["strike"]:
        return None
    if sp["strike"] >= S or sc["strike"] <= S:
        return None
    sp_bid, sc_bid = float(sp["bid"]), float(sc["bid"])
    lp_ask, lc_ask = float(lp["ask"]), float(lc["ask"])
    if sp_bid <= 0 or sc_bid <= 0:
        return None
    net_credit = (sp_bid + sc_bid) - (lp_ask + lc_ask)
    if net_credit <= 0.05:
        return None
    pw = sp["strike"] - lp["strike"]
    cw = lc["strike"] - sc["strike"]
    max_loss = max(pw, cw) - net_credit
    if max_loss <= 0:
        return None
    return {
        "sp_K": float(sp["strike"]), "lp_K": float(lp["strike"]),
        "sc_K": float(sc["strike"]), "lc_K": float(lc["strike"]),
        "sp_bid": sp_bid, "lp_ask": lp_ask,
        "sc_bid": sc_bid, "lc_ask": lc_ask,
        "net_credit": net_credit,
        "sp_delta": float(sp["delta"]), "lp_delta": float(lp["delta"]),
        "sc_delta": float(sc["delta"]), "lc_delta": float(lc["delta"]),
    }


def try_close_real(exp_chain: pd.DataFrame, legs: Dict) -> Optional[Dict]:
    """Try to close IC using real chain. Returns close info or None."""
    sp = find_strike_by_value(exp_chain, legs["sp_K"], "p")
    lp = find_strike_by_value(exp_chain, legs["lp_K"], "p")
    sc = find_strike_by_value(exp_chain, legs["sc_K"], "c")
    lc = find_strike_by_value(exp_chain, legs["lc_K"], "c")
    if any(x is None for x in [sp, lp, sc, lc]):
        return None
    sp_ask = float(sp["ask"])
    lp_bid = float(lp["bid"])
    sc_ask = float(sc["ask"])
    lc_bid = float(lc["bid"])
    cost = (sp_ask + sc_ask) - (lp_bid + lc_bid)
    return {
        "sp_ask": sp_ask, "lp_bid": lp_bid,
        "sc_ask": sc_ask, "lc_bid": lc_bid,
        "cost": cost,
    }


def close_intrinsic(legs: Dict, S: float) -> Dict:
    """Close at intrinsic (expiry settlement)."""
    sp_val = max(legs["sp_K"] - S, 0)
    lp_val = max(legs["lp_K"] - S, 0)
    sc_val = max(S - legs["sc_K"], 0)
    lc_val = max(S - legs["lc_K"], 0)
    cost = (sp_val + sc_val) - (lp_val + lc_val)
    return {
        "sp_ask": sp_val, "lp_bid": lp_val,
        "sc_ask": sc_val, "lc_bid": lc_val,
        "cost": cost,
    }


def run_pure_real_backtest():
    """Run backtest where ALL entries use real chain data.
    For exits, use real chain when available, intrinsic at expiry.
    Never use BS."""

    print("[load] SPY daily prices + VIX ...")
    daily = load_spy_daily()

    print("[load] SPY option chains (Dolt real data) ...")
    chains = load_chains()

    # Index: (obs_date, expiry) -> chain
    chains_idx = {}
    for (d, exp), grp in chains.groupby(["date", "expiration"]):
        chains_idx[(d, exp)] = grp

    # expiry -> sorted list of observation dates
    exp_obs = defaultdict(list)
    for (d, exp) in chains_idx.keys():
        exp_obs[exp].append(d)
    for exp in exp_obs:
        exp_obs[exp] = sorted(exp_obs[exp])

    # All observation dates sorted
    obs_dates = sorted(set(d for d, _ in chains_idx.keys()))
    print(f"[load] {len(obs_dates)} chain observation dates, {len(exp_obs)} unique expirations")

    # Get VIX on each date
    vix_lookup = daily["vix"].to_dict()

    trades: List[ICTrade] = []
    pos = None  # current position: dict with legs + open info
    skipped = 0

    for obs_date in obs_dates:
        S_row = daily.loc[:obs_date]
        if S_row.empty:
            continue
        S = float(S_row.iloc[-1]["close"])
        vix = vix_lookup.get(obs_date, vix_lookup.get(S_row.index[-1], 20.0))
        if pd.isna(vix):
            vix = 20.0

        # ── Manage existing position ──
        if pos is not None:
            dte = (pos["expiry"] - obs_date).days

            # Try real close
            exp_chain = chains_idx.get((obs_date, pos["expiry"]))

            should_close = False
            close_reason = ""

            if dte <= 0:
                # Expired — intrinsic settlement
                close_info = close_intrinsic(pos["legs"], S)
                should_close = True
                close_reason = "expiry"
                pricing = "intrinsic"
            elif exp_chain is not None:
                close_info = try_close_real(exp_chain, pos["legs"])
                if close_info is not None:
                    profit_frac = (pos["legs"]["net_credit"] - close_info["cost"]) / pos["legs"]["net_credit"]
                    if profit_frac >= PROFIT_TAKE_FRAC:
                        should_close = True
                        close_reason = "profit_take"
                    elif dte <= DTE_CLOSE_TRIGGER:
                        should_close = True
                        close_reason = "dte_close"
                    pricing = "real_chain"
                else:
                    close_info = None
            else:
                close_info = None

            # If DTE triggered but no real chain for this expiry, check if this is the
            # last observation before expiry — if so, close at whatever we can get
            if not should_close and dte <= DTE_CLOSE_TRIGGER and close_info is None:
                # Check if there's another obs date before expiry with this exp
                future_obs = [d for d in exp_obs.get(pos["expiry"], []) if d > obs_date]
                if not future_obs:
                    # No more chain data — will expire; handle on expiry date
                    pass

            if should_close and close_info is not None:
                pnl_ps = pos["legs"]["net_credit"] - close_info["cost"]
                trade = ICTrade(
                    open_date=pos["open_date"], close_date=obs_date,
                    expiry=pos["expiry"],
                    short_put_K=pos["legs"]["sp_K"], long_put_K=pos["legs"]["lp_K"],
                    short_call_K=pos["legs"]["sc_K"], long_call_K=pos["legs"]["lc_K"],
                    open_sp_bid=pos["legs"]["sp_bid"], open_lp_ask=pos["legs"]["lp_ask"],
                    open_sc_bid=pos["legs"]["sc_bid"], open_lc_ask=pos["legs"]["lc_ask"],
                    open_net_credit=pos["legs"]["net_credit"],
                    close_sp_ask=close_info["sp_ask"], close_lp_bid=close_info["lp_bid"],
                    close_sc_ask=close_info["sc_ask"], close_lc_bid=close_info["lc_bid"],
                    close_cost=close_info["cost"],
                    pnl_per_share=pnl_ps, pnl_per_contract=pnl_ps * 100,
                    close_reason=close_reason,
                    entry_deltas={
                        "sp": pos["legs"]["sp_delta"], "lp": pos["legs"]["lp_delta"],
                        "sc": pos["legs"]["sc_delta"], "lc": pos["legs"]["lc_delta"],
                    },
                    close_pricing=pricing,
                )
                trades.append(trade)
                pos = None

        # Handle expiry on non-obs dates (check daily)
        if pos is not None:
            dte = (pos["expiry"] - obs_date).days
            if dte <= 0:
                close_info = close_intrinsic(pos["legs"], S)
                pnl_ps = pos["legs"]["net_credit"] - close_info["cost"]
                trade = ICTrade(
                    open_date=pos["open_date"], close_date=obs_date,
                    expiry=pos["expiry"],
                    short_put_K=pos["legs"]["sp_K"], long_put_K=pos["legs"]["lp_K"],
                    short_call_K=pos["legs"]["sc_K"], long_call_K=pos["legs"]["lc_K"],
                    open_sp_bid=pos["legs"]["sp_bid"], open_lp_ask=pos["legs"]["lp_ask"],
                    open_sc_bid=pos["legs"]["sc_bid"], open_lc_ask=pos["legs"]["lc_ask"],
                    open_net_credit=pos["legs"]["net_credit"],
                    close_sp_ask=close_info["sp_ask"], close_lp_bid=close_info["lp_bid"],
                    close_sc_ask=close_info["sc_ask"], close_lc_bid=close_info["lc_bid"],
                    close_cost=close_info["cost"],
                    pnl_per_share=pnl_ps, pnl_per_contract=pnl_ps * 100,
                    close_reason="expiry",
                    entry_deltas={
                        "sp": pos["legs"]["sp_delta"], "lp": pos["legs"]["lp_delta"],
                        "sc": pos["legs"]["sc_delta"], "lc": pos["legs"]["lc_delta"],
                    },
                    close_pricing="intrinsic",
                )
                trades.append(trade)
                pos = None

        # ── Open new position ──
        if pos is None and vix <= VIX_GATE_OPEN:
            # Find available expirations on this obs date
            available_exps = [exp for (d, exp) in chains_idx if d == obs_date]
            best_legs = None
            best_exp = None
            best_dte_diff = 999

            for exp in available_exps:
                dte = (exp - obs_date).days
                if DTE_OPEN_MIN <= dte <= DTE_OPEN_MAX:
                    # Check if this expiry has a future observation date for close
                    future_obs = [d for d in exp_obs[exp] if d > obs_date]
                    if not future_obs:
                        continue  # can't close with real data — skip
                    diff = abs(dte - DTE_OPEN_TARGET)
                    if diff < best_dte_diff:
                        exp_chain = chains_idx[(obs_date, exp)]
                        legs = try_build_ic(exp_chain, S)
                        if legs is not None:
                            best_legs = legs
                            best_exp = exp
                            best_dte_diff = diff

            if best_legs is not None:
                pos = {
                    "open_date": obs_date,
                    "expiry": best_exp,
                    "legs": best_legs,
                }
            else:
                skipped += 1

    # Force close any remaining position at intrinsic
    if pos is not None:
        S = float(daily.iloc[-1]["close"])
        close_info = close_intrinsic(pos["legs"], S)
        pnl_ps = pos["legs"]["net_credit"] - close_info["cost"]
        trade = ICTrade(
            open_date=pos["open_date"], close_date=daily.index[-1],
            expiry=pos["expiry"],
            short_put_K=pos["legs"]["sp_K"], long_put_K=pos["legs"]["lp_K"],
            short_call_K=pos["legs"]["sc_K"], long_call_K=pos["legs"]["lc_K"],
            open_sp_bid=pos["legs"]["sp_bid"], open_lp_ask=pos["legs"]["lp_ask"],
            open_sc_bid=pos["legs"]["sc_bid"], open_lc_ask=pos["legs"]["lc_ask"],
            open_net_credit=pos["legs"]["net_credit"],
            close_sp_ask=close_info["sp_ask"], close_lp_bid=close_info["lp_bid"],
            close_sc_ask=close_info["sc_ask"], close_lc_bid=close_info["lc_bid"],
            close_cost=close_info["cost"],
            pnl_per_share=pnl_ps, pnl_per_contract=pnl_ps * 100,
            close_reason="end_of_data",
            entry_deltas={
                "sp": pos["legs"]["sp_delta"], "lp": pos["legs"]["lp_delta"],
                "sc": pos["legs"]["sc_delta"], "lc": pos["legs"]["lc_delta"],
            },
            close_pricing="intrinsic",
        )
        trades.append(trade)

    return trades, daily, skipped


def build_equity_curve(trades: List[ICTrade], daily: pd.DataFrame,
                       fixed_contracts: int = 1) -> pd.DataFrame:
    """Build equity curve with fixed position sizing (no compounding distortion)."""
    # Create daily PnL series from trades
    trade_pnl = {}
    for t in trades:
        d = t.close_date
        pnl = t.pnl_per_contract * fixed_contracts
        trade_pnl[d] = trade_pnl.get(d, 0) + pnl

    # Build curve
    start = trades[0].open_date if trades else daily.index[0]
    end = trades[-1].close_date if trades else daily.index[-1]
    sub = daily.loc[start:end].copy()

    equity = STARTING_CASH
    curve = []
    for d in sub.index:
        if d in trade_pnl:
            equity += trade_pnl[d]
        curve.append({
            "date": d, "equity": equity,
            "S": float(sub.loc[d, "close"]),
            "spy_ret": float(sub.loc[d, "spy_ret"]) if pd.notna(sub.loc[d, "spy_ret"]) else np.nan,
        })

    eq_df = pd.DataFrame(curve).set_index("date")
    eq_df["ret"] = eq_df["equity"].pct_change()
    return eq_df


def build_equity_curve_compound(trades: List[ICTrade], daily: pd.DataFrame,
                                alloc_frac: float = 0.50) -> pd.DataFrame:
    """Build equity curve with compounding: alloc_frac of equity as risk budget."""
    start = trades[0].open_date if trades else daily.index[0]
    end = trades[-1].close_date if trades else daily.index[-1]
    sub = daily.loc[start:end].copy()

    # Map trades by close date
    trade_by_close = defaultdict(list)
    for t in trades:
        trade_by_close[t.close_date].append(t)

    equity = STARTING_CASH
    curve = []
    for d in sub.index:
        for t in trade_by_close.get(d, []):
            # How many contracts would we have traded at open?
            risk_per = t.max_risk_per_share * 100
            if risk_per > 0:
                contracts = max(1, int((equity * alloc_frac) / risk_per))
            else:
                contracts = 1
            pnl = t.pnl_per_contract * contracts
            equity += pnl

        curve.append({
            "date": d, "equity": equity,
            "S": float(sub.loc[d, "close"]),
            "spy_ret": float(sub.loc[d, "spy_ret"]) if pd.notna(sub.loc[d, "spy_ret"]) else np.nan,
        })

    eq_df = pd.DataFrame(curve).set_index("date")
    eq_df["ret"] = eq_df["equity"].pct_change()
    return eq_df


def compute_metrics(eq_df):
    rets = eq_df["ret"].dropna()
    if len(rets) < 2 or eq_df["equity"].iloc[0] <= 0:
        return {}
    years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
    final_eq = max(eq_df["equity"].iloc[-1], 1e-6)
    cagr = (final_eq / eq_df["equity"].iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    mu, sd = rets.mean(), rets.std()
    downside = rets[rets < 0].std()
    sharpe = (mu / sd) * math.sqrt(TRADING_DAYS) if sd > 0 else float("nan")
    sortino = (mu / downside) * math.sqrt(TRADING_DAYS) if downside and downside > 0 else float("nan")
    peak = eq_df["equity"].cummax()
    dd = eq_df["equity"] / peak - 1.0
    max_dd = float(dd.min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else float("nan")
    wr_daily = float((rets > 0).mean())
    pos_sum = rets[rets > 0].sum()
    neg_sum = abs(rets[rets < 0].sum())
    pf = float(pos_sum / neg_sum) if neg_sum > 0 else float("nan")
    return {
        "n_days": int(len(rets)),
        "years": float(years),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
        "win_rate_daily": wr_daily,
        "profit_factor": pf,
        "total_return_pct": float((final_eq / eq_df["equity"].iloc[0] - 1.0) * 100.0),
        "final_equity": float(final_eq),
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
        "n_green": int(len(green)), "n_red": int(len(red)), "n_flat": int(len(flat)),
        "sharpe_green": float(sh_g) if np.isfinite(sh_g) else None,
        "sharpe_red": float(sh_r) if np.isfinite(sh_r) else None,
        "sharpe_flat": float(sh_f) if np.isfinite(sh_f) else None,
        "regime_gap": float(gap),
        "hc428_r1_pass": bool(gap <= 0.50),
    }


def main():
    print("=" * 70)
    print("IRON CONDOR BACKTEST — PURE REAL OPTION CHAIN DATA")
    print("  Entry: REAL bid/ask from Dolt")
    print("  Exit: REAL bid/ask from Dolt (or intrinsic at expiry)")
    print("  NO Black-Scholes in P&L — zero synthetic pricing")
    print("  Commission: $0 (Robinhood)")
    print("=" * 70)
    print()

    trades, daily, skipped = run_pure_real_backtest()

    if not trades:
        print("NO TRADES GENERATED — check data availability")
        return

    # ── Trade-level statistics ──
    n_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl_per_share > 0)
    losses = sum(1 for t in trades if t.pnl_per_share <= 0)
    total_pnl_1lot = sum(t.pnl_per_contract for t in trades)
    avg_pnl = np.mean([t.pnl_per_contract for t in trades])
    avg_win = np.mean([t.pnl_per_contract for t in trades if t.pnl_per_share > 0]) if wins > 0 else 0
    avg_loss = np.mean([t.pnl_per_contract for t in trades if t.pnl_per_share <= 0]) if losses > 0 else 0
    avg_credit = np.mean([t.open_net_credit for t in trades])
    avg_hold = np.mean([(t.close_date - t.open_date).days for t in trades])
    real_closes = sum(1 for t in trades if t.close_pricing == "real_chain")
    intrinsic_closes = sum(1 for t in trades if t.close_pricing == "intrinsic")

    print("=" * 70)
    print("TRADE-LEVEL RESULTS (1 contract per trade, no compounding)")
    print("=" * 70)
    print(f"Period:          {trades[0].open_date.date()} -> {trades[-1].close_date.date()}")
    print(f"Total trades:    {n_trades}")
    print(f"Win/Loss:        {wins}W / {losses}L ({wins/n_trades*100:.0f}% WR)")
    print(f"Total PnL (1-lot): ${total_pnl_1lot:+,.2f}")
    print(f"Avg PnL/trade:   ${avg_pnl:+,.2f}")
    print(f"Avg winner:      ${avg_win:+,.2f}")
    print(f"Avg loser:       ${avg_loss:+,.2f}")
    print(f"Avg credit recv: ${avg_credit:.2f}/share")
    print(f"Avg hold period: {avg_hold:.1f} days")
    print(f"Close pricing:   {real_closes} real chain, {intrinsic_closes} intrinsic")
    print(f"Skipped (no IC):  {skipped}")

    # ── Per-trade return on risk ──
    rors = [t.return_on_risk for t in trades]
    avg_ror = np.mean(rors)
    print(f"Avg return on risk: {avg_ror*100:+.2f}%")

    # ── Fixed 1-contract equity curve ──
    eq_fixed = build_equity_curve(trades, daily, fixed_contracts=1)
    m_fixed = compute_metrics(eq_fixed)

    print()
    print("── FIXED 1-CONTRACT EQUITY CURVE ──")
    print(f"Starting:  ${STARTING_CASH:,.0f}")
    print(f"Final:     ${m_fixed.get('final_equity', 0):,.2f}")
    print(f"CAGR:      {m_fixed.get('cagr', 0)*100:+.2f}%")
    print(f"Sharpe:    {m_fixed.get('sharpe', 0):.2f}")
    print(f"Sortino:   {m_fixed.get('sortino', 0):.2f}")
    print(f"MaxDD:     {m_fixed.get('max_dd', 0)*100:.1f}%")
    print(f"Calmar:    {m_fixed.get('calmar', 0):.2f}")

    # ── Compounding equity curve (50% alloc) ──
    eq_compound = build_equity_curve_compound(trades, daily, alloc_frac=0.50)
    m_compound = compute_metrics(eq_compound)

    print()
    print("── COMPOUND 50% ALLOC EQUITY CURVE ──")
    print(f"Starting:  ${STARTING_CASH:,.0f}")
    print(f"Final:     ${m_compound.get('final_equity', 0):,.2f}")
    print(f"CAGR:      {m_compound.get('cagr', 0)*100:+.2f}%")
    print(f"Sharpe:    {m_compound.get('sharpe', 0):.2f}")
    print(f"Sortino:   {m_compound.get('sortino', 0):.2f}")
    print(f"MaxDD:     {m_compound.get('max_dd', 0)*100:.1f}%")
    print(f"Calmar:    {m_compound.get('calmar', 0):.2f}")

    # ── Regime gate on fixed curve ──
    gate = regime_gate(eq_fixed)
    print()
    print("── REGIME GATE (HC #428 R1) — fixed sizing ──")
    for k in ["sharpe_green", "sharpe_red", "sharpe_flat", "regime_gap"]:
        v = gate.get(k)
        if v is not None:
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: N/A")
    print(f"  R1 PASS: {gate.get('hc428_r1_pass', False)}")

    # ── SPY comparison ──
    spy_start = daily.loc[:trades[0].open_date].iloc[-1]["close"]
    spy_end = daily.loc[:trades[-1].close_date].iloc[-1]["close"]
    spy_years = (trades[-1].close_date - trades[0].open_date).days / 365.25
    spy_cagr = (spy_end / spy_start) ** (1.0 / spy_years) - 1 if spy_years > 0 else 0

    print()
    print(f"SPY Buy&Hold CAGR: {spy_cagr*100:.2f}%")

    # ── Print all trades ──
    print()
    print("ALL TRADES:")
    print(f"{'Open':>12s} {'Close':>12s} {'Days':>4s} {'Strikes':>32s} {'Credit':>7s} "
          f"{'PnL':>10s} {'RoR':>7s} {'Reason':>14s} {'Pricing':>12s}")
    for t in trades:
        strikes = f"{t.long_put_K}/{t.short_put_K}/{t.short_call_K}/{t.long_call_K}"
        days = (t.close_date - t.open_date).days
        print(f"{str(t.open_date.date()):>12s} {str(t.close_date.date()):>12s} {days:>4d} "
              f"{strikes:>32s} {t.open_net_credit:>7.2f} "
              f"${t.pnl_per_contract:>+9.2f} {t.return_on_risk*100:>+6.1f}% "
              f"{t.close_reason:>14s} {t.close_pricing:>12s}")

    # ── Comparison ──
    print()
    print("=" * 70)
    print("COMPARISON: BS-synthetic (207% CAGR) vs REAL chain pricing")
    print("=" * 70)
    print(f"  BS version CAGR (compound):  207%")
    print(f"  Real chain CAGR (1-lot):     {m_fixed.get('cagr',0)*100:+.2f}%")
    print(f"  Real chain CAGR (compound):  {m_compound.get('cagr',0)*100:+.2f}%")

    # ── Save ──
    eq_fixed.to_parquet(OUT_DIR / "equity_fixed.parquet")
    eq_fixed.to_csv(OUT_DIR / "equity_fixed.csv")
    eq_compound.to_parquet(OUT_DIR / "equity_compound.parquet")
    eq_compound.to_csv(OUT_DIR / "equity_compound.csv")

    ledger_rows = []
    for t in trades:
        ledger_rows.append({
            "open_date": t.open_date, "close_date": t.close_date,
            "expiry": t.expiry,
            "short_put_K": t.short_put_K, "long_put_K": t.long_put_K,
            "short_call_K": t.short_call_K, "long_call_K": t.long_call_K,
            "open_net_credit": t.open_net_credit,
            "close_cost": t.close_cost,
            "pnl_per_share": t.pnl_per_share,
            "pnl_per_contract": t.pnl_per_contract,
            "return_on_risk": t.return_on_risk,
            "days_held": (t.close_date - t.open_date).days,
            "close_reason": t.close_reason,
            "close_pricing": t.close_pricing,
            "sp_delta_entry": t.entry_deltas["sp"],
            "lp_delta_entry": t.entry_deltas["lp"],
            "sc_delta_entry": t.entry_deltas["sc"],
            "lc_delta_entry": t.entry_deltas["lc"],
        })
    ledger_df = pd.DataFrame(ledger_rows)
    ledger_df.to_parquet(OUT_DIR / "ledger.parquet")
    ledger_df.to_csv(OUT_DIR / "ledger.csv", index=False)

    results = {
        "strategy": "iron_condor_spy_PURE_real_chains",
        "pricing": "REAL Dolt bid/ask ONLY — no BS anywhere in P&L",
        "commission": "$0 (Robinhood HC #694)",
        "period": f"{trades[0].open_date.date()} -> {trades[-1].close_date.date()}",
        "trade_stats": {
            "total_trades": n_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / n_trades,
            "total_pnl_1lot": total_pnl_1lot,
            "avg_pnl_per_trade": avg_pnl,
            "avg_winner": avg_win,
            "avg_loser": avg_loss,
            "avg_credit": avg_credit,
            "avg_hold_days": avg_hold,
            "avg_return_on_risk": avg_ror,
            "real_closes": real_closes,
            "intrinsic_closes": intrinsic_closes,
            "skipped": skipped,
        },
        "fixed_1lot_metrics": m_fixed,
        "compound_50pct_metrics": m_compound,
        "regime_gate": gate,
        "spy_buy_hold_cagr": spy_cagr,
        "comparison": {
            "bs_cagr_pct": 207.0,
            "real_fixed_cagr_pct": m_fixed.get("cagr", 0) * 100,
            "real_compound_cagr_pct": m_compound.get("cagr", 0) * 100,
        },
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR}")
    return results


if __name__ == "__main__":
    main()
