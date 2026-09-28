"""
macro_regime.regime_rotation_backtest — regime-conditioned sector rotation.

Universe: XLK XLF XLE XLV XLI XLP XLY XLU XLB XLRE XLC + cash + TLT (for stress)
Rules per regime (tilts, top-K=3):
  EARLY:     cyclicals      → XLF XLI XLY XLB
  MID:       growth+tech    → XLK XLC XLY
  LATE:      defensives+E   → XLE XLP XLU XLV
  RECESSION: cash+defense   → CASH XLP XLU TLT (or all-cash if STRESS gate triggers)

Rebalance weekly (every 5 trading days). Equal weight within holdings. No leverage.
Costs: 1bp slippage + $0.005/share commission (commission applied as
       est_commission_ticks = COMM_PER_SHARE / mean_share_price * 1.0 → bps).

Backtest 2018-01-01 → 2025-12-31.

Metrics:
  CAGR, MaxDD, Sharpe, Sortino, Calmar, PF, WR
  HC #428 R1: per-day Sharpe stratified by SPY green/red/flat days
  Day-concentration: max(top_day_contribution / total_pnl)
  Regime transitions per year
  vs SPY buy-and-hold AND vs leader rotation baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "strategy"))
sys.path.insert(0, str(ROOT))

from macro_regime.regime_classifier import (
    build_feature_panel, classify_regime, regime_transitions,
)

PRICES_V2 = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"
SECTOR_ETFS_PATH = ROOT / "wheel_strategy_v1/data/cache/sector_etfs.parquet"

UNIVERSE = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC", "TLT"]

REGIME_TILTS = {
    "EARLY":     {"long": ["XLF", "XLI", "XLY", "XLB"], "topk": 3, "cash_frac": 0.0},
    "MID":       {"long": ["XLK", "XLC", "XLY"],         "topk": 3, "cash_frac": 0.0},
    "LATE":      {"long": ["XLE", "XLP", "XLU", "XLV"], "topk": 3, "cash_frac": 0.0},
    "RECESSION": {"long": ["XLP", "XLU", "TLT"],         "topk": 3, "cash_frac": 0.50},
}

# Baseline (regime-agnostic): all 11 sector ETFs, equal weight, rebalance weekly
BASELINE_UNIVERSE = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]

TRADING_DAYS = 252
COMM_PER_SHARE = 0.005
SLIPPAGE_BPS = 1.0


def load_prices() -> pd.DataFrame:
    """Long-form (date, ticker, close) for all needed tickers."""
    df1 = pd.read_parquet(PRICES_V2)
    df1 = df1[["date", "ticker", "close"]]
    df1["date"] = pd.to_datetime(df1["date"])
    needed = set(UNIVERSE) | {"SPY"}
    df = df1[df1["ticker"].isin(needed)].copy()
    # Pivot wide
    wide = df.pivot(index="date", columns="ticker", values="close").sort_index()
    return wide


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)


def _txn_cost(n_legs_changed: int, avg_px: float, gross_w_changed: float) -> float:
    """Return cost as a fraction of portfolio NAV.
       Commission: $0.005/share * (gross_$ / avg_px) → cost_$ → /NAV
       Slippage: SLIPPAGE_BPS on the gross dollars changed.
    For simplicity we use a single avg_px ~ 100 (sector ETFs typically $50-200);
    final commission magnitude is ~ 0.005/100 = 5bps of dollar turnover. We
    explicitly apply both:
      cost_frac = gross_w_changed * (SLIPPAGE_BPS/10000 + COMM_PER_SHARE/avg_px)
    """
    comm_bps = (COMM_PER_SHARE / max(avg_px, 1.0)) * 10000.0
    total_bps = SLIPPAGE_BPS + comm_bps
    return gross_w_changed * total_bps / 10000.0


def run_strategy(returns: pd.DataFrame,
                 regime: pd.Series,
                 start: str = "2018-01-01",
                 end: str = "2025-12-31",
                 rebalance_n: int = 5,
                 use_regime: bool = True,
                 prices: pd.DataFrame | None = None) -> dict:
    """Run the regime-rotation (or unconditional baseline) backtest."""
    ret = returns.loc[start:end].copy()
    reg = regime.reindex(ret.index, method="ffill").fillna("MID")

    # Track weights, daily returns
    cols = ret.columns.tolist()
    weights = pd.DataFrame(0.0, index=ret.index, columns=cols + ["CASH"])
    book_ret = pd.Series(0.0, index=ret.index)
    cost_log = pd.Series(0.0, index=ret.index)
    holdings_log = []  # list of dicts

    rebal_idx = list(range(0, len(ret.index), rebalance_n))
    target_w = pd.Series(0.0, index=cols + ["CASH"])
    last_avg_px = 100.0  # rough

    for i, date in enumerate(ret.index):
        # Rebalance decision
        if i in rebal_idx:
            cur_regime = reg.loc[date]
            if use_regime:
                tilts = REGIME_TILTS[cur_regime]
                pool = [t for t in tilts["long"] if t in cols]
                # Top-K selection within the pool: use 21d momentum among the pool
                if i >= 21:
                    mom = ret.iloc[i-21:i][pool].add(1).prod() - 1.0
                    chosen = mom.nlargest(min(tilts["topk"], len(pool))).index.tolist()
                else:
                    chosen = pool[:tilts["topk"]]
                cash_w = tilts["cash_frac"]
                eq_w = (1.0 - cash_w) / max(len(chosen), 1)
                new_w = pd.Series(0.0, index=cols + ["CASH"])
                for t in chosen:
                    new_w[t] = eq_w
                new_w["CASH"] = cash_w
            else:
                # baseline: equal weight across BASELINE_UNIVERSE, regime-agnostic
                pool = [t for t in BASELINE_UNIVERSE if t in cols]
                if i >= 21:
                    mom = ret.iloc[i-21:i][pool].add(1).prod() - 1.0
                    chosen = mom.nlargest(3).index.tolist()
                else:
                    chosen = pool[:3]
                new_w = pd.Series(0.0, index=cols + ["CASH"])
                eq_w = 1.0 / max(len(chosen), 1)
                for t in chosen:
                    new_w[t] = eq_w
                new_w["CASH"] = 0.0

            # Estimate avg px for commission
            if prices is not None:
                px_today = prices.loc[date, [t for t in chosen if t in prices.columns]] if chosen else None
                if px_today is not None and len(px_today) > 0:
                    last_avg_px = float(px_today.mean())

            # txn cost on weight delta
            gross_changed = float((new_w - target_w).abs().sum())
            cost_frac = _txn_cost(len(chosen), last_avg_px, gross_changed)
            cost_log.loc[date] = cost_frac
            target_w = new_w.copy()
            holdings_log.append({"date": str(date.date()), "regime": cur_regime,
                                  "chosen": chosen, "cash_w": float(new_w["CASH"])})

        weights.loc[date] = target_w.values
        # Daily P&L: dot of weights and returns (CASH returns 0 — no money market modelled)
        day_r = float((target_w[cols] * ret.loc[date, cols]).sum())
        book_ret.loc[date] = day_r - cost_log.loc[date]

    return {
        "daily_ret": book_ret,
        "weights": weights,
        "regime": reg,
        "cost_log": cost_log,
        "holdings_log": holdings_log,
    }


def perf_metrics(daily_ret: pd.Series, anchor: float = 20_000) -> dict:
    r = daily_ret.dropna()
    if r.empty:
        return {}
    eq = (1.0 + r).cumprod()
    nav = anchor * eq
    n = len(r)
    years = n / TRADING_DAYS
    cagr = float(eq.iloc[-1] ** (1 / max(years, 1e-9)) - 1.0) if eq.iloc[-1] > 0 else float("nan")
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std(ddof=1) > 0 else float("nan")
    downside = r[r < 0]
    sortino = float(r.mean() / downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 and downside.std(ddof=1) > 0 else float("nan")
    dd = (eq / eq.cummax() - 1.0)
    maxdd = float(dd.min())
    calmar = float(cagr / abs(maxdd)) if maxdd < 0 else float("nan")
    wr = float((r > 0).mean())
    pf_num = float(r[r > 0].sum())
    pf_den = float(-r[r < 0].sum())
    pf = pf_num / pf_den if pf_den > 0 else float("nan")
    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": maxdd,
        "calmar": calmar,
        "wr": wr,
        "pf": pf,
        "final_nav": float(nav.iloc[-1]),
        "n_days": n,
        "years": years,
    }


def hc428_r1_check(daily_ret: pd.Series, spy_ret: pd.Series) -> dict:
    """Stratified Sharpe by SPY green/red/flat days. Gap test."""
    df = pd.DataFrame({"r": daily_ret, "spy": spy_ret}).dropna()
    df["regime"] = "flat"
    df.loc[df["spy"] > 0.0010, "regime"] = "green"
    df.loc[df["spy"] < -0.0010, "regime"] = "red"

    out = {}
    for cat in ("green", "red", "flat"):
        sub = df[df["regime"] == cat]["r"]
        if len(sub) < 5:
            out[cat] = {"sharpe": float("nan"), "n": len(sub), "mean": float("nan")}
            continue
        sd = sub.std(ddof=1)
        sh = float(sub.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
        out[cat] = {"sharpe": sh, "n": int(len(sub)), "mean": float(sub.mean())}

    sg = out["green"]["sharpe"]
    sr = out["red"]["sharpe"]
    if np.isfinite(sg) and np.isfinite(sr) and max(abs(sg), abs(sr)) > 1e-9:
        gap = abs(sg - sr) / max(abs(sg), abs(sr))
    else:
        gap = float("nan")
    out["gap"] = gap
    out["pass_R1"] = bool(np.isfinite(gap) and gap <= 0.50)
    return out


def day_concentration(daily_ret: pd.Series) -> float:
    r = daily_ret.dropna()
    if r.empty:
        return float("nan")
    top1 = r.abs().max()
    tot = r.abs().sum()
    return float(top1 / tot) if tot > 0 else float("nan")


def perf_per_regime(daily_ret: pd.Series, regime: pd.Series) -> dict:
    df = pd.DataFrame({"r": daily_ret, "reg": regime.reindex(daily_ret.index, method="ffill")}).dropna()
    out = {}
    for reg in ("EARLY", "MID", "LATE", "RECESSION"):
        sub = df[df["reg"] == reg]["r"]
        if len(sub) < 5:
            out[reg] = {"n": int(len(sub)), "sharpe": float("nan"),
                        "mean_daily": float("nan"), "cum": 0.0}
            continue
        sd = sub.std(ddof=1)
        sh = float(sub.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
        out[reg] = {
            "n": int(len(sub)),
            "sharpe": sh,
            "mean_daily": float(sub.mean()),
            "cum": float((1 + sub).prod() - 1.0),
        }
    return out
