#!/usr/bin/env python3
"""
wheel_param_sweep.py — Parameter sweep for wheel strategy across 4 configs.

HC #660 R2: Find higher-return configurations by sweeping delta/DTE params
on the 41 Tier-1 tickers.

Configs:
  conservative:      PUT_DELTA=0.25, CALL_DELTA=0.30, DTE=30 (baseline)
  aggressive_delta:  PUT_DELTA=0.35, CALL_DELTA=0.25, DTE=30 (closer strikes)
  weekly:            PUT_DELTA=0.25, CALL_DELTA=0.30, DTE=14 (shorter cycles)
  aggressive_weekly: PUT_DELTA=0.35, CALL_DELTA=0.25, DTE=14 (max premium)
"""
from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------- paths ----------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "wheel_param_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIER1_FILE = ROOT / "output" / "wheel_expanded_universe" / "tiered_universe.json"

# ----------------------------- global config --------------------------------
START_DATE = pd.Timestamp("2019-01-01")
STARTING_CASH = 20_000.0
TRADING_DAYS = 252
RISK_FREE = 0.04
PROFIT_TAKE = 0.50
VIX_MAX = 35.0
COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03

# ----------------------------- sweep configs --------------------------------
CONFIGS = {
    "conservative": {
        "PUT_DELTA": 0.25,
        "CALL_DELTA": 0.30,
        "DTE_TARGET": 30,
        "DTE_MIN": 25,
        "DTE_MAX": 35,
    },
    "aggressive_delta": {
        "PUT_DELTA": 0.35,
        "CALL_DELTA": 0.25,
        "DTE_TARGET": 30,
        "DTE_MIN": 25,
        "DTE_MAX": 35,
    },
    "weekly": {
        "PUT_DELTA": 0.25,
        "CALL_DELTA": 0.30,
        "DTE_TARGET": 14,
        "DTE_MIN": 10,
        "DTE_MAX": 18,
    },
    "aggressive_weekly": {
        "PUT_DELTA": 0.35,
        "CALL_DELTA": 0.25,
        "DTE_TARGET": 14,
        "DTE_MIN": 10,
        "DTE_MAX": 18,
    },
}


# ----------------------------- BS pricing -----------------------------------
def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
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


def slippage(premium):
    if premium is None or premium <= 0:
        return 0.0
    return max(SLIPPAGE_MIN, SLIPPAGE_FRAC * premium)


# ----------------------------- expiry finder --------------------------------
def find_expiry(open_date, dte_min, dte_max, dte_target):
    best, best_dist = None, 10_000
    for d_off in range(dte_min, dte_max + 1):
        cand = open_date + pd.Timedelta(days=d_off)
        shift = (4 - cand.weekday()) % 7
        cand_fri = cand + pd.Timedelta(days=shift)
        dte = (cand_fri - open_date).days
        if dte < dte_min or dte > dte_max:
            continue
        dist = abs(dte - dte_target)
        if dist < best_dist:
            best, best_dist = cand_fri, dist
    return best


# ----------------------------- wheel engine ---------------------------------
@dataclass
class Position:
    side: str
    strike: float
    expiry: pd.Timestamp
    open_date: pd.Timestamp
    open_price: float
    contracts: int
    share_basis: float = 0.0


def run_wheel(dates, closes, sigmas, vix_arr, cfg):
    """Run wheel on a single ticker with given config. Returns dict of metrics or None."""
    put_delta = cfg["PUT_DELTA"]
    call_delta = cfg["CALL_DELTA"]
    dte_target = cfg["DTE_TARGET"]
    dte_min = cfg["DTE_MIN"]
    dte_max = cfg["DTE_MAX"]

    n = len(dates)
    cash = STARTING_CASH
    position = None
    state = None

    equity_series = np.empty(n, dtype=np.float64)
    n_trades = 0
    wins = 0
    total_premium = 0.0
    n_assignments = 0

    for i in range(n):
        S = closes[i]
        sigma = sigmas[i]
        vix = vix_arr[i]

        # --- process existing position ---
        if position is not None:
            T = max((position.expiry - dates[i]).days, 0) / 365.0

            if state == "short_put":
                opt = bs_price(S, position.strike, T, sigma, kind="put")
                pnl_ps = position.open_price - opt
                pf = pnl_ps / position.open_price if position.open_price > 0 else 0.0

                is_expiry = dates[i] >= position.expiry
                close_profit = pf >= PROFIT_TAKE and not is_expiry

                if close_profit:
                    slip = slippage(opt)
                    cost = (opt + slip) * 100 * position.contracts + COST_PER_CONTRACT * position.contracts
                    realized = position.open_price * 100 * position.contracts - cost
                    cash += realized
                    n_trades += 1
                    if realized > 0:
                        wins += 1
                    position = None
                    state = None
                elif is_expiry:
                    if S < position.strike:
                        # assigned
                        cost = position.strike * 100 * position.contracts
                        cash -= cost
                        basis = position.strike - position.open_price
                        n_trades += 1
                        wins += 1
                        n_assignments += 1
                        position = Position(
                            side="long_shares", strike=basis,
                            expiry=dates[i], open_date=dates[i],
                            open_price=basis, contracts=position.contracts,
                            share_basis=basis,
                        )
                        state = "long_shares"
                    else:
                        realized = position.open_price * 100 * position.contracts - COST_PER_CONTRACT * position.contracts
                        n_trades += 1
                        if realized > 0:
                            wins += 1
                        position = None
                        state = None

            elif state == "short_call":
                opt = bs_price(S, position.strike, T, sigma, kind="call")
                pnl_ps = position.open_price - opt
                pf = pnl_ps / position.open_price if position.open_price > 0 else 0.0

                is_expiry = dates[i] >= position.expiry
                close_profit = pf >= PROFIT_TAKE and not is_expiry

                if close_profit:
                    slip = slippage(opt)
                    cost = (opt + slip) * 100 * position.contracts + COST_PER_CONTRACT * position.contracts
                    realized = position.open_price * 100 * position.contracts - cost
                    cash += realized
                    n_trades += 1
                    if realized > 0:
                        wins += 1
                    position = Position(
                        side="long_shares", strike=position.share_basis,
                        expiry=dates[i], open_date=dates[i],
                        open_price=position.share_basis, contracts=position.contracts,
                        share_basis=position.share_basis,
                    )
                    state = "long_shares"
                elif is_expiry:
                    if S > position.strike:
                        # called away
                        proceeds = position.strike * 100 * position.contracts
                        premium_kept = position.open_price * 100 * position.contracts - COST_PER_CONTRACT * position.contracts
                        share_pnl = (position.strike - position.share_basis) * 100 * position.contracts
                        cash += proceeds + premium_kept
                        n_trades += 1
                        n_assignments += 1
                        if (premium_kept + share_pnl) > 0:
                            wins += 1
                        position = None
                        state = None
                    else:
                        realized = position.open_price * 100 * position.contracts - COST_PER_CONTRACT * position.contracts
                        cash += realized
                        n_trades += 1
                        if realized > 0:
                            wins += 1
                        position = Position(
                            side="long_shares", strike=position.share_basis,
                            expiry=dates[i], open_date=dates[i],
                            open_price=position.share_basis, contracts=position.contracts,
                            share_basis=position.share_basis,
                        )
                        state = "long_shares"

        # --- open new legs ---
        if state is None and position is None:
            if vix <= VIX_MAX:
                expiry = find_expiry(dates[i], dte_min, dte_max, dte_target)
                if expiry is not None:
                    T = (expiry - dates[i]).days / 365.0
                    K = strike_from_delta(S, T, sigma, put_delta, kind="put")
                    prem = bs_price(S, K, T, sigma, kind="put")
                    slip = slippage(prem)
                    net_prem = prem - slip
                    contracts = int((cash * 0.95) // (K * 100))
                    if contracts >= 1 and net_prem > 0.05:
                        credit = net_prem * 100 * contracts - COST_PER_CONTRACT * contracts
                        cash += credit
                        total_premium += net_prem * 100 * contracts
                        position = Position(
                            side="short_put", strike=K, expiry=expiry,
                            open_date=dates[i], open_price=net_prem,
                            contracts=contracts, share_basis=0.0,
                        )
                        state = "short_put"

        elif state == "long_shares" and position is not None:
            expiry = find_expiry(dates[i], dte_min, dte_max, dte_target)
            if expiry is not None:
                T = (expiry - dates[i]).days / 365.0
                K = strike_from_delta(S, T, sigma, call_delta, kind="call")
                prem = bs_price(S, K, T, sigma, kind="call")
                slip = slippage(prem)
                net_prem = prem - slip
                if net_prem > 0.05:
                    credit = net_prem * 100 * position.contracts - COST_PER_CONTRACT * position.contracts
                    cash += credit
                    total_premium += net_prem * 100 * position.contracts
                    position = Position(
                        side="short_call", strike=K, expiry=expiry,
                        open_date=dates[i], open_price=net_prem,
                        contracts=position.contracts,
                        share_basis=position.share_basis,
                    )
                    state = "short_call"

        # --- MTM equity ---
        equity = cash
        if position is not None:
            T = max((position.expiry - dates[i]).days, 0) / 365.0
            if state == "short_put":
                opt = bs_price(S, position.strike, T, sigma, kind="put")
                equity += (position.open_price - opt) * 100 * position.contracts
            elif state == "long_shares":
                equity += (S - position.share_basis) * 100 * position.contracts
            elif state == "short_call":
                opt = bs_price(S, position.strike, T, sigma, kind="call")
                equity += (S - position.share_basis) * 100 * position.contracts
                equity += (position.open_price - opt) * 100 * position.contracts

        equity_series[i] = equity

    # --- compute metrics ---
    if n < 10:
        return None

    rets = np.diff(equity_series) / equity_series[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) < 20:
        return None

    years = (dates[-1] - dates[0]).days / 365.25
    total_ret = equity_series[-1] / equity_series[0] - 1.0
    ann_ret = (1 + total_ret) ** (1.0 / max(years, 0.01)) - 1.0
    mu = np.mean(rets)
    sd = np.std(rets, ddof=1)
    sharpe = (mu / sd) * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan
    downside = np.std(rets[rets < 0], ddof=1) if np.sum(rets < 0) > 1 else np.nan
    sortino = (mu / downside) * np.sqrt(TRADING_DAYS) if downside and downside > 0 else np.nan
    peak = np.maximum.accumulate(equity_series)
    dd = equity_series / peak - 1.0
    max_dd = float(np.min(dd))
    wr = float(np.mean(rets > 0))
    premium_pct = total_premium / STARTING_CASH * 100.0
    assignment_rate = n_assignments / n_trades if n_trades > 0 else 0.0

    return {
        "total_return_pct": total_ret * 100.0,
        "ann_return_pct": ann_ret * 100.0,
        "sharpe": float(sharpe),
        "sortino": float(sortino) if np.isfinite(sortino) else np.nan,
        "max_dd_pct": max_dd * 100.0,
        "win_rate": wr,
        "n_trades": n_trades,
        "n_wins": wins,
        "premium_income_pct": premium_pct,
        "assignment_rate": assignment_rate,
        "n_assignments": n_assignments,
        "final_equity": float(equity_series[-1]),
        "years": years,
        "n_days": n,
    }


# ----------------------------- data loading ---------------------------------
def load_data_and_tickers():
    """Load prices, compute IV, load VIX, and return Tier 1 ticker list."""
    # Load Tier 1 tickers
    with open(TIER1_FILE) as f:
        tier_data = json.load(f)
    tier1_tickers = tier_data["tier1_premium"]["tickers"]
    print(f"[load] {len(tier1_tickers)} Tier-1 tickers loaded")

    # Load universe metadata
    universe = pd.read_parquet(CACHE / "universe_expanded.parquet")
    ticker_meta = {}
    for _, row in universe.iterrows():
        ticker_meta[row["ticker"]] = {
            "sector": row["sector"],
            "beta": row.get("beta", np.nan),
        }

    # Load prices
    print("[load] Loading price data ...")
    p1 = pd.read_parquet(CACHE / "prices.parquet")[["ticker", "date", "close"]].copy()
    p1["date"] = pd.to_datetime(p1["date"], utc=False)
    if p1["date"].dt.tz is not None:
        p1["date"] = p1["date"].dt.tz_localize(None)

    p2 = pd.read_parquet(CACHE / "prices_expanded.parquet")
    p2 = p2.rename(columns={"Close": "close"})[["ticker", "date", "close"]].copy()
    p2["date"] = pd.to_datetime(p2["date"], utc=False)
    if p2["date"].dt.tz is not None:
        p2["date"] = p2["date"].dt.tz_localize(None)

    prices = pd.concat([p1, p2], ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices = prices.dropna(subset=["close"])
    prices = prices[prices["close"] > 0]
    prices = prices[prices["date"] >= START_DATE].copy()

    # Filter to Tier 1 only
    prices = prices[prices["ticker"].isin(tier1_tickers)].copy()

    # VIX
    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"], utc=False)
    if macro["date"].dt.tz is not None:
        macro["date"] = macro["date"].dt.tz_localize(None)
    prices = prices.merge(macro, on="date", how="left")
    prices["vix"] = prices["vix"].ffill().fillna(20.0)

    # 20-day realized vol
    print("[load] Computing realized vol ...")
    prices["log_ret"] = prices.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))
    prices["sigma"] = prices.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(20, min_periods=15).std() * np.sqrt(252)
    )
    prices["sigma"] = prices["sigma"].clip(lower=0.05, upper=2.0)
    prices = prices.dropna(subset=["sigma"]).reset_index(drop=True)

    available = sorted(prices["ticker"].unique())
    print(f"[load] {len(available)} tickers with data, "
          f"{len(prices)} rows, {prices['date'].min().date()} to {prices['date'].max().date()}")

    return prices, ticker_meta, tier1_tickers, available


# ----------------------------- main -----------------------------------------
def main():
    t0 = time.time()
    prices, ticker_meta, tier1_tickers, available_tickers = load_data_and_tickers()

    all_results = {}  # config_name -> list of dicts

    for cfg_name, cfg in CONFIGS.items():
        print(f"\n{'='*80}")
        print(f"  CONFIG: {cfg_name}")
        print(f"  PUT_DELTA={cfg['PUT_DELTA']}, CALL_DELTA={cfg['CALL_DELTA']}, "
              f"DTE={cfg['DTE_TARGET']} ({cfg['DTE_MIN']}-{cfg['DTE_MAX']})")
        print(f"{'='*80}")

        results = []
        skipped = []

        for idx, tk in enumerate(available_tickers):
            df_t = prices[prices["ticker"] == tk].sort_values("date")
            if len(df_t) < 252:
                skipped.append(tk)
                continue

            dates = df_t["date"].values.astype("datetime64[ns]")
            dates_ts = pd.DatetimeIndex(dates)
            closes = df_t["close"].values.astype(np.float64)
            sigmas = df_t["sigma"].values.astype(np.float64)
            vix_arr = df_t["vix"].values.astype(np.float64)

            m = run_wheel(dates_ts, closes, sigmas, vix_arr, cfg)
            if m is None:
                skipped.append(tk)
                continue

            meta = ticker_meta.get(tk, {})
            m["ticker"] = tk
            m["sector"] = meta.get("sector", "Unknown")
            m["beta"] = meta.get("beta", np.nan)
            m["config"] = cfg_name
            results.append(m)

        all_results[cfg_name] = results
        elapsed = time.time() - t0
        print(f"  -> {len(results)} tickers completed, {len(skipped)} skipped ({elapsed:.0f}s)")

        # Per-config summary
        if results:
            df_cfg = pd.DataFrame(results)
            df_cfg = df_cfg.sort_values("sharpe", ascending=False)
            df_cfg.to_csv(OUT_DIR / f"{cfg_name}_results.csv", index=False)

            print(f"\n  Top 10 by Sharpe ({cfg_name}):")
            print(f"  {'Ticker':<8} {'Sharpe':>7} {'AnnRet%':>8} {'MaxDD%':>8} "
                  f"{'WR%':>6} {'AssignR':>7} {'Prem%':>7}")
            print(f"  {'-'*55}")
            for _, row in df_cfg.head(10).iterrows():
                print(f"  {row['ticker']:<8} {row['sharpe']:>7.2f} "
                      f"{row['ann_return_pct']:>7.1f}% {row['max_dd_pct']:>7.1f}% "
                      f"{row['win_rate']*100:>5.1f}% {row['assignment_rate']*100:>6.1f}% "
                      f"{row['premium_income_pct']:>6.0f}%")

    # ====================== CROSS-CONFIG COMPARISON =========================
    print("\n\n" + "=" * 90)
    print("CROSS-CONFIG COMPARISON")
    print("=" * 90)

    comparison = {}
    for cfg_name, results in all_results.items():
        if not results:
            continue
        df_c = pd.DataFrame(results)
        comparison[cfg_name] = {
            "n_tickers": len(df_c),
            "median_sharpe": float(df_c["sharpe"].median()),
            "mean_sharpe": float(df_c["sharpe"].mean()),
            "median_ann_return": float(df_c["ann_return_pct"].median()),
            "mean_ann_return": float(df_c["ann_return_pct"].mean()),
            "median_max_dd": float(df_c["max_dd_pct"].median()),
            "mean_max_dd": float(df_c["max_dd_pct"].mean()),
            "median_win_rate": float(df_c["win_rate"].median()),
            "median_premium_pct": float(df_c["premium_income_pct"].median()),
            "mean_premium_pct": float(df_c["premium_income_pct"].mean()),
            "median_assignment_rate": float(df_c["assignment_rate"].median()),
            "mean_assignment_rate": float(df_c["assignment_rate"].mean()),
            "sharpe_gt1_pct": float((df_c["sharpe"] > 1.0).mean() * 100),
            "positive_sharpe_pct": float((df_c["sharpe"] > 0).mean() * 100),
        }

    print(f"\n{'Config':<20} {'MedSharpe':>10} {'MeanSharpe':>11} {'MedAnnRet%':>11} "
          f"{'MeanAnnRet%':>12} {'MedMaxDD%':>10} {'MedWR%':>7} {'MedPrem%':>9} {'MedAssign%':>11}")
    print("-" * 110)
    for cfg_name in CONFIGS:
        if cfg_name not in comparison:
            continue
        c = comparison[cfg_name]
        print(f"{cfg_name:<20} {c['median_sharpe']:>10.3f} {c['mean_sharpe']:>11.3f} "
              f"{c['median_ann_return']:>10.1f}% {c['mean_ann_return']:>11.1f}% "
              f"{c['median_max_dd']:>9.1f}% {c['median_win_rate']*100:>6.1f}% "
              f"{c['median_premium_pct']:>8.0f}% {c['median_assignment_rate']*100:>10.1f}%")

    # ====================== SECTOR BREAKDOWN PER CONFIG =====================
    print("\n\n" + "=" * 90)
    print("SECTOR-LEVEL BREAKDOWN BY CONFIG")
    print("=" * 90)

    sector_comparison = {}
    for cfg_name, results in all_results.items():
        if not results:
            continue
        df_c = pd.DataFrame(results)
        sector_agg = df_c.groupby("sector").agg(
            n=("ticker", "count"),
            avg_sharpe=("sharpe", "mean"),
            med_sharpe=("sharpe", "median"),
            avg_ann_ret=("ann_return_pct", "mean"),
            avg_max_dd=("max_dd_pct", "mean"),
            avg_assignment=("assignment_rate", "mean"),
        ).sort_values("avg_sharpe", ascending=False)

        sector_agg.to_csv(OUT_DIR / f"{cfg_name}_sector.csv")
        sector_comparison[cfg_name] = sector_agg.reset_index().to_dict("records")

        print(f"\n  {cfg_name}:")
        print(f"  {'Sector':<25} {'N':>3} {'AvgSharpe':>10} {'MedSharpe':>10} "
              f"{'AvgAnnRet%':>11} {'AvgMaxDD%':>10} {'AvgAssign%':>11}")
        print(f"  {'-'*85}")
        for _, row in sector_agg.iterrows():
            print(f"  {row.name:<25} {row['n']:>3} {row['avg_sharpe']:>10.2f} "
                  f"{row['med_sharpe']:>10.2f} {row['avg_ann_ret']:>10.1f}% "
                  f"{row['avg_max_dd']:>9.1f}% {row['avg_assignment']*100:>10.1f}%")

    # ====================== DELTA COMPARISON (per ticker) ===================
    print("\n\n" + "=" * 90)
    print("PER-TICKER: AGGRESSIVE vs CONSERVATIVE (return uplift)")
    print("=" * 90)

    # Build merged per-ticker view
    cons_df = pd.DataFrame(all_results.get("conservative", []))
    agg_df = pd.DataFrame(all_results.get("aggressive_weekly", []))

    if len(cons_df) > 0 and len(agg_df) > 0:
        merged = cons_df[["ticker", "sharpe", "ann_return_pct", "max_dd_pct"]].merge(
            agg_df[["ticker", "sharpe", "ann_return_pct", "max_dd_pct"]],
            on="ticker", suffixes=("_cons", "_aggw"),
        )
        merged["return_uplift"] = merged["ann_return_pct_aggw"] - merged["ann_return_pct_cons"]
        merged["sharpe_change"] = merged["sharpe_aggw"] - merged["sharpe_cons"]
        merged["dd_change"] = merged["max_dd_pct_aggw"] - merged["max_dd_pct_cons"]
        merged = merged.sort_values("return_uplift", ascending=False)

        print(f"{'Ticker':<8} {'ConsRet%':>9} {'AggWRet%':>9} {'Uplift%':>8} "
              f"{'ConsSharpe':>11} {'AggWSharpe':>11} {'ConsDD%':>8} {'AggWDD%':>8}")
        print("-" * 85)
        for _, row in merged.iterrows():
            print(f"{row['ticker']:<8} {row['ann_return_pct_cons']:>8.1f}% "
                  f"{row['ann_return_pct_aggw']:>8.1f}% {row['return_uplift']:>7.1f}% "
                  f"{row['sharpe_cons']:>11.2f} {row['sharpe_aggw']:>11.2f} "
                  f"{row['max_dd_pct_cons']:>7.1f}% {row['max_dd_pct_aggw']:>7.1f}%")

        # Winners/losers summary
        winners = (merged["return_uplift"] > 0).sum()
        losers = (merged["return_uplift"] <= 0).sum()
        avg_uplift = merged["return_uplift"].mean()
        med_uplift = merged["return_uplift"].median()
        print(f"\nReturn uplift (aggressive_weekly vs conservative):")
        print(f"  Winners: {winners}, Losers: {losers}")
        print(f"  Mean uplift: {avg_uplift:+.1f}%, Median uplift: {med_uplift:+.1f}%")
        print(f"  Mean Sharpe change: {merged['sharpe_change'].mean():+.3f}")
        print(f"  Mean DD change: {merged['dd_change'].mean():+.1f}%")

    # ====================== SAVE COMPARISON JSON ============================
    output = {
        "generated": datetime.now().isoformat(),
        "configs": {name: dict(cfg) for name, cfg in CONFIGS.items()},
        "global_params": {
            "starting_cash": STARTING_CASH,
            "profit_take": PROFIT_TAKE,
            "vix_gate": VIX_MAX,
            "cost_per_contract": COST_PER_CONTRACT,
            "iv_model": "rv_20",
        },
        "comparison": comparison,
        "sector_breakdown": sector_comparison,
    }

    with open(OUT_DIR / "comparison.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n[done] Total runtime: {elapsed:.1f}s")
    print(f"[done] Results saved to {OUT_DIR}")

    return comparison


if __name__ == "__main__":
    main()
