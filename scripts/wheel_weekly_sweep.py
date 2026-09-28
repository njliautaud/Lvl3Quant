#!/usr/bin/env python3
"""
wheel_expanded_sweep.py — Wheel backtest across expanded 197-name universe.

HC #660: Run simplified wheel (CSP + CC cycle) on all 197 tickers independently,
rank by Sharpe/return, aggregate by sector and beta bucket.

IV modeled as 20-day realized vol (no external IV data needed).
"""
from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------- paths ----------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "wheel_weekly_full_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------- config ---------------------------------------
START_DATE = pd.Timestamp("2019-01-01")
STARTING_CASH = 20_000.0
TRADING_DAYS = 252
RISK_FREE = 0.04

PUT_DELTA = 0.25
CALL_DELTA = 0.30
DTE_MIN = 10
DTE_MAX = 18
DTE_TARGET = 14  # Weekly — param sweep confirmed +13% Sharpe vs monthly
PROFIT_TAKE = 0.50
VIX_MAX = 35.0

COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03


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
    # Rational approximation of inverse normal CDF (Beasley-Springer-Moro)
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
def find_expiry(open_date, dte_min=DTE_MIN, dte_max=DTE_MAX, dte_target=DTE_TARGET):
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
    side: str       # 'short_put' | 'long_shares' | 'short_call'
    strike: float
    expiry: pd.Timestamp
    open_date: pd.Timestamp
    open_price: float
    contracts: int
    share_basis: float = 0.0


def run_wheel(dates, closes, sigmas, vix_arr):
    """Run wheel on a single ticker. Returns dict of metrics or None if insufficient data."""
    n = len(dates)
    cash = STARTING_CASH
    position = None  # at most one active position/state
    state = None     # None=cash, 'short_put', 'long_shares', 'short_call'

    equity_series = np.empty(n, dtype=np.float64)
    n_trades = 0
    wins = 0
    total_premium = 0.0

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
                    # buy back for profit
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
                        wins += 1  # premium kept
                        position = Position(
                            side="long_shares", strike=basis,
                            expiry=dates[i], open_date=dates[i],
                            open_price=basis, contracts=position.contracts,
                            share_basis=basis,
                        )
                        state = "long_shares"
                    else:
                        # expired worthless
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
                    # revert to long_shares
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
                        if (premium_kept + share_pnl) > 0:
                            wins += 1
                        position = None
                        state = None
                    else:
                        # CC expired, keep shares + premium
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
            # try new CSP
            if vix <= VIX_MAX:
                expiry = find_expiry(dates[i])
                if expiry is not None:
                    T = (expiry - dates[i]).days / 365.0
                    K = strike_from_delta(S, T, sigma, PUT_DELTA, kind="put")
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
            # sell CC on shares
            expiry = find_expiry(dates[i])
            if expiry is not None:
                T = (expiry - dates[i]).days / 365.0
                K = strike_from_delta(S, T, sigma, CALL_DELTA, kind="call")
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
        "final_equity": float(equity_series[-1]),
        "years": years,
        "n_days": n,
    }


# ----------------------------- data loading ---------------------------------
def load_all_data():
    """Load and merge prices for all 197 tickers, compute IV, load VIX."""
    print("[load] Loading universe metadata ...")
    universe = pd.read_parquet(CACHE / "universe_expanded.parquet")
    ticker_meta = {}
    for _, row in universe.iterrows():
        ticker_meta[row["ticker"]] = {
            "sector": row["sector"],
            "beta": row.get("beta", np.nan),
            "market_cap": row.get("market_cap", 0),
        }

    print("[load] Loading original prices (70 tickers) ...")
    p1 = pd.read_parquet(CACHE / "prices.parquet")[["ticker", "date", "close"]].copy()
    p1["date"] = pd.to_datetime(p1["date"], utc=False)
    # Strip timezone if present
    if p1["date"].dt.tz is not None:
        p1["date"] = p1["date"].dt.tz_localize(None)

    print("[load] Loading expanded prices (127 tickers) ...")
    p2 = pd.read_parquet(CACHE / "prices_expanded.parquet")
    p2 = p2.rename(columns={"Close": "close"})[["ticker", "date", "close"]].copy()
    p2["date"] = pd.to_datetime(p2["date"], utc=False)
    if p2["date"].dt.tz is not None:
        p2["date"] = p2["date"].dt.tz_localize(None)

    print("[load] Merging price datasets ...")
    prices = pd.concat([p1, p2], ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices = prices.dropna(subset=["close"])
    prices = prices[prices["close"] > 0]

    # Filter to start date onward
    prices = prices[prices["date"] >= START_DATE].copy()

    print("[load] Loading macro/VIX data ...")
    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"], utc=False)
    if macro["date"].dt.tz is not None:
        macro["date"] = macro["date"].dt.tz_localize(None)

    prices = prices.merge(macro, on="date", how="left")
    prices["vix"] = prices["vix"].ffill().fillna(20.0)

    # Compute 20-day realized vol as IV proxy
    print("[load] Computing 20-day realized vol per ticker ...")
    prices["log_ret"] = prices.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))
    prices["sigma"] = prices.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(20, min_periods=15).std() * np.sqrt(252)
    )
    prices["sigma"] = prices["sigma"].clip(lower=0.05, upper=2.0)

    # Drop rows without sigma
    prices = prices.dropna(subset=["sigma"]).reset_index(drop=True)

    all_tickers = sorted(prices["ticker"].unique())
    print(f"[load] {len(all_tickers)} tickers, {len(prices)} total rows, "
          f"date range {prices['date'].min().date()} to {prices['date'].max().date()}")

    return prices, ticker_meta, all_tickers


# ----------------------------- main -----------------------------------------
def main():
    t0 = time.time()
    prices, ticker_meta, all_tickers = load_all_data()

    results = []
    skipped = []
    n_total = len(all_tickers)

    print(f"\n[sweep] Running wheel on {n_total} tickers ...")

    for idx, tk in enumerate(all_tickers):
        if (idx + 1) % 20 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{n_total}] processing {tk} ... ({elapsed:.0f}s elapsed)")

        df_t = prices[prices["ticker"] == tk].sort_values("date")

        if len(df_t) < 252:
            skipped.append(tk)
            continue

        dates = df_t["date"].values.astype("datetime64[ns]")
        dates_ts = pd.DatetimeIndex(dates)
        closes = df_t["close"].values.astype(np.float64)
        sigmas = df_t["sigma"].values.astype(np.float64)
        vix_arr = df_t["vix"].values.astype(np.float64)

        m = run_wheel(dates_ts, closes, sigmas, vix_arr)
        if m is None:
            skipped.append(tk)
            continue

        meta = ticker_meta.get(tk, {})
        m["ticker"] = tk
        m["sector"] = meta.get("sector", "Unknown")
        m["beta"] = meta.get("beta", np.nan)
        m["market_cap"] = meta.get("market_cap", 0)
        results.append(m)

    elapsed = time.time() - t0
    print(f"\n[sweep] Completed {len(results)} tickers, skipped {len(skipped)}, "
          f"in {elapsed:.1f}s")
    if skipped:
        print(f"  Skipped: {', '.join(skipped[:20])}{'...' if len(skipped) > 20 else ''}")

    # ---- build results dataframe ----
    df = pd.DataFrame(results)
    df = df.sort_values("sharpe", ascending=False).reset_index(drop=True)

    # ---- Top 20 by Sharpe ----
    print("\n" + "=" * 80)
    print("TOP 20 BY SHARPE RATIO")
    print("=" * 80)
    top_sharpe = df.head(20)
    print(f"{'Rank':>4} {'Ticker':<8} {'Sector':<25} {'Sharpe':>7} {'AnnRet%':>8} "
          f"{'MaxDD%':>8} {'WR%':>6} {'Trades':>6} {'Prem%':>7} {'Beta':>5}")
    print("-" * 100)
    for i, row in top_sharpe.iterrows():
        beta_str = f"{row['beta']:.2f}" if np.isfinite(row['beta']) else "n/a"
        print(f"{i+1:>4} {row['ticker']:<8} {row['sector']:<25} {row['sharpe']:>7.2f} "
              f"{row['ann_return_pct']:>7.1f}% {row['max_dd_pct']:>7.1f}% "
              f"{row['win_rate']*100:>5.1f}% {row['n_trades']:>6} "
              f"{row['premium_income_pct']:>6.0f}% {beta_str:>5}")

    # ---- Top 20 by annualized return ----
    print("\n" + "=" * 80)
    print("TOP 20 BY ANNUALIZED RETURN")
    print("=" * 80)
    top_ret = df.sort_values("ann_return_pct", ascending=False).head(20)
    print(f"{'Rank':>4} {'Ticker':<8} {'Sector':<25} {'AnnRet%':>8} {'Sharpe':>7} "
          f"{'MaxDD%':>8} {'WR%':>6} {'Trades':>6}")
    print("-" * 80)
    for rank, (i, row) in enumerate(top_ret.iterrows()):
        print(f"{rank+1:>4} {row['ticker']:<8} {row['sector']:<25} "
              f"{row['ann_return_pct']:>7.1f}% {row['sharpe']:>7.2f} "
              f"{row['max_dd_pct']:>7.1f}% {row['win_rate']*100:>5.1f}% {row['n_trades']:>6}")

    # ---- Sector aggregation ----
    print("\n" + "=" * 80)
    print("SECTOR-LEVEL AGGREGATION")
    print("=" * 80)
    sector_agg = df.groupby("sector").agg(
        n_tickers=("ticker", "count"),
        avg_sharpe=("sharpe", "mean"),
        med_sharpe=("sharpe", "median"),
        avg_ann_ret=("ann_return_pct", "mean"),
        avg_max_dd=("max_dd_pct", "mean"),
        avg_wr=("win_rate", "mean"),
        avg_premium_pct=("premium_income_pct", "mean"),
    ).sort_values("avg_sharpe", ascending=False)

    print(f"{'Sector':<25} {'N':>3} {'AvgSharpe':>10} {'MedSharpe':>10} "
          f"{'AvgAnnRet%':>11} {'AvgMaxDD%':>10} {'AvgWR%':>7} {'AvgPrem%':>9}")
    print("-" * 95)
    for sector, row in sector_agg.iterrows():
        print(f"{sector:<25} {row['n_tickers']:>3} {row['avg_sharpe']:>10.2f} "
              f"{row['med_sharpe']:>10.2f} {row['avg_ann_ret']:>10.1f}% "
              f"{row['avg_max_dd']:>9.1f}% {row['avg_wr']*100:>6.1f}% "
              f"{row['avg_premium_pct']:>8.0f}%")

    # ---- Beta bucket analysis ----
    print("\n" + "=" * 80)
    print("BETA BUCKET ANALYSIS")
    print("=" * 80)
    df_with_beta = df[df["beta"].notna() & np.isfinite(df["beta"])].copy()

    if len(df_with_beta) > 0:
        bins = [0, 0.8, 1.2, 1.8, float("inf")]
        labels = ["Low (<0.8)", "Mid (0.8-1.2)", "High (1.2-1.8)", "Extreme (>1.8)"]
        df_with_beta["beta_bucket"] = pd.cut(df_with_beta["beta"], bins=bins, labels=labels, right=False)

        beta_agg = df_with_beta.groupby("beta_bucket", observed=True).agg(
            n_tickers=("ticker", "count"),
            avg_sharpe=("sharpe", "mean"),
            med_sharpe=("sharpe", "median"),
            avg_ann_ret=("ann_return_pct", "mean"),
            avg_max_dd=("max_dd_pct", "mean"),
            avg_wr=("win_rate", "mean"),
            avg_beta=("beta", "mean"),
        )

        print(f"{'Bucket':<20} {'N':>3} {'AvgBeta':>8} {'AvgSharpe':>10} {'MedSharpe':>10} "
              f"{'AvgAnnRet%':>11} {'AvgMaxDD%':>10} {'AvgWR%':>7}")
        print("-" * 90)
        for bucket, row in beta_agg.iterrows():
            print(f"{bucket:<20} {row['n_tickers']:>3} {row['avg_beta']:>8.2f} "
                  f"{row['avg_sharpe']:>10.2f} {row['med_sharpe']:>10.2f} "
                  f"{row['avg_ann_ret']:>10.1f}% {row['avg_max_dd']:>9.1f}% "
                  f"{row['avg_wr']*100:>6.1f}%")
    else:
        print("  No beta data available for bucketing.")

    # ---- Distribution summary ----
    print("\n" + "=" * 80)
    print("DISTRIBUTION SUMMARY (all tickers)")
    print("=" * 80)
    for col, label in [("sharpe", "Sharpe"), ("ann_return_pct", "Ann Return %"),
                       ("max_dd_pct", "Max DD %"), ("win_rate", "Win Rate"),
                       ("n_trades", "N Trades")]:
        vals = df[col].dropna()
        if col == "win_rate":
            vals = vals * 100
        print(f"  {label:<15}: mean={vals.mean():>8.2f}  median={vals.median():>8.2f}  "
              f"std={vals.std():>8.2f}  min={vals.min():>8.2f}  max={vals.max():>8.2f}")

    # Positive Sharpe count
    pos_sharpe = (df["sharpe"] > 0).sum()
    gt1_sharpe = (df["sharpe"] > 1.0).sum()
    print(f"\n  Positive Sharpe: {pos_sharpe}/{len(df)} ({pos_sharpe/len(df)*100:.0f}%)")
    print(f"  Sharpe > 1.0:   {gt1_sharpe}/{len(df)} ({gt1_sharpe/len(df)*100:.0f}%)")

    # ---- Bottom 10 ----
    print("\n" + "=" * 80)
    print("BOTTOM 10 BY SHARPE (worst performers)")
    print("=" * 80)
    bottom = df.tail(10).iloc[::-1]
    print(f"{'Ticker':<8} {'Sector':<25} {'Sharpe':>7} {'AnnRet%':>8} {'MaxDD%':>8}")
    print("-" * 60)
    for _, row in bottom.iterrows():
        print(f"{row['ticker']:<8} {row['sector']:<25} {row['sharpe']:>7.2f} "
              f"{row['ann_return_pct']:>7.1f}% {row['max_dd_pct']:>7.1f}%")

    # ---- Save results ----
    print(f"\n[save] Writing results to {OUT_DIR} ...")

    # Full results CSV
    df.to_csv(OUT_DIR / "all_tickers_results.csv", index=False)

    # Sector aggregation
    sector_agg.to_csv(OUT_DIR / "sector_aggregation.csv")

    # JSON summary
    summary = {
        "generated": datetime.now().isoformat(),
        "config": {
            "start_date": str(START_DATE.date()),
            "starting_cash": STARTING_CASH,
            "put_delta": PUT_DELTA,
            "call_delta": CALL_DELTA,
            "dte_target": DTE_TARGET,
            "profit_take": PROFIT_TAKE,
            "vix_gate": VIX_MAX,
            "cost_per_contract": COST_PER_CONTRACT,
            "iv_model": "rv_20 (20-day realized vol)",
        },
        "summary": {
            "n_tickers_run": len(results),
            "n_skipped": len(skipped),
            "skipped_tickers": skipped,
            "positive_sharpe_pct": float(pos_sharpe / len(df) * 100),
            "sharpe_gt1_pct": float(gt1_sharpe / len(df) * 100),
            "median_sharpe": float(df["sharpe"].median()),
            "median_ann_return": float(df["ann_return_pct"].median()),
            "median_max_dd": float(df["max_dd_pct"].median()),
        },
        "top10_sharpe": df.head(10)[["ticker", "sector", "sharpe", "ann_return_pct",
                                      "max_dd_pct", "win_rate", "n_trades"]].to_dict("records"),
        "sector_ranking": sector_agg[["n_tickers", "avg_sharpe", "med_sharpe",
                                       "avg_ann_ret"]].reset_index().to_dict("records"),
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n[done] Total runtime: {elapsed:.1f}s")
    print(f"[done] Results saved to {OUT_DIR}")

    return df


if __name__ == "__main__":
    main()
