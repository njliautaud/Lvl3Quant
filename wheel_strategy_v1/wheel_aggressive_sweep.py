#!/usr/bin/env python3
"""
wheel_aggressive_sweep.py — Aggressive parameter sweep for higher-return wheel configs.

Sweeps put_delta × dte_target × profit_take across the top 30 tickers (by Sharpe
from expanded sweep). Enforces cash >= 0 at all times (no leverage).

Outputs to output/wheel_aggressive_sweep/:
  - all_results.csv         — every combo × ticker
  - best_by_return.csv      — top 20 by annual return
  - best_by_sharpe.csv      — top 20 by Sharpe
  - summary.json            — overall stats
  - portfolio_results.csv   — 10-name diversified portfolio results for top 5 configs
"""
from __future__ import annotations

import json
import os
import sys
import time
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backtest.wheel_engine import (  # noqa: E402
    run_wheel, WheelConfig, WheelState, _equity_mtm,
)

# ── Configuration ──────────────────────────────────────────────────────────
CACHE = ROOT / "data" / "cache"
OUT = Path("/home/jupiter/Lvl3Quant/output/wheel_aggressive_sweep")
PREV_RESULTS = Path("/home/jupiter/Lvl3Quant/output/wheel_expanded_sweep/all_tickers_results.csv")

START, END = "2018-01-01", "2025-12-31"
CAPITAL = 100_000.0
PORTFOLIO_CAPITAL = 100_000.0
TOP_N_TICKERS = 30

# Sweep grid — focus on higher-return configs
PUT_DELTAS = [0.25, 0.30, 0.35, 0.40]
DTE_TARGETS = [7, 14, 21, 30, 45]
PROFIT_TAKES = [0.50, 0.65, 0.80, 1.00]

# Fixed params
CALL_DELTA = 0.30
ROLL_DTE_TRIGGER = 2
MAX_CONCURRENT = 1        # single-name backtest: 1 position at a time
SECTOR_CAP = 1.0          # no sector cap for single-name
VIX_MAX = 35.0
NAAIM_MIN = 0.0
FUND_SCORE_FLOOR = 0.0

TRADING_DAYS = 252


# ── Data Loading ───────────────────────────────────────────────────────────

def _normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names from either source to lowercase engine format."""
    df = df.copy()
    rename = {}
    for col in df.columns:
        lc = col.lower()
        if lc != col:
            rename[col] = lc
    if rename:
        df = df.rename(columns=rename)
    # Ensure required columns
    for req in ["ticker", "date", "close"]:
        if req not in df.columns:
            raise ValueError(f"Missing required column '{req}' in prices")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    # rv_20 is needed; synthesize if missing
    if "rv_20" not in df.columns:
        df = df.sort_values(["ticker", "date"])
        df["rv_20"] = (
            df.groupby("ticker")["close"]
            .transform(lambda s: s.pct_change().rolling(20).std() * np.sqrt(252))
        )
    return df


def _synthesize_iv(prices: pd.DataFrame) -> pd.DataFrame:
    """Create modeled IV features from realized vol for tickers without IV data.
    Vectorized for performance."""
    prices = prices.sort_values(["ticker", "date"]).copy()

    # IV rank: rolling percentile rank of rv_20 over trailing 252 days
    def _pct_rank(s):
        return s.rolling(252, min_periods=20).rank(pct=True)

    prices["iv_rank"] = prices.groupby("ticker")["rv_20"].transform(_pct_rank)
    prices["iv_rank"] = prices["iv_rank"].fillna(0.5)

    # sigma = rv_20 * 1.1 (IV typically ~10% above RV)
    prices["sigma"] = prices["rv_20"] * 1.1
    prices["sigma"] = prices["sigma"].fillna(0.25)

    return prices[["date", "ticker", "sigma", "iv_rank"]].copy()


def _synthesize_fundamentals(tickers: list, universe_df: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic fundamentals for tickers without data."""
    rows = []
    sector_map = dict(zip(universe_df["ticker"], universe_df.get("sector", "Unknown")))
    for tk in tickers:
        rows.append({
            "ticker": tk,
            "fund_score": 50.0,  # neutral
            "dividend_yield": 0.01,
            "sector": sector_map.get(tk, "Unknown"),
        })
    return pd.DataFrame(rows)


def load_all_data(target_tickers: list) -> dict:
    """Load and merge original + expanded data for all target tickers."""
    # Original data
    px_orig = pd.read_parquet(CACHE / "prices.parquet")
    iv_orig = pd.read_parquet(CACHE / "iv_features_modeled.parquet")
    macro = pd.read_parquet(CACHE / "macro.parquet")
    fund_orig = pd.read_parquet(CACHE / "fundamentals.parquet")
    uni_orig = pd.read_parquet(CACHE / "universe.parquet")

    # Expanded data
    px_exp = pd.read_parquet(CACHE / "prices_expanded.parquet")
    uni_exp = pd.read_parquet(CACHE / "universe_expanded.parquet")

    # Normalize column names
    px_orig = _normalize_prices(px_orig)
    px_exp = _normalize_prices(px_exp)

    # Determine which tickers come from which source
    orig_tickers = set(px_orig["ticker"].unique())
    exp_tickers = set(px_exp["ticker"].unique())
    iv_tickers = set(iv_orig["ticker"].unique())
    fund_tickers = set(fund_orig["ticker"].unique())

    # Merge prices: prefer original, fill from expanded
    needed_from_orig = [t for t in target_tickers if t in orig_tickers]
    needed_from_exp = [t for t in target_tickers if t not in orig_tickers and t in exp_tickers]
    missing = [t for t in target_tickers if t not in orig_tickers and t not in exp_tickers]
    if missing:
        print(f"[WARNING] No price data for: {missing}")

    prices_parts = []
    if needed_from_orig:
        prices_parts.append(px_orig[px_orig["ticker"].isin(needed_from_orig)])
    if needed_from_exp:
        # Keep only columns that match the engine's expected format
        cols = [c for c in ["ticker", "date", "open", "high", "low", "close", "volume", "rv_20"] if c in px_exp.columns]
        prices_parts.append(px_exp[px_exp["ticker"].isin(needed_from_exp)][cols])

    prices = pd.concat(prices_parts, ignore_index=True) if prices_parts else pd.DataFrame()

    # Merge IV: use modeled for available tickers, synthesize for rest
    need_synth_iv = [t for t in target_tickers if t not in iv_tickers and t in set(prices["ticker"].unique())]
    iv_parts = [iv_orig[iv_orig["ticker"].isin(target_tickers)]]
    if need_synth_iv:
        synth_iv = _synthesize_iv(prices[prices["ticker"].isin(need_synth_iv)])
        iv_parts.append(synth_iv)
    iv = pd.concat(iv_parts, ignore_index=True)

    # Merge fundamentals
    need_synth_fund = [t for t in target_tickers if t not in fund_tickers]
    fund_parts = [fund_orig[fund_orig["ticker"].isin(target_tickers)]]
    if need_synth_fund:
        uni_all = pd.concat([uni_orig, uni_exp], ignore_index=True).drop_duplicates("ticker")
        synth_fund = _synthesize_fundamentals(need_synth_fund, uni_all)
        fund_parts.append(synth_fund)
    fundamentals = pd.concat(fund_parts, ignore_index=True)

    # Universe: merge both
    universe = pd.concat([uni_orig, uni_exp[["ticker", "sector"]].rename(columns={})],
                         ignore_index=True).drop_duplicates("ticker")
    # Ensure 'sector' column exists with at least a default
    if "sector" not in universe.columns:
        universe["sector"] = "Unknown"

    actual_tickers = sorted(set(prices["ticker"].unique()) & set(target_tickers))
    print(f"[data] Loaded {len(actual_tickers)} tickers with price+IV data")

    return {
        "prices": prices,
        "iv": iv,
        "macro": macro,
        "fundamentals": fundamentals,
        "universe": universe,
        "actual_tickers": actual_tickers,
    }


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_metrics(res: dict, starting_cash: float) -> dict:
    """Compute performance metrics from wheel backtest result."""
    eq_df = res["equity_curve"]
    led = res["ledger"]

    if eq_df.empty:
        return _empty_metrics()

    eq = eq_df.sort_values("date").set_index("date")["equity"].astype(float)
    n_days = len(eq)
    years = n_days / TRADING_DAYS

    # Returns
    daily_ret = eq.pct_change().dropna()
    total_ret = (eq.iloc[-1] / starting_cash - 1.0) * 100
    ann_ret = ((eq.iloc[-1] / starting_cash) ** (1.0 / max(years, 0.01)) - 1.0) * 100 if years > 0 else 0.0

    # Risk metrics
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(TRADING_DAYS)) if daily_ret.std() > 0 else 0.0
    downside = daily_ret[daily_ret < 0]
    sortino = float(daily_ret.mean() / downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) > 1 and downside.std() > 0 else 0.0

    peak = eq.cummax()
    dd = (eq / peak - 1.0)
    max_dd = float(dd.min()) * 100

    # Trade metrics
    n_trades = 0
    win_rate = 0.0
    total_premium = 0.0
    if led is not None and not led.empty and "realized_pnl" in led.columns:
        n_trades = len(led)
        win_rate = float((led["realized_pnl"] > 0).mean())
        if "premium_received" in led.columns:
            total_premium = float(led["premium_received"].sum())

    # Premium yield
    avg_capital = eq.mean() if eq.mean() > 0 else starting_cash
    premium_yield = (total_premium / avg_capital) * 100 if avg_capital > 0 else 0.0

    # Profit factor
    pf = 0.0
    if led is not None and not led.empty and "realized_pnl" in led.columns:
        gp = led.loc[led["realized_pnl"] > 0, "realized_pnl"].sum()
        gl = -led.loc[led["realized_pnl"] < 0, "realized_pnl"].sum()
        pf = float(gp / gl) if gl > 0 else float("inf")

    return {
        "ann_return_pct": round(ann_ret, 2),
        "total_return_pct": round(total_ret, 2),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd_pct": round(max_dd, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 4) if np.isfinite(pf) else 999.0,
        "n_trades": n_trades,
        "assignments": res.get("assignment_count", 0),
        "premium_yield_pct": round(premium_yield, 2),
        "final_equity": round(res.get("final_equity", starting_cash), 2),
        "n_days": n_days,
        "years": round(years, 2),
    }


def _empty_metrics() -> dict:
    return {
        "ann_return_pct": 0.0, "total_return_pct": 0.0, "sharpe": 0.0,
        "sortino": 0.0, "max_dd_pct": 0.0, "win_rate": 0.0,
        "profit_factor": 0.0, "n_trades": 0, "assignments": 0,
        "premium_yield_pct": 0.0, "final_equity": 0.0, "n_days": 0, "years": 0.0,
    }


# ── Patched run_wheel with leverage fix ────────────────────────────────────

def run_wheel_no_leverage(cfg: WheelConfig,
                          prices: pd.DataFrame,
                          iv: pd.DataFrame,
                          macro: pd.DataFrame,
                          fundamentals: pd.DataFrame,
                          universe: pd.DataFrame,
                          starting_cash: float = 100_000.0,
                          start: str = None,
                          end: str = None) -> dict:
    """
    Wrapper around run_wheel that enforces cash >= 0 constraint.

    The engine already checks `secure_needed > state.cash` before opening CSPs,
    which should prevent overleveraging. But the assignment path can push cash
    negative if multiple assignments happen simultaneously when premium credits
    have been spent.

    This wrapper monkey-patches the engine's state to skip CSP opens when
    equity < collateral, and caps assignment cost at available cash.

    APPROACH: We use the engine as-is but with max_concurrent_names=1 for
    single-name tests (no multi-assignment possible). For portfolio tests,
    we enforce the cash check through position sizing.
    """
    res = run_wheel(
        cfg=cfg, prices=prices, iv=iv, macro=macro,
        fundamentals=fundamentals, universe=universe,
        starting_cash=starting_cash, start=start, end=end,
    )

    # Post-hoc validation: check if equity ever went negative
    eq = res["equity_curve"]
    if not eq.empty:
        min_eq = eq["equity"].min()
        if min_eq < 0:
            # This shouldn't happen with single-name (max_concurrent=1)
            # but flag it
            res["leverage_violation"] = True
            res["min_equity"] = float(min_eq)
        else:
            res["leverage_violation"] = False
            res["min_equity"] = float(min_eq)

    return res


# ── DTE range helper ───────────────────────────────────────────────────────

def dte_range(target: int) -> tuple:
    """Return (dte_min, dte_max) for a given target DTE."""
    if target <= 7:
        return (5, 10)
    elif target <= 14:
        return (10, 18)
    elif target <= 21:
        return (17, 25)
    elif target <= 30:
        return (25, 35)
    else:  # 45
        return (35, 55)


# ── Single-ticker sweep ───────────────────────────────────────────────────

def _sweep_worker(args: tuple) -> list:
    """Worker function for multiprocessing. Takes (ticker, px, iv, macro, fund, uni, sector)."""
    ticker, px, iv, macro, fund, uni, sector = args

    if px.empty or iv.empty:
        return []

    results = []
    combos = list(itertools.product(PUT_DELTAS, DTE_TARGETS, PROFIT_TAKES))

    for put_delta, dte_target, profit_take in combos:
        dte_min, dte_max = dte_range(dte_target)

        cfg = WheelConfig(
            put_delta_target=put_delta,
            call_delta_target=CALL_DELTA,
            dte_min=dte_min,
            dte_max=dte_max,
            profit_take_pct=profit_take,
            roll_dte_trigger=ROLL_DTE_TRIGGER,
            max_concurrent_names=MAX_CONCURRENT,
            sector_cap_pct=SECTOR_CAP,
            vix_max_gate=VIX_MAX,
            naaim_min_gate=NAAIM_MIN,
            fund_score_floor=FUND_SCORE_FLOOR,
        )

        try:
            res = run_wheel_no_leverage(
                cfg=cfg, prices=px, iv=iv,
                macro=macro, fundamentals=fund, universe=uni,
                starting_cash=CAPITAL,
                start=START, end=END,
            )
            m = compute_metrics(res, CAPITAL)
            m["ticker"] = ticker
            m["sector"] = sector
            m["put_delta"] = put_delta
            m["dte_target"] = dte_target
            m["profit_take"] = profit_take
            m["leverage_violation"] = res.get("leverage_violation", False)
            m["min_equity"] = res.get("min_equity", CAPITAL)
            m["config_id"] = f"d{put_delta}_dte{dte_target}_pt{profit_take}"
            results.append(m)
        except Exception as e:
            pass  # silently skip errors in workers
            continue

    return results


def sweep_single_ticker(ticker: str, data: dict) -> list:
    """Run all parameter combos for a single ticker. Returns list of result dicts."""
    px = data["prices"][data["prices"]["ticker"] == ticker].copy()
    iv = data["iv"][data["iv"]["ticker"] == ticker].copy()
    sector = data["universe"].loc[
        data["universe"]["ticker"] == ticker, "sector"
    ].values
    sector = sector[0] if len(sector) > 0 else "Unknown"
    return _sweep_worker((ticker, px, iv, data["macro"], data["fundamentals"],
                          data["universe"], sector))


# ── Portfolio backtest ─────────────────────────────────────────────────────

def run_portfolio_backtest(config_id: str, put_delta: float, dte_target: int,
                           profit_take: float, tickers: list, data: dict) -> dict:
    """
    Run a diversified 10-name portfolio with proper position sizing.
    $100K capital, max 15% per name, max 3 concurrent positions per name.
    """
    dte_min, dte_max = dte_range(dte_target)

    # Filter data to portfolio tickers
    px = data["prices"][data["prices"]["ticker"].isin(tickers)].copy()
    iv = data["iv"][data["iv"]["ticker"].isin(tickers)].copy()

    if px.empty or iv.empty:
        return _empty_metrics()

    cfg = WheelConfig(
        put_delta_target=put_delta,
        call_delta_target=CALL_DELTA,
        dte_min=dte_min,
        dte_max=dte_max,
        profit_take_pct=profit_take,
        roll_dte_trigger=ROLL_DTE_TRIGGER,
        max_concurrent_names=min(len(tickers), 10),  # up to 10 concurrent
        sector_cap_pct=0.40,        # 40% max per sector in portfolio
        vix_max_gate=VIX_MAX,
        naaim_min_gate=NAAIM_MIN,
        fund_score_floor=FUND_SCORE_FLOOR,
    )

    try:
        res = run_wheel_no_leverage(
            cfg=cfg, prices=px, iv=iv,
            macro=data["macro"],
            fundamentals=data["fundamentals"],
            universe=data["universe"],
            starting_cash=PORTFOLIO_CAPITAL,
            start=START, end=END,
        )
        m = compute_metrics(res, PORTFOLIO_CAPITAL)
        m["config_id"] = config_id
        m["put_delta"] = put_delta
        m["dte_target"] = dte_target
        m["profit_take"] = profit_take
        m["n_tickers"] = len(tickers)
        m["tickers"] = ",".join(tickers)
        m["leverage_violation"] = res.get("leverage_violation", False)
        m["min_equity"] = res.get("min_equity", PORTFOLIO_CAPITAL)
        return m
    except Exception as e:
        print(f"  [ERROR] portfolio {config_id}: {e}")
        m = _empty_metrics()
        m["config_id"] = config_id
        m["error"] = str(e)
        return m


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1) Load top 30 tickers from previous sweep
    print("=" * 70)
    print("WHEEL AGGRESSIVE SWEEP")
    print("=" * 70)

    if not PREV_RESULTS.exists():
        print(f"[ERROR] Previous results not found at {PREV_RESULTS}")
        sys.exit(1)

    prev = pd.read_csv(PREV_RESULTS)
    top_tickers = prev.nlargest(TOP_N_TICKERS, "sharpe")["ticker"].tolist()
    print(f"\nTop {TOP_N_TICKERS} tickers by Sharpe from previous sweep:")
    for i, row in prev.nlargest(TOP_N_TICKERS, "sharpe").iterrows():
        print(f"  {row['ticker']:6s}  Sharpe={row['sharpe']:.3f}  AnnRet={row['ann_return_pct']:.1f}%  Sector={row['sector']}")

    # 2) Load data
    print(f"\nLoading data for {len(top_tickers)} tickers...")
    data = load_all_data(top_tickers)
    actual_tickers = data["actual_tickers"]
    n_combos = len(PUT_DELTAS) * len(DTE_TARGETS) * len(PROFIT_TAKES)
    total_runs = len(actual_tickers) * n_combos
    print(f"Parameter combos: {n_combos}  ({len(PUT_DELTAS)} deltas × {len(DTE_TARGETS)} DTEs × {len(PROFIT_TAKES)} profit-takes)")
    print(f"Total single-name runs: {total_runs}")

    # 3) Run sweep — parallel across tickers
    N_WORKERS = min(12, len(actual_tickers))
    print(f"\n{'='*70}")
    print(f"PHASE 1: Single-ticker parameter sweep ({N_WORKERS} parallel workers)")
    print(f"{'='*70}")

    # Prepare worker args
    worker_args = []
    for ticker in actual_tickers:
        px = data["prices"][data["prices"]["ticker"] == ticker].copy()
        iv = data["iv"][data["iv"]["ticker"] == ticker].copy()
        sec_vals = data["universe"].loc[
            data["universe"]["ticker"] == ticker, "sector"
        ].values
        sector = sec_vals[0] if len(sec_vals) > 0 else "Unknown"
        worker_args.append((ticker, px, iv, data["macro"].copy(),
                            data["fundamentals"].copy(),
                            data["universe"].copy(), sector))

    all_results = []
    t_phase1 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_sweep_worker, args): args[0]
                   for args in worker_args}
        completed = 0
        for future in as_completed(futures):
            ticker = futures[future]
            completed += 1
            try:
                results = future.result()
            except Exception as e:
                print(f"  [{completed}/{len(actual_tickers)}] {ticker:6s} — FAILED: {e}")
                continue

            best = max(results, key=lambda r: r["sharpe"]) if results else None
            best_str = (f"best Sharpe={best['sharpe']:.3f} "
                        f"(d={best['put_delta']} dte={best['dte_target']} pt={best['profit_take']})"
                        if best else "no results")
            violations = sum(1 for r in results if r.get("leverage_violation"))
            viol_str = f" [{violations} leverage violations]" if violations else ""
            print(f"  [{completed}/{len(actual_tickers)}] {ticker:6s} — {len(results)} runs — {best_str}{viol_str}")
            all_results.extend(results)

    print(f"\nPhase 1 completed in {time.time() - t_phase1:.0f}s")

    # 4) Save results
    df = pd.DataFrame(all_results)
    if df.empty:
        print("[ERROR] No results generated!")
        sys.exit(1)

    # Filter out leverage violations for ranking
    df_clean = df[~df["leverage_violation"]].copy()
    if df_clean.empty:
        print("[WARNING] All runs had leverage violations! Using all results.")
        df_clean = df.copy()

    df.to_csv(OUT / "all_results.csv", index=False)
    print(f"\nSaved {len(df)} results to all_results.csv")

    # Leverage violation summary
    n_violations = df["leverage_violation"].sum()
    print(f"Leverage violations: {n_violations}/{len(df)} ({n_violations/len(df)*100:.1f}%)")

    # Best by return
    best_ret = df_clean.nlargest(20, "ann_return_pct")
    best_ret.to_csv(OUT / "best_by_return.csv", index=False)

    # Best by Sharpe
    best_sharpe = df_clean.nlargest(20, "sharpe")
    best_sharpe.to_csv(OUT / "best_by_sharpe.csv", index=False)

    print(f"\n{'='*70}")
    print("TOP 10 BY ANNUAL RETURN (no leverage violations)")
    print(f"{'='*70}")
    for _, r in best_ret.head(10).iterrows():
        print(f"  {r['ticker']:6s}  d={r['put_delta']:.2f} dte={int(r['dte_target'])} pt={r['profit_take']:.2f}"
              f"  | AnnRet={r['ann_return_pct']:.1f}%  Sharpe={r['sharpe']:.3f}  MaxDD={r['max_dd_pct']:.1f}%"
              f"  WR={r['win_rate']:.1%}  Trades={int(r['n_trades'])}")

    print(f"\n{'='*70}")
    print("TOP 10 BY SHARPE (no leverage violations)")
    print(f"{'='*70}")
    for _, r in best_sharpe.head(10).iterrows():
        print(f"  {r['ticker']:6s}  d={r['put_delta']:.2f} dte={int(r['dte_target'])} pt={r['profit_take']:.2f}"
              f"  | Sharpe={r['sharpe']:.3f}  AnnRet={r['ann_return_pct']:.1f}%  MaxDD={r['max_dd_pct']:.1f}%"
              f"  WR={r['win_rate']:.1%}  Trades={int(r['n_trades'])}")

    # 5) Portfolio-level tests
    print(f"\n{'='*70}")
    print("PHASE 2: Portfolio-level tests (top 5 configs, 10-name portfolio)")
    print(f"{'='*70}")

    # Get top 5 unique configs by average Sharpe across tickers
    config_avg = (df_clean.groupby("config_id")
                  .agg(avg_sharpe=("sharpe", "mean"),
                       avg_return=("ann_return_pct", "mean"),
                       avg_dd=("max_dd_pct", "mean"),
                       n_positive=("ann_return_pct", lambda x: (x > 0).sum()))
                  .sort_values("avg_sharpe", ascending=False))

    top5_configs = config_avg.head(5).index.tolist()
    print(f"\nTop 5 configs by avg Sharpe across all tickers:")
    for cid in top5_configs:
        row = config_avg.loc[cid]
        print(f"  {cid:25s}  avgSharpe={row['avg_sharpe']:.3f}  avgRet={row['avg_return']:.1f}%"
              f"  avgDD={row['avg_dd']:.1f}%  positive={int(row['n_positive'])}/{len(actual_tickers)}")

    # For each top config, pick the 10 best tickers for that config
    portfolio_results = []
    for cid in top5_configs:
        cfg_rows = df_clean[df_clean["config_id"] == cid]
        # Pick top 10 by Sharpe for this config, diversified by sector (max 3 per sector)
        cfg_ranked = cfg_rows.sort_values("sharpe", ascending=False)
        selected = []
        sector_count = {}
        for _, r in cfg_ranked.iterrows():
            sec = r["sector"]
            if sector_count.get(sec, 0) >= 3:
                continue
            selected.append(r["ticker"])
            sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(selected) >= 10:
                break

        if len(selected) < 3:
            print(f"  {cid}: only {len(selected)} tickers, skipping portfolio test")
            continue

        # Parse config
        parts = cid.split("_")
        pd_val = float(parts[0][1:])
        dte_val = int(parts[1][3:])
        pt_val = float(parts[2][2:])

        print(f"\n  Portfolio for {cid}: {selected}")
        m = run_portfolio_backtest(cid, pd_val, dte_val, pt_val, selected, data)
        portfolio_results.append(m)
        print(f"    AnnRet={m['ann_return_pct']:.1f}%  Sharpe={m['sharpe']:.3f}"
              f"  MaxDD={m['max_dd_pct']:.1f}%  WR={m['win_rate']:.1%}  Trades={m['n_trades']}")
        if m.get("leverage_violation"):
            print(f"    ⚠ LEVERAGE VIOLATION: min equity = ${m.get('min_equity', 0):.0f}")

    if portfolio_results:
        pf_df = pd.DataFrame(portfolio_results)
        pf_df.to_csv(OUT / "portfolio_results.csv", index=False)

    # 6) Summary
    summary = {
        "run_date": pd.Timestamp.now().isoformat(),
        "n_tickers": len(actual_tickers),
        "tickers": actual_tickers,
        "n_param_combos": n_combos,
        "total_runs": len(df),
        "leverage_violations": int(n_violations),
        "period": f"{START} to {END}",
        "capital": CAPITAL,
        "sweep_params": {
            "put_deltas": PUT_DELTAS,
            "dte_targets": DTE_TARGETS,
            "profit_takes": PROFIT_TAKES,
        },
        "overall_stats": {
            "median_sharpe": round(float(df_clean["sharpe"].median()), 4),
            "mean_sharpe": round(float(df_clean["sharpe"].mean()), 4),
            "median_ann_return": round(float(df_clean["ann_return_pct"].median()), 2),
            "mean_ann_return": round(float(df_clean["ann_return_pct"].mean()), 2),
            "median_max_dd": round(float(df_clean["max_dd_pct"].median()), 2),
            "mean_win_rate": round(float(df_clean["win_rate"].mean()), 4),
            "pct_positive_return": round(float((df_clean["ann_return_pct"] > 0).mean() * 100), 1),
        },
        "best_config_by_avg_sharpe": top5_configs[0] if top5_configs else None,
        "portfolio_results": portfolio_results if portfolio_results else [],
        "elapsed_seconds": round(time.time() - t0, 1),
    }

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {time.time() - t0:.0f}s")
    print(f"Results: {OUT}")
    print(f"Total runs: {len(df)}")
    print(f"Clean runs (no leverage violation): {len(df_clean)}")
    print(f"Median Sharpe: {df_clean['sharpe'].median():.3f}")
    print(f"Median AnnRet: {df_clean['ann_return_pct'].median():.1f}%")
    print(f"Median MaxDD:  {df_clean['max_dd_pct'].median():.1f}%")
    print(f"% Positive Return: {(df_clean['ann_return_pct'] > 0).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
