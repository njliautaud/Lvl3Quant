#!/usr/bin/env python3
"""
Multi-Day Exhaustion Selling Strategy — v1
============================================

OBSERVATION:
  When a stock drops for 3+ consecutive days AND the cumulative drop exceeds
  a threshold, AND volume is increasing on the down days (panic selling),
  the exhaustion may predict a bounce.

HYPOTHESIS:
  Increasing volume on consecutive down days = capitulation. Smart money
  absorbs supply. Bounce follows once selling pressure exhausts.

SIGNAL VARIANTS (54 total):
  - Consecutive down days: 3, 4, 5
  - Cumulative drop threshold: 5%, 8%, 12%
  - Volume filter: ON (avg vol last 3d > 1.5x avg vol last 20d) or OFF
  - Hold periods: 5, 10, 21 days

VALIDATION (HC #428 + HC #432):
  - Regime analysis: SPY close vs 200 SMA → GREEN/RED
  - Regime gap test: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) ≤ 0.50
  - Permutation test: 1000 shuffles, p < 0.05
  - Year-by-year breakdown for passing variants
  - Volume filter comparison: does volume actually add value?

Usage:
    python3 research/strategies/iv_term_structure_v1.py

Author: Claude Opus 4.6 / Teleclaude Research
"""

import datetime as dt
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

COST_BPS_RT = 10  # 5 bps each way for equities
NUM_PERMUTATIONS = 1000
REGIME_GAP_THRESHOLD = 0.50
PERM_P_THRESHOLD = 0.05
MIN_TRADES = 30  # minimum trades to consider a variant valid
DATA_YEARS = 12  # years of data to download

# S&P 500 tickers via Wikipedia
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Output
OUTPUT_DIR = Path(os.environ.get("LVL3_ROOT", "/home/nick/Lvl3Quant")) / "output" / "exhaustion_selling_v1"

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL VARIANT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Build all 54 variants
VARIANTS = []
for consec_days in [3, 4, 5]:
    for drop_pct in [5, 8, 12]:
        for vol_filter in [False, True]:
            for hold_days in [5, 10, 21]:
                vf_tag = "VOL" if vol_filter else "NOVOL"
                name = f"CD{consec_days}_D{drop_pct}__{vf_tag}__H{hold_days}"
                VARIANTS.append({
                    "name": name,
                    "consec_days": consec_days,
                    "drop_pct": drop_pct / 100.0,
                    "vol_filter": vol_filter,
                    "hold_days": hold_days,
                })

print(f"Testing {len(VARIANTS)} variants")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html(SP500_URL)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"Got {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        print(f"Failed to get S&P 500 list from Wikipedia: {e}")
        # Fallback: top ~220 tickers by market cap
        return _fallback_tickers()


def _fallback_tickers():
    """Hardcoded fallback if Wikipedia scraping fails."""
    return [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "TSLA",
        "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
        "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
        "TMO", "ACN", "DHR", "ABT", "NEE", "LIN", "ADBE", "TXN", "PM",
        "CRM", "NKE", "RTX", "ORCL", "HON", "QCOM", "UPS", "LOW", "MS",
        "IBM", "GS", "CAT", "BA", "GE", "AMGN", "INTC", "SBUX", "BLK",
        "INTU", "AMD", "DE", "MDT", "GILD", "ADP", "ISRG", "BKNG", "SYK",
        "MDLZ", "VRTX", "ADI", "TJX", "PLD", "MMC", "CI", "CB", "PNC",
        "ZTS", "REGN", "SO", "DUK", "USB", "CME", "BDX", "MO", "CL",
        "EL", "MMM", "SLB", "WM", "FIS", "ICE", "AON", "NSC", "D",
        "APD", "EMR", "SHW", "FCX", "NOC", "GD", "CCI", "ITW", "PXD",
        "ATVI", "HUM", "KLAC", "SPG", "EQIX", "SNPS", "CDNS", "KMB",
        "AEP", "SRE", "FTNT", "ORLY", "TGT", "A", "CMG", "DXCM",
        "AIG", "AFL", "PSA", "DLR", "STZ", "MPC", "F", "GM", "DAL",
        "NFLX", "DIS", "PYPL", "COP", "EOG", "VLO", "OXY", "HES",
        "DVN", "APA", "MRO", "FANG", "HAL", "BKR", "PSX", "BIIB",
        "MRNA", "BMY", "PFE", "ZM", "DOCU", "ROKU", "SNAP", "PINS",
        "SQ", "SHOP", "SPOT", "UBER", "LYFT", "ABNB", "COIN", "RIVN",
        "LCID", "PLTR", "SOFI", "HOOD", "RBLX", "DKNG", "CRWD", "ZS",
        "NET", "SNOW", "MNDY", "BILL", "HUBS", "VEEV", "PANW", "OKTA",
        "TWLO", "MDB", "ESTC", "CFLT", "PATH", "U", "TEAM", "NOW",
        "WDAY", "SPLK", "DDOG", "FIVN", "COUP", "TT", "TRMB", "CARR",
        "OTIS", "IEX", "IR", "KEYS", "FTV", "ZBRA", "ROK", "TER",
        "WAB", "DOV", "NDSN", "AME", "XYL", "GGG", "RBC", "PNR",
    ]


def download_data(tickers, start_date, end_date):
    """Download daily OHLCV for all tickers via yfinance."""
    import yfinance as yf

    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    t0 = time.time()

    # Download in batches to avoid timeouts
    batch_size = 50
    all_data = {}
    failed = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"  Batch {i // batch_size + 1}/{(len(tickers) + batch_size - 1) // batch_size}: {batch[0]}..{batch[-1]}")
        try:
            data = yf.download(
                batch,
                start=start_date,
                end=end_date,
                group_by="ticker",
                auto_adjust=True,
                threads=True,
            )
            if len(batch) == 1:
                # yfinance returns single-level columns for single ticker
                ticker = batch[0]
                if not data.empty and len(data) > 100:
                    all_data[ticker] = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                for ticker in batch:
                    try:
                        df = data[ticker][["Open", "High", "Low", "Close", "Volume"]].copy()
                        df = df.dropna(subset=["Close"])
                        if len(df) > 100:
                            all_data[ticker] = df
                    except (KeyError, TypeError):
                        failed.append(ticker)
        except Exception as e:
            print(f"  Batch failed: {e}")
            failed.extend(batch)
        time.sleep(0.5)

    elapsed = time.time() - t0
    print(f"Downloaded {len(all_data)} tickers in {elapsed:.0f}s ({len(failed)} failed)")
    return all_data


def download_spy(start_date, end_date):
    """Download SPY for regime classification."""
    import yfinance as yf
    spy = yf.download("SPY", start=start_date, end=end_date, auto_adjust=True)
    # yfinance may return DataFrame with MultiIndex columns for single ticker
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
    # Ensure Close and other columns are Series, not DataFrames
    for col in spy.columns:
        if isinstance(spy[col], pd.DataFrame):
            spy[col] = spy[col].iloc[:, 0]
    spy["SMA200"] = spy["Close"].rolling(200).mean()
    spy["regime"] = np.where(spy["Close"] > spy["SMA200"], "GREEN", "RED")
    return spy[["Close", "SMA200", "regime"]].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def find_exhaustion_signals(df, consec_days, drop_pct, vol_filter):
    """
    Find exhaustion selling signals in a single stock's OHLCV data.

    Returns list of (entry_date_idx, entry_price) tuples.
    """
    close = df["Close"].values
    volume = df["Volume"].values
    dates = df.index

    # Daily returns
    daily_ret = np.diff(close) / close[:-1]
    # Prepend NaN so indices align
    daily_ret = np.concatenate([[np.nan], daily_ret])

    signals = []

    for i in range(consec_days + 20, len(close)):
        # Check consecutive down days ending at day i
        all_down = True
        for d in range(consec_days):
            idx = i - (consec_days - 1 - d)
            if np.isnan(daily_ret[idx]) or daily_ret[idx] >= 0:
                all_down = False
                break
        if not all_down:
            continue

        # Check cumulative drop
        start_price = close[i - consec_days]
        end_price = close[i]
        cum_drop = (start_price - end_price) / start_price
        if cum_drop < drop_pct:
            continue

        # Volume filter
        if vol_filter:
            avg_vol_3d = np.mean(volume[i - 2:i + 1])
            avg_vol_20d = np.mean(volume[i - 20:i])
            if avg_vol_20d <= 0 or avg_vol_3d / avg_vol_20d < 1.5:
                continue

        # Entry at next day's open (i+1)
        if i + 1 < len(close):
            signals.append((i + 1, dates[i + 1]))

    return signals


def compute_trades(df, signals, hold_days, cost_bps_rt=COST_BPS_RT):
    """
    Compute trade returns for a list of signals.
    Entry: next day open. Exit: hold_days later close.
    Returns DataFrame of trades.
    """
    close = df["Close"].values
    open_prices = df["Open"].values
    dates = df.index

    trades = []
    for entry_idx, entry_date in signals:
        if entry_idx >= len(close) or entry_idx + hold_days >= len(close):
            continue

        entry_price = open_prices[entry_idx]
        exit_price = close[entry_idx + hold_days]

        if entry_price <= 0:
            continue

        gross_ret = (exit_price - entry_price) / entry_price
        cost = cost_bps_rt / 10000.0
        net_ret = gross_ret - cost

        trades.append({
            "entry_date": entry_date,
            "exit_date": dates[entry_idx + hold_days],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
        })

    return pd.DataFrame(trades)


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(trades_df, hold_days):
    """Compute strategy statistics from trades DataFrame."""
    if trades_df.empty or len(trades_df) < MIN_TRADES:
        return None

    rets = trades_df["net_ret"].values
    n = len(rets)
    wins = np.sum(rets > 0)
    wr = wins / n

    mean_ret = np.mean(rets)
    median_ret = np.median(rets)

    # Annualized Sharpe (assuming hold_days per trade)
    trades_per_year = 252 / hold_days
    if np.std(rets) > 0:
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Profit factor
    gross_wins = np.sum(rets[rets > 0])
    gross_losses = abs(np.sum(rets[rets < 0]))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside) if len(downside) > 1 else 1e-9
    sortino = (np.mean(rets) / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0.0

    return {
        "n_trades": int(n),
        "win_rate": round(wr, 4),
        "mean_ret": round(mean_ret, 6),
        "median_ret": round(median_ret, 6),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(min(pf, 99.0), 3),
    }


def regime_analysis(trades_df, spy_regime, hold_days):
    """
    Split trades into GREEN/RED regimes based on SPY vs 200 SMA.
    Returns regime stats and gap test result.
    """
    if trades_df.empty:
        return None

    # Map entry dates to regime
    trades_df = trades_df.copy()
    trades_df["entry_date_dt"] = pd.to_datetime(trades_df["entry_date"])

    # Find nearest regime date for each entry
    regime_dates = spy_regime.index
    regimes = []
    for ed in trades_df["entry_date_dt"]:
        # Find closest date <= entry date
        mask = regime_dates <= ed
        if mask.any():
            closest = regime_dates[mask][-1]
            regimes.append(spy_regime.loc[closest, "regime"])
        else:
            regimes.append("UNKNOWN")

    trades_df["regime"] = regimes

    result = {}
    for regime in ["GREEN", "RED"]:
        subset = trades_df[trades_df["regime"] == regime]
        if len(subset) >= 10:
            stats = compute_stats(subset, hold_days)
            if stats:
                result[regime] = stats
            else:
                result[regime] = {"n_trades": len(subset), "sharpe": 0.0}
        else:
            result[regime] = {"n_trades": len(subset), "sharpe": 0.0}

    # Regime gap test
    sg = result.get("GREEN", {}).get("sharpe", 0.0)
    sr = result.get("RED", {}).get("sharpe", 0.0)
    denom = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / denom if denom > 0 else 0.0

    result["gap"] = round(gap, 4)
    result["gap_pass"] = gap <= REGIME_GAP_THRESHOLD

    return result


def permutation_test(trades_df, hold_days, n_perms=NUM_PERMUTATIONS):
    """
    Permutation test: shuffle returns, compute Sharpe, compare to actual.
    """
    if trades_df.empty or len(trades_df) < MIN_TRADES:
        return {"p_value": 1.0, "pass": False}

    rets = trades_df["net_ret"].values
    trades_per_year = 252 / hold_days
    actual_sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(trades_per_year) if np.std(rets) > 0 else 0.0

    rng = np.random.default_rng(42)
    count_better = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(rets)
        if np.std(shuffled) > 0:
            perm_sharpe = (np.mean(shuffled) / np.std(shuffled)) * np.sqrt(trades_per_year)
        else:
            perm_sharpe = 0.0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    p_val = count_better / n_perms
    return {
        "p_value": round(p_val, 4),
        "actual_sharpe": round(actual_sharpe, 3),
        "pass": p_val < PERM_P_THRESHOLD,
    }


def year_by_year(trades_df, hold_days):
    """Year-by-year breakdown of stats."""
    if trades_df.empty:
        return {}

    trades_df = trades_df.copy()
    trades_df["year"] = pd.to_datetime(trades_df["entry_date"]).dt.year

    result = {}
    for year, group in trades_df.groupby("year"):
        if len(group) >= 5:
            stats = compute_stats(group, hold_days)
            if stats:
                result[str(year)] = stats
            else:
                result[str(year)] = {"n_trades": len(group), "note": "too few for stats"}
        else:
            result[str(year)] = {"n_trades": len(group), "note": "too few"}

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    # Setup output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Date range
    end_date = dt.date.today().isoformat()
    start_date = (dt.date.today() - dt.timedelta(days=DATA_YEARS * 365)).isoformat()

    # Get tickers and download data
    tickers = get_sp500_tickers()
    all_data = download_data(tickers, start_date, end_date)
    spy_regime = download_spy(start_date, end_date)

    print(f"\nLoaded {len(all_data)} stocks, SPY regime data {spy_regime.index[0].date()} to {spy_regime.index[-1].date()}")
    print(f"Regime distribution: {spy_regime['regime'].value_counts().to_dict()}")
    print(f"\n{'='*80}")
    print(f"Running {len(VARIANTS)} variants across {len(all_data)} stocks...")
    print(f"{'='*80}\n")

    # ── Run all variants ──────────────────────────────────────────────────────
    results = {}
    passing_variants = []

    for vi, variant in enumerate(VARIANTS):
        vname = variant["name"]
        print(f"[{vi+1}/{len(VARIANTS)}] {vname} ... ", end="", flush=True)

        # Collect trades across all stocks
        all_trades = []
        for ticker, df in all_data.items():
            signals = find_exhaustion_signals(
                df,
                consec_days=variant["consec_days"],
                drop_pct=variant["drop_pct"],
                vol_filter=variant["vol_filter"],
            )
            if signals:
                trades = compute_trades(df, signals, variant["hold_days"])
                if not trades.empty:
                    trades["ticker"] = ticker
                    all_trades.append(trades)

        if not all_trades:
            print("0 trades")
            results[vname] = {"n_trades": 0, "status": "NO_TRADES"}
            continue

        trades_df = pd.concat(all_trades, ignore_index=True)

        # Overall stats
        stats = compute_stats(trades_df, variant["hold_days"])
        if stats is None:
            print(f"{len(trades_df)} trades (below minimum {MIN_TRADES})")
            results[vname] = {"n_trades": len(trades_df), "status": "TOO_FEW_TRADES"}
            continue

        # Regime analysis
        regime = regime_analysis(trades_df, spy_regime, variant["hold_days"])

        # Permutation test
        perm = permutation_test(trades_df, variant["hold_days"])

        # Determine pass/fail
        passes_regime = regime["gap_pass"] if regime else False
        passes_perm = perm["pass"]
        passes_all = passes_regime and passes_perm

        status_str = "PASS" if passes_all else "FAIL"
        reason = []
        if not passes_regime:
            reason.append(f"regime_gap={regime['gap'] if regime else 'N/A'}")
        if not passes_perm:
            reason.append(f"perm_p={perm['p_value']}")

        print(f"N={stats['n_trades']} WR={stats['win_rate']:.1%} Sharpe={stats['sharpe']:.2f} "
              f"PF={stats['profit_factor']:.2f} [{status_str}]"
              + (f" ({', '.join(reason)})" if reason else ""))

        entry = {
            "variant": variant,
            "stats": stats,
            "regime": regime,
            "permutation": perm,
            "status": status_str,
        }

        # Year-by-year for passing variants
        if passes_all:
            entry["year_by_year"] = year_by_year(trades_df, variant["hold_days"])
            passing_variants.append(entry)

        results[vname] = entry

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    print(f"\n{'='*80}")
    print(f"EXHAUSTION SELLING v1 — RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"Total variants tested: {len(VARIANTS)}")
    print(f"Passing variants (regime + perm): {len(passing_variants)}")
    print(f"Elapsed: {elapsed:.0f}s\n")

    if passing_variants:
        print("PASSING VARIANTS:")
        print(f"{'Name':<30} {'N':>5} {'WR':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
              f"{'gap':>6} {'p_val':>6}")
        print("-" * 80)
        for pv in sorted(passing_variants, key=lambda x: x["stats"]["sharpe"], reverse=True):
            s = pv["stats"]
            r = pv["regime"]
            p = pv["permutation"]
            print(f"{pv['variant']['name']:<30} {s['n_trades']:>5} {s['win_rate']:>6.1%} "
                  f"{s['sharpe']:>7.2f} {s['sortino']:>8.2f} {s['profit_factor']:>6.2f} "
                  f"{r['gap']:>6.3f} {p['p_value']:>6.3f}")

        # Volume filter comparison
        print(f"\n{'='*80}")
        print("VOLUME FILTER COMPARISON")
        print(f"{'='*80}")
        vol_pass = [p for p in passing_variants if p["variant"]["vol_filter"]]
        novol_pass = [p for p in passing_variants if not p["variant"]["vol_filter"]]
        print(f"  With volume filter: {len(vol_pass)} passing variants")
        print(f"  Without volume filter: {len(novol_pass)} passing variants")

        # Compare matched pairs (same consec/drop/hold, different vol_filter)
        print(f"\n  Matched pair comparison (avg Sharpe):")
        for consec in [3, 4, 5]:
            for drop in [5, 8, 12]:
                for hold in [5, 10, 21]:
                    vol_name = f"CD{consec}_D{drop}__VOL__H{hold}"
                    novol_name = f"CD{consec}_D{drop}__NOVOL__H{hold}"
                    vol_r = results.get(vol_name, {})
                    novol_r = results.get(novol_name, {})
                    vs = vol_r.get("stats", {}).get("sharpe", None) if isinstance(vol_r, dict) and "stats" in vol_r else None
                    ns = novol_r.get("stats", {}).get("sharpe", None) if isinstance(novol_r, dict) and "stats" in novol_r else None
                    if vs is not None and ns is not None:
                        better = "VOL" if vs > ns else "NOVOL"
                        print(f"    CD{consec} D{drop}% H{hold}: VOL={vs:.2f} vs NOVOL={ns:.2f} → {better} wins")
    else:
        print("NO VARIANTS PASSED both regime and permutation tests.")

    # ── Save results ──────────────────────────────────────────────────────────
    output_file = OUTPUT_DIR / "results.json"

    # Make results JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, dt.date, dt.datetime)):
            return str(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    output = {
        "strategy": "exhaustion_selling_v1",
        "description": "Multi-day exhaustion selling with volume capitulation filter",
        "run_date": dt.datetime.now().isoformat(),
        "universe": f"S&P 500 ({len(all_data)} stocks loaded)",
        "data_range": f"{start_date} to {end_date}",
        "total_variants": len(VARIANTS),
        "passing_variants": len(passing_variants),
        "elapsed_seconds": round(elapsed, 1),
        "results": make_serializable(results),
        "summary": {
            "best_variant": passing_variants[0]["variant"]["name"] if passing_variants else None,
            "best_sharpe": passing_variants[0]["stats"]["sharpe"] if passing_variants else None,
            "volume_filter_passing": len([p for p in passing_variants if p["variant"]["vol_filter"]]),
            "no_volume_filter_passing": len([p for p in passing_variants if not p["variant"]["vol_filter"]]),
        }
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print(f"Total elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
