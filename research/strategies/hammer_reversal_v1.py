#!/usr/bin/env python3
"""
Hammer / Rejection Candle Reversal Strategy — v1
==================================================

OBSERVATION:
  When a stock drops significantly intraday (low is 3%+ below open) but CLOSES
  near the high of the day (close > 75th percentile of the day's range), this
  "rejection of lows" suggests buyers stepped in hard. The intraday recovery
  signals demand at lower prices.

HYPOTHESIS:
  This hammer/rejection pattern predicts next-day/next-week continuation higher.
  Combined with vol compression, this could be a high-conviction mean reversion
  signal — consistent with our finding that mean reversion is regime-agnostic
  while momentum is not.

SIGNAL:
  intraday_drop = (low - open) / open          → how far it fell intraday
  recovery      = (close - low) / (high - low)  → where in range it closed

  Entry: intraday_drop < -threshold AND recovery > recovery_threshold

VARIANTS (18 base + vol-compression overlays):
  Drop thresholds: 3%, 5%, 7%
  Recovery thresholds: 0.60, 0.75
  Hold periods: 5, 10, 21 days
  Best 2-3 non-VC variants also tested with vol_pctile < 15

VALIDATION (HC #428 + HC #432):
  - Regime analysis: SPY close vs 200 SMA → GREEN/RED
  - Regime gap test: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) ≤ 0.50
  - Permutation test: 1000 shuffles, p < 0.05
  - Year-by-year breakdown

Usage:
    python3 research/strategies/hammer_reversal_v1.py

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
LOOKBACK_VOL_DAYS = 20  # realized vol window
VOL_HISTORY_DAYS = 252  # percentile ranking lookback
DATA_YEARS = 12

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

OUTPUT_DIR = Path(os.environ.get("LVL3_ROOT", "/home/nick/Lvl3Quant")) / "output" / "hammer_reversal_v1"

# ─────────────────────────────────────────────────────────────────────────────
# VARIANT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

# (name, drop_pct, recovery_thresh, hold_days, vol_pctile_thresh)
# vol_pctile_thresh=None means no vol filter
BASE_VARIANTS = []
for drop_pct in [3, 5, 7]:
    for rec_thresh in [0.60, 0.75]:
        for hold in [5, 10, 21]:
            name = f"D{drop_pct}_R{int(rec_thresh*100)}_H{hold}"
            BASE_VARIANTS.append((name, drop_pct, rec_thresh, hold, None))

# Vol-compression overlay variants will be added dynamically for top performers

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html(SP500_URL)
        df = tables[0]
        tickers = df["Symbol"].tolist()
        # Clean tickers: BRK.B -> BRK-B for yfinance
        tickers = [t.replace(".", "-") for t in tickers]
        print(f"[INFO] Got {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        print(f"[WARN] Wikipedia fetch failed ({e}), using fallback list")
        # Top 100 by market cap as fallback
        return [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B",
            "UNH", "LLY", "JPM", "XOM", "JNJ", "V", "PG", "MA", "AVGO", "HD",
            "MRK", "COST", "ABBV", "CVX", "PEP", "KO", "ADBE", "WMT", "CRM",
            "MCD", "CSCO", "ACN", "TMO", "BAC", "ABT", "NFLX", "LIN", "AMD",
            "DHR", "ORCL", "CMCSA", "TXN", "PM", "WFC", "NEE", "DIS", "INTC",
            "RTX", "HON", "UPS", "QCOM", "INTU", "LOW", "AMGN", "SPGI", "CAT",
            "GS", "BA", "PFE", "ISRG", "BLK", "DE", "T", "ELV", "AXP", "BKNG",
            "SYK", "GILD", "MDLZ", "ADI", "VRTX", "MMC", "LRCX", "TMUS", "ADP",
            "REGN", "ETN", "CI", "SCHW", "NOW", "BSX", "CB", "MU", "ZTS", "SLB",
            "FI", "SO", "PGR", "PANW", "BDX", "DUK", "SNPS", "CME", "ITW",
            "CL", "AON", "KLAC", "ICE", "SHW", "MCO", "EQIX", "MO", "CDNS",
        ]


def download_data(tickers, start_date, end_date):
    """Download OHLCV data for all tickers via yfinance."""
    import yfinance as yf

    print(f"[INFO] Downloading data for {len(tickers)} tickers from {start_date} to {end_date}")
    all_data = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        ticker_str = " ".join(batch)
        try:
            data = yf.download(ticker_str, start=start_date, end=end_date,
                               group_by="ticker", progress=False, threads=True)
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        df = data.copy()
                    else:
                        df = data[ticker].copy()
                    df = df.dropna(subset=["Open", "High", "Low", "Close"])
                    if len(df) > 100:
                        all_data[ticker] = df
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARN] Batch download failed: {e}")
        if i > 0 and i % 200 == 0:
            print(f"  ... downloaded {i}/{len(tickers)} tickers ({len(all_data)} valid)")
            time.sleep(1)

    print(f"[INFO] Downloaded {len(all_data)} valid tickers")
    return all_data


def download_spy(start_date, end_date):
    """Download SPY for regime classification."""
    import yfinance as yf
    spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
    # Flatten MultiIndex columns if present
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    close = spy["Close"].astype(float)
    spy["SMA200"] = close.rolling(200).mean()
    spy["regime"] = np.where(close > spy["SMA200"], "GREEN", "RED")
    return spy[["Close", "SMA200", "regime"]].dropna()


def compute_vol_percentile(close_series):
    """Compute rolling realized vol percentile."""
    ret = close_series.pct_change()
    vol = ret.rolling(LOOKBACK_VOL_DAYS).std() * np.sqrt(252)
    vol_pctile = vol.rolling(VOL_HISTORY_DAYS).apply(
        lambda x: (x.iloc[-1] <= x).sum() / len(x) * 100 if len(x) >= 60 else np.nan,
        raw=False,
    )
    return vol_pctile


def compute_signals(df):
    """Compute hammer/rejection candle signals for a single stock.

    Returns DataFrame with columns: intraday_drop, recovery, vol_pctile
    """
    out = pd.DataFrame(index=df.index)

    # Flatten MultiIndex columns if present (yfinance sometimes returns multi-level)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)

    # Intraday drop: how far low is below open
    out["intraday_drop"] = (l - o) / o  # negative = dropped

    # Recovery: where in day's range did it close? 1.0 = closed at high
    day_range = h - l
    out["recovery"] = np.where(day_range > 0, (c - l) / day_range, 0.5)

    # Vol percentile
    out["vol_pctile"] = compute_vol_percentile(c)

    # Forward returns for various hold periods
    for hold in [5, 10, 21]:
        out[f"fwd_ret_{hold}d"] = c.pct_change(hold).shift(-hold)

    return out


def compute_variant_stats(trades_returns, cost_bps=COST_BPS_RT):
    """Compute stats for a set of trade returns."""
    if len(trades_returns) < MIN_TRADES:
        return None

    cost = cost_bps / 10000
    net_returns = trades_returns - cost

    n = len(net_returns)
    wr = (net_returns > 0).mean()
    mean_ret = net_returns.mean()
    median_ret = float(np.median(net_returns))

    # Annualized Sharpe (approximate — assume ~252/hold_days trades per year per stock)
    if net_returns.std() > 0:
        sharpe = mean_ret / net_returns.std() * np.sqrt(52)  # rough annualization
    else:
        sharpe = 0.0

    # Profit factor
    gross_profit = net_returns[net_returns > 0].sum()
    gross_loss = abs(net_returns[net_returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "n_trades": n,
        "win_rate": round(wr, 4),
        "mean_return": round(mean_ret, 6),
        "median_return": round(median_ret, 6),
        "sharpe": round(sharpe, 4),
        "profit_factor": round(pf, 4),
        "total_return": round(net_returns.sum(), 6),
    }


def regime_analysis(trade_dates, trade_returns, spy_regime, cost_bps=COST_BPS_RT):
    """Split trades by regime and compute regime gap."""
    cost = cost_bps / 10000
    net_returns = trade_returns - cost

    # Align dates
    regime_series = spy_regime["regime"]
    green_mask = []
    red_mask = []
    for d in trade_dates:
        # Find nearest regime date
        idx = regime_series.index.get_indexer([d], method="ffill")[0]
        if idx >= 0:
            r = regime_series.iloc[idx]
            green_mask.append(r == "GREEN")
            red_mask.append(r == "RED")
        else:
            green_mask.append(False)
            red_mask.append(False)

    green_mask = np.array(green_mask)
    red_mask = np.array(red_mask)

    green_rets = net_returns[green_mask]
    red_rets = net_returns[red_mask]

    result = {
        "n_green": int(green_mask.sum()),
        "n_red": int(red_mask.sum()),
    }

    if len(green_rets) >= 10 and green_rets.std() > 0:
        sharpe_green = green_rets.mean() / green_rets.std() * np.sqrt(52)
    else:
        sharpe_green = 0.0

    if len(red_rets) >= 10 and red_rets.std() > 0:
        sharpe_red = red_rets.mean() / red_rets.std() * np.sqrt(52)
    else:
        sharpe_red = 0.0

    result["sharpe_green"] = round(sharpe_green, 4)
    result["sharpe_red"] = round(sharpe_red, 4)

    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    if max_sharpe > 0:
        regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe
    else:
        regime_gap = 0.0

    result["regime_gap"] = round(regime_gap, 4)
    result["regime_pass"] = regime_gap <= REGIME_GAP_THRESHOLD

    return result


def permutation_test(trade_returns, n_perms=NUM_PERMUTATIONS, cost_bps=COST_BPS_RT):
    """Permutation test: shuffle assignment of returns, compute p-value."""
    cost = cost_bps / 10000
    net_returns = trade_returns - cost
    observed_mean = net_returns.mean()

    count_greater = 0
    all_values = np.array(net_returns).copy()
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        rng.shuffle(all_values)
        # Use same number of trades, shuffled
        perm_mean = all_values[:len(net_returns)].mean()
        if perm_mean >= observed_mean:
            count_greater += 1

    p_value = count_greater / n_perms
    return {
        "p_value": round(p_value, 4),
        "perm_pass": p_value < PERM_P_THRESHOLD,
        "observed_mean": round(observed_mean, 6),
    }


def year_by_year(trade_dates, trade_returns, cost_bps=COST_BPS_RT):
    """Year-by-year breakdown."""
    cost = cost_bps / 10000
    net_returns = trade_returns - cost

    df = pd.DataFrame({"date": trade_dates, "ret": net_returns})
    df["year"] = pd.to_datetime(df["date"]).dt.year

    results = {}
    for year, group in df.groupby("year"):
        if len(group) < 3:
            continue
        wr = (group["ret"] > 0).mean()
        mean_r = group["ret"].mean()
        results[int(year)] = {
            "n_trades": len(group),
            "win_rate": round(wr, 4),
            "mean_return": round(mean_r, 6),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HAMMER / REJECTION CANDLE REVERSAL STRATEGY — v1")
    print("=" * 70)
    print(f"Start time: {dt.datetime.now()}")
    print()

    # --- 1. Get tickers & download data ---
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=365 * DATA_YEARS + 200)  # extra for SMA warmup

    tickers = get_sp500_tickers()
    all_data = download_data(tickers, start_date.isoformat(), end_date.isoformat())
    spy_regime = download_spy(start_date.isoformat(), end_date.isoformat())

    print(f"\n[INFO] Computing signals for {len(all_data)} stocks...")

    # --- 2. Compute signals for all stocks ---
    all_signals = {}
    for ticker, df in all_data.items():
        try:
            sig = compute_signals(df)
            all_signals[ticker] = sig
        except Exception as e:
            pass

    print(f"[INFO] Signals computed for {len(all_signals)} stocks")

    # --- 3. Run base variants ---
    print(f"\n[INFO] Running {len(BASE_VARIANTS)} base variants...")
    results = {}

    for name, drop_pct, rec_thresh, hold, vol_thresh in BASE_VARIANTS:
        drop_threshold = -drop_pct / 100  # e.g., -0.03

        trade_returns = []
        trade_dates = []

        for ticker, sig in all_signals.items():
            # Entry conditions
            mask = (sig["intraday_drop"] <= drop_threshold) & (sig["recovery"] >= rec_thresh)

            if vol_thresh is not None:
                mask = mask & (sig["vol_pctile"] <= vol_thresh)

            ret_col = f"fwd_ret_{hold}d"
            valid = sig.loc[mask & sig[ret_col].notna(), ret_col]

            if len(valid) > 0:
                trade_returns.extend(valid.values.tolist())
                trade_dates.extend(valid.index.tolist())

        trade_returns = np.array(trade_returns)
        trade_dates = np.array(trade_dates)

        stats = compute_variant_stats(trade_returns)
        if stats is None:
            print(f"  {name}: SKIP (< {MIN_TRADES} trades)")
            continue

        # Regime analysis
        regime = regime_analysis(trade_dates, trade_returns, spy_regime)

        # Permutation test
        perm = permutation_test(trade_returns)

        # Year breakdown
        yearly = year_by_year(trade_dates, trade_returns)

        variant_result = {
            "config": {
                "drop_pct": drop_pct,
                "recovery_thresh": rec_thresh,
                "hold_days": hold,
                "vol_pctile_thresh": vol_thresh,
            },
            "stats": stats,
            "regime": regime,
            "permutation": perm,
            "year_by_year": yearly,
            "pass_regime": regime["regime_pass"],
            "pass_perm": perm["perm_pass"],
            "pass_all": regime["regime_pass"] and perm["perm_pass"],
        }

        results[name] = variant_result

        flag = "PASS" if variant_result["pass_all"] else "FAIL"
        print(f"  {name}: N={stats['n_trades']:>5}, WR={stats['win_rate']:.1%}, "
              f"Sharpe={stats['sharpe']:>6.2f}, PF={stats['profit_factor']:>5.2f}, "
              f"RG={regime['regime_gap']:.2f}, p={perm['p_value']:.3f} [{flag}]")

    # --- 4. Find top non-VC variants and add vol-compression overlays ---
    print(f"\n[INFO] Finding top base variants for vol-compression overlay...")

    # Rank by Sharpe, pick top 3 that have reasonable N
    ranked = sorted(
        [(k, v) for k, v in results.items() if v["stats"]["n_trades"] >= MIN_TRADES],
        key=lambda x: x[1]["stats"]["sharpe"],
        reverse=True,
    )

    top_for_vc = ranked[:3]
    print(f"  Top {len(top_for_vc)} for VC overlay: {[t[0] for t in top_for_vc]}")

    vc_variants = []
    for name, res in top_for_vc:
        cfg = res["config"]
        vc_name = f"{name}_VC15"
        vc_variants.append((vc_name, cfg["drop_pct"], cfg["recovery_thresh"], cfg["hold_days"], 15))

    print(f"\n[INFO] Running {len(vc_variants)} vol-compression overlay variants...")

    for name, drop_pct, rec_thresh, hold, vol_thresh in vc_variants:
        drop_threshold = -drop_pct / 100

        trade_returns = []
        trade_dates = []

        for ticker, sig in all_signals.items():
            mask = (
                (sig["intraday_drop"] <= drop_threshold)
                & (sig["recovery"] >= rec_thresh)
                & (sig["vol_pctile"] <= vol_thresh)
            )

            ret_col = f"fwd_ret_{hold}d"
            valid = sig.loc[mask & sig[ret_col].notna(), ret_col]

            if len(valid) > 0:
                trade_returns.extend(valid.values.tolist())
                trade_dates.extend(valid.index.tolist())

        trade_returns = np.array(trade_returns)
        trade_dates = np.array(trade_dates)

        stats = compute_variant_stats(trade_returns)
        if stats is None:
            print(f"  {name}: SKIP (< {MIN_TRADES} trades)")
            continue

        regime = regime_analysis(trade_dates, trade_returns, spy_regime)
        perm = permutation_test(trade_returns)
        yearly = year_by_year(trade_dates, trade_returns)

        variant_result = {
            "config": {
                "drop_pct": drop_pct,
                "recovery_thresh": rec_thresh,
                "hold_days": hold,
                "vol_pctile_thresh": vol_thresh,
            },
            "stats": stats,
            "regime": regime,
            "permutation": perm,
            "year_by_year": yearly,
            "pass_regime": regime["regime_pass"],
            "pass_perm": perm["perm_pass"],
            "pass_all": regime["regime_pass"] and perm["perm_pass"],
        }

        results[name] = variant_result

        flag = "PASS" if variant_result["pass_all"] else "FAIL"
        print(f"  {name}: N={stats['n_trades']:>5}, WR={stats['win_rate']:.1%}, "
              f"Sharpe={stats['sharpe']:>6.2f}, PF={stats['profit_factor']:>5.2f}, "
              f"RG={regime['regime_gap']:.2f}, p={perm['p_value']:.3f} [{flag}]")

    # --- 5. Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passing = {k: v for k, v in results.items() if v["pass_all"]}
    failing = {k: v for k, v in results.items() if not v["pass_all"]}

    print(f"\nPASSING variants: {len(passing)} / {len(results)}")
    for name, res in sorted(passing.items(), key=lambda x: x[1]["stats"]["sharpe"], reverse=True):
        s = res["stats"]
        print(f"  {name:25s}  N={s['n_trades']:>5}  WR={s['win_rate']:.1%}  "
              f"Sharpe={s['sharpe']:>6.2f}  PF={s['profit_factor']:>5.2f}  "
              f"MeanRet={s['mean_return']:>8.4%}")

    print(f"\nFAILING variants: {len(failing)} / {len(results)}")
    for name, res in sorted(failing.items(), key=lambda x: x[1]["stats"]["sharpe"], reverse=True):
        s = res["stats"]
        reasons = []
        if not res["pass_regime"]:
            reasons.append(f"RG={res['regime']['regime_gap']:.2f}")
        if not res["pass_perm"]:
            reasons.append(f"p={res['permutation']['p_value']:.3f}")
        print(f"  {name:25s}  N={s['n_trades']:>5}  WR={s['win_rate']:.1%}  "
              f"Sharpe={s['sharpe']:>6.2f}  [{', '.join(reasons)}]")

    # --- 6. Save results ---
    output = {
        "metadata": {
            "strategy": "hammer_reversal_v1",
            "description": "Hammer/rejection candle reversal — intraday drop with close near high",
            "run_time": str(dt.datetime.now()),
            "n_tickers": len(all_signals),
            "data_years": DATA_YEARS,
            "cost_bps_rt": COST_BPS_RT,
        },
        "variants": results,
        "summary": {
            "total_variants": len(results),
            "passing": len(passing),
            "failing": len(failing),
            "best_variant": max(results.items(), key=lambda x: x[1]["stats"]["sharpe"])[0] if results else None,
        },
    }

    output_path = OUTPUT_DIR / "results.json"

    # Custom serializer for numpy/pandas types
    def default_serializer(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date)):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=default_serializer)

    elapsed = time.time() - t0
    print(f"\n[DONE] Results saved to {output_path}")
    print(f"[DONE] Total runtime: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
