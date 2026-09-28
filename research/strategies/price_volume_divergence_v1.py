#!/usr/bin/env python3
"""
Price-Volume Divergence Strategy — v1
=======================================

HYPOTHESIS (HC #737 R4):
  Price-volume divergence — when price direction diverges from flow indicators,
  it signals asymmetric institutional positioning.

  BULLISH DIVERGENCE: Price declining over N days BUT volume indicators (OBV, MFI)
  rising = institutions accumulating while retail panics. Buy.

  Key difference from smart_money_accumulation_v1: this uses MULTIPLE flow
  indicators simultaneously and requires explicit divergence (price direction ≠
  flow direction), not just OBV slope.

SIGNAL VARIANTS (~90):
  Divergence lookback:  10d, 21d, 42d
  Price condition:      price down >0%, >2%, >5% over lookback
  Flow divergence type:
    (a) OBV rising while price falling (OBV slope > 0)
    (b) MFI rising while price falling (MFI slope > 0)
    (c) BOTH OBV AND MFI rising while price falling (strongest confluence)
  RSI filter:           RSI<40, or no filter
  Hold periods:         5d, 10d, 21d

VALIDATION (HC #428 + HC #432):
  - Regime analysis: SPY close vs 200 SMA -> GREEN/RED
  - Regime gap test: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
  - Permutation test: 1000 shuffles, p < 0.05
  - Year-by-year breakdown: >75% years profitable
  - Min trades: 30+
  - Cost: 10 bps RT

Usage:
    python3 research/strategies/price_volume_divergence_v1.py

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
# MLflow setup
# ─────────────────────────────────────────────────────────────────────────────
try:
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    mlflow.set_experiment("price_volume_divergence_v1")
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False
    print("[WARN] MLflow not available, will skip tracking")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

COST_BPS_RT = 10  # 5 bps each way for equities
NUM_PERMUTATIONS = 1000
REGIME_GAP_THRESHOLD = 0.50
PERM_P_THRESHOLD = 0.05
MIN_TRADES = 30
MFI_PERIOD = 14
RSI_PERIOD = 14
DATA_YEARS = 13

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

OUTPUT_DIR = Path(os.environ.get("LVL3_ROOT", "/home/nick/Lvl3Quant")) / "output" / "price_volume_divergence_v1"

# ─────────────────────────────────────────────────────────────────────────────
# VARIANT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

VARIANTS = []
for lookback in [10, 21, 42]:
    for price_drop_pct in [0, 2, 5]:  # price down >X% over lookback
        for div_type in ["obv", "mfi", "both"]:  # which flow indicator(s) must diverge
            for rsi_filter in [None, 40]:  # None = no RSI filter, 40 = RSI<40
                for hold in [5, 10, 21]:
                    rsi_str = "noRSI" if rsi_filter is None else f"RSI{rsi_filter}"
                    name = f"LB{lookback}_Dn{price_drop_pct}_{div_type}_{rsi_str}_H{hold}"
                    VARIANTS.append({
                        "name": name,
                        "lookback": lookback,
                        "price_drop_pct": price_drop_pct,
                        "div_type": div_type,
                        "rsi_filter": rsi_filter,
                        "hold": hold,
                    })

print(f"[INFO] Total variants to test: {len(VARIANTS)}")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html(SP500_URL)
        df = tables[0]
        tickers = df["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        print(f"[INFO] Got {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        print(f"[WARN] Wikipedia fetch failed ({e}), using fallback list")
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
                    if len(df) > 100 and "Volume" in df.columns:
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
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    close = spy["Close"].astype(float)
    spy["SMA200"] = close.rolling(200).mean()
    spy["regime"] = np.where(close > spy["SMA200"], "GREEN", "RED")
    return spy[["Close", "SMA200", "regime"]].dropna()


def compute_obv(close, volume):
    """Compute On-Balance Volume (OBV)."""
    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0).cumsum()
    return obv


def compute_mfi(high, low, close, volume, period=MFI_PERIOD):
    """Compute Money Flow Index (MFI)."""
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume

    tp_diff = typical_price.diff()
    pos_flow = pd.Series(np.where(tp_diff > 0, raw_money_flow, 0), index=close.index)
    neg_flow = pd.Series(np.where(tp_diff < 0, raw_money_flow, 0), index=close.index)

    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()

    money_ratio = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_ratio))
    return mfi


def compute_rsi(close, period=RSI_PERIOD):
    """Compute Relative Strength Index (RSI)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_slope(series, window):
    """Compute linear regression slope over rolling window using vectorized approach."""
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(vals):
        if len(vals) < window or np.isnan(vals).any():
            return np.nan
        y_mean = vals.mean()
        return ((x * (vals - y_mean)).sum()) / x_var

    return series.rolling(window).apply(_slope, raw=True)


def compute_signals(df):
    """Compute price-volume divergence signals for a single stock.

    Returns DataFrame with columns for each lookback:
        price_ret_{lb}d, obv_slope_{lb}d, mfi_slope_{lb}d, rsi, fwd_ret_Xd
    """
    out = pd.DataFrame(index=df.index)

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)

    # Compute OBV, MFI, RSI
    obv = compute_obv(c, v)
    mfi = compute_mfi(h, l, c, v)
    rsi = compute_rsi(c)
    out["rsi"] = rsi

    # For each lookback, compute price return and flow indicator slopes
    for lb in [10, 21, 42]:
        out[f"price_ret_{lb}d"] = c.pct_change(lb)
        out[f"obv_slope_{lb}d"] = compute_slope(obv, lb)
        out[f"mfi_slope_{lb}d"] = compute_slope(mfi, lb)

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

    if net_returns.std() > 0:
        sharpe = mean_ret / net_returns.std() * np.sqrt(52)
    else:
        sharpe = 0.0

    # Sortino
    downside = net_returns[net_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = mean_ret / downside.std() * np.sqrt(52)
    else:
        sortino = 0.0

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
        "sortino": round(sortino, 4),
        "profit_factor": round(pf, 4),
        "total_return": round(net_returns.sum(), 6),
    }


def regime_analysis(trade_dates, trade_returns, spy_regime, cost_bps=COST_BPS_RT):
    """Split trades by regime and compute regime gap."""
    cost = cost_bps / 10000
    net_returns = trade_returns - cost

    regime_series = spy_regime["regime"]
    green_mask = []
    red_mask = []
    for d in trade_dates:
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
        perm_mean = all_values[:len(net_returns)].mean()
        if perm_mean >= observed_mean:
            count_greater += 1

    p_value = count_greater / n_perms
    return {
        "perm_p": round(p_value, 4),
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
    n_profitable_years = 0
    n_years = 0
    for year, group in df.groupby("year"):
        if len(group) < 3:
            continue
        wr = (group["ret"] > 0).mean()
        mean_r = group["ret"].mean()
        profitable = mean_r > 0
        if profitable:
            n_profitable_years += 1
        n_years += 1
        results[int(year)] = {
            "n_trades": len(group),
            "win_rate": round(wr, 4),
            "mean_return": round(mean_r, 6),
            "profitable": bool(profitable),
        }

    year_consistency = n_profitable_years / n_years if n_years > 0 else 0
    return results, round(year_consistency, 4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PRICE-VOLUME DIVERGENCE STRATEGY — v1")
    print("=" * 70)
    print(f"Start time: {dt.datetime.now()}")
    print()

    # --- 1. Get tickers & download data ---
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=365 * DATA_YEARS + 200)

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
        except Exception:
            pass

    print(f"[INFO] Signals computed for {len(all_signals)} stocks")

    # --- 3. Run all variants ---
    print(f"\n[INFO] Running {len(VARIANTS)} variants...")
    results = {}
    passing_count = 0

    for vi, var in enumerate(VARIANTS):
        name = var["name"]
        lookback = var["lookback"]
        price_drop_pct = var["price_drop_pct"]
        div_type = var["div_type"]
        rsi_filter = var["rsi_filter"]
        hold = var["hold"]

        trade_returns = []
        trade_dates = []

        price_col = f"price_ret_{lookback}d"
        obv_col = f"obv_slope_{lookback}d"
        mfi_col = f"mfi_slope_{lookback}d"

        for ticker, sig in all_signals.items():
            # Core signal: price declining over lookback
            if price_drop_pct == 0:
                price_mask = sig[price_col] < 0  # any decline
            else:
                price_mask = sig[price_col] < (-price_drop_pct / 100)  # down >X%

            # Flow divergence: indicator(s) rising while price falling
            if div_type == "obv":
                div_mask = sig[obv_col] > 0  # OBV slope positive
            elif div_type == "mfi":
                div_mask = sig[mfi_col] > 0  # MFI slope positive
            else:  # "both" — strongest confluence
                div_mask = (sig[obv_col] > 0) & (sig[mfi_col] > 0)

            mask = price_mask & div_mask

            # Optional RSI filter
            if rsi_filter is not None:
                mask = mask & (sig["rsi"] < rsi_filter)

            ret_col = f"fwd_ret_{hold}d"
            valid = sig.loc[mask & sig[ret_col].notna(), ret_col]

            if len(valid) > 0:
                trade_returns.extend(valid.values.tolist())
                trade_dates.extend(valid.index.tolist())

        trade_returns = np.array(trade_returns)
        trade_dates = np.array(trade_dates)

        stats = compute_variant_stats(trade_returns)
        if stats is None:
            if vi % 20 == 0:
                print(f"  [{vi+1}/{len(VARIANTS)}] {name}: SKIP (< {MIN_TRADES} trades)")
            continue

        # Regime analysis
        regime = regime_analysis(trade_dates, trade_returns, spy_regime)

        # Permutation test
        perm = permutation_test(trade_returns)

        # Year breakdown + consistency
        yearly, year_consistency = year_by_year(trade_dates, trade_returns)
        year_pass = year_consistency >= 0.75

        variant_result = {
            "config": {
                "lookback": lookback,
                "price_drop_pct": price_drop_pct,
                "div_type": div_type,
                "rsi_filter": rsi_filter,
                "hold_days": hold,
            },
            "stats": stats,
            "regime": regime,
            "permutation": perm,
            "perm_p": perm["perm_p"],
            "year_by_year": yearly,
            "year_consistency": year_consistency,
            "pass_regime": bool(regime["regime_pass"]),
            "pass_perm": bool(perm["perm_pass"]),
            "pass_year": bool(year_pass),
            "pass_all": bool(regime["regime_pass"] and perm["perm_pass"] and year_pass),
        }

        results[name] = variant_result

        if variant_result["pass_all"]:
            passing_count += 1

        flag = "PASS" if variant_result["pass_all"] else "FAIL"
        print(f"  [{vi+1}/{len(VARIANTS)}] {name}: N={stats['n_trades']:>5}, WR={stats['win_rate']:.1%}, "
              f"Sharpe={stats['sharpe']:>6.2f}, PF={stats['profit_factor']:>5.2f}, "
              f"RG={regime['regime_gap']:.2f}, p={perm['perm_p']:.3f}, "
              f"YC={year_consistency:.0%} [{flag}]")

    # --- 4. Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passing = {k: v for k, v in results.items() if v["pass_all"]}
    failing = {k: v for k, v in results.items() if not v["pass_all"]}

    print(f"\nTotal variants evaluated: {len(results)}")
    print(f"PASSING (regime + perm + year consistency): {len(passing)}")
    print(f"FAILING: {len(failing)}")

    if passing:
        print(f"\n{'Name':45s} {'N':>5} {'WR':>6} {'Sharpe':>7} {'Sortino':>7} {'PF':>6} {'MeanRet':>9}")
        print("-" * 90)
        for name, res in sorted(passing.items(), key=lambda x: x[1]["stats"]["sharpe"], reverse=True):
            s = res["stats"]
            print(f"  {name:43s} {s['n_trades']:>5} {s['win_rate']:.1%} "
                  f"{s['sharpe']:>7.2f} {s['sortino']:>7.2f} {s['profit_factor']:>6.2f} "
                  f"{s['mean_return']:>8.4%}")

    if failing:
        print(f"\nTop 10 FAILING by Sharpe:")
        for name, res in sorted(failing.items(), key=lambda x: x[1]["stats"]["sharpe"], reverse=True)[:10]:
            s = res["stats"]
            reasons = []
            if not res["pass_regime"]:
                reasons.append(f"RG={res['regime']['regime_gap']:.2f}")
            if not res["pass_perm"]:
                reasons.append(f"p={res['perm_p']:.3f}")
            if not res["pass_year"]:
                reasons.append(f"YC={res['year_consistency']:.0%}")
            print(f"  {name:43s} N={s['n_trades']:>5} Sharpe={s['sharpe']:>6.2f} [{', '.join(reasons)}]")

    # Best variant overall
    best_name = None
    if results:
        if passing:
            best_name = max(passing.items(), key=lambda x: x[1]["stats"]["sharpe"])[0]
            print(f"\nBEST PASSING: {best_name}")
            best = passing[best_name]
        else:
            best_name = max(results.items(), key=lambda x: x[1]["stats"]["sharpe"])[0]
            print(f"\nBEST OVERALL (no variants passed all gates): {best_name}")
            best = results[best_name]

        s = best["stats"]
        print(f"  Trades: {s['n_trades']}, WR: {s['win_rate']:.1%}, Sharpe: {s['sharpe']:.2f}, "
              f"Sortino: {s['sortino']:.2f}, PF: {s['profit_factor']:.2f}")
        print(f"  Regime gap: {best['regime']['regime_gap']:.2f}, Perm p: {best['perm_p']:.4f}, "
              f"Year consistency: {best['year_consistency']:.0%}")

    # --- 5. Save results ---
    output = {
        "metadata": {
            "strategy": "price_volume_divergence_v1",
            "description": "Price-volume divergence — price declining but flow indicators (OBV/MFI) rising signals institutional accumulation",
            "hypothesis": "Price direction diverging from volume flow direction = asymmetric institutional positioning = buy",
            "run_time": str(dt.datetime.now()),
            "n_tickers": len(all_signals),
            "data_years": DATA_YEARS,
            "cost_bps_rt": COST_BPS_RT,
            "n_permutations": NUM_PERMUTATIONS,
            "regime_gap_threshold": REGIME_GAP_THRESHOLD,
            "perm_p_threshold": PERM_P_THRESHOLD,
        },
        "variants": results,
        "summary": {
            "total_variants_tested": len(VARIANTS),
            "total_variants_evaluated": len(results),
            "passing": len(passing),
            "failing": len(failing),
            "best_variant": best_name,
        },
    }

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

    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=default_serializer)

    elapsed = time.time() - t0
    print(f"\n[DONE] Results saved to {output_path}")
    print(f"[DONE] Total runtime: {elapsed / 60:.1f} minutes")

    # --- 6. MLflow logging ---
    if HAS_MLFLOW:
        try:
            with mlflow.start_run(run_name="price_volume_divergence_v1"):
                mlflow.log_param("n_tickers", len(all_signals))
                mlflow.log_param("n_variants_tested", len(VARIANTS))
                mlflow.log_param("data_years", DATA_YEARS)
                mlflow.log_param("cost_bps_rt", COST_BPS_RT)
                mlflow.log_param("n_permutations", NUM_PERMUTATIONS)

                mlflow.log_metric("n_variants_evaluated", len(results))
                mlflow.log_metric("n_passing", len(passing))
                mlflow.log_metric("n_failing", len(failing))

                if best_name and best_name in results:
                    best_res = results[best_name]
                    mlflow.log_metric("best_sharpe", best_res["stats"]["sharpe"])
                    mlflow.log_metric("best_sortino", best_res["stats"]["sortino"])
                    mlflow.log_metric("best_pf", best_res["stats"]["profit_factor"])
                    mlflow.log_metric("best_wr", best_res["stats"]["win_rate"])
                    mlflow.log_metric("best_n_trades", best_res["stats"]["n_trades"])
                    mlflow.log_metric("best_regime_gap", best_res["regime"]["regime_gap"])
                    mlflow.log_metric("best_perm_p", best_res["perm_p"])
                    mlflow.log_metric("best_year_consistency", best_res["year_consistency"])
                    mlflow.log_param("best_variant", best_name)

                try:
                    mlflow.log_artifact(str(output_path))
                except Exception as e:
                    print(f"[WARN] MLflow artifact upload failed: {e}")

                print("[INFO] MLflow logged successfully")
        except Exception as e:
            print(f"[WARN] MLflow logging failed: {e}")


if __name__ == "__main__":
    main()
