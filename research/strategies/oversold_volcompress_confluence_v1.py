#!/usr/bin/env python3
"""
Oversold + Vol Compression Confluence Strategy — v1
====================================================

PROVEN INDEPENDENT EDGES (both pass permutation + regime gates):
  1. Oversold bounce: RSI(5)<20 or 1-day drop>5%, hold 10d. WR 55%, PF 2.05,
     regime gap 0.41, perm p=0.000
  2. Vol compression → breakout: hist vol at <10th pctile, then breakout.
     WR 81%, PF 8.26, regime gap passes

HYPOTHESIS:
  When BOTH conditions are present simultaneously (oversold AND vol compressed),
  the resulting move is stronger and more reliable. The intersection filters
  out noise from each signal independently. These stocks are "coiled springs
  being pushed down" — when they bounce, the move should be larger.

SIGNAL VARIANTS (12 total):
  - RSI(5)<20 + vol_pctile<10, hold 5/10/21 days
  - RSI(5)<25 + vol_pctile<15, hold 5/10/21 days
  - drop_1d>3% + vol_pctile<10, hold 5/10/21 days
  - drop_1d>5% + vol_pctile<10, hold 5/10/21 days

VALIDATION (HC #428 + HC #432):
  - Regime analysis: SPY close vs 200 SMA → GREEN/RED
  - Regime gap test: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) ≤ 0.50
  - Permutation test: 1000 shuffles, p < 0.05
  - Year-by-year breakdown
  - Comparison vs individual signals (RSI-only, vol-only)

Usage:
    python3 research/strategies/oversold_volcompress_confluence_v1.py

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
RSI_PERIOD = 5
DATA_YEARS = 12  # years of data to download

# S&P 500 — we'll get a recent list via Wikipedia
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Output
OUTPUT_DIR = Path(os.environ.get("LVL3_ROOT", "/home/nick/Lvl3Quant")) / "output" / "oversold_volcompress_confluence_v1"

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL VARIANT DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

CONFLUENCE_VARIANTS = [
    # (name, rsi_thresh, vol_pctile_thresh, drop_thresh, hold_days)
    # RSI + vol compression
    ("RSI20_VP10_H5",   20, 10, None, 5),
    ("RSI20_VP10_H10",  20, 10, None, 10),
    ("RSI20_VP10_H21",  20, 10, None, 21),
    ("RSI25_VP15_H5",   25, 15, None, 5),
    ("RSI25_VP15_H10",  25, 15, None, 10),
    ("RSI25_VP15_H21",  25, 15, None, 21),
    # Drop + vol compression
    ("DROP3_VP10_H5",   None, 10, 0.03, 5),
    ("DROP3_VP10_H10",  None, 10, 0.03, 10),
    ("DROP3_VP10_H21",  None, 10, 0.03, 21),
    ("DROP5_VP10_H5",   None, 10, 0.05, 5),
    ("DROP5_VP10_H10",  None, 10, 0.05, 10),
    ("DROP5_VP10_H21",  None, 10, 0.05, 21),
]

# Individual signal variants for comparison
INDIVIDUAL_VARIANTS = [
    # RSI-only
    ("RSI20_ONLY_H5",   20, None, None, 5),
    ("RSI20_ONLY_H10",  20, None, None, 10),
    ("RSI25_ONLY_H10",  25, None, None, 10),
    # Vol compression only
    ("VP10_ONLY_H5",    None, 10, None, 5),
    ("VP10_ONLY_H10",   None, 10, None, 10),
    ("VP15_ONLY_H10",   None, 15, None, 10),
    # Drop only
    ("DROP3_ONLY_H10",  None, None, 0.03, 10),
    ("DROP5_ONLY_H10",  None, None, 0.05, 10),
]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html(SP500_URL)
        df = tables[0]
        tickers = df["Symbol"].tolist()
        # Clean: replace . with - for yfinance (e.g., BRK.B -> BRK-B)
        tickers = [t.replace(".", "-") for t in tickers]
        print(f"[INFO] Retrieved {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        print(f"[WARN] Failed to get S&P 500 list from Wikipedia: {e}")
        print("[WARN] Falling back to hardcoded top 200 tickers")
        return _fallback_tickers()


def _fallback_tickers():
    """Fallback list of ~200 liquid large-caps."""
    return [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
        "UNH", "JNJ", "V", "JPM", "XOM", "PG", "MA", "HD", "CVX", "LLY",
        "ABBV", "MRK", "PEP", "KO", "AVGO", "COST", "TMO", "MCD", "WMT",
        "CSCO", "ACN", "ABT", "DHR", "NEE", "LIN", "TXN", "PM", "UNP",
        "BMY", "RTX", "AMGN", "LOW", "HON", "ORCL", "UPS", "QCOM", "MS",
        "GS", "ELV", "BA", "SBUX", "MDLZ", "ADP", "BLK", "GILD", "ADI",
        "DE", "SYK", "MMC", "CB", "PLD", "AMT", "CI", "ISRG", "REGN",
        "VRTX", "SO", "DUK", "MO", "BSX", "CL", "CME", "SLB", "BDX",
        "NOC", "EOG", "APD", "SCHW", "ITW", "CSX", "FDX", "WM", "PNC",
        "USB", "CCI", "NSC", "ICE", "EMR", "MCK", "SHW", "ATVI", "HUM",
        "ORLY", "GD", "F", "GM", "AIG", "MET", "ALL", "TRV", "PRU",
        "AFL", "AEP", "D", "EXC", "SRE", "XEL", "WEC", "ES", "ED",
        "FE", "PEG", "ETR", "AES", "PPL", "CMS", "CNP", "NI", "EVRG",
        "ATO", "OGE", "PNW", "NRG", "AWK", "WMB", "OKE", "KMI", "ET",
        "TRGP", "MPC", "PSX", "VLO", "PBF", "DK", "HES", "FANG", "DVN",
        "COP", "OXY", "APA", "MRO", "HAL", "BKR", "NOV", "RIG", "HP",
        "CLR", "MGM", "WYNN", "CZR", "LVS", "NCLH", "CCL", "RCL",
        "MAR", "HLT", "H", "IHG", "DAL", "UAL", "LUV", "AAL", "ALK",
        "JBLU", "HA", "SAVE", "DIS", "CMCSA", "NFLX", "T", "VZ",
        "TMUS", "CHTR", "DISH", "LUMN", "CAT", "MMM", "GE", "HON",
        "EMR", "ROK", "ETN", "PH", "IR", "DOV", "AME", "SWK", "ROP",
        "FTV", "GRMN", "TDY", "WAB", "GNRC", "NDSN", "XYL", "AOS",
        "IEX", "RBC", "HWM", "TT", "CARR", "OTIS", "LHX", "HII",
    ]


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_vol_percentile(prices: pd.Series, vol_window: int = 20,
                            hist_window: int = 252) -> pd.Series:
    """
    Compute realized vol percentile: current vol_window-day vol ranked against
    its own hist_window-day history of vol_window-day vol readings.
    """
    log_ret = np.log(prices / prices.shift(1))
    realized_vol = log_ret.rolling(vol_window).std() * np.sqrt(252)

    def _pctile(s):
        if len(s) < hist_window:
            return np.nan
        current = s.iloc[-1]
        history = s.iloc[:-1]
        if np.isnan(current) or history.isna().all():
            return np.nan
        return (history < current).sum() / len(history) * 100

    vol_pctile = realized_vol.rolling(hist_window + 1).apply(_pctile, raw=False)
    return vol_pctile


def compute_regime(spy_close: pd.Series) -> pd.Series:
    """GREEN if SPY close > 200 SMA, else RED."""
    sma200 = spy_close.rolling(200).mean()
    regime = pd.Series("RED", index=spy_close.index)
    regime[spy_close > sma200] = "GREEN"
    regime[sma200.isna()] = np.nan
    return regime


def annualized_sharpe(returns: np.ndarray, trades_per_year: float = 50) -> float:
    """Annualized Sharpe from per-trade returns."""
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year)


def profit_factor(returns: np.ndarray) -> float:
    """Sum of gains / sum of losses."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def permutation_test(returns: np.ndarray, n_perms: int = 1000) -> float:
    """
    Permutation test: shuffle sign of returns, compute fraction where
    random Sharpe >= actual Sharpe.
    """
    actual_sharpe = annualized_sharpe(returns)
    count_ge = 0
    abs_returns = np.abs(returns)
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(returns))
        shuffled = abs_returns * signs
        shuf_sharpe = annualized_sharpe(shuffled)
        if shuf_sharpe >= actual_sharpe:
            count_ge += 1
    return count_ge / n_perms


def regime_gap(sharpe_green: float, sharpe_red: float) -> float:
    """Regime gap metric. REJECT if > 0.50."""
    denom = max(abs(sharpe_green), abs(sharpe_red))
    if denom == 0:
        return 0.0
    return abs(sharpe_green - sharpe_red) / denom


# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_data(tickers: list, years: int = 12) -> tuple:
    """
    Download daily OHLCV data for all tickers + SPY.
    Returns dict of {ticker: DataFrame} and SPY DataFrame.
    """
    import yfinance as yf

    end = dt.datetime.now()
    start = end - dt.timedelta(days=years * 365)

    # Ensure SPY is included
    all_tickers = list(set(tickers + ["SPY"]))

    print(f"[INFO] Downloading {len(all_tickers)} tickers, {start.date()} to {end.date()}")
    t0 = time.time()

    # Download in batches to avoid timeout
    batch_size = 50
    all_data = {}

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        batch_str = " ".join(batch)
        try:
            data = yf.download(batch_str, start=start, end=end,
                             group_by="ticker", progress=False, threads=True)

            if len(batch) == 1:
                ticker = batch[0]
                if not data.empty:
                    all_data[ticker] = data
            else:
                for ticker in batch:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            df = data[ticker].dropna(how="all")
                            if len(df) > 252:  # at least 1 year
                                all_data[ticker] = df
                    except Exception:
                        pass
        except Exception as e:
            print(f"[WARN] Batch {i//batch_size + 1} failed: {e}")

        if (i // batch_size + 1) % 5 == 0:
            print(f"[INFO] Downloaded {min(i + batch_size, len(all_tickers))}/{len(all_tickers)} tickers...")

    elapsed = time.time() - t0
    print(f"[INFO] Downloaded {len(all_data)} tickers in {elapsed:.0f}s")

    spy_df = all_data.pop("SPY", None)
    if spy_df is None:
        raise RuntimeError("Failed to download SPY data")

    return all_data, spy_df


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_signals(stock_data: dict, spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each stock on each date, compute:
    - RSI(5)
    - vol_percentile (20d vol ranked in 252d history)
    - 1-day return (drop)
    - forward returns (5d, 10d, 21d)
    - regime (GREEN/RED from SPY)

    Returns a DataFrame of all signal dates across all stocks.
    """
    spy_regime = compute_regime(spy_df["Close"].squeeze())

    all_signals = []
    processed = 0

    for ticker, df in stock_data.items():
        try:
            close = df["Close"].squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]

            if len(close) < 300:
                continue

            # Compute features
            rsi = compute_rsi(close, RSI_PERIOD)
            vol_pctile = compute_vol_percentile(close, LOOKBACK_VOL_DAYS, VOL_HISTORY_DAYS)
            daily_ret = close.pct_change()

            # Forward returns (from next open, approximated as next close)
            fwd_5d = close.shift(-5) / close - 1
            fwd_10d = close.shift(-10) / close - 1
            fwd_21d = close.shift(-21) / close - 1

            # Build signal frame
            sig = pd.DataFrame({
                "ticker": ticker,
                "close": close,
                "rsi5": rsi,
                "vol_pctile": vol_pctile,
                "daily_ret": daily_ret,
                "fwd_5d": fwd_5d,
                "fwd_10d": fwd_10d,
                "fwd_21d": fwd_21d,
            }, index=close.index)

            # Map regime
            sig["regime"] = sig.index.map(spy_regime)
            sig["year"] = sig.index.year

            # Drop rows without enough data
            sig = sig.dropna(subset=["rsi5", "vol_pctile", "daily_ret"])

            all_signals.append(sig)
            processed += 1

        except Exception as e:
            pass  # skip problematic tickers silently

    print(f"[INFO] Computed signals for {processed} stocks")

    if not all_signals:
        raise RuntimeError("No signals generated — check data download")

    return pd.concat(all_signals, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_variant(name: str, rsi_thresh, vol_pctile_thresh, drop_thresh,
                     hold_days: int, signals_df: pd.DataFrame) -> dict:
    """
    Filter signals for the variant, compute metrics.
    """
    mask = pd.Series(True, index=signals_df.index)

    # Apply filters
    if rsi_thresh is not None:
        mask &= (signals_df["rsi5"] < rsi_thresh)
    if vol_pctile_thresh is not None:
        mask &= (signals_df["vol_pctile"] < vol_pctile_thresh)
    if drop_thresh is not None:
        mask &= (signals_df["daily_ret"] < -drop_thresh)

    # At least one filter must be active (for individual signals)
    # For confluence: at least two filters active

    trades = signals_df[mask].copy()

    # Select appropriate forward return
    fwd_col = f"fwd_{hold_days}d"
    if fwd_col not in trades.columns:
        return {"name": name, "status": "SKIP", "reason": f"No {fwd_col} column"}

    trades = trades.dropna(subset=[fwd_col])

    # Apply transaction costs
    cost_frac = COST_BPS_RT / 10000  # 10 bps round-trip
    returns = trades[fwd_col].values - cost_frac

    n_trades = len(returns)
    if n_trades < MIN_TRADES:
        return {
            "name": name,
            "status": "INSUFFICIENT_TRADES",
            "n_trades": n_trades,
            "min_required": MIN_TRADES,
        }

    # Core metrics
    wr = (returns > 0).mean()
    mean_ret = returns.mean()
    median_ret = np.median(returns)
    sharpe = annualized_sharpe(returns)
    pf = profit_factor(returns)

    # Regime analysis
    green_mask = trades["regime"] == "GREEN"
    red_mask = trades["regime"] == "RED"

    returns_green = returns[green_mask.values]
    returns_red = returns[red_mask.values]

    sharpe_green = annualized_sharpe(returns_green) if len(returns_green) >= 10 else np.nan
    sharpe_red = annualized_sharpe(returns_red) if len(returns_red) >= 10 else np.nan

    if not np.isnan(sharpe_green) and not np.isnan(sharpe_red):
        rgap = regime_gap(sharpe_green, sharpe_red)
        regime_pass = rgap <= REGIME_GAP_THRESHOLD
    else:
        rgap = np.nan
        regime_pass = None

    # Permutation test
    perm_p = permutation_test(returns, NUM_PERMUTATIONS)
    perm_pass = perm_p < PERM_P_THRESHOLD

    # Year-by-year breakdown
    yearly = {}
    for year in sorted(trades["year"].unique()):
        yr_mask = trades["year"] == year
        yr_returns = returns[yr_mask.values]
        if len(yr_returns) >= 5:
            yearly[int(year)] = {
                "n_trades": int(len(yr_returns)),
                "wr": float(round((yr_returns > 0).mean(), 3)),
                "mean_ret": float(round(yr_returns.mean() * 100, 2)),
                "sharpe": float(round(annualized_sharpe(yr_returns), 2)),
            }

    # Overall verdict
    passes_all = True
    if regime_pass is not None and not regime_pass:
        passes_all = False
    if not perm_pass:
        passes_all = False
    if sharpe <= 0:
        passes_all = False

    return {
        "name": name,
        "status": "PASS" if passes_all else "FAIL",
        "n_trades": int(n_trades),
        "n_green": int(green_mask.sum()),
        "n_red": int(red_mask.sum()),
        "hold_days": hold_days,
        "wr": float(round(wr, 4)),
        "mean_ret_pct": float(round(mean_ret * 100, 3)),
        "median_ret_pct": float(round(median_ret * 100, 3)),
        "sharpe": float(round(sharpe, 3)),
        "profit_factor": float(round(pf, 3)),
        "sharpe_green": float(round(sharpe_green, 3)) if not np.isnan(sharpe_green) else None,
        "sharpe_red": float(round(sharpe_red, 3)) if not np.isnan(sharpe_red) else None,
        "regime_gap": float(round(rgap, 3)) if not np.isnan(rgap) else None,
        "regime_pass": regime_pass,
        "perm_p": float(round(perm_p, 4)),
        "perm_pass": perm_pass,
        "yearly": yearly,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print("=" * 80)
    print("OVERSOLD + VOL COMPRESSION CONFLUENCE STRATEGY — v1")
    print("=" * 80)
    print()

    # 1. Get universe
    tickers = get_sp500_tickers()

    # 2. Download data
    stock_data, spy_df = download_data(tickers, years=DATA_YEARS)

    if len(stock_data) < 100:
        print(f"[WARN] Only {len(stock_data)} stocks downloaded, expected 200+")

    # 3. Generate signals
    print("\n[INFO] Computing signals (RSI, vol percentile, daily returns)...")
    t_sig = time.time()
    signals_df = generate_signals(stock_data, spy_df)
    print(f"[INFO] Signal computation done in {time.time() - t_sig:.0f}s")
    print(f"[INFO] Total signal rows: {len(signals_df):,}")

    # Free memory
    del stock_data

    # 4. Evaluate confluence variants
    print("\n" + "=" * 80)
    print("CONFLUENCE VARIANTS (oversold + vol compression)")
    print("=" * 80)

    confluence_results = []
    for name, rsi_t, vol_t, drop_t, hold in CONFLUENCE_VARIANTS:
        print(f"\n--- Evaluating {name} ---")
        result = evaluate_variant(name, rsi_t, vol_t, drop_t, hold, signals_df)
        confluence_results.append(result)

        if result.get("status") == "INSUFFICIENT_TRADES":
            print(f"  SKIP: Only {result['n_trades']} trades (need {MIN_TRADES})")
        else:
            print(f"  N={result['n_trades']}, WR={result['wr']:.1%}, "
                  f"Mean={result['mean_ret_pct']:.2f}%, Sharpe={result['sharpe']:.2f}, "
                  f"PF={result['profit_factor']:.2f}")
            print(f"  Regime gap={result.get('regime_gap', 'N/A')}, "
                  f"Perm p={result['perm_p']:.4f}, "
                  f"Status={result['status']}")

    # 5. Evaluate individual signal variants (for comparison)
    print("\n" + "=" * 80)
    print("INDIVIDUAL SIGNAL VARIANTS (for comparison)")
    print("=" * 80)

    individual_results = []
    for name, rsi_t, vol_t, drop_t, hold in INDIVIDUAL_VARIANTS:
        print(f"\n--- Evaluating {name} ---")
        result = evaluate_variant(name, rsi_t, vol_t, drop_t, hold, signals_df)
        individual_results.append(result)

        if result.get("status") == "INSUFFICIENT_TRADES":
            print(f"  SKIP: Only {result['n_trades']} trades (need {MIN_TRADES})")
        else:
            print(f"  N={result['n_trades']}, WR={result['wr']:.1%}, "
                  f"Mean={result['mean_ret_pct']:.2f}%, Sharpe={result['sharpe']:.2f}, "
                  f"PF={result['profit_factor']:.2f}")
            print(f"  Regime gap={result.get('regime_gap', 'N/A')}, "
                  f"Perm p={result['perm_p']:.4f}, "
                  f"Status={result['status']}")

    # 6. Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    header = f"{'Variant':<22} {'N':>6} {'WR':>6} {'Mean%':>7} {'Sharpe':>7} {'PF':>7} {'RGap':>6} {'Perm-p':>7} {'Pass':>5}"
    print(header)
    print("-" * len(header))

    all_results = confluence_results + individual_results

    for r in all_results:
        if r.get("status") in ("INSUFFICIENT_TRADES", "SKIP"):
            print(f"{r['name']:<22} {'SKIP':>6}  (insufficient trades: {r.get('n_trades', '?')})")
            continue

        rgap_str = f"{r['regime_gap']:.3f}" if r.get("regime_gap") is not None else "N/A"
        status = "YES" if r["status"] == "PASS" else "NO"

        print(f"{r['name']:<22} {r['n_trades']:>6} {r['wr']:>5.1%} "
              f"{r['mean_ret_pct']:>7.2f} {r['sharpe']:>7.2f} "
              f"{r['profit_factor']:>7.2f} {rgap_str:>6} {r['perm_p']:>7.4f} {status:>5}")

    # 7. Confluence vs Individual comparison
    print("\n" + "=" * 80)
    print("CONFLUENCE VALUE-ADD ANALYSIS")
    print("=" * 80)

    # Compare RSI20_VP10_H10 (confluence) vs RSI20_ONLY_H10 and VP10_ONLY_H10
    conf_h10 = next((r for r in confluence_results if r["name"] == "RSI20_VP10_H10"), None)
    rsi_only = next((r for r in individual_results if r["name"] == "RSI20_ONLY_H10"), None)
    vol_only = next((r for r in individual_results if r["name"] == "VP10_ONLY_H10"), None)

    if conf_h10 and conf_h10.get("status") not in ("INSUFFICIENT_TRADES", "SKIP"):
        print(f"\nConfluence RSI20+VP10 H10:")
        print(f"  Sharpe={conf_h10['sharpe']:.2f}, WR={conf_h10['wr']:.1%}, "
              f"Mean={conf_h10['mean_ret_pct']:.2f}%, N={conf_h10['n_trades']}")

    if rsi_only and rsi_only.get("status") not in ("INSUFFICIENT_TRADES", "SKIP"):
        print(f"\nRSI20-only H10:")
        print(f"  Sharpe={rsi_only['sharpe']:.2f}, WR={rsi_only['wr']:.1%}, "
              f"Mean={rsi_only['mean_ret_pct']:.2f}%, N={rsi_only['n_trades']}")

    if vol_only and vol_only.get("status") not in ("INSUFFICIENT_TRADES", "SKIP"):
        print(f"\nVP10-only H10:")
        print(f"  Sharpe={vol_only['sharpe']:.2f}, WR={vol_only['wr']:.1%}, "
              f"Mean={vol_only['mean_ret_pct']:.2f}%, N={vol_only['n_trades']}")

    if (conf_h10 and rsi_only and vol_only and
        all(r.get("status") not in ("INSUFFICIENT_TRADES", "SKIP")
            for r in [conf_h10, rsi_only, vol_only])):

        sharpe_lift_vs_rsi = conf_h10["sharpe"] - rsi_only["sharpe"]
        sharpe_lift_vs_vol = conf_h10["sharpe"] - vol_only["sharpe"]

        print(f"\nSharpe lift from confluence:")
        print(f"  vs RSI-only: {sharpe_lift_vs_rsi:+.2f}")
        print(f"  vs Vol-only: {sharpe_lift_vs_vol:+.2f}")

        if sharpe_lift_vs_rsi > 0 and sharpe_lift_vs_vol > 0:
            print("  VERDICT: Confluence ADDS VALUE over both individual signals")
        elif sharpe_lift_vs_rsi > 0 or sharpe_lift_vs_vol > 0:
            print("  VERDICT: Confluence adds value over ONE signal but not both")
        else:
            print("  VERDICT: Confluence does NOT add value — individual signals may be better")

    # 8. Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.json"

    output = {
        "timestamp": dt.datetime.now().isoformat(),
        "strategy": "oversold_volcompress_confluence_v1",
        "universe": "S&P 500",
        "data_years": DATA_YEARS,
        "cost_bps_rt": COST_BPS_RT,
        "n_permutations": NUM_PERMUTATIONS,
        "regime_gap_threshold": REGIME_GAP_THRESHOLD,
        "perm_p_threshold": PERM_P_THRESHOLD,
        "confluence_results": confluence_results,
        "individual_results": individual_results,
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[INFO] Results saved to {results_path}")
    print(f"[INFO] Total runtime: {time.time() - t_start:.0f}s")

    # 9. Final verdicts
    print("\n" + "=" * 80)
    print("FINAL VERDICTS")
    print("=" * 80)

    passing = [r for r in confluence_results if r.get("status") == "PASS"]
    failing = [r for r in confluence_results if r.get("status") == "FAIL"]
    skipped = [r for r in confluence_results if r.get("status") in ("INSUFFICIENT_TRADES", "SKIP")]

    print(f"\nConfluence variants: {len(passing)} PASS, {len(failing)} FAIL, {len(skipped)} SKIP")

    if passing:
        best = max(passing, key=lambda r: r["sharpe"])
        print(f"\nBest confluence variant: {best['name']}")
        print(f"  Sharpe={best['sharpe']:.2f}, WR={best['wr']:.1%}, "
              f"PF={best['profit_factor']:.2f}, N={best['n_trades']}")
        print(f"  Regime gap={best.get('regime_gap', 'N/A')}, Perm p={best['perm_p']:.4f}")
    else:
        print("\nNo confluence variant passed all gates.")

    print("\nDone.")
    return output


if __name__ == "__main__":
    main()
