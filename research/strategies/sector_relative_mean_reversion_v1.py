#!/usr/bin/env python3
"""
Sector-Relative Mean Reversion Strategy — v1
==============================================

HYPOTHESIS:
  When a stock underperforms its own sector ETF significantly over 5-10 days
  (while the sector is stable or up), the stock-specific weakness may revert.
  This is different from absolute oversold (RSI) because it controls for
  market/sector moves — the stock is weak RELATIVE to peers, not just weak
  in a down market.

  The key insight: absolute oversold signals (RSI<20) fire mostly in broad
  drawdowns. Sector-relative underperformance isolates stock-specific weakness
  (earnings miss, analyst downgrade, fund rebalancing) which has stronger
  mean-reversion properties because the cause is often transient.

SIGNAL VARIANTS (12 total):
  - 3 underperformance thresholds: stock 5d return < sector 5d return by 3%, 5%, 7%
  - Extra filter: sector ETF 5d return > -2% (sector isn't crashing)
  - With/without vol compression overlay (vol_pctile < 15)
  - Hold periods: 5, 10, 21 days

SECTOR ETF MAPPING:
  XLK (Tech), XLF (Financials), XLV (Health Care), XLE (Energy),
  XLI (Industrials), XLC (Communication), XLY (Consumer Disc),
  XLP (Consumer Staples), XLU (Utilities), XLB (Materials), XLRE (Real Estate)

VALIDATION (HC #428 + HC #432):
  - Regime analysis: SPY close vs 200 SMA -> GREEN/RED
  - Regime gap test: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
  - Permutation test: 1000 shuffles, p < 0.05
  - Year-by-year breakdown

Usage:
    python3 research/strategies/sector_relative_mean_reversion_v1.py

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

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

COST_BPS_RT = 10  # 5 bps each way for equities
NUM_PERMUTATIONS = 1000
REGIME_GAP_THRESHOLD = 0.50
PERM_P_THRESHOLD = 0.05
MIN_TRADES = 30
LOOKBACK_REL = 5  # 5-day relative performance lookback
LOOKBACK_VOL_DAYS = 20
VOL_HISTORY_DAYS = 252
DATA_YEARS = 12

# Underperformance thresholds to test (stock return - sector return < -threshold)
UNDERPERF_THRESHOLDS = [0.03, 0.05, 0.07]

# Sector must not be crashing: sector 5d return > this
SECTOR_FLOOR = -0.02

# Vol compression percentile threshold (for filtered variants)
VOL_PCTILE_THRESH = 15

HOLD_PERIODS = [5, 10, 21]

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

OUTPUT_DIR = (
    Path(os.environ.get("LVL3_ROOT", "/home/nick/Lvl3Quant"))
    / "output"
    / "sector_relative_meanrev_v1"
)

# ---------------------------------------------------------------------------
# SECTOR ETF MAPPING
# ---------------------------------------------------------------------------
# GICS sector name -> sector ETF ticker
SECTOR_ETF_MAP = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
    "Real Estate": "XLRE",
}

ALL_SECTOR_ETFS = list(set(SECTOR_ETF_MAP.values()))


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def get_sp500_with_sectors():
    """Get S&P 500 tickers with their GICS sector from Wikipedia.
    Returns list of (ticker, sector_etf) tuples.
    """
    try:
        tables = pd.read_html(SP500_URL)
        df = tables[0]
        tickers_sectors = []
        unmapped = set()
        for _, row in df.iterrows():
            ticker = str(row["Symbol"]).replace(".", "-")
            sector = str(row.get("GICS Sector", ""))
            etf = SECTOR_ETF_MAP.get(sector)
            if etf:
                tickers_sectors.append((ticker, etf))
            else:
                unmapped.add(sector)
        if unmapped:
            print(f"[WARN] Unmapped sectors: {unmapped}")
        print(f"[INFO] Retrieved {len(tickers_sectors)} S&P 500 tickers with sector mapping")
        return tickers_sectors
    except Exception as e:
        print(f"[WARN] Failed to get S&P 500 from Wikipedia: {e}")
        print("[WARN] Falling back to hardcoded mapping")
        return _fallback_tickers_with_sectors()


def _fallback_tickers_with_sectors():
    """Hardcoded fallback: ~200 tickers with sector ETFs."""
    mapping = {
        "XLK": [
            "AAPL", "MSFT", "NVDA", "AVGO", "CSCO", "ACN", "TXN", "ORCL",
            "QCOM", "ADI", "INTC", "AMAT", "LRCX", "KLAC", "MCHP", "FTNT",
            "CDNS", "SNPS", "ANSS", "HPQ", "IBM", "NOW", "CRM", "ADBE",
            "INTU", "PYPL", "FISV", "FIS", "GPN", "ADP",
        ],
        "XLF": [
            "JPM", "V", "MA", "BRK-B", "BAC", "WFC", "GS", "MS", "SCHW",
            "BLK", "CB", "MMC", "AIG", "MET", "ALL", "TRV", "PRU", "AFL",
            "PNC", "USB", "CME", "ICE", "COF", "AXP", "DFS",
        ],
        "XLV": [
            "UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR",
            "BMY", "AMGN", "GILD", "SYK", "CI", "ELV", "ISRG", "REGN",
            "VRTX", "BSX", "BDX", "HUM", "MCK", "ZBH", "BAX", "DXCM",
        ],
        "XLE": [
            "XOM", "CVX", "SLB", "EOG", "COP", "OXY", "MPC", "PSX",
            "VLO", "HES", "FANG", "DVN", "APA", "MRO", "HAL", "BKR",
        ],
        "XLI": [
            "CAT", "GE", "HON", "UNP", "BA", "RTX", "DE", "UPS",
            "FDX", "WM", "NOC", "GD", "LHX", "ITW", "EMR", "ROK",
            "ETN", "PH", "CSX", "NSC", "IR", "DOV", "AME", "WAB",
        ],
        "XLC": [
            "GOOGL", "META", "DIS", "CMCSA", "NFLX", "T", "VZ",
            "TMUS", "CHTR", "EA", "TTWO", "MTCH", "OMC", "IPG",
        ],
        "XLY": [
            "AMZN", "TSLA", "HD", "MCD", "LOW", "SBUX", "NKE", "TJX",
            "CMG", "ORLY", "ROST", "MAR", "HLT", "F", "GM", "DHI",
            "LEN", "BKNG", "ABNB", "EXPE", "CCL", "RCL", "WYNN", "MGM",
        ],
        "XLP": [
            "PG", "PEP", "KO", "COST", "WMT", "PM", "MO", "MDLZ",
            "CL", "KMB", "GIS", "SJM", "HSY", "K", "STZ", "TAP",
            "KR", "SYY", "ADM", "TSN",
        ],
        "XLU": [
            "NEE", "SO", "DUK", "AEP", "D", "EXC", "SRE", "XEL",
            "WEC", "ED", "FE", "PEG", "ETR", "AES", "PPL", "CMS",
            "CNP", "ES", "NI", "EVRG", "ATO", "NRG", "AWK",
        ],
        "XLB": [
            "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "NUE",
            "VMC", "MLM", "PPG", "IFF", "CF", "MOS", "ALB",
        ],
        "XLRE": [
            "PLD", "AMT", "CCI", "EQIX", "PSA", "O", "SPG", "DLR",
            "WELL", "AVB", "EQR", "VTR", "ARE", "MAA", "UDR",
        ],
    }
    result = []
    for etf, tickers in mapping.items():
        for t in tickers:
            result.append((t, etf))
    return result


def compute_vol_percentile(prices: pd.Series, vol_window: int = 20,
                            hist_window: int = 252) -> pd.Series:
    """Realized vol percentile: current vol ranked in its own history."""
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
    """Permutation test: shuffle sign of returns, compare Sharpe."""
    actual_sharpe = annualized_sharpe(returns)
    count_ge = 0
    abs_returns = np.abs(returns)
    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(returns))
        shuffled = abs_returns * signs
        if annualized_sharpe(shuffled) >= actual_sharpe:
            count_ge += 1
    return count_ge / n_perms


def regime_gap(sharpe_green: float, sharpe_red: float) -> float:
    """Regime gap metric. REJECT if > 0.50."""
    denom = max(abs(sharpe_green), abs(sharpe_red))
    if denom == 0:
        return 0.0
    return abs(sharpe_green - sharpe_red) / denom


# ---------------------------------------------------------------------------
# DATA DOWNLOAD
# ---------------------------------------------------------------------------

def download_data(tickers_with_sectors: list, years: int = 12) -> tuple:
    """
    Download daily close data for all stocks, sector ETFs, and SPY.
    Returns:
      stock_closes: dict of {ticker: pd.Series (close prices)}
      sector_closes: dict of {etf: pd.Series (close prices)}
      spy_close: pd.Series
      ticker_to_sector: dict of {ticker: sector_etf}
    """
    import yfinance as yf

    end = dt.datetime.now()
    start = end - dt.timedelta(days=years * 365)

    stock_tickers = [t for t, _ in tickers_with_sectors]
    ticker_to_sector = {t: s for t, s in tickers_with_sectors}

    # Build full download list
    all_tickers = list(set(stock_tickers + ALL_SECTOR_ETFS + ["SPY"]))

    print(f"[INFO] Downloading {len(all_tickers)} tickers, {start.date()} to {end.date()}")
    t0 = time.time()

    batch_size = 50
    all_data = {}

    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i + batch_size]
        batch_str = " ".join(batch)
        try:
            data = yf.download(batch_str, start=start, end=end,
                               group_by="ticker", progress=False, threads=True)

            if len(batch) == 1:
                ticker = batch[0]
                if not data.empty:
                    close = data["Close"].squeeze()
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    all_data[ticker] = close.dropna()
            else:
                for ticker in batch:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            close = data[ticker]["Close"].squeeze()
                            if isinstance(close, pd.DataFrame):
                                close = close.iloc[:, 0]
                            close = close.dropna()
                            if len(close) > 252:
                                all_data[ticker] = close
                    except Exception:
                        pass
        except Exception as e:
            print(f"[WARN] Batch {i // batch_size + 1} failed: {e}")

        if (i // batch_size + 1) % 5 == 0:
            print(f"[INFO] Downloaded {min(i + batch_size, len(all_tickers))}/{len(all_tickers)} tickers...")

    elapsed = time.time() - t0
    print(f"[INFO] Downloaded {len(all_data)} tickers in {elapsed:.0f}s")

    # Separate into stocks, sector ETFs, SPY
    spy_close = all_data.pop("SPY", None)
    if spy_close is None:
        raise RuntimeError("Failed to download SPY data")

    sector_closes = {}
    for etf in ALL_SECTOR_ETFS:
        if etf in all_data:
            sector_closes[etf] = all_data.pop(etf)
        else:
            print(f"[WARN] Missing sector ETF: {etf}")

    # Filter stocks: must have sector ETF data available
    stock_closes = {}
    for ticker, close in all_data.items():
        sector_etf = ticker_to_sector.get(ticker)
        if sector_etf and sector_etf in sector_closes:
            stock_closes[ticker] = close

    print(f"[INFO] {len(stock_closes)} stocks with sector ETF mapping")
    print(f"[INFO] {len(sector_closes)} sector ETFs loaded")

    return stock_closes, sector_closes, spy_close, ticker_to_sector


# ---------------------------------------------------------------------------
# SIGNAL GENERATION
# ---------------------------------------------------------------------------

def generate_signals(stock_closes: dict, sector_closes: dict,
                     spy_close: pd.Series, ticker_to_sector: dict) -> pd.DataFrame:
    """
    For each stock on each date, compute:
    - stock 5-day return
    - sector ETF 5-day return
    - relative underperformance (stock_5d - sector_5d)
    - vol percentile
    - forward returns (5d, 10d, 21d)
    - regime (GREEN/RED)

    Returns DataFrame of all signal rows.
    """
    spy_regime = compute_regime(spy_close)

    all_signals = []
    processed = 0

    for ticker, close in stock_closes.items():
        try:
            sector_etf = ticker_to_sector[ticker]
            sector_close = sector_closes[sector_etf]

            # Align dates
            common_idx = close.index.intersection(sector_close.index)
            if len(common_idx) < 300:
                continue

            stk = close.reindex(common_idx)
            sec = sector_close.reindex(common_idx)

            # 5-day returns
            stk_5d_ret = stk / stk.shift(LOOKBACK_REL) - 1
            sec_5d_ret = sec / sec.shift(LOOKBACK_REL) - 1

            # Relative underperformance
            rel_underperf = stk_5d_ret - sec_5d_ret  # negative = stock lagging sector

            # Vol percentile
            vol_pctile = compute_vol_percentile(stk, LOOKBACK_VOL_DAYS, VOL_HISTORY_DAYS)

            # Forward returns
            fwd_5d = stk.shift(-5) / stk - 1
            fwd_10d = stk.shift(-10) / stk - 1
            fwd_21d = stk.shift(-21) / stk - 1

            sig = pd.DataFrame({
                "ticker": ticker,
                "sector_etf": sector_etf,
                "close": stk,
                "stk_5d_ret": stk_5d_ret,
                "sec_5d_ret": sec_5d_ret,
                "rel_underperf": rel_underperf,
                "vol_pctile": vol_pctile,
                "fwd_5d": fwd_5d,
                "fwd_10d": fwd_10d,
                "fwd_21d": fwd_21d,
            }, index=common_idx)

            sig["regime"] = sig.index.map(spy_regime)
            sig["year"] = sig.index.year

            sig = sig.dropna(subset=["stk_5d_ret", "sec_5d_ret", "rel_underperf"])
            all_signals.append(sig)
            processed += 1

        except Exception:
            pass

    print(f"[INFO] Computed signals for {processed} stocks")

    if not all_signals:
        raise RuntimeError("No signals generated")

    return pd.concat(all_signals, axis=0)


# ---------------------------------------------------------------------------
# VARIANT DEFINITIONS
# ---------------------------------------------------------------------------

def build_variants():
    """Build 12 variant definitions:
    3 thresholds x (with/without vol filter) x hold 5/10/21 = 3*2 * ... wait,
    actually: 3 thresholds x 4 combos: hold 5/10/21 and with/without vol.
    Let's do: 3 thresholds x 2 (vol filter on/off) x ... but we want 12 total
    per the spec: 3 thresholds x hold 5,10,21 = 9 base + 3 best-hold with vol = 12?

    Actually the spec says: 3 thresholds x 4 combos (with/without vol compression filter)
    = 12. But 3 x 4 doesn't work. Let me re-read:
    "3 thresholds x 4 combos: with/without vol compression filter"
    Probably: 3 thresholds x 2 (vol on/off) x ... hmm, that gives 6 per hold.

    I'll interpret as: 3 thresholds x 2 (vol filter on/off) = 6 signal variants,
    each tested at hold 5 and 10 (main holds) = 12. But spec also says hold 21.
    Let's do all: 3 x 2 x 3 = 18 variants. More data is better.
    """
    variants = []
    for thresh in UNDERPERF_THRESHOLDS:
        for use_vol in [False, True]:
            for hold in HOLD_PERIODS:
                pct = int(thresh * 100)
                vol_tag = "_VC" if use_vol else ""
                name = f"REL{pct}{vol_tag}_H{hold}"
                variants.append((name, thresh, use_vol, hold))
    return variants


# ---------------------------------------------------------------------------
# VARIANT EVALUATION
# ---------------------------------------------------------------------------

def evaluate_variant(name: str, underperf_thresh: float, use_vol_filter: bool,
                     hold_days: int, signals_df: pd.DataFrame) -> dict:
    """Evaluate one variant."""

    # Entry: stock underperforms sector by >= threshold
    mask = signals_df["rel_underperf"] < -underperf_thresh

    # Sector floor: sector isn't crashing
    mask &= signals_df["sec_5d_ret"] > SECTOR_FLOOR

    # Optional vol compression filter
    if use_vol_filter:
        mask &= signals_df["vol_pctile"] < VOL_PCTILE_THRESH

    trades = signals_df[mask].copy()

    fwd_col = f"fwd_{hold_days}d"
    if fwd_col not in trades.columns:
        return {"name": name, "status": "SKIP", "reason": f"No {fwd_col} column"}

    trades = trades.dropna(subset=[fwd_col])

    # Cost
    cost_frac = COST_BPS_RT / 10000
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
                "mean_ret_pct": float(round(yr_returns.mean() * 100, 2)),
                "sharpe": float(round(annualized_sharpe(yr_returns), 2)),
            }

    # Sector breakdown
    sector_stats = {}
    for etf in sorted(trades["sector_etf"].unique()):
        s_mask = trades["sector_etf"] == etf
        s_returns = returns[s_mask.values]
        if len(s_returns) >= 10:
            sector_stats[etf] = {
                "n_trades": int(len(s_returns)),
                "wr": float(round((s_returns > 0).mean(), 3)),
                "mean_ret_pct": float(round(s_returns.mean() * 100, 2)),
                "sharpe": float(round(annualized_sharpe(s_returns), 2)),
            }

    # Verdict
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
        "underperf_thresh_pct": float(underperf_thresh * 100),
        "vol_filter": use_vol_filter,
        "hold_days": hold_days,
        "n_trades": int(n_trades),
        "n_green": int(green_mask.sum()),
        "n_red": int(red_mask.sum()),
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
        "sector_breakdown": sector_stats,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print("=" * 80)
    print("SECTOR-RELATIVE MEAN REVERSION STRATEGY — v1")
    print("=" * 80)
    print()
    print("Hypothesis: stocks that underperform their sector ETF by N% over 5 days")
    print("(while the sector is stable/up) exhibit stronger mean reversion than")
    print("absolute oversold signals, because the weakness is stock-specific.")
    print()

    # 1. Get universe with sector mapping
    tickers_with_sectors = get_sp500_with_sectors()
    if len(tickers_with_sectors) < 100:
        print(f"[WARN] Only {len(tickers_with_sectors)} tickers mapped, expected 200+")

    # 2. Download data
    stock_closes, sector_closes, spy_close, ticker_to_sector = download_data(
        tickers_with_sectors, years=DATA_YEARS
    )

    if len(stock_closes) < 100:
        print(f"[WARN] Only {len(stock_closes)} stocks with data, expected 200+")

    # 3. Generate signals
    print("\n[INFO] Computing sector-relative signals...")
    t_sig = time.time()
    signals_df = generate_signals(stock_closes, sector_closes, spy_close, ticker_to_sector)
    print(f"[INFO] Signal computation done in {time.time() - t_sig:.0f}s")
    print(f"[INFO] Total signal rows: {len(signals_df):,}")

    # Stats on relative underperformance distribution
    rel = signals_df["rel_underperf"].dropna()
    print(f"[INFO] Relative underperformance distribution:")
    print(f"  Mean: {rel.mean():.4f}, Std: {rel.std():.4f}")
    print(f"  P5: {rel.quantile(0.05):.4f}, P10: {rel.quantile(0.10):.4f}")
    print(f"  P1: {rel.quantile(0.01):.4f}")
    print(f"  Signals < -3%: {(rel < -0.03).sum():,}")
    print(f"  Signals < -5%: {(rel < -0.05).sum():,}")
    print(f"  Signals < -7%: {(rel < -0.07).sum():,}")

    # Free memory
    del stock_closes, sector_closes

    # 4. Build and evaluate variants
    variants = build_variants()

    print(f"\n{'=' * 80}")
    print(f"EVALUATING {len(variants)} VARIANTS")
    print("=" * 80)

    results = []
    for name, thresh, use_vol, hold in variants:
        print(f"\n--- {name} (underperf>{thresh*100:.0f}%, vol_filter={use_vol}, hold={hold}d) ---")
        result = evaluate_variant(name, thresh, use_vol, hold, signals_df)
        results.append(result)

        if result.get("status") == "INSUFFICIENT_TRADES":
            print(f"  SKIP: Only {result['n_trades']} trades (need {MIN_TRADES})")
        elif result.get("status") == "SKIP":
            print(f"  SKIP: {result.get('reason', 'unknown')}")
        else:
            print(f"  N={result['n_trades']}, WR={result['wr']:.1%}, "
                  f"Mean={result['mean_ret_pct']:.2f}%, Sharpe={result['sharpe']:.2f}, "
                  f"PF={result['profit_factor']:.2f}")
            print(f"  Regime gap={result.get('regime_gap', 'N/A')}, "
                  f"Perm p={result['perm_p']:.4f}, "
                  f"Status={result['status']}")

    # 5. Summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print("=" * 80)

    header = (f"{'Variant':<18} {'N':>6} {'WR':>6} {'Mean%':>7} {'Sharpe':>7} "
              f"{'PF':>7} {'RGap':>6} {'Perm-p':>7} {'Pass':>5}")
    print(header)
    print("-" * len(header))

    for r in results:
        if r.get("status") in ("INSUFFICIENT_TRADES", "SKIP"):
            print(f"{r['name']:<18} {'SKIP':>6}  (insufficient trades: {r.get('n_trades', '?')})")
            continue

        rgap_str = f"{r['regime_gap']:.3f}" if r.get("regime_gap") is not None else "N/A"
        status = "YES" if r["status"] == "PASS" else "NO"

        print(f"{r['name']:<18} {r['n_trades']:>6} {r['wr']:>5.1%} "
              f"{r['mean_ret_pct']:>7.2f} {r['sharpe']:>7.2f} "
              f"{r['profit_factor']:>7.2f} {rgap_str:>6} {r['perm_p']:>7.4f} {status:>5}")

    # 6. Sector-level analysis for best variant
    passing = [r for r in results if r.get("status") == "PASS"]

    if passing:
        best = max(passing, key=lambda r: r["sharpe"])
        print(f"\n{'=' * 80}")
        print(f"BEST VARIANT SECTOR BREAKDOWN: {best['name']}")
        print("=" * 80)

        if best.get("sector_breakdown"):
            sec_header = f"{'Sector ETF':<12} {'N':>6} {'WR':>6} {'Mean%':>7} {'Sharpe':>7}"
            print(sec_header)
            print("-" * len(sec_header))
            for etf, stats in sorted(best["sector_breakdown"].items(),
                                      key=lambda x: x[1]["sharpe"], reverse=True):
                print(f"{etf:<12} {stats['n_trades']:>6} {stats['wr']:>5.1%} "
                      f"{stats['mean_ret_pct']:>7.2f} {stats['sharpe']:>7.2f}")

        # Year-by-year for best
        if best.get("yearly"):
            print(f"\nYear-by-year for {best['name']}:")
            yr_header = f"{'Year':<6} {'N':>6} {'WR':>6} {'Mean%':>7} {'Sharpe':>7}"
            print(yr_header)
            print("-" * len(yr_header))
            for year, stats in sorted(best["yearly"].items()):
                print(f"{year:<6} {stats['n_trades']:>6} {stats['wr']:>5.1%} "
                      f"{stats['mean_ret_pct']:>7.2f} {stats['sharpe']:>7.2f}")

    # 7. Vol compression value-add analysis
    print(f"\n{'=' * 80}")
    print("VOL COMPRESSION FILTER VALUE-ADD")
    print("=" * 80)

    for thresh in UNDERPERF_THRESHOLDS:
        pct = int(thresh * 100)
        for hold in HOLD_PERIODS:
            base_name = f"REL{pct}_H{hold}"
            vc_name = f"REL{pct}_VC_H{hold}"
            base = next((r for r in results if r["name"] == base_name), None)
            vc = next((r for r in results if r["name"] == vc_name), None)

            if (base and vc and
                base.get("status") not in ("INSUFFICIENT_TRADES", "SKIP") and
                vc.get("status") not in ("INSUFFICIENT_TRADES", "SKIP")):
                delta = vc["sharpe"] - base["sharpe"]
                print(f"  {vc_name} vs {base_name}: Sharpe {delta:+.2f} "
                      f"({vc['sharpe']:.2f} vs {base['sharpe']:.2f}), "
                      f"N: {vc['n_trades']} vs {base['n_trades']}")

    # 8. Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "results.json"

    output = {
        "timestamp": dt.datetime.now().isoformat(),
        "strategy": "sector_relative_mean_reversion_v1",
        "hypothesis": (
            "Stocks underperforming their sector ETF by N% over 5 days "
            "(with sector stable/up) revert due to stock-specific weakness."
        ),
        "universe": "S&P 500",
        "data_years": DATA_YEARS,
        "cost_bps_rt": COST_BPS_RT,
        "n_permutations": NUM_PERMUTATIONS,
        "regime_gap_threshold": REGIME_GAP_THRESHOLD,
        "perm_p_threshold": PERM_P_THRESHOLD,
        "sector_floor_5d": SECTOR_FLOOR,
        "underperf_lookback_days": LOOKBACK_REL,
        "results": results,
        "runtime_seconds": round(time.time() - t_start, 1),
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[INFO] Results saved to {results_path}")
    print(f"[INFO] Total runtime: {time.time() - t_start:.0f}s")

    # 9. Final verdicts
    print(f"\n{'=' * 80}")
    print("FINAL VERDICTS")
    print("=" * 80)

    failing = [r for r in results if r.get("status") == "FAIL"]
    skipped = [r for r in results if r.get("status") in ("INSUFFICIENT_TRADES", "SKIP")]

    print(f"\nVariants: {len(passing)} PASS, {len(failing)} FAIL, {len(skipped)} SKIP")

    if passing:
        best = max(passing, key=lambda r: r["sharpe"])
        print(f"\nBest variant: {best['name']}")
        print(f"  Underperf threshold: {best['underperf_thresh_pct']:.0f}%")
        print(f"  Vol filter: {best['vol_filter']}")
        print(f"  Hold: {best['hold_days']}d")
        print(f"  Sharpe={best['sharpe']:.2f}, WR={best['wr']:.1%}, "
              f"PF={best['profit_factor']:.2f}, N={best['n_trades']}")
        print(f"  Regime gap={best.get('regime_gap', 'N/A')}, Perm p={best['perm_p']:.4f}")
        print(f"  Sharpe GREEN={best.get('sharpe_green', 'N/A')}, "
              f"RED={best.get('sharpe_red', 'N/A')}")
    else:
        print("\nNo variant passed all gates.")
        # Show best failing for insight
        valid = [r for r in results
                 if r.get("status") not in ("INSUFFICIENT_TRADES", "SKIP")]
        if valid:
            best_fail = max(valid, key=lambda r: r["sharpe"])
            print(f"\nBest failing variant: {best_fail['name']}")
            print(f"  Sharpe={best_fail['sharpe']:.2f}, WR={best_fail['wr']:.1%}, "
                  f"PF={best_fail['profit_factor']:.2f}")
            print(f"  Regime gap={best_fail.get('regime_gap', 'N/A')}, "
                  f"Perm p={best_fail['perm_p']:.4f}")
            reasons = []
            if best_fail.get("regime_pass") is False:
                reasons.append("regime gap too high")
            if not best_fail.get("perm_pass"):
                reasons.append("permutation test failed")
            if best_fail["sharpe"] <= 0:
                reasons.append("negative Sharpe")
            print(f"  Failure reasons: {', '.join(reasons)}")

    print("\nDone.")
    return output


if __name__ == "__main__":
    main()
