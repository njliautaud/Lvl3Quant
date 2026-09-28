#!/usr/bin/env python3
"""
Momentum V2 Backtester — Production Quality
============================================
Expanded universe (S&P 500 + S&P 400 MidCap), dynamic exits, walk-forward,
regime-agnostic validation.

HC #684 exit rules:
  - Trailing stop: 2x ATR(20)
  - Momentum breakdown: 10-day returns negative 3 consecutive days
  - Volume dry-up: volume < 50% of 20-day avg for 3 consecutive days

Walk-forward: 3-year train, 1-year OOT, sliding windows.
Regime test: bull/bear/flat by SPY monthly returns, gap < 0.50.
"""

import os
import sys
import json
import time
import logging
import warnings
import datetime as dt
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import requests
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/momentum_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "backtest.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# Backtest parameters
START_DATE = "2015-01-01"
END_DATE = "2026-07-01"
LOOKBACK_PERIODS = [63, 126, 252]  # ~3mo, 6mo, 12mo trading days
SKIP_LAST_MONTH = 21  # Jegadeesh-Titman: skip most recent month
TOP_N_STOCKS = 50  # portfolio size
REBALANCE_FREQ = 21  # monthly (~21 trading days)

# Exit parameters (HC #684)
ATR_PERIOD = 20
ATR_TRAILING_MULT = 2.0
MOMENTUM_BREAKDOWN_WINDOW = 10
MOMENTUM_BREAKDOWN_CONSEC = 3
VOLUME_DRYUP_RATIO = 0.50
VOLUME_DRYUP_CONSEC = 3

# Walk-forward
WF_TRAIN_YEARS = 3
WF_TEST_YEARS = 1

# Position sizing modes
SIZING_MODES = ["equal_weight", "inverse_vol"]

# Transaction costs
COMMISSION_BPS = 5  # 5 bps round-trip for equities
SLIPPAGE_BPS = 5    # 5 bps slippage estimate


# ── Universe Construction ──────────────────────────────────────────────────

def _wiki_read_html(url: str) -> list:
    """Read HTML tables from Wikipedia with proper headers to avoid 403."""
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return pd.read_html(resp.text)


def get_sp500_tickers() -> list:
    """Pull S&P 500 tickers from Wikipedia."""
    try:
        tables = _wiki_read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        log.info(f"S&P 500: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        log.error(f"Failed to get S&P 500 tickers: {e}")
        return []


def get_sp400_tickers() -> list:
    """Pull S&P 400 MidCap tickers from Wikipedia."""
    try:
        tables = _wiki_read_html("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")
        df = tables[0]
        # Column name varies — try common names
        for col in ["Symbol", "Ticker symbol", "Ticker"]:
            if col in df.columns:
                tickers = df[col].str.replace(".", "-", regex=False).tolist()
                log.info(f"S&P 400: {len(tickers)} tickers")
                return tickers
        log.warning(f"S&P 400 columns: {df.columns.tolist()}")
        return []
    except Exception as e:
        log.error(f"Failed to get S&P 400 tickers: {e}")
        return []


def _extract_ticker_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Extract single-ticker OHLCV from yfinance MultiIndex DataFrame."""
    cols_needed = ["Open", "High", "Low", "Close", "Volume"]
    if df.columns.nlevels == 2:
        # MultiIndex columns — check both orderings
        level_names = [n for n in df.columns.names]
        level0_vals = df.columns.get_level_values(0).unique().tolist()

        if ticker in level0_vals:
            # Ticker is level 0 (group_by='ticker' format)
            sub = df[ticker]
        else:
            # Ticker might be level 1 (default format for single ticker)
            try:
                sub = df.xs(ticker, level=1, axis=1)
            except KeyError:
                # Try the other level
                sub = df.xs(ticker, level="Ticker", axis=1) if "Ticker" in level_names else None

        if sub is not None:
            # Flatten any remaining MultiIndex
            if sub.columns.nlevels > 1:
                sub.columns = sub.columns.get_level_values(0)
            out = pd.DataFrame(index=sub.index)
            for c in cols_needed:
                if c in sub.columns:
                    out[c] = sub[c]
            return out.dropna(how="all")
    else:
        # Simple columns
        if all(c in df.columns for c in cols_needed):
            return df[cols_needed].copy().dropna(how="all")
    return pd.DataFrame()


def download_price_data(tickers: list, start: str, end: str) -> dict:
    """Download daily OHLCV data for all tickers using yfinance batch download.
    Returns dict of ticker -> DataFrame.
    """
    cache_file = OUTPUT_DIR / "price_cache.pkl"
    if cache_file.exists():
        log.info("Loading cached price data...")
        data = pd.read_pickle(cache_file)
        return data

    log.info(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")

    # Download in batches to avoid timeout
    batch_size = 50
    all_data = {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        batch_str = " ".join(batch)
        log.info(f"  Batch {i // batch_size + 1}/{(len(tickers) + batch_size - 1) // batch_size} ({len(batch)} tickers)...")
        try:
            df = yf.download(batch_str, start=start, end=end, group_by="ticker",
                             auto_adjust=True, threads=True, progress=False)
            if df.empty:
                continue

            # Detect tickers present in the download
            if df.columns.nlevels == 2:
                # Figure out which level has tickers
                l0 = df.columns.get_level_values(0).unique().tolist()
                l1 = df.columns.get_level_values(1).unique().tolist()
                # Tickers are whichever level has more unique values
                if len(l0) > len(l1):
                    present_tickers = [t for t in batch if t in l0]
                else:
                    present_tickers = [t for t in batch if t in l1]
                # Fallback: try both
                if not present_tickers:
                    present_tickers = [t for t in batch if t in l0 or t in l1]
            else:
                present_tickers = batch[:1] if len(batch) == 1 else []

            for ticker in (present_tickers if present_tickers else batch):
                try:
                    ticker_df = _extract_ticker_df(df, ticker)
                    if len(ticker_df) > 252:
                        all_data[ticker] = ticker_df
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"  Batch download failed: {e}")
        time.sleep(0.3)

    log.info(f"Downloaded data for {len(all_data)} tickers with sufficient history")

    # Also get SPY for regime classification
    if "SPY" not in all_data:
        spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
        if not spy.empty:
            spy_df = _extract_ticker_df(spy, "SPY")
            if len(spy_df) > 0:
                all_data["SPY"] = spy_df

    # Cache
    pd.to_pickle(all_data, cache_file)
    log.info(f"Cached price data to {cache_file}")

    return all_data


# ── Feature Computation ───────────────────────────────────────────────────

def compute_momentum_score(close: pd.Series, lookback: int, skip: int = 21) -> pd.Series:
    """Compute momentum score: return over [t-lookback, t-skip]."""
    if len(close) < lookback + skip:
        return pd.Series(np.nan, index=close.index)
    lagged_close = close.shift(skip)
    far_close = close.shift(lookback)
    return (lagged_close / far_close) - 1.0


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_signals_for_ticker(args):
    """Compute momentum scores and exit features for a single ticker.
    Used in multiprocessing.
    """
    ticker, df, lookback_periods = args
    try:
        close = df["Close"].squeeze() if isinstance(df["Close"], pd.DataFrame) else df["Close"]
        high = df["High"].squeeze() if isinstance(df["High"], pd.DataFrame) else df["High"]
        low = df["Low"].squeeze() if isinstance(df["Low"], pd.DataFrame) else df["Low"]
        volume = df["Volume"].squeeze() if isinstance(df["Volume"], pd.DataFrame) else df["Volume"]

        result = pd.DataFrame(index=df.index)
        result["close"] = close
        result["high"] = high
        result["low"] = low
        result["volume"] = volume

        # Momentum scores for each lookback
        for lb in lookback_periods:
            result[f"mom_{lb}"] = compute_momentum_score(close, lb, SKIP_LAST_MONTH)

        # ATR for trailing stop
        result["atr"] = compute_atr(high, low, close, ATR_PERIOD)

        # 20-day realized volatility for inverse-vol sizing
        result["vol_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)

        # 20-day average volume
        result["vol_avg_20d"] = volume.rolling(20).mean()

        # 10-day returns for momentum breakdown check
        result["ret_10d"] = close.pct_change(10)

        # Daily returns
        result["daily_ret"] = close.pct_change()

        return ticker, result
    except Exception as e:
        return ticker, None


# ── Exit Logic ─────────────────────────────────────────────────────────────

class DynamicExitManager:
    """Manages daily exit checks for active positions."""

    def __init__(self):
        self.trailing_highs = {}  # ticker -> highest close since entry
        self.momentum_neg_streak = {}  # ticker -> consecutive days of negative 10d ret
        self.volume_low_streak = {}  # ticker -> consecutive days of low volume

    def init_position(self, ticker: str, entry_price: float):
        self.trailing_highs[ticker] = entry_price
        self.momentum_neg_streak[ticker] = 0
        self.volume_low_streak[ticker] = 0

    def remove_position(self, ticker: str):
        self.trailing_highs.pop(ticker, None)
        self.momentum_neg_streak.pop(ticker, None)
        self.volume_low_streak.pop(ticker, None)

    def check_exits(self, ticker: str, row: dict) -> tuple:
        """Check if position should be exited.
        Returns (should_exit: bool, reason: str).
        """
        close = row.get("close", np.nan)
        atr = row.get("atr", np.nan)
        ret_10d = row.get("ret_10d", np.nan)
        volume = row.get("volume", np.nan)
        vol_avg = row.get("vol_avg_20d", np.nan)

        if ticker not in self.trailing_highs:
            return False, ""

        # Update trailing high
        if not np.isnan(close):
            self.trailing_highs[ticker] = max(self.trailing_highs[ticker], close)

        # 1. Trailing stop: 2x ATR below trailing high
        if not np.isnan(atr) and atr > 0:
            stop_level = self.trailing_highs[ticker] - ATR_TRAILING_MULT * atr
            if close < stop_level:
                return True, "trailing_stop"

        # 2. Momentum breakdown: 10-day returns negative for N consecutive days
        if not np.isnan(ret_10d):
            if ret_10d < 0:
                self.momentum_neg_streak[ticker] = self.momentum_neg_streak.get(ticker, 0) + 1
            else:
                self.momentum_neg_streak[ticker] = 0
            if self.momentum_neg_streak.get(ticker, 0) >= MOMENTUM_BREAKDOWN_CONSEC:
                return True, "momentum_breakdown"

        # 3. Volume dry-up: volume < 50% of 20-day avg for N consecutive days
        if not np.isnan(volume) and not np.isnan(vol_avg) and vol_avg > 0:
            if volume < VOLUME_DRYUP_RATIO * vol_avg:
                self.volume_low_streak[ticker] = self.volume_low_streak.get(ticker, 0) + 1
            else:
                self.volume_low_streak[ticker] = 0
            if self.volume_low_streak.get(ticker, 0) >= VOLUME_DRYUP_CONSEC:
                return True, "volume_dryup"

        return False, ""


# ── Backtester Core ────────────────────────────────────────────────────────

def run_backtest(
    signals: dict,
    trading_dates: pd.DatetimeIndex,
    lookback: int,
    sizing: str = "equal_weight",
    top_n: int = TOP_N_STOCKS,
    cost_bps: float = COMMISSION_BPS + SLIPPAGE_BPS,
) -> pd.DataFrame:
    """Run the momentum backtest with dynamic exits.

    Args:
        signals: dict of ticker -> DataFrame with features
        trading_dates: date index to iterate over
        lookback: momentum lookback period
        sizing: 'equal_weight' or 'inverse_vol'
        top_n: number of stocks to hold
        cost_bps: total round-trip cost in bps

    Returns:
        DataFrame with daily portfolio returns and metadata
    """
    mom_col = f"mom_{lookback}"
    exit_mgr = DynamicExitManager()

    # Current portfolio: {ticker: weight}
    portfolio = {}
    portfolio_entry_prices = {}

    results = []
    last_rebalance_idx = -REBALANCE_FREQ  # force first rebalance
    exit_counts = {"trailing_stop": 0, "momentum_breakdown": 0, "volume_dryup": 0, "rebalance": 0}

    for i, date in enumerate(trading_dates):
        day_returns = {}
        day_exit_tickers = set()

        # ── Daily exit checks for held positions ──
        for ticker in list(portfolio.keys()):
            if ticker not in signals or date not in signals[ticker].index:
                continue
            row = signals[ticker].loc[date]
            row_dict = {
                "close": row.get("close", np.nan),
                "atr": row.get("atr", np.nan),
                "ret_10d": row.get("ret_10d", np.nan),
                "volume": row.get("volume", np.nan),
                "vol_avg_20d": row.get("vol_avg_20d", np.nan),
            }

            # Check exits
            should_exit, reason = exit_mgr.check_exits(ticker, row_dict)
            if should_exit:
                day_exit_tickers.add(ticker)
                exit_counts[reason] = exit_counts.get(reason, 0) + 1

        # Remove exited positions
        for ticker in day_exit_tickers:
            exit_mgr.remove_position(ticker)
            del portfolio[ticker]
            portfolio_entry_prices.pop(ticker, None)

        # ── Monthly rebalance ──
        is_rebalance = (i - last_rebalance_idx) >= REBALANCE_FREQ

        if is_rebalance:
            # Score all tickers
            scores = {}
            vols = {}
            for ticker, sig_df in signals.items():
                if ticker == "SPY":
                    continue
                if date not in sig_df.index:
                    continue
                row = sig_df.loc[date]
                score = row.get(mom_col, np.nan)
                vol = row.get("vol_20d", np.nan)
                close = row.get("close", np.nan)
                if not np.isnan(score) and not np.isnan(close) and close > 5.0:  # penny stock filter
                    scores[ticker] = score
                    vols[ticker] = vol if not np.isnan(vol) and vol > 0 else 0.30  # default vol

            tc = 0.0
            turnover_weight = 0.0

            if len(scores) >= top_n:
                # Select top N by momentum score
                sorted_tickers = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
                new_portfolio_tickers = sorted_tickers[:top_n]

                # Compute weights
                if sizing == "inverse_vol":
                    inv_vols = {t: 1.0 / vols[t] for t in new_portfolio_tickers}
                    total_inv_vol = sum(inv_vols.values())
                    new_weights = {t: inv_vols[t] / total_inv_vol for t in new_portfolio_tickers}
                else:
                    w = 1.0 / top_n
                    new_weights = {t: w for t in new_portfolio_tickers}

                # Compute turnover for cost
                old_tickers = set(portfolio.keys())
                new_tickers = set(new_portfolio_tickers)
                exited = old_tickers - new_tickers
                entered = new_tickers - old_tickers

                # Count rebalanced exits
                for t in exited:
                    exit_mgr.remove_position(t)
                    exit_counts["rebalance"] += 1

                turnover_weight = 0.0
                for t in exited:
                    turnover_weight += portfolio.get(t, 0)
                for t in entered:
                    turnover_weight += new_weights.get(t, 0)
                # Weight changes for kept positions
                for t in old_tickers & new_tickers:
                    turnover_weight += abs(new_weights.get(t, 0) - portfolio.get(t, 0))

                # Initialize new entries in exit manager
                for t in entered:
                    if t in signals and date in signals[t].index:
                        entry_price = signals[t].loc[date].get("close", 0)
                        exit_mgr.init_position(t, entry_price)
                        portfolio_entry_prices[t] = entry_price

                # Update portfolio to new weights (but apply returns AFTER)
                portfolio = new_weights
                last_rebalance_idx = i

                # Transaction cost for this day
                tc = turnover_weight * cost_bps / 10000.0
        else:
            tc = 0.0
            turnover_weight = 0.0

        # ── Compute portfolio return for this day ──
        port_ret = 0.0
        active_weight = 0.0
        for ticker, weight in portfolio.items():
            if ticker not in signals or date not in signals[ticker].index:
                continue
            daily_r = signals[ticker].loc[date].get("daily_ret", 0.0)
            if np.isnan(daily_r):
                daily_r = 0.0
            port_ret += weight * daily_r
            active_weight += weight

        # Normalize if some positions have no data
        if active_weight > 0 and active_weight < 0.5:
            port_ret = port_ret / active_weight

        # Subtract transaction costs
        net_ret = port_ret - tc

        results.append({
            "date": date,
            "gross_return": port_ret,
            "net_return": net_ret,
            "tc": tc,
            "n_positions": len(portfolio),
            "turnover": turnover_weight if is_rebalance else 0.0,
        })

    results_df = pd.DataFrame(results).set_index("date")
    results_df.attrs["exit_counts"] = exit_counts
    return results_df


# ── Performance Metrics ────────────────────────────────────────────────────

def compute_metrics(returns: pd.Series, rf_annual: float = 0.04) -> dict:
    """Compute standard performance metrics."""
    if len(returns) < 30:
        return {}

    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    excess = returns - rf_daily

    ann_ret = (1 + returns.mean()) ** 252 - 1
    ann_vol = returns.std() * np.sqrt(252)

    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    downside = excess[excess < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-9
    sortino = excess.mean() * 252 / downside_vol if downside_vol > 0 else 0

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # CAGR
    n_years = len(returns) / 252
    total_ret = cum.iloc[-1]
    cagr = total_ret ** (1 / n_years) - 1 if n_years > 0 else 0

    # Win rate
    wr = (returns > 0).mean()

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "CAGR": round(cagr * 100, 2),
        "Ann_Return": round(ann_ret * 100, 2),
        "Ann_Vol": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "MaxDD": round(max_dd * 100, 2),
        "Calmar": round(calmar, 3),
        "WinRate": round(wr * 100, 2),
        "ProfitFactor": round(pf, 3),
        "TotalReturn": round((total_ret - 1) * 100, 2),
        "N_Days": len(returns),
    }


# ── Regime Classification ─────────────────────────────────────────────────

def classify_regimes(spy_data: pd.DataFrame) -> pd.DataFrame:
    """Classify each month as bull/bear/flat by SPY returns."""
    spy_close = spy_data["Close"].squeeze() if isinstance(spy_data["Close"], pd.DataFrame) else spy_data["Close"]
    monthly = spy_close.resample("ME").last().pct_change()

    regimes = []
    for date, ret in monthly.items():
        if np.isnan(ret):
            regime = "flat"
        elif ret > 0.02:
            regime = "bull"
        elif ret < -0.02:
            regime = "bear"
        else:
            regime = "flat"
        regimes.append({"date": date, "regime": regime, "spy_ret": ret})

    return pd.DataFrame(regimes).set_index("date")


def regime_stratified_sharpe(returns: pd.Series, regime_df: pd.DataFrame) -> dict:
    """Compute Sharpe per regime. Returns dict with sharpes and gap test."""
    # Map each trading day to its month-end regime
    returns_df = returns.to_frame("ret")
    returns_df["month_end"] = returns_df.index.to_period("M").to_timestamp("ME")

    merged = returns_df.merge(regime_df[["regime"]], left_on="month_end", right_index=True, how="left")
    merged["regime"] = merged["regime"].fillna("flat")

    sharpes = {}
    for regime in ["bull", "bear", "flat"]:
        mask = merged["regime"] == regime
        if mask.sum() > 30:
            r = merged.loc[mask, "ret"]
            s = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0
            sharpes[regime] = round(s, 3)
        else:
            sharpes[regime] = None

    # Gap test: |best - worst| / max(|best|, |worst|) < 0.50
    valid_sharpes = [v for v in sharpes.values() if v is not None]
    if len(valid_sharpes) >= 2:
        best = max(valid_sharpes)
        worst = min(valid_sharpes)
        denom = max(abs(best), abs(worst))
        gap = abs(best - worst) / denom if denom > 0 else 0
    else:
        gap = None

    return {
        "sharpe_bull": sharpes.get("bull"),
        "sharpe_bear": sharpes.get("bear"),
        "sharpe_flat": sharpes.get("flat"),
        "regime_gap": round(gap, 3) if gap is not None else None,
        "regime_pass": gap < 0.50 if gap is not None else None,
    }


# ── Walk-Forward Validation ───────────────────────────────────────────────

def walk_forward_backtest(
    signals: dict,
    all_dates: pd.DatetimeIndex,
    lookback: int,
    sizing: str,
    spy_data: pd.DataFrame,
) -> dict:
    """Run walk-forward with 3yr train / 1yr OOT sliding windows."""
    # Determine year boundaries
    years = sorted(all_dates.year.unique())
    min_year = years[0]
    max_year = years[-1]

    wf_results = []
    all_oot_returns = []

    for test_start_year in range(min_year + WF_TRAIN_YEARS, max_year + 1):
        train_end_year = test_start_year - 1
        train_start_year = test_start_year - WF_TRAIN_YEARS

        test_end_year = test_start_year + WF_TEST_YEARS - 1
        if test_end_year > max_year:
            break

        train_start = pd.Timestamp(f"{train_start_year}-01-01")
        train_end = pd.Timestamp(f"{train_end_year}-12-31")
        test_start = pd.Timestamp(f"{test_start_year}-01-01")
        test_end = pd.Timestamp(f"{test_end_year}-12-31")

        # In walk-forward for momentum, "training" = selecting optimal lookback/params
        # For now we use fixed lookback, so train phase just validates the lookback works
        train_dates = all_dates[(all_dates >= train_start) & (all_dates <= train_end)]
        test_dates = all_dates[(all_dates >= test_start) & (all_dates <= test_end)]

        if len(test_dates) < 100:
            continue

        # Run backtest on OOT period
        oot_returns = run_backtest(signals, test_dates, lookback, sizing)

        metrics = compute_metrics(oot_returns["net_return"])
        metrics["test_period"] = f"{test_start_year}"
        metrics["train_period"] = f"{train_start_year}-{train_end_year}"
        wf_results.append(metrics)

        all_oot_returns.append(oot_returns["net_return"])

        log.info(
            f"  WF {train_start_year}-{train_end_year} -> {test_start_year}: "
            f"Sharpe={metrics.get('Sharpe', 'N/A')}, CAGR={metrics.get('CAGR', 'N/A')}%, "
            f"MaxDD={metrics.get('MaxDD', 'N/A')}%"
        )

    # Concatenate all OOT returns
    if all_oot_returns:
        concat_oot = pd.concat(all_oot_returns)
        # Remove duplicates (overlapping windows)
        concat_oot = concat_oot[~concat_oot.index.duplicated(keep="first")]
        concat_oot = concat_oot.sort_index()
        overall_oot_metrics = compute_metrics(concat_oot)
    else:
        overall_oot_metrics = {}
        concat_oot = pd.Series(dtype=float)

    return {
        "wf_folds": wf_results,
        "overall_oot_metrics": overall_oot_metrics,
        "oot_returns": concat_oot,
    }


# ── SPY Benchmark ─────────────────────────────────────────────────────────

def compute_spy_benchmark(spy_data: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """Buy-and-hold SPY performance over the same period."""
    spy_close = spy_data["Close"].squeeze() if isinstance(spy_data["Close"], pd.DataFrame) else spy_data["Close"]
    spy_ret = spy_close.pct_change().reindex(dates).fillna(0)
    return compute_metrics(spy_ret)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("MOMENTUM V2 BACKTESTER — Starting")
    log.info("=" * 70)

    # 1. Get universe
    log.info("Step 1: Building universe...")
    sp500 = get_sp500_tickers()
    sp400 = get_sp400_tickers()
    all_tickers = list(set(sp500 + sp400))
    # Always include SPY for regime classification
    if "SPY" not in all_tickers:
        all_tickers.append("SPY")
    log.info(f"Total universe: {len(all_tickers)} unique tickers")

    # 2. Download data
    log.info("Step 2: Downloading price data...")
    price_data = download_price_data(all_tickers, START_DATE, END_DATE)
    log.info(f"Got data for {len(price_data)} tickers")

    if len(price_data) < 100:
        log.error("Insufficient data. Aborting.")
        return

    # 3. Compute signals (parallel)
    log.info("Step 3: Computing momentum signals and exit features...")
    args_list = [(ticker, df, LOOKBACK_PERIODS) for ticker, df in price_data.items()]

    n_workers = min(cpu_count(), 12)
    with Pool(n_workers) as pool:
        results = pool.map(compute_signals_for_ticker, args_list)

    signals = {}
    for ticker, sig_df in results:
        if sig_df is not None and len(sig_df) > 252:
            signals[ticker] = sig_df

    log.info(f"Computed signals for {len(signals)} tickers")

    # Get common trading dates
    all_dates_sets = [set(df.index) for df in signals.values()]
    if not all_dates_sets:
        log.error("No signals computed. Aborting.")
        return

    # Use SPY dates as reference
    if "SPY" in signals:
        trading_dates = signals["SPY"].index.sort_values()
    else:
        # Fallback: intersection of most common dates
        from collections import Counter
        date_counts = Counter()
        for dates_set in all_dates_sets:
            for d in dates_set:
                date_counts[d] += 1
        # Keep dates that appear in at least 50% of tickers
        threshold = len(signals) * 0.5
        common_dates = [d for d, c in date_counts.items() if c >= threshold]
        trading_dates = pd.DatetimeIndex(sorted(common_dates))

    log.info(f"Trading dates: {trading_dates[0].date()} to {trading_dates[-1].date()} ({len(trading_dates)} days)")

    # Get SPY data for regime classification
    spy_data = price_data.get("SPY", signals.get("SPY"))
    if spy_data is None:
        log.error("No SPY data for regime classification. Aborting.")
        return

    regime_df = classify_regimes(spy_data)
    log.info(f"Regime classification: {regime_df['regime'].value_counts().to_dict()}")

    # 4. Run backtests for all combinations
    log.info("Step 4: Running backtests...")
    all_results = {}
    summary_rows = []

    for lookback in LOOKBACK_PERIODS:
        for sizing in SIZING_MODES:
            config_name = f"mom{lookback}d_{sizing}"
            log.info(f"\n{'='*50}")
            log.info(f"Config: {config_name}")
            log.info(f"{'='*50}")

            # Full-period backtest
            log.info("  Running full-period backtest...")
            full_results = run_backtest(signals, trading_dates, lookback, sizing)
            full_metrics = compute_metrics(full_results["net_return"])
            exit_counts = full_results.attrs.get("exit_counts", {})
            log.info(f"  Full period: Sharpe={full_metrics.get('Sharpe')}, "
                     f"CAGR={full_metrics.get('CAGR')}%, MaxDD={full_metrics.get('MaxDD')}%")
            log.info(f"  Exit counts: {exit_counts}")

            # Regime-stratified analysis
            regime_results = regime_stratified_sharpe(full_results["net_return"], regime_df)
            log.info(f"  Regime Sharpe: bull={regime_results['sharpe_bull']}, "
                     f"bear={regime_results['sharpe_bear']}, flat={regime_results['sharpe_flat']}")
            log.info(f"  Regime gap: {regime_results['regime_gap']} "
                     f"({'PASS' if regime_results['regime_pass'] else 'FAIL'})")

            # Walk-forward
            log.info("  Running walk-forward validation...")
            wf = walk_forward_backtest(signals, trading_dates, lookback, sizing, spy_data)
            log.info(f"  WF Overall OOT: {wf['overall_oot_metrics']}")

            # Regime test on OOT returns
            if len(wf["oot_returns"]) > 0:
                wf_regime = regime_stratified_sharpe(wf["oot_returns"], regime_df)
                log.info(f"  WF OOT Regime gap: {wf_regime['regime_gap']} "
                         f"({'PASS' if wf_regime['regime_pass'] else 'FAIL'})")
            else:
                wf_regime = {}

            # SPY benchmark over same dates
            spy_bench = compute_spy_benchmark(spy_data, trading_dates)

            # Store results
            config_result = {
                "config": config_name,
                "lookback": lookback,
                "sizing": sizing,
                "full_period": full_metrics,
                "exit_counts": exit_counts,
                "regime": regime_results,
                "wf_overall_oot": wf["overall_oot_metrics"],
                "wf_oot_regime": wf_regime,
                "wf_folds": wf["wf_folds"],
                "spy_benchmark": spy_bench,
            }
            all_results[config_name] = config_result

            # Save equity curve
            eq_curve = (1 + full_results["net_return"]).cumprod()
            eq_curve.to_csv(OUTPUT_DIR / f"equity_{config_name}.csv")

            # Summary row
            summary_rows.append({
                "Config": config_name,
                "CAGR%": full_metrics.get("CAGR"),
                "Sharpe": full_metrics.get("Sharpe"),
                "Sortino": full_metrics.get("Sortino"),
                "MaxDD%": full_metrics.get("MaxDD"),
                "WR%": full_metrics.get("WinRate"),
                "PF": full_metrics.get("ProfitFactor"),
                "RegimeGap": regime_results.get("regime_gap"),
                "RegimePass": regime_results.get("regime_pass"),
                "WF_OOT_Sharpe": wf["overall_oot_metrics"].get("Sharpe"),
                "WF_OOT_CAGR%": wf["overall_oot_metrics"].get("CAGR"),
                "SPY_CAGR%": spy_bench.get("CAGR"),
                "SPY_Sharpe": spy_bench.get("Sharpe"),
                "Exits_TrailingStop": exit_counts.get("trailing_stop", 0),
                "Exits_MomBreakdown": exit_counts.get("momentum_breakdown", 0),
                "Exits_VolDryup": exit_counts.get("volume_dryup", 0),
                "Exits_Rebalance": exit_counts.get("rebalance", 0),
            })

    # 5. Print summary
    log.info("\n" + "=" * 80)
    log.info("SUMMARY — ALL CONFIGURATIONS")
    log.info("=" * 80)

    summary_df = pd.DataFrame(summary_rows)
    log.info("\n" + summary_df.to_string(index=False))

    # Best config by WF OOT Sharpe (regime-passing only)
    passing = summary_df[summary_df["RegimePass"] == True]
    if len(passing) > 0:
        best_idx = passing["WF_OOT_Sharpe"].idxmax()
        best = passing.loc[best_idx]
        log.info(f"\nBEST REGIME-PASSING CONFIG: {best['Config']}")
        log.info(f"  WF OOT Sharpe: {best['WF_OOT_Sharpe']}")
        log.info(f"  Full-period CAGR: {best['CAGR%']}%")
        log.info(f"  Regime gap: {best['RegimeGap']}")
    else:
        log.info("\nNO CONFIG PASSED REGIME TEST (gap < 0.50)")
        best_idx = summary_df["WF_OOT_Sharpe"].idxmax() if len(summary_df) > 0 else None
        if best_idx is not None:
            best = summary_df.loc[best_idx]
            log.info(f"Best by WF OOT Sharpe (FAILING regime): {best['Config']}, "
                     f"Sharpe={best['WF_OOT_Sharpe']}, gap={best['RegimeGap']}")

    # 6. Save results
    log.info("\nSaving results...")

    # Summary JSON
    summary_json = {
        "run_timestamp": dt.datetime.now().isoformat(),
        "universe_size": len(signals),
        "date_range": f"{trading_dates[0].date()} to {trading_dates[-1].date()}",
        "n_trading_days": len(trading_dates),
        "configs": all_results,
        "regime_distribution": regime_df["regime"].value_counts().to_dict(),
    }

    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_types(i) for i in obj]
        return obj

    summary_json = convert_types(summary_json)

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2, default=str)

    summary_df.to_csv(OUTPUT_DIR / "summary_table.csv", index=False)

    elapsed = time.time() - t0
    log.info(f"\nCompleted in {elapsed / 60:.1f} minutes")
    log.info(f"Results saved to {OUTPUT_DIR}")

    # Print final key metrics to stdout
    print("\n" + "=" * 80)
    print("MOMENTUM V2 BACKTEST RESULTS")
    print("=" * 80)
    print(f"\nUniverse: {len(signals)} stocks | Period: {trading_dates[0].date()} to {trading_dates[-1].date()}")
    print(f"\n{summary_df.to_string(index=False)}")
    if len(passing) > 0:
        print(f"\nBest regime-passing config: {best['Config']}")
    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
