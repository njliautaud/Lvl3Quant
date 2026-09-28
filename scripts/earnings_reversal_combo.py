#!/usr/bin/env python3
"""
Earnings Reversal + Sector-Relative Mean Reversion COMBO Backtester
====================================================================
Combines two strongest signals: earnings events + sector-relative oversold bounces.

Core insight: After earnings, stocks gap significantly. If a stock gaps DOWN more
than its sector, that idiosyncratic post-earnings drop may mean-revert.

Six variants tested:
  A. Post-Earnings Dip Buy (>5% drop on earnings day, hold 10d)
  B. Sector-Relative Earnings Dip (>3% worse than sector ETF, hold 10d)
  C. Earnings Gap-Down + Quality (B + above 200-SMA, hold 15d)
  D. Post-Earnings Drift Fade (>7% gap down, contrarian, hold 20d)
  E. Earnings Vol Crush Recovery (vol drops, price stable, hold 10d)
  F. Multi-Day Post-Earnings Reversal (wait for exhaustion, RSI<30, hold 10d)

Regime hedge: Half-size when SPY < 200-SMA.
Max positions: 5 simultaneous.
Starting capital: $645.
Slippage: 0.02% each way.
OOT: 2022-01-01 to 2026-07-25.

5-gate validation:
  1. Sharpe > 0.5
  2. Permutation test p < 0.05 (500 iterations)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/earnings_reversal_combo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

DATA_START = "2020-06-01"
OOT_START = "2022-01-01"
OOT_END = "2026-07-25"
STARTING_CAPITAL = 645.0
SLIPPAGE_BPS = 2  # 0.02% each way = 2 bps
MAX_POSITIONS = 5
PERMUTATION_ITERS = 500

# ─── Universe ────────────────────────────────────────────────────────────────

UNIVERSE = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B", "UNH", "JNJ",
    "JPM", "V", "PG", "XOM", "HD", "MA", "CVX", "MRK", "ABBV", "LLY",
    "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "WMT", "CSCO", "ACN", "ABT",
    "DHR", "CRM", "NKE", "ADBE", "TXN", "NEE", "PM", "UNP", "RTX", "HON",
    "LOW", "INTC", "UPS", "QCOM", "BA", "AMGN", "CAT", "IBM", "GE", "SBUX",
    "INTU", "ISRG", "BLK", "PLD", "MDLZ", "ADP", "GILD", "ADI", "SYK", "MMC",
    "DE", "LMT", "TJX", "CB", "REGN", "MO", "CI", "SO", "DUK", "CL",
    "CME", "ICE", "PGR", "SHW", "ZTS", "BSX", "VRTX", "FISV", "APD", "MCK",
    "EL", "AON", "HUM", "EMR", "ECL", "SLB", "ORLY", "AIG", "WM", "PSA",
    "SPG", "NSC", "F", "GM", "USB", "TFC", "PNC", "MS", "GS", "SCHW",
]

# Stock -> sector ETF mapping
STOCK_SECTOR = {
    # Technology
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AVGO": "XLK", "ADBE": "XLK",
    "CRM": "XLK", "CSCO": "XLK", "ACN": "XLK", "TXN": "XLK", "INTC": "XLK",
    "QCOM": "XLK", "IBM": "XLK", "INTU": "XLK", "ADP": "XLK", "ADI": "XLK",
    "FISV": "XLK",
    # Financials
    "JPM": "XLF", "V": "XLF", "MA": "XLF", "BRK-B": "XLF", "BLK": "XLF",
    "MMC": "XLF", "CB": "XLF", "AIG": "XLF", "AON": "XLF", "CME": "XLF",
    "ICE": "XLF", "PGR": "XLF", "USB": "XLF", "TFC": "XLF", "PNC": "XLF",
    "MS": "XLF", "GS": "XLF", "SCHW": "XLF",
    # Health Care
    "UNH": "XLV", "JNJ": "XLV", "LLY": "XLV", "MRK": "XLV", "ABBV": "XLV",
    "TMO": "XLV", "ABT": "XLV", "DHR": "XLV", "AMGN": "XLV", "ISRG": "XLV",
    "SYK": "XLV", "GILD": "XLV", "REGN": "XLV", "CI": "XLV", "BSX": "XLV",
    "VRTX": "XLV", "MCK": "XLV", "HUM": "XLV", "ZTS": "XLV",
    # Consumer Discretionary
    "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "MCD": "XLY", "NKE": "XLY",
    "LOW": "XLY", "SBUX": "XLY", "TJX": "XLY", "ORLY": "XLY", "F": "XLY",
    "GM": "XLY",
    # Consumer Staples
    "PG": "XLP", "PEP": "XLP", "KO": "XLP", "COST": "XLP", "WMT": "XLP",
    "PM": "XLP", "MO": "XLP", "CL": "XLP", "MDLZ": "XLP", "EL": "XLP",
    # Energy
    "XOM": "XLE", "CVX": "XLE", "SLB": "XLE",
    # Industrials
    "UNP": "XLI", "RTX": "XLI", "HON": "XLI", "UPS": "XLI", "BA": "XLI",
    "CAT": "XLI", "GE": "XLI", "DE": "XLI", "LMT": "XLI", "EMR": "XLI",
    "WM": "XLI", "NSC": "XLI",
    # Communication Services
    "GOOGL": "XLC", "META": "XLC",
    # Utilities
    "NEE": "XLU", "SO": "XLU", "DUK": "XLU",
    # Real Estate
    "PLD": "XLRE", "PSA": "XLRE", "SPG": "XLRE",
    # Materials
    "SHW": "XLB", "APD": "XLB", "ECL": "XLB",
}

SECTOR_ETFS = list(set(STOCK_SECTOR.values()))


# ─── Data Download ───────────────────────────────────────────────────────────

def download_data(tickers, start, end):
    """Download OHLCV data with caching."""
    cache_file = CACHE_DIR / f"stock_data_{start}_{end}.pkl"
    if cache_file.exists():
        log.info("Loading cached stock data...")
        data = pd.read_pickle(cache_file)
        if len(data) >= len(tickers) * 0.6:
            return data

    log.info(f"Downloading {len(tickers)} tickers...")
    data = {}
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(" ".join(batch), start=start, end=end,
                              group_by="ticker", auto_adjust=True, progress=False, threads=True)
            for t in batch:
                try:
                    if len(batch) > 1:
                        df = raw[t].copy()
                    else:
                        df = raw.copy()
                    df = df.dropna(subset=["Close", "Volume"])
                    if len(df) > 100:
                        data[t] = df
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"Batch download failed: {e}")

    log.info(f"Downloaded {len(data)} tickers successfully")
    pd.to_pickle(data, cache_file)
    return data


def download_etfs(etf_list, start, end):
    """Download sector ETF data with caching."""
    cache_file = CACHE_DIR / f"etf_data_{start}_{end}.pkl"
    if cache_file.exists():
        log.info("Loading cached ETF data...")
        data = pd.read_pickle(cache_file)
        if len(data) >= len(etf_list) * 0.8:
            return data

    log.info(f"Downloading {len(etf_list)} ETFs...")
    tickers_str = " ".join(etf_list + ["SPY"])
    raw = yf.download(tickers_str, start=start, end=end,
                      group_by="ticker", auto_adjust=True, progress=False, threads=True)
    data = {}
    for t in etf_list + ["SPY"]:
        try:
            df = raw[t].copy()
            df = df.dropna(subset=["Close", "Volume"])
            if len(df) > 100:
                data[t] = df
        except Exception:
            pass

    log.info(f"Downloaded {len(data)} ETFs")
    pd.to_pickle(data, cache_file)
    return data


# ─── Feature Computation ────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_sma(series, period):
    return series.rolling(period).mean()


def compute_realized_vol(series, window=10):
    """Annualized realized volatility from log returns."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def detect_earnings_days(df):
    """
    Detect likely earnings days: abs(daily return) > 3% AND volume > 2x 20-day avg.
    Returns boolean Series.
    """
    prev_close = df["Close"].shift(1)
    daily_ret = (df["Close"] - prev_close) / prev_close
    vol_avg_20 = df["Volume"].rolling(20).mean()
    vol_ratio = df["Volume"] / vol_avg_20

    is_earnings = (daily_ret.abs() > 0.03) & (vol_ratio > 2.0)
    return is_earnings, daily_ret, vol_ratio


def precompute_features(df):
    """Compute all needed features for a stock."""
    prev_close = df["Close"].shift(1)
    daily_ret = (df["Close"] - prev_close) / prev_close
    gap_pct = (df["Open"] - prev_close) / prev_close  # overnight gap
    vol_avg_20 = df["Volume"].rolling(20).mean()
    vol_ratio = df["Volume"] / vol_avg_20

    sma_200 = compute_sma(df["Close"], 200)
    rsi_5 = compute_rsi(df["Close"], 5)
    rsi_14 = compute_rsi(df["Close"], 14)

    # Pre-earnings realized vol (20-day before) vs post (10-day after)
    rvol_pre = compute_realized_vol(df["Close"], 20)
    rvol_post = compute_realized_vol(df["Close"], 10)

    is_earnings, _, _ = detect_earnings_days(df)

    return pd.DataFrame({
        "close": df["Close"],
        "open": df["Open"],
        "high": df["High"],
        "low": df["Low"],
        "volume": df["Volume"],
        "prev_close": prev_close,
        "daily_ret": daily_ret,
        "gap_pct": gap_pct,
        "vol_ratio": vol_ratio,
        "sma_200": sma_200,
        "rsi_5": rsi_5,
        "rsi_14": rsi_14,
        "rvol_pre": rvol_pre,
        "rvol_post": rvol_post,
        "is_earnings": is_earnings,
    }, index=df.index)


# ─── Position & Portfolio ────────────────────────────────────────────────────

class Position:
    def __init__(self, ticker, entry_date, entry_price, shares, variant, hold_days):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.variant = variant
        self.max_hold = hold_days
        self.days_held = 0
        self.exit_date = None
        self.exit_price = None

    def pnl(self, price):
        return (price - self.entry_price) * self.shares

    def pnl_pct(self, price):
        return (price / self.entry_price) - 1.0


def apply_slippage(price, direction="buy"):
    """Apply 0.02% slippage each way."""
    if direction == "buy":
        return price * (1 + SLIPPAGE_BPS / 10000)
    else:
        return price * (1 - SLIPPAGE_BPS / 10000)


# ─── Backtest Engine ─────────────────────────────────────────────────────────

def run_variant(variant_name, features, etf_data, spy_close, spy_sma200, all_dates):
    """
    Run a single variant backtest.
    Returns: trades list, equity curve, daily returns.
    """
    capital = STARTING_CAPITAL
    positions = []
    trades = []
    equity_curve = []
    daily_returns = []
    prev_equity = capital

    for i, date in enumerate(all_dates):
        # ── Check exits ──
        closed_today = []
        for pos in positions:
            pos.days_held += 1
            if date not in features.get(pos.ticker, pd.DataFrame()).index:
                continue
            feat = features[pos.ticker].loc[date]
            current_price = feat["close"]

            if pos.days_held >= pos.max_hold:
                exit_price = apply_slippage(current_price, "sell")
                pnl = (exit_price - pos.entry_price) * pos.shares
                capital += pos.shares * exit_price
                trades.append({
                    "ticker": pos.ticker,
                    "variant": pos.variant,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(date.date()),
                    "entry_price": round(pos.entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "shares": pos.shares,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((exit_price / pos.entry_price - 1) * 100, 2),
                    "days_held": pos.days_held,
                })
                closed_today.append(pos)

        positions = [p for p in positions if p not in closed_today]

        # ── Regime check ──
        spy_val = spy_close.get(date, np.nan) if isinstance(spy_close, dict) else spy_close.reindex([date]).values[0] if date in spy_close.index else np.nan
        sma200_val = spy_sma200.get(date, np.nan) if isinstance(spy_sma200, dict) else spy_sma200.reindex([date]).values[0] if date in spy_sma200.index else np.nan
        bear_market = False
        if not np.isnan(spy_val) and not np.isnan(sma200_val):
            bear_market = spy_val < sma200_val

        size_mult = 0.5 if bear_market else 1.0

        # ── Check entries ──
        if len(positions) < MAX_POSITIONS:
            candidates = _generate_signals(variant_name, features, etf_data, date, positions)
            for ticker, hold_days in candidates:
                if len(positions) >= MAX_POSITIONS:
                    break
                if any(p.ticker == ticker for p in positions):
                    continue

                feat = features[ticker].loc[date]
                entry_price = apply_slippage(feat["close"], "buy")

                # Position sizing: equal weight, regime-adjusted
                alloc = capital * (1.0 / MAX_POSITIONS) * size_mult
                shares = alloc / entry_price if entry_price > 0 else 0
                if shares < 0.01 or alloc < 1.0:
                    continue

                capital -= shares * entry_price
                positions.append(Position(
                    ticker=ticker,
                    entry_date=date,
                    entry_price=entry_price,
                    shares=shares,
                    variant=variant_name,
                    hold_days=hold_days,
                ))

        # ── Mark to market ──
        portfolio_value = capital
        for pos in positions:
            if date in features.get(pos.ticker, pd.DataFrame()).index:
                price = features[pos.ticker].loc[date]["close"]
                portfolio_value += pos.shares * price
            else:
                portfolio_value += pos.shares * pos.entry_price

        daily_ret = (portfolio_value / prev_equity - 1) if prev_equity > 0 else 0
        daily_returns.append(daily_ret)
        equity_curve.append({"date": str(date.date()), "equity": round(portfolio_value, 2)})
        prev_equity = portfolio_value

    # Close any remaining positions at last date
    last_date = all_dates[-1]
    for pos in positions:
        if last_date in features.get(pos.ticker, pd.DataFrame()).index:
            exit_price = apply_slippage(features[pos.ticker].loc[last_date]["close"], "sell")
        else:
            exit_price = pos.entry_price
        pnl = (exit_price - pos.entry_price) * pos.shares
        trades.append({
            "ticker": pos.ticker,
            "variant": pos.variant,
            "entry_date": str(pos.entry_date.date()),
            "exit_date": str(last_date.date()),
            "entry_price": round(pos.entry_price, 4),
            "exit_price": round(exit_price, 4),
            "shares": pos.shares,
            "pnl": round(pnl, 2),
            "pnl_pct": round((exit_price / pos.entry_price - 1) * 100, 2),
            "days_held": pos.days_held,
        })

    return trades, equity_curve, daily_returns


def _generate_signals(variant, features, etf_data, date, current_positions):
    """Generate entry signals for a given variant on a given date."""
    signals = []
    held_tickers = {p.ticker for p in current_positions}

    for ticker in UNIVERSE:
        if ticker in held_tickers:
            continue
        if ticker not in features:
            continue
        feat = features[ticker]
        if date not in feat.index:
            continue
        row = feat.loc[date]

        # Skip if NaN in critical fields
        if pd.isna(row["daily_ret"]) or pd.isna(row["vol_ratio"]):
            continue

        if variant == "A":
            # Post-Earnings Dip Buy: drop >5% on earnings day, hold 10d
            if row["is_earnings"] and row["daily_ret"] < -0.05:
                signals.append((ticker, 10))

        elif variant == "B":
            # Sector-Relative Earnings Dip: drop >3% more than sector
            if row["is_earnings"] and row["daily_ret"] < -0.03:
                sector_etf = STOCK_SECTOR.get(ticker)
                if sector_etf and sector_etf in etf_data:
                    etf_feat = etf_data[sector_etf]
                    if date in etf_feat.index:
                        etf_prev = etf_feat.loc[:date]["Close"].shift(1)
                        if date in etf_prev.index and not pd.isna(etf_prev.loc[date]):
                            etf_ret = (etf_feat.loc[date]["Close"] - etf_prev.loc[date]) / etf_prev.loc[date]
                            relative_drop = row["daily_ret"] - etf_ret
                            if relative_drop < -0.03:
                                signals.append((ticker, 10))

        elif variant == "C":
            # Sector-Relative + Quality: same as B but above 200-SMA, hold 15d
            if row["is_earnings"] and row["daily_ret"] < -0.03:
                if not pd.isna(row["sma_200"]) and row["close"] > row["sma_200"]:
                    sector_etf = STOCK_SECTOR.get(ticker)
                    if sector_etf and sector_etf in etf_data:
                        etf_feat = etf_data[sector_etf]
                        if date in etf_feat.index:
                            etf_prev = etf_feat.loc[:date]["Close"].shift(1)
                            if date in etf_prev.index and not pd.isna(etf_prev.loc[date]):
                                etf_ret = (etf_feat.loc[date]["Close"] - etf_prev.loc[date]) / etf_prev.loc[date]
                                relative_drop = row["daily_ret"] - etf_ret
                                if relative_drop < -0.03:
                                    signals.append((ticker, 15))

        elif variant == "D":
            # Post-Earnings Drift Fade: gap down >7%, contrarian, hold 20d
            if row["is_earnings"] and row["daily_ret"] < -0.07:
                signals.append((ticker, 20))

        elif variant == "E":
            # Earnings Vol Crush Recovery: post-earnings realized vol drops,
            # stock within 5% of pre-earnings price, hold 10d
            if row["is_earnings"]:
                # Look back: was there an earnings event 3-7 days ago?
                pass  # handled below

            # Check if 3-7 days ago was an earnings day
            idx_pos = feat.index.get_loc(date) if date in feat.index else None
            if idx_pos is not None and idx_pos >= 7:
                for lookback in range(3, 8):
                    if idx_pos - lookback < 0:
                        continue
                    past_date = feat.index[idx_pos - lookback]
                    past_row = feat.loc[past_date]
                    if past_row["is_earnings"]:
                        # Check vol crush: current rvol < pre-earnings rvol
                        if (not pd.isna(row["rvol_post"]) and not pd.isna(past_row["rvol_pre"])
                                and row["rvol_post"] < past_row["rvol_pre"]):
                            # Stock within 5% of pre-earnings close
                            pre_earnings_close = past_row["prev_close"]
                            if not pd.isna(pre_earnings_close):
                                price_diff = abs(row["close"] / pre_earnings_close - 1)
                                if price_diff < 0.05:
                                    signals.append((ticker, 10))
                        break  # only check most recent earnings event

        elif variant == "F":
            # Multi-Day Post-Earnings Reversal: wait 3-5 days after earnings miss,
            # stock dropped >8% from pre-earnings close AND RSI(5) < 30, hold 10d
            idx_pos = feat.index.get_loc(date) if date in feat.index else None
            if idx_pos is not None and idx_pos >= 7:
                for lookback in range(3, 6):
                    if idx_pos - lookback < 0:
                        continue
                    past_date = feat.index[idx_pos - lookback]
                    past_row = feat.loc[past_date]
                    if past_row["is_earnings"] and past_row["daily_ret"] < -0.03:
                        # Check cumulative drop from pre-earnings close
                        pre_earnings_close = past_row["prev_close"]
                        if not pd.isna(pre_earnings_close) and pre_earnings_close > 0:
                            cum_drop = (row["close"] / pre_earnings_close) - 1
                            if cum_drop < -0.08 and not pd.isna(row["rsi_5"]) and row["rsi_5"] < 30:
                                signals.append((ticker, 10))
                        break

    return signals


# ─── Validation ──────────────────────────────────────────────────────────────

def compute_metrics(trades, daily_returns, equity_curve):
    """Compute strategy metrics."""
    if not trades or len(daily_returns) < 20:
        return None

    dr = np.array(daily_returns)
    n_trades = len(trades)
    winners = [t for t in trades if t["pnl"] > 0]
    losers = [t for t in trades if t["pnl"] <= 0]

    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = len(winners) / n_trades if n_trades > 0 else 0

    gross_profit = sum(t["pnl"] for t in winners) if winners else 0
    gross_loss = abs(sum(t["pnl"] for t in losers)) if losers else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = np.mean([t["pnl_pct"] for t in winners]) if winners else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losers]) if losers else 0

    # Annualized metrics
    mean_daily = np.mean(dr)
    std_daily = np.std(dr, ddof=1) if len(dr) > 1 else 1e-9
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0

    downside = dr[dr < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_daily / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    # Max drawdown
    eq = [e["equity"] for e in equity_curve]
    peak = eq[0]
    max_dd = 0
    for v in eq:
        peak = max(peak, v)
        dd = (v - peak) / peak
        max_dd = min(max_dd, dd)

    # CAGR
    start_eq = equity_curve[0]["equity"]
    end_eq = equity_curve[-1]["equity"]
    n_years = len(equity_curve) / 252
    cagr = (end_eq / start_eq) ** (1 / n_years) - 1 if n_years > 0 and start_eq > 0 else 0

    return {
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "total_pnl": round(total_pnl, 2),
        "profit_factor": round(profit_factor, 3),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd, 4),
        "cagr": round(cagr, 4),
        "start_equity": start_eq,
        "end_equity": round(end_eq, 2),
    }


def permutation_test(trades, daily_returns, n_iter=500):
    """
    Sign-flip permutation test on trade returns.
    Randomly flip the sign of each trade's return to test if the mean
    trade return is significantly positive (not random).
    """
    if not trades or len(trades) < 5:
        return 1.0

    trade_rets = np.array([t["pnl_pct"] for t in trades])
    actual_mean = np.mean(trade_rets)

    count_above = 0
    rng = np.random.RandomState(42)

    for _ in range(n_iter):
        # Randomly flip sign of each trade return
        signs = rng.choice([-1, 1], size=len(trade_rets))
        shuffled_mean = np.mean(trade_rets * signs)
        if shuffled_mean >= actual_mean:
            count_above += 1

    return count_above / n_iter


def regime_gap_test(trades, spy_close, spy_sma200):
    """
    Compute Sharpe in bull vs bear regime.
    Regime gap = |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|)
    """
    bull_pnls = []
    bear_pnls = []

    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        # Find closest spy date
        if entry_date in spy_close.index:
            spy_val = spy_close.loc[entry_date]
            sma_val = spy_sma200.loc[entry_date] if entry_date in spy_sma200.index else np.nan
        else:
            spy_val = np.nan
            sma_val = np.nan

        if pd.isna(spy_val) or pd.isna(sma_val):
            bull_pnls.append(t["pnl_pct"])  # default to bull
            continue

        if spy_val >= sma_val:
            bull_pnls.append(t["pnl_pct"])
        else:
            bear_pnls.append(t["pnl_pct"])

    def _sharpe(pnls):
        if len(pnls) < 3:
            return 0.0
        arr = np.array(pnls)
        std = np.std(arr, ddof=1)
        return np.mean(arr) / std * np.sqrt(len(arr)) if std > 0 else 0

    sharpe_bull = _sharpe(bull_pnls)
    sharpe_bear = _sharpe(bear_pnls)

    denom = max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)
    regime_gap = abs(sharpe_bull - sharpe_bear) / denom

    return round(regime_gap, 4), round(sharpe_bull, 3), round(sharpe_bear, 3), len(bull_pnls), len(bear_pnls)


def validate_5_gates(metrics, trades, daily_returns, spy_close, spy_sma200):
    """Apply 5-gate validation. Returns dict with gate results."""
    if metrics is None:
        return {"passed": False, "reason": "No metrics (too few trades or data)"}

    gates = {}

    # Gate 1: Sharpe > 0.5
    gates["sharpe_pass"] = metrics["sharpe"] > 0.5
    gates["sharpe_val"] = metrics["sharpe"]

    # Gate 2: Permutation test p < 0.05
    p_val = permutation_test(trades, daily_returns, PERMUTATION_ITERS)
    gates["perm_p"] = round(p_val, 4)
    gates["perm_pass"] = p_val < 0.05

    # Gate 3: Regime gap < 0.5
    regime_gap, sharpe_bull, sharpe_bear, n_bull, n_bear = regime_gap_test(trades, spy_close, spy_sma200)
    gates["regime_gap"] = regime_gap
    gates["regime_pass"] = regime_gap < 0.5
    gates["sharpe_bull"] = sharpe_bull
    gates["sharpe_bear"] = sharpe_bear
    gates["n_bull_trades"] = n_bull
    gates["n_bear_trades"] = n_bear

    # Gate 4: MaxDD > -50%
    gates["max_dd"] = metrics["max_dd"]
    gates["max_dd_pass"] = metrics["max_dd"] > -0.50

    # Gate 5: >= 20 trades
    gates["n_trades"] = metrics["n_trades"]
    gates["n_trades_pass"] = metrics["n_trades"] >= 20

    gates["passed"] = all([
        gates["sharpe_pass"],
        gates["perm_pass"],
        gates["regime_pass"],
        gates["max_dd_pass"],
        gates["n_trades_pass"],
    ])

    return gates


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("EARNINGS REVERSAL + SECTOR-RELATIVE MEAN REVERSION COMBO BACKTEST")
    log.info("=" * 70)

    # Download data
    all_tickers = UNIVERSE + SECTOR_ETFS + ["SPY"]
    all_tickers = list(set(all_tickers))

    stock_data = download_data(UNIVERSE, DATA_START, OOT_END)
    etf_data = download_etfs(SECTOR_ETFS, DATA_START, OOT_END)

    spy_df = etf_data.get("SPY")
    if spy_df is None:
        log.info("Downloading SPY separately...")
        spy_df = yf.download("SPY", start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

    # Flatten MultiIndex columns if needed
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.get_level_values(0)

    spy_close = spy_df["Close"].squeeze() if isinstance(spy_df["Close"], pd.DataFrame) else spy_df["Close"]
    spy_sma200 = spy_close.rolling(200).mean()

    # Pre-compute features for all stocks
    log.info("Pre-computing features for all stocks...")
    features = {}
    for ticker, df in stock_data.items():
        try:
            # Flatten columns if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            features[ticker] = precompute_features(df)
        except Exception as e:
            log.warning(f"Failed to compute features for {ticker}: {e}")

    log.info(f"Features computed for {len(features)} stocks")

    # Get OOT trading dates
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)
    all_dates = spy_df.index[(spy_df.index >= oot_start) & (spy_df.index <= oot_end)]
    log.info(f"OOT period: {all_dates[0].date()} to {all_dates[-1].date()} ({len(all_dates)} days)")

    # Run all variants
    variants = ["A", "B", "C", "D", "E", "F"]
    variant_names = {
        "A": "Post-Earnings Dip Buy",
        "B": "Sector-Relative Earnings Dip",
        "C": "Earnings Gap-Down + Quality",
        "D": "Post-Earnings Drift Fade (contrarian)",
        "E": "Earnings Vol Crush Recovery",
        "F": "Multi-Day Post-Earnings Reversal",
    }

    results = {}
    for v in variants:
        log.info(f"\n{'─' * 60}")
        log.info(f"Running Variant {v}: {variant_names[v]}")
        log.info(f"{'─' * 60}")

        trades, equity_curve, daily_returns = run_variant(v, features, etf_data, spy_close, spy_sma200, all_dates)
        metrics = compute_metrics(trades, daily_returns, equity_curve)

        if metrics:
            gates = validate_5_gates(metrics, trades, daily_returns, spy_close, spy_sma200)

            log.info(f"  Trades: {metrics['n_trades']}, WR: {metrics['win_rate']:.1%}, "
                     f"PF: {metrics['profit_factor']:.2f}")
            log.info(f"  Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}, "
                     f"MaxDD: {metrics['max_dd']:.1%}")
            log.info(f"  Total PnL: ${metrics['total_pnl']:.2f}, "
                     f"End Equity: ${metrics['end_equity']:.2f}")
            log.info(f"  Regime gap: {gates['regime_gap']:.3f} "
                     f"(bull Sharpe={gates['sharpe_bull']:.2f}, bear Sharpe={gates['sharpe_bear']:.2f})")
            log.info(f"  Permutation p-val: {gates['perm_p']:.4f}")
            log.info(f"  5-GATE: {'✓ PASS' if gates['passed'] else '✗ FAIL'}")

            for g in ["sharpe_pass", "perm_pass", "regime_pass", "max_dd_pass", "n_trades_pass"]:
                status = "PASS" if gates[g] else "FAIL"
                log.info(f"    {g}: {status}")

            results[v] = {
                "name": variant_names[v],
                "metrics": metrics,
                "gates": gates,
                "n_trades": metrics["n_trades"],
                "sample_trades": trades[:10] if trades else [],
                "equity_start": equity_curve[0] if equity_curve else None,
                "equity_end": equity_curve[-1] if equity_curve else None,
            }
        else:
            log.info(f"  NO TRADES or insufficient data for variant {v}")
            results[v] = {
                "name": variant_names[v],
                "metrics": None,
                "gates": {"passed": False, "reason": "No trades"},
                "n_trades": 0,
            }

    # ── Summary ──
    log.info(f"\n{'=' * 70}")
    log.info("SUMMARY — EARNINGS REVERSAL COMBO BACKTEST")
    log.info(f"{'=' * 70}")
    log.info(f"{'Variant':<8} {'Name':<40} {'Trades':>6} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'5-Gate':>7}")
    log.info("-" * 90)

    any_passed = False
    for v in variants:
        r = results[v]
        m = r.get("metrics")
        g = r.get("gates", {})
        if m:
            status = "PASS" if g.get("passed") else "FAIL"
            log.info(f"  {v:<6} {r['name']:<40} {m['n_trades']:>6} {m['sharpe']:>7.3f} "
                     f"{m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {m['max_dd']:>6.1%} {status:>7}")
            if g.get("passed"):
                any_passed = True
        else:
            log.info(f"  {v:<6} {r['name']:<40} {'N/A':>6} {'N/A':>7} {'N/A':>6} {'N/A':>6} {'N/A':>7} {'FAIL':>7}")

    # Save results
    output = {
        "strategy": "Earnings Reversal + Sector-Relative Mean Reversion Combo",
        "oot_period": f"{OOT_START} to {OOT_END}",
        "starting_capital": STARTING_CAPITAL,
        "universe_size": len(UNIVERSE),
        "max_positions": MAX_POSITIONS,
        "slippage_bps": SLIPPAGE_BPS,
        "regime_hedge": "Half-size when SPY < 200-SMA",
        "run_timestamp": datetime.now().isoformat(),
        "variants": results,
        "any_variant_passed": any_passed,
    }

    output_path = Path("/home/jupiter/Lvl3Quant/data/earnings_reversal_combo_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nResults saved to {output_path}")

    return output


if __name__ == "__main__":
    main()
