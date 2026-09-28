#!/usr/bin/env python3
"""
Trend Reversal Detection Backtest
==================================
Thesis: The hardest problem in trading is detecting when a downtrend ends.
Tests 6 technical reversal detection methods on growth + safe-haven universe.

Variants:
  A) MACD Crossover + RSI Confluence
  B) Bollinger Band Bounce
  C) Volume Exhaustion + Price Reversal
  D) Moving Average Convergence
  E) Multi-Signal Confluence (Best of All)
  F) Adaptive Reversal on Best Stock

Walk-forward OOT: Jan 2022 - Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

# -- Config --
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "CRM",
    "GLD", "TLT", "SHY",
    "SPY", "QQQ",
]
GROWTH_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "CRM"]
START = "2021-06-01"  # extra warmup for 200-SMA
END = "2026-07-30"
OOT_START = "2022-01-01"
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
PERMUTATION_ITERS = 1000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)


# -- Data Download --
def download_data():
    all_tickers = list(set(TICKERS + ["SPY"]))
    print(f"Downloading {len(all_tickers)} tickers from {START} to {END}...")
    data = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
            if df is not None and len(df) > 50:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                print(f"  {ticker}: {len(df)} bars")
            else:
                print(f"  {ticker}: insufficient data, skipping")
        except Exception as e:
            print(f"  {ticker}: download failed ({e})")

    spy = data.get("SPY")
    return data, spy


# -- Indicator Helpers --
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def compute_bollinger(series, period=20, num_std=2.0):
    sma = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    lower = sma - num_std * std
    upper = sma + num_std * std
    return sma, lower, upper


def compute_sma(series, period):
    return series.rolling(window=period, min_periods=period).mean()


# -- Portfolio --
class Portfolio:
    def __init__(self, capital=CAPITAL):
        self.initial_capital = capital
        self.cash = capital
        self.position = None  # single position: {ticker, shares, entry_price, entry_date, entry_bar_idx}
        self.trades = []
        self.equity_curve = []

    def is_flat(self):
        return self.position is None

    def open_position(self, ticker, price, date, bar_idx):
        if not self.is_flat():
            return False
        slipped_price = price * (1 + SLIPPAGE_PCT)
        shares = self.cash / slipped_price  # full capital
        cost = shares * slipped_price
        if cost > self.cash:
            shares = self.cash / slipped_price
            cost = shares * slipped_price
        self.cash -= cost
        self.position = {
            "ticker": ticker,
            "shares": shares,
            "entry_price": slipped_price,
            "entry_date": date,
            "entry_bar_idx": bar_idx,
        }
        return True

    def close_position(self, price, date):
        if self.is_flat():
            return None
        pos = self.position
        slipped_price = price * (1 - SLIPPAGE_PCT)
        proceeds = pos["shares"] * slipped_price
        pnl = proceeds - pos["shares"] * pos["entry_price"]
        pnl_pct = (slipped_price / pos["entry_price"] - 1) * 100

        self.cash += proceeds
        trade = {
            "ticker": pos["ticker"],
            "entry_date": str(pos["entry_date"])[:10],
            "exit_date": str(date)[:10],
            "entry_price": float(pos["entry_price"]),
            "exit_price": float(slipped_price),
            "shares": float(pos["shares"]),
            "pnl": float(pnl),
            "pnl_pct": float(pnl_pct),
            "hold_days": int((pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days),
        }
        self.trades.append(trade)
        self.position = None
        return trade

    def mark_to_market(self, prices_dict, date):
        equity = self.cash
        if self.position is not None:
            ticker = self.position["ticker"]
            if ticker in prices_dict:
                equity += self.position["shares"] * prices_dict[ticker]
            else:
                equity += self.position["shares"] * self.position["entry_price"]
        self.equity_curve.append({"date": str(date)[:10], "equity": float(equity)})
        return equity


# -- Strategy Implementations --

def run_variant_a(data, all_dates, oot_start_idx):
    """A) MACD Crossover + RSI Confluence: Buy QQQ when MACD crosses above signal AND RSI(14)<40. Hold 10 days."""
    print("\n[A] MACD Crossover + RSI Confluence...")

    ticker = "QQQ"
    if ticker not in data:
        print("  QQQ not available!")
        return Portfolio()

    df = data[ticker]
    macd_line, signal_line = compute_macd(df["Close"])
    rsi = compute_rsi(df["Close"], period=14)

    portfolio = Portfolio()

    for i, date in enumerate(all_dates):
        if i < oot_start_idx:
            # warmup period, just mark to market
            prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
            portfolio.mark_to_market(prices, date)
            continue

        # Exit: hold 10 trading days
        if not portfolio.is_flat():
            bars_held = i - portfolio.position["entry_bar_idx"]
            if bars_held >= 10:
                if date in df.index:
                    portfolio.close_position(float(df.loc[date, "Close"]), date)

        # Entry
        if portfolio.is_flat() and date in df.index and date in macd_line.index:
            idx_pos = df.index.get_loc(date)
            if idx_pos < 1:
                prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
                portfolio.mark_to_market(prices, date)
                continue

            prev_date = df.index[idx_pos - 1]

            macd_today = macd_line.loc[date] if date in macd_line.index else np.nan
            signal_today = signal_line.loc[date] if date in signal_line.index else np.nan
            macd_prev = macd_line.loc[prev_date] if prev_date in macd_line.index else np.nan
            signal_prev = signal_line.loc[prev_date] if prev_date in signal_line.index else np.nan
            rsi_today = rsi.loc[date] if date in rsi.index else np.nan

            if (not any(np.isnan([macd_today, signal_today, macd_prev, signal_prev, rsi_today]))):
                # MACD crosses above signal line
                crossover = (macd_prev <= signal_prev) and (macd_today > signal_today)
                rsi_ok = rsi_today < 40

                if crossover and rsi_ok:
                    portfolio.open_position(ticker, float(df.loc[date, "Close"]), date, i)

        prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
        portfolio.mark_to_market(prices, date)

    # Close remaining
    if not portfolio.is_flat():
        last_date = all_dates[-1]
        if ticker in data and last_date in data[ticker].index:
            portfolio.close_position(float(data[ticker].loc[last_date, "Close"]), last_date)

    return portfolio


def run_variant_b(data, all_dates, oot_start_idx):
    """B) Bollinger Band Bounce: Buy QQQ at lower BB touch, sell at middle band or 15 days."""
    print("\n[B] Bollinger Band Bounce...")

    ticker = "QQQ"
    if ticker not in data:
        return Portfolio()

    df = data[ticker]
    sma, lower, upper = compute_bollinger(df["Close"], period=20, num_std=2.0)

    portfolio = Portfolio()

    for i, date in enumerate(all_dates):
        if i < oot_start_idx:
            prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
            portfolio.mark_to_market(prices, date)
            continue

        # Exit: price reaches middle band OR 15 days
        if not portfolio.is_flat():
            bars_held = i - portfolio.position["entry_bar_idx"]
            should_exit = False

            if date in df.index and date in sma.index:
                price = float(df.loc[date, "Close"])
                sma_val = sma.loc[date]
                if not np.isnan(sma_val) and price >= sma_val:
                    should_exit = True

            if bars_held >= 15:
                should_exit = True

            if should_exit and date in df.index:
                portfolio.close_position(float(df.loc[date, "Close"]), date)

        # Entry: price touches lower BB then closes above it next day (bounce confirmation)
        if portfolio.is_flat() and date in df.index:
            idx_pos = df.index.get_loc(date)
            if idx_pos >= 1:
                prev_date = df.index[idx_pos - 1]

                if prev_date in lower.index and date in lower.index:
                    prev_close = float(df.loc[prev_date, "Close"])
                    prev_lower = lower.loc[prev_date]
                    today_close = float(df.loc[date, "Close"])
                    today_lower = lower.loc[date]

                    if (not np.isnan(prev_lower) and not np.isnan(today_lower)):
                        # Yesterday touched/went below lower BB, today closed above it
                        touched_yesterday = prev_close <= prev_lower
                        bounced_today = today_close > today_lower

                        if touched_yesterday and bounced_today:
                            portfolio.open_position(ticker, today_close, date, i)

        prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
        portfolio.mark_to_market(prices, date)

    if not portfolio.is_flat():
        last_date = all_dates[-1]
        if ticker in data and last_date in data[ticker].index:
            portfolio.close_position(float(data[ticker].loc[last_date, "Close"]), last_date)

    return portfolio


def run_variant_c(data, all_dates, oot_start_idx):
    """C) Volume Exhaustion + Price Reversal: Buy QQQ on new 20-day low with declining volume + bullish candle."""
    print("\n[C] Volume Exhaustion + Price Reversal...")

    ticker = "QQQ"
    if ticker not in data:
        return Portfolio()

    df = data[ticker]
    portfolio = Portfolio()

    for i, date in enumerate(all_dates):
        if i < oot_start_idx:
            prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
            portfolio.mark_to_market(prices, date)
            continue

        # Exit: 10 trading days
        if not portfolio.is_flat():
            bars_held = i - portfolio.position["entry_bar_idx"]
            if bars_held >= 10 and date in df.index:
                portfolio.close_position(float(df.loc[date, "Close"]), date)

        # Entry
        if portfolio.is_flat() and date in df.index:
            idx_pos = df.index.get_loc(date)
            if idx_pos >= 20:
                close_today = float(df.iloc[idx_pos]["Close"])
                low_today = float(df.iloc[idx_pos]["Low"])
                high_today = float(df.iloc[idx_pos]["High"])
                open_today = float(df.iloc[idx_pos]["Open"])
                vol_today = float(df.iloc[idx_pos]["Volume"])

                # New 20-day low
                past_20_lows = df["Low"].iloc[idx_pos - 20:idx_pos]
                is_new_low = low_today <= past_20_lows.min()

                # Declining volume: today's volume < average of last 5 days
                avg_vol_5d = df["Volume"].iloc[idx_pos - 5:idx_pos].mean()
                declining_volume = vol_today < avg_vol_5d

                # Bullish reversal candle: close in upper half of day's range
                day_range = high_today - low_today
                if day_range > 0:
                    close_position_in_range = (close_today - low_today) / day_range
                    bullish_candle = close_position_in_range >= 0.5
                else:
                    bullish_candle = False

                if is_new_low and declining_volume and bullish_candle:
                    portfolio.open_position(ticker, close_today, date, i)

        prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
        portfolio.mark_to_market(prices, date)

    if not portfolio.is_flat():
        last_date = all_dates[-1]
        if ticker in data and last_date in data[ticker].index:
            portfolio.close_position(float(data[ticker].loc[last_date, "Close"]), last_date)

    return portfolio


def run_variant_d(data, all_dates, oot_start_idx):
    """D) Moving Average Convergence: Buy QQQ when 10-SMA crosses above 20-SMA after being below 10+ days. Hold 15 days."""
    print("\n[D] Moving Average Convergence...")

    ticker = "QQQ"
    if ticker not in data:
        return Portfolio()

    df = data[ticker]
    sma10 = compute_sma(df["Close"], 10)
    sma20 = compute_sma(df["Close"], 20)

    portfolio = Portfolio()

    for i, date in enumerate(all_dates):
        if i < oot_start_idx:
            prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
            portfolio.mark_to_market(prices, date)
            continue

        # Exit: 15 trading days
        if not portfolio.is_flat():
            bars_held = i - portfolio.position["entry_bar_idx"]
            if bars_held >= 15 and date in df.index:
                portfolio.close_position(float(df.loc[date, "Close"]), date)

        # Entry
        if portfolio.is_flat() and date in df.index and date in sma10.index and date in sma20.index:
            idx_pos = df.index.get_loc(date)
            if idx_pos < 1:
                prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
                portfolio.mark_to_market(prices, date)
                continue

            prev_date = df.index[idx_pos - 1]

            s10_today = sma10.loc[date] if date in sma10.index else np.nan
            s20_today = sma20.loc[date] if date in sma20.index else np.nan
            s10_prev = sma10.loc[prev_date] if prev_date in sma10.index else np.nan
            s20_prev = sma20.loc[prev_date] if prev_date in sma20.index else np.nan

            if not any(np.isnan([s10_today, s20_today, s10_prev, s20_prev])):
                # Crossover: 10-SMA crosses above 20-SMA
                crossover = (s10_prev <= s20_prev) and (s10_today > s20_today)

                if crossover:
                    # Check if 10-SMA was below 20-SMA for at least 10 consecutive days prior
                    below_count = 0
                    for lookback in range(2, min(idx_pos + 1, 60)):
                        lb_date = df.index[idx_pos - lookback]
                        if lb_date in sma10.index and lb_date in sma20.index:
                            s10_lb = sma10.loc[lb_date]
                            s20_lb = sma20.loc[lb_date]
                            if not np.isnan(s10_lb) and not np.isnan(s20_lb) and s10_lb <= s20_lb:
                                below_count += 1
                            else:
                                break
                        else:
                            break

                    if below_count >= 10:
                        portfolio.open_position(ticker, float(df.loc[date, "Close"]), date, i)

        prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
        portfolio.mark_to_market(prices, date)

    if not portfolio.is_flat():
        last_date = all_dates[-1]
        if ticker in data and last_date in data[ticker].index:
            portfolio.close_position(float(data[ticker].loc[last_date, "Close"]), last_date)

    return portfolio


def run_variant_e(data, all_dates, oot_start_idx):
    """E) Multi-Signal Confluence: Buy when 3+ of: RSI(5)<30, MACD bullish cross, below lower BB, volume declining 3+ days. Hold 10 days."""
    print("\n[E] Multi-Signal Confluence...")

    ticker = "QQQ"
    if ticker not in data:
        return Portfolio()

    df = data[ticker]
    rsi5 = compute_rsi(df["Close"], period=5)
    macd_line, signal_line = compute_macd(df["Close"])
    sma_bb, lower_bb, upper_bb = compute_bollinger(df["Close"], period=20, num_std=2.0)

    portfolio = Portfolio()

    for i, date in enumerate(all_dates):
        if i < oot_start_idx:
            prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
            portfolio.mark_to_market(prices, date)
            continue

        # Exit: 10 trading days
        if not portfolio.is_flat():
            bars_held = i - portfolio.position["entry_bar_idx"]
            if bars_held >= 10 and date in df.index:
                portfolio.close_position(float(df.loc[date, "Close"]), date)

        # Entry
        if portfolio.is_flat() and date in df.index:
            idx_pos = df.index.get_loc(date)
            if idx_pos < 3:
                prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
                portfolio.mark_to_market(prices, date)
                continue

            signals = 0

            # 1. RSI(5) < 30
            rsi_val = rsi5.loc[date] if date in rsi5.index else np.nan
            if not np.isnan(rsi_val) and rsi_val < 30:
                signals += 1

            # 2. MACD bullish cross
            prev_date = df.index[idx_pos - 1]
            macd_t = macd_line.loc[date] if date in macd_line.index else np.nan
            sig_t = signal_line.loc[date] if date in signal_line.index else np.nan
            macd_p = macd_line.loc[prev_date] if prev_date in macd_line.index else np.nan
            sig_p = signal_line.loc[prev_date] if prev_date in signal_line.index else np.nan
            if not any(np.isnan([macd_t, sig_t, macd_p, sig_p])):
                if (macd_p <= sig_p) and (macd_t > sig_t):
                    signals += 1

            # 3. Price below lower BB
            price = float(df.loc[date, "Close"])
            lb_val = lower_bb.loc[date] if date in lower_bb.index else np.nan
            if not np.isnan(lb_val) and price < lb_val:
                signals += 1

            # 4. Volume declining for 3+ consecutive days
            if idx_pos >= 3:
                vol_declining = True
                for k in range(1, 4):
                    v_curr = float(df.iloc[idx_pos - k + 1]["Volume"])
                    v_prev = float(df.iloc[idx_pos - k]["Volume"])
                    if v_curr >= v_prev:
                        vol_declining = False
                        break
                if vol_declining:
                    signals += 1

            if signals >= 3:
                portfolio.open_position(ticker, price, date, i)

        prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
        portfolio.mark_to_market(prices, date)

    if not portfolio.is_flat():
        last_date = all_dates[-1]
        if ticker in data and last_date in data[ticker].index:
            portfolio.close_position(float(data[ticker].loc[last_date, "Close"]), last_date)

    return portfolio


def run_variant_f(data, all_dates, oot_start_idx):
    """F) Adaptive Reversal on Best Stock: Among growth stocks, find RSI(5)<25 + positive 60-day momentum. Hold 7 days."""
    print("\n[F] Adaptive Reversal on Best Stock...")

    # Precompute RSI(5) and 60-day momentum for all growth stocks
    rsi_data = {}
    mom_data = {}
    for ticker in GROWTH_TICKERS:
        if ticker in data:
            rsi_data[ticker] = compute_rsi(data[ticker]["Close"], period=5)
            mom_data[ticker] = data[ticker]["Close"].pct_change(60)

    portfolio = Portfolio()

    for i, date in enumerate(all_dates):
        if i < oot_start_idx:
            prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
            portfolio.mark_to_market(prices, date)
            continue

        # Exit: 7 trading days
        if not portfolio.is_flat():
            bars_held = i - portfolio.position["entry_bar_idx"]
            if bars_held >= 7 and date in data.get(portfolio.position["ticker"], pd.DataFrame()).index:
                price = float(data[portfolio.position["ticker"]].loc[date, "Close"])
                portfolio.close_position(price, date)

        # Entry: find most oversold growth stock with positive long-term momentum
        if portfolio.is_flat():
            candidates = []
            for ticker in GROWTH_TICKERS:
                if ticker not in data or date not in data[ticker].index:
                    continue
                if ticker not in rsi_data or date not in rsi_data[ticker].index:
                    continue
                if ticker not in mom_data or date not in mom_data[ticker].index:
                    continue

                rsi_val = rsi_data[ticker].loc[date]
                mom_val = mom_data[ticker].loc[date]

                if np.isnan(rsi_val) or np.isnan(mom_val):
                    continue

                # RSI(5) < 25 AND positive 60-day momentum (long-term uptrend intact)
                if rsi_val < 25 and mom_val > 0:
                    candidates.append((ticker, float(rsi_val)))

            if candidates:
                # Pick the most oversold (lowest RSI)
                candidates.sort(key=lambda x: x[1])
                best_ticker = candidates[0][0]
                price = float(data[best_ticker].loc[date, "Close"])
                portfolio.open_position(best_ticker, price, date, i)

        prices = {t: float(data[t].loc[date, "Close"]) for t in data if date in data[t].index}
        portfolio.mark_to_market(prices, date)

    if not portfolio.is_flat():
        last_date = all_dates[-1]
        ticker = portfolio.position["ticker"]
        if ticker in data and last_date in data[ticker].index:
            portfolio.close_position(float(data[ticker].loc[last_date, "Close"]), last_date)

    return portfolio


# -- Analytics --
def compute_metrics(portfolio, oot_start_date=None):
    trades = portfolio.trades
    ec = portfolio.equity_curve

    # Filter to OOT period only
    if oot_start_date:
        trades = [t for t in trades if t["entry_date"] >= oot_start_date]
        ec = [e for e in ec if e["date"] >= oot_start_date]

    if len(trades) == 0:
        return {
            "n_trades": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0,
            "sharpe": 0, "sortino": 0, "profit_factor": 0,
            "max_drawdown_pct": 0, "avg_hold_days": 0,
            "final_equity": CAPITAL, "total_return_pct": 0,
        }

    pnls = [t["pnl"] for t in trades]
    pnl_pcts = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    equities = [e["equity"] for e in ec]
    if len(equities) > 1:
        daily_returns = pd.Series(equities).pct_change().dropna()
        daily_returns = daily_returns.replace([np.inf, -np.inf], 0).fillna(0)
    else:
        daily_returns = pd.Series([0.0])

    if daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = float(daily_returns.mean() / downside.std() * np.sqrt(252))
    else:
        sortino = 0.0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else float(gross_profit) if gross_profit > 0 else 0.0

    eq_series = pd.Series(equities)
    rolling_max = eq_series.cummax()
    drawdown = (eq_series - rolling_max) / rolling_max * 100
    max_dd = float(drawdown.min())

    return {
        "n_trades": len(trades),
        "win_rate": float(len(wins) / len(trades) * 100),
        "avg_pnl": float(np.mean(pnls)),
        "avg_pnl_pct": float(np.mean(pnl_pcts)),
        "total_pnl": float(sum(pnls)),
        "total_return_pct": float((equities[-1] / equities[0] - 1) * 100) if equities else 0,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_hold_days": round(float(np.mean([t["hold_days"] for t in trades])), 1),
        "final_equity": round(float(equities[-1]), 2) if equities else CAPITAL,
    }


def permutation_test(portfolio, data, all_dates, n_iters=PERMUTATION_ITERS, oot_start_date=None):
    trades = portfolio.trades
    if oot_start_date:
        trades = [t for t in trades if t["entry_date"] >= oot_start_date]

    if len(trades) < 5:
        return {"p_value": 1.0, "actual_sharpe": 0.0, "null_mean_sharpe": 0.0}

    actual_metrics = compute_metrics(portfolio, oot_start_date)
    actual_sharpe = actual_metrics["sharpe"]

    hold_bars_list = [max(1, int(t["hold_days"] * 5 / 7)) for t in trades]
    avg_hold_bars = int(np.mean(hold_bars_list))
    n_trades = len(trades)

    rng = np.random.RandomState(RANDOM_SEED)
    ticker_list = list(data.keys())
    warmup = 30

    null_sharpes = []

    for _ in range(n_iters):
        pf = Portfolio()
        entries = []
        for _ in range(n_trades):
            bar_idx = rng.randint(warmup, max(warmup + 1, len(all_dates) - avg_hold_bars - 1))
            ticker = ticker_list[rng.randint(0, len(ticker_list))]
            entries.append((bar_idx, ticker))
        entries.sort(key=lambda x: x[0])

        entry_ptr = 0
        for i_bar, date in enumerate(all_dates):
            if not pf.is_flat():
                if i_bar - pf.position["entry_bar_idx"] >= avg_hold_bars:
                    tk = pf.position["ticker"]
                    if tk in data and date in data[tk].index:
                        pf.close_position(float(data[tk].loc[date, "Close"]), date)

            while entry_ptr < len(entries) and pf.is_flat():
                target_bar, tk = entries[entry_ptr]
                if target_bar > i_bar:
                    break
                entry_ptr += 1
                if target_bar != i_bar:
                    continue
                if tk not in data or date not in data[tk].index:
                    continue
                if len(pf.trades) >= n_trades:
                    break
                pf.open_position(tk, float(data[tk].loc[date, "Close"]), date, i_bar)

            prices = {tk: float(data[tk].loc[date, "Close"]) for tk in data if date in data[tk].index}
            pf.mark_to_market(prices, date)

        if not pf.is_flat():
            last_date = all_dates[-1]
            tk = pf.position["ticker"]
            if tk in data and last_date in data[tk].index:
                pf.close_position(float(data[tk].loc[last_date, "Close"]), last_date)

        m = compute_metrics(pf)
        null_sharpes.append(m["sharpe"])

    null_sharpes = np.array(null_sharpes)
    p_value = float(np.mean(null_sharpes >= actual_sharpe))

    return {
        "p_value": round(p_value, 4),
        "actual_sharpe": round(actual_sharpe, 3),
        "null_mean_sharpe": round(float(np.mean(null_sharpes)), 3),
        "null_std_sharpe": round(float(np.std(null_sharpes)), 3),
    }


def regime_analysis(portfolio, spy_df, oot_start_date=None):
    trades = portfolio.trades
    if oot_start_date:
        trades = [t for t in trades if t["entry_date"] >= oot_start_date]

    if len(trades) == 0:
        return {"bull_sharpe": 0, "bear_sharpe": 0, "regime_gap": 0, "bull_trades": 0, "bear_trades": 0}

    spy_sma200 = spy_df["Close"].rolling(200).mean()

    bull_pnls = []
    bear_pnls = []

    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        if entry_date in spy_sma200.index:
            sma_val = spy_sma200.loc[entry_date]
            spy_price = float(spy_df.loc[entry_date, "Close"])
        else:
            idx = spy_sma200.index.get_indexer([entry_date], method="nearest")[0]
            if idx < 0 or idx >= len(spy_sma200):
                bull_pnls.append(t["pnl_pct"])
                continue
            sma_val = spy_sma200.iloc[idx]
            spy_price = float(spy_df.iloc[idx]["Close"])

        if np.isnan(sma_val):
            bull_pnls.append(t["pnl_pct"])
            continue

        if spy_price > sma_val:
            bull_pnls.append(t["pnl_pct"])
        else:
            bear_pnls.append(t["pnl_pct"])

    def _sharpe_from_pnls(pnls):
        if len(pnls) < 2:
            return 0.0
        arr = np.array(pnls)
        if arr.std() == 0:
            return 0.0
        return float(arr.mean() / arr.std())

    bull_sharpe = _sharpe_from_pnls(bull_pnls)
    bear_sharpe = _sharpe_from_pnls(bear_pnls)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_trades": len(bull_pnls),
        "bear_trades": len(bear_pnls),
    }


def validate(metrics, perm_result, regime_result):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "permutation_p_lt_0.05": perm_result["p_value"] < 0.05,
        "regime_gap_lt_0.5": regime_result["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# -- Main --
def main():
    print("=" * 70)
    print("TREND REVERSAL DETECTION BACKTEST")
    print("=" * 70)

    data, spy_df = download_data()

    # Build common date index
    all_dates_set = set()
    for df in data.values():
        all_dates_set.update(df.index.tolist())
    all_dates = sorted(all_dates_set)
    print(f"\nTotal trading days: {len(all_dates)}")

    # Find OOT start index
    oot_start_ts = pd.Timestamp(OOT_START)
    oot_start_idx = 0
    for i, d in enumerate(all_dates):
        if pd.Timestamp(d) >= oot_start_ts:
            oot_start_idx = i
            break
    print(f"OOT starts at index {oot_start_idx} ({all_dates[oot_start_idx]})")
    oot_start_date = str(all_dates[oot_start_idx])[:10]

    # Run all variants
    variant_runners = {
        "A_MACD_RSI_Confluence": lambda: run_variant_a(data, all_dates, oot_start_idx),
        "B_Bollinger_Bounce": lambda: run_variant_b(data, all_dates, oot_start_idx),
        "C_Volume_Exhaustion": lambda: run_variant_c(data, all_dates, oot_start_idx),
        "D_MA_Convergence": lambda: run_variant_d(data, all_dates, oot_start_idx),
        "E_Multi_Confluence": lambda: run_variant_e(data, all_dates, oot_start_idx),
        "F_Adaptive_Best_Stock": lambda: run_variant_f(data, all_dates, oot_start_idx),
    }

    variants = {}
    for name, runner in variant_runners.items():
        variants[name] = runner()

    # Compute all results
    results = {}

    for name, pf in variants.items():
        print(f"\n{'=' * 50}")
        print(f"Analyzing {name}...")

        metrics = compute_metrics(pf, oot_start_date)
        print(f"  Trades: {metrics['n_trades']}, WR: {metrics['win_rate']:.1f}%, "
              f"Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}")

        print(f"  Running permutation test ({PERMUTATION_ITERS} iterations)...")
        perm = permutation_test(pf, data, all_dates, oot_start_date=oot_start_date)
        print(f"  p-value: {perm['p_value']:.4f}")

        regime = regime_analysis(pf, spy_df, oot_start_date)
        print(f"  Bull Sharpe: {regime['bull_sharpe']:.3f}, Bear Sharpe: {regime['bear_sharpe']:.3f}, "
              f"Gap: {regime['regime_gap']:.3f}")

        gates = validate(metrics, perm, regime)
        passed = sum(1 for k, v in gates.items() if v and k != "all_passed")
        total = len(gates) - 1
        print(f"  Validation: {passed}/{total} gates passed. ALL={'PASS' if gates['all_passed'] else 'FAIL'}")

        # Get OOT-only trades for sample
        oot_trades = [t for t in pf.trades if t["entry_date"] >= oot_start_date]

        results[name] = {
            "metrics": metrics,
            "permutation_test": perm,
            "regime_analysis": regime,
            "validation_gates": gates,
            "sample_trades": oot_trades[:10] if oot_trades else [],
        }

    # Summary
    print("\n" + "=" * 70)
    print("TREND REVERSAL DETECTION — SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<25} {'Trades':>6} {'WR%':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'MaxDD%':>7} {'Return%':>8} {'Pass':>5}")
    print("-" * 85)
    for name, r in results.items():
        m = r["metrics"]
        v = r["validation_gates"]
        passed = sum(1 for k, val in v.items() if val and k != "all_passed")
        print(f"{name:<25} {m['n_trades']:>6} {m['win_rate']:>5.1f}% {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['profit_factor']:>6.2f} {m['max_drawdown_pct']:>6.1f}% {m.get('total_return_pct', 0):>7.1f}% {passed:>3}/5")

    # Save output
    output = {
        "metadata": {
            "strategy": "Trend Reversal Detection",
            "thesis": "Detecting when downtrends end and uptrends begin for optimal re-entry timing",
            "universe": TICKERS,
            "growth_tickers": GROWTH_TICKERS,
            "period": f"{OOT_START} to {END}",
            "warmup_from": START,
            "initial_capital": CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "commission": 0.0,
            "position_sizing": "Full $645 in single position",
            "permutation_iterations": PERMUTATION_ITERS,
            "regime_definition": "Bull = SPY > 200-SMA, Bear = SPY < 200-SMA",
            "run_timestamp": datetime.now().isoformat(),
        },
        "variants": results,
    }

    output_path = "/home/jupiter/Lvl3Quant/data/trend_reversal_detection_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
