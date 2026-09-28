#!/usr/bin/env python3
"""
Mean Reversion / Oversold Bounce Backtest
==========================================
Academic basis: Jegadeesh (1990) — short-term reversal effects.
Tests 6 variants (A-F) on top 30 liquid stocks, 2022-2026.

Validation: Sharpe, permutation test, regime analysis, drawdown, trade count.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AMD", "AVGO", "CRM",
    "NFLX", "ADBE", "INTC", "QCOM", "MU", "PANW", "CRWD", "PLTR", "HOOD", "COIN",
    "UBER", "ABNB", "SNAP", "ROKU", "PINS", "SOFI", "NET", "ZS", "TTD", "DASH",
]
START = "2022-01-01"
END = "2026-07-28"
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
MAX_CONCURRENT = 3
POSITION_SIZE = 200.0  # equal weight per position
PERMUTATION_ITERS = 1000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ── Custom JSON encoder for numpy types ─────────────────────────────────
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


# ── Data Download ───────────────────────────────────────────────────────
def download_data():
    print(f"Downloading {len(TICKERS)} tickers from {START} to {END}...")
    data = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
            if df is not None and len(df) > 50:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                print(f"  {ticker}: {len(df)} bars")
            else:
                print(f"  {ticker}: insufficient data, skipping")
        except Exception as e:
            print(f"  {ticker}: download failed ({e})")

    # Download SPY for regime classification
    spy = yf.download("SPY", start=START, end=END, progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    return data, spy


# ── Indicator Helpers ───────────────────────────────────────────────────
def compute_rsi(series, period=5):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger(series, period=20, num_std=2.5):
    sma = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    lower = sma - num_std * std
    upper = sma + num_std * std
    return sma, lower, upper


# ── Position Tracker ────────────────────────────────────────────────────
class Portfolio:
    def __init__(self, capital=CAPITAL):
        self.initial_capital = capital
        self.cash = capital
        self.positions = {}  # ticker -> {shares, entry_price, entry_date, entry_bar_idx}
        self.trades = []
        self.equity_curve = []

    def n_positions(self):
        return len(self.positions)

    def can_open(self):
        return self.n_positions() < MAX_CONCURRENT

    def open_position(self, ticker, price, date, bar_idx):
        if ticker in self.positions:
            return False
        if not self.can_open():
            return False

        slipped_price = price * (1 + SLIPPAGE_PCT)
        shares = POSITION_SIZE / slipped_price
        cost = shares * slipped_price

        if cost > self.cash:
            return False

        self.cash -= cost
        self.positions[ticker] = {
            "shares": shares,
            "entry_price": slipped_price,
            "entry_date": date,
            "entry_bar_idx": bar_idx,
        }
        return True

    def close_position(self, ticker, price, date):
        if ticker not in self.positions:
            return None

        pos = self.positions[ticker]
        slipped_price = price * (1 - SLIPPAGE_PCT)
        proceeds = pos["shares"] * slipped_price
        pnl = proceeds - pos["shares"] * pos["entry_price"]
        pnl_pct = (slipped_price / pos["entry_price"] - 1) * 100

        self.cash += proceeds

        trade = {
            "ticker": ticker,
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
        del self.positions[ticker]
        return trade

    def mark_to_market(self, prices_dict, date):
        """Calculate total equity given current prices."""
        equity = self.cash
        for ticker, pos in self.positions.items():
            if ticker in prices_dict:
                equity += pos["shares"] * prices_dict[ticker]
            else:
                equity += pos["shares"] * pos["entry_price"]
        self.equity_curve.append({"date": str(date)[:10], "equity": float(equity)})
        return equity


# ── Strategy Implementations ────────────────────────────────────────────
def run_variant_a(data, all_dates):
    """RSI(5) Oversold Bounce: Buy RSI<20, Sell RSI>50."""
    print("\n[A] RSI(5) Oversold Bounce...")

    # Precompute RSI for all tickers
    rsi_data = {}
    for ticker, df in data.items():
        rsi_data[ticker] = compute_rsi(df["Close"], period=5)

    portfolio = Portfolio()

    for i, date in enumerate(all_dates):
        # Check exits first
        tickers_to_close = []
        for ticker in list(portfolio.positions.keys()):
            if ticker in rsi_data and date in rsi_data[ticker].index:
                rsi_val = rsi_data[ticker].loc[date]
                if not np.isnan(rsi_val) and rsi_val > 50:
                    tickers_to_close.append(ticker)

        for ticker in tickers_to_close:
            if date in data[ticker].index:
                portfolio.close_position(ticker, float(data[ticker].loc[date, "Close"]), date)

        # Check entries
        if portfolio.can_open():
            candidates = []
            for ticker, df in data.items():
                if ticker in portfolio.positions:
                    continue
                if date not in df.index:
                    continue
                if ticker not in rsi_data or date not in rsi_data[ticker].index:
                    continue
                rsi_val = rsi_data[ticker].loc[date]
                if not np.isnan(rsi_val) and rsi_val < 20:
                    candidates.append((ticker, float(rsi_val)))

            # Buy most oversold first
            candidates.sort(key=lambda x: x[1])
            for ticker, _ in candidates:
                if not portfolio.can_open():
                    break
                price = float(data[ticker].loc[date, "Close"])
                portfolio.open_position(ticker, price, date, i)

        # Mark to market
        prices = {}
        for ticker, df in data.items():
            if date in df.index:
                prices[ticker] = float(df.loc[date, "Close"])
        portfolio.mark_to_market(prices, date)

    # Close remaining positions at last date
    last_date = all_dates[-1]
    for ticker in list(portfolio.positions.keys()):
        if ticker in data and last_date in data[ticker].index:
            portfolio.close_position(ticker, float(data[ticker].loc[last_date, "Close"]), last_date)

    return portfolio


def run_variant_b(data, all_dates):
    """5-Day Drawdown Reversal: Buy on >10% 5-day drop, hold 5 days."""
    print("\n[B] 5-Day Drawdown Reversal...")

    portfolio = Portfolio()

    for i, date in enumerate(all_dates):
        # Check exits (hold 5 trading days)
        tickers_to_close = []
        for ticker, pos in list(portfolio.positions.items()):
            bars_held = i - pos["entry_bar_idx"]
            if bars_held >= 5:
                tickers_to_close.append(ticker)

        for ticker in tickers_to_close:
            if date in data[ticker].index:
                portfolio.close_position(ticker, float(data[ticker].loc[date, "Close"]), date)

        # Check entries
        if portfolio.can_open():
            candidates = []
            for ticker, df in data.items():
                if ticker in portfolio.positions:
                    continue
                if date not in df.index:
                    continue

                idx = df.index.get_loc(date)
                if idx < 5:
                    continue

                price_now = float(df.iloc[idx]["Close"])
                price_5d_ago = float(df.iloc[idx - 5]["Close"])
                drawdown = (price_now / price_5d_ago - 1)

                if drawdown < -0.10:
                    candidates.append((ticker, drawdown))

            candidates.sort(key=lambda x: x[1])  # worst drawdown first
            for ticker, _ in candidates:
                if not portfolio.can_open():
                    break
                price = float(data[ticker].loc[date, "Close"])
                portfolio.open_position(ticker, price, date, i)

        prices = {}
        for ticker, df in data.items():
            if date in df.index:
                prices[ticker] = float(df.loc[date, "Close"])
        portfolio.mark_to_market(prices, date)

    last_date = all_dates[-1]
    for ticker in list(portfolio.positions.keys()):
        if ticker in data and last_date in data[ticker].index:
            portfolio.close_position(ticker, float(data[ticker].loc[last_date, "Close"]), last_date)

    return portfolio


def run_variant_c(data, all_dates):
    """Bollinger Band Bounce: Buy at lower BB(20,2.5), sell at middle band."""
    print("\n[C] Bollinger Band Bounce...")

    bb_data = {}
    for ticker, df in data.items():
        sma, lower, upper = compute_bollinger(df["Close"], period=20, num_std=2.5)
        bb_data[ticker] = {"sma": sma, "lower": lower, "upper": upper}

    portfolio = Portfolio()

    for i, date in enumerate(all_dates):
        # Check exits
        tickers_to_close = []
        for ticker in list(portfolio.positions.keys()):
            if ticker in data and date in data[ticker].index:
                price = float(data[ticker].loc[date, "Close"])
                if ticker in bb_data and date in bb_data[ticker]["sma"].index:
                    sma_val = bb_data[ticker]["sma"].loc[date]
                    if not np.isnan(sma_val) and price >= sma_val:
                        tickers_to_close.append(ticker)

        for ticker in tickers_to_close:
            portfolio.close_position(ticker, float(data[ticker].loc[date, "Close"]), date)

        # Check entries
        if portfolio.can_open():
            candidates = []
            for ticker, df in data.items():
                if ticker in portfolio.positions:
                    continue
                if date not in df.index:
                    continue
                if ticker not in bb_data or date not in bb_data[ticker]["lower"].index:
                    continue

                price = float(df.loc[date, "Close"])
                lower_val = bb_data[ticker]["lower"].loc[date]

                if not np.isnan(lower_val) and price <= lower_val:
                    # Rank by how far below the band
                    distance = (price - lower_val) / lower_val
                    candidates.append((ticker, distance))

            candidates.sort(key=lambda x: x[1])
            for ticker, _ in candidates:
                if not portfolio.can_open():
                    break
                price = float(data[ticker].loc[date, "Close"])
                portfolio.open_position(ticker, price, date, i)

        prices = {}
        for ticker, df in data.items():
            if date in df.index:
                prices[ticker] = float(df.loc[date, "Close"])
        portfolio.mark_to_market(prices, date)

    last_date = all_dates[-1]
    for ticker in list(portfolio.positions.keys()):
        if ticker in data and last_date in data[ticker].index:
            portfolio.close_position(ticker, float(data[ticker].loc[last_date, "Close"]), last_date)

    return portfolio


def run_variant_d(data, all_dates):
    """Gap Down Recovery: Buy on >5% gap down, hold 3 days."""
    print("\n[D] Gap Down Recovery...")

    portfolio = Portfolio()

    for i, date in enumerate(all_dates):
        # Check exits (hold 3 trading days)
        tickers_to_close = []
        for ticker, pos in list(portfolio.positions.items()):
            bars_held = i - pos["entry_bar_idx"]
            if bars_held >= 3:
                tickers_to_close.append(ticker)

        for ticker in tickers_to_close:
            if date in data[ticker].index:
                portfolio.close_position(ticker, float(data[ticker].loc[date, "Close"]), date)

        # Check entries
        if portfolio.can_open():
            candidates = []
            for ticker, df in data.items():
                if ticker in portfolio.positions:
                    continue
                if date not in df.index:
                    continue

                idx = df.index.get_loc(date)
                if idx < 1:
                    continue

                open_price = float(df.iloc[idx]["Open"])
                prev_close = float(df.iloc[idx - 1]["Close"])
                gap = (open_price / prev_close - 1)

                if gap < -0.05:
                    candidates.append((ticker, gap, open_price))

            candidates.sort(key=lambda x: x[1])  # biggest gap down first
            for ticker, _, open_price in candidates:
                if not portfolio.can_open():
                    break
                # Buy at open price (gap down entry)
                portfolio.open_position(ticker, open_price, date, i)

        prices = {}
        for ticker, df in data.items():
            if date in df.index:
                prices[ticker] = float(df.loc[date, "Close"])
        portfolio.mark_to_market(prices, date)

    last_date = all_dates[-1]
    for ticker in list(portfolio.positions.keys()):
        if ticker in data and last_date in data[ticker].index:
            portfolio.close_position(ticker, float(data[ticker].loc[last_date, "Close"]), last_date)

    return portfolio


def run_variant_e(data, all_dates):
    """Multi-Signal Confluence: 2+ of RSI<25, below lower BB(20,2), down >8% in 5d. Sell RSI>50 or +5%."""
    print("\n[E] Multi-Signal Confluence...")

    rsi_data = {}
    bb_data = {}
    for ticker, df in data.items():
        rsi_data[ticker] = compute_rsi(df["Close"], period=5)
        sma, lower, upper = compute_bollinger(df["Close"], period=20, num_std=2.0)
        bb_data[ticker] = {"sma": sma, "lower": lower}

    portfolio = Portfolio()

    for i, date in enumerate(all_dates):
        # Check exits
        tickers_to_close = []
        for ticker, pos in list(portfolio.positions.items()):
            if ticker in data and date in data[ticker].index:
                price = float(data[ticker].loc[date, "Close"])
                pnl_pct = (price / pos["entry_price"] - 1) * 100

                rsi_exit = False
                if ticker in rsi_data and date in rsi_data[ticker].index:
                    rsi_val = rsi_data[ticker].loc[date]
                    if not np.isnan(rsi_val) and rsi_val > 50:
                        rsi_exit = True

                if rsi_exit or pnl_pct >= 5.0:
                    tickers_to_close.append(ticker)

        for ticker in tickers_to_close:
            portfolio.close_position(ticker, float(data[ticker].loc[date, "Close"]), date)

        # Check entries
        if portfolio.can_open():
            candidates = []
            for ticker, df in data.items():
                if ticker in portfolio.positions:
                    continue
                if date not in df.index:
                    continue

                idx = df.index.get_loc(date)
                signals = 0

                # Signal 1: RSI(5) < 25
                if ticker in rsi_data and date in rsi_data[ticker].index:
                    rsi_val = rsi_data[ticker].loc[date]
                    if not np.isnan(rsi_val) and rsi_val < 25:
                        signals += 1

                # Signal 2: Below lower BB(20, 2)
                if ticker in bb_data and date in bb_data[ticker]["lower"].index:
                    lower_val = bb_data[ticker]["lower"].loc[date]
                    price = float(df.loc[date, "Close"])
                    if not np.isnan(lower_val) and price < lower_val:
                        signals += 1

                # Signal 3: Down >8% in 5 days
                if idx >= 5:
                    price_now = float(df.iloc[idx]["Close"])
                    price_5d = float(df.iloc[idx - 5]["Close"])
                    if (price_now / price_5d - 1) < -0.08:
                        signals += 1

                if signals >= 2:
                    candidates.append((ticker, signals))

            candidates.sort(key=lambda x: -x[1])  # most signals first
            for ticker, _ in candidates:
                if not portfolio.can_open():
                    break
                price = float(data[ticker].loc[date, "Close"])
                portfolio.open_position(ticker, price, date, i)

        prices = {}
        for ticker, df in data.items():
            if date in df.index:
                prices[ticker] = float(df.loc[date, "Close"])
        portfolio.mark_to_market(prices, date)

    last_date = all_dates[-1]
    for ticker in list(portfolio.positions.keys()):
        if ticker in data and last_date in data[ticker].index:
            portfolio.close_position(ticker, float(data[ticker].loc[last_date, "Close"]), last_date)

    return portfolio


def run_variant_f(data, all_dates, reference_portfolio):
    """ADVERSARIAL: Random trades with same avg holding period and trade count."""
    print("\n[F] Random Adversarial...")

    ref_trades = reference_portfolio.trades
    if len(ref_trades) == 0:
        print("  No reference trades to match, skipping.")
        portfolio = Portfolio()
        for date in all_dates:
            prices = {}
            for ticker, df in data.items():
                if date in df.index:
                    prices[ticker] = float(df.loc[date, "Close"])
            portfolio.mark_to_market(prices, date)
        return portfolio

    n_trades = len(ref_trades)
    avg_hold = max(1, int(np.mean([t["hold_days"] for t in ref_trades])))
    # Convert avg_hold calendar days to approximate bar count
    avg_hold_bars = max(1, int(avg_hold * 5 / 7))  # rough calendar->trading day conversion

    portfolio = Portfolio()
    ticker_list = list(data.keys())

    # Pre-generate random entry points
    rng = np.random.RandomState(RANDOM_SEED + 99)
    warmup = 30
    valid_range = len(all_dates) - warmup - avg_hold_bars - 1
    if valid_range < 1:
        valid_range = 1

    random_entries = []
    for _ in range(n_trades * 3):  # generate extra in case some fail
        bar_idx = rng.randint(warmup, warmup + valid_range)
        ticker = ticker_list[rng.randint(0, len(ticker_list))]
        random_entries.append((bar_idx, ticker))

    entry_idx = 0

    for i, date in enumerate(all_dates):
        # Check exits
        tickers_to_close = []
        for ticker, pos in list(portfolio.positions.items()):
            bars_held = i - pos["entry_bar_idx"]
            if bars_held >= avg_hold_bars:
                tickers_to_close.append(ticker)

        for ticker in tickers_to_close:
            if date in data[ticker].index:
                portfolio.close_position(ticker, float(data[ticker].loc[date, "Close"]), date)

        # Check if we should enter (random)
        while entry_idx < len(random_entries) and portfolio.can_open():
            target_bar, ticker = random_entries[entry_idx]
            if target_bar > i:
                break
            entry_idx += 1
            if target_bar != i:
                continue
            if ticker in portfolio.positions:
                continue
            if ticker not in data or date not in data[ticker].index:
                continue
            if len(portfolio.trades) + portfolio.n_positions() >= n_trades:
                break
            price = float(data[ticker].loc[date, "Close"])
            portfolio.open_position(ticker, price, date, i)

        prices = {}
        for ticker, df in data.items():
            if date in df.index:
                prices[ticker] = float(df.loc[date, "Close"])
        portfolio.mark_to_market(prices, date)

    last_date = all_dates[-1]
    for ticker in list(portfolio.positions.keys()):
        if ticker in data and last_date in data[ticker].index:
            portfolio.close_position(ticker, float(data[ticker].loc[last_date, "Close"]), last_date)

    return portfolio


# ── Analytics ───────────────────────────────────────────────────────────
def compute_metrics(portfolio):
    """Compute performance metrics from portfolio."""
    trades = portfolio.trades
    ec = portfolio.equity_curve

    if len(trades) == 0:
        return {
            "n_trades": 0, "win_rate": 0, "avg_pnl": 0, "total_pnl": 0,
            "sharpe": 0, "sortino": 0, "profit_factor": 0,
            "max_drawdown_pct": 0, "avg_hold_days": 0,
            "final_equity": CAPITAL,
        }

    pnls = [t["pnl"] for t in trades]
    pnl_pcts = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Equity-based daily returns
    equities = [e["equity"] for e in ec]
    if len(equities) > 1:
        daily_returns = pd.Series(equities).pct_change().dropna()
        daily_returns = daily_returns.replace([np.inf, -np.inf], 0).fillna(0)
    else:
        daily_returns = pd.Series([0.0])

    # Sharpe (annualized, 252 trading days)
    if daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = float(daily_returns.mean() / downside.std() * np.sqrt(252))
    else:
        sortino = 0.0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else float(gross_profit) if gross_profit > 0 else 0.0

    # Max drawdown
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
        "total_return_pct": float((equities[-1] / CAPITAL - 1) * 100),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_hold_days": round(float(np.mean([t["hold_days"] for t in trades])), 1),
        "final_equity": round(float(equities[-1]), 2),
    }


def permutation_test(portfolio, data, all_dates, n_iters=PERMUTATION_ITERS):
    """Shuffle entry dates and re-run to get null distribution of Sharpe."""
    trades = portfolio.trades
    if len(trades) < 5:
        return {"p_value": 1.0, "actual_sharpe": 0.0, "null_mean_sharpe": 0.0}

    actual_metrics = compute_metrics(portfolio)
    actual_sharpe = actual_metrics["sharpe"]

    hold_bars_list = []
    for t in trades:
        hold_days = t["hold_days"]
        hold_bars_list.append(max(1, int(hold_days * 5 / 7)))

    avg_hold_bars = int(np.mean(hold_bars_list))
    n_trades = len(trades)

    rng = np.random.RandomState(RANDOM_SEED)
    ticker_list = list(data.keys())
    warmup = 30

    null_sharpes = []

    for iteration in range(n_iters):
        pf = Portfolio()

        # Generate random entry points
        entries = []
        for _ in range(n_trades):
            bar_idx = rng.randint(warmup, max(warmup + 1, len(all_dates) - avg_hold_bars - 1))
            ticker = ticker_list[rng.randint(0, len(ticker_list))]
            entries.append((bar_idx, ticker))
        entries.sort(key=lambda x: x[0])

        entry_ptr = 0
        for i, date in enumerate(all_dates):
            # Exits
            for tk in list(pf.positions.keys()):
                if i - pf.positions[tk]["entry_bar_idx"] >= avg_hold_bars:
                    if tk in data and date in data[tk].index:
                        pf.close_position(tk, float(data[tk].loc[date, "Close"]), date)

            # Entries
            while entry_ptr < len(entries) and pf.can_open():
                target_bar, tk = entries[entry_ptr]
                if target_bar > i:
                    break
                entry_ptr += 1
                if target_bar != i or tk in pf.positions:
                    continue
                if tk not in data or date not in data[tk].index:
                    continue
                if len(pf.trades) + pf.n_positions() >= n_trades:
                    break
                pf.open_position(tk, float(data[tk].loc[date, "Close"]), date, i)

            prices = {}
            for tk, df in data.items():
                if date in df.index:
                    prices[tk] = float(df.loc[date, "Close"])
            pf.mark_to_market(prices, date)

        # Close remaining
        last_date = all_dates[-1]
        for tk in list(pf.positions.keys()):
            if tk in data and last_date in data[tk].index:
                pf.close_position(tk, float(data[tk].loc[last_date, "Close"]), last_date)

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


def regime_analysis(portfolio, spy_df):
    """Split trades by bull/bear regime (SPY > or < 200-SMA)."""
    trades = portfolio.trades
    if len(trades) == 0:
        return {"bull_sharpe": 0, "bear_sharpe": 0, "regime_gap": 0, "bull_trades": 0, "bear_trades": 0}

    spy_sma200 = spy_df["Close"].rolling(200).mean()

    bull_pnls = []
    bear_pnls = []

    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        # Find closest date in SPY
        if entry_date in spy_sma200.index:
            sma_val = spy_sma200.loc[entry_date]
            spy_price = float(spy_df.loc[entry_date, "Close"])
        else:
            # Find nearest
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
    """Apply 5 validation gates."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "permutation_p_lt_0.05": perm_result["p_value"] < 0.05,
        "regime_gap_lt_0.5": regime_result["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("MEAN REVERSION / OVERSOLD BOUNCE BACKTEST")
    print("=" * 70)

    data, spy_df = download_data()

    # Build common date index
    all_dates_set = set()
    for df in data.values():
        all_dates_set.update(df.index.tolist())
    all_dates = sorted(all_dates_set)
    print(f"\nTotal trading days: {len(all_dates)}")

    # Run all variants
    variants = {}

    variants["A_RSI5_Oversold"] = run_variant_a(data, all_dates)
    variants["B_5Day_Drawdown"] = run_variant_b(data, all_dates)
    variants["C_Bollinger_Bounce"] = run_variant_c(data, all_dates)
    variants["D_Gap_Down"] = run_variant_d(data, all_dates)
    variants["E_Multi_Confluence"] = run_variant_e(data, all_dates)

    # Find best variant (by Sharpe) for adversarial reference
    best_variant_name = None
    best_sharpe = -999
    for name, pf in variants.items():
        m = compute_metrics(pf)
        if m["sharpe"] > best_sharpe:
            best_sharpe = m["sharpe"]
            best_variant_name = name

    print(f"\nBest variant for adversarial reference: {best_variant_name} (Sharpe={best_sharpe:.3f})")
    variants["F_Random_Adversarial"] = run_variant_f(data, all_dates, variants[best_variant_name])

    # Compute all results
    results = {}

    for name, pf in variants.items():
        print(f"\n{'=' * 50}")
        print(f"Analyzing {name}...")

        metrics = compute_metrics(pf)
        print(f"  Trades: {metrics['n_trades']}, WR: {metrics['win_rate']:.1f}%, "
              f"Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}")

        print(f"  Running permutation test ({PERMUTATION_ITERS} iterations)...")
        perm = permutation_test(pf, data, all_dates)
        print(f"  p-value: {perm['p_value']:.4f}")

        regime = regime_analysis(pf, spy_df)
        print(f"  Bull Sharpe: {regime['bull_sharpe']:.3f}, Bear Sharpe: {regime['bear_sharpe']:.3f}, "
              f"Gap: {regime['regime_gap']:.3f}")

        gates = validate(metrics, perm, regime)
        passed = sum(1 for v in gates.values() if v and isinstance(v, bool)) - (1 if gates["all_passed"] else 0)
        total = len(gates) - 1
        print(f"  Validation: {passed}/{total} gates passed. ALL={'PASS' if gates['all_passed'] else 'FAIL'}")

        results[name] = {
            "metrics": metrics,
            "permutation_test": perm,
            "regime_analysis": regime,
            "validation_gates": gates,
            "sample_trades": pf.trades[:10] if pf.trades else [],
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<25} {'Trades':>6} {'WR%':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'MaxDD%':>7} {'Return%':>8} {'Pass':>5}")
    print("-" * 85)
    for name, r in results.items():
        m = r["metrics"]
        v = r["validation_gates"]
        passed = sum(1 for k, val in v.items() if val and k != "all_passed")
        print(f"{name:<25} {m['n_trades']:>6} {m['win_rate']:>5.1f}% {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['profit_factor']:>6.2f} {m['max_drawdown_pct']:>6.1f}% {m.get('total_return_pct', 0):>7.1f}% {passed:>3}/5")

    # Add metadata
    output = {
        "metadata": {
            "strategy": "Mean Reversion / Oversold Bounces",
            "academic_basis": "Jegadeesh (1990) — short-term reversal",
            "universe": TICKERS,
            "period": f"{START} to {END}",
            "initial_capital": CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "max_concurrent_positions": MAX_CONCURRENT,
            "position_size": POSITION_SIZE,
            "permutation_iterations": PERMUTATION_ITERS,
            "run_timestamp": datetime.now().isoformat(),
        },
        "variants": results,
    }

    output_path = "/home/jupiter/Lvl3Quant/data/mean_reversion_bounce_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()
