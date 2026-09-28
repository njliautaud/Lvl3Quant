#!/usr/bin/env python3
"""
Idiosyncratic Volatility Strategies Backtest
=============================================
Key insight: strategies that survive rigorous testing isolate IDIOSYNCRATIC
(stock-specific) signals by removing market and sector beta.

Apply this to VOLATILITY: stocks with unusually low/high idiosyncratic vol
(relative to sector) exhibit predictable patterns.

Variants:
  A: Vol Compression Buy (idio vol < 20th pctl -> buy, 10d hold)
  B: Vol Expansion Short-Term (idio vol > 80th pctl + down >3% -> buy capitulation, 5d)
  C: Vol Ratio Mean Reversion (stock_vol/sector_vol extremes, 10d)
  D: Low Idiosyncratic Vol Momentum (top quintile momentum + below-median idio vol, 15d)
  E: Post-Volatility-Crush Recovery (vol crush >50% + near 50d high, 10d)
  F: Regime-Adjusted Vol Breakout (Bollinger break + rising idio vol + falling sector vol, 7d)

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation test p < 0.05 (500 iterations)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Universe ────────────────────────────────────────────────────────────
TICKERS = [
    'AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','BRK-B','UNH','JNJ',
    'JPM','V','PG','XOM','HD','MA','CVX','MRK','ABBV','LLY',
    'PEP','KO','COST','AVGO','TMO','MCD','WMT','CSCO','ACN','ABT',
    'DHR','CRM','NKE','ADBE','TXN','NEE','PM','UNP','RTX','HON',
    'LOW','INTC','UPS','QCOM','BA','AMGN','CAT','IBM','GE','SBUX',
    'INTU','ISRG','BLK','PLD','MDLZ','ADP','GILD','ADI','SYK','MMC',
    'DE','LMT','TJX','CB','REGN','MO','CI','SO','DUK','CL',
    'CME','ICE','PGR','SHW','ZTS','BSX','VRTX','FISV','APD','MCK',
    'EL','AON','HUM','EMR','ECL','SLB','ORLY','AIG','WM','PSA',
    'SPG','NSC','F','GM','USB','TFC','PNC','MS','GS','SCHW',
]

SECTOR_MAP = {
    'AAPL':'XLK','MSFT':'XLK','AMZN':'XLY','GOOGL':'XLC','META':'XLC',
    'NVDA':'XLK','TSLA':'XLY','BRK-B':'XLF','UNH':'XLV','JNJ':'XLV',
    'JPM':'XLF','V':'XLK','PG':'XLP','XOM':'XLE','HD':'XLY',
    'MA':'XLK','CVX':'XLE','MRK':'XLV','ABBV':'XLV','LLY':'XLV',
    'PEP':'XLP','KO':'XLP','COST':'XLP','AVGO':'XLK','TMO':'XLV',
    'MCD':'XLY','WMT':'XLP','CSCO':'XLK','ACN':'XLK','ABT':'XLV',
    'DHR':'XLV','CRM':'XLK','NKE':'XLY','ADBE':'XLK','TXN':'XLK',
    'NEE':'XLU','PM':'XLP','UNP':'XLI','RTX':'XLI','HON':'XLI',
    'LOW':'XLY','INTC':'XLK','UPS':'XLI','QCOM':'XLK','BA':'XLI',
    'AMGN':'XLV','CAT':'XLI','IBM':'XLK','GE':'XLI','SBUX':'XLY',
    'INTU':'XLK','ISRG':'XLV','BLK':'XLF','PLD':'XLRE','MDLZ':'XLP',
    'ADP':'XLK','GILD':'XLV','ADI':'XLK','SYK':'XLV','MMC':'XLF',
    'DE':'XLI','LMT':'XLI','TJX':'XLY','CB':'XLF','REGN':'XLV',
    'MO':'XLP','CI':'XLV','SO':'XLU','DUK':'XLU','CL':'XLP',
    'CME':'XLF','ICE':'XLF','PGR':'XLF','SHW':'XLB','ZTS':'XLV',
    'BSX':'XLV','VRTX':'XLV','FISV':'XLK','APD':'XLB','MCK':'XLV',
    'EL':'XLP','AON':'XLF','HUM':'XLV','EMR':'XLI','ECL':'XLB',
    'SLB':'XLE','ORLY':'XLY','AIG':'XLF','WM':'XLI','PSA':'XLRE',
    'SPG':'XLRE','NSC':'XLI','F':'XLY','GM':'XLY','USB':'XLF',
    'TFC':'XLF','PNC':'XLF','MS':'XLF','GS':'XLF','SCHW':'XLF',
}

SECTOR_ETFS = list(set(SECTOR_MAP.values()))
SLIPPAGE_PCT = 0.0002  # 0.02% each way
MAX_POSITIONS = 5
STARTING_CAPITAL = 645.0
OOT_START = pd.Timestamp('2022-01-01')
OOT_END = pd.Timestamp('2026-07-25')
PERM_ITERATIONS = 500


# ── Data Download ─────────────────────────────────────────────────────────
def download_data():
    """Download all required price data with lookback for indicators."""
    all_tickers = list(set(TICKERS + ['SPY'] + SECTOR_ETFS))
    print(f"Downloading data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start='2021-01-01', end='2026-07-29',
                       group_by='ticker', auto_adjust=True, threads=True)

    closes = pd.DataFrame()
    volumes = pd.DataFrame()
    for t in all_tickers:
        try:
            if len(all_tickers) > 1:
                if t in data.columns.get_level_values(0):
                    closes[t] = data[t]['Close']
                    volumes[t] = data[t]['Volume']
            else:
                closes[t] = data['Close']
                volumes[t] = data['Volume']
        except Exception:
            pass

    closes = closes.dropna(how='all')
    volumes = volumes.dropna(how='all')
    print(f"Data range: {closes.index[0].date()} to {closes.index[-1].date()}")
    print(f"Tickers with data: {len([c for c in closes.columns if closes[c].notna().sum() > 100])}")
    return closes, volumes


# ── Volatility Helpers ───────────────────────────────────────────────────
def realized_vol(prices, window=10):
    """Annualized realized vol from log returns over a rolling window."""
    log_ret = np.log(prices / prices.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


def idiosyncratic_vol(stock_vol, sector_vol, beta=0.8):
    """Idiosyncratic vol = stock vol - beta * sector vol."""
    return stock_vol - beta * sector_vol


# ── Signal Generators ─────────────────────────────────────────────────────
def generate_signals_A(closes, volumes, spy_close):
    """
    Variant A: Vol Compression Buy.
    Buy when stock's idiosyncratic vol drops below its 20th percentile (60d lookback).
    Hold 10 days.
    """
    signals = []
    stock_tickers = [t for t in TICKERS if t in closes.columns]

    # Pre-compute 10d realized vol for all stocks and sector ETFs
    stock_vols = {}
    for t in stock_tickers:
        stock_vols[t] = realized_vol(closes[t], 10)

    sector_vols = {}
    for etf in SECTOR_ETFS:
        if etf in closes.columns:
            sector_vols[etf] = realized_vol(closes[etf], 10)

    for date in closes.index:
        if date < closes.index[80]:  # need 60d lookback + 10d vol + buffer
            continue
        for ticker in stock_tickers:
            sector_etf = SECTOR_MAP.get(ticker)
            if not sector_etf or sector_etf not in sector_vols:
                continue
            try:
                sv = stock_vols[ticker].at[date]
                ev = sector_vols[sector_etf].at[date]
            except (KeyError, ValueError):
                continue
            if pd.isna(sv) or pd.isna(ev):
                continue

            idio_vol = sv - 0.8 * ev

            # 60-day lookback for percentile
            date_idx = closes.index.get_loc(date)
            lookback_start = max(0, date_idx - 60)
            lookback_dates = closes.index[lookback_start:date_idx + 1]

            idio_hist = []
            for d in lookback_dates:
                try:
                    s = stock_vols[ticker].at[d]
                    e = sector_vols[sector_etf].at[d]
                    if pd.notna(s) and pd.notna(e):
                        idio_hist.append(s - 0.8 * e)
                except (KeyError, ValueError):
                    pass

            if len(idio_hist) < 30:
                continue

            pctl_20 = np.percentile(idio_hist, 20)
            if idio_vol < pctl_20:
                signals.append({
                    'date': date, 'ticker': ticker,
                    'idio_vol': idio_vol, 'pctl_20': pctl_20,
                })

    return pd.DataFrame(signals), 10


def generate_signals_B(closes, volumes, spy_close):
    """
    Variant B: Vol Expansion Short-Term.
    Buy when idiosyncratic vol > 80th percentile (60d) AND stock down >3% in past 5d.
    Hold 5 days. (Capitulation buy.)
    """
    signals = []
    stock_tickers = [t for t in TICKERS if t in closes.columns]
    ret_5d = closes.pct_change(5)

    stock_vols = {}
    for t in stock_tickers:
        stock_vols[t] = realized_vol(closes[t], 10)

    sector_vols = {}
    for etf in SECTOR_ETFS:
        if etf in closes.columns:
            sector_vols[etf] = realized_vol(closes[etf], 10)

    for date in closes.index:
        if date < closes.index[80]:
            continue
        for ticker in stock_tickers:
            sector_etf = SECTOR_MAP.get(ticker)
            if not sector_etf or sector_etf not in sector_vols:
                continue
            try:
                sv = stock_vols[ticker].at[date]
                ev = sector_vols[sector_etf].at[date]
                r5 = ret_5d.at[date, ticker]
            except (KeyError, ValueError):
                continue
            if pd.isna(sv) or pd.isna(ev) or pd.isna(r5):
                continue

            idio_vol = sv - 0.8 * ev

            # 60d lookback percentile
            date_idx = closes.index.get_loc(date)
            lookback_start = max(0, date_idx - 60)
            lookback_dates = closes.index[lookback_start:date_idx + 1]

            idio_hist = []
            for d in lookback_dates:
                try:
                    s = stock_vols[ticker].at[d]
                    e = sector_vols[sector_etf].at[d]
                    if pd.notna(s) and pd.notna(e):
                        idio_hist.append(s - 0.8 * e)
                except (KeyError, ValueError):
                    pass

            if len(idio_hist) < 30:
                continue

            pctl_80 = np.percentile(idio_hist, 80)
            if idio_vol > pctl_80 and r5 < -0.03:
                signals.append({
                    'date': date, 'ticker': ticker,
                    'idio_vol': idio_vol, 'ret_5d': r5,
                })

    return pd.DataFrame(signals), 5


def generate_signals_C(closes, volumes, spy_close):
    """
    Variant C: Vol Ratio Mean Reversion.
    Ratio = stock_vol / sector_vol.
    Buy when ratio < 0.5 (stock unusually calm vs sector).
    Short when ratio > 2.0 (stock unusually volatile vs sector).
    Hold 10 days.
    """
    signals = []
    stock_tickers = [t for t in TICKERS if t in closes.columns]

    stock_vols = {}
    for t in stock_tickers:
        stock_vols[t] = realized_vol(closes[t], 10)

    sector_vols = {}
    for etf in SECTOR_ETFS:
        if etf in closes.columns:
            sector_vols[etf] = realized_vol(closes[etf], 10)

    for date in closes.index:
        if date < closes.index[30]:
            continue
        for ticker in stock_tickers:
            sector_etf = SECTOR_MAP.get(ticker)
            if not sector_etf or sector_etf not in sector_vols:
                continue
            try:
                sv = stock_vols[ticker].at[date]
                ev = sector_vols[sector_etf].at[date]
            except (KeyError, ValueError):
                continue
            if pd.isna(sv) or pd.isna(ev) or ev <= 0:
                continue

            ratio = sv / ev
            if ratio < 0.5:
                signals.append({
                    'date': date, 'ticker': ticker,
                    'vol_ratio': ratio, 'direction': 'long',
                })
            elif ratio > 2.0:
                signals.append({
                    'date': date, 'ticker': ticker,
                    'vol_ratio': ratio, 'direction': 'short',
                })

    return pd.DataFrame(signals), 10


def generate_signals_D(closes, volumes, spy_close):
    """
    Variant D: Low Idiosyncratic Vol Momentum.
    Buy stocks in top quintile of 20d momentum ONLY when their idiosyncratic vol
    is below median. Filters momentum for steady movers. Hold 15 days.
    """
    signals = []
    stock_tickers = [t for t in TICKERS if t in closes.columns]
    ret_20d = closes[stock_tickers].pct_change(20)

    stock_vols = {}
    for t in stock_tickers:
        stock_vols[t] = realized_vol(closes[t], 10)

    sector_vols = {}
    for etf in SECTOR_ETFS:
        if etf in closes.columns:
            sector_vols[etf] = realized_vol(closes[etf], 10)

    for date in closes.index:
        if date < closes.index[80]:
            continue

        # Compute idiosyncratic vol for all stocks on this date
        idio_vols = {}
        mom_vals = {}
        for ticker in stock_tickers:
            sector_etf = SECTOR_MAP.get(ticker)
            if not sector_etf or sector_etf not in sector_vols:
                continue
            try:
                sv = stock_vols[ticker].at[date]
                ev = sector_vols[sector_etf].at[date]
                m = ret_20d.at[date, ticker]
            except (KeyError, ValueError):
                continue
            if pd.notna(sv) and pd.notna(ev) and pd.notna(m):
                idio_vols[ticker] = sv - 0.8 * ev
                mom_vals[ticker] = m

        if len(mom_vals) < 10:
            continue

        # Top quintile momentum
        mom_series = pd.Series(mom_vals)
        idio_series = pd.Series(idio_vols)
        mom_threshold = mom_series.quantile(0.80)
        idio_median = idio_series.median()

        top_mom = mom_series[mom_series >= mom_threshold].index
        for ticker in top_mom:
            if ticker in idio_series and idio_series[ticker] < idio_median:
                signals.append({
                    'date': date, 'ticker': ticker,
                    'momentum_20d': mom_vals[ticker],
                    'idio_vol': idio_vols[ticker],
                })

    return pd.DataFrame(signals), 15


def generate_signals_E(closes, volumes, spy_close):
    """
    Variant E: Post-Volatility-Crush Recovery.
    After stock's 5d realized vol drops >50% from its 20d avg vol (vol crush),
    buy if the stock is within 5% of its 50d high (healthy consolidation).
    Hold 10 days.
    """
    signals = []
    stock_tickers = [t for t in TICKERS if t in closes.columns]

    vol_5d = {}
    vol_20d = {}
    high_50d = {}
    for t in stock_tickers:
        vol_5d[t] = realized_vol(closes[t], 5)
        vol_20d[t] = realized_vol(closes[t], 20)
        high_50d[t] = closes[t].rolling(50).max()

    for date in closes.index:
        if date < closes.index[60]:
            continue
        for ticker in stock_tickers:
            try:
                v5 = vol_5d[ticker].at[date]
                v20 = vol_20d[ticker].at[date]
                h50 = high_50d[ticker].at[date]
                price = closes.at[date, ticker]
            except (KeyError, ValueError):
                continue
            if pd.isna(v5) or pd.isna(v20) or pd.isna(h50) or pd.isna(price):
                continue
            if v20 <= 0 or h50 <= 0:
                continue

            vol_drop = (v5 - v20) / v20
            pct_from_high = (price - h50) / h50

            # Vol crush >50% AND within 5% of 50d high
            if vol_drop < -0.50 and pct_from_high > -0.05:
                signals.append({
                    'date': date, 'ticker': ticker,
                    'vol_5d': v5, 'vol_20d': v20,
                    'vol_crush_pct': vol_drop,
                    'pct_from_50d_high': pct_from_high,
                })

    return pd.DataFrame(signals), 10


def generate_signals_F(closes, volumes, spy_close):
    """
    Variant F: Regime-Adjusted Vol Breakout.
    Buy when stock breaks above 20d Bollinger Band AND its idiosyncratic vol
    is rising (today > 5d avg) while sector vol is falling.
    Hold 7 days.
    """
    signals = []
    stock_tickers = [t for t in TICKERS if t in closes.columns]

    # Pre-compute Bollinger Bands (20d, 2 std)
    bb_mid = {}
    bb_upper = {}
    for t in stock_tickers:
        bb_mid[t] = closes[t].rolling(20).mean()
        bb_std = closes[t].rolling(20).std()
        bb_upper[t] = bb_mid[t] + 2 * bb_std

    stock_vols = {}
    stock_vol_5d_avg = {}
    for t in stock_tickers:
        stock_vols[t] = realized_vol(closes[t], 10)
        stock_vol_5d_avg[t] = stock_vols[t].rolling(5).mean()

    sector_vols = {}
    sector_vol_5d_avg = {}
    for etf in SECTOR_ETFS:
        if etf in closes.columns:
            sector_vols[etf] = realized_vol(closes[etf], 10)
            sector_vol_5d_avg[etf] = sector_vols[etf].rolling(5).mean()

    for date in closes.index:
        if date < closes.index[40]:
            continue
        for ticker in stock_tickers:
            sector_etf = SECTOR_MAP.get(ticker)
            if not sector_etf or sector_etf not in sector_vols:
                continue
            try:
                price = closes.at[date, ticker]
                upper = bb_upper[ticker].at[date]
                sv_today = stock_vols[ticker].at[date]
                sv_5d = stock_vol_5d_avg[ticker].at[date]
                ev_today = sector_vols[sector_etf].at[date]
                ev_5d = sector_vol_5d_avg[sector_etf].at[date]
            except (KeyError, ValueError):
                continue
            if any(pd.isna(x) for x in [price, upper, sv_today, sv_5d, ev_today, ev_5d]):
                continue

            # Stock breaks above upper Bollinger
            # Idiosyncratic vol rising (today > 5d avg)
            # Sector vol falling (today < 5d avg)
            idio_today = sv_today - 0.8 * ev_today
            idio_5d = sv_5d - 0.8 * ev_5d

            if (price > upper and
                    idio_today > idio_5d and
                    ev_today < ev_5d):
                signals.append({
                    'date': date, 'ticker': ticker,
                    'price': price, 'bb_upper': upper,
                    'idio_vol': idio_today,
                })

    return pd.DataFrame(signals), 7


# ── Backtester ────────────────────────────────────────────────────────────
def backtest(signals_df, hold_days, closes, spy_close, has_direction=False):
    """
    Run backtest with position limits and regime hedging.
    supports long and short signals via 'direction' column.
    Returns trade-level results and equity curve.
    """
    if signals_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    spy_sma200 = spy_close.rolling(200).mean()

    signals_df = signals_df.sort_values('date').reset_index(drop=True)

    # OOT filter
    signals_df = signals_df[(signals_df['date'] >= OOT_START) & (signals_df['date'] <= OOT_END)]
    if signals_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    trades = []
    active_positions = []  # (exit_date, ticker)
    dates = closes.index

    for _, sig in signals_df.iterrows():
        entry_date = sig['date']
        ticker = sig['ticker']
        direction = sig.get('direction', 'long') if has_direction else 'long'

        # Clean active positions
        active_positions = [(ed, t) for ed, t in active_positions if ed > entry_date]
        if len(active_positions) >= MAX_POSITIONS:
            continue
        if ticker in [t for _, t in active_positions]:
            continue

        try:
            entry_idx = dates.get_loc(entry_date)
        except (KeyError, ValueError):
            continue

        exit_idx = min(entry_idx + hold_days, len(dates) - 1)
        if exit_idx <= entry_idx:
            continue
        exit_date = dates[exit_idx]

        try:
            entry_price = closes.at[entry_date, ticker]
            exit_price = closes.at[exit_date, ticker]
        except (KeyError, ValueError):
            continue

        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            continue

        # Regime hedge
        try:
            spy_val = spy_close.at[entry_date]
            sma_val = spy_sma200.at[entry_date]
            regime_bear = pd.notna(sma_val) and spy_val < sma_val
        except (KeyError, ValueError):
            regime_bear = False

        size_mult = 0.5 if regime_bear else 1.0

        # Return calc
        if direction == 'long':
            raw_ret = (exit_price / entry_price) - 1
        else:
            raw_ret = (entry_price / exit_price) - 1  # short

        net_ret = raw_ret - 2 * SLIPPAGE_PCT
        weighted_ret = net_ret * size_mult

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'ticker': ticker,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'raw_ret': raw_ret,
            'net_ret': net_ret,
            'weighted_ret': weighted_ret,
            'size_mult': size_mult,
            'regime': 'bear' if regime_bear else 'bull',
        })
        active_positions.append((exit_date, ticker))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, pd.Series(dtype=float)

    # Build daily equity curve
    all_dates = closes.loc[OOT_START:OOT_END].index
    daily_rets = pd.Series(0.0, index=all_dates)

    for _, trade in trades_df.iterrows():
        t_dates = closes.loc[trade['entry_date']:trade['exit_date']].index
        if len(t_dates) < 2:
            continue
        ticker = trade['ticker']
        direction = trade['direction']
        size = trade['size_mult'] / MAX_POSITIONS

        for i in range(1, len(t_dates)):
            d = t_dates[i]
            try:
                p0 = closes.at[t_dates[i - 1], ticker]
                p1 = closes.at[d, ticker]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    day_ret = ((p1 / p0) - 1) * size
                    if direction == 'short':
                        day_ret = -day_ret
                    daily_rets.at[d] += day_ret
            except (KeyError, ValueError):
                pass

    equity = (1 + daily_rets).cumprod() * STARTING_CAPITAL
    return trades_df, equity


# ── Validation Gates ──────────────────────────────────────────────────────
def compute_sharpe(trades_df):
    if trades_df.empty or len(trades_df) < 2:
        return 0.0
    rets = trades_df['weighted_ret'].values
    if rets.std() == 0:
        return 0.0
    years = max((trades_df['exit_date'].max() - trades_df['entry_date'].min()).days / 365.25, 0.5)
    trades_per_year = len(trades_df) / years
    return (rets.mean() / rets.std()) * np.sqrt(trades_per_year)


def compute_sortino(trades_df):
    if trades_df.empty or len(trades_df) < 2:
        return 0.0
    rets = trades_df['weighted_ret'].values
    downside = rets[rets < 0]
    if len(downside) < 2:
        return 10.0
    down_std = downside.std()
    if down_std == 0:
        return 0.0
    years = max((trades_df['exit_date'].max() - trades_df['entry_date'].min()).days / 365.25, 0.5)
    trades_per_year = len(trades_df) / years
    return (rets.mean() / down_std) * np.sqrt(trades_per_year)


def compute_max_dd(equity):
    if equity.empty:
        return -1.0
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return dd.min()


def permutation_test(trades_df, n_perms=PERM_ITERATIONS):
    """Shuffle entry dates (sign-flip test) to compute p-value."""
    if trades_df.empty or len(trades_df) < 5:
        return 1.0
    rets = trades_df['weighted_ret'].values
    observed_mean = rets.mean()
    count_above = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(rets))
        perm_mean = (rets * signs).mean()
        if perm_mean >= observed_mean:
            count_above += 1
    return count_above / n_perms


def regime_gap(trades_df):
    bull = trades_df[trades_df['regime'] == 'bull']
    bear = trades_df[trades_df['regime'] == 'bear']
    if bull.empty or bear.empty:
        return 0.0  # only one regime, can't compute gap
    s_bull = bull['weighted_ret'].mean() / max(bull['weighted_ret'].std(), 1e-8)
    s_bear = bear['weighted_ret'].mean() / max(bear['weighted_ret'].std(), 1e-8)
    denom = max(abs(s_bull), abs(s_bear), 1e-8)
    return abs(s_bull - s_bear) / denom


def validate_5gate(trades_df, equity, variant_name):
    n_trades = len(trades_df)
    if n_trades == 0:
        return {
            'variant': variant_name, 'n_trades': 0,
            'sharpe': 0, 'sortino': 0, 'perm_p': 1.0,
            'regime_gap': 1.0, 'max_dd_pct': -100.0,
            'win_rate_pct': 0, 'avg_ret_pct': 0, 'profit_factor': 0,
            'final_equity': STARTING_CAPITAL,
            'gate_1_sharpe_gt_0.5': False, 'gate_2_perm_p_lt_0.05': False,
            'gate_3_regime_gap_lt_0.5': False, 'gate_4_maxdd_gt_neg50': False,
            'gate_5_trades_gte_20': False,
            'gates_passed': 0, 'all_gates_pass': False,
        }

    sharpe = compute_sharpe(trades_df)
    sortino = compute_sortino(trades_df)
    perm_p = permutation_test(trades_df, n_perms=PERM_ITERATIONS)
    rg = regime_gap(trades_df)
    mdd = compute_max_dd(equity) if not equity.empty else -1.0

    wr = (trades_df['weighted_ret'] > 0).mean()
    avg_ret = trades_df['weighted_ret'].mean()
    wins = trades_df[trades_df['weighted_ret'] > 0]['weighted_ret'].sum()
    losses = abs(trades_df[trades_df['weighted_ret'] < 0]['weighted_ret'].sum())
    pf = wins / max(losses, 1e-8)

    final_eq = equity.iloc[-1] if not equity.empty else STARTING_CAPITAL

    bull_trades = trades_df[trades_df['regime'] == 'bull']
    bear_trades = trades_df[trades_df['regime'] == 'bear']

    g1 = sharpe > 0.5
    g2 = perm_p < 0.05
    g3 = rg < 0.5
    g4 = mdd > -0.50
    g5 = n_trades >= 20

    return {
        'variant': variant_name,
        'n_trades': n_trades,
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'perm_p': round(float(perm_p), 3),
        'regime_gap': round(float(rg), 3),
        'max_dd_pct': round(float(mdd * 100), 2),
        'win_rate_pct': round(float(wr * 100), 1),
        'avg_ret_pct': round(float(avg_ret * 100), 3),
        'profit_factor': round(float(pf), 2),
        'final_equity': round(float(final_eq), 2),
        'n_bull_trades': int(len(bull_trades)),
        'n_bear_trades': int(len(bear_trades)),
        'bull_wr_pct': round(float((bull_trades['weighted_ret'] > 0).mean() * 100), 1) if len(bull_trades) > 0 else 0,
        'bear_wr_pct': round(float((bear_trades['weighted_ret'] > 0).mean() * 100), 1) if len(bear_trades) > 0 else 0,
        'gate_1_sharpe_gt_0.5': bool(g1),
        'gate_2_perm_p_lt_0.05': bool(g2),
        'gate_3_regime_gap_lt_0.5': bool(g3),
        'gate_4_maxdd_gt_neg50': bool(g4),
        'gate_5_trades_gte_20': bool(g5),
        'gates_passed': int(sum([g1, g2, g3, g4, g5])),
        'all_gates_pass': bool(all([g1, g2, g3, g4, g5])),
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("IDIOSYNCRATIC VOLATILITY STRATEGIES BACKTEST")
    print("=" * 70)
    print(f"Universe: {len(TICKERS)} S&P 500 stocks")
    print(f"OOT Period: {OOT_START.date()} to {OOT_END.date()}")
    print(f"Starting Capital: ${STARTING_CAPITAL}")
    print(f"Max Positions: {MAX_POSITIONS}")
    print(f"Slippage: {SLIPPAGE_PCT*100:.2f}% each way")
    print(f"Permutation iterations: {PERM_ITERATIONS}")

    closes, volumes = download_data()
    spy_close = closes['SPY'].copy()

    generators = {
        'A': ('Vol Compression Buy (idio vol < 20th pctl, 10d)', generate_signals_A, False),
        'B': ('Vol Expansion Capitulation (idio vol > 80th + down >3%, 5d)', generate_signals_B, False),
        'C': ('Vol Ratio Mean Reversion (ratio extremes, 10d)', generate_signals_C, True),
        'D': ('Low Idio Vol Momentum (top quintile mom + low vol, 15d)', generate_signals_D, False),
        'E': ('Post-Vol-Crush Recovery (vol crush >50% + near high, 10d)', generate_signals_E, False),
        'F': ('Regime-Adj Vol Breakout (BB break + rising idio vol, 7d)', generate_signals_F, False),
    }

    all_results = {}
    variant_summaries = {}

    for key, (desc, gen_func, has_dir) in generators.items():
        print(f"\n{'=' * 60}")
        print(f"Variant {key}: {desc}")
        print(f"{'=' * 60}")

        signals_df, default_hold = gen_func(closes, volumes, spy_close)
        print(f"  Raw signals: {len(signals_df)}")

        if signals_df.empty:
            print("  NO SIGNALS -- skipping")
            result = validate_5gate(pd.DataFrame(), pd.Series(dtype=float), f"Variant_{key}")
            variant_summaries[key] = result
            all_results[f"Variant_{key}"] = result
            continue

        trades_df, equity = backtest(signals_df, default_hold, closes, spy_close, has_direction=has_dir)
        result = validate_5gate(trades_df, equity, f"Variant_{key}")

        print(f"  Trades (OOT): {result['n_trades']}")
        print(f"  Sharpe:       {result['sharpe']}")
        print(f"  Sortino:      {result['sortino']}")
        print(f"  Win Rate:     {result['win_rate_pct']}%")
        print(f"  Avg Return:   {result['avg_ret_pct']}%")
        print(f"  Profit Factor:{result['profit_factor']}")
        print(f"  Max DD:       {result['max_dd_pct']}%")
        print(f"  Perm Test p:  {result['perm_p']}")
        print(f"  Regime Gap:   {result['regime_gap']}")
        print(f"  Final Equity: ${result['final_equity']}")
        print(f"  Bull/Bear:    {result['n_bull_trades']}/{result['n_bear_trades']} trades")
        print(f"  Gates Passed: {result['gates_passed']}/5")
        print(f"  ALL PASS:     {'YES' if result['all_gates_pass'] else 'NO'}")

        variant_summaries[key] = result
        all_results[f"Variant_{key}"] = result

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("SUMMARY: 5-GATE VALIDATION")
    print(f"{'=' * 80}")
    print(f"{'Var':<4} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'PF':>6} "
          f"{'MaxDD%':>7} {'PermP':>6} {'RGap':>6} {'N':>5} {'$Final':>8} {'Pass':>6}")
    print("-" * 80)

    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        if key in variant_summaries:
            r = variant_summaries[key]
            status = "PASS" if r['all_gates_pass'] else f"{r['gates_passed']}/5"
            print(f"  {key:<3} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['win_rate_pct']:>5.1f}% "
                  f"{r['profit_factor']:>6.2f} {r['max_dd_pct']:>6.2f}% {r['perm_p']:>6.3f} "
                  f"{r['regime_gap']:>6.3f} {r['n_trades']:>5d} {r['final_equity']:>7.2f} {status:>6}")

    # Find best
    best_key = None
    best_gates = 0
    best_sharpe = -999
    for key, r in variant_summaries.items():
        if r['gates_passed'] > best_gates or (r['gates_passed'] == best_gates and r['sharpe'] > best_sharpe):
            best_key = key
            best_gates = r['gates_passed']
            best_sharpe = r['sharpe']

    if best_key:
        print(f"\nBest Variant: {best_key} ({best_gates}/5 gates, Sharpe={best_sharpe:.3f})")

    # ── Save Results ──────────────────────────────────────────────────
    output = {
        'strategy': 'Idiosyncratic Volatility Strategies',
        'oot_period': f'{OOT_START.date()} to {OOT_END.date()}',
        'universe': f'S&P 500 ({len(TICKERS)} large-cap)',
        'cost_model': f'0.02% slippage each way',
        'max_positions': MAX_POSITIONS,
        'starting_capital': STARTING_CAPITAL,
        'regime_hedge': 'Half-size when SPY < 200-SMA',
        'permutation_iterations': PERM_ITERATIONS,
        'run_timestamp': datetime.now().isoformat(),
        'variant_descriptions': {
            'A': 'Vol Compression Buy: idiosyncratic vol < 20th percentile (60d lookback), hold 10d',
            'B': 'Vol Expansion Capitulation: idio vol > 80th percentile + stock down >3% in 5d, hold 5d',
            'C': 'Vol Ratio Mean Reversion: stock_vol/sector_vol < 0.5 (buy) or > 2.0 (short), hold 10d',
            'D': 'Low Idio Vol Momentum: top quintile 20d momentum + below-median idio vol, hold 15d',
            'E': 'Post-Vol-Crush Recovery: 5d vol drops >50% from 20d avg + within 5% of 50d high, hold 10d',
            'F': 'Regime-Adj Vol Breakout: breaks 20d Bollinger + rising idio vol + falling sector vol, hold 7d',
        },
        'variant_results': {},
    }

    for key, r in variant_summaries.items():
        output['variant_results'][key] = r

    out_path = Path('/home/jupiter/Lvl3Quant/data/idiosyncratic_vol_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
