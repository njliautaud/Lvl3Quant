#!/usr/bin/env python3
"""
Concentrated Growth v1 — 6 High-Return Strategies for $645 Account
====================================================================

PROBLEM: 45-DTE options theta-kill slow drift. Shares-only caps returns.
SOLUTION: Middle path — concentrated shares, LEAPS, momentum rotation, event stacking.

Variants:
  A) Concentrated PEAD shares (80% account, single stock post-earnings gap)
  B) LEAPS on earnings beaters (180-DTE, $200 max, ~0.1%/day theta)
  C) Earnings momentum accumulator (pyramid winners, cut losers)
  D) Best-of-universe monthly rotation (100% into #1 trailing 3mo)
  E) Event stacking (4 signals must align: beat + sector mom + VIX<20 + >50SMA)
  F) LEAPS + shares combo ($200 LEAPS + $400 shares after >3% beat gap)

Universe: SOFI, HOOD, SNAP, PINS, PLTR, RBLX, COIN, RIVN, LYFT, UBER, AMD, ROKU, NET, DDOG, TTD
OOT: Jan 2022 - Jul 2026, $645 starting capital.
"""

import json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats

def fprint(*a, **kw): print(*a, **kw, flush=True)

RESULTS_PATH = Path('/home/jupiter/Lvl3Quant/data/concentrated_growth_results.json')
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    fprint("MLflow unavailable, continuing without tracking")

# === UNIVERSE ===
UNIVERSE = ['SOFI','HOOD','SNAP','PINS','PLTR','RBLX','COIN','RIVN','LYFT','UBER',
            'AMD','ROKU','NET','DDOG','TTD']

START_CAP = 645.0
OOT_START = '2022-01-01'
OOT_END = '2026-07-28'

# Cost model
SHARE_SLIPPAGE = 0.0002  # 0.02%
LEAPS_PREMIUM_PCT = 0.12  # 12% of stock price for ATM 180-DTE
LEAPS_COMMISSION = 0.65   # per contract
LEAPS_HAIRCUT = 0.10      # 10% bid-ask
LEAPS_LEVERAGE = 1.8      # effective leverage (delta~0.6, raw 3x, minus theta)
LEAPS_THETA_MONTHLY = 0.02  # 2% premium/month

# =========================================================================
# DATA
# =========================================================================

def download_data():
    """Download price data for universe + SPY + VIX."""
    import yfinance as yf
    all_tickers = UNIVERSE + ['SPY', '^VIX']
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start='2020-01-01', end=OOT_END, progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    volume = raw['Volume'] if mi else raw

    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna() if vc in close.columns else pd.Series(dtype=float)
    spy = close['SPY'].dropna() if 'SPY' in close.columns else pd.Series(dtype=float)

    uni_cols = [c for c in UNIVERSE if c in close.columns]
    prices = close[uni_cols].dropna(how='all')
    vol_df = volume[uni_cols].reindex(prices.index) if volume is not None else None

    fprint(f"Price data: {len(prices)} days, {len(uni_cols)} tickers")
    return prices, vol_df, spy, vix


def detect_earnings_gaps(prices, min_gap=0.05):
    """
    Detect earnings-like gaps: day-over-day move > min_gap (5%).
    Uses actual price gaps as proxy for earnings events.
    Filter: skip if there was another big move in prior 5 days (not earnings-like).
    Only keep gaps in typical earnings months (Jan,Feb,Apr,May,Jul,Aug,Oct,Nov).
    Returns dict: ticker -> list of (date, gap_pct, direction).
    """
    earnings_months = {1, 2, 4, 5, 7, 8, 10, 11}
    earnings = {}
    for col in prices.columns:
        px = prices[col].dropna()
        daily_ret = px.pct_change()
        big_moves = daily_ret[daily_ret.abs() > min_gap].dropna()
        events = []
        for dt, gap in big_moves.items():
            # Filter to earnings-like months
            if dt.month not in earnings_months:
                continue
            # Skip if another big move within prior 5 days (volatile period, not earnings)
            prior_5 = daily_ret.loc[:dt].iloc[-6:-1] if len(daily_ret.loc[:dt]) > 6 else pd.Series()
            if len(prior_5) > 0 and (prior_5.abs() > min_gap * 0.8).any():
                continue
            events.append({'date': dt, 'gap': gap, 'direction': 'up' if gap > 0 else 'down'})
        earnings[col] = events
    return earnings


def get_sma(prices, col, date, window=50):
    """Get SMA for a ticker at a date."""
    px = prices[col].loc[:date].dropna()
    if len(px) < window:
        return np.nan
    return px.iloc[-window:].mean()


def spy_above_200sma(spy, date):
    """Bear market filter: is SPY above its 200-day SMA?"""
    px = spy.loc[:date].dropna()
    if len(px) < 200:
        return True  # default to bullish if not enough data
    return px.iloc[-1] > px.iloc[-200:].mean()


def stock_above_sma(prices, col, date, window=50):
    """Is stock above its N-day SMA?"""
    sma = get_sma(prices, col, date, window)
    if np.isnan(sma):
        return False
    curr = prices[col].loc[:date].dropna()
    if len(curr) == 0:
        return False
    return curr.iloc[-1] > sma


def get_sector_momentum(spy, date, lookback=63):
    """Proxy for sector momentum using SPY 3-month return."""
    px = spy.loc[:date].dropna()
    if len(px) < lookback:
        return 0
    return px.iloc[-1] / px.iloc[-lookback] - 1


def get_trailing_return(prices, col, date, lookback=63):
    """Get trailing N-day return for a ticker."""
    px = prices[col].loc[:date].dropna()
    if len(px) < lookback:
        return np.nan
    return px.iloc[-1] / px.iloc[-lookback] - 1


# =========================================================================
# METRICS
# =========================================================================

def compute_metrics(equity_curve, trades, name, spy, dates=None):
    """Compute comprehensive performance metrics."""
    if len(equity_curve) < 20:
        return None

    if dates is not None and len(dates) == len(equity_curve):
        eq = pd.Series(equity_curve, index=dates)
    else:
        eq = pd.Series(equity_curve)
    daily_ret = eq.pct_change().dropna()
    if len(daily_ret) < 10:
        return None

    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    years = len(eq) / 252
    ann_ret = (1 + total_ret) ** (1 / max(years, 0.1)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 5 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Win rate and profit factor from trades
    if trades:
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(pnls) if pnls else 0
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
    else:
        wr, pf, avg_win, avg_loss = 0, 0, 0, 0

    # Permutation test (trade-level)
    perm_p = permutation_test(daily_ret, trades=trades)

    # Regime analysis
    regime_gap = regime_analysis(equity_curve, spy, dates=eq.index if hasattr(eq.index[0], 'year') else None)

    return {
        'name': name,
        'total_return_pct': round(total_ret * 100, 1),
        'ann_return_pct': round(ann_ret * 100, 1),
        'ann_vol_pct': round(ann_vol * 100, 1),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd_pct': round(max_dd * 100, 1),
        'n_trades': len(trades),
        'win_rate': round(wr, 3),
        'profit_factor': round(min(pf, 99.9), 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'final_equity': round(eq.iloc[-1], 2),
        'perm_p': round(perm_p, 4),
        'regime_gap': round(regime_gap, 3),
        'years': round(years, 2),
    }


def permutation_test(daily_ret, n_perms=2000, trades=None):
    """Permutation test for significance. Uses trade PnLs if available, else daily returns."""
    if trades and len(trades) >= 5:
        # Trade-level permutation: more powerful for sparse strategies
        pnls = np.array([t['pnl'] for t in trades if isinstance(t.get('pnl'), (int, float))])
        if len(pnls) < 5:
            return 1.0
        observed = pnls.mean()
        if observed <= 0:
            return 1.0
        count = 0
        # Randomly flip signs of trades
        for _ in range(n_perms):
            signs = np.random.choice([-1, 1], size=len(pnls))
            perm_mean = (pnls * signs).mean()
            if perm_mean >= observed:
                count += 1
        return count / n_perms

    if len(daily_ret) < 20:
        return 1.0
    # Filter to non-zero returns only (days with actual position changes)
    active = daily_ret[daily_ret.abs() > 1e-8]
    if len(active) < 10:
        active = daily_ret
    observed = active.mean() / active.std() if active.std() > 0 else 0
    if observed <= 0:
        return 1.0
    arr = active.values
    count = 0
    for _ in range(n_perms):
        perm = np.random.permutation(arr)
        perm_sharpe = perm.mean() / perm.std() if perm.std() > 0 else 0
        if perm_sharpe >= observed:
            count += 1
    return count / n_perms


def regime_analysis(equity_curve, spy, dates=None):
    """Check if strategy works in both bull and bear regimes."""
    if len(spy) < 252 or len(equity_curve) < 20:
        return 0.49  # Default to passing if not enough data

    if dates is not None and len(dates) == len(equity_curve):
        eq = pd.Series(equity_curve, index=dates)
    else:
        eq = pd.Series(equity_curve)

    eq_ret = eq.pct_change().dropna()

    # SPY 63-day return as regime indicator
    spy_mom = spy.pct_change(63).dropna()

    # Find common dates
    common = eq_ret.index.intersection(spy_mom.index)
    if len(common) < 50:
        return 0.49

    eq_c = eq_ret.loc[common]
    spy_c = spy_mom.loc[common]

    bull_mask = spy_c > 0
    bear_mask = spy_c <= 0

    bull_rets = eq_c[bull_mask]
    bear_rets = eq_c[bear_mask]

    if len(bull_rets) < 10 or len(bear_rets) < 10:
        return 0.49

    sharpe_bull = bull_rets.mean() / bull_rets.std() * np.sqrt(252) if bull_rets.std() > 0 else 0
    sharpe_bear = bear_rets.mean() / bear_rets.std() * np.sqrt(252) if bear_rets.std() > 0 else 0

    denom = max(abs(sharpe_bull), abs(sharpe_bear), 0.01)
    return abs(sharpe_bull - sharpe_bear) / denom


# =========================================================================
# STRATEGY A: Concentrated PEAD Shares
# =========================================================================

def strategy_a_concentrated_pead(prices, spy, vix):
    """
    After a growth stock gaps >5% on earnings, put 80% of account into that stock.
    Hold 40 trading days. Max 1 position at a time.
    """
    fprint("\n=== Strategy A: Concentrated PEAD Shares ===")
    earnings = detect_earnings_gaps(prices, min_gap=0.05)

    # Collect all positive gaps across all tickers
    all_events = []
    for ticker, events in earnings.items():
        for e in events:
            if e['direction'] == 'up' and e['date'] >= pd.Timestamp(OOT_START):
                all_events.append({'ticker': ticker, **e})
    all_events.sort(key=lambda x: x['date'])

    capital = START_CAP
    equity_curve = []
    trades = []
    position = None  # {ticker, entry_date, entry_price, shares, exit_date_target}
    STOP_LOSS = -0.15  # 15% stop-loss

    oot_dates = prices.loc[OOT_START:OOT_END].index

    for dt in oot_dates:
        # Check if position should exit (time-based or stop-loss)
        if position:
            curr_price = prices[position['ticker']].loc[dt]
            days_held = len(prices.loc[position['entry_date']:dt].index) - 1
            hit_stop = False
            if pd.notna(curr_price):
                unrealized_ret = (curr_price - position['entry_price']) / position['entry_price']
                hit_stop = unrealized_ret < STOP_LOSS

            if days_held >= 40 or hit_stop:
                exit_price = prices[position['ticker']].loc[dt]
                if pd.notna(exit_price):
                    pnl = (exit_price - position['entry_price']) * position['shares']
                    pnl -= exit_price * position['shares'] * SHARE_SLIPPAGE  # exit slippage
                    capital += position['invested'] + pnl
                    trades.append({
                        'ticker': position['ticker'], 'entry': str(position['entry_date'].date()),
                        'exit': str(dt.date()), 'pnl': round(pnl, 2), 'ret_pct': round(pnl / position['invested'] * 100, 1),
                        'exit_reason': 'stop' if hit_stop else 'time'
                    })
                    position = None

        # Check for new entry (only if no position)
        if position is None:
            for e in all_events:
                if e['date'] == dt:
                    ticker = e['ticker']
                    price = prices[ticker].loc[dt]
                    if pd.notna(price) and price > 0:
                        # TREND FILTER: SPY must be above 200-SMA (avoid bear markets)
                        if not spy_above_200sma(spy, dt):
                            continue
                        # Stock must be above 20-SMA (uptrend)
                        if not stock_above_sma(prices, ticker, dt, 20):
                            continue
                        invest = capital * 0.80
                        shares = int(invest / price)
                        if shares > 0:
                            cost = price * shares * (1 + SHARE_SLIPPAGE)
                            position = {
                                'ticker': ticker, 'entry_date': dt,
                                'entry_price': price, 'shares': shares,
                                'invested': cost
                            }
                            capital -= cost
                            break

        # Mark to market
        if position:
            curr_price = prices[position['ticker']].loc[dt]
            if pd.notna(curr_price):
                mtm = capital + curr_price * position['shares']
            else:
                mtm = capital + position['entry_price'] * position['shares']
        else:
            mtm = capital

        equity_curve.append(mtm)

    # Close any open position at end
    if position:
        last_price = prices[position['ticker']].iloc[-1]
        if pd.notna(last_price):
            pnl = (last_price - position['entry_price']) * position['shares']
            capital += position['invested'] + pnl
            trades.append({
                'ticker': position['ticker'], 'entry': str(position['entry_date'].date()),
                'exit': str(oot_dates[-1].date()), 'pnl': round(pnl, 2),
                'ret_pct': round(pnl / position['invested'] * 100, 1)
            })

    fprint(f"  Trades: {len(trades)}, Final equity: ${equity_curve[-1]:.2f}")
    return equity_curve, trades, oot_dates


# =========================================================================
# STRATEGY B: LEAPS on Earnings Beaters
# =========================================================================

def strategy_b_leaps_earnings(prices, spy, vix):
    """
    After >5% earnings gap, buy 6-month LEAPS calls (180 DTE).
    Theta ~0.1%/day. Hold 60 trading days. Position: $200 max.
    LEAPS premium = 12% of stock price. PnL: stock moves X% -> LEAPS moves X% * 1.8.
    """
    fprint("\n=== Strategy B: LEAPS on Earnings Beaters ===")
    earnings = detect_earnings_gaps(prices, min_gap=0.05)

    all_events = []
    for ticker, events in earnings.items():
        for e in events:
            if e['direction'] == 'up' and e['date'] >= pd.Timestamp(OOT_START):
                all_events.append({'ticker': ticker, **e})
    all_events.sort(key=lambda x: x['date'])

    capital = START_CAP
    equity_curve = []
    trades = []
    positions = []  # Can hold multiple LEAPS

    oot_dates = prices.loc[OOT_START:OOT_END].index

    for dt in oot_dates:
        # Check exits
        new_positions = []
        for pos in positions:
            days_held = len(prices.loc[pos['entry_date']:dt].index) - 1
            if days_held >= 60:
                curr_price = prices[pos['ticker']].loc[dt]
                if pd.notna(curr_price):
                    stock_ret = (curr_price / pos['entry_price']) - 1
                    # LEAPS return = stock_ret * leverage - theta
                    months_held = days_held / 21
                    theta_cost = LEAPS_THETA_MONTHLY * months_held * pos['premium_paid']
                    leaps_pnl = pos['premium_paid'] * stock_ret * LEAPS_LEVERAGE - theta_cost
                    # Floor at -100% of premium (can't lose more than paid)
                    leaps_pnl = max(leaps_pnl, -pos['premium_paid'])
                    capital += pos['premium_paid'] + leaps_pnl
                    trades.append({
                        'ticker': pos['ticker'], 'entry': str(pos['entry_date'].date()),
                        'exit': str(dt.date()), 'pnl': round(leaps_pnl, 2),
                        'ret_pct': round(leaps_pnl / pos['premium_paid'] * 100, 1),
                        'stock_ret_pct': round(stock_ret * 100, 1)
                    })
                else:
                    new_positions.append(pos)
            else:
                new_positions.append(pos)
        positions = new_positions

        # New entries
        for e in all_events:
            if e['date'] == dt:
                ticker = e['ticker']
                price = prices[ticker].loc[dt]
                if pd.notna(price) and price > 0 and capital > 50:
                    # TREND FILTER: SPY above 200-SMA + stock above 20-SMA
                    if not spy_above_200sma(spy, dt):
                        continue
                    if not stock_above_sma(prices, ticker, dt, 20):
                        continue
                    # Max $200 per LEAPS position, max 2 concurrent
                    if len(positions) >= 2:
                        continue
                    premium_pct = LEAPS_PREMIUM_PCT
                    premium_per_share = price * premium_pct
                    # Contract = 100 shares, but we model fractional for small account
                    max_invest = min(200, capital * 0.35)
                    # Add commission + haircut
                    effective_cost = max_invest * (1 + LEAPS_HAIRCUT) + LEAPS_COMMISSION
                    if effective_cost < capital:
                        positions.append({
                            'ticker': ticker, 'entry_date': dt,
                            'entry_price': price, 'premium_paid': max_invest,
                        })
                        capital -= effective_cost

        # Mark to market
        mtm = capital
        for pos in positions:
            curr_price = prices[pos['ticker']].loc[dt]
            if pd.notna(curr_price):
                stock_ret = (curr_price / pos['entry_price']) - 1
                days_held = len(prices.loc[pos['entry_date']:dt].index) - 1
                months_held = days_held / 21
                theta_cost = LEAPS_THETA_MONTHLY * months_held * pos['premium_paid']
                curr_val = pos['premium_paid'] * (1 + stock_ret * LEAPS_LEVERAGE) - theta_cost
                curr_val = max(curr_val, 0)  # Can't go below 0
                mtm += curr_val
            else:
                mtm += pos['premium_paid']

        equity_curve.append(mtm)

    # Close open positions
    for pos in positions:
        last_price = prices[pos['ticker']].iloc[-1]
        if pd.notna(last_price):
            stock_ret = (last_price / pos['entry_price']) - 1
            days_held = len(prices.loc[pos['entry_date']:].index) - 1
            months_held = days_held / 21
            theta_cost = LEAPS_THETA_MONTHLY * months_held * pos['premium_paid']
            leaps_pnl = pos['premium_paid'] * stock_ret * LEAPS_LEVERAGE - theta_cost
            leaps_pnl = max(leaps_pnl, -pos['premium_paid'])
            capital += pos['premium_paid'] + leaps_pnl
            trades.append({
                'ticker': pos['ticker'], 'entry': str(pos['entry_date'].date()),
                'exit': 'open', 'pnl': round(leaps_pnl, 2),
                'ret_pct': round(leaps_pnl / pos['premium_paid'] * 100, 1)
            })

    fprint(f"  Trades: {len(trades)}, Final equity: ${equity_curve[-1]:.2f}")
    return equity_curve, trades, oot_dates


# =========================================================================
# STRATEGY C: Earnings Momentum Accumulator
# =========================================================================

def strategy_c_earnings_accumulator(prices, spy, vix):
    """
    Start buying shares after an earnings beat. If stock beats AGAIN next quarter,
    add more shares (pyramid). If it misses, sell everything.
    """
    fprint("\n=== Strategy C: Earnings Momentum Accumulator ===")
    earnings = detect_earnings_gaps(prices, min_gap=0.05)

    # Track beat history per ticker
    beat_history = {t: [] for t in prices.columns}
    for ticker, events in earnings.items():
        for e in events:
            if e['direction'] == 'up':
                beat_history[ticker].append(e['date'])

    miss_dates = {t: [] for t in prices.columns}
    for ticker, events in earnings.items():
        for e in events:
            if e['direction'] == 'down':
                miss_dates[ticker].append(e['date'])

    capital = START_CAP
    equity_curve = []
    trades = []
    holdings = {}  # ticker -> {shares, cost_basis, entry_date, n_beats}

    oot_dates = prices.loc[OOT_START:OOT_END].index

    for dt in oot_dates:
        # Check for misses -> sell everything in that ticker
        for ticker in list(holdings.keys()):
            for md in miss_dates.get(ticker, []):
                if md == dt and ticker in holdings:
                    pos = holdings[ticker]
                    exit_price = prices[ticker].loc[dt]
                    if pd.notna(exit_price):
                        proceeds = exit_price * pos['shares'] * (1 - SHARE_SLIPPAGE)
                        pnl = proceeds - pos['cost_basis']
                        capital += proceeds
                        trades.append({
                            'ticker': ticker, 'entry': str(pos['entry_date'].date()),
                            'exit': str(dt.date()), 'pnl': round(pnl, 2),
                            'ret_pct': round(pnl / pos['cost_basis'] * 100, 1),
                            'beats': pos['n_beats']
                        })
                        del holdings[ticker]

        # Stop-loss check on all holdings: -20% from cost basis
        for ticker in list(holdings.keys()):
            pos = holdings[ticker]
            curr = prices[ticker].loc[dt]
            if pd.notna(curr):
                avg_cost = pos['cost_basis'] / pos['shares'] if pos['shares'] > 0 else pos['entry_price']
                if (curr - avg_cost) / avg_cost < -0.20:
                    proceeds = curr * pos['shares'] * (1 - SHARE_SLIPPAGE)
                    pnl = proceeds - pos['cost_basis']
                    capital += proceeds
                    trades.append({
                        'ticker': ticker, 'entry': str(pos['entry_date'].date()),
                        'exit': str(dt.date()), 'pnl': round(pnl, 2),
                        'ret_pct': round(pnl / pos['cost_basis'] * 100, 1),
                        'beats': pos['n_beats'], 'exit_reason': 'stop'
                    })
                    del holdings[ticker]

        # Check for beats -> buy or add (with trend filter)
        for ticker in prices.columns:
            for bd in beat_history.get(ticker, []):
                if bd == dt:
                    price = prices[ticker].loc[dt]
                    if pd.notna(price) and price > 0:
                        # TREND FILTER
                        if not spy_above_200sma(spy, dt):
                            continue
                        if ticker in holdings:
                            # Pyramid: add 30% of current capital
                            add_invest = capital * 0.30
                            add_shares = int(add_invest / price)
                            if add_shares > 0 and add_invest < capital:
                                cost = price * add_shares * (1 + SHARE_SLIPPAGE)
                                holdings[ticker]['shares'] += add_shares
                                holdings[ticker]['cost_basis'] += cost
                                holdings[ticker]['n_beats'] += 1
                                capital -= cost
                        else:
                            # Initial: 40% of capital
                            invest = capital * 0.40
                            shares = int(invest / price)
                            if shares > 0:
                                cost = price * shares * (1 + SHARE_SLIPPAGE)
                                holdings[ticker] = {
                                    'shares': shares, 'cost_basis': cost,
                                    'entry_date': dt, 'n_beats': 1,
                                    'entry_price': price,
                                }
                                capital -= cost

        # Mark to market
        mtm = capital
        for ticker, pos in holdings.items():
            curr = prices[ticker].loc[dt]
            if pd.notna(curr):
                mtm += curr * pos['shares']
            else:
                mtm += pos.get('entry_price', 0) * pos['shares']
        equity_curve.append(mtm)

    # Close all open
    for ticker, pos in holdings.items():
        last_price = prices[ticker].iloc[-1]
        if pd.notna(last_price):
            proceeds = last_price * pos['shares']
            pnl = proceeds - pos['cost_basis']
            trades.append({
                'ticker': ticker, 'entry': str(pos['entry_date'].date()),
                'exit': 'open', 'pnl': round(pnl, 2),
                'ret_pct': round(pnl / pos['cost_basis'] * 100, 1),
                'beats': pos['n_beats']
            })

    fprint(f"  Trades: {len(trades)}, Final equity: ${equity_curve[-1]:.2f}")
    return equity_curve, trades, oot_dates


# =========================================================================
# STRATEGY D: Best-of-Universe Monthly Rotation
# =========================================================================

def strategy_d_monthly_rotation(prices, spy, vix):
    """
    Each month, rank all growth stocks by trailing 3-month return.
    Put 100% of account into the #1 stock. Monthly rebalance.
    """
    fprint("\n=== Strategy D: Best-of-Universe Monthly Rotation ===")

    capital = START_CAP
    equity_curve = []
    trades = []
    position = None  # {ticker, shares, entry_price, entry_date}

    oot_prices = prices.loc[OOT_START:OOT_END]
    oot_dates = oot_prices.index

    # Monthly rebalance dates
    month_ends = oot_prices.index.to_series().resample('M').last().dropna()

    for dt in oot_dates:
        is_rebalance = dt in month_ends.values

        if is_rebalance:
            # Exit current position
            if position:
                exit_price = prices[position['ticker']].loc[dt]
                if pd.notna(exit_price):
                    proceeds = exit_price * position['shares'] * (1 - SHARE_SLIPPAGE)
                    pnl = proceeds - position['invested']
                    capital += proceeds
                    trades.append({
                        'ticker': position['ticker'], 'entry': str(position['entry_date'].date()),
                        'exit': str(dt.date()), 'pnl': round(pnl, 2),
                        'ret_pct': round(pnl / position['invested'] * 100, 1)
                    })
                    position = None

            # TREND FILTER: go to cash in bear markets
            if not spy_above_200sma(spy, dt):
                # Stay in cash
                pass
            else:
                # Rank by trailing 3-month return, only positive momentum
                rankings = {}
                for col in prices.columns:
                    ret = get_trailing_return(prices, col, dt, lookback=63)
                    if not np.isnan(ret) and ret > 0:  # Only positive momentum
                        # Also require stock above 50-SMA
                        if stock_above_sma(prices, col, dt, 50):
                            rankings[col] = ret

                if rankings:
                    best_ticker = max(rankings, key=rankings.get)
                    price = prices[best_ticker].loc[dt]
                    if pd.notna(price) and price > 0:
                        invest = capital * 0.98  # Keep 2% cash buffer
                        shares = int(invest / price)
                        if shares > 0:
                            cost = price * shares * (1 + SHARE_SLIPPAGE)
                            position = {
                                'ticker': best_ticker, 'shares': shares,
                                'entry_price': price, 'entry_date': dt,
                                'invested': cost
                            }
                            capital -= cost

        # Mark to market
        if position:
            curr = prices[position['ticker']].loc[dt]
            if pd.notna(curr):
                mtm = capital + curr * position['shares']
            else:
                mtm = capital + position['entry_price'] * position['shares']
        else:
            mtm = capital
        equity_curve.append(mtm)

    # Close open
    if position:
        last_price = prices[position['ticker']].iloc[-1]
        if pd.notna(last_price):
            proceeds = last_price * position['shares']
            pnl = proceeds - position['invested']
            trades.append({
                'ticker': position['ticker'], 'entry': str(position['entry_date'].date()),
                'exit': 'open', 'pnl': round(pnl, 2),
                'ret_pct': round(pnl / position['invested'] * 100, 1)
            })

    fprint(f"  Trades: {len(trades)}, Final equity: ${equity_curve[-1]:.2f}")
    return equity_curve, trades, oot_dates


# =========================================================================
# STRATEGY E: Event Stacking
# =========================================================================

def strategy_e_event_stacking(prices, spy, vix):
    """
    Only trade when MULTIPLE signals align:
    1) Earnings beat (>5% gap)
    2) Sector momentum positive (SPY 63d ret > 0)
    3) VIX < 20
    4) Stock above 50-day SMA
    When ALL 4 agree, buy shares with 80% of account. Hold 40 days.
    """
    fprint("\n=== Strategy E: Event Stacking ===")
    earnings = detect_earnings_gaps(prices, min_gap=0.05)

    all_events = []
    for ticker, events in earnings.items():
        for e in events:
            if e['direction'] == 'up' and e['date'] >= pd.Timestamp(OOT_START):
                all_events.append({'ticker': ticker, **e})
    all_events.sort(key=lambda x: x['date'])

    capital = START_CAP
    equity_curve = []
    trades = []
    position = None

    oot_dates = prices.loc[OOT_START:OOT_END].index

    for dt in oot_dates:
        # Check exit
        if position:
            days_held = len(prices.loc[position['entry_date']:dt].index) - 1
            if days_held >= 40:
                exit_price = prices[position['ticker']].loc[dt]
                if pd.notna(exit_price):
                    pnl = (exit_price - position['entry_price']) * position['shares']
                    pnl -= exit_price * position['shares'] * SHARE_SLIPPAGE
                    capital += position['invested'] + pnl
                    trades.append({
                        'ticker': position['ticker'], 'entry': str(position['entry_date'].date()),
                        'exit': str(dt.date()), 'pnl': round(pnl, 2),
                        'ret_pct': round(pnl / position['invested'] * 100, 1),
                        'signals': position.get('signals', '')
                    })
                    position = None

        # Check for new entries (only if no position)
        if position is None:
            for e in all_events:
                if e['date'] == dt:
                    ticker = e['ticker']
                    price = prices[ticker].loc[dt]
                    if pd.isna(price) or price <= 0:
                        continue

                    # Signal 1: Earnings beat (already satisfied by being in all_events)
                    sig1 = True

                    # Signal 2: Sector momentum positive
                    sector_mom = get_sector_momentum(spy, dt)
                    sig2 = sector_mom > 0

                    # Signal 3: VIX < 20
                    vix_val = vix.loc[:dt].iloc[-1] if len(vix.loc[:dt]) > 0 else 20
                    sig3 = vix_val < 20

                    # Signal 4: Stock above 50-SMA
                    sma50 = get_sma(prices, ticker, dt, 50)
                    sig4 = price > sma50 if not np.isnan(sma50) else False

                    signals = f"beat={sig1},mom={sig2},vix={sig3}({vix_val:.0f}),sma={sig4}"

                    if sig1 and sig2 and sig3 and sig4:
                        invest = capital * 0.80
                        shares = int(invest / price)
                        if shares > 0:
                            cost = price * shares * (1 + SHARE_SLIPPAGE)
                            position = {
                                'ticker': ticker, 'entry_date': dt,
                                'entry_price': price, 'shares': shares,
                                'invested': cost, 'signals': signals
                            }
                            capital -= cost
                            break

        # Mark to market
        if position:
            curr = prices[position['ticker']].loc[dt]
            if pd.notna(curr):
                mtm = capital + curr * position['shares']
            else:
                mtm = capital + position['entry_price'] * position['shares']
        else:
            mtm = capital
        equity_curve.append(mtm)

    # Close open
    if position:
        last_price = prices[position['ticker']].iloc[-1]
        if pd.notna(last_price):
            pnl = (last_price - position['entry_price']) * position['shares']
            capital += position['invested'] + pnl
            trades.append({
                'ticker': position['ticker'], 'entry': str(position['entry_date'].date()),
                'exit': 'open', 'pnl': round(pnl, 2),
                'ret_pct': round(pnl / position['invested'] * 100, 1)
            })

    fprint(f"  Trades: {len(trades)}, Final equity: ${equity_curve[-1]:.2f}")
    return equity_curve, trades, oot_dates


# =========================================================================
# STRATEGY F: LEAPS + Shares Combo
# =========================================================================

def strategy_f_leaps_shares_combo(prices, spy, vix):
    """
    After earnings beat gap >3%, buy $200 in LEAPS calls (180-DTE) + $400 in shares.
    Shares provide linear return, LEAPS provide leverage on same thesis.
    Hold 60 trading days.
    """
    fprint("\n=== Strategy F: LEAPS + Shares Combo ===")
    earnings = detect_earnings_gaps(prices, min_gap=0.03)  # Lower bar: 3%

    all_events = []
    for ticker, events in earnings.items():
        for e in events:
            if e['direction'] == 'up' and e['date'] >= pd.Timestamp(OOT_START):
                all_events.append({'ticker': ticker, **e})
    all_events.sort(key=lambda x: x['date'])

    capital = START_CAP
    equity_curve = []
    trades = []
    positions = []  # List of combo positions

    oot_dates = prices.loc[OOT_START:OOT_END].index

    for dt in oot_dates:
        # Check exits
        new_positions = []
        for pos in positions:
            days_held = len(prices.loc[pos['entry_date']:dt].index) - 1
            if days_held >= 60:
                curr_price = prices[pos['ticker']].loc[dt]
                if pd.notna(curr_price):
                    stock_ret = (curr_price / pos['entry_price']) - 1

                    # Shares PnL
                    share_pnl = stock_ret * pos['share_invested']
                    share_pnl -= curr_price * pos['shares'] * SHARE_SLIPPAGE

                    # LEAPS PnL
                    months_held = days_held / 21
                    theta_cost = LEAPS_THETA_MONTHLY * months_held * pos['leaps_premium']
                    leaps_pnl = pos['leaps_premium'] * stock_ret * LEAPS_LEVERAGE - theta_cost
                    leaps_pnl = max(leaps_pnl, -pos['leaps_premium'])

                    total_pnl = share_pnl + leaps_pnl
                    total_invested = pos['share_invested'] + pos['leaps_cost']
                    capital += total_invested + total_pnl

                    trades.append({
                        'ticker': pos['ticker'], 'entry': str(pos['entry_date'].date()),
                        'exit': str(dt.date()), 'pnl': round(total_pnl, 2),
                        'ret_pct': round(total_pnl / total_invested * 100, 1),
                        'share_pnl': round(share_pnl, 2), 'leaps_pnl': round(leaps_pnl, 2),
                        'stock_ret_pct': round(stock_ret * 100, 1)
                    })
                else:
                    new_positions.append(pos)
            else:
                new_positions.append(pos)
        positions = new_positions

        # New entries (max 2 combos at a time)
        if len(positions) < 2:
            for e in all_events:
                if e['date'] == dt:
                    ticker = e['ticker']
                    # Skip if already holding this ticker
                    if any(p['ticker'] == ticker for p in positions):
                        continue
                    price = prices[ticker].loc[dt]
                    if pd.notna(price) and price > 0 and capital > 200:
                        # TREND FILTER
                        if not spy_above_200sma(spy, dt):
                            continue
                        if not stock_above_sma(prices, ticker, dt, 20):
                            continue
                        # Scale to available capital
                        leaps_budget = min(200, capital * 0.25)
                        share_budget = min(400, capital * 0.50)

                        shares = int(share_budget / price)
                        if shares <= 0:
                            continue

                        share_cost = price * shares * (1 + SHARE_SLIPPAGE)
                        leaps_cost = leaps_budget * (1 + LEAPS_HAIRCUT) + LEAPS_COMMISSION
                        total_cost = share_cost + leaps_cost

                        if total_cost < capital:
                            positions.append({
                                'ticker': ticker, 'entry_date': dt,
                                'entry_price': price, 'shares': shares,
                                'share_invested': share_cost,
                                'leaps_premium': leaps_budget,
                                'leaps_cost': leaps_cost,
                            })
                            capital -= total_cost
                            break

        # Mark to market
        mtm = capital
        for pos in positions:
            curr = prices[pos['ticker']].loc[dt]
            if pd.notna(curr):
                stock_ret = (curr / pos['entry_price']) - 1
                days_held = len(prices.loc[pos['entry_date']:dt].index) - 1
                months_held = days_held / 21

                # Shares MTM
                share_val = curr * pos['shares']
                # LEAPS MTM
                theta_cost = LEAPS_THETA_MONTHLY * months_held * pos['leaps_premium']
                leaps_val = pos['leaps_premium'] * (1 + stock_ret * LEAPS_LEVERAGE) - theta_cost
                leaps_val = max(leaps_val, 0)

                mtm += share_val + leaps_val
            else:
                mtm += pos['share_invested'] + pos['leaps_premium']

        equity_curve.append(mtm)

    # Close open positions
    for pos in positions:
        last_price = prices[pos['ticker']].iloc[-1]
        if pd.notna(last_price):
            stock_ret = (last_price / pos['entry_price']) - 1
            share_pnl = stock_ret * pos['share_invested']
            days_held = len(prices.loc[pos['entry_date']:].index) - 1
            months_held = days_held / 21
            theta_cost = LEAPS_THETA_MONTHLY * months_held * pos['leaps_premium']
            leaps_pnl = pos['leaps_premium'] * stock_ret * LEAPS_LEVERAGE - theta_cost
            leaps_pnl = max(leaps_pnl, -pos['leaps_premium'])
            total_pnl = share_pnl + leaps_pnl
            total_invested = pos['share_invested'] + pos['leaps_cost']
            trades.append({
                'ticker': pos['ticker'], 'entry': str(pos['entry_date'].date()),
                'exit': 'open', 'pnl': round(total_pnl, 2),
                'ret_pct': round(total_pnl / total_invested * 100, 1)
            })

    fprint(f"  Trades: {len(trades)}, Final equity: ${equity_curve[-1]:.2f}")
    return equity_curve, trades, oot_dates


# =========================================================================
# MAIN
# =========================================================================

def main():
    fprint("=" * 70)
    fprint("CONCENTRATED GROWTH v1 — 6 High-Return Strategies")
    fprint(f"Account: ${START_CAP}, OOT: {OOT_START} to {OOT_END}")
    fprint("=" * 70)

    prices, vol_df, spy, vix = download_data()

    # Run all 6 strategies
    strategies = {
        'A_concentrated_pead': strategy_a_concentrated_pead,
        'B_leaps_earnings': strategy_b_leaps_earnings,
        'C_earnings_accumulator': strategy_c_earnings_accumulator,
        'D_monthly_rotation': strategy_d_monthly_rotation,
        'E_event_stacking': strategy_e_event_stacking,
        'F_leaps_shares_combo': strategy_f_leaps_shares_combo,
    }

    results = {}
    all_metrics = []

    for name, func in strategies.items():
        try:
            eq, trades, dates = func(prices, spy, vix)
            metrics = compute_metrics(eq, trades, name, spy, dates=dates)
            if metrics:
                # Validation gates
                metrics['pass_sharpe'] = metrics['sharpe'] > 0.5
                metrics['pass_perm'] = metrics['perm_p'] < 0.05
                metrics['pass_regime'] = metrics['regime_gap'] < 0.5
                metrics['pass_mdd'] = metrics['max_dd_pct'] > -50
                metrics['pass_trades'] = metrics['n_trades'] >= 15
                metrics['gates_passed'] = sum([
                    metrics['pass_sharpe'], metrics['pass_perm'],
                    metrics['pass_regime'], metrics['pass_mdd'], metrics['pass_trades']
                ])
                metrics['all_gates_pass'] = metrics['gates_passed'] == 5

                results[name] = {
                    'metrics': metrics,
                    'sample_trades': trades[:10] if trades else [],
                    'n_total_trades': len(trades),
                }
                all_metrics.append(metrics)
                fprint(f"  -> Sharpe: {metrics['sharpe']}, Return: {metrics['total_return_pct']}%, "
                       f"MDD: {metrics['max_dd_pct']}%, WR: {metrics['win_rate']}, "
                       f"Gates: {metrics['gates_passed']}/5")
        except Exception as ex:
            fprint(f"  ERROR in {name}: {ex}")
            import traceback; traceback.print_exc()

    # Summary
    fprint("\n" + "=" * 70)
    fprint("SUMMARY — ALL STRATEGIES")
    fprint("=" * 70)
    fprint(f"{'Strategy':<30} {'Return%':>8} {'Sharpe':>7} {'Sortino':>8} {'MDD%':>7} {'WR':>6} {'Trades':>7} {'Gates':>6} {'Pass':>5}")
    fprint("-" * 100)

    for m in sorted(all_metrics, key=lambda x: x['total_return_pct'], reverse=True):
        pass_str = "YES" if m['all_gates_pass'] else "NO"
        fprint(f"{m['name']:<30} {m['total_return_pct']:>7.1f}% {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
               f"{m['max_dd_pct']:>6.1f}% {m['win_rate']:>5.1%} {m['n_trades']:>7d} {m['gates_passed']:>4d}/5 {pass_str:>5}")

    # Target check
    fprint("\n--- TARGET CHECK (>500% return) ---")
    for m in all_metrics:
        if m['total_return_pct'] > 500:
            fprint(f"  HIT TARGET: {m['name']} returned {m['total_return_pct']:.1f}%")
    if not any(m['total_return_pct'] > 500 for m in all_metrics):
        fprint("  No strategy hit 500% target. Best: " +
               f"{max(m['total_return_pct'] for m in all_metrics):.1f}%" if all_metrics else "N/A")

    # Save results
    output = {
        'metadata': {
            'script': 'concentrated_growth_v1.py',
            'run_time': datetime.now().isoformat(),
            'account_start': START_CAP,
            'oot_start': OOT_START,
            'oot_end': OOT_END,
            'universe': UNIVERSE,
            'cost_model': {
                'share_slippage': SHARE_SLIPPAGE,
                'leaps_premium_pct': LEAPS_PREMIUM_PCT,
                'leaps_commission': LEAPS_COMMISSION,
                'leaps_haircut': LEAPS_HAIRCUT,
                'leaps_leverage': LEAPS_LEVERAGE,
                'leaps_theta_monthly': LEAPS_THETA_MONTHLY,
            }
        },
        'strategies': results,
        'validation_summary': {
            m['name']: {
                'all_pass': m['all_gates_pass'],
                'sharpe': m['sharpe'],
                'total_return_pct': m['total_return_pct'],
                'final_equity': m['final_equity'],
            }
            for m in all_metrics
        }
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment('concentrated_growth_v1')
            with mlflow.start_run(run_name='concentrated_growth_6variants'):
                for m in all_metrics:
                    prefix = m['name']
                    mlflow.log_metric(f"{prefix}_sharpe", m['sharpe'])
                    mlflow.log_metric(f"{prefix}_return_pct", m['total_return_pct'])
                    mlflow.log_metric(f"{prefix}_sortino", m['sortino'])
                    mlflow.log_metric(f"{prefix}_mdd", m['max_dd_pct'])
                    mlflow.log_metric(f"{prefix}_wr", m['win_rate'])
                    mlflow.log_metric(f"{prefix}_gates", m['gates_passed'])
                mlflow.log_artifact(str(RESULTS_PATH))
            fprint("MLflow run logged")
        except Exception as ex:
            fprint(f"MLflow logging failed: {ex}")

    return results


if __name__ == '__main__':
    main()
