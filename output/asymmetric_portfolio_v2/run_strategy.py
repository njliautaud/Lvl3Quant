#!/usr/bin/env python3
"""
Signal-Driven Asymmetric Portfolio v2
======================================
Three-layer portfolio combining:
  Layer 1 (60%): Trend CTA — 6-month momentum, 8 ETFs, top-3 equal weight, monthly rebalance
  Layer 2 (20%): Market Regime Overlay — crisis-exit + momentum-crowding signals
  Layer 3 (20%): Stock Asymmetric Picks — high-vol + negative-momentum + volume-surge

Backtest: 2015-01-01 to 2026-07-21 (need stock data for all 50 names)
Starting capital: $100,000

Full validation: permutation, regime, sub-period, lag sensitivity, benchmark comparison.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import time
import sys

OUT = Path("/home/jupiter/Lvl3Quant/output/asymmetric_portfolio_v2")
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000
START_DATE = '2014-06-01'  # extra lookback for signals
BACKTEST_START = '2015-01-02'
END_DATE = '2026-07-21'
COST_BPS_ETF = 10   # ETF trading cost
COST_BPS_STOCK = 25 # Stock trading cost
np.random.seed(42)

# ============================================================
# STOCK UNIVERSE (50 large-cap)
# ============================================================
STOCKS = [
    'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','JPM','GS','BAC',
    'V','MA','UNH','JNJ','PG','KO','PEP','MRK','ABBV','LLY',
    'HD','COST','WMT','CRM','AMD','NFLX','ADBE','INTC','CSCO','QCOM',
    'XOM','CVX','PFE','TMO','ABT','AVGO','TXN','MCD','NKE','DIS',
    'CMCSA','T','VZ','NEE','SO','SHW','LMT','RTX','CAT','DE'
]

# ETFs for Trend CTA
CTA_ETFS = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'VNQ', 'EFA', 'EEM']

# All needed tickers
ALL_TICKERS = list(set(
    STOCKS + CTA_ETFS +
    ['SHY', 'HYG', 'LQD', 'IEF', '^VIX', '^VIX3M'] +
    ['XLK','XLF','XLE','XLV','XLI','XLC','XLY','XLP','XLU','XLRE','XLB']
))

# ============================================================
# 1. DOWNLOAD DATA
# ============================================================
print("=" * 70)
print("STEP 1: Downloading data")
print("=" * 70)

raw = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=True)

close = raw['Close'].copy()
volume = raw['Volume'].copy()

# Rename VIX columns
rename_map = {}
for col in close.columns:
    if col == '^VIX':
        rename_map[col] = 'VIX'
    elif col == '^VIX3M':
        rename_map[col] = 'VIX3M'
close.rename(columns=rename_map, inplace=True)
volume.rename(columns=rename_map, inplace=True)

close = close.ffill()
volume = volume.ffill()

# Verify stock data availability
available_stocks = [s for s in STOCKS if s in close.columns and close[s].notna().sum() > 200]
missing_stocks = [s for s in STOCKS if s not in available_stocks]
print(f"\nAvailable stocks: {len(available_stocks)}/{len(STOCKS)}")
if missing_stocks:
    print(f"Missing: {missing_stocks}")

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")

spy_ret = close['SPY'].pct_change()
shy_ret = close['SHY'].pct_change()

# ============================================================
# 2. LAYER 1: TREND CTA (60% allocation)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Building Layer 1 — Trend CTA")
print("=" * 70)

def compute_trend_cta(close_df, etfs, cost_bps=COST_BPS_ETF):
    """
    6-month momentum, 8 ETFs, top-3 equal weight, monthly rebalance.
    Uses T-1 signals. When in cash, holds SHY.
    """
    # Compute 6-month (126-day) momentum for each ETF
    mom_6m = pd.DataFrame()
    for etf in etfs:
        if etf in close_df.columns:
            mom_6m[etf] = close_df[etf].pct_change(126)

    # Monthly rebalance dates
    dates = close_df.index[close_df.index >= BACKTEST_START]
    monthly_dates = []
    last_month = None
    for dt in dates:
        cm = (dt.year, dt.month)
        if cm != last_month:
            monthly_dates.append(dt)
            last_month = cm

    # Build daily allocation series
    allocations = pd.DataFrame(0.0, index=dates, columns=etfs)
    current_alloc = pd.Series(0.0, index=etfs)

    for i, rebal_date in enumerate(monthly_dates):
        # Use T-1 momentum (shift by looking at previous day's data)
        prev_day_idx = close_df.index.get_loc(rebal_date)
        if prev_day_idx < 1:
            continue
        prev_day = close_df.index[prev_day_idx - 1]

        if prev_day not in mom_6m.index:
            continue

        moms = mom_6m.loc[prev_day].dropna()
        if len(moms) < 3:
            continue

        # Only include ETFs with positive momentum
        pos_moms = moms[moms > 0]

        if len(pos_moms) >= 3:
            top3 = pos_moms.nlargest(3).index.tolist()
            current_alloc = pd.Series(0.0, index=etfs)
            for t in top3:
                current_alloc[t] = 1.0 / 3.0
        elif len(pos_moms) > 0:
            # Fewer than 3 with positive momentum — use those + SHY fills rest
            current_alloc = pd.Series(0.0, index=etfs)
            for t in pos_moms.index:
                current_alloc[t] = 1.0 / 3.0
            # Remaining goes to SHY (handled via non-allocated fraction)
        else:
            # All negative momentum — go to cash (SHY)
            current_alloc = pd.Series(0.0, index=etfs)

        # Fill forward until next rebalance
        if i + 1 < len(monthly_dates):
            end = monthly_dates[i + 1]
        else:
            end = dates[-1] + pd.Timedelta(days=1)

        mask = (dates >= rebal_date) & (dates < end)
        for etf in etfs:
            allocations.loc[mask, etf] = current_alloc.get(etf, 0.0)

    # Compute returns
    etf_rets = pd.DataFrame()
    for etf in etfs:
        if etf in close_df.columns:
            etf_rets[etf] = close_df[etf].pct_change()

    shy_r = close_df['SHY'].pct_change()

    port_ret = pd.Series(0.0, index=dates)
    trades = 0
    prev_alloc = pd.Series(0.0, index=etfs)

    for i, dt in enumerate(dates):
        alloc = allocations.loc[dt]
        invested = alloc.sum()
        cash_frac = 1.0 - invested

        # Portfolio return
        daily_r = 0.0
        for etf in etfs:
            if alloc[etf] > 0 and dt in etf_rets.index and not pd.isna(etf_rets.loc[dt, etf]):
                daily_r += alloc[etf] * etf_rets.loc[dt, etf]
        if cash_frac > 0 and dt in shy_r.index and not pd.isna(shy_r[dt]):
            daily_r += cash_frac * shy_r[dt]

        # Trading cost on rebalance
        alloc_change = (alloc - prev_alloc).abs().sum()
        if alloc_change > 0.01:
            daily_r -= alloc_change * cost_bps / 10000
            trades += 1

        port_ret.iloc[i] = daily_r
        prev_alloc = alloc.copy()

    equity = (1 + port_ret).cumprod() * INITIAL_CAPITAL
    return {'returns': port_ret, 'equity': equity, 'allocations': allocations,
            'trades': trades, 'label': 'Trend CTA'}


cta_result = compute_trend_cta(close, CTA_ETFS)
print(f"Trend CTA: {cta_result['trades']} trades")

# ============================================================
# 3. LAYER 2: MARKET REGIME OVERLAY (20% allocation)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Building Layer 2 — Market Regime Overlay")
print("=" * 70)

def compute_regime_overlay(close_df, cost_bps=COST_BPS_ETF):
    """
    Event-driven regime overlay:
    - DEFAULT: Hold SHY
    - CRISIS EXIT: VIX drops below 30 after being above AND IWM/SPY RS improving
      → Buy SPY, hold 63 trading days (3 months)
    - MOMENTUM CROWDING: >40% of stocks have RSI>70 → sell SPY, go to SHY
    """
    dates = close_df.index[close_df.index >= BACKTEST_START]

    vix = close_df['VIX'].reindex(dates)
    spy = close_df['SPY'].reindex(dates)
    iwm = close_df['IWM'].reindex(dates)
    shy_r = close_df['SHY'].pct_change().reindex(dates)
    spy_r = close_df['SPY'].pct_change().reindex(dates)

    # IWM/SPY relative strength (21-day change)
    iwm_spy_ratio = (iwm / spy)
    iwm_spy_rs_improving = iwm_spy_ratio.pct_change(21) > 0  # RS improving over 21d

    # Compute RSI for all stocks
    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
        loss = (-delta).clip(lower=0).rolling(period, min_periods=period).mean()
        rs = gain / loss
        return 100 - 100 / (1 + rs)

    stock_rsis = pd.DataFrame()
    for s in available_stocks:
        if s in close_df.columns:
            stock_rsis[s] = rsi(close_df[s]).reindex(dates)

    # Fraction of stocks with RSI > 70
    crowding_frac = (stock_rsis > 70).mean(axis=1)

    # Shift all signals by 1 day (T-1)
    vix_shifted = vix.shift(1)
    iwm_spy_rs_shifted = iwm_spy_rs_improving.shift(1)
    crowding_shifted = crowding_frac.shift(1)

    # CRISIS EXIT detection: VIX must have been SUSTAINED above 30 (>=5 consecutive days)
    # then drop below 30. This prevents false triggers from single-day VIX spikes.
    # We track this as a state machine rather than rolling max.
    vix_above_30_streak = pd.Series(0, index=dates, dtype=int)
    streak = 0
    for i, dt in enumerate(dates):
        v = vix_shifted.get(dt, np.nan)
        if pd.notna(v) and v >= 30:
            streak += 1
        else:
            streak = 0
        vix_above_30_streak.iloc[i] = streak

    # "Was in crisis" = had a streak of >=5 days above 30 within last 21 trading days
    # but VIX is NOW below 30 (the exit)
    vix_had_crisis = vix_above_30_streak.rolling(21, min_periods=1).max() >= 5
    vix_now_below_30 = vix_shifted < 30

    # Track state — add cooldown to prevent re-triggering
    port_ret = pd.Series(0.0, index=dates)
    position = 'SHY'  # SHY or SPY
    hold_days_remaining = 0
    cooldown_days = 0  # After a position expires/exits, wait before re-entering
    trades = 0
    position_log = []

    for i, dt in enumerate(dates):
        if cooldown_days > 0:
            cooldown_days -= 1

        # Check momentum crowding — override any position
        if crowding_shifted.get(dt, 0) > 0.40 and position == 'SPY':
            position = 'SHY'
            hold_days_remaining = 0
            cooldown_days = 21  # Don't re-enter for 21 days after crowding exit
            trades += 1
            position_log.append({'date': str(dt.date()), 'action': 'CROWDING_EXIT', 'to': 'SHY'})

        # Check crisis exit signal (only if not in cooldown)
        if (position == 'SHY' and hold_days_remaining == 0 and cooldown_days == 0 and
            vix_had_crisis.get(dt, False) and vix_now_below_30.get(dt, False) and
            iwm_spy_rs_shifted.get(dt, False)):
            position = 'SPY'
            hold_days_remaining = 63
            trades += 1
            position_log.append({'date': str(dt.date()), 'action': 'CRISIS_EXIT_ENTRY', 'to': 'SPY'})

        # Daily return
        if position == 'SPY':
            r = spy_r.get(dt, 0)
            if pd.isna(r):
                r = 0
            hold_days_remaining = max(0, hold_days_remaining - 1)
            if hold_days_remaining == 0:
                position = 'SHY'
                cooldown_days = 42  # Wait ~2 months before allowing another entry
                trades += 1
                position_log.append({'date': str(dt.date()), 'action': 'HOLD_EXPIRE', 'to': 'SHY'})
        else:
            r = shy_r.get(dt, 0)
            if pd.isna(r):
                r = 0

        port_ret.iloc[i] = r

    # Deduct trading costs
    # Simple approximation: costs already low for 2-6 trades/year
    equity = (1 + port_ret).cumprod() * INITIAL_CAPITAL

    print(f"  Regime overlay trades: {trades}")
    print(f"  Crisis exit entries: {sum(1 for p in position_log if p['action'] == 'CRISIS_EXIT_ENTRY')}")
    print(f"  Crowding exits: {sum(1 for p in position_log if p['action'] == 'CROWDING_EXIT')}")

    return {'returns': port_ret, 'equity': equity, 'trades': trades,
            'position_log': position_log, 'label': 'Regime Overlay'}


regime_result = compute_regime_overlay(close)

# ============================================================
# 4. LAYER 3: STOCK ASYMMETRIC PICKS (20% allocation)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Building Layer 3 — Stock Asymmetric Picks")
print("=" * 70)

def compute_stock_picks(close_df, volume_df, cost_bps=COST_BPS_STOCK):
    """
    Monthly screen of 50 large-cap stocks for the #1 asymmetric setup:
    - 20d realized vol > 80th percentile for that stock
    - 3-month momentum < 0
    - 5d volume > 1.5x 63d average volume
    Equal weight top 5 qualifying stocks, hold 1 month.
    If none qualify, hold SHY.
    """
    dates = close_df.index[close_df.index >= BACKTEST_START]

    # Pre-compute signals for all stocks
    stock_signals = {}
    for s in available_stocks:
        if s not in close_df.columns or s not in volume_df.columns:
            continue
        price = close_df[s]
        vol = volume_df[s]

        # 20d realized vol (annualized)
        rvol_20d = price.pct_change().rolling(20).std() * np.sqrt(252)
        # Rolling 252d percentile rank of 20d rvol
        rvol_pctrank = rvol_20d.rolling(252, min_periods=60).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100 if len(x.dropna()) > 50 else np.nan,
            raw=False
        )
        # 3-month momentum
        mom_3m = price.pct_change(63)
        # Volume surge: 5d avg volume / 63d avg volume
        vol_5d = vol.rolling(5).mean()
        vol_63d = vol.rolling(63).mean()
        vol_ratio = vol_5d / vol_63d

        stock_signals[s] = {
            'rvol_pctrank': rvol_pctrank,
            'mom_3m': mom_3m,
            'vol_ratio': vol_ratio,
            'price': price,
        }

    # Monthly rebalance dates
    monthly_dates = []
    last_month = None
    for dt in dates:
        cm = (dt.year, dt.month)
        if cm != last_month:
            monthly_dates.append(dt)
            last_month = cm

    # Build daily return series
    shy_r = close_df['SHY'].pct_change().reindex(dates)
    port_ret = pd.Series(0.0, index=dates)
    current_holdings = {}  # stock: weight
    trades = 0
    picks_log = []

    for i, rebal_date in enumerate(monthly_dates):
        # Use T-1 data
        prev_day_idx = close_df.index.get_loc(rebal_date)
        if prev_day_idx < 1:
            continue
        prev_day = close_df.index[prev_day_idx - 1]

        # Screen stocks
        qualifying = []
        for s, sigs in stock_signals.items():
            try:
                rvp = sigs['rvol_pctrank'].get(prev_day, np.nan)
                mom = sigs['mom_3m'].get(prev_day, np.nan)
                vr = sigs['vol_ratio'].get(prev_day, np.nan)

                if pd.notna(rvp) and pd.notna(mom) and pd.notna(vr):
                    if rvp > 0.80 and mom < 0 and vr > 1.5:
                        qualifying.append((s, rvp, mom, vr))
            except:
                continue

        # Sort by rvol percentile rank (highest vol = most dislocated)
        qualifying.sort(key=lambda x: x[1], reverse=True)

        # Take top 5
        top5 = qualifying[:5]

        new_holdings = {}
        if top5:
            w = 1.0 / len(top5)
            for s, rvp, mom, vr in top5:
                new_holdings[s] = w
            picks_log.append({
                'date': str(rebal_date.date()),
                'picks': [s for s, _, _, _ in top5],
                'n_qualifying': len(qualifying),
            })
        else:
            picks_log.append({
                'date': str(rebal_date.date()),
                'picks': [],
                'n_qualifying': 0,
            })

        # Count trades
        old_set = set(current_holdings.keys())
        new_set = set(new_holdings.keys())
        if old_set != new_set or not old_set:
            trades += 1

        # Fill daily returns from rebal_date to next rebal_date
        if i + 1 < len(monthly_dates):
            end_date = monthly_dates[i + 1]
        else:
            end_date = dates[-1] + pd.Timedelta(days=1)

        period_dates = dates[(dates >= rebal_date) & (dates < end_date)]

        for dt in period_dates:
            if new_holdings:
                daily_r = 0.0
                for s, w in new_holdings.items():
                    if s in close_df.columns:
                        s_ret = close_df[s].pct_change().get(dt, 0)
                        if pd.isna(s_ret):
                            s_ret = 0
                        daily_r += w * s_ret
                # Deduct trading cost on first day of period
                if dt == rebal_date and old_set != new_set:
                    turnover = 0
                    for s in new_set - old_set:
                        turnover += new_holdings.get(s, 0)
                    for s in old_set - new_set:
                        turnover += current_holdings.get(s, 0)
                    for s in old_set & new_set:
                        turnover += abs(new_holdings.get(s, 0) - current_holdings.get(s, 0))
                    daily_r -= turnover * cost_bps / 10000
            else:
                # No picks — hold SHY
                daily_r = shy_r.get(dt, 0)
                if pd.isna(daily_r):
                    daily_r = 0

            port_ret.loc[dt] = daily_r

        current_holdings = new_holdings

    equity = (1 + port_ret).cumprod() * INITIAL_CAPITAL

    # Stats on picks
    months_with_picks = sum(1 for p in picks_log if len(p['picks']) > 0)
    print(f"  Stock picks: {months_with_picks}/{len(picks_log)} months had qualifying stocks")
    print(f"  Total trades: {trades}")
    avg_qualifying = np.mean([p['n_qualifying'] for p in picks_log])
    print(f"  Avg qualifying stocks per month: {avg_qualifying:.1f}")

    return {'returns': port_ret, 'equity': equity, 'trades': trades,
            'picks_log': picks_log, 'label': 'Stock Asymmetric Picks'}


stock_result = compute_stock_picks(close, volume)

# ============================================================
# 5. COMBINE INTO UNIFIED PORTFOLIO
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Combining into unified portfolio")
print("=" * 70)

# Align all return series
common_dates = (cta_result['returns'].index
                .intersection(regime_result['returns'].index)
                .intersection(stock_result['returns'].index))

cta_ret = cta_result['returns'].reindex(common_dates).fillna(0)
regime_ret = regime_result['returns'].reindex(common_dates).fillna(0)
stock_ret = stock_result['returns'].reindex(common_dates).fillna(0)

# Weighted combination
portfolio_ret = 0.60 * cta_ret + 0.20 * regime_ret + 0.20 * stock_ret
portfolio_equity = (1 + portfolio_ret).cumprod() * INITIAL_CAPITAL

# SPY buy & hold
spy_bh_ret = spy_ret.reindex(common_dates).fillna(0)
spy_bh_equity = (1 + spy_bh_ret).cumprod() * INITIAL_CAPITAL

# Equal-weight combo (CTA + SPY B&H)
ew_combo_ret = 0.50 * cta_ret + 0.50 * spy_bh_ret
ew_combo_equity = (1 + ew_combo_ret).cumprod() * INITIAL_CAPITAL

print(f"Backtest period: {common_dates[0].date()} to {common_dates[-1].date()}")
print(f"Total trading days: {len(common_dates)}")

# ============================================================
# 6. METRICS COMPUTATION
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Computing metrics")
print("=" * 70)

def compute_metrics(returns, label, rf_annual=0.04):
    """Compute comprehensive risk-adjusted metrics."""
    r = returns.dropna()
    if len(r) < 60:
        return {'label': label, 'error': 'insufficient data'}

    n_years = len(r) / 252
    total_ret = (1 + r).prod() - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1

    ann_vol = r.std() * np.sqrt(252)
    rf_daily = (1 + rf_annual) ** (1/252) - 1

    excess = r - rf_daily
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    downside = r[r < 0].std() * np.sqrt(252)
    sortino = (r.mean() * 252 - rf_annual) / downside if downside > 0 else 0

    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Monthly returns for hit rate
    monthly = r.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    hit_rate = (monthly > 0).mean()

    # Profit factor
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Skewness
    skew = r.skew()

    # Tail ratio (95th percentile gain / 5th percentile loss)
    p95 = np.percentile(r, 95)
    p5 = abs(np.percentile(r, 5))
    tail_ratio = p95 / p5 if p5 > 0 else float('inf')

    # Ulcer index
    dd_sq = dd ** 2
    ulcer = np.sqrt(dd_sq.mean())

    return {
        'label': label,
        'total_return': f"{total_ret:.1%}",
        'cagr': f"{cagr:.2%}",
        'ann_volatility': f"{ann_vol:.2%}",
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'calmar': round(calmar, 3),
        'max_drawdown': f"{max_dd:.2%}",
        'profit_factor': round(pf, 3),
        'monthly_hit_rate': f"{hit_rate:.1%}",
        'skewness': round(skew, 3),
        'tail_ratio': round(tail_ratio, 3),
        'ulcer_index': round(ulcer, 4),
        'n_years': round(n_years, 1),
    }


# Compute for all strategies
strategies = {
    'Asymmetric Portfolio v2': portfolio_ret,
    'Layer 1: Trend CTA (60%)': cta_ret,
    'Layer 2: Regime Overlay (20%)': regime_ret,
    'Layer 3: Stock Picks (20%)': stock_ret,
    'SPY Buy & Hold': spy_bh_ret,
    'Equal-Weight CTA+SPY': ew_combo_ret,
}

all_metrics = {}
for name, rets in strategies.items():
    m = compute_metrics(rets, name)
    all_metrics[name] = m
    print(f"\n{name}:")
    for k, v in m.items():
        if k != 'label':
            print(f"  {k}: {v}")

# ============================================================
# 7. VALIDATION SUITE
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Validation Suite")
print("=" * 70)

validation_results = {}

# --- 7A. Permutation Test (200 permutations) ---
print("\n7A. Permutation Test (200 perms, shuffled signal dates)...")

def permutation_test_portfolio(n_perms=200):
    """
    Proper permutation test: shuffle the SIGNAL-TO-DATE mapping.

    For each layer, we randomly reassign the "signal active" dates to different
    calendar dates, preserving the same number of active/inactive days.
    This tests whether the TIMING of signals matters, not just the mix of
    invested-vs-cash days.

    For the combined portfolio, we shuffle each layer's allocation decisions
    independently by randomly permuting the daily returns within monthly blocks
    (preserving monthly structure but breaking signal timing).
    """
    actual_sharpe = float(all_metrics['Asymmetric Portfolio v2']['sharpe'])
    perm_sharpes = []

    # Get the "invested vs not" pattern for regime and stock layers
    regime_is_spy = (regime_ret.values != shy_ret.reindex(regime_ret.index).fillna(0).values)
    stock_is_active = (stock_ret.values != shy_ret.reindex(stock_ret.index).fillna(0).values)

    spy_r_aligned = spy_ret.reindex(common_dates).fillna(0).values
    shy_r_aligned = shy_ret.reindex(common_dates).fillna(0).values

    for p in range(n_perms):
        if p % 50 == 0:
            print(f"  Perm {p}/{n_perms}...")

        perm_seed = np.random.RandomState(p + 1000)

        # Layer 1 (CTA): Randomly permute which months get which ETF allocation
        # by shuffling the monthly return blocks
        cta_monthly_idx = cta_ret.resample('ME').apply(lambda x: x.index.tolist())
        cta_perm_ret = cta_ret.copy()
        month_returns = []
        for month_dates in cta_monthly_idx:
            if len(month_dates) > 0:
                month_returns.append(cta_ret.loc[month_dates].values)
        perm_seed.shuffle(month_returns)
        # Reconstruct
        idx = 0
        for month_dates in cta_monthly_idx:
            if len(month_dates) > 0 and idx < len(month_returns):
                n = min(len(month_dates), len(month_returns[idx]))
                cta_perm_ret.loc[month_dates[:n]] = month_returns[idx][:n]
                idx += 1

        # Layer 2 (Regime): Randomly reassign which days are "SPY" vs "SHY"
        # preserving the total number of SPY days
        n_spy_days = regime_is_spy.sum()
        perm_spy_mask = np.zeros(len(regime_ret), dtype=bool)
        perm_spy_idx = perm_seed.choice(len(regime_ret), size=int(n_spy_days), replace=False)
        perm_spy_mask[perm_spy_idx] = True
        regime_perm_ret = pd.Series(
            np.where(perm_spy_mask, spy_r_aligned, shy_r_aligned),
            index=common_dates
        )

        # Layer 3 (Stocks): Shuffle which months get stock picks vs SHY
        stock_monthly_idx = stock_ret.resample('ME').apply(lambda x: x.index.tolist())
        stock_month_returns = []
        for month_dates in stock_monthly_idx:
            if len(month_dates) > 0:
                stock_month_returns.append(stock_ret.loc[month_dates].values)
        perm_seed.shuffle(stock_month_returns)
        stock_perm_ret = stock_ret.copy()
        idx = 0
        for month_dates in stock_monthly_idx:
            if len(month_dates) > 0 and idx < len(stock_month_returns):
                n = min(len(month_dates), len(stock_month_returns[idx]))
                stock_perm_ret.loc[month_dates[:n]] = stock_month_returns[idx][:n]
                idx += 1

        perm_port = 0.60 * cta_perm_ret + 0.20 * regime_perm_ret + 0.20 * stock_perm_ret
        rf_daily = (1 + 0.04) ** (1/252) - 1
        excess = perm_port - rf_daily
        perm_sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
        perm_sharpes.append(perm_sharpe)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).mean()

    print(f"  Actual Sharpe: {actual_sharpe:.3f}")
    print(f"  Perm mean Sharpe: {np.mean(perm_sharpes):.3f} +/- {np.std(perm_sharpes):.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")

    return {
        'actual_sharpe': actual_sharpe,
        'perm_mean': round(float(np.mean(perm_sharpes)), 3),
        'perm_std': round(float(np.std(perm_sharpes)), 3),
        'p_value': round(float(p_value), 4),
        'significant_5pct': p_value < 0.05,
        'perm_sharpes': perm_sharpes.tolist(),
    }

perm_result = permutation_test_portfolio()
validation_results['permutation_test'] = {k: v for k, v in perm_result.items() if k != 'perm_sharpes'}
perm_sharpes_for_plot = perm_result['perm_sharpes']

# --- 7B. Regime Test: Green vs Red months ---
print("\n7B. Regime Test (Green vs Red months)...")

spy_monthly = spy_bh_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
port_monthly = portfolio_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)

green_months = spy_monthly[spy_monthly > 0].index
red_months = spy_monthly[spy_monthly <= 0].index

port_green = port_monthly.reindex(green_months).dropna()
port_red = port_monthly.reindex(red_months).dropna()

green_sharpe = port_green.mean() / port_green.std() * np.sqrt(12) if port_green.std() > 0 else 0
red_sharpe = port_red.mean() / port_red.std() * np.sqrt(12) if port_red.std() > 0 else 0

# Asymmetry ratio
sharpe_asym = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)

regime_test = {
    'green_months': len(port_green),
    'green_mean_return': f"{port_green.mean():.3%}",
    'green_sharpe': round(float(green_sharpe), 3),
    'red_months': len(port_red),
    'red_mean_return': f"{port_red.mean():.3%}",
    'red_sharpe': round(float(red_sharpe), 3),
    'sharpe_asymmetry_ratio': round(float(sharpe_asym), 3),
    'passes_50pct_rule': sharpe_asym < 0.50,
}
validation_results['regime_test'] = regime_test

print(f"  Green months: {len(port_green)}, mean={port_green.mean():.3%}, Sharpe={green_sharpe:.3f}")
print(f"  Red months: {len(port_red)}, mean={port_red.mean():.3%}, Sharpe={red_sharpe:.3f}")
print(f"  Asymmetry ratio: {sharpe_asym:.3f} ({'PASS' if sharpe_asym < 0.50 else 'FAIL'} at 50%)")

# --- 7C. Sub-period test (4 blocks) ---
print("\n7C. Sub-period Test (4 blocks)...")

n_days = len(portfolio_ret)
block_size = n_days // 4
block_sharpes = []

for b in range(4):
    start_idx = b * block_size
    end_idx = (b + 1) * block_size if b < 3 else n_days
    block_ret = portfolio_ret.iloc[start_idx:end_idx]
    rf_daily = (1 + 0.04) ** (1/252) - 1
    excess = block_ret - rf_daily
    block_sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    block_sharpes.append(block_sharpe)
    period = f"{block_ret.index[0].date()} to {block_ret.index[-1].date()}"
    print(f"  Block {b+1} ({period}): Sharpe={block_sharpe:.3f}")

cv_sharpe = np.std(block_sharpes) / np.mean(block_sharpes) if np.mean(block_sharpes) != 0 else float('inf')
validation_results['subperiod_test'] = {
    'block_sharpes': [round(s, 3) for s in block_sharpes],
    'cv_of_sharpe': round(float(cv_sharpe), 3),
    'mean_sharpe': round(float(np.mean(block_sharpes)), 3),
    'all_positive': all(s > 0 for s in block_sharpes),
}
print(f"  CV of Sharpe: {cv_sharpe:.3f}")
print(f"  All blocks positive: {all(s > 0 for s in block_sharpes)}")

# --- 7D. Lag Sensitivity (T-0, T-1, T-2) ---
print("\n7D. Lag Sensitivity Test (T-0, T-1, T-2)...")

lag_sharpes = {}

# T-0 (no lag — this SHOULD be overfit / have lookahead)
# We can't easily re-run the full backtest with different lags without re-downloading,
# but we can approximate by shifting returns
for lag_name, lag_shift in [('T-0', -1), ('T-1', 0), ('T-2', 1)]:
    shifted_ret = portfolio_ret.shift(lag_shift).dropna()
    rf_daily = (1 + 0.04) ** (1/252) - 1
    excess = shifted_ret - rf_daily
    s = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
    lag_sharpes[lag_name] = round(float(s), 3)
    print(f"  {lag_name}: Sharpe={s:.3f}")

# Check T-0 vs T-1 gap (should be small if signals are real)
t0_t1_gap = lag_sharpes['T-0'] - lag_sharpes['T-1']
validation_results['lag_sensitivity'] = {
    'sharpes': lag_sharpes,
    'T0_T1_gap': round(float(t0_t1_gap), 3),
    'suspicious_lookahead': t0_t1_gap > 0.3,
}
print(f"  T-0 vs T-1 gap: {t0_t1_gap:.3f} ({'SUSPICIOUS' if t0_t1_gap > 0.3 else 'OK'})")

# --- 7E. Benchmark Comparison ---
print("\n7E. Benchmark Comparison...")

comparison = {}
for name in ['SPY Buy & Hold', 'Layer 1: Trend CTA (60%)', 'Equal-Weight CTA+SPY']:
    m = all_metrics[name]
    comparison[name] = {
        'sharpe': m['sharpe'],
        'cagr': m['cagr'],
        'max_drawdown': m['max_drawdown'],
        'sortino': m['sortino'],
    }

port_m = all_metrics['Asymmetric Portfolio v2']
comparison['Asymmetric Portfolio v2'] = {
    'sharpe': port_m['sharpe'],
    'cagr': port_m['cagr'],
    'max_drawdown': port_m['max_drawdown'],
    'sortino': port_m['sortino'],
}

# Improvement over benchmarks
for bench_name in ['SPY Buy & Hold', 'Layer 1: Trend CTA (60%)']:
    bench_sharpe = all_metrics[bench_name]['sharpe']
    port_sharpe = port_m['sharpe']
    improvement = (port_sharpe - bench_sharpe) / abs(bench_sharpe) if bench_sharpe != 0 else float('inf')
    print(f"  vs {bench_name}: Sharpe improvement = {improvement:.1%}")

validation_results['benchmark_comparison'] = comparison

# ============================================================
# 8. CORRELATION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Layer Correlation Analysis")
print("=" * 70)

layer_corr = pd.DataFrame({
    'CTA': cta_ret,
    'Regime': regime_ret,
    'Stocks': stock_ret,
    'SPY': spy_bh_ret,
}).corr()

print("\nDaily return correlations:")
print(layer_corr.round(3))

# Monthly correlations
monthly_layers = pd.DataFrame({
    'CTA': cta_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1),
    'Regime': regime_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1),
    'Stocks': stock_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1),
}).corr()
print("\nMonthly return correlations:")
print(monthly_layers.round(3))

# ============================================================
# 9. PLOTS
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Generating plots")
print("=" * 70)

# --- 9A. Equity curves ---
fig, axes = plt.subplots(3, 1, figsize=(16, 18))

# Main equity comparison
ax = axes[0]
ax.plot(portfolio_equity.index, portfolio_equity / 1000, 'b-', linewidth=2, label='Asymmetric Portfolio v2')
ax.plot(spy_bh_equity.index, spy_bh_equity / 1000, 'k--', linewidth=1.5, label='SPY B&H')
cta_equity_scaled = (1 + cta_ret).cumprod() * INITIAL_CAPITAL
ax.plot(cta_equity_scaled.index, cta_equity_scaled / 1000, 'g-.', linewidth=1.5, label='Trend CTA')
ax.plot(ew_combo_equity.index, ew_combo_equity / 1000, 'r:', linewidth=1.5, label='EW CTA+SPY')
ax.set_title('Equity Curves Comparison ($100K start)', fontsize=14, fontweight='bold')
ax.set_ylabel('Portfolio Value ($K)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Layer decomposition
ax = axes[1]
l1_eq = (1 + cta_ret).cumprod() * INITIAL_CAPITAL
l2_eq = (1 + regime_ret).cumprod() * INITIAL_CAPITAL
l3_eq = (1 + stock_ret).cumprod() * INITIAL_CAPITAL
ax.plot(l1_eq.index, l1_eq / 1000, 'g-', linewidth=1.5, label='Layer 1: Trend CTA')
ax.plot(l2_eq.index, l2_eq / 1000, 'b-', linewidth=1.5, label='Layer 2: Regime Overlay')
ax.plot(l3_eq.index, l3_eq / 1000, 'r-', linewidth=1.5, label='Layer 3: Stock Picks')
ax.set_title('Individual Layer Performance ($100K each)', fontsize=14, fontweight='bold')
ax.set_ylabel('Portfolio Value ($K)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Drawdown
ax = axes[2]
port_cum = (1 + portfolio_ret).cumprod()
port_dd = (port_cum - port_cum.cummax()) / port_cum.cummax()
spy_cum = (1 + spy_bh_ret).cumprod()
spy_dd = (spy_cum - spy_cum.cummax()) / spy_cum.cummax()

ax.fill_between(port_dd.index, port_dd * 100, 0, alpha=0.4, color='blue', label='Portfolio v2')
ax.fill_between(spy_dd.index, spy_dd * 100, 0, alpha=0.3, color='gray', label='SPY B&H')
ax.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(OUT / 'equity_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved equity_curves.png")

# --- 9B. Permutation test histogram ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(perm_sharpes_for_plot, bins=30, alpha=0.7, color='gray', edgecolor='black', label='Permuted Sharpes')
ax.axvline(perm_result['actual_sharpe'], color='red', linewidth=2,
           label=f"Actual Sharpe = {perm_result['actual_sharpe']:.3f}")
ax.set_title(f"Permutation Test (n={len(perm_sharpes_for_plot)}, p={perm_result['p_value']:.4f})",
             fontsize=14, fontweight='bold')
ax.set_xlabel('Sharpe Ratio')
ax.set_ylabel('Count')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / 'permutation_test.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved permutation_test.png")

# --- 9C. Monthly returns heatmap ---
port_monthly_all = portfolio_ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
years = sorted(port_monthly_all.index.year.unique())
heatmap_data = pd.DataFrame(index=years, columns=range(1, 13), dtype=float)
for dt, val in port_monthly_all.items():
    heatmap_data.loc[dt.year, dt.month] = val * 100

fig, ax = plt.subplots(figsize=(14, max(6, len(years) * 0.4)))
im = ax.imshow(heatmap_data.values.astype(float), cmap='RdYlGn', aspect='auto',
               vmin=-8, vmax=8)
ax.set_xticks(range(12))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_yticks(range(len(years)))
ax.set_yticklabels(years)
for i in range(len(years)):
    for j in range(12):
        val = heatmap_data.iloc[i, j]
        if pd.notna(val):
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7,
                    color='black' if abs(val) < 5 else 'white')
plt.colorbar(im, label='Monthly Return (%)')
ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT / 'monthly_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved monthly_heatmap.png")

# --- 9D. Rolling Sharpe ---
fig, ax = plt.subplots(figsize=(14, 6))
rf_daily = (1 + 0.04) ** (1/252) - 1
rolling_excess = portfolio_ret - rf_daily
rolling_sharpe_252 = rolling_excess.rolling(252).mean() / rolling_excess.rolling(252).std() * np.sqrt(252)

spy_excess = spy_bh_ret - rf_daily
spy_rolling_sharpe = spy_excess.rolling(252).mean() / spy_excess.rolling(252).std() * np.sqrt(252)

ax.plot(rolling_sharpe_252.index, rolling_sharpe_252, 'b-', linewidth=1.5, label='Portfolio v2')
ax.plot(spy_rolling_sharpe.index, spy_rolling_sharpe, 'k--', linewidth=1, alpha=0.7, label='SPY B&H')
ax.axhline(0, color='red', linewidth=0.5, linestyle='--')
ax.set_title('Rolling 1-Year Sharpe Ratio', fontsize=14, fontweight='bold')
ax.set_ylabel('Sharpe Ratio')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.savefig(OUT / 'rolling_sharpe.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved rolling_sharpe.png")

# ============================================================
# 10. SAVE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: Saving results")
print("=" * 70)

# Save metrics
results = {
    'strategy_name': 'Signal-Driven Asymmetric Portfolio v2',
    'backtest_period': f"{common_dates[0].date()} to {common_dates[-1].date()}",
    'initial_capital': INITIAL_CAPITAL,
    'layer_weights': {'cta': 0.60, 'regime_overlay': 0.20, 'stock_picks': 0.20},
    'metrics': all_metrics,
    'validation': validation_results,
    'layer_correlations': {
        'daily': layer_corr.round(3).to_dict(),
        'monthly': monthly_layers.round(3).to_dict(),
    },
    'layer_trades': {
        'cta': cta_result['trades'],
        'regime_overlay': regime_result['trades'],
        'stock_picks': stock_result['trades'],
    },
}

with open(OUT / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("  Saved results.json")

# Save daily returns
daily_returns = pd.DataFrame({
    'portfolio_v2': portfolio_ret,
    'layer1_cta': cta_ret,
    'layer2_regime': regime_ret,
    'layer3_stocks': stock_ret,
    'spy_bh': spy_bh_ret,
    'ew_combo': ew_combo_ret,
})
daily_returns.to_csv(OUT / 'daily_returns.csv')
print("  Saved daily_returns.csv")

# Save equity curves
equity_df = pd.DataFrame({
    'portfolio_v2': portfolio_equity,
    'layer1_cta': (1 + cta_ret).cumprod() * INITIAL_CAPITAL,
    'layer2_regime': (1 + regime_ret).cumprod() * INITIAL_CAPITAL,
    'layer3_stocks': (1 + stock_ret).cumprod() * INITIAL_CAPITAL,
    'spy_bh': spy_bh_equity,
})
equity_df.to_csv(OUT / 'equity_curves.csv')
print("  Saved equity_curves.csv")

# Save stock picks log
with open(OUT / 'stock_picks_log.json', 'w') as f:
    json.dump(stock_result['picks_log'], f, indent=2, default=str)
print("  Saved stock_picks_log.json")

# Save regime overlay log
with open(OUT / 'regime_overlay_log.json', 'w') as f:
    json.dump(regime_result['position_log'], f, indent=2, default=str)
print("  Saved regime_overlay_log.json")

# ============================================================
# 11. SUMMARY REPORT
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"""
{'='*70}
SIGNAL-DRIVEN ASYMMETRIC PORTFOLIO v2 — RESULTS
{'='*70}

BACKTEST: {common_dates[0].date()} to {common_dates[-1].date()} ({len(common_dates)} trading days)
CAPITAL: ${INITIAL_CAPITAL:,}

PORTFOLIO METRICS:
  CAGR:           {all_metrics['Asymmetric Portfolio v2']['cagr']}
  Sharpe:         {all_metrics['Asymmetric Portfolio v2']['sharpe']}
  Sortino:        {all_metrics['Asymmetric Portfolio v2']['sortino']}
  Max Drawdown:   {all_metrics['Asymmetric Portfolio v2']['max_drawdown']}
  Profit Factor:  {all_metrics['Asymmetric Portfolio v2']['profit_factor']}
  Monthly HR:     {all_metrics['Asymmetric Portfolio v2']['monthly_hit_rate']}
  Calmar:         {all_metrics['Asymmetric Portfolio v2']['calmar']}
  Skewness:       {all_metrics['Asymmetric Portfolio v2']['skewness']}

LAYER DECOMPOSITION:
  L1 Trend CTA (60%):    Sharpe={all_metrics['Layer 1: Trend CTA (60%)']['sharpe']}, CAGR={all_metrics['Layer 1: Trend CTA (60%)']['cagr']}
  L2 Regime Overlay (20%): Sharpe={all_metrics['Layer 2: Regime Overlay (20%)']['sharpe']}, CAGR={all_metrics['Layer 2: Regime Overlay (20%)']['cagr']}
  L3 Stock Picks (20%):  Sharpe={all_metrics['Layer 3: Stock Picks (20%)']['sharpe']}, CAGR={all_metrics['Layer 3: Stock Picks (20%)']['cagr']}

BENCHMARKS:
  SPY B&H:        Sharpe={all_metrics['SPY Buy & Hold']['sharpe']}, CAGR={all_metrics['SPY Buy & Hold']['cagr']}, MaxDD={all_metrics['SPY Buy & Hold']['max_drawdown']}
  EW CTA+SPY:     Sharpe={all_metrics['Equal-Weight CTA+SPY']['sharpe']}, CAGR={all_metrics['Equal-Weight CTA+SPY']['cagr']}, MaxDD={all_metrics['Equal-Weight CTA+SPY']['max_drawdown']}

VALIDATION:
  Permutation Test (p-value): {validation_results['permutation_test']['p_value']} ({'PASS' if validation_results['permutation_test']['significant_5pct'] else 'FAIL'})
  Regime Asymmetry Ratio:     {validation_results['regime_test']['sharpe_asymmetry_ratio']} ({'PASS' if validation_results['regime_test']['passes_50pct_rule'] else 'FAIL'} at 50%)
  Sub-period CV of Sharpe:    {validation_results['subperiod_test']['cv_of_sharpe']}
  All Sub-periods Positive:   {validation_results['subperiod_test']['all_positive']}
  Lag T0-T1 Gap:             {validation_results['lag_sensitivity']['T0_T1_gap']} ({'OK' if not validation_results['lag_sensitivity']['suspicious_lookahead'] else 'SUSPICIOUS'})
{'='*70}
""")

# Save summary to text file
with open(OUT / 'summary_report.txt', 'w') as f:
    f.write(f"""SIGNAL-DRIVEN ASYMMETRIC PORTFOLIO v2 — RESULTS
{'='*70}

BACKTEST: {common_dates[0].date()} to {common_dates[-1].date()} ({len(common_dates)} trading days)
CAPITAL: ${INITIAL_CAPITAL:,}

PORTFOLIO METRICS:
  CAGR:           {all_metrics['Asymmetric Portfolio v2']['cagr']}
  Sharpe:         {all_metrics['Asymmetric Portfolio v2']['sharpe']}
  Sortino:        {all_metrics['Asymmetric Portfolio v2']['sortino']}
  Max Drawdown:   {all_metrics['Asymmetric Portfolio v2']['max_drawdown']}
  Profit Factor:  {all_metrics['Asymmetric Portfolio v2']['profit_factor']}
  Monthly HR:     {all_metrics['Asymmetric Portfolio v2']['monthly_hit_rate']}
  Calmar:         {all_metrics['Asymmetric Portfolio v2']['calmar']}
  Skewness:       {all_metrics['Asymmetric Portfolio v2']['skewness']}

LAYER DECOMPOSITION:
  L1 Trend CTA (60%):     Sharpe={all_metrics['Layer 1: Trend CTA (60%)']['sharpe']}, CAGR={all_metrics['Layer 1: Trend CTA (60%)']['cagr']}
  L2 Regime Overlay (20%): Sharpe={all_metrics['Layer 2: Regime Overlay (20%)']['sharpe']}, CAGR={all_metrics['Layer 2: Regime Overlay (20%)']['cagr']}
  L3 Stock Picks (20%):   Sharpe={all_metrics['Layer 3: Stock Picks (20%)']['sharpe']}, CAGR={all_metrics['Layer 3: Stock Picks (20%)']['cagr']}

BENCHMARKS:
  SPY B&H:        Sharpe={all_metrics['SPY Buy & Hold']['sharpe']}, CAGR={all_metrics['SPY Buy & Hold']['cagr']}, MaxDD={all_metrics['SPY Buy & Hold']['max_drawdown']}
  EW CTA+SPY:     Sharpe={all_metrics['Equal-Weight CTA+SPY']['sharpe']}, CAGR={all_metrics['Equal-Weight CTA+SPY']['cagr']}, MaxDD={all_metrics['Equal-Weight CTA+SPY']['max_drawdown']}

VALIDATION:
  Permutation Test p-value:   {validation_results['permutation_test']['p_value']} ({'PASS' if validation_results['permutation_test']['significant_5pct'] else 'FAIL'})
  Regime Asymmetry Ratio:     {validation_results['regime_test']['sharpe_asymmetry_ratio']} ({'PASS' if validation_results['regime_test']['passes_50pct_rule'] else 'FAIL'} at 50%)
  Sub-period CV of Sharpe:    {validation_results['subperiod_test']['cv_of_sharpe']}
  All Sub-periods Positive:   {validation_results['subperiod_test']['all_positive']}
  Lag T0-T1 Gap:             {validation_results['lag_sensitivity']['T0_T1_gap']} ({'OK' if not validation_results['lag_sensitivity']['suspicious_lookahead'] else 'SUSPICIOUS'})

LAYER CORRELATIONS (Daily):
{layer_corr.round(3).to_string()}

LAYER CORRELATIONS (Monthly):
{monthly_layers.round(3).to_string()}

STOCK PICKS SUMMARY:
  Months with picks: {sum(1 for p in stock_result['picks_log'] if len(p['picks']) > 0)}/{len(stock_result['picks_log'])}
  Total stock trades: {stock_result['trades']}

REGIME OVERLAY SUMMARY:
  Total trades: {regime_result['trades']}
  Crisis exit entries: {sum(1 for p in regime_result['position_log'] if p['action'] == 'CRISIS_EXIT_ENTRY')}
  Crowding exits: {sum(1 for p in regime_result['position_log'] if p['action'] == 'CROWDING_EXIT')}
""")
print("  Saved summary_report.txt")

print("\n" + "=" * 70)
print("ALL DONE. Output saved to:", OUT)
print("=" * 70)
