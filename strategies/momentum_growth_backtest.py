#!/usr/bin/env python3
"""
Aggressive Momentum / Growth Rotation Backtest
================================================
ALWAYS-INVESTED strategy targeting high absolute CAGR (beat SPY ~19% ann.).

Strategies:
  A) Sector ETF momentum rotation (6m lookback, top 3 sectors, monthly rebal)
  B) Dual momentum (absolute + relative) with trend filter
  C) Accelerating momentum (rate-of-change of momentum — buy acceleration)
  D) Sector momentum + trailing stop (ride trends, cut losers)
  E) Cross-sectional RS with vol-weighting (strong stocks, size by inverse vol)
  F) Multi-timeframe momentum (1m + 3m + 6m blend)

All strategies are ALWAYS INVESTED — no cash parking.
Walk-forward: 12-month sliding train, 1-month OOS.
Backtest: Jan 2019 — Aug 2026.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import json
import os
import time

# ── Config ──────────────────────────────────────────────────────────────────
SECTOR_ETFS = [
    'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLU',
    'SMH', 'SOXX', 'QQQ', 'IWM', 'XBI', 'ARKK', 'IGV',
]

INDIVIDUAL_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AVGO', 'AMD', 'TSM',
    'LLY', 'UNH', 'JPM', 'V', 'MA', 'COST', 'CRM', 'ORCL',
    'NFLX', 'ADBE', 'NOW', 'UBER', 'ABNB', 'PLTR', 'COIN',
]

BENCHMARK = 'SPY'

START_DATE = '2019-01-01'
END_DATE = '2026-08-26'

TRAIN_MONTHS = 12
OOS_MONTHS = 1
TOP_N_SECTORS = 4
TOP_N_STOCKS = 8
REBAL_COST_BPS = 10  # 10 bps round-trip for ETFs
N_PERMUTATIONS = 500
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ── Data Download ───────────────────────────────────────────────────────────
def download_data(tickers):
    """Download OHLCV data with retries."""
    print("Downloading data...")
    data = {}
    all_tickers = list(set(tickers + [BENCHMARK]))
    for ticker in all_tickers:
        for attempt in range(3):
            try:
                df = yf.download(ticker, start=START_DATE, end=END_DATE,
                                 progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df) > 100:
                    data[ticker] = df
                    print(f"  {ticker}: {len(df)} bars")
                    break
                else:
                    time.sleep(1)
            except Exception as e:
                print(f"  {ticker} attempt {attempt+1} failed: {e}")
                time.sleep(2)
        if ticker not in data:
            print(f"  WARNING: {ticker} download FAILED — skipping")
    return data


# ── Feature Engineering ─────────────────────────────────────────────────────
def build_close_df(data, tickers):
    """Build aligned close price DataFrame."""
    available = [t for t in tickers if t in data]
    close_df = pd.DataFrame({t: data[t]['Close'] for t in available})
    close_df = close_df.dropna(how='all')
    return close_df, available


def momentum_features(close_df, windows=[21, 63, 126]):
    """Compute momentum (total return) over multiple lookback windows."""
    features = {}
    for w in windows:
        features[f'mom_{w}d'] = close_df.pct_change(w)
    return features


def relative_strength(close_df, spy_close, window=126):
    """Relative strength vs SPY."""
    asset_ret = close_df.pct_change(window)
    spy_ret = spy_close.pct_change(window)
    rs = asset_ret.subtract(spy_ret, axis=0)
    return rs


def trailing_stop_signal(close_df, atr_window=20, atr_mult=2.5):
    """Compute trailing stop levels using ATR-like proxy (high-low range)."""
    # Using close-based range as proxy since we only have close
    rolling_range = close_df.rolling(atr_window).std() * atr_mult
    stop_level = close_df.rolling(atr_window).max() - rolling_range
    return stop_level


# ── Strategy Implementations ───────────────────────────────────────────────

def get_rebal_dates(close_df, freq='ME'):
    """Get month-end rebalance dates."""
    dates = close_df.resample(freq).last().index
    return [d for d in dates if d in close_df.index or
            (close_df.index >= d - timedelta(days=5)).any()]


def strategy_a_sector_momentum(close_df, tickers, train_end, top_n=TOP_N_SECTORS):
    """
    Strategy A: Simple 6-month momentum rotation.
    Buy the top N sectors by 126-day return. Equal weight. Monthly rebalance.
    Always fully invested.
    """
    mom = close_df.pct_change(126)
    if train_end not in mom.index:
        # Find closest date
        valid = mom.index[mom.index <= train_end]
        if len(valid) == 0:
            return {}
        train_end = valid[-1]

    row = mom.loc[train_end, tickers].dropna()
    if len(row) < top_n:
        top_n = max(1, len(row))
    top = row.nlargest(top_n).index.tolist()
    return {t: 1.0 / len(top) for t in top}


def strategy_b_dual_momentum(close_df, spy_close, tickers, train_end, top_n=TOP_N_SECTORS):
    """
    Strategy B: Dual momentum — absolute + relative.
    Absolute: only buy assets with positive 6m return.
    Relative: rank by return vs SPY.
    If fewer than top_n pass absolute filter, fill remaining with strongest anyway
    (we MUST stay invested).
    """
    mom_6m = close_df.pct_change(126)
    spy_mom = spy_close.pct_change(126)

    if train_end not in mom_6m.index:
        valid = mom_6m.index[mom_6m.index <= train_end]
        if len(valid) == 0:
            return {}
        train_end = valid[-1]

    row = mom_6m.loc[train_end, tickers].dropna()
    spy_val = spy_mom.loc[train_end] if train_end in spy_mom.index else 0

    # Absolute filter: positive 6m return
    abs_pass = row[row > 0]

    # Relative: excess return vs SPY
    excess = row - spy_val
    ranked = excess.sort_values(ascending=False)

    # Prefer those passing absolute, but always stay invested
    if len(abs_pass) >= top_n:
        candidates = abs_pass.index
        rel_ranked = excess.loc[candidates].sort_values(ascending=False)
        top = rel_ranked.head(top_n).index.tolist()
    else:
        # Fill with strongest momentum even if negative (stay invested)
        top = ranked.head(top_n).index.tolist()

    return {t: 1.0 / len(top) for t in top}


def strategy_c_accelerating_momentum(close_df, tickers, train_end, top_n=TOP_N_SECTORS):
    """
    Strategy C: Accelerating momentum.
    Score = 1m_momentum - 3m_momentum/3 (momentum is increasing).
    Buy stocks where momentum is accelerating, not just high.
    """
    mom_1m = close_df.pct_change(21)
    mom_3m = close_df.pct_change(63)

    if train_end not in mom_1m.index:
        valid = mom_1m.index[mom_1m.index <= train_end]
        if len(valid) == 0:
            return {}
        train_end = valid[-1]

    r1 = mom_1m.loc[train_end, tickers].dropna()
    r3 = mom_3m.loc[train_end, tickers].dropna()
    common = r1.index.intersection(r3.index)
    if len(common) < top_n:
        top_n = max(1, len(common))

    # Acceleration = recent momentum exceeding the trend
    accel = r1[common] - r3[common] / 3
    top = accel.nlargest(top_n).index.tolist()
    return {t: 1.0 / len(top) for t in top}


def strategy_d_momentum_trailing_stop(close_df, tickers, train_end,
                                       top_n=TOP_N_SECTORS, stop_pct=0.10):
    """
    Strategy D: Momentum selection + trailing stop risk management.
    Select top momentum, but kick out any ticker whose price is >10% below
    its 63d high. Replace with next-best momentum.
    """
    mom_6m = close_df.pct_change(126)
    high_63 = close_df.rolling(63).max()

    if train_end not in mom_6m.index:
        valid = mom_6m.index[mom_6m.index <= train_end]
        if len(valid) == 0:
            return {}
        train_end = valid[-1]

    row = mom_6m.loc[train_end, tickers].dropna()
    prices = close_df.loc[train_end, tickers].dropna()
    highs = high_63.loc[train_end, tickers].dropna()

    common = row.index.intersection(prices.index).intersection(highs.index)
    if len(common) == 0:
        return {}

    # Filter: price within stop_pct of 63d high
    drawdown = (prices[common] - highs[common]) / highs[common]
    ok = drawdown[drawdown > -stop_pct].index
    ranked = row[common].sort_values(ascending=False)

    # Prefer those not stopped out
    selected = [t for t in ranked.index if t in ok][:top_n]
    # Fill if needed
    if len(selected) < top_n:
        remaining = [t for t in ranked.index if t not in selected]
        selected += remaining[:top_n - len(selected)]

    if len(selected) == 0:
        return {}
    return {t: 1.0 / len(selected) for t in selected}


def strategy_e_rs_vol_weighted(close_df, spy_close, tickers, train_end, top_n=TOP_N_SECTORS):
    """
    Strategy E: Relative strength + inverse-vol weighting.
    Select top RS vs SPY, weight by inverse realized vol (risk parity within winners).
    """
    rs = relative_strength(close_df, spy_close, window=126)
    vol = close_df.pct_change().rolling(63).std()

    if train_end not in rs.index:
        valid = rs.index[rs.index <= train_end]
        if len(valid) == 0:
            return {}
        train_end = valid[-1]

    rs_row = rs.loc[train_end, tickers].dropna()
    vol_row = vol.loc[train_end, tickers].dropna()
    common = rs_row.index.intersection(vol_row.index)

    if len(common) < top_n:
        top_n = max(1, len(common))

    top = rs_row[common].nlargest(top_n).index.tolist()

    # Inverse-vol weighting
    vols = vol_row[top].replace(0, np.nan).dropna()
    if len(vols) == 0:
        return {t: 1.0/len(top) for t in top}

    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()
    return weights.to_dict()


def strategy_f_multi_timeframe(close_df, tickers, train_end, top_n=TOP_N_SECTORS):
    """
    Strategy F: Multi-timeframe momentum blend.
    Score = 0.2 * rank(1m_mom) + 0.3 * rank(3m_mom) + 0.5 * rank(6m_mom)
    Emphasizes longer-term trends but gives credit to recent acceleration.
    """
    mom_1m = close_df.pct_change(21)
    mom_3m = close_df.pct_change(63)
    mom_6m = close_df.pct_change(126)

    if train_end not in mom_1m.index:
        valid = mom_1m.index[mom_1m.index <= train_end]
        if len(valid) == 0:
            return {}
        train_end = valid[-1]

    r1 = mom_1m.loc[train_end, tickers].dropna()
    r3 = mom_3m.loc[train_end, tickers].dropna()
    r6 = mom_6m.loc[train_end, tickers].dropna()

    common = r1.index.intersection(r3.index).intersection(r6.index)
    if len(common) < top_n:
        top_n = max(1, len(common))

    rank1 = r1[common].rank(pct=True)
    rank3 = r3[common].rank(pct=True)
    rank6 = r6[common].rank(pct=True)

    composite = 0.2 * rank1 + 0.3 * rank3 + 0.5 * rank6
    top = composite.nlargest(top_n).index.tolist()
    return {t: 1.0 / len(top) for t in top}


# ── Walk-Forward Engine ─────────────────────────────────────────────────────

STRATEGIES = {
    'A_sector_momentum': strategy_a_sector_momentum,
    'B_dual_momentum': strategy_b_dual_momentum,
    'C_accelerating_momentum': strategy_c_accelerating_momentum,
    'D_momentum_trailing_stop': strategy_d_momentum_trailing_stop,
    'E_rs_vol_weighted': strategy_e_rs_vol_weighted,
    'F_multi_timeframe': strategy_f_multi_timeframe,
}


def run_walk_forward(close_df, spy_close, tickers, strategy_name, strategy_fn,
                     train_months=TRAIN_MONTHS, oos_months=OOS_MONTHS):
    """
    Sliding walk-forward:
    - Train window = train_months (used for lookback signal computation)
    - OOS = oos_months (trade this month using signal from end of train)
    - Slide forward oos_months at a time
    """
    all_dates = close_df.index
    start = all_dates[0]

    # Build rebalance schedule
    results_returns = []
    all_trades = []

    # Start after enough lookback
    first_rebal = start + pd.DateOffset(months=train_months)
    rebal_dates = pd.date_range(first_rebal, all_dates[-1], freq='ME')
    rebal_dates = [d for d in rebal_dates if d <= all_dates[-1]]

    for i, rebal_date in enumerate(rebal_dates):
        # Find closest actual trading date
        valid_dates = all_dates[all_dates <= rebal_date]
        if len(valid_dates) == 0:
            continue
        train_end = valid_dates[-1]

        # OOS period: from train_end to next rebal
        if i + 1 < len(rebal_dates):
            next_rebal = rebal_dates[i + 1]
            valid_next = all_dates[all_dates <= next_rebal]
            if len(valid_next) == 0:
                continue
            oos_end = valid_next[-1]
        else:
            oos_end = all_dates[-1]

        oos_mask = (all_dates > train_end) & (all_dates <= oos_end)
        oos_dates = all_dates[oos_mask]
        if len(oos_dates) == 0:
            continue

        # Get strategy weights
        if strategy_name in ['B_dual_momentum', 'E_rs_vol_weighted']:
            weights = strategy_fn(close_df, spy_close, tickers, train_end)
        else:
            weights = strategy_fn(close_df, tickers, train_end)

        if not weights:
            continue

        # Compute OOS returns
        daily_ret = close_df[list(weights.keys())].pct_change()
        oos_ret = daily_ret.loc[oos_dates]

        port_ret = pd.Series(0.0, index=oos_dates)
        for t, w in weights.items():
            if t in oos_ret.columns:
                port_ret += w * oos_ret[t].fillna(0)

        # Subtract rebalance cost (proportional to turnover)
        turnover_cost = REBAL_COST_BPS / 10000.0  # Apply once at rebal
        if len(port_ret) > 0:
            port_ret.iloc[0] -= turnover_cost

        results_returns.append(port_ret)
        all_trades.append({
            'date': str(train_end.date()),
            'holdings': weights,
            'oos_days': len(oos_dates),
        })

    if not results_returns:
        return pd.Series(dtype=float), []

    full_returns = pd.concat(results_returns).sort_index()
    # Remove duplicates (overlapping periods)
    full_returns = full_returns[~full_returns.index.duplicated(keep='first')]
    return full_returns, all_trades


# ── Performance Metrics ─────────────────────────────────────────────────────

def compute_metrics(returns, label="Strategy"):
    """Compute comprehensive performance metrics."""
    returns = returns.dropna()
    if len(returns) < 30:
        return None

    ann_factor = 252
    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / ann_factor
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = cagr / (ann_vol + 1e-10)

    # Sortino
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(ann_factor) if len(downside) > 0 else 1e-10
    sortino = cagr / (downside_vol + 1e-10)

    # Max drawdown
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()

    # Win rate (monthly)
    monthly_ret = returns.resample('ME').apply(lambda x: (1+x).prod()-1)
    wr = (monthly_ret > 0).mean() if len(monthly_ret) > 0 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / (gross_loss + 1e-10)

    # Calmar
    calmar = cagr / (abs(max_dd) + 1e-10)

    return {
        'label': label,
        'total_return': f"{total_ret:.1%}",
        'cagr': f"{cagr:.1%}",
        'cagr_raw': cagr,
        'ann_vol': f"{ann_vol:.1%}",
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_dd': f"{max_dd:.1%}",
        'calmar': round(calmar, 2),
        'profit_factor': round(pf, 2),
        'win_rate_monthly': f"{wr:.0%}",
        'n_months': len(monthly_ret),
        'n_years': round(n_years, 1),
    }


def permutation_test(strat_returns, bench_returns, n_perms=N_PERMUTATIONS):
    """Permutation test: is strategy alpha statistically significant?"""
    # Align
    common = strat_returns.index.intersection(bench_returns.index)
    s = strat_returns.loc[common].values
    b = bench_returns.loc[common].values
    excess = s - b

    observed_mean = excess.mean()
    count = 0
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(excess))
        if (signs * excess).mean() >= observed_mean:
            count += 1
    p_value = count / n_perms
    return p_value


# ── Main Execution ──────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("AGGRESSIVE MOMENTUM / GROWTH ROTATION BACKTEST")
    print("=" * 80)
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Walk-forward: {TRAIN_MONTHS}m sliding train, {OOS_MONTHS}m OOS")
    print()

    # Download data for both universes
    all_tickers = list(set(SECTOR_ETFS + INDIVIDUAL_STOCKS + [BENCHMARK]))
    data = download_data(all_tickers)

    # Build close DataFrames
    sector_close, avail_sectors = build_close_df(data, SECTOR_ETFS)
    stock_close, avail_stocks = build_close_df(data, INDIVIDUAL_STOCKS)
    spy_close = data[BENCHMARK]['Close'] if BENCHMARK in data else None

    if spy_close is None:
        print("FATAL: SPY data not available")
        return

    # SPY benchmark returns
    spy_ret = spy_close.pct_change().dropna()

    print(f"\nAvailable sector ETFs: {len(avail_sectors)}")
    print(f"Available individual stocks: {len(avail_stocks)}")
    print()

    # ── Run all strategies ──────────────────────────────────────────────────
    results = {}

    # Strategies A-F on sector ETFs
    print("=" * 60)
    print("UNIVERSE 1: SECTOR ETFs (always invested, monthly rotation)")
    print("=" * 60)

    for name, fn in STRATEGIES.items():
        print(f"\nRunning {name}...")
        returns, trades = run_walk_forward(sector_close, spy_close, avail_sectors, name, fn)
        if len(returns) > 30:
            metrics = compute_metrics(returns, label=f"ETF_{name}")
            if metrics:
                p_val = permutation_test(returns, spy_ret)
                metrics['p_value'] = round(p_val, 3)
                metrics['alpha_sig'] = "YES" if p_val < 0.05 else "no"
                results[f"ETF_{name}"] = {'metrics': metrics, 'returns': returns, 'trades': trades}
                print(f"  CAGR={metrics['cagr']}  Sharpe={metrics['sharpe']}  "
                      f"Sortino={metrics['sortino']}  MaxDD={metrics['max_dd']}  "
                      f"PF={metrics['profit_factor']}  WR={metrics['win_rate_monthly']}")

    # Strategies on individual stocks (top_n = TOP_N_STOCKS)
    print()
    print("=" * 60)
    print("UNIVERSE 2: INDIVIDUAL STOCKS (top momentum names, monthly)")
    print("=" * 60)

    stock_strategies = {
        'A_sector_momentum': lambda df, tickers, te: strategy_a_sector_momentum(df, tickers, te, top_n=TOP_N_STOCKS),
        'C_accelerating_momentum': lambda df, tickers, te: strategy_c_accelerating_momentum(df, tickers, te, top_n=TOP_N_STOCKS),
        'D_momentum_trailing_stop': lambda df, tickers, te: strategy_d_momentum_trailing_stop(df, tickers, te, top_n=TOP_N_STOCKS),
        'F_multi_timeframe': lambda df, tickers, te: strategy_f_multi_timeframe(df, tickers, te, top_n=TOP_N_STOCKS),
    }
    stock_strategies_dual = {
        'B_dual_momentum': lambda df, spy, tickers, te: strategy_b_dual_momentum(df, spy, tickers, te, top_n=TOP_N_STOCKS),
        'E_rs_vol_weighted': lambda df, spy, tickers, te: strategy_e_rs_vol_weighted(df, spy, tickers, te, top_n=TOP_N_STOCKS),
    }

    for name, fn in stock_strategies.items():
        print(f"\nRunning STOCK_{name}...")
        returns, trades = run_walk_forward(stock_close, spy_close, avail_stocks, name, fn)
        if len(returns) > 30:
            metrics = compute_metrics(returns, label=f"STOCK_{name}")
            if metrics:
                p_val = permutation_test(returns, spy_ret)
                metrics['p_value'] = round(p_val, 3)
                metrics['alpha_sig'] = "YES" if p_val < 0.05 else "no"
                results[f"STOCK_{name}"] = {'metrics': metrics, 'returns': returns, 'trades': trades}
                print(f"  CAGR={metrics['cagr']}  Sharpe={metrics['sharpe']}  "
                      f"Sortino={metrics['sortino']}  MaxDD={metrics['max_dd']}  "
                      f"PF={metrics['profit_factor']}  WR={metrics['win_rate_monthly']}")

    for name, fn in stock_strategies_dual.items():
        print(f"\nRunning STOCK_{name}...")
        returns, trades = run_walk_forward(stock_close, spy_close, avail_stocks, name, fn)
        if len(returns) > 30:
            metrics = compute_metrics(returns, label=f"STOCK_{name}")
            if metrics:
                p_val = permutation_test(returns, spy_ret)
                metrics['p_value'] = round(p_val, 3)
                metrics['alpha_sig'] = "YES" if p_val < 0.05 else "no"
                results[f"STOCK_{name}"] = {'metrics': metrics, 'returns': returns, 'trades': trades}
                print(f"  CAGR={metrics['cagr']}  Sharpe={metrics['sharpe']}  "
                      f"Sortino={metrics['sortino']}  MaxDD={metrics['max_dd']}  "
                      f"PF={metrics['profit_factor']}  WR={metrics['win_rate_monthly']}")

    # ── SPY Benchmark ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("BENCHMARK: SPY BUY-AND-HOLD")
    print("=" * 60)

    # Align SPY to same OOS period as strategies
    if results:
        first_key = list(results.keys())[0]
        first_date = results[first_key]['returns'].index[0]
        spy_aligned = spy_ret[spy_ret.index >= first_date]
    else:
        spy_aligned = spy_ret

    spy_metrics = compute_metrics(spy_aligned, label="SPY_BuyHold")
    if spy_metrics:
        print(f"  CAGR={spy_metrics['cagr']}  Sharpe={spy_metrics['sharpe']}  "
              f"Sortino={spy_metrics['sortino']}  MaxDD={spy_metrics['max_dd']}  "
              f"PF={spy_metrics['profit_factor']}  WR={spy_metrics['win_rate_monthly']}")

    # ── Summary Table ───────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("FINAL RESULTS — SORTED BY CAGR (DESCENDING)")
    print("=" * 80)

    all_metrics = []
    for k, v in results.items():
        all_metrics.append(v['metrics'])
    if spy_metrics:
        spy_metrics['p_value'] = '-'
        spy_metrics['alpha_sig'] = '-'
        all_metrics.append(spy_metrics)

    # Sort by CAGR
    all_metrics.sort(key=lambda x: x.get('cagr_raw', 0), reverse=True)

    # Print table
    header = f"{'Strategy':<35} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'PF':>6} {'WR':>6} {'Calmar':>7} {'p-val':>6} {'Alpha':>6}"
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        beat_spy = ""
        if spy_metrics and m['label'] != 'SPY_BuyHold':
            if m.get('cagr_raw', 0) > spy_metrics.get('cagr_raw', 0):
                beat_spy = " ** BEATS SPY"
        print(f"{m['label']:<35} {m['cagr']:>7} {m['sharpe']:>7} {m['sortino']:>8} "
              f"{m['max_dd']:>8} {m['profit_factor']:>6} {m['win_rate_monthly']:>6} "
              f"{m['calmar']:>7} {str(m.get('p_value','-')):>6} {str(m.get('alpha_sig','-')):>6}{beat_spy}")

    # ── Regime Analysis (simple: green/red months) ──────────────────────────
    print()
    print("=" * 60)
    print("REGIME ANALYSIS: Performance in UP vs DOWN SPY months")
    print("=" * 60)

    spy_monthly = spy_ret.resample('ME').apply(lambda x: (1+x).prod()-1)
    up_months = spy_monthly[spy_monthly > 0].index
    down_months = spy_monthly[spy_monthly <= 0].index

    for k, v in results.items():
        ret = v['returns']
        monthly = ret.resample('ME').apply(lambda x: (1+x).prod()-1)
        up_ret = monthly[monthly.index.isin(up_months)]
        down_ret = monthly[monthly.index.isin(down_months)]

        up_mean = up_ret.mean() * 12 if len(up_ret) > 0 else 0
        down_mean = down_ret.mean() * 12 if len(down_ret) > 0 else 0
        ratio = abs(up_mean / (down_mean + 1e-10)) if down_mean != 0 else float('inf')

        print(f"  {k:<35} UP={up_mean:+.1%}/yr  DOWN={down_mean:+.1%}/yr  ratio={ratio:.1f}")

    # ── Year-by-Year Breakdown (top 3 strategies) ──────────────────────────
    print()
    print("=" * 60)
    print("YEAR-BY-YEAR CAGR — Top strategies vs SPY")
    print("=" * 60)

    # Pick top 3 by CAGR
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics'].get('cagr_raw', 0), reverse=True)
    top3 = sorted_results[:3]

    years = range(2020, 2027)
    header = f"{'Year':<6}"
    for k, _ in top3:
        short = k[:25]
        header += f" {short:>25}"
    header += f" {'SPY':>10}"
    print(header)
    print("-" * len(header))

    for yr in years:
        row = f"{yr:<6}"
        for k, v in top3:
            yr_ret = v['returns'][v['returns'].index.year == yr]
            if len(yr_ret) > 0:
                yr_total = (1 + yr_ret).prod() - 1
                row += f" {yr_total:>24.1%}"
            else:
                row += f" {'N/A':>25}"
        # SPY
        spy_yr = spy_ret[spy_ret.index.year == yr]
        if len(spy_yr) > 0:
            spy_yr_total = (1 + spy_yr).prod() - 1
            row += f" {spy_yr_total:>9.1%}"
        else:
            row += f" {'N/A':>10}"
        print(row)

    # ── Save results ────────────────────────────────────────────────────────
    save_data = {
        'run_date': datetime.now().isoformat(),
        'config': {
            'start': START_DATE, 'end': END_DATE,
            'train_months': TRAIN_MONTHS, 'oos_months': OOS_MONTHS,
            'top_n_sectors': TOP_N_SECTORS, 'top_n_stocks': TOP_N_STOCKS,
            'rebal_cost_bps': REBAL_COST_BPS,
        },
        'strategies': {k: v['metrics'] for k, v in results.items()},
        'benchmark': spy_metrics,
    }

    out_path = os.path.join(os.path.dirname(__file__), 'momentum_growth_results.json')
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ── Bottom Line ─────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    if all_metrics and spy_metrics:
        best = all_metrics[0]
        spy_cagr = spy_metrics.get('cagr_raw', 0)
        best_cagr = best.get('cagr_raw', 0)
        if best_cagr > spy_cagr:
            print(f"BEST STRATEGY: {best['label']}")
            print(f"  CAGR {best['cagr']} vs SPY {spy_metrics['cagr']} "
                  f"(+{best_cagr - spy_cagr:.1%} excess)")
            print(f"  Sharpe {best['sharpe']}  Sortino {best['sortino']}  MaxDD {best['max_dd']}")
        else:
            print("No strategy beat SPY on CAGR in this backtest period.")
            print("Consider: leveraged variants, bi-weekly rebalance, or combined stock+ETF universe.")
    print("=" * 80)


if __name__ == '__main__':
    main()
