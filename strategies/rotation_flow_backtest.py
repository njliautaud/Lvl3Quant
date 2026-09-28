#!/usr/bin/env python3
"""
Sector Rotation + Money Flow Backtest
======================================
6 strategy variants with 5-gate validation.
Walk-forward: 12-month sliding train, 1-month OOS.
Backtest: Jan 2020 — Jul 2026.

Strategies:
  A) Relative Strength Momentum
  B) Flow-Weighted Rotation (MFI/OBV)
  C) Price-Flow Divergence (accumulation detection)
  D) Mean-Reversion Rotation
  E) Dual Momentum (absolute + relative)
  F) Composite (A + B + C blend)
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
import sys
import time

# ── Config ──────────────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLU']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]

START_DATE = '2020-01-01'
END_DATE = '2026-08-01'

TRAIN_MONTHS = 12
OOS_MONTHS = 1
TOP_N = 3          # sectors to hold in rotation
REBAL_FREQ = 'M'   # monthly rebalance
OPTION_SPREAD_COST = 0.01  # 1% bid-ask for options overlay estimate
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

# ── Data Download ───────────────────────────────────────────────────────────
def download_data():
    """Download OHLCV data for all sector ETFs + SPY."""
    print("Downloading data...")
    data = {}
    for ticker in ALL_TICKERS:
        for attempt in range(3):
            try:
                df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
                if len(df) > 100:
                    # Flatten multi-level columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    data[ticker] = df
                    print(f"  {ticker}: {len(df)} bars")
                    break
                else:
                    print(f"  {ticker}: insufficient data ({len(df)} bars), retrying...")
                    time.sleep(1)
            except Exception as e:
                print(f"  {ticker} attempt {attempt+1} failed: {e}")
                time.sleep(2)
        if ticker not in data:
            print(f"  WARNING: {ticker} download FAILED")
    return data

# ── Feature Engineering ─────────────────────────────────────────────────────
def compute_features(data):
    """Compute rotation and money flow features for all sectors."""
    close_df = pd.DataFrame({t: data[t]['Close'] for t in SECTOR_ETFS if t in data})
    volume_df = pd.DataFrame({t: data[t]['Volume'] for t in SECTOR_ETFS if t in data})
    high_df = pd.DataFrame({t: data[t]['High'] for t in SECTOR_ETFS if t in data})
    low_df = pd.DataFrame({t: data[t]['Low'] for t in SECTOR_ETFS if t in data})

    spy_close = data[BENCHMARK]['Close'] if BENCHMARK in data else close_df.mean(axis=1)

    # Align all dataframes
    idx = close_df.index
    volume_df = volume_df.reindex(idx)
    high_df = high_df.reindex(idx)
    low_df = low_df.reindex(idx)
    if isinstance(spy_close, pd.Series):
        spy_close = spy_close.reindex(idx)

    features = {}

    # 1. Returns at various lookbacks
    for lb in [5, 10, 21, 63]:
        features[f'ret_{lb}d'] = close_df.pct_change(lb)

    # 2. Relative strength vs SPY
    for lb in [21, 63]:
        sector_ret = close_df.pct_change(lb)
        spy_ret = spy_close.pct_change(lb)
        features[f'rel_str_{lb}d'] = sector_ret.subtract(spy_ret, axis=0)

    # 3. On-Balance Volume (OBV) trend
    obv = {}
    for t in close_df.columns:
        direction = np.sign(close_df[t].diff())
        obv[t] = (direction * volume_df[t]).cumsum()
    obv_df = pd.DataFrame(obv)
    # OBV slope (21-day regression slope, normalized)
    obv_slope = obv_df.rolling(21).apply(
        lambda x: np.polyfit(np.arange(len(x)), x / (np.abs(x).mean() + 1e-10), 1)[0] if np.abs(x).mean() > 0 else 0,
        raw=True
    )
    features['obv_slope'] = obv_slope

    # 4. Money Flow Index (14-day)
    tp = (high_df + low_df + close_df) / 3  # typical price
    raw_mf = tp * volume_df
    mf_direction = tp.diff()
    pos_mf = raw_mf.where(mf_direction > 0, 0).rolling(14).sum()
    neg_mf = raw_mf.where(mf_direction <= 0, 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
    features['mfi'] = mfi

    # 5. Chaikin Money Flow (21-day)
    mfm = ((close_df - low_df) - (high_df - close_df)) / (high_df - low_df + 1e-10)
    mfv = mfm * volume_df
    cmf = mfv.rolling(21).sum() / (volume_df.rolling(21).sum() + 1e-10)
    features['cmf'] = cmf

    # 6. Accumulation/Distribution line trend
    ad = mfv.cumsum()
    ad_slope = ad.rolling(21).apply(
        lambda x: np.polyfit(np.arange(len(x)), x / (np.abs(x).mean() + 1e-10), 1)[0] if np.abs(x).mean() > 0 else 0,
        raw=True
    )
    features['ad_slope'] = ad_slope

    # 7. Volume trend (21d SMA ratio)
    vol_ratio = volume_df.rolling(5).mean() / (volume_df.rolling(21).mean() + 1e-10)
    features['vol_ratio'] = vol_ratio

    # 8. Drawdown from 63d high
    rolling_max = close_df.rolling(63).max()
    dd = (close_df - rolling_max) / rolling_max
    features['dd_63d'] = dd

    # 9. Price-flow divergence: MFI rising but price flat/down
    price_chg_21 = close_df.pct_change(21)
    mfi_chg_21 = mfi.diff(21)
    # Positive divergence: flow up, price flat/down
    features['pf_divergence'] = mfi_chg_21.subtract(price_chg_21.multiply(100), axis='index')

    # 10. SPY regime (bull/bear based on 200d SMA)
    spy_sma200 = spy_close.rolling(200).mean()
    regime = (spy_close > spy_sma200).astype(int)  # 1=bull, 0=bear

    return features, close_df, volume_df, spy_close, regime

# ── Strategy Implementations ────────────────────────────────────────────────

def get_monthly_rebal_dates(close_df):
    """Get month-end rebalance dates."""
    monthly = close_df.resample('ME').last()
    return monthly.index

def strategy_A_relative_strength(features, close_df, rebal_dates, top_n=TOP_N):
    """A) Relative Strength Momentum: buy top N sectors by 1-month return."""
    positions = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    trades = []

    ret_21 = features['ret_21d']

    for i, date in enumerate(rebal_dates):
        if date not in ret_21.index:
            continue
        row = ret_21.loc[date].dropna()
        if len(row) < top_n:
            continue
        top = row.nlargest(top_n).index.tolist()

        # Hold until next rebal
        next_date = rebal_dates[i+1] if i+1 < len(rebal_dates) else close_df.index[-1]
        mask = (close_df.index > date) & (close_df.index <= next_date)
        for t in top:
            positions.loc[mask, t] = 1.0 / top_n
            trades.append({'date': date, 'ticker': t, 'side': 'long', 'signal': 'rel_str'})

    return positions, trades

def strategy_B_flow_weighted(features, close_df, rebal_dates, top_n=TOP_N):
    """B) Flow-Weighted Rotation: rank by MFI + OBV slope composite."""
    positions = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    trades = []

    mfi = features['mfi']
    obv_slope = features['obv_slope']

    for i, date in enumerate(rebal_dates):
        if date not in mfi.index:
            continue
        mfi_row = mfi.loc[date].dropna()
        obv_row = obv_slope.loc[date].dropna()
        common = mfi_row.index.intersection(obv_row.index)
        if len(common) < top_n:
            continue

        # Composite: rank MFI + rank OBV slope
        mfi_rank = mfi_row[common].rank()
        obv_rank = obv_row[common].rank()
        composite = mfi_rank + obv_rank
        top = composite.nlargest(top_n).index.tolist()

        next_date = rebal_dates[i+1] if i+1 < len(rebal_dates) else close_df.index[-1]
        mask = (close_df.index > date) & (close_df.index <= next_date)
        for t in top:
            positions.loc[mask, t] = 1.0 / top_n
            trades.append({'date': date, 'ticker': t, 'side': 'long', 'signal': 'flow'})

    return positions, trades

def strategy_C_divergence(features, close_df, rebal_dates, top_n=TOP_N):
    """C) Price-Flow Divergence: buy sectors with rising flow but flat/down price."""
    positions = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    trades = []

    div = features['pf_divergence']

    for i, date in enumerate(rebal_dates):
        if date not in div.index:
            continue
        row = div.loc[date].dropna()
        if len(row) < top_n:
            continue

        # Highest positive divergence = most accumulation
        top = row.nlargest(top_n).index.tolist()
        # Only take if divergence is actually positive
        top = [t for t in top if row[t] > 0]
        if not top:
            continue

        next_date = rebal_dates[i+1] if i+1 < len(rebal_dates) else close_df.index[-1]
        mask = (close_df.index > date) & (close_df.index <= next_date)
        weight = 1.0 / max(len(top), 1)
        for t in top:
            positions.loc[mask, t] = weight
            trades.append({'date': date, 'ticker': t, 'side': 'long', 'signal': 'divergence'})

    return positions, trades

def strategy_D_mean_reversion(features, close_df, rebal_dates, top_n=TOP_N):
    """D) Mean-Reversion: buy sectors in >5% drawdown, sell after recovery."""
    positions = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    trades = []

    dd = features['dd_63d']
    ret_5 = features['ret_5d']

    for i, date in enumerate(rebal_dates):
        if date not in dd.index:
            continue
        dd_row = dd.loc[date].dropna()
        ret_row = ret_5.loc[date].dropna()

        # Sectors in drawdown > 5% AND showing recent bounce (5d ret > 0)
        candidates = dd_row[dd_row < -0.05].index
        bouncing = [t for t in candidates if t in ret_row.index and ret_row[t] > 0]

        if not bouncing:
            # Just take deepest drawdowns
            if len(dd_row[dd_row < -0.05]) > 0:
                bouncing = dd_row[dd_row < -0.05].nsmallest(top_n).index.tolist()

        if not bouncing:
            continue

        bouncing = bouncing[:top_n]
        next_date = rebal_dates[i+1] if i+1 < len(rebal_dates) else close_df.index[-1]
        mask = (close_df.index > date) & (close_df.index <= next_date)
        weight = 1.0 / len(bouncing)
        for t in bouncing:
            positions.loc[mask, t] = weight
            trades.append({'date': date, 'ticker': t, 'side': 'long', 'signal': 'mean_rev'})

    return positions, trades

def strategy_E_dual_momentum(features, close_df, rebal_dates, top_n=TOP_N):
    """E) Dual Momentum: absolute + relative. Only buy if sector has positive
    absolute momentum AND is in the top relative ranking."""
    positions = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    trades = []

    ret_21 = features['ret_21d']
    rel_str = features['rel_str_21d']

    for i, date in enumerate(rebal_dates):
        if date not in ret_21.index:
            continue
        abs_row = ret_21.loc[date].dropna()
        rel_row = rel_str.loc[date].dropna()
        common = abs_row.index.intersection(rel_row.index)

        # Absolute filter: must have positive 21d return
        pos_mom = [t for t in common if abs_row[t] > 0]
        if not pos_mom:
            continue  # go to cash (no position)

        # Relative rank among positive-momentum sectors
        rel_sub = rel_row[pos_mom]
        top = rel_sub.nlargest(min(top_n, len(pos_mom))).index.tolist()

        next_date = rebal_dates[i+1] if i+1 < len(rebal_dates) else close_df.index[-1]
        mask = (close_df.index > date) & (close_df.index <= next_date)
        weight = 1.0 / len(top)
        for t in top:
            positions.loc[mask, t] = weight
            trades.append({'date': date, 'ticker': t, 'side': 'long', 'signal': 'dual_mom'})

    return positions, trades

def strategy_F_composite(features, close_df, rebal_dates, top_n=TOP_N):
    """F) Composite: blend of relative strength, flow, and divergence scores."""
    positions = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    trades = []

    ret_21 = features['ret_21d']
    mfi = features['mfi']
    obv_slope = features['obv_slope']
    div = features['pf_divergence']
    rel_str = features['rel_str_21d']

    for i, date in enumerate(rebal_dates):
        if date not in ret_21.index:
            continue

        # Get all available data
        scores = {}
        for t in close_df.columns:
            s = 0
            n = 0
            if t in ret_21.columns and date in ret_21.index and not pd.isna(ret_21.loc[date, t]):
                n += 1
            if t in mfi.columns and date in mfi.index and not pd.isna(mfi.loc[date, t]):
                n += 1
            if t in obv_slope.columns and date in obv_slope.index and not pd.isna(obv_slope.loc[date, t]):
                n += 1
            if n < 2:
                continue
            scores[t] = 0

        if len(scores) < top_n:
            continue

        # Rank each signal and combine
        valid_tickers = list(scores.keys())

        # Momentum rank
        mom_vals = {t: ret_21.loc[date, t] for t in valid_tickers if not pd.isna(ret_21.loc[date, t])}
        if mom_vals:
            mom_rank = pd.Series(mom_vals).rank()
            for t in mom_rank.index:
                scores[t] += mom_rank[t]

        # MFI rank
        mfi_vals = {t: mfi.loc[date, t] for t in valid_tickers if not pd.isna(mfi.loc[date, t])}
        if mfi_vals:
            mfi_rank = pd.Series(mfi_vals).rank()
            for t in mfi_rank.index:
                scores[t] += mfi_rank[t]

        # OBV slope rank
        obv_vals = {t: obv_slope.loc[date, t] for t in valid_tickers if not pd.isna(obv_slope.loc[date, t])}
        if obv_vals:
            obv_rank = pd.Series(obv_vals).rank()
            for t in obv_rank.index:
                scores[t] += obv_rank[t]

        # Divergence rank
        div_vals = {t: div.loc[date, t] for t in valid_tickers if not pd.isna(div.loc[date, t])}
        if div_vals:
            div_rank = pd.Series(div_vals).rank()
            for t in div_rank.index:
                scores[t] += div_rank[t]

        # Relative strength rank
        rs_vals = {t: rel_str.loc[date, t] for t in valid_tickers if not pd.isna(rel_str.loc[date, t])}
        if rs_vals:
            rs_rank = pd.Series(rs_vals).rank()
            for t in rs_rank.index:
                scores[t] += rs_rank[t]

        score_series = pd.Series(scores)
        top = score_series.nlargest(top_n).index.tolist()

        next_date = rebal_dates[i+1] if i+1 < len(rebal_dates) else close_df.index[-1]
        mask = (close_df.index > date) & (close_df.index <= next_date)
        weight = 1.0 / len(top)
        for t in top:
            positions.loc[mask, t] = weight
            trades.append({'date': date, 'ticker': t, 'side': 'long', 'signal': 'composite'})

    return positions, trades

# ── Performance Metrics ─────────────────────────────────────────────────────

def compute_strategy_returns(positions, close_df):
    """Compute daily strategy returns from position weights."""
    daily_ret = close_df.pct_change()
    # Strategy return = sum of (weight * daily return) across sectors
    strat_ret = (positions.shift(1) * daily_ret).sum(axis=1)
    # Apply options spread cost at each rebalance (position change)
    pos_change = positions.diff().abs().sum(axis=1)
    cost = pos_change * OPTION_SPREAD_COST * 0.5  # half-spread per side
    strat_ret = strat_ret - cost
    return strat_ret

def compute_metrics(returns, regime_series=None):
    """Compute comprehensive performance metrics."""
    returns = returns.dropna()
    if len(returns) < 30:
        return None

    # Annualize
    ann_factor = 252

    total_ret = (1 + returns).prod() - 1
    ann_ret = (1 + total_ret) ** (ann_factor / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / (ann_vol + 1e-10)

    # Sortino
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(ann_factor) if len(downside) > 0 else 1e-10
    sortino = ann_ret / (downside_vol + 1e-10)

    # Max drawdown
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()

    # Win rate (monthly)
    monthly_ret = returns.resample('ME').sum()
    wr = (monthly_ret > 0).mean() if len(monthly_ret) > 0 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / (gross_loss + 1e-10)

    # Calmar
    calmar = ann_ret / (abs(max_dd) + 1e-10)

    metrics = {
        'total_return': total_ret,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'win_rate': wr,
        'profit_factor': pf,
        'calmar': calmar,
        'n_days': len(returns),
    }

    # Regime analysis
    if regime_series is not None:
        regime_aligned = regime_series.reindex(returns.index).ffill()
        bull_ret = returns[regime_aligned == 1]
        bear_ret = returns[regime_aligned == 0]

        if len(bull_ret) > 20:
            bull_ann = bull_ret.mean() * ann_factor
            bull_vol = bull_ret.std() * np.sqrt(ann_factor)
            metrics['sharpe_bull'] = bull_ann / (bull_vol + 1e-10)
        else:
            metrics['sharpe_bull'] = np.nan

        if len(bear_ret) > 20:
            bear_ann = bear_ret.mean() * ann_factor
            bear_vol = bear_ret.std() * np.sqrt(ann_factor)
            metrics['sharpe_bear'] = bear_ann / (bear_vol + 1e-10)
        else:
            metrics['sharpe_bear'] = np.nan

        # Regime gap
        if not np.isnan(metrics['sharpe_bull']) and not np.isnan(metrics['sharpe_bear']):
            max_sharpe = max(abs(metrics['sharpe_bull']), abs(metrics['sharpe_bear']))
            if max_sharpe > 0:
                metrics['regime_gap'] = abs(metrics['sharpe_bull'] - metrics['sharpe_bear']) / max_sharpe
            else:
                metrics['regime_gap'] = 0
        else:
            metrics['regime_gap'] = np.nan

    return metrics

def permutation_test(strategy_returns, close_df, positions, n_perms=N_PERMUTATIONS):
    """Permutation test: randomly rotate sector ASSIGNMENTS at each rebalance,
    keeping the rebalance timing fixed. Tests whether sector SELECTION matters.
    Vectorized for speed."""
    np.random.seed(RANDOM_SEED)
    strategy_returns = strategy_returns.dropna()
    actual_sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-10) * np.sqrt(252)

    daily_ret = close_df.pct_change().values  # (T, N_sectors)
    sectors = close_df.columns.tolist()
    n_sectors = len(sectors)
    T = len(close_df)

    # Find rebalance segments: contiguous blocks with same allocation
    pos_arr = positions.values  # (T, N_sectors)
    pos_changes = np.abs(np.diff(pos_arr, axis=0)).sum(axis=1)
    rebal_idx = np.where(pos_changes > 0.01)[0] + 1  # +1 because diff shifts by 1
    rebal_idx = np.concatenate([[0], rebal_idx])

    # For each segment, record: start_idx, end_idx, n_held
    segments = []
    for i, start in enumerate(rebal_idx):
        end = rebal_idx[i+1] if i+1 < len(rebal_idx) else T
        n_held = int((pos_arr[start] > 0.01).sum())
        segments.append((start, end, n_held))

    # Precompute: for each segment, equal-weight random portfolio returns
    # For each permutation, pick random sectors per segment
    shuffled_sharpes = np.empty(n_perms)
    for p in range(n_perms):
        perm_ret = np.zeros(T)
        for start, end, n_held in segments:
            if n_held == 0 or start >= T - 1:
                continue
            chosen = np.random.choice(n_sectors, size=n_held, replace=False)
            weight = 1.0 / n_held
            # Use shifted positions (t-1 weight * t return)
            s = max(start + 1, 1)  # shift by 1 for positions.shift(1)
            e = min(end + 1, T)
            perm_ret[s:e] += weight * daily_ret[s:e, chosen].sum(axis=1)

        perm_ret_clean = perm_ret[~np.isnan(perm_ret)]
        if len(perm_ret_clean) > 0 and perm_ret_clean.std() > 0:
            shuffled_sharpes[p] = perm_ret_clean.mean() / perm_ret_clean.std() * np.sqrt(252)
        else:
            shuffled_sharpes[p] = 0.0

    p_val = np.mean(shuffled_sharpes >= actual_sharpe)
    return p_val, actual_sharpe

def walk_forward_test(strategy_func, features, close_df, rebal_dates, regime, **kwargs):
    """Run walk-forward validation with sliding 12-month train, 1-month OOS."""
    # For rotation strategies, walk-forward is embedded in the monthly rebalance
    # The "training" is the lookback period used to compute signals
    # The "OOS" is the next month's performance
    # We already use lookback-based signals, so the backtest IS walk-forward by construction

    positions, trades = strategy_func(features, close_df, rebal_dates, **kwargs)
    returns = compute_strategy_returns(positions, close_df)
    metrics = compute_metrics(returns, regime)

    if metrics is None:
        return None, None, None, None

    # Permutation test (sector-selection shuffle)
    p_val, _ = permutation_test(returns, close_df, positions)
    metrics['perm_p_value'] = p_val
    metrics['n_trades'] = len(trades)

    return metrics, returns, positions, trades

# ── 5-Gate Validation ───────────────────────────────────────────────────────

def validate_5_gates(metrics, name):
    """Apply 5-gate test. Returns dict of gate results."""
    gates = {}

    if metrics is None:
        return {f'G{i}': False for i in range(1, 6)}, "NO DATA"

    # G1: Sharpe > 0.5
    gates['G1_sharpe'] = metrics['sharpe'] > 0.5

    # G2: Permutation test p < 0.05
    gates['G2_perm'] = metrics.get('perm_p_value', 1.0) < 0.05

    # G3: Regime balance (gap < 0.50)
    regime_gap = metrics.get('regime_gap', np.nan)
    gates['G3_regime'] = not np.isnan(regime_gap) and regime_gap < 0.50

    # G4: Max drawdown < 30%
    gates['G4_mdd'] = metrics['max_dd'] > -0.30

    # G5: Minimum 30 trades
    gates['G5_trades'] = metrics.get('n_trades', 0) >= 30

    passed = sum(gates.values())
    status = f"{passed}/5 PASS" + (" *** VIABLE ***" if passed == 5 else "")

    return gates, status

# ── Options Overlay Analysis ────────────────────────────────────────────────

def options_overlay_analysis(trades, close_df, features):
    """Analyze what options trades each signal implies."""
    print("\n" + "="*80)
    print("OPTIONS OVERLAY ANALYSIS")
    print("="*80)

    # Group trades by signal type and outcome
    if not trades:
        print("  No trades to analyze.")
        return

    trade_df = pd.DataFrame(trades)

    # For each trade, compute the forward 1-month return
    results = []
    for _, t in trade_df.iterrows():
        date = t['date']
        ticker = t['ticker']
        if date not in close_df.index:
            continue
        # Find entry price
        entry_idx = close_df.index.get_loc(date)
        if entry_idx + 21 >= len(close_df):
            continue
        entry_price = close_df.iloc[entry_idx][ticker]
        exit_price = close_df.iloc[min(entry_idx + 21, len(close_df)-1)][ticker]
        ret = (exit_price - entry_price) / entry_price
        results.append({
            'signal': t['signal'],
            'ticker': ticker,
            'date': date,
            'ret': ret,
            'win': ret > OPTION_SPREAD_COST  # net of spread
        })

    if not results:
        return

    res_df = pd.DataFrame(results)

    print(f"\n  Total signal-trades: {len(res_df)}")
    print(f"  Win rate (net of 1% spread): {res_df['win'].mean():.1%}")
    print(f"  Avg 21d return: {res_df['ret'].mean():.2%}")
    print(f"  Median 21d return: {res_df['ret'].median():.2%}")

    print("\n  By signal type:")
    for sig, grp in res_df.groupby('signal'):
        print(f"    {sig:15s}: WR={grp['win'].mean():.1%}, avg_ret={grp['ret'].mean():+.2%}, "
              f"med_ret={grp['ret'].median():+.2%}, n={len(grp)}")

    print("\n  OPTIONS TRADE MAPPING:")
    print("    - Rotating-IN sectors (positive momentum + flow): BUY CALLS (30-45 DTE, ATM or slightly OTM)")
    print("    - Rotating-OUT sectors (negative momentum + flow): BUY PUTS (30-45 DTE, ATM or slightly OTM)")
    print("    - Divergence setups (flow up, price flat): BUY CALLS with longer DTE (60-90d) for breakout")
    print("    - Mean-reversion bounces: SELL PUTS (cash-secured) on deeply oversold sectors")

    # Best sector for options
    by_ticker = res_df.groupby('ticker').agg({'ret': ['mean', 'count'], 'win': 'mean'})
    by_ticker.columns = ['avg_ret', 'n_trades', 'win_rate']
    by_ticker = by_ticker.sort_values('avg_ret', ascending=False)
    print("\n  Sector ETF options ranking (by avg signal return):")
    for t in by_ticker.index:
        r = by_ticker.loc[t]
        print(f"    {t}: avg_ret={r['avg_ret']:+.2%}, WR={r['win_rate']:.1%}, n={int(r['n_trades'])}")

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print("="*80)
    print("SECTOR ROTATION + MONEY FLOW BACKTEST")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Sectors: {', '.join(SECTOR_ETFS)}")
    print(f"Benchmark: {BENCHMARK}")
    print("="*80)

    # Download data
    data = download_data()
    if len(data) < 6:
        print("ERROR: Insufficient data downloaded. Aborting.")
        sys.exit(1)

    # Compute features
    print("\nComputing features (MFI, OBV, CMF, divergence, drawdown)...")
    features, close_df, volume_df, spy_close, regime = compute_features(data)

    # Get rebalance dates
    rebal_dates = get_monthly_rebal_dates(close_df)
    print(f"Rebalance dates: {len(rebal_dates)} months")

    # Benchmark returns
    spy_ret = spy_close.pct_change().dropna()
    spy_metrics = compute_metrics(spy_ret)

    print(f"\nSPY Benchmark: Sharpe={spy_metrics['sharpe']:.2f}, "
          f"Sortino={spy_metrics['sortino']:.2f}, MDD={spy_metrics['max_dd']:.1%}, "
          f"Total={spy_metrics['total_return']:.1%}")

    # Run all strategies
    strategies = {
        'A) Rel Strength Mom': (strategy_A_relative_strength, {}),
        'B) Flow-Weighted': (strategy_B_flow_weighted, {}),
        'C) Price-Flow Div': (strategy_C_divergence, {}),
        'D) Mean Reversion': (strategy_D_mean_reversion, {}),
        'E) Dual Momentum': (strategy_E_dual_momentum, {}),
        'F) Composite': (strategy_F_composite, {}),
    }

    all_results = {}
    all_trades = []

    print("\n" + "="*80)
    print("STRATEGY RESULTS")
    print("="*80)

    for name, (func, kwargs) in strategies.items():
        print(f"\n{'─'*60}")
        print(f"  {name}")
        print(f"{'─'*60}")

        metrics, returns, positions, trades = walk_forward_test(
            func, features, close_df, rebal_dates, regime, **kwargs
        )

        if metrics is None:
            print("  ** NO VALID RESULTS **")
            continue

        gates, status = validate_5_gates(metrics, name)

        print(f"  Sharpe:        {metrics['sharpe']:+.3f}")
        print(f"  Sortino:       {metrics['sortino']:+.3f}")
        print(f"  Ann Return:    {metrics['ann_return']:+.1%}")
        print(f"  Max Drawdown:  {metrics['max_dd']:.1%}")
        print(f"  Win Rate (mo): {metrics['win_rate']:.1%}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Calmar:        {metrics['calmar']:.2f}")
        print(f"  Total Return:  {metrics['total_return']:+.1%}")
        print(f"  Trades:        {metrics['n_trades']}")
        print(f"  Sharpe (Bull): {metrics.get('sharpe_bull', float('nan')):+.3f}")
        print(f"  Sharpe (Bear): {metrics.get('sharpe_bear', float('nan')):+.3f}")
        print(f"  Regime Gap:    {metrics.get('regime_gap', float('nan')):.3f}")
        print(f"  Perm p-value:  {metrics.get('perm_p_value', float('nan')):.4f}")
        print(f"  5-Gate:        {status}")
        for g, v in gates.items():
            print(f"    {g}: {'PASS' if v else 'FAIL'}")

        all_results[name] = {
            'metrics': metrics,
            'gates': gates,
            'status': status,
        }
        all_trades.extend(trades)

    # Options overlay
    options_overlay_analysis(all_trades, close_df, features)

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Strategy':<25} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MDD':>7} "
          f"{'Trades':>7} {'RegGap':>7} {'Perm-p':>7} {'Gates':>7}")
    print("-"*95)

    for name, res in all_results.items():
        m = res['metrics']
        print(f"{name:<25} {m['sharpe']:>+7.3f} {m['sortino']:>+8.3f} {m['win_rate']:>5.1%} "
              f"{m['profit_factor']:>6.2f} {m['max_dd']:>6.1%} {m['n_trades']:>7} "
              f"{m.get('regime_gap', float('nan')):>7.3f} {m.get('perm_p_value', float('nan')):>7.4f} "
              f"{res['status'].split(' ')[0]:>7}")

    print(f"\n{'SPY (benchmark)':<25} {spy_metrics['sharpe']:>+7.3f} {spy_metrics['sortino']:>+8.3f} "
          f"{spy_metrics['win_rate']:>5.1%} {spy_metrics['profit_factor']:>6.2f} "
          f"{spy_metrics['max_dd']:>6.1%}")

    # Viable strategies
    viable = [n for n, r in all_results.items() if all(r['gates'].values())]
    print(f"\n{'='*80}")
    if viable:
        print(f"VIABLE STRATEGIES (passed all 5 gates): {', '.join(viable)}")
    else:
        partially = [(n, sum(r['gates'].values())) for n, r in all_results.items()]
        partially.sort(key=lambda x: -x[1])
        print("NO strategy passed all 5 gates.")
        print(f"Best partial: {partially[0][0]} ({partially[0][1]}/5)")
        print("\nACTIONABLE FINDINGS:")
        for name, score in partially[:3]:
            m = all_results[name]['metrics']
            g = all_results[name]['gates']
            fails = [k for k, v in g.items() if not v]
            print(f"  {name} ({score}/5): failing {', '.join(fails)}")

    # Save results
    output_path = '/home/jupiter/Lvl3Quant/strategies/rotation_flow_results.json'
    save_results = {}
    for name, res in all_results.items():
        save_results[name] = {
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in res['metrics'].items()},
            'gates': {k: bool(v) for k, v in res['gates'].items()},
            'status': res['status'],
        }
    save_results['spy_benchmark'] = {
        'sharpe': float(spy_metrics['sharpe']),
        'sortino': float(spy_metrics['sortino']),
        'max_dd': float(spy_metrics['max_dd']),
        'total_return': float(spy_metrics['total_return']),
    }
    save_results['backtest_config'] = {
        'start': START_DATE,
        'end': END_DATE,
        'sectors': SECTOR_ETFS,
        'top_n': TOP_N,
        'train_months': TRAIN_MONTHS,
        'n_permutations': N_PERMUTATIONS,
        'option_spread_cost': OPTION_SPREAD_COST,
    }

    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
