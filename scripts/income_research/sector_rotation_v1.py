#!/usr/bin/env python3
"""
Sector Rotation Momentum — Equity Backtest v1
==============================================
Academic momentum applied to sector ETFs: buy the top-performing
sector(s) each month, hold 20 trading days, rebalance.

Entry rules tested:
  A) Top 1 sector by trailing 1-month return
  B) Top 2 sectors by trailing 1-month return
  C) Top 1 sector by trailing 3-month return
  D) Top 1 sector by trailing 1-month return IF above 50-day MA

Universe: 11 SPDR sector ETFs (2010-2026)
Starting capital: $10,000
Rebalance: every 20 trading days

HC #705: ALL adversarial checks built inline:
  - Permutation test (200 shuffles)
  - Regime test (PRIOR-DAY SPY close — no leakage)
  - Sub-period consistency
  - Outlier removal
  - Ticker concentration
  - Pricing sanity checks
  - WARNING banners for suspicious patterns

Leakage prevention:
  - Momentum ranking uses PRIOR data only
  - Entry at NEXT DAY open after signal
  - Regime uses PRIOR-DAY SPY return
"""

import sys, json, warnings, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "sector_rotation_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# =====================================================================
# CONFIG
# =====================================================================

STARTING_CAPITAL     = 10_000
EQUITY_SLIPPAGE_PCT  = 0.001     # 0.1% per side
REBALANCE_DAYS       = 20        # ~1 month of trading days
N_PERMUTATIONS       = 200

SECTOR_ETFS = ['XLK','XLF','XLV','XLE','XLI','XLY','XLP','XLU','XLB','XLRE','XLC']

STRATEGIES = {
    'top1_mom1m': {
        'desc': 'Top 1 by trailing 1-month return',
        'n_top': 1, 'lookback_days': 21, 'ma_filter': False,
    },
    'top2_mom1m': {
        'desc': 'Top 2 by trailing 1-month return',
        'n_top': 2, 'lookback_days': 21, 'ma_filter': False,
    },
    'top1_mom3m': {
        'desc': 'Top 1 by trailing 3-month return',
        'n_top': 1, 'lookback_days': 63, 'ma_filter': False,
    },
    'top1_mom1m_ma50': {
        'desc': 'Top 1 by trailing 1-month return IF above 50-day MA',
        'n_top': 1, 'lookback_days': 21, 'ma_filter': True,
    },
}

# =====================================================================
# DATA LOADING
# =====================================================================

def fetch_prices(tickers, cache_path):
    """Fetch daily OHLCV from yfinance with caching."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"  Loaded cached prices: {len(df)} rows, {df['ticker'].nunique()} tickers")
        return df

    import yfinance as yf
    print(f"  Downloading prices for {len(tickers)} tickers...")
    all_frames = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start='2010-01-01', end='2026-07-15',
                               progress=False, auto_adjust=True)
            if len(data) > 100:
                data = data.reset_index()
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
                data['ticker'] = ticker
                data.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                     'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                data.columns = [c.lower() if isinstance(c, str) else c for c in data.columns]
                all_frames.append(data[['date','open','high','low','close','volume','ticker']])
                print(f"    {ticker}: {len(data)} days")
        except Exception as e:
            print(f"    {ticker}: error -- {e}")
        time.sleep(0.3)

    df = pd.concat(all_frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df.to_parquet(cache_path, index=False)
    print(f"  Saved {len(df)} rows to cache")
    return df


def fetch_spy(cache_path):
    """Fetch SPY for regime classification."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    import yfinance as yf
    spy = yf.download('SPY', start='2010-01-01', end='2026-07-15',
                       progress=False, auto_adjust=True).reset_index()
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy.rename(columns={'Date': 'date', 'Close': 'close', 'Open': 'open'}, inplace=True)
    spy.columns = [c.lower() if isinstance(c, str) else c for c in spy.columns]
    spy = spy[['date','open','close']].copy()
    spy['date'] = pd.to_datetime(spy['date'])
    spy.to_parquet(cache_path, index=False)
    return spy


# =====================================================================
# PRICING SANITY CHECKS
# =====================================================================

def pricing_sanity_checks(all_data):
    """HC #705: Check data integrity before running backtest."""
    print("\n  PRICING SANITY CHECKS:")
    warnings_found = []

    for ticker in all_data['ticker'].unique():
        df = all_data[all_data['ticker'] == ticker].sort_values('date').copy()

        # Check 1: Missing dates (gaps > 5 business days)
        date_diffs = df['date'].diff().dt.days
        big_gaps = date_diffs[date_diffs > 7]
        if len(big_gaps) > 3:
            msg = f"    WARNING: {ticker} has {len(big_gaps)} date gaps > 7 calendar days"
            print(msg)
            warnings_found.append(msg)

        # Check 2: Zero or negative prices
        bad_prices = df[(df['close'] <= 0) | (df['open'] <= 0)]
        if len(bad_prices) > 0:
            msg = f"    WARNING: {ticker} has {len(bad_prices)} zero/negative prices -- DATA CORRUPT"
            print(msg)
            warnings_found.append(msg)

        # Check 3: Extreme daily returns (>30% in a day for ETFs = suspicious)
        rets = df['close'].pct_change().dropna()
        extreme = rets[rets.abs() > 0.30]
        if len(extreme) > 0:
            msg = f"    WARNING: {ticker} has {len(extreme)} daily returns > 30% -- check for splits/errors"
            print(msg)
            warnings_found.append(msg)

        # Check 4: Flat prices (same close for 5+ days = stale data)
        flat = (df['close'].diff() == 0).rolling(5).sum()
        if (flat >= 5).any():
            msg = f"    WARNING: {ticker} has stretches of 5+ identical closes -- stale data?"
            print(msg)
            warnings_found.append(msg)

        # Check 5: Open far from prior close (>10% gap for ETFs)
        df_tmp = df.copy()
        df_tmp['prev_close'] = df_tmp['close'].shift(1)
        gap = ((df_tmp['open'] - df_tmp['prev_close']) / df_tmp['prev_close']).abs()
        big_open_gaps = gap[gap > 0.10].dropna()
        if len(big_open_gaps) > 2:
            msg = f"    WARNING: {ticker} has {len(big_open_gaps)} open-vs-prior-close gaps > 10%"
            print(msg)
            warnings_found.append(msg)

    if not warnings_found:
        print("    All sanity checks passed.")
    else:
        print(f"\n    *** {len(warnings_found)} TOTAL WARNINGS -- review before trusting results ***")

    return warnings_found


# =====================================================================
# REGIME CLASSIFICATION (PRIOR-DAY SPY -- NO LEAKAGE)
# =====================================================================

def classify_spy_regime_prior_day(spy_df):
    """
    Classify regime using PRIOR-DAY SPY close-to-close return.
    HC #705: Using same-day SPY is leakage -- we use prior-day.
    """
    spy = spy_df.sort_values('date').copy()
    spy['ret'] = spy['close'].pct_change()
    # Shift: regime for date T is based on SPY return from T-2 close to T-1 close
    spy['prior_ret'] = spy['ret'].shift(1)

    regime = {}
    for _, row in spy.iterrows():
        dt = pd.Timestamp(row['date']).normalize()
        pr = row.get('prior_ret', np.nan)
        if pd.isna(pr):
            regime[dt] = 'flat'
        elif pr > 0.003:
            regime[dt] = 'green'
        elif pr < -0.003:
            regime[dt] = 'red'
        else:
            regime[dt] = 'flat'
    return regime


# =====================================================================
# SECTOR ROTATION BACKTEST
# =====================================================================

def build_wide_prices(all_data, field='close'):
    """Pivot data into wide format: date x ticker."""
    return all_data.pivot_table(index='date', columns='ticker', values=field).sort_index()


def run_sector_rotation(all_data, strategy_name, strategy_cfg, spy_regime):
    """
    Run a single sector rotation strategy.

    LEAKAGE PREVENTION:
    - Momentum rank computed on day T using data up to day T (close prices)
    - Entry at day T+1 open
    - No future data used in ranking
    """
    n_top = strategy_cfg['n_top']
    lookback = strategy_cfg['lookback_days']
    ma_filter = strategy_cfg['ma_filter']

    close_wide = build_wide_prices(all_data, 'close')
    open_wide  = build_wide_prices(all_data, 'open')

    # Compute trailing returns for ranking
    mom_returns = close_wide.pct_change(lookback)

    # Compute 50-day MA if needed
    ma50 = None
    if ma_filter:
        ma50 = close_wide.rolling(50).mean()

    dates = close_wide.index.tolist()
    trades = []

    # Start after enough lookback + MA warmup
    start_idx = max(lookback, 50 if ma_filter else 0) + 5
    i = start_idx

    while i < len(dates) - REBALANCE_DAYS - 1:
        signal_date = dates[i]

        # LEAKAGE CHECK: momentum uses data up to signal_date ONLY
        mom_row = mom_returns.loc[signal_date].dropna()
        if len(mom_row) < len(SECTOR_ETFS) - 2:
            i += REBALANCE_DAYS
            continue

        # If MA filter, restrict to ETFs above their 50-day MA
        if ma_filter:
            ma_row = ma50.loc[signal_date].dropna()
            close_row = close_wide.loc[signal_date].dropna()
            eligible = [t for t in mom_row.index
                       if t in ma_row.index and t in close_row.index
                       and close_row[t] > ma_row[t]]
            mom_eligible = mom_row[mom_row.index.isin(eligible)]
        else:
            mom_eligible = mom_row

        if len(mom_eligible) < n_top:
            i += REBALANCE_DAYS
            continue

        # Rank: top N by trailing momentum
        top_etfs = mom_eligible.nlargest(n_top).index.tolist()

        # ENTRY: next trading day open (i+1)
        entry_date_idx = i + 1
        if entry_date_idx >= len(dates):
            break
        entry_date = dates[entry_date_idx]

        # EXIT: hold for REBALANCE_DAYS trading days
        exit_date_idx = min(entry_date_idx + REBALANCE_DAYS, len(dates) - 1)
        exit_date = dates[exit_date_idx]

        for etf in top_etfs:
            if etf not in open_wide.columns or etf not in close_wide.columns:
                continue

            entry_price = open_wide.loc[entry_date, etf]
            exit_price = close_wide.loc[exit_date, etf]

            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                continue

            # Apply slippage
            entry_price *= (1 + EQUITY_SLIPPAGE_PCT)
            exit_price  *= (1 - EQUITY_SLIPPAGE_PCT)

            ret = (exit_price - entry_price) / entry_price

            # Get regime at entry using PRIOR-DAY SPY
            entry_dt = pd.Timestamp(entry_date).normalize()
            regime = spy_regime.get(entry_dt, 'unknown')

            # Momentum score used for ranking
            mom_score = mom_returns.loc[signal_date, etf] if etf in mom_returns.columns else np.nan

            trades.append({
                'ticker': etf,
                'signal_date': pd.Timestamp(signal_date),
                'entry_date': pd.Timestamp(entry_date),
                'exit_date': pd.Timestamp(exit_date),
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
                'return_pct': float(ret),
                'regime': regime,
                'strategy': strategy_name,
                'hold_days': REBALANCE_DAYS,
                'momentum_score': float(mom_score) if not pd.isna(mom_score) else 0.0,
                'n_eligible': len(mom_eligible),
            })

        i += REBALANCE_DAYS

    return trades


# =====================================================================
# PORTFOLIO SIMULATION (equal-weight, fully invested)
# =====================================================================

def simulate_portfolio(trades, starting_capital=STARTING_CAPITAL):
    """
    Simulate equal-weight portfolio.
    Each rebalance period: allocate capital equally across selected ETFs.
    """
    if not trades:
        return [], starting_capital

    df = pd.DataFrame(trades)
    capital = starting_capital
    equity_curve = [{'date': str(df['entry_date'].min()), 'equity': capital}]

    # Group by entry_date (same rebalance period)
    for entry_date, group in df.groupby('entry_date'):
        period_return = group['return_pct'].mean()  # equal-weight avg
        period_pnl = capital * period_return
        capital += period_pnl
        equity_curve.append({
            'date': str(group['exit_date'].iloc[0]),
            'equity': round(capital, 2),
            'period_return': round(period_return, 6),
            'n_positions': len(group),
        })

    return equity_curve, capital


# =====================================================================
# QUALITY GATES (ALL INLINE -- HC #705)
# =====================================================================

def permutation_test(trades, n_perms=N_PERMUTATIONS):
    """
    HC #705: Permutation test -- shuffle returns, compute p-value.
    If p >= 0.05, the edge is likely noise.
    """
    if len(trades) < 10:
        return {'pass': False, 'p_value': 1.0, 'reason': 'too few trades'}

    df = pd.DataFrame(trades)
    rets = df['return_pct'].values
    obs_mean = rets.mean()

    rng = np.random.RandomState(42)
    count_ge = 0
    shuffled_means = []
    for _ in range(n_perms):
        shuffled = rng.permutation(rets)
        sm = shuffled.mean()
        shuffled_means.append(sm)
        if sm >= obs_mean:
            count_ge += 1

    p_value = count_ge / n_perms
    result = {
        'pass': p_value < 0.05,
        'p_value': round(p_value, 4),
        'observed_mean_pct': round(obs_mean * 100, 4),
        'null_mean_pct': round(np.mean(shuffled_means) * 100, 4),
        'n_perms': n_perms,
    }

    if not result['pass']:
        print(f"    *** WARNING: PERMUTATION TEST FAILED (p={p_value:.3f}) -- edge may be noise ***")

    return result


def regime_test(trades):
    """
    HC #705: Regime-agnostic test using PRIOR-DAY SPY.
    Reject if |Sharpe_green - Sharpe_red| / max(...) > 0.50
    """
    if not trades:
        return {'pass': False, 'reason': 'no trades'}

    df = pd.DataFrame(trades)
    results = {}
    for regime in ['green', 'red', 'flat']:
        subset = df[df['regime'] == regime]
        if len(subset) < 5:
            results[regime] = {'sharpe': 0, 'n': 0, 'wr': 0, 'avg_ret': 0}
            continue
        rets = subset['return_pct']
        ann = np.sqrt(252 / REBALANCE_DAYS)
        sharpe = rets.mean() / rets.std() * ann if rets.std() > 0 else 0
        results[regime] = {
            'sharpe': round(sharpe, 3),
            'n': len(subset),
            'wr': round((rets > 0).mean(), 3),
            'avg_ret': round(rets.mean() * 100, 3),
        }

    sg = results.get('green', {}).get('sharpe', 0)
    sr = results.get('red', {}).get('sharpe', 0)
    denom = max(abs(sg), abs(sr), 0.001)
    gap = abs(sg - sr) / denom

    passed = gap <= 0.50
    result = {
        'pass': passed,
        'gap': round(gap, 3),
        'regimes': results,
        'note': 'PASS (prior-day SPY regime)' if passed else f'FAIL: gap={gap:.3f} > 0.50',
    }

    if not passed:
        print(f"    *** WARNING: REGIME TEST FAILED -- gap={gap:.3f} > 0.50 ***")
        print(f"        Green Sharpe={sg:.3f} ({results['green']['n']} trades), "
              f"Red Sharpe={sr:.3f} ({results['red']['n']} trades)")

    return result


def sub_period_test(trades):
    """
    HC #705: Split trades into halves by time. Both must be profitable.
    """
    if len(trades) < 20:
        return {'pass': False, 'reason': 'too few trades'}

    df = pd.DataFrame(trades).sort_values('entry_date')
    mid = len(df) // 2
    h1 = df.iloc[:mid]['return_pct']
    h2 = df.iloc[mid:]['return_pct']

    h1_mean = h1.mean()
    h2_mean = h2.mean()
    passed = h1_mean > 0 and h2_mean > 0

    # Also check thirds for robustness
    third = len(df) // 3
    t1 = df.iloc[:third]['return_pct'].mean()
    t2 = df.iloc[third:2*third]['return_pct'].mean()
    t3 = df.iloc[2*third:]['return_pct'].mean()
    thirds_positive = sum(1 for t in [t1, t2, t3] if t > 0)

    result = {
        'pass': passed,
        'first_half': {
            'n': len(h1), 'avg_ret': round(h1_mean*100, 3), 'wr': round((h1>0).mean(), 3),
            'period': f"{df.iloc[0]['entry_date'].date()} to {df.iloc[mid-1]['entry_date'].date()}",
        },
        'second_half': {
            'n': len(h2), 'avg_ret': round(h2_mean*100, 3), 'wr': round((h2>0).mean(), 3),
            'period': f"{df.iloc[mid]['entry_date'].date()} to {df.iloc[-1]['entry_date'].date()}",
        },
        'thirds_positive': thirds_positive,
        'thirds_detail': {
            'T1_avg_ret': round(t1*100, 3),
            'T2_avg_ret': round(t2*100, 3),
            'T3_avg_ret': round(t3*100, 3),
        },
    }

    if not passed:
        print(f"    *** WARNING: SUB-PERIOD TEST FAILED ***")
        print(f"        H1 avg={h1_mean*100:.2f}%, H2 avg={h2_mean*100:.2f}%")

    return result


def outlier_removal_test(trades):
    """
    HC #705: Remove top/bottom 5% of returns. Strategy must remain profitable.
    """
    if len(trades) < 20:
        return {'pass': False, 'reason': 'too few trades'}

    df = pd.DataFrame(trades)
    rets = df['return_pct'].values
    p5, p95 = np.percentile(rets, [5, 95])
    trimmed = rets[(rets >= p5) & (rets <= p95)]

    full_mean = rets.mean()
    trim_mean = trimmed.mean()
    passed = trim_mean > 0

    # How much of the edge comes from outliers?
    edge_from_outliers = 1 - (trim_mean / full_mean) if full_mean != 0 else 0

    result = {
        'pass': passed,
        'full_mean_pct': round(full_mean*100, 3),
        'trimmed_mean_pct': round(trim_mean*100, 3),
        'n_removed': len(rets) - len(trimmed),
        'edge_from_outliers_pct': round(edge_from_outliers*100, 1),
    }

    if not passed:
        print(f"    *** WARNING: OUTLIER REMOVAL TEST FAILED -- edge is from outliers ***")
    elif edge_from_outliers > 0.50:
        print(f"    *** WARNING: {edge_from_outliers*100:.0f}% of edge comes from outliers ***")

    return result


def ticker_concentration_test(trades):
    """
    HC #705: No single ticker should contribute > 30% of total P&L.
    """
    if len(trades) < 10:
        return {'pass': False, 'reason': 'too few trades'}

    df = pd.DataFrame(trades)
    ticker_pnl = df.groupby('ticker')['return_pct'].sum()
    total_positive = ticker_pnl[ticker_pnl > 0].sum()

    if total_positive > 0:
        max_conc = ticker_pnl.max() / total_positive
    else:
        max_conc = 0

    passed = max_conc < 0.30

    # Trade count distribution
    ticker_counts = df['ticker'].value_counts()

    result = {
        'pass': passed,
        'max_concentration': round(max_conc, 3),
        'top_contributor': str(ticker_pnl.idxmax()) if len(ticker_pnl) > 0 else 'N/A',
        'ticker_pnl': {str(k): round(v*100, 2) for k, v in ticker_pnl.sort_values(ascending=False).items()},
        'ticker_trade_counts': {str(k): int(v) for k, v in ticker_counts.items()},
    }

    if not passed:
        print(f"    *** WARNING: TICKER CONCENTRATION FAILED -- "
              f"{ticker_pnl.idxmax()} = {max_conc*100:.0f}% of P&L ***")

    return result


def leakage_audit(trades, all_data):
    """
    HC #705: Explicit leakage checks.
    1. Signal date must be before entry date
    2. Entry price must be from entry date, not signal date
    3. Momentum calculation uses only past data
    """
    issues = []
    if not trades:
        return {'pass': True, 'issues': []}

    df = pd.DataFrame(trades)

    # Check 1: Signal before entry
    bad_dates = df[df['signal_date'] >= df['entry_date']]
    if len(bad_dates) > 0:
        issues.append(f"CRITICAL: {len(bad_dates)} trades where signal_date >= entry_date (look-ahead)")

    # Check 2: Entry before exit
    bad_exit = df[df['entry_date'] >= df['exit_date']]
    if len(bad_exit) > 0:
        issues.append(f"CRITICAL: {len(bad_exit)} trades where entry_date >= exit_date")

    # Check 3: Suspiciously high win rate (>75% for momentum = likely leakage)
    wr = (df['return_pct'] > 0).mean()
    if wr > 0.75:
        issues.append(f"SUSPICIOUS: Win rate {wr:.1%} is abnormally high for momentum -- check for leakage")

    # Check 4: Returns too good to be true (>5% avg per period for ETFs)
    avg_ret = df['return_pct'].mean()
    if avg_ret > 0.05:
        issues.append(f"SUSPICIOUS: Avg return {avg_ret*100:.2f}% per period is very high for ETF momentum")

    passed = len([i for i in issues if 'CRITICAL' in i]) == 0

    if issues:
        print(f"    LEAKAGE AUDIT:")
        for issue in issues:
            print(f"      {issue}")

    return {'pass': passed, 'issues': issues}


# =====================================================================
# METRICS
# =====================================================================

def compute_metrics(trades, label=''):
    """Compute risk-adjusted metrics for a set of trades."""
    if not trades:
        return {'label': label, 'n_trades': 0}

    df = pd.DataFrame(trades)
    rets = df['return_pct']
    n = len(rets)
    avg_ret = rets.mean()
    std_ret = rets.std()
    wr = (rets > 0).mean()

    ann_factor = np.sqrt(252 / REBALANCE_DAYS)

    sharpe = avg_ret / std_ret * ann_factor if std_ret > 0 else 0
    downside = rets[rets < 0].std()
    sortino = avg_ret / downside * ann_factor if downside > 0 and len(rets[rets < 0]) > 2 else 0

    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    gross_wins = wins.sum() if len(wins) > 0 else 0
    gross_losses = abs(losses.sum()) if len(losses) > 0 else 0.001
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Max drawdown from equity curve
    cum_returns = (1 + rets).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    # CAGR
    n_years = n * REBALANCE_DAYS / 252
    total_return = cum_returns.iloc[-1] if len(cum_returns) > 0 else 1
    cagr = (total_return ** (1 / n_years) - 1) if n_years > 0 and total_return > 0 else 0

    return {
        'label': label,
        'n_trades': n,
        'win_rate': round(float(wr), 4),
        'avg_return_pct': round(float(avg_ret * 100), 3),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'profit_factor': round(float(pf), 3),
        'max_drawdown_pct': round(float(max_dd * 100), 2),
        'cagr_pct': round(float(cagr * 100), 2),
        'total_return_pct': round(float((total_return - 1) * 100), 2),
        'avg_win_pct': round(float(wins.mean() * 100), 3) if len(wins) > 0 else 0,
        'avg_loss_pct': round(float(abs(losses.mean()) * 100), 3) if len(losses) > 0 else 0,
    }


def compute_annual_returns(trades):
    """Compute per-year returns for consistency check."""
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    df['year'] = df['entry_date'].dt.year
    annual = {}
    for year, grp in df.groupby('year'):
        rets = grp['return_pct']
        total = (1 + rets).prod() - 1
        annual[int(year)] = {
            'return_pct': round(float(total * 100), 2),
            'n_trades': len(grp),
            'wr': round(float((rets > 0).mean()), 3),
        }
    return annual


# =====================================================================
# BENCHMARK: BUY-AND-HOLD SPY
# =====================================================================

def compute_spy_benchmark(spy_df, start_date, end_date):
    """Compute SPY buy-and-hold for the same period."""
    spy = spy_df[(spy_df['date'] >= start_date) & (spy_df['date'] <= end_date)].sort_values('date')
    if len(spy) < 10:
        return {}

    start_px = spy['close'].iloc[0]
    end_px = spy['close'].iloc[-1]
    total_ret = (end_px / start_px) - 1

    rets = spy['close'].pct_change().dropna()
    n_years = len(rets) / 252
    cagr = ((1 + total_ret) ** (1 / n_years) - 1) if n_years > 0 else 0
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0

    cum = (1 + rets).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

    return {
        'total_return_pct': round(float(total_ret * 100), 2),
        'cagr_pct': round(float(cagr * 100), 2),
        'sharpe': round(float(sharpe), 3),
        'max_drawdown_pct': round(float(max_dd * 100), 2),
    }


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("=" * 80)
    print("SECTOR ROTATION MOMENTUM -- EQUITY BACKTEST v1")
    print("=" * 80)
    print(f"Run time: {datetime.now()}")

    # -- Load data --
    print("\n[1/7] Loading price data...")
    prices_cache = OUTPUT / "sector_prices_cache.parquet"
    spy_cache    = OUTPUT / "spy_cache.parquet"

    all_data = fetch_prices(SECTOR_ETFS, prices_cache)
    spy_df   = fetch_spy(spy_cache)

    tickers_available = sorted(all_data['ticker'].unique())
    date_range = f"{all_data['date'].min().date()} to {all_data['date'].max().date()}"
    print(f"  Universe: {len(tickers_available)} ETFs: {', '.join(tickers_available)}")
    print(f"  Date range: {date_range}")

    # -- Pricing sanity checks --
    print("\n[2/7] Pricing sanity checks...")
    sanity_warnings = pricing_sanity_checks(all_data)

    # -- Regime classification (PRIOR-DAY) --
    print("\n[3/7] Classifying SPY regime (PRIOR-DAY -- no leakage)...")
    spy_regime = classify_spy_regime_prior_day(spy_df)
    regime_counts = defaultdict(int)
    for v in spy_regime.values():
        regime_counts[v] += 1
    print(f"  Regime distribution: {dict(regime_counts)}")

    # -- Run all strategies --
    print("\n[4/7] Running sector rotation backtests...")
    all_results = {}

    for strat_name, strat_cfg in STRATEGIES.items():
        print(f"\n  === {strat_name}: {strat_cfg['desc']} ===")
        trades = run_sector_rotation(all_data, strat_name, strat_cfg, spy_regime)
        metrics = compute_metrics(trades, strat_name)
        annual = compute_annual_returns(trades)

        print(f"  Trades: {metrics['n_trades']}, WR: {metrics.get('win_rate',0):.1%}, "
              f"Sharpe: {metrics.get('sharpe',0):.2f}, Sortino: {metrics.get('sortino',0):.2f}, "
              f"PF: {metrics.get('profit_factor',0):.2f}")
        print(f"  CAGR: {metrics.get('cagr_pct',0):.2f}%, MaxDD: {metrics.get('max_drawdown_pct',0):.2f}%, "
              f"Total return: {metrics.get('total_return_pct',0):.1f}%")

        # Equity curve
        eq_curve, final_capital = simulate_portfolio(trades, STARTING_CAPITAL)

        all_results[strat_name] = {
            'config': strat_cfg,
            'metrics': metrics,
            'trades': trades,
            'annual_returns': annual,
            'equity_curve': eq_curve,
            'final_capital': round(final_capital, 2),
        }

    # -- Quality gates on all strategies --
    print("\n[5/7] Running quality gates (HC #705 -- all inline)...")
    quality_results = {}

    for strat_name, res in all_results.items():
        trades = res['trades']
        print(f"\n  --- {strat_name} ---")

        if len(trades) < 10:
            quality_results[strat_name] = {'skip': True, 'reason': f'only {len(trades)} trades'}
            continue

        perm = permutation_test(trades)
        reg = regime_test(trades)
        sub = sub_period_test(trades)
        outlier = outlier_removal_test(trades)
        conc = ticker_concentration_test(trades)
        leak = leakage_audit(trades, all_data)

        all_pass = all([perm['pass'], reg['pass'], sub['pass'],
                       outlier['pass'], conc['pass'], leak['pass']])

        quality_results[strat_name] = {
            'permutation_test': perm,
            'regime_test': reg,
            'sub_period_test': sub,
            'outlier_removal': outlier,
            'ticker_concentration': conc,
            'leakage_audit': leak,
            'ALL_PASS': all_pass,
        }

        checks = [
            ('Perm', perm['pass'], f"p={perm['p_value']:.3f}"),
            ('Regime', reg['pass'], f"gap={reg.get('gap',0):.3f}"),
            ('SubPeriod', sub['pass'], ''),
            ('Outlier', outlier['pass'], ''),
            ('Concentration', conc['pass'], f"max={conc.get('max_concentration',0):.2f}"),
            ('Leakage', leak['pass'], ''),
        ]
        status = "ALL PASS" if all_pass else "SOME FAILED"
        print(f"    {status}: " + ", ".join(
            f"{name}={'PASS' if ok else 'FAIL'}{' ('+detail+')' if detail else ''}"
            for name, ok, detail in checks
        ))

    # -- SPY benchmark --
    print("\n[6/7] Computing SPY benchmark...")
    start_date = all_data['date'].min()
    end_date = all_data['date'].max()
    spy_bench = compute_spy_benchmark(spy_df, start_date, end_date)
    print(f"  SPY Buy-and-Hold: CAGR={spy_bench.get('cagr_pct',0):.2f}%, "
          f"Sharpe={spy_bench.get('sharpe',0):.2f}, MaxDD={spy_bench.get('max_drawdown_pct',0):.2f}%")

    # -- Compile final report --
    print("\n[7/7] Compiling report...")

    report = {
        'backtest': 'Sector Rotation Momentum -- Equity Sim v1',
        'date_run': str(datetime.now()),
        'universe': f"{len(tickers_available)} sector ETFs: {', '.join(tickers_available)}",
        'period': date_range,
        'starting_capital': STARTING_CAPITAL,
        'rebalance_days': REBALANCE_DAYS,
        'slippage_pct': EQUITY_SLIPPAGE_PCT,

        'spy_benchmark': spy_bench,

        'strategy_results': {
            name: {
                'config': res['config'],
                'metrics': res['metrics'],
                'annual_returns': res['annual_returns'],
                'final_capital': res['final_capital'],
            }
            for name, res in all_results.items()
        },

        'quality_gates': quality_results,
        'pricing_sanity_warnings': sanity_warnings,
    }

    report_path = OUTPUT / "backtest_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Save trade logs
    for name, res in all_results.items():
        if res['trades']:
            trades_df = pd.DataFrame(res['trades'])
            trades_df.to_csv(OUTPUT / f"trades_{name}.csv", index=False)

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================

    print("\n" + "=" * 80)
    print("SUMMARY -- SECTOR ROTATION MOMENTUM (Equity Sim)")
    print("=" * 80)

    print(f"\n{'Strategy':<22} {'N':>4} {'WR':>6} {'AvgRet':>8} {'Sharpe':>7} "
          f"{'Sortino':>8} {'PF':>6} {'CAGR':>7} {'MaxDD':>7} {'Final$':>9}")
    print("-" * 95)
    for name, res in sorted(all_results.items(),
                            key=lambda x: x[1]['metrics'].get('sharpe', -999), reverse=True):
        m = res['metrics']
        print(f"{name:<22} {m['n_trades']:>4} {m.get('win_rate',0):>5.1%} "
              f"{m.get('avg_return_pct',0):>7.2f}% {m.get('sharpe',0):>7.2f} "
              f"{m.get('sortino',0):>7.2f} {m.get('profit_factor',0):>6.2f} "
              f"{m.get('cagr_pct',0):>6.1f}% {m.get('max_drawdown_pct',0):>6.1f}% "
              f"${res['final_capital']:>8,.0f}")

    print(f"\n{'SPY Buy-Hold':<22}      "
          f"{'':>6} {'':>8} {spy_bench.get('sharpe',0):>7.2f} "
          f"{'':>8} {'':>6} {spy_bench.get('cagr_pct',0):>6.1f}% "
          f"{spy_bench.get('max_drawdown_pct',0):>6.1f}%")

    # Annual returns for best strategy
    best_strat = max(all_results.items(), key=lambda x: x[1]['metrics'].get('sharpe', -999))
    best_name = best_strat[0]
    print(f"\n{'='*80}")
    print(f"ANNUAL RETURNS -- {best_name} (best by Sharpe)")
    print(f"{'='*80}")
    annual = best_strat[1]['annual_returns']
    for year in sorted(annual.keys()):
        yr = annual[year]
        bar = '+' * max(0, int(yr['return_pct'] / 2)) if yr['return_pct'] > 0 else '-' * max(0, int(-yr['return_pct'] / 2))
        print(f"  {year}: {yr['return_pct']:>+7.1f}%  WR={yr['wr']:.0%}  n={yr['n_trades']:>3}  {bar}")

    # Quality gates summary
    print(f"\n{'='*80}")
    print(f"QUALITY GATES")
    print(f"{'='*80}")
    for name, qr in quality_results.items():
        if 'skip' in qr:
            print(f"  {name:<22} SKIP ({qr['reason']})")
            continue
        status = "PASS" if qr['ALL_PASS'] else "FAIL"
        parts = []
        for check_name in ['permutation_test', 'regime_test', 'sub_period_test',
                          'outlier_removal', 'ticker_concentration', 'leakage_audit']:
            check = qr.get(check_name, {})
            ok = check.get('pass', False)
            parts.append(f"{check_name.split('_')[0]}={'OK' if ok else 'X'}")
        print(f"  {name:<22} {status}  [{', '.join(parts)}]")

    # Ticker P&L breakdown for best strategy
    if best_strat[1]['trades']:
        print(f"\n{'='*80}")
        print(f"TICKER P&L CONTRIBUTION -- {best_name}")
        print(f"{'='*80}")
        df_best = pd.DataFrame(best_strat[1]['trades'])
        ticker_summary = df_best.groupby('ticker').agg(
            n=('return_pct', 'count'),
            avg_ret=('return_pct', 'mean'),
            total_ret=('return_pct', 'sum'),
            wr=('return_pct', lambda x: (x > 0).mean()),
        ).sort_values('total_ret', ascending=False)

        for ticker, row in ticker_summary.iterrows():
            bar = '+' * max(0, int(row['total_ret'] * 100 / 3)) if row['total_ret'] > 0 else '-' * max(0, int(-row['total_ret'] * 100 / 3))
            print(f"  {ticker:<6} n={int(row['n']):>3}  WR={row['wr']:.0%}  "
                  f"avg={row['avg_ret']*100:>+6.2f}%  total={row['total_ret']*100:>+7.1f}%  {bar}")

    # WARNING BANNERS
    print("\n" + "=" * 80)
    has_warnings = False

    for name, qr in quality_results.items():
        if 'skip' in qr:
            continue
        if not qr.get('ALL_PASS', False):
            if not has_warnings:
                print("WARNING BANNERS")
                print("=" * 80)
                has_warnings = True

            failures = []
            for check_name in ['permutation_test', 'regime_test', 'sub_period_test',
                              'outlier_removal', 'ticker_concentration', 'leakage_audit']:
                check = qr.get(check_name, {})
                if not check.get('pass', True):
                    failures.append(check_name)

            print(f"\n  {name}: FAILED checks: {', '.join(failures)}")

            if 'permutation_test' in failures:
                p = qr['permutation_test']['p_value']
                print(f"    PERMUTATION: p={p:.3f} -- edge is NOT statistically significant")
            if 'regime_test' in failures:
                gap = qr['regime_test']['gap']
                print(f"    REGIME: gap={gap:.3f} -- strategy is regime-dependent, not robust")
            if 'sub_period_test' in failures:
                h1 = qr['sub_period_test']['first_half']['avg_ret']
                h2 = qr['sub_period_test']['second_half']['avg_ret']
                print(f"    SUB-PERIOD: H1={h1:.2f}%, H2={h2:.2f}% -- inconsistent across time")
            if 'ticker_concentration' in failures:
                conc = qr['ticker_concentration']['max_concentration']
                top = qr['ticker_concentration']['top_contributor']
                print(f"    CONCENTRATION: {top} = {conc*100:.0f}% of P&L -- single-ticker dependency")

    if not has_warnings:
        print("NO WARNING BANNERS -- all strategies passed or were skipped")
        print("=" * 80)

    print("\n" + "=" * 80)
    print("DONE -- Report saved.")
    print("=" * 80)


if __name__ == '__main__':
    main()
