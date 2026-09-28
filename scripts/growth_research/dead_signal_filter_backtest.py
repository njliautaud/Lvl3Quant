#!/usr/bin/env python3
"""
Dead Signal Filter Backtest (HC #772)
======================================
Tests dead signals as REJECTION FILTERS on our 3 strongest validated strategies.

Concept: Instead of requiring both signals to fire (confluence), require the
dead signal to NOT be bearish. Filter OUT entries where the dead signal is
actively negative.

Base Strategies (strongest validated):
  1. IV-RV Gap (#10): VIX > 20d RV by 5pts AND stock >5% below 20-SMA AND RSI<40
  2. Bond Yield (#9): ^TNX drops >0.1% over 5d AND stock >5% below 20-SMA
  3. RSI Divergence (#8): Price lower low vs 14d ago but RSI higher low AND volume declining

Dead-Signal Filters (binary "not bearish" check):
  A. VIX Term Structure: skip entry if VIX/VIX3M > 1.05 (backwardation = panic)
  B. Credit Stress: skip entry if HYG/LQD 5d change < -0.5%
  C. Breadth: skip entry if SPY RSI(14) < 25 (deeply oversold)
  D. Momentum Context: skip entry if SPY 20d return < -8%

12 combinations (3 strategies x 4 filters) + 3 solo baselines = 15 tests.
5-gate validation: Sharpe, WR, PF, MDD, regime gap < 0.50, perm p < 0.05.
Key metric: filter alpha = filtered Sharpe - solo Sharpe.
"""

import os, sys, json, warnings, time, functools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

# --- Config ---
START_DATE = '2019-01-01'
END_DATE = '2026-07-01'
BACKTEST_START = '2020-01-01'
POS_SIZE = 300.0
MAX_CONCURRENT = 2
HOLD_DAYS = 21
PROFIT_TARGET = 0.10
STOP_LOSS = -0.15
SPREAD_COST_PCT = 0.001  # 10bps RT
N_PERMS = 200
REGIME_GAP_LIMIT = 0.50

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/dead_signal_filter')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]

MACRO_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', '^TNX', 'HYG', 'LQD']

print("=" * 80)
print("DEAD SIGNAL FILTER BACKTEST (HC #772)")
print("=" * 80)

# =====================================================================
# 1. DATA DOWNLOAD
# =====================================================================
def download_data():
    import yfinance as yf
    cache_file = OUTPUT_DIR / '_dead_filter_cache.pkl'
    if cache_file.exists():
        import pickle
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded cache: {len(data['close'].columns)} stocks, {len(data['close'])} days")
        return data

    print(f"\n[1] Downloading {len(UNIVERSE)} stocks + {len(MACRO_TICKERS)} macro tickers...")
    all_tickers = UNIVERSE + MACRO_TICKERS

    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
        high = raw['High']
        low = raw['Low']
        volume = raw['Volume']
    else:
        close = high = low = volume = raw

    # Clean column names
    for df in [close, high, low, volume]:
        if hasattr(df.columns, 'droplevel'):
            try:
                df.columns = df.columns.droplevel(1)
            except:
                pass

    close = close.ffill().dropna(how='all')

    data = {
        'close': close,
        'high': high.reindex(close.index).ffill(),
        'low': low.reindex(close.index).ffill(),
        'volume': volume.reindex(close.index).ffill().fillna(0),
    }

    import pickle
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Downloaded: {len(close.columns)} tickers, {len(close)} days")
    return data


# =====================================================================
# 2. HELPER FUNCTIONS
# =====================================================================

def _rsi(series, period=14):
    """RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _realized_vol(returns, window=21):
    """Annualized realized vol."""
    return returns.rolling(window).std() * np.sqrt(252) * 100


# =====================================================================
# 3. BASE SIGNAL GENERATORS
# =====================================================================

def signal_iv_rv_gap(data, stock_tickers):
    """IV-RV Gap: VIX > 20d realized vol by 5pts AND stock >5% below 20-SMA AND RSI<40."""
    spy_close = data['close']['SPY']
    spy_ret = spy_close.pct_change()
    vix = data['close']['^VIX']
    rv_20 = _realized_vol(spy_ret, 20)
    gap = vix - rv_20

    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        sma20 = close.rolling(20).mean()
        rsi = _rsi(close, 14)
        drawdown_from_sma = (close - sma20) / sma20

        for i in range(20, len(close)):
            day = close.index[i]
            if day not in gap.index:
                continue
            try:
                g = gap.at[day]
            except:
                continue
            if pd.isna(g) or g <= 5:
                continue
            if drawdown_from_sma.iloc[i] > -0.05:
                continue
            if pd.isna(rsi.iloc[i]) or rsi.iloc[i] >= 40:
                continue
            signals[(day, t)] = True

    return signals, 'IV-RV Gap'


def signal_bond_yield(data, stock_tickers):
    """Bond Yield: ^TNX drops >0.1% over 5d AND stock >5% below 20-SMA."""
    if '^TNX' not in data['close'].columns:
        # Fallback to TLT proxy
        tlt = data['close']['TLT']
        yield_change = tlt.pct_change(5)
        fire_mask = yield_change > 0.02  # TLT up 2% ~ yields down ~0.1%
    else:
        tnx = data['close']['^TNX']
        yield_change = tnx.diff(5)
        fire_mask = yield_change < -0.10

    fire_days = set(fire_mask[fire_mask].index)

    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        sma20 = close.rolling(20).mean()
        drawdown_from_sma = (close - sma20) / sma20

        for i in range(20, len(close)):
            day = close.index[i]
            if day not in fire_days:
                continue
            if drawdown_from_sma.iloc[i] > -0.05:
                continue
            signals[(day, t)] = True

    return signals, 'Bond Yield'


def signal_rsi_divergence(data, stock_tickers):
    """RSI Divergence: Price lower low vs 14d ago but RSI higher low AND volume declining."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, 14)
        vol = data['volume'].get(t)
        if vol is None:
            continue
        vol = vol.reindex(close.index).fillna(0)

        for i in range(20, len(close)):
            if i < 14:
                continue
            # Price: current close < close 14 days ago (lower low)
            if close.iloc[i] >= close.iloc[i - 14]:
                continue
            # RSI: current RSI > RSI 14 days ago (higher low = divergence)
            r_now = rsi.iloc[i]
            r_prev = rsi.iloc[i - 14]
            if pd.isna(r_now) or pd.isna(r_prev):
                continue
            if r_now <= r_prev:
                continue
            # Volume declining: current 5d avg volume < prior 5d avg volume
            vol_now = vol.iloc[max(0, i-4):i+1].mean()
            vol_prev = vol.iloc[max(0, i-9):i-4].mean()
            if vol_prev > 0 and vol_now >= vol_prev:
                continue
            signals[(close.index[i], t)] = True

    return signals, 'RSI Divergence'


# =====================================================================
# 4. DEAD SIGNAL FILTERS (binary "not bearish" checks)
# =====================================================================

def build_filter_series(data):
    """Pre-compute all filter series for efficiency.
    Returns dict of filter_name -> pd.Series of dates where entry is BLOCKED."""

    filters = {}

    # A. VIX Term Structure: bearish if VIX/VIX3M > 1.05
    vix = data['close'].get('^VIX')
    vix3m = data['close'].get('^VIX3M')
    if vix is not None and vix3m is not None:
        ratio = vix / vix3m
        filters['vix_term_structure'] = ratio > 1.05  # True = blocked
    else:
        filters['vix_term_structure'] = pd.Series(False, index=data['close'].index)

    # B. Credit Stress: bearish if HYG/LQD 5d change < -0.5%
    hyg = data['close'].get('HYG')
    lqd = data['close'].get('LQD')
    if hyg is not None and lqd is not None:
        credit_ratio = hyg / lqd
        credit_chg = credit_ratio.pct_change(5) * 100  # percentage
        filters['credit_stress'] = credit_chg < -0.5  # True = blocked
    else:
        filters['credit_stress'] = pd.Series(False, index=data['close'].index)

    # C. Breadth: bearish if SPY RSI(14) < 25
    spy_close = data['close']['SPY']
    spy_rsi = _rsi(spy_close, 14)
    filters['breadth'] = spy_rsi < 25  # True = blocked

    # D. Momentum Context: bearish if SPY 20d return < -8%
    spy_ret_20d = spy_close.pct_change(20) * 100  # percentage
    filters['momentum_context'] = spy_ret_20d < -8  # True = blocked

    return filters


def apply_filter(signal_entries, filter_series, filter_name):
    """Remove entries where the filter is bearish (True = blocked).
    Returns filtered signal dict."""
    filtered = {}
    blocked_count = 0
    for (day, ticker) in signal_entries:
        try:
            is_blocked = filter_series.at[day]
        except (KeyError, ValueError):
            is_blocked = False
        if pd.isna(is_blocked):
            is_blocked = False
        if is_blocked:
            blocked_count += 1
            continue
        filtered[(day, ticker)] = True
    return filtered, blocked_count


# =====================================================================
# 5. BACKTESTER
# =====================================================================

def run_backtest(signal_entries, data, stock_tickers, label=''):
    """Run backtest on a set of signal entries {(date, ticker): True}."""
    close = data['close']
    spy_close = close['SPY']

    bt_start = pd.Timestamp(BACKTEST_START)
    entries = [(d, t) for (d, t) in signal_entries if d >= bt_start]
    entries.sort(key=lambda x: x[0])

    if not entries:
        return None

    trades = []
    open_positions = []

    for entry_date, ticker in entries:
        # Check max concurrent
        open_positions = [p for p in open_positions if p[3] > entry_date]
        if len(open_positions) >= MAX_CONCURRENT:
            continue

        try:
            entry_price = close.at[entry_date, ticker]
        except:
            continue
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # Find exit
        future_dates = close.index[close.index > entry_date]
        if len(future_dates) == 0:
            continue

        exit_price = None
        exit_date = None
        exit_reason = 'hold_expiry'

        for j, fdate in enumerate(future_dates[:HOLD_DAYS]):
            try:
                price = close.at[fdate, ticker]
            except:
                continue
            if pd.isna(price):
                continue
            ret = (price - entry_price) / entry_price
            if ret >= PROFIT_TARGET:
                exit_price = price
                exit_date = fdate
                exit_reason = 'profit_target'
                break
            elif ret <= STOP_LOSS:
                exit_price = price
                exit_date = fdate
                exit_reason = 'stop_loss'
                break

        if exit_price is None:
            hold_end = min(HOLD_DAYS, len(future_dates))
            if hold_end > 0:
                exit_date = future_dates[hold_end - 1]
                try:
                    exit_price = close.at[exit_date, ticker]
                except:
                    continue
                if pd.isna(exit_price):
                    continue

        if exit_price is None:
            continue

        raw_ret = (exit_price - entry_price) / entry_price
        net_ret = raw_ret - SPREAD_COST_PCT
        pnl = POS_SIZE * net_ret

        # SPY regime on entry
        try:
            spy_sma = spy_close.rolling(200).mean()
            spy_regime = 'bull' if spy_close.at[entry_date] > spy_sma.at[entry_date] else 'bear'
        except:
            spy_regime = 'unknown'

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'ticker': ticker,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': net_ret,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'regime': spy_regime,
            'hold_days': (exit_date - entry_date).days,
        })

        open_positions.append((entry_date, ticker, entry_price, exit_date))

    if not trades:
        return None

    return pd.DataFrame(trades)


# =====================================================================
# 6. VALIDATION (5-gate)
# =====================================================================

def validate_strategy(trades_df, label='', run_perm=True, all_signal_entries=None,
                      data=None, stock_tickers=None):
    """5-gate validation: Sharpe, WR, PF, MDD, regime gap, perm test."""
    if trades_df is None or len(trades_df) < 10:
        return {
            'label': label, 'n_trades': 0 if trades_df is None else len(trades_df),
            'sharpe': 0, 'wr': 0, 'pf': 0, 'mdd': 0, 'regime_gap': 1.0,
            'perm_p': 1.0, 'passed': False, 'failed_gates': ['insufficient_trades'],
            'reason': 'insufficient trades',
            'mean_ret': 0, 'bull_n': 0, 'bear_n': 0, 'avg_hold': 0,
        }

    rets = trades_df['return'].values
    n = len(rets)

    # Sharpe (annualized)
    years = max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    trades_per_year = n / years
    if trades_per_year < 1:
        trades_per_year = 1
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Win rate
    wr = np.mean(rets > 0)

    # Profit factor
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    # Max drawdown (cumulative PnL based)
    cum_pnl = np.cumsum(trades_df['pnl'].values)
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    mdd = np.min(dd) if len(dd) > 0 else 0

    # Regime gap
    bull_trades = trades_df[trades_df['regime'] == 'bull']
    bear_trades = trades_df[trades_df['regime'] == 'bear']
    if len(bull_trades) >= 3 and len(bear_trades) >= 3:
        bull_sharpe_raw = np.mean(bull_trades['return']) / max(np.std(bull_trades['return'], ddof=1), 1e-6)
        bear_sharpe_raw = np.mean(bear_trades['return']) / max(np.std(bear_trades['return'], ddof=1), 1e-6)
        max_abs = max(abs(bull_sharpe_raw), abs(bear_sharpe_raw), 1e-6)
        regime_gap = abs(bull_sharpe_raw - bear_sharpe_raw) / max_abs
    else:
        regime_gap = 0.0

    # Permutation test: random entry timing
    perm_p = 1.0
    if run_perm and n >= 10 and all_signal_entries is not None and data is not None and stock_tickers is not None:
        actual_sharpe = sharpe
        perm_sharpes = []
        bt_start = pd.Timestamp(BACKTEST_START)
        valid_dates = data['close'].index[data['close'].index >= bt_start]
        valid_dates = valid_dates[:-HOLD_DAYS - 5]

        for _ in range(N_PERMS):
            n_raw = len(all_signal_entries)
            rand_dates = np.random.choice(valid_dates, size=min(n_raw, len(valid_dates)), replace=True)
            rand_tickers = np.random.choice(stock_tickers, size=min(n_raw, len(valid_dates)), replace=True)
            rand_signals = {(d, t): True for d, t in zip(rand_dates, rand_tickers)}

            rand_trades = run_backtest(rand_signals, data, stock_tickers)
            if rand_trades is not None and len(rand_trades) >= 5:
                r_rets = rand_trades['return'].values
                r_years = max(1, (rand_trades['entry_date'].max() - rand_trades['entry_date'].min()).days / 365.25)
                r_tpy = len(r_rets) / r_years
                if r_tpy < 1:
                    r_tpy = 1
                r_mean = np.mean(r_rets)
                r_std = np.std(r_rets, ddof=1)
                r_sharpe = (r_mean / r_std) * np.sqrt(r_tpy) if r_std > 0 else 0
            else:
                r_sharpe = 0.0
            perm_sharpes.append(r_sharpe)
        perm_p = np.mean(np.array(perm_sharpes) >= actual_sharpe)

    # 5-gate check
    gates = {
        'sharpe': sharpe > 0.3,
        'wr': wr > 0.45,
        'pf': pf > 1.0,
        'mdd': mdd > -POS_SIZE * 5,
        'regime_gap': regime_gap < REGIME_GAP_LIMIT,
    }
    perm_pass = perm_p < 0.05

    passed = all(gates.values()) and perm_pass
    failed_gates = [k for k, v in gates.items() if not v]
    if not perm_pass:
        failed_gates.append('perm_test')

    return {
        'label': label,
        'n_trades': n,
        'sharpe': round(sharpe, 3),
        'wr': round(wr, 3),
        'pf': round(pf, 3),
        'mdd': round(mdd, 2),
        'regime_gap': round(regime_gap, 3),
        'perm_p': round(perm_p, 4),
        'passed': passed,
        'failed_gates': failed_gates,
        'mean_ret': round(mean_ret * 100, 2),
        'bull_n': len(bull_trades),
        'bear_n': len(bear_trades),
        'avg_hold': round(trades_df['hold_days'].mean(), 1),
        'reason': 'PASSED' if passed else f"FAILED: {', '.join(failed_gates)}",
    }


# =====================================================================
# 7. MAIN EXECUTION
# =====================================================================

def main():
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Stock universe: {len(stock_tickers)} tickers")

    # -- Generate base signals --
    print("\n[2] Generating base strategy signals...")
    base_generators = {
        'iv_rv_gap': signal_iv_rv_gap,
        'bond_yield': signal_bond_yield,
        'rsi_divergence': signal_rsi_divergence,
    }

    base_signals = {}
    for key, gen_func in base_generators.items():
        t0 = time.time()
        sigs, name = gen_func(data, stock_tickers)
        elapsed = time.time() - t0
        print(f"  {name}: {len(sigs)} raw entries ({elapsed:.1f}s)")
        base_signals[key] = sigs

    # -- Build filter series --
    print("\n[3] Building dead-signal filter series...")
    filter_series = build_filter_series(data)
    for fname, fseries in filter_series.items():
        blocked_days = fseries.sum() if hasattr(fseries, 'sum') else 0
        total_days = len(fseries)
        print(f"  {fname}: {int(blocked_days)}/{total_days} days blocked ({100*blocked_days/max(1,total_days):.1f}%)")

    # -- Run solo backtests (baselines) --
    print("\n[4] Running SOLO backtests (baselines)...")
    solo_results = {}
    solo_trades = {}
    for key, sigs in base_signals.items():
        trades_df = run_backtest(sigs, data, stock_tickers, label=key)
        result = validate_strategy(trades_df, label=f"SOLO: {key}", run_perm=True,
                                   all_signal_entries=sigs, data=data, stock_tickers=stock_tickers)
        solo_results[key] = result
        solo_trades[key] = trades_df
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {key:20s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
              f"WR={result['wr']:.3f} | PF={result['pf']:.3f} | perm_p={result['perm_p']:.4f} | {status}")

    # -- Define all 12 filter combinations --
    filter_combos = [
        # (base_strategy_key, filter_key, label_letter, label)
        ('iv_rv_gap', 'vix_term_structure', 'A', 'IV-RV + VIX TermStr filter'),
        ('iv_rv_gap', 'credit_stress',     'B', 'IV-RV + Credit Stress filter'),
        ('iv_rv_gap', 'breadth',           'C', 'IV-RV + Breadth filter'),
        ('iv_rv_gap', 'momentum_context',  'D', 'IV-RV + Momentum filter'),
        ('bond_yield', 'vix_term_structure', 'E', 'Bond Yld + VIX TermStr filter'),
        ('bond_yield', 'credit_stress',     'F', 'Bond Yld + Credit Stress filter'),
        ('bond_yield', 'breadth',           'G', 'Bond Yld + Breadth filter'),
        ('bond_yield', 'momentum_context',  'H', 'Bond Yld + Momentum filter'),
        ('rsi_divergence', 'vix_term_structure', 'I', 'RSI Div + VIX TermStr filter'),
        ('rsi_divergence', 'credit_stress',     'J', 'RSI Div + Credit Stress filter'),
        ('rsi_divergence', 'breadth',           'K', 'RSI Div + Breadth filter'),
        ('rsi_divergence', 'momentum_context',  'L', 'RSI Div + Momentum filter'),
    ]

    # -- Run filtered backtests --
    print(f"\n[5] Running {len(filter_combos)} FILTERED backtests...")
    filtered_results = []

    for base_key, filt_key, letter, combo_label in filter_combos:
        raw_sigs = base_signals[base_key]
        filt_s = filter_series[filt_key]

        filtered_sigs, blocked = apply_filter(raw_sigs, filt_s, filt_key)
        remaining = len(filtered_sigs)
        total = len(raw_sigs)
        pct_blocked = 100 * blocked / max(1, total)

        if remaining < 5:
            print(f"  {letter}) {combo_label:38s} | SKIP ({remaining} entries after filter, blocked {blocked}/{total})")
            filtered_results.append({
                'letter': letter, 'label': combo_label,
                'base_strategy': base_key, 'filter': filt_key,
                'n_raw': total, 'n_blocked': blocked, 'n_remaining': remaining,
                'n_trades': 0, 'sharpe': 0, 'wr': 0, 'pf': 0,
                'perm_p': 1.0, 'passed': False,
                'filter_alpha': 0, 'solo_sharpe': solo_results[base_key]['sharpe'],
                'reason': 'insufficient entries after filter',
            })
            continue

        trades_df = run_backtest(filtered_sigs, data, stock_tickers, label=combo_label)
        result = validate_strategy(trades_df, label=combo_label, run_perm=True,
                                   all_signal_entries=filtered_sigs, data=data,
                                   stock_tickers=stock_tickers)

        solo_sharpe = solo_results[base_key]['sharpe']
        filter_alpha = round(result['sharpe'] - solo_sharpe, 3)

        result['letter'] = letter
        result['base_strategy'] = base_key
        result['filter'] = filt_key
        result['n_raw'] = total
        result['n_blocked'] = blocked
        result['n_remaining'] = remaining
        result['solo_sharpe'] = solo_sharpe
        result['filter_alpha'] = filter_alpha

        filtered_results.append(result)

        alpha_tag = f"alpha={filter_alpha:+.3f}"
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {letter}) {combo_label:38s} | n={result['n_trades']:4d} (blocked {blocked:4d}/{total:4d}, "
              f"{pct_blocked:.0f}%) | Sharpe={result['sharpe']:6.3f} (solo {solo_sharpe:.3f}) | "
              f"WR={result['wr']:.3f} | PF={result['pf']:.3f} | p={result['perm_p']:.4f} | "
              f"{alpha_tag} | {status}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 120)
    print("RESULTS SUMMARY: DEAD SIGNAL FILTER BACKTEST")
    print("=" * 120)

    # Solo baselines
    print("\n-- SOLO BASELINES --")
    print(f"{'Strategy':20s} | {'N':>5s} | {'Sharpe':>7s} | {'WR':>5s} | {'PF':>6s} | {'MDD':>8s} | {'RegGap':>6s} | {'Perm p':>7s} | {'Status':>6s}")
    print("-" * 95)
    for key in ['iv_rv_gap', 'bond_yield', 'rsi_divergence']:
        r = solo_results[key]
        status = "PASS" if r['passed'] else "FAIL"
        print(f"{key:20s} | {r['n_trades']:5d} | {r['sharpe']:7.3f} | {r['wr']:.3f} | {r['pf']:6.3f} | "
              f"{r['mdd']:8.2f} | {r['regime_gap']:6.3f} | {r['perm_p']:7.4f} | {status:>6s}")

    # Filtered results
    print(f"\n-- FILTERED COMBINATIONS (sorted by filter alpha) --")
    print(f"{'#':>2s} {'Combo':40s} | {'N':>5s} | {'Sharpe':>7s} | {'Solo':>6s} | {'Alpha':>7s} | "
          f"{'WR':>5s} | {'PF':>6s} | {'Perm p':>7s} | {'Blocked%':>8s} | {'Pass':>4s}")
    print("-" * 130)

    sorted_filtered = sorted(filtered_results, key=lambda x: x.get('filter_alpha', 0), reverse=True)
    for r in sorted_filtered:
        status = "PASS" if r.get('passed', False) else "FAIL"
        alpha = r.get('filter_alpha', 0)
        solo = r.get('solo_sharpe', 0)
        pct_blk = 100 * r.get('n_blocked', 0) / max(1, r.get('n_raw', 1))
        letter = r.get('letter', '?')
        print(f"{letter:>2s} {r['label']:40s} | {r.get('n_trades',0):5d} | {r.get('sharpe',0):7.3f} | {solo:6.3f} | "
              f"{alpha:+7.3f} | {r.get('wr',0):.3f} | {r.get('pf',0):6.3f} | {r.get('perm_p',1):7.4f} | "
              f"{pct_blk:7.1f}% | {status:>4s}")

    # Highlight winners
    print(f"\n-- HIGHLIGHTS --")

    # 1. Filter alpha > 0 (filter helps)
    helpers = [r for r in sorted_filtered if r.get('filter_alpha', 0) > 0 and r.get('n_trades', 0) >= 10]
    print(f"\n  Filters that HELP (alpha > 0):")
    if helpers:
        for r in helpers:
            print(f"    {r.get('letter','?')}) {r['label']}: alpha={r['filter_alpha']:+.3f}, "
                  f"Sharpe {r['sharpe']:.3f} vs solo {r['solo_sharpe']:.3f}")
    else:
        print("    None found.")

    # 2. Filtered passes 5-gate even if solo didn't
    promoted = []
    for r in sorted_filtered:
        if r.get('passed', False):
            base_passed = solo_results.get(r.get('base_strategy', ''), {}).get('passed', False)
            if not base_passed:
                promoted.append(r)
    print(f"\n  Filters that PROMOTE (solo failed but filtered passed):")
    if promoted:
        for r in promoted:
            print(f"    {r.get('letter','?')}) {r['label']}: Sharpe={r['sharpe']:.3f}, "
                  f"WR={r['wr']:.1%}, PF={r['pf']:.2f}, p={r['perm_p']:.4f}")
    else:
        print("    None found.")

    # 3. Filter alpha > 0.30
    strong = [r for r in sorted_filtered if r.get('filter_alpha', 0) > 0.30 and r.get('n_trades', 0) >= 10]
    print(f"\n  STRONG filter alpha (> 0.30):")
    if strong:
        for r in strong:
            print(f"    ** {r.get('letter','?')}) {r['label']}: alpha={r['filter_alpha']:+.3f}, "
                  f"Sharpe={r['sharpe']:.3f}, WR={r['wr']:.1%}, PF={r['pf']:.2f}")
    else:
        print("    None found.")

    # Save results
    def _sanitize(v):
        if isinstance(v, (pd.Timestamp,)):
            return str(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    def _sanitize_dict(d):
        return {k: _sanitize(v) if not isinstance(v, list) else [_sanitize(x) for x in v] for k, v in d.items()}

    results_data = {
        'run_date': datetime.now().isoformat(),
        'config': {
            'universe_size': len(stock_tickers),
            'backtest_start': BACKTEST_START,
            'backtest_end': END_DATE,
            'pos_size': POS_SIZE,
            'max_concurrent': MAX_CONCURRENT,
            'hold_days': HOLD_DAYS,
            'profit_target': PROFIT_TARGET,
            'stop_loss': STOP_LOSS,
            'spread_cost': SPREAD_COST_PCT,
            'n_perms': N_PERMS,
            'regime_gap_limit': REGIME_GAP_LIMIT,
        },
        'solo_results': {k: _sanitize_dict(v) for k, v in solo_results.items()},
        'filtered_results': [_sanitize_dict(r) for r in sorted_filtered],
        'helpers': [_sanitize_dict(r) for r in helpers],
        'promoted': [_sanitize_dict(r) for r in promoted],
        'strong_alpha': [_sanitize_dict(r) for r in strong],
    }

    results_file = OUTPUT_DIR / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 80)
    print("DONE -- Dead Signal Filter Backtest Complete")
    print("=" * 80)

    return results_data


if __name__ == '__main__':
    main()
