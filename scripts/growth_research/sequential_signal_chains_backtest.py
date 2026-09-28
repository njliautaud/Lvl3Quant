#!/usr/bin/env python3
"""
Sequential Signal Chains Backtest
==================================
Tests setup-then-trigger patterns: Signal A fires on day 1 (setup),
Signal B fires within 1-5 days (trigger), enter on trigger day.

Hypothesis: a setup-then-confirm pattern should be stronger than
same-day confluence (which was already tested and failed: 0/9 pairs passed).

6 Chain Variants:
  A) IV-RV Setup -> RSI Trigger
  B) Bond Yield Setup -> Dip Trigger
  C) Liquidity Setup -> Volume Trigger
  D) VIX Spike Setup -> Quality Dip Trigger
  E) RSI Divergence Setup -> Bond Yield Trigger
  F) Multi-Setup Chain (VIX+RV + yield declining + RSI dip)

5-Gate validation: Sharpe, WR, PF, MDD, regime gap (<0.50), perm test (p<0.05)
"""

import os, sys, json, warnings, time, functools, pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

# ─── Config ───
START_DATE = '2019-01-01'
END_DATE = '2026-07-01'
BACKTEST_START = '2020-01-01'
POS_SIZE = 300.0
MAX_CONCURRENT = 2
HOLD_DAYS = 21
PROFIT_TARGET = 0.10
STOP_LOSS = -0.15
SPREAD_COST_PCT = 0.001
N_PERMS = 200
REGIME_GAP_LIMIT = 0.50

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/sequential_signal_chains')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]

MACRO_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', '^TNX', 'HYG', 'LQD']

print("=" * 80)
print("SEQUENTIAL SIGNAL CHAINS BACKTEST")
print("Setup-then-Trigger pattern testing")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════
def download_data():
    import yfinance as yf
    cache_file = OUTPUT_DIR / '_seq_chains_cache.pkl'
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded cache: {len(data['close'].columns)} tickers, {len(data['close'])} days")
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
            except Exception:
                pass

    close = close.ffill().dropna(how='all')

    data = {
        'close': close,
        'high': high.reindex(close.index).ffill(),
        'low': low.reindex(close.index).ffill(),
        'volume': volume.reindex(close.index).ffill().fillna(0),
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Downloaded: {len(close.columns)} tickers, {len(close)} days")
    return data

# ═══════════════════════════════════════════════════════════════════════
# 2. INDICATOR HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _rsi(series, period=14):
    """Standard RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _realized_vol(returns, window=20):
    """Annualized realized vol from returns. 20-day rolling std * sqrt(252) * 100."""
    return returns.rolling(window).std() * np.sqrt(252) * 100

def _sma(series, window=20):
    """Simple moving average."""
    return series.rolling(window).mean()

# ═══════════════════════════════════════════════════════════════════════
# 3. SETUP SIGNAL GENERATORS
#    Each returns {date: True} for macro setups, or {(date, ticker): True} for stock-specific
# ═══════════════════════════════════════════════════════════════════════

def setup_iv_rv(data):
    """Setup: VIX > 20d realized vol of SPY by 5+ points."""
    spy_close = data['close']['SPY']
    spy_ret = spy_close.pct_change()
    vix = data['close']['^VIX']
    rv_20 = _realized_vol(spy_ret, 20)
    gap = vix - rv_20
    fire_days = gap[gap > 5].index
    return set(fire_days), 'IV-RV Gap Setup'

def setup_bond_yield_drop(data):
    """Setup: 10Y yield drops >0.1% over 5 days."""
    if '^TNX' not in data['close'].columns:
        # Fallback: TLT up > 2% in 5d as proxy
        tlt = data['close']['TLT']
        change = tlt.pct_change(5)
        fire_days = change[change > 0.02].index
    else:
        tnx = data['close']['^TNX']
        yield_change = tnx.diff(5)
        fire_days = yield_change[yield_change < -0.10].index
    return set(fire_days), 'Bond Yield Drop Setup'

def setup_liquidity_narrow(data, stock_tickers):
    """Setup: HL spread narrows below 60d average (per stock)."""
    setups = {}  # {(date, ticker): True}
    for t in stock_tickers:
        if t not in data['close'].columns or t not in data['high'].columns:
            continue
        high = data['high'][t].dropna()
        low = data['low'][t].dropna()
        close = data['close'][t].dropna()
        idx = high.index.intersection(low.index).intersection(close.index)
        if len(idx) < 65:
            continue
        high_s, low_s, close_s = high.loc[idx], low.loc[idx], close.loc[idx]
        hl_spread = (high_s - low_s) / close_s
        avg_60 = hl_spread.rolling(60).mean()
        narrow = hl_spread < avg_60
        for i in range(60, len(idx)):
            if narrow.iloc[i]:
                setups[(idx[i], t)] = True
    return setups, 'Liquidity Narrow Setup'

def setup_vix_spike(data):
    """Setup: VIX rises >20% in 5 days."""
    vix = data['close']['^VIX']
    vix_change_pct = vix.pct_change(5)
    fire_days = vix_change_pct[vix_change_pct > 0.20].index
    return set(fire_days), 'VIX Spike Setup'

def setup_rsi_divergence(data, stock_tickers):
    """Setup: Bullish RSI divergence - price makes lower low but RSI makes higher low over 14 days."""
    setups = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, 14)
        for i in range(14, len(close)):
            if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i - 14]):
                continue
            # Price: current close < close 14 days ago (lower low)
            price_lower_low = close.iloc[i] < close.iloc[i - 14]
            # RSI: current RSI > RSI 14 days ago (higher low)
            rsi_higher_low = rsi.iloc[i] > rsi.iloc[i - 14]
            if price_lower_low and rsi_higher_low:
                setups[(close.index[i], t)] = True
    return setups, 'RSI Divergence Setup'

def setup_multi(data, stock_tickers):
    """Multi-Setup: VIX > 20 AND VIX > realized vol AND 10Y yield declining.
    Returns dates where ALL macro conditions hold."""
    vix = data['close']['^VIX']
    spy_ret = data['close']['SPY'].pct_change()
    rv_20 = _realized_vol(spy_ret, 20)

    # VIX > 20
    cond1 = vix > 20
    # VIX > realized vol
    cond2 = vix > rv_20
    # 10Y yield declining (over 5 days)
    if '^TNX' in data['close'].columns:
        tnx = data['close']['^TNX']
        yield_declining = tnx.diff(5) < 0
    else:
        tlt = data['close']['TLT']
        yield_declining = tlt.pct_change(5) > 0  # TLT up = yields down
    cond3 = yield_declining

    combined = cond1 & cond2 & cond3
    fire_days = combined[combined == True].index
    return set(fire_days), 'Multi-Setup (VIX+RV+Yield)'

# ═══════════════════════════════════════════════════════════════════════
# 4. TRIGGER SIGNAL GENERATORS
#    Each returns {(date, ticker): True}
# ═══════════════════════════════════════════════════════════════════════

def trigger_rsi_below_35(data, stock_tickers):
    """Trigger: RSI(14) drops below 35."""
    triggers = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, 14)
        for i in range(14, len(close)):
            if pd.isna(rsi.iloc[i]):
                continue
            if rsi.iloc[i] < 35:
                triggers[(close.index[i], t)] = True
    return triggers, 'RSI < 35 Trigger'

def trigger_dip_below_sma(data, stock_tickers):
    """Trigger: stock drops >5% below 20-SMA."""
    triggers = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        sma20 = _sma(close, 20)
        for i in range(20, len(close)):
            if pd.isna(sma20.iloc[i]):
                continue
            pct_below = (close.iloc[i] - sma20.iloc[i]) / sma20.iloc[i]
            if pct_below < -0.05:
                triggers[(close.index[i], t)] = True
    return triggers, 'Dip >5% below 20-SMA Trigger'

def trigger_volume_spike(data, stock_tickers):
    """Trigger: Volume spikes >2x 20d average."""
    triggers = {}
    for t in stock_tickers:
        if t not in data['close'].columns or t not in data['volume'].columns:
            continue
        vol = data['volume'][t].dropna()
        if len(vol) < 25:
            continue
        avg_20 = vol.rolling(20).mean()
        for i in range(20, len(vol)):
            if pd.isna(avg_20.iloc[i]) or avg_20.iloc[i] <= 0:
                continue
            if vol.iloc[i] > 2.0 * avg_20.iloc[i]:
                triggers[(vol.index[i], t)] = True
    return triggers, 'Volume Spike >2x Trigger'

def trigger_quality_dip(data, stock_tickers):
    """Trigger: any quality stock drops >7% below 20-SMA."""
    triggers = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        sma20 = _sma(close, 20)
        for i in range(20, len(close)):
            if pd.isna(sma20.iloc[i]):
                continue
            pct_below = (close.iloc[i] - sma20.iloc[i]) / sma20.iloc[i]
            if pct_below < -0.07:
                triggers[(close.index[i], t)] = True
    return triggers, 'Quality Dip >7% below 20-SMA Trigger'

def trigger_bond_yield_drop_small(data):
    """Trigger: 10Y yield drops >0.05% within 5 days."""
    if '^TNX' not in data['close'].columns:
        tlt = data['close']['TLT']
        change = tlt.pct_change(5)
        fire_days = change[change > 0.01].index  # smaller threshold
    else:
        tnx = data['close']['^TNX']
        yield_change = tnx.diff(5)
        fire_days = yield_change[yield_change < -0.05].index
    return set(fire_days), 'Bond Yield Drop >0.05% Trigger'

def trigger_multi_entry(data, stock_tickers):
    """Multi trigger: RSI<35 + >5% below 20-SMA (both must hold on same day)."""
    triggers = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        rsi = _rsi(close, 14)
        sma20 = _sma(close, 20)
        for i in range(20, len(close)):
            if pd.isna(rsi.iloc[i]) or pd.isna(sma20.iloc[i]):
                continue
            pct_below = (close.iloc[i] - sma20.iloc[i]) / sma20.iloc[i]
            if rsi.iloc[i] < 35 and pct_below < -0.05:
                triggers[(close.index[i], t)] = True
    return triggers, 'RSI<35 + Dip>5% Trigger'

# ═══════════════════════════════════════════════════════════════════════
# 5. SEQUENTIAL CHAIN MATCHER
# ═══════════════════════════════════════════════════════════════════════

def match_sequential_chain(setup_dates, triggers, max_delay_days, stock_tickers, data):
    """Match setup-then-trigger sequences.

    Args:
        setup_dates: either set of dates (macro setup) or dict of {(date, ticker): True} (stock-specific setup)
        triggers: dict of {(date, ticker): True}
        max_delay_days: max calendar days between setup and trigger
        stock_tickers: list of tickers
        data: full data dict

    Returns:
        dict of {(trigger_date, ticker): True} for matched entries
    """
    # Determine if setup is macro (set of dates) or stock-specific (dict with (date, ticker) keys)
    is_macro_setup = isinstance(setup_dates, set)

    # Build trigger index by ticker -> sorted list of dates
    trigger_by_ticker = defaultdict(list)
    for (d, t) in triggers:
        trigger_by_ticker[t].append(d)
    for t in trigger_by_ticker:
        trigger_by_ticker[t].sort()

    entries = {}

    if is_macro_setup:
        # Macro setup: fires for all stocks, look for per-stock trigger within max_delay_days
        sorted_setup_dates = sorted(setup_dates)
        for setup_date in sorted_setup_dates:
            for t in stock_tickers:
                if t not in trigger_by_ticker:
                    continue
                # Find triggers within [setup_date, setup_date + max_delay_days]
                for trig_date in trigger_by_ticker[t]:
                    delta = (trig_date - setup_date).days
                    if delta < 0:
                        continue
                    if delta > max_delay_days:
                        break
                    # Valid chain: setup on setup_date, trigger on trig_date for ticker t
                    entries[(trig_date, t)] = True
    else:
        # Stock-specific setup: match per-stock
        setup_by_ticker = defaultdict(list)
        for (d, t) in setup_dates:
            setup_by_ticker[t].append(d)
        for t in setup_by_ticker:
            setup_by_ticker[t].sort()

        for t in stock_tickers:
            if t not in setup_by_ticker or t not in trigger_by_ticker:
                continue
            s_dates = setup_by_ticker[t]
            t_dates = trigger_by_ticker[t]
            # For each setup date, find triggers within window
            ti = 0
            for sd in s_dates:
                while ti < len(t_dates) and t_dates[ti] < sd:
                    ti += 1
                j = ti
                while j < len(t_dates):
                    delta = (t_dates[j] - sd).days
                    if delta > max_delay_days:
                        break
                    entries[(t_dates[j], t)] = True
                    j += 1

    return entries

def match_sequential_chain_mixed(setup_dates_macro, triggers_macro_dates, triggers_stock, max_delay_days, stock_tickers, data):
    """For chain E: stock-specific setup + macro trigger.

    setup_dates_macro: dict {(date, ticker): True} (stock-specific RSI divergence)
    triggers_macro_dates: set of dates (macro bond yield trigger)
    triggers_stock: not used here, kept for signature consistency

    Entry: on trigger date for the specific stock that had the setup.
    """
    # Build setup index by ticker
    setup_by_ticker = defaultdict(list)
    for (d, t) in setup_dates_macro:
        setup_by_ticker[t].append(d)
    for t in setup_by_ticker:
        setup_by_ticker[t].sort()

    sorted_trigger_dates = sorted(triggers_macro_dates)

    entries = {}
    for t in stock_tickers:
        if t not in setup_by_ticker:
            continue
        s_dates = setup_by_ticker[t]
        for sd in s_dates:
            for td in sorted_trigger_dates:
                delta = (td - sd).days
                if delta < 0:
                    continue
                if delta > max_delay_days:
                    break
                entries[(td, t)] = True

    return entries

# ═══════════════════════════════════════════════════════════════════════
# 6. BACKTESTER
# ═══════════════════════════════════════════════════════════════════════

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
        except Exception:
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
            except Exception:
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
                except Exception:
                    continue
                if pd.isna(exit_price):
                    continue

        if exit_price is None:
            continue

        raw_ret = (exit_price - entry_price) / entry_price
        net_ret = raw_ret - SPREAD_COST_PCT
        pnl = POS_SIZE * net_ret

        # Regime: bull if SPY close > SPY previous close on entry day
        try:
            spy_prev_idx = spy_close.index.get_loc(entry_date)
            if spy_prev_idx > 0:
                spy_today = spy_close.iloc[spy_prev_idx]
                spy_yesterday = spy_close.iloc[spy_prev_idx - 1]
                regime = 'bull' if spy_today > spy_yesterday else 'bear'
            else:
                regime = 'unknown'
        except Exception:
            regime = 'unknown'

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'ticker': ticker,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': net_ret,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'regime': regime,
            'hold_days': (exit_date - entry_date).days,
        })

        open_positions.append((entry_date, ticker, entry_price, exit_date))

    if not trades:
        return None

    return pd.DataFrame(trades)

# ═══════════════════════════════════════════════════════════════════════
# 7. VALIDATION (5-Gate + Regime + Perm)
# ═══════════════════════════════════════════════════════════════════════

def compute_sortino(rets, trades_per_year):
    """Annualized Sortino ratio."""
    mean_ret = np.mean(rets)
    downside = rets[rets < 0]
    if len(downside) < 2:
        return 99.0
    down_std = np.std(downside, ddof=1)
    if down_std <= 0:
        return 99.0
    return (mean_ret / down_std) * np.sqrt(trades_per_year)

def validate_strategy(trades_df, label='', run_perm=True, all_signal_entries=None, data=None, stock_tickers=None):
    """5-gate validation: Sharpe, WR, PF, MDD, regime gap, perm test."""
    if trades_df is None or len(trades_df) < 10:
        return {
            'label': label, 'n_trades': 0 if trades_df is None else len(trades_df),
            'sharpe': 0, 'sortino': 0, 'wr': 0, 'pf': 0, 'mdd': 0,
            'regime_gap': 1.0, 'perm_p': 1.0, 'passed': False,
            'reason': 'insufficient trades (<10)'
        }

    rets = trades_df['return'].values
    n = len(rets)

    # Sharpe (annualized)
    years_span = max(1, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    trades_per_year = n / years_span
    if trades_per_year < 1:
        trades_per_year = 1
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    sortino = compute_sortino(rets, trades_per_year)

    # Win rate
    wr = np.mean(rets > 0)

    # Profit factor
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = np.abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    # Max drawdown (cumulative PnL)
    cum_pnl = np.cumsum(trades_df['pnl'].values)
    peak = np.maximum.accumulate(cum_pnl)
    dd = cum_pnl - peak
    mdd = np.min(dd) if len(dd) > 0 else 0
    mdd_pct = (mdd / max(peak.max(), 1)) * 100 if peak.max() > 0 else 0

    # Regime gap: bull day = SPY close > SPY prev close (already classified per trade)
    bull_trades = trades_df[trades_df['regime'] == 'bull']
    bear_trades = trades_df[trades_df['regime'] == 'bear']
    if len(bull_trades) >= 3 and len(bear_trades) >= 3:
        bull_sharpe_raw = np.mean(bull_trades['return']) / max(np.std(bull_trades['return'], ddof=1), 1e-6)
        bear_sharpe_raw = np.mean(bear_trades['return']) / max(np.std(bear_trades['return'], ddof=1), 1e-6)
        max_abs = max(abs(bull_sharpe_raw), abs(bear_sharpe_raw), 1e-6)
        regime_gap = abs(bull_sharpe_raw - bear_sharpe_raw) / max_abs
    else:
        regime_gap = 0.0  # Can't assess, pass by default

    # Permutation test: shuffle signal dates, compare Sharpes
    perm_p = 1.0
    if run_perm and n >= 10 and all_signal_entries is not None and data is not None and stock_tickers is not None:
        actual_sharpe = sharpe
        perm_sharpes = []
        bt_start = pd.Timestamp(BACKTEST_START)
        valid_dates = data['close'].index[data['close'].index >= bt_start]
        valid_dates = valid_dates[:-HOLD_DAYS - 5] if len(valid_dates) > HOLD_DAYS + 5 else valid_dates

        n_raw = len(all_signal_entries)
        for _ in range(N_PERMS):
            rand_dates = np.random.choice(valid_dates, size=min(n_raw, len(valid_dates)), replace=True)
            rand_tickers = np.random.choice(stock_tickers, size=min(n_raw, len(valid_dates)), replace=True)
            rand_signals = {(d, t): True for d, t in zip(rand_dates, rand_tickers)}

            rand_trades = run_backtest(rand_signals, data, stock_tickers)
            if rand_trades is not None and len(rand_trades) >= 5:
                r_rets = rand_trades['return'].values
                r_tpy = len(r_rets) / max(1, (rand_trades['entry_date'].max() - rand_trades['entry_date'].min()).days / 365.25)
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
        'sortino': round(sortino, 3),
        'wr': round(wr, 3),
        'pf': round(pf, 3),
        'mdd': round(mdd, 2),
        'mdd_pct': round(mdd_pct, 2),
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

# ═══════════════════════════════════════════════════════════════════════
# 8. CHAIN DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

def build_chains(data, stock_tickers):
    """Build all 6 sequential chain variants and return matched entries."""

    print("\n[2] Generating setup and trigger signals...")

    # ── Generate all setup signals ──
    t0 = time.time()
    iv_rv_setup, iv_rv_name = setup_iv_rv(data)
    print(f"  {iv_rv_name}: {len(iv_rv_setup)} setup days ({time.time()-t0:.1f}s)")

    t0 = time.time()
    bond_setup, bond_name = setup_bond_yield_drop(data)
    print(f"  {bond_name}: {len(bond_setup)} setup days ({time.time()-t0:.1f}s)")

    t0 = time.time()
    liq_setup, liq_name = setup_liquidity_narrow(data, stock_tickers)
    print(f"  {liq_name}: {len(liq_setup)} setup signals ({time.time()-t0:.1f}s)")

    t0 = time.time()
    vix_spike_setup, vix_spike_name = setup_vix_spike(data)
    print(f"  {vix_spike_name}: {len(vix_spike_setup)} setup days ({time.time()-t0:.1f}s)")

    t0 = time.time()
    rsi_div_setup, rsi_div_name = setup_rsi_divergence(data, stock_tickers)
    print(f"  {rsi_div_name}: {len(rsi_div_setup)} setup signals ({time.time()-t0:.1f}s)")

    t0 = time.time()
    multi_setup, multi_name = setup_multi(data, stock_tickers)
    print(f"  {multi_name}: {len(multi_setup)} setup days ({time.time()-t0:.1f}s)")

    # ── Generate all trigger signals ──
    print("\n  Generating trigger signals...")

    t0 = time.time()
    rsi35_trigger, rsi35_name = trigger_rsi_below_35(data, stock_tickers)
    print(f"  {rsi35_name}: {len(rsi35_trigger)} trigger signals ({time.time()-t0:.1f}s)")

    t0 = time.time()
    dip_trigger, dip_name = trigger_dip_below_sma(data, stock_tickers)
    print(f"  {dip_name}: {len(dip_trigger)} trigger signals ({time.time()-t0:.1f}s)")

    t0 = time.time()
    vol_spike_trigger, vol_spike_name = trigger_volume_spike(data, stock_tickers)
    print(f"  {vol_spike_name}: {len(vol_spike_trigger)} trigger signals ({time.time()-t0:.1f}s)")

    t0 = time.time()
    quality_dip_trigger, quality_dip_name = trigger_quality_dip(data, stock_tickers)
    print(f"  {quality_dip_name}: {len(quality_dip_trigger)} trigger signals ({time.time()-t0:.1f}s)")

    t0 = time.time()
    bond_small_trigger, bond_small_name = trigger_bond_yield_drop_small(data)
    print(f"  {bond_small_name}: {len(bond_small_trigger)} trigger days ({time.time()-t0:.1f}s)")

    t0 = time.time()
    multi_entry_trigger, multi_entry_name = trigger_multi_entry(data, stock_tickers)
    print(f"  {multi_entry_name}: {len(multi_entry_trigger)} trigger signals ({time.time()-t0:.1f}s)")

    # ── Match chains ──
    print("\n[3] Matching sequential chains...")

    chains = {}

    # A) IV-RV Setup -> RSI Trigger (macro setup, stock trigger, 5-day window)
    t0 = time.time()
    chain_a = match_sequential_chain(iv_rv_setup, rsi35_trigger, 5, stock_tickers, data)
    chains['A_IVRV_RSI'] = (chain_a, 'A) IV-RV Setup -> RSI<35 Trigger (5d)')
    print(f"  Chain A: {len(chain_a)} entries ({time.time()-t0:.1f}s)")

    # B) Bond Yield Setup -> Dip Trigger (macro setup, stock trigger, 5-day window)
    t0 = time.time()
    chain_b = match_sequential_chain(bond_setup, dip_trigger, 5, stock_tickers, data)
    chains['B_Bond_Dip'] = (chain_b, 'B) Bond Yield Drop -> Dip>5% Trigger (5d)')
    print(f"  Chain B: {len(chain_b)} entries ({time.time()-t0:.1f}s)")

    # C) Liquidity Setup -> Volume Trigger (stock-specific setup, stock trigger, 3-day window)
    t0 = time.time()
    chain_c = match_sequential_chain(liq_setup, vol_spike_trigger, 3, stock_tickers, data)
    chains['C_Liq_Vol'] = (chain_c, 'C) Liquidity Narrow -> Volume Spike Trigger (3d)')
    print(f"  Chain C: {len(chain_c)} entries ({time.time()-t0:.1f}s)")

    # D) VIX Spike Setup -> Quality Dip Trigger (macro setup, stock trigger, 5-day window)
    t0 = time.time()
    chain_d = match_sequential_chain(vix_spike_setup, quality_dip_trigger, 5, stock_tickers, data)
    chains['D_VIXSpike_QualDip'] = (chain_d, 'D) VIX Spike -> Quality Dip>7% Trigger (5d)')
    print(f"  Chain D: {len(chain_d)} entries ({time.time()-t0:.1f}s)")

    # E) RSI Divergence Setup -> Bond Yield Trigger (stock setup, macro trigger, 5-day window)
    t0 = time.time()
    # Convert macro trigger dates to per-stock entries for the stocks that had setup
    chain_e = match_sequential_chain_mixed(rsi_div_setup, bond_small_trigger, None, 5, stock_tickers, data)
    chains['E_RSIDiv_Bond'] = (chain_e, 'E) RSI Divergence -> Bond Yield Drop Trigger (5d)')
    print(f"  Chain E: {len(chain_e)} entries ({time.time()-t0:.1f}s)")

    # F) Multi-Setup Chain (macro multi-setup, stock multi-trigger, 7-day window)
    t0 = time.time()
    chain_f = match_sequential_chain(multi_setup, multi_entry_trigger, 7, stock_tickers, data)
    chains['F_MultiChain'] = (chain_f, 'F) Multi-Setup -> RSI<35+Dip>5% Trigger (7d)')
    print(f"  Chain F: {len(chain_f)} entries ({time.time()-t0:.1f}s)")

    return chains

# ═══════════════════════════════════════════════════════════════════════
# 9. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════

def main():
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Stock universe: {len(stock_tickers)} tickers")

    chains = build_chains(data, stock_tickers)

    # ── Run backtests ──
    print(f"\n[4] Running backtests for {len(chains)} chain variants...")
    results = []

    for key, (entries, label) in chains.items():
        if len(entries) < 5:
            print(f"  {label:55s} | SKIP (only {len(entries)} entries)")
            results.append({
                'label': label, 'key': key, 'n_trades': len(entries),
                'sharpe': 0, 'sortino': 0, 'wr': 0, 'pf': 0, 'mdd': 0,
                'mdd_pct': 0, 'regime_gap': 1.0, 'perm_p': 1.0,
                'passed': False, 'reason': 'insufficient entries',
            })
            continue

        t0 = time.time()
        trades_df = run_backtest(entries, data, stock_tickers, label=label)
        result = validate_strategy(
            trades_df, label=label, run_perm=True,
            all_signal_entries=entries, data=data, stock_tickers=stock_tickers
        )
        result['key'] = key
        elapsed = time.time() - t0

        status = "PASS" if result['passed'] else "FAIL"
        print(f"  {label:55s} | n={result['n_trades']:4d} | Sharpe={result['sharpe']:6.3f} | "
              f"Sortino={result['sortino']:6.3f} | WR={result['wr']:.3f} | PF={result['pf']:.3f} | "
              f"MDD={result['mdd_pct']:.1f}% | RegGap={result['regime_gap']:.3f} | "
              f"p={result['perm_p']:.4f} | {status} ({elapsed:.0f}s)")

        results.append(result)

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("RESULTS SUMMARY: SEQUENTIAL SIGNAL CHAINS")
    print("=" * 120)

    print(f"\n{'Variant':55s} | {'N':>5s} | {'Sharpe':>7s} | {'Sortino':>7s} | {'WR%':>5s} | "
          f"{'PF':>6s} | {'MDD%':>6s} | {'RegGap':>6s} | {'Perm p':>7s} | {'Result':>6s}")
    print("-" * 120)

    sorted_results = sorted(results, key=lambda x: x.get('sharpe', 0), reverse=True)
    for r in sorted_results:
        status = "PASS" if r.get('passed', False) else "FAIL"
        print(f"{r['label']:55s} | {r['n_trades']:5d} | {r.get('sharpe',0):7.3f} | "
              f"{r.get('sortino',0):7.3f} | {r.get('wr',0)*100:5.1f} | {r.get('pf',0):6.3f} | "
              f"{r.get('mdd_pct',0):6.1f} | {r.get('regime_gap',1):6.3f} | "
              f"{r.get('perm_p',1):7.4f} | {status:>6s}")

    # Winners
    winners = [r for r in sorted_results if r.get('passed', False)]
    print(f"\n── WINNERS (passed all 5 gates + perm test) ──")
    if winners:
        for r in winners:
            print(f"  ** {r['label']}: Sharpe={r['sharpe']:.3f}, Sortino={r['sortino']:.3f}, "
                  f"WR={r['wr']:.1%}, PF={r['pf']:.2f}, perm_p={r['perm_p']:.4f}")
    else:
        print("  No chain variants passed ALL gates.")
        near = [r for r in sorted_results if r.get('sharpe', 0) > 0.2 and r['n_trades'] >= 10]
        if near:
            print("  Best near-misses:")
            for r in near[:3]:
                print(f"    {r['label']}: Sharpe={r.get('sharpe',0):.3f}, WR={r.get('wr',0):.1%}, "
                      f"perm_p={r.get('perm_p',1):.4f}, failed={r.get('failed_gates', [])}")

    # Save results
    def sanitize(v):
        if isinstance(v, (pd.Timestamp,)):
            return str(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v

    results_data = {
        'run_date': datetime.now().isoformat(),
        'concept': 'Sequential signal chains: setup on day 1, trigger within N days, enter on trigger day',
        'hypothesis': 'Setup-then-confirm should be stronger than same-day confluence (which failed 0/9)',
        'config': {
            'universe_size': len(stock_tickers),
            'backtest_start': BACKTEST_START,
            'backtest_end': END_DATE,
            'pos_size': POS_SIZE,
            'max_concurrent': MAX_CONCURRENT,
            'hold_days': HOLD_DAYS,
            'profit_target': PROFIT_TARGET,
            'stop_loss': STOP_LOSS,
            'spread_cost_pct': SPREAD_COST_PCT,
            'n_perms': N_PERMS,
            'regime_gap_limit': REGIME_GAP_LIMIT,
        },
        'chain_results': [
            {k: sanitize(v) for k, v in r.items()} for r in sorted_results
        ],
        'winners': [
            {k: sanitize(v) for k, v in r.items()} for r in winners
        ],
        'n_passed': len(winners),
        'n_tested': len(results),
    }

    results_file = OUTPUT_DIR / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 80)
    print("DONE - Sequential Signal Chains Backtest Complete")
    print(f"  {len(winners)}/{len(results)} chain variants passed all gates")
    print("=" * 80)

    return results_data


if __name__ == '__main__':
    main()
