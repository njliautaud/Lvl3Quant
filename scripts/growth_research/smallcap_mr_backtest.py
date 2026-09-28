#!/usr/bin/env python3
"""
Small/Mid-Cap Mean Reversion Dip-Buying Backtest
=================================================
Tests whether MR dip-buying works in small/mid-cap stocks.
All 12 validated strategies were built on quality megacaps.
Small caps have different dynamics — more volatile, less efficient,
potentially MORE alpha but also more risk.

6 strategy variants:
  A) Base MR (adapted): RSI<30 + >10% below 20-SMA
  B) Sector Dip: stock drops >15% from 60d high, sector ETF holding up
  C) Volume Exhaustion: 5-day drop >10%, climax volume, next-day vol drops
  D) Small-Cap Momentum Reversal: top 20% momentum, then drops >12% in 10d
  E) Relative Value Dip: >8% below 20-SMA while IWM near its 20-SMA
  F) IV-Implied Dip: >10% below 20-SMA + VIX > 20d realized vol by 3+ pts

5-gate validation: Sharpe, WR, PF, MDD, regime gap (<0.50), perm test (p<0.05)
Regime: bull/bear by SPY close-to-close (same as megacap framework).
"""

import os, sys, json, warnings, time, functools
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

# ─── Config ───
START_DATE = '2019-01-01'
END_DATE = '2026-07-01'
BACKTEST_START = '2020-01-01'
POS_SIZE = 300.0
MAX_CONCURRENT = 2
HOLD_DAYS = 21
PROFIT_TARGET = 0.15
STOP_LOSS = -0.20
SPREAD_COST_PCT = 0.002  # 20bps RT — wider spreads in small caps
N_PERMS = 200
REGIME_GAP_LIMIT = 0.50

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/smallcap_mr')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    # Mining/Metals
    'FCX', 'NEM', 'CLF', 'AA', 'GOLD', 'AEM',
    # Equipment/Industrial
    'CAT', 'DE', 'PCAR', 'CMI', 'EMR',
    # Healthcare/Biotech
    'ISRG', 'VRTX', 'DXCM', 'REGN', 'MRNA', 'GILD',
    # Small-cap diversified
    'ROKU', 'SNAP', 'CRSP', 'PLUG', 'ENPH', 'SEDG',
    # Mid-cap tech
    'FTNT', 'PANW', 'DDOG', 'NET', 'ZS', 'BILL',
]

MACRO_TICKERS = ['SPY', 'IWM', '^VIX', '^VIX3M', 'TLT', '^TNX']

# Sector ETF mappings for Strategy B
SECTOR_ETF_MAP = {
    'FCX': 'XME', 'NEM': 'XME', 'CLF': 'XME', 'AA': 'XME', 'GOLD': 'XME', 'AEM': 'XME',
    'CAT': 'XLI', 'DE': 'XLI', 'PCAR': 'XLI', 'CMI': 'XLI', 'EMR': 'XLI',
    'ISRG': 'XLV', 'VRTX': 'XLV', 'DXCM': 'XLV', 'REGN': 'XBI', 'MRNA': 'XBI', 'GILD': 'XBI',
    'ROKU': 'IWM', 'SNAP': 'IWM', 'CRSP': 'IWM', 'PLUG': 'IWM', 'ENPH': 'IWM', 'SEDG': 'IWM',
    'FTNT': 'IWM', 'PANW': 'IWM', 'DDOG': 'IWM', 'NET': 'IWM', 'ZS': 'IWM', 'BILL': 'IWM',
}
SECTOR_ETFS = list(set(SECTOR_ETF_MAP.values()))  # XME, XLI, XLV, XBI, IWM

print("=" * 80)
print("SMALL/MID-CAP MEAN REVERSION DIP-BUYING BACKTEST")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════
def download_data():
    import yfinance as yf
    cache_file = OUTPUT_DIR / '_smallcap_mr_cache.pkl'
    if cache_file.exists():
        import pickle
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded cache: {len(data['close'].columns)} tickers, {len(data['close'])} days")
        return data

    all_tickers = list(set(UNIVERSE + MACRO_TICKERS + SECTOR_ETFS))
    print(f"\n[1] Downloading {len(all_tickers)} tickers...")

    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
        high = raw['High']
        low = raw['Low']
        volume = raw['Volume']
    else:
        close = high = low = volume = raw

    # Clean column names if multi-level
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

    import pickle
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Downloaded: {len(close.columns)} tickers, {len(close)} days")
    return data

# ═══════════════════════════════════════════════════════════════════════
# 2. INDICATOR HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _rsi(series, period=14):
    """Standard RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _sma(series, period):
    return series.rolling(period).mean()

def _realized_vol(returns, window=20):
    """Annualized realized vol (in pct points like VIX)."""
    return returns.rolling(window).std() * np.sqrt(252) * 100

# ═══════════════════════════════════════════════════════════════════════
# 3. SIGNAL GENERATORS (6 variants)
# ═══════════════════════════════════════════════════════════════════════

def signal_a_base_mr(data, stock_tickers):
    """A) Base MR (adapted): RSI(14) < 30 AND stock > 10% below 20-SMA."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        if len(close) < 30:
            continue
        rsi = _rsi(close, 14)
        sma20 = _sma(close, 20)
        pct_below_sma = (close - sma20) / sma20
        for i in range(20, len(close)):
            if pd.isna(rsi.iloc[i]) or pd.isna(sma20.iloc[i]):
                continue
            if rsi.iloc[i] < 30 and pct_below_sma.iloc[i] < -0.10:
                signals[(close.index[i], t)] = True
    return signals, 'A) Base MR'

def signal_b_sector_dip(data, stock_tickers):
    """B) Sector Dip: stock drops >15% from 60d high AND sector ETF NOT down >10%."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        if len(close) < 65:
            continue
        high_60 = close.rolling(60).max()
        stock_dd = (close - high_60) / high_60

        etf_ticker = SECTOR_ETF_MAP.get(t)
        if etf_ticker is None or etf_ticker not in data['close'].columns:
            continue
        etf_close = data['close'][etf_ticker].dropna()
        etf_high_60 = etf_close.rolling(60).max()
        etf_dd = (etf_close - etf_high_60) / etf_high_60

        # Align indices
        common = close.index.intersection(etf_close.index)
        for day in common:
            if day not in stock_dd.index or day not in etf_dd.index:
                continue
            s_dd = stock_dd.get(day, np.nan)
            e_dd = etf_dd.get(day, np.nan)
            if pd.isna(s_dd) or pd.isna(e_dd):
                continue
            # Stock down >15% from 60d high, sector ETF NOT down >10%
            if s_dd < -0.15 and e_dd > -0.10:
                signals[(day, t)] = True
    return signals, 'B) Sector Dip'

def signal_c_volume_exhaustion(data, stock_tickers):
    """C) Volume Exhaustion: stock drops >10% over 5 days, last down-day volume >3x 20d avg,
    next day volume drops below average (selling climax)."""
    signals = {}
    for t in stock_tickers:
        if t not in data['close'].columns or t not in data['volume'].columns:
            continue
        close = data['close'][t].dropna()
        volume = data['volume'][t].dropna()
        if len(close) < 30:
            continue
        # Align
        common = close.index.intersection(volume.index)
        close = close.loc[common]
        volume = volume.loc[common]

        ret_5d = close.pct_change(5)
        vol_sma20 = _sma(volume, 20)

        for i in range(25, len(common) - 1):
            if pd.isna(ret_5d.iloc[i]) or pd.isna(vol_sma20.iloc[i]):
                continue
            if vol_sma20.iloc[i] <= 0:
                continue
            # Stock dropped >10% over 5 days
            if ret_5d.iloc[i] < -0.10:
                # Last day is a down day
                day_ret = close.iloc[i] / close.iloc[i-1] - 1
                if day_ret < 0:
                    # Volume on this day >3x 20d average
                    if volume.iloc[i] > 3.0 * vol_sma20.iloc[i]:
                        # Next day volume drops below average
                        if i + 1 < len(common):
                            if volume.iloc[i+1] < vol_sma20.iloc[i]:
                                # Signal fires on the NEXT day (when vol drops)
                                signals[(common[i+1], t)] = True
    return signals, 'C) Volume Exhaustion'

def signal_d_momentum_reversal(data, stock_tickers):
    """D) Small-Cap Momentum Reversal: stock was in top 20% by 60d momentum,
    then drops >12% in 10 days. 'Fallen angel' pattern."""
    signals = {}
    close_df = data['close']

    # Compute 60d returns for all stocks
    valid_tickers = [t for t in stock_tickers if t in close_df.columns]
    if len(valid_tickers) < 5:
        return signals, 'D) Momentum Reversal'

    ret_60d = close_df[valid_tickers].pct_change(60)
    ret_10d = close_df[valid_tickers].pct_change(10)

    for i in range(70, len(close_df)):
        day = close_df.index[i]
        # Get 60d returns as of 10 days ago (when they WERE leaders)
        if i - 10 < 0:
            continue
        prior_idx = i - 10
        prior_rets = ret_60d.iloc[prior_idx]
        valid_rets = prior_rets.dropna()
        if len(valid_rets) < 5:
            continue

        # Top 20% threshold
        threshold = valid_rets.quantile(0.80)

        for t in valid_tickers:
            if pd.isna(prior_rets.get(t, np.nan)):
                continue
            # Was in top 20% by momentum 10 days ago
            if prior_rets[t] >= threshold:
                # Now dropped >12% in last 10 days
                current_10d = ret_10d.iloc[i].get(t, np.nan) if i < len(ret_10d) else np.nan
                if pd.notna(current_10d) and current_10d < -0.12:
                    signals[(day, t)] = True
    return signals, 'D) Momentum Reversal'

def signal_e_relative_value_dip(data, stock_tickers):
    """E) Relative Value Dip: stock >8% below 20-SMA while IWM within 3% of its 20-SMA."""
    signals = {}
    if 'IWM' not in data['close'].columns:
        return signals, 'E) Relative Value Dip'

    iwm = data['close']['IWM'].dropna()
    iwm_sma20 = _sma(iwm, 20)
    iwm_pct = (iwm - iwm_sma20) / iwm_sma20

    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        if len(close) < 25:
            continue
        sma20 = _sma(close, 20)
        pct_below = (close - sma20) / sma20

        common = close.index.intersection(iwm.index)
        for day in common:
            s_pct = pct_below.get(day, np.nan)
            i_pct = iwm_pct.get(day, np.nan)
            if pd.isna(s_pct) or pd.isna(i_pct):
                continue
            # Stock >8% below its 20-SMA
            if s_pct < -0.08:
                # IWM within 3% of its 20-SMA (not broken down)
                if abs(i_pct) < 0.03:
                    signals[(day, t)] = True
    return signals, 'E) Relative Value Dip'

def signal_f_iv_implied_dip(data, stock_tickers):
    """F) IV-Implied Dip: stock >10% below 20-SMA AND VIX > 20d realized vol by 3+ pts."""
    signals = {}
    if '^VIX' not in data['close'].columns or 'SPY' not in data['close'].columns:
        return signals, 'F) IV-Implied Dip'

    spy = data['close']['SPY'].dropna()
    vix = data['close']['^VIX'].dropna()
    spy_ret = spy.pct_change()
    rv_20 = _realized_vol(spy_ret, 20)
    iv_rv_gap = vix - rv_20

    for t in stock_tickers:
        if t not in data['close'].columns:
            continue
        close = data['close'][t].dropna()
        if len(close) < 25:
            continue
        sma20 = _sma(close, 20)
        pct_below = (close - sma20) / sma20

        common = close.index.intersection(vix.index)
        for day in common:
            s_pct = pct_below.get(day, np.nan)
            gap = iv_rv_gap.get(day, np.nan)
            if pd.isna(s_pct) or pd.isna(gap):
                continue
            # Stock >10% below 20-SMA AND VIX-RV gap > 3 pts
            if s_pct < -0.10 and gap > 3.0:
                signals[(day, t)] = True
    return signals, 'F) IV-Implied Dip'

# ═══════════════════════════════════════════════════════════════════════
# 4. BACKTESTER
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(signal_entries, data, stock_tickers, label=''):
    """Run backtest on signal entries {(date, ticker): True}."""
    close = data['close']
    spy_close = close.get('SPY')

    bt_start = pd.Timestamp(BACKTEST_START)
    entries = sorted(
        [(d, t) for (d, t) in signal_entries if d >= bt_start],
        key=lambda x: x[0]
    )

    if not entries:
        return None

    trades = []
    open_positions = []

    # Pre-compute SPY 200-SMA for regime
    spy_sma200 = spy_close.rolling(200).mean() if spy_close is not None else None

    for entry_date, ticker in entries:
        # Check max concurrent — drop expired positions
        open_positions = [p for p in open_positions if p[3] > entry_date]
        if len(open_positions) >= MAX_CONCURRENT:
            continue

        try:
            entry_price = close.at[entry_date, ticker]
        except (KeyError, TypeError):
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

        for fdate in future_dates[:HOLD_DAYS]:
            try:
                price = close.at[fdate, ticker]
            except (KeyError, TypeError):
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
                except (KeyError, TypeError):
                    continue
                if pd.isna(exit_price):
                    continue

        if exit_price is None:
            continue

        raw_ret = (exit_price - entry_price) / entry_price
        net_ret = raw_ret - SPREAD_COST_PCT
        pnl = POS_SIZE * net_ret

        # SPY regime
        spy_regime = 'unknown'
        if spy_close is not None and spy_sma200 is not None:
            try:
                if spy_close.at[entry_date] > spy_sma200.at[entry_date]:
                    spy_regime = 'bull'
                else:
                    spy_regime = 'bear'
            except (KeyError, TypeError):
                pass

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'ticker': ticker,
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'return': float(net_ret),
            'pnl': float(pnl),
            'exit_reason': exit_reason,
            'regime': spy_regime,
            'hold_days': (exit_date - entry_date).days,
        })

        open_positions.append((entry_date, ticker, entry_price, exit_date))

    if not trades:
        return None

    return pd.DataFrame(trades)

# ═══════════════════════════════════════════════════════════════════════
# 5. VALIDATION (5-gate + permutation)
# ═══════════════════════════════════════════════════════════════════════

def validate_strategy(trades_df, label='', run_perm=True,
                      all_signal_entries=None, data=None, stock_tickers=None):
    """5-gate validation: Sharpe, WR, PF, MDD, regime gap, perm test."""
    if trades_df is None or len(trades_df) < 10:
        return {
            'label': label,
            'n_trades': 0 if trades_df is None else len(trades_df),
            'sharpe': 0, 'sortino': 0, 'wr': 0, 'pf': 0, 'mdd': 0,
            'regime_gap': 1.0, 'perm_p': 1.0, 'passed': False,
            'reason': 'insufficient trades (<10)',
            'failed_gates': ['insufficient'],
            'mean_ret_pct': 0, 'bull_n': 0, 'bear_n': 0, 'avg_hold': 0,
        }

    rets = trades_df['return'].values
    n = len(rets)

    # Sharpe (annualized by trade frequency)
    years = max(0.5, (trades_df['entry_date'].max() - trades_df['entry_date'].min()).days / 365.25)
    trades_per_year = n / years
    if trades_per_year < 1:
        trades_per_year = 1
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1.0
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

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
    mdd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Regime gap (SPY-based)
    bull_trades = trades_df[trades_df['regime'] == 'bull']
    bear_trades = trades_df[trades_df['regime'] == 'bear']
    if len(bull_trades) >= 3 and len(bear_trades) >= 3:
        bull_sr = np.mean(bull_trades['return']) / max(np.std(bull_trades['return'], ddof=1), 1e-6)
        bear_sr = np.mean(bear_trades['return']) / max(np.std(bear_trades['return'], ddof=1), 1e-6)
        max_abs = max(abs(bull_sr), abs(bear_sr), 1e-6)
        regime_gap = abs(bull_sr - bear_sr) / max_abs
    else:
        regime_gap = 0.0  # can't assess

    # Permutation test — random entry timing
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
                r_yrs = max(0.5, (rand_trades['entry_date'].max() - rand_trades['entry_date'].min()).days / 365.25)
                r_tpy = len(r_rets) / r_yrs
                if r_tpy < 1:
                    r_tpy = 1
                r_std = np.std(r_rets, ddof=1)
                r_sharpe = (np.mean(r_rets) / r_std) * np.sqrt(r_tpy) if r_std > 0 else 0
            else:
                r_sharpe = 0.0
            perm_sharpes.append(r_sharpe)

        perm_p = float(np.mean(np.array(perm_sharpes) >= actual_sharpe))

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
        'n_trades': int(n),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'wr': round(float(wr), 3),
        'pf': round(float(pf), 3),
        'mdd': round(float(mdd), 2),
        'regime_gap': round(float(regime_gap), 3),
        'perm_p': round(float(perm_p), 4),
        'passed': bool(passed),
        'failed_gates': failed_gates,
        'reason': 'PASSED' if passed else f"FAILED: {', '.join(failed_gates)}",
        'mean_ret_pct': round(float(mean_ret * 100), 2),
        'bull_n': int(len(bull_trades)),
        'bear_n': int(len(bear_trades)),
        'avg_hold': round(float(trades_df['hold_days'].mean()), 1),
    }

# ═══════════════════════════════════════════════════════════════════════
# 6. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    data = download_data()
    stock_tickers = [t for t in UNIVERSE if t in data['close'].columns]
    print(f"  Stock universe: {len(stock_tickers)}/{len(UNIVERSE)} tickers available")

    missing = [t for t in UNIVERSE if t not in data['close'].columns]
    if missing:
        print(f"  Missing tickers: {missing}")

    # Check macro tickers
    for mt in MACRO_TICKERS + SECTOR_ETFS:
        if mt not in data['close'].columns:
            print(f"  WARNING: macro/sector ticker {mt} not in data")

    # ── Generate all signals ──
    print("\n[2] Generating signals for 6 strategy variants...")
    signal_generators = [
        ('A', signal_a_base_mr),
        ('B', signal_b_sector_dip),
        ('C', signal_c_volume_exhaustion),
        ('D', signal_d_momentum_reversal),
        ('E', signal_e_relative_value_dip),
        ('F', signal_f_iv_implied_dip),
    ]

    all_signals = {}
    for key, gen_func in signal_generators:
        t0 = time.time()
        sigs, name = gen_func(data, stock_tickers)
        elapsed = time.time() - t0
        # Count entries in backtest period
        bt_start = pd.Timestamp(BACKTEST_START)
        bt_count = sum(1 for (d, _) in sigs if d >= bt_start)
        print(f"  {name:30s} | {len(sigs):5d} raw | {bt_count:5d} in BT window | {elapsed:.1f}s")
        all_signals[key] = sigs

    # ── Run backtests ──
    print(f"\n[3] Running backtests with {N_PERMS} permutations each...")
    print(f"    Config: POS_SIZE=${POS_SIZE:.0f}, MAX_CONCURRENT={MAX_CONCURRENT}, "
          f"HOLD={HOLD_DAYS}d, TP={PROFIT_TARGET:.0%}, SL={STOP_LOSS:.0%}, "
          f"spread={SPREAD_COST_PCT*100:.1f}bps")
    print()

    results = {}
    for key, gen_func in signal_generators:
        sigs = all_signals[key]
        _, name = gen_func.__doc__.split(')')[0] + ')', gen_func.__doc__.split(':')[0]
        label = gen_func(data, stock_tickers)[1]  # get the label

        t0 = time.time()
        trades_df = run_backtest(sigs, data, stock_tickers, label=label)
        result = validate_strategy(
            trades_df, label=label, run_perm=True,
            all_signal_entries=sigs, data=data, stock_tickers=stock_tickers
        )
        elapsed = time.time() - t0

        results[key] = result
        status = "PASS ✓" if result['passed'] else "FAIL"
        print(f"  {result['label']:30s} | n={result['n_trades']:4d} | "
              f"Sharpe={result['sharpe']:7.3f} | Sortino={result['sortino']:7.3f} | "
              f"WR={result['wr']:.3f} | PF={result['pf']:6.3f} | "
              f"MDD=${result['mdd']:8.2f} | RG={result['regime_gap']:.3f} | "
              f"p={result['perm_p']:.4f} | {status} ({elapsed:.0f}s)")

        # Print trade breakdown if trades exist
        if trades_df is not None and len(trades_df) > 0:
            exits = trades_df['exit_reason'].value_counts()
            exit_str = ', '.join(f"{r}={c}" for r, c in exits.items())
            top_tickers = trades_df['ticker'].value_counts().head(5)
            ticker_str = ', '.join(f"{t}={c}" for t, c in top_tickers.items())
            print(f"    Exits: {exit_str}")
            print(f"    Top tickers: {ticker_str}")
            print(f"    Bull trades: {result['bull_n']}, Bear trades: {result['bear_n']}, "
                  f"Avg hold: {result['avg_hold']:.1f}d, Mean ret: {result['mean_ret_pct']:+.2f}%")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════
    elapsed_total = time.time() - t_start

    print("\n" + "=" * 120)
    print("SMALL/MID-CAP MEAN REVERSION — RESULTS SUMMARY")
    print("=" * 120)
    print(f"{'Strategy':30s} | {'N':>5s} | {'Sharpe':>7s} | {'Sortino':>7s} | {'WR':>5s} | "
          f"{'PF':>6s} | {'MDD':>9s} | {'RegGap':>6s} | {'Perm p':>7s} | {'AvgHold':>7s} | {'Status':>6s}")
    print("-" * 120)

    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = results[key]
        status = "PASS" if r['passed'] else "FAIL"
        print(f"{r['label']:30s} | {r['n_trades']:5d} | {r['sharpe']:7.3f} | {r['sortino']:7.3f} | "
              f"{r['wr']:.3f} | {r['pf']:6.3f} | {r['mdd']:9.2f} | {r['regime_gap']:6.3f} | "
              f"{r['perm_p']:7.4f} | {r['avg_hold']:5.1f}d | {status:>6s}")

    # Winners
    winners = {k: v for k, v in results.items() if v['passed']}
    print(f"\n── WINNERS (passed all 5 gates + permutation test) ──")
    if winners:
        for k, r in winners.items():
            print(f"  ** {r['label']}: Sharpe={r['sharpe']:.3f}, Sortino={r['sortino']:.3f}, "
                  f"WR={r['wr']:.1%}, PF={r['pf']:.2f}, perm_p={r['perm_p']:.4f}")
    else:
        print("  No strategies passed ALL gates.")
        # Near misses
        near = [(k, r) for k, r in results.items()
                if r['n_trades'] >= 10 and r['sharpe'] > 0.2]
        near.sort(key=lambda x: x[1]['sharpe'], reverse=True)
        if near:
            print("  Near-misses (Sharpe > 0.2, n >= 10):")
            for k, r in near:
                print(f"    {r['label']}: Sharpe={r['sharpe']:.3f}, WR={r['wr']:.1%}, "
                      f"p={r['perm_p']:.4f}, failed={r['failed_gates']}")

    # Comparison note
    print(f"\n── COMPARISON NOTE ──")
    print(f"  These small/mid-cap results use WIDER thresholds than megacap:")
    print(f"    TP={PROFIT_TARGET:.0%} vs 10% megacap | SL={STOP_LOSS:.0%} vs -15% megacap")
    print(f"    Spread={SPREAD_COST_PCT*10000:.0f}bps vs 10bps megacap")
    print(f"    RSI<30 vs RSI<35 megacap | >10% below SMA vs >5% megacap")
    print(f"  If no strategies pass, small-cap MR may require different holding periods")
    print(f"  or position sizing to compensate for higher volatility and wider spreads.")

    # ── Save results ──
    results_data = {
        'run_date': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed_total, 1),
        'config': {
            'universe': UNIVERSE,
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
        'strategies': {k: v for k, v in results.items()},
        'winners': list(winners.keys()) if winners else [],
        'summary': {
            'total_strategies': 6,
            'passed': len(winners),
            'best_sharpe': max(r['sharpe'] for r in results.values()),
            'best_strategy': max(results.items(), key=lambda x: x[1]['sharpe'])[1]['label'],
        },
    }

    results_file = OUTPUT_DIR / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    print(f"\nTotal runtime: {elapsed_total:.0f}s")
    print("=" * 80)
    print("DONE — Small/Mid-Cap Mean Reversion Backtest Complete")
    print("=" * 80)

    return results_data


if __name__ == '__main__':
    main()
