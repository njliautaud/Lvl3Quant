#!/usr/bin/env python3
"""
Flow-Signal Confluence Backtest
================================
Tests OPTIONS FLOW proxy signals paired with dip-buying signals.
Theory: a dip + unusual options flow (smart money positioning) = stronger signal.

Flow proxies (from price/volume data):
  1. Unusual Volume Surge: volume > 2x 20d avg on a dip day
  2. Put/Call Proxy: large gap-down with quick recovery (range > 2x ATR, close near high)
  3. Accumulation Pattern: OBV rising while price falling
  4. Volatility Crush: 20d realized vol drops >20% while stock still in dip
  5. Relative Volume Dip: down-day volume < 50% of up-day volume over 10d window

Dip signals:
  A. IV-RV Gap: VIX > SPY 20d realized vol by 5+ pts
  B. RSI Divergence: price lower low + RSI higher low
  C. Base MR: RSI(14) < 30 AND >7% below 52w high

Test matrix: 5 flow x 3 dip = 15 combos + 1 kitchen sink
5-gate: Sharpe>0.5, WR>55%, PF>1.5, regime gap<0.50, perm p<0.05
"""

import os, sys, json, warnings, functools, pickle, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

# ─── Config ───
START_DATE = '2019-01-01'  # extra lookback for indicators
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
CONFLUENCE_WINDOW = 3  # days

OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/flow_confluence')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'ADBE',
]

MACRO_TICKERS = ['SPY', '^VIX']

print("=" * 80)
print("FLOW-SIGNAL CONFLUENCE BACKTEST")
print("=" * 80)
print(f"Universe: {len(UNIVERSE)} stocks | Period: {BACKTEST_START} to {END_DATE}")
print(f"Position: ${POS_SIZE} | Max concurrent: {MAX_CONCURRENT}")
print(f"Exit: {HOLD_DAYS}d hold / {PROFIT_TARGET:.0%} TP / {abs(STOP_LOSS):.0%} SL")
print(f"Confluence window: {CONFLUENCE_WINDOW} days")
print()

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════
def download_data():
    import yfinance as yf
    cache_file = OUTPUT_DIR / '_flow_confluence_cache.pkl'
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded cache: {len(data['close'].columns)} stocks, {len(data['close'])} days")
        return data

    print(f"[1] Downloading {len(UNIVERSE)} stocks + macro tickers...")
    all_tickers = UNIVERSE + MACRO_TICKERS

    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False, threads=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
        high = raw['High']
        low = raw['Low']
        volume = raw['Volume']
        opn = raw['Open']
    else:
        close = raw[['Close']].rename(columns={'Close': all_tickers[0]})
        high = raw[['High']].rename(columns={'High': all_tickers[0]})
        low = raw[['Low']].rename(columns={'Low': all_tickers[0]})
        volume = raw[['Volume']].rename(columns={'Volume': all_tickers[0]})
        opn = raw[['Open']].rename(columns={'Open': all_tickers[0]})

    # Flatten multi-index if needed
    for df in [close, high, low, volume, opn]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

    data = {
        'close': close, 'high': high, 'low': low,
        'volume': volume, 'open': opn,
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Downloaded: {len(close.columns)} tickers, {len(close)} days")
    return data

# ═══════════════════════════════════════════════════════════════════════
# 2. INDICATOR COMPUTATION
# ═══════════════════════════════════════════════════════════════════════
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_obv(close, volume):
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_realized_vol(close, window=20):
    """Annualized realized vol from log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252) * 100  # in pct points

def compute_indicators(data):
    """Compute all needed indicators for each stock."""
    close = data['close']
    high = data['high']
    low = data['low']
    volume = data['volume']
    opn = data['open']

    indicators = {}
    stocks = [t for t in UNIVERSE if t in close.columns]

    for ticker in stocks:
        c = close[ticker].dropna()
        h = high[ticker].reindex(c.index)
        l = low[ticker].reindex(c.index)
        v = volume[ticker].reindex(c.index)
        o = opn[ticker].reindex(c.index)

        ind = pd.DataFrame(index=c.index)
        ind['close'] = c
        ind['high'] = h
        ind['low'] = l
        ind['open'] = o
        ind['volume'] = v
        ind['ret'] = c.pct_change()

        # RSI
        ind['rsi'] = compute_rsi(c, 14)

        # ATR
        ind['atr'] = compute_atr(h, l, c, 14)

        # OBV
        ind['obv'] = compute_obv(c, v)

        # Volume averages
        ind['vol_20d_avg'] = v.rolling(20).mean()
        ind['vol_ratio'] = v / ind['vol_20d_avg'].replace(0, np.nan)

        # 52-week high
        ind['high_52w'] = c.rolling(252).max()
        ind['pct_from_52w_high'] = (c - ind['high_52w']) / ind['high_52w']

        # Realized vol (20d)
        ind['rvol_20d'] = compute_realized_vol(c, 20)
        ind['rvol_20d_prev'] = ind['rvol_20d'].shift(5)  # 5d ago for crush detection

        # Daily range relative to ATR
        ind['range_atr'] = (h - l) / ind['atr'].replace(0, np.nan)

        # Close position within day's range
        day_range = h - l
        ind['close_position'] = (c - l) / day_range.replace(0, np.nan)

        # Up/down day volumes
        ind['is_up'] = (c > c.shift(1)).astype(float)
        ind['is_down'] = (c < c.shift(1)).astype(float)
        up_vol = (v * ind['is_up']).rolling(10).sum()
        down_vol = (v * ind['is_down']).rolling(10).sum()
        ind['down_up_vol_ratio'] = down_vol / up_vol.replace(0, np.nan)

        # OBV trend (5d slope)
        obv = ind['obv']
        ind['obv_slope_5d'] = obv.rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0,
            raw=True
        )
        # Price slope (5d)
        ind['price_slope_5d'] = c.rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0,
            raw=True
        )

        # Gap from open
        ind['gap_pct'] = (o - c.shift(1)) / c.shift(1)

        indicators[ticker] = ind

    return indicators, stocks

# ═══════════════════════════════════════════════════════════════════════
# 3. SIGNAL GENERATORS
# ═══════════════════════════════════════════════════════════════════════

# --- FLOW PROXY SIGNALS ---

def flow_unusual_volume(ind):
    """1. Volume > 2x 20d avg on a down day."""
    return (ind['vol_ratio'] > 2.0) & (ind['ret'] < 0)

def flow_putcall_proxy(ind):
    """2. Large gap-down with quick intraday recovery.
    Range > 2x ATR but close in top 30% of range."""
    return (
        (ind['gap_pct'] < -0.01) &      # gapped down >1%
        (ind['range_atr'] > 2.0) &       # wide range day
        (ind['close_position'] > 0.70)   # closed near high (call buying recovery)
    )

def flow_accumulation(ind):
    """3. OBV rising while price falling over 5d (smart money accumulation)."""
    return (ind['obv_slope_5d'] > 0) & (ind['price_slope_5d'] < 0) & (ind['ret'] < -0.005)

def flow_vol_crush(ind):
    """4. Realized vol drops >20% while stock is in a dip (>5% below 52w high)."""
    vol_change = (ind['rvol_20d'] - ind['rvol_20d_prev']) / ind['rvol_20d_prev'].replace(0, np.nan)
    return (vol_change < -0.20) & (ind['pct_from_52w_high'] < -0.05)

def flow_relative_volume_dip(ind):
    """5. Down-day volume < 50% of up-day volume over 10d window."""
    return (ind['down_up_vol_ratio'] < 0.50) & (ind['ret'] < -0.005)

FLOW_SIGNALS = {
    '1_VolSurge': flow_unusual_volume,
    '2_PCProxy': flow_putcall_proxy,
    '3_Accum': flow_accumulation,
    '4_VolCrush': flow_vol_crush,
    '5_RelVol': flow_relative_volume_dip,
}

# --- DIP SIGNALS ---

def dip_iv_rv_gap(ind, spy_data):
    """A. VIX > SPY 20d realized vol by 5+ pts."""
    # Returns a series indexed like ind
    vix = spy_data['vix'].reindex(ind.index)
    spy_rvol = spy_data['spy_rvol'].reindex(ind.index)
    gap = vix - spy_rvol
    return gap > 5.0

def dip_rsi_divergence(ind):
    """B. Price lower low + RSI higher low (bullish divergence) over 10d lookback."""
    price_ll = ind['close'] < ind['close'].rolling(10).min().shift(1)
    rsi_hl = ind['rsi'] > ind['rsi'].rolling(10).min().shift(1)
    return price_ll & rsi_hl & (ind['rsi'] < 40)  # only when RSI is low-ish

def dip_base_mr(ind):
    """C. RSI(14) < 30 AND >7% below 52-week high."""
    return (ind['rsi'] < 30) & (ind['pct_from_52w_high'] < -0.07)


# ═══════════════════════════════════════════════════════════════════════
# 4. BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════

def run_backtest(signal_dates, price_data, spy_close, label=""):
    """
    Run backtest on a list of (date, ticker) entry signals.
    Returns dict of metrics.
    """
    if not signal_dates:
        return None

    trades = []
    active_positions = []  # list of (exit_date, ticker)

    # Sort by date
    signal_dates.sort(key=lambda x: x[0])

    for entry_date, ticker in signal_dates:
        if entry_date < pd.Timestamp(BACKTEST_START):
            continue

        # Check concurrent limit
        active_positions = [(ed, tk) for ed, tk in active_positions if ed > entry_date]
        if len(active_positions) >= MAX_CONCURRENT:
            continue

        # Get price series from entry
        if ticker not in price_data.columns:
            continue
        prices = price_data[ticker]
        if entry_date not in prices.index:
            # Find next available date
            future = prices.index[prices.index >= entry_date]
            if len(future) == 0:
                continue
            entry_date = future[0]

        entry_price = prices.loc[entry_date]
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # Simulate hold period
        future_idx = prices.index.get_loc(entry_date)
        exit_price = None
        exit_date = None
        exit_reason = 'hold'

        for d in range(1, HOLD_DAYS + 1):
            if future_idx + d >= len(prices):
                break
            p = prices.iloc[future_idx + d]
            ret = (p - entry_price) / entry_price

            if ret >= PROFIT_TARGET:
                exit_price = p
                exit_date = prices.index[future_idx + d]
                exit_reason = 'TP'
                break
            elif ret <= STOP_LOSS:
                exit_price = p
                exit_date = prices.index[future_idx + d]
                exit_reason = 'SL'
                break

        if exit_price is None:
            last_idx = min(future_idx + HOLD_DAYS, len(prices) - 1)
            exit_price = prices.iloc[last_idx]
            exit_date = prices.index[last_idx]

        gross_ret = (exit_price - entry_price) / entry_price
        net_ret = gross_ret - SPREAD_COST_PCT
        pnl = POS_SIZE * net_ret

        # Regime: SPY return over trade period
        spy_entry = spy_close.asof(entry_date) if entry_date in spy_close.index or True else np.nan
        spy_exit = spy_close.asof(exit_date) if exit_date is not None else np.nan
        spy_ret = (spy_exit - spy_entry) / spy_entry if spy_entry > 0 else 0
        regime = 'bull' if spy_ret > 0.005 else ('bear' if spy_ret < -0.005 else 'flat')

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'ticker': ticker,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'gross_ret': gross_ret,
            'net_ret': net_ret,
            'pnl': pnl,
            'exit_reason': exit_reason,
            'regime': regime,
        })

        active_positions.append((exit_date, ticker))

    if len(trades) < 5:
        return None

    df = pd.DataFrame(trades)
    returns = df['net_ret'].values

    # Metrics
    n_trades = len(df)
    win_rate = (returns > 0).mean()
    total_pnl = df['pnl'].sum()
    avg_ret = returns.mean()

    gross_wins = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else 99.0

    # Sharpe (annualized from per-trade returns, ~12 trades/yr assumption)
    if returns.std() > 0:
        trades_per_year = max(n_trades / max((df['entry_date'].max() - df['entry_date'].min()).days / 365.25, 0.5), 1)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 1:
        downside_std = downside.std()
        sortino = (returns.mean() / downside_std) * np.sqrt(max(n_trades / max((df['entry_date'].max() - df['entry_date'].min()).days / 365.25, 0.5), 1)) if downside_std > 0 else 0
    else:
        sortino = sharpe * 1.5  # approximate

    # Max drawdown (cumulative PnL)
    cum_pnl = df['pnl'].cumsum()
    peak = cum_pnl.cummax()
    dd = cum_pnl - peak
    max_dd = dd.min()

    # Regime analysis
    regime_sharpes = {}
    for regime in ['bull', 'bear', 'flat']:
        r = returns[df['regime'].values == regime]
        if len(r) >= 3 and r.std() > 0:
            regime_sharpes[regime] = r.mean() / r.std()
        else:
            regime_sharpes[regime] = np.nan

    bull_s = regime_sharpes.get('bull', np.nan)
    bear_s = regime_sharpes.get('bear', np.nan)
    if not np.isnan(bull_s) and not np.isnan(bear_s):
        max_s = max(abs(bull_s), abs(bear_s))
        regime_gap = abs(bull_s - bear_s) / max_s if max_s > 0 else 0
    else:
        regime_gap = np.nan

    return {
        'label': label,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_ret': avg_ret,
        'total_pnl': total_pnl,
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': profit_factor,
        'max_dd': max_dd,
        'regime_gap': regime_gap,
        'regime_sharpes': regime_sharpes,
        'trades_df': df,
        'returns': returns,
    }

def permutation_test(returns, n_perms=200):
    """Permutation test: p-value for mean return being > 0."""
    if len(returns) < 5:
        return 1.0
    observed = returns.mean()
    count = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        shuffled = returns.copy()
        rng.shuffle(shuffled)
        # Randomly flip signs to test if direction matters
        signs = rng.choice([-1, 1], size=len(shuffled))
        if (shuffled * signs).mean() >= observed:
            count += 1
    return count / n_perms

def five_gate_check(result):
    """Apply 5-gate validation. Returns (pass, details)."""
    if result is None:
        return False, "No trades"

    gates = {
        'Sharpe > 0.5': result['sharpe'] > 0.5,
        'WR > 55%': result['win_rate'] > 0.55,
        'PF > 1.5': result['profit_factor'] > 1.5,
        'Regime gap < 0.50': (np.isnan(result['regime_gap']) or result['regime_gap'] < REGIME_GAP_LIMIT),
    }

    # Permutation test
    p_val = permutation_test(result['returns'], N_PERMS)
    gates['Perm p < 0.05'] = p_val < 0.05
    result['perm_p'] = p_val

    passed = all(gates.values())
    detail_str = ' | '.join(f"{'PASS' if v else 'FAIL'} {k}" for k, v in gates.items())
    return passed, detail_str

# ═══════════════════════════════════════════════════════════════════════
# 5. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # Download data
    data = download_data()
    close = data['close']

    # SPY data for regime + dip signals
    spy_close = close['SPY'].dropna() if 'SPY' in close.columns else pd.Series(dtype=float)
    vix = close['^VIX'].dropna() if '^VIX' in close.columns else pd.Series(dtype=float)

    spy_rvol = compute_realized_vol(spy_close, 20)
    spy_data = {
        'vix': vix,
        'spy_rvol': spy_rvol,
    }

    # Compute indicators
    print("\n[2] Computing indicators for all stocks...")
    indicators, stocks = compute_indicators(data)
    print(f"  Computed indicators for {len(stocks)} stocks")

    # Generate all signals
    print("\n[3] Generating flow and dip signals...")

    # Store signals as dict of {ticker: boolean series}
    flow_signals_all = {}
    for fname, ffunc in FLOW_SIGNALS.items():
        flow_signals_all[fname] = {}
        for ticker in stocks:
            flow_signals_all[fname][ticker] = ffunc(indicators[ticker])

    dip_signals_all = {}
    # A. IV-RV Gap (market-level, same for all stocks)
    dip_signals_all['A_IVRV'] = {}
    for ticker in stocks:
        dip_signals_all['A_IVRV'][ticker] = dip_iv_rv_gap(indicators[ticker], spy_data)

    # B. RSI Divergence
    dip_signals_all['B_RSIDiv'] = {}
    for ticker in stocks:
        dip_signals_all['B_RSIDiv'][ticker] = dip_rsi_divergence(indicators[ticker])

    # C. Base MR
    dip_signals_all['C_BaseMR'] = {}
    for ticker in stocks:
        dip_signals_all['C_BaseMR'][ticker] = dip_base_mr(indicators[ticker])

    # Count signals
    for fname in FLOW_SIGNALS:
        total = sum(s.sum() for s in flow_signals_all[fname].values())
        print(f"  Flow {fname}: {int(total)} signals")
    for dname in dip_signals_all:
        total = sum(s.sum() for s in dip_signals_all[dname].values())
        print(f"  Dip  {dname}: {int(total)} signals")

    # ─── Build confluence entries ───
    print("\n[4] Building confluence pairs (3-day window)...")

    results_all = []

    # Solo baselines first
    print("\n  --- SOLO BASELINES ---")
    for fname in FLOW_SIGNALS:
        entries = []
        for ticker in stocks:
            sig = flow_signals_all[fname][ticker]
            dates = sig[sig].index.tolist()
            entries.extend([(d, ticker) for d in dates])
        res = run_backtest(entries, close, spy_close, label=f"Solo_{fname}")
        if res:
            passed, detail = five_gate_check(res)
            results_all.append(res)
            print(f"  {fname:20s}: {res['n_trades']:4d} trades | Sharpe {res['sharpe']:+.2f} | WR {res['win_rate']:.1%} | PF {res['profit_factor']:.2f} | {'5G-PASS' if passed else '5G-FAIL'}")

    for dname in dip_signals_all:
        entries = []
        for ticker in stocks:
            sig = dip_signals_all[dname][ticker]
            dates = sig[sig].index.tolist()
            entries.extend([(d, ticker) for d in dates])
        res = run_backtest(entries, close, spy_close, label=f"Solo_{dname}")
        if res:
            passed, detail = five_gate_check(res)
            results_all.append(res)
            print(f"  {dname:20s}: {res['n_trades']:4d} trades | Sharpe {res['sharpe']:+.2f} | WR {res['win_rate']:.1%} | PF {res['profit_factor']:.2f} | {'5G-PASS' if passed else '5G-FAIL'}")

    # ─── 15 Confluence Combos ───
    print("\n  --- CONFLUENCE PAIRS (flow x dip, 3d window) ---")

    confluence_results = []

    for fname in FLOW_SIGNALS:
        for dname in dip_signals_all:
            label = f"{fname}_x_{dname}"
            entries = []

            for ticker in stocks:
                flow_sig = flow_signals_all[fname][ticker]
                dip_sig = dip_signals_all[dname][ticker]

                flow_dates = flow_sig[flow_sig].index
                dip_dates = dip_sig[dip_sig].index

                if len(flow_dates) == 0 or len(dip_dates) == 0:
                    continue

                # For each flow signal, check if any dip signal within CONFLUENCE_WINDOW days
                for fd in flow_dates:
                    window_start = fd - pd.Timedelta(days=CONFLUENCE_WINDOW)
                    window_end = fd + pd.Timedelta(days=CONFLUENCE_WINDOW)
                    nearby_dips = dip_dates[(dip_dates >= window_start) & (dip_dates <= window_end)]
                    if len(nearby_dips) > 0:
                        # Entry on the later of the two signals
                        entry = max(fd, nearby_dips[0])
                        entries.append((entry, ticker))

            # Deduplicate (same ticker/date)
            entries = list(set(entries))

            res = run_backtest(entries, close, spy_close, label=label)
            if res:
                passed, detail = five_gate_check(res)
                res['passed_5g'] = passed
                res['gate_detail'] = detail
                confluence_results.append(res)
                results_all.append(res)

                status = "*** 5G-PASS ***" if passed else "5G-FAIL"
                print(f"  {label:35s}: {res['n_trades']:4d} trades | Sharpe {res['sharpe']:+.2f} | WR {res['win_rate']:.1%} | PF {res['profit_factor']:.2f} | {status}")
            else:
                print(f"  {label:35s}: <5 trades (skip)")

    # ─── Kitchen Sink ───
    print("\n  --- KITCHEN SINK (any 2 flow + any 1 dip, 5d window) ---")

    ks_entries = []
    KS_WINDOW = 5

    for ticker in stocks:
        # Get all flow signal dates for this ticker
        all_flow_dates = {}
        for fname in FLOW_SIGNALS:
            sig = flow_signals_all[fname][ticker]
            all_flow_dates[fname] = set(sig[sig].index.tolist())

        # Get all dip signal dates
        all_dip_dates = set()
        for dname in dip_signals_all:
            sig = dip_signals_all[dname][ticker]
            all_dip_dates.update(sig[sig].index.tolist())

        # For each date in the backtest range, check if 2+ flow signals and 1+ dip within window
        idx = indicators[ticker].index
        idx = idx[idx >= pd.Timestamp(BACKTEST_START)]

        for dt in idx:
            window_start = dt - pd.Timedelta(days=KS_WINDOW)
            window_end = dt

            # Count flow signals in window
            flow_count = 0
            for fname, fdates in all_flow_dates.items():
                if any(window_start <= d <= window_end for d in fdates):
                    flow_count += 1

            # Check dip signal in window
            has_dip = any(window_start <= d <= window_end for d in all_dip_dates)

            if flow_count >= 2 and has_dip:
                ks_entries.append((dt, ticker))

    ks_entries = list(set(ks_entries))
    res_ks = run_backtest(ks_entries, close, spy_close, label="KitchenSink_2flow+1dip")
    if res_ks:
        passed, detail = five_gate_check(res_ks)
        res_ks['passed_5g'] = passed
        res_ks['gate_detail'] = detail
        confluence_results.append(res_ks)
        results_all.append(res_ks)
        status = "*** 5G-PASS ***" if passed else "5G-FAIL"
        print(f"  KitchenSink_2flow+1dip         : {res_ks['n_trades']:4d} trades | Sharpe {res_ks['sharpe']:+.2f} | WR {res_ks['win_rate']:.1%} | PF {res_ks['profit_factor']:.2f} | {status}")
    else:
        print(f"  KitchenSink: <5 trades (skip)")

    # ═══════════════════════════════════════════════════════════════════
    # 6. CONFLUENCE ALPHA ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("CONFLUENCE ALPHA ANALYSIS")
    print("=" * 80)

    # For each passing confluence pair, compare to solo components
    solo_sharpes = {}
    for r in results_all:
        if r['label'].startswith('Solo_'):
            solo_sharpes[r['label'].replace('Solo_', '')] = r['sharpe']

    print(f"\n{'Combo':<40s} {'Sharpe':>7s} {'Solo1':>7s} {'Solo2':>7s} {'Alpha':>7s} {'5-Gate':>8s}")
    print("-" * 80)

    passing_combos = []
    for r in confluence_results:
        label = r['label']
        parts = label.split('_x_')
        if len(parts) == 2:
            s1 = solo_sharpes.get(parts[0], np.nan)
            s2 = solo_sharpes.get(parts[1], np.nan)
            best_solo = max(s1, s2) if not (np.isnan(s1) or np.isnan(s2)) else np.nan
            alpha = r['sharpe'] - best_solo if not np.isnan(best_solo) else np.nan
        else:
            s1 = np.nan
            s2 = np.nan
            alpha = np.nan

        passed = r.get('passed_5g', False)
        status = "PASS" if passed else "FAIL"
        print(f"  {label:<38s} {r['sharpe']:+7.2f} {s1:+7.2f} {s2:+7.2f} {alpha:+7.2f} {status:>8s}")

        if passed:
            passing_combos.append(r)

    # ═══════════════════════════════════════════════════════════════════
    # 7. SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FULL SUMMARY TABLE — ALL CONFLUENCE COMBOS")
    print("=" * 80)

    header = f"{'Combo':<40s} {'Trades':>6s} {'WR':>6s} {'Sharpe':>7s} {'Sortino':>8s} {'PF':>6s} {'PnL':>9s} {'MaxDD':>8s} {'RGap':>6s} {'Perm-p':>7s} {'5G':>5s}"
    print(header)
    print("-" * len(header))

    for r in sorted(confluence_results, key=lambda x: x['sharpe'], reverse=True):
        passed = r.get('passed_5g', False)
        perm_p = r.get('perm_p', 1.0)
        rg = f"{r['regime_gap']:.2f}" if not np.isnan(r.get('regime_gap', np.nan)) else "N/A"
        print(f"  {r['label']:<38s} {r['n_trades']:6d} {r['win_rate']:5.1%} {r['sharpe']:+7.2f} {r['sortino']:+8.2f} {r['profit_factor']:6.2f} ${r['total_pnl']:8.0f} ${r['max_dd']:7.0f} {rg:>6s} {perm_p:7.3f} {'PASS' if passed else 'FAIL':>5s}")

    # Solo baselines
    print(f"\n{'SOLO BASELINES':<40s}")
    print("-" * len(header))
    for r in sorted([x for x in results_all if x['label'].startswith('Solo_')], key=lambda x: x['sharpe'], reverse=True):
        perm_p = r.get('perm_p', 1.0)
        rg = f"{r['regime_gap']:.2f}" if not np.isnan(r.get('regime_gap', np.nan)) else "N/A"
        print(f"  {r['label']:<38s} {r['n_trades']:6d} {r['win_rate']:5.1%} {r['sharpe']:+7.2f} {r['sortino']:+8.2f} {r['profit_factor']:6.2f} ${r['total_pnl']:8.0f} ${r['max_dd']:7.0f} {rg:>6s} {perm_p:7.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # 8. REGIME BREAKDOWN FOR PASSING COMBOS
    # ═══════════════════════════════════════════════════════════════════
    if passing_combos:
        print("\n" + "=" * 80)
        print("REGIME BREAKDOWN — PASSING COMBOS")
        print("=" * 80)

        for r in passing_combos:
            df = r['trades_df']
            print(f"\n  {r['label']}:")
            for regime in ['bull', 'bear', 'flat']:
                rdf = df[df['regime'] == regime]
                if len(rdf) > 0:
                    wr = (rdf['net_ret'] > 0).mean()
                    avg = rdf['net_ret'].mean()
                    rs = r['regime_sharpes'].get(regime, np.nan)
                    print(f"    {regime:5s}: {len(rdf):4d} trades | WR {wr:.1%} | Avg {avg:+.2%} | Sharpe-proxy {rs:+.2f}" if not np.isnan(rs) else f"    {regime:5s}: {len(rdf):4d} trades | WR {wr:.1%} | Avg {avg:+.2%}")

    # ═══════════════════════════════════════════════════════════════════
    # 9. CONCLUSION
    # ═══════════════════════════════════════════════════════════════════
    n_passed = len(passing_combos)
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"CONCLUSION: {n_passed}/{len(confluence_results)} combos passed 5-gate validation")
    print(f"Elapsed: {elapsed:.0f}s")
    print("=" * 80)

    if n_passed > 0:
        best = max(passing_combos, key=lambda x: x['sharpe'])
        print(f"\nBest passing combo: {best['label']}")
        print(f"  Sharpe: {best['sharpe']:+.2f} | Sortino: {best['sortino']:+.2f}")
        print(f"  WR: {best['win_rate']:.1%} | PF: {best['profit_factor']:.2f}")
        print(f"  Total PnL: ${best['total_pnl']:.0f} over {best['n_trades']} trades")
        print(f"  Regime gap: {best.get('regime_gap', np.nan):.2f}")
        print(f"  Perm test p: {best.get('perm_p', 1.0):.3f}")
    else:
        # Find closest to passing
        if confluence_results:
            best = max(confluence_results, key=lambda x: x['sharpe'])
            print(f"\nClosest to passing: {best['label']}")
            print(f"  Sharpe: {best['sharpe']:+.2f} | WR: {best['win_rate']:.1%} | PF: {best['profit_factor']:.2f}")
            print(f"  Gate detail: {best.get('gate_detail', 'N/A')}")

    # Save results
    save_data = []
    for r in results_all:
        d = {k: v for k, v in r.items() if k not in ('trades_df', 'returns', 'regime_sharpes')}
        d['regime_sharpes'] = {k: float(v) if not np.isnan(v) else None for k, v in r.get('regime_sharpes', {}).items()}
        save_data.append(d)

    with open(OUTPUT_DIR / 'flow_confluence_results.json', 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_DIR / 'flow_confluence_results.json'}")


if __name__ == '__main__':
    main()
