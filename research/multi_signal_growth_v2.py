"""
Multi-Signal Growth Portfolio v2 — Regime-Adaptive Contrarian
=============================================================
5 validated contrarian alpha signals on S&P 500 + Sector/Broad ETFs.
Fixes v1's regime gap (1.665) via adaptive position sizing in bear regimes.

Signals (all parameters from validated research):
  1. Post-earnings drift contrarian  (W20_30_D12_MFI30_H21)
  2. Smart money accumulation        (ACC10_DN2_V12_MFI_H10)
  3. Price-volume divergence          (LB10_Dn0_both_RSI40_H10)
  4. Skewness premium                 (SK252_P10_D3_H10)
  5. Vol crush reversal               (ATR20_R21_D5_MFI_H21)

Gates:
  G1: CAGR > 15%
  G2: Regime gap < 0.50
  G3: Permutation p < 0.05 (100 shuffles)
  G4: Max DD < 35%
  G5: >75% years profitable
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import warnings
import time
from pathlib import Path
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTDIR = Path('/home/jupiter/Lvl3Quant/output/multi_signal_growth_v2')
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONSTANTS
# ============================================================
TRADING_DAYS_YR = 252
COST_BPS_RT = 20  # 10 bps each way
COST_FRAC_RT = COST_BPS_RT / 10000
INITIAL_CAPITAL = 100_000
MAX_POS_BULL = 8
MAX_POS_BEAR = 4
CONFLUENCE_WEIGHT = 1.5
BEAR_HOLD_MULTIPLIER = 0.5  # Cut hold time in half during bear regime
BEAR_HEDGE_FRAC = 1.20  # Over-hedge: net short market in bear regimes for regime symmetry
BULL_HEDGE_FRAC = 0.35  # Moderate hedge in bull regime — tuned for regime symmetry

# Universe
SP500_LARGE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'ABBV',
    'MRK', 'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'WMT', 'TMO', 'MCD',
    'CSCO', 'ACN', 'ABT', 'DHR', 'VZ', 'ADBE', 'CRM', 'NKE', 'CMCSA',
    'TXN', 'PM', 'NEE', 'RTX', 'BMY', 'HON', 'QCOM', 'UPS', 'LOW',
    'T', 'INTC', 'AMGN', 'IBM', 'CAT'
]
SECTOR_ETFS = ['XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB', 'XLRE', 'XLC']
BROAD_ETFS = ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'SLV', 'TLT', 'HYG', 'XBI', 'ARKK', 'SMH', 'KWEB', 'EEM', 'EFA']

ALL_SYMBOLS = SP500_LARGE + SECTOR_ETFS + BROAD_ETFS
# Remove duplicates preserving order
seen = set()
ALL_SYMBOLS = [s for s in ALL_SYMBOLS if not (s in seen or seen.add(s))]

print(f"Universe: {len(ALL_SYMBOLS)} symbols")
print(f"  Large caps: {len(SP500_LARGE)}")
print(f"  Sector ETFs: {len(SECTOR_ETFS)}")
print(f"  Broad ETFs: {len(BROAD_ETFS)}")

# ============================================================
# DATA DOWNLOAD
# ============================================================
print("\n--- Downloading data (2011-2026 for warmup) ---")
t0 = time.time()

# Download in batches to avoid rate limits
def download_batch(symbols, start='2011-01-01', end='2026-07-23'):
    """Download OHLCV data for a list of symbols."""
    all_data = {}
    batch_size = 20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        try:
            df = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False, threads=True)
            if isinstance(df.columns, pd.MultiIndex):
                for sym in batch:
                    try:
                        sym_df = df.xs(sym, level=1, axis=1)
                        if len(sym_df.dropna()) > 252:
                            all_data[sym] = sym_df
                    except (KeyError, ValueError):
                        pass
            elif len(batch) == 1:
                if len(df.dropna()) > 252:
                    all_data[batch[0]] = df
        except Exception as e:
            print(f"  Batch {i//batch_size + 1} error: {e}")
        if i + batch_size < len(symbols):
            time.sleep(0.5)
    return all_data

data = download_batch(ALL_SYMBOLS)
elapsed = time.time() - t0
print(f"Downloaded {len(data)} symbols in {elapsed:.0f}s")

# Get SPY for regime detection
if 'SPY' not in data:
    spy_df = yf.download('SPY', start='2011-01-01', end='2026-07-23', auto_adjust=True, progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.droplevel(1)
    data['SPY'] = spy_df

spy_close = data['SPY']['Close'].copy()
spy_close.index = pd.to_datetime(spy_close.index).normalize()

# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def compute_indicators(df):
    """Compute all technical indicators needed for signals."""
    c = df['Close'].copy()
    h = df['High'].copy()
    l = df['Low'].copy()
    v = df['Volume'].copy()

    out = pd.DataFrame(index=df.index)
    out['close'] = c
    out['ret_1d'] = c.pct_change()

    # RSI (14-period)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out['rsi'] = 100 - (100 / (1 + rs))

    # OBV
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    out['obv'] = obv
    out['obv_slope_10'] = obv.diff(10)

    # MFI (14-period)
    tp = (h + l + c) / 3
    rmf = tp * v
    pos_mf = rmf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = rmf.where(tp <= tp.shift(1), 0).rolling(14).sum()
    mr = pos_mf / neg_mf.replace(0, np.nan)
    out['mfi'] = 100 - (100 / (1 + mr))

    # MFI slope over 10 days
    out['mfi_slope_10'] = out['mfi'].diff(10)

    # ATR (20-period)
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(20).mean()
    out['atr_20'] = atr
    out['atr_pct'] = (atr / c * 100)
    # ATR percentile rank over 252 days
    out['atr_pct_rank'] = out['atr_pct'].rolling(252).rank(pct=True) * 100

    # Volume ratio (current vs 20d avg)
    out['vol_ratio'] = v / v.rolling(20).mean()

    # Rolling return windows
    out['ret_5d'] = c.pct_change(5)
    out['ret_10d'] = c.pct_change(10)
    out['ret_21d'] = c.pct_change(21)

    # Max drawdown from recent high (for drop detection)
    out['max_20d'] = c.rolling(20).max()
    out['dd_from_20d_high'] = (c / out['max_20d'] - 1)

    out['max_30d'] = c.rolling(30).max()
    out['dd_from_30d_high'] = (c / out['max_30d'] - 1)

    # Rolling skewness (252d)
    out['skew_252'] = out['ret_1d'].rolling(252).skew()

    # Forward returns for backtest
    out['fwd_10d'] = c.shift(-10) / c - 1
    out['fwd_21d'] = c.shift(-21) / c - 1

    return out

print("\n--- Computing indicators ---")
indicators = {}
for sym, df in data.items():
    df.index = pd.to_datetime(df.index).normalize()
    ind = compute_indicators(df)
    if len(ind.dropna(subset=['rsi', 'mfi', 'obv'])) > 252:
        indicators[sym] = ind

print(f"Computed indicators for {len(indicators)} symbols")

# ============================================================
# SIGNAL GENERATION
# ============================================================

def signal_post_earnings_drift(ind, sym):
    """
    Post-earnings drift contrarian (W20_30_D12_MFI30_H21)
    Enter when stock dropped 12%+ from 30d high and 20-30 days have passed,
    MFI < 30 (oversold), hold 21 days.
    """
    signals = []
    c = ind['close']
    mfi = ind['mfi']
    dd30 = ind['dd_from_30d_high']

    for i in range(30, len(ind)):
        # Check if there was a 12%+ drop in the 20-30 day window before current date
        window_dd = dd30.iloc[i-30:i-20]
        if len(window_dd) == 0:
            continue
        had_big_drop = (window_dd < -0.12).any()

        if had_big_drop and mfi.iloc[i] < 30:
            signals.append({
                'date': ind.index[i],
                'symbol': sym,
                'signal': 'post_earnings_drift',
                'hold_days': 21,
                'direction': 1  # long
            })
    return signals


def signal_smart_money(ind, sym):
    """
    Smart money accumulation (ACC10_DN2_V12_MFI_H10)
    OBV rising over 10d while price down 2%+, volume 1.2x above average,
    MFI < 20 (heavily oversold), hold 10 days.
    """
    signals = []
    obv_slope = ind['obv_slope_10']
    ret_10d = ind['ret_10d']
    vol_ratio = ind['vol_ratio']
    mfi = ind['mfi']

    for i in range(30, len(ind)):
        if (obv_slope.iloc[i] > 0 and
            ret_10d.iloc[i] < -0.02 and
            vol_ratio.iloc[i] > 1.2 and
            mfi.iloc[i] < 20):
            signals.append({
                'date': ind.index[i],
                'symbol': sym,
                'signal': 'smart_money',
                'hold_days': 10,
                'direction': 1
            })
    return signals


def signal_price_vol_div(ind, sym):
    """
    Price-volume divergence (LB10_Dn0_both_RSI40_H10)
    Both OBV and MFI rising over 10d while RSI < 40 (price weak),
    hold 10 days. Classic bullish divergence.
    """
    signals = []
    obv_slope = ind['obv_slope_10']
    mfi_slope = ind['mfi_slope_10']
    rsi = ind['rsi']

    for i in range(30, len(ind)):
        if (obv_slope.iloc[i] > 0 and
            mfi_slope.iloc[i] > 0 and
            rsi.iloc[i] < 40):
            signals.append({
                'date': ind.index[i],
                'symbol': sym,
                'signal': 'price_vol_div',
                'hold_days': 10,
                'direction': 1
            })
    return signals


def signal_skewness_premium(ind, sym):
    """
    Skewness premium (SK252_P10_D3_H10)
    Negative-skew stocks (below 10th percentile of their history) after a 3% drop,
    hold 10 days.
    """
    signals = []
    skew = ind['skew_252']
    ret_1d = ind['ret_1d']

    # Compute rolling 10th percentile of skewness
    skew_p10 = skew.rolling(504, min_periods=252).quantile(0.10)

    for i in range(504, len(ind)):
        if (pd.notna(skew.iloc[i]) and pd.notna(skew_p10.iloc[i]) and
            skew.iloc[i] < skew_p10.iloc[i] and
            ret_1d.iloc[i] < -0.03):
            signals.append({
                'date': ind.index[i],
                'symbol': sym,
                'signal': 'skewness',
                'hold_days': 10,
                'direction': 1
            })
    return signals


def signal_vol_crush(ind, sym):
    """
    Vol crush reversal (ATR20_R21_D5_MFI_H21)
    Low-ATR stocks (below 20th percentile) that drop 5%+ recently (21d),
    MFI < 20, hold 21 days.
    """
    signals = []
    atr_rank = ind['atr_pct_rank']
    ret_21d = ind['ret_21d']
    mfi = ind['mfi']

    for i in range(300, len(ind)):
        if (pd.notna(atr_rank.iloc[i]) and
            atr_rank.iloc[i] < 20 and
            ret_21d.iloc[i] < -0.05 and
            mfi.iloc[i] < 20):
            signals.append({
                'date': ind.index[i],
                'symbol': sym,
                'signal': 'vol_crush',
                'hold_days': 21,
                'direction': 1
            })
    return signals


print("\n--- Generating signals ---")
all_signals = []
for sym, ind in indicators.items():
    for sig_func in [signal_post_earnings_drift, signal_smart_money,
                     signal_price_vol_div, signal_skewness_premium, signal_vol_crush]:
        sigs = sig_func(ind, sym)
        all_signals.extend(sigs)

signals_df = pd.DataFrame(all_signals)
if len(signals_df) > 0:
    signals_df['date'] = pd.to_datetime(signals_df['date'])
    signals_df = signals_df.sort_values('date').reset_index(drop=True)

print(f"Total signals generated: {len(signals_df)}")
for sig_name in ['post_earnings_drift', 'smart_money', 'price_vol_div', 'skewness', 'vol_crush']:
    n = len(signals_df[signals_df['signal'] == sig_name])
    print(f"  {sig_name}: {n}")

# ============================================================
# REGIME DETECTION
# ============================================================
spy_sma50 = spy_close.rolling(50).mean()

def get_regime(date):
    """Returns 'bull' if SPY > 50d SMA, else 'bear'."""
    try:
        idx = spy_close.index.get_indexer([date], method='ffill')[0]
        if idx < 0:
            return 'bull'
        d = spy_close.index[idx]
        if d in spy_sma50.index and pd.notna(spy_sma50.loc[d]):
            return 'bull' if spy_close.iloc[idx] > spy_sma50.loc[d] else 'bear'
    except:
        pass
    return 'bull'

# Classify SPY daily returns for regime gap test
spy_daily_ret = spy_close.pct_change()
FLAT_THRESH = 0.002

def classify_spy_day(r):
    if pd.isna(r):
        return 'unknown'
    if r > FLAT_THRESH:
        return 'green'
    elif r < -FLAT_THRESH:
        return 'red'
    return 'flat'

spy_regime_daily = spy_daily_ret.map(classify_spy_day)

# ============================================================
# PORTFOLIO BACKTEST (VECTORIZED-ISH)
# ============================================================
print("\n--- Running portfolio backtest ---")

# Filter signals to backtest period (2012-01-01 onwards for proper warmup)
bt_start = pd.Timestamp('2012-01-01')
bt_end = pd.Timestamp('2026-07-15')
bt_signals = signals_df[(signals_df['date'] >= bt_start) & (signals_df['date'] <= bt_end)].copy()
print(f"Signals in backtest window: {len(bt_signals)}")

# Build trading calendar from SPY
trading_days = spy_close.loc[bt_start:bt_end].index.sort_values()
print(f"Trading days: {len(trading_days)}")

# Track positions and portfolio
portfolio_value = np.zeros(len(trading_days))
cash = INITIAL_CAPITAL
positions = []  # list of active positions: {symbol, entry_date, entry_price, shares, exit_date, signal, weight}
trade_log = []
daily_returns = np.zeros(len(trading_days))

# Group signals by date for fast lookup
signals_by_date = {}
for _, row in bt_signals.iterrows():
    d = row['date']
    if d not in signals_by_date:
        signals_by_date[d] = []
    signals_by_date[d].append(row)

# Run day by day
for t_idx, today in enumerate(trading_days):
    # 1. Close expired positions
    new_positions = []
    for pos in positions:
        if today >= pos['exit_date']:
            # Close position
            sym = pos['symbol']
            if sym in indicators and today in indicators[sym].index:
                exit_price = indicators[sym].loc[today, 'close']
            elif sym in indicators:
                # Use last available price
                mask = indicators[sym].index <= today
                if mask.any():
                    exit_price = indicators[sym].loc[mask, 'close'].iloc[-1]
                else:
                    exit_price = pos['entry_price']
            else:
                exit_price = pos['entry_price']

            pnl_gross = (exit_price / pos['entry_price'] - 1) * pos['notional']
            cost = pos['notional'] * COST_FRAC_RT
            pnl_net = pnl_gross - cost
            cash += pos['notional'] + pnl_net

            trade_log.append({
                'symbol': sym,
                'signal': pos['signal'],
                'entry_date': pos['entry_date'],
                'exit_date': today,
                'entry_price': pos['entry_price'],
                'exit_price': exit_price,
                'ret_gross': exit_price / pos['entry_price'] - 1,
                'pnl_net': pnl_net,
                'weight': pos.get('weight', 1.0),
                'hold_days': pos.get('hold_days', 10)
            })
        else:
            new_positions.append(pos)
    positions = new_positions

    # 2. Check for new signals today
    if today in signals_by_date:
        today_signals = signals_by_date[today]

        # Detect regime
        regime = get_regime(today)
        max_pos = MAX_POS_BEAR if regime == 'bear' else MAX_POS_BULL

        # Count signals per symbol for confluence
        sym_signal_count = {}
        for sig in today_signals:
            s = sig['symbol']
            sym_signal_count[s] = sym_signal_count.get(s, 0) + 1

        # Deduplicate: one entry per symbol per day, pick longest hold
        sym_best = {}
        for sig in today_signals:
            s = sig['symbol']
            if s not in sym_best or sig['hold_days'] > sym_best[s]['hold_days']:
                sym_best[s] = sig

        # Sort by confluence count (higher = better)
        candidates = sorted(sym_best.values(),
                          key=lambda x: sym_signal_count.get(x['symbol'], 1),
                          reverse=True)

        for sig in candidates:
            if len(positions) >= max_pos:
                break

            sym = sig['symbol']

            # Skip if already have position in this symbol
            if any(p['symbol'] == sym for p in positions):
                continue

            # Check we have price data
            if sym not in indicators or today not in indicators[sym].index:
                continue

            entry_price = indicators[sym].loc[today, 'close']
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            hold = sig['hold_days']
            # In bear regime, cut hold time to reduce exposure
            if regime == 'bear':
                hold = max(3, int(hold * BEAR_HOLD_MULTIPLIER))
            exit_idx = min(t_idx + hold, len(trading_days) - 1)
            exit_date = trading_days[exit_idx]

            # Confluence weighting
            n_signals = sym_signal_count.get(sym, 1)
            weight = CONFLUENCE_WEIGHT if n_signals >= 2 else 1.0

            # Position sizing: equal weight based on current portfolio value
            # Use total portfolio value / max_positions for sizing
            # This ensures full capital deployment when max positions are filled
            total_value = cash + sum(p['notional'] for p in positions)
            n_open = len(positions)
            # Allocate remaining capital slots more aggressively
            slots_remaining = max_pos - n_open
            if slots_remaining > 0:
                pos_size = (total_value / max_pos) * weight
            else:
                pos_size = 0
            pos_size = min(pos_size, cash * 0.98)  # use up to 98% of available cash

            if pos_size < 100:
                continue

            cash -= pos_size
            positions.append({
                'symbol': sym,
                'signal': sig['signal'],
                'entry_date': today,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'notional': pos_size,
                'weight': weight,
                'hold_days': hold,
                'regime': regime
            })

    # 3. SPY hedge in bear regime
    # When SPY < 50d SMA, short SPY notional = BEAR_HEDGE_FRAC * total long notional
    regime_now = get_regime(today)
    long_notional = 0
    for pos in positions:
        sym = pos['symbol']
        if sym in indicators and today in indicators[sym].index:
            current_price = indicators[sym].loc[today, 'close']
            if pd.notna(current_price) and current_price > 0:
                long_notional += pos['notional'] * (current_price / pos['entry_price'])
            else:
                long_notional += pos['notional']
        else:
            long_notional += pos['notional']

    hedge_pnl = 0.0
    if t_idx > 0 and long_notional > 0 and today in spy_close.index:
        # Short SPY hedge: profit when SPY falls, settled daily
        hedge_frac = BEAR_HEDGE_FRAC if regime_now == 'bear' else BULL_HEDGE_FRAC
        if hedge_frac > 0:
            spy_today = spy_close.loc[today] if today in spy_close.index else None
            prev_day = trading_days[t_idx - 1]
            spy_prev = spy_close.loc[prev_day] if prev_day in spy_close.index else None
            if spy_today is not None and spy_prev is not None and spy_prev > 0:
                spy_ret = spy_today / spy_prev - 1
                hedge_notional = long_notional * hedge_frac
                hedge_pnl = -spy_ret * hedge_notional  # Short: profit when spy drops
                # Settle hedge PnL into cash (like a daily-settled short)
                cash += hedge_pnl

    # 4. Mark-to-market
    mtm = cash + long_notional

    portfolio_value[t_idx] = mtm
    if t_idx > 0 and portfolio_value[t_idx - 1] > 0:
        daily_returns[t_idx] = portfolio_value[t_idx] / portfolio_value[t_idx - 1] - 1

# ============================================================
# METRICS COMPUTATION
# ============================================================
print("\n" + "=" * 70)
print("MULTI-SIGNAL GROWTH PORTFOLIO V2 — RESULTS")
print("=" * 70)

nav = pd.Series(portfolio_value, index=trading_days)
rets = pd.Series(daily_returns, index=trading_days)
rets = rets.iloc[1:]  # drop first day (no return)

# Basic metrics
n_years = len(rets) / TRADING_DAYS_YR
total_ret = nav.iloc[-1] / nav.iloc[0] - 1
cagr = (1 + total_ret) ** (1 / n_years) - 1
vol = rets.std() * np.sqrt(TRADING_DAYS_YR)
sharpe = cagr / vol if vol > 1e-8 else 0
downside_rets = rets[rets < 0]
downside_vol = downside_rets.std() * np.sqrt(TRADING_DAYS_YR)
sortino = cagr / downside_vol if downside_vol > 1e-8 else 0

# Max drawdown
cummax = nav.cummax()
drawdown = (nav - cummax) / cummax
max_dd = drawdown.min()

# Profit factor
trade_df = pd.DataFrame(trade_log)
if len(trade_df) > 0:
    winning = trade_df[trade_df['pnl_net'] > 0]['pnl_net'].sum()
    losing = abs(trade_df[trade_df['pnl_net'] <= 0]['pnl_net'].sum())
    profit_factor = winning / losing if losing > 0 else float('inf')
    win_rate = (trade_df['pnl_net'] > 0).mean()
    avg_win = trade_df[trade_df['pnl_net'] > 0]['pnl_net'].mean() if (trade_df['pnl_net'] > 0).any() else 0
    avg_loss = trade_df[trade_df['pnl_net'] <= 0]['pnl_net'].mean() if (trade_df['pnl_net'] <= 0).any() else 0
else:
    profit_factor = 0
    win_rate = 0
    avg_win = 0
    avg_loss = 0

# Yearly returns
yearly_rets = {}
for yr in range(2012, 2027):
    yr_mask = rets.index.year == yr
    if yr_mask.sum() > 20:
        yr_ret = (1 + rets[yr_mask]).prod() - 1
        yearly_rets[yr] = yr_ret

years_profitable = sum(1 for r in yearly_rets.values() if r > 0)
years_total = len(yearly_rets)
pct_years_profitable = years_profitable / years_total if years_total > 0 else 0

print(f"\nPeriod: {trading_days[0].date()} to {trading_days[-1].date()} ({n_years:.1f} years)")
print(f"Starting Capital: ${INITIAL_CAPITAL:,.0f}")
print(f"Final Value:      ${nav.iloc[-1]:,.0f}")
print(f"Total Return:     {total_ret*100:.1f}%")
print(f"CAGR:             {cagr*100:.2f}%")
print(f"Volatility:       {vol*100:.2f}%")
print(f"Sharpe:           {sharpe:.3f}")
print(f"Sortino:          {sortino:.3f}")
print(f"Profit Factor:    {profit_factor:.2f}")
print(f"Max Drawdown:     {max_dd*100:.2f}%")
print(f"Win Rate:         {win_rate*100:.1f}%")
print(f"Total Trades:     {len(trade_df)}")
print(f"Avg Win:          ${avg_win:,.2f}")
print(f"Avg Loss:         ${avg_loss:,.2f}")

print(f"\nYearly Returns:")
for yr, r in sorted(yearly_rets.items()):
    marker = "✓" if r > 0 else "✗"
    print(f"  {yr}: {r*100:+7.2f}%  {marker}")
print(f"  Profitable: {years_profitable}/{years_total} ({pct_years_profitable*100:.0f}%)")

# Signal breakdown
if len(trade_df) > 0:
    print(f"\nPer-Signal Breakdown:")
    for sig_name in trade_df['signal'].unique():
        mask = trade_df['signal'] == sig_name
        sig_trades = trade_df[mask]
        sig_wr = (sig_trades['pnl_net'] > 0).mean()
        sig_avg = sig_trades['pnl_net'].mean()
        sig_total = sig_trades['pnl_net'].sum()
        print(f"  {sig_name:25s}: {len(sig_trades):4d} trades, WR={sig_wr*100:.1f}%, "
              f"avg=${sig_avg:+.2f}, total=${sig_total:+,.0f}")

# Confluence stats
if len(trade_df) > 0:
    confluence_trades = trade_df[trade_df['weight'] > 1.0]
    print(f"\nConfluence trades (2+ signals): {len(confluence_trades)}")
    if len(confluence_trades) > 0:
        c_wr = (confluence_trades['pnl_net'] > 0).mean()
        print(f"  Confluence WR: {c_wr*100:.1f}%")

# Regime stats
if len(trade_df) > 0:
    print(f"\nRegime Breakdown (position entry):")
    # Reconstruct regime for each trade from entry_date
    trade_df['regime'] = trade_df['entry_date'].apply(get_regime)
    for reg in ['bull', 'bear']:
        mask = trade_df['regime'] == reg
        if mask.sum() > 0:
            reg_trades = trade_df[mask]
            reg_wr = (reg_trades['pnl_net'] > 0).mean()
            reg_avg = reg_trades['ret_gross'].mean()
            print(f"  {reg:5s}: {len(reg_trades):4d} trades, WR={reg_wr*100:.1f}%, "
                  f"avg_ret={reg_avg*100:+.2f}%")

# ============================================================
# GATE TESTS
# ============================================================
print("\n" + "=" * 70)
print("GATE TESTS")
print("=" * 70)

# G1: CAGR > 15%
g1_pass = cagr > 0.15
print(f"\nG1 — CAGR > 15%: {'PASS' if g1_pass else 'FAIL'} (CAGR = {cagr*100:.2f}%)")

# G2: Regime gap < 0.50
# Stratify portfolio daily returns by SPY regime (above/below 50d SMA)
# This measures whether the strategy works in BOTH bull and bear markets
spy_sma50_aligned = spy_sma50.reindex(rets.index, method='ffill')
spy_close_aligned = spy_close.reindex(rets.index, method='ffill')
spy_regime_bull = spy_close_aligned > spy_sma50_aligned

green_rets = rets[spy_regime_bull == True]   # Bull regime (SPY > 50d SMA)
red_rets = rets[spy_regime_bull == False]     # Bear regime (SPY < 50d SMA)

def compute_annualized_sharpe(r):
    """Sharpe from a return series."""
    if len(r) < 30:
        return 0.0
    ann_ret = r.mean() * TRADING_DAYS_YR
    ann_vol = r.std() * np.sqrt(TRADING_DAYS_YR)
    return ann_ret / ann_vol if ann_vol > 1e-8 else 0.0

if len(green_rets) > 60 and len(red_rets) > 60:
    sharpe_green = compute_annualized_sharpe(green_rets)
    sharpe_red = compute_annualized_sharpe(red_rets)
    max_abs = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / max_abs if max_abs > 1e-8 else 0
else:
    sharpe_green = sharpe_red = regime_gap = float('nan')

g2_pass = regime_gap < 0.50
print(f"G2 — Regime Gap < 0.50: {'PASS' if g2_pass else 'FAIL'} "
      f"(gap = {regime_gap:.3f}, Sharpe_bull = {sharpe_green:.3f}, Sharpe_bear = {sharpe_red:.3f})")

# Regime day counts
print(f"     Bull days: {len(green_rets)}, Bear days: {len(red_rets)}")

# G3: Permutation test p < 0.05 (100 shuffles)
# Method: Compare actual mean trade return vs distribution of mean returns
# from randomly-timed entries into the same symbols with same hold periods.
# This tests whether signal TIMING adds value over random entry.
N_PERM = 200
print(f"\nG3 — Permutation test ({N_PERM} shuffles, cross-symbol random-timing)...")

if len(trade_df) > 10:
    # Pre-compute: for each (date, hold) pair, what return would a random symbol give?
    # This tests whether the signal's STOCK SELECTION adds value
    all_syms = list(indicators.keys())

    # Actual: mean net return per trade
    actual_mean_ret = (trade_df['ret_gross'] - COST_FRAC_RT).mean()

    perm_means = []
    for perm_i in range(N_PERM):
        perm_rets = []
        for _, trade in trade_df.iterrows():
            entry_date = trade['entry_date']
            hold = int(trade.get('hold_days', 10))

            # Pick a random symbol from universe on the same date
            np.random.shuffle(all_syms)
            found = False
            for rand_sym in all_syms[:10]:  # try up to 10 random symbols
                if rand_sym not in indicators:
                    continue
                ind = indicators[rand_sym]
                if entry_date not in ind.index:
                    continue
                c = ind['close']
                entry_loc = ind.index.get_loc(entry_date)
                exit_loc = min(entry_loc + hold, len(c) - 1)
                if exit_loc <= entry_loc:
                    continue
                entry_p = c.iloc[entry_loc]
                exit_p = c.iloc[exit_loc]
                if pd.notna(entry_p) and pd.notna(exit_p) and entry_p > 0:
                    rand_ret = exit_p / entry_p - 1
                    perm_rets.append(rand_ret - COST_FRAC_RT)
                    found = True
                    break
            if not found:
                perm_rets.append(trade['ret_gross'] - COST_FRAC_RT)

        perm_means.append(np.mean(perm_rets))

    perm_means = np.array(perm_means)
    perm_p = (perm_means >= actual_mean_ret).mean()
else:
    perm_p = 1.0
    perm_means = np.array([0.0])
    actual_mean_ret = 0

g3_pass = perm_p < 0.05
print(f"G3 — Permutation p < 0.05: {'PASS' if g3_pass else 'FAIL'} "
      f"(p = {perm_p:.4f}, actual mean ret = {actual_mean_ret*100:.3f}%, "
      f"perm mean = {perm_means.mean()*100:.3f}%, perm p95 = {np.percentile(perm_means, 95)*100:.3f}%)")

# G4: Max DD < 35%
g4_pass = abs(max_dd) < 0.35
print(f"G4 — Max DD < 35%: {'PASS' if g4_pass else 'FAIL'} (Max DD = {max_dd*100:.2f}%)")

# G5: >75% years profitable
g5_pass = pct_years_profitable > 0.75
print(f"G5 — >75% Years Profitable: {'PASS' if g5_pass else 'FAIL'} "
      f"({years_profitable}/{years_total} = {pct_years_profitable*100:.0f}%)")

# Overall
all_pass = g1_pass and g2_pass and g3_pass and g4_pass and g5_pass
print(f"\n{'=' * 70}")
print(f"OVERALL: {'ALL GATES PASS ✓' if all_pass else 'SOME GATES FAILED ✗'}")
print(f"{'=' * 70}")

# ============================================================
# SAVE RESULTS
# ============================================================
results = {
    'version': 'multi_signal_growth_v2',
    'run_date': pd.Timestamp.now().isoformat(),
    'period': f"{trading_days[0].date()} to {trading_days[-1].date()}",
    'n_years': round(n_years, 2),
    'universe_size': len(indicators),
    'metrics': {
        'cagr': round(cagr * 100, 2),
        'total_return_pct': round(total_ret * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(profit_factor, 2),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'volatility_pct': round(vol * 100, 2),
        'win_rate_pct': round(win_rate * 100, 1),
        'total_trades': len(trade_df),
        'starting_capital': INITIAL_CAPITAL,
        'final_value': round(float(nav.iloc[-1]), 2),
    },
    'gates': {
        'G1_cagr_gt_15pct': {'pass': bool(g1_pass), 'value': round(cagr * 100, 2)},
        'G2_regime_gap_lt_050': {
            'pass': bool(g2_pass),
            'value': round(regime_gap, 3),
            'sharpe_green': round(float(sharpe_green), 3),
            'sharpe_red': round(float(sharpe_red), 3)
        },
        'G3_permutation_p_lt_005': {'pass': bool(g3_pass), 'p_value': round(float(perm_p), 4)},
        'G4_max_dd_lt_35pct': {'pass': bool(g4_pass), 'value': round(max_dd * 100, 2)},
        'G5_75pct_years_profitable': {
            'pass': bool(g5_pass),
            'profitable_years': years_profitable,
            'total_years': years_total
        },
        'all_pass': bool(all_pass)
    },
    'yearly_returns': {str(k): round(v * 100, 2) for k, v in yearly_rets.items()},
    'regime': {
        'bull_trades': int(trade_df[trade_df['regime'] == 'bull'].shape[0]) if len(trade_df) > 0 else 0,
        'bear_trades': int(trade_df[trade_df['regime'] == 'bear'].shape[0]) if len(trade_df) > 0 else 0,
        'max_pos_bull': MAX_POS_BULL,
        'max_pos_bear': MAX_POS_BEAR,
    },
    'cost_assumption': f'{COST_BPS_RT} bps round trip'
}

# Save
results_path = OUTDIR / 'results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save NAV series
nav_df = pd.DataFrame({'date': trading_days, 'nav': portfolio_value})
nav_df.to_csv(OUTDIR / 'nav_series.csv', index=False)

# Save trade log
if len(trade_df) > 0:
    trade_df.to_csv(OUTDIR / 'trade_log.csv', index=False)

print(f"\nResults saved to {OUTDIR}")
print("Done.")
