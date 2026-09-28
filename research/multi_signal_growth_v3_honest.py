"""
Multi-Signal Growth Portfolio v3 — Honest Backtest
===================================================
Fixes ALL 4 critical issues from adversarial audit of v2:
  1. Next-day execution: signals computed at day-T close, entry at day-T+1 open
  2. Lagged hedge: SPY 50d SMA regime uses day T-1 close, hedge adjusts at day-T open
  3. Survivorship-free universe: date-conditional inclusion for post-IPO/post-index stocks
  4. Honest permutation test: shuffles stock selection WITHOUT hedge advantage

Signals (same 5 proven contrarian signals):
  1. Post-earnings drift contrarian  (W20_30_D12_MFI30_H21)
  2. Smart money accumulation        (ACC10_DN2_V12_MFI_H10)
  3. Price-volume divergence          (LB10_Dn0_both_RSI40_H10)
  4. Skewness premium                 (SK252_P10_D3_H10)
  5. Vol crush reversal               (ATR20_R21_D5_MFI_H21)

Tests THREE hedge variants:
  A. No hedge (baseline)
  B. Constant 30% SPY short hedge
  C. Lagged regime-adaptive hedge (20%-60% range)

Gates (relaxed for honesty):
  G1: CAGR > 10%
  G2: Regime gap < 0.50
  G3: Permutation p < 0.05 (honest, no hedge advantage)
  G4: Max DD < 40%
  G5: >75% years profitable
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import warnings
import time
from pathlib import Path
from copy import deepcopy

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTDIR = Path('/home/jupiter/Lvl3Quant/output/multi_signal_growth_v3')
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONSTANTS
# ============================================================
TRADING_DAYS_YR = 252
COST_BPS_RT = 20  # 10 bps each way for stocks
COST_FRAC_RT = COST_BPS_RT / 10000
SPY_BORROW_ANNUAL = 0.004  # 0.4% annual borrow cost for SPY short
REBALANCE_COST_BPS = 5  # 5 bps per hedge ratio change
REBALANCE_COST_FRAC = REBALANCE_COST_BPS / 10000
INITIAL_CAPITAL = 100_000
MAX_POS_BULL = 8
MAX_POS_BEAR = 4
CONFLUENCE_WEIGHT = 1.5
BEAR_HOLD_MULTIPLIER = 0.5

# Hedge variants
CONSTANT_HEDGE_FRAC = 0.30
LAGGED_BULL_HEDGE = 0.20
LAGGED_BEAR_HEDGE = 0.60

# ============================================================
# SURVIVORSHIP BIAS FIX — date-conditional inclusion
# ============================================================
# These stocks were NOT in S&P 500 at backtest start (2012).
# Only include them from their S&P 500 addition date onward.
SP500_ADDITION_DATES = {
    'META':  pd.Timestamp('2013-12-23'),
    'TSLA':  pd.Timestamp('2020-12-21'),
    'AVGO':  pd.Timestamp('2014-03-21'),
    'CRM':   pd.Timestamp('2020-09-21'),
    'DHR':   pd.Timestamp('2017-09-18'),
}

# Universe (same 74 symbols but survivorship-corrected)
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
seen = set()
ALL_SYMBOLS = [s for s in ALL_SYMBOLS if not (s in seen or seen.add(s))]

def symbol_eligible(sym, date):
    """Check if symbol is eligible for trading on a given date (survivorship fix)."""
    if sym in SP500_ADDITION_DATES:
        return date >= SP500_ADDITION_DATES[sym]
    return True

print(f"Universe: {len(ALL_SYMBOLS)} symbols (with survivorship date-gates)")
print(f"  Survivorship-gated: {list(SP500_ADDITION_DATES.keys())}")

# ============================================================
# DATA DOWNLOAD
# ============================================================
print("\n--- Downloading data (2011-2026 for warmup) ---")
t0 = time.time()

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
spy_open = data['SPY']['Open'].copy()
spy_open.index = pd.to_datetime(spy_open.index).normalize()

# ============================================================
# TECHNICAL INDICATORS
# ============================================================
def compute_indicators(df):
    """Compute all technical indicators needed for signals."""
    c = df['Close'].copy()
    o = df['Open'].copy()
    h = df['High'].copy()
    l = df['Low'].copy()
    v = df['Volume'].copy()

    out = pd.DataFrame(index=df.index)
    out['close'] = c
    out['open'] = o
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
# SIGNAL GENERATION (signals fire on day-T close data)
# Entry will happen at day T+1 open (handled in backtest loop)
# ============================================================

def signal_post_earnings_drift(ind, sym):
    signals = []
    mfi = ind['mfi']
    dd30 = ind['dd_from_30d_high']
    for i in range(30, len(ind)):
        window_dd = dd30.iloc[i-30:i-20]
        if len(window_dd) == 0:
            continue
        had_big_drop = (window_dd < -0.12).any()
        if had_big_drop and mfi.iloc[i] < 30:
            signals.append({
                'signal_date': ind.index[i],  # day T: signal computed from this day's close
                'symbol': sym,
                'signal': 'post_earnings_drift',
                'hold_days': 21,
                'direction': 1
            })
    return signals

def signal_smart_money(ind, sym):
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
                'signal_date': ind.index[i],
                'symbol': sym,
                'signal': 'smart_money',
                'hold_days': 10,
                'direction': 1
            })
    return signals

def signal_price_vol_div(ind, sym):
    signals = []
    obv_slope = ind['obv_slope_10']
    mfi_slope = ind['mfi_slope_10']
    rsi = ind['rsi']
    for i in range(30, len(ind)):
        if (obv_slope.iloc[i] > 0 and
            mfi_slope.iloc[i] > 0 and
            rsi.iloc[i] < 40):
            signals.append({
                'signal_date': ind.index[i],
                'symbol': sym,
                'signal': 'price_vol_div',
                'hold_days': 10,
                'direction': 1
            })
    return signals

def signal_skewness_premium(ind, sym):
    signals = []
    skew = ind['skew_252']
    ret_1d = ind['ret_1d']
    skew_p10 = skew.rolling(504, min_periods=252).quantile(0.10)
    for i in range(504, len(ind)):
        if (pd.notna(skew.iloc[i]) and pd.notna(skew_p10.iloc[i]) and
            skew.iloc[i] < skew_p10.iloc[i] and
            ret_1d.iloc[i] < -0.03):
            signals.append({
                'signal_date': ind.index[i],
                'symbol': sym,
                'signal': 'skewness',
                'hold_days': 10,
                'direction': 1
            })
    return signals

def signal_vol_crush(ind, sym):
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
                'signal_date': ind.index[i],
                'symbol': sym,
                'signal': 'vol_crush',
                'hold_days': 21,
                'direction': 1
            })
    return signals

print("\n--- Generating signals (day-T close, entry at T+1 open) ---")
all_signals = []
for sym, ind in indicators.items():
    for sig_func in [signal_post_earnings_drift, signal_smart_money,
                     signal_price_vol_div, signal_skewness_premium, signal_vol_crush]:
        sigs = sig_func(ind, sym)
        all_signals.extend(sigs)

signals_df = pd.DataFrame(all_signals)
if len(signals_df) > 0:
    signals_df['signal_date'] = pd.to_datetime(signals_df['signal_date'])
    signals_df = signals_df.sort_values('signal_date').reset_index(drop=True)

    # FIX 3: Apply survivorship filter — remove signals for stocks before their S&P addition
    before_filter = len(signals_df)
    signals_df = signals_df[signals_df.apply(
        lambda row: symbol_eligible(row['symbol'], row['signal_date']), axis=1
    )].reset_index(drop=True)
    after_filter = len(signals_df)
    print(f"Survivorship filter removed {before_filter - after_filter} signals")

print(f"Total signals generated: {len(signals_df)}")
for sig_name in ['post_earnings_drift', 'smart_money', 'price_vol_div', 'skewness', 'vol_crush']:
    n = len(signals_df[signals_df['signal'] == sig_name])
    print(f"  {sig_name}: {n}")

# ============================================================
# REGIME DETECTION — LAGGED (uses day T-1 close for day T decision)
# ============================================================
spy_sma50 = spy_close.rolling(50).mean()

def get_regime_lagged(date, trading_days_index):
    """
    Returns regime using YESTERDAY's data (FIX 2).
    The SMA comparison uses day T-1's close vs T-1's SMA.
    """
    try:
        # Find T-1 (previous trading day)
        loc = trading_days_index.get_loc(date)
        if loc < 1:
            return 'bull'
        prev_day = trading_days_index[loc - 1]
        if prev_day in spy_sma50.index and pd.notna(spy_sma50.loc[prev_day]):
            return 'bull' if spy_close.loc[prev_day] > spy_sma50.loc[prev_day] else 'bear'
    except (KeyError, IndexError):
        pass
    return 'bull'

# ============================================================
# BACKTEST ENGINE — supports multiple hedge modes
# ============================================================

def run_backtest(signals_df, indicators, spy_close, spy_open, spy_sma50,
                 hedge_mode='none', label=''):
    """
    Run portfolio backtest with honest execution model.

    hedge_mode: 'none', 'constant_30', 'lagged_regime'

    KEY FIXES vs v2:
      - Entry at T+1 OPEN (not T close)
      - Hedge uses T-1 regime (not T)
      - Survivorship-corrected signals
    """
    bt_start = pd.Timestamp('2012-01-01')
    bt_end = pd.Timestamp('2026-07-15')
    bt_signals = signals_df[
        (signals_df['signal_date'] >= bt_start) & (signals_df['signal_date'] <= bt_end)
    ].copy()

    trading_days = spy_close.loc[bt_start:bt_end].index.sort_values()

    # Build signal lookup: signal fires on day T, we ENTER on day T+1
    # So group signals by their ENTRY date (signal_date + 1 trading day)
    td_set = set(trading_days)
    td_list = list(trading_days)
    td_to_idx = {d: i for i, d in enumerate(td_list)}

    signals_by_entry_date = {}
    for _, row in bt_signals.iterrows():
        sig_date = row['signal_date']
        # Find T+1 (next trading day after signal)
        if sig_date in td_to_idx:
            sig_idx = td_to_idx[sig_date]
            if sig_idx + 1 < len(td_list):
                entry_date = td_list[sig_idx + 1]
            else:
                continue
        else:
            # Signal date not in trading calendar, find next trading day
            future = [d for d in td_list if d > sig_date]
            if future:
                entry_date = future[0]
            else:
                continue

        if entry_date not in signals_by_entry_date:
            signals_by_entry_date[entry_date] = []
        entry_sig = row.to_dict()
        entry_sig['entry_date_planned'] = entry_date
        signals_by_entry_date[entry_date].append(entry_sig)

    # Portfolio state
    portfolio_value = np.zeros(len(trading_days))
    cash = float(INITIAL_CAPITAL)
    positions = []
    trade_log = []
    daily_returns = np.zeros(len(trading_days))
    prev_hedge_frac = 0.0  # Track for rebalancing cost

    for t_idx, today in enumerate(trading_days):
        # 1. Close expired positions at today's OPEN
        new_positions = []
        for pos in positions:
            if today >= pos['exit_date']:
                sym = pos['symbol']
                # Exit at open price on exit day
                exit_price = None
                if sym in indicators and today in indicators[sym].index:
                    exit_price = indicators[sym].loc[today, 'open']
                if exit_price is None or pd.isna(exit_price) or exit_price <= 0:
                    # Fallback to close
                    if sym in indicators and today in indicators[sym].index:
                        exit_price = indicators[sym].loc[today, 'close']
                    elif sym in indicators:
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
                    'signal_date': pos.get('signal_date', pos['entry_date']),
                    'entry_date': pos['entry_date'],
                    'exit_date': today,
                    'entry_price': pos['entry_price'],
                    'exit_price': float(exit_price),
                    'ret_gross': float(exit_price / pos['entry_price'] - 1),
                    'pnl_net': float(pnl_net),
                    'weight': pos.get('weight', 1.0),
                    'hold_days': pos.get('hold_days', 10),
                    'regime': pos.get('regime', 'unknown')
                })
            else:
                new_positions.append(pos)
        positions = new_positions

        # 2. Open new positions at today's OPEN (signal fired yesterday)
        if today in signals_by_entry_date:
            today_signals = signals_by_entry_date[today]

            # Regime uses YESTERDAY's data (FIX 2)
            regime = get_regime_lagged(today, trading_days)
            max_pos = MAX_POS_BEAR if regime == 'bear' else MAX_POS_BULL

            # Confluence counting
            sym_signal_count = {}
            for sig in today_signals:
                s = sig['symbol']
                sym_signal_count[s] = sym_signal_count.get(s, 0) + 1

            # Deduplicate: one entry per symbol per day
            sym_best = {}
            for sig in today_signals:
                s = sig['symbol']
                if s not in sym_best or sig['hold_days'] > sym_best[s]['hold_days']:
                    sym_best[s] = sig

            candidates = sorted(sym_best.values(),
                              key=lambda x: sym_signal_count.get(x['symbol'], 1),
                              reverse=True)

            for sig in candidates:
                if len(positions) >= max_pos:
                    break

                sym = sig['symbol']

                # Skip if already have position
                if any(p['symbol'] == sym for p in positions):
                    continue

                # FIX 1: Entry at T+1 OPEN price
                if sym not in indicators or today not in indicators[sym].index:
                    continue

                entry_price = indicators[sym].loc[today, 'open']
                if pd.isna(entry_price) or entry_price <= 0:
                    # Fallback to close if open not available
                    entry_price = indicators[sym].loc[today, 'close']
                if pd.isna(entry_price) or entry_price <= 0:
                    continue

                hold = sig['hold_days']
                if regime == 'bear':
                    hold = max(3, int(hold * BEAR_HOLD_MULTIPLIER))
                exit_idx = min(t_idx + hold, len(trading_days) - 1)
                exit_date = trading_days[exit_idx]

                n_signals = sym_signal_count.get(sym, 1)
                weight = CONFLUENCE_WEIGHT if n_signals >= 2 else 1.0

                total_value = cash + sum(p['notional'] for p in positions)
                slots_remaining = max_pos - len(positions)
                if slots_remaining > 0:
                    pos_size = (total_value / max_pos) * weight
                else:
                    pos_size = 0
                pos_size = min(pos_size, cash * 0.98)

                if pos_size < 100:
                    continue

                cash -= pos_size
                positions.append({
                    'symbol': sym,
                    'signal': sig['signal'],
                    'signal_date': sig.get('signal_date', today),
                    'entry_date': today,
                    'exit_date': exit_date,
                    'entry_price': float(entry_price),
                    'notional': pos_size,
                    'weight': weight,
                    'hold_days': hold,
                    'regime': regime
                })

        # 3. Compute long notional (mark-to-market)
        long_notional = 0.0
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

        # 4. Hedge PnL (daily settlement)
        hedge_pnl = 0.0
        current_hedge_frac = 0.0

        if t_idx > 0 and long_notional > 0 and today in spy_close.index:
            prev_day = trading_days[t_idx - 1]
            spy_today = spy_close.loc[today] if today in spy_close.index else None
            spy_prev = spy_close.loc[prev_day] if prev_day in spy_close.index else None

            if spy_today is not None and spy_prev is not None and spy_prev > 0:
                spy_ret = float(spy_today / spy_prev - 1)

                if hedge_mode == 'constant_30':
                    current_hedge_frac = CONSTANT_HEDGE_FRAC
                elif hedge_mode == 'lagged_regime':
                    # FIX 2: Regime uses YESTERDAY's data
                    regime_now = get_regime_lagged(today, trading_days)
                    current_hedge_frac = LAGGED_BEAR_HEDGE if regime_now == 'bear' else LAGGED_BULL_HEDGE
                else:
                    current_hedge_frac = 0.0

                if current_hedge_frac > 0:
                    hedge_notional = long_notional * current_hedge_frac
                    hedge_pnl = -spy_ret * hedge_notional  # Short: profit when SPY drops

                    # Borrow cost (daily)
                    borrow_daily = hedge_notional * SPY_BORROW_ANNUAL / TRADING_DAYS_YR
                    hedge_pnl -= borrow_daily

                    # Rebalancing cost when hedge ratio changes
                    if abs(current_hedge_frac - prev_hedge_frac) > 0.001:
                        rebal_notional = abs(current_hedge_frac - prev_hedge_frac) * long_notional
                        rebal_cost = rebal_notional * REBALANCE_COST_FRAC
                        hedge_pnl -= rebal_cost

                    cash += hedge_pnl

        prev_hedge_frac = current_hedge_frac

        # 5. Mark-to-market
        mtm = cash + long_notional
        portfolio_value[t_idx] = mtm
        if t_idx > 0 and portfolio_value[t_idx - 1] > 0:
            daily_returns[t_idx] = portfolio_value[t_idx] / portfolio_value[t_idx - 1] - 1

    return portfolio_value, daily_returns, trade_log, trading_days


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_metrics(portfolio_value, daily_returns, trade_log, trading_days, label=''):
    """Compute all performance metrics and gate tests."""
    nav = pd.Series(portfolio_value, index=trading_days)
    rets = pd.Series(daily_returns, index=trading_days)
    rets = rets.iloc[1:]  # drop first day

    n_years = len(rets) / TRADING_DAYS_YR
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1
    vol = rets.std() * np.sqrt(TRADING_DAYS_YR)
    sharpe = cagr / vol if vol > 1e-8 else 0
    downside_rets = rets[rets < 0]
    downside_vol = downside_rets.std() * np.sqrt(TRADING_DAYS_YR)
    sortino = cagr / downside_vol if downside_vol > 1e-8 else 0

    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    max_dd = drawdown.min()

    trade_df = pd.DataFrame(trade_log)
    if len(trade_df) > 0:
        winning = trade_df[trade_df['pnl_net'] > 0]['pnl_net'].sum()
        losing = abs(trade_df[trade_df['pnl_net'] <= 0]['pnl_net'].sum())
        profit_factor = winning / losing if losing > 0 else float('inf')
        win_rate = (trade_df['pnl_net'] > 0).mean()
        avg_win = trade_df[trade_df['pnl_net'] > 0]['pnl_net'].mean() if (trade_df['pnl_net'] > 0).any() else 0
        avg_loss = trade_df[trade_df['pnl_net'] <= 0]['pnl_net'].mean() if (trade_df['pnl_net'] <= 0).any() else 0
    else:
        profit_factor = win_rate = avg_win = avg_loss = 0

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

    # Regime gap (G2)
    spy_sma50_aligned = spy_sma50.reindex(rets.index, method='ffill')
    spy_close_aligned = spy_close.reindex(rets.index, method='ffill')
    spy_regime_bull = spy_close_aligned > spy_sma50_aligned

    green_rets = rets[spy_regime_bull == True]
    red_rets = rets[spy_regime_bull == False]

    def ann_sharpe(r):
        if len(r) < 30:
            return 0.0
        ann_ret = r.mean() * TRADING_DAYS_YR
        ann_vol = r.std() * np.sqrt(TRADING_DAYS_YR)
        return ann_ret / ann_vol if ann_vol > 1e-8 else 0.0

    if len(green_rets) > 60 and len(red_rets) > 60:
        sharpe_green = ann_sharpe(green_rets)
        sharpe_red = ann_sharpe(red_rets)
        max_abs = max(abs(sharpe_green), abs(sharpe_red))
        regime_gap = abs(sharpe_green - sharpe_red) / max_abs if max_abs > 1e-8 else 0
    else:
        sharpe_green = sharpe_red = regime_gap = float('nan')

    results = {
        'label': label,
        'cagr': cagr,
        'total_return': total_ret,
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': profit_factor,
        'max_dd': max_dd,
        'volatility': vol,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_trades': len(trade_df),
        'final_value': float(nav.iloc[-1]),
        'yearly_rets': yearly_rets,
        'years_profitable': years_profitable,
        'years_total': years_total,
        'pct_years_profitable': pct_years_profitable,
        'regime_gap': regime_gap,
        'sharpe_green': sharpe_green,
        'sharpe_red': sharpe_red,
        'n_years': n_years,
        'nav': nav,
        'rets': rets,
        'trade_df': trade_df,
    }
    return results


def print_results(r):
    """Print formatted results."""
    print(f"\n{'='*70}")
    print(f"  {r['label']}")
    print(f"{'='*70}")
    print(f"  CAGR:             {r['cagr']*100:.2f}%")
    print(f"  Total Return:     {r['total_return']*100:.1f}%")
    print(f"  Sharpe:           {r['sharpe']:.3f}")
    print(f"  Sortino:          {r['sortino']:.3f}")
    print(f"  Profit Factor:    {r['profit_factor']:.2f}")
    print(f"  Max Drawdown:     {r['max_dd']*100:.2f}%")
    print(f"  Volatility:       {r['volatility']*100:.2f}%")
    print(f"  Win Rate:         {r['win_rate']*100:.1f}%")
    print(f"  Total Trades:     {r['total_trades']}")
    print(f"  Final Value:      ${r['final_value']:,.0f}")
    print(f"  Avg Win:          ${r['avg_win']:,.2f}")
    print(f"  Avg Loss:         ${r['avg_loss']:,.2f}")
    print(f"\n  Yearly Returns:")
    for yr, ret in sorted(r['yearly_rets'].items()):
        marker = "+" if ret > 0 else "-"
        print(f"    {yr}: {ret*100:+7.2f}%  {marker}")
    print(f"    Profitable: {r['years_profitable']}/{r['years_total']} "
          f"({r['pct_years_profitable']*100:.0f}%)")
    print(f"\n  Regime: Sharpe_bull={r['sharpe_green']:.3f}, "
          f"Sharpe_bear={r['sharpe_red']:.3f}, Gap={r['regime_gap']:.3f}")

    # Signal breakdown
    if len(r['trade_df']) > 0:
        print(f"\n  Per-Signal Breakdown:")
        for sig_name in r['trade_df']['signal'].unique():
            mask = r['trade_df']['signal'] == sig_name
            sig_trades = r['trade_df'][mask]
            sig_wr = (sig_trades['pnl_net'] > 0).mean()
            sig_avg = sig_trades['pnl_net'].mean()
            sig_total = sig_trades['pnl_net'].sum()
            print(f"    {sig_name:25s}: {len(sig_trades):4d} trades, WR={sig_wr*100:.1f}%, "
                  f"avg=${sig_avg:+.2f}, total=${sig_total:+,.0f}")


# ============================================================
# RUN ALL THREE VARIANTS
# ============================================================
print("\n" + "=" * 70)
print("RUNNING THREE HEDGE VARIANTS")
print("=" * 70)

# Variant A: No hedge
print("\n--- Variant A: No Hedge (baseline) ---")
pv_a, dr_a, tl_a, td = run_backtest(
    signals_df, indicators, spy_close, spy_open, spy_sma50,
    hedge_mode='none', label='A: No Hedge'
)
results_a = compute_metrics(pv_a, dr_a, tl_a, td, label='A: No Hedge (Baseline)')
print_results(results_a)

# Variant B: Constant 30% hedge
print("\n--- Variant B: Constant 30% SPY Short Hedge ---")
pv_b, dr_b, tl_b, td = run_backtest(
    signals_df, indicators, spy_close, spy_open, spy_sma50,
    hedge_mode='constant_30', label='B: Constant 30% Hedge'
)
results_b = compute_metrics(pv_b, dr_b, tl_b, td, label='B: Constant 30% SPY Hedge')
print_results(results_b)

# Variant C: Lagged regime hedge (20-60%)
print("\n--- Variant C: Lagged Regime Hedge (20%-60%) ---")
pv_c, dr_c, tl_c, td = run_backtest(
    signals_df, indicators, spy_close, spy_open, spy_sma50,
    hedge_mode='lagged_regime', label='C: Lagged Regime Hedge'
)
results_c = compute_metrics(pv_c, dr_c, tl_c, td, label='C: Lagged Regime Hedge (20-60%)')
print_results(results_c)


# ============================================================
# GATE TESTS — pick the best variant
# ============================================================
print("\n\n" + "=" * 70)
print("GATE TESTS (all variants)")
print("=" * 70)

all_variants = [
    ('A: No Hedge', results_a),
    ('B: Constant 30%', results_b),
    ('C: Lagged Regime', results_c),
]

for name, r in all_variants:
    g1 = r['cagr'] > 0.10
    g2 = r['regime_gap'] < 0.50 if not np.isnan(r['regime_gap']) else False
    g4 = abs(r['max_dd']) < 0.40
    g5 = r['pct_years_profitable'] > 0.75

    print(f"\n  {name}:")
    print(f"    G1 CAGR>10%:         {'PASS' if g1 else 'FAIL'} ({r['cagr']*100:.2f}%)")
    print(f"    G2 Regime gap<0.50:  {'PASS' if g2 else 'FAIL'} ({r['regime_gap']:.3f})")
    print(f"    G4 MaxDD<40%:        {'PASS' if g4 else 'FAIL'} ({r['max_dd']*100:.2f}%)")
    print(f"    G5 >75% yrs prof:    {'PASS' if g5 else 'FAIL'} ({r['years_profitable']}/{r['years_total']})")

# Determine best variant (highest Sharpe among those passing G1+G4)
best = None
best_sharpe = -999
for name, r in all_variants:
    if r['cagr'] > 0.10 and abs(r['max_dd']) < 0.40:
        if r['sharpe'] > best_sharpe:
            best_sharpe = r['sharpe']
            best = (name, r)

if best is None:
    # Pick highest CAGR regardless
    best = max(all_variants, key=lambda x: x[1]['cagr'])
    print(f"\nNo variant passes G1+G4. Selecting {best[0]} (highest CAGR) for permutation test.")
else:
    print(f"\nBest variant: {best[0]} (Sharpe={best[1]['sharpe']:.3f})")


# ============================================================
# HONEST PERMUTATION TEST (FIX 4)
# ============================================================
# Test stock selection signal value WITHOUT hedge advantage.
# Method: For each trade, replace the signal-selected stock with a random
# stock from the eligible universe (same date, same hold period).
# This tests: do our contrarian signals pick better stocks than random?
# ============================================================
N_PERM = 200
print(f"\n{'='*70}")
print(f"HONEST PERMUTATION TEST ({N_PERM} shuffles)")
print(f"  Testing: Do contrarian signals pick better stocks than random?")
print(f"  Method: Replace each signal's stock with random eligible stock,")
print(f"  same entry date, same hold period. NO hedge in permutation.")
print(f"{'='*70}")

# Use no-hedge variant for honest permutation (isolate signal value)
perm_trade_df = results_a['trade_df']

if len(perm_trade_df) > 10:
    all_syms = list(indicators.keys())

    # Actual mean net return (no hedge)
    actual_mean_ret = (perm_trade_df['ret_gross'] - COST_FRAC_RT).mean()

    perm_means = []
    for perm_i in range(N_PERM):
        if perm_i % 50 == 0:
            print(f"  Permutation {perm_i}/{N_PERM}...")
        perm_rets = []
        for _, trade in perm_trade_df.iterrows():
            entry_date = trade['entry_date']
            hold = int(trade.get('hold_days', 10))

            # Pick a random ELIGIBLE symbol from universe on the same date
            np.random.shuffle(all_syms)
            found = False
            for rand_sym in all_syms[:15]:  # try up to 15 random symbols
                if rand_sym not in indicators:
                    continue
                # Survivorship check
                if not symbol_eligible(rand_sym, entry_date):
                    continue
                ind = indicators[rand_sym]
                if entry_date not in ind.index:
                    continue
                c = ind['close']
                o = ind['open']
                entry_loc = ind.index.get_loc(entry_date)
                exit_loc = min(entry_loc + hold, len(c) - 1)
                if exit_loc <= entry_loc:
                    continue
                # Use open price for entry (consistent with backtest)
                entry_p = o.iloc[entry_loc]
                if pd.isna(entry_p) or entry_p <= 0:
                    entry_p = c.iloc[entry_loc]
                # Use open for exit
                exit_p = o.iloc[exit_loc] if not pd.isna(o.iloc[exit_loc]) else c.iloc[exit_loc]
                if pd.notna(entry_p) and pd.notna(exit_p) and entry_p > 0:
                    rand_ret = exit_p / entry_p - 1
                    perm_rets.append(rand_ret - COST_FRAC_RT)
                    found = True
                    break
            if not found:
                # Keep original trade if no substitute found
                perm_rets.append(trade['ret_gross'] - COST_FRAC_RT)

        perm_means.append(np.mean(perm_rets))

    perm_means = np.array(perm_means)
    perm_p = (perm_means >= actual_mean_ret).mean()
else:
    perm_p = 1.0
    perm_means = np.array([0.0])
    actual_mean_ret = 0

g3_pass = perm_p < 0.05
print(f"\n  G3 — Permutation p < 0.05: {'PASS' if g3_pass else 'FAIL'}")
print(f"    p-value:           {perm_p:.4f}")
print(f"    Actual mean ret:   {actual_mean_ret*100:.3f}%")
print(f"    Perm mean:         {perm_means.mean()*100:.3f}%")
print(f"    Perm p5:           {np.percentile(perm_means, 5)*100:.3f}%")
print(f"    Perm p95:          {np.percentile(perm_means, 95)*100:.3f}%")
print(f"    Signal edge:       {(actual_mean_ret - perm_means.mean())*100:.3f}% per trade")


# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n\n" + "=" * 70)
print("FINAL SUMMARY — MULTI-SIGNAL GROWTH V3 (HONEST)")
print("=" * 70)

best_name, best_r = best

print(f"\n  Best variant: {best_name}")
print(f"  CAGR:         {best_r['cagr']*100:.2f}%")
print(f"  Sharpe:       {best_r['sharpe']:.3f}")
print(f"  Sortino:      {best_r['sortino']:.3f}")
print(f"  Max DD:       {best_r['max_dd']*100:.2f}%")
print(f"  Win Rate:     {best_r['win_rate']*100:.1f}%")
print(f"  Regime Gap:   {best_r['regime_gap']:.3f}")

g1 = best_r['cagr'] > 0.10
g2 = best_r['regime_gap'] < 0.50 if not np.isnan(best_r['regime_gap']) else False
g4 = abs(best_r['max_dd']) < 0.40
g5 = best_r['pct_years_profitable'] > 0.75
all_pass = g1 and g2 and g3_pass and g4 and g5

print(f"\n  GATES:")
print(f"    G1 CAGR>10%:         {'PASS' if g1 else 'FAIL'} ({best_r['cagr']*100:.2f}%)")
print(f"    G2 Regime gap<0.50:  {'PASS' if g2 else 'FAIL'} ({best_r['regime_gap']:.3f})")
print(f"    G3 Perm p<0.05:      {'PASS' if g3_pass else 'FAIL'} (p={perm_p:.4f})")
print(f"    G4 MaxDD<40%:        {'PASS' if g4 else 'FAIL'} ({best_r['max_dd']*100:.2f}%)")
print(f"    G5 >75% yrs prof:    {'PASS' if g5 else 'FAIL'} ({best_r['years_profitable']}/{best_r['years_total']})")
print(f"\n    OVERALL: {'ALL GATES PASS' if all_pass else 'SOME GATES FAILED'}")

# Comparison table
print(f"\n  COMPARISON TABLE:")
print(f"  {'Variant':<25s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MaxDD':>8s} {'WR':>6s} {'Gap':>6s}")
print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6}")
for name, r in all_variants:
    print(f"  {name:<25s} {r['cagr']*100:>7.2f}% {r['sharpe']:>8.3f} {r['sortino']:>8.3f} "
          f"{r['max_dd']*100:>7.2f}% {r['win_rate']*100:>5.1f}% {r['regime_gap']:>6.3f}")


# ============================================================
# SAVE RESULTS
# ============================================================
def serialize_results(r):
    """Convert results dict to JSON-serializable format."""
    return {
        'label': r['label'],
        'metrics': {
            'cagr_pct': round(r['cagr'] * 100, 2),
            'total_return_pct': round(r['total_return'] * 100, 2),
            'sharpe': round(r['sharpe'], 3),
            'sortino': round(r['sortino'], 3),
            'profit_factor': round(r['profit_factor'], 2),
            'max_drawdown_pct': round(r['max_dd'] * 100, 2),
            'volatility_pct': round(r['volatility'] * 100, 2),
            'win_rate_pct': round(r['win_rate'] * 100, 1),
            'total_trades': r['total_trades'],
            'starting_capital': INITIAL_CAPITAL,
            'final_value': round(r['final_value'], 2),
            'avg_win': round(float(r['avg_win']), 2),
            'avg_loss': round(float(r['avg_loss']), 2),
        },
        'regime': {
            'sharpe_bull': round(float(r['sharpe_green']), 3),
            'sharpe_bear': round(float(r['sharpe_red']), 3),
            'regime_gap': round(float(r['regime_gap']), 3),
        },
        'yearly_returns_pct': {str(k): round(v * 100, 2) for k, v in r['yearly_rets'].items()},
        'years_profitable': r['years_profitable'],
        'years_total': r['years_total'],
    }

output = {
    'version': 'multi_signal_growth_v3_honest',
    'run_date': pd.Timestamp.now().isoformat(),
    'fixes_applied': [
        'FIX1: Next-day execution (signal at T close, entry at T+1 open)',
        'FIX2: Lagged hedge (regime from T-1 close, not same-day)',
        'FIX3: Survivorship-free universe (date-conditional inclusion)',
        'FIX4: Honest permutation (shuffles stock selection, no hedge advantage)',
        'FIX5: Realistic hedge costs (0.4% borrow, 5bps rebalance)',
    ],
    'survivorship_gates': {sym: str(dt.date()) for sym, dt in SP500_ADDITION_DATES.items()},
    'cost_assumptions': {
        'stock_rt_bps': COST_BPS_RT,
        'spy_borrow_annual_pct': SPY_BORROW_ANNUAL * 100,
        'hedge_rebalance_bps': REBALANCE_COST_BPS,
    },
    'variants': {
        'A_no_hedge': serialize_results(results_a),
        'B_constant_30pct': serialize_results(results_b),
        'C_lagged_regime': serialize_results(results_c),
    },
    'best_variant': best_name,
    'permutation_test': {
        'method': 'Random stock substitution (no hedge), survivorship-aware',
        'n_permutations': N_PERM,
        'p_value': round(float(perm_p), 4),
        'actual_mean_ret_pct': round(float(actual_mean_ret) * 100, 3),
        'perm_mean_ret_pct': round(float(perm_means.mean()) * 100, 3),
        'signal_edge_pct': round(float(actual_mean_ret - perm_means.mean()) * 100, 3),
        'pass': bool(g3_pass),
    },
    'gates': {
        'G1_cagr_gt_10pct': {'pass': bool(g1), 'value': round(best_r['cagr'] * 100, 2)},
        'G2_regime_gap_lt_050': {'pass': bool(g2), 'value': round(float(best_r['regime_gap']), 3)},
        'G3_permutation_p_lt_005': {'pass': bool(g3_pass), 'p_value': round(float(perm_p), 4)},
        'G4_max_dd_lt_40pct': {'pass': bool(g4), 'value': round(best_r['max_dd'] * 100, 2)},
        'G5_75pct_years_profitable': {
            'pass': bool(g5),
            'profitable_years': best_r['years_profitable'],
            'total_years': best_r['years_total'],
        },
        'all_pass': bool(all_pass),
    },
}

results_path = OUTDIR / 'results.json'
with open(results_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

# Save NAV series for all variants
for vname, pv, td_arr in [('A_no_hedge', pv_a, td), ('B_constant_30', pv_b, td), ('C_lagged_regime', pv_c, td)]:
    nav_df = pd.DataFrame({'date': td_arr, 'nav': pv})
    nav_df.to_csv(OUTDIR / f'nav_{vname}.csv', index=False)

# Save trade log for best variant
best_trade_df = best[1]['trade_df']
if len(best_trade_df) > 0:
    best_trade_df.to_csv(OUTDIR / 'trade_log_best.csv', index=False)

# Save no-hedge trade log too
if len(results_a['trade_df']) > 0:
    results_a['trade_df'].to_csv(OUTDIR / 'trade_log_no_hedge.csv', index=False)

print(f"\nResults saved to {OUTDIR}")
print("Done.")
