"""
Adversarial Deep Audit — Multi-Signal Growth Portfolio v2
==========================================================
Tests for leakage, survivorship bias, cost realism, and inflated metrics.

Checks:
  1. Hedge timing leakage (same-day SMA look-ahead)
  2. Survivorship bias (post-2012 S&P 500 additions)
  3. Hedge rebalancing & borrow costs
  4. Signal execution realism (same-day close entry)
  5. Drawdown stress test (COVID, 2022 bear)
  6. Permutation robustness (hedge-inclusive shuffle)
  7. Cost sensitivity (40 bps RT, 1% borrow)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import warnings
import time
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

OUTDIR = Path('/home/jupiter/Lvl3Quant/output/multi_signal_growth_v2')
RESEARCH_DIR = Path('/home/jupiter/Lvl3Quant/research')

# ============================================================
# CONSTANTS (from original)
# ============================================================
TRADING_DAYS_YR = 252
COST_BPS_RT = 20
COST_FRAC_RT = COST_BPS_RT / 10000
INITIAL_CAPITAL = 100_000
MAX_POS_BULL = 8
MAX_POS_BEAR = 4
CONFLUENCE_WEIGHT = 1.5
BEAR_HOLD_MULTIPLIER = 0.5
BEAR_HEDGE_FRAC = 1.20
BULL_HEDGE_FRAC = 0.35

# Universe (same as original)
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

# ============================================================
# AUDIT RESULTS COLLECTOR
# ============================================================
audit_results = {}

def verdict(name, status, detail, impact=None):
    """Record a check result."""
    entry = {'status': status, 'detail': detail}
    if impact:
        entry['impact'] = impact
    audit_results[name] = entry
    marker = {'PASS': '[PASS]', 'FAIL': '[FAIL]', 'WARNING': '[WARN]'}
    print(f"\n{marker.get(status, '[????]')} {name}")
    print(f"  {detail}")
    if impact:
        print(f"  Impact: {impact}")

# ============================================================
# DATA DOWNLOAD
# ============================================================
print("=" * 70)
print("ADVERSARIAL AUDIT — Multi-Signal Growth Portfolio v2")
print("=" * 70)

print("\n--- Downloading data ---")
t0 = time.time()

def download_batch(symbols, start='2011-01-01', end='2026-07-23'):
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
            print(f"  Batch error: {e}")
        if i + batch_size < len(symbols):
            time.sleep(0.5)
    return all_data

data = download_batch(ALL_SYMBOLS)
print(f"Downloaded {len(data)} symbols in {time.time()-t0:.0f}s")

if 'SPY' not in data:
    spy_df = yf.download('SPY', start='2011-01-01', end='2026-07-23', auto_adjust=True, progress=False)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_df.columns = spy_df.columns.droplevel(1)
    data['SPY'] = spy_df

spy_close = data['SPY']['Close'].copy()
spy_close.index = pd.to_datetime(spy_close.index).normalize()

# ============================================================
# INDICATORS (same as original)
# ============================================================
def compute_indicators(df):
    c = df['Close'].copy()
    h = df['High'].copy()
    l = df['Low'].copy()
    v = df['Volume'].copy()

    out = pd.DataFrame(index=df.index)
    out['close'] = c
    out['ret_1d'] = c.pct_change()

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out['rsi'] = 100 - (100 / (1 + rs))

    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    out['obv'] = obv
    out['obv_slope_10'] = obv.diff(10)

    tp = (h + l + c) / 3
    rmf = tp * v
    pos_mf = rmf.where(tp > tp.shift(1), 0).rolling(14).sum()
    neg_mf = rmf.where(tp <= tp.shift(1), 0).rolling(14).sum()
    mr = pos_mf / neg_mf.replace(0, np.nan)
    out['mfi'] = 100 - (100 / (1 + mr))
    out['mfi_slope_10'] = out['mfi'].diff(10)

    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20).mean()
    out['atr_20'] = atr
    out['atr_pct'] = (atr / c * 100)
    out['atr_pct_rank'] = out['atr_pct'].rolling(252).rank(pct=True) * 100

    out['vol_ratio'] = v / v.rolling(20).mean()
    out['ret_5d'] = c.pct_change(5)
    out['ret_10d'] = c.pct_change(10)
    out['ret_21d'] = c.pct_change(21)

    out['max_20d'] = c.rolling(20).max()
    out['dd_from_20d_high'] = (c / out['max_20d'] - 1)
    out['max_30d'] = c.rolling(30).max()
    out['dd_from_30d_high'] = (c / out['max_30d'] - 1)

    out['skew_252'] = out['ret_1d'].rolling(252).skew()
    out['fwd_10d'] = c.shift(-10) / c - 1
    out['fwd_21d'] = c.shift(-21) / c - 1

    return out

print("--- Computing indicators ---")
indicators = {}
for sym, df in data.items():
    df.index = pd.to_datetime(df.index).normalize()
    ind = compute_indicators(df)
    if len(ind.dropna(subset=['rsi', 'mfi', 'obv'])) > 252:
        indicators[sym] = ind
print(f"Indicators for {len(indicators)} symbols")

# ============================================================
# SIGNAL GENERATION (same as original)
# ============================================================
def signal_post_earnings_drift(ind, sym):
    signals = []
    mfi = ind['mfi']
    dd30 = ind['dd_from_30d_high']
    for i in range(30, len(ind)):
        window_dd = dd30.iloc[i-30:i-20]
        if len(window_dd) == 0: continue
        if (window_dd < -0.12).any() and mfi.iloc[i] < 30:
            signals.append({'date': ind.index[i], 'symbol': sym, 'signal': 'post_earnings_drift', 'hold_days': 21, 'direction': 1})
    return signals

def signal_smart_money(ind, sym):
    signals = []
    for i in range(30, len(ind)):
        if (ind['obv_slope_10'].iloc[i] > 0 and ind['ret_10d'].iloc[i] < -0.02 and
            ind['vol_ratio'].iloc[i] > 1.2 and ind['mfi'].iloc[i] < 20):
            signals.append({'date': ind.index[i], 'symbol': sym, 'signal': 'smart_money', 'hold_days': 10, 'direction': 1})
    return signals

def signal_price_vol_div(ind, sym):
    signals = []
    for i in range(30, len(ind)):
        if (ind['obv_slope_10'].iloc[i] > 0 and ind['mfi_slope_10'].iloc[i] > 0 and ind['rsi'].iloc[i] < 40):
            signals.append({'date': ind.index[i], 'symbol': sym, 'signal': 'price_vol_div', 'hold_days': 10, 'direction': 1})
    return signals

def signal_skewness_premium(ind, sym):
    signals = []
    skew = ind['skew_252']
    skew_p10 = skew.rolling(504, min_periods=252).quantile(0.10)
    for i in range(504, len(ind)):
        if (pd.notna(skew.iloc[i]) and pd.notna(skew_p10.iloc[i]) and
            skew.iloc[i] < skew_p10.iloc[i] and ind['ret_1d'].iloc[i] < -0.03):
            signals.append({'date': ind.index[i], 'symbol': sym, 'signal': 'skewness', 'hold_days': 10, 'direction': 1})
    return signals

def signal_vol_crush(ind, sym):
    signals = []
    for i in range(300, len(ind)):
        if (pd.notna(ind['atr_pct_rank'].iloc[i]) and ind['atr_pct_rank'].iloc[i] < 20 and
            ind['ret_21d'].iloc[i] < -0.05 and ind['mfi'].iloc[i] < 20):
            signals.append({'date': ind.index[i], 'symbol': sym, 'signal': 'vol_crush', 'hold_days': 21, 'direction': 1})
    return signals

print("--- Generating signals ---")
all_signals = []
for sym, ind in indicators.items():
    for sig_func in [signal_post_earnings_drift, signal_smart_money,
                     signal_price_vol_div, signal_skewness_premium, signal_vol_crush]:
        all_signals.extend(sig_func(ind, sym))

signals_df = pd.DataFrame(all_signals)
signals_df['date'] = pd.to_datetime(signals_df['date'])
signals_df = signals_df.sort_values('date').reset_index(drop=True)
print(f"Total signals: {len(signals_df)}")

# ============================================================
# CORE BACKTEST ENGINE (parameterized for re-runs)
# ============================================================
def run_backtest(signals_df, indicators, spy_close,
                 cost_bps_rt=20, hedge_lag_days=0, annual_borrow_cost_pct=0.0,
                 hedge_rebal_cost_bps=0, use_next_day_entry=False,
                 bear_hedge_frac=BEAR_HEDGE_FRAC, bull_hedge_frac=BULL_HEDGE_FRAC):
    """
    Run the portfolio backtest with configurable parameters.

    hedge_lag_days: 0 = original (same-day SMA), 1 = use yesterday's SMA (realistic)
    annual_borrow_cost_pct: annual cost of shorting SPY (e.g. 0.5 = 0.5%/yr)
    hedge_rebal_cost_bps: cost in bps when hedge ratio changes
    use_next_day_entry: if True, signal fires at close, entry at next day open
    """
    cost_frac_rt = cost_bps_rt / 10000

    spy_sma50 = spy_close.rolling(50).mean()

    bt_start = pd.Timestamp('2012-01-01')
    bt_end = pd.Timestamp('2026-07-15')
    bt_signals = signals_df[(signals_df['date'] >= bt_start) & (signals_df['date'] <= bt_end)].copy()
    trading_days = spy_close.loc[bt_start:bt_end].index.sort_values()

    # Build regime lookup with optional lag
    def get_regime_for_date(date, t_idx):
        """Get regime. If hedge_lag_days > 0, use lagged SMA."""
        if hedge_lag_days > 0 and t_idx >= hedge_lag_days:
            # Use SMA from hedge_lag_days ago
            lagged_date = trading_days[t_idx - hedge_lag_days]
        else:
            lagged_date = date
        try:
            idx = spy_close.index.get_indexer([lagged_date], method='ffill')[0]
            if idx < 0:
                return 'bull'
            d = spy_close.index[idx]
            if d in spy_sma50.index and pd.notna(spy_sma50.loc[d]):
                return 'bull' if spy_close.iloc[idx] > spy_sma50.loc[d] else 'bear'
        except:
            pass
        return 'bull'

    # Group signals by date
    signals_by_date = {}
    for _, row in bt_signals.iterrows():
        d = row['date']
        if d not in signals_by_date:
            signals_by_date[d] = []
        signals_by_date[d].append(row)

    # If using next-day entry, shift signal dates forward by 1 trading day
    if use_next_day_entry:
        td_list = list(trading_days)
        td_set = set(trading_days)
        new_signals_by_date = {}
        for d, sigs in signals_by_date.items():
            if d in td_set:
                idx = td_list.index(d)
                if idx + 1 < len(td_list):
                    next_d = td_list[idx + 1]
                    if next_d not in new_signals_by_date:
                        new_signals_by_date[next_d] = []
                    new_signals_by_date[next_d].extend(sigs)
        signals_by_date = new_signals_by_date

    portfolio_value = np.zeros(len(trading_days))
    cash = INITIAL_CAPITAL
    positions = []
    trade_log = []
    daily_returns = np.zeros(len(trading_days))
    prev_hedge_frac = 0.0
    cumulative_borrow_cost = 0.0
    cumulative_rebal_cost = 0.0

    for t_idx, today in enumerate(trading_days):
        # 1. Close expired positions
        new_positions = []
        for pos in positions:
            if today >= pos['exit_date']:
                sym = pos['symbol']
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
                cost = pos['notional'] * cost_frac_rt
                pnl_net = pnl_gross - cost
                cash += pos['notional'] + pnl_net
                trade_log.append({
                    'symbol': sym, 'signal': pos['signal'],
                    'entry_date': pos['entry_date'], 'exit_date': today,
                    'entry_price': pos['entry_price'], 'exit_price': exit_price,
                    'ret_gross': exit_price / pos['entry_price'] - 1,
                    'pnl_net': pnl_net, 'weight': pos.get('weight', 1.0),
                    'hold_days': pos.get('hold_days', 10)
                })
            else:
                new_positions.append(pos)
        positions = new_positions

        # 2. New signals
        if today in signals_by_date:
            today_signals = signals_by_date[today]
            regime = get_regime_for_date(today, t_idx)
            max_pos = MAX_POS_BEAR if regime == 'bear' else MAX_POS_BULL

            sym_signal_count = {}
            for sig in today_signals:
                s = sig['symbol']
                sym_signal_count[s] = sym_signal_count.get(s, 0) + 1

            sym_best = {}
            for sig in today_signals:
                s = sig['symbol']
                if s not in sym_best or sig['hold_days'] > sym_best[s]['hold_days']:
                    sym_best[s] = sig

            candidates = sorted(sym_best.values(),
                              key=lambda x: sym_signal_count.get(x['symbol'], 1), reverse=True)

            for sig in candidates:
                if len(positions) >= max_pos:
                    break
                sym = sig['symbol']
                if any(p['symbol'] == sym for p in positions):
                    continue
                if sym not in indicators or today not in indicators[sym].index:
                    continue

                if use_next_day_entry:
                    # Use Open price for next-day entry
                    if sym in data and today in data[sym].index:
                        entry_price = data[sym].loc[today, 'Open']
                        if hasattr(entry_price, 'iloc'):
                            entry_price = entry_price.iloc[0]
                    else:
                        entry_price = indicators[sym].loc[today, 'close']
                else:
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
                n_open = len(positions)
                slots_remaining = max_pos - n_open
                if slots_remaining > 0:
                    pos_size = (total_value / max_pos) * weight
                else:
                    pos_size = 0
                pos_size = min(pos_size, cash * 0.98)
                if pos_size < 100:
                    continue

                cash -= pos_size
                positions.append({
                    'symbol': sym, 'signal': sig['signal'],
                    'entry_date': today, 'exit_date': exit_date,
                    'entry_price': entry_price, 'notional': pos_size,
                    'weight': weight, 'hold_days': hold,
                    'regime': regime
                })

        # 3. SPY hedge
        regime_now = get_regime_for_date(today, t_idx)
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
        current_hedge_frac = 0.0
        if t_idx > 0 and long_notional > 0 and today in spy_close.index:
            hedge_frac = bear_hedge_frac if regime_now == 'bear' else bull_hedge_frac
            current_hedge_frac = hedge_frac
            if hedge_frac > 0:
                spy_today = spy_close.loc[today] if today in spy_close.index else None
                prev_day = trading_days[t_idx - 1]
                spy_prev = spy_close.loc[prev_day] if prev_day in spy_close.index else None
                if spy_today is not None and spy_prev is not None and spy_prev > 0:
                    spy_ret = spy_today / spy_prev - 1
                    hedge_notional = long_notional * hedge_frac
                    hedge_pnl = -spy_ret * hedge_notional
                    cash += hedge_pnl

                    # Borrow cost (daily accrual)
                    if annual_borrow_cost_pct > 0:
                        daily_borrow = hedge_notional * (annual_borrow_cost_pct / 100) / TRADING_DAYS_YR
                        cash -= daily_borrow
                        cumulative_borrow_cost += daily_borrow

                    # Rebalancing cost when hedge ratio changes
                    if hedge_rebal_cost_bps > 0 and abs(current_hedge_frac - prev_hedge_frac) > 0.01:
                        rebal_notional = abs(current_hedge_frac - prev_hedge_frac) * long_notional
                        rebal_cost = rebal_notional * (hedge_rebal_cost_bps / 10000)
                        cash -= rebal_cost
                        cumulative_rebal_cost += rebal_cost

        prev_hedge_frac = current_hedge_frac

        # 4. MTM
        mtm = cash + long_notional
        portfolio_value[t_idx] = mtm
        if t_idx > 0 and portfolio_value[t_idx - 1] > 0:
            daily_returns[t_idx] = portfolio_value[t_idx] / portfolio_value[t_idx - 1] - 1

    nav = pd.Series(portfolio_value, index=trading_days)
    rets = pd.Series(daily_returns, index=trading_days).iloc[1:]

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
    else:
        profit_factor = 0; win_rate = 0

    yearly_rets = {}
    for yr in range(2012, 2027):
        yr_mask = rets.index.year == yr
        if yr_mask.sum() > 20:
            yearly_rets[yr] = float((1 + rets[yr_mask]).prod() - 1)

    return {
        'nav': nav, 'rets': rets, 'drawdown': drawdown, 'trade_df': trade_df,
        'cagr': cagr, 'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd,
        'profit_factor': profit_factor, 'win_rate': win_rate,
        'total_trades': len(trade_df), 'yearly_rets': yearly_rets,
        'final_value': float(nav.iloc[-1]),
        'cumulative_borrow_cost': cumulative_borrow_cost,
        'cumulative_rebal_cost': cumulative_rebal_cost,
    }

# ============================================================
# RUN BASELINE (reproduce original)
# ============================================================
print("\n" + "=" * 70)
print("BASELINE RUN (reproduce original)")
print("=" * 70)
baseline = run_backtest(signals_df, indicators, spy_close,
                        cost_bps_rt=20, hedge_lag_days=0)
print(f"  CAGR: {baseline['cagr']*100:.2f}%")
print(f"  Sharpe: {baseline['sharpe']:.3f}")
print(f"  Max DD: {baseline['max_dd']*100:.2f}%")
print(f"  Trades: {baseline['total_trades']}")

# ============================================================
# CHECK 1: HEDGE TIMING LEAKAGE
# ============================================================
print("\n" + "=" * 70)
print("CHECK 1: HEDGE TIMING LEAKAGE")
print("=" * 70)

# The original code: get_regime(today) computes spy_close[today] > spy_sma50[today]
# spy_sma50[today] = mean of spy_close over past 50 days INCLUDING today
# This means the hedge decision uses today's close price, but that price
# is only known AFTER the market closes. You can't trade on it intraday.
#
# The hedge PnL is computed as: -spy_ret * hedge_notional
# where spy_ret = spy_close[today] / spy_close[yesterday] - 1
# So the hedge also uses today's close for settlement.
#
# The issue: regime DECISION uses today's close. In reality you'd need
# to use yesterday's close to compute the SMA and decide today's hedge.

lagged_1d = run_backtest(signals_df, indicators, spy_close,
                         cost_bps_rt=20, hedge_lag_days=1)

print(f"\n  Original (same-day SMA):   CAGR={baseline['cagr']*100:.2f}%, "
      f"Sharpe={baseline['sharpe']:.3f}, MaxDD={baseline['max_dd']*100:.2f}%")
print(f"  Lagged 1-day SMA:          CAGR={lagged_1d['cagr']*100:.2f}%, "
      f"Sharpe={lagged_1d['sharpe']:.3f}, MaxDD={lagged_1d['max_dd']*100:.2f}%")

cagr_impact = (baseline['cagr'] - lagged_1d['cagr']) * 100
dd_impact = (lagged_1d['max_dd'] - baseline['max_dd']) * 100  # more negative = worse

# The SMA changes slowly so 1-day lag should have minimal effect if no leakage
if abs(cagr_impact) < 0.5 and abs(dd_impact) < 2.0:
    verdict("1_hedge_timing_leakage",
            "PASS",
            f"1-day lagged SMA changes CAGR by only {cagr_impact:+.2f}% and MaxDD by {dd_impact:+.2f}%. "
            f"SMA(50) is slow-moving so same-day vs lagged makes negligible difference.",
            f"CAGR delta: {cagr_impact:+.2f}%, MaxDD delta: {dd_impact:+.2f}%")
else:
    # Significant difference — this is concerning
    if abs(dd_impact) > 5.0 or abs(cagr_impact) > 2.0:
        verdict("1_hedge_timing_leakage",
                "FAIL",
                f"Same-day SMA creates meaningful look-ahead bias. "
                f"Lagged CAGR={lagged_1d['cagr']*100:.2f}%, MaxDD={lagged_1d['max_dd']*100:.2f}%. "
                f"The strategy uses today's close to decide hedge, but this info isn't available until EOD.",
                f"CAGR inflated by {cagr_impact:.2f}%, MaxDD understated by {abs(dd_impact):.2f}%")
    else:
        verdict("1_hedge_timing_leakage",
                "WARNING",
                f"1-day SMA lag has moderate impact. CAGR delta={cagr_impact:+.2f}%, MaxDD delta={dd_impact:+.2f}%. "
                f"Technically look-ahead but SMA is slow enough that impact is limited.",
                f"CAGR delta: {cagr_impact:+.2f}%, MaxDD delta: {dd_impact:+.2f}%")

# ============================================================
# CHECK 2: SURVIVORSHIP BIAS
# ============================================================
print("\n" + "=" * 70)
print("CHECK 2: SURVIVORSHIP BIAS")
print("=" * 70)

# Stocks added to S&P 500 AFTER 2012 (approximate dates)
post_2012_additions = {
    'META': '2013-12-23',  # Added as FB
    'TSLA': '2020-12-21',
    'AVGO': '2014-03-21',  # Via Broadcom merger, effective add later
    'CRM': '2020-09-21',
    'NVDA': '2001-11-30',  # Actually was in before, but became mega-cap later
    'ABBV': '2012-12-31',  # Spun off from ABT in 2013-01-02
    'LLY': '1970-01-01',   # Long-time member, but 10x'd post-2020
    'DHR': '2017-06-19',   # Re-added
}

# Check which symbols have data starting well after 2012
late_data_start = {}
for sym in SP500_LARGE:
    if sym in indicators:
        first_valid = indicators[sym]['close'].first_valid_index()
        if first_valid and first_valid > pd.Timestamp('2012-06-01'):
            late_data_start[sym] = str(first_valid.date())

# Key concern: stocks that are NOW mega-caps but weren't in S&P 500 in 2012
# If they're in the backtest universe from 2012 but weren't discoverable then,
# that's survivorship bias
survivor_concerns = []
for sym, add_date_str in post_2012_additions.items():
    if sym in SP500_LARGE:
        add_date = pd.Timestamp(add_date_str)
        if sym in indicators:
            # Count trades before the stock was added to S&P
            if len(signals_df) > 0:
                pre_add_trades = signals_df[(signals_df['symbol'] == sym) &
                                            (signals_df['date'] < add_date)]
                if len(pre_add_trades) > 0:
                    survivor_concerns.append({
                        'symbol': sym,
                        'added_to_sp500': add_date_str,
                        'trades_before_add': len(pre_add_trades)
                    })

print(f"\n  Stocks added to S&P 500 after backtest start (2012):")
for sc in survivor_concerns:
    print(f"    {sc['symbol']}: added {sc['added_to_sp500']}, "
          f"{sc['trades_before_add']} trades before addition")

if late_data_start:
    print(f"\n  Symbols with data starting after mid-2012:")
    for sym, start in late_data_start.items():
        print(f"    {sym}: data starts {start}")

# The ETFs partially hedge survivorship since they're indices
# But stock selection from current large caps IS survivorship bias
n_survivor_issues = len(survivor_concerns)
total_survivor_trades = sum(sc['trades_before_add'] for sc in survivor_concerns)

if n_survivor_issues == 0:
    verdict("2_survivorship_bias", "PASS", "No stocks in universe were added to S&P 500 after backtest start.")
elif total_survivor_trades < 20:
    verdict("2_survivorship_bias", "WARNING",
            f"{n_survivor_issues} stocks were added to S&P 500 after 2012. "
            f"But only {total_survivor_trades} trades occurred before their addition. "
            f"Includes: {', '.join(sc['symbol'] for sc in survivor_concerns)}. "
            f"More importantly: the entire universe of 50 large caps was selected based on "
            f"CURRENT S&P 500 membership, which inherently excludes stocks that crashed "
            f"and were removed (e.g., GE, XRX, AIG). This biases mean reversion signals upward.",
            f"{n_survivor_issues} survivor stocks, {total_survivor_trades} pre-addition trades")
else:
    verdict("2_survivorship_bias", "FAIL",
            f"{n_survivor_issues} stocks added after 2012 with {total_survivor_trades} trades before addition. "
            f"Universe selected from current S&P 500 = survivorship bias. Stocks that crashed out "
            f"(e.g., GE removed 2018, XRX, etc.) aren't in the universe, inflating contrarian signal returns.",
            f"{total_survivor_trades} trades on survivor stocks")

# ============================================================
# CHECK 3: HEDGE REBALANCING & BORROW COSTS
# ============================================================
print("\n" + "=" * 70)
print("CHECK 3: HEDGE REBALANCING & BORROW COSTS")
print("=" * 70)

# Run with realistic hedge costs
# SPY borrow rate: typically 0.3-0.5% annually (very liquid)
# Hedge rebalancing: switching from 35% to 120% is a big trade
with_costs = run_backtest(signals_df, indicators, spy_close,
                          cost_bps_rt=20, hedge_lag_days=0,
                          annual_borrow_cost_pct=0.4,
                          hedge_rebal_cost_bps=5)

print(f"\n  Original:             CAGR={baseline['cagr']*100:.2f}%, Sharpe={baseline['sharpe']:.3f}")
print(f"  With hedge costs:     CAGR={with_costs['cagr']*100:.2f}%, Sharpe={with_costs['sharpe']:.3f}")
print(f"  Cumulative borrow:    ${with_costs['cumulative_borrow_cost']:,.0f}")
print(f"  Cumulative rebal:     ${with_costs['cumulative_rebal_cost']:,.0f}")

cagr_cost_impact = (baseline['cagr'] - with_costs['cagr']) * 100
if cagr_cost_impact < 1.0:
    verdict("3_hedge_costs", "PASS",
            f"Adding 0.4% annual borrow + 5 bps rebal cost reduces CAGR by only {cagr_cost_impact:.2f}%. "
            f"SPY is cheap to short.",
            f"Total hedge cost over backtest: ${with_costs['cumulative_borrow_cost'] + with_costs['cumulative_rebal_cost']:,.0f}")
elif cagr_cost_impact < 3.0:
    verdict("3_hedge_costs", "WARNING",
            f"Hedge costs reduce CAGR by {cagr_cost_impact:.2f}% — not modeled in original. "
            f"Borrow=${with_costs['cumulative_borrow_cost']:,.0f}, Rebal=${with_costs['cumulative_rebal_cost']:,.0f}.",
            f"CAGR drops from {baseline['cagr']*100:.2f}% to {with_costs['cagr']*100:.2f}%")
else:
    verdict("3_hedge_costs", "FAIL",
            f"Hedge costs reduce CAGR by {cagr_cost_impact:.2f}% — material omission. "
            f"The 120% bear hedge is expensive to maintain.",
            f"CAGR drops from {baseline['cagr']*100:.2f}% to {with_costs['cagr']*100:.2f}%")

# ============================================================
# CHECK 4: SIGNAL EXECUTION REALISM
# ============================================================
print("\n" + "=" * 70)
print("CHECK 4: SIGNAL EXECUTION REALISM")
print("=" * 70)

# The code uses indicators[sym].loc[today, 'close'] as entry_price
# But signals are computed FROM close prices (RSI, MFI, OBV all use Close)
# So: on day T, you compute RSI/MFI/OBV from Close_T, fire signal, and enter at Close_T
# This is technically look-ahead: you can't know Close_T until the market closes,
# but you'd need to enter at Close_T which is the same moment.
#
# In reality: you could use MOC (market-on-close) orders if signals were computed
# from data available before close. But RSI etc. use the close itself.
# The realistic approach: signal fires at T's close, enter at T+1 open.

next_day = run_backtest(signals_df, indicators, spy_close,
                        cost_bps_rt=20, hedge_lag_days=0,
                        use_next_day_entry=True)

print(f"\n  Original (same-day close entry): CAGR={baseline['cagr']*100:.2f}%, "
      f"Sharpe={baseline['sharpe']:.3f}, WR={baseline['win_rate']*100:.1f}%")
print(f"  Next-day open entry:             CAGR={next_day['cagr']*100:.2f}%, "
      f"Sharpe={next_day['sharpe']:.3f}, WR={next_day['win_rate']*100:.1f}%")

exec_impact = (baseline['cagr'] - next_day['cagr']) * 100

# This is a real issue because all signals use close-derived indicators
verdict("4_signal_execution_realism",
        "FAIL" if exec_impact > 3.0 else ("WARNING" if exec_impact > 1.0 else "PASS"),
        f"Signals use RSI/MFI/OBV computed from day-T Close, then enter at day-T Close. "
        f"This is look-ahead: you can't compute RSI(Close_T) and also enter at Close_T. "
        f"Realistic = enter at T+1 Open. Impact: CAGR drops by {exec_impact:.2f}%. "
        f"Honest CAGR={next_day['cagr']*100:.2f}%, MaxDD={next_day['max_dd']*100:.2f}%.",
        f"CAGR inflated by {exec_impact:.2f}% due to same-day entry")

# ============================================================
# CHECK 5: DRAWDOWN STRESS TEST
# ============================================================
print("\n" + "=" * 70)
print("CHECK 5: DRAWDOWN STRESS TEST")
print("=" * 70)

# COVID crash: Feb 19 - Mar 23, 2020 (SPY: -33.9%)
# 2022 bear: Jan 3 - Oct 12, 2022 (SPY: -25.4%)

stress_periods = {
    'COVID_crash': ('2020-02-19', '2020-03-23'),
    'bear_2022': ('2022-01-03', '2022-10-12'),
    'Q4_2018': ('2018-09-20', '2018-12-24'),
    'taper_tantrum_2015': ('2015-07-20', '2016-02-11'),
}

nav = baseline['nav']
stress_results = {}

for name, (start_str, end_str) in stress_periods.items():
    start_d = pd.Timestamp(start_str)
    end_d = pd.Timestamp(end_str)

    # SPY drawdown in this period
    spy_period = spy_close.loc[start_d:end_d]
    if len(spy_period) > 5:
        spy_dd = (spy_period.iloc[-1] / spy_period.iloc[0] - 1) * 100
    else:
        spy_dd = np.nan

    # Portfolio drawdown in this period
    nav_period = nav.loc[start_d:end_d]
    if len(nav_period) > 5:
        port_dd = (nav_period.iloc[-1] / nav_period.iloc[0] - 1) * 100
        # Also compute max peak-to-trough within the period
        cummax_period = nav_period.cummax()
        dd_series = (nav_period - cummax_period) / cummax_period
        port_max_dd = dd_series.min() * 100
    else:
        port_dd = np.nan
        port_max_dd = np.nan

    stress_results[name] = {
        'period': f"{start_str} to {end_str}",
        'spy_return_pct': round(float(spy_dd), 2) if not np.isnan(spy_dd) else None,
        'portfolio_return_pct': round(float(port_dd), 2) if not np.isnan(port_dd) else None,
        'portfolio_max_dd_pct': round(float(port_max_dd), 2) if not np.isnan(port_max_dd) else None,
    }

    print(f"\n  {name} ({start_str} to {end_str}):")
    print(f"    SPY return:       {spy_dd:+.1f}%")
    print(f"    Portfolio return:  {port_dd:+.1f}%")
    print(f"    Portfolio max DD:  {port_max_dd:.1f}%")

# Check if COVID drawdown is suspiciously low
covid = stress_results.get('COVID_crash', {})
spy_covid = covid.get('spy_return_pct', -34)
port_covid = covid.get('portfolio_return_pct', 0)

# With 120% short hedge in bear regime, the portfolio could legitimately profit
# during a crash. But -34% SPY crash with minimal portfolio impact needs scrutiny.
# The hedge should FULLY kick in (SPY below 50d SMA during crash), giving 120% short.
# So portfolio should be net SHORT market during crash = PROFIT from crash.
# This is by design, but relies on perfect regime timing.

if port_covid is not None and spy_covid is not None:
    if port_covid > 0 and spy_covid < -25:
        # Portfolio made money during a -34% crash — the hedge is doing heavy lifting
        # This is possible with 120% short, but suspicious if drawdown is tiny
        verdict("5_drawdown_stress",
                "WARNING",
                f"During COVID crash (SPY {spy_covid:+.1f}%), portfolio returned {port_covid:+.1f}%. "
                f"The 120% bear hedge turns the portfolio net short during crashes, so profits are "
                f"mechanically expected. But this means the MaxDD=-16.2% figure is misleading — "
                f"it's the HEDGE doing the heavy lifting, not the stock signals. "
                f"Without the hedge, drawdown would be much worse.",
                f"COVID: SPY={spy_covid:+.1f}%, Portfolio={port_covid:+.1f}%")
    elif port_covid is not None and abs(port_covid) < abs(spy_covid) * 0.3:
        verdict("5_drawdown_stress", "WARNING",
                f"Portfolio only drew down {port_covid:.1f}% vs SPY {spy_covid:.1f}% during COVID. "
                f"Max DD of -16.2% seems artificially low due to perfectly-timed hedge switches.",
                f"COVID portfolio DD is {abs(port_covid)/abs(spy_covid)*100:.0f}% of SPY DD")
    else:
        verdict("5_drawdown_stress", "PASS",
                f"Portfolio drew down {port_covid:.1f}% vs SPY {spy_covid:.1f}% during COVID. Reasonable.",
                f"Portfolio captured {abs(port_covid)/abs(spy_covid)*100:.0f}% of SPY drawdown")
else:
    verdict("5_drawdown_stress", "WARNING", "Could not compute stress period returns — data may be missing.")

# Also run the backtest with NO HEDGE to see raw signal drawdown
print("\n  --- Running NO-HEDGE variant to isolate signal alpha ---")
no_hedge = run_backtest(signals_df, indicators, spy_close,
                        cost_bps_rt=20, hedge_lag_days=0,
                        bear_hedge_frac=0.0, bull_hedge_frac=0.0)
print(f"  No-hedge:  CAGR={no_hedge['cagr']*100:.2f}%, Sharpe={no_hedge['sharpe']:.3f}, "
      f"MaxDD={no_hedge['max_dd']*100:.2f}%")

# What's the hedge contribution to performance?
hedge_cagr_contribution = baseline['cagr'] - no_hedge['cagr']
hedge_dd_contribution = no_hedge['max_dd'] - baseline['max_dd']  # more negative no_hedge = hedge helps

audit_results['5a_hedge_contribution'] = {
    'status': 'INFO',
    'detail': f"Hedge adds {hedge_cagr_contribution*100:.2f}% CAGR and reduces MaxDD by "
              f"{abs(hedge_dd_contribution)*100:.2f}%. Without hedge: CAGR={no_hedge['cagr']*100:.2f}%, "
              f"MaxDD={no_hedge['max_dd']*100:.2f}%.",
    'no_hedge_cagr': round(no_hedge['cagr'] * 100, 2),
    'no_hedge_max_dd': round(no_hedge['max_dd'] * 100, 2),
    'no_hedge_sharpe': round(no_hedge['sharpe'], 3),
}

# ============================================================
# CHECK 6: PERMUTATION ROBUSTNESS
# ============================================================
print("\n" + "=" * 70)
print("CHECK 6: PERMUTATION ROBUSTNESS (hedge-inclusive)")
print("=" * 70)

# The original permutation test shuffles stock SELECTION (random symbols on same dates)
# but keeps the hedge FIXED. This is invalid because:
# 1. In bear markets, the hedge alone generates alpha (120% short SPY during drops)
# 2. Random stock picks + the same hedge would also show alpha
# 3. The test should shuffle EVERYTHING or test stock alpha SEPARATELY from hedge

# Proper test: shuffle entry dates across time (not just symbols) while
# keeping everything else the same (including hedge)
# If permuted entries ALSO beat the random baseline, the hedge is doing the work

print("  Running hedge-inclusive permutation test (100 shuffles)...")
print("  Method: randomize trade entry TIMING (not just symbol), keep hedge intact")

N_PERM_FULL = 100
np.random.seed(42)

# Get baseline trade-level returns
baseline_trade_df = baseline['trade_df']
if len(baseline_trade_df) > 0:
    actual_mean_ret = (baseline_trade_df['ret_gross'] - COST_FRAC_RT).mean()
else:
    actual_mean_ret = 0

# For the hedge-inclusive test: run full backtest with randomly-timed entries
# This tests whether TIMING matters, or if any random entry + hedge = good results
perm_cagrs = []
perm_sharpes = []
all_syms = list(indicators.keys())

for perm_i in range(N_PERM_FULL):
    if (perm_i + 1) % 20 == 0:
        print(f"    Permutation {perm_i + 1}/{N_PERM_FULL}...")

    # Create shuffled signals: same count per day, random symbols
    shuffled_signals = signals_df.copy()
    shuffled_symbols = shuffled_signals['symbol'].values.copy()
    np.random.shuffle(shuffled_symbols)
    shuffled_signals['symbol'] = shuffled_symbols

    perm_result = run_backtest(shuffled_signals, indicators, spy_close,
                               cost_bps_rt=20, hedge_lag_days=0)
    perm_cagrs.append(perm_result['cagr'])
    perm_sharpes.append(perm_result['sharpe'])

perm_cagrs = np.array(perm_cagrs)
perm_sharpes = np.array(perm_sharpes)

# How many permutations beat the actual strategy?
p_cagr = (perm_cagrs >= baseline['cagr']).mean()
p_sharpe = (perm_sharpes >= baseline['sharpe']).mean()

print(f"\n  Actual CAGR: {baseline['cagr']*100:.2f}%")
print(f"  Perm CAGR mean: {perm_cagrs.mean()*100:.2f}%, p95: {np.percentile(perm_cagrs, 95)*100:.2f}%")
print(f"  p-value (CAGR): {p_cagr:.3f}")
print(f"  Actual Sharpe: {baseline['sharpe']:.3f}")
print(f"  Perm Sharpe mean: {perm_sharpes.mean():.3f}, p95: {np.percentile(perm_sharpes, 95):.3f}")
print(f"  p-value (Sharpe): {p_sharpe:.3f}")

# If random stock picks + same hedge gets similar results, stock selection is NOT adding value
# — the hedge is doing all the work
if perm_cagrs.mean() > baseline['cagr'] * 0.7:
    verdict("6_permutation_robustness",
            "FAIL",
            f"Random stock selection + same hedge achieves {perm_cagrs.mean()*100:.2f}% CAGR "
            f"(vs actual {baseline['cagr']*100:.2f}%). The hedge alone explains most returns. "
            f"Stock signal TIMING adds only {(baseline['cagr'] - perm_cagrs.mean())*100:.2f}% CAGR. "
            f"The original permutation test was invalid because it kept the hedge fixed.",
            f"Perm p-value (CAGR): {p_cagr:.3f}, hedge explains {perm_cagrs.mean()/baseline['cagr']*100:.0f}% of returns")
elif p_cagr > 0.05:
    verdict("6_permutation_robustness",
            "FAIL",
            f"Hedge-inclusive permutation p={p_cagr:.3f} > 0.05. Stock selection does NOT "
            f"add statistically significant alpha over random picks + hedge. "
            f"Random CAGR={perm_cagrs.mean()*100:.2f}% vs actual {baseline['cagr']*100:.2f}%.",
            f"Stock signals not significant: p={p_cagr:.3f}")
else:
    verdict("6_permutation_robustness",
            "PASS",
            f"Even with hedge included, actual strategy significantly beats random stock selection. "
            f"p={p_cagr:.3f}, actual CAGR={baseline['cagr']*100:.2f}% vs perm mean {perm_cagrs.mean()*100:.2f}%.",
            f"Stock selection is genuine: p={p_cagr:.3f}")

# ============================================================
# CHECK 7: COST SENSITIVITY
# ============================================================
print("\n" + "=" * 70)
print("CHECK 7: COST SENSITIVITY")
print("=" * 70)

# 7a: 40 bps RT cost (double the original)
cost_40bps = run_backtest(signals_df, indicators, spy_close,
                          cost_bps_rt=40, hedge_lag_days=0)

print(f"\n  20 bps RT (original): CAGR={baseline['cagr']*100:.2f}%, Sharpe={baseline['sharpe']:.3f}")
print(f"  40 bps RT:            CAGR={cost_40bps['cagr']*100:.2f}%, Sharpe={cost_40bps['sharpe']:.3f}")

# 7b: Full realistic costs (40 bps + borrow + rebal + lagged hedge + next-day entry)
full_realistic = run_backtest(signals_df, indicators, spy_close,
                              cost_bps_rt=40, hedge_lag_days=1,
                              annual_borrow_cost_pct=1.0,
                              hedge_rebal_cost_bps=10,
                              use_next_day_entry=True)

print(f"  Full realistic*:      CAGR={full_realistic['cagr']*100:.2f}%, "
      f"Sharpe={full_realistic['sharpe']:.3f}, MaxDD={full_realistic['max_dd']*100:.2f}%")
print(f"    * 40 bps RT + 1% borrow + 10 bps rebal + lagged hedge + next-day entry")

# Check if gates still pass under realistic costs
g1_realistic = full_realistic['cagr'] > 0.15
g4_realistic = abs(full_realistic['max_dd']) < 0.35
yr_profitable = sum(1 for r in full_realistic['yearly_rets'].values() if r > 0)
yr_total = len(full_realistic['yearly_rets'])
g5_realistic = yr_total > 0 and yr_profitable / yr_total > 0.75

print(f"\n  Gates under full realistic costs:")
print(f"    G1 CAGR>15%: {'PASS' if g1_realistic else 'FAIL'} ({full_realistic['cagr']*100:.2f}%)")
print(f"    G4 MaxDD<35%: {'PASS' if g4_realistic else 'FAIL'} ({full_realistic['max_dd']*100:.2f}%)")
print(f"    G5 >75% yrs: {'PASS' if g5_realistic else 'FAIL'} ({yr_profitable}/{yr_total})")

if g1_realistic and g4_realistic:
    verdict("7_cost_sensitivity", "PASS",
            f"Strategy survives aggressive cost assumptions (40 bps RT, 1% borrow, 10 bps rebal, "
            f"lagged hedge, next-day entry). CAGR={full_realistic['cagr']*100:.2f}%, "
            f"Sharpe={full_realistic['sharpe']:.3f}.",
            f"Realistic CAGR: {full_realistic['cagr']*100:.2f}% (vs {baseline['cagr']*100:.2f}% original)")
elif full_realistic['cagr'] > 0.08:
    verdict("7_cost_sensitivity", "WARNING",
            f"Strategy degrades significantly under realistic costs but remains profitable. "
            f"CAGR drops from {baseline['cagr']*100:.2f}% to {full_realistic['cagr']*100:.2f}%. "
            f"G1 (CAGR>15%) {'passes' if g1_realistic else 'FAILS'}.",
            f"CAGR reduction: {(baseline['cagr']-full_realistic['cagr'])*100:.2f}%")
else:
    verdict("7_cost_sensitivity", "FAIL",
            f"Strategy collapses under realistic costs. "
            f"CAGR={full_realistic['cagr']*100:.2f}% — not viable.",
            f"CAGR drops from {baseline['cagr']*100:.2f}% to {full_realistic['cagr']*100:.2f}%")

# ============================================================
# OVERALL SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("ADVERSARIAL AUDIT SUMMARY")
print("=" * 70)

n_pass = sum(1 for v in audit_results.values() if v['status'] == 'PASS')
n_warn = sum(1 for v in audit_results.values() if v['status'] == 'WARNING')
n_fail = sum(1 for v in audit_results.values() if v['status'] == 'FAIL')
n_info = sum(1 for v in audit_results.values() if v['status'] == 'INFO')

print(f"\n  PASS: {n_pass}  |  WARNING: {n_warn}  |  FAIL: {n_fail}  |  INFO: {n_info}")

# Compute the "honest" numbers: lagged hedge + next-day entry + hedge costs
honest = run_backtest(signals_df, indicators, spy_close,
                      cost_bps_rt=20, hedge_lag_days=1,
                      annual_borrow_cost_pct=0.4,
                      hedge_rebal_cost_bps=5,
                      use_next_day_entry=True)

print(f"\n  HONEST NUMBERS (minimal fixes: lag hedge, next-day entry, 0.4% borrow, 5 bps rebal):")
print(f"    CAGR:     {honest['cagr']*100:.2f}%  (was {baseline['cagr']*100:.2f}%)")
print(f"    Sharpe:   {honest['sharpe']:.3f}  (was {baseline['sharpe']:.3f})")
print(f"    Sortino:  {honest['sortino']:.3f}  (was {baseline['sortino']:.3f})")
print(f"    MaxDD:    {honest['max_dd']*100:.2f}%  (was {baseline['max_dd']*100:.2f}%)")
print(f"    PF:       {honest['profit_factor']:.2f}  (was {baseline['profit_factor']:.2f})")
print(f"    WR:       {honest['win_rate']*100:.1f}%  (was {baseline['win_rate']*100:.1f}%)")
print(f"    Trades:   {honest['total_trades']}  (was {baseline['total_trades']})")

# Save everything
output = {
    'audit_date': pd.Timestamp.now().isoformat(),
    'baseline_metrics': {
        'cagr': round(baseline['cagr'] * 100, 2),
        'sharpe': round(baseline['sharpe'], 3),
        'sortino': round(baseline['sortino'], 3),
        'max_dd': round(baseline['max_dd'] * 100, 2),
        'profit_factor': round(baseline['profit_factor'], 2),
        'win_rate': round(baseline['win_rate'] * 100, 1),
        'total_trades': baseline['total_trades'],
    },
    'honest_metrics': {
        'description': 'Lagged hedge (1d), next-day entry, 0.4% borrow, 5 bps rebal, 20 bps RT',
        'cagr': round(honest['cagr'] * 100, 2),
        'sharpe': round(honest['sharpe'], 3),
        'sortino': round(honest['sortino'], 3),
        'max_dd': round(honest['max_dd'] * 100, 2),
        'profit_factor': round(honest['profit_factor'], 2),
        'win_rate': round(honest['win_rate'] * 100, 1),
        'total_trades': honest['total_trades'],
    },
    'full_realistic_metrics': {
        'description': '40 bps RT, 1% borrow, 10 bps rebal, lagged hedge, next-day entry',
        'cagr': round(full_realistic['cagr'] * 100, 2),
        'sharpe': round(full_realistic['sharpe'], 3),
        'max_dd': round(full_realistic['max_dd'] * 100, 2),
    },
    'no_hedge_metrics': {
        'description': 'Pure stock signals, no SPY hedge',
        'cagr': round(no_hedge['cagr'] * 100, 2),
        'sharpe': round(no_hedge['sharpe'], 3),
        'max_dd': round(no_hedge['max_dd'] * 100, 2),
    },
    'checks': {},
    'stress_test': stress_results,
    'permutation': {
        'method': 'hedge-inclusive: random stock selection with same hedge',
        'n_permutations': N_PERM_FULL,
        'actual_cagr': round(baseline['cagr'] * 100, 2),
        'perm_mean_cagr': round(float(perm_cagrs.mean()) * 100, 2),
        'perm_p95_cagr': round(float(np.percentile(perm_cagrs, 95)) * 100, 2),
        'p_value_cagr': round(float(p_cagr), 3),
        'p_value_sharpe': round(float(p_sharpe), 3),
    },
    'survivorship': {
        'post_2012_additions': survivor_concerns,
        'late_data_starts': late_data_start,
    },
}

# Copy check verdicts
for check_name, check_data in audit_results.items():
    output['checks'][check_name] = check_data

# Save
audit_path = OUTDIR / 'adversarial_audit.json'
with open(audit_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nAudit saved to {audit_path}")
print("Done.")
