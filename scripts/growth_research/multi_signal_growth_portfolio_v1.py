#!/usr/bin/env python3
"""
Multi-Signal Growth Portfolio v1
=================================
Combines 5 individually-validated alpha signals into a unified growth portfolio.
Each signal passed adversarial testing (permutation p<0.05, regime gap <0.50).

SIGNALS (best variants from SESSION_STATE entries 791-799):
  1. Skewness Premium (SK252_P10_D3_H10): Sharpe 1.023
  2. Post-Earnings Drift Contrarian (W20_30_D12_MFI30_H21): Sharpe 2.110
  3. Smart Money Accumulation (ACC10_DN2_V12_MFI_H10): Sharpe 1.605
  4. Price-Volume Divergence (LB10_Dn0_both_RSI40_H10): Sharpe 1.234
  5. Volatility Crush Reversal (ATR20_R21_D5_MFI_H21): Sharpe 0.962

OUTPUT: /home/jupiter/Lvl3Quant/output/multi_signal_growth_portfolio_v1/
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from scipy import stats as scipy_stats
warnings.filterwarnings('ignore')

import functools
print = functools.partial(print, flush=True)

sys.path.insert(0, '/home/jupiter/Lvl3Quant')

# ─── Config ───
START_DATE = '2013-01-01'
END_DATE = '2026-07-01'
BACKTEST_START = '2014-01-01'
INITIAL_CAPITAL = 100_000
SPREAD_COST_PCT = 0.0005  # 5bps per side
N_PERMS = 50
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/multi_signal_growth_portfolio_v1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLD_DAYS = {
    'skewness_premium': 10,
    'post_earnings_contrarian': 21,
    'smart_money_accumulation': 10,
    'price_volume_divergence': 10,
    'vol_crush_reversal': 21,
}

print(f"{'='*70}")
print(f"MULTI-SIGNAL GROWTH PORTFOLIO v1")
print(f"{'='*70}")

# ═══════════════════════════════════════════════════════════════════════
# DATA ACQUISITION
# ═══════════════════════════════════════════════════════════════════════

def get_sp500_tickers():
    import urllib.request
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=15).read().decode()
        tables = pd.read_html(html)
        df = tables[0]
        tickers = [t.replace('.', '-') for t in df['Symbol'].tolist()]
        print(f"  Loaded {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"  Wikipedia failed: {e}, using fallback")
    return [
        'AAPL','MSFT','GOOGL','META','NVDA','AVGO','ADBE','CRM','CSCO','ACN',
        'ORCL','TXN','QCOM','AMD','INTC','INTU','IBM','NOW','AMAT','MU',
        'UNH','JNJ','LLY','PFE','ABBV','MRK','TMO','ABT','DHR','BMY',
        'AMGN','MDT','ISRG','SYK','GILD','VRTX','REGN','BSX','BRK-B','JPM',
        'V','MA','BAC','WFC','GS','MS','SPGI','BLK','AXP','C',
        'AMZN','TSLA','HD','MCD','NKE','LOW','SBUX','TJX','BKNG','CMG',
        'PG','KO','PEP','COST','WMT','PM','MO','MDLZ','CL',
        'CAT','UNP','UPS','HON','RTX','BA','DE','LMT','GE','GD',
        'XOM','CVX','COP','EOG','SLB','MPC','PSX','VLO','OXY',
        'LIN','APD','SHW','ECL','FCX','NUE','NEM','DOW','DD',
        'NEE','SO','DUK','D','SRE','AEP','EXC','XEL',
        'PLD','AMT','CCI','EQIX','PSA','SPG','O','WELL',
        'DIS','CMCSA','NFLX','T','VZ','CHTR','TMUS','EA',
        'PYPL','SHOP','ZS','CRWD','DDOG','NET','ABNB','UBER',
    ]


def download_ohlcv_data(tickers, start, end):
    import yfinance as yf
    import pickle

    cache_file = OUTPUT_DIR / '_ohlcv_cache.pkl'
    if cache_file.exists():
        print("\n  Loading cached OHLCV data...")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"  Cache: {len(data['close'].columns)} tickers, {len(data['close'])} days")
        return data

    print(f"\n  Downloading OHLCV for {len(tickers)} tickers...")
    batch_size = 50
    all_close, all_high, all_low, all_volume = {}, {}, {}, {}

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            raw = yf.download(' '.join(batch), start=start, end=end,
                            auto_adjust=True, progress=False, threads=True)
            if isinstance(raw.columns, pd.MultiIndex):
                for field, store in [('Close', all_close), ('High', all_high),
                                     ('Low', all_low), ('Volume', all_volume)]:
                    if field in raw.columns.get_level_values(0):
                        df_f = raw[field]
                        for col in df_f.columns:
                            if df_f[col].notna().sum() > 252:
                                store[col] = df_f[col]
            elif len(batch) == 1:
                t = batch[0]
                for field, store in [('Close', all_close), ('High', all_high),
                                     ('Low', all_low), ('Volume', all_volume)]:
                    if field in raw.columns and raw[field].notna().sum() > 252:
                        store[t] = raw[field]
        except Exception:
            pass
        n_batches = (len(tickers) - 1) // batch_size + 1
        print(f"    Batch {i//batch_size + 1}/{n_batches}: {len(all_close)} tickers")

    data = {
        'close': pd.DataFrame(all_close),
        'high': pd.DataFrame(all_high),
        'low': pd.DataFrame(all_low),
        'volume': pd.DataFrame(all_volume),
    }
    print(f"  Downloaded {data['close'].shape[1]} tickers, {data['close'].shape[0]} days")

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    return data


# ═══════════════════════════════════════════════════════════════════════
# VECTORIZED SIGNAL GENERATORS (no row-by-row loops)
# ═══════════════════════════════════════════════════════════════════════

def compute_obv(close, volume):
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()


def compute_mfi(high, low, close, volume, period=14):
    tp = (high + low + close) / 3
    mf = tp * volume
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0).rolling(period).sum()
    neg_mf = mf.where(delta < 0, 0).rolling(period).sum()
    return 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def signal_skewness_premium(close):
    """SK252_P10_D3_H10: Bottom 10% 252d skewness + 3% drop in 5d."""
    returns = close.pct_change()
    skew_252 = returns.rolling(252, min_periods=200).skew()
    ret_5d = close.pct_change(5)

    # Cross-sectional 10th percentile threshold per day
    threshold = skew_252.quantile(0.10, axis=1)
    # Boolean: skewness <= threshold for that day
    neg_skew = skew_252.le(threshold, axis=0)
    # Dropped 3%+
    dropped = ret_5d <= -0.03

    signals = (neg_skew & dropped).astype(int)
    # Zero out warmup period
    signals.loc[signals.index < pd.Timestamp(BACKTEST_START)] = 0
    return signals


def signal_post_earnings_contrarian(close, volume, mfi):
    """W20_30_D12_MFI30_H21: -12% day on 2x volume, enter 20-30d later if MFI<30."""
    daily_ret = close.pct_change()
    vol_avg = volume.rolling(20).mean()

    # Event detection: drop >= 12% on volume >= 2x average
    event_mask = (daily_ret <= -0.12) & (volume >= 2.0 * vol_avg)

    # Create signal: for each event, fire signal on days 20-30 after
    signals = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=np.int8)

    # Vectorized: shift event_mask by 20-30 days and OR them
    for lag in range(20, 31):
        shifted = event_mask.shift(lag).fillna(False)
        signals = signals | shifted.astype(int)

    # Apply MFI < 30 filter
    mfi_ok = mfi < 30
    signals = (signals & mfi_ok.reindex(columns=signals.columns, fill_value=False)).astype(int)
    signals.loc[signals.index < pd.Timestamp(BACKTEST_START)] = 0
    return signals


def signal_smart_money_accumulation(close, volume, mfi):
    """ACC10_DN2_V12_MFI_H10: OBV rising 10d, price down 2%+, vol 1.2x, MFI<20."""
    obv = compute_obv(close, volume)
    vol_avg = volume.rolling(20).mean()

    obv_rising = obv.diff(10) > 0
    price_down = close.pct_change(10) <= -0.02
    vol_elevated = volume >= 1.2 * vol_avg
    mfi_oversold = mfi < 20

    signals = (obv_rising & price_down & vol_elevated & mfi_oversold).astype(int)
    signals.loc[signals.index < pd.Timestamp(BACKTEST_START)] = 0
    return signals


def signal_price_volume_divergence(close, volume, mfi):
    """LB10_Dn0_both_RSI40_H10: OBV + MFI both rising, price flat/down, RSI<40."""
    obv = compute_obv(close, volume)
    rsi = compute_rsi(close, 14)

    obv_rising = obv.diff(10) > 0
    mfi_rising = mfi.diff(10) > 0
    price_flat_down = close.pct_change(10) <= 0
    rsi_low = rsi < 40

    signals = (obv_rising & mfi_rising & price_flat_down & rsi_low).astype(int)
    signals.loc[signals.index < pd.Timestamp(BACKTEST_START)] = 0
    return signals


def signal_vol_crush_reversal(close, high, low, mfi):
    """ATR20_R21_D5_MFI_H21: ATR<20th pctile within 21d, dropped 5%+, MFI<20."""
    # Compute ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=0)
    # Actually need element-wise max across 3 components
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1.stack(), tr2.stack(), tr3.stack()], axis=1).max(axis=1).unstack()
    atr = tr.rolling(14).mean()

    # Rolling 252d rank of ATR (percentile)
    # Use rank-based approach: for each day, ATR rank within last 252 days
    atr_rank = atr.rolling(252, min_periods=100).apply(
        lambda x: scipy_stats.percentileofscore(x[:-1], x[-1]) / 100 if len(x) > 1 else 0.5,
        raw=True
    )
    print(f"      ATR percentile computed")

    # Was ATR < 20th percentile at any point in last 21 days?
    atr_low_recently = (atr_rank < 0.20).rolling(21, min_periods=1).max().astype(bool)

    ret_5d = close.pct_change(5)
    dropped = ret_5d <= -0.05
    mfi_oversold = mfi < 20

    signals = (atr_low_recently & dropped & mfi_oversold).astype(int)
    signals.loc[signals.index < pd.Timestamp(BACKTEST_START)] = 0
    return signals


# ═══════════════════════════════════════════════════════════════════════
# PORTFOLIO SIMULATION
# ═══════════════════════════════════════════════════════════════════════

def simulate_portfolio(all_signals, close, mode='equal_weight', max_pos_per_signal=10):
    """Simulate portfolio from combined signals."""
    signal_names = list(all_signals.keys())
    n_signals = len(signal_names)
    dates = close.index.tolist()
    bt_start = pd.Timestamp(BACKTEST_START)

    positions = []
    equity_curve = []
    daily_returns = []
    trade_log = []
    capital = INITIAL_CAPITAL

    for i in range(len(dates)):
        date = dates[i]
        prev_date = dates[i-1] if i > 0 else date

        if date < bt_start:
            equity_curve.append(capital)
            daily_returns.append(0.0)
            continue

        # --- Close expired positions & mark-to-market ---
        new_positions = []
        daily_pnl = 0.0
        for pos in positions:
            trading_days_held = sum(1 for d in dates if pos['entry_date'] < d <= date)

            if trading_days_held >= pos['hold_days']:
                # Exit
                exit_price = close.loc[date, pos['ticker']] if pos['ticker'] in close.columns else np.nan
                if pd.notna(exit_price) and pos['entry_price'] > 0:
                    gross_ret = (exit_price / pos['entry_price']) - 1
                    net_ret = gross_ret - SPREAD_COST_PCT
                    pnl = pos['notional'] * net_ret
                    daily_pnl += pnl
                    trade_log.append({
                        'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                        'exit_date': date.strftime('%Y-%m-%d'),
                        'ticker': pos['ticker'],
                        'signal': pos['signal'],
                        'entry_price': pos['entry_price'],
                        'exit_price': float(exit_price),
                        'gross_return': float(gross_ret),
                        'net_return': float(net_ret),
                        'pnl': float(pnl),
                        'hold_days': trading_days_held,
                    })
            else:
                # Mark-to-market
                cur_price = close.loc[date, pos['ticker']] if pos['ticker'] in close.columns else np.nan
                prev_price = close.loc[prev_date, pos['ticker']] if pos['ticker'] in close.columns else np.nan
                if pd.notna(cur_price) and pd.notna(prev_price) and prev_price > 0:
                    day_ret = (cur_price / prev_price) - 1
                    daily_pnl += pos['notional'] * day_ret
                    pos['notional'] *= (1 + day_ret)
                new_positions.append(pos)

        positions = new_positions
        capital += daily_pnl

        # --- Open new positions ---
        held_tickers = {p['ticker'] for p in positions}

        if mode == 'equal_weight':
            alloc_per_signal = capital / n_signals
            for sig_name in signal_names:
                sig = all_signals[sig_name]
                if date not in sig.index:
                    continue
                fired = sig.loc[date]
                fired_tickers = [t for t in fired[fired > 0].index if t not in held_tickers]
                if not fired_tickers:
                    continue
                existing = sum(1 for p in positions if p['signal'] == sig_name)
                slots = max(0, max_pos_per_signal - existing)
                if slots == 0:
                    continue
                new_tickers = fired_tickers[:slots]
                pos_size = alloc_per_signal / max(len(new_tickers), 1)
                pos_size = min(pos_size, capital * 0.05)
                for ticker in new_tickers:
                    ep = close.loc[date, ticker] if ticker in close.columns else np.nan
                    if pd.isna(ep) or ep <= 0:
                        continue
                    positions.append({
                        'ticker': ticker, 'entry_date': date, 'entry_price': float(ep),
                        'hold_days': HOLD_DAYS[sig_name], 'signal': sig_name,
                        'notional': pos_size * (1 - SPREAD_COST_PCT),
                    })

        elif mode == 'confluence':
            ticker_counts = {}
            ticker_signals = {}
            for sig_name in signal_names:
                sig = all_signals[sig_name]
                if date not in sig.index:
                    continue
                fired = sig.loc[date]
                for t in fired[fired > 0].index:
                    if t not in held_tickers:
                        ticker_counts[t] = ticker_counts.get(t, 0) + 1
                        ticker_signals.setdefault(t, []).append(sig_name)

            if ticker_counts:
                total_weight = sum(ticker_counts.values())
                alloc = capital * 0.8
                for ticker, count in sorted(ticker_counts.items(), key=lambda x: -x[1])[:50]:
                    pos_size = alloc * (count / total_weight)
                    pos_size = min(pos_size, capital * 0.10)
                    ep = close.loc[date, ticker] if ticker in close.columns else np.nan
                    if pd.isna(ep) or ep <= 0:
                        continue
                    sigs = ticker_signals.get(ticker, [signal_names[0]])
                    avg_hold = int(np.mean([HOLD_DAYS[s] for s in sigs]))
                    positions.append({
                        'ticker': ticker, 'entry_date': date, 'entry_price': float(ep),
                        'hold_days': avg_hold, 'signal': '+'.join(sigs),
                        'notional': pos_size * (1 - SPREAD_COST_PCT),
                    })

        daily_ret = daily_pnl / max(equity_curve[-1] if equity_curve else capital, 1)
        equity_curve.append(capital)
        daily_returns.append(daily_ret)

    return {
        'equity_curve': equity_curve,
        'daily_returns': daily_returns,
        'trade_log': trade_log,
        'dates': dates,
    }


def compute_metrics(result):
    dates = result['dates']
    rets = np.array(result['daily_returns'])
    eq = np.array(result['equity_curve'])

    bt_mask = np.array([d >= pd.Timestamp(BACKTEST_START) for d in dates])
    rets = rets[bt_mask]
    eq = eq[bt_mask]
    trades = result['trade_log']

    if len(rets) == 0 or np.std(rets) == 0:
        return {'sharpe': 0, 'sortino': 0, 'cagr': 0, 'max_dd': -1,
                'win_rate': 0, 'profit_factor': 0, 'n_trades': 0,
                'total_return': 0, 'avg_trade_return': 0}

    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252)
    downside = rets[rets < 0]
    sortino = np.mean(rets) / np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) > 0 else 0

    n_years = len(rets) / 252
    total_ret = eq[-1] / INITIAL_CAPITAL
    cagr = (total_ret ** (1 / max(n_years, 0.01))) - 1 if total_ret > 0 else -1

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()

    if trades:
        trade_rets = [t['net_return'] for t in trades]
        wins = [r for r in trade_rets if r > 0]
        losses = [r for r in trade_rets if r <= 0]
        win_rate = len(wins) / len(trade_rets)
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        profit_factor = gross_profit / gross_loss
    else:
        win_rate = 0
        profit_factor = 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 2),
        'max_dd': round(max_dd * 100, 2),
        'win_rate': round(win_rate * 100, 1),
        'profit_factor': round(profit_factor, 2),
        'n_trades': len(trades),
        'total_return': round((total_ret - 1) * 100, 2),
        'avg_trade_return': round(np.mean([t['net_return'] for t in trades]) * 100, 3) if trades else 0,
    }


def regime_test(result, spy_close):
    dates = result['dates']
    rets = np.array(result['daily_returns'])
    bt_mask = np.array([d >= pd.Timestamp(BACKTEST_START) for d in dates])
    dates_bt = [d for d, m in zip(dates, bt_mask) if m]
    rets_bt = rets[bt_mask]

    spy_rets = spy_close.pct_change()
    green_rets, red_rets = [], []

    for i, date in enumerate(dates_bt):
        if date in spy_rets.index:
            spy_r = spy_rets.loc[date]
            if isinstance(spy_r, pd.Series):
                spy_r = spy_r.iloc[0]
            if pd.notna(spy_r):
                if spy_r >= 0:
                    green_rets.append(rets_bt[i])
                else:
                    red_rets.append(rets_bt[i])

    green_rets = np.array(green_rets)
    red_rets = np.array(red_rets)

    if len(green_rets) < 50 or len(red_rets) < 50:
        return {'pass': False, 'reason': 'Insufficient regime days'}

    sharpe_g = np.mean(green_rets) / np.std(green_rets) * np.sqrt(252) if np.std(green_rets) > 0 else 0
    sharpe_r = np.mean(red_rets) / np.std(red_rets) * np.sqrt(252) if np.std(red_rets) > 0 else 0
    denom = max(abs(sharpe_g), abs(sharpe_r))
    gap = abs(sharpe_g - sharpe_r) / denom if denom > 0 else 0

    return {
        'sharpe_green': round(sharpe_g, 3),
        'sharpe_red': round(sharpe_r, 3),
        'regime_gap': round(gap, 3),
        'pass': gap <= 0.50,
        'n_green_days': len(green_rets),
        'n_red_days': len(red_rets),
    }


def yearly_breakdown(result):
    dates = result['dates']
    rets = np.array(result['daily_returns'])
    bt_mask = np.array([d >= pd.Timestamp(BACKTEST_START) for d in dates])
    dates_bt = [d for d, m in zip(dates, bt_mask) if m]
    rets_bt = rets[bt_mask]

    yearly = {}
    for d, r in zip(dates_bt, rets_bt):
        yr = d.year
        yearly.setdefault(yr, []).append(r)

    breakdown = {}
    for yr, yr_rets in sorted(yearly.items()):
        yr_rets = np.array(yr_rets)
        cum_ret = np.prod(1 + yr_rets) - 1
        sharpe = np.mean(yr_rets) / np.std(yr_rets) * np.sqrt(252) if np.std(yr_rets) > 0 else 0
        breakdown[yr] = {
            'return_pct': round(cum_ret * 100, 2),
            'sharpe': round(sharpe, 2),
            'n_days': len(yr_rets),
        }
    return breakdown


def permutation_test(all_signals, close, mode, n_perms=50):
    print(f"\n  Running {n_perms} permutations ({mode})...")
    real_result = simulate_portfolio(all_signals, close, mode=mode)
    real_sharpe = compute_metrics(real_result)['sharpe']

    perm_sharpes = []
    for p in range(n_perms):
        if (p + 1) % 10 == 0:
            print(f"    Perm {p+1}/{n_perms}...")
        shuffled = {}
        for sig_name, sig_df in all_signals.items():
            offset = np.random.randint(20, len(sig_df) - 20)
            shifted = sig_df.copy()
            shifted.values[:] = np.roll(sig_df.values, offset, axis=0)
            shuffled[sig_name] = shifted
        perm_result = simulate_portfolio(shuffled, close, mode=mode)
        perm_sharpes.append(compute_metrics(perm_result)['sharpe'])

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= real_sharpe)

    return {
        'real_sharpe': real_sharpe,
        'perm_mean': round(np.mean(perm_sharpes), 3),
        'perm_p95': round(np.percentile(perm_sharpes, 95), 3),
        'p_value': round(p_value, 3),
        'pass': p_value < 0.05,
    }


def signal_contribution(trade_log):
    by_signal = {}
    for trade in trade_log:
        sig = trade['signal']
        by_signal.setdefault(sig, []).append(trade)

    results = {}
    for sig, trades in by_signal.items():
        rets = [t['net_return'] for t in trades]
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r <= 0]
        results[sig] = {
            'n_trades': len(trades),
            'avg_return_pct': round(np.mean(rets) * 100, 3),
            'win_rate': round(len(wins) / len(rets) * 100, 1) if rets else 0,
            'profit_factor': round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) != 0 else 999,
            'total_pnl': round(sum(t['pnl'] for t in trades), 2),
        }
    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

t0 = time.time()

print("\nPHASE 0: DATA")
print("="*70)

tickers = get_sp500_tickers()
tickers_with_spy = list(set(tickers + ['SPY']))
data = download_ohlcv_data(tickers_with_spy, START_DATE, END_DATE)

close = data['close']
high = data['high']
low = data['low']
volume = data['volume']

spy_close = close[['SPY']].copy() if 'SPY' in close.columns else None

# Align to common tickers
common = sorted(set(close.columns) & set(high.columns) & set(low.columns) & set(volume.columns) - {'SPY'})
print(f"  Universe: {len(common)} tickers with full OHLCV")

close_s = close[common]
high_s = high[common]
low_s = low[common]
volume_s = volume[common]

# Pre-compute MFI
print("  Computing MFI...")
t_ind = time.time()
mfi = compute_mfi(high_s, low_s, close_s, volume_s, period=14)
print(f"  MFI done in {time.time()-t_ind:.1f}s")

# --- Signals ---
print("\nPHASE 1: SIGNALS (vectorized)")
print("="*70)

all_signals = {}

print("  [1/5] Skewness Premium...")
t1 = time.time()
all_signals['skewness_premium'] = signal_skewness_premium(close_s)
n = (all_signals['skewness_premium'] > 0).sum().sum()
print(f"    {n} fires in {time.time()-t1:.1f}s")

print("  [2/5] Post-Earnings Drift Contrarian...")
t1 = time.time()
all_signals['post_earnings_contrarian'] = signal_post_earnings_contrarian(close_s, volume_s, mfi)
n = (all_signals['post_earnings_contrarian'] > 0).sum().sum()
print(f"    {n} fires in {time.time()-t1:.1f}s")

print("  [3/5] Smart Money Accumulation...")
t1 = time.time()
all_signals['smart_money_accumulation'] = signal_smart_money_accumulation(close_s, volume_s, mfi)
n = (all_signals['smart_money_accumulation'] > 0).sum().sum()
print(f"    {n} fires in {time.time()-t1:.1f}s")

print("  [4/5] Price-Volume Divergence...")
t1 = time.time()
all_signals['price_volume_divergence'] = signal_price_volume_divergence(close_s, volume_s, mfi)
n = (all_signals['price_volume_divergence'] > 0).sum().sum()
print(f"    {n} fires in {time.time()-t1:.1f}s")

print("  [5/5] Volatility Crush Reversal...")
t1 = time.time()
all_signals['vol_crush_reversal'] = signal_vol_crush_reversal(close_s, high_s, low_s, mfi)
n = (all_signals['vol_crush_reversal'] > 0).sum().sum()
print(f"    {n} fires in {time.time()-t1:.1f}s")

print("\n  Signal Summary:")
for sig_name, sig_df in all_signals.items():
    n = (sig_df > 0).sum().sum()
    print(f"    {sig_name}: {n} fires")

# --- Simulation ---
print("\nPHASE 2: SIMULATION")
print("="*70)

print("  [A] Equal-Weight Portfolio...")
t1 = time.time()
result_ew = simulate_portfolio(all_signals, close_s, mode='equal_weight')
metrics_ew = compute_metrics(result_ew)
print(f"    Done in {time.time()-t1:.1f}s")
print(f"    Sharpe={metrics_ew['sharpe']}, CAGR={metrics_ew['cagr']}%, MaxDD={metrics_ew['max_dd']}%")
print(f"    WR={metrics_ew['win_rate']}%, PF={metrics_ew['profit_factor']}, Trades={metrics_ew['n_trades']}")

print("  [B] Confluence Portfolio...")
t1 = time.time()
result_conf = simulate_portfolio(all_signals, close_s, mode='confluence')
metrics_conf = compute_metrics(result_conf)
print(f"    Done in {time.time()-t1:.1f}s")
print(f"    Sharpe={metrics_conf['sharpe']}, CAGR={metrics_conf['cagr']}%, MaxDD={metrics_conf['max_dd']}%")
print(f"    WR={metrics_conf['win_rate']}%, PF={metrics_conf['profit_factor']}, Trades={metrics_conf['n_trades']}")

# --- Regime Test ---
print("\nPHASE 3: R1 REGIME TEST")
print("="*70)

if spy_close is not None:
    regime_ew = regime_test(result_ew, spy_close)
    regime_conf = regime_test(result_conf, spy_close)
    print(f"  EW:  Sharpe_green={regime_ew['sharpe_green']}, Sharpe_red={regime_ew['sharpe_red']}, "
          f"gap={regime_ew['regime_gap']} {'PASS' if regime_ew['pass'] else 'FAIL'}")
    print(f"  Conf: Sharpe_green={regime_conf['sharpe_green']}, Sharpe_red={regime_conf['sharpe_red']}, "
          f"gap={regime_conf['regime_gap']} {'PASS' if regime_conf['pass'] else 'FAIL'}")
else:
    regime_ew = regime_conf = {'pass': False, 'reason': 'No SPY'}

# --- Yearly ---
print("\nPHASE 4: YEARLY BREAKDOWN")
print("="*70)

yearly_ew = yearly_breakdown(result_ew)
yearly_conf = yearly_breakdown(result_conf)

print(f"  {'Year':<6} {'EW Ret%':>8} {'EW Shp':>7} | {'Conf Ret%':>9} {'Conf Shp':>8}")
for yr in sorted(set(list(yearly_ew.keys()) + list(yearly_conf.keys()))):
    ew_s = yearly_ew.get(yr, {'return_pct': 0, 'sharpe': 0})
    cf_s = yearly_conf.get(yr, {'return_pct': 0, 'sharpe': 0})
    print(f"  {yr:<6} {ew_s['return_pct']:>7.1f}% {ew_s['sharpe']:>7.2f} | "
          f"{cf_s['return_pct']:>8.1f}% {cf_s['sharpe']:>8.2f}")

# --- Signal Contribution ---
print("\nPHASE 5: SIGNAL CONTRIBUTIONS (EW)")
print("="*70)

contrib_ew = signal_contribution(result_ew['trade_log'])
print(f"  {'Signal':<30} {'Trades':>6} {'AvgRet%':>8} {'WR%':>6} {'PF':>6} {'PnL$':>10}")
for sig, s in contrib_ew.items():
    print(f"  {sig:<30} {s['n_trades']:>6} {s['avg_return_pct']:>8.3f} "
          f"{s['win_rate']:>5.1f}% {s['profit_factor']:>6.2f} ${s['total_pnl']:>9.0f}")

# --- Permutation ---
print("\nPHASE 6: PERMUTATION TEST")
print("="*70)

perm_ew = permutation_test(all_signals, close_s, 'equal_weight', n_perms=N_PERMS)
print(f"  EW: real={perm_ew['real_sharpe']}, p95={perm_ew['perm_p95']}, "
      f"p={perm_ew['p_value']} {'PASS' if perm_ew['pass'] else 'FAIL'}")

# --- Save ---
print("\nPHASE 7: SAVE")
print("="*70)

results = {
    'timestamp': datetime.now().isoformat(),
    'config': {
        'start_date': BACKTEST_START, 'end_date': END_DATE,
        'initial_capital': INITIAL_CAPITAL, 'spread_cost_pct': SPREAD_COST_PCT,
        'n_tickers': len(common), 'signals': list(all_signals.keys()),
        'hold_days': HOLD_DAYS,
    },
    'equal_weight': {
        'metrics': metrics_ew, 'regime_test': regime_ew,
        'yearly': {str(k): v for k, v in yearly_ew.items()},
        'signal_contributions': contrib_ew, 'permutation': perm_ew,
    },
    'confluence': {
        'metrics': metrics_conf, 'regime_test': regime_conf,
        'yearly': {str(k): v for k, v in yearly_conf.items()},
    },
    'runtime_seconds': round(time.time() - t0, 1),
}

results_file = OUTPUT_DIR / 'results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Results saved")

# Equity curves
eq_df = pd.DataFrame({
    'date': [d.strftime('%Y-%m-%d') for d in result_ew['dates']],
    'equal_weight': result_ew['equity_curve'],
    'confluence': result_conf['equity_curve'],
})
eq_df.to_csv(OUTPUT_DIR / 'equity_curves.csv', index=False)

# Trade logs
if result_ew['trade_log']:
    pd.DataFrame(result_ew['trade_log']).to_csv(OUTPUT_DIR / 'trades_equal_weight.csv', index=False)
if result_conf['trade_log']:
    pd.DataFrame(result_conf['trade_log']).to_csv(OUTPUT_DIR / 'trades_confluence.csv', index=False)

# MLflow
try:
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("multi_signal_growth_portfolio")
    with mlflow.start_run(run_name="v1_5signals"):
        mlflow.log_params({'n_signals': len(all_signals), 'n_tickers': len(common),
                          'start_date': BACKTEST_START, 'spread_cost_pct': SPREAD_COST_PCT})
        for prefix, m in [('ew_', metrics_ew), ('conf_', metrics_conf)]:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f'{prefix}{k}', v)
        mlflow.log_metric('ew_regime_gap', regime_ew.get('regime_gap', -1))
        mlflow.log_metric('conf_regime_gap', regime_conf.get('regime_gap', -1))
        mlflow.log_artifact(str(results_file))
    print("  MLflow logged")
except Exception as e:
    print(f"  MLflow skipped: {e}")

# --- Final ---
elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"FINAL SUMMARY ({elapsed/60:.1f} min)")
print(f"{'='*70}")
print(f"  Universe: {len(common)} S&P 500 stocks, {BACKTEST_START} to {END_DATE}")
print(f"\n  {'Metric':<25} {'Equal-Weight':>15} {'Confluence':>15}")
print(f"  {'-'*55}")
for m in ['sharpe', 'sortino', 'cagr', 'max_dd', 'win_rate', 'profit_factor', 'n_trades']:
    ew_v, cf_v = metrics_ew[m], metrics_conf[m]
    if m in ['cagr', 'max_dd', 'win_rate']:
        print(f"  {m:<25} {ew_v:>14.1f}% {cf_v:>14.1f}%")
    else:
        print(f"  {m:<25} {ew_v:>15} {cf_v:>15}")

print(f"\n  R1 Regime: EW gap={regime_ew.get('regime_gap','?')} {'PASS' if regime_ew.get('pass') else 'FAIL'} | "
      f"Conf gap={regime_conf.get('regime_gap','?')} {'PASS' if regime_conf.get('pass') else 'FAIL'}")
print(f"  Permutation: p={perm_ew['p_value']} {'PASS' if perm_ew['pass'] else 'FAIL'}")

best = 'equal_weight' if metrics_ew['sharpe'] >= metrics_conf['sharpe'] else 'confluence'
bm = metrics_ew if best == 'equal_weight' else metrics_conf
br = regime_ew if best == 'equal_weight' else regime_conf
print(f"\n  BEST: {best}")
print(f"  25% CAGR target: {'PASS' if bm['cagr'] > 25 else 'FAIL'} ({bm['cagr']}%)")
print(f"  10% min bar: {'PASS' if bm['cagr'] > 10 else 'FAIL'}")
print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
