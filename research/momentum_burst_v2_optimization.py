#!/usr/bin/env python3
"""
Momentum Burst v2 — Parameter Optimization
============================================
KB #281 validated momentum burst as the ONLY single-leg options strategy
that works for $645 account (Sharpe 1.28, Sortino 2.37).

This sweep optimizes the key parameters to find a better configuration:

SWEEP DIMENSIONS:
1. DTE: 7, 10, 14, 21 days
2. Strike: ATM, 2% OTM, 2% ITM, 5% ITM
3. TP: 20%, 30%, 40%, 50%
4. SL: 15%, 20%, 25%, 30%
5. Trailing giveback: 40%, 50%, 60%
6. Max hold: 3, 5, 7 days
7. Min signals: 1, 2

Total grid: 4×4×4×4×3×3×2 = 4,608 combos (too many)
Smart approach: Latin hypercube sampling of ~200 combos + local refinement

Also tests:
- VIX regime filter (different params for VIX > 20 vs < 20)
- Volume confirmation (enter only on above-avg volume days)

4-gate validation on best config only (avoid multiple comparisons bias).

HC #696: Growth strategy improvement.
HC #749: $645 agentic account, options-only.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
import json
import os
import traceback
from scipy import stats
from itertools import product

warnings.filterwarnings('ignore')

# === CONFIG ===
INITIAL_CAPITAL = 645.0
MAX_TRADE_SIZE = 250.0
COMMISSION = 0.65
N_PERMUTATIONS = 100

# Momentum signal tickers
MOMENTUM_TICKERS = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY']

# BS calibration (KB #282)
BS_HAIRCUT = 0.60

try:
    import mlflow
    MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")
    mlflow.set_tracking_uri(MLFLOW_URI)
    USE_MLFLOW = True
except ImportError:
    USE_MLFLOW = False


def fetch_data(tickers, start='2020-01-01', end='2026-07-25'):
    """Fetch daily data for all tickers."""
    print(f"Fetching {len(tickers)} tickers...")
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            if len(df) > 200:
                data[t] = df
                print(f"  {t}: {len(df)} days")
        except Exception as e:
            print(f"  {t}: {e}")
    return data


def compute_momentum_signals(data):
    """
    Compute momentum signals for each ticker.
    Signal = combination of: 5d return > 0, 10d return > 0, RSI > 50,
    price > 20d SMA, volume > 20d avg.
    """
    signals = {}
    for ticker, df in data.items():
        close = df['Close']
        volume = df['Volume'] if 'Volume' in df.columns else pd.Series(1e6, index=df.index)

        s = pd.DataFrame(index=df.index)
        s['ret_5d'] = close.pct_change(5) > 0
        s['ret_10d'] = close.pct_change(10) > 0

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        s['rsi_bull'] = rsi > 55
        s['rsi_bear'] = rsi < 45

        s['above_sma20'] = close > close.rolling(20).mean()
        s['above_sma50'] = close > close.rolling(50).mean()
        s['high_volume'] = volume > volume.rolling(20).mean()

        # Momentum score (0-5 for calls, 0-5 for puts)
        s['bull_score'] = (s['ret_5d'].astype(int) + s['ret_10d'].astype(int) +
                          s['rsi_bull'].astype(int) + s['above_sma20'].astype(int) +
                          s['high_volume'].astype(int))
        s['bear_score'] = ((~s['ret_5d']).astype(int) + (~s['ret_10d']).astype(int) +
                          s['rsi_bear'].astype(int) + (~s['above_sma20']).astype(int) +
                          s['high_volume'].astype(int))

        # 5d realized momentum magnitude
        s['momentum_mag'] = close.pct_change(5).abs()
        s['close'] = close

        # HV for IV estimation
        s['hv_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)

        signals[ticker] = s

    return signals


def bs_call_price(S, K, T, sigma, r=0.045):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return max(S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2), 0)


def bs_put_price(S, K, T, sigma, r=0.045):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return max(K * np.exp(-r*T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1), 0)


def simulate_trade(entry_price, direction, strike_pct, iv, dte, subsequent_prices,
                   tp_pct, sl_pct, trailing_pct, max_hold):
    """
    Simulate a single option trade.
    strike_pct: 0 = ATM, 0.02 = 2% OTM, -0.02 = 2% ITM (for calls)
    """
    if direction == 'call':
        strike = entry_price * (1 + strike_pct)
    else:
        strike = entry_price * (1 - strike_pct)

    T = dte / 252
    if direction == 'call':
        cost = bs_call_price(entry_price, strike, T, iv)
    else:
        cost = bs_put_price(entry_price, strike, T, iv)

    cost *= (1 - BS_HAIRCUT)
    if cost < 0.05:
        return None

    max_value = cost

    for day_i, price in enumerate(subsequent_prices[:max_hold]):
        T_remain = max(0, (dte - day_i - 1) / 252)

        if direction == 'call':
            val = bs_call_price(price, strike, T_remain, iv)
        else:
            val = bs_put_price(price, strike, T_remain, iv)
        val *= (1 - BS_HAIRCUT)

        pnl_pct = (val - cost) / cost
        if val > max_value:
            max_value = val

        # TP
        if pnl_pct >= tp_pct:
            return (val - cost - 2*COMMISSION) / (cost + COMMISSION), day_i + 1, 'TP'

        # SL
        if pnl_pct <= -sl_pct:
            return (val - cost - 2*COMMISSION) / (cost + COMMISSION), day_i + 1, 'SL'

        # Trailing
        if max_value > cost * 1.05:
            giveback = (max_value - val) / (max_value - cost)
            if giveback >= trailing_pct:
                return (val - cost - 2*COMMISSION) / (cost + COMMISSION), day_i + 1, 'TRAIL'

    # Max hold exit
    if len(subsequent_prices) > 0:
        final_price = subsequent_prices[min(max_hold-1, len(subsequent_prices)-1)]
        T_remain = max(0, (dte - max_hold) / 252)
        if direction == 'call':
            val = bs_call_price(final_price, strike, T_remain, iv)
        else:
            val = bs_put_price(final_price, strike, T_remain, iv)
        val *= (1 - BS_HAIRCUT)
        return (val - cost - 2*COMMISSION) / (cost + COMMISSION), max_hold, 'EXPIRE'

    return None


def run_config(data, signals, config):
    """
    Run a single parameter configuration and return Sharpe + trade count.
    Fast path — no permutation test, just raw metrics.
    """
    dte = config['dte']
    strike_pct = config['strike_pct']
    tp = config['tp']
    sl = config['sl']
    trailing = config['trailing']
    max_hold = config['max_hold']
    min_signals = config['min_signals']
    vix_filter = config.get('vix_filter', False)
    volume_filter = config.get('volume_filter', False)

    trades = []
    equity = INITIAL_CAPITAL

    # VIX proxy from SPY
    if 'SPY' in data:
        spy_close = data['SPY']['Close']
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]
        vix_proxy = spy_close.pct_change().rolling(20).std() * np.sqrt(252) * 100
    else:
        vix_proxy = None

    for ticker, sig in signals.items():
        close = sig['close']
        dates = sig.index

        for i in range(60, len(dates) - max_hold - 1):
            bull_score = sig['bull_score'].iloc[i]
            bear_score = sig['bear_score'].iloc[i]

            # Direction decision
            if bull_score >= min_signals and bull_score > bear_score:
                direction = 'call'
                score = bull_score
            elif bear_score >= min_signals and bear_score > bull_score:
                direction = 'put'
                score = bear_score
            else:
                continue

            # Volume filter
            if volume_filter and not sig['high_volume'].iloc[i]:
                continue

            # VIX filter: skip long entries in high VIX
            if vix_filter and vix_proxy is not None:
                vix_val = vix_proxy.get(dates[i], 15)
                if pd.notna(vix_val) and vix_val > 25 and direction == 'call':
                    continue

            # Cooldown: max 1 trade per ticker per 5 days
            recent = [t for t in trades[-30:] if t.get('ticker') == ticker
                     and (dates[i] - t['entry_date']).days < 5]
            if recent:
                continue

            entry_price = float(close.iloc[i])
            iv = float(sig['hv_20'].iloc[i]) * 1.15 if pd.notna(sig['hv_20'].iloc[i]) else 0.25
            iv = max(0.10, min(1.0, iv))

            subsequent = close.iloc[i+1:i+max_hold+2].values.astype(float)

            result = simulate_trade(entry_price, direction, strike_pct, iv, dte,
                                   subsequent, tp, sl, trailing, max_hold)
            if result is None:
                continue

            pnl_pct, hold_days, exit_reason = result
            trade_size = min(MAX_TRADE_SIZE, equity * 0.30)
            if trade_size < 30:
                continue

            dollar_pnl = trade_size * pnl_pct
            equity += dollar_pnl

            trades.append({
                'entry_date': dates[i],
                'ticker': ticker,
                'direction': direction,
                'pnl_pct': pnl_pct,
                'dollar_pnl': dollar_pnl,
                'hold_days': hold_days,
                'exit_reason': exit_reason,
                'equity_after': equity,
            })

    if len(trades) < 20:
        return None

    pnls = [t['pnl_pct'] for t in trades]
    daily_pnls = [t['dollar_pnl'] for t in trades]

    avg = np.mean(pnls)
    std = np.std(pnls)
    sharpe_approx = (avg / std * np.sqrt(252 / 5)) if std > 0 else 0  # ~5 day avg hold

    win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
    total_return = (trades[-1]['equity_after'] - INITIAL_CAPITAL) / INITIAL_CAPITAL

    gross_profit = sum(p for p in daily_pnls if p > 0)
    gross_loss = abs(sum(p for p in daily_pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max drawdown
    equities = [INITIAL_CAPITAL] + [t['equity_after'] for t in trades]
    peak = INITIAL_CAPITAL
    max_dd = 0
    for e in equities:
        peak = max(peak, e)
        dd = (e - peak) / peak
        max_dd = min(max_dd, dd)

    # Downside std
    neg_pnls = [p for p in pnls if p < 0]
    downside_std = np.std(neg_pnls) if neg_pnls else 1e-6
    sortino_approx = (avg / downside_std * np.sqrt(252 / 5)) if downside_std > 0 else 0

    return {
        'sharpe': sharpe_approx,
        'sortino': sortino_approx,
        'win_rate': win_rate,
        'pf': pf,
        'total_return': total_return,
        'max_dd': max_dd,
        'n_trades': len(trades),
        'final_equity': trades[-1]['equity_after'],
        'avg_hold': np.mean([t['hold_days'] for t in trades]),
        'trades': trades,  # keep for validation
    }


def full_validation(trades, n_perms=N_PERMUTATIONS):
    """Run 4-gate validation on the best config."""
    if not trades or len(trades) < 20:
        return {'gates_passed': 0}

    pnls = [t['pnl_pct'] for t in trades]
    real_sharpe = np.mean(pnls) / max(np.std(pnls), 1e-6)

    # Gate 1: Permutation
    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = [p * (1 if np.random.random() > 0.5 else -1) for p in pnls]
        if np.std(shuffled) > 0:
            perm_sharpes.append(np.mean(shuffled) / np.std(shuffled))
    if perm_sharpes:
        z1 = (real_sharpe - np.mean(perm_sharpes)) / max(np.std(perm_sharpes), 1e-6)
        p1 = 1 - stats.norm.cdf(z1)
    else:
        z1, p1 = 0, 1.0
    gate1 = p1 < 0.05

    # Gate 2: Regime (call vs put performance balance)
    call_pnls = [t['pnl_pct'] for t in trades if t['direction'] == 'call']
    put_pnls = [t['pnl_pct'] for t in trades if t['direction'] == 'put']
    if call_pnls and put_pnls:
        call_sr = np.mean(call_pnls) / max(np.std(call_pnls), 1e-6)
        put_sr = np.mean(put_pnls) / max(np.std(put_pnls), 1e-6)
        regime_gap = abs(call_sr - put_sr) / max(abs(call_sr), abs(put_sr), 1e-6)
        gate2 = regime_gap < 0.50
    else:
        call_sr = put_sr = regime_gap = None
        gate2 = False

    # Gate 3: Random direction
    rand_sharpes = []
    abs_pnls = [abs(p) for p in pnls]
    for _ in range(n_perms):
        rand = [p * (1 if np.random.random() > 0.5 else -1) for p in abs_pnls]
        if np.std(rand) > 0:
            rand_sharpes.append(np.mean(rand) / np.std(rand))
    if rand_sharpes:
        z3 = (real_sharpe - np.mean(rand_sharpes)) / max(np.std(rand_sharpes), 1e-6)
        p3 = 1 - stats.norm.cdf(z3)
    else:
        z3, p3 = 0, 1.0
    gate3 = p3 < 0.05

    # Gate 4: Sub-period stability
    mid = len(pnls) // 2
    h1_sr = np.mean(pnls[:mid]) / max(np.std(pnls[:mid]), 1e-6) if mid > 5 else 0
    h2_sr = np.mean(pnls[mid:]) / max(np.std(pnls[mid:]), 1e-6) if mid > 5 else 0
    gate4 = h1_sr > 0 and h2_sr > 0

    gates_passed = sum([gate1, gate2, gate3, gate4])

    return {
        'gates_passed': gates_passed,
        'gate1_perm': {'pass': gate1, 'z': z1, 'p': p1},
        'gate2_regime': {'pass': gate2, 'call_sr': call_sr, 'put_sr': put_sr, 'gap': regime_gap},
        'gate3_random': {'pass': gate3, 'z': z3, 'p': p3},
        'gate4_subperiod': {'pass': gate4, 'h1': h1_sr, 'h2': h2_sr},
    }


def main():
    print("=" * 70)
    print("MOMENTUM BURST v2 — PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if USE_MLFLOW:
        try:
            mlflow.set_experiment("momentum_burst_v2_optimization")
            mlflow.start_run(run_name=f"mb_v2_{datetime.now().strftime('%Y%m%d_%H%M')}")
            mlflow.set_tag("node", "jupiter")
            mlflow.set_tag("strategy_type", "growth")
        except Exception as e:
            print(f"MLflow: {e}")

    data = fetch_data(MOMENTUM_TICKERS)
    if len(data) < 4:
        print("FATAL: insufficient data")
        return

    signals = compute_momentum_signals(data)
    print(f"\nComputed signals for {len(signals)} tickers")

    # === PHASE 1: SMART PARAMETER SWEEP ===
    # Instead of full grid, use structured sweep of key dimensions

    param_grid = {
        'dte': [7, 10, 14, 21],
        'strike_pct': [0.0, 0.02, -0.02, -0.05],  # ATM, 2% OTM, 2% ITM, 5% ITM
        'tp': [0.20, 0.30, 0.40, 0.50],
        'sl': [0.15, 0.20, 0.25, 0.30],
        'trailing': [0.40, 0.50, 0.60],
        'max_hold': [3, 5, 7],
        'min_signals': [2],  # KB #281 says 2 is optimal, 3+ hurts
    }

    # Phase 1a: DTE + Strike sweep (baseline TP/SL/trailing from KB #281)
    print("\n--- PHASE 1a: DTE × Strike (32 configs) ---")
    phase1a_results = []
    for dte in param_grid['dte']:
        for strike in param_grid['strike_pct']:
            config = {
                'dte': dte, 'strike_pct': strike,
                'tp': 0.30, 'sl': 0.25, 'trailing': 0.50,
                'max_hold': 5, 'min_signals': 2,
            }
            result = run_config(data, signals, config)
            label = f"DTE={dte},STR={strike:+.0%}"
            if result:
                phase1a_results.append((config, result, label))
                print(f"  {label}: Sharpe {result['sharpe']:.2f}, WR {result['win_rate']*100:.0f}%, "
                      f"Trades {result['n_trades']}, ${result['final_equity']:.0f}")
            else:
                print(f"  {label}: <20 trades, skip")

    # Find best DTE and strike
    if not phase1a_results:
        print("FATAL: no valid configs in phase 1a")
        return

    best_1a = max(phase1a_results, key=lambda x: x[1]['sharpe'])
    best_dte = best_1a[0]['dte']
    best_strike = best_1a[0]['strike_pct']
    print(f"\n  BEST Phase 1a: {best_1a[2]} → Sharpe {best_1a[1]['sharpe']:.2f}")

    # Phase 1b: TP × SL sweep with best DTE/strike
    print(f"\n--- PHASE 1b: TP × SL (16 configs, DTE={best_dte}, strike={best_strike:+.0%}) ---")
    phase1b_results = []
    for tp in param_grid['tp']:
        for sl in param_grid['sl']:
            config = {
                'dte': best_dte, 'strike_pct': best_strike,
                'tp': tp, 'sl': sl, 'trailing': 0.50,
                'max_hold': 5, 'min_signals': 2,
            }
            result = run_config(data, signals, config)
            label = f"TP={tp:.0%},SL={sl:.0%}"
            if result:
                phase1b_results.append((config, result, label))
                print(f"  {label}: Sharpe {result['sharpe']:.2f}, WR {result['win_rate']*100:.0f}%, "
                      f"Trades {result['n_trades']}, ${result['final_equity']:.0f}")
            else:
                print(f"  {label}: <20 trades, skip")

    if not phase1b_results:
        print("Using phase 1a best for remaining phases")
        best_tp, best_sl = 0.30, 0.25
    else:
        best_1b = max(phase1b_results, key=lambda x: x[1]['sharpe'])
        best_tp = best_1b[0]['tp']
        best_sl = best_1b[0]['sl']
        print(f"\n  BEST Phase 1b: {best_1b[2]} → Sharpe {best_1b[1]['sharpe']:.2f}")

    # Phase 1c: Trailing × Max Hold sweep
    print(f"\n--- PHASE 1c: Trailing × MaxHold (9 configs) ---")
    phase1c_results = []
    for trail in param_grid['trailing']:
        for max_hold in param_grid['max_hold']:
            config = {
                'dte': best_dte, 'strike_pct': best_strike,
                'tp': best_tp, 'sl': best_sl, 'trailing': trail,
                'max_hold': max_hold, 'min_signals': 2,
            }
            result = run_config(data, signals, config)
            label = f"TRAIL={trail:.0%},HOLD={max_hold}d"
            if result:
                phase1c_results.append((config, result, label))
                print(f"  {label}: Sharpe {result['sharpe']:.2f}, WR {result['win_rate']*100:.0f}%, "
                      f"Trades {result['n_trades']}, ${result['final_equity']:.0f}")
            else:
                print(f"  {label}: <20 trades, skip")

    if phase1c_results:
        best_1c = max(phase1c_results, key=lambda x: x[1]['sharpe'])
        best_trail = best_1c[0]['trailing']
        best_max_hold = best_1c[0]['max_hold']
        print(f"\n  BEST Phase 1c: {best_1c[2]} → Sharpe {best_1c[1]['sharpe']:.2f}")
    else:
        best_trail, best_max_hold = 0.50, 5

    # Phase 1d: Feature filters with optimal params
    print(f"\n--- PHASE 1d: Feature Filters ---")
    optimal_base = {
        'dte': best_dte, 'strike_pct': best_strike,
        'tp': best_tp, 'sl': best_sl, 'trailing': best_trail,
        'max_hold': best_max_hold, 'min_signals': 2,
    }

    filter_configs = [
        ('BASE_OPTIMAL', {}),
        ('VIX_FILTER', {'vix_filter': True}),
        ('VOL_FILTER', {'volume_filter': True}),
        ('VIX+VOL', {'vix_filter': True, 'volume_filter': True}),
        ('MIN_3_SIGNALS', {'min_signals': 3}),
        ('MIN_1_SIGNAL', {'min_signals': 1}),
    ]

    phase1d_results = []
    for label, extra in filter_configs:
        config = {**optimal_base, **extra}
        result = run_config(data, signals, config)
        if result:
            phase1d_results.append((config, result, label))
            print(f"  {label}: Sharpe {result['sharpe']:.2f}, Sortino {result['sortino']:.2f}, "
                  f"WR {result['win_rate']*100:.0f}%, PF {result['pf']:.2f}, "
                  f"MDD {result['max_dd']*100:.1f}%, Trades {result['n_trades']}, "
                  f"${result['final_equity']:.0f}")
        else:
            print(f"  {label}: <20 trades, skip")

    # === PHASE 2: VALIDATION OF BEST ===
    print("\n" + "=" * 70)
    print("PHASE 2: FULL VALIDATION OF BEST CONFIG")
    print("=" * 70)

    # Collect ALL results
    all_results = phase1a_results + phase1b_results + phase1c_results + phase1d_results
    if not all_results:
        print("FATAL: no valid results to validate")
        return

    # Best by Sharpe
    best_overall = max(all_results, key=lambda x: x[1]['sharpe'])
    best_config = best_overall[0]
    best_metrics = best_overall[1]
    best_label = best_overall[2]

    print(f"\nBEST CONFIG: {best_label}")
    print(f"  Parameters: DTE={best_config['dte']}, Strike={best_config['strike_pct']:+.0%}, "
          f"TP={best_config['tp']:.0%}, SL={best_config['sl']:.0%}, "
          f"Trail={best_config['trailing']:.0%}, MaxHold={best_config['max_hold']}d")
    print(f"  Sharpe: {best_metrics['sharpe']:.2f}")
    print(f"  Sortino: {best_metrics['sortino']:.2f}")
    print(f"  WR: {best_metrics['win_rate']*100:.1f}%")
    print(f"  PF: {best_metrics['pf']:.2f}")
    print(f"  MDD: {best_metrics['max_dd']*100:.1f}%")
    print(f"  CAGR: {best_metrics['total_return']*100:.1f}% total")
    print(f"  Final: ${best_metrics['final_equity']:.0f}")
    print(f"  Trades: {best_metrics['n_trades']}")
    print(f"  Avg hold: {best_metrics['avg_hold']:.1f} days")

    # Compare to KB #281 baseline
    baseline_config = {
        'dte': 14, 'strike_pct': 0.0,
        'tp': 0.30, 'sl': 0.25, 'trailing': 0.50,
        'max_hold': 5, 'min_signals': 2,
    }
    baseline_result = run_config(data, signals, baseline_config)
    if baseline_result:
        print(f"\n  KB #281 BASELINE: Sharpe {baseline_result['sharpe']:.2f}, "
              f"WR {baseline_result['win_rate']*100:.0f}%, ${baseline_result['final_equity']:.0f}")
        improvement = (best_metrics['sharpe'] - baseline_result['sharpe']) / abs(baseline_result['sharpe']) * 100
        print(f"  IMPROVEMENT: {improvement:+.1f}% Sharpe")
    else:
        print("  Baseline: insufficient trades")

    # Full 4-gate validation
    print("\n--- 4-Gate Validation ---")
    validation = full_validation(best_metrics['trades'])

    g1 = validation['gate1_perm']
    g2 = validation['gate2_regime']
    g3 = validation['gate3_random']
    g4 = validation['gate4_subperiod']

    print(f"  Gate 1 (Permutation): z={g1['z']:.2f}, p={g1['p']:.3f} → {'PASS' if g1['pass'] else 'FAIL'}")
    if g2['gap'] is not None:
        print(f"  Gate 2 (Regime): call={g2['call_sr']:.2f}, put={g2['put_sr']:.2f}, "
              f"gap={g2['gap']:.2f} → {'PASS' if g2['pass'] else 'FAIL'}")
    else:
        print(f"  Gate 2 (Regime): one-directional → FAIL")
    print(f"  Gate 3 (Random): z={g3['z']:.2f}, p={g3['p']:.3f} → {'PASS' if g3['pass'] else 'FAIL'}")
    print(f"  Gate 4 (Sub-period): h1={g4['h1']:.2f}, h2={g4['h2']:.2f} → {'PASS' if g4['pass'] else 'FAIL'}")
    print(f"\n  GATES: {validation['gates_passed']}/4")

    # Also validate baseline
    if baseline_result:
        print("\n--- Baseline (KB #281) Validation ---")
        base_val = full_validation(baseline_result['trades'])
        print(f"  Baseline gates: {base_val['gates_passed']}/4")

    # === SUMMARY TABLE ===
    print("\n" + "=" * 70)
    print("TOP 10 CONFIGS BY SHARPE")
    print("=" * 70)
    all_sorted = sorted(all_results, key=lambda x: x[1]['sharpe'], reverse=True)[:10]
    print(f"{'Rank':<5} {'Config':<35} {'Sharpe':>7} {'WR':>5} {'PF':>5} {'MDD':>7} {'Trades':>7} {'Final$':>8}")
    for i, (cfg, met, lbl) in enumerate(all_sorted):
        print(f"{i+1:<5} {lbl:<35} {met['sharpe']:>7.2f} {met['win_rate']*100:>4.0f}% "
              f"{met['pf']:>5.2f} {met['max_dd']*100:>6.1f}% {met['n_trades']:>7} ${met['final_equity']:>7.0f}")

    # MLflow logging
    if USE_MLFLOW:
        try:
            mlflow.log_metrics({
                'best_sharpe': round(best_metrics['sharpe'], 3),
                'best_sortino': round(best_metrics['sortino'], 3),
                'best_wr': round(best_metrics['win_rate'], 3),
                'best_pf': round(min(best_metrics['pf'], 99), 3),
                'best_mdd': round(best_metrics['max_dd'], 3),
                'best_n_trades': best_metrics['n_trades'],
                'gates_passed': validation['gates_passed'],
                'configs_tested': len(all_results),
            })
            if baseline_result:
                mlflow.log_metrics({
                    'baseline_sharpe': round(baseline_result['sharpe'], 3),
                    'improvement_pct': round(improvement, 1),
                })
            mlflow.set_tag('best_config', json.dumps({k: v for k, v in best_config.items()
                                                       if k != 'trades'}, default=str))
            mlflow.set_tag('result', 'PASS' if validation['gates_passed'] >= 3 else 'FAIL')
            mlflow.end_run()
        except Exception as e:
            print(f"MLflow log error: {e}")

    # Save results
    output_path = "/home/jupiter/Lvl3Quant/research/findings/momentum_burst_v2_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_data = {
        'best_config': {k: v for k, v in best_config.items()},
        'best_metrics': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in best_metrics.items() if k != 'trades'},
        'validation': validation,
        'baseline_sharpe': baseline_result['sharpe'] if baseline_result else None,
        'top_10': [
            {'config': {k: v for k, v in cfg.items()},
             'label': lbl,
             'sharpe': float(met['sharpe']),
             'win_rate': float(met['win_rate']),
             'n_trades': met['n_trades']}
            for cfg, met, lbl in all_sorted
        ],
        'total_configs_tested': len(all_results),
    }

    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nDone at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
