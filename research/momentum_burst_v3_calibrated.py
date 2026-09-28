#!/usr/bin/env python3
"""
Momentum Burst v3 — Calibrated ATM Pricing
=============================================
The v2 sweep failed (ALL negative Sharpe) because it applied a 60% BS haircut
designed for 4% OTM sector ETFs to ATM options.

ATM options have significant intrinsic value → BS is much more accurate.
Expected BS accuracy by moneyness:
  Deep ITM: 95-100%
  ATM: 80-90%  ← THIS is what KB #281 uses
  2% OTM: 50-75%
  5% OTM: 30-50% (KB #282 calibration territory)

This v3 uses moneyness-dependent calibration:
  ATM: 15% haircut (BS captures ~85% of market)
  2% OTM: 35% haircut
  5% OTM: 60% haircut

Tests the KB #281 optimal config (DTE=14, ATM, 30% TP, 25% SL, trailing 50%,
5-day max hold) with corrected pricing, then sweeps best DTE/TP/SL.

4-gate validation on best config.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
import json
import os
from scipy import stats

warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 645.0
MAX_TRADE_SIZE = 250.0
COMMISSION = 0.65
N_PERMUTATIONS = 200  # more robust

TICKERS = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY']

# Moneyness-dependent BS calibration
HAIRCUTS = {
    'deep_itm': 0.05,  # BS captures 95%
    'atm': 0.15,       # BS captures 85%
    'otm_2': 0.35,     # BS captures 65%
    'otm_5': 0.60,     # BS captures 40%
}

try:
    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000"))
    USE_MLFLOW = True
except ImportError:
    USE_MLFLOW = False


def fetch_data(tickers):
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start='2020-01-01', end='2026-07-25', progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            if len(df) > 200:
                data[t] = df
                print(f"  {t}: {len(df)} days")
        except:
            pass
    return data


def get_haircut(strike_pct):
    """Get moneyness-appropriate BS haircut."""
    abs_pct = abs(strike_pct)
    if abs_pct <= 0.005:
        return HAIRCUTS['atm']
    elif abs_pct <= 0.025:
        return HAIRCUTS['otm_2']
    elif abs_pct <= 0.055:
        return HAIRCUTS['otm_5']
    else:
        return HAIRCUTS['otm_5']


def bs_call(S, K, T, sigma, r=0.045):
    if T <= 0 or sigma <= 0: return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return max(S * stats.norm.cdf(d1) - K * np.exp(-r*T) * stats.norm.cdf(d2), 0)


def bs_put(S, K, T, sigma, r=0.045):
    if T <= 0 or sigma <= 0: return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return max(K * np.exp(-r*T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1), 0)


def compute_signals(data):
    signals = {}
    for ticker, df in data.items():
        close = df['Close']
        vol = df['Volume'] if 'Volume' in df.columns else pd.Series(1e6, index=df.index)

        s = pd.DataFrame(index=df.index)
        s['ret_5d'] = close.pct_change(5) > 0
        s['ret_10d'] = close.pct_change(10) > 0

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        s['rsi_bull'] = rsi > 55
        s['rsi_bear'] = rsi < 45
        s['above_sma20'] = close > close.rolling(20).mean()
        s['high_volume'] = vol > vol.rolling(20).mean()

        s['bull_score'] = (s['ret_5d'].astype(int) + s['ret_10d'].astype(int) +
                          s['rsi_bull'].astype(int) + s['above_sma20'].astype(int) +
                          s['high_volume'].astype(int))
        s['bear_score'] = ((~s['ret_5d']).astype(int) + (~s['ret_10d']).astype(int) +
                          s['rsi_bear'].astype(int) + (~s['above_sma20']).astype(int) +
                          s['high_volume'].astype(int))
        s['close'] = close
        s['hv_20'] = close.pct_change().rolling(20).std() * np.sqrt(252)
        signals[ticker] = s
    return signals


def simulate_trade(S, direction, strike_pct, iv, dte, prices_after, tp, sl, trailing, max_hold):
    """Simulate with moneyness-calibrated pricing."""
    haircut = get_haircut(strike_pct)
    cal = 1 - haircut  # calibration factor

    if direction == 'call':
        K = round(S * (1 + strike_pct), 2)
        cost = bs_call(S, K, dte/252, iv) * cal
    else:
        K = round(S * (1 - strike_pct), 2)
        cost = bs_put(S, K, dte/252, iv) * cal

    if cost < 0.10:
        return None

    max_val = cost
    for i, price in enumerate(prices_after[:max_hold]):
        T = max(0, (dte - i - 1) / 252)
        if direction == 'call':
            val = bs_call(price, K, T, iv) * cal
        else:
            val = bs_put(price, K, T, iv) * cal

        pnl_pct = (val - cost) / cost
        if val > max_val:
            max_val = val

        if pnl_pct >= tp:
            return ((val - cost - 2*COMMISSION) / (cost + COMMISSION), i+1, 'TP')
        if pnl_pct <= -sl:
            return ((val - cost - 2*COMMISSION) / (cost + COMMISSION), i+1, 'SL')
        if max_val > cost * 1.05:
            giveback = (max_val - val) / (max_val - cost)
            if giveback >= trailing:
                return ((val - cost - 2*COMMISSION) / (cost + COMMISSION), i+1, 'TRAIL')

    # Max hold exit
    if len(prices_after) >= max_hold:
        T = max(0, (dte - max_hold) / 252)
        price = prices_after[min(max_hold-1, len(prices_after)-1)]
        if direction == 'call':
            val = bs_call(price, K, T, iv) * cal
        else:
            val = bs_put(price, K, T, iv) * cal
        return ((val - cost - 2*COMMISSION) / (cost + COMMISSION), max_hold, 'EXPIRE')
    return None


def run_config(data, signals, cfg):
    """Run single config, return metrics."""
    trades = []
    equity = INITIAL_CAPITAL

    for ticker, sig in signals.items():
        close = sig['close']
        dates = sig.index
        for i in range(60, len(dates) - cfg['max_hold'] - 2):
            bs = sig['bull_score'].iloc[i]
            brs = sig['bear_score'].iloc[i]

            if bs >= cfg['min_sig'] and bs > brs:
                direction = 'call'
            elif brs >= cfg['min_sig'] and brs > bs:
                direction = 'put'
            else:
                continue

            # 5-day cooldown per ticker
            recent = [t for t in trades[-30:] if t.get('tk') == ticker
                     and (dates[i] - t['d']).days < 5]
            if recent:
                continue

            S = float(close.iloc[i])
            iv = float(sig['hv_20'].iloc[i]) * 1.15 if pd.notna(sig['hv_20'].iloc[i]) else 0.25
            iv = max(0.10, min(1.0, iv))

            after = close.iloc[i+1:i+cfg['max_hold']+2].values.astype(float)
            result = simulate_trade(S, direction, cfg['strike'], iv, cfg['dte'],
                                   after, cfg['tp'], cfg['sl'], cfg['trail'], cfg['max_hold'])
            if result is None:
                continue

            pnl_pct, hold, exit_r = result
            sz = min(MAX_TRADE_SIZE, equity * 0.30)
            if sz < 30:
                continue
            dpnl = sz * pnl_pct
            equity += dpnl
            trades.append({
                'd': dates[i], 'tk': ticker, 'dir': direction,
                'pnl': pnl_pct, 'dpnl': dpnl, 'hold': hold,
                'exit': exit_r, 'eq': equity,
            })

    if len(trades) < 15:
        return None

    pnls = [t['pnl'] for t in trades]
    avg = np.mean(pnls)
    std = np.std(pnls)
    avg_hold = np.mean([t['hold'] for t in trades])
    sharpe = (avg / std * np.sqrt(252 / max(avg_hold, 1))) if std > 0 else 0
    wr = sum(1 for p in pnls if p > 0) / len(pnls)

    dpnls = [t['dpnl'] for t in trades]
    gp = sum(p for p in dpnls if p > 0)
    gl = abs(sum(p for p in dpnls if p < 0))
    pf = gp / gl if gl > 0 else 0

    equities = [INITIAL_CAPITAL] + [t['eq'] for t in trades]
    peak = INITIAL_CAPITAL
    mdd = 0
    for e in equities:
        peak = max(peak, e)
        mdd = min(mdd, (e - peak) / peak)

    neg = [p for p in pnls if p < 0]
    ds = np.std(neg) if neg else 1e-6
    sortino = (avg / ds * np.sqrt(252 / max(avg_hold, 1))) if ds > 0 else 0

    total_ret = (trades[-1]['eq'] - INITIAL_CAPITAL) / INITIAL_CAPITAL

    return {
        'sharpe': sharpe, 'sortino': sortino, 'wr': wr, 'pf': pf,
        'mdd': mdd, 'n': len(trades), 'final': trades[-1]['eq'],
        'avg_hold': avg_hold, 'total_ret': total_ret, 'trades': trades,
    }


def validate(trades, n_perms=N_PERMUTATIONS):
    """4-gate validation."""
    pnls = [t['pnl'] for t in trades]
    real_sr = np.mean(pnls) / max(np.std(pnls), 1e-6)

    # G1: Permutation
    psrs = []
    for _ in range(n_perms):
        sh = [p * (1 if np.random.random() > 0.5 else -1) for p in pnls]
        if np.std(sh) > 0: psrs.append(np.mean(sh) / np.std(sh))
    z1 = (real_sr - np.mean(psrs)) / max(np.std(psrs), 1e-6) if psrs else 0
    p1 = 1 - stats.norm.cdf(z1)
    g1 = p1 < 0.05

    # G2: Regime (call vs put)
    cp = [t['pnl'] for t in trades if t['dir'] == 'call']
    pp = [t['pnl'] for t in trades if t['dir'] == 'put']
    if cp and pp:
        csr = np.mean(cp) / max(np.std(cp), 1e-6)
        psr = np.mean(pp) / max(np.std(pp), 1e-6)
        gap = abs(csr - psr) / max(abs(csr), abs(psr), 1e-6)
        g2 = gap < 0.50
    else:
        csr = psr = gap = None; g2 = False

    # G3: Random direction
    rsrs = []
    ap = [abs(p) for p in pnls]
    for _ in range(n_perms):
        r = [p * (1 if np.random.random() > 0.5 else -1) for p in ap]
        if np.std(r) > 0: rsrs.append(np.mean(r) / np.std(r))
    z3 = (real_sr - np.mean(rsrs)) / max(np.std(rsrs), 1e-6) if rsrs else 0
    p3 = 1 - stats.norm.cdf(z3)
    g3 = p3 < 0.05

    # G4: Sub-period
    mid = len(pnls) // 2
    h1 = np.mean(pnls[:mid]) / max(np.std(pnls[:mid]), 1e-6) if mid > 5 else 0
    h2 = np.mean(pnls[mid:]) / max(np.std(pnls[mid:]), 1e-6) if mid > 5 else 0
    g4 = h1 > 0 and h2 > 0

    return {
        'passed': sum([g1, g2, g3, g4]),
        'g1': {'pass': g1, 'z': float(z1), 'p': float(p1)},
        'g2': {'pass': g2, 'call_sr': float(csr) if csr else None,
               'put_sr': float(psr) if psr else None, 'gap': float(gap) if gap else None},
        'g3': {'pass': g3, 'z': float(z3), 'p': float(p3)},
        'g4': {'pass': g4, 'h1': float(h1), 'h2': float(h2)},
    }


def main():
    print("=" * 70)
    print("MOMENTUM BURST v3 — CALIBRATED ATM PRICING")
    print("=" * 70)
    print(f"Start: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Haircuts: ATM={HAIRCUTS['atm']:.0%}, OTM2%={HAIRCUTS['otm_2']:.0%}, "
          f"OTM5%={HAIRCUTS['otm_5']:.0%}")

    if USE_MLFLOW:
        try:
            mlflow.set_experiment("momentum_burst_v3_calibrated")
            mlflow.start_run(run_name=f"mb_v3_{datetime.now().strftime('%Y%m%d_%H%M')}")
            mlflow.set_tag("node", "jupiter")
        except: pass

    print("\nFetching data...")
    data = fetch_data(TICKERS)
    signals = compute_signals(data)
    print(f"Signals computed for {len(signals)} tickers")

    # === SWEEP ===
    configs = []

    # Phase A: DTE × Strike at baseline exits
    for dte in [7, 10, 14, 21, 28]:
        for strike in [0.0, 0.02, -0.02, -0.05]:
            configs.append({
                'dte': dte, 'strike': strike,
                'tp': 0.30, 'sl': 0.25, 'trail': 0.50,
                'max_hold': 5, 'min_sig': 2,
                'label': f'DTE={dte},STR={strike:+.0%}',
            })

    # Phase B: TP × SL at ATM DTE=14 (KB #281 baseline)
    for tp in [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        for sl in [0.10, 0.15, 0.20, 0.25, 0.30]:
            configs.append({
                'dte': 14, 'strike': 0.0,
                'tp': tp, 'sl': sl, 'trail': 0.50,
                'max_hold': 5, 'min_sig': 2,
                'label': f'TP={tp:.0%},SL={sl:.0%}',
            })

    # Phase C: MaxHold × Trailing at best
    for mh in [2, 3, 5, 7, 10]:
        for trail in [0.30, 0.40, 0.50, 0.60, 0.70]:
            configs.append({
                'dte': 14, 'strike': 0.0,
                'tp': 0.30, 'sl': 0.25, 'trail': trail,
                'max_hold': mh, 'min_sig': 2,
                'label': f'HOLD={mh}d,TRAIL={trail:.0%}',
            })

    # Phase D: Filters
    for msig in [1, 2, 3, 4]:
        configs.append({
            'dte': 14, 'strike': 0.0,
            'tp': 0.30, 'sl': 0.25, 'trail': 0.50,
            'max_hold': 5, 'min_sig': msig,
            'label': f'MIN_SIG={msig}',
        })

    print(f"\nSweeping {len(configs)} configs...")
    results = []

    for i, cfg in enumerate(configs):
        result = run_config(data, signals, cfg)
        if result:
            results.append((cfg, result))
            if len(results) % 20 == 0:
                print(f"  [{i+1}/{len(configs)}] {len(results)} valid configs so far...")

    print(f"\n{len(results)} valid configs (of {len(configs)} tested)")

    if not results:
        print("FATAL: no valid configs")
        return

    # Sort by Sharpe
    results.sort(key=lambda x: x[1]['sharpe'], reverse=True)

    # Top 15
    print("\n" + "=" * 70)
    print("TOP 15 CONFIGS BY SHARPE")
    print("=" * 70)
    print(f"{'#':<3} {'Config':<30} {'Sharpe':>7} {'Sort':>7} {'WR':>5} {'PF':>5} "
          f"{'MDD':>7} {'N':>4} {'Final$':>8}")
    print("-" * 85)

    for i, (cfg, met) in enumerate(results[:15]):
        print(f"{i+1:<3} {cfg['label']:<30} {met['sharpe']:>7.2f} {met['sortino']:>7.2f} "
              f"{met['wr']*100:>4.0f}% {met['pf']:>5.2f} {met['mdd']*100:>6.1f}% "
              f"{met['n']:>4} ${met['final']:>7.0f}")

    # Bottom 5 (worst)
    print("\nBOTTOM 5:")
    for i, (cfg, met) in enumerate(results[-5:]):
        print(f"  {cfg['label']:<30} Sharpe {met['sharpe']:>7.2f}, WR {met['wr']*100:.0f}%, ${met['final']:.0f}")

    # === VALIDATE BEST ===
    best_cfg, best_met = results[0]
    print(f"\n{'='*70}")
    print(f"BEST CONFIG: {best_cfg['label']}")
    print(f"{'='*70}")
    print(f"  DTE={best_cfg['dte']}, Strike={best_cfg['strike']:+.0%}, "
          f"TP={best_cfg['tp']:.0%}, SL={best_cfg['sl']:.0%}, "
          f"Trail={best_cfg['trail']:.0%}, MaxHold={best_cfg['max_hold']}d, "
          f"MinSig={best_cfg['min_sig']}")
    print(f"  Sharpe: {best_met['sharpe']:.2f}")
    print(f"  Sortino: {best_met['sortino']:.2f}")
    print(f"  WR: {best_met['wr']*100:.1f}%")
    print(f"  PF: {best_met['pf']:.2f}")
    print(f"  MDD: {best_met['mdd']*100:.1f}%")
    print(f"  Total Return: {best_met['total_ret']*100:.1f}%")
    print(f"  Final: ${best_met['final']:.0f} (from $645)")
    print(f"  Trades: {best_met['n']}, Avg hold: {best_met['avg_hold']:.1f} days")

    # KB #281 baseline comparison
    kb281_cfg = {'dte': 14, 'strike': 0.0, 'tp': 0.30, 'sl': 0.25,
                 'trail': 0.50, 'max_hold': 5, 'min_sig': 2, 'label': 'KB281_baseline'}
    kb281 = run_config(data, signals, kb281_cfg)
    if kb281:
        print(f"\n  KB #281 BASELINE (calibrated): Sharpe {kb281['sharpe']:.2f}, "
              f"WR {kb281['wr']*100:.0f}%, ${kb281['final']:.0f}")
        if kb281['sharpe'] != 0:
            impr = (best_met['sharpe'] - kb281['sharpe']) / abs(kb281['sharpe']) * 100
            print(f"  IMPROVEMENT: {impr:+.1f}%")

    # 4-gate validation
    print(f"\n--- 4-Gate Validation (best config) ---")
    val = validate(best_met['trades'])
    print(f"  G1 Permutation: z={val['g1']['z']:.2f}, p={val['g1']['p']:.3f} → {'PASS' if val['g1']['pass'] else 'FAIL'}")
    g2 = val['g2']
    if g2['gap'] is not None:
        print(f"  G2 Regime: call={g2['call_sr']:.2f}, put={g2['put_sr']:.2f}, "
              f"gap={g2['gap']:.2f} → {'PASS' if g2['pass'] else 'FAIL'}")
    else:
        print(f"  G2 Regime: one-directional → FAIL")
    print(f"  G3 Random: z={val['g3']['z']:.2f}, p={val['g3']['p']:.3f} → {'PASS' if val['g3']['pass'] else 'FAIL'}")
    print(f"  G4 Sub-period: h1={val['g4']['h1']:.2f}, h2={val['g4']['h2']:.2f} → {'PASS' if val['g4']['pass'] else 'FAIL'}")
    print(f"\n  GATES: {val['passed']}/4")

    # Also validate KB #281 baseline
    if kb281:
        print(f"\n--- KB #281 Baseline Validation ---")
        bval = validate(kb281['trades'])
        print(f"  Gates: {bval['passed']}/4")
        print(f"  G1 perm: z={bval['g1']['z']:.2f}, p={bval['g1']['p']:.3f}")

    # Exit breakdown for best
    exits = {}
    for t in best_met['trades']:
        exits[t['exit']] = exits.get(t['exit'], 0) + 1
    print(f"\n  Exit breakdown: {exits}")

    # Direction breakdown
    calls = [t for t in best_met['trades'] if t['dir'] == 'call']
    puts = [t for t in best_met['trades'] if t['dir'] == 'put']
    print(f"  Calls: {len(calls)} (WR {sum(1 for t in calls if t['pnl']>0)/max(len(calls),1)*100:.0f}%)")
    print(f"  Puts: {len(puts)} (WR {sum(1 for t in puts if t['pnl']>0)/max(len(puts),1)*100:.0f}%)")

    # MLflow
    if USE_MLFLOW:
        try:
            mlflow.log_metrics({
                'best_sharpe': round(best_met['sharpe'], 3),
                'best_sortino': round(best_met['sortino'], 3),
                'best_wr': round(best_met['wr'], 3),
                'best_pf': round(min(best_met['pf'], 99), 3),
                'best_mdd': round(best_met['mdd'], 3),
                'gates_passed': val['passed'],
                'configs_valid': len(results),
                'configs_tested': len(configs),
                'kb281_sharpe': round(kb281['sharpe'], 3) if kb281 else 0,
            })
            mlflow.set_tag('best_config', best_cfg['label'])
            mlflow.set_tag('result', 'PASS' if val['passed'] >= 3 else 'FAIL')
            mlflow.set_tag('pricing', 'moneyness_calibrated')
            mlflow.end_run()
        except: pass

    # Save
    save = {
        'best_config': {k: v for k, v in best_cfg.items()},
        'best_metrics': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in best_met.items() if k != 'trades'},
        'validation': val,
        'kb281_baseline': {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                          for k, v in kb281.items() if k != 'trades'} if kb281 else None,
        'top_10': [
            {'config': c['label'], 'sharpe': float(m['sharpe']), 'wr': float(m['wr']),
             'n': m['n'], 'final': float(m['final'])}
            for c, m in results[:10]
        ],
        'pricing_model': HAIRCUTS,
        'total_configs': len(configs),
        'valid_configs': len(results),
    }
    out = '/home/jupiter/Lvl3Quant/research/findings/momentum_burst_v3_results.json'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(save, f, indent=2, default=str)

    print(f"\nDone at {datetime.now().strftime('%H:%M:%S')}")


if __name__ == '__main__':
    main()
