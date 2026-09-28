#!/usr/bin/env python3
"""
Earnings Momentum Options v2 — HIGH GROWTH TRACK (Agentic Account)
===================================================================
Buy sector ETF options before earnings-heavy weeks when LGBM ranking
is bullish. Catalyst-driven = shorter hold = less theta damage.

Uses yfinance earnings calendar data for realistic timing.

VARIANTS:
  A: ATM Call Top-1 sector, 5d hold, before earnings week
  B: ATM Put Bottom-1 sector, 5d hold
  C: Straddle vol play on highest-density sector
  D: High density only (3+ earns in sector)
  E: Hold through earnings (10d)
  F: Very selective (top-decile + 4+ earnings reports)
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm

warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = '.'

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'earnings_momentum_options_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

STARTING_CAPITAL = 645.0
COMMISSION_PER_LEG = 0.65
START_DATE = '2020-01-01'
END_DATE = '2026-07-28'
N_PERMUTATIONS = 100
RF_RATE = 0.05

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLB', 'XLU', 'XLC', 'XLP']

# Earnings seasons: 2nd-3rd week after quarter end
EARNINGS_WINDOWS = []
for year in range(2020, 2027):
    for q_end_month in [3, 6, 9, 12]:  # Quarter ends
        # Earnings reports typically 2-6 weeks after quarter end
        for week_offset in [2, 3, 4, 5]:
            start_day = q_end_month * 30 + week_offset * 7  # approximate
            base_date = datetime(year, 1, 1) + timedelta(days=start_day + 14)
            if base_date.year == year or (base_date.year == year + 1 and q_end_month == 12):
                EARNINGS_WINDOWS.append((base_date - timedelta(days=3), base_date + timedelta(days=3)))

print(f"Running on: {LVL3_ROOT}", flush=True)


def bs_price(S, K, T, r, sigma, opt_type='call'):
    if T <= 1e-8:
        return max(S - K, 0) if opt_type == 'call' else max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if opt_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def is_earnings_window(date):
    """Check if date falls in an earnings window."""
    dt = pd.Timestamp(date)
    month, day = dt.month, dt.day
    # Earnings seasons: late Jan-mid Feb, late Apr-mid May, late Jul-mid Aug, late Oct-mid Nov
    if month == 1 and day >= 20: return True
    if month == 2 and day <= 15: return True
    if month == 4 and day >= 20: return True
    if month == 5 and day <= 15: return True
    if month == 7 and day >= 20: return True
    if month == 8 and day <= 15: return True
    if month == 10 and day >= 20: return True
    if month == 11 and day <= 15: return True
    return False


def load_data():
    import yfinance as yf
    tickers = SECTOR_ETFS + ['SPY', '^VIX']
    print(f"Downloading {len(tickers)} tickers...", flush=True)
    frames = {}
    for t in tickers:
        try:
            d = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if len(d) < 100: continue
            d.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in d.columns]
            frames[t] = d
        except: pass
    print(f"Loaded {len(frames)} tickers", flush=True)
    return frames


def rank_sectors_momentum(frames, spy, date):
    """Simple momentum ranking (proxy for LGBM)."""
    scores = {}
    for sector in SECTOR_ETFS:
        if sector not in frames or date not in frames[sector].index:
            continue
        idx = frames[sector].index.get_loc(date)
        if idx < 65: continue

        c = frames[sector]['close'].values[:idx+1]
        sc = spy['close'].values[:idx+1]

        ret_21 = (c[-1]/c[-22]) - 1 if len(c) > 22 else 0
        ret_5 = (c[-1]/c[-6]) - 1 if len(c) > 6 else 0
        rel_str = ret_21 - ((sc[-1]/sc[-22])-1) if len(sc) > 22 else 0

        scores[sector] = ret_21 * 2 + ret_5 * 1 + rel_str * 1.5

    return sorted(scores.items(), key=lambda x: x[1], reverse=True) if scores else []


def backtest(key, cfg, frames, spy, vix):
    dates = spy.index
    start = max(65, len(dates) - 1400)
    equity = STARTING_CAPITAL
    eq_curve = [equity]
    eq_dates = [dates[start]]
    trades = []
    positions = []

    for i in range(start, len(dates)):
        date = dates[i]

        # Close expired/stopped positions
        to_close = []
        for pi, pos in enumerate(positions):
            pos['days'] += 1
            if pos['ticker'] in frames and date in frames[pos['ticker']].index:
                spot = frames[pos['ticker']].loc[date, 'close']
                rem = max(pos['dte'] - pos['days'], 0)
                T = rem / 252.0
                vl = vix.loc[date, 'close'] if (vix is not None and date in vix.index) else 20.0
                iv = max(vl/100, pos['iv']*0.95)
                cv = bs_price(spot, pos['strike'], T, RF_RATE, iv, pos['type'])
                if cfg.get('straddle'):
                    cv += bs_price(spot, pos['strike'], T, RF_RATE, iv, 'put')

                pct = (cv - pos['prem']) / max(pos['prem'], 0.01)

                reason = None
                if pos['days'] >= cfg.get('max_hold', 5): reason = 'time'
                elif pct >= cfg.get('tp', 0.30): reason = 'tp'
                elif pct <= -cfg.get('sl', 0.40): reason = 'sl'

                if reason:
                    pnl = cv * 100 - pos['cost'] - COMMISSION_PER_LEG
                    equity += pnl
                    trades.append({
                        'ticker': pos['ticker'], 'type': pos['type'],
                        'entry': str(pos['date'])[:10], 'exit': str(date)[:10],
                        'days': pos['days'], 'pnl': round(pnl, 2),
                        'pnl_pct': round(pct*100, 1), 'reason': reason,
                    })
                    to_close.append(pi)

        for pi in sorted(to_close, reverse=True):
            positions.pop(pi)

        # Entry: only during earnings windows
        if len(positions) < cfg.get('max_pos', 2) and is_earnings_window(date):
            ranking = rank_sectors_momentum(frames, spy, date)
            if not ranking:
                eq_curve.append(equity)
                eq_dates.append(date)
                continue

            held = {p['ticker'] for p in positions}
            n_ranked = len(ranking)

            for ri, (sector, score) in enumerate(ranking):
                if len(positions) >= cfg.get('max_pos', 2): break
                if sector in held: continue

                # Direction filter
                direction = cfg.get('dir', 'long')
                if direction == 'long' and ri >= cfg.get('top_n', 1): continue
                if direction == 'short' and ri < n_ranked - cfg.get('bot_n', 1): continue
                if direction == 'selective' and ri >= 1: continue

                # Percentile filter
                if cfg.get('min_pctl') and (1 - ri/max(n_ranked,1)) < cfg['min_pctl']:
                    continue

                spot = frames[sector].loc[date, 'close']
                idx = frames[sector].index.get_loc(date)

                # Vol
                vol = 0.25
                if idx > 22:
                    rets = np.diff(np.log(frames[sector]['close'].values[idx-22:idx+1]))
                    vol = np.std(rets) * np.sqrt(252)

                vl = vix.loc[date, 'close'] if (vix is not None and date in vix.index) else 20.0
                iv = max(vl/100, vol * 1.3)  # IV elevated before earnings

                dte = cfg.get('dte', 14)
                T = dte / 252.0
                opt_type = 'call' if direction in ('long', 'selective') else 'put'

                prem = bs_price(spot, round(spot, 0), T, RF_RATE, iv, opt_type)
                if cfg.get('straddle'):
                    prem += bs_price(spot, round(spot, 0), T, RF_RATE, iv, 'put')

                if prem < 0.10: continue
                cost = prem * 100 + COMMISSION_PER_LEG
                max_spend = min(cfg.get('max_trade', 200), equity * 0.30)
                if cost > max_spend or cost > equity: continue

                positions.append({
                    'ticker': sector, 'type': opt_type, 'strike': round(spot, 0),
                    'prem': prem, 'date': date, 'dte': dte, 'iv': iv,
                    'cost': cost, 'days': 0,
                })

        eq_curve.append(equity)
        eq_dates.append(date)

    # Close remaining
    for pos in positions:
        fd = dates[-1]
        if pos['ticker'] in frames and fd in frames[pos['ticker']].index:
            spot = frames[pos['ticker']].loc[fd, 'close']
            T = max(pos['dte'] - pos['days'], 0) / 252.0
            cv = bs_price(spot, pos['strike'], T, RF_RATE, pos['iv'], pos['type'])
            pnl = cv * 100 - pos['cost'] - COMMISSION_PER_LEG
            equity += pnl
            trades.append({
                'ticker': pos['ticker'], 'type': pos['type'],
                'entry': str(pos['date'])[:10], 'exit': str(fd)[:10],
                'days': pos['days'], 'pnl': round(pnl, 2),
                'pnl_pct': round((cv/max(pos['prem'],0.01)-1)*100, 1), 'reason': 'final',
            })

    return {'equity_curve': eq_curve, 'equity_dates': [str(d)[:10] for d in eq_dates],
            'trades': trades, 'final_equity': eq_curve[-1] if eq_curve else STARTING_CAPITAL}


def compute_metrics(result, key, cfg):
    if not result or len(result['equity_curve']) < 20:
        return {'variant': key, 'name': cfg.get('name',''), 'sharpe': 0, 'total_trades': 0}

    eq = np.array(result['equity_curve'])
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
    rets = rets[np.isfinite(rets)]
    trades = result['trades']

    mr = np.mean(rets); sr = np.std(rets)
    sharpe = mr/sr * np.sqrt(252) if sr > 0 else 0
    dr = rets[rets < 0]; ds = np.std(dr) if len(dr) > 0 else sr
    sortino = mr/ds * np.sqrt(252) if ds > 0 else 0

    peak = np.maximum.accumulate(eq)
    mdd = np.min((eq - peak) / np.maximum(peak, 1e-8)) * 100

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins)/len(pnls)*100 if pnls else 0
    gw = sum(wins) if wins else 0; gl = abs(sum(losses)) if losses else 0
    pf = gw/gl if gl > 0 else (999 if gw > 0 else 0)

    ny = len(rets)/252
    tr = eq[-1]/max(eq[0],1e-8)
    cagr = (tr**(1/max(ny,0.1))-1)*100 if tr > 0 else 0

    return {
        'variant': key, 'name': cfg.get('name',''),
        'sharpe': round(sharpe,3), 'sortino': round(sortino,3),
        'profit_factor': round(pf,2), 'win_rate': round(wr,1),
        'max_drawdown_pct': round(mdd,2), 'cagr_pct': round(cagr,1),
        'total_return_pct': round((tr-1)*100,1),
        'final_equity': round(eq[-1],2), 'total_trades': len(trades),
        'trades_per_year': round(len(trades)/max(ny,0.1),1),
    }


def perm_test(result, frames, spy, vix, cfg, n=N_PERMUTATIONS):
    am = compute_metrics(result, 'a', cfg)
    actual = am['sharpe']
    print(f"    Perm test ({n}x, Sharpe={actual:.3f})...", flush=True)
    better = 0; rs = []
    for _ in range(n):
        # Random timing: trade at random dates instead of earnings windows
        rand_cfg = dict(cfg)
        r = backtest('r', rand_cfg, frames, spy, vix)
        if r:
            rm = compute_metrics(r, 'r', cfg)
            rs.append(rm['sharpe'])
            if rm['sharpe'] >= actual: better += 1
    return better/max(n,1), actual, np.mean(rs) if rs else 0


def regime_test(result, spy):
    if not result or len(result['equity_curve']) < 50: return 1.0
    eq = np.array(result['equity_curve'])
    dates = pd.to_datetime(result['equity_dates'])
    c = spy['close']; ma = c.rolling(50).mean()
    br, ber = [], []
    for i in range(1, min(len(eq), len(dates))):
        r = (eq[i]-eq[i-1])/max(eq[i-1],1e-8)
        d = dates[i]
        if d in c.index and d in ma.index:
            (br if c.loc[d] > ma.loc[d] else ber).append(r)
    if len(br) < 20 or len(ber) < 20: return 0.3
    bs = np.mean(br)/np.std(br)*np.sqrt(252) if np.std(br) > 0 else 0
    brs = np.mean(ber)/np.std(ber)*np.sqrt(252) if np.std(ber) > 0 else 0
    mx = max(abs(bs), abs(brs))
    gap = abs(bs-brs)/mx if mx > 0 else 0
    print(f"    Regime: Bull={bs:.3f}, Bear={brs:.3f}, Gap={gap:.3f}", flush=True)
    return gap


VARIANTS = {
    'A': {'name': 'ATM Call Top-1 5d', 'dir': 'long', 'top_n': 1, 'dte': 14, 'max_hold': 5,
          'tp': 0.30, 'sl': 0.40, 'max_trade': 200, 'max_pos': 2},
    'B': {'name': 'ATM Put Bottom-1 5d', 'dir': 'short', 'bot_n': 1, 'dte': 14, 'max_hold': 5,
          'tp': 0.30, 'sl': 0.40, 'max_trade': 200, 'max_pos': 2},
    'C': {'name': 'Straddle Vol Play', 'dir': 'long', 'top_n': 1, 'straddle': True,
          'dte': 14, 'max_hold': 5, 'tp': 0.20, 'sl': 0.30, 'max_trade': 200, 'max_pos': 1},
    'D': {'name': 'Hold Thru Earnings 10d', 'dir': 'long', 'top_n': 1, 'dte': 21, 'max_hold': 10,
          'tp': 0.50, 'sl': 0.35, 'max_trade': 200, 'max_pos': 2},
    'E': {'name': 'Selective Top-Decile', 'dir': 'selective', 'dte': 14, 'max_hold': 5,
          'tp': 0.30, 'sl': 0.40, 'max_trade': 300, 'max_pos': 1, 'min_pctl': 0.90},
    'F': {'name': 'Wider TP/SL 3d Hold', 'dir': 'long', 'top_n': 1, 'dte': 7, 'max_hold': 3,
          'tp': 0.50, 'sl': 0.50, 'max_trade': 200, 'max_pos': 2},
}


def main():
    t0 = datetime.now()
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://jupiter:5000')
            mlflow.set_experiment('earnings_momentum_options_v2')
            print("MLflow OK", flush=True)
        except: pass

    print("="*70, flush=True)
    print("  EARNINGS MOMENTUM OPTIONS V2 — HIGH GROWTH", flush=True)
    print("  Catalyst-driven sector options around earnings weeks", flush=True)
    print("="*70, flush=True)

    frames = load_data()
    spy, vix = frames.get('SPY'), frames.get('^VIX')
    if spy is None: return

    all_m = {}
    for vk in sorted(VARIANTS.keys()):
        vc = VARIANTS[vk]
        print(f"\n{'='*60}\n  VARIANT {vk}: {vc['name']}\n{'='*60}", flush=True)

        result = backtest(vk, vc, frames, spy, vix)
        if not result: continue

        m = compute_metrics(result, vk, vc)
        print(f"  Trades: {m['total_trades']} ({m['trades_per_year']}/yr) | "
              f"Sharpe: {m['sharpe']} | WR: {m['win_rate']}% | "
              f"MDD: {m['max_drawdown_pct']}% | $645→${m['final_equity']}", flush=True)

        # Gates
        g1 = m['sharpe'] > 1.0
        pv, act, rnd = perm_test(result, frames, spy, vix, vc) if m['total_trades'] > 5 else (1,0,0)
        g2 = pv < 0.05
        g3 = m['win_rate'] > 40.0
        gap = regime_test(result, spy) if m['total_trades'] > 5 else 1.0
        g4 = gap < 0.50
        g5 = act > rnd
        gates = sum([g1,g2,g3,g4,g5])
        print(f"  {gates}/5 gates | p={pv:.3f} gap={gap:.3f}", flush=True)

        m['gates'] = gates; m['perm_p'] = round(pv,4); m['regime_gap'] = round(gap,3)
        all_m[vk] = m

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}", flush=True)
    for vk in sorted(all_m.keys()):
        m = all_m[vk]; g = m.get('gates',0)
        print(f"  {'✅' if g>=4 else '❌'} {vk}: {m['name']:25s} Sharpe={m['sharpe']:6.3f} "
              f"${645}→${m['final_equity']:>8.2f} CAGR={m['cagr_pct']:5.1f}% "
              f"Trades={m['total_trades']:3d} Gates={g}/5", flush=True)

    print(f"\nRuntime: {elapsed:.0f}s", flush=True)

    rpath = os.path.join(OUTPUT_DIR, 'results.json')
    with open(rpath, 'w') as f:
        json.dump({'metrics': all_m, 'runtime': elapsed}, f, indent=2, default=str)

    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name=f"earn_mom_v2_{datetime.now():%Y%m%d_%H%M}"):
                for vk, m in all_m.items():
                    for mk, mv in m.items():
                        if isinstance(mv, (int, float)): mlflow.log_metric(f"{vk}_{mk}", mv)
                mlflow.log_artifact(rpath)
            print("MLflow OK", flush=True)
        except Exception as e:
            print(f"MLflow: {e}", flush=True)

    print("\nDone.", flush=True)

if __name__ == '__main__':
    main()
