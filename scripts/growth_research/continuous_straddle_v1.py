#!/usr/bin/env python3
"""
Continuous Straddle Rotation v1
==================================

BUY straddles on the most volatile growth stocks continuously,
not just before earnings. Rotate every 14 trading days.

MOTIVATION:
- IV Run-Up adversarial showed random-timed straddles on growth stocks
  are profitable (Sharpe 1.45) even without earnings timing
- This tests whether a continuous straddle rotation captures that edge
- If it works, it provides non-earnings-dependent income

VARIANTS:
A — Top-3 highest 21d vol, straddle, 14d hold, monthly rotation
B — Top-3 highest vol, 7d hold (faster rotation, less theta)
C — Top-3 highest vol + momentum filter (vol AND momentum aligned)
D — Top-5 highest vol, 14d hold (more diversified)
E — Top-3 vol, only when VIX < 20 (low vol environment = more upside)
F — Top-3 vol, AVOID earnings window (pure non-event vol capture)

UNIVERSE: 52 growth stocks
CAPITAL: $645 (agentic account)
MAX POS: $200 per trade, max 3 concurrent
PERIOD: 2022-01-01 to 2026-07-01
"""

import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = '.'

LOG_DIR = os.path.join(LVL3_ROOT, 'scripts', 'growth_research', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'continuous_straddle_v1.log')),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI',
]

INITIAL_CAPITAL = 645.0
MAX_POS_COST = 200.0
COMMISSION_RT = 1.30
RISK_FREE_RATE = 0.05
BS_HAIRCUT = 0.85

OOT_START = '2022-01-01'
OOT_END = '2026-07-01'

VARIANTS = {
    'A_Top3_14d': {
        'desc': 'Top-3 vol, straddle, 14d hold',
        'top_k': 3, 'hold_days': 14, 'filters': {},
    },
    'B_Top3_7d': {
        'desc': 'Top-3 vol, straddle, 7d hold (faster)',
        'top_k': 3, 'hold_days': 7, 'filters': {},
    },
    'C_MomFilter': {
        'desc': 'Top-3 vol + momentum filter',
        'top_k': 3, 'hold_days': 14, 'filters': {'momentum': True},
    },
    'D_Top5_14d': {
        'desc': 'Top-5 vol, 14d hold (diversified)',
        'top_k': 5, 'hold_days': 14, 'filters': {},
    },
    'E_LowVIX': {
        'desc': 'Top-3 vol, only VIX < 20',
        'top_k': 3, 'hold_days': 14, 'filters': {'vix_max': 20},
    },
    'F_NoEarnings': {
        'desc': 'Top-3 vol, avoid earnings window (±10d)',
        'top_k': 3, 'hold_days': 14, 'filters': {'avoid_earnings': True},
    },
}


def bs_call(S, K, T, r, sigma):
    if T <= 1e-8: return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 1e-8: return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def straddle_price(S, K, T, r, sigma):
    return (bs_call(S, K, T, r, sigma) + bs_put(S, K, T, r, sigma)) * BS_HAIRCUT


def load_data():
    import yfinance as yf
    cache_path = os.path.join(LVL3_ROOT, 'data', 'continuous_straddle_prices.parquet')
    earnings_cache = os.path.join(LVL3_ROOT, 'data', 'continuous_straddle_earnings.json')

    if os.path.exists(cache_path):
        prices = pd.read_parquet(cache_path)
        log.info(f"Loaded prices: {len(prices)} rows")
    else:
        log.info("Downloading...")
        frames = []
        for t in STOCK_UNIVERSE + ['SPY', '^VIX']:
            try:
                df = yf.download(t, start='2020-01-01', progress=False, auto_adjust=True)
                if len(df) < 50: continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                df['ticker'] = t
                df.index.name = 'date'
                frames.append(df.reset_index())
            except:
                pass
        prices = pd.concat(frames, ignore_index=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices.to_parquet(cache_path)

    earnings = {}
    if os.path.exists(earnings_cache):
        with open(earnings_cache) as f:
            earnings = json.load(f)
    else:
        for t in STOCK_UNIVERSE:
            try:
                stock = yf.Ticker(t)
                dates = stock.get_earnings_dates(limit=40)
                if dates is not None:
                    earnings[t] = [str(d.date()) if hasattr(d, 'date') else str(d)[:10] for d in dates.index]
            except:
                pass
        with open(earnings_cache, 'w') as f:
            json.dump(earnings, f)

    return prices, earnings


def run_variant(vname, config, prices, earnings):
    log.info(f"\n{'='*60}")
    log.info(f"  {vname}: {config['desc']}")
    log.info(f"{'='*60}")

    top_k = config['top_k']
    hold_days = config['hold_days']
    filters = config.get('filters', {})

    capital = INITIAL_CAPITAL
    trades = []
    equity_curve = [capital]
    equity_dates = [pd.Timestamp(OOT_START)]

    # Get VIX
    vix_df = prices[prices['ticker'] == '^VIX'].sort_values('date').reset_index(drop=True)
    vix_dates = vix_df['date'].values
    vix_close = vix_df['close'].values

    # Pre-compute per-ticker data
    ticker_data = {}
    for t in STOCK_UNIVERSE:
        tdf = prices[prices['ticker'] == t].sort_values('date').reset_index(drop=True)
        if len(tdf) < 60:
            continue
        ticker_data[t] = tdf

    # Generate rebalance dates (every hold_days trading days)
    # Use SPY trading days as reference calendar
    spy_df = prices[prices['ticker'] == 'SPY'].sort_values('date').reset_index(drop=True)
    spy_dates = spy_df['date'].values
    oot_mask = (spy_dates >= np.datetime64(OOT_START)) & (spy_dates <= np.datetime64(OOT_END))
    oot_dates = spy_dates[oot_mask]

    rebalance_indices = list(range(0, len(oot_dates), hold_days))

    for i, reb_idx in enumerate(rebalance_indices):
        if reb_idx + hold_days >= len(oot_dates):
            break

        entry_date = pd.Timestamp(oot_dates[reb_idx])
        exit_date = pd.Timestamp(oot_dates[min(reb_idx + hold_days, len(oot_dates) - 1)])

        # VIX filter
        if filters.get('vix_max'):
            vix_loc = np.searchsorted(vix_dates, np.datetime64(entry_date))
            if vix_loc > 0 and vix_loc <= len(vix_close):
                current_vix = float(vix_close[min(vix_loc, len(vix_close)-1)])
                if current_vix > filters['vix_max']:
                    continue

        # Rank stocks by 21-day realized vol
        vol_ranking = []
        for t, tdf in ticker_data.items():
            td = tdf['date'].values
            loc = np.searchsorted(td, np.datetime64(entry_date))
            if loc < 22 or loc >= len(tdf):
                continue

            close_window = tdf['close'].values[loc-21:loc+1]
            if len(close_window) < 22:
                continue

            rets = np.diff(np.log(close_window))
            vol = float(np.std(rets) * np.sqrt(252))

            # Momentum filter
            if filters.get('momentum'):
                mom_5d = float(close_window[-1] / close_window[-6] - 1) if len(close_window) >= 6 else 0
                # Only pick stocks where vol is high AND 5d momentum is strong (either direction)
                if abs(mom_5d) < 0.02:  # need at least 2% move in 5 days
                    continue

            # Avoid earnings window
            if filters.get('avoid_earnings'):
                earn_dates = earnings.get(t, [])
                near_earnings = False
                for ed in earn_dates:
                    days_diff = abs((pd.Timestamp(ed) - entry_date).days)
                    if days_diff <= 10:
                        near_earnings = True
                        break
                if near_earnings:
                    continue

            entry_price = float(tdf['close'].iloc[loc])

            vol_ranking.append({
                'ticker': t,
                'vol': vol,
                'entry_idx': loc,
                'entry_price': entry_price,
            })

        # Sort by vol descending, pick top-K
        vol_ranking.sort(key=lambda x: x['vol'], reverse=True)
        picks = vol_ranking[:top_k]

        # Trade each pick
        for pick in picks:
            t = pick['ticker']
            tdf = ticker_data[t]
            entry_idx = pick['entry_idx']
            entry_price = pick['entry_price']

            # Find exit index
            exit_loc = np.searchsorted(tdf['date'].values, np.datetime64(exit_date))
            if exit_loc >= len(tdf):
                exit_loc = len(tdf) - 1

            exit_price = float(tdf['close'].iloc[exit_loc])

            # Price straddle at entry
            strike = round(entry_price)
            vol_entry = pick['vol']
            iv_entry = vol_entry * 1.05  # small IV premium over realized
            dte = hold_days + 7  # extend past hold for liquidity
            T_entry = dte / 252.0

            straddle_entry = straddle_price(entry_price, strike, T_entry, RISK_FREE_RATE, iv_entry)
            entry_cost = straddle_entry * 100 + COMMISSION_RT

            if entry_cost <= 0 or entry_cost > MAX_POS_COST or entry_cost > capital:
                continue

            # Price at exit
            remaining_dte = max(dte - hold_days, 1)
            T_exit = remaining_dte / 252.0

            # At exit, IV ~ realized vol (no structural expansion)
            exit_loc_for_vol = min(exit_loc, len(tdf) - 1)
            if exit_loc_for_vol >= 22:
                exit_rets = np.diff(np.log(tdf['close'].values[exit_loc_for_vol-21:exit_loc_for_vol+1]))
                vol_exit = float(np.std(exit_rets) * np.sqrt(252)) if len(exit_rets) > 5 else vol_entry
            else:
                vol_exit = vol_entry
            iv_exit = vol_exit * 1.05

            straddle_exit = straddle_price(exit_price, strike, T_exit, RISK_FREE_RATE, iv_exit)
            exit_value = straddle_exit * 100 - COMMISSION_RT

            pnl = exit_value - entry_cost
            pnl_pct = pnl / entry_cost if entry_cost > 0 else 0
            capital += pnl

            stock_move = abs(exit_price / entry_price - 1) * 100

            trades.append({
                'ticker': t,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exit_price, 2),
                'stock_move_abs_pct': round(stock_move, 2),
                'vol_21d': round(pick['vol'] * 100, 1),
                'straddle_entry': round(straddle_entry, 2),
                'straddle_exit': round(straddle_exit, 2),
                'entry_cost': round(entry_cost, 2),
                'exit_value': round(exit_value, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct * 100, 2),
                'capital': round(capital, 2),
            })

            equity_curve.append(capital)
            equity_dates.append(exit_date)

    return trades, equity_curve, equity_dates


def evaluate(name, config, trades, equity_curve, equity_dates):
    n = len(trades)
    if n == 0:
        log.info(f"  {name}: NO TRADES")
        return {'name': name, 'n_trades': 0, 'gates_passed': 0}

    pnls = [t['pnl'] for t in trades]
    pnl_pcts = [t['pnl_pct'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n * 100
    total_pnl = sum(pnls)
    final = equity_curve[-1]
    total_ret = (final / INITIAL_CAPITAL - 1) * 100

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    years = max((equity_dates[-1] - equity_dates[0]).days / 365.25, 0.5) if len(equity_dates) > 1 else 1

    sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(n / years) if np.std(pnl_pcts) > 0 else 0
    downside = [p for p in pnl_pcts if p < 0]
    sortino = np.mean(pnl_pcts) / np.std(downside) * np.sqrt(n / years) if downside and np.std(downside) > 0 else sharpe

    peak = INITIAL_CAPITAL
    mdd = 0
    for eq in equity_curve:
        if eq > peak: peak = eq
        dd = (eq - peak) / peak
        if dd < mdd: mdd = dd
    mdd_pct = mdd * 100

    avg_stock_move = np.mean([t['stock_move_abs_pct'] for t in trades])

    # Gates (4 for agentic — concentration waived per HC #761)
    gates = 0
    gate_results = {}

    gate_results['sharpe'] = sharpe >= 1.0
    if gate_results['sharpe']: gates += 1

    # Perm test
    if n >= 10:
        obs = np.mean(pnl_pcts)
        count = sum(1 for _ in range(1000) if np.mean(np.random.choice([-1,1], n) * np.abs(pnl_pcts)) >= obs)
        perm_p = count / 1000
    else:
        perm_p = 1.0
    gate_results['perm'] = perm_p < 0.05
    if gate_results['perm']: gates += 1

    # Regime
    first = [t['pnl_pct'] for t in trades if t['entry_date'] < '2023-07-01']
    second = [t['pnl_pct'] for t in trades if t['entry_date'] >= '2023-07-01']
    if first and second:
        s1 = np.mean(first) / max(np.std(first), 0.01)
        s2 = np.mean(second) / max(np.std(second), 0.01)
        regime_gap = abs(s1 - s2) / max(abs(s1), abs(s2), 0.01)
    else:
        regime_gap = 1.0
    gate_results['regime'] = regime_gap < 0.50
    if gate_results['regime']: gates += 1

    # MDD
    gate_results['mdd'] = abs(mdd_pct) < 50
    if gate_results['mdd']: gates += 1

    result = {
        'name': name, 'desc': config['desc'], 'n_trades': n,
        'wins': wins, 'wr': round(wr, 1),
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3), 'pf': round(pf, 2),
        'final': round(final, 2), 'total_return': round(total_ret, 1),
        'mdd': round(mdd_pct, 1), 'perm_p': round(perm_p, 4),
        'regime_gap': round(regime_gap, 3),
        'avg_stock_move_pct': round(avg_stock_move, 2),
        'gates_passed': gates, 'gate_results': gate_results,
    }

    log.info(f"\n  --- {name} ---")
    log.info(f"  Trades: {n} (W:{wins} L:{n-wins} WR:{wr:.0f}%)")
    log.info(f"  Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | PF: {pf:.2f}")
    log.info(f"  Final: ${final:.0f} ({total_ret:+.1f}%) | MDD: {mdd_pct:.1f}%")
    log.info(f"  Perm p: {perm_p:.4f} | Regime gap: {regime_gap:.3f}")
    log.info(f"  Avg |stock move|: {avg_stock_move:.2f}%")
    log.info(f"  GATES: {gates}/4 {'✅' if gates >= 3 else '❌'}")
    for g, v in gate_results.items():
        log.info(f"    {g}: {'PASS' if v else 'FAIL'}")

    return result


def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("  Continuous Straddle Rotation v1")
    log.info("=" * 60)

    prices, earnings = load_data()
    results = []

    for vname, vconfig in VARIANTS.items():
        try:
            trades, eq, dates = run_variant(vname, vconfig, prices, earnings)
            result = evaluate(vname, vconfig, trades, eq, dates)
            results.append(result)
        except Exception as e:
            log.error(f"  {vname}: FAILED ({e})")
            import traceback; traceback.print_exc()
            results.append({'name': vname, 'gates_passed': 0, 'error': str(e)})

    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"  SUMMARY ({elapsed:.0f}s)")
    log.info(f"{'='*60}")
    for r in sorted(results, key=lambda x: x.get('gates_passed', 0), reverse=True):
        log.info(f"  {r['name']}: {r.get('gates_passed',0)}/4 | Sharpe {r.get('sharpe',0):.3f} | "
                 f"{r.get('n_trades',0)} trades | WR {r.get('wr',0):.0f}% | ${r.get('final', INITIAL_CAPITAL):.0f}")

    out = os.path.join(LVL3_ROOT, 'research', 'findings', 'continuous_straddle_v1.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    try:
        import mlflow
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment('continuous_straddle_v1')
        for r in results:
            with mlflow.start_run(run_name=r.get('name', '?')):
                for k, v in r.items():
                    if isinstance(v, (int, float)): mlflow.log_metric(k, v)
                mlflow.log_param('variant', r.get('name', ''))
        log.info("MLflow logged")
    except Exception as e:
        log.warning(f"MLflow: {e}")


if __name__ == '__main__':
    main()
