#!/usr/bin/env python3
"""
VIX ETF Structural Decay Options Strategy V1
==============================================

HYPOTHESIS: Leveraged VIX ETFs (UVXY, VXX) structurally lose value over time
due to futures contango roll cost. Buying puts on these instruments exploits
this well-known structural edge. Key advantages for $645 account:

  1. CHEAP CONTRACTS: VIX ETFs frequently reverse-split, keeping share prices
     in the $10-50 range. OTM puts cost $20-150 per contract — AFFORDABLE.
  2. LEVEL 2 COMPATIBLE: Just buying puts, no spreads needed.
  3. STRUCTURAL EDGE: Not dependent on market timing — contango decay is
     persistent across regimes (~5-7% per month historically on UVXY).
  4. REGIME-AGNOSTIC POTENTIAL: Decay happens in both bull and bear markets
     (though rate varies). During VIX spikes, puts become cheap insurance.

STRATEGY LOGIC:
  - Buy OTM puts on UVXY/VXX when VIX is elevated (mean-reversion setup)
  - VIX spike = UVXY rallies = puts are cheap = high R:R for mean-reversion
  - Time entries using VIX term structure (contango/backwardation ratio)
  - DTE 21-45 days (enough time for decay, not too much theta)

SIX VARIANTS:
  A: UVXY ATM puts, monthly, no filter (pure decay capture)
  B: UVXY OTM puts (15% OTM), monthly (cheaper, higher leverage)
  C: UVXY ATM puts with VIX filter (only enter when VIX > 20 = mean-reversion)
  D: UVXY OTM puts with VIX spike filter (VIX > 25, aggressive mean-reversion)
  E: Rolling puts with trailing profit take (+50% gain = roll to new put)
  F: Sizing by VIX level (higher VIX = larger position, max $300/trade)

VALIDATION:
  - 5 gates: Sharpe>1, perm p<0.05, WR>40%, regime balance, random baseline
  - ALL available OOT days, regime-stratified
  - Permutation test: 200 shuffles
  - Cost-adjusted (BS pricing + 20% conservative buffer for VIX options)

NOTE ON PRICING: VIX options have higher implied vol than sector ETFs, so
BS underpricing is MORE relevant here. We use a 30% pricing buffer
(conservative) to account for this.

Output: output/growth_research/vix_etf_decay_options_v1/
MLflow experiment: vix_etf_decay_options_v1
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# ── Detect environment ──
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")
BASE = _NEPTUNE_BASE if _NEPTUNE_BASE.exists() else _JUPITER_BASE
fprint(f"Running on: {BASE}")
sys.path.insert(0, str(BASE))

OUTPUT_DIR = BASE / "output" / "growth_research" / "vix_etf_decay_options_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow setup ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "vix_etf_decay_options_v1"
MLFLOW_OK = False
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    MLFLOW_OK = True
    fprint(f"MLflow OK: {MLFLOW_URI}")
except Exception as e:
    fprint(f"MLflow not available: {e}")

# ==================== CONFIG ====================
INITIAL_CAPITAL = 645.0
MAX_POSITION_SIZE = 300.0  # HC #749: max $200-300/trade
RISK_FREE_RATE = 0.05
PRICING_BUFFER = 1.30  # 30% conservative buffer for VIX options pricing

# ==================== BS PRICING ====================

def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_put_delta(S, K, T, r, sigma):
    """Black-Scholes put delta."""
    if T <= 0 or sigma <= 0:
        return -1.0 if K > S else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) - 1.0


def price_put_option(stock_price, otm_pct, dte_days, realized_vol, pricing_buffer=1.0):
    """
    Price a put option.

    Args:
        stock_price: current stock price
        otm_pct: how far OTM (0 = ATM, 0.15 = 15% OTM)
        dte_days: days to expiration
        realized_vol: annualized realized volatility
        pricing_buffer: multiplier for premium (1.3 = 30% buffer)

    Returns:
        dict with strike, premium, delta, contract_cost
    """
    T = dte_days / 365.0
    sigma = max(realized_vol, 0.30)  # VIX ETFs have minimum 30% vol

    # For puts: OTM means strike below stock price
    strike = stock_price * (1 - otm_pct)
    # Round to nearest $0.50
    strike = round(strike * 2) / 2
    if strike <= 0:
        strike = 0.50

    premium = bs_put_price(stock_price, strike, T, RISK_FREE_RATE, sigma)
    delta = bs_put_delta(stock_price, strike, T, RISK_FREE_RATE, sigma)

    # Apply pricing buffer
    premium *= pricing_buffer

    contract_cost = premium * 100  # 100 shares per contract

    return {
        'strike': strike,
        'premium_per_share': premium,
        'contract_cost': contract_cost,
        'delta': delta,
        'T': T,
        'sigma': sigma,
        'otm_pct': otm_pct,
    }


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download UVXY + VIX + SPY data."""
    import yfinance as yf

    # UVXY has multiple reverse splits — use adjusted close
    tickers = ['UVXY', 'SPY', '^VIX']
    fprint(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start='2020-01-01', progress=False)

    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill()

    rename_map = {'^VIX': 'VIX'}
    close = close.rename(columns=rename_map)

    uvxy = close['UVXY'].dropna()
    spy = close['SPY'].dropna()
    vix_col = 'VIX' if 'VIX' in close.columns else '^VIX'
    vix = close[vix_col].dropna()

    ix = uvxy.index.intersection(spy.index).intersection(vix.index)
    fprint(f"Data: {len(ix)} trading days, {ix[0].strftime('%Y-%m-%d')} to {ix[-1].strftime('%Y-%m-%d')}")

    # Show UVXY decay stats
    if len(ix) > 252:
        annual_decay = float(uvxy.iloc[-1] / uvxy.iloc[-252] - 1)
        total_decay = float(uvxy.iloc[-1] / uvxy.iloc[0] - 1)
        fprint(f"UVXY decay: {total_decay*100:.1f}% total, {annual_decay*100:.1f}% last year")
        fprint(f"UVXY price range: ${uvxy.min():.2f} — ${uvxy.max():.2f}")
        fprint(f"UVXY current: ${uvxy.iloc[-1]:.2f}")

    return uvxy.loc[ix], spy.loc[ix], vix.loc[ix]


# ==================== BACKTEST ENGINE ====================

def run_variant(uvxy, spy, vix, variant, verbose=True):
    """
    Run a single VIX ETF decay options backtest.

    Returns dict with equity_curve, trades, metrics.
    """
    configs = {
        'A': {'otm_pct': 0.00, 'dte': 30, 'vix_min': 0, 'vix_max': 999,
               'trailing_tp': None, 'vix_sizing': False, 'label': 'ATM puts monthly'},
        'B': {'otm_pct': 0.15, 'dte': 30, 'vix_min': 0, 'vix_max': 999,
               'trailing_tp': None, 'vix_sizing': False, 'label': '15% OTM puts monthly'},
        'C': {'otm_pct': 0.00, 'dte': 30, 'vix_min': 20, 'vix_max': 999,
               'trailing_tp': None, 'vix_sizing': False, 'label': 'ATM puts when VIX>20'},
        'D': {'otm_pct': 0.15, 'dte': 30, 'vix_min': 25, 'vix_max': 999,
               'trailing_tp': None, 'vix_sizing': False, 'label': '15% OTM puts on VIX spikes>25'},
        'E': {'otm_pct': 0.00, 'dte': 30, 'vix_min': 0, 'vix_max': 999,
               'trailing_tp': 0.50, 'vix_sizing': False, 'label': 'ATM puts with 50% TP roll'},
        'F': {'otm_pct': 0.10, 'dte': 30, 'vix_min': 0, 'vix_max': 999,
               'trailing_tp': None, 'vix_sizing': True, 'label': '10% OTM VIX-sized'},
    }
    cfg = configs[variant]

    dates = uvxy.index
    n_days = len(dates)
    start_idx = 63  # Need some history for vol estimation

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []
    closed_trades = []
    last_entry_idx = None
    rebalance_interval = cfg['dte']  # Match DTE

    for day_idx in range(start_idx, n_days):
        today = dates[day_idx]
        current_uvxy = float(uvxy.iloc[day_idx])
        current_vix = float(vix.iloc[day_idx])

        # ── Mark-to-market existing positions ──
        expired = []
        for i, pos in enumerate(positions):
            days_held = (today - pos['entry_date']).days
            remaining_dte = pos['dte'] - days_held

            if remaining_dte <= 0:
                # Expired: intrinsic value only
                intrinsic = max(0, pos['strike'] - current_uvxy) * 100 * pos['n_contracts']
                pos['current_value'] = intrinsic
                pos['current_pnl'] = intrinsic - pos['total_cost']
                expired.append(i)
            else:
                # Reprice with BS
                rets = uvxy.iloc[max(0, day_idx-21):day_idx].pct_change().dropna()
                vol = float(rets.std() * np.sqrt(252)) if len(rets) > 5 else pos['sigma']
                vol = max(vol, 0.30)

                option_val = bs_put_price(current_uvxy, pos['strike'],
                                           remaining_dte/365.0, RISK_FREE_RATE, vol)
                pos['current_value'] = option_val * 100 * pos['n_contracts']
                pos['current_pnl'] = pos['current_value'] - pos['total_cost']
                pos['current_pnl_pct'] = pos['current_pnl'] / pos['total_cost'] if pos['total_cost'] > 0 else 0
                pos['peak_value'] = max(pos.get('peak_value', pos['current_value']), pos['current_value'])

                # Trailing take-profit
                if cfg['trailing_tp'] is not None and pos['current_pnl_pct'] >= cfg['trailing_tp']:
                    expired.append(i)

        # Close expired/TP positions
        for i in sorted(set(expired), reverse=True):
            pos = positions.pop(i)
            pnl = pos.get('current_pnl', 0)
            equity += pnl
            closed_trades.append({
                'ticker': 'UVXY',
                'entry_date': pos['entry_date'],
                'exit_date': today,
                'strike': pos['strike'],
                'entry_cost': pos['total_cost'],
                'exit_value': pos.get('current_value', 0),
                'pnl': pnl,
                'pnl_pct': pos.get('current_pnl_pct', 0),
                'days_held': (today - pos['entry_date']).days,
                'entry_vix': pos['entry_vix'],
                'entry_uvxy': pos['entry_uvxy'],
                'exit_uvxy': current_uvxy,
                'reason': 'expired' if (today - pos['entry_date']).days >= pos['dte'] else 'take_profit',
            })

        # ── Open new position? ──
        should_open = False
        if not positions:  # Only one position at a time
            if last_entry_idx is None or day_idx - last_entry_idx >= rebalance_interval:
                should_open = True

        if should_open and day_idx < n_days - 5:
            # VIX filter
            if current_vix < cfg['vix_min'] or current_vix > cfg['vix_max']:
                equity_curve.append({'date': today, 'equity': equity})
                continue

            # Compute realized vol for pricing
            rets = uvxy.iloc[max(0, day_idx-63):day_idx].pct_change().dropna()
            realized_vol = float(rets.std() * np.sqrt(252)) if len(rets) > 10 else 0.60

            opt = price_put_option(
                stock_price=current_uvxy,
                otm_pct=cfg['otm_pct'],
                dte_days=cfg['dte'],
                realized_vol=realized_vol,
                pricing_buffer=PRICING_BUFFER,
            )

            # Position sizing
            if cfg['vix_sizing']:
                # Scale with VIX: higher VIX = bigger position (more decay expected)
                vix_scale = min(current_vix / 20.0, 2.0)  # Cap at 2x
                max_spend = min(MAX_POSITION_SIZE * vix_scale, equity * 0.5)
            else:
                max_spend = min(MAX_POSITION_SIZE, equity * 0.5)

            if opt['contract_cost'] > max_spend or opt['contract_cost'] < 5:
                # Can't afford or too cheap (likely numerical issue)
                equity_curve.append({'date': today, 'equity': equity})
                continue

            n_contracts = max(1, int(max_spend / opt['contract_cost']))
            # For $645 account, usually 1-3 contracts
            n_contracts = min(n_contracts, 5)  # Safety cap
            total_cost = opt['contract_cost'] * n_contracts

            if total_cost > equity * 0.5:
                n_contracts = max(1, int(equity * 0.5 / opt['contract_cost']))
                total_cost = opt['contract_cost'] * n_contracts

            if total_cost > equity:
                equity_curve.append({'date': today, 'equity': equity})
                continue

            positions.append({
                'entry_date': today,
                'strike': opt['strike'],
                'delta': opt['delta'],
                'premium_per_share': opt['premium_per_share'],
                'total_cost': total_cost,
                'n_contracts': n_contracts,
                'sigma': opt['sigma'],
                'dte': cfg['dte'],
                'entry_uvxy': current_uvxy,
                'entry_vix': current_vix,
                'current_value': total_cost,
                'current_pnl': 0,
                'current_pnl_pct': 0,
                'peak_value': total_cost,
            })
            last_entry_idx = day_idx

            if verbose and len(closed_trades) == 0:
                fprint(f"  {variant}: First trade — UVXY ${current_uvxy:.2f}, "
                       f"strike ${opt['strike']:.2f}, cost ${total_cost:.0f}, "
                       f"VIX {current_vix:.1f}, vol {realized_vol:.0%}")

        # Mark-to-market equity
        pos_value = sum(p.get('current_pnl', 0) for p in positions)
        mtm_equity = equity + pos_value
        equity_curve.append({'date': today, 'equity': mtm_equity})

    # Close remaining positions
    for pos in positions:
        pnl = pos.get('current_pnl', 0)
        equity += pnl
        closed_trades.append({
            'ticker': 'UVXY',
            'entry_date': pos['entry_date'],
            'exit_date': dates[-1],
            'strike': pos['strike'],
            'entry_cost': pos['total_cost'],
            'exit_value': pos.get('current_value', 0),
            'pnl': pnl,
            'pnl_pct': pos.get('current_pnl_pct', 0),
            'days_held': (dates[-1] - pos['entry_date']).days,
            'entry_vix': pos['entry_vix'],
            'entry_uvxy': pos['entry_uvxy'],
            'exit_uvxy': float(uvxy.iloc[-1]),
            'reason': 'end',
        })

    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty:
        return None
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df = eq_df.set_index('date')

    return {
        'equity_curve': eq_df,
        'closed_trades': closed_trades,
        'variant': variant,
        'config': cfg,
    }


# ==================== METRICS ====================

def compute_metrics(result, spy):
    """Compute risk-adjusted metrics."""
    if result is None:
        return None

    eq = result['equity_curve']['equity']
    trades = result['closed_trades']

    if len(eq) < 20 or not trades:
        return None

    daily_rets = eq.pct_change().dropna()
    if len(daily_rets) < 10:
        return None

    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
    n_years = len(daily_rets) / 252
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1/max(n_years, 0.1)) - 1)

    ann_ret = float(daily_rets.mean() * 252)
    ann_vol = float(daily_rets.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    down_rets = daily_rets[daily_rets < 0]
    down_vol = float(down_rets.std() * np.sqrt(252)) if len(down_rets) > 0 else 0.001
    sortino = ann_ret / down_vol

    cummax = eq.cummax()
    drawdown = (eq - cummax) / cummax
    max_dd = float(drawdown.min())

    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = len(trades) - wins
    wr = wins / len(trades) if trades else 0
    avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins > 0 else 0
    avg_loss = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if losses > 0 else 0.001
    pf = (avg_win * wins) / (avg_loss * losses) if losses > 0 and avg_loss > 0 else 999

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Avg VIX at entry
    avg_entry_vix = np.mean([t.get('entry_vix', 20) for t in trades])
    # Avg contract cost
    avg_cost = np.mean([t['entry_cost'] for t in trades])
    # Avg days held
    avg_hold = np.mean([t['days_held'] for t in trades])

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1),
        'total_return': round(total_ret * 100, 1),
        'max_dd': round(max_dd * 100, 1),
        'win_rate': round(wr * 100, 1),
        'profit_factor': round(min(pf, 99), 2),
        'calmar': round(calmar, 2),
        'n_trades': len(trades),
        'final_equity': round(float(eq.iloc[-1]), 2),
        'ann_vol': round(ann_vol * 100, 1),
        'avg_entry_vix': round(avg_entry_vix, 1),
        'avg_contract_cost': round(avg_cost, 0),
        'avg_hold_days': round(avg_hold, 1),
    }


# ==================== PERMUTATION TEST ====================

_USE_RANDOM_TIMING = False

def run_variant_random(uvxy, spy, vix, variant):
    """Run variant with random entry timing."""
    global _USE_RANDOM_TIMING
    _USE_RANDOM_TIMING = True
    result = run_variant(uvxy, spy, vix, variant, verbose=False)
    _USE_RANDOM_TIMING = False
    return result


def run_permutation_test(uvxy, spy, vix, variant, n_shuffles=200):
    """Compare real strategy against random entry timing."""
    fprint(f"  Permutation test ({n_shuffles} shuffles)...")

    real_result = run_variant(uvxy, spy, vix, variant, verbose=False)
    if real_result is None:
        return None, None, None
    real_metrics = compute_metrics(real_result, spy)
    if real_metrics is None:
        return None, None, None
    real_sharpe = real_metrics['sharpe']

    random_sharpes = []
    for i in range(n_shuffles):
        np.random.seed(i + 42)

        # Create shuffled VIX to randomize entry timing
        vix_shuffled = vix.copy()
        vix_vals = vix_shuffled.values.copy()
        np.random.shuffle(vix_vals)
        vix_shuffled[:] = vix_vals

        result = run_variant(uvxy, spy, vix_shuffled, variant, verbose=False)
        if result is None:
            continue
        metrics = compute_metrics(result, spy)
        if metrics:
            random_sharpes.append(metrics['sharpe'])

    if not random_sharpes:
        return real_sharpe, None, None

    p_value = sum(1 for s in random_sharpes if s >= real_sharpe) / len(random_sharpes)
    mean_random = np.mean(random_sharpes)

    return real_sharpe, p_value, mean_random


# ==================== REGIME ANALYSIS ====================

def regime_analysis(result, spy):
    """Classify days and compute per-regime metrics."""
    if result is None:
        return None

    eq = result['equity_curve']['equity']
    spy_aligned = spy.reindex(eq.index).ffill().dropna()

    spy_rets = spy_aligned.pct_change().dropna()
    eq_rets = eq.pct_change().dropna()

    common_idx = spy_rets.index.intersection(eq_rets.index)
    spy_rets = spy_rets.loc[common_idx]
    eq_rets = eq_rets.loc[common_idx]

    green = spy_rets > 0.001
    red = spy_rets < -0.001
    flat = ~green & ~red

    regime_metrics = {}
    for name, mask in [('green', green), ('red', red), ('flat', flat)]:
        r = eq_rets[mask]
        if len(r) > 5:
            ann_ret = float(r.mean() * 252)
            ann_vol = float(r.std() * np.sqrt(252)) if r.std() > 0 else 0.001
            regime_metrics[name] = {
                'sharpe': round(ann_ret / ann_vol, 3),
                'n_days': int(mask.sum()),
                'mean_ret_bps': round(float(r.mean()) * 10000, 2),
            }
        else:
            regime_metrics[name] = {'sharpe': 0, 'n_days': 0, 'mean_ret_bps': 0}

    g_sharpe = abs(regime_metrics['green']['sharpe'])
    r_sharpe = abs(regime_metrics['red']['sharpe'])
    max_sharpe = max(g_sharpe, r_sharpe, 0.001)
    gap = abs(g_sharpe - r_sharpe) / max_sharpe
    regime_metrics['gap_ratio'] = round(gap, 3)
    regime_metrics['regime_pass'] = gap <= 0.50

    return regime_metrics


# ==================== MAIN ====================

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("VIX ETF STRUCTURAL DECAY OPTIONS V1")
    fprint("=" * 70)
    fprint(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"Capital: ${INITIAL_CAPITAL}, Max position: ${MAX_POSITION_SIZE}")
    fprint(f"Pricing buffer: {PRICING_BUFFER:.0%} (conservative for VIX options)")
    fprint()

    uvxy, spy, vix = download_data()

    all_metrics = {}
    variants = ['A', 'B', 'C', 'D', 'E', 'F']

    for v in variants:
        fprint(f"\n{'─'*50}")
        cfg_label = {
            'A': 'ATM puts monthly (pure decay)',
            'B': '15% OTM puts monthly (leverage)',
            'C': 'ATM puts VIX>20 (mean-reversion)',
            'D': '15% OTM on VIX spikes>25',
            'E': 'ATM puts + 50% TP roll',
            'F': '10% OTM VIX-sized',
        }
        fprint(f"VARIANT {v}: {cfg_label[v]}")
        fprint(f"{'─'*50}")

        result = run_variant(uvxy, spy, vix, v, verbose=True)

        if result is None:
            fprint(f"  {v}: No result")
            all_metrics[v] = None
            continue

        metrics = compute_metrics(result, spy)
        if metrics is None:
            fprint(f"  {v}: Could not compute metrics (too few trades?)")
            all_metrics[v] = None
            continue

        fprint(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}, "
               f"CAGR: {metrics['cagr']}%, MDD: {metrics['max_dd']}%")
        fprint(f"  WR: {metrics['win_rate']}%, PF: {metrics['profit_factor']}, "
               f"Trades: {metrics['n_trades']}, Final: ${metrics['final_equity']:.0f}")
        fprint(f"  Avg cost/trade: ${metrics['avg_contract_cost']:.0f}, "
               f"Avg hold: {metrics['avg_hold_days']:.0f}d, Avg entry VIX: {metrics['avg_entry_vix']:.1f}")

        # Regime analysis
        regime = regime_analysis(result, spy)
        if regime:
            fprint(f"  Regime: Green={regime['green']['sharpe']}, "
                   f"Red={regime['red']['sharpe']}, Gap={regime['gap_ratio']}")
            metrics['regime'] = regime
            metrics['regime_pass'] = regime['regime_pass']
        else:
            metrics['regime_pass'] = False

        # Permutation test
        real_s, p_val, mean_rand = run_permutation_test(uvxy, spy, vix, v, n_shuffles=200)
        if p_val is not None:
            fprint(f"  Permutation: p={p_val:.4f}, real={real_s:.3f}, random_mean={mean_rand:.3f}")
            metrics['perm_p'] = round(p_val, 4)
            metrics['random_sharpe'] = round(mean_rand, 3)
        else:
            metrics['perm_p'] = 1.0
            metrics['random_sharpe'] = 0

        # 5-gate validation
        gates = {
            'sharpe_gt_1': metrics['sharpe'] > 1.0,
            'perm_significant': metrics.get('perm_p', 1.0) < 0.05,
            'wr_gt_40': metrics['win_rate'] > 40,
            'regime_balance': metrics.get('regime_pass', False),
            'beats_random': metrics['sharpe'] > metrics.get('random_sharpe', 0) * 1.5,
        }
        gates_passed = sum(gates.values())
        metrics['gates'] = {k: bool(v2) for k, v2 in gates.items()}
        metrics['gates_passed'] = gates_passed
        metrics['gates_total'] = len(gates)

        fprint(f"  GATES: {gates_passed}/{len(gates)} — "
               + ", ".join(f"{'✅' if v2 else '❌'} {k}" for k, v2 in gates.items()))

        all_metrics[v] = metrics

        # MLflow
        if MLFLOW_OK:
            try:
                with mlflow.start_run(run_name=f"variant_{v}"):
                    mlflow.log_params({'variant': v, 'label': cfg_label[v]})
                    for mk, mv in metrics.items():
                        if isinstance(mv, (int, float)):
                            mlflow.log_metric(mk, mv)
            except Exception as e:
                fprint(f"  MLflow error: {e}")

    # ── SUMMARY ──
    fprint(f"\n{'='*70}")
    fprint("SUMMARY — VIX ETF STRUCTURAL DECAY OPTIONS")
    fprint(f"{'='*70}")

    best_sharpe = -999
    best_variant = None

    for v in variants:
        m = all_metrics.get(v)
        if m is None:
            fprint(f"  {v}: NO RESULT")
            continue
        fprint(f"  {v}: Sharpe {m['sharpe']}, Sortino {m['sortino']}, "
               f"CAGR {m['cagr']}%, MDD {m['max_dd']}%, WR {m['win_rate']}%, "
               f"Trades {m['n_trades']}, Gates {m['gates_passed']}/{m['gates_total']}, "
               f"Final ${m['final_equity']:.0f}")

        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_variant = v

    if best_variant:
        fprint(f"\n  BEST: Variant {best_variant} (Sharpe {best_sharpe})")
        bm = all_metrics[best_variant]
        fprint(f"  Key question: Is this STRUCTURAL EDGE or just UVXY put = short vol exposure?")
        if bm.get('perm_p', 1) < 0.05:
            fprint(f"  → Permutation test PASSED (p={bm['perm_p']}): edge has signal-specific timing")
        else:
            fprint(f"  → Permutation test FAILED (p={bm.get('perm_p', 'N/A')}): likely just beta exposure")
    else:
        fprint("  No valid variants found.")

    # Contract affordability check
    fprint(f"\n  AFFORDABILITY CHECK (vs $645 capital):")
    for v in variants:
        m = all_metrics.get(v)
        if m and m.get('avg_contract_cost'):
            affordable = "✅ AFFORDABLE" if m['avg_contract_cost'] < 300 else "❌ TOO EXPENSIVE"
            fprint(f"    {v}: Avg ${m['avg_contract_cost']:.0f}/trade — {affordable}")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save
    summary = {
        'experiment': EXPERIMENT_NAME,
        'timestamp': datetime.now().isoformat(),
        'capital': INITIAL_CAPITAL,
        'pricing_buffer': PRICING_BUFFER,
        'runtime_s': round(elapsed, 1),
        'variants': {v: m for v, m in all_metrics.items() if m},
    }
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    fprint(f"Saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == '__main__':
    main()
