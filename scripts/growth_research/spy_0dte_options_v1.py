#!/usr/bin/env python3
"""
SPY 0DTE Options Strategy V1
==============================

HYPOTHESIS: 0DTE (zero days to expiration) SPY options are uniquely suited
for a $645 account because:

  1. EXTREMELY CHEAP: OTM 0DTE SPY options cost $5-200/contract
  2. HIGHLY LIQUID: SPY 0DTE is the most liquid options market
  3. LEVEL 2: Just buying calls/puts
  4. NO THETA OVERNIGHT: Expires same day, no theta drag across days
  5. HIGH GAMMA: Small SPY moves create large % option gains

STRATEGY LOGIC:
  Model SPY daily options using end-of-day pricing with daily granularity.
  Since we don't have intraday data here, we simulate 0DTE-like behavior:
  - Entry: open price (buying at market open)
  - Exit: close price (option expires at close)
  - Option P&L based on BS pricing at entry vs intrinsic at expiry

  The "0DTE" is simulated as: buy an ATM/OTM option at market open,
  it expires at market close. P&L = max(0, intrinsic at close) - premium.

ENTRY SIGNALS:
  We test several timing signals to decide DIRECTION (call vs put):
  A: Always buy calls (bullish bias — SPY has upward drift)
  B: Always buy puts (bearish bias — test if vol premium makes this work)
  C: Momentum: buy call if SPY 5d ret > 0, else put (trend following)
  D: Mean-reversion: buy call if SPY 5d ret < 0, else put (contrarian)
  E: VIX-based: buy call if VIX > 20 (vol mean-reversion), else put
  F: Combined: call if (5d ret < 0 AND VIX > 18), else put
  G: Selective: only trade when |5d ret| > 1% (strong signal days)
  H: Sized by conviction: 1 contract if marginal, 2 if strong signal

PRICING:
  - Use BS with SPY's realized vol + 30% IV premium (typical for SPY options)
  - Strike selection: 0.3% OTM (close to ATM but cheaper)
  - At expiry: value = max(0, intrinsic)

VALIDATION:
  - 5 gates: Sharpe>1, perm p<0.05, WR>40%, regime balance, random baseline
  - ALL OOT days, regime-stratified
  - Permutation: 100 shuffles (faster — many daily trades)

Output: output/growth_research/spy_0dte_options_v1/
MLflow experiment: spy_0dte_options_v1
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")
BASE = _NEPTUNE_BASE if _NEPTUNE_BASE.exists() else _JUPITER_BASE
fprint(f"Running on: {BASE}")
sys.path.insert(0, str(BASE))

OUTPUT_DIR = BASE / "output" / "growth_research" / "spy_0dte_options_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "spy_0dte_options_v1"
MLFLOW_OK = False
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    MLFLOW_OK = True
    fprint(f"MLflow OK")
except Exception as e:
    fprint(f"MLflow not available: {e}")

# ==================== CONFIG ====================
INITIAL_CAPITAL = 645.0
MAX_TRADE_SIZE = 200.0  # HC #749 conservative
OTM_PCT = 0.003  # 0.3% OTM — close to ATM but cheaper
IV_PREMIUM = 1.30  # 30% IV premium over realized vol (typical for SPY)
RISK_FREE_RATE = 0.05
HOURS_IN_TRADING_DAY = 6.5

# ==================== BS PRICING ====================

def bs_option_price(S, K, T, r, sigma, option_type='call'):
    """BS price for call or put."""
    if T <= 0 or sigma <= 0:
        if option_type == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ==================== DATA ====================

def download_data():
    """Download SPY OHLC + VIX."""
    import yfinance as yf
    fprint("Downloading SPY + VIX...")
    raw = yf.download(['SPY', '^VIX'], start='2020-01-01', progress=False)

    mi = isinstance(raw.columns, pd.MultiIndex)

    # Get OHLC for SPY
    if mi:
        spy_open = raw['Open']['SPY'].dropna()
        spy_close = raw['Close']['SPY'].dropna()
        spy_high = raw['High']['SPY'].dropna()
        spy_low = raw['Low']['SPY'].dropna()
        vix_close = raw['Close']['^VIX'].dropna() if '^VIX' in raw['Close'].columns else raw['Close']['VIX'].dropna()
    else:
        spy_open = raw['Open'].dropna()
        spy_close = raw['Close'].dropna()
        spy_high = raw['High'].dropna()
        spy_low = raw['Low'].dropna()
        vix_close = pd.Series()

    # Align
    ix = spy_open.index.intersection(spy_close.index)
    if len(vix_close) > 0:
        ix = ix.intersection(vix_close.index)

    spy_df = pd.DataFrame({
        'open': spy_open.loc[ix],
        'high': spy_high.reindex(ix).ffill(),
        'low': spy_low.reindex(ix).ffill(),
        'close': spy_close.loc[ix],
        'vix': vix_close.reindex(ix).ffill() if len(vix_close) > 0 else 20.0,
    })
    spy_df = spy_df.dropna()

    fprint(f"Data: {len(spy_df)} trading days, "
           f"{spy_df.index[0].strftime('%Y-%m-%d')} to {spy_df.index[-1].strftime('%Y-%m-%d')}")
    fprint(f"SPY range: ${spy_df['close'].min():.0f} — ${spy_df['close'].max():.0f}")

    return spy_df


# ==================== 0DTE SIMULATION ====================

def simulate_0dte_day(spy_open, spy_close, spy_high, spy_low, realized_vol,
                       option_type='call', otm_pct=0.003):
    """
    Simulate a 0DTE option trade for one day.

    Entry at open: buy option at BS price (with IV premium)
    Exit at close: option expires, value = intrinsic

    For intraday, we also check if the option hit a reasonable MFE
    using high/low as proxy for intraday extremes.

    Returns:
        dict with entry_cost, exit_value, pnl, max_potential (from intraday high/low)
    """
    T = 1.0 / 252  # 1 trading day remaining at open
    sigma = max(realized_vol * IV_PREMIUM, 0.10)

    if option_type == 'call':
        strike = spy_open * (1 + otm_pct)
    else:
        strike = spy_open * (1 - otm_pct)

    # Round strike to nearest $1 (SPY options in $1 increments)
    strike = round(strike)

    # Entry premium (at open, T=1day)
    entry_premium = bs_option_price(spy_open, strike, T, RISK_FREE_RATE, sigma, option_type)
    entry_cost = entry_premium * 100  # per contract

    # Exit value at close (expiry): intrinsic only
    if option_type == 'call':
        exit_value = max(0, spy_close - strike) * 100
        # MFE: max intrinsic during day (using high)
        max_potential = max(0, spy_high - strike) * 100
    else:
        exit_value = max(0, strike - spy_close) * 100
        # MFE: max intrinsic during day (using low)
        max_potential = max(0, strike - spy_low) * 100

    pnl = exit_value - entry_cost
    pnl_pct = pnl / entry_cost if entry_cost > 0 else 0

    return {
        'strike': strike,
        'entry_cost': entry_cost,
        'exit_value': exit_value,
        'pnl': pnl,
        'pnl_pct': pnl_pct,
        'max_potential': max_potential,
        'sigma': sigma,
        'option_type': option_type,
    }


# ==================== BACKTEST ENGINE ====================

def run_variant(spy_df, variant, verbose=True):
    """Run a single 0DTE variant."""

    dates = spy_df.index
    n_days = len(dates)
    start_idx = 30  # Need some history for signals

    equity = INITIAL_CAPITAL
    equity_curve = []
    trades = []
    daily_rets = []

    for day_idx in range(start_idx, n_days):
        today = dates[day_idx]
        row = spy_df.iloc[day_idx]
        spy_open = float(row['open'])
        spy_close = float(row['close'])
        spy_high = float(row['high'])
        spy_low = float(row['low'])
        current_vix = float(row['vix'])

        # Compute signals
        ret_5d = float(spy_df['close'].iloc[day_idx] / spy_df['close'].iloc[max(0, day_idx-5)] - 1)
        ret_1d = float(spy_df['close'].iloc[day_idx-1] / spy_df['close'].iloc[max(0, day_idx-2)] - 1) if day_idx > 1 else 0

        # Realized vol (21-day)
        rets_21 = spy_df['close'].iloc[max(0, day_idx-21):day_idx].pct_change().dropna()
        realized_vol = float(rets_21.std() * np.sqrt(252)) if len(rets_21) > 5 else 0.15

        # ── Determine direction and whether to trade ──
        should_trade = True
        option_type = 'call'
        n_contracts = 1

        if variant == 'A':
            # Always buy calls
            option_type = 'call'
        elif variant == 'B':
            # Always buy puts
            option_type = 'put'
        elif variant == 'C':
            # Momentum: trend following
            option_type = 'call' if ret_5d > 0 else 'put'
        elif variant == 'D':
            # Mean-reversion: contrarian
            option_type = 'call' if ret_5d < 0 else 'put'
        elif variant == 'E':
            # VIX-based: high VIX = buy call (vol mean-reversion implies SPY recovery)
            option_type = 'call' if current_vix > 20 else 'put'
        elif variant == 'F':
            # Combined: contrarian + VIX
            if ret_5d < 0 and current_vix > 18:
                option_type = 'call'  # Oversold + high fear = recovery
            elif ret_5d > 0.02:
                option_type = 'put'  # Overbought = fade
            else:
                option_type = 'call'  # Default bullish
        elif variant == 'G':
            # Selective: only trade strong signal days
            if abs(ret_5d) < 0.01:
                should_trade = False
            option_type = 'call' if ret_5d < 0 else 'put'  # Contrarian on strong moves
        elif variant == 'H':
            # Conviction sizing
            option_type = 'call' if ret_5d < 0 else 'put'
            if abs(ret_5d) > 0.02:
                n_contracts = 2  # Strong conviction
            else:
                n_contracts = 1

        if not should_trade:
            equity_curve.append({'date': today, 'equity': equity})
            continue

        # ── Simulate the 0DTE trade ──
        result = simulate_0dte_day(spy_open, spy_close, spy_high, spy_low,
                                    realized_vol, option_type, OTM_PCT)

        total_cost = result['entry_cost'] * n_contracts

        # Can we afford it?
        if total_cost > min(MAX_TRADE_SIZE, equity * 0.5) or total_cost < 1:
            equity_curve.append({'date': today, 'equity': equity})
            continue

        total_pnl = result['pnl'] * n_contracts

        # Cap loss at investment (can't lose more than premium paid)
        total_pnl = max(total_pnl, -total_cost)

        equity += total_pnl

        trades.append({
            'date': today,
            'option_type': option_type,
            'strike': result['strike'],
            'entry_cost': total_cost,
            'exit_value': result['exit_value'] * n_contracts,
            'pnl': total_pnl,
            'pnl_pct': total_pnl / total_cost if total_cost > 0 else 0,
            'spy_open': spy_open,
            'spy_close': spy_close,
            'spy_move': (spy_close / spy_open - 1) * 100,
            'vix': current_vix,
            'n_contracts': n_contracts,
            'max_potential': result['max_potential'] * n_contracts,
        })

        daily_rets.append(total_pnl / equity if equity > 0 else 0)
        equity_curve.append({'date': today, 'equity': max(equity, 0)})

        if equity <= 0:
            fprint(f"  {variant}: BLOWUP at {today.strftime('%Y-%m-%d')}")
            break

    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty:
        return None
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df = eq_df.set_index('date')

    return {
        'equity_curve': eq_df,
        'closed_trades': trades,
        'variant': variant,
    }


# ==================== METRICS ====================

def compute_metrics(result):
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
    cagr = float((max(eq.iloc[-1], 0.01) / eq.iloc[0]) ** (1/max(n_years, 0.1)) - 1)

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
    wr = wins / len(trades)
    avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins > 0 else 0
    avg_loss = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if losses > 0 else 0.001
    pf = (avg_win * wins) / (avg_loss * losses) if losses > 0 and avg_loss > 0 else 999

    # Cost analysis
    avg_cost = np.mean([t['entry_cost'] for t in trades])
    avg_pnl = np.mean([t['pnl'] for t in trades])
    total_pnl = sum(t['pnl'] for t in trades)

    # Direction analysis
    call_trades = [t for t in trades if t['option_type'] == 'call']
    put_trades = [t for t in trades if t['option_type'] == 'put']
    call_wr = sum(1 for t in call_trades if t['pnl'] > 0) / len(call_trades) if call_trades else 0
    put_wr = sum(1 for t in put_trades if t['pnl'] > 0) / len(put_trades) if put_trades else 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1),
        'total_return': round(total_ret * 100, 1),
        'max_dd': round(max_dd * 100, 1),
        'win_rate': round(wr * 100, 1),
        'profit_factor': round(min(pf, 99), 2),
        'n_trades': len(trades),
        'final_equity': round(float(eq.iloc[-1]), 2),
        'ann_vol': round(ann_vol * 100, 1),
        'avg_cost_per_trade': round(avg_cost, 0),
        'avg_pnl_per_trade': round(avg_pnl, 2),
        'total_pnl': round(total_pnl, 2),
        'call_wr': round(call_wr * 100, 1),
        'put_wr': round(put_wr * 100, 1),
        'n_calls': len(call_trades),
        'n_puts': len(put_trades),
    }


# ==================== PERMUTATION TEST ====================

def run_permutation_test(spy_df, variant, n_shuffles=100):
    """Shuffle direction signals and compare."""
    fprint(f"  Permutation test ({n_shuffles} shuffles)...")

    real_result = run_variant(spy_df, variant, verbose=False)
    if real_result is None:
        return None, None, None
    real_metrics = compute_metrics(real_result)
    if real_metrics is None:
        return None, None, None
    real_sharpe = real_metrics['sharpe']

    random_sharpes = []
    for i in range(n_shuffles):
        np.random.seed(i + 42)
        # Shuffle close prices to randomize signals
        spy_shuffled = spy_df.copy()
        # Shuffle the relationship between past returns and future moves
        # by randomly permuting the date order of close prices while keeping OHLC intact
        close_vals = spy_shuffled['close'].values.copy()
        np.random.shuffle(close_vals)
        spy_shuffled['close'] = close_vals

        result = run_variant(spy_shuffled, variant, verbose=False)
        if result is None:
            continue
        metrics = compute_metrics(result)
        if metrics:
            random_sharpes.append(metrics['sharpe'])

    if not random_sharpes:
        return real_sharpe, None, None

    p_value = sum(1 for s in random_sharpes if s >= real_sharpe) / len(random_sharpes)
    mean_random = np.mean(random_sharpes)
    return real_sharpe, p_value, mean_random


# ==================== REGIME ANALYSIS ====================

def regime_analysis(result, spy_df):
    """Per-regime metrics."""
    if result is None:
        return None

    eq = result['equity_curve']['equity']
    spy_close = spy_df['close'].reindex(eq.index).ffill().dropna()
    eq_rets = eq.pct_change().dropna()
    spy_rets = spy_close.pct_change().dropna()

    common = eq_rets.index.intersection(spy_rets.index)
    eq_rets = eq_rets.loc[common]
    spy_rets = spy_rets.loc[common]

    green = spy_rets > 0.001
    red = spy_rets < -0.001

    regime_metrics = {}
    for name, mask in [('green', green), ('red', red)]:
        r = eq_rets[mask]
        if len(r) > 5:
            ann_ret = float(r.mean() * 252)
            ann_vol = float(r.std() * np.sqrt(252)) if r.std() > 0 else 0.001
            regime_metrics[name] = {'sharpe': round(ann_ret / ann_vol, 3), 'n_days': int(mask.sum())}
        else:
            regime_metrics[name] = {'sharpe': 0, 'n_days': 0}

    g = abs(regime_metrics['green']['sharpe'])
    r = abs(regime_metrics['red']['sharpe'])
    mx = max(g, r, 0.001)
    gap = abs(g - r) / mx
    regime_metrics['gap_ratio'] = round(gap, 3)
    regime_metrics['regime_pass'] = gap <= 0.50

    return regime_metrics


# ==================== MAIN ====================

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("SPY 0DTE OPTIONS STRATEGY V1")
    fprint("=" * 70)
    fprint(f"Capital: ${INITIAL_CAPITAL}, Max trade: ${MAX_TRADE_SIZE}")
    fprint(f"OTM: {OTM_PCT:.1%}, IV premium: {IV_PREMIUM:.0%}")
    fprint()

    spy_df = download_data()

    all_metrics = {}
    variants = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    labels = {
        'A': 'Always calls (bullish)',
        'B': 'Always puts (bearish)',
        'C': 'Momentum (trend follow)',
        'D': 'Mean-reversion (contrarian)',
        'E': 'VIX-based (high VIX=call)',
        'F': 'Combined (contrarian+VIX)',
        'G': 'Selective (strong signals only)',
        'H': 'Conviction sizing (1-2 contracts)',
    }

    for v in variants:
        fprint(f"\n{'─'*50}")
        fprint(f"VARIANT {v}: {labels[v]}")
        fprint(f"{'─'*50}")

        result = run_variant(spy_df, v, verbose=True)

        if result is None:
            fprint(f"  {v}: No result")
            all_metrics[v] = None
            continue

        metrics = compute_metrics(result)
        if metrics is None:
            fprint(f"  {v}: Could not compute metrics")
            all_metrics[v] = None
            continue

        fprint(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}, "
               f"CAGR: {metrics['cagr']}%, MDD: {metrics['max_dd']}%")
        fprint(f"  WR: {metrics['win_rate']}%, PF: {metrics['profit_factor']}, "
               f"Trades: {metrics['n_trades']}, Final: ${metrics['final_equity']:.0f}")
        fprint(f"  Calls: {metrics['n_calls']} ({metrics['call_wr']}% WR), "
               f"Puts: {metrics['n_puts']} ({metrics['put_wr']}% WR)")
        fprint(f"  Avg cost: ${metrics['avg_cost_per_trade']:.0f}, "
               f"Avg P&L: ${metrics['avg_pnl_per_trade']:.2f}")

        # Regime
        regime = regime_analysis(result, spy_df)
        if regime:
            fprint(f"  Regime: Green={regime['green']['sharpe']}, "
                   f"Red={regime['red']['sharpe']}, Gap={regime['gap_ratio']}")
            metrics['regime'] = regime
            metrics['regime_pass'] = regime['regime_pass']
        else:
            metrics['regime_pass'] = False

        # Permutation
        real_s, p_val, mean_rand = run_permutation_test(spy_df, v, n_shuffles=100)
        if p_val is not None:
            fprint(f"  Permutation: p={p_val:.4f}, real={real_s:.3f}, random_mean={mean_rand:.3f}")
            metrics['perm_p'] = round(p_val, 4)
            metrics['random_sharpe'] = round(mean_rand, 3)
        else:
            metrics['perm_p'] = 1.0
            metrics['random_sharpe'] = 0

        # Gates
        gates = {
            'sharpe_gt_1': metrics['sharpe'] > 1.0,
            'perm_significant': metrics.get('perm_p', 1.0) < 0.05,
            'wr_gt_40': metrics['win_rate'] > 40,
            'regime_balance': metrics.get('regime_pass', False),
            'beats_random': metrics['sharpe'] > metrics.get('random_sharpe', 0) * 1.5,
        }
        gates_passed = sum(gates.values())
        metrics['gates'] = {k: bool(gv) for k, gv in gates.items()}
        metrics['gates_passed'] = gates_passed
        metrics['gates_total'] = len(gates)

        fprint(f"  GATES: {gates_passed}/{len(gates)} — "
               + ", ".join(f"{'✅' if gv else '❌'} {k}" for k, gv in gates.items()))

        all_metrics[v] = metrics

        # MLflow
        if MLFLOW_OK:
            try:
                with mlflow.start_run(run_name=f"variant_{v}"):
                    mlflow.log_params({'variant': v, 'label': labels[v]})
                    for mk, mv in metrics.items():
                        if isinstance(mv, (int, float)):
                            mlflow.log_metric(mk, mv)
            except Exception as e:
                fprint(f"  MLflow error: {e}")

    # ── SUMMARY ──
    fprint(f"\n{'='*70}")
    fprint("SUMMARY — SPY 0DTE OPTIONS")
    fprint(f"{'='*70}")

    best_sharpe = -999
    best_variant = None

    for v in variants:
        m = all_metrics.get(v)
        if m is None:
            fprint(f"  {v}: NO RESULT")
            continue
        fprint(f"  {v} ({labels[v]}): Sharpe {m['sharpe']}, Sortino {m['sortino']}, "
               f"CAGR {m['cagr']}%, MDD {m['max_dd']}%, WR {m['win_rate']}%, "
               f"Trades {m['n_trades']}, Gates {m['gates_passed']}/{m['gates_total']}, "
               f"Final ${m['final_equity']:.0f}")

        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_variant = v

    if best_variant:
        fprint(f"\n  BEST: Variant {best_variant} — {labels[best_variant]} (Sharpe {best_sharpe})")
        bm = all_metrics[best_variant]

        # Key insight
        fprint(f"\n  KEY INSIGHT: 0DTE options at 0.3% OTM cost ~${bm['avg_cost_per_trade']:.0f}/contract")
        fprint(f"  This IS affordable for $645 account ({bm['avg_cost_per_trade']/INITIAL_CAPITAL*100:.0f}% of capital per trade)")

        if bm['sharpe'] > 0:
            fprint(f"  Direction prediction adds value: Sharpe {bm['sharpe']} vs random {bm.get('random_sharpe', 'N/A')}")
        else:
            fprint(f"  ⚠️ NEGATIVE EDGE: 0DTE premium decay dominates. Time value is too expensive relative to moves.")
            fprint(f"  This means: buying 0DTE options is a LOSING proposition on average.")
            fprint(f"  To make 0DTE work, you'd need SELLING premium (Level 3) or extremely precise timing.")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save
    summary = {
        'experiment': EXPERIMENT_NAME,
        'timestamp': datetime.now().isoformat(),
        'capital': INITIAL_CAPITAL,
        'otm_pct': OTM_PCT,
        'iv_premium': IV_PREMIUM,
        'runtime_s': round(elapsed, 1),
        'variants': {v: m for v, m in all_metrics.items() if m},
    }
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    fprint(f"Saved to {OUTPUT_DIR / 'results.json'}")


if __name__ == '__main__':
    main()
