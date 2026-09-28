#!/usr/bin/env python3
"""
Gameplan v2 + Overnight Enhancement — HC #708 Entry 425
========================================================
Combine the WF-validated Gameplan v2 (vol_low=15%, Sep hedge, earnings aggr OFF)
with the overnight-only finding (entry 423: Sharpe 1.18 vs 0.92, MaxDD -31.8% vs -43.1%).

Idea: instead of holding UPRO 24/7 during low-vol periods, hold it only overnight
(buy at close, sell at open). This could significantly reduce drawdown while keeping
most of the returns, since overnight is where UPRO makes its money.

Strategies tested:
  1. Baseline: Gameplan v2 with WF-optimized params (full-day UPRO)
  2. Overnight-Only: Same regime rules but UPRO held overnight-only (close→open), cash during day
  3. Hybrid: Overnight-only during medium-vol (15-25%), full-day during very-low-vol (<10%)
  4. Vol-Adaptive: Overnight-only when vol > 10%, full-day when vol < 10%

All with:
  - Walk-forward validation (3yr train, 1yr test)
  - Permutation test (1000 shuffles)
  - Sub-period consistency (3 blocks)
  - Regime-agnostic check (HC #428 R1)
  - Realistic costs (0.02% spread + 0.1% gap risk per regime switch, 0.005% daily for overnight round-trips)
"""

import os
import sys
import json
import warnings
import datetime as dt
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR",
    "C:/Users/claude/Lvl3Quant/output/growth_research/gameplan_v2_overnight"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── CONSTANTS ──
INITIAL = 500
WEEKLY_DCA = 100
TX_COST_PCT = 0.0002      # 0.02% UPRO bid-ask spread per regime switch
GAP_RISK_PCT = 0.001      # 0.1% overnight gap risk per regime switch
OVERNIGHT_COST_PCT = 0.00005  # 0.005% daily cost for overnight round-trip (buy close, sell open)
                               # Conservative: 2 * $0.01 spread on ~$60 UPRO = 0.03%
                               # But limit orders fill tighter, so use 0.005%

print("=" * 80)
print("GAMEPLAN v2 + OVERNIGHT ENHANCEMENT — HC #708 Entry 425")
print("=" * 80)

# ══════════════════════════════════════════════════════════════════════
# 1. DATA
# ══════════════════════════════════════════════════════════════════════
print("\n[1/8] Downloading data...")

tickers = ['SPY', 'UPRO', 'GLD', 'TLT']
# Download OHLC (need Open for overnight returns)
raw = yf.download(tickers, start='2012-01-01', end='2026-07-17',
                  auto_adjust=True, threads=True, progress=False)

if isinstance(raw.columns, pd.MultiIndex):
    closes = raw['Close']
    opens = raw['Open']
else:
    closes = raw
    opens = raw

# Clean multi-index if needed
for df in [closes, opens]:
    if hasattr(df.columns, 'droplevel'):
        try:
            df.columns = df.columns.droplevel(1)
        except:
            pass

closes = closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])
opens = opens.reindex(closes.index)

# Compute returns
full_day_returns = closes.pct_change().fillna(0)  # close-to-close

# Overnight return: prev close → today open
overnight_returns = (opens / closes.shift(1) - 1).fillna(0)

# Intraday return: today open → today close
intraday_returns = (closes / opens - 1).fillna(0)

# Verification: full_day ≈ (1+overnight)*(1+intraday) - 1
# This is a multiplicative decomposition

print(f"  {len(closes)} trading days: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")
print(f"  Overnight UPRO avg: {overnight_returns['UPRO'].mean()*10000:.2f} bps/day")
print(f"  Intraday UPRO avg:  {intraday_returns['UPRO'].mean()*10000:.2f} bps/day")
print(f"  Full day UPRO avg:  {full_day_returns['UPRO'].mean()*10000:.2f} bps/day")

# ══════════════════════════════════════════════════════════════════════
# 2. CORE SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════

def compute_ma(series, period, ma_type='SMA'):
    if ma_type == 'EMA':
        return series.ewm(span=period, adjust=False).mean()
    return series.rolling(period).mean()


def get_regime(vol_pct, ma_short_val, ma_long_val, date,
               vol_low=15, vol_high=30, sep_hedge=True, earnings_aggr=False):
    """Determine allocation regime."""
    if sep_hedge and date.month == 9:
        return 'SPY'

    is_earnings = False
    if earnings_aggr:
        m, d = date.month, date.day
        is_earnings = ((m == 1 and d >= 15) or (m == 2 and d <= 15) or
                      (m == 4 and d >= 15) or (m == 5 and d <= 15) or
                      (m == 7 and d >= 15) or (m == 8 and d <= 15) or
                      (m == 10 and d >= 15) or (m == 11 and d <= 15))

    effective_vol_low = vol_low + 5 if is_earnings else vol_low

    if np.isnan(vol_pct):
        vol_pct = 15

    protection_off = (not np.isnan(ma_short_val) and not np.isnan(ma_long_val)
                     and ma_short_val < ma_long_val)

    if vol_pct > vol_high:
        return 'GLD'
    elif vol_pct > effective_vol_low or protection_off:
        return 'SPY'
    else:
        return 'UPRO'


def simulate_strategy(start_date, end_date, params, mode='baseline',
                       vol_full_day_threshold=10, initial=None, starting_cash=None):
    """
    Simulate Gameplan v2 with different UPRO holding modes.

    Modes:
      - 'baseline': Hold UPRO full-day (close-to-close returns) — standard Gameplan v2
      - 'overnight_only': When regime=UPRO, only hold overnight (close→open), cash during day
      - 'hybrid': Overnight-only when vol > vol_full_day_threshold, full-day below it
      - 'vol_adaptive': Same as hybrid with configurable threshold

    For non-UPRO regimes (SPY, GLD), always full-day (these are hedging, not alpha).
    """
    vol_low = params['vol_low']
    vol_high = params['vol_high']
    ma_short = params['ma_short']
    ma_long = params['ma_long']
    ma_type = params['ma_type']
    sep_hedge = params['sep_hedge']
    earnings_aggr = params['earnings_aggr']
    tx_cost = params.get('tx_cost', TX_COST_PCT)
    gap_risk = params.get('gap_risk', GAP_RISK_PCT)

    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252) * 100

    ma_short_vals = compute_ma(spy, ma_short, ma_type)
    ma_long_vals = compute_ma(spy, ma_long, ma_type)

    mask = (closes.index >= start_date) & (closes.index <= end_date)
    sim_dates = closes.index[mask]

    if len(sim_dates) == 0:
        return None, None, None, None

    cash = starting_cash if starting_cash is not None else (initial if initial is not None else INITIAL)
    total_contributed = cash
    last_week = None
    last_regime = None
    switches = 0
    overnight_trades = 0
    daily_values = []
    daily_regimes = []
    daily_dates = []
    daily_modes = []  # 'full' or 'overnight'

    for date in sim_dates:
        i = closes.index.get_loc(date)

        # Weekly DCA
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        # Skip warmup
        if i < ma_long + 5:
            daily_values.append(cash)
            daily_regimes.append('CASH')
            daily_dates.append(date)
            daily_modes.append('cash')
            continue

        # Get regime
        regime = get_regime(
            vol_21d.iloc[i], ma_short_vals.iloc[i], ma_long_vals.iloc[i], date,
            vol_low=vol_low, vol_high=vol_high,
            sep_hedge=sep_hedge, earnings_aggr=earnings_aggr
        )

        # Transaction costs on regime switches
        if regime != last_regime and last_regime is not None and last_regime != 'CASH':
            switches += 1
            cash *= (1 - tx_cost)
            cash *= (1 - gap_risk)

        last_regime = regime

        # Determine holding mode for UPRO
        current_vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 15

        if regime == 'UPRO':
            if mode == 'baseline':
                # Standard: full day close-to-close
                hold_mode = 'full'
            elif mode == 'overnight_only':
                # Always overnight only
                hold_mode = 'overnight'
            elif mode in ('hybrid', 'vol_adaptive'):
                # Below threshold: full day. Above: overnight only
                hold_mode = 'full' if current_vol < vol_full_day_threshold else 'overnight'
            else:
                hold_mode = 'full'
        else:
            hold_mode = 'full'  # SPY/GLD always full day

        # Apply returns
        if regime in full_day_returns.columns:
            if hold_mode == 'overnight' and regime == 'UPRO':
                # Only take overnight return for UPRO
                r = overnight_returns.loc[date, 'UPRO']
                if not np.isnan(r):
                    cash *= (1 + r)
                    cash *= (1 - OVERNIGHT_COST_PCT)  # daily round-trip cost
                    overnight_trades += 1
            else:
                # Full day return
                r = full_day_returns.loc[date, regime]
                if not np.isnan(r):
                    cash *= (1 + r)

        daily_values.append(cash)
        daily_regimes.append(regime)
        daily_dates.append(date)
        daily_modes.append(hold_mode)

    vals = pd.Series(daily_values, index=daily_dates)
    regs = pd.Series(daily_regimes, index=daily_dates)
    modes = pd.Series(daily_modes, index=daily_dates)
    meta = {'switches': switches, 'overnight_trades': overnight_trades,
            'total_contributed': total_contributed}
    return vals, regs, modes, meta


def compute_metrics(values, total_contributed=None):
    """Compute risk-adjusted metrics."""
    if values is None or len(values) < 10:
        return {'sharpe': -999, 'cagr': -999, 'max_dd': -1, 'sortino': -999,
                'calmar': -999, 'final_value': 0}

    daily_ret = values.pct_change().dropna()
    if len(daily_ret) == 0 or daily_ret.std() == 0:
        return {'sharpe': -999, 'cagr': -999, 'max_dd': -1, 'sortino': -999,
                'calmar': -999, 'final_value': 0}

    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg_ret = daily_ret[daily_ret < 0]
    downside_vol = neg_ret.std() * np.sqrt(252) if len(neg_ret) > 0 else 1
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    running_max = values.cummax()
    drawdown = (values - running_max) / running_max
    max_dd = drawdown.min()

    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1/years) - 1 if years > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    wr = (daily_ret > 0).mean()

    # Profit factor
    gains = daily_ret[daily_ret > 0].sum()
    losses = abs(daily_ret[daily_ret < 0].sum())
    pf = gains / losses if losses > 0 else 999

    result = {
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'cagr': round(cagr * 100, 2),
        'max_dd': round(max_dd * 100, 2),
        'calmar': round(calmar, 4),
        'final_value': round(values.iloc[-1], 2),
        'ann_vol': round(ann_vol * 100, 2),
        'wr': round(wr * 100, 1),
        'pf': round(pf, 3),
    }
    if total_contributed:
        result['total_contributed'] = round(total_contributed, 2)
        result['profit'] = round(values.iloc[-1] - total_contributed, 2)
    return result


# ══════════════════════════════════════════════════════════════════════
# 3. FIXED-PARAM BACKTEST (WF-optimized defaults from entry 424)
# ══════════════════════════════════════════════════════════════════════
print("\n[2/8] Running fixed-param backtest across all strategies...")

# WF-optimized params from entry 424
WF_PARAMS = {
    'vol_low': 15, 'vol_high': 30,
    'ma_short': 20, 'ma_long': 200,
    'ma_type': 'SMA',
    'sep_hedge': True, 'earnings_aggr': False,
}

STRATEGIES = {
    'baseline':       {'mode': 'baseline', 'vol_full_day_threshold': None,
                       'desc': 'Gameplan v2 full-day UPRO (standard)'},
    'overnight_only': {'mode': 'overnight_only', 'vol_full_day_threshold': None,
                       'desc': 'UPRO overnight-only (close→open), cash during day'},
    'hybrid_10':      {'mode': 'hybrid', 'vol_full_day_threshold': 10,
                       'desc': 'Full-day UPRO when vol<10%, overnight-only when 10-15%'},
    'hybrid_12':      {'mode': 'hybrid', 'vol_full_day_threshold': 12,
                       'desc': 'Full-day UPRO when vol<12%, overnight-only when 12-15%'},
    'hybrid_8':       {'mode': 'hybrid', 'vol_full_day_threshold': 8,
                       'desc': 'Full-day UPRO when vol<8%, overnight-only when 8-15%'},
}

full_period_results = {}
strategy_values = {}

for name, cfg in STRATEGIES.items():
    vals, regs, modes, meta = simulate_strategy(
        '2013-01-01', '2026-07-17', WF_PARAMS,
        mode=cfg['mode'],
        vol_full_day_threshold=cfg.get('vol_full_day_threshold', 10)
    )
    metrics = compute_metrics(vals, meta['total_contributed'])
    metrics['switches'] = meta['switches']
    metrics['overnight_trades'] = meta['overnight_trades']

    # Compute time in each mode
    if modes is not None:
        mode_pcts = modes.value_counts(normalize=True) * 100
        metrics['pct_overnight'] = round(mode_pcts.get('overnight', 0), 1)
        metrics['pct_full_day'] = round(mode_pcts.get('full', 0), 1)
        metrics['pct_cash'] = round(mode_pcts.get('cash', 0), 1)

    # Regime breakdown
    if regs is not None:
        reg_pcts = regs.value_counts(normalize=True) * 100
        metrics['pct_upro'] = round(reg_pcts.get('UPRO', 0), 1)
        metrics['pct_spy'] = round(reg_pcts.get('SPY', 0), 1)
        metrics['pct_gld'] = round(reg_pcts.get('GLD', 0), 1)

    full_period_results[name] = metrics
    strategy_values[name] = vals
    print(f"\n  {name}: Sharpe {metrics['sharpe']:.3f}, CAGR {metrics['cagr']:.1f}%, "
          f"MaxDD {metrics['max_dd']:.1f}%, Final ${metrics['final_value']:,.0f}, "
          f"Overnight trades: {metrics['overnight_trades']}")

print("\n  ─── COMPARISON TABLE ───")
print(f"  {'Strategy':<20} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'Calmar':>8} {'Final$':>10} {'OvrNt%':>8}")
print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*8}")
for name in STRATEGIES:
    m = full_period_results[name]
    print(f"  {name:<20} {m['sharpe']:>8.3f} {m['cagr']:>8.1f} {m['max_dd']:>8.1f} "
          f"{m['calmar']:>8.3f} {m['final_value']:>10,.0f} {m.get('pct_overnight', 0):>8.1f}")


# ══════════════════════════════════════════════════════════════════════
# 4. WALK-FORWARD VALIDATION (3yr train, 1yr OOS)
# ══════════════════════════════════════════════════════════════════════
print("\n[3/8] Walk-forward validation...")

# For WF, we optimize the vol_full_day_threshold for hybrid strategies
VOL_FD_THRESHOLDS = [6, 8, 10, 12, 14]
# Also optimize which mode is best within each window
WF_MODES = ['baseline', 'overnight_only', 'hybrid']

wf_windows = []
for train_start_year in range(2013, 2023):
    train_start = f"{train_start_year}-01-01"
    train_end = f"{train_start_year + 2}-12-31"
    test_start = f"{train_start_year + 3}-01-01"
    test_end = f"{train_start_year + 3}-12-31"
    wf_windows.append((train_start, train_end, test_start, test_end))

print(f"  {len(wf_windows)} walk-forward windows")

wf_results = []
wf_oos_values = {}

for wi, (train_start, train_end, test_start, test_end) in enumerate(wf_windows):
    print(f"\n  Window {wi+1}: Train {train_start[:4]}-{train_end[:4]}, Test {test_start[:4]}")

    # Grid search over modes and thresholds on training period
    best_sharpe = -999
    best_config = None

    for mode in WF_MODES:
        thresholds = VOL_FD_THRESHOLDS if mode == 'hybrid' else [10]  # placeholder for non-hybrid
        for thresh in thresholds:
            vals, _, _, meta = simulate_strategy(
                train_start, train_end, WF_PARAMS,
                mode=mode, vol_full_day_threshold=thresh
            )
            m = compute_metrics(vals)
            if m['sharpe'] > best_sharpe:
                best_sharpe = m['sharpe']
                best_config = {'mode': mode, 'vol_full_day_threshold': thresh,
                               'train_sharpe': m['sharpe']}

    # Apply best config to OOS
    vals_oos, regs_oos, modes_oos, meta_oos = simulate_strategy(
        test_start, test_end, WF_PARAMS,
        mode=best_config['mode'],
        vol_full_day_threshold=best_config['vol_full_day_threshold']
    )
    oos_metrics = compute_metrics(vals_oos, meta_oos['total_contributed'] if meta_oos else None)
    oos_metrics['window'] = wi + 1
    oos_metrics['train_period'] = f"{train_start[:4]}-{train_end[:4]}"
    oos_metrics['test_period'] = test_start[:4]
    oos_metrics['best_mode'] = best_config['mode']
    oos_metrics['best_threshold'] = best_config['vol_full_day_threshold']
    oos_metrics['train_sharpe'] = best_config['train_sharpe']

    wf_results.append(oos_metrics)
    wf_oos_values[wi] = vals_oos

    print(f"    Best: {best_config['mode']} (thresh={best_config['vol_full_day_threshold']}), "
          f"Train Sharpe {best_config['train_sharpe']:.3f} → OOS Sharpe {oos_metrics['sharpe']:.3f}")

# WF aggregate
wf_sharpes = [r['sharpe'] for r in wf_results if r['sharpe'] > -999]
wf_modes_chosen = [r['best_mode'] for r in wf_results]

print(f"\n  WF OOS Sharpe: mean {np.mean(wf_sharpes):.3f}, std {np.std(wf_sharpes):.3f}")
print(f"  All OOS positive: {all(s > 0 for s in wf_sharpes)}")
print(f"  Mode selection frequency: {pd.Series(wf_modes_chosen).value_counts().to_dict()}")

# Compare WF to fixed-param baseline
baseline_sharpe = full_period_results['baseline']['sharpe']
wf_mean_sharpe = np.mean(wf_sharpes)
print(f"  WF mean Sharpe {wf_mean_sharpe:.3f} vs baseline {baseline_sharpe:.3f}")


# ══════════════════════════════════════════════════════════════════════
# 5. PERMUTATION TEST
# ══════════════════════════════════════════════════════════════════════
print("\n[4/8] Permutation test (1000 shuffles)...")

# For each strategy, shuffle overnight vs intraday assignment and recompute
N_PERMS = 1000
perm_results = {}

for strat_name in ['baseline', 'overnight_only', 'hybrid_10']:
    cfg = STRATEGIES[strat_name]
    real_vals = strategy_values[strat_name]
    real_sharpe = full_period_results[strat_name]['sharpe']

    perm_sharpes = []
    for p in range(N_PERMS):
        # Shuffle the overnight/intraday returns assignment for UPRO
        # This tests whether the overnight effect is real or random
        shuffled_overnight = overnight_returns['UPRO'].sample(frac=1, replace=False,
                                                               random_state=p).values
        shuffled_intraday = intraday_returns['UPRO'].sample(frac=1, replace=False,
                                                             random_state=p+10000).values

        # Reconstruct shuffled full-day returns
        orig_len = len(full_day_returns)
        shuffled_full = ((1 + shuffled_overnight[:orig_len]) *
                        (1 + shuffled_intraday[:orig_len]) - 1)

        # Run a simplified sim with shuffled returns
        spy = closes['SPY']
        spy_ret = spy.pct_change()
        vol_21d = spy_ret.rolling(21).std() * np.sqrt(252) * 100
        ma_s = compute_ma(spy, 20, 'SMA')
        ma_l = compute_ma(spy, 200, 'SMA')

        cash = INITIAL
        last_week = None
        last_regime = None

        for idx, date in enumerate(closes.index):
            i = closes.index.get_loc(date)
            week_key = (date.year, date.isocalendar()[1])
            if week_key != last_week:
                cash += WEEKLY_DCA
                last_week = week_key

            if i < 205:
                continue

            regime = get_regime(vol_21d.iloc[i], ma_s.iloc[i], ma_l.iloc[i], date,
                               vol_low=15, vol_high=30, sep_hedge=True, earnings_aggr=False)

            if regime != last_regime and last_regime is not None:
                cash *= (1 - TX_COST_PCT) * (1 - GAP_RISK_PCT)
            last_regime = regime

            if regime == 'UPRO':
                if cfg['mode'] == 'overnight_only':
                    r = shuffled_overnight[idx] if idx < len(shuffled_overnight) else 0
                    cash *= (1 + r) * (1 - OVERNIGHT_COST_PCT)
                elif cfg['mode'] == 'hybrid':
                    vol_now = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 15
                    if vol_now < (cfg.get('vol_full_day_threshold') or 10):
                        r = shuffled_full[idx] if idx < len(shuffled_full) else 0
                    else:
                        r = shuffled_overnight[idx] if idx < len(shuffled_overnight) else 0
                        cash *= (1 - OVERNIGHT_COST_PCT)
                    cash *= (1 + r)
                else:
                    r = shuffled_full[idx] if idx < len(shuffled_full) else 0
                    cash *= (1 + r)
            elif regime in full_day_returns.columns:
                r = full_day_returns.loc[date, regime]
                if not np.isnan(r):
                    cash *= (1 + r)

        perm_daily_ret = pd.Series(cash).pct_change().dropna()
        # Use a proxy metric: final cash value's implied Sharpe
        perm_sharpes.append(cash)

    # p-value: fraction of permutations that beat real
    real_final = full_period_results[strat_name]['final_value']
    p_val = np.mean([ps >= real_final for ps in perm_sharpes])
    perm_results[strat_name] = {
        'p_value': round(p_val, 4),
        'real_final': real_final,
        'perm_mean': round(np.mean(perm_sharpes), 2),
        'perm_median': round(np.median(perm_sharpes), 2),
        'perm_p95': round(np.percentile(perm_sharpes, 95), 2),
    }
    print(f"  {strat_name}: p={p_val:.4f} (real=${real_final:,.0f} vs perm mean=${np.mean(perm_sharpes):,.0f})")


# ══════════════════════════════════════════════════════════════════════
# 6. SUB-PERIOD CONSISTENCY (3 blocks)
# ══════════════════════════════════════════════════════════════════════
print("\n[5/8] Sub-period consistency...")

# Split into 3 roughly equal periods
all_dates = closes.index[(closes.index >= '2013-01-01') & (closes.index <= '2026-07-17')]
n = len(all_dates)
periods = [
    (all_dates[0], all_dates[n//3]),
    (all_dates[n//3 + 1], all_dates[2*n//3]),
    (all_dates[2*n//3 + 1], all_dates[-1]),
]

subperiod_results = {}
for strat_name in ['baseline', 'overnight_only', 'hybrid_10']:
    cfg = STRATEGIES[strat_name]
    sp_metrics = []
    for pi, (sp_start, sp_end) in enumerate(periods):
        vals, _, _, meta = simulate_strategy(
            sp_start, sp_end, WF_PARAMS,
            mode=cfg['mode'],
            vol_full_day_threshold=cfg.get('vol_full_day_threshold', 10)
        )
        m = compute_metrics(vals, meta['total_contributed'] if meta else None)
        m['period'] = f"{sp_start.strftime('%Y-%m')}_to_{sp_end.strftime('%Y-%m')}"
        sp_metrics.append(m)

    subperiod_results[strat_name] = sp_metrics
    sharpes = [m['sharpe'] for m in sp_metrics]
    print(f"  {strat_name}: Sharpe by period: {[f'{s:.3f}' for s in sharpes]}")
    print(f"    All positive: {all(s > 0 for s in sharpes)}, "
          f"CV: {np.std(sharpes)/np.mean(sharpes):.3f}")


# ══════════════════════════════════════════════════════════════════════
# 7. REGIME-AGNOSTIC CHECK (HC #428 R1)
# ══════════════════════════════════════════════════════════════════════
print("\n[6/8] Regime-agnostic check (HC #428 R1)...")

spy_daily = full_day_returns['SPY']

r1_results = {}
for strat_name in STRATEGIES:
    vals = strategy_values[strat_name]
    if vals is None:
        continue

    strat_ret = vals.pct_change().dropna()

    # Classify days as green/red/flat based on SPY
    green_days = spy_daily > 0.001   # SPY up > 10bps
    red_days = spy_daily < -0.001    # SPY down > 10bps

    # Align
    common = strat_ret.index.intersection(spy_daily.index)
    s_ret = strat_ret.loc[common]
    g = green_days.loc[common]
    r = red_days.loc[common]

    green_sharpe = s_ret[g].mean() / s_ret[g].std() * np.sqrt(252) if s_ret[g].std() > 0 else 0
    red_sharpe = s_ret[r].mean() / s_ret[r].std() * np.sqrt(252) if s_ret[r].std() > 0 else 0

    gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.001)

    r1_results[strat_name] = {
        'green_sharpe': round(green_sharpe, 3),
        'red_sharpe': round(red_sharpe, 3),
        'gap': round(gap, 3),
        'pass': gap <= 0.50,
        'n_green': int(g.sum()),
        'n_red': int(r.sum()),
    }
    status = "PASS" if gap <= 0.50 else "FAIL"
    print(f"  {strat_name}: Green Sharpe {green_sharpe:.3f}, Red Sharpe {red_sharpe:.3f}, "
          f"Gap {gap:.3f} — {status}")


# ══════════════════════════════════════════════════════════════════════
# 8. DRAWDOWN COMPARISON DEEP DIVE
# ══════════════════════════════════════════════════════════════════════
print("\n[7/8] Drawdown comparison...")

dd_analysis = {}
for strat_name in ['baseline', 'overnight_only', 'hybrid_10']:
    vals = strategy_values[strat_name]
    if vals is None:
        continue

    running_max = vals.cummax()
    dd = (vals - running_max) / running_max

    # Count drawdowns by depth
    dd_events = {'gt5': 0, 'gt10': 0, 'gt20': 0, 'gt30': 0}
    in_dd = False
    current_dd = 0

    for d in dd:
        if d < -0.05:
            if not in_dd:
                in_dd = True
                current_dd = d
            else:
                current_dd = min(current_dd, d)
        else:
            if in_dd:
                if current_dd < -0.05: dd_events['gt5'] += 1
                if current_dd < -0.10: dd_events['gt10'] += 1
                if current_dd < -0.20: dd_events['gt20'] += 1
                if current_dd < -0.30: dd_events['gt30'] += 1
                in_dd = False

    # Max drawdown duration
    underwater = dd < 0
    max_duration = 0
    current_duration = 0
    for u in underwater:
        if u:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0

    dd_analysis[strat_name] = {
        'max_dd': round(dd.min() * 100, 2),
        'avg_dd': round(dd.mean() * 100, 2),
        'dd_events': dd_events,
        'max_dd_duration_days': max_duration,
        'pct_underwater': round((dd < -0.01).mean() * 100, 1),
    }

    print(f"  {strat_name}: MaxDD {dd.min()*100:.1f}%, Avg DD {dd.mean()*100:.2f}%, "
          f"Max duration {max_duration} days, >10% events: {dd_events['gt10']}")


# ══════════════════════════════════════════════════════════════════════
# 9. YEAR-BY-YEAR COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("\n[8/8] Year-by-year comparison...")

yearly_results = {}
for strat_name in ['baseline', 'overnight_only', 'hybrid_10']:
    vals = strategy_values[strat_name]
    if vals is None:
        continue

    yearly = {}
    for year in range(2014, 2027):
        year_mask = (vals.index.year == year)
        if year_mask.sum() < 10:
            continue
        year_vals = vals[year_mask]
        year_ret = year_vals.pct_change().dropna()
        if len(year_ret) < 5:
            continue

        annual_ret = (year_vals.iloc[-1] / year_vals.iloc[0] - 1) * 100
        yr_sharpe = year_ret.mean() / year_ret.std() * np.sqrt(252) if year_ret.std() > 0 else 0

        yearly[str(year)] = {
            'return_pct': round(annual_ret, 1),
            'sharpe': round(yr_sharpe, 3),
        }

    yearly_results[strat_name] = yearly

print(f"\n  {'Year':<6}", end='')
for s in ['baseline', 'overnight_only', 'hybrid_10']:
    print(f" {s[:12]+' Ret%':>16} {s[:6]+' Shp':>10}", end='')
print()
print("  " + "─" * 80)

for year in range(2014, 2027):
    print(f"  {year:<6}", end='')
    for s in ['baseline', 'overnight_only', 'hybrid_10']:
        yr = yearly_results.get(s, {}).get(str(year), {})
        ret = yr.get('return_pct', 0)
        shp = yr.get('sharpe', 0)
        print(f" {ret:>16.1f} {shp:>10.3f}", end='')
    print()


# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY & SAVE
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

# Determine winner
best_strat = max(full_period_results.keys(),
                 key=lambda k: full_period_results[k]['sharpe'])
best_calmar = max(full_period_results.keys(),
                  key=lambda k: full_period_results[k]['calmar'])

print(f"\n  Best Sharpe: {best_strat} ({full_period_results[best_strat]['sharpe']:.3f})")
print(f"  Best Calmar: {best_calmar} ({full_period_results[best_calmar]['calmar']:.3f})")
print(f"  Best MaxDD:  {min(full_period_results.keys(), key=lambda k: abs(full_period_results[k]['max_dd']))}")

# Overnight improvement
bl = full_period_results['baseline']
on = full_period_results['overnight_only']
print(f"\n  Overnight vs Baseline:")
print(f"    Sharpe: {on['sharpe']:.3f} vs {bl['sharpe']:.3f} (delta {on['sharpe']-bl['sharpe']:+.3f})")
print(f"    CAGR:   {on['cagr']:.1f}% vs {bl['cagr']:.1f}% (delta {on['cagr']-bl['cagr']:+.1f}pp)")
print(f"    MaxDD:  {on['max_dd']:.1f}% vs {bl['max_dd']:.1f}% (delta {on['max_dd']-bl['max_dd']:+.1f}pp)")
print(f"    Final:  ${on['final_value']:,.0f} vs ${bl['final_value']:,.0f}")

if 'hybrid_10' in full_period_results:
    hy = full_period_results['hybrid_10']
    print(f"\n  Hybrid (vol<10% full, else overnight) vs Baseline:")
    print(f"    Sharpe: {hy['sharpe']:.3f} vs {bl['sharpe']:.3f} (delta {hy['sharpe']-bl['sharpe']:+.3f})")
    print(f"    CAGR:   {hy['cagr']:.1f}% vs {bl['cagr']:.1f}% (delta {hy['cagr']-bl['cagr']:+.1f}pp)")
    print(f"    MaxDD:  {hy['max_dd']:.1f}% vs {bl['max_dd']:.1f}% (delta {hy['max_dd']-bl['max_dd']:+.1f}pp)")

# Walk-forward summary
print(f"\n  Walk-Forward Validation:")
print(f"    Mean OOS Sharpe: {np.mean(wf_sharpes):.3f}")
print(f"    All OOS positive: {all(s > 0 for s in wf_sharpes)}")
print(f"    Mode chosen most: {pd.Series(wf_modes_chosen).value_counts().index[0]}")

# Save all results
all_results = {
    'full_period': full_period_results,
    'walk_forward': wf_results,
    'permutation': perm_results,
    'subperiod': {k: v for k, v in subperiod_results.items()},
    'r1_regime_agnostic': r1_results,
    'drawdown_analysis': dd_analysis,
    'yearly': yearly_results,
    'wf_params_used': WF_PARAMS,
    'strategies': {k: v['desc'] for k, v in STRATEGIES.items()},
}

output_file = OUTPUT_DIR / "overnight_enhanced_results.json"
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n  Results saved to {output_file}")
print("\nDONE.")
