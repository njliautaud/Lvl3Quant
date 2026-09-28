#!/usr/bin/env python3
"""
Integrated System Backtest — Full Portfolio Simulation
======================================================
Combines ALL validated findings into a single coherent system:

Phase 1 ($500-$2K):   100% UPRO protected (SPY > SMA50)
Phase 2 ($2K-$10K):   70% UPRO protected + 30% CTA trend
Phase 3 ($10K-$50K):  Risk parity (inverse-vol, 126d lookback, threshold 15% rebalance)
Phase 4 ($50K+):      Risk parity + dynamic SH hedge (VIXY momentum)

Dynamic features:
- Protection overlay: SPY > SMA50 → invest, else cash
- DCA: $100/week contributions
- Threshold rebalancing: only when weight drifts >15%
- Dynamic hedge: 20% SH when VIXY > 5-day SMA (Phase 4+)
- Transaction costs: 10 bps round-trip

This is the DEFINITIVE backtest of our full system.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/integrated_system'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TX_COST = 0.001  # 10 bps round-trip
WEEKLY_DCA = 100  # $100/week
INITIAL_CAPITAL = 500

# Phase transition thresholds
PHASE_2_THRESHOLD = 2000
PHASE_3_THRESHOLD = 10000
PHASE_4_THRESHOLD = 50000

def download_data():
    """Download all required tickers."""
    tickers = [
        'SPY', 'UPRO', 'TQQQ',    # Core equity
        'TMF', 'GLD', 'SLV',       # Risk parity components
        'USO', 'UUP',              # Commodities, dollar
        'SH',                       # Inverse for hedging
        'VIXY',                     # VIX proxy for signals
        'TLT', 'HYG', 'IWM',      # Regime signals
    ]

    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start='2012-01-01', period='max',
                       auto_adjust=True, threads=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data

    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except:
            pass

    closes = closes.dropna(how='all').dropna(subset=['UPRO', 'SPY'])
    print(f"  Data: {len(closes)} days")
    print(f"  Range: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")
    return closes

def compute_cta_signal(closes, lookback=200):
    """CTA trend following on SPY."""
    spy = closes['SPY']
    sma = spy.rolling(lookback).mean()
    # Long when price > SMA, else cash
    signal = (spy > sma).astype(float)
    return signal

def compute_risk_parity_weights(returns, lookback=126):
    """Inverse-vol weighting."""
    vols = returns.iloc[-lookback:].std() * np.sqrt(252)
    inv_vol = 1 / vols.clip(lower=0.01)
    weights = inv_vol / inv_vol.sum()
    return weights

def simulate_integrated_system(closes):
    """
    Full simulation with phase transitions, DCA, and all validated rules.
    """
    returns = closes.pct_change().fillna(0)

    spy = closes['SPY']
    spy_sma50 = spy.rolling(50).mean()
    cta_signal = compute_cta_signal(closes, 200)

    # VIXY momentum for dynamic hedge
    vixy = closes['VIXY'] if 'VIXY' in closes.columns else None
    vixy_sma5 = vixy.rolling(5).mean() if vixy is not None else None

    # Risk parity tickers
    rp_tickers = [t for t in ['UPRO', 'TQQQ', 'TMF', 'GLD', 'SLV', 'USO', 'UUP']
                  if t in closes.columns]

    # Track portfolio
    warmup = 252  # Need history for signals
    start_idx = warmup

    portfolio_value = INITIAL_CAPITAL
    cash = INITIAL_CAPITAL
    holdings = {}  # ticker -> dollar value

    # Track metrics
    daily_values = []
    daily_dates = []
    daily_phases = []
    total_contributed = INITIAL_CAPITAL
    n_rebalances = 0
    total_tx_paid = 0

    # Current phase and weights
    current_phase = 1
    last_rebalance_weights = None
    last_rebalance_date = None

    # Weekly DCA tracking
    last_dca_week = None

    for i in range(start_idx, len(closes)):
        date = closes.index[i]

        # --- Weekly DCA ---
        week_num = date.isocalendar()[1]
        year = date.year
        week_key = (year, week_num)
        if week_key != last_dca_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_dca_week = week_key

        # --- Update holdings with daily returns ---
        for ticker in list(holdings.keys()):
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    holdings[ticker] *= (1 + r)

        # --- Calculate total portfolio value ---
        portfolio_value = cash + sum(holdings.values())

        # --- Determine current phase ---
        if portfolio_value >= PHASE_4_THRESHOLD:
            current_phase = 4
        elif portfolio_value >= PHASE_3_THRESHOLD:
            current_phase = 3
        elif portfolio_value >= PHASE_2_THRESHOLD:
            current_phase = 2
        else:
            current_phase = 1

        # --- Protection overlay ---
        protection_on = True
        if not np.isnan(spy_sma50.iloc[i]):
            protection_on = spy.iloc[i] > spy_sma50.iloc[i]

        # --- Dynamic hedge signal (Phase 4+) ---
        hedge_on = False
        if current_phase >= 4 and vixy is not None and vixy_sma5 is not None:
            if not np.isnan(vixy_sma5.iloc[i]):
                hedge_on = vixy.iloc[i] > vixy_sma5.iloc[i]

        # --- Determine target allocation ---
        if not protection_on:
            # Go to cash
            target_alloc = {'CASH': 1.0}
        elif current_phase == 1:
            target_alloc = {'UPRO': 1.0}
        elif current_phase == 2:
            cta = cta_signal.iloc[i] if not np.isnan(cta_signal.iloc[i]) else 1.0
            if cta > 0:
                target_alloc = {'UPRO': 0.7, 'SPY': 0.3}  # CTA = long SPY when trending
            else:
                target_alloc = {'UPRO': 0.7}  # CTA signal off, just UPRO
                # Put CTA portion in cash
        elif current_phase >= 3:
            # Risk parity weights
            avail_tickers = [t for t in rp_tickers if t in returns.columns]
            if len(avail_tickers) >= 3:
                rp_returns = returns[avail_tickers].iloc[max(0, i-126):i]
                weights = compute_risk_parity_weights(rp_returns, min(126, len(rp_returns)))
                target_alloc = {t: float(w) for t, w in zip(avail_tickers, weights.values)}
            else:
                target_alloc = {'UPRO': 0.7, 'SPY': 0.3}

            # Phase 4: dynamic hedge overlay
            if current_phase >= 4 and hedge_on:
                # Shift 20% to SH
                for t in target_alloc:
                    target_alloc[t] *= 0.8
                target_alloc['SH'] = 0.2

        # --- Check if rebalance needed ---
        should_rebalance = False

        if last_rebalance_weights is None:
            should_rebalance = True
        else:
            # Threshold 15% drift check
            invested = sum(holdings.values())
            if invested > 0:
                current_alloc = {t: v / invested for t, v in holdings.items()}
                max_drift = 0
                for t in set(list(target_alloc.keys()) + list(current_alloc.keys())):
                    if t == 'CASH':
                        continue
                    cur = current_alloc.get(t, 0)
                    tgt = target_alloc.get(t, 0)
                    max_drift = max(max_drift, abs(cur - tgt))

                if max_drift > 0.15:
                    should_rebalance = True

            # Also rebalance on phase transition
            if last_rebalance_weights is not None:
                old_phase = daily_phases[-1] if daily_phases else 1
                if current_phase != old_phase:
                    should_rebalance = True

            # Always rebalance if going to/from cash (protection)
            if 'CASH' in target_alloc and holdings:
                should_rebalance = True
            elif 'CASH' not in target_alloc and not holdings and cash > WEEKLY_DCA * 2:
                should_rebalance = True

        # --- Execute rebalance ---
        if should_rebalance:
            invested_value = cash + sum(holdings.values())

            if 'CASH' in target_alloc:
                # Liquidate everything to cash
                turnover = sum(holdings.values()) / invested_value if invested_value > 0 else 0
                cost = turnover * TX_COST * invested_value
                total_tx_paid += cost
                cash = invested_value - cost
                holdings = {}
            else:
                # Allocate to targets
                new_holdings = {}
                total_turnover = 0

                for ticker, weight in target_alloc.items():
                    target_val = invested_value * weight
                    current_val = holdings.get(ticker, 0)
                    trade_val = abs(target_val - current_val)
                    total_turnover += trade_val
                    new_holdings[ticker] = target_val

                cost = (total_turnover / invested_value) * TX_COST * invested_value if invested_value > 0 else 0
                total_tx_paid += cost

                # Deduct cost proportionally
                cost_per_holding = cost / len(new_holdings) if new_holdings else 0
                for t in new_holdings:
                    new_holdings[t] -= cost_per_holding

                holdings = new_holdings
                cash = 0  # Fully invested

            last_rebalance_weights = target_alloc.copy()
            last_rebalance_date = date
            n_rebalances += 1

        # Invest new DCA money (if we have cash and should be invested)
        if cash > WEEKLY_DCA * 0.5 and 'CASH' not in target_alloc and holdings:
            # Distribute cash according to current target allocation
            for ticker, weight in target_alloc.items():
                if ticker != 'CASH':
                    add_val = cash * weight
                    holdings[ticker] = holdings.get(ticker, 0) + add_val
            cash = 0

        portfolio_value = cash + sum(holdings.values())
        daily_values.append(portfolio_value)
        daily_dates.append(date)
        daily_phases.append(current_phase)

    # Build results DataFrame
    portfolio = pd.Series(daily_values, index=daily_dates)
    phases = pd.Series(daily_phases, index=daily_dates)

    return portfolio, phases, total_contributed, n_rebalances, total_tx_paid

def compute_metrics(portfolio, total_contributed):
    """Compute comprehensive metrics."""
    returns = portfolio.pct_change().dropna()
    r = returns

    years = len(r) / 252
    final_value = portfolio.iloc[-1]
    total_return = (final_value - total_contributed) / total_contributed

    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg = r[r < 0]
    downside_vol = neg.std() * np.sqrt(252) if len(neg) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    peak = portfolio.expanding().max()
    dd = (portfolio - peak) / peak
    max_dd = dd.min()
    max_dd_date = dd.idxmin()

    # Time underwater
    underwater = dd < 0
    if underwater.any():
        underwater_periods = []
        in_dd = False
        dd_start = None
        for idx, val in underwater.items():
            if val and not in_dd:
                in_dd = True
                dd_start = idx
            elif not val and in_dd:
                in_dd = False
                underwater_periods.append((dd_start, idx))
        if in_dd:
            underwater_periods.append((dd_start, underwater.index[-1]))

        if underwater_periods:
            max_underwater = max((end - start).days for start, end in underwater_periods)
        else:
            max_underwater = 0
    else:
        max_underwater = 0

    wr = (r > 0).mean()
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    cagr = (final_value / portfolio.iloc[0]) ** (1/years) - 1

    return {
        'years': float(years),
        'initial': float(portfolio.iloc[0]),
        'final_value': float(final_value),
        'total_contributed': float(total_contributed),
        'profit': float(final_value - total_contributed),
        'total_return_on_contributions': float(total_return * 100),
        'cagr': float(cagr * 100),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd * 100),
        'max_dd_date': str(max_dd_date.date()) if hasattr(max_dd_date, 'date') else str(max_dd_date),
        'max_underwater_days': int(max_underwater),
        'win_rate': float(wr * 100),
        'profit_factor': float(pf),
        'ann_vol': float(ann_vol * 100),
    }

def run_comparison_benchmarks(closes, daily_dates, total_contributed_per_day):
    """Run simple benchmark strategies for comparison."""
    benchmarks = {}

    # SPY buy-and-hold with same DCA
    spy = closes['SPY']
    spy_ret = spy.pct_change().fillna(0)

    portfolio_val = INITIAL_CAPITAL
    last_week = None
    total_contributed = INITIAL_CAPITAL
    spy_values = []

    for date in daily_dates:
        if date not in spy_ret.index:
            spy_values.append(portfolio_val)
            continue

        # DCA
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            portfolio_val += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        portfolio_val *= (1 + spy_ret.loc[date])
        spy_values.append(portfolio_val)

    spy_portfolio = pd.Series(spy_values, index=daily_dates)
    benchmarks['SPY DCA'] = compute_metrics(spy_portfolio, total_contributed)

    # UPRO unprotected with same DCA
    upro_ret = closes['UPRO'].pct_change().fillna(0)
    portfolio_val = INITIAL_CAPITAL
    last_week = None
    total_contributed = INITIAL_CAPITAL
    upro_values = []

    for date in daily_dates:
        if date not in upro_ret.index:
            upro_values.append(portfolio_val)
            continue

        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            portfolio_val += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        portfolio_val *= (1 + upro_ret.loc[date])
        upro_values.append(portfolio_val)

    upro_portfolio = pd.Series(upro_values, index=daily_dates)
    benchmarks['UPRO DCA (no protection)'] = compute_metrics(upro_portfolio, total_contributed)

    return benchmarks

def main():
    print("="*70)
    print("INTEGRATED SYSTEM BACKTEST — FULL PORTFOLIO SIMULATION")
    print("="*70)
    print(f"\n  Initial capital: ${INITIAL_CAPITAL}")
    print(f"  Weekly DCA: ${WEEKLY_DCA}")
    print(f"  Phase transitions: $2K → $10K → $50K")
    print(f"  TX cost: {TX_COST*10000:.0f} bps")

    closes = download_data()

    # Run main simulation
    print("\nRunning integrated system simulation...")
    portfolio, phases, total_contributed, n_rebalances, total_tx = simulate_integrated_system(closes)

    metrics = compute_metrics(portfolio, total_contributed)

    print("\n" + "="*70)
    print("INTEGRATED SYSTEM RESULTS")
    print("="*70)

    print(f"\n  Duration: {metrics['years']:.1f} years")
    print(f"  Total contributed: ${total_contributed:,.0f}")
    print(f"  Final value: ${metrics['final_value']:,.0f}")
    print(f"  Profit: ${metrics['profit']:,.0f}")
    print(f"  Return on contributions: {metrics['total_return_on_contributions']:.1f}%")
    print(f"\n  CAGR: {metrics['cagr']:.1f}%")
    print(f"  Sharpe: {metrics['sharpe']:.3f}")
    print(f"  Sortino: {metrics['sortino']:.3f}")
    print(f"  Max Drawdown: {metrics['max_dd']:.1f}% (on {metrics['max_dd_date']})")
    print(f"  Max Underwater: {metrics['max_underwater_days']} days")
    print(f"  Win Rate: {metrics['win_rate']:.1f}%")
    print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
    print(f"\n  Rebalances: {n_rebalances}")
    print(f"  Total TX cost: ${total_tx:.2f}")

    # Phase analysis
    print("\n" + "="*70)
    print("PHASE PROGRESSION")
    print("="*70)

    phase_transitions = []
    current = phases.iloc[0]
    for i, (date, phase) in enumerate(phases.items()):
        if phase != current:
            phase_transitions.append((date, int(current), int(phase), float(portfolio.loc[date])))
            current = phase

    if phase_transitions:
        for date, old, new, value in phase_transitions:
            print(f"  {date.strftime('%Y-%m-%d')}: Phase {old} → Phase {new} (${value:,.0f})")
    else:
        print(f"  Stayed in Phase {phases.iloc[-1]} throughout")

    # Time in each phase
    phase_counts = phases.value_counts().sort_index()
    total_days = len(phases)
    print(f"\n  Time in each phase:")
    for phase, count in phase_counts.items():
        print(f"    Phase {phase}: {count} days ({count/total_days*100:.1f}%)")

    # Benchmarks
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON")
    print("="*70)

    benchmarks = run_comparison_benchmarks(closes, portfolio.index.tolist(), total_contributed)

    print(f"\n  {'Strategy':<30s} {'Final $':>10s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'Profit':>10s}")
    print("  " + "-"*72)

    # Our system first
    print(f"  {'INTEGRATED SYSTEM':<30s} ${metrics['final_value']:>9,.0f} {metrics['cagr']:>6.1f}% "
          f"{metrics['sharpe']:>7.3f} {metrics['max_dd']:>6.1f}% ${metrics['profit']:>9,.0f}")

    for name, bm in benchmarks.items():
        print(f"  {name:<30s} ${bm['final_value']:>9,.0f} {bm['cagr']:>6.1f}% "
              f"{bm['sharpe']:>7.3f} {bm['max_dd']:>6.1f}% ${bm['profit']:>9,.0f}")

    # Yearly breakdown
    print("\n" + "="*70)
    print("YEARLY PERFORMANCE")
    print("="*70)

    yearly_returns = portfolio.resample('YE').last().pct_change().dropna()
    yearly_contributions = pd.Series(0.0, index=yearly_returns.index)

    print(f"\n  {'Year':>6s} {'Return':>8s} {'Value':>10s} {'Phase':>6s}")
    print("  " + "-"*35)

    for date in portfolio.resample('YE').last().index:
        year = date.year
        val = portfolio.asof(date)
        phase = phases.asof(date)
        if date in yearly_returns.index:
            ret = yearly_returns.loc[date]
            print(f"  {year:>6d} {ret*100:>7.1f}% ${val:>9,.0f}    P{int(phase)}")

    # Save results
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'parameters': {
            'initial_capital': INITIAL_CAPITAL,
            'weekly_dca': WEEKLY_DCA,
            'tx_cost_bps': TX_COST * 10000,
            'phase_thresholds': [PHASE_2_THRESHOLD, PHASE_3_THRESHOLD, PHASE_4_THRESHOLD],
        },
        'metrics': metrics,
        'phase_transitions': [(str(d.date()), o, n, v) for d, o, n, v in phase_transitions],
        'benchmarks': benchmarks,
        'n_rebalances': n_rebalances,
        'total_tx_cost': float(total_tx),
    }

    output_path = os.path.join(OUTPUT_DIR, 'integrated_system_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved.")
    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    main()
