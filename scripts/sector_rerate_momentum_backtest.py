#!/usr/bin/env python3
"""
SECTOR RE-RATING MOMENTUM BACKTEST
====================================
Hypothesis: Sectors whose excess return vs SPY (risk-adjusted by their own vol)
is accelerating are experiencing "fundamental re-rating" — analyst conviction
flowing in before full market pricing. Go long top-3 sectors for 5 days.

Signal: Re-Rating Speed = 20d change in (sector_return - SPY_return) / sector_vol
  - Captures which sectors are being bid up vs market faster than volatility predicts
  - High values = institutional conviction / analyst upgrades flowing in

Walk-forward: sliding 60-day estimation window, 1-day step
Cost: 0.20% round-trip (ETF options spread proxy)
Hold: 5 trading days
OOT: ALL available days from 2021-01-01 onward (60d warmup from 2020)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
import sys
import json

warnings.filterwarnings('ignore')
np.random.seed(42)

def pprint(*args, **kwargs):
    kwargs.setdefault('flush', True)
    __builtins__['print'](*args, **kwargs) if isinstance(__builtins__, dict) else print(*args, flush=True, **kwargs)

# ── Config ──────────────────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLU', 'XLP', 'XLY', 'XLV', 'XLI', 'XLB', 'XLC', 'XLRE']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]

LOOKBACK = 20          # days for re-rating speed calc
ESTIMATION_WINDOW = 60 # sliding window for ranking calibration
HOLD_DAYS = 5          # 5 trading days
TOP_N = 3              # long top 3 sectors
COST_RT = 0.0020       # 0.20% round-trip cost
CAPITAL = 100000.0     # notional

DATA_START = '2020-01-01'
DATA_END = '2026-08-21'

# ── Data Download ───────────────────────────────────────────────────────────────
def download_data():
    print("Downloading sector ETF + SPY daily data...", flush=True)
    raw = yf.download(ALL_TICKERS, start=DATA_START, end=DATA_END,
                       progress=True, auto_adjust=True, group_by='ticker', threads=True)

    closes = {}
    for ticker in ALL_TICKERS:
        try:
            if len(ALL_TICKERS) > 1:
                s = raw[ticker]['Close'].dropna().squeeze()
            else:
                s = raw['Close'].dropna().squeeze()
            if len(s) > 100:
                closes[ticker] = s
        except Exception as e:
            print(f"  WARNING: Failed to get {ticker}: {e}", flush=True)

    df = pd.DataFrame(closes).dropna()
    print(f"  Got {len(df)} trading days, {len(df.columns)} tickers", flush=True)
    return df


# ── Signal: Re-Rating Speed ────────────────────────────────────────────────────
def compute_rerate_speed(prices_df, lookback=LOOKBACK):
    """
    For each sector ETF on each day:
      rerate_speed = (sector_20d_ret - spy_20d_ret) / sector_20d_vol

    This is essentially a risk-adjusted relative momentum score.
    High = sector outperforming SPY more than its vol would predict = conviction.
    """
    returns = prices_df.pct_change()
    spy_ret = returns[BENCHMARK]

    signals = pd.DataFrame(index=prices_df.index, columns=SECTOR_ETFS, dtype=float)

    for etf in SECTOR_ETFS:
        if etf not in returns.columns:
            continue
        etf_ret = returns[etf]

        # 20-day cumulative excess return
        etf_cum = etf_ret.rolling(lookback).sum()
        spy_cum = spy_ret.rolling(lookback).sum()
        excess_ret = etf_cum - spy_cum

        # 20-day realized volatility
        etf_vol = etf_ret.rolling(lookback).std() * np.sqrt(252)

        # Re-rating speed = excess return / vol (information-ratio-like)
        signals[etf] = excess_ret / etf_vol.replace(0, np.nan)

    return signals


def compute_rerate_acceleration(prices_df, lookback=LOOKBACK):
    """
    Variant 2: ACCELERATION of re-rating speed.
    Change in re-rating speed over the last 5 days.
    Captures sectors where conviction is INCREASING.
    """
    rerate = compute_rerate_speed(prices_df, lookback)
    accel = rerate.diff(5)  # 5-day change in re-rating speed
    return accel


def compute_vol_adjusted_momentum(prices_df, lookback=LOOKBACK):
    """
    Variant 3: Momentum / Volatility (simple vol-adjusted momentum).
    Classic risk-parity style momentum signal.
    """
    returns = prices_df.pct_change()
    signals = pd.DataFrame(index=prices_df.index, columns=SECTOR_ETFS, dtype=float)

    for etf in SECTOR_ETFS:
        if etf not in returns.columns:
            continue
        mom = returns[etf].rolling(lookback).sum()
        vol = returns[etf].rolling(lookback).std()
        signals[etf] = mom / vol.replace(0, np.nan)

    return signals


# ── Backtest Engine ─────────────────────────────────────────────────────────────
def run_backtest(prices_df, signal_func, label, top_n=TOP_N, hold=HOLD_DAYS):
    """
    Walk-forward sector rotation backtest.
    Each day: rank sectors by signal, go long top_n, hold for `hold` days.
    Positions are equal-weight, rebalanced every `hold` days.
    Sliding window: only use last ESTIMATION_WINDOW days for signal calc context.
    """
    signals = signal_func(prices_df)
    returns = prices_df[SECTOR_ETFS].pct_change()
    spy_returns = prices_df[BENCHMARK].pct_change()

    # Need warmup: lookback + estimation_window
    warmup = LOOKBACK + ESTIMATION_WINDOW
    tradeable_dates = prices_df.index[warmup:]

    if len(tradeable_dates) < 40:
        print(f"  {label}: Not enough OOT days ({len(tradeable_dates)}). SKIP.", flush=True)
        return None

    print(f"  {label}: {len(tradeable_dates)} OOT days available", flush=True)

    # Strategy returns (daily)
    strat_daily_rets = []
    strat_dates = []
    trade_log = []

    # Rebalance every `hold` days
    current_holdings = []  # list of ETF tickers
    days_since_rebal = hold  # force rebalance on first day

    for i, date in enumerate(tradeable_dates):
        idx = prices_df.index.get_loc(date)

        if days_since_rebal >= hold:
            # Rebalance: rank sectors by signal, pick top N
            day_signals = signals.loc[date]
            valid = day_signals.dropna()

            if len(valid) >= top_n:
                ranked = valid.sort_values(ascending=False)
                new_holdings = list(ranked.index[:top_n])

                # Log trade
                if set(new_holdings) != set(current_holdings):
                    trade_log.append({
                        'date': str(date.date()),
                        'action': 'rebalance',
                        'holdings': new_holdings,
                        'signals': {k: round(float(v), 4) for k, v in ranked.head(top_n).items()}
                    })

                current_holdings = new_holdings
                days_since_rebal = 0
            # else: keep current holdings

        # Compute daily return of portfolio
        if current_holdings:
            daily_rets = returns.loc[date, current_holdings]
            port_ret = daily_rets.mean()  # equal weight

            # Apply cost on rebalance days (prorated)
            if days_since_rebal == 0:
                port_ret -= COST_RT / hold  # spread cost over hold period

            strat_daily_rets.append(float(port_ret))
        else:
            strat_daily_rets.append(0.0)

        strat_dates.append(date)
        days_since_rebal += 1

    # Build results
    strat_rets = pd.Series(strat_daily_rets, index=strat_dates)
    spy_rets_oot = spy_returns.loc[strat_dates]

    return {
        'label': label,
        'strat_rets': strat_rets,
        'spy_rets': spy_rets_oot,
        'trade_log': trade_log,
        'n_oot_days': len(tradeable_dates),
        'n_rebalances': len(trade_log)
    }


# ── Metrics ─────────────────────────────────────────────────────────────────────
def compute_metrics(result):
    """Full metrics suite."""
    rets = result['strat_rets']
    spy = result['spy_rets']

    n_days = len(rets)

    # Sharpe
    sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0

    # Sortino
    downside = rets[rets < 0]
    sortino = (rets.mean() / downside.std()) * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

    # Cumulative return
    cum_ret = (1 + rets).prod() - 1

    # Max drawdown
    cum_equity = (1 + rets).cumprod()
    peak = cum_equity.expanding().max()
    dd = (cum_equity - peak) / peak
    max_dd = dd.min()

    # Win rate (daily)
    wr = (rets > 0).mean()

    # Profit factor (daily)
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / losses if losses > 0 else (999 if gains > 0 else 0)

    # Win rate on 5-day blocks (trade-level)
    block_rets = []
    for i in range(0, len(rets) - HOLD_DAYS + 1, HOLD_DAYS):
        block = rets.iloc[i:i+HOLD_DAYS]
        block_rets.append(block.sum())
    block_rets = np.array(block_rets)
    trade_wr = (block_rets > 0).mean() if len(block_rets) > 0 else 0
    trade_pf_gains = block_rets[block_rets > 0].sum() if len(block_rets) > 0 else 0
    trade_pf_losses = abs(block_rets[block_rets < 0].sum()) if len(block_rets) > 0 else 0
    trade_pf = trade_pf_gains / trade_pf_losses if trade_pf_losses > 0 else 0

    # SPY comparison
    spy_cum = (1 + spy).prod() - 1
    spy_sharpe = (spy.mean() / spy.std()) * np.sqrt(252) if spy.std() > 0 else 0

    # Excess return
    excess_ret = cum_ret - spy_cum

    # Regime analysis: green days (SPY up) vs red days (SPY down)
    green_mask = spy > 0
    red_mask = spy < 0

    green_rets = rets[green_mask]
    red_rets = rets[red_mask]

    sharpe_green = (green_rets.mean() / green_rets.std()) * np.sqrt(252) if len(green_rets) > 10 and green_rets.std() > 0 else 0
    sharpe_red = (red_rets.mean() / red_rets.std()) * np.sqrt(252) if len(red_rets) > 10 and red_rets.std() > 0 else 0

    # Regime gap
    max_regime = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / max_regime if max_regime > 0 else 0

    # Day concentration: max single-day contribution to total P&L
    if abs(rets.sum()) > 0:
        day_conc = rets.abs().max() / rets.abs().sum()
    else:
        day_conc = 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cum_return_pct': round(cum_ret * 100, 2),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'daily_win_rate': round(wr * 100, 1),
        'daily_profit_factor': round(min(pf, 99), 3),
        'trade_win_rate': round(trade_wr * 100, 1),
        'trade_profit_factor': round(min(trade_pf, 99), 3),
        'n_trades': len(block_rets),
        'n_oot_days': n_days,
        'spy_cum_return_pct': round(spy_cum * 100, 2),
        'spy_sharpe': round(spy_sharpe, 3),
        'excess_return_pct': round(excess_ret * 100, 2),
        'sharpe_green_days': round(sharpe_green, 3),
        'sharpe_red_days': round(sharpe_red, 3),
        'regime_gap': round(regime_gap, 4),
        'day_concentration': round(day_conc, 4),
        'n_green_days': int(green_mask.sum()),
        'n_red_days': int(red_mask.sum()),
        'avg_daily_ret_bps': round(rets.mean() * 10000, 2),
        'annual_return_pct': round(rets.mean() * 252 * 100, 2),
    }


# ── Permutation Test ────────────────────────────────────────────────────────────
def permutation_test(prices_df, signal_func, real_sharpe, n_perms=500):
    """
    Shuffle sector labels each rebalance to break signal-sector link.
    Returns p-value.
    """
    returns = prices_df[SECTOR_ETFS].pct_change()
    signals = signal_func(prices_df)
    warmup = LOOKBACK + ESTIMATION_WINDOW
    tradeable_dates = prices_df.index[warmup:]

    perm_sharpes = []

    for p in range(n_perms):
        rng = np.random.RandomState(p + 1000)
        strat_rets = []
        days_since = HOLD_DAYS
        holdings = []

        for date in tradeable_dates:
            if days_since >= HOLD_DAYS:
                day_signals = signals.loc[date].dropna()
                if len(day_signals) >= TOP_N:
                    # Shuffle: randomly pick TOP_N sectors
                    available = list(day_signals.index)
                    rng.shuffle(available)
                    holdings = available[:TOP_N]
                days_since = 0

            if holdings:
                dr = returns.loc[date, holdings]
                port_ret = dr.mean()
                if days_since == 0:
                    port_ret -= COST_RT / HOLD_DAYS
                strat_rets.append(float(port_ret))
            else:
                strat_rets.append(0.0)

            days_since += 1

        sr = pd.Series(strat_rets)
        if sr.std() > 0:
            perm_sharpes.append((sr.mean() / sr.std()) * np.sqrt(252))
        else:
            perm_sharpes.append(0.0)

    count_ge = sum(1 for s in perm_sharpes if s >= real_sharpe)
    return round(count_ge / n_perms, 4), perm_sharpes


# ── Yearly Breakdown ────────────────────────────────────────────────────────────
def yearly_breakdown(result):
    """Per-year metrics."""
    rets = result['strat_rets']
    spy = result['spy_rets']

    years = sorted(rets.index.year.unique())
    rows = []
    for y in years:
        mask = rets.index.year == y
        yr = rets[mask]
        sr = spy[mask]
        if len(yr) < 10:
            continue
        cum = (1 + yr).prod() - 1
        spy_cum = (1 + sr).prod() - 1
        sh = (yr.mean() / yr.std()) * np.sqrt(252) if yr.std() > 0 else 0
        rows.append({
            'year': y,
            'return_pct': round(cum * 100, 2),
            'spy_return_pct': round(spy_cum * 100, 2),
            'excess_pct': round((cum - spy_cum) * 100, 2),
            'sharpe': round(sh, 3),
            'days': len(yr)
        })
    return rows


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80, flush=True)
    print("SECTOR RE-RATING MOMENTUM BACKTEST", flush=True)
    print("Hypothesis: Sectors with fastest risk-adjusted excess returns", flush=True)
    print("  are experiencing 'fundamental re-rating' and will outperform 5 more days.", flush=True)
    print("=" * 80, flush=True)

    prices = download_data()
    print(f"\nDate range: {prices.index[0].date()} to {prices.index[-1].date()}", flush=True)
    print(f"Sectors available: {[c for c in SECTOR_ETFS if c in prices.columns]}", flush=True)

    # ── Run all 3 signal variants + adversarial ─────────────────────────────────
    variants = [
        ('A: Re-Rating Speed (20d excess/vol)', lambda df: compute_rerate_speed(df, LOOKBACK)),
        ('B: Re-Rating Acceleration (5d chg)', lambda df: compute_rerate_acceleration(df, LOOKBACK)),
        ('C: Vol-Adjusted Momentum (20d)', lambda df: compute_vol_adjusted_momentum(df, LOOKBACK)),
    ]

    all_results = {}

    for label, sig_func in variants:
        print(f"\n{'─' * 70}", flush=True)
        print(f"Running: {label}", flush=True)
        print(f"{'─' * 70}", flush=True)

        result = run_backtest(prices, sig_func, label)
        if result is None:
            continue

        metrics = compute_metrics(result)

        print(f"  OOT days: {metrics['n_oot_days']}", flush=True)
        print(f"  Trades (5d blocks): {metrics['n_trades']}", flush=True)
        print(f"  Cumulative return: {metrics['cum_return_pct']:.2f}%", flush=True)
        print(f"  SPY return: {metrics['spy_cum_return_pct']:.2f}%", flush=True)
        print(f"  Excess return: {metrics['excess_return_pct']:.2f}%", flush=True)
        print(f"  Sharpe: {metrics['sharpe']:.3f}", flush=True)
        print(f"  Sortino: {metrics['sortino']:.3f}", flush=True)
        print(f"  Trade WR: {metrics['trade_win_rate']:.1f}%", flush=True)
        print(f"  Trade PF: {metrics['trade_profit_factor']:.3f}", flush=True)
        print(f"  Max DD: {metrics['max_drawdown_pct']:.2f}%", flush=True)
        print(f"  Avg daily ret: {metrics['avg_daily_ret_bps']:.2f} bps", flush=True)
        print(f"  Annualized return: {metrics['annual_return_pct']:.2f}%", flush=True)
        print(f"  Sharpe (green SPY days): {metrics['sharpe_green_days']:.3f}", flush=True)
        print(f"  Sharpe (red SPY days): {metrics['sharpe_red_days']:.3f}", flush=True)
        print(f"  Regime gap: {metrics['regime_gap']:.4f}", flush=True)
        print(f"  Day concentration: {metrics['day_concentration']:.4f}", flush=True)

        # Yearly breakdown
        yearly = yearly_breakdown(result)
        if yearly:
            print(f"\n  Per-year breakdown:", flush=True)
            print(f"  {'Year':<6} {'Return':>8} {'SPY':>8} {'Excess':>8} {'Sharpe':>8} {'Days':>6}", flush=True)
            for row in yearly:
                print(f"  {row['year']:<6} {row['return_pct']:>7.2f}% {row['spy_return_pct']:>7.2f}% {row['excess_pct']:>7.2f}% {row['sharpe']:>8.3f} {row['days']:>6}", flush=True)

        # Permutation test
        if metrics['sharpe'] > 0:
            print(f"\n  Running permutation test (500 shuffles)...", flush=True)
            perm_p, perm_dist = permutation_test(prices, sig_func, metrics['sharpe'])
            print(f"  Permutation p-value: {perm_p:.4f}", flush=True)
            print(f"  Perm Sharpe distribution: mean={np.mean(perm_dist):.3f}, std={np.std(perm_dist):.3f}", flush=True)
        else:
            perm_p = 1.0
            print(f"  Sharpe <= 0, skipping permutation test", flush=True)

        metrics['perm_p'] = perm_p

        # Gate check
        gates = {
            'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
            'regime_gap_lt_0.5': metrics['regime_gap'] < 0.5,
            'perm_p_lt_0.05': perm_p < 0.05,
            'mdd_gt_neg30': metrics['max_drawdown_pct'] > -30,
            'trades_gte_40': metrics['n_trades'] >= 40,
            'day_conc_lt_0.70': metrics['day_concentration'] < 0.70,
        }
        gates['all_pass'] = all(gates.values())

        print(f"\n  GATES:", flush=True)
        for g, v in gates.items():
            status = "PASS" if v else "FAIL"
            print(f"    {g}: {status}", flush=True)

        all_results[label] = {
            'metrics': metrics,
            'gates': gates,
            'yearly': yearly,
            'sample_trades': result['trade_log'][:10]
        }

    # ── Adversarial: random sector pick ─────────────────────────────────────────
    print(f"\n{'─' * 70}", flush=True)
    print(f"Running: ADVERSARIAL (random sector selection)", flush=True)
    print(f"{'─' * 70}", flush=True)

    # Random signal: just random noise
    def random_signal(df):
        rng = np.random.RandomState(42)
        idx = df.index
        cols = SECTOR_ETFS
        return pd.DataFrame(rng.randn(len(idx), len(cols)), index=idx, columns=cols)

    adv_result = run_backtest(prices, random_signal, 'ADVERSARIAL')
    if adv_result:
        adv_metrics = compute_metrics(adv_result)
        print(f"  Sharpe: {adv_metrics['sharpe']:.3f}", flush=True)
        print(f"  Cum return: {adv_metrics['cum_return_pct']:.2f}%", flush=True)
        all_results['ADVERSARIAL'] = {'metrics': adv_metrics}

    # ── Summary ─────────────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 100}", flush=True)
    print("SUMMARY TABLE", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"{'Variant':<42} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'SPY':>8} {'Excess':>8} {'TrWR':>6} {'TrPF':>6} {'MDD':>7} {'RGap':>6} {'Perm_p':>7} {'Pass':>5}", flush=True)
    print("-" * 130, flush=True)

    for lbl, r in all_results.items():
        m = r['metrics']
        perm = m.get('perm_p', '-')
        perm_str = f"{perm:.4f}" if isinstance(perm, float) else perm
        pass_str = "YES" if r.get('gates', {}).get('all_pass', False) else "NO"
        tr_wr = m.get('trade_win_rate', 0)
        tr_pf = m.get('trade_profit_factor', 0)
        rg = m.get('regime_gap', '-')
        rg_str = f"{rg:.4f}" if isinstance(rg, (int, float)) else rg
        print(f"{lbl:<42} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['cum_return_pct']:>7.2f}% {m['spy_cum_return_pct']:>7.2f}% {m['excess_return_pct']:>7.2f}% {tr_wr:>5.1f}% {tr_pf:>6.3f} {m['max_drawdown_pct']:>6.2f}% {rg_str:>6} {perm_str:>7} {pass_str:>5}", flush=True)

    print("-" * 130, flush=True)

    # ── Verdict ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}", flush=True)
    print("VERDICT", flush=True)
    print(f"{'=' * 80}", flush=True)

    passing = [k for k, v in all_results.items() if v.get('gates', {}).get('all_pass', False) and 'ADVERSARIAL' not in k]

    if passing:
        print(f"PASSING VARIANTS: {', '.join(passing)}", flush=True)
        for k in passing:
            m = all_results[k]['metrics']
            print(f"  {k}: Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
                  f"WR={m['trade_win_rate']:.1f}%, PF={m['trade_profit_factor']:.3f}, "
                  f"Excess={m['excess_return_pct']:.2f}%", flush=True)
    else:
        print("NO VARIANTS PASSED ALL GATES.", flush=True)
        # Best variant
        real_variants = {k: v for k, v in all_results.items() if 'ADVERSARIAL' not in k}
        if real_variants:
            best_k = max(real_variants, key=lambda k: real_variants[k]['metrics']['sharpe'])
            m = real_variants[best_k]['metrics']
            failed = [g for g, v in all_results[best_k].get('gates', {}).items() if not v and g != 'all_pass']
            print(f"Best: {best_k} (Sharpe={m['sharpe']:.3f})", flush=True)
            print(f"Failed gates: {', '.join(failed)}", flush=True)

    # Adversarial check
    adv_sharpe = all_results.get('ADVERSARIAL', {}).get('metrics', {}).get('sharpe', 0)
    real_sharpes = [v['metrics']['sharpe'] for k, v in all_results.items() if 'ADVERSARIAL' not in k]
    best_real = max(real_sharpes) if real_sharpes else 0

    if adv_sharpe >= best_real:
        print(f"\nWARNING: Random baseline (Sharpe={adv_sharpe:.3f}) matched best strategy ({best_real:.3f}). NO EDGE.", flush=True)
    else:
        print(f"\nAdversarial Sharpe: {adv_sharpe:.3f} vs Best: {best_real:.3f} "
              f"(+{best_real - adv_sharpe:.3f} advantage)", flush=True)

    if best_real < 0.5:
        print(f"\nBest Sharpe {best_real:.3f} < 0.5 threshold. SIGNAL IS DEAD for 5-day sector rotation.", flush=True)

    # Save results
    output = {
        'backtest': 'Sector Re-Rating Momentum',
        'run_date': str(datetime.now()),
        'config': {
            'sectors': SECTOR_ETFS,
            'lookback': LOOKBACK,
            'hold_days': HOLD_DAYS,
            'top_n': TOP_N,
            'cost_rt': COST_RT,
            'estimation_window': ESTIMATION_WINDOW,
        },
        'results': {}
    }
    for k, v in all_results.items():
        output['results'][k] = {
            'metrics': v['metrics'],
            'gates': v.get('gates', {}),
            'yearly': v.get('yearly', []),
        }

    out_path = '/home/jupiter/Lvl3Quant/data/sector_rerate_momentum_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved.", flush=True)


if __name__ == '__main__':
    main()
