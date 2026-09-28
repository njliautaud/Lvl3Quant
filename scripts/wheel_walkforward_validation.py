#!/usr/bin/env python3
"""
wheel_walkforward_validation.py — Walk-Forward Out-of-Sample Validation

PURPOSE: The wheel strategy's 21.3% CAGR / Sharpe 1.06 was measured over the
FULL 2019-2026 period. But parameter selection (delta, DTE, margin cap, etc.)
was optimized over that same period — potential overfit.

THIS TEST: Split into rolling windows:
  - Train: optimize params on first N years
  - Test: run those params on the next year (unseen)
  - Report in-sample vs out-of-sample degradation

Walk-forward windows:
  1. Train 2019-2021, Test 2022
  2. Train 2019-2022, Test 2023
  3. Train 2019-2023, Test 2024
  4. Train 2019-2024, Test 2025
  5. Train 2019-2025, Test 2026 (partial)

Also tests: fixed "best" params (30-delta, 14 DTE, 40% margin, 3% per name, 65% PT)
across each OOS year separately — are 2022 (bear) and 2024 (bull) consistent?
"""
import sys
import time
import math
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from collections import deque
from dataclasses import dataclass

ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = ROOT / "output" / "wheel_walkforward"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))

logging.basicConfig(
    format='%(asctime)s [WF-VAL] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger('WF-VAL')

RISK_FREE = 0.04
COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03
MARGIN_REQ_PCT = 0.20
CALL_DELTA = 0.30

def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0, (K - S) if kind == "put" else (S - K))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
    else:
        return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)

def find_strike(S, sigma, T, target_delta, r=RISK_FREE, kind="put"):
    if T <= 0 or sigma <= 0:
        return S
    lo, hi = S * 0.5, S * 1.5
    for _ in range(50):
        K = (lo + hi) / 2
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        if kind == "put":
            delta_abs = abs(_Phi(d1) - 1)
        else:
            delta_abs = _Phi(d1)
        if delta_abs < target_delta:
            if kind == "put":
                hi = K
            else:
                lo = K
        else:
            if kind == "put":
                lo = K
            else:
                hi = K
    return round((lo + hi) / 2, 2)


@dataclass
class CSP:
    ticker: str
    open_date: object
    strike: float
    premium: float
    dte: int
    expiry: object
    margin: float

@dataclass
class SharePos:
    ticker: str
    assign_date: object
    shares: int
    cost_basis: float
    cc_strike: float = 0
    cc_premium: float = 0
    cc_expiry: object = None


def run_wheel_period(price_map, sigma_map, spy_bear, dates, tickers,
                     starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
                     put_delta=0.30, dte_target=14, profit_take=0.65,
                     bear_mode="liq_csp_only", max_assignments_5d=3,
                     max_share_positions=5, loss_cut_pct=-0.15,
                     min_price=10.0, max_price=500.0):
    """Run wheel simulation over a specific date range. Returns metrics dict + equity df."""
    import random

    cash = float(starting_cash)
    csps = []
    shares = []
    assignment_log = deque(maxlen=100)
    daily_equity = []
    trades = []

    for di, date in enumerate(dates):
        is_bear = spy_bear.get(date, 0)

        # Check CSP expiries
        expired_csps = [c for c in csps if date >= c.expiry]
        active_csps = [c for c in csps if date < c.expiry]

        for c in expired_csps:
            px = price_map.get((c.ticker, date))
            if px is None:
                # No price data — assume expired worthless, margin returned
                cash += c.margin
                trades.append({'ticker': c.ticker, 'type': 'csp_expire', 'date': date, 'pnl': c.premium * 100})
                continue

            if px < c.strike:
                # ITM at expiry — assignment or force-close
                recent_assigns = sum(1 for ad in assignment_log if (date - ad).days <= 5)
                if recent_assigns >= max_assignments_5d or len(shares) >= max_share_positions:
                    # Force close: pay intrinsic value to close
                    intrinsic = (c.strike - px) * 100
                    # Premium was already received at open. Now we pay intrinsic to close.
                    cash += c.margin  # Get margin back
                    cash -= intrinsic  # Pay to close ITM put
                    cash -= COST_PER_CONTRACT  # Close commission
                    pnl = c.premium * 100 - intrinsic - COST_PER_CONTRACT
                    trades.append({'ticker': c.ticker, 'type': 'csp_force_close', 'date': date, 'pnl': pnl})
                else:
                    # Assignment: take shares at strike
                    assignment_log.append(date)
                    cost_basis = c.strike - c.premium
                    # Premium already received at open. Now pay strike for shares.
                    cash += c.margin  # Get margin back
                    cash -= c.strike * 100  # Buy shares at strike
                    shares.append(SharePos(
                        ticker=c.ticker, assign_date=date,
                        shares=100, cost_basis=cost_basis
                    ))
                    trades.append({'ticker': c.ticker, 'type': 'assigned', 'date': date, 'pnl': 0})
            else:
                # OTM at expiry — expires worthless, premium kept (already received at open)
                cash += c.margin  # Get margin back
                trades.append({'ticker': c.ticker, 'type': 'csp_profit', 'date': date, 'pnl': c.premium * 100})

        csps = active_csps

        # Profit-take on active CSPs
        new_csps = []
        for c in csps:
            px = price_map.get((c.ticker, date))
            sigma = sigma_map.get((c.ticker, date), 0.3)
            if px is not None:
                remaining = max((c.expiry - date).days / 365.0, 1/365)
                current_val = bs_price(px, c.strike, remaining, sigma)
                if current_val <= c.premium * (1 - profit_take):
                    # Buy back put at current_val (premium already received at open)
                    buyback = current_val * 100 + COST_PER_CONTRACT
                    cash += c.margin  # Get margin back
                    cash -= buyback  # Pay to close
                    pnl = (c.premium - current_val) * 100 - 2 * COST_PER_CONTRACT
                    trades.append({'ticker': c.ticker, 'type': 'csp_profit_take', 'date': date, 'pnl': pnl})
                    continue
            new_csps.append(c)
        csps = new_csps

        # Bear mode
        if is_bear and bear_mode == "liq_csp_only":
            for c in csps:
                px = price_map.get((c.ticker, date))
                sigma = sigma_map.get((c.ticker, date), 0.3)
                if px is not None:
                    remaining = max((c.expiry - date).days / 365.0, 1/365)
                    current_val = bs_price(px, c.strike, remaining, sigma)
                    # Buy back put at current_val
                    buyback = current_val * 100 + COST_PER_CONTRACT
                    cash += c.margin  # Get margin back
                    cash -= buyback  # Pay to close
                    pnl = (c.premium - current_val) * 100 - 2 * COST_PER_CONTRACT
                    trades.append({'ticker': c.ticker, 'type': 'bear_close', 'date': date, 'pnl': pnl})
                else:
                    cash += c.margin
            csps = []

        # Share management
        new_shares = []
        for s in shares:
            px = price_map.get((s.ticker, date))
            if px is None:
                new_shares.append(s)
                continue
            unrealized = (px - s.cost_basis) / s.cost_basis
            if unrealized < loss_cut_pct:
                pnl = (px - s.cost_basis) * s.shares
                cash += px * s.shares
                trades.append({'ticker': s.ticker, 'type': 'share_loss_cut', 'date': date, 'pnl': pnl})
                continue
            if s.cc_expiry and date >= s.cc_expiry:
                if px >= s.cc_strike and s.cc_strike > 0:
                    pnl = (s.cc_strike - s.cost_basis) * s.shares + s.cc_premium * 100
                    cash += s.cc_strike * s.shares + s.cc_premium * 100
                    trades.append({'ticker': s.ticker, 'type': 'cc_called', 'date': date, 'pnl': pnl})
                    continue
                else:
                    s.cc_strike = 0
                    s.cc_premium = 0
                    s.cc_expiry = None
            if s.cc_strike == 0:
                sigma = sigma_map.get((s.ticker, date), 0.3)
                T = dte_target / 365.0
                cc_strike = find_strike(px, sigma, T, CALL_DELTA, kind="call")
                cc_prem = bs_price(px, cc_strike, T, sigma, kind="call")
                slip = max(cc_prem * SLIPPAGE_FRAC, SLIPPAGE_MIN)
                cc_prem = max(cc_prem - slip, 0.01)
                s.cc_strike = cc_strike
                s.cc_premium = cc_prem
                s.cc_expiry = date + timedelta(days=dte_target)
            new_shares.append(s)
        shares = new_shares

        # Open new CSPs
        if not is_bear or bear_mode != "liq_csp_only":
            current_margin_used = sum(c.margin for c in csps)
            csp_tickers = set(c.ticker for c in csps)
            share_tickers = set(s.ticker for s in shares)
            total_equity_est = cash + sum(
                price_map.get((s.ticker, date), s.cost_basis) * s.shares for s in shares
            )
            random.seed(int(date.timestamp()) if hasattr(date, 'timestamp') else di)
            candidates = [t for t in tickers if t not in csp_tickers and t not in share_tickers]
            random.shuffle(candidates)

            for ticker in candidates:
                if current_margin_used >= total_equity_est * margin_cap:
                    break
                px = price_map.get((ticker, date))
                sigma = sigma_map.get((ticker, date))
                if px is None or sigma is None or px < min_price or px > max_price:
                    continue
                margin_needed = px * 100 * MARGIN_REQ_PCT
                per_name_limit = total_equity_est * per_name_pct
                if margin_needed > per_name_limit:
                    continue
                if current_margin_used + margin_needed > total_equity_est * margin_cap:
                    continue
                if cash < margin_needed:
                    continue
                T = dte_target / 365.0
                strike = find_strike(px, sigma, T, put_delta)
                premium = bs_price(px, strike, T, sigma, kind="put")
                slip = max(premium * SLIPPAGE_FRAC, SLIPPAGE_MIN)
                premium = max(premium - slip, 0.01)
                if premium < 0.10:
                    continue
                csps.append(CSP(
                    ticker=ticker, open_date=date, strike=strike,
                    premium=premium, dte=dte_target,
                    expiry=date + timedelta(days=dte_target),
                    margin=margin_needed
                ))
                current_margin_used += margin_needed
                cash -= margin_needed  # Reserve margin from cash
                cash += premium * 100  # Receive premium
                cash -= COST_PER_CONTRACT  # Commission

        # Daily equity
        share_val = sum(
            price_map.get((s.ticker, date), s.cost_basis) * s.shares for s in shares
        )
        csp_margin = sum(c.margin for c in csps)
        equity = cash + share_val + csp_margin
        daily_equity.append({'date': date, 'equity': equity, 'n_csps': len(csps),
                            'n_shares': len(shares), 'cash': cash})

    eq_df = pd.DataFrame(daily_equity)
    if len(eq_df) < 10:
        return None, eq_df

    eq_df['ret'] = eq_df['equity'].pct_change()
    eq_df = eq_df.dropna(subset=['ret'])

    total_ret = (eq_df['equity'].iloc[-1] / starting_cash - 1) * 100
    years = len(eq_df) / 252
    cagr = ((eq_df['equity'].iloc[-1] / starting_cash) ** (1/max(years, 0.1)) - 1) * 100
    mu = eq_df['ret'].mean() * 252
    std = eq_df['ret'].std() * np.sqrt(252)
    sharpe = mu / std if std > 0 else 0
    downside = eq_df['ret'][eq_df['ret'] < 0].std() * np.sqrt(252)
    sortino = mu / downside if downside > 0 else 0
    cummax = eq_df['equity'].cummax()
    dd = (eq_df['equity'] / cummax - 1)
    max_dd = dd.min() * 100
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    daily_wr = (eq_df['ret'] > 0).mean()
    pos_days = eq_df['ret'][eq_df['ret'] > 0].sum()
    neg_days = abs(eq_df['ret'][eq_df['ret'] < 0].sum())
    pf = pos_days / neg_days if neg_days > 0 else 999

    # Regime
    eq_df['bear'] = eq_df['date'].map(spy_bear).fillna(0).astype(int)
    bull_ret = eq_df[eq_df['bear'] == 0]['ret']
    bear_ret = eq_df[eq_df['bear'] == 1]['ret']
    bull_sharpe = (bull_ret.mean() * 252) / (bull_ret.std() * np.sqrt(252)) if len(bull_ret) > 20 else 0
    bear_sharpe = (bear_ret.mean() * 252) / (bear_ret.std() * np.sqrt(252)) if len(bear_ret) > 20 else 0

    return {
        'total_return_pct': round(total_ret, 1),
        'cagr_pct': round(cagr, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_dd_pct': round(max_dd, 1),
        'calmar': round(calmar, 2),
        'daily_wr': round(daily_wr, 3),
        'profit_factor': round(pf, 2),
        'final_equity': round(eq_df['equity'].iloc[-1], 2),
        'n_days': len(eq_df),
        'n_trades': len(trades),
        'bull_sharpe': round(bull_sharpe, 2),
        'bear_sharpe': round(bear_sharpe, 2),
    }, eq_df


def main():
    from wheel_universe_v3_expand import load_all_data

    log.info("=" * 70)
    log.info("WHEEL WALK-FORWARD OUT-OF-SAMPLE VALIDATION")
    log.info("=" * 70)

    # Load data
    log.info("Loading price data...")
    prices, spy_regime, sector_map = load_all_data(start_date="2019-01-01")
    tickers = [t for t in prices['ticker'].unique() if t not in ('SPY', 'VIX', '^VIX')]

    # Filter by price
    last_prices = prices.groupby('ticker')['close'].last()
    tickers = [t for t in tickers if t in last_prices.index and 10 <= last_prices[t] <= 500]
    log.info(f"Universe: {len(tickers)} tickers")

    # Build lookup maps (vectorized for speed)
    log.info("Building price/vol lookup maps...")
    price_map = {}
    sigma_map = {}
    for ticker in tickers:
        tdf = prices[prices['ticker'] == ticker]
        for _, row in tdf.iterrows():
            key = (ticker, row['date'])
            price_map[key] = row['close']
            if 'sigma' in row.index and not pd.isna(row.get('sigma', np.nan)):
                sigma_map[key] = row['sigma']

    # Also add SPY for regime
    spy_df = prices[prices['ticker'] == 'SPY']
    for _, row in spy_df.iterrows():
        price_map[('SPY', row['date'])] = row['close']

    spy_bear = dict(zip(spy_regime['date'], spy_regime['bear']))
    all_dates = sorted(prices['date'].unique())

    log.info(f"Price map: {len(price_map)} entries, date range: {all_dates[0]} to {all_dates[-1]}")

    # ========== TEST 1: Per-Year Performance (fixed best params) ==========
    log.info("\n" + "=" * 70)
    log.info("TEST 1: PER-YEAR PERFORMANCE (fixed params: d=0.30, DTE=14, margin=40%, PT=65%)")
    log.info("=" * 70)

    year_results = {}
    for year in range(2019, 2027):
        year_dates = [d for d in all_dates
                      if pd.Timestamp(d).year == year]
        if len(year_dates) < 20:
            continue

        log.info(f"\n  Running {year} ({len(year_dates)} days)...")
        t0 = time.time()
        result, eq = run_wheel_period(
            price_map, sigma_map, spy_bear, year_dates, tickers,
            starting_cash=100_000, put_delta=0.30, dte_target=14,
            margin_cap=0.40, per_name_pct=0.03, profit_take=0.65,
        )
        elapsed = time.time() - t0

        if result is not None:
            year_results[year] = result
            log.info(f"  {year}: CAGR={result['cagr_pct']:+.1f}% Sharpe={result['sharpe']:.2f} "
                     f"MaxDD={result['max_dd_pct']:.1f}% WR={result['daily_wr']:.3f} "
                     f"PF={result['profit_factor']:.2f} ({elapsed:.0f}s)")
        else:
            log.info(f"  {year}: insufficient data")

    # ========== TEST 2: Walk-Forward (expanding train, 1-year OOS) ==========
    log.info("\n" + "=" * 70)
    log.info("TEST 2: WALK-FORWARD (train on expanding window, test on next year)")
    log.info("Params tested per window: delta in {0.25, 0.30, 0.35}, PT in {0.50, 0.65, 0.80}")
    log.info("=" * 70)

    param_grid = [
        {'put_delta': d, 'profit_take': pt}
        for d in [0.25, 0.30, 0.35]
        for pt in [0.50, 0.65, 0.80]
    ]

    wf_results = []
    for test_year in range(2022, 2027):
        train_dates = [d for d in all_dates if pd.Timestamp(d).year < test_year]
        test_dates = [d for d in all_dates if pd.Timestamp(d).year == test_year]

        if len(train_dates) < 100 or len(test_dates) < 20:
            continue

        log.info(f"\n  --- Window: Train 2019-{test_year-1}, Test {test_year} ---")
        log.info(f"  Train days: {len(train_dates)}, Test days: {len(test_dates)}")

        # Find best params on train set
        best_train_sharpe = -999
        best_params = None
        train_results_all = []

        for params in param_grid:
            result, _ = run_wheel_period(
                price_map, sigma_map, spy_bear, train_dates, tickers,
                starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
                dte_target=14, **params,
            )
            if result is not None:
                train_results_all.append({**params, **result})
                if result['sharpe'] > best_train_sharpe:
                    best_train_sharpe = result['sharpe']
                    best_params = params

        if best_params is None:
            continue

        log.info(f"  Best train params: delta={best_params['put_delta']}, "
                 f"PT={best_params['profit_take']} → Sharpe={best_train_sharpe:.2f}")

        # Run best params on OOS test year
        oos_result, oos_eq = run_wheel_period(
            price_map, sigma_map, spy_bear, test_dates, tickers,
            starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
            dte_target=14, **best_params,
        )

        # Also run fixed params (d=0.30, PT=0.65) on test year for comparison
        fixed_result, _ = run_wheel_period(
            price_map, sigma_map, spy_bear, test_dates, tickers,
            starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
            dte_target=14, put_delta=0.30, profit_take=0.65,
        )

        if oos_result is not None:
            wf_entry = {
                'test_year': test_year,
                'train_end': test_year - 1,
                'best_delta': best_params['put_delta'],
                'best_pt': best_params['profit_take'],
                'train_sharpe': round(best_train_sharpe, 2),
                'oos_sharpe': oos_result['sharpe'],
                'oos_cagr': oos_result['cagr_pct'],
                'oos_max_dd': oos_result['max_dd_pct'],
                'oos_wr': oos_result['daily_wr'],
                'oos_pf': oos_result['profit_factor'],
                'oos_sortino': oos_result['sortino'],
                'fixed_sharpe': fixed_result['sharpe'] if fixed_result else None,
                'fixed_cagr': fixed_result['cagr_pct'] if fixed_result else None,
                'degradation': round(best_train_sharpe - oos_result['sharpe'], 2),
            }
            wf_results.append(wf_entry)

            log.info(f"  OOS {test_year}: Sharpe={oos_result['sharpe']:.2f} "
                     f"CAGR={oos_result['cagr_pct']:+.1f}% MaxDD={oos_result['max_dd_pct']:.1f}%")
            log.info(f"  Fixed params OOS: Sharpe={fixed_result['sharpe']:.2f}" if fixed_result else "")
            log.info(f"  Train→OOS degradation: {wf_entry['degradation']:.2f}")

    # ========== SUMMARY ==========
    log.info("\n" + "=" * 70)
    log.info("WALK-FORWARD SUMMARY")
    log.info("=" * 70)

    log.info(f"\n{'Year':>6} {'Delta':>6} {'PT':>5} {'Train':>7} {'OOS':>7} {'Fixed':>7} {'Degrad':>7} {'OOS DD':>7}")
    log.info("-" * 60)
    for wf in wf_results:
        log.info(f"{wf['test_year']:>6} {wf['best_delta']:>6.2f} {wf['best_pt']:>5.2f} "
                 f"{wf['train_sharpe']:>7.2f} {wf['oos_sharpe']:>7.2f} "
                 f"{wf['fixed_sharpe']:>7.2f} {wf['degradation']:>7.2f} "
                 f"{wf['oos_max_dd']:>6.1f}%")

    if wf_results:
        avg_oos_sharpe = np.mean([w['oos_sharpe'] for w in wf_results])
        avg_fixed_sharpe = np.mean([w['fixed_sharpe'] for w in wf_results if w['fixed_sharpe'] is not None])
        avg_degradation = np.mean([w['degradation'] for w in wf_results])
        log.info(f"\nAvg OOS Sharpe: {avg_oos_sharpe:.2f}")
        log.info(f"Avg Fixed-Param Sharpe: {avg_fixed_sharpe:.2f}")
        log.info(f"Avg Train→OOS degradation: {avg_degradation:.2f}")
        log.info(f"OOS vs Fixed: {'Optimized better' if avg_oos_sharpe > avg_fixed_sharpe else 'Fixed params just as good (not overfit)'}")

    log.info(f"\nPer-Year (fixed params):")
    log.info(f"{'Year':>6} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'WR':>6} {'PF':>6}")
    log.info("-" * 45)
    for year, r in sorted(year_results.items()):
        log.info(f"{year:>6} {r['cagr_pct']:>6.1f}% {r['sharpe']:>7.2f} {r['max_dd_pct']:>6.1f}% "
                 f"{r['daily_wr']:>5.3f} {r['profit_factor']:>5.2f}")

    # Consistency check
    positive_years = sum(1 for r in year_results.values() if r['cagr_pct'] > 0)
    total_years = len(year_results)
    log.info(f"\nPositive years: {positive_years}/{total_years}")

    # Save results
    all_output = {
        'per_year': {str(k): v for k, v in year_results.items()},
        'walk_forward': wf_results,
        'summary': {
            'avg_oos_sharpe': round(avg_oos_sharpe, 2) if wf_results else None,
            'avg_fixed_sharpe': round(avg_fixed_sharpe, 2) if wf_results else None,
            'avg_degradation': round(avg_degradation, 2) if wf_results else None,
            'positive_years': positive_years,
            'total_years': total_years,
        }
    }

    out_file = OUT_DIR / "walkforward_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_output, f, indent=2, default=str)
    log.info(f"\nResults saved to {out_file}")

    return all_output


if __name__ == "__main__":
    main()
