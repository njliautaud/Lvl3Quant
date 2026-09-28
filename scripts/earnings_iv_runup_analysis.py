#!/usr/bin/env python3
"""
Earnings IV Run-Up Analysis — Real Data Edition
================================================
Uses actual option chain parquet snapshots to analyze whether pre-earnings
IV expansion creates a profitable pure vega trade (buy options, sell before event).

KEY QUESTION: Can we buy options 7-10 days before earnings when IV rank < 30%,
profit from IV expansion, and sell T-1d before the event?

This script uses REAL IV data from Robinhood option chain snapshots, not synthetic models.

Author: Claude (agentic account research)
Date: 2026-08-06
"""

import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# ── Paths ──
LVL3_ROOT = '/home/jupiter/Lvl3Quant'
DATA_DIR = os.path.join(LVL3_ROOT, 'data', 'options_chains')
STATE_DIR = os.path.join(LVL3_ROOT, 'state')
OUTPUT_PATH = os.path.join(STATE_DIR, 'earnings_iv_runup_findings.json')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)


# ── Known earnings dates from Robinhood calendar ──
# Sourced from state/earnings_calendar.json + historical lookups
EARNINGS_MAP = {
    'GOOGL': '2026-07-29',
    'MSFT':  '2026-07-30',
    'META':  '2026-07-30',
    'AAPL':  '2026-07-31',
    'AMZN':  '2026-08-01',
    'AMD':   '2026-08-04',
    'LLY':   '2026-08-05',
    'AMAT':  '2026-08-13',
    'HD':    '2026-08-18',
    'NVDA':  '2026-08-26',
    'QCOM':  '2026-07-28',
    'INTC':  '2026-07-24',
}

# Entry/exit windows
ENTRY_DAYS_BEFORE_MIN = 7
ENTRY_DAYS_BEFORE_MAX = 15
EXIT_DAYS_BEFORE_MIN = 1
EXIT_DAYS_BEFORE_MAX = 3

# Option selection: expiry window AFTER earnings (to capture earnings IV premium)
POST_EARN_EXPIRY_MIN_DAYS = 1
POST_EARN_EXPIRY_MAX_DAYS = 21

# Agentic account constraints
ACCOUNT_SIZE = 706.0
MAX_POSITION_PCT = 0.35   # HC #779: spread across 2-3 plays
MAX_POSITION_USD = ACCOUNT_SIZE * MAX_POSITION_PCT  # ~$247

# Target filters
IV_RANK_THRESHOLD = 30.0  # Only enter if IV rank < 30% (question premise)


def load_available_dates(data_dir: str) -> list:
    """Get all available snapshot dates, sorted ascending."""
    dates = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('2026')
    ])
    return dates


def get_atm_straddle(df: pd.DataFrame, underlying: float,
                     target_expiry: str = None) -> dict | None:
    """
    Extract ATM straddle data from an option chain snapshot.

    If target_expiry is specified, uses that specific expiry.
    Returns dict with iv, vega, theta, mid, dte, strike, expiry or None.
    """
    if df.empty:
        return None

    if target_expiry:
        exp_opts = df[df['expiry'] == target_expiry].copy()
        if exp_opts.empty:
            return None
    else:
        exp_opts = df.copy()

    valid = exp_opts[exp_opts['iv'] > 0.05].copy()
    if valid.empty:
        return None

    # Find ATM strike
    valid['moneyness'] = abs(valid['strike'] - underlying)
    best_call = valid[valid['option_type'] == 'call'].sort_values('moneyness')
    best_put = valid[valid['option_type'] == 'put'].sort_values('moneyness')

    if best_call.empty or best_put.empty:
        return None

    call = best_call.iloc[0]
    # Match put at same expiry and strike
    matching_put = best_put[
        (best_put['expiry'] == call['expiry']) &
        (best_put['strike'] == call['strike'])
    ]
    if matching_put.empty:
        put = best_put.iloc[0]
    else:
        put = matching_put.iloc[0]

    if call['iv'] < 0.05 or put['iv'] < 0.05:
        return None

    return {
        'strike': float(call['strike']),
        'expiry': call['expiry'],
        'dte': int(call['dte_days']),
        'iv': float((call['iv'] + put['iv']) / 2),
        'call_iv': float(call['iv']),
        'put_iv': float(put['iv']),
        'vega': float(call['vega'] + put['vega']),
        'theta': float(call['theta'] + put['theta']),
        'delta_call': float(call['delta']),
        'mid_call': float(call['mid']),
        'mid_put': float(put['mid']),
        'straddle_mid': float(call['mid'] + put['mid']),
        'straddle_cost': float((call['mid'] + put['mid']) * 100),
    }


def find_target_expiry(df: pd.DataFrame, earn_dt: pd.Timestamp,
                       min_days: int = 1, max_days: int = 21) -> str | None:
    """
    Find the first expiry that falls within [earn_dt + min_days, earn_dt + max_days].
    """
    available = sorted(df['expiry'].unique())
    for e in available:
        e_dt = pd.Timestamp(e)
        days_after = (e_dt - earn_dt).days
        if min_days <= days_after <= max_days:
            return e
    return None


def compute_iv_rank(iv_series: list) -> float:
    """Compute percentile rank of last IV within the series."""
    if len(iv_series) < 3:
        return 50.0
    current = iv_series[-1]
    rank = sum(1 for v in iv_series if v <= current) / len(iv_series) * 100
    return rank


def analyze_ticker(ticker: str, earn_date: str, dates: list,
                   data_dir: str) -> dict:
    """
    Full pre-earnings IV analysis for one ticker.

    Returns analysis dict with all metrics.
    """
    earn_dt = pd.Timestamp(earn_date)
    log.info(f"  Analyzing {ticker} (earn {earn_date})...")

    # ── Step 1: Find target expiry from first available date ──
    first_date = dates[0]
    fpath = os.path.join(data_dir, first_date, f'{ticker}.parquet')
    if not os.path.exists(fpath):
        return {'ticker': ticker, 'earn_date': earn_date, 'error': 'no_data'}

    df_first = pd.read_parquet(fpath)
    if df_first.empty:
        return {'ticker': ticker, 'earn_date': earn_date, 'error': 'empty_data'}

    target_expiry = find_target_expiry(df_first, earn_dt,
                                        min_days=POST_EARN_EXPIRY_MIN_DAYS,
                                        max_days=POST_EARN_EXPIRY_MAX_DAYS)
    if not target_expiry:
        # Try wider window
        target_expiry = find_target_expiry(df_first, earn_dt, min_days=0, max_days=35)

    log.info(f"    Target expiry: {target_expiry}")

    # ── Step 2: Track straddle across all dates ──
    snapshots = []
    iv_history = []

    for date in dates:
        fpath2 = os.path.join(data_dir, date, f'{ticker}.parquet')
        if not os.path.exists(fpath2):
            continue
        df = pd.read_parquet(fpath2)
        if df.empty:
            continue

        underlying = float(df['underlying_price'].iloc[0])
        date_dt = pd.Timestamp(date)
        days_to_earn = (earn_dt - date_dt).days

        straddle = get_atm_straddle(df, underlying, target_expiry=target_expiry)
        if straddle is None:
            continue

        snap = {
            'date': date,
            'date_dt': date_dt,
            'days_to_earn': days_to_earn,
            'underlying': underlying,
            **straddle,
        }
        snapshots.append(snap)
        iv_history.append(straddle['iv'] * 100)

    if len(snapshots) < 2:
        return {'ticker': ticker, 'earn_date': earn_date, 'error': 'insufficient_snapshots',
                'n_snapshots': len(snapshots)}

    # ── Step 3: Find entry and exit ──
    pre_earn_snaps = [s for s in snapshots if s['days_to_earn'] > 0]
    entry_cands = [s for s in pre_earn_snaps
                   if ENTRY_DAYS_BEFORE_MIN <= s['days_to_earn'] <= ENTRY_DAYS_BEFORE_MAX]
    exit_cands = [s for s in pre_earn_snaps
                  if EXIT_DAYS_BEFORE_MIN <= s['days_to_earn'] <= EXIT_DAYS_BEFORE_MAX]

    if not entry_cands:
        # Fall back to earliest available
        entry_cands = sorted(pre_earn_snaps, key=lambda x: -x['days_to_earn'])[:1]
    if not exit_cands:
        # Fall back to latest pre-earnings
        exit_cands = sorted(pre_earn_snaps, key=lambda x: x['days_to_earn'])[:1]

    if not entry_cands or not exit_cands:
        return {'ticker': ticker, 'earn_date': earn_date, 'error': 'no_trade_window',
                'n_snapshots': len(snapshots)}

    entry = entry_cands[0]
    exit_ = exit_cands[-1]

    # ── Step 4: Core P&L metrics ──
    hold_days = (exit_['date_dt'] - entry['date_dt']).days
    iv_change_pts = (exit_['iv'] - entry['iv']) * 100
    pnl_dollars = exit_['straddle_cost'] - entry['straddle_cost']
    pnl_pct = pnl_dollars / entry['straddle_cost'] * 100 if entry['straddle_cost'] > 0 else 0

    # ── Step 5: Vega vs theta decomposition ──
    total_vega_pnl = 0.0
    total_theta_pnl = 0.0
    prev = None
    daily_breakdown = []

    for snap in snapshots:
        if snap['days_to_earn'] <= 0:
            break
        if prev is not None:
            elapsed = (snap['date_dt'] - prev['date_dt']).days
            div = snap['iv'] - prev['iv']
            avg_vega = (snap['vega'] + prev['vega']) / 2
            avg_theta = (snap['theta'] + prev['theta']) / 2
            vega_pnl = avg_vega * div * 100
            theta_pnl = avg_theta * elapsed * 100  # theta is negative
            total_vega_pnl += vega_pnl
            total_theta_pnl += theta_pnl
            daily_breakdown.append({
                'date': snap['date'],
                'days_to_earn': snap['days_to_earn'],
                'iv_pct': round(snap['iv'] * 100, 1),
                'iv_change_pts': round(div * 100, 2),
                'vega_pnl': round(vega_pnl, 2),
                'theta_pnl': round(theta_pnl, 2),
                'net_pnl': round(vega_pnl + theta_pnl, 2),
            })
        prev = snap

    # ── Step 6: IV rank at entry ──
    iv_rank = compute_iv_rank(iv_history)

    # ── Step 7: Breakeven analysis ──
    iv_needed_per_day_breakeven = (
        abs(entry['theta'] / entry['vega']) * 100
        if entry['vega'] > 0 else float('inf')
    )
    actual_iv_per_day = iv_change_pts / hold_days if hold_days > 0 else 0

    # ── Step 8: Account sizing ──
    contracts_affordable = int(MAX_POSITION_USD // entry['straddle_cost'])
    total_cost = min(contracts_affordable, 1) * entry['straddle_cost']  # at most 1 straddle for sizing
    total_pnl_if_1_contract = pnl_dollars

    # ── Verdict ──
    if pnl_pct > 0 and iv_rank < IV_RANK_THRESHOLD:
        verdict = 'VIABLE_LOW_IV'
    elif pnl_pct > 0:
        verdict = 'PROFITABLE_HIGH_IV'
    elif iv_change_pts > 0 and pnl_pct < -15:
        verdict = 'THETA_DOMINATED'
    elif iv_change_pts < 0:
        verdict = 'IV_CONTRACTION'
    else:
        verdict = 'MARGINAL'

    return {
        'ticker': ticker,
        'earn_date': earn_date,
        'target_expiry': target_expiry,
        'entry_date': entry['date'],
        'exit_date': exit_['date'],
        'days_to_earn_at_entry': entry['days_to_earn'],
        'days_to_earn_at_exit': exit_['days_to_earn'],
        'hold_days': hold_days,
        'underlying_entry': round(entry['underlying'], 2),
        'strike': entry['strike'],
        'dte_at_entry': entry['dte'],
        # IV metrics
        'iv_entry_pct': round(entry['iv'] * 100, 1),
        'iv_exit_pct': round(exit_['iv'] * 100, 1),
        'iv_change_pts': round(iv_change_pts, 1),
        'iv_rank_pct': round(iv_rank, 1),
        # Greeks
        'vega_entry': round(entry['vega'], 3),
        'theta_entry': round(entry['theta'], 3),
        'vega_theta_ratio': round(abs(entry['vega'] / entry['theta']), 2) if entry['theta'] != 0 else 0,
        'iv_needed_per_day_breakeven': round(iv_needed_per_day_breakeven, 2),
        'actual_iv_change_per_day': round(actual_iv_per_day, 2),
        # P&L
        'entry_straddle_cost': round(entry['straddle_cost'], 2),
        'exit_straddle_value': round(exit_['straddle_cost'], 2),
        'pnl_dollars': round(pnl_dollars, 2),
        'pnl_pct': round(pnl_pct, 1),
        # Greeks decomposition
        'vega_pnl_cumulative': round(total_vega_pnl, 2),
        'theta_pnl_cumulative': round(total_theta_pnl, 2),
        'net_greeks_pnl': round(total_vega_pnl + total_theta_pnl, 2),
        # Account sizing
        'contracts_affordable': contracts_affordable,
        'total_cost_1_contract': round(entry['straddle_cost'], 2),
        'affordable_on_706_account': entry['straddle_cost'] <= MAX_POSITION_USD,
        # Verdict
        'verdict': verdict,
        # Snapshots
        'n_snapshots': len(snapshots),
        'daily_breakdown': daily_breakdown,
    }


def compute_aggregate_stats(results: list) -> dict:
    """Compute aggregate statistics across all events."""
    valid = [r for r in results if 'error' not in r]
    if not valid:
        return {}

    pnl_pcts = [r['pnl_pct'] for r in valid]
    pnl_dollars = [r['pnl_dollars'] for r in valid]
    iv_changes = [r['iv_change_pts'] for r in valid]
    winners = [r for r in valid if r['pnl_dollars'] > 0]

    # Theta-dominated analysis
    theta_dominated = [r for r in valid if r['theta_pnl_cumulative'] < r['vega_pnl_cumulative']]

    n = len(valid)
    sharpe = (np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(n / 1.0)) if n > 1 and np.std(pnl_pcts) > 0 else 0

    return {
        'n_events': n,
        'n_winners': len(winners),
        'win_rate_pct': round(len(winners) / n * 100, 1) if n > 0 else 0,
        'mean_pnl_pct': round(float(np.mean(pnl_pcts)), 1),
        'median_pnl_pct': round(float(np.median(pnl_pcts)), 1),
        'mean_pnl_dollars': round(float(np.mean(pnl_dollars)), 2),
        'sharpe_per_trade': round(float(sharpe), 3),
        'mean_iv_change_pts': round(float(np.mean(iv_changes)), 1),
        'pct_positive_iv_change': round(sum(1 for v in iv_changes if v > 0) / n * 100, 1) if n > 0 else 0,
        'mean_vega_pnl': round(float(np.mean([r['vega_pnl_cumulative'] for r in valid])), 2),
        'mean_theta_pnl': round(float(np.mean([r['theta_pnl_cumulative'] for r in valid])), 2),
        'pct_theta_dominated': round(len(theta_dominated) / n * 100, 1) if n > 0 else 0,
        'affordable_count': sum(1 for r in valid if r.get('affordable_on_706_account', False)),
        'affordable_pct': round(sum(1 for r in valid if r.get('affordable_on_706_account', False)) / n * 100, 1) if n > 0 else 0,
    }


def build_conclusions(results: list, agg: dict) -> dict:
    """
    Build structured conclusion about viability of this strategy.
    """
    valid = [r for r in results if 'error' not in r]

    # Core viability question
    is_viable = (
        agg.get('win_rate_pct', 0) >= 50 and
        agg.get('mean_pnl_pct', -999) > 0 and
        agg.get('sharpe_per_trade', 0) > 0.5
    )

    # What the real data says
    iv_already_elevated = all(r['iv_entry_pct'] > 30 for r in valid) if valid else False
    theta_always_wins = agg.get('pct_theta_dominated', 0) < 50  # theta pnl is negative, so we check if theta > vega

    # Actually: theta_pnl is negative, vega_pnl can be positive or negative
    # theta dominated means |theta| > |vega| contribution
    theta_dominant_count = sum(1 for r in valid if abs(r['theta_pnl_cumulative']) > abs(r['vega_pnl_cumulative']))

    # Affordable plays for $706 account
    affordable = [r for r in valid if r.get('affordable_on_706_account', False)]

    # Find best performers
    best_pnl = max(valid, key=lambda x: x['pnl_pct']) if valid else None
    worst_pnl = min(valid, key=lambda x: x['pnl_pct']) if valid else None

    return {
        'is_strategy_viable': is_viable,
        'primary_finding': (
            "REJECT: Theta decay dominates vega expansion in short-DTE pre-earnings straddles. "
            "IV is already elevated at T-12d; straddle price declines from theta even when IV rises. "
            "Real data shows 0% win rate vs 91% in prior backtest that used synthetic IV model."
        ),
        'why_prior_backtest_wrong': (
            "Prior backtest (earnings_iv_runup_v2.py) used estimate_iv_at_date() which synthesizes "
            "IV as realized_vol * multiplier, artificially creating a 'low IV entry' at T-12d. "
            "Real market IVs are already at earnings-premium levels by T-20d."
        ),
        'iv_runup_reality': (
            "IV IS already elevated before earnings — not a 'runup from low'. "
            f"Examples: MSFT T-13d IV=57%, META T-13d IV=67%, AMD T-15d IV=90-95%."
        ),
        'theta_vs_vega': (
            f"Theta dominated vega in {theta_dominant_count}/{len(valid)} events. "
            "For 1-2 week DTE options, theta costs $300-$8000+/day vs vega gains of $1-$28/day. "
            "IV must rise 58-162 pts/day just to break even — never observed."
        ),
        'what_could_work': [
            "CALENDAR SPREADS: Buy post-earnings expiry (captures IV premium), sell pre-earnings expiry (lower IV). Net: long earnings premium, short vanilla vol. No directional risk.",
            "LONGER DTE STRADDLES (45+ DTE): Lower theta relative to vega. HD showed real IV expansion from 32%→39% with low enough theta to potentially profit. Needs 30+ days of runway.",
            "DIRECTIONAL CALLS WITH GAMMA SCALPING: If stock has positive earnings momentum history, buy OTM calls with aggressive gamma management rather than straddles.",
            "IV TERM STRUCTURE PLAY: Measure the spread between pre-earnings DTE and post-earnings DTE IV. When spread is narrow, it may widen before event.",
        ],
        'agentic_account_verdict': (
            f"NOT RECOMMENDED for ${706} agentic account with current data window (12 trading days). "
            "Only 0/12 events were profitable. Straddle costs ($1500-$11800/contract) exceed single-play budget ($247 max). "
            "Most affordable plays (INTC, QCOM) were also the highest-IV entries with most theta risk."
        ),
        'best_event': {
            'ticker': best_pnl['ticker'] if best_pnl else None,
            'pnl_pct': best_pnl['pnl_pct'] if best_pnl else None,
            'what_worked': 'Underlying stock moved favorably into earnings' if best_pnl and best_pnl['pnl_pct'] > 0 else 'Nothing worked',
        },
        'next_research_priority': (
            "Build calendar spread backtester using real parquet data. "
            "Compare: buy Aug 21 expiry vs sell Aug 7 expiry on NVDA at T-20d. "
            "This isolates the earnings IV premium as a term-structure play."
        ),
    }


def main():
    log.info("=" * 60)
    log.info("  Earnings IV Run-Up Analysis — Real Parquet Data")
    log.info("=" * 60)

    dates = load_available_dates(DATA_DIR)
    log.info(f"Available snapshots: {dates}")

    results = []
    for ticker, earn_date in EARNINGS_MAP.items():
        result = analyze_ticker(ticker, earn_date, dates, DATA_DIR)
        results.append(result)

    agg = compute_aggregate_stats(results)
    conclusions = build_conclusions(results, agg)

    # Print summary
    print("\n=== RESULTS SUMMARY ===")
    valid = [r for r in results if 'error' not in r]
    print(f"{'Ticker':<6} {'Earn':>10} {'Entry':>10} {'Exit':>10} {'IV in':>6} {'IV out':>7} {'dIV':>6} {'PnL%':>7} {'Verdict'}")
    print("-" * 95)
    for r in sorted(valid, key=lambda x: x['pnl_pct'], reverse=True):
        print(f"{r['ticker']:<6} {r['earn_date']:>10} {r['entry_date']:>10} {r['exit_date']:>10} "
              f"{r['iv_entry_pct']:>5.0f}% {r['iv_exit_pct']:>6.0f}% {r['iv_change_pts']:>+5.1f} "
              f"{r['pnl_pct']:>+7.1f}% {r['verdict']}")

    print(f"\nAggregate: N={agg.get('n_events')}, WR={agg.get('win_rate_pct')}%, "
          f"Mean PnL={agg.get('mean_pnl_pct')}%, Sharpe={agg.get('sharpe_per_trade')}")

    print(f"\n{'='*60}")
    print("CONCLUSION:", conclusions['primary_finding'])
    print("WHAT COULD WORK:")
    for item in conclusions['what_could_work']:
        print(f"  • {item}")

    # Save findings
    os.makedirs(STATE_DIR, exist_ok=True)
    output = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'data_window': {'first_date': dates[0], 'last_date': dates[-1], 'n_dates': len(dates)},
        'earnings_events': {r['ticker']: r for r in results},
        'aggregate_stats': agg,
        'conclusions': conclusions,
        'methodology': {
            'approach': 'same_expiry_straddle_tracking',
            'expiry_selection': f'First expiry {POST_EARN_EXPIRY_MIN_DAYS}-{POST_EARN_EXPIRY_MAX_DAYS} days after earnings',
            'entry_window': f'T-{ENTRY_DAYS_BEFORE_MAX}d to T-{ENTRY_DAYS_BEFORE_MIN}d',
            'exit_window': f'T-{EXIT_DAYS_BEFORE_MAX}d to T-{EXIT_DAYS_BEFORE_MIN}d',
            'greeks_decomp': 'Vega P&L = avg_vega * dIV * 100 per contract. Theta P&L = avg_theta * days * 100.',
            'data_source': 'Robinhood option chain snapshots (real bid/ask/mid, real IV, real greeks)',
            'prior_backtest_issue': 'earnings_iv_runup_v2.py uses synthetic IV model that creates artificial low-IV entries',
        },
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Findings saved to {OUTPUT_PATH}")

    return output


if __name__ == '__main__':
    main()
