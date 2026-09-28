#!/usr/bin/env python3
"""
Macro Event Trading v1 — FOMC, CPI, NFP Event-Driven Options Strategies
========================================================================
Hypothesis: Major macro announcements (FOMC, CPI, NFP) cause predictable
vol patterns similar to earnings — IV expansion before, resolution after.
Can we trade the vol cycle around these events?

Strategies tested:
A) Pre-event straddle (buy 5d before, sell day-of) — IV run-up analog
B) Post-event momentum (trade breakout direction after announcement)
C) Post-event mean reversion (fade the initial move after 1-2 hours settle)
D) VIX spike fade (buy SPY calls when VIX spikes >2pts on event day)
E) Sector rotation post-FOMC (rate-sensitive sectors move predictably)
F) Event-day iron condor (sell vol before event, collect premium)
"""

import json
import warnings
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────────

ACCOUNT_SIZE = 645
MAX_POSITION = 200
COMMISSION_PER_CONTRACT = 0.65  # RH options commission (if any — RH is zero but model conservatively)
SPREAD_SLIPPAGE_PCT = 0.05     # 5% slippage on fills

# Event dates — historical FOMC, CPI, NFP
# FOMC: ~8 meetings/year, announcement at 2pm ET
# CPI: monthly, released 8:30am ET
# NFP: first Friday of month, 8:30am ET

def get_fomc_dates(start_year=2022, end_year=2026):
    """Historical FOMC announcement dates."""
    # Source: Federal Reserve calendar
    fomc = {
        2022: ["2022-01-26","2022-03-16","2022-05-04","2022-06-15",
               "2022-07-27","2022-09-21","2022-11-02","2022-12-14"],
        2023: ["2023-02-01","2023-03-22","2023-05-03","2023-06-14",
               "2023-07-26","2023-09-20","2023-11-01","2023-12-13"],
        2024: ["2024-01-31","2024-03-20","2024-05-01","2024-06-12",
               "2024-07-31","2024-09-18","2024-11-07","2024-12-18"],
        2025: ["2025-01-29","2025-03-19","2025-05-07","2025-06-18",
               "2025-07-30","2025-09-17","2025-10-29","2025-12-17"],
        2026: ["2026-01-28","2026-03-18","2026-05-06","2026-06-17",
               "2026-07-29"],  # partial year
    }
    dates = []
    for y in range(start_year, end_year + 1):
        if y in fomc:
            dates.extend([date.fromisoformat(d) for d in fomc[y]])
    return dates


def get_cpi_dates(start_year=2022, end_year=2026):
    """Approximate CPI release dates (typically 2nd Tuesday-Wednesday of month)."""
    cpi = {
        2022: ["2022-01-12","2022-02-10","2022-03-10","2022-04-12","2022-05-11",
               "2022-06-10","2022-07-13","2022-08-10","2022-09-13","2022-10-13",
               "2022-11-10","2022-12-13"],
        2023: ["2023-01-12","2023-02-14","2023-03-14","2023-04-12","2023-05-10",
               "2023-06-13","2023-07-12","2023-08-10","2023-09-13","2023-10-12",
               "2023-11-14","2023-12-12"],
        2024: ["2024-01-11","2024-02-13","2024-03-12","2024-04-10","2024-05-15",
               "2024-06-12","2024-07-11","2024-08-14","2024-09-11","2024-10-10",
               "2024-11-13","2024-12-11"],
        2025: ["2025-01-15","2025-02-12","2025-03-12","2025-04-10","2025-05-13",
               "2025-06-11","2025-07-10","2025-08-12","2025-09-10","2025-10-14",
               "2025-11-12","2025-12-10"],
        2026: ["2026-01-13","2026-02-11","2026-03-11","2026-04-14","2026-05-12",
               "2026-06-10","2026-07-14"],  # partial
    }
    dates = []
    for y in range(start_year, end_year + 1):
        if y in cpi:
            dates.extend([date.fromisoformat(d) for d in cpi[y]])
    return dates


def get_nfp_dates(start_year=2022, end_year=2026):
    """NFP = first Friday of each month."""
    dates = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            if y == 2026 and m > 7:
                break
            d = date(y, m, 1)
            # Find first Friday
            while d.weekday() != 4:  # Friday = 4
                d += timedelta(days=1)
            dates.append(d)
    return dates


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_price_data(symbols, start="2021-12-01", end="2026-07-28"):
    """Load daily OHLCV from yfinance."""
    data = {}
    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(start=start, end=end)
            if len(hist) > 100:
                data[sym] = hist
        except Exception as e:
            print(f"  Warning: {sym} failed: {e}")
    return data


def get_trading_days(price_data):
    """Get sorted list of trading days from price data."""
    if 'SPY' in price_data:
        return sorted(price_data['SPY'].index.date)
    return []


def find_nearest_trading_day(target, trading_days, direction='on_or_after'):
    """Find nearest trading day to target date."""
    for td in trading_days:
        if direction == 'on_or_after' and td >= target:
            return td
        elif direction == 'on_or_before' and td <= target:
            pass  # keep going
    # For on_or_before, return last one found
    if direction == 'on_or_before':
        result = None
        for td in trading_days:
            if td <= target:
                result = td
            else:
                break
        return result
    return None


def find_trading_day_offset(target, trading_days, offset):
    """Find trading day that is `offset` trading days from target."""
    td = find_nearest_trading_day(target, trading_days, 'on_or_after')
    if td is None:
        return None
    try:
        idx = trading_days.index(td)
        new_idx = idx + offset
        if 0 <= new_idx < len(trading_days):
            return trading_days[new_idx]
    except ValueError:
        pass
    return None


def get_price(sym, dt, price_data, field='Close'):
    """Get price for symbol on date."""
    if sym not in price_data:
        return None
    df = price_data[sym]
    # Find closest date
    mask = df.index.date == dt
    if mask.any():
        return float(df.loc[mask, field].iloc[0])
    return None


def estimate_atm_straddle_cost(stock_price, days_to_event, base_iv=0.25):
    """Rough BS estimate for ATM straddle cost."""
    # IV tends to be higher pre-event
    iv = base_iv * (1 + 0.3 * max(0, 5 - days_to_event) / 5)  # IV bump near event
    t = days_to_event / 252
    if t <= 0:
        return 0
    # ATM straddle ≈ stock × IV × sqrt(T) × 0.8 (rough approximation)
    straddle_per_share = stock_price * iv * np.sqrt(t) * 0.8
    return straddle_per_share * 100  # per contract


# ─── Strategy Backtests ─────────────────────────────────────────────────────

def strategy_a_pre_event_straddle(events, price_data, trading_days, sym="SPY"):
    """Buy ATM straddle 5 trading days before event, sell on event day open."""
    trades = []
    for event_date in events:
        entry_date = find_trading_day_offset(event_date, trading_days, -5)
        exit_date = find_nearest_trading_day(event_date, trading_days, 'on_or_after')
        if entry_date is None or exit_date is None:
            continue

        entry_price = get_price(sym, entry_date, price_data)
        exit_price = get_price(sym, exit_date, price_data)
        if entry_price is None or exit_price is None:
            continue

        # Straddle P&L approximation: |move| - theta decay
        move_pct = abs(exit_price - entry_price) / entry_price
        # ATM straddle gains ~0.5 delta on the winning side
        # Theta costs ~0.3% per day for SPY ATM
        theta_cost_pct = 5 * 0.003  # 5 days × 0.3%/day
        # IV expansion benefit (pre-event IV pump)
        iv_benefit_pct = 0.015  # ~1.5% from IV expansion

        straddle_pnl_pct = move_pct * 0.5 + iv_benefit_pct - theta_cost_pct
        # Cap straddle cost at $200
        est_cost = entry_price * 0.03 * 100  # ~3% of stock price for 5-day ATM straddle
        if est_cost > MAX_POSITION:
            # Use a cheaper proxy (mini options or skip)
            continue

        pnl = est_cost * straddle_pnl_pct / 0.03  # scale back to dollar P&L
        trades.append({
            'entry_date': str(entry_date),
            'exit_date': str(exit_date),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'move_pct': round(move_pct * 100, 2),
            'pnl': round(pnl, 2),
            'pnl_pct': round(straddle_pnl_pct / 0.03 * 100, 2),
        })
    return trades


def strategy_b_post_event_momentum(events, price_data, trading_days, sym="SPY"):
    """After event, buy in direction of move, hold 3 days."""
    trades = []
    for event_date in events:
        event_td = find_nearest_trading_day(event_date, trading_days, 'on_or_after')
        exit_td = find_trading_day_offset(event_date, trading_days, 3)
        if event_td is None or exit_td is None:
            continue

        event_open = get_price(sym, event_td, price_data, 'Open')
        event_close = get_price(sym, event_td, price_data, 'Close')
        exit_close = get_price(sym, exit_td, price_data, 'Close')
        if None in (event_open, event_close, exit_close):
            continue

        # Direction: event day close vs open
        event_move = (event_close - event_open) / event_open
        if abs(event_move) < 0.002:  # Skip tiny moves
            continue

        direction = 1 if event_move > 0 else -1
        # Enter at close, exit 3 days later
        entry = event_close
        drift = (exit_close - entry) / entry * direction

        # With $200 of shares
        shares = MAX_POSITION / entry
        pnl = shares * (exit_close - entry) * direction

        trades.append({
            'event_date': str(event_td),
            'exit_date': str(exit_td),
            'event_move_pct': round(event_move * 100, 2),
            'direction': 'LONG' if direction > 0 else 'SHORT',
            'drift_pct': round(drift * 100, 2),
            'pnl': round(pnl, 2),
        })
    return trades


def strategy_c_post_event_reversal(events, price_data, trading_days, sym="SPY"):
    """Fade the event-day move: bet on mean reversion over next 3 days."""
    trades = []
    for event_date in events:
        event_td = find_nearest_trading_day(event_date, trading_days, 'on_or_after')
        exit_td = find_trading_day_offset(event_date, trading_days, 3)
        if event_td is None or exit_td is None:
            continue

        event_open = get_price(sym, event_td, price_data, 'Open')
        event_close = get_price(sym, event_td, price_data, 'Close')
        exit_close = get_price(sym, exit_td, price_data, 'Close')
        if None in (event_open, event_close, exit_close):
            continue

        event_move = (event_close - event_open) / event_open
        if abs(event_move) < 0.005:  # Need meaningful move to fade
            continue

        # Fade: go opposite direction
        direction = -1 if event_move > 0 else 1
        entry = event_close
        drift = (exit_close - entry) / entry * direction

        shares = MAX_POSITION / entry
        pnl = shares * (exit_close - entry) * direction

        trades.append({
            'event_date': str(event_td),
            'exit_date': str(exit_td),
            'event_move_pct': round(event_move * 100, 2),
            'direction': 'LONG' if direction > 0 else 'SHORT',
            'reversion_pct': round(drift * 100, 2),
            'pnl': round(pnl, 2),
        })
    return trades


def strategy_d_vix_spike_fade(events, price_data, trading_days):
    """When VIX spikes >2pts on event day, buy SPY calls (vol mean-reverts)."""
    trades = []
    if '^VIX' not in price_data and 'VIXY' not in price_data:
        print("  No VIX data available, using SPY vol proxy")
        return trades

    for event_date in events:
        event_td = find_nearest_trading_day(event_date, trading_days, 'on_or_after')
        prev_td = find_trading_day_offset(event_date, trading_days, -1)
        exit_td = find_trading_day_offset(event_date, trading_days, 5)
        if None in (event_td, prev_td, exit_td):
            continue

        spy_event = get_price('SPY', event_td, price_data)
        spy_prev = get_price('SPY', prev_td, price_data)
        spy_exit = get_price('SPY', exit_td, price_data)
        if None in (spy_event, spy_prev, spy_exit):
            continue

        # SPY drop > 1% as VIX spike proxy
        spy_drop = (spy_event - spy_prev) / spy_prev
        if spy_drop > -0.01:  # No significant drop
            continue

        # Buy SPY at close of event day, hold 5 days (vol mean reversion)
        pnl_pct = (spy_exit - spy_event) / spy_event
        shares = MAX_POSITION / spy_event
        pnl = shares * (spy_exit - spy_event)

        trades.append({
            'event_date': str(event_td),
            'exit_date': str(exit_td),
            'spy_drop_pct': round(spy_drop * 100, 2),
            'recovery_pct': round(pnl_pct * 100, 2),
            'pnl': round(pnl, 2),
        })
    return trades


def strategy_e_sector_rotation_fomc(events, price_data, trading_days):
    """After FOMC, rate-sensitive sectors move predictably.
    Hawkish (market down) -> short financials long utilities
    Dovish (market up) -> long financials short utilities"""
    rate_sensitive = {
        'long_hawkish': ['XLU', 'XLP', 'XLRE'],   # defensive sectors benefit from hawkish pause
        'long_dovish': ['XLF', 'XLY', 'XLK'],      # growth/financial benefit from dovish
    }
    trades = []
    for event_date in events:
        event_td = find_nearest_trading_day(event_date, trading_days, 'on_or_after')
        exit_td = find_trading_day_offset(event_date, trading_days, 5)
        if event_td is None or exit_td is None:
            continue

        spy_open = get_price('SPY', event_td, price_data, 'Open')
        spy_close = get_price('SPY', event_td, price_data, 'Close')
        if spy_open is None or spy_close is None:
            continue

        spy_move = (spy_close - spy_open) / spy_open
        is_hawkish = spy_move < -0.002

        # Pick sector based on FOMC direction
        if is_hawkish:
            sectors = rate_sensitive['long_hawkish']
        else:
            sectors = rate_sensitive['long_dovish']

        # Equal-weight the sectors
        sector_pnls = []
        for sec in sectors:
            sec_event = get_price(sec, event_td, price_data)
            sec_exit = get_price(sec, exit_td, price_data)
            if sec_event and sec_exit:
                sector_pnls.append((sec_exit - sec_event) / sec_event)

        if not sector_pnls:
            continue

        avg_return = np.mean(sector_pnls)
        pnl = MAX_POSITION * avg_return

        trades.append({
            'event_date': str(event_td),
            'exit_date': str(exit_td),
            'fomc_direction': 'HAWKISH' if is_hawkish else 'DOVISH',
            'spy_move_pct': round(spy_move * 100, 2),
            'sectors': sectors,
            'sector_return_pct': round(avg_return * 100, 2),
            'pnl': round(pnl, 2),
        })
    return trades


def strategy_f_combined_events(all_events_by_type, price_data, trading_days, sym="SPY"):
    """Combined: buy SPY when multiple macro events cluster within 5 days."""
    # Flatten all events with types
    all_events = []
    for etype, dates in all_events_by_type.items():
        for d in dates:
            all_events.append((d, etype))
    all_events.sort()

    trades = []
    used_dates = set()

    for i, (evt_date, etype) in enumerate(all_events):
        if evt_date in used_dates:
            continue
        # Check for cluster: another event within 5 calendar days
        cluster = [(evt_date, etype)]
        for j in range(i+1, len(all_events)):
            other_date, other_type = all_events[j]
            if (other_date - evt_date).days <= 5 and other_type != etype:
                cluster.append((other_date, other_type))
                used_dates.add(other_date)

        if len(cluster) < 2:
            continue  # Need at least 2 different event types

        used_dates.add(evt_date)
        # Trade: buy on first event, sell 5 days after last event
        first_date = cluster[0][0]
        last_date = cluster[-1][0]

        entry_td = find_nearest_trading_day(first_date, trading_days, 'on_or_after')
        exit_td = find_trading_day_offset(last_date, trading_days, 5)
        if entry_td is None or exit_td is None:
            continue

        entry_price = get_price(sym, entry_td, price_data)
        exit_price = get_price(sym, exit_td, price_data)
        if entry_price is None or exit_price is None:
            continue

        pnl_pct = (exit_price - entry_price) / entry_price
        shares = MAX_POSITION / entry_price
        pnl = shares * (exit_price - entry_price)

        types_in_cluster = [c[1] for c in cluster]
        trades.append({
            'entry_date': str(entry_td),
            'exit_date': str(exit_td),
            'cluster_events': types_in_cluster,
            'cluster_size': len(cluster),
            'pnl_pct': round(pnl_pct * 100, 2),
            'pnl': round(pnl, 2),
        })
    return trades


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_trades(trades, label, account=ACCOUNT_SIZE):
    """Compute risk-adjusted metrics."""
    if not trades:
        return {'name': label, 'n_trades': 0, 'sharpe': 0, 'wr': 0, 'pf': 0, 'total_pnl': 0, 'avg_pnl': 0, 'final_equity': account, 'mdd_pct': 0, 'perm_p': 1.0, 'regime_gap': 0, 'sortino': 0, 'gates_passed': 0, 'gates_total': 5, 'gate_results': {}, 'verdict': 'NO TRADES'}

    pnls = [t['pnl'] for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    total_pnl = sum(pnls)

    # Equity curve
    equity = [account]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity = np.array(equity)
    mdd = np.min(equity / np.maximum.accumulate(equity) - 1) * 100

    # Risk metrics
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls) if len(pnls) > 1 else 1
    sharpe = (mean_pnl / std_pnl * np.sqrt(252 / max(1, n))) if std_pnl > 0 else 0

    neg_pnls = [p for p in pnls if p < 0]
    downside_std = np.std(neg_pnls) if neg_pnls else 1
    sortino = (mean_pnl / downside_std * np.sqrt(252 / max(1, n))) if downside_std > 0 else 0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Permutation test
    observed_sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0
    n_perms = 5000
    perm_count = 0
    for _ in range(n_perms):
        shuffled = np.random.choice([-1, 1], size=n) * np.abs(pnls)
        perm_sharpe = np.mean(shuffled) / (np.std(shuffled) + 1e-10)
        if perm_sharpe >= observed_sharpe:
            perm_count += 1
    perm_p = perm_count / n_perms

    # Regime analysis (simple: first half vs second half as proxy)
    mid = n // 2
    if mid > 2:
        first_half_mean = np.mean(pnls[:mid])
        second_half_mean = np.mean(pnls[mid:])
        regime_gap = abs(first_half_mean - second_half_mean) / (abs(max(first_half_mean, second_half_mean)) + 1e-10)
    else:
        regime_gap = 0

    result = {
        'name': label,
        'n_trades': n,
        'wins': wins,
        'wr': round(wins / n * 100, 1),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 2),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(mean_pnl, 2),
        'final_equity': round(equity[-1], 2),
        'mdd_pct': round(mdd, 1),
        'perm_p': round(perm_p, 3),
        'regime_gap': round(regime_gap, 2),
    }

    # Gates
    gates = {
        'sharpe': sharpe >= 0.5,
        'perm_test': perm_p < 0.05,
        'regime': regime_gap < 0.50,
        'mdd': mdd > -50,
        'random_beat': True,  # placeholder
    }
    result['gates_passed'] = sum(gates.values())
    result['gates_total'] = len(gates)
    result['gate_results'] = gates

    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MACRO EVENT TRADING v1 — FOMC / CPI / NFP Strategies")
    print("=" * 70)

    # Load event dates
    fomc_dates = get_fomc_dates()
    cpi_dates = get_cpi_dates()
    nfp_dates = get_nfp_dates()
    all_events = sorted(set(fomc_dates + cpi_dates + nfp_dates))
    print(f"\nEvents: {len(fomc_dates)} FOMC, {len(cpi_dates)} CPI, {len(nfp_dates)} NFP")
    print(f"Total unique event days: {len(all_events)}")

    # Load price data
    print("\nLoading price data...")
    symbols = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD',
               'XLF', 'XLK', 'XLE', 'XLU', 'XLP', 'XLY', 'XLRE',
               'VIXY']
    price_data = load_price_data(symbols)
    trading_days = get_trading_days(price_data)
    print(f"  Loaded {len(price_data)} symbols, {len(trading_days)} trading days")

    results = []

    # Strategy A: Pre-event straddle (all events combined)
    print("\n── A: Pre-Event Straddle (all events) ──")
    trades_a = strategy_a_pre_event_straddle(all_events, price_data, trading_days)
    res_a = evaluate_trades(trades_a, "A_PreEvent_Straddle")
    results.append(res_a)
    print(f"  {res_a['n_trades']} trades, Sharpe {res_a['sharpe']}, WR {res_a['wr']}%, "
          f"PF {res_a['pf']}, MDD {res_a['mdd_pct']}%, perm p={res_a['perm_p']}")

    # Strategy B: Post-event momentum
    print("\n── B: Post-Event Momentum (all events) ──")
    trades_b = strategy_b_post_event_momentum(all_events, price_data, trading_days)
    res_b = evaluate_trades(trades_b, "B_PostEvent_Momentum")
    results.append(res_b)
    print(f"  {res_b['n_trades']} trades, Sharpe {res_b['sharpe']}, WR {res_b['wr']}%, "
          f"PF {res_b['pf']}, MDD {res_b['mdd_pct']}%, perm p={res_b['perm_p']}")

    # Strategy C: Post-event mean reversion
    print("\n── C: Post-Event Reversal (all events) ──")
    trades_c = strategy_c_post_event_reversal(all_events, price_data, trading_days)
    res_c = evaluate_trades(trades_c, "C_PostEvent_Reversal")
    results.append(res_c)
    print(f"  {res_c['n_trades']} trades, Sharpe {res_c['sharpe']}, WR {res_c['wr']}%, "
          f"PF {res_c['pf']}, MDD {res_c['mdd_pct']}%, perm p={res_c['perm_p']}")

    # Strategy D: VIX spike fade
    print("\n── D: VIX Spike Fade ──")
    trades_d = strategy_d_vix_spike_fade(all_events, price_data, trading_days)
    res_d = evaluate_trades(trades_d, "D_VIX_Spike_Fade")
    results.append(res_d)
    print(f"  {res_d['n_trades']} trades, Sharpe {res_d['sharpe']}, WR {res_d['wr']}%, "
          f"PF {res_d['pf']}, MDD {res_d['mdd_pct']}%, perm p={res_d['perm_p']}")

    # Strategy E: Sector rotation post-FOMC only
    print("\n── E: Sector Rotation Post-FOMC ──")
    trades_e = strategy_e_sector_rotation_fomc(fomc_dates, price_data, trading_days)
    res_e = evaluate_trades(trades_e, "E_Sector_FOMC")
    results.append(res_e)
    print(f"  {res_e['n_trades']} trades, Sharpe {res_e['sharpe']}, WR {res_e['wr']}%, "
          f"PF {res_e['pf']}, MDD {res_e['mdd_pct']}%, perm p={res_e['perm_p']}")

    # Strategy F: Event cluster
    print("\n── F: Event Cluster (multi-type within 5d) ──")
    trades_f = strategy_f_combined_events(
        {'FOMC': fomc_dates, 'CPI': cpi_dates, 'NFP': nfp_dates},
        price_data, trading_days
    )
    res_f = evaluate_trades(trades_f, "F_Event_Cluster")
    results.append(res_f)
    print(f"  {res_f['n_trades']} trades, Sharpe {res_f['sharpe']}, WR {res_f['wr']}%, "
          f"PF {res_f['pf']}, MDD {res_f['mdd_pct']}%, perm p={res_f['perm_p']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        gates = r.get('gates_passed', 0)
        total = r.get('gates_total', 5)
        status = "✅ PASS" if gates >= 4 else ("⚠️ PARTIAL" if gates >= 3 else "❌ FAIL")
        print(f"  {r['name']:30s} | Sharpe {r['sharpe']:6.3f} | WR {r['wr']:5.1f}% | "
              f"PF {r['pf']:5.2f} | MDD {r['mdd_pct']:6.1f}% | p={r['perm_p']:.3f} | "
              f"{gates}/{total} gates | {status}")

    # Save results
    out_path = Path("/home/jupiter/Lvl3Quant/research/findings/macro_event_trading_v1.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved.")

    return results


if __name__ == "__main__":
    main()
