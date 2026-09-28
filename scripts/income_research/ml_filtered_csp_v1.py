#!/usr/bin/env python3
"""
ML-Filtered Cash-Secured Put (CSP) Backtest v1
===============================================
Hypothesis: Selling puts ONLY on stocks our model predicts will outperform SPY
(high confidence) improves risk-adjusted returns vs selling puts on all available
stocks blindly.

Strategy:
- Universe: stocks in both v2 model OOT predictions AND real options chain data (52 tickers)
- Every Friday (options-like weekly settlement), evaluate candidates
- ML-filtered arm: only sell CSPs on stocks with pred_proba >= CONF_THRESHOLD
- Unfiltered arm: sell CSPs on all available stocks (control)
- Delta ~25 puts, 20-50 DTE, mid-price (Robinhood commission-free, HC #694)
- Full wheel on assignment: sell CCs until called away
- Risk controls: 200MA filter, VIX < 35 gate, max 20% capital per name, max 5 concurrent
- Hard 15% stop-loss on assigned shares

Walk-forward integrity: all predictions are OOT (from the saved walk-forward output).
The prediction model uses 60-day embargo so no label leakage.

Validation:
- HC #428 R1 regime test: |Sharpe_green - Sharpe_red| / max(...) <= 0.50
- 100-trial permutation test (shuffle NAV returns, not trades)
- Per-regime, per-year breakdown

Output: /home/jupiter/Lvl3Quant/output/ml_filtered_csp_v1/
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG
# ============================================================================

CONF_THRESHOLD_060 = 0.60     # ML filter tier 1 (~63% precision)
CONF_THRESHOLD_065 = 0.65     # ML filter tier 2 (~70% precision)
CONF_THRESHOLD_070 = 0.70     # ML filter tier 3 (~80% precision)

TARGET_DELTA = -0.25           # Target put delta
DTE_MIN = 20
DTE_MAX = 55
MAX_POSITIONS = 5              # Max concurrent CSP/wheel positions
MAX_CAPITAL_PER_POS_PCT = 0.20 # Max 20% of portfolio per position (collateral)
STOP_LOSS_PCT = 0.15           # 15% stop on assigned shares
VIX_GATE = 35.0                # No new positions when VIX > 35
INITIAL_CAPITAL = 100_000

# Paths
OOT_PRED_FILE = Path("/home/jupiter/Lvl3Quant/output/growth_research/stock_prediction/v2_relative/oot_predictions_target_excess_60d_3pct.parquet")
OPTIONS_DIR = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/options_real/chains")
PRICE_CACHE = Path("/home/jupiter/Lvl3Quant/output/growth_research/stock_prediction/cache/price_data.parquet")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/ml_filtered_csp_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_predictions():
    print("Loading ML predictions (OOT walk-forward)...")
    df = pd.read_parquet(OOT_PRED_FILE)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  {len(df):,} rows | {df['date'].min().date()} to {df['date'].max().date()} | {df['ticker'].nunique()} tickers")

    # Check precision by threshold
    base = df['target_excess_60d_3pct'].mean()
    print(f"  Base rate (60d excess >3%): {base:.3f}")
    for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        m = df[df['pred_proba'] >= t]
        if len(m) > 10:
            prec = m['target_excess_60d_3pct'].mean()
            print(f"    {t:.2f}+: N={len(m):,}, precision={prec:.3f}, lift={prec/base:.2f}x")
    return df


def load_all_options():
    """Load all put options chains into memory (per-ticker dict)."""
    print("Loading options chains...")
    frames = {}
    for fpath in sorted(OPTIONS_DIR.glob("*.parquet")):
        ticker = fpath.stem
        df = pd.read_parquet(fpath)
        df['date'] = pd.to_datetime(df['date'])
        df['expiration'] = pd.to_datetime(df['expiration'])
        puts = df[df['type'] == 'p'].copy()
        if len(puts) > 0:
            frames[ticker] = puts
    print(f"  Loaded {len(frames)} tickers")
    return frames


def load_prices():
    """Load price data, compute 200MA, return price_lookup dict."""
    print("Loading price data and computing 200MA...")
    raw = pd.read_parquet(PRICE_CACHE)
    raw['date'] = pd.to_datetime(raw['date'])

    price_lookup = {}
    ma200_lookup = {}
    for ticker, grp in raw.groupby('ticker'):
        grp = grp.sort_values('date').set_index('date')
        price_lookup[ticker] = grp['close']
        ma200_lookup[ticker] = grp['close'].rolling(200, min_periods=150).mean()
    print(f"  {len(price_lookup)} tickers")
    return price_lookup, ma200_lookup


def load_vix():
    """Load VIX from yfinance with fallback."""
    print("Loading VIX...")
    try:
        import yfinance as yf
        raw = yf.download("^VIX", start="2019-01-01", end="2026-12-31", progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            vix = raw['Close'].iloc[:, 0]
        else:
            vix = raw['Close']
        vix.index = pd.to_datetime(vix.index)
        vix = vix.squeeze()
        print(f"  VIX loaded {vix.index.min().date()} to {vix.index.max().date()}")
        return vix
    except Exception as e:
        print(f"  VIX load failed ({e}), using flat VIX=20")
        return pd.Series(20.0, index=pd.date_range("2019-01-01", "2026-12-31", freq='B'))


# ============================================================================
# HELPERS
# ============================================================================

def get_price_on_date(series, date, tolerance_days=5):
    """Get nearest price on or before date, within tolerance."""
    mask = series.index <= date
    if not mask.any():
        return None
    nearest = series.index[mask][-1]
    if (date - nearest).days > tolerance_days:
        return None
    val = series.loc[nearest]
    return float(val) if not pd.isna(val) else None


def get_scalar_on_date(series, date, tolerance_days=5):
    """Same as get_price_on_date — get scalar from series."""
    return get_price_on_date(series, date, tolerance_days)


def find_put(puts_df, trade_date, stock_price=None, target_delta=TARGET_DELTA,
             dte_min=DTE_MIN, dte_max=DTE_MAX):
    """
    Find best put to sell near target_delta on trade_date.
    stock_price: current adjusted price — used to filter out split-artifact strikes.
    IMPORTANT: options chains may have unadjusted (nominal) strikes while price_cache
    uses split-adjusted prices. We reject any option where strike / stock_price > 2.0
    or < 0.50, which flags post-split artifacts. (e.g. AAPL options at $450 when stock
    is $120 post-split adjustment.)
    Returns dict with strike, expiration, mid, delta, dte or None.
    """
    # Try trade_date first, then nearby days
    for offset in [0, 1, -1, 2, -2, 3]:
        check_date = trade_date + pd.Timedelta(days=offset)
        day = puts_df[puts_df['date'] == check_date]
        if len(day) > 0:
            break
    else:
        return None

    # Filter DTE and valid delta
    day = day[(day['dte'] >= dte_min) & (day['dte'] <= dte_max)]
    day = day[day['delta'] < -0.01].copy()
    day = day[day['mid'] > 0].copy()

    if len(day) == 0:
        return None

    # Split-artifact filter: reject strikes that are implausibly far from current price
    if stock_price is not None and stock_price > 0:
        ratio = day['strike'] / stock_price
        day = day[(ratio >= 0.50) & (ratio <= 2.0)]

    if len(day) == 0:
        return None

    day = day.copy()
    day['delta_dist'] = (day['delta'] - target_delta).abs()
    best = day.nsmallest(1, 'delta_dist').iloc[0]

    return {
        'strike': float(best['strike']),
        'expiration': pd.Timestamp(best['expiration']),
        'mid': float(best['mid']),
        'delta': float(best['delta']),
        'dte': int(best['dte']),
    }


def find_call(calls_df, trade_date, cost_basis, target_delta=0.30,
              dte_min=14, dte_max=50):
    """
    Find best call to sell at or above cost_basis (for covered call on assigned shares).
    """
    if calls_df is None:
        return None

    for offset in [0, 1, -1, 2, -2, 3]:
        check_date = trade_date + pd.Timedelta(days=offset)
        day = calls_df[calls_df['date'] == check_date]
        if len(day) > 0:
            break
    else:
        return None

    day = day[(day['dte'] >= dte_min) & (day['dte'] <= dte_max)]
    day = day[day['delta'] > 0.05].copy()
    day = day[day['mid'] > 0].copy()
    # Only calls at or above cost basis (don't lock in a loss)
    day = day[day['strike'] >= cost_basis * 0.995]

    if len(day) == 0:
        return None

    day = day.copy()
    day['delta_dist'] = (day['delta'] - target_delta).abs()
    best = day.nsmallest(1, 'delta_dist').iloc[0]

    return {
        'strike': float(best['strike']),
        'expiration': pd.Timestamp(best['expiration']),
        'mid': float(best['mid']),
        'delta': float(best['delta']),
        'dte': int(best['dte']),
    }


# ============================================================================
# CORE BACKTEST
# ============================================================================

def run_backtest(
    oot_preds, options_puts, options_calls, price_lookup, ma200_lookup, vix,
    ml_filter=True, conf_threshold=CONF_THRESHOLD_060,
    label="ML-Filtered", verbose=False,
):
    """
    Run CSP/wheel backtest.

    Cash accounting (clear and correct):
    - At CSP entry: cash += premium; cash -= strike*100*n (collateral reserved)
    - At CSP expiry worthless: cash += strike*100*n (release collateral)
    - At CSP assignment: 'collateral' converts to shares at cost_basis=strike
      (no additional cash move — cash already had collateral removed)
    - Shares: tracked separately. Stop-loss or CC-called-away adds proceeds to cash.
    - CC entry: cash += cc_premium (shares already held as collateral for CC)
    - CC expiry worthless: continue holding shares
    - CC assignment (called away): cash += strike*shares (deliver shares at strike price)
    - NAV = cash + current_share_value + csp_collateral_reserved
    """
    print(f"\n{'=' * 65}")
    print(f"BACKTEST: {label}")
    print(f"  ML filter: {ml_filter}  |  Threshold: {conf_threshold if ml_filter else 'N/A'}")
    print(f"{'=' * 65}")

    # Build signal lookup: date -> {ticker: proba}
    # For filtered: only high-confidence predictions
    # For unfiltered: all tickers get "signal" every date we have options for them
    if ml_filter:
        signal_lookup = {}
        for _, row in oot_preds[oot_preds['pred_proba'] >= conf_threshold].iterrows():
            d = row['date']
            if d not in signal_lookup:
                signal_lookup[d] = {}
            signal_lookup[d][row['ticker']] = float(row['pred_proba'])
        signal_dates = sorted(signal_lookup.keys())
        print(f"  Signal dates: {len(signal_dates)} | Total signals: {sum(len(v) for v in signal_lookup.values())}")
    else:
        signal_lookup = None
        print(f"  No ML filter — all {len(options_puts)} tickers eligible every week")

    # Backtest date range
    start_date = oot_preds['date'].min()
    end_date = min(oot_preds['date'].max(), pd.Timestamp('2026-06-01'))

    # Weekly rebalance on Fridays (options settle Fridays)
    rebalance_dates = pd.date_range(start_date, end_date, freq='W-FRI')

    # State
    cash = INITIAL_CAPITAL
    # Open positions: list of dicts
    # Each position: type, ticker, strike, expiration, n_contracts, premium, cost_basis, shares, ml_flag
    open_positions = []

    nav_history = []   # [{date, nav, cash, n_pos, vix}]
    trade_log = []     # Each closed trade

    def current_nav(date):
        """Compute NAV = cash + share_market_value + csp_reserved_collateral."""
        nav = cash
        for pos in open_positions:
            if pos['type'] == 'csp':
                # Collateral is already OUT of cash — add it back to NAV
                nav += pos['strike'] * pos['n_contracts'] * 100
            elif pos['type'] in ('shares', 'cc'):
                sp = get_price_on_date(price_lookup.get(pos['ticker'], pd.Series(dtype=float)), date)
                if sp is None:
                    sp = pos['cost_basis']
                nav += sp * pos['shares']
        return nav

    def get_vix_val(date):
        v = get_scalar_on_date(vix, date, tolerance_days=5)
        return v if v is not None else 20.0

    tickers_in_options = set(options_puts.keys())
    n_puts_sold = 0
    n_assignments = 0
    n_expired = 0
    n_called_away = 0
    n_stops = 0
    n_vix_blocked = 0

    for rebalance_date in rebalance_dates:

        # -------------------------------------------------------------------
        # 1. Process expired / hit positions
        # -------------------------------------------------------------------
        still_open = []
        for pos in open_positions:
            sp = get_price_on_date(
                price_lookup.get(pos['ticker'], pd.Series(dtype=float)),
                rebalance_date
            )

            if pos['type'] == 'csp' and rebalance_date >= pos['expiration']:
                # Sanity: if strike is >2x stock price, this is a split-artifact position
                # (should have been prevented by find_put filter, but double-check here)
                if sp is not None and pos['strike'] > sp * 2.5:
                    # Release collateral back — treat as data error, skip
                    cash += pos['strike'] * pos['n_contracts'] * 100
                    continue
                if sp is not None and sp < pos['strike']:
                    # ASSIGNED: convert collateral to shares
                    # cash was already reduced by collateral at entry; no change to cash
                    n_contracts = pos['n_contracts']
                    shares = n_contracts * 100
                    n_assignments += 1
                    new_pos = {
                        'type': 'shares',
                        'ticker': pos['ticker'],
                        'strike': pos['strike'],
                        'expiration': None,
                        'n_contracts': n_contracts,
                        'premium': 0.0,
                        'cost_basis': pos['strike'],
                        'shares': shares,
                        'ml_flag': pos['ml_flag'],
                        'cc_expiration': None,
                        'cc_strike': None,
                        'cc_premium': 0.0,
                        'total_premium': pos['premium'],  # Track all premium collected
                    }
                    still_open.append(new_pos)
                    trade_log.append({
                        'event': 'csp_assigned', 'ticker': pos['ticker'],
                        'date': rebalance_date,
                        'strike': pos['strike'],
                        'n_contracts': pos['n_contracts'],
                        'premium_collected': pos['premium'] * pos['n_contracts'] * 100,
                        'stock_price_at_event': sp,
                        'pnl': pos['premium'] * pos['n_contracts'] * 100,  # Premium kept; shares at strike
                        'ml_flag': pos['ml_flag'],
                    })
                else:
                    # EXPIRED WORTHLESS: release collateral
                    collateral = pos['strike'] * pos['n_contracts'] * 100
                    nonlocal_cash_add = collateral  # Will add to cash below
                    pnl = pos['premium'] * pos['n_contracts'] * 100
                    n_expired += 1
                    # Add collateral back
                    # (can't modify 'cash' directly in loop — use a flag mechanism)
                    still_open.append({'_release_cash': collateral})
                    trade_log.append({
                        'event': 'csp_expired', 'ticker': pos['ticker'],
                        'date': rebalance_date,
                        'strike': pos['strike'],
                        'n_contracts': pos['n_contracts'],
                        'premium_collected': pnl,
                        'stock_price_at_event': sp,
                        'pnl': pnl,
                        'ml_flag': pos['ml_flag'],
                    })

            elif pos['type'] == 'shares':
                if sp is None:
                    still_open.append(pos)
                    continue

                # Check stop-loss
                loss_pct = (sp - pos['cost_basis']) / pos['cost_basis']
                if loss_pct <= -STOP_LOSS_PCT:
                    proceeds = sp * pos['shares']
                    # cash was already debited for collateral (=shares at cost_basis) at assignment
                    # We get back proceeds (at current price, lower)
                    still_open.append({'_release_cash': proceeds})
                    n_stops += 1
                    pnl = (sp - pos['cost_basis']) * pos['shares'] + pos.get('total_premium', 0)
                    trade_log.append({
                        'event': 'stop_loss', 'ticker': pos['ticker'],
                        'date': rebalance_date,
                        'strike': pos['cost_basis'],
                        'n_contracts': pos['n_contracts'],
                        'premium_collected': pos.get('total_premium', 0),
                        'stock_price_at_event': sp,
                        'pnl': pnl,
                        'ml_flag': pos['ml_flag'],
                    })
                else:
                    # Try to sell a covered call if not already running one
                    calls_df = options_calls.get(pos['ticker'])
                    if calls_df is not None:
                        call = find_call(calls_df, rebalance_date, pos['cost_basis'])
                        if call is not None:
                            cc_premium_total = call['mid'] * pos['n_contracts'] * 100
                            still_open.append({'_release_cash': cc_premium_total})
                            pos = dict(pos)  # copy
                            pos['type'] = 'cc'
                            pos['cc_expiration'] = call['expiration']
                            pos['cc_strike'] = call['strike']
                            pos['cc_premium'] = call['mid']
                            pos['total_premium'] = pos.get('total_premium', 0) + cc_premium_total
                        still_open.append(pos)
                    else:
                        still_open.append(pos)

            elif pos['type'] == 'cc':
                if sp is None:
                    still_open.append(pos)
                    continue

                loss_pct = (sp - pos['cost_basis']) / pos['cost_basis']
                if loss_pct <= -STOP_LOSS_PCT:
                    # Stop-loss even on CC
                    proceeds = sp * pos['shares']
                    still_open.append({'_release_cash': proceeds})
                    n_stops += 1
                    pnl = (sp - pos['cost_basis']) * pos['shares'] + pos.get('total_premium', 0)
                    trade_log.append({
                        'event': 'stop_loss_cc', 'ticker': pos['ticker'],
                        'date': rebalance_date,
                        'strike': pos['cost_basis'],
                        'n_contracts': pos['n_contracts'],
                        'premium_collected': pos.get('total_premium', 0),
                        'stock_price_at_event': sp,
                        'pnl': pnl,
                        'ml_flag': pos['ml_flag'],
                    })
                elif pos['cc_expiration'] is not None and rebalance_date >= pos['cc_expiration']:
                    if sp > pos['cc_strike']:
                        # Called away
                        proceeds = pos['cc_strike'] * pos['shares']
                        still_open.append({'_release_cash': proceeds})
                        n_called_away += 1
                        pnl = (pos['cc_strike'] - pos['cost_basis']) * pos['shares'] + pos.get('total_premium', 0)
                        trade_log.append({
                            'event': 'called_away', 'ticker': pos['ticker'],
                            'date': rebalance_date,
                            'strike': pos['cc_strike'],
                            'n_contracts': pos['n_contracts'],
                            'premium_collected': pos.get('total_premium', 0),
                            'stock_price_at_event': sp,
                            'pnl': pnl,
                            'ml_flag': pos['ml_flag'],
                        })
                    else:
                        # CC expired worthless — keep shares, reset to 'shares' type
                        pos = dict(pos)
                        pos['type'] = 'shares'
                        pos['cc_expiration'] = None
                        still_open.append(pos)
                else:
                    still_open.append(pos)

            else:
                still_open.append(pos)

        # Apply cash releases
        new_still_open = []
        for item in still_open:
            if isinstance(item, dict) and '_release_cash' in item:
                cash += item['_release_cash']
            else:
                new_still_open.append(item)
        open_positions = new_still_open

        # -------------------------------------------------------------------
        # 2. VIX gate
        # -------------------------------------------------------------------
        current_vix = get_vix_val(rebalance_date)
        if current_vix > VIX_GATE:
            n_vix_blocked += 1
            nav_history.append({
                'date': rebalance_date, 'nav': current_nav(rebalance_date),
                'cash': cash, 'n_positions': len(open_positions), 'vix': current_vix,
            })
            continue

        # -------------------------------------------------------------------
        # 3. Build candidate list for this week
        # -------------------------------------------------------------------
        positioned_tickers = {p['ticker'] for p in open_positions}

        if ml_filter:
            # Get signals from the nearest prediction date (within 7 days prior)
            candidates = {}
            for offset in range(0, 8):
                check_date = rebalance_date - pd.Timedelta(days=offset)
                if check_date in signal_lookup:
                    for ticker, proba in signal_lookup[check_date].items():
                        if ticker in tickers_in_options and ticker not in positioned_tickers:
                            if ticker not in candidates or proba > candidates[ticker]:
                                candidates[ticker] = proba
        else:
            # All available tickers
            candidates = {t: 0.5 for t in tickers_in_options if t not in positioned_tickers}

        # -------------------------------------------------------------------
        # 4. 200MA filter
        # -------------------------------------------------------------------
        qualified = {}
        for ticker, proba in candidates.items():
            if ticker not in price_lookup:
                continue
            sp = get_price_on_date(price_lookup[ticker], rebalance_date)
            ma = get_scalar_on_date(ma200_lookup.get(ticker, pd.Series(dtype=float)), rebalance_date)
            if sp is None:
                continue
            if ma is not None and sp < ma:
                continue  # Below 200MA — skip
            qualified[ticker] = (proba, sp)

        # -------------------------------------------------------------------
        # 5. Open new CSP positions (sorted by confidence, best first)
        # -------------------------------------------------------------------
        sorted_candidates = sorted(qualified.items(), key=lambda x: -x[1][0])

        nav_est = current_nav(rebalance_date)

        for ticker, (proba, sp) in sorted_candidates:
            if len(open_positions) >= MAX_POSITIONS:
                break

            max_collateral = nav_est * MAX_CAPITAL_PER_POS_PCT

            # Find best put
            if ticker not in options_puts:
                continue
            put = find_put(options_puts[ticker], rebalance_date, stock_price=sp)
            if put is None:
                continue

            # How many contracts can we fit?
            collateral_per_contract = put['strike'] * 100
            if collateral_per_contract <= 0:
                continue

            n_contracts = int(max_collateral / collateral_per_contract)
            n_contracts = max(1, min(n_contracts, int(cash / collateral_per_contract)))

            if n_contracts < 1 or cash < collateral_per_contract:
                continue

            # Enter position
            total_collateral = put['strike'] * n_contracts * 100
            total_premium = put['mid'] * n_contracts * 100

            cash -= total_collateral
            cash += total_premium

            open_positions.append({
                'type': 'csp',
                'ticker': ticker,
                'strike': put['strike'],
                'expiration': put['expiration'],
                'n_contracts': n_contracts,
                'premium': put['mid'],
                'cost_basis': put['strike'],
                'shares': 0,
                'ml_flag': ml_filter,
                'cc_expiration': None,
                'cc_strike': None,
                'cc_premium': 0.0,
                'total_premium': total_premium,
                'entry_stock_price': sp,
                'entry_date': rebalance_date,
            })
            positioned_tickers.add(ticker)
            n_puts_sold += 1

            if verbose:
                print(f"  [{rebalance_date.date()}] SELL PUT {ticker} "
                      f"strike={put['strike']:.0f} exp={put['expiration'].date()} "
                      f"mid={put['mid']:.2f} delta={put['delta']:.3f} "
                      f"n={n_contracts} premium=${total_premium:.0f}")

        # -------------------------------------------------------------------
        # 6. Record NAV
        # -------------------------------------------------------------------
        nav_history.append({
            'date': rebalance_date,
            'nav': current_nav(rebalance_date),
            'cash': cash,
            'n_positions': len(open_positions),
            'vix': current_vix,
        })

    # Close all remaining positions at end of backtest
    for pos in open_positions:
        if pos['type'] == 'csp':
            # Release collateral
            cash += pos['strike'] * pos['n_contracts'] * 100
            trade_log.append({
                'event': 'csp_eob', 'ticker': pos['ticker'],
                'date': end_date, 'pnl': pos['total_premium'],
                'ml_flag': pos['ml_flag'],
                'n_contracts': pos['n_contracts'],
                'premium_collected': pos['total_premium'],
                'strike': pos['strike'], 'stock_price_at_event': None,
            })
        elif pos['type'] in ('shares', 'cc'):
            sp = get_price_on_date(price_lookup.get(pos['ticker'], pd.Series(dtype=float)), end_date) or pos['cost_basis']
            cash += sp * pos['shares']
            pnl = (sp - pos['cost_basis']) * pos['shares'] + pos.get('total_premium', 0)
            trade_log.append({
                'event': 'shares_eob', 'ticker': pos['ticker'],
                'date': end_date, 'pnl': pnl,
                'ml_flag': pos['ml_flag'],
                'n_contracts': pos['n_contracts'],
                'premium_collected': pos.get('total_premium', 0),
                'strike': pos['cost_basis'], 'stock_price_at_event': sp,
            })

    # Final NAV
    nav_history.append({
        'date': end_date,
        'nav': cash,
        'cash': cash,
        'n_positions': 0,
        'vix': 20.0,
    })

    nav_df = pd.DataFrame(nav_history).drop_duplicates('date').sort_values('date')
    trades_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

    print(f"  Puts sold: {n_puts_sold} | Assignments: {n_assignments} | "
          f"Expired: {n_expired} | Called away: {n_called_away} | "
          f"Stops: {n_stops} | VIX blocks: {n_vix_blocked}")
    print(f"  Final NAV: ${cash:,.0f} (started ${INITIAL_CAPITAL:,.0f})")

    stats = {
        'n_puts_sold': n_puts_sold,
        'n_assignments': n_assignments,
        'n_expired': n_expired,
        'n_called_away': n_called_away,
        'n_stops': n_stops,
        'n_vix_blocked': n_vix_blocked,
    }

    return nav_df, trades_df, stats


# ============================================================================
# PERFORMANCE ANALYTICS
# ============================================================================

def compute_metrics(nav_df, label="Strategy"):
    """Full risk-adjusted metrics from weekly NAV series."""
    if nav_df is None or len(nav_df) < 4:
        print(f"  {label}: insufficient data")
        return {}

    nav = nav_df.set_index('date').sort_index()['nav']
    weekly_rets = nav.pct_change().dropna()
    ann = np.sqrt(52)

    total_ret = nav.iloc[-1] / INITIAL_CAPITAL - 1
    n_years = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.1)
    cagr = (nav.iloc[-1] / INITIAL_CAPITAL) ** (1 / n_years) - 1

    mu = weekly_rets.mean()
    sigma = weekly_rets.std()
    sharpe = mu / sigma * ann if sigma > 0 else 0

    downside_std = weekly_rets[weekly_rets < 0].std()
    sortino = mu / downside_std * ann if downside_std > 0 else 0

    rolling_max = nav.cummax()
    dd = nav / rolling_max - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0

    win_rate = (weekly_rets > 0).mean()
    pf_num = weekly_rets[weekly_rets > 0].sum()
    pf_den = abs(weekly_rets[weekly_rets < 0].sum())
    profit_factor = pf_num / pf_den if pf_den > 0 else float('inf')

    m = {
        'label': label,
        'total_return': float(total_ret),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd),
        'calmar': float(calmar),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor) if np.isfinite(profit_factor) else 999.0,
        'n_weeks': int(len(weekly_rets)),
        'n_years': float(n_years),
        'final_nav': float(nav.iloc[-1]),
    }

    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")
    print(f"  Total Return:  {total_ret*100:+.1f}%")
    print(f"  CAGR:          {cagr*100:+.1f}%")
    print(f"  Sharpe:        {sharpe:+.3f}")
    print(f"  Sortino:       {sortino:+.3f}")
    print(f"  Max Drawdown:  {max_dd*100:.1f}%")
    print(f"  Calmar:        {calmar:+.3f}")
    print(f"  Win Rate:      {win_rate*100:.1f}% (weekly NAV)")
    print(f"  Profit Factor: {profit_factor:.2f}")
    return m


def regime_analysis(nav_df, price_lookup, label=""):
    """
    HC #428 R1: Classify each week by prior SPY monthly return.
    Green: SPY month > +2%, Red: SPY month < -2%, Flat: otherwise.
    Compare Sharpe across regimes.
    """
    spy_prices = price_lookup.get('SPY', pd.Series(dtype=float))

    nav = nav_df.set_index('date').sort_index()['nav']
    weekly_rets = nav.pct_change().dropna()

    spy_monthly = spy_prices.resample('ME').last().pct_change()

    def classify(date):
        m = date.to_period('M')
        idx = spy_monthly.index[spy_monthly.index.to_period('M') == m]
        if len(idx) == 0:
            return 'flat'
        r = spy_monthly.loc[idx[0]]
        if r > 0.02:
            return 'green'
        elif r < -0.02:
            return 'red'
        return 'flat'

    wr_series = weekly_rets.copy()
    wr_series.index = pd.to_datetime(wr_series.index)
    regimes = wr_series.index.map(classify)

    ann = np.sqrt(52)
    result = {}
    print(f"\n  Regime breakdown [{label}]:")
    for regime in ['green', 'red', 'flat']:
        sub = wr_series[regimes == regime]
        if len(sub) < 3:
            continue
        sh = sub.mean() / sub.std() * ann if sub.std() > 0 else 0
        wr = (sub > 0).mean()
        result[regime] = {'sharpe': float(sh), 'win_rate': float(wr), 'n': int(len(sub))}
        print(f"    {regime:>5}: N={len(sub):>3}, Sharpe={sh:+.3f}, WR={wr*100:.1f}%")

    if 'green' in result and 'red' in result:
        sg, sr = result['green']['sharpe'], result['red']['sharpe']
        denom = max(abs(sg), abs(sr), 0.001)
        skew = abs(sg - sr) / denom
        r1_pass = skew <= 0.50
        print(f"    R1 skew: {skew:.3f} {'PASS' if r1_pass else 'FAIL'} (gate: <= 0.50)")
        result['r1_skew'] = float(skew)
        result['r1_pass'] = bool(r1_pass)

    return result


def per_year_analysis(nav_df, price_lookup, label=""):
    """Per-year metrics including SPY comparison."""
    spy_prices = price_lookup.get('SPY', pd.Series(dtype=float))
    nav = nav_df.set_index('date').sort_index()['nav']
    weekly_rets = nav.pct_change().dropna()

    ann = np.sqrt(52)
    print(f"\n  Per-year breakdown [{label}]:")
    print(f"    {'Year':>5} {'Return':>9} {'Sharpe':>8} {'WR':>7} {'N':>4}")

    yearly = {}
    for year in sorted(weekly_rets.index.year.unique()):
        sub = weekly_rets[weekly_rets.index.year == year]
        if len(sub) < 3:
            continue
        yr_ret = (1 + sub).prod() - 1
        yr_sh = sub.mean() / sub.std() * ann if sub.std() > 0 else 0
        yr_wr = (sub > 0).mean()
        print(f"    {year:>5} {yr_ret*100:>+8.1f}% {yr_sh:>8.2f} {yr_wr*100:>6.1f}% {len(sub):>4}")
        yearly[year] = {'return': float(yr_ret), 'sharpe': float(yr_sh), 'win_rate': float(yr_wr)}

    return yearly


def permutation_test_nav(nav_df, n_perms=100, label=""):
    """
    Permutation test on weekly returns. Shuffle returns to build null Sharpe distribution.
    Real Sharpe vs null distribution Sharpe.
    """
    nav = nav_df.set_index('date').sort_index()['nav']
    weekly_rets = nav.pct_change().dropna().values
    ann = np.sqrt(52)

    if len(weekly_rets) == 0 or weekly_rets.std() == 0:
        return {'z_score': 0.0, 'p_value': 1.0}

    real_sharpe = weekly_rets.mean() / weekly_rets.std() * ann

    perm_sharpes = []
    for i in range(n_perms):
        rng = np.random.default_rng(i)
        perm_rets = rng.permutation(weekly_rets)
        perm_sharpes.append(perm_rets.mean() / perm_rets.std() * ann if perm_rets.std() > 0 else 0)

    perm_arr = np.array(perm_sharpes)
    z = (real_sharpe - perm_arr.mean()) / (perm_arr.std() + 1e-10)
    p = (perm_arr >= real_sharpe).mean()

    print(f"\n  Permutation test [{label}]:")
    print(f"    Real Sharpe:  {real_sharpe:+.3f}")
    print(f"    Perm Sharpe:  {perm_arr.mean():+.3f} +/- {perm_arr.std():.3f}")
    print(f"    Z-score:      {z:+.2f}")
    print(f"    p-value:      {p:.4f}  {'SIGNIFICANT' if p < 0.05 else ('MARGINAL' if p < 0.10 else 'not sig')}")

    return {'real_sharpe': float(real_sharpe), 'perm_mean': float(perm_arr.mean()),
            'z_score': float(z), 'p_value': float(p)}


# ============================================================================
# COMPARISON TABLE
# ============================================================================

def print_comparison_table(results):
    """Print clean comparison table across all strategies."""
    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON: ML-FILTERED CSP vs UNFILTERED CSP")
    print(f"{'=' * 80}")

    labels = [r['label'] for r in results]
    header = f"{'Metric':<28}" + "".join(f"{l:>17}" for l in labels)
    print(header)
    print("-" * 80)

    def fmt_row(name, key, scale=1, fmt_str="{:+.2f}"):
        row = f"  {name:<26}"
        for r in results:
            v = r.get('metrics', {}).get(key, float('nan'))
            if np.isnan(v):
                row += f"{'N/A':>17}"
            else:
                try:
                    row += f"{fmt_str.format(v * scale):>17}"
                except Exception:
                    row += f"{'ERR':>17}"
        print(row)

    fmt_row("CAGR", "cagr", 100, "{:+.1f}%")
    fmt_row("Total Return", "total_return", 100, "{:+.1f}%")
    fmt_row("Sharpe Ratio", "sharpe", 1, "{:+.3f}")
    fmt_row("Sortino Ratio", "sortino", 1, "{:+.3f}")
    fmt_row("Max Drawdown", "max_dd", 100, "{:.1f}%")
    fmt_row("Calmar Ratio", "calmar", 1, "{:+.3f}")
    fmt_row("Win Rate (wkly)", "win_rate", 100, "{:.1f}%")
    fmt_row("Profit Factor", "profit_factor", 1, "{:.2f}")
    fmt_row("Final NAV", "final_nav", 1, "${:,.0f}")

    print("-" * 80)
    print(f"{'Trades':<28}" + "".join(f"{'':>17}" for _ in results))

    for stat_key, stat_label in [("n_puts_sold", "Puts sold"),
                                   ("n_assignments", "Assignments"),
                                   ("n_expired", "Expired worthless"),
                                   ("n_stops", "Stop-losses hit"),
                                   ("n_called_away", "Called away")]:
        row = f"  {stat_label:<26}"
        for r in results:
            v = r.get('stats', {}).get(stat_key, 0)
            row += f"{v:>17}"
        print(row)

    print("-" * 80)
    print(f"{'Regime (Sharpe)':<28}" + "".join(f"{'':>17}" for _ in results))

    for regime in ['green', 'red', 'flat']:
        row = f"  {regime.capitalize() + ' months':<26}"
        for r in results:
            v = r.get('regime', {}).get(regime, {}).get('sharpe', float('nan'))
            row += f"{v:>+17.3f}" if not np.isnan(v) else f"{'N/A':>17}"
        print(row)

    row_r1 = f"  {'R1 gate (pass/fail)':<26}"
    for r in results:
        rp = r.get('regime', {}).get('r1_pass', None)
        row_r1 += f"{'PASS' if rp else 'FAIL':>17}"
    print(row_r1)

    row_pv = f"  {'Permutation p-value':<26}"
    for r in results:
        pv = r.get('perm', {}).get('p_value', float('nan'))
        row_pv += f"{pv:>17.4f}" if not np.isnan(pv) else f"{'N/A':>17}"
    print(row_pv)

    print(f"\n{'=' * 80}")

    # Verdict
    filtered = [r for r in results if 'ML' in r.get('label', '')]
    unfiltered = [r for r in results if 'Unfiltered' in r.get('label', '')]
    if filtered and unfiltered:
        sh_f = filtered[0].get('metrics', {}).get('sharpe', 0)
        sh_u = unfiltered[0].get('metrics', {}).get('sharpe', 0)
        dd_f = filtered[0].get('metrics', {}).get('max_dd', 0)
        dd_u = unfiltered[0].get('metrics', {}).get('max_dd', 0)
        print("VERDICT:")
        if sh_f > sh_u:
            print(f"  ML filter (0.60+) IMPROVES Sharpe: {sh_f:+.3f} vs {sh_u:+.3f} unfiltered (+{sh_f - sh_u:.3f})")
        else:
            print(f"  Unfiltered beats ML filter by Sharpe: {sh_u:+.3f} vs {sh_f:+.3f} (+{sh_u - sh_f:.3f})")
        if dd_f > dd_u:
            print(f"  ML filter reduces max drawdown: {dd_f*100:.1f}% vs {dd_u*100:.1f}%")
        print(f"{'=' * 80}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = datetime.now()
    print("=" * 70)
    print("ML-FILTERED CSP BACKTEST v1")
    print(f"Started: {t0.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Load all data
    oot_preds = load_predictions()
    options_puts = load_all_options()

    # Load call options separately for wheel strategy
    print("Loading call options for wheel (CC) leg...")
    options_calls = {}
    for fpath in sorted(OPTIONS_DIR.glob("*.parquet")):
        ticker = fpath.stem
        df = pd.read_parquet(fpath)
        df['date'] = pd.to_datetime(df['date'])
        df['expiration'] = pd.to_datetime(df['expiration'])
        calls = df[df['type'] == 'c'].copy()
        if len(calls) > 0:
            options_calls[ticker] = calls

    price_lookup, ma200_lookup = load_prices()
    vix = load_vix()

    # Common universe
    oot_tickers = set(oot_preds['ticker'].unique())
    opt_tickers = set(options_puts.keys())
    common = oot_tickers & opt_tickers
    print(f"\nUniverse overlap: {len(common)} tickers")

    # Filter to common universe
    oot_preds = oot_preds[oot_preds['ticker'].isin(common)].copy()
    options_puts_common = {t: options_puts[t] for t in common if t in options_puts}
    options_calls_common = {t: options_calls[t] for t in common if t in options_calls}

    # -----------------------------------------------------------------------
    # Run backtests
    # -----------------------------------------------------------------------
    all_results = []

    # A: ML-filtered (0.60+ threshold)
    nav_f60, trades_f60, stats_f60 = run_backtest(
        oot_preds, options_puts_common, options_calls_common,
        price_lookup, ma200_lookup, vix,
        ml_filter=True, conf_threshold=CONF_THRESHOLD_060,
        label="ML-Filtered (0.60+)",
    )

    # B: ML-filtered (0.65+ threshold)
    nav_f65, trades_f65, stats_f65 = run_backtest(
        oot_preds, options_puts_common, options_calls_common,
        price_lookup, ma200_lookup, vix,
        ml_filter=True, conf_threshold=CONF_THRESHOLD_065,
        label="ML-Filtered (0.65+)",
    )

    # C: ML-filtered (0.70+ threshold, ~80% precision)
    nav_f70, trades_f70, stats_f70 = run_backtest(
        oot_preds, options_puts_common, options_calls_common,
        price_lookup, ma200_lookup, vix,
        ml_filter=True, conf_threshold=CONF_THRESHOLD_070,
        label="ML-Filtered (0.70+)",
    )

    # D: Unfiltered control
    nav_unf, trades_unf, stats_unf = run_backtest(
        oot_preds, options_puts_common, options_calls_common,
        price_lookup, ma200_lookup, vix,
        ml_filter=False,
        label="Unfiltered (all stocks)",
    )

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 65}")
    print("PERFORMANCE METRICS")
    print(f"{'=' * 65}")

    for label, nav_df, stats in [
        ("ML-Filtered (0.60+)", nav_f60, stats_f60),
        ("ML-Filtered (0.65+)", nav_f65, stats_f65),
        ("ML-Filtered (0.70+)", nav_f70, stats_f70),
        ("Unfiltered", nav_unf, stats_unf),
    ]:
        m = compute_metrics(nav_df, label=label)
        regime = regime_analysis(nav_df, price_lookup, label=label)
        yearly = per_year_analysis(nav_df, price_lookup, label=label)
        perm = permutation_test_nav(nav_df, n_perms=100, label=label)
        all_results.append({
            'label': label, 'metrics': m, 'stats': stats,
            'regime': regime, 'yearly': yearly, 'perm': perm,
        })

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    nav_f60.to_csv(OUTPUT_DIR / "nav_ml_filtered_060.csv", index=False)
    nav_f65.to_csv(OUTPUT_DIR / "nav_ml_filtered_065.csv", index=False)
    nav_f70.to_csv(OUTPUT_DIR / "nav_ml_filtered_070.csv", index=False)
    nav_unf.to_csv(OUTPUT_DIR / "nav_unfiltered.csv", index=False)

    if len(trades_f60) > 0:
        trades_f60.to_csv(OUTPUT_DIR / "trades_ml_filtered_060.csv", index=False)
    if len(trades_unf) > 0:
        trades_unf.to_csv(OUTPUT_DIR / "trades_unfiltered.csv", index=False)

    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj) if np.isfinite(obj) else None
        elif isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    summary = {
        'run_date': t0.strftime('%Y-%m-%d %H:%M'),
        'runtime_seconds': (datetime.now() - t0).total_seconds(),
        'config': {
            'conf_threshold_060': CONF_THRESHOLD_060,
            'conf_threshold_065': CONF_THRESHOLD_065,
            'conf_threshold_070': CONF_THRESHOLD_070,
            'max_positions': MAX_POSITIONS,
            'max_capital_per_pos_pct': MAX_CAPITAL_PER_POS_PCT,
            'stop_loss_pct': STOP_LOSS_PCT,
            'vix_gate': VIX_GATE,
            'initial_capital': INITIAL_CAPITAL,
        },
        'results': clean_for_json(all_results),
    }

    with open(OUTPUT_DIR / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    print_comparison_table(all_results)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\nRuntime: {elapsed:.1f}s")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("DONE.")


if __name__ == "__main__":
    main()
