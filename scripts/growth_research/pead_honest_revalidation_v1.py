#!/usr/bin/env python3
"""
PEAD Honest Revalidation v1 — Hold-to-Expiry with Intrinsic Value Only
========================================================================

The PEAD (post-earnings announcement drift) call spread strategy was validated
on July 26 with Sharpe 1.38, but used the OLD pricing framework that may have
inflated results (early exit with BS mid-life pricing, take-profit logic, etc).

This revalidation applies the HONEST pricing framework:
  1. HOLD TO EXPIRY ONLY — no early exit, no take-profit
  2. At expiry: INTRINSIC VALUE ONLY (no BS time value)
  3. 15% haircut on ENTRY only (expiry = auto-exercise, no spread crossing)
  4. Commission: $2.60/spread round trip
  5. $645 starting capital, max $200/trade

Strategy: After earnings announcements with gap > threshold, buy call spreads
on the gapping stock in the gap direction. 30 large-cap stocks, quarterly events.

Variants:
  A: Gap > 3%, 30 DTE, hold to expiry
  B: Gap > 5%, 30 DTE, hold to expiry
  C: Gap > 5%, 45 DTE, hold to expiry (original best)
  D: Gap > 3%, 45 DTE, hold to expiry

Uses standardized tools from research.tools.
"""

import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Standardized tools
sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread, estimate_iv, compute_atr,
    COMMISSION_RT_SPREAD, DEFAULT_HAIRCUT
)
from research.tools.adversarial_validator import validate_trades

BASE = Path('/home/jupiter/Lvl3Quant')
CACHE_DIR = BASE / 'output' / 'pead_honest_reval'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def fprint(*a, **kw):
    print(*a, **kw, flush=True)


# ─── MLflow ─────────────────────────────────────────────────────────

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
    fprint("MLflow connected")
except Exception:
    fprint("MLflow unavailable — will log to disk only")


# ─── Constants ──────────────────────────────────────────────────────

TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'MA',
    'JNJ', 'UNH', 'PG', 'HD', 'DIS', 'NFLX', 'CRM', 'ADBE', 'PYPL', 'INTC',
    'AMD', 'COST', 'WMT', 'TGT', 'NKE', 'SBUX', 'MCD', 'BA', 'CAT', 'GS',
]

CAP = 645.0
MAX_POSITION = 200.0
SPREAD_WIDTH_PCT = 0.03  # 3% width
EARNINGS_MONTHS = {1, 2, 4, 5, 7, 8, 10, 11}

# Random baseline tickers (different set, same market cap tier)
RANDOM_TICKERS = [
    'PFE', 'ABBV', 'KO', 'PEP', 'ABT', 'LLY', 'TMO', 'DHR', 'BMY', 'AMGN',
    'LOW', 'TJX', 'ROST', 'CMG', 'SYK', 'ISRG', 'ZTS', 'VRTX', 'REGN', 'GILD',
    'NOW', 'INTU', 'PANW', 'SNPS', 'CDNS', 'KLAC', 'LRCX', 'AMAT', 'MRVL', 'AVGO',
]


# ═══════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════

def download_prices(tickers, label="main"):
    """Download OHLCV price data for a list of tickers + SPY."""
    import yfinance as yf

    cache = CACHE_DIR / f'prices_{label}.parquet'
    if cache.exists():
        df = pd.read_parquet(cache)
        fprint(f"  Loaded cached {label} prices: {len(df)} rows")
        return df

    fprint(f"  Downloading {len(tickers)} tickers for {label}...")
    all_data = []
    for tk in tickers + ['SPY']:
        try:
            d = yf.download(tk, start='2016-01-01', end='2026-07-27', progress=False, auto_adjust=True)
            if len(d) > 0:
                d = d.reset_index()
                # Handle multi-level columns from yfinance
                cols = []
                for c in d.columns:
                    if isinstance(c, tuple):
                        cols.append(c[0].lower())
                    else:
                        cols.append(c.lower())
                d.columns = cols
                d['ticker'] = tk
                d = d[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
                all_data.append(d)
                fprint(f"    {tk}: {len(d)} days")
        except Exception as e:
            fprint(f"    {tk}: FAILED {e}")
        time.sleep(0.12)

    df = pd.concat(all_data, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df.to_parquet(cache)
    fprint(f"  Cached {len(df)} rows for {label}")
    return df


def download_vix():
    """Download VIX data."""
    import yfinance as yf

    cache = CACHE_DIR / 'vix.parquet'
    if cache.exists():
        return pd.read_parquet(cache)

    d = yf.download('^VIX', start='2016-01-01', end='2026-07-27', progress=False)
    if isinstance(d.columns, pd.MultiIndex):
        close = d['Close'].iloc[:, 0] if len(d['Close'].columns) > 0 else d['Close']
    else:
        close = d['Close']
    close = close.dropna()
    vdf = pd.DataFrame({'date': close.index, 'vix': close.values.flatten()}).dropna()
    vdf['date'] = pd.to_datetime(vdf['date'])
    vdf.to_parquet(cache)
    return vdf


# ═══════════════════════════════════════════════════════════════════
# EARNINGS DETECTION
# ═══════════════════════════════════════════════════════════════════

def detect_earnings_gaps(prices_df, ticker, min_gap_pct=2.0):
    """Detect earnings events from overnight gaps during earnings season.

    Heuristic: if |gap| > min_gap_pct% during earnings months (Jan/Feb, Apr/May,
    Jul/Aug, Oct/Nov), it's likely an earnings event.
    """
    tk_data = prices_df[prices_df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
    if len(tk_data) < 60:
        return []

    events = []
    for i in range(1, len(tk_data)):
        row = tk_data.iloc[i]
        prev = tk_data.iloc[i - 1]
        month = row['date'].month

        if month not in EARNINGS_MONTHS:
            continue

        gap = (row['open'] - prev['close']) / prev['close']

        if abs(gap) < min_gap_pct / 100.0:
            continue

        # Avoid double-counting: no gap within last 60 days
        recent = [e for e in events if (row['date'] - e['date']).days < 60]
        if recent:
            continue

        events.append({
            'date': row['date'],
            'ticker': ticker,
            'gap_pct': gap * 100,
            'prev_close': prev['close'],
            'open_price': row['open'],
            'close_price': row['close'],
            'idx': i,
        })

    return events


# ═══════════════════════════════════════════════════════════════════
# HONEST SIMULATION — HOLD TO EXPIRY, INTRINSIC VALUE ONLY
# ═══════════════════════════════════════════════════════════════════

def simulate_pead_honest(
    name: str,
    events: list,
    prices_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    spy_prices: pd.Series,
    gap_threshold: float = 5.0,
    dte: int = 45,
):
    """Simulate PEAD call spread strategy with HONEST pricing.

    HONEST RULES:
    1. Hold to expiry — no early exit
    2. At expiry: intrinsic value only (max(S-K1, 0) - max(S-K2, 0))
    3. 15% haircut on ENTRY only
    4. Commission: $2.60/spread RT
    """
    fprint(f"\n{'='*60}")
    fprint(f"  {name}")
    fprint(f"  Gap >= {gap_threshold}%, DTE={dte}, Hold to Expiry, Intrinsic Only")
    fprint(f"{'='*60}")

    equity = CAP
    trades = []

    # Prepare VIX lookup
    vix_series = vix_df.set_index('date')['vix'] if 'date' in vix_df.columns else vix_df

    for event in sorted(events, key=lambda x: x['date']):
        gap = event['gap_pct']
        tk = event['ticker']
        dt = event['date']
        idx = event['idx']

        # Only trade positive gaps (bull call spread for upward drift)
        if gap < gap_threshold:
            continue

        # Get ticker data
        tk_data = prices_df[prices_df['ticker'] == tk].sort_values('date').reset_index(drop=True)

        # Need enough forward data to reach expiry
        if idx + dte + 5 >= len(tk_data):
            continue

        # Position sizing: max $200/trade, at most 35% of equity
        pos_budget = min(MAX_POSITION, equity * 0.35)
        if pos_budget < 30:
            continue

        # Entry price = opening price on gap day
        S_entry = float(event['open_price'])

        # ATR for IV estimation (use pre-earnings data)
        lookback_start = max(0, idx - 30)
        tk_slice = tk_data.iloc[lookback_start:idx]
        if len(tk_slice) < 14:
            continue
        atr = compute_atr(tk_slice['high'], tk_slice['low'], tk_slice['close'], period=14)

        # VIX at entry
        vix_at_entry = 20.0
        nearest_vix = vix_series.index[vix_series.index.get_indexer([dt], method='nearest')]
        if len(nearest_vix) > 0:
            vix_at_entry = float(vix_series.loc[nearest_vix[0]])

        # Post-earnings IV crush: reduce IV by ~35% from pre-earnings level
        # (earnings vol is priced in pre-event; post-event vol drops sharply)
        base_iv = estimate_iv(atr, S_entry, vix_at_entry)
        post_earnings_iv = base_iv * 0.65  # 35% crush

        # Bull call spread: ATM strike at gap close, 3% width
        K1 = round(S_entry, 2)
        K2 = round(S_entry * (1 + SPREAD_WIDTH_PCT), 2)

        entry_cost_per_share, max_profit_per_share = price_bull_call_spread(
            S=S_entry, K1=K1, K2=K2, dte=dte,
            atr=atr, vix=vix_at_entry,
            haircut=DEFAULT_HAIRCUT,
            sigma=post_earnings_iv,
        )

        # Total cost per contract (x100) + commission
        entry_cost_total = entry_cost_per_share * 100 + COMMISSION_RT_SPREAD

        if entry_cost_total <= 0 or entry_cost_total > pos_budget:
            continue

        # Number of contracts (usually 1 at this capital level)
        n_contracts = max(1, int(pos_budget / entry_cost_total))
        total_cost = entry_cost_total * n_contracts

        if total_cost > pos_budget:
            n_contracts = 1
            total_cost = entry_cost_total

        # ──── HOLD TO EXPIRY ────
        # Find the trading day closest to DTE days from entry
        expiry_idx = min(idx + dte, len(tk_data) - 1)
        S_expiry = float(tk_data['close'].iloc[expiry_idx])
        expiry_date = tk_data['date'].iloc[expiry_idx]

        # INTRINSIC VALUE ONLY at expiry (no BS pricing, no haircut)
        intrinsic_long = max(S_expiry - K1, 0.0)
        intrinsic_short = max(S_expiry - K2, 0.0)
        exit_value_per_share = intrinsic_long - intrinsic_short

        # PnL = (exit_value - entry_cost) * 100 * n_contracts - commission
        # Commission already included in entry_cost_total, so:
        pnl = (exit_value_per_share * 100 * n_contracts) - total_cost

        equity += pnl

        trades.append({
            'pnl': round(pnl, 2),
            'entry_date': str(dt.date()),
            'exit_date': str(expiry_date.date()) if hasattr(expiry_date, 'date') else str(expiry_date),
            'ticker': tk,
            'gap_pct': round(gap, 1),
            'entry_price': round(S_entry, 2),
            'expiry_price': round(S_expiry, 2),
            'K1': K1,
            'K2': K2,
            'entry_cost': round(total_cost, 2),
            'exit_value': round(exit_value_per_share * 100 * n_contracts, 2),
            'n_contracts': n_contracts,
            'dte': dte,
            'iv_used': round(post_earnings_iv, 4),
        })

    fprint(f"  Trades: {len(trades)}, Final equity: ${equity:.2f}")
    if trades:
        wins = sum(1 for t in trades if t['pnl'] > 0)
        fprint(f"  Win rate: {wins}/{len(trades)} = {wins/len(trades)*100:.1f}%")
        fprint(f"  Total PnL: ${sum(t['pnl'] for t in trades):.2f}")

    return trades


# ═══════════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ═══════════════════════════════════════════════════════════════════

def simulate_random_baseline(
    pead_trades: list,
    random_prices_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    dte: int = 45,
    n_trials: int = 5,
):
    """Random baseline: use same entry dates but random stocks.

    This measures how much of PEAD's edge comes from the earnings-gap selection
    vs. general market drift.
    """
    fprint(f"\n  Random baseline ({n_trials} trials, DTE={dte})...")

    if not pead_trades:
        return []

    # Get entry dates from PEAD trades
    entry_dates = [pd.Timestamp(t['entry_date']) for t in pead_trades]

    # Available random tickers
    available_tickers = random_prices_df['ticker'].unique()
    available_tickers = [t for t in available_tickers if t != 'SPY']

    vix_series = vix_df.set_index('date')['vix'] if 'date' in vix_df.columns else vix_df

    trial_sharpes = []

    for trial in range(n_trials):
        np.random.seed(42 + trial)
        equity = CAP
        trades = []

        for entry_dt in entry_dates:
            # Pick a random ticker
            tk = np.random.choice(available_tickers)
            tk_data = random_prices_df[random_prices_df['ticker'] == tk].sort_values('date').reset_index(drop=True)

            if len(tk_data) < 60:
                continue

            # Find nearest trading day
            date_diffs = (tk_data['date'] - entry_dt).abs()
            nearest_idx = date_diffs.idxmin()

            if nearest_idx + dte + 5 >= len(tk_data):
                continue
            if nearest_idx < 30:
                continue

            S_entry = float(tk_data['close'].iloc[nearest_idx])

            # ATR
            lookback_start = max(0, nearest_idx - 30)
            tk_slice = tk_data.iloc[lookback_start:nearest_idx]
            if len(tk_slice) < 14:
                continue
            atr = compute_atr(tk_slice['high'], tk_slice['low'], tk_slice['close'], period=14)

            vix_at_entry = 20.0
            nearest_vix = vix_series.index[vix_series.index.get_indexer([entry_dt], method='nearest')]
            if len(nearest_vix) > 0:
                vix_at_entry = float(vix_series.loc[nearest_vix[0]])

            iv = estimate_iv(atr, S_entry, vix_at_entry)

            K1 = round(S_entry, 2)
            K2 = round(S_entry * (1 + SPREAD_WIDTH_PCT), 2)

            entry_cost_ps, _ = price_bull_call_spread(
                S=S_entry, K1=K1, K2=K2, dte=dte,
                atr=atr, vix=vix_at_entry, sigma=iv,
            )
            entry_total = entry_cost_ps * 100 + COMMISSION_RT_SPREAD

            pos_budget = min(MAX_POSITION, equity * 0.35)
            if entry_total <= 0 or entry_total > pos_budget or pos_budget < 30:
                continue

            # Hold to expiry
            expiry_idx = min(nearest_idx + dte, len(tk_data) - 1)
            S_expiry = float(tk_data['close'].iloc[expiry_idx])
            expiry_date = tk_data['date'].iloc[expiry_idx]

            intrinsic = max(S_expiry - K1, 0) - max(S_expiry - K2, 0)
            pnl = intrinsic * 100 - entry_total
            equity += pnl

            trades.append({
                'pnl': round(pnl, 2),
                'entry_date': str(entry_dt.date()),
                'exit_date': str(expiry_date.date()) if hasattr(expiry_date, 'date') else str(expiry_date),
            })

        if len(trades) >= 10:
            result = validate_trades(
                trades, initial_capital=CAP,
                strategy_name=f"Random_Trial_{trial}",
                n_perms=200,
            )
            trial_sharpes.append(result.sharpe)
            fprint(f"    Trial {trial}: {len(trades)} trades, Sharpe={result.sharpe:.2f}")
        else:
            fprint(f"    Trial {trial}: {len(trades)} trades (too few)")

    if trial_sharpes:
        fprint(f"  Random baseline avg Sharpe: {np.mean(trial_sharpes):.2f} "
               f"(range {min(trial_sharpes):.2f} to {max(trial_sharpes):.2f})")
    return trial_sharpes


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    fprint("=" * 65)
    fprint("  PEAD HONEST REVALIDATION v1")
    fprint("  Hold-to-Expiry, Intrinsic Value Only, 15% Entry Haircut")
    fprint("=" * 65)
    fprint(f"  Start time: {datetime.now()}")
    fprint(f"  Capital: ${CAP}, Max position: ${MAX_POSITION}")
    fprint(f"  Commission: ${COMMISSION_RT_SPREAD}/spread RT")
    fprint(f"  Entry haircut: {DEFAULT_HAIRCUT*100:.0f}%")
    fprint(f"  Spread width: {SPREAD_WIDTH_PCT*100:.0f}%")
    fprint()

    # ──── Download data ────
    fprint("Step 1: Downloading price data...")
    prices_df = download_prices(TICKERS, "pead")
    vix_df = download_vix()

    # SPY prices for regime gate
    spy_data = prices_df[prices_df['ticker'] == 'SPY'].set_index('date')['close'].sort_index()

    # ──── Detect earnings ────
    fprint("\nStep 2: Detecting earnings gaps...")
    all_events = []
    for tk in TICKERS:
        events = detect_earnings_gaps(prices_df, tk, min_gap_pct=2.0)
        all_events.extend(events)
        if events:
            fprint(f"  {tk}: {len(events)} earnings gaps found")

    fprint(f"  Total earnings events: {len(all_events)}")
    gap_3 = sum(1 for e in all_events if e['gap_pct'] >= 3.0)
    gap_5 = sum(1 for e in all_events if e['gap_pct'] >= 5.0)
    fprint(f"  Gap >= 3%: {gap_3}, Gap >= 5%: {gap_5}")

    # ──── Variants ────
    fprint("\nStep 3: Running 4 honest variants...")

    variants = {
        'A_Gap3_30DTE': {'gap_threshold': 3.0, 'dte': 30},
        'B_Gap5_30DTE': {'gap_threshold': 5.0, 'dte': 30},
        'C_Gap5_45DTE': {'gap_threshold': 5.0, 'dte': 45},
        'D_Gap3_45DTE': {'gap_threshold': 3.0, 'dte': 45},
    }

    results = {}

    for vname, params in variants.items():
        trades = simulate_pead_honest(
            name=f"PEAD_{vname}",
            events=all_events,
            prices_df=prices_df,
            vix_df=vix_df,
            spy_prices=spy_data,
            **params,
        )

        if len(trades) < 10:
            fprint(f"  {vname}: Only {len(trades)} trades — skipping validation")
            results[vname] = {'n_trades': len(trades), 'error': 'too_few_trades'}
            continue

        # Run adversarial validation
        val_result = validate_trades(
            trades,
            initial_capital=CAP,
            spy_prices=spy_data,
            strategy_name=f"PEAD_{vname}_Honest",
            n_perms=2000,
        )
        val_result.print_summary()

        results[vname] = {
            'params': params,
            'validation': val_result.to_dict(),
            'trades': trades,
        }

    # ──── Random baseline for best variant ────
    fprint("\nStep 4: Random baseline comparison...")

    # Download random ticker data
    random_prices = download_prices(RANDOM_TICKERS, "random")

    # Find best variant by Sharpe
    best_variant = None
    best_sharpe = -999
    for vname, res in results.items():
        if 'validation' in res and res['validation'].get('sharpe', -999) > best_sharpe:
            best_sharpe = res['validation']['sharpe']
            best_variant = vname

    if best_variant and 'trades' in results[best_variant]:
        best_params = results[best_variant]['params']
        fprint(f"\n  Best variant: {best_variant} (Sharpe={best_sharpe:.2f})")
        fprint(f"  Running random baseline with DTE={best_params['dte']}...")

        random_sharpes = simulate_random_baseline(
            pead_trades=results[best_variant]['trades'],
            random_prices_df=random_prices,
            vix_df=vix_df,
            dte=best_params['dte'],
            n_trials=5,
        )

        if random_sharpes:
            results['random_baseline'] = {
                'avg_sharpe': round(float(np.mean(random_sharpes)), 3),
                'sharpes': [round(s, 3) for s in random_sharpes],
                'pead_sharpe': round(best_sharpe, 3),
                'incremental_pct': round(
                    (best_sharpe - np.mean(random_sharpes)) / max(abs(np.mean(random_sharpes)), 0.01) * 100, 1
                ),
            }
    else:
        fprint("  No valid variant to compare — skipping baseline")

    # ──── MLflow logging ────
    if MLFLOW_OK:
        fprint("\nStep 5: Logging to MLflow...")
        try:
            mlflow.set_experiment("pead_honest_revalidation")
            with mlflow.start_run(run_name="pead_honest_reval_v1"):
                mlflow.log_param("capital", CAP)
                mlflow.log_param("max_position", MAX_POSITION)
                mlflow.log_param("spread_width_pct", SPREAD_WIDTH_PCT)
                mlflow.log_param("haircut", DEFAULT_HAIRCUT)
                mlflow.log_param("commission_rt", COMMISSION_RT_SPREAD)
                mlflow.log_param("pricing", "hold_to_expiry_intrinsic_only")
                mlflow.log_param("n_tickers", len(TICKERS))
                mlflow.log_param("total_earnings_events", len(all_events))

                for vname, res in results.items():
                    if vname == 'random_baseline':
                        mlflow.log_metric("random_avg_sharpe", res['avg_sharpe'])
                        mlflow.log_metric("pead_incremental_pct", res['incremental_pct'])
                    elif 'validation' in res:
                        v = res['validation']
                        mlflow.log_metric(f"{vname}_sharpe", v['sharpe'])
                        mlflow.log_metric(f"{vname}_sortino", v['sortino'])
                        mlflow.log_metric(f"{vname}_cagr", v['cagr'])
                        mlflow.log_metric(f"{vname}_max_dd", v['max_dd'])
                        mlflow.log_metric(f"{vname}_win_rate", v['win_rate'])
                        mlflow.log_metric(f"{vname}_pf", v['profit_factor'])
                        mlflow.log_metric(f"{vname}_n_trades", v['n_trades'])
                        mlflow.log_metric(f"{vname}_gates_passed",
                                          v['gates_passed'])

            fprint("  MLflow logged successfully")
        except Exception as e:
            fprint(f"  MLflow logging failed: {e}")

    # ──── Summary ────
    fprint("\n" + "=" * 65)
    fprint("  FINAL COMPARISON: HONEST vs ORIGINAL")
    fprint("=" * 65)
    fprint(f"  {'Variant':<20} {'Sharpe':>8} {'CAGR':>8} {'WR':>7} {'MDD':>8} {'Trades':>7} {'Gates':>7}")
    fprint(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*7}")

    for vname, res in results.items():
        if vname == 'random_baseline':
            continue
        if 'validation' in res:
            v = res['validation']
            fprint(f"  {vname:<20} {v['sharpe']:>8.2f} {v['cagr']*100:>7.1f}% "
                   f"{v['win_rate']*100:>6.1f}% {v['max_dd']*100:>7.1f}% "
                   f"{v['n_trades']:>7} {v['gates_passed']}/{v['gates_total']}")
        elif 'error' in res:
            fprint(f"  {vname:<20} {'N/A':>8} {'N/A':>8} {'N/A':>7} {'N/A':>8} "
                   f"{res.get('n_trades', 0):>7} {'N/A':>7}")

    fprint(f"\n  Original C_Gap5_45DTE (old pricing): Sharpe=1.38, CAGR=55.9%, WR=65.3%, MDD=-29.1%")

    if 'random_baseline' in results:
        rb = results['random_baseline']
        fprint(f"\n  Random baseline avg Sharpe: {rb['avg_sharpe']:.2f}")
        fprint(f"  PEAD incremental value: {rb['incremental_pct']:.1f}%")

    # ──── Save results ────
    save_results = {}
    for vname, res in results.items():
        if vname == 'random_baseline':
            save_results[vname] = res
        elif 'validation' in res:
            save_results[vname] = {
                'params': res['params'],
                'validation': res['validation'],
                'n_trades': len(res.get('trades', [])),
            }
        else:
            save_results[vname] = res

    results_path = RESULTS_DIR / 'pead_honest_revalidation_v1.json'
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\n  Results saved to {results_path}")

    fprint(f"\n  Completed: {datetime.now()}")
    fprint("=" * 65)

    return results


if __name__ == '__main__':
    main()
