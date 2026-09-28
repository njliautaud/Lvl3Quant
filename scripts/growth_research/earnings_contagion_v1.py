#!/usr/bin/env python3
"""
Earnings Contagion v1 — Cross-Stock Earnings Signal Propagation
================================================================

HYPOTHESIS: When an early reporter in a sector beats/misses earnings and gaps,
late reporters in the same sector tend to move in the same direction BEFORE
their own earnings. This is "earnings contagion" — exploiting information
spillover from correlated companies.

Example: MSFT beats and gaps +5% → buy AAPL (reports 2 days later) → AAPL
tends to drift up before its own report, reflecting sector-wide optimism.

6 VARIANTS:
  A: Simple contagion — buy late reporters when early reporter gaps up, hold 2 days
  B: Magnitude-weighted — scale position by early reporter's gap size
  C: Sector consensus — wait for 2+ early reporters to agree, then trade late reporters
  D: Earnings anticipation — buy calls on late reporters (leverage the directional bet)
  E: Contagion + ML filter (LGBM predicts which contagion signals work)
  F: Reverse contagion — fade the contagion (contrarian: late reporters mean-revert)

UNIVERSE: 20 mega-cap tech + growth stocks in reporting clusters
PERIOD: 2022-01-01 to 2026-07-28
ACCOUNT: $645, equity + options
VALIDATION: 5-gate
"""

import sys
import os
import json
import warnings
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy import stats as sp_stats
from collections import defaultdict

warnings.filterwarnings('ignore')

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

LVL3_ROOT = '/home/jupiter/Lvl3Quant'
sys.path.insert(0, LVL3_ROOT)

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'earnings_contagion_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# EARNINGS REPORTING CLUSTERS
# ============================================================
# Companies that report in the same week/adjacent weeks and are correlated
# Format: {sector: [(ticker, typical_report_order_in_quarter), ...]}

SECTOR_CLUSTERS = {
    'mega_tech': ['MSFT', 'GOOGL', 'META', 'AAPL', 'AMZN'],
    'semis': ['AMD', 'NVDA', 'INTC', 'AVGO', 'QCOM'],
    'cloud_saas': ['CRM', 'NOW', 'SNOW', 'DDOG', 'NET'],
    'ecommerce': ['SHOP', 'MELI', 'SE', 'BABA', 'PDD'],
    'fintech': ['PYPL', 'SQ', 'COIN', 'HOOD', 'SOFI'],
    'social_media': ['SNAP', 'PINS', 'ROKU', 'TTD', 'RBLX'],
    'ev_energy': ['TSLA', 'RIVN', 'ENPH', 'FSLR', 'NIO'],
    'enterprise': ['ORCL', 'ADBE', 'INTU', 'PANW', 'CRWD'],
}

ALL_TICKERS = sorted(set(t for cluster in SECTOR_CLUSTERS.values() for t in cluster))

STARTING_CAPITAL = 645.0
MIN_GAP_PCT = 0.03  # Lower threshold — we want to detect sector signals, not just big moves
START_DATE = '2022-01-01'
END_DATE = '2026-07-28'
RISK_FREE_RATE = 0.05
N_PERMUTATIONS = 200

# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    import yfinance as yf

    cache_path = os.path.join(LVL3_ROOT, 'data', 'earnings_contagion_prices.parquet')
    earnings_cache = os.path.join(LVL3_ROOT, 'data', 'earnings_contagion_earnings.json')

    if os.path.exists(cache_path) and os.path.exists(earnings_cache):
        prices_df = pd.read_parquet(cache_path)
        with open(earnings_cache) as f:
            earnings_dates = json.load(f)
        fprint(f"Loaded cached data: {len(prices_df)} price rows, {len(earnings_dates)} tickers")
        return prices_df, earnings_dates

    fprint(f"Downloading {len(ALL_TICKERS)} tickers + SPY + VIX...")
    download_tickers = list(set(ALL_TICKERS)) + ['SPY', '^VIX']
    frames = []
    for t in download_tickers:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if len(df) < 50: continue
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            df['ticker'] = t
            df.index.name = 'date'
            frames.append(df)
        except Exception as e:
            fprint(f"  SKIP {t}: {e}")

    prices_df = pd.concat(frames).reset_index().set_index(['ticker', 'date']).sort_index()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    prices_df.to_parquet(cache_path)
    fprint(f"Saved {len(prices_df)} price rows")

    fprint("Fetching earnings dates...")
    earnings_dates = {}
    for t in ALL_TICKERS:
        try:
            stock = yf.Ticker(t)
            dates = stock.get_earnings_dates(limit=30)
            if dates is not None and len(dates) > 0:
                earnings_dates[t] = sorted([str(d.date()) for d in dates.index])
        except Exception:
            pass

    with open(earnings_cache, 'w') as f:
        json.dump(earnings_dates, f)
    fprint(f"Cached {len(earnings_dates)} tickers' earnings dates")
    return prices_df, earnings_dates


# ============================================================
# BUILD CONTAGION EVENTS
# ============================================================

def find_earnings_gaps(prices_df, earnings_dates, ticker):
    """Find all earnings gaps for a ticker."""
    try:
        ticker_prices = prices_df.loc[ticker].sort_index()
    except KeyError:
        return []

    if len(ticker_prices) < 30:
        return []

    trading_days = ticker_prices.index.sort_values()
    gaps = []

    if ticker not in earnings_dates:
        return []

    for earn_date_str in earnings_dates[ticker]:
        earn_date = pd.Timestamp(earn_date_str)

        post_mask = trading_days >= earn_date
        if not post_mask.any(): continue
        post_days = trading_days[post_mask]
        if len(post_days) < 6: continue

        pre_mask = trading_days < earn_date
        if not pre_mask.any(): continue
        pre_days = trading_days[pre_mask]
        if len(pre_days) < 5: continue

        close_before = float(ticker_prices.loc[pre_days[-1], 'close'])
        open_after = float(ticker_prices.loc[post_days[0], 'open'])

        if close_before <= 0 or open_after <= 0: continue

        gap_pct = (open_after / close_before) - 1.0

        # Get post-earnings closes for drift measurement
        post_close = [float(ticker_prices.loc[d, 'close']) for d in post_days[:6]]

        gaps.append({
            'ticker': ticker,
            'earn_date': earn_date_str,
            'earn_ts': earn_date,
            'gap_pct': gap_pct,
            'abs_gap': abs(gap_pct),
            'close_before': close_before,
            'open_after': open_after,
            'post_closes': post_close,
        })

    return gaps


def build_contagion_events(prices_df, earnings_dates):
    """
    For each sector cluster, find cases where an early reporter's earnings
    gap can be used to trade a late reporter.

    A contagion event = early reporter gapped, late reporter reports within 14 days.
    We buy/sell late reporter AFTER early reporter's gap, BEFORE late reporter's earnings.
    """
    fprint("\n--- Building Contagion Events ---")

    # Get all earnings gaps for all tickers
    all_gaps = {}
    for ticker in ALL_TICKERS:
        gaps = find_earnings_gaps(prices_df, earnings_dates, ticker)
        if gaps:
            all_gaps[ticker] = gaps

    fprint(f"Found earnings gaps for {len(all_gaps)} tickers")

    contagion_events = []
    available_tickers = set(prices_df.index.get_level_values(0).unique())

    # For each sector cluster, find early→late reporter pairs
    for sector, tickers in SECTOR_CLUSTERS.items():
        sector_tickers = [t for t in tickers if t in all_gaps]
        if len(sector_tickers) < 2:
            continue

        # Collect all earnings events in this sector
        sector_events = []
        for t in sector_tickers:
            for gap in all_gaps[t]:
                sector_events.append(gap)

        # Sort by date
        sector_events.sort(key=lambda x: x['earn_ts'])

        # For each pair: early reporter → late reporter within 14 calendar days
        for i, early in enumerate(sector_events):
            for j, late in enumerate(sector_events):
                if i == j:
                    continue
                if early['ticker'] == late['ticker']:
                    continue

                days_diff = (late['earn_ts'] - early['earn_ts']).days
                if days_diff < 1 or days_diff > 14:
                    continue

                # Early reporter had a meaningful gap
                if early['abs_gap'] < MIN_GAP_PCT:
                    continue

                # We want to trade the late reporter AFTER early reporter's earnings
                # Entry: day after early reporter's earnings
                # Exit: day before late reporter's earnings (or hold through)
                late_ticker = late['ticker']
                if late_ticker not in available_tickers:
                    continue

                try:
                    late_prices = prices_df.loc[late_ticker].sort_index()
                except KeyError:
                    continue

                trading_days = late_prices.index.sort_values()

                # Find entry day (first trading day after early reporter's earnings)
                entry_mask = trading_days > early['earn_ts']
                if not entry_mask.any():
                    continue
                entry_days = trading_days[entry_mask]
                if len(entry_days) < 3:
                    continue

                entry_date = entry_days[0]
                entry_price = float(late_prices.loc[entry_date, 'open'])
                if entry_price <= 0:
                    continue

                # Exit: 2 trading days after entry OR 1 day before late reporter's earnings
                # (whichever comes first)
                exit_idx = min(2, len(entry_days) - 1)
                # Don't hold through late reporter's own earnings (that's a different bet)
                for k in range(exit_idx + 1):
                    if entry_days[k] >= late['earn_ts']:
                        exit_idx = max(k - 1, 0)
                        break

                if exit_idx < 1:
                    # Not enough time to trade
                    continue

                exit_date = entry_days[exit_idx]
                exit_price = float(late_prices.loc[exit_date, 'close'])
                if exit_price <= 0:
                    continue

                # Signal: trade in direction of early reporter's gap
                signal_direction = 1 if early['gap_pct'] > 0 else -1

                if signal_direction == 1:
                    pnl_pct = (exit_price / entry_price) - 1.0
                else:
                    pnl_pct = 1.0 - (exit_price / entry_price)

                # Get SPY for regime
                try:
                    spy = prices_df.loc['SPY']
                    spy_close = spy.loc[spy.index <= entry_date, 'close']
                    if len(spy_close) >= 22:
                        spy_mom = float(spy_close.iloc[-1] / spy_close.iloc[-21] - 1)
                    else:
                        spy_mom = 0
                except:
                    spy_mom = 0

                # Get VIX
                try:
                    vix_data = prices_df.loc['^VIX']
                    vix_close = vix_data.loc[vix_data.index <= entry_date, 'close']
                    vix_level = float(vix_close.iloc[-1]) if len(vix_close) > 0 else 20
                except:
                    vix_level = 20

                # Features for ML
                feats = {
                    'early_gap_pct': early['gap_pct'],
                    'early_abs_gap': early['abs_gap'],
                    'early_direction': 1 if early['gap_pct'] > 0 else -1,
                    'days_between': days_diff,
                    'sector': sector,
                    'early_ticker': early['ticker'],
                    'late_ticker': late_ticker,
                    'entry_date': str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date),
                    'exit_date': str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'signal_direction': signal_direction,
                    'pnl_pct': pnl_pct,
                    'hold_days': exit_idx,
                    'spy_momentum': spy_mom,
                    'vix_level': vix_level,
                    'spy_regime': 'bull' if spy_mom > 0 else 'bear',
                    # ML features
                    'early_gap_magnitude_tier': 3 if early['abs_gap'] >= 0.15 else (2 if early['abs_gap'] >= 0.08 else 1),
                    'vix_normalized': vix_level / 20.0,
                    'same_sector_correlation': 1.0,  # placeholder
                }
                contagion_events.append(feats)

    fprint(f"\nTotal contagion events: {len(contagion_events)}")

    # Analyze the raw signal
    if contagion_events:
        pnls = [e['pnl_pct'] for e in contagion_events]
        wins = [p for p in pnls if p > 0]
        fprint(f"  Raw signal: WR {len(wins)/len(pnls)*100:.1f}%, avg PnL {np.mean(pnls)*100:.3f}%")
        fprint(f"  Sectors: {len(SECTOR_CLUSTERS)}, tickers: {len(ALL_TICKERS)}")

        # Per-sector breakdown
        fprint("\n  Per-sector breakdown:")
        for sector in SECTOR_CLUSTERS:
            sec_events = [e for e in contagion_events if e['sector'] == sector]
            if sec_events:
                sec_pnls = [e['pnl_pct'] for e in sec_events]
                sec_wr = len([p for p in sec_pnls if p > 0]) / len(sec_pnls) * 100
                fprint(f"    {sector}: {len(sec_events)} events, WR {sec_wr:.1f}%, avg {np.mean(sec_pnls)*100:.3f}%")

        # Direction analysis
        up_events = [e for e in contagion_events if e['signal_direction'] == 1]
        dn_events = [e for e in contagion_events if e['signal_direction'] == -1]
        if up_events:
            up_pnls = [e['pnl_pct'] for e in up_events]
            fprint(f"\n  Up contagion: {len(up_events)} events, WR {len([p for p in up_pnls if p > 0])/len(up_pnls)*100:.1f}%, avg {np.mean(up_pnls)*100:.3f}%")
        if dn_events:
            dn_pnls = [e['pnl_pct'] for e in dn_events]
            fprint(f"  Down contagion: {len(dn_events)} events, WR {len([p for p in dn_pnls if p > 0])/len(dn_pnls)*100:.1f}%, avg {np.mean(dn_pnls)*100:.3f}%")

    return contagion_events


# ============================================================
# ML FEATURES FOR CONTAGION
# ============================================================

ML_FEATURES = [
    'early_abs_gap', 'early_direction', 'days_between',
    'early_gap_magnitude_tier', 'vix_normalized', 'spy_momentum',
]


def train_contagion_lgbm(X_train, y_train):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=80, max_depth=2, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# BS PRICING FOR OPTIONS VARIANT
# ============================================================

def bs_call(S, K, T, r, sigma):
    if T <= 1e-8: return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 1e-8: return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ============================================================
# VARIANT CONFIGS
# ============================================================

VARIANT_CONFIGS = {
    'A': {
        'name': 'Simple Contagion (follow early reporter)',
        'trade_type': 'equity',
        'position_pct': 0.20,
        'min_early_gap': 0.03,
        'consensus_required': 1,
        'use_ml': False,
        'fade': False,
    },
    'B': {
        'name': 'Magnitude-Weighted Contagion',
        'trade_type': 'equity',
        'position_pct': 0.15,
        'min_early_gap': 0.03,
        'consensus_required': 1,
        'use_ml': False,
        'fade': False,
        'magnitude_weight': True,
    },
    'C': {
        'name': 'Sector Consensus (2+ agree)',
        'trade_type': 'equity',
        'position_pct': 0.25,
        'min_early_gap': 0.03,
        'consensus_required': 2,
        'use_ml': False,
        'fade': False,
    },
    'D': {
        'name': 'Options Contagion (buy calls/puts)',
        'trade_type': 'options',
        'position_pct': 0.15,
        'min_early_gap': 0.05,
        'consensus_required': 1,
        'use_ml': False,
        'fade': False,
    },
    'E': {
        'name': 'ML-Filtered Contagion (LGBM)',
        'trade_type': 'equity',
        'position_pct': 0.20,
        'min_early_gap': 0.03,
        'consensus_required': 1,
        'use_ml': True,
        'ml_threshold': 0.55,
        'fade': False,
    },
    'F': {
        'name': 'Reverse Contagion (fade the signal)',
        'trade_type': 'equity',
        'position_pct': 0.20,
        'min_early_gap': 0.05,
        'consensus_required': 1,
        'use_ml': False,
        'fade': True,
    },
}


# ============================================================
# WALK-FORWARD SIMULATION
# ============================================================

def run_variant(variant_key, cfg, contagion_events):
    fprint(f"\n{'='*60}")
    fprint(f"  VARIANT {variant_key}: {cfg['name']}")
    fprint(f"{'='*60}")

    # Filter by minimum early gap
    events = [e for e in contagion_events if e['early_abs_gap'] >= cfg['min_early_gap']]
    events.sort(key=lambda x: x['entry_date'])

    if not events:
        fprint("  No qualifying events")
        return None

    # For consensus variant, group events by late_ticker + entry_date
    # and require N early reporters to agree
    if cfg['consensus_required'] > 1:
        grouped = defaultdict(list)
        for e in events:
            key = (e['late_ticker'], e['entry_date'])
            grouped[key].append(e)

        consensus_events = []
        for key, group in grouped.items():
            directions = [e['signal_direction'] for e in group]
            # Check if consensus_required agree on direction
            up_count = sum(1 for d in directions if d == 1)
            dn_count = sum(1 for d in directions if d == -1)
            if up_count >= cfg['consensus_required']:
                best = max(group, key=lambda x: x['early_abs_gap'])
                best['consensus_count'] = up_count
                consensus_events.append(best)
            elif dn_count >= cfg['consensus_required']:
                best = max(group, key=lambda x: x['early_abs_gap'])
                best['signal_direction'] = -1
                best['consensus_count'] = dn_count
                consensus_events.append(best)
        events = sorted(consensus_events, key=lambda x: x['entry_date'])
        fprint(f"  After consensus filter: {len(events)} events")

    if not events:
        fprint("  No events after filtering")
        return None

    # Walk-forward
    min_train = 20 if cfg['use_ml'] else 0
    equity = STARTING_CAPITAL
    trades = []
    ticker_trade_count = defaultdict(int)
    peak_equity = equity
    max_dd = 0

    # Deduplicate: don't trade same late_ticker on same date twice
    seen = set()

    for i in range(min_train, len(events)):
        event = events[i]

        dedup_key = (event['late_ticker'], event['entry_date'])
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        # Ticker trade cap
        if ticker_trade_count[event['late_ticker']] >= 5:
            continue

        # ML filter
        if cfg['use_ml']:
            train_events = events[:i]
            X_train = pd.DataFrame(train_events)[ML_FEATURES].fillna(0).values
            y_train = np.array([1 if e['pnl_pct'] > 0 else 0 for e in train_events])

            if len(np.unique(y_train)) < 2:
                continue

            model = train_contagion_lgbm(X_train, y_train)
            X_test = pd.DataFrame([event])[ML_FEATURES].fillna(0).values
            prob = model.predict_proba(X_test)[0][1]

            if prob < cfg.get('ml_threshold', 0.55):
                continue
        else:
            prob = None

        # Position sizing
        base_size = equity * cfg['position_pct']

        if cfg.get('magnitude_weight'):
            # Scale by early reporter's gap magnitude
            mag_mult = min(event['early_abs_gap'] / 0.05, 3.0)
            base_size *= mag_mult

        base_size = min(base_size, equity * 0.30)  # cap
        if base_size < 10:
            continue

        # Determine direction
        signal_dir = event['signal_direction']
        if cfg.get('fade'):
            signal_dir = -signal_dir  # reverse

        # Compute PnL
        if cfg['trade_type'] == 'equity':
            if signal_dir == 1:
                pnl_pct = (event['exit_price'] / event['entry_price']) - 1.0
            else:
                pnl_pct = 1.0 - (event['exit_price'] / event['entry_price'])

            # Slippage
            shares = base_size / max(event['entry_price'], 1)
            slippage = shares * 0.01 * 2
            trade_pnl = base_size * pnl_pct - slippage

        elif cfg['trade_type'] == 'options':
            # Buy ATM call or put, hold 2 days
            S = event['entry_price']
            K = round(S)  # ATM
            T_entry = 14 / 365.0  # ~2 weeks to expiry
            T_exit = max((14 - event['hold_days']) / 365.0, 1/365.0)
            iv = max(event['vix_level'] / 100 * 1.3, 0.20)  # BS*1.3 haircut

            if signal_dir == 1:
                entry_premium = bs_call(S, K, T_entry, RISK_FREE_RATE, iv) * 1.3  # haircut
                exit_premium = bs_call(event['exit_price'], K, T_exit, RISK_FREE_RATE, iv)
            else:
                entry_premium = bs_put(S, K, T_entry, RISK_FREE_RATE, iv) * 1.3
                exit_premium = bs_put(event['exit_price'], K, T_exit, RISK_FREE_RATE, iv)

            if entry_premium < 0.10:
                continue

            n_contracts = max(int(base_size / (entry_premium * 100)), 1)
            commission = 0.65 * n_contracts * 2  # RT
            trade_pnl = (exit_premium - entry_premium) * 100 * n_contracts - commission

            pnl_pct = trade_pnl / (entry_premium * 100 * n_contracts) if entry_premium > 0 else 0

        equity += trade_pnl
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        max_dd = min(max_dd, dd)

        if equity <= 0:
            fprint("  BLOWN OUT")
            break

        trade = {
            'date': event['entry_date'],
            'early_ticker': event['early_ticker'],
            'late_ticker': event['late_ticker'],
            'sector': event['sector'],
            'direction': 'long' if signal_dir == 1 else 'short',
            'early_gap_pct': event['early_gap_pct'],
            'pnl_pct': pnl_pct,
            'pnl_usd': trade_pnl,
            'equity': equity,
            'position_size': base_size,
            'ml_prob': prob,
            'spy_regime': event.get('spy_regime', 'unknown'),
        }
        trades.append(trade)
        ticker_trade_count[event['late_ticker']] += 1

    if not trades:
        fprint("  No trades generated")
        return None

    return analyze_results(variant_key, cfg, trades)


# ============================================================
# ANALYSIS
# ============================================================

def analyze_results(variant_key, cfg, trades):
    n_trades = len(trades)
    pnls = [t['pnl_usd'] for t in trades]
    pnl_pcts = [t['pnl_pct'] for t in trades]
    final_equity = trades[-1]['equity']

    total_return = (final_equity / STARTING_CAPITAL - 1) * 100
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    win_rate = len(winners) / n_trades * 100
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0
    profit_factor = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float('inf')

    # Drawdown
    peak = STARTING_CAPITAL
    max_dd = 0
    for t in trades:
        peak = max(peak, t['equity'])
        dd = (t['equity'] - peak) / peak
        max_dd = min(max_dd, dd)

    # Sharpe / Sortino
    if len(pnl_pcts) > 1:
        trades_per_year = max(n_trades / 4.5, 1)
        annualization = np.sqrt(max(trades_per_year, 1))
        mean_ret = np.mean(pnl_pcts)
        std_ret = np.std(pnl_pcts) if np.std(pnl_pcts) > 0 else 1e-6
        sharpe = (mean_ret / std_ret) * annualization
        downside = np.std([r for r in pnl_pcts if r < 0]) if any(r < 0 for r in pnl_pcts) else std_ret
        sortino = (mean_ret / downside) * annualization if downside > 0 else sharpe
    else:
        sharpe = sortino = 0
        annualization = 1

    fprint(f"\n  Trades: {n_trades}, WR: {win_rate:.1f}%, PF: {profit_factor:.2f}")
    fprint(f"  ${STARTING_CAPITAL:.0f} -> ${final_equity:.0f} ({total_return:+.1f}%)")
    fprint(f"  Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}, MDD: {max_dd*100:.1f}%")

    # Sector breakdown
    fprint("\n  Sector breakdown:")
    for sector in SECTOR_CLUSTERS:
        sec_trades = [t for t in trades if t['sector'] == sector]
        if sec_trades:
            sec_wr = len([t for t in sec_trades if t['pnl_usd'] > 0]) / len(sec_trades) * 100
            sec_pnl = sum(t['pnl_usd'] for t in sec_trades)
            fprint(f"    {sector}: {len(sec_trades)} trades, WR {sec_wr:.1f}%, PnL ${sec_pnl:.2f}")

    # Direction breakdown
    long_trades = [t for t in trades if t['direction'] == 'long']
    short_trades = [t for t in trades if t['direction'] == 'short']
    fprint(f"\n  Long: {len(long_trades)}, Short: {len(short_trades)}")

    # ============================================================
    # 5-GATE VALIDATION
    # ============================================================
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates['sharpe_pass'] = sharpe > 0.5
    fprint(f"\n  Gate 1 (Sharpe > 0.5): {'PASS' if gates['sharpe_pass'] else 'FAIL'} ({sharpe:.2f})")

    # Gate 2: Permutation test
    np.random.seed(42)
    perm_sharpes = []
    for _ in range(N_PERMUTATIONS):
        shuffled = np.random.permutation(pnl_pcts)
        if np.std(shuffled) > 0:
            perm_sharpes.append(np.mean(shuffled) / np.std(shuffled) * annualization)
        else:
            perm_sharpes.append(0)
    p_value = np.mean([1 for ps in perm_sharpes if ps >= sharpe])
    gates['perm_pass'] = p_value < 0.05
    fprint(f"  Gate 2 (Perm p<0.05): {'PASS' if gates['perm_pass'] else 'FAIL'} (p={p_value:.3f})")

    # Gate 3: Beats random
    random_sharpes = []
    for _ in range(100):
        random_rets = np.random.choice(pnl_pcts, size=n_trades, replace=True)
        if np.std(random_rets) > 0:
            random_sharpes.append(np.mean(random_rets) / np.std(random_rets) * annualization)
    beats_random = np.mean([1 for rs in random_sharpes if sharpe > rs])
    gates['random_pass'] = beats_random > 0.6
    fprint(f"  Gate 3 (Beats random): {'PASS' if gates['random_pass'] else 'FAIL'} ({beats_random:.1%})")

    # Gate 4: Regime gap
    bull_trades = [t['pnl_pct'] for t in trades if t.get('spy_regime') == 'bull']
    bear_trades = [t['pnl_pct'] for t in trades if t.get('spy_regime') == 'bear']
    if bull_trades and bear_trades and len(bull_trades) >= 3 and len(bear_trades) >= 3:
        bull_sharpe = np.mean(bull_trades) / max(np.std(bull_trades), 1e-6) * annualization
        bear_sharpe = np.mean(bear_trades) / max(np.std(bear_trades), 1e-6) * annualization
        regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)
        gates['regime_pass'] = regime_gap < 0.50
        fprint(f"  Gate 4 (Regime gap<0.50): {'PASS' if gates['regime_pass'] else 'FAIL'} "
               f"(gap={regime_gap:.2f}, bull={bull_sharpe:.2f}, bear={bear_sharpe:.2f})")
    else:
        gates['regime_pass'] = False
        fprint(f"  Gate 4 (Regime gap<0.50): FAIL (insufficient regime data: bull={len(bull_trades)}, bear={len(bear_trades)})")

    # Gate 5: MDD > -50%
    gates['mdd_pass'] = max_dd > -0.50
    fprint(f"  Gate 5 (MDD > -50%): {'PASS' if gates['mdd_pass'] else 'FAIL'} ({max_dd*100:.1f}%)")

    gates_passed = sum(gates.values())
    fprint(f"\n  GATES PASSED: {gates_passed}/5")

    result = {
        'variant': variant_key,
        'name': cfg['name'],
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'sortino': sortino,
        'total_return_pct': total_return,
        'final_equity': final_equity,
        'max_drawdown': max_dd,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'gates_passed': gates_passed,
        'gates': gates,
        'p_value': p_value,
        'trades': trades,
    }
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    fprint("=" * 70)
    fprint("  EARNINGS CONTAGION v1")
    fprint("  Testing: Do early reporters' earnings predict late reporters?")
    fprint("=" * 70)
    start_time = time.time()

    prices_df, earnings_dates = load_data()
    fprint(f"\nPrices: {len(prices_df)} rows, Earnings: {len(earnings_dates)} tickers")

    contagion_events = build_contagion_events(prices_df, earnings_dates)
    if not contagion_events:
        fprint("ERROR: No contagion events found")
        return

    # MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri('sqlite:////home/jupiter/teleclaude-main/mlflow.db')
        mlflow.set_experiment('earnings_contagion_v1')

    results = {}
    for vk in sorted(VARIANT_CONFIGS.keys()):
        cfg = VARIANT_CONFIGS[vk]
        try:
            result = run_variant(vk, cfg, contagion_events)
            if result:
                results[vk] = result

                if MLFLOW_AVAILABLE:
                    with mlflow.start_run(run_name=f"variant_{vk}_{cfg['name'][:30]}"):
                        mlflow.log_params({
                            'variant': vk,
                            'trade_type': cfg['trade_type'],
                            'consensus_required': cfg['consensus_required'],
                            'use_ml': cfg.get('use_ml', False),
                            'fade': cfg.get('fade', False),
                            'min_early_gap': cfg['min_early_gap'],
                        })
                        mlflow.log_metrics({
                            'sharpe': result['sharpe'],
                            'sortino': result['sortino'],
                            'win_rate': result['win_rate'],
                            'profit_factor': min(result['profit_factor'], 100),
                            'total_return_pct': result['total_return_pct'],
                            'max_drawdown': result['max_drawdown'],
                            'n_trades': result['n_trades'],
                            'gates_passed': result['gates_passed'],
                            'p_value': result.get('p_value', 1.0),
                        })
        except Exception as e:
            fprint(f"\n  ERROR in variant {vk}: {e}")
            traceback.print_exc()

    # ============================================================
    # SUMMARY
    # ============================================================
    fprint(f"\n{'='*70}")
    fprint(f"  SUMMARY — EARNINGS CONTAGION")
    fprint(f"{'='*70}")

    if not results:
        fprint("No variants produced results")
        return

    header = f"{'Var':>3} | {'Name':<40} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'Sharpe':>6} | {'Sortino':>7} | {'Return%':>8} | {'MDD%':>6} | {'Gates':>5}"
    fprint(header)
    fprint("-" * len(header))
    for vk in sorted(results.keys()):
        r = results[vk]
        fprint(f"  {vk} | {r['name']:<40} | {r['n_trades']:>6} | {r['win_rate']:>5.1f} | {r['profit_factor']:>5.2f} | {r['sharpe']:>6.2f} | {r['sortino']:>7.2f} | {r['total_return_pct']:>+7.1f}% | {r['max_drawdown']*100:>5.1f}% | {r['gates_passed']}/5")

    best_key = max(results, key=lambda k: results[k]['sharpe'])
    best = results[best_key]
    fprint(f"\n  BEST: Variant {best_key} ({best['name']})")
    fprint(f"  Sharpe {best['sharpe']:.2f}, ${STARTING_CAPITAL:.0f} -> ${best['final_equity']:.0f}, "
           f"{best['gates_passed']}/5 gates")

    # Save results
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'trades'} for k, v in results.items()}
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\n  Results saved to output/growth_research/earnings_contagion_v1/")

    elapsed = time.time() - start_time
    fprint(f"\n  Total time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
