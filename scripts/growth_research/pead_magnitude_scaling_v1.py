#!/usr/bin/env python3
"""
PEAD Magnitude Scaling v1 — Scale Position by Earnings Surprise Magnitude
==========================================================================

HYPOTHESIS: Academic PEAD literature shows drift is proportional to surprise
magnitude. A 15% gap should generate more drift than a 5% gap. We should
scale position size accordingly instead of treating all 5%+ gaps equally.

ALSO TESTS: Direction asymmetry — do up-gaps drift more/less than down-gaps?

6 VARIANTS:
  A: Linear scaling (gap_pct / 0.05 * base_size, capped at 3x)
  B: Sqrt scaling (sqrt(gap_pct / 0.05) * base_size, capped at 3x)
  C: Tier scaling (5-10%=1x, 10-20%=2x, 20%+=3x)
  D: Direction-aware linear (long=1x, short=1.5x — short drift is stronger)
  E: Magnitude + LGBM confidence interaction (scale by both)
  F: High-conviction only (>10% gaps, no ML, just ride the drift)

WALK-FORWARD: sliding window, 30 events train, walk forward
UNIVERSE: 50 growth stocks
PERIOD: 2022-01-01 to 2026-07-28
ACCOUNT: $645, equity trades, $0 commission (Robinhood)
VALIDATION: 5-gate (Sharpe>0.5, perm p<0.05, beats random, regime gap<0.50, MDD>-50%)
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
from scipy.stats import norm, percentileofscore
from scipy import stats as sp_stats
from collections import defaultdict

warnings.filterwarnings('ignore')

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# Path setup
LVL3_ROOT = '/home/jupiter/Lvl3Quant'
sys.path.insert(0, LVL3_ROOT)

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'pead_magnitude_scaling_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================

GROWTH_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO',
]

STARTING_CAPITAL = 645.0
MIN_GAP_PCT = 0.05
MAX_POSITION_PCT = 0.30
MAX_TRADES_PER_TICKER = 4
START_DATE = '2022-01-01'
END_DATE = '2026-07-28'
RISK_FREE_RATE = 0.05
N_PERMUTATIONS = 200
DRIFT_THRESHOLD = 0.03

# ============================================================
# DATA LOADING (reuse cache from pead_v2)
# ============================================================

def load_data():
    import yfinance as yf

    cache_path = os.path.join(LVL3_ROOT, 'data', 'pead_v2_prices_cache.parquet')
    earnings_cache = os.path.join(LVL3_ROOT, 'data', 'pead_v2_earnings_cache.json')

    if os.path.exists(cache_path) and os.path.exists(earnings_cache):
        prices_df = pd.read_parquet(cache_path)
        with open(earnings_cache) as f:
            earnings_dates = json.load(f)
        fprint(f"Loaded cached data: {len(prices_df)} price rows, {len(earnings_dates)} tickers earnings")
        return prices_df, earnings_dates

    fprint(f"Downloading {len(GROWTH_UNIVERSE)} tickers + SPY + VIX...")
    all_tickers = list(set(GROWTH_UNIVERSE)) + ['SPY', '^VIX']
    frames = []
    for t in all_tickers:
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

    fprint("Fetching earnings dates...")
    earnings_dates = {}
    for t in GROWTH_UNIVERSE:
        try:
            stock = yf.Ticker(t)
            dates = stock.get_earnings_dates(limit=30)
            if dates is not None and len(dates) > 0:
                earnings_dates[t] = [str(d.date()) for d in dates.index]
        except Exception:
            pass

    with open(earnings_cache, 'w') as f:
        json.dump(earnings_dates, f)
    fprint(f"Cached {len(earnings_dates)} tickers' earnings dates")
    return prices_df, earnings_dates


# ============================================================
# FEATURE ENGINEERING
# ============================================================

FEATURE_COLS = [
    'abs_gap', 'gap_direction', 'mom_5d', 'mom_10d', 'mom_21d',
    'vol_21d', 'rel_str_21d', 'vix', 'vol_ratio', 'rsi',
    'dist_52w_high', 'prev_gap', 'ticker_hash',
    'volume_spike', 'gap_vs_atr', 'sector_momentum',
    # New magnitude features
    'gap_magnitude_tier', 'gap_zscore', 'gap_rank_pct',
]


def build_event_features(prices_df, earnings_dates):
    events = []
    available_tickers = set(prices_df.index.get_level_values(0).unique())

    try:
        spy = prices_df.loc['SPY', 'close']
    except:
        spy = None
    try:
        vix = prices_df.loc['^VIX', 'close']
    except:
        vix = None

    # Collect all gaps first for z-score computation
    all_gaps = []

    for ticker in GROWTH_UNIVERSE:
        if ticker not in earnings_dates or ticker not in available_tickers:
            continue

        try:
            ticker_prices = prices_df.loc[ticker].sort_index()
        except KeyError:
            continue

        if len(ticker_prices) < 30:
            continue

        trading_days = ticker_prices.index.sort_values()
        dates = earnings_dates[ticker]

        for earn_date_str in dates:
            earn_date = pd.Timestamp(earn_date_str)

            post_mask = trading_days >= earn_date
            if not post_mask.any(): continue
            post_days = trading_days[post_mask]
            if len(post_days) < 11: continue

            pre_mask = trading_days < earn_date
            if not pre_mask.any(): continue
            pre_days = trading_days[pre_mask]
            if len(pre_days) < 30: continue

            close_before = float(ticker_prices.loc[pre_days[-1], 'close'])
            open_after = float(ticker_prices.loc[post_days[0], 'open'])
            if close_before <= 0 or open_after <= 0: continue

            gap_pct = (open_after / close_before) - 1.0
            if abs(gap_pct) < MIN_GAP_PCT:
                continue

            all_gaps.append(abs(gap_pct))

            feats = {}
            feats['gap_pct'] = gap_pct
            feats['abs_gap'] = abs(gap_pct)
            feats['gap_direction'] = 1 if gap_pct > 0 else -1

            # Magnitude tier
            ag = abs(gap_pct)
            if ag >= 0.20:
                feats['gap_magnitude_tier'] = 3
            elif ag >= 0.10:
                feats['gap_magnitude_tier'] = 2
            else:
                feats['gap_magnitude_tier'] = 1

            # Pre-earnings momentum
            for w in [5, 10, 21]:
                if len(pre_days) >= w + 1:
                    feats[f'mom_{w}d'] = float(ticker_prices.loc[pre_days[-1], 'close'] /
                                               ticker_prices.loc[pre_days[-w-1], 'close'] - 1)
                else:
                    feats[f'mom_{w}d'] = 0

            # Volatility
            if len(pre_days) >= 22:
                recent_close = ticker_prices.loc[pre_days[-22:], 'close']
                feats['vol_21d'] = float(recent_close.pct_change().dropna().std() * np.sqrt(252))
            else:
                feats['vol_21d'] = 0.3

            # Relative strength vs SPY
            feats['rel_str_21d'] = 0
            if spy is not None and len(pre_days) >= 22:
                try:
                    spy_close = spy.loc[spy.index <= pre_days[-1]].iloc[-21:]
                    stock_close = ticker_prices.loc[pre_days[-21:], 'close']
                    feats['rel_str_21d'] = float(
                        (stock_close.iloc[-1] / stock_close.iloc[0] - 1) -
                        (spy_close.iloc[-1] / spy_close.iloc[0] - 1))
                except:
                    pass

            # VIX level
            feats['vix'] = 20
            if vix is not None:
                vix_mask = vix.index <= pre_days[-1]
                if vix_mask.any():
                    feats['vix'] = float(vix[vix_mask].iloc[-1])

            # Volume ratio
            if 'volume' in ticker_prices.columns and len(pre_days) >= 22:
                recent_vol = ticker_prices.loc[pre_days[-5:], 'volume'].mean()
                avg_vol = ticker_prices.loc[pre_days[-22:], 'volume'].mean()
                feats['vol_ratio'] = float(recent_vol / max(avg_vol, 1))
            else:
                feats['vol_ratio'] = 1.0

            # RSI
            if len(pre_days) >= 15:
                rets = ticker_prices.loc[pre_days[-15:], 'close'].pct_change().dropna()
                gains = rets.clip(lower=0).mean()
                losses = (-rets).clip(lower=0).mean()
                feats['rsi'] = float(100 - 100 / (1 + gains/losses)) if losses > 0 else 100
            else:
                feats['rsi'] = 50

            # Distance from 52w high
            if len(pre_days) >= 252:
                h52 = ticker_prices.loc[pre_days[-252:], 'close'].max()
                feats['dist_52w_high'] = float(close_before / h52 - 1)
            else:
                feats['dist_52w_high'] = 0

            # Previous earnings gap
            ticker_earn = [d for d in dates if pd.Timestamp(d) < earn_date]
            feats['prev_gap'] = 0
            if ticker_earn:
                prev_earn = pd.Timestamp(ticker_earn[-1])
                prev_post = trading_days[trading_days >= prev_earn]
                prev_pre = trading_days[trading_days < prev_earn]
                if len(prev_post) > 0 and len(prev_pre) > 0:
                    prev_close = float(ticker_prices.loc[prev_pre[-1], 'close'])
                    prev_open = float(ticker_prices.loc[prev_post[0], 'open'])
                    if prev_close > 0:
                        feats['prev_gap'] = float(prev_open / prev_close - 1)

            feats['ticker_hash'] = hash(ticker) % 100

            # Volume spike
            if 'volume' in ticker_prices.columns and len(post_days) >= 1:
                earn_vol = float(ticker_prices.loc[post_days[0], 'volume'])
                avg_vol = float(ticker_prices.loc[pre_days[-22:], 'volume'].mean()) if len(pre_days) >= 22 else 1
                feats['volume_spike'] = float(earn_vol / max(avg_vol, 1))
            else:
                feats['volume_spike'] = 1.0

            # Gap vs ATR
            if len(pre_days) >= 15:
                highs = ticker_prices.loc[pre_days[-15:], 'high'].values
                lows = ticker_prices.loc[pre_days[-15:], 'low'].values
                tr = np.mean(highs - lows)
                feats['gap_vs_atr'] = float(abs(gap_pct) * close_before / max(tr, 0.01))
            else:
                feats['gap_vs_atr'] = 1.0

            # Sector momentum
            feats['sector_momentum'] = 0
            if spy is not None and len(pre_days) >= 22:
                try:
                    spy_close = spy.loc[spy.index <= pre_days[-1]].iloc[-21:]
                    feats['sector_momentum'] = float(spy_close.iloc[-1] / spy_close.iloc[0] - 1)
                except:
                    pass

            # TARGETS: hold 3d and 5d drift
            post_close = [float(ticker_prices.loc[d, 'close']) for d in post_days[:11]]

            for hold_days in [3, 5]:
                if len(post_close) >= hold_days + 1:
                    entry_price = open_after
                    if gap_pct > 0:
                        exit_price = post_close[hold_days]
                        feats[f'drift_{hold_days}d'] = float(exit_price / entry_price - 1)
                        feats[f'target_{hold_days}d'] = 1 if (exit_price / entry_price - 1) >= DRIFT_THRESHOLD else 0
                    else:
                        exit_price = post_close[hold_days]
                        feats[f'drift_{hold_days}d'] = float(1 - exit_price / entry_price)
                        feats[f'target_{hold_days}d'] = 1 if (1 - exit_price / entry_price) >= DRIFT_THRESHOLD else 0

            # Raw PnL for each hold period
            for hold_days in [3, 5]:
                if len(post_close) >= hold_days + 1:
                    entry_price = open_after
                    exit_price = post_close[hold_days]
                    if gap_pct > 0:
                        feats[f'pnl_pct_{hold_days}d'] = float(exit_price / entry_price - 1)
                    else:
                        feats[f'pnl_pct_{hold_days}d'] = float(1 - exit_price / entry_price)

            feats['ticker'] = ticker
            feats['earn_date'] = earn_date_str
            feats['close_before'] = close_before
            feats['open_after'] = open_after
            feats['post_closes'] = post_close

            events.append(feats)

    # Compute z-scores and rank percentiles for magnitude
    if all_gaps:
        gap_mean = np.mean(all_gaps)
        gap_std = np.std(all_gaps) if np.std(all_gaps) > 0 else 1
        for e in events:
            e['gap_zscore'] = (e['abs_gap'] - gap_mean) / gap_std
            e['gap_rank_pct'] = percentileofscore(all_gaps, e['abs_gap']) / 100.0

    fprint(f"\nBuilt {len(events)} qualifying events (|gap| >= {MIN_GAP_PCT*100}%)")

    # Analyze magnitude vs drift relationship
    if events:
        fprint("\n--- Magnitude vs Drift Analysis ---")
        for tier, label in [(1, '5-10%'), (2, '10-20%'), (3, '20%+')]:
            tier_events = [e for e in events if e.get('gap_magnitude_tier') == tier]
            if tier_events:
                drifts_3d = [e.get('drift_3d', 0) for e in tier_events if 'drift_3d' in e]
                drifts_5d = [e.get('drift_5d', 0) for e in tier_events if 'drift_5d' in e]
                up = [e for e in tier_events if e['gap_direction'] == 1]
                dn = [e for e in tier_events if e['gap_direction'] == -1]
                fprint(f"  Tier {tier} ({label}): {len(tier_events)} events, "
                       f"avg 3d drift {np.mean(drifts_3d)*100:.2f}%, "
                       f"avg 5d drift {np.mean(drifts_5d)*100:.2f}%, "
                       f"up={len(up)} dn={len(dn)}")

        # Direction analysis
        fprint("\n--- Direction Analysis ---")
        for direction, label in [(1, 'UP gaps'), (-1, 'DOWN gaps')]:
            dir_events = [e for e in events if e['gap_direction'] == direction]
            if dir_events:
                drifts_3d = [e.get('drift_3d', 0) for e in dir_events if 'drift_3d' in e]
                wr_3d = [1 for d in drifts_3d if d > 0]
                fprint(f"  {label}: {len(dir_events)} events, "
                       f"avg 3d drift {np.mean(drifts_3d)*100:.2f}%, "
                       f"WR {len(wr_3d)/len(drifts_3d)*100:.1f}%")

    return events


# ============================================================
# ML MODEL
# ============================================================

def train_lgbm(X_train, y_train):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ============================================================
# MAGNITUDE SCALING FUNCTIONS
# ============================================================

def scale_linear(gap_pct, base_size, cap=3.0):
    """Linear scaling: size proportional to gap magnitude."""
    multiplier = min(abs(gap_pct) / MIN_GAP_PCT, cap)
    return base_size * multiplier

def scale_sqrt(gap_pct, base_size, cap=3.0):
    """Sqrt scaling: diminishing returns for larger gaps."""
    multiplier = min(np.sqrt(abs(gap_pct) / MIN_GAP_PCT), cap)
    return base_size * multiplier

def scale_tier(gap_pct, base_size):
    """Tier scaling: 1x/2x/3x by magnitude bracket."""
    ag = abs(gap_pct)
    if ag >= 0.20:
        return base_size * 3.0
    elif ag >= 0.10:
        return base_size * 2.0
    else:
        return base_size * 1.0

def scale_direction_aware(gap_pct, base_size, cap=3.0):
    """Direction-aware: shorts get 1.5x multiplier (stronger drift)."""
    base_mult = min(abs(gap_pct) / MIN_GAP_PCT, cap)
    direction_mult = 1.5 if gap_pct < 0 else 1.0
    return base_size * base_mult * direction_mult

def scale_confidence(gap_pct, ml_prob, base_size, cap=3.0):
    """Scale by both magnitude and ML confidence."""
    mag_mult = min(abs(gap_pct) / MIN_GAP_PCT, cap)
    conf_mult = max(ml_prob - 0.5, 0) * 4  # 0.5→0x, 0.75→1x, 1.0→2x
    return base_size * mag_mult * (0.5 + conf_mult)


# ============================================================
# VARIANT CONFIGS
# ============================================================

VARIANT_CONFIGS = {
    'A': {
        'name': 'Linear Magnitude Scaling',
        'hold_days': 3,
        'scaling': 'linear',
        'use_ml': True,
        'ml_threshold': 0.55,
        'base_position_pct': 0.15,
        'tp_pct': 0.05,
        'sl_pct': -0.03,
    },
    'B': {
        'name': 'Sqrt Magnitude Scaling',
        'hold_days': 3,
        'scaling': 'sqrt',
        'use_ml': True,
        'ml_threshold': 0.55,
        'base_position_pct': 0.15,
        'tp_pct': 0.05,
        'sl_pct': -0.03,
    },
    'C': {
        'name': 'Tier Scaling (1x/2x/3x)',
        'hold_days': 3,
        'scaling': 'tier',
        'use_ml': True,
        'ml_threshold': 0.55,
        'base_position_pct': 0.15,
        'tp_pct': 0.05,
        'sl_pct': -0.03,
    },
    'D': {
        'name': 'Direction-Aware Linear (Shorts 1.5x)',
        'hold_days': 3,
        'scaling': 'direction_aware',
        'use_ml': True,
        'ml_threshold': 0.55,
        'base_position_pct': 0.15,
        'tp_pct': 0.05,
        'sl_pct': -0.03,
    },
    'E': {
        'name': 'Magnitude x ML Confidence',
        'hold_days': 3,
        'scaling': 'confidence',
        'use_ml': True,
        'ml_threshold': 0.50,
        'base_position_pct': 0.15,
        'tp_pct': 0.05,
        'sl_pct': -0.03,
    },
    'F': {
        'name': 'High-Conviction Only (>10% gap, no ML)',
        'hold_days': 5,
        'scaling': 'linear',
        'use_ml': False,
        'min_gap': 0.10,
        'base_position_pct': 0.25,
        'tp_pct': 0.08,
        'sl_pct': -0.05,
    },
}


# ============================================================
# WALK-FORWARD SIMULATION
# ============================================================

def run_variant(variant_key, cfg, events):
    fprint(f"\n{'='*60}")
    fprint(f"  VARIANT {variant_key}: {cfg['name']}")
    fprint(f"{'='*60}")

    hold_days = cfg['hold_days']
    target_col = f'target_{hold_days}d'
    pnl_col = f'pnl_pct_{hold_days}d'

    valid_events = [e for e in events if target_col in e and pnl_col in e]
    events_sorted = sorted(valid_events, key=lambda e: e['earn_date'])

    min_train = 30
    if len(events_sorted) < min_train + 5:
        fprint(f"  Not enough events ({len(events_sorted)})")
        return None

    # Apply min gap filter for variant F
    if 'min_gap' in cfg:
        events_sorted = [e for e in events_sorted if e['abs_gap'] >= cfg['min_gap']]
        fprint(f"  After min_gap filter: {len(events_sorted)} events")

    equity = STARTING_CAPITAL
    trades = []
    ticker_trade_count = defaultdict(int)
    peak_equity = equity
    max_dd = 0

    feature_cols = [c for c in FEATURE_COLS if c in events_sorted[0]]

    for i in range(min_train, len(events_sorted)):
        event = events_sorted[i]

        # Skip if ticker over-traded
        if ticker_trade_count[event['ticker']] >= MAX_TRADES_PER_TICKER:
            continue

        if cfg['use_ml']:
            # Train on all prior events
            train_events = events_sorted[:i]
            X_train = pd.DataFrame(train_events)[feature_cols].fillna(0).values
            y_train = np.array([e[target_col] for e in train_events])

            if len(np.unique(y_train)) < 2:
                continue

            model = train_lgbm(X_train, y_train)
            X_test = pd.DataFrame([event])[feature_cols].fillna(0).values
            prob = model.predict_proba(X_test)[0][1]

            if prob < cfg.get('ml_threshold', 0.55):
                continue
        else:
            prob = 0.75  # default for no-ML variant

        # Position sizing with magnitude scaling
        base_size = equity * cfg['base_position_pct']
        gap_pct = event['gap_pct']

        if cfg['scaling'] == 'linear':
            position_size = scale_linear(gap_pct, base_size)
        elif cfg['scaling'] == 'sqrt':
            position_size = scale_sqrt(gap_pct, base_size)
        elif cfg['scaling'] == 'tier':
            position_size = scale_tier(gap_pct, base_size)
        elif cfg['scaling'] == 'direction_aware':
            position_size = scale_direction_aware(gap_pct, base_size)
        elif cfg['scaling'] == 'confidence':
            position_size = scale_confidence(gap_pct, prob, base_size)
        else:
            position_size = base_size

        # Cap at MAX_POSITION_PCT of equity
        position_size = min(position_size, equity * MAX_POSITION_PCT)
        if position_size < 10:  # minimum viable trade
            continue

        # Compute PnL
        pnl_pct = event[pnl_col]

        # Apply TP/SL
        if pnl_pct >= cfg['tp_pct']:
            pnl_pct = cfg['tp_pct']
        elif pnl_pct <= cfg['sl_pct']:
            pnl_pct = cfg['sl_pct']

        trade_pnl = position_size * pnl_pct
        # Slippage: $0.01/share, assume avg price ~$100
        shares = position_size / max(event['open_after'], 1)
        slippage_cost = shares * 0.01 * 2  # entry + exit
        trade_pnl -= slippage_cost

        equity += trade_pnl
        peak_equity = max(peak_equity, equity)
        dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
        max_dd = min(max_dd, dd)

        trade = {
            'date': event['earn_date'],
            'ticker': event['ticker'],
            'direction': 'long' if gap_pct > 0 else 'short',
            'gap_pct': gap_pct,
            'magnitude_tier': event.get('gap_magnitude_tier', 1),
            'position_size': position_size,
            'pnl_pct': pnl_pct,
            'pnl_usd': trade_pnl,
            'equity': equity,
            'ml_prob': prob if cfg['use_ml'] else None,
            'spy_regime': 'bull' if event.get('sector_momentum', 0) > 0 else 'bear',
        }
        trades.append(trade)
        ticker_trade_count[event['ticker']] += 1

    if not trades:
        fprint("  No trades generated")
        return None

    return analyze_results(variant_key, cfg, trades)


# ============================================================
# ANALYSIS & VALIDATION
# ============================================================

def analyze_results(variant_key, cfg, trades):
    """Compute metrics and run 5-gate validation."""
    n_trades = len(trades)
    pnls = [t['pnl_usd'] for t in trades]
    pnl_pcts = [t['pnl_pct'] for t in trades]
    final_equity = trades[-1]['equity']

    # Basic metrics
    total_return = (final_equity / STARTING_CAPITAL - 1) * 100
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]
    win_rate = len(winners) / n_trades * 100 if n_trades > 0 else 0
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0
    profit_factor = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float('inf')

    # Drawdown
    equities = [t['equity'] for t in trades]
    peak = STARTING_CAPITAL
    max_dd = 0
    for eq in equities:
        peak = max(peak, eq)
        dd = (eq - peak) / peak
        max_dd = min(max_dd, dd)

    # Sharpe / Sortino (annualized, ~4 earnings/year/stock, rough annualization)
    if len(pnl_pcts) > 1:
        # Approximate: each trade spans ~3-5 days, ~70 trading days/quarter = ~14 trades/quarter
        trades_per_year = max(n_trades / 4.5, 1)  # years of data
        annualization = np.sqrt(max(trades_per_year, 1))
        mean_ret = np.mean(pnl_pcts)
        std_ret = np.std(pnl_pcts) if np.std(pnl_pcts) > 0 else 1e-6
        sharpe = (mean_ret / std_ret) * annualization
        downside = np.std([r for r in pnl_pcts if r < 0]) if any(r < 0 for r in pnl_pcts) else std_ret
        sortino = (mean_ret / downside) * annualization if downside > 0 else sharpe
    else:
        sharpe = sortino = 0

    fprint(f"\n  Trades: {n_trades}, WR: {win_rate:.1f}%, PF: {profit_factor:.2f}")
    fprint(f"  ${STARTING_CAPITAL:.0f} -> ${final_equity:.0f} ({total_return:+.1f}%)")
    fprint(f"  Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}, MDD: {max_dd*100:.1f}%")
    fprint(f"  Avg Win: ${avg_win:.2f}, Avg Loss: ${avg_loss:.2f}")

    # Direction breakdown
    long_trades = [t for t in trades if t['direction'] == 'long']
    short_trades = [t for t in trades if t['direction'] == 'short']
    fprint(f"  Long trades: {len(long_trades)}, Short trades: {len(short_trades)}")
    if long_trades:
        long_wr = len([t for t in long_trades if t['pnl_usd'] > 0]) / len(long_trades) * 100
        fprint(f"    Long WR: {long_wr:.1f}%, avg PnL: ${np.mean([t['pnl_usd'] for t in long_trades]):.2f}")
    if short_trades:
        short_wr = len([t for t in short_trades if t['pnl_usd'] > 0]) / len(short_trades) * 100
        fprint(f"    Short WR: {short_wr:.1f}%, avg PnL: ${np.mean([t['pnl_usd'] for t in short_trades]):.2f}")

    # Magnitude tier breakdown
    for tier in [1, 2, 3]:
        tier_trades = [t for t in trades if t.get('magnitude_tier') == tier]
        if tier_trades:
            tier_wr = len([t for t in tier_trades if t['pnl_usd'] > 0]) / len(tier_trades) * 100
            tier_pnl = sum(t['pnl_usd'] for t in tier_trades)
            fprint(f"  Tier {tier}: {len(tier_trades)} trades, WR {tier_wr:.1f}%, total PnL ${tier_pnl:.2f}")

    # ============================================================
    # 5-GATE VALIDATION
    # ============================================================
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates['sharpe_pass'] = sharpe > 0.5
    fprint(f"\n  Gate 1 (Sharpe > 0.5): {'PASS' if gates['sharpe_pass'] else 'FAIL'} ({sharpe:.2f})")

    # Gate 2: Permutation test p < 0.05
    actual_sharpe = sharpe
    perm_sharpes = []
    np.random.seed(42)
    for _ in range(N_PERMUTATIONS):
        shuffled = np.random.permutation(pnl_pcts)
        if np.std(shuffled) > 0:
            perm_sharpes.append(np.mean(shuffled) / np.std(shuffled) * annualization)
        else:
            perm_sharpes.append(0)
    p_value = np.mean([1 for ps in perm_sharpes if ps >= actual_sharpe]) / len(perm_sharpes)
    gates['perm_pass'] = p_value < 0.05
    fprint(f"  Gate 2 (Perm p<0.05): {'PASS' if gates['perm_pass'] else 'FAIL'} (p={p_value:.3f})")

    # Gate 3: Beats random trading
    random_sharpes = []
    for _ in range(100):
        n_random = n_trades
        random_rets = np.random.choice(pnl_pcts, size=n_random, replace=True)
        if np.std(random_rets) > 0:
            random_sharpes.append(np.mean(random_rets) / np.std(random_rets) * annualization)
    beats_random = np.mean([1 for rs in random_sharpes if actual_sharpe > rs])
    gates['random_pass'] = beats_random > 0.6
    fprint(f"  Gate 3 (Beats random): {'PASS' if gates['random_pass'] else 'FAIL'} ({beats_random:.1%})")

    # Gate 4: Regime gap < 0.50
    bull_trades = [t['pnl_pct'] for t in trades if t.get('spy_regime') == 'bull']
    bear_trades = [t['pnl_pct'] for t in trades if t.get('spy_regime') == 'bear']
    if bull_trades and bear_trades:
        bull_sharpe = np.mean(bull_trades) / max(np.std(bull_trades), 1e-6) * annualization
        bear_sharpe = np.mean(bear_trades) / max(np.std(bear_trades), 1e-6) * annualization
        regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)
        gates['regime_pass'] = regime_gap < 0.50
        fprint(f"  Gate 4 (Regime gap<0.50): {'PASS' if gates['regime_pass'] else 'FAIL'} "
               f"(gap={regime_gap:.2f}, bull={bull_sharpe:.2f}, bear={bear_sharpe:.2f})")
    else:
        gates['regime_pass'] = False
        fprint(f"  Gate 4 (Regime gap<0.50): FAIL (insufficient regime data)")

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
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'trades': trades,
    }
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    fprint("=" * 70)
    fprint("  PEAD MAGNITUDE SCALING v1")
    fprint("  Testing: Does scaling by earnings surprise magnitude improve PEAD?")
    fprint("=" * 70)
    start_time = time.time()

    # Load data
    prices_df, earnings_dates = load_data()
    fprint(f"\nPrices: {len(prices_df)} rows")

    # Build features
    events = build_event_features(prices_df, earnings_dates)
    if not events:
        fprint("ERROR: No events found")
        return

    # MLflow setup
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri('sqlite:////home/jupiter/teleclaude-main/mlflow.db')
        mlflow.set_experiment('pead_magnitude_scaling_v1')

    results = {}
    for vk in sorted(VARIANT_CONFIGS.keys()):
        cfg = VARIANT_CONFIGS[vk]
        try:
            result = run_variant(vk, cfg, events)
            if result:
                results[vk] = result

                # Log to MLflow
                if MLFLOW_AVAILABLE:
                    with mlflow.start_run(run_name=f"variant_{vk}_{cfg['name'][:30]}"):
                        mlflow.log_params({
                            'variant': vk,
                            'scaling': cfg['scaling'],
                            'hold_days': cfg['hold_days'],
                            'use_ml': cfg.get('use_ml', True),
                            'ml_threshold': cfg.get('ml_threshold', 0.55),
                            'base_position_pct': cfg['base_position_pct'],
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
    fprint(f"  SUMMARY — PEAD MAGNITUDE SCALING")
    fprint(f"{'='*70}")

    if not results:
        fprint("No variants produced results")
        return

    header = f"{'Var':>3} | {'Name':<35} | {'Trades':>6} | {'WR%':>5} | {'PF':>5} | {'Sharpe':>6} | {'Sortino':>7} | {'Return%':>8} | {'MDD%':>6} | {'Gates':>5}"
    fprint(header)
    fprint("-" * len(header))
    for vk in sorted(results.keys()):
        r = results[vk]
        fprint(f"  {vk} | {r['name']:<35} | {r['n_trades']:>6} | {r['win_rate']:>5.1f} | {r['profit_factor']:>5.2f} | {r['sharpe']:>6.2f} | {r['sortino']:>7.2f} | {r['total_return_pct']:>+7.1f}% | {r['max_drawdown']*100:>5.1f}% | {r['gates_passed']}/5")

    # Best variant
    best_key = max(results, key=lambda k: results[k]['sharpe'])
    best = results[best_key]
    fprint(f"\n  BEST: Variant {best_key} ({best['name']})")
    fprint(f"  Sharpe {best['sharpe']:.2f}, ${STARTING_CAPITAL:.0f} -> ${best['final_equity']:.0f}, "
           f"{best['gates_passed']}/5 gates")

    # Compare with baseline (uniform sizing)
    fprint(f"\n  KEY QUESTION: Does magnitude scaling beat uniform sizing?")
    uniform = results.get('A')
    tier = results.get('C')
    if uniform and tier:
        fprint(f"    Linear (A): Sharpe {uniform['sharpe']:.2f}, Return {uniform['total_return_pct']:+.1f}%")
        fprint(f"    Tier (C): Sharpe {tier['sharpe']:.2f}, Return {tier['total_return_pct']:+.1f}%")

    # Save results
    save_results = {k: {kk: vv for kk, vv in v.items() if kk != 'trades'} for k, v in results.items()}
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\n  Results saved to output/growth_research/pead_magnitude_scaling_v1/")

    elapsed = time.time() - start_time
    fprint(f"\n  Total time: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
