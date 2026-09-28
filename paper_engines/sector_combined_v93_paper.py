#!/usr/bin/env python3
"""
Sector Combined V9.3 Paper Trading Engine
===========================================

V9.3 UPGRADE from V9.2 — key change: 50% PROFIT TARGET EXIT

This is our single biggest finding: adding a 50% profit target nearly doubles
Sharpe (2.36 -> 5.12) and is confirmed robust across all thresholds 30-70%.

V9.3 vs V9.2 changes:
  - 50% profit target: every weekday at 4:30 PM, check each open spread.
    If unrealized gain >= 50% of max profit, CLOSE IT early.
  - Close cost: $2.60 commission for the early close
  - Daily cron (weekdays) instead of monthly-only — checks profit targets daily
  - Still monthly rebalance for opening NEW positions (first Friday of month)
  - Everything else same: DTE=28, 3% OTM, adaptive max($3,3%), LGBM 17 features

Key parameters:
  - 11 sectors, 17 LGBM momentum features
  - Monthly rebalance (first Friday of month) for new positions
  - Daily profit target check at 4:30 PM (50% of max profit)
  - DTE=28 (monthly options), hold to expiry if not hitting target
  - 3% OTM moneyness
  - $645 starting capital, $200 max per trade
  - Commission: $2.60 per spread RT, 15% entry haircut
  - TLT risk parity overlay in low-VIX mode

Backtest reference (V9.3): Sharpe 5.12, robust across 30-70% thresholds

Usage:
    python sector_combined_v93_paper.py              # normal daily run
    python sector_combined_v93_paper.py --dry-run    # simulate without state changes
"""
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Standardized tools ──
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    exit_spread_value,
    bs_call_price,
    bs_put_price,
    estimate_iv,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: LightGBM not available. Cannot run LGBM ranking.")

LOG_DIR = Path(__file__).resolve().parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)
STATE_DIR = BASE / 'state'
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / 'sector_combined_v93_paper_state.json'
TRADE_LOG = LOG_DIR / 'sector_combined_v93_trades.jsonl'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'sector_combined_v93_paper.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ==================== DRY-RUN MODE ====================
DRY_RUN = '--dry-run' in sys.argv

# ==================== STRATEGY CONFIG (V9.3 Profit Target) ====================
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
EXTRA_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', 'SHY', 'HYG', 'GLD']
INITIAL_CAPITAL = 645.0
REBALANCE_INTERVAL_DAYS = 28
SPREAD_PCT = 3.0
MIN_SPREAD_WIDTH = 3.0
DTE = 28
MONEYNESS_PCT = 3.0
MAX_POS_SIZE = 200
MAX_POS_PCT = 0.40
HAIRCUT = 0.15
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # $2.60 round trip

# V9.3: Profit target — close when unrealized gain >= this fraction of max profit
PROFIT_TARGET_PCT = 0.50  # 50% of max profit

# VIX threshold — switches between modes
VIX_THRESHOLD = 20.0

# High-VIX mode (bull spreads only): top-2 sectors, regime gate
HIGH_VIX_TOP_K = 2
REGIME_BULL_THRESHOLD = 0.4

# Low-VIX mode (pair trades): top-3 long + bottom-3 short
LOW_VIX_TOP_K = 3
LOW_VIX_BOTTOM_K = 3

# Risk parity overlay allocation (low-VIX mode only)
RISK_PARITY_ALLOC = {'SPY': 0.333, 'TLT': 0.333, 'GLD': 0.334}

# Regime predictions file
REGIME_FILE = BASE / 'output' / 'regime_detector_v1' / 'regime_predictions_v1.npz'

# ==================== FEATURES (17 momentum features) ====================
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
]


# ==================== REGIME MODEL ====================

def load_regime_predictions():
    """Load GRU regime predictions from pre-computed NPZ file."""
    if not REGIME_FILE.exists():
        log.warning("Regime file not found -- will use VIX-based proxy")
        return None

    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data['dates'])
    scores = data['regime_scores']
    regime_series = pd.Series(scores, index=dates, name='regime_score')
    regime_series = regime_series[~regime_series.index.duplicated(keep='last')]
    log.info(f"Regime predictions loaded: {len(regime_series)} days "
             f"({regime_series.index[0].date()} to {regime_series.index[-1].date()})")
    return regime_series


def get_regime_score(regime_series, dt):
    """Get regime score at a given date, with nearest-date fallback."""
    if regime_series is None:
        return None
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method='ffill')]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return None


def vix_regime_proxy(vix_val):
    """Fallback: map VIX to a pseudo regime_score."""
    if vix_val >= 25:
        return 0.7
    elif vix_val >= 20:
        return 0.4 + (vix_val - 20) * 0.06
    else:
        return 0.15 + vix_val * 0.0125


# ==================== MONTHLY REBALANCE LOGIC ====================

def is_first_friday(dt):
    """Check if a date is the first Friday of its month."""
    return dt.weekday() == 4 and dt.day <= 7


def should_rebalance(today, state):
    """V9.2 monthly rebalance logic (unchanged in V9.3)."""
    is_friday = today.weekday() == 4
    if not is_friday:
        return False, "not Friday"

    has_open = len(state['open_positions']) > 0
    last_rebal = state.get('last_rebalance')

    if is_first_friday(today):
        return True, "first Friday of month"

    if not has_open and last_rebal:
        last_dt = pd.Timestamp(last_rebal)
        days_since = (today - last_dt).days
        if days_since >= REBALANCE_INTERVAL_DAYS:
            return True, f"no positions + {days_since}d since last rebalance"

    if not has_open and last_rebal is None:
        return True, "initial rebalance (no prior history)"

    return False, "not first Friday, positions open or recent rebalance"


# ==================== STATE MANAGEMENT ====================

def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        defaults = _default_state()
        for k, v in defaults.items():
            if k not in state:
                state[k] = v
        return state
    return _default_state()


def _default_state():
    return {
        'config_version': 'v93_profit_target_50pct',
        'equity': INITIAL_CAPITAL,
        'open_positions': [],
        'risk_parity_positions': [],
        'closed_trades': [],
        'last_rebalance': None,
        'total_trades': 0,
        'total_pnl': 0,
        'wins': 0,
        'losses': 0,
        'long_wins': 0,
        'long_losses': 0,
        'short_wins': 0,
        'short_losses': 0,
        'early_exits': 0,
        'current_mode': None,
        'created': datetime.now().isoformat(),
    }


def save_state(state):
    if DRY_RUN:
        log.info("[DRY-RUN] State NOT saved (dry-run mode)")
        return
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(trade_record):
    if DRY_RUN:
        log.info(f"[DRY-RUN] Trade NOT logged: {trade_record.get('action')} "
                 f"{trade_record.get('ticker', 'N/A')} ({trade_record.get('mode', 'N/A')})")
        return
    with open(TRADE_LOG, 'a') as f:
        f.write(json.dumps(trade_record, default=str) + '\n')


# ==================== DATA DOWNLOAD ====================

def download_data():
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    raw = yf.download(all_tickers, start='2024-01-01', progress=False)

    if raw is None or raw.empty:
        raise ValueError("yfinance returned empty data — possible rate limit or network error")

    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)

    close = close.ffill()
    high = high.ffill()
    low = low.ffill()

    rename_map = {'^VIX': 'VIX', '^VIX3M': 'VIX3M'}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)

    vc = 'VIX' if 'VIX' in close.columns else ('^VIX' if '^VIX' in close.columns else None)
    if vc is None:
        raise ValueError("VIX data not available")
    vix = close[vc].dropna()
    if vix.empty:
        raise ValueError("VIX series is empty after dropna")
    spy = close['SPY'].dropna()
    if spy.empty:
        raise ValueError("SPY series is empty after dropna")
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    if sc.empty:
        raise ValueError("No sector data available after dropna")
    ix = sc.index.intersection(vix.index).intersection(spy.index)
    if len(ix) == 0:
        raise ValueError("No overlapping dates between sectors, VIX, and SPY")
    return close.loc[ix], sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]


# ==================== FEATURE ENGINEERING (17 momentum features) ====================

def compute_features(px):
    """Compute the 17 momentum features for a single sector ETF."""
    from scipy import stats
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, 'ret_5d'), (10, 'ret_10d'), (21, 'ret_21d'),
                   (63, 'ret_63d'), (126, 'ret_126d'), (252, 'ret_252d')]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:] / pk63) - 1).min())
    f['pct_52w_high'] = float(px.iloc[-1] / px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d'] / 3

    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)

    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val ** 2
        f['trend_slope_63d'] = slope * 252
    else:
        f['trend_r2_63d'] = 0.0
        f['trend_slope_63d'] = 0.0

    return f


# ==================== LGBM RANKING (17 features) ====================

def run_lgbm_ranking(sc, close_df):
    """Run walk-forward LGBM ranking using trailing data with 17 momentum features."""
    if not HAS_LGBM:
        log.warning("No LightGBM -- using simple momentum ranking")
        rets_21d = sc.pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    records = []
    all_dates = sc.index[-400:]
    rebal_dates = all_dates[::20]

    for dt in rebal_dates[:-1]:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx + 1].dropna()
            feats = compute_features(px)
            if not feats:
                continue
            fi = min(idx + 28, len(sc) - 1)
            feats.update({
                'date': dt, 'ticker': tk,
                'fwd_ret': float(sc[tk].iloc[fi] / sc[tk].iloc[idx] - 1)
            })
            records.append(feats)

    df = pd.DataFrame(records)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)

    if len(df) < 50:
        log.warning(f"Not enough training data ({len(df)} rows). Falling back to momentum.")
        rets_21d = sc.pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)

    X_train = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
    y_train = df['rank_label'].values.astype(np.float32)

    m = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1
    )
    m.fit(X_train, y_train)

    current_feats = {}
    for tk in sc.columns:
        px = sc[tk].dropna()
        feats = compute_features(px)
        if feats:
            current_feats[tk] = feats

    if not current_feats:
        return {}

    pred_df = pd.DataFrame(current_feats).T
    for c in FEAT_COLS:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    X_pred = np.nan_to_num(pred_df[FEAT_COLS].values.astype(np.float32))
    scores = m.predict(X_pred)

    return dict(zip(pred_df.index, scores))


# ==================== CONFLUENCE CHECK (high-VIX mode only) ====================

def confluence_check(tk, sc, spy, vix):
    """Multi-signal confluence for bull call spreads. Min 2 independent signals required."""
    signals = []
    px = sc[tk].dropna()
    if len(px) < 63:
        return False, 0, []

    ret_21d = float(px.iloc[-1] / px.iloc[-21] - 1)
    if ret_21d > 0:
        signals.append('mom_21d_pos')

    sma50 = px.rolling(50).mean()
    if not pd.isna(sma50.iloc[-1]) and px.iloc[-1] > sma50.iloc[-1]:
        signals.append('above_sma50')

    if len(spy) > 21:
        common = px.index.intersection(spy.index)
        if len(common) > 21:
            rel = px.loc[common] / spy.loc[common]
            rel_ret = float(rel.iloc[-1] / rel.iloc[-21] - 1)
            if rel_ret > 0:
                signals.append('rel_str_pos')

    rets = px.pct_change().dropna()
    if len(rets) > 14:
        gains = rets.clip(lower=0).rolling(14).mean()
        losses = (-rets.clip(upper=0)).rolling(14).mean()
        rs = gains / (losses + 1e-10)
        rsi = 100 - 100 / (1 + rs)
        r = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        if r < 80:
            signals.append('rsi_ok')

    if len(px) > 63:
        ret_63d = float(px.iloc[-1] / px.iloc[-63] - 1)
        accel = ret_21d - ret_63d / 3
        if accel > 0:
            signals.append('mom_accel_pos')

    return len(signals) >= 2, len(signals), signals


# ==================== SPREAD PRICING ====================

def price_bull_spread(S, spread_pct, dte, sh_tk, sl_tk, sc_tk, vix_val):
    """Price a bull call spread with 3% OTM moneyness and V9 adaptive width."""
    K1 = round(S * (1 + MONEYNESS_PCT / 100))
    pct_width = K1 * spread_pct / 100
    adaptive_width = max(MIN_SPREAD_WIDTH, pct_width)
    K2 = round(K1 + adaptive_width)

    if sh_tk is not None and len(sh_tk) >= 14:
        atr = compute_atr(sh_tk, sl_tk, sc_tk, period=14)
    else:
        atr = S * 0.015

    entry_cost_ps, max_profit_ps = price_bull_call_spread(
        S=S, K1=K1, K2=K2, dte=dte, atr=atr, vix=vix_val, haircut=HAIRCUT
    )

    cost_dollars = entry_cost_ps * 100 + SPREAD_COMM
    max_profit_dollars = max_profit_ps * 100 - SPREAD_COMM

    return cost_dollars, max_profit_dollars, K1, K2, entry_cost_ps


def price_bear_spread(S, spread_pct, dte, sh_tk, sl_tk, sc_tk, vix_val):
    """Price a bear put spread with 3% OTM moneyness and V9 adaptive width."""
    K1 = round(S * (1 - MONEYNESS_PCT / 100))
    pct_width = K1 * spread_pct / 100
    adaptive_width = max(MIN_SPREAD_WIDTH, pct_width)
    K2 = round(K1 - adaptive_width)

    if K2 >= K1:
        K2 = K1 - 1

    if sh_tk is not None and len(sh_tk) >= 14:
        atr = compute_atr(sh_tk, sl_tk, sc_tk, period=14)
    else:
        atr = S * 0.015

    entry_cost_ps, max_profit_ps = price_bear_put_spread(
        S=S, K1=K2, K2=K1, dte=dte, atr=atr, vix=vix_val, haircut=HAIRCUT
    )

    cost_dollars = entry_cost_ps * 100 + SPREAD_COMM
    max_profit_dollars = max_profit_ps * 100 - SPREAD_COMM

    return cost_dollars, max_profit_dollars, K2, K1, entry_cost_ps


# ==================== V9.3: PROFIT TARGET EXIT LOGIC ====================

def estimate_current_spread_value(pos, current_price, current_vix, days_held):
    """Estimate current value of an open spread position using Black-Scholes.

    For bull call spread: value = BS_call(K1) - BS_call(K2)
    For bear put spread: value = BS_put(K2) - BS_put(K1)  (K2 > K1)

    Uses current VIX as vol estimate, remaining DTE for time decay.
    Returns value per share (before haircut/commission).
    """
    remaining_dte = max(DTE - days_held, 0)
    T = remaining_dte / 365.0

    # Use VIX/100 as annualized vol proxy, floor at 10%
    sigma = max(current_vix / 100.0, 0.10)

    K1, K2 = pos['K1'], pos['K2']
    S = current_price

    if pos['mode'] == 'bull':
        # Bull call spread: long call at K1, short call at K2
        value_ps = bs_call_price(S, K1, T, sigma=sigma) - bs_call_price(S, K2, T, sigma=sigma)
    else:
        # Bear put spread: long put at K2 (higher), short put at K1 (lower)
        value_ps = bs_put_price(S, K2, T, sigma=sigma) - bs_put_price(S, K1, T, sigma=sigma)

    return max(value_ps, 0.0)


def check_profit_target(pos, current_price, current_vix, days_held):
    """V9.3: Check if a position has reached 50% profit target.

    Current value estimated via BS. If:
      current_value_dollars >= cost_paid + 50% * max_profit_dollars
    then close for early exit.

    Returns (should_exit, pnl_dollars, reason)
    """
    entry_cost_ps = pos.get('entry_cost_ps', (pos['cost'] - SPREAD_COMM) / 100.0)
    max_profit_dollars = pos.get('max_profit', 0)

    # Don't check on day 0 (just entered)
    if days_held < 1:
        return False, 0, 'hold'

    # Estimate current spread value using BS
    current_value_ps = estimate_current_spread_value(pos, current_price, current_vix, days_held)
    current_value_dollars = current_value_ps * 100

    # Entry premium (without entry commission)
    cost_paid_dollars = pos['cost']  # includes entry commission
    entry_cost_no_comm = cost_paid_dollars - SPREAD_COMM

    # Unrealized gain relative to entry premium
    unrealized_gain = current_value_dollars - entry_cost_no_comm

    # Target: gain >= 50% of max_profit
    target_gain = PROFIT_TARGET_PCT * max_profit_dollars

    if target_gain <= 0:
        return False, 0, 'hold'

    pct_of_max = unrealized_gain / max_profit_dollars if max_profit_dollars > 0 else 0

    if unrealized_gain >= target_gain:
        # Close: credit = current spread value - $2.60 close commission
        exit_credit = current_value_dollars - SPREAD_COMM
        # PnL = exit_credit - cost_paid (cost_paid includes entry commission)
        pnl = exit_credit - cost_paid_dollars
        log.info(f"  PROFIT TARGET HIT {pos['ticker']} {pos['mode']}: "
                 f"gain ${unrealized_gain:.2f} = {pct_of_max:.0%} of max ${max_profit_dollars:.2f} "
                 f"(target {PROFIT_TARGET_PCT:.0%}), PnL ${pnl:.2f}")
        return True, pnl, 'early_exit'

    return False, 0, 'hold'


# ==================== POSITION EXIT (expiry + profit target) ====================

def check_bull_exit(pos, current_price, days_held, current_vix):
    """Bull call spread exit: profit target OR expiry."""
    # V9.3: Check profit target first
    should_exit, pnl, reason = check_profit_target(pos, current_price, current_vix, days_held)
    if should_exit:
        return True, pnl, reason

    # Original expiry check
    if days_held < DTE:
        return False, 0, 'hold'

    S = current_price
    K1, K2 = pos['K1'], pos['K2']
    entry_cost_ps = pos.get('entry_cost_ps', (pos['cost'] - SPREAD_COMM) / 100.0)

    intrinsic = max(S - K1, 0.0) - max(S - K2, 0.0)
    pnl = (intrinsic - entry_cost_ps) * 100 - SPREAD_COMM
    return True, pnl, 'expiry'


def check_bear_exit(pos, current_price, days_held, current_vix):
    """Bear put spread exit: profit target OR expiry."""
    # V9.3: Check profit target first
    should_exit, pnl, reason = check_profit_target(pos, current_price, current_vix, days_held)
    if should_exit:
        return True, pnl, reason

    # Original expiry check
    if days_held < DTE:
        return False, 0, 'hold'

    S = current_price
    K1, K2 = pos['K1'], pos['K2']
    entry_cost_ps = pos.get('entry_cost_ps', (pos['cost'] - SPREAD_COMM) / 100.0)

    intrinsic = max(K2 - S, 0.0) - max(K1 - S, 0.0)
    pnl = (intrinsic - entry_cost_ps) * 100 - SPREAD_COMM
    return True, pnl, 'expiry'


# ==================== RISK PARITY OVERLAY (low-VIX mode) ====================

def enter_risk_parity(state, close_df, today):
    """When VIX < 20, allocate spare capital to SPY/TLT/GLD risk parity."""
    available = state['equity'] - sum(p['cost'] for p in state['open_positions'])
    if available < 50:
        log.info("Insufficient capital for risk parity overlay")
        return state

    if state['risk_parity_positions']:
        log.info("Risk parity positions already open -- holding")
        return state

    for ticker, weight in RISK_PARITY_ALLOC.items():
        if ticker not in close_df.columns:
            log.warning(f"  Risk parity ticker {ticker} not in data -- skipping")
            continue
        tk_data = close_df[ticker].dropna()
        if tk_data.empty:
            log.warning(f"  Risk parity ticker {ticker} has no price data -- skipping")
            continue
        price = float(tk_data.iloc[-1])
        alloc = available * weight
        shares = int(alloc / price)
        if shares <= 0:
            continue

        position = {
            'ticker': ticker,
            'type': 'risk_parity',
            'entry_date': str(today.date()),
            'entry_price': price,
            'shares': shares,
            'cost': round(shares * price, 2),
        }
        state['risk_parity_positions'].append(position)
        log.info(f"  RISK PARITY: {shares} shares {ticker} @ ${price:.2f} "
                 f"(${shares * price:.2f}, {weight*100:.0f}% alloc)")

        log_trade({
            'action': 'OPEN_RP',
            'date': str(today.date()),
            'ticker': ticker,
            'type': 'risk_parity',
            'shares': shares,
            'entry_price': price,
            'cost': round(shares * price, 2),
        })

    return state


def exit_risk_parity(state, close_df, today, reason='mode_switch'):
    """Exit risk parity positions."""
    if not state['risk_parity_positions']:
        return state

    for pos in state['risk_parity_positions']:
        tk = pos['ticker']
        if tk not in close_df.columns:
            continue
        tk_data = close_df[tk].dropna()
        if tk_data.empty:
            log.warning(f"  Risk parity ticker {tk} has no price data -- skipping exit")
            continue
        current_price = float(tk_data.iloc[-1])
        pnl = (current_price - pos['entry_price']) * pos['shares']
        state['equity'] += pnl
        state['total_pnl'] += pnl

        log.info(f"  EXIT RISK PARITY: {pos['shares']} shares {tk} @ ${current_price:.2f} "
                 f"(PnL: ${pnl:.2f}, reason: {reason})")

        log_trade({
            'action': 'CLOSE_RP',
            'date': str(today.date()),
            'ticker': tk,
            'type': 'risk_parity',
            'shares': pos['shares'],
            'entry_price': pos['entry_price'],
            'exit_price': current_price,
            'pnl': round(pnl, 2),
            'exit_reason': reason,
            'equity_after': round(state['equity'], 2),
        })

    state['risk_parity_positions'] = []
    return state


# ==================== POSITION SIZING ====================

def compute_position_size(state, max_concurrent):
    """Position size scales with equity growth."""
    equity_ratio = state['equity'] / INITIAL_CAPITAL
    scaled_max = MAX_POS_SIZE * equity_ratio
    equity_cap = state['equity'] * MAX_POS_PCT
    return min(scaled_max, equity_cap, state['equity'] / max(max_concurrent, 2))


# ==================== MAIN DAILY RUN ====================

def run_daily():
    """Main daily run. Called every weekday at 4:30 PM.
    V9.3: checks profit targets daily, opens new positions monthly."""
    state = load_state()
    if DRY_RUN:
        log.info("=" * 60)
        log.info("  DRY-RUN MODE -- no state changes will be persisted")
        log.info("=" * 60)

    log.info("=== Sector Combined V9.3 Paper Engine (50% Profit Target) ===")
    log.info(f"Equity: ${state['equity']:.2f} | Open spreads: {len(state['open_positions'])} | "
             f"Open RP: {len(state['risk_parity_positions'])} | "
             f"Trades: {state['total_trades']} | W/L: {state['wins']}/{state['losses']} | "
             f"Early exits: {state.get('early_exits', 0)}")

    # Load regime predictions
    regime_series = load_regime_predictions()

    # Download latest data
    try:
        close_df, sc, sh, sl, spy, vix = download_data()
    except Exception as e:
        log.error(f"Data download failed: {e}")
        return

    today = sc.index[-1]
    current_vix = float(vix.iloc[-1])
    high_vix_mode = current_vix >= VIX_THRESHOLD

    # Determine mode
    if high_vix_mode:
        rscore = get_regime_score(regime_series, today)
        if rscore is None:
            rscore = vix_regime_proxy(current_vix)
            log.info(f"Date: {today.date()} | VIX: {current_vix:.1f} | "
                     f"MODE: HIGH-VIX (bull spreads) | Regime: {rscore:.3f} (VIX proxy)")
        else:
            log.info(f"Date: {today.date()} | VIX: {current_vix:.1f} | "
                     f"MODE: HIGH-VIX (bull spreads) | Regime: {rscore:.3f} (GRU)")
        regime_active = rscore > REGIME_BULL_THRESHOLD
    else:
        rscore = None
        regime_active = True
        log.info(f"Date: {today.date()} | VIX: {current_vix:.1f} | "
                 f"MODE: LOW-VIX (pair trades + risk parity)")

    # ==================== CHECK EXISTING SPREAD POSITIONS ====================
    # V9.3: check profit targets + expiry on EVERY run (daily)
    positions_to_close = []
    for i, pos in enumerate(state['open_positions']):
        entry_date = pd.Timestamp(pos['entry_date'])
        days_held = len(sc.index[(sc.index > entry_date) & (sc.index <= today)])

        if pos['ticker'] not in sc.columns:
            continue

        current_price = float(sc[pos['ticker']].iloc[-1])

        if pos['mode'] == 'bull':
            should_exit, pnl, reason = check_bull_exit(pos, current_price, days_held, current_vix)
        else:
            should_exit, pnl, reason = check_bear_exit(pos, current_price, days_held, current_vix)

        if should_exit:
            positions_to_close.append((i, pnl, reason, days_held))
            log.info(f"  EXIT {pos['ticker']} {pos['mode']} spread: "
                     f"PnL ${pnl:.2f} ({reason}, held {days_held}d)")

    # Close positions (reverse order to preserve indices)
    for i, pnl, reason, days_held in reversed(positions_to_close):
        pos = state['open_positions'].pop(i)
        state['equity'] += pnl
        state['total_pnl'] += pnl

        is_win = pnl > 0
        if is_win:
            state['wins'] += 1
        else:
            state['losses'] += 1

        if pos['mode'] == 'bull':
            if is_win:
                state['long_wins'] += 1
            else:
                state['long_losses'] += 1
        else:
            if is_win:
                state['short_wins'] += 1
            else:
                state['short_losses'] += 1

        # V9.3: track early exits
        if reason == 'early_exit':
            state['early_exits'] = state.get('early_exits', 0) + 1

        trade_record = {
            'action': 'CLOSE',
            'date': str(today.date()),
            'ticker': pos['ticker'],
            'mode': pos['mode'],
            'entry_date': pos['entry_date'],
            'days_held': days_held,
            'cost': pos['cost'],
            'pnl': round(pnl, 2),
            'exit_reason': reason,
            'equity_after': round(state['equity'], 2),
            'vix_mode': 'high_vix' if high_vix_mode else 'low_vix',
        }
        state['closed_trades'].append(trade_record)
        log_trade(trade_record)

    # ==================== HANDLE MODE TRANSITIONS ====================
    prev_mode = state.get('current_mode')
    new_mode = 'high_vix' if high_vix_mode else 'low_vix'

    if prev_mode and prev_mode != new_mode:
        log.info(f"MODE SWITCH: {prev_mode} -> {new_mode}")
        if new_mode == 'high_vix' and state['risk_parity_positions']:
            state = exit_risk_parity(state, close_df, today, reason='switch_to_high_vix')

    state['current_mode'] = new_mode

    # ==================== CHECK FOR MONTHLY REBALANCE ====================
    do_rebalance, rebal_reason = should_rebalance(today, state)

    if not do_rebalance:
        log.info(f"Not rebalance day ({rebal_reason}). Profit target check complete.")
        _print_summary(state)
        save_state(state)
        # Write execution signals for HOLD positions + any CLOSEs from today
        _write_execution_signals(state, today, closed_today=positions_to_close)
        return

    log.info(f"=== REBALANCE DAY ({rebal_reason}) ===")
    state['last_rebalance'] = str(today.date())

    # ==================== HIGH-VIX MODE: Bull spreads on top-2 ====================
    if high_vix_mode:
        if not regime_active:
            log.info(f"Regime score {rscore:.3f} < {REGIME_BULL_THRESHOLD} -- "
                     f"HIGH-VIX but regime inactive. No new spreads.")
            _print_summary(state)
            save_state(state)
            return

        rankings = run_lgbm_ranking(sc, close_df)
        if not rankings:
            log.warning("No rankings available. Skipping rebalance.")
            save_state(state)
            return

        ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:HIGH_VIX_TOP_K]]
        log.info(f"HIGH-VIX bull picks (top {HIGH_VIX_TOP_K}): {picks}")
        log.info(f"Rankings: {', '.join(f'{t}={s:.3f}' for t, s in ranked)}")

        max_pos = compute_position_size(state, HIGH_VIX_TOP_K)
        if max_pos < 30:
            log.warning(f"Position size too small (${max_pos:.0f}). Skipping entries.")
            _print_summary(state)
            save_state(state)
            return

        entries = 0
        for tk in picks:
            if entries >= HIGH_VIX_TOP_K:
                break
            if tk not in sc.columns:
                continue
            if any(p['ticker'] == tk and p['mode'] == 'bull' for p in state['open_positions']):
                log.info(f"  SKIP {tk}: already holding bull spread")
                continue

            passes, n_signals, signal_names = confluence_check(tk, sc, spy, vix)
            if not passes:
                log.info(f"  SKIP {tk}: confluence FAIL ({n_signals} signals: {signal_names})")
                continue
            log.info(f"  PASS {tk}: confluence OK ({n_signals} signals: {signal_names})")

            S = float(sc[tk].iloc[-1])
            sh_tk = sh[tk] if tk in sh.columns else None
            sl_tk = sl[tk] if tk in sl.columns else None
            sc_tk = sc[tk] if tk in sc.columns else None
            cost, max_profit, K1, K2, entry_cost_ps = price_bull_spread(
                S, SPREAD_PCT, DTE, sh_tk, sl_tk, sc_tk, current_vix
            )

            if cost <= 0 or cost > max_pos or cost > state['equity'] * MAX_POS_PCT:
                log.info(f"  SKIP {tk}: cost ${cost:.2f} exceeds limits")
                continue

            spread_width = K2 - K1
            if spread_width > 0 and entry_cost_ps / spread_width > 0.50:
                log.info(f"  SKIP {tk}: cost/width {entry_cost_ps/spread_width:.0%} > 50% "
                         f"(entry ${entry_cost_ps:.4f}, width ${spread_width:.2f})")
                continue

            position = {
                'ticker': tk,
                'mode': 'bull',
                'entry_date': str(today.date()),
                'entry_price': S,
                'K1': K1,
                'K2': K2,
                'cost': round(cost, 2),
                'entry_cost_ps': round(entry_cost_ps, 6),
                'max_profit': round(max_profit, 2),
                'vix_at_entry': current_vix,
                'vix_mode': 'high_vix',
                'regime_score': round(rscore, 4) if rscore else None,
                'lgbm_score': round(rankings.get(tk, 0), 4),
                'signals': signal_names,
                'n_signals': n_signals,
            }
            state['open_positions'].append(position)
            state['total_trades'] += 1
            entries += 1

            log_trade({
                'action': 'OPEN',
                'date': str(today.date()),
                'ticker': tk,
                'mode': 'bull',
                'vix_mode': 'high_vix',
                'entry_price': S,
                'strikes': f"{K1}/{K2}",
                'cost': round(cost, 2),
                'max_profit': round(max_profit, 2),
                'vix': current_vix,
                'regime_score': round(rscore, 4) if rscore else None,
                'lgbm_score': round(rankings.get(tk, 0), 4),
                'signals': signal_names,
                'equity': round(state['equity'], 2),
            })
            log.info(f"  ENTER {tk} BULL call spread {K1}/{K2}: cost ${cost:.2f}, "
                     f"max profit ${max_profit:.2f}, VIX {current_vix:.1f}")

        if entries == 0:
            log.info("No entries this rebalance (high-VIX mode)")

    # ==================== LOW-VIX MODE: Pair trades + risk parity ====================
    else:
        rankings = run_lgbm_ranking(sc, close_df)
        if not rankings:
            log.warning("No rankings available. Skipping rebalance.")
            save_state(state)
            return

        ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        long_picks = [t for t, _ in ranked[:LOW_VIX_TOP_K]]
        short_picks = [t for t, _ in ranked[-LOW_VIX_BOTTOM_K:]]

        log.info(f"Rankings: {', '.join(f'{t}={s:.3f}' for t, s in ranked)}")
        log.info(f"LOW-VIX LONG picks (top {LOW_VIX_TOP_K}): {long_picks}")
        log.info(f"LOW-VIX SHORT picks (bottom {LOW_VIX_BOTTOM_K}): {short_picks}")

        max_concurrent = LOW_VIX_TOP_K + LOW_VIX_BOTTOM_K
        max_pos = compute_position_size(state, max_concurrent)
        if max_pos < 30:
            log.warning(f"Position size too small (${max_pos:.0f}). Skipping entries.")
            _print_summary(state)
            save_state(state)
            return

        entries = 0

        # --- LONG SIDE: Bull call spreads on top-ranked ---
        for tk in long_picks:
            if tk not in sc.columns:
                continue
            if any(p['ticker'] == tk and p['mode'] == 'bull' for p in state['open_positions']):
                log.info(f"  SKIP {tk} bull: already holding")
                continue

            S = float(sc[tk].iloc[-1])
            sh_tk = sh[tk] if tk in sh.columns else None
            sl_tk = sl[tk] if tk in sl.columns else None
            sc_tk = sc[tk] if tk in sc.columns else None
            cost, max_profit, K1, K2, entry_cost_ps = price_bull_spread(
                S, SPREAD_PCT, DTE, sh_tk, sl_tk, sc_tk, current_vix
            )

            if cost <= 0 or cost > max_pos or cost > state['equity'] * MAX_POS_PCT:
                log.info(f"  SKIP {tk} bull: cost ${cost:.2f} exceeds limits")
                continue

            spread_width = K2 - K1
            if spread_width > 0 and entry_cost_ps / spread_width > 0.50:
                log.info(f"  SKIP {tk} bull: cost/width {entry_cost_ps/spread_width:.0%} > 50% "
                         f"(entry ${entry_cost_ps:.4f}, width ${spread_width:.2f})")
                continue

            position = {
                'ticker': tk,
                'mode': 'bull',
                'entry_date': str(today.date()),
                'entry_price': S,
                'K1': K1,
                'K2': K2,
                'cost': round(cost, 2),
                'entry_cost_ps': round(entry_cost_ps, 6),
                'max_profit': round(max_profit, 2),
                'vix_at_entry': current_vix,
                'vix_mode': 'low_vix',
                'lgbm_score': round(rankings.get(tk, 0), 4),
            }
            state['open_positions'].append(position)
            state['total_trades'] += 1
            entries += 1

            log_trade({
                'action': 'OPEN',
                'date': str(today.date()),
                'ticker': tk,
                'mode': 'bull',
                'vix_mode': 'low_vix',
                'entry_price': S,
                'strikes': f"{K1}/{K2}",
                'cost': round(cost, 2),
                'max_profit': round(max_profit, 2),
                'vix': current_vix,
                'lgbm_score': round(rankings.get(tk, 0), 4),
                'equity': round(state['equity'], 2),
            })
            log.info(f"  ENTER {tk} BULL call spread {K1}/{K2}: cost ${cost:.2f}, "
                     f"max profit ${max_profit:.2f}")

        # --- SHORT SIDE: Bear put spreads on bottom-ranked ---
        for tk in short_picks:
            if tk not in sc.columns:
                continue
            if any(p['ticker'] == tk and p['mode'] == 'bear' for p in state['open_positions']):
                log.info(f"  SKIP {tk} bear: already holding")
                continue

            S = float(sc[tk].iloc[-1])
            sh_tk = sh[tk] if tk in sh.columns else None
            sl_tk = sl[tk] if tk in sl.columns else None
            sc_tk = sc[tk] if tk in sc.columns else None
            cost, max_profit, K1, K2, entry_cost_ps = price_bear_spread(
                S, SPREAD_PCT, DTE, sh_tk, sl_tk, sc_tk, current_vix
            )

            if cost <= 0 or cost > max_pos or cost > state['equity'] * MAX_POS_PCT:
                log.info(f"  SKIP {tk} bear: cost ${cost:.2f} exceeds limits")
                continue

            spread_width = abs(K1 - K2)
            if spread_width > 0 and entry_cost_ps / spread_width > 0.50:
                log.info(f"  SKIP {tk} bear: cost/width {entry_cost_ps/spread_width:.0%} > 50% "
                         f"(entry ${entry_cost_ps:.4f}, width ${spread_width:.2f})")
                continue

            position = {
                'ticker': tk,
                'mode': 'bear',
                'entry_date': str(today.date()),
                'entry_price': S,
                'K1': K1,
                'K2': K2,
                'cost': round(cost, 2),
                'entry_cost_ps': round(entry_cost_ps, 6),
                'max_profit': round(max_profit, 2),
                'vix_at_entry': current_vix,
                'vix_mode': 'low_vix',
                'lgbm_score': round(rankings.get(tk, 0), 4),
            }
            state['open_positions'].append(position)
            state['total_trades'] += 1
            entries += 1

            log_trade({
                'action': 'OPEN',
                'date': str(today.date()),
                'ticker': tk,
                'mode': 'bear',
                'vix_mode': 'low_vix',
                'entry_price': S,
                'strikes': f"{K1}/{K2}",
                'cost': round(cost, 2),
                'max_profit': round(max_profit, 2),
                'vix': current_vix,
                'lgbm_score': round(rankings.get(tk, 0), 4),
                'equity': round(state['equity'], 2),
            })
            log.info(f"  ENTER {tk} BEAR put spread {K1}/{K2}: cost ${cost:.2f}, "
                     f"max profit ${max_profit:.2f}")

        if entries == 0:
            log.info("No entries this rebalance (low-VIX mode)")

        # Risk parity overlay in low-VIX mode
        state = enter_risk_parity(state, close_df, today)

    _print_summary(state)
    save_state(state)

    # HC #774: Write execution signals — OPEN for new positions, HOLD for existing, CLOSE for exited
    _write_execution_signals(state, today, closed_today=positions_to_close)

    log.info("State saved. Done.")


def _print_summary(state):
    """Print summary of current portfolio state."""
    total = state['wins'] + state['losses']
    wr = state['wins'] / total * 100 if total > 0 else 0
    long_total = state['long_wins'] + state['long_losses']
    short_total = state['short_wins'] + state['short_losses']
    long_wr = state['long_wins'] / long_total * 100 if long_total > 0 else 0
    short_wr = state['short_wins'] / short_total * 100 if short_total > 0 else 0

    log.info(f"\n=== Summary (V9.3 Profit Target) ===")
    log.info(f"Equity: ${state['equity']:.2f} (P&L: ${state['total_pnl']:.2f})")
    log.info(f"Mode: {state.get('current_mode', 'N/A')} | Last rebalance: {state.get('last_rebalance', 'never')}")
    log.info(f"Trades: {state['total_trades']} | W: {state['wins']} L: {state['losses']} | WR: {wr:.1f}%")
    log.info(f"  Long side:  W: {state['long_wins']} L: {state['long_losses']} | WR: {long_wr:.1f}%")
    log.info(f"  Short side: W: {state['short_wins']} L: {state['short_losses']} | WR: {short_wr:.1f}%")
    log.info(f"  Early exits (50% target): {state.get('early_exits', 0)}")

    bulls = [p for p in state['open_positions'] if p['mode'] == 'bull']
    bears = [p for p in state['open_positions'] if p['mode'] == 'bear']
    log.info(f"Open positions: {len(bulls)} bull + {len(bears)} bear = {len(state['open_positions'])} total")
    for p in state['open_positions']:
        log.info(f"  {p['ticker']} {p['mode']} {p['K1']}/{p['K2']} "
                 f"(entry {p['entry_date']}, cost ${p['cost']:.2f}, "
                 f"max_profit ${p.get('max_profit', 0):.2f}, "
                 f"LGBM {p.get('lgbm_score', 'N/A')}, via {p.get('vix_mode', 'N/A')})")

    if state['risk_parity_positions']:
        log.info(f"Risk parity positions: {len(state['risk_parity_positions'])}")
        for p in state['risk_parity_positions']:
            log.info(f"  {p['ticker']} {p['shares']} shares @ ${p['entry_price']:.2f}")


def _write_execution_signals(state, today, closed_today=None):
    """Write V93 execution bridge file on EVERY run.

    HC #774: V93 signals must flow to the spread execution bridge.
    - OPEN: new positions entered today (rebalance day)
    - HOLD: existing positions still open
    - CLOSE: positions that hit profit target or expired today

    The file is ALWAYS written, even if positions list is empty.
    """
    bridge_file = STATE_DIR / 'v93_execution_signals.json'
    closed_today = closed_today or []
    # Build set of tickers closed today for lookup
    closed_tickers = set()
    signals = []

    # 1) CLOSE signals for positions exited today
    for i, pnl, reason, days_held in closed_today:
        # The position was already popped from state, reconstruct from closed_trades
        for ct in reversed(state.get('closed_trades', [])):
            if str(ct.get('date', '')) == str(today.date()) and ct.get('exit_reason') == reason:
                tk = ct['ticker']
                if tk in closed_tickers:
                    continue  # already emitted
                closed_tickers.add(tk)
                signals.append({
                    "ticker": tk,
                    "action": "CLOSE",
                    "direction": ct.get('mode', 'unknown'),
                    "spread_type": "bull_call_spread" if ct.get('mode') == 'bull' else "bear_put_spread",
                    "pnl": ct.get('pnl', 0),
                    "exit_reason": reason,
                    "days_held": days_held,
                    "source": "v93_profit_target" if reason == 'early_exit' else "v93_expiry",
                })
                break

    # 2) OPEN or HOLD signals for current open positions
    today_str = str(today.date())
    for pos in state['open_positions']:
        is_new = str(pos.get('entry_date', '')) == today_str
        action = "OPEN" if is_new else "HOLD"
        spread_type = "bull_call_spread" if pos['mode'] == 'bull' else "bear_put_spread"
        entry_date = pd.Timestamp(pos['entry_date'])
        days_held = (today - entry_date).days

        sig = {
            "ticker": pos['ticker'],
            "action": action,
            "direction": "bull" if pos['mode'] == 'bull' else "bear",
            "spread_type": spread_type,
            "buy_strike": pos['K1'],
            "sell_strike": pos['K2'],
            "paper_cost": pos['cost'],
            "max_profit": pos.get('max_profit', 0),
            "lgbm_score": pos.get('lgbm_score', 0),
            "entry_date": pos.get('entry_date', ''),
            "days_held": days_held,
            "confidence": 0.80,
            "source": "v93_rebalance" if is_new else "v93_daily",
            "exit_rules": {
                "profit_target_pct": 50,
                "max_hold_days": 28,
                "stop_loss_pct": -100,
            },
        }
        signals.append(sig)

    output = {
        "date": today_str,
        "engine": "sector_combined_v93",
        "equity": state['equity'],
        "n_signals": len(signals),
        "positions": signals,
    }
    if not DRY_RUN:
        with open(bridge_file, 'w') as f:
            json.dump(output, f, indent=2)
    log.info(f"Wrote {len(signals)} execution signals to {bridge_file.name} "
             f"({sum(1 for s in signals if s['action']=='OPEN')} OPEN, "
             f"{sum(1 for s in signals if s['action']=='HOLD')} HOLD, "
             f"{sum(1 for s in signals if s['action']=='CLOSE')} CLOSE)")


if __name__ == '__main__':
    run_daily()
