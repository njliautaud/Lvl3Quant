#!/usr/bin/env python3
"""
V10 Calibrated Spread Backtest V1
====================================

CRITICAL QUESTION: What is V10's REAL Sharpe when using properly calibrated
option pricing for SPREADS (not single legs)?

Context:
  - V10 reports Sharpe 6.32 using ATR-based BS pricing with 15% haircut
  - BS calibration found ATR-based IV underestimates real IV by 72.7% median
  - BUT that was tested on single-leg options. V10 uses spreads where the
    cost error partially cancels between long and short legs.
  - For spreads: entry cost = price(short_leg) - price(long_leg), so if both
    legs are underpriced by similar amounts, the spread cost error is SMALLER.

THREE PRICING MODES:
  A) BASELINE: ATR-based BS with 15% haircut (should reproduce ~Sharpe 6.32)
  B) CALIBRATED: Apply multivariate calibration model to BOTH legs, then
     compute spread cost from corrected leg prices. Partial cancellation
     expected since both legs shift up.
  C) MARKET_IV: Use VIX/100 * 1.5 as IV proxy instead of ATR-based IV.
     Tests if the root cause fix (better IV) solves the problem.

V10 Config (exact match to production):
  - 11 sectors, 17 LGBM momentum features
  - 8 positions: 4 top (bull call spreads) + 4 bottom (bear put spreads)
  - 4% OTM moneyness, 3% width (min $3), DTE=28
  - 30% profit target, monthly rebalance
  - $645 starting capital, $2.60 commission RT
  - 1/rank-weighted sizing
  - Walk-forward: 60-day sliding window

Output: output/growth_research/v10_calibrated_spread_backtest_v1/
MLflow experiment: v10_calibrated_spread_backtest
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# ── Detect environment ──
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")

if _NEPTUNE_BASE.exists():
    BASE = _NEPTUNE_BASE
    fprint(f"Running on Neptune: {BASE}")
else:
    BASE = _JUPITER_BASE
    fprint(f"Running on Jupiter: {BASE}")

sys.path.insert(0, str(BASE))
from research.tools.options_pricer import (
    bs_call_price, bs_put_price, estimate_iv, compute_atr,
    price_bull_call_spread, price_bear_put_spread,
    COMMISSION_RT_SPREAD, DEFAULT_HAIRCUT, RISK_FREE_RATE,
)

OUTPUT_DIR = BASE / "output" / "growth_research" / "v10_calibrated_spread_backtest_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CALIB_FILE = BASE / "output" / "growth_research" / "bs_pricing_calibration_v1" / "calibration_results.json"

# ── MLflow setup ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v10_calibrated_spread_backtest"
MLFLOW_OK = False
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    MLFLOW_OK = True
    fprint(f"MLflow OK: {MLFLOW_URI}, experiment={EXPERIMENT_NAME}")
except Exception as e:
    fprint(f"MLflow not available: {e}")

# ── LightGBM ──
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    fprint("WARNING: LightGBM not available. Will use simple momentum ranking.")

# ==================== V10 CONFIG ====================
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
EXTRA_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', 'SHY', 'HYG', 'GLD']
INITIAL_CAPITAL = 645.0
SPREAD_PCT = 3.0
MIN_SPREAD_WIDTH = 3.0
DTE = 28
MONEYNESS_PCT = 4.0
MAX_POS_SIZE = 200
MAX_POS_PCT = 0.40
HAIRCUT = 0.15
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # $2.60 round trip
PROFIT_TARGET_PCT = 0.30
VIX_THRESHOLD = 20.0
HIGH_VIX_TOP_K = 4
LOW_VIX_TOP_K = 4
LOW_VIX_BOTTOM_K = 4
REGIME_BULL_THRESHOLD = 0.4
REBALANCE_INTERVAL_DAYS = 28

# 17 LGBM momentum features
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
]

# ── Calibration model coefficients ──
# From calibration_results.json multivariate model
CALIB_COEF = {
    'bs_price': 1.1200588438223305,
    'abs_moneyness': -95.09807003557364,
    'dte_days': 0.18295721155595857,
    'bs_iv_estimate': 71.40839566666281,
}
CALIB_INTERCEPT = -6.160192231790862

# Load from file if available (overrides hardcoded)
if CALIB_FILE.exists():
    with open(CALIB_FILE) as f:
        calib_data = json.load(f)
    mv = calib_data['calibration_models']['multivariate']
    CALIB_COEF = mv['coefficients']
    CALIB_INTERCEPT = mv['intercept']
    fprint(f"Loaded calibration from {CALIB_FILE}")
    fprint(f"  R2={mv['r2']:.4f}, MAE={mv['mae']:.2f}")
    fprint(f"  Coefficients: {CALIB_COEF}")
else:
    fprint("Using hardcoded calibration coefficients")


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download sector ETF + VIX data via yfinance."""
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    raw = yf.download(all_tickers, start='2023-06-01', progress=False)
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
    spy = close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index)
    return close.loc[ix], sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]


# ==================== FEATURE ENGINEERING ====================

def compute_features(px):
    """Compute the 17 momentum features for a single sector ETF."""
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


# ==================== LGBM RANKING ====================

def run_lgbm_ranking(sc, idx_end):
    """Walk-forward LGBM ranking using data up to idx_end."""
    if not HAS_LGBM:
        rets_21d = sc.iloc[:idx_end + 1].pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    records = []
    all_idx = list(range(max(260, idx_end - 400), idx_end))
    rebal_idx = all_idx[::20]

    for i in rebal_idx[:-1]:
        for tk in sc.columns:
            px = sc[tk].iloc[:i + 1].dropna()
            feats = compute_features(px)
            if not feats:
                continue
            fi = min(i + 28, len(sc) - 1)
            feats.update({
                'date_idx': i, 'ticker': tk,
                'fwd_ret': float(sc[tk].iloc[fi] / sc[tk].iloc[i] - 1)
            })
            records.append(feats)

    df = pd.DataFrame(records)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)

    if len(df) < 50:
        rets_21d = sc.iloc[:idx_end + 1].pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    df['rank_label'] = df.groupby('date_idx')['fwd_ret'].rank(pct=True)
    X_train = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
    y_train = df['rank_label'].values.astype(np.float32)

    m = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1
    )
    m.fit(X_train, y_train)

    current_feats = {}
    for tk in sc.columns:
        px = sc[tk].iloc[:idx_end + 1].dropna()
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


# ==================== RANK-WEIGHTED SIZING ====================

def compute_rank_weights(n_picks):
    """1/rank weighting. Rank 1 gets most capital."""
    if n_picks <= 0:
        return []
    raw = [1.0 / (i + 1) for i in range(n_picks)]
    total = sum(raw)
    return [w / total for w in raw]


# ==================== PRICING ENGINES ====================

def calibrate_leg_price(bs_price, abs_moneyness, dte_days, bs_iv_estimate):
    """Apply multivariate calibration model to a single leg BS price.

    market_mid ≈ coef_bs * bs_price + coef_money * abs_moneyness
                 + coef_dte * dte_days + coef_iv * bs_iv + intercept

    Returns calibrated price (floored at bs_price to avoid negative correction).
    """
    calibrated = (
        CALIB_COEF['bs_price'] * bs_price
        + CALIB_COEF['abs_moneyness'] * abs_moneyness
        + CALIB_COEF['dte_days'] * dte_days
        + CALIB_COEF['bs_iv_estimate'] * bs_iv_estimate
        + CALIB_INTERCEPT
    )
    # Floor at a small positive value -- calibration should not make prices negative
    # but for deep OTM the moneyness term can dominate
    return max(calibrated, bs_price * 0.5, 0.01)


def price_spread_baseline(S, K_long, K_short, dte, atr, vix_val, spread_type='bull_call'):
    """Mode A: ATR-based BS with 15% haircut (production V10 pricing).

    Bull call spread: buy call K_long (lower), sell call K_short (higher)
    Bear put spread: buy put K_long (higher), sell put K_short (lower)

    Returns (entry_cost_ps, max_profit_ps, sigma_used).
    """
    sigma = estimate_iv(atr, S, vix_val)
    T = dte / 365.0

    if spread_type == 'bull_call':
        long_price = bs_call_price(S, K_long, T, sigma=sigma)
        short_price = bs_call_price(S, K_short, T, sigma=sigma)
        fair_spread = long_price - short_price
        fair_spread = max(fair_spread, 0.001)
        entry_cost = fair_spread * (1.0 + HAIRCUT)
        width = K_short - K_long
    else:  # bear_put
        long_price = bs_put_price(S, K_long, T, sigma=sigma)   # K_long is higher
        short_price = bs_put_price(S, K_short, T, sigma=sigma)  # K_short is lower
        fair_spread = long_price - short_price
        fair_spread = max(fair_spread, 0.001)
        entry_cost = fair_spread * (1.0 + HAIRCUT)
        width = K_long - K_short

    max_profit = width - entry_cost
    return entry_cost, max_profit, sigma


def price_spread_calibrated(S, K_long, K_short, dte, atr, vix_val, spread_type='bull_call'):
    """Mode B: Apply calibration correction to BOTH legs, then compute spread.

    The key insight: both legs get corrected upward by similar amounts,
    so the spread cost increase is SMALLER than the single-leg error.

    Calibration model per leg:
      market_mid ≈ 1.12 * bs_price - 95.1 * abs_moneyness + 0.18 * dte + 71.4 * bs_iv - 6.16

    Returns (entry_cost_ps, max_profit_ps, sigma_used).
    """
    sigma = estimate_iv(atr, S, vix_val)
    T = dte / 365.0

    if spread_type == 'bull_call':
        # Bull call spread: buy call at K_long (lower), sell call at K_short (higher)
        bs_long = bs_call_price(S, K_long, T, sigma=sigma)
        bs_short = bs_call_price(S, K_short, T, sigma=sigma)

        # Moneyness for each leg (fraction OTM)
        moneyness_long = abs(K_long / S - 1.0)
        moneyness_short = abs(K_short / S - 1.0)

        # Calibrate each leg independently
        # The calibration model was trained on per-share prices (not per-contract)
        calib_long = calibrate_leg_price(bs_long, moneyness_long, dte, sigma)
        calib_short = calibrate_leg_price(bs_short, moneyness_short, dte, sigma)

        # Spread cost = long leg - short leg (you buy the more expensive near-ATM leg)
        fair_spread = calib_long - calib_short
        fair_spread = max(fair_spread, 0.001)

        # No additional haircut -- calibration already maps to market mid,
        # but add a small 5% haircut for bid-ask on the spread itself
        entry_cost = fair_spread * 1.05
        width = K_short - K_long

    else:  # bear_put
        # Bear put spread: buy put at K_long (higher strike), sell put at K_short (lower)
        bs_long = bs_put_price(S, K_long, T, sigma=sigma)
        bs_short = bs_put_price(S, K_short, T, sigma=sigma)

        moneyness_long = abs(K_long / S - 1.0)
        moneyness_short = abs(K_short / S - 1.0)

        calib_long = calibrate_leg_price(bs_long, moneyness_long, dte, sigma)
        calib_short = calibrate_leg_price(bs_short, moneyness_short, dte, sigma)

        fair_spread = calib_long - calib_short
        fair_spread = max(fair_spread, 0.001)
        entry_cost = fair_spread * 1.05
        width = K_long - K_short

    max_profit = width - entry_cost
    return entry_cost, max_profit, sigma


def price_spread_market_iv(S, K_long, K_short, dte, atr, vix_val, spread_type='bull_call'):
    """Mode C: Use VIX-based IV proxy instead of ATR-based IV.

    sigma = VIX/100 * 1.5 (empirical multiplier for sector ETF IV vs index IV).
    This tests whether the ROOT CAUSE (bad IV estimation) is the problem.

    Still applies 15% haircut like production.

    Returns (entry_cost_ps, max_profit_ps, sigma_used).
    """
    # VIX/100 gives SPX IV. Sector ETFs have higher IV (beta > 1 for most).
    # 1.5x multiplier approximates the sector-to-index IV ratio.
    sigma = max(vix_val / 100.0 * 1.5, 0.10)
    T = dte / 365.0

    if spread_type == 'bull_call':
        long_price = bs_call_price(S, K_long, T, sigma=sigma)
        short_price = bs_call_price(S, K_short, T, sigma=sigma)
        fair_spread = long_price - short_price
        fair_spread = max(fair_spread, 0.001)
        entry_cost = fair_spread * (1.0 + HAIRCUT)
        width = K_short - K_long
    else:  # bear_put
        long_price = bs_put_price(S, K_long, T, sigma=sigma)
        short_price = bs_put_price(S, K_short, T, sigma=sigma)
        fair_spread = long_price - short_price
        fair_spread = max(fair_spread, 0.001)
        entry_cost = fair_spread * (1.0 + HAIRCUT)
        width = K_long - K_short

    max_profit = width - entry_cost
    return entry_cost, max_profit, sigma


# ==================== CONFLUENCE CHECK ====================

def confluence_check(tk, sc, spy, idx_end):
    """Multi-signal confluence for high-VIX bull spreads."""
    px = sc[tk].iloc[:idx_end + 1].dropna()
    if len(px) < 63:
        return False, 0, []

    signals = []
    ret_21d = float(px.iloc[-1] / px.iloc[-21] - 1)
    if ret_21d > 0:
        signals.append('mom_21d_pos')

    sma50 = px.rolling(50).mean()
    if not pd.isna(sma50.iloc[-1]) and px.iloc[-1] > sma50.iloc[-1]:
        signals.append('above_sma50')

    spy_slice = spy.iloc[:idx_end + 1]
    if len(spy_slice) > 21:
        common = px.index.intersection(spy_slice.index)
        if len(common) > 21:
            rel = px.loc[common] / spy_slice.loc[common]
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


# ==================== REGIME ====================

def load_regime_predictions():
    """Load GRU regime predictions."""
    regime_file = BASE / 'output' / 'regime_detector_v1' / 'regime_predictions_v1.npz'
    if not regime_file.exists():
        return None
    data = np.load(regime_file, allow_pickle=True)
    dates = pd.to_datetime(data['dates'])
    scores = data['regime_scores']
    regime_series = pd.Series(scores, index=dates, name='regime_score')
    regime_series = regime_series[~regime_series.index.duplicated(keep='last')]
    return regime_series


def get_regime_score(regime_series, dt):
    """Get regime score with ffill fallback."""
    if regime_series is None:
        return None
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method='ffill')]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return None


def vix_regime_proxy(vix_val):
    """Map VIX to pseudo regime_score when GRU not available."""
    if vix_val >= 25:
        return 0.7
    elif vix_val >= 20:
        return 0.4 + (vix_val - 20) * 0.06
    else:
        return 0.15 + vix_val * 0.0125


# ==================== EXPIRY / PROFIT TARGET ====================

def estimate_current_spread_value(mode, S, K1, K2, remaining_dte, sigma):
    """Estimate current spread value via BS for profit target check."""
    T = max(remaining_dte, 0) / 365.0
    if mode == 'bull':
        return max(bs_call_price(S, K1, T, sigma=sigma) - bs_call_price(S, K2, T, sigma=sigma), 0.0)
    else:
        return max(bs_put_price(S, K2, T, sigma=sigma) - bs_put_price(S, K1, T, sigma=sigma), 0.0)


def compute_intrinsic(mode, S, K1, K2):
    """Compute intrinsic value at expiry."""
    if mode == 'bull':
        return max(S - K1, 0.0) - max(S - K2, 0.0)
    else:  # bear
        return max(K2 - S, 0.0) - max(K1 - S, 0.0)


# ==================== MAIN BACKTEST ENGINE ====================

def run_backtest(sc, sh, sl, spy, vix, regime_series, pricing_mode='baseline'):
    """Run full V10 backtest with specified pricing mode.

    pricing_mode: 'baseline' | 'calibrated' | 'market_iv'

    Returns dict with trades, equity curve, and metrics.
    """
    fprint(f"\n{'='*70}")
    fprint(f"  BACKTEST MODE: {pricing_mode.upper()}")
    fprint(f"{'='*70}")

    pricing_fn = {
        'baseline': price_spread_baseline,
        'calibrated': price_spread_calibrated,
        'market_iv': price_spread_market_iv,
    }[pricing_mode]

    # State
    equity = INITIAL_CAPITAL
    equity_curve = []
    open_positions = []
    closed_trades = []
    last_rebalance_idx = None

    entry_costs_ps = []  # Track entry costs for comparison

    dates = sc.index
    n_days = len(dates)

    # Need at least 260 days of history for features
    start_idx = 280  # Warm up with enough data

    for day_idx in range(start_idx, n_days):
        today = dates[day_idx]
        current_vix = float(vix.iloc[day_idx]) if day_idx < len(vix) else 20.0
        high_vix_mode = current_vix >= VIX_THRESHOLD

        # ── Check exits (profit target + expiry) ──
        positions_to_close = []
        for i, pos in enumerate(open_positions):
            entry_idx = pos['entry_idx']
            days_held = day_idx - entry_idx

            tk = pos['ticker']
            if tk not in sc.columns:
                continue
            S = float(sc[tk].iloc[day_idx])

            K1, K2 = pos['K1'], pos['K2']
            entry_cost_ps = pos['entry_cost_ps']
            max_profit_dollars = pos['max_profit_dollars']

            # Use VIX/100 as vol for mark-to-market (same as production)
            sigma_mtm = max(current_vix / 100.0, 0.10)
            remaining_dte = max(DTE - days_held, 0)

            # Check profit target (after day 0)
            if days_held >= 1 and max_profit_dollars > 0:
                current_value_ps = estimate_current_spread_value(
                    pos['mode'], S, K1, K2, remaining_dte, sigma_mtm
                )
                current_value_dollars = current_value_ps * 100
                entry_cost_no_comm = pos['cost_dollars'] - SPREAD_COMM
                unrealized_gain = current_value_dollars - entry_cost_no_comm
                target_gain = PROFIT_TARGET_PCT * max_profit_dollars

                if unrealized_gain >= target_gain:
                    exit_credit = current_value_dollars - SPREAD_COMM
                    pnl = exit_credit - pos['cost_dollars']
                    positions_to_close.append((i, pnl, 'early_exit', days_held))
                    continue

            # Check expiry
            if days_held >= DTE:
                intrinsic = compute_intrinsic(pos['mode'], S, K1, K2)
                pnl = (intrinsic - entry_cost_ps) * 100 - SPREAD_COMM
                positions_to_close.append((i, pnl, 'expiry', days_held))

        # Close positions (reverse order)
        for i, pnl, reason, days_held in reversed(positions_to_close):
            pos = open_positions.pop(i)
            equity += pnl
            trade_record = {
                'ticker': pos['ticker'],
                'mode': pos['mode'],
                'entry_date': str(dates[pos['entry_idx']].date()),
                'exit_date': str(today.date()),
                'days_held': days_held,
                'cost_dollars': pos['cost_dollars'],
                'entry_cost_ps': pos['entry_cost_ps'],
                'pnl': pnl,
                'exit_reason': reason,
                'vix_at_entry': pos['vix_at_entry'],
                'vix_at_exit': current_vix,
                'sigma_used': pos['sigma_used'],
                'K1': pos['K1'],
                'K2': pos['K2'],
                'entry_price': pos['entry_price'],
                'exit_price': float(sc[pos['ticker']].iloc[day_idx]),
            }
            closed_trades.append(trade_record)

        # Record equity
        equity_curve.append({'date': today, 'equity': equity, 'n_open': len(open_positions)})

        # ── Check for rebalance ──
        is_friday = today.weekday() == 4
        if not is_friday:
            continue

        is_first_fri = today.day <= 7
        has_open = len(open_positions) > 0

        do_rebalance = False
        if is_first_fri:
            do_rebalance = True
        elif not has_open and last_rebalance_idx is not None:
            days_since = day_idx - last_rebalance_idx
            if days_since >= REBALANCE_INTERVAL_DAYS:
                do_rebalance = True
        elif not has_open and last_rebalance_idx is None:
            do_rebalance = True

        if not do_rebalance:
            continue

        last_rebalance_idx = day_idx

        # ── LGBM Ranking ──
        rankings = run_lgbm_ranking(sc, day_idx)
        if not rankings:
            continue

        ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)

        if high_vix_mode:
            # Regime gate
            rscore = get_regime_score(regime_series, today)
            if rscore is None:
                rscore = vix_regime_proxy(current_vix)
            if rscore <= REGIME_BULL_THRESHOLD:
                continue

            picks = [(t, 'bull') for t, _ in ranked[:HIGH_VIX_TOP_K]]
            rank_weights = compute_rank_weights(HIGH_VIX_TOP_K)
            max_concurrent = HIGH_VIX_TOP_K
        else:
            long_picks = [(t, 'bull') for t, _ in ranked[:LOW_VIX_TOP_K]]
            short_picks = [(t, 'bear') for t, _ in ranked[-LOW_VIX_BOTTOM_K:]]
            picks = long_picks + short_picks
            long_weights = compute_rank_weights(LOW_VIX_TOP_K)
            short_weights = compute_rank_weights(LOW_VIX_BOTTOM_K)
            rank_weights = long_weights + short_weights
            max_concurrent = LOW_VIX_TOP_K + LOW_VIX_BOTTOM_K

        # Position sizing
        equity_ratio = equity / INITIAL_CAPITAL
        scaled_max = MAX_POS_SIZE * equity_ratio
        equity_cap = equity * MAX_POS_PCT
        max_pos = min(scaled_max, equity_cap, equity / max(max_concurrent, 2))

        if max_pos < 30:
            continue

        entries = 0
        for rank_idx, (tk, mode) in enumerate(picks):
            if tk not in sc.columns:
                continue
            if any(p['ticker'] == tk and p['mode'] == mode for p in open_positions):
                continue

            # Confluence check for high-VIX bull spreads
            if high_vix_mode and mode == 'bull':
                passes, _, _ = confluence_check(tk, sc, spy, day_idx)
                if not passes:
                    continue

            # Rank-weighted sizing
            rw = rank_weights[rank_idx] if rank_idx < len(rank_weights) else rank_weights[-1]
            n_side = HIGH_VIX_TOP_K if high_vix_mode else (LOW_VIX_TOP_K if mode == 'bull' else LOW_VIX_BOTTOM_K)
            weighted_max_pos = max(30, max_pos * rw * n_side)

            S = float(sc[tk].iloc[day_idx])

            # Compute strikes (exact V10 logic)
            if mode == 'bull':
                K1 = round(S * (1 + MONEYNESS_PCT / 100))  # long call (lower)
                pct_width = K1 * SPREAD_PCT / 100
                adaptive_width = max(MIN_SPREAD_WIDTH, pct_width)
                K2 = round(K1 + adaptive_width)  # short call (higher)
                spread_type = 'bull_call'
            else:
                K2 = round(S * (1 - MONEYNESS_PCT / 100))  # long put (this is K1 in production bear_spread)
                pct_width = K2 * SPREAD_PCT / 100
                adaptive_width = max(MIN_SPREAD_WIDTH, pct_width)
                K1 = round(K2 - adaptive_width)  # short put (lower)
                if K1 >= K2:
                    K1 = K2 - 1
                spread_type = 'bear_put'
                # For bear put: K_long=K2 (higher, buy put), K_short=K1 (lower, sell put)

            # ATR
            sh_tk = sh[tk].iloc[:day_idx + 1] if tk in sh.columns else None
            sl_tk = sl[tk].iloc[:day_idx + 1] if tk in sl.columns else None
            sc_tk = sc[tk].iloc[:day_idx + 1] if tk in sc.columns else None

            if sh_tk is not None and len(sh_tk) >= 14:
                atr = compute_atr(sh_tk, sl_tk, sc_tk, period=14)
            else:
                atr = S * 0.015

            # Price the spread
            if spread_type == 'bull_call':
                entry_cost_ps, max_profit_ps, sigma_used = pricing_fn(
                    S, K1, K2, DTE, atr, current_vix, spread_type='bull_call'
                )
                width = K2 - K1
            else:
                entry_cost_ps, max_profit_ps, sigma_used = pricing_fn(
                    S, K2, K1, DTE, atr, current_vix, spread_type='bear_put'
                )
                width = K2 - K1

            cost_dollars = entry_cost_ps * 100 + SPREAD_COMM
            max_profit_dollars = max_profit_ps * 100 - SPREAD_COMM

            # Filters
            if cost_dollars <= 0 or cost_dollars > weighted_max_pos or cost_dollars > equity * MAX_POS_PCT:
                continue

            if width > 0 and entry_cost_ps / width > 0.50:
                continue

            position = {
                'ticker': tk,
                'mode': mode,
                'entry_idx': day_idx,
                'entry_price': S,
                'K1': K1,
                'K2': K2,
                'cost_dollars': round(cost_dollars, 2),
                'entry_cost_ps': round(entry_cost_ps, 6),
                'max_profit_dollars': round(max_profit_dollars, 2),
                'vix_at_entry': current_vix,
                'sigma_used': round(sigma_used, 4),
            }
            open_positions.append(position)
            entry_costs_ps.append(entry_cost_ps)
            entries += 1

    # ── Close any remaining open positions at last available price ──
    for pos in open_positions:
        tk = pos['ticker']
        S = float(sc[tk].iloc[-1])
        intrinsic = compute_intrinsic(pos['mode'], S, pos['K1'], pos['K2'])
        pnl = (intrinsic - pos['entry_cost_ps']) * 100 - SPREAD_COMM
        equity += pnl
        closed_trades.append({
            'ticker': tk,
            'mode': pos['mode'],
            'entry_date': str(dates[pos['entry_idx']].date()),
            'exit_date': str(dates[-1].date()),
            'days_held': n_days - 1 - pos['entry_idx'],
            'cost_dollars': pos['cost_dollars'],
            'entry_cost_ps': pos['entry_cost_ps'],
            'pnl': pnl,
            'exit_reason': 'force_close',
            'vix_at_entry': pos['vix_at_entry'],
            'vix_at_exit': float(vix.iloc[-1]),
            'sigma_used': pos['sigma_used'],
            'K1': pos['K1'],
            'K2': pos['K2'],
            'entry_price': pos['entry_price'],
            'exit_price': S,
        })

    equity_curve.append({'date': dates[-1], 'equity': equity, 'n_open': 0})

    # ── Compute metrics ──
    eq_df = pd.DataFrame(equity_curve)
    eq_df = eq_df.set_index('date')
    eq_df = eq_df[~eq_df.index.duplicated(keep='last')]

    metrics = compute_metrics(closed_trades, eq_df, pricing_mode)
    metrics['entry_costs_ps'] = entry_costs_ps

    return {
        'trades': closed_trades,
        'equity_curve': eq_df,
        'metrics': metrics,
        'pricing_mode': pricing_mode,
    }


# ==================== METRICS ====================

def compute_metrics(trades, eq_df, mode_name):
    """Compute comprehensive performance metrics."""
    if not trades:
        return {
            'mode': mode_name, 'n_trades': 0, 'sharpe': 0, 'sortino': 0,
            'pf': 0, 'wr': 0, 'mdd': 0, 'total_return': 0,
            'mean_entry_cost_ps': 0, 'mean_pnl': 0,
        }

    pnls = [t['pnl'] for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    wr = wins / n if n > 0 else 0

    total_pnl = sum(pnls)
    mean_pnl = np.mean(pnls)
    std_pnl = np.std(pnls) if n > 1 else 1.0

    # Sharpe (annualized, assuming ~12 trades/year for monthly rebalance)
    trades_per_year = 12.0 * 8  # ~8 positions per month
    sharpe = (mean_pnl / (std_pnl + 1e-10)) * np.sqrt(trades_per_year) if std_pnl > 0 else 0

    # Sortino
    downside = [p for p in pnls if p < 0]
    downside_std = np.std(downside) if len(downside) > 1 else std_pnl
    sortino = (mean_pnl / (downside_std + 1e-10)) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Profit Factor
    gross_wins = sum(p for p in pnls if p > 0)
    gross_losses = abs(sum(p for p in pnls if p < 0))
    pf = gross_wins / (gross_losses + 1e-10)

    # MDD from equity curve
    if len(eq_df) > 0:
        peak = eq_df['equity'].cummax()
        dd = (eq_df['equity'] - peak) / peak
        mdd = float(dd.min())
    else:
        mdd = 0

    # Total return
    total_return = total_pnl / INITIAL_CAPITAL

    # Mean entry cost
    entry_costs = [t['entry_cost_ps'] for t in trades]
    mean_entry_cost_ps = np.mean(entry_costs)

    # Early exit stats
    early_exits = sum(1 for t in trades if t['exit_reason'] == 'early_exit')

    metrics = {
        'mode': mode_name,
        'n_trades': n,
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'pf': round(pf, 2),
        'wr': round(wr * 100, 1),
        'mdd': round(mdd * 100, 1),
        'total_return': round(total_return * 100, 1),
        'total_pnl': round(total_pnl, 2),
        'mean_pnl': round(mean_pnl, 2),
        'std_pnl': round(std_pnl, 2),
        'mean_entry_cost_ps': round(mean_entry_cost_ps, 4),
        'wins': wins,
        'losses': losses,
        'early_exits': early_exits,
    }
    return metrics


def compute_regime_breakdown(trades, spy):
    """Per-regime (green/red/flat) breakdown using SPY close-to-close."""
    if not trades or len(spy) < 2:
        return {}

    # Classify each trade's entry date as green/red/flat
    spy_rets = spy.pct_change().dropna()

    regime_trades = {'green': [], 'red': [], 'flat': []}
    for t in trades:
        entry_date = pd.Timestamp(t['entry_date'])
        # Find the closest SPY return
        if entry_date in spy_rets.index:
            ret = float(spy_rets.loc[entry_date])
        else:
            closest = spy_rets.index[spy_rets.index.get_indexer([entry_date], method='nearest')]
            if len(closest) > 0:
                ret = float(spy_rets.loc[closest[0]])
            else:
                ret = 0.0

        # Also look at the return over the trade's holding period
        exit_date = pd.Timestamp(t['exit_date'])
        mask = (spy.index >= entry_date) & (spy.index <= exit_date)
        period_spy = spy[mask]
        if len(period_spy) >= 2:
            period_ret = float(period_spy.iloc[-1] / period_spy.iloc[0] - 1)
        else:
            period_ret = ret

        if period_ret > 0.005:
            regime_trades['green'].append(t)
        elif period_ret < -0.005:
            regime_trades['red'].append(t)
        else:
            regime_trades['flat'].append(t)

    breakdown = {}
    for regime, rtrades in regime_trades.items():
        if not rtrades:
            breakdown[regime] = {'n': 0, 'sharpe': 0, 'wr': 0, 'mean_pnl': 0}
            continue
        pnls = [t['pnl'] for t in rtrades]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if n > 1 else 1.0
        sharpe = (mean_pnl / (std_pnl + 1e-10)) * np.sqrt(96) if std_pnl > 0 else 0  # ~96 trades/yr
        breakdown[regime] = {
            'n': n,
            'sharpe': round(sharpe, 2),
            'wr': round(wins / n * 100, 1) if n > 0 else 0,
            'mean_pnl': round(mean_pnl, 2),
            'total_pnl': round(sum(pnls), 2),
        }
    return breakdown


# ==================== MAIN ====================

def main():
    t0 = time.time()

    fprint("=" * 70)
    fprint("  V10 CALIBRATED SPREAD BACKTEST")
    fprint("  Question: What is V10's REAL Sharpe with calibrated spread pricing?")
    fprint("=" * 70)

    # Download data
    fprint("\nDownloading data...")
    close_df, sc, sh, sl, spy, vix = download_data()
    fprint(f"Data: {sc.index[0].date()} to {sc.index[-1].date()}, "
           f"{len(sc)} days, {len(sc.columns)} sectors")

    # Load regime
    regime_series = load_regime_predictions()
    if regime_series is not None:
        fprint(f"Regime predictions loaded: {len(regime_series)} days")
    else:
        fprint("No regime predictions — using VIX proxy")

    # Run three backtests
    results = {}
    for mode in ['baseline', 'calibrated', 'market_iv']:
        results[mode] = run_backtest(sc, sh, sl, spy, vix, regime_series, pricing_mode=mode)

    # Regime breakdown
    fprint("\n" + "=" * 70)
    fprint("  REGIME BREAKDOWNS")
    fprint("=" * 70)

    for mode in ['baseline', 'calibrated', 'market_iv']:
        breakdown = compute_regime_breakdown(results[mode]['trades'], spy)
        results[mode]['regime_breakdown'] = breakdown
        fprint(f"\n  {mode.upper()} per-regime:")
        for regime in ['green', 'red', 'flat']:
            b = breakdown.get(regime, {})
            fprint(f"    {regime:5s}: n={b.get('n',0):3d}  Sharpe={b.get('sharpe',0):6.2f}  "
                   f"WR={b.get('wr',0):5.1f}%  mean_PnL=${b.get('mean_pnl',0):7.2f}  "
                   f"total_PnL=${b.get('total_pnl',0):8.2f}")

    # ── Summary comparison ──
    fprint("\n" + "=" * 70)
    fprint("  COMPARISON SUMMARY")
    fprint("=" * 70)
    fprint(f"{'Mode':<15} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
           f"{'WR%':>6} {'MDD%':>6} {'TotRet%':>8} {'AvgCost':>8} {'AvgPnL':>8}")
    fprint("-" * 90)

    for mode in ['baseline', 'calibrated', 'market_iv']:
        m = results[mode]['metrics']
        fprint(f"{mode:<15} {m['n_trades']:>6} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
               f"{m['pf']:>6.2f} {m['wr']:>6.1f} {m['mdd']:>6.1f} {m['total_return']:>8.1f} "
               f"${m['mean_entry_cost_ps']:>7.4f} ${m['mean_pnl']:>7.2f}")

    # ── Entry cost analysis ──
    fprint("\n" + "=" * 70)
    fprint("  ENTRY COST ANALYSIS (per-share spread cost)")
    fprint("=" * 70)

    for mode in ['baseline', 'calibrated', 'market_iv']:
        costs = results[mode]['metrics'].get('entry_costs_ps', [])
        if costs:
            fprint(f"  {mode:<15}: mean=${np.mean(costs):.4f}  median=${np.median(costs):.4f}  "
                   f"std=${np.std(costs):.4f}  min=${np.min(costs):.4f}  max=${np.max(costs):.4f}")

    # Cost inflation ratio
    baseline_costs = results['baseline']['metrics'].get('entry_costs_ps', [])
    calib_costs = results['calibrated']['metrics'].get('entry_costs_ps', [])
    if baseline_costs and calib_costs:
        ratio = np.mean(calib_costs) / np.mean(baseline_costs) if np.mean(baseline_costs) > 0 else 0
        fprint(f"\n  Calibrated/Baseline cost ratio: {ratio:.2f}x")
        fprint(f"  (If ratio < 1.73, partial cancellation is working — "
               f"single-leg error was 72.7% but spread error is smaller)")

    # ── Regime-agnostic validation (HC #428) ──
    fprint("\n" + "=" * 70)
    fprint("  REGIME-AGNOSTIC VALIDATION (HC #428)")
    fprint("=" * 70)

    for mode in ['baseline', 'calibrated', 'market_iv']:
        bd = results[mode].get('regime_breakdown', {})
        sharpe_green = bd.get('green', {}).get('sharpe', 0)
        sharpe_red = bd.get('red', {}).get('sharpe', 0)
        max_s = max(abs(sharpe_green), abs(sharpe_red), 0.01)
        regime_skew = abs(sharpe_green - sharpe_red) / max_s
        passes = regime_skew <= 0.50
        fprint(f"  {mode:<15}: Sharpe_green={sharpe_green:6.2f}  Sharpe_red={sharpe_red:6.2f}  "
               f"skew={regime_skew:.2f}  {'PASS' if passes else 'FAIL'}")

    # ── Key conclusions ──
    fprint("\n" + "=" * 70)
    fprint("  KEY CONCLUSIONS")
    fprint("=" * 70)

    baseline_sharpe = results['baseline']['metrics']['sharpe']
    calib_sharpe = results['calibrated']['metrics']['sharpe']
    mktiv_sharpe = results['market_iv']['metrics']['sharpe']

    fprint(f"  Baseline (ATR+15% haircut) Sharpe: {baseline_sharpe:.2f}")
    fprint(f"  Calibrated (multivariate model)  Sharpe: {calib_sharpe:.2f}")
    fprint(f"  Market IV (VIX*1.5 proxy)       Sharpe: {mktiv_sharpe:.2f}")

    if baseline_sharpe > 0:
        calib_retention = calib_sharpe / baseline_sharpe * 100
        mktiv_retention = mktiv_sharpe / baseline_sharpe * 100
        fprint(f"\n  Sharpe retention (calibrated vs baseline): {calib_retention:.0f}%")
        fprint(f"  Sharpe retention (market_iv vs baseline):  {mktiv_retention:.0f}%")

    if calib_sharpe > 2.0:
        fprint("\n  VERDICT: V10 edge SURVIVES calibration. Spread cancellation works.")
        fprint(f"  Real Sharpe ~{calib_sharpe:.1f} (was {baseline_sharpe:.1f} with BS pricing)")
    elif calib_sharpe > 0:
        fprint(f"\n  VERDICT: V10 edge REDUCED but positive. Real Sharpe ~{calib_sharpe:.1f}")
        fprint(f"  Spread cancellation helps but doesn't fully offset the pricing error.")
    else:
        fprint(f"\n  VERDICT: V10 edge DOES NOT SURVIVE calibration. Sharpe {calib_sharpe:.1f}")
        fprint(f"  The pricing error is too large even with spread cancellation.")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s")

    # ── Save results ──
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'runtime_seconds': round(elapsed, 1),
        'data_range': f"{sc.index[0].date()} to {sc.index[-1].date()}",
        'n_days': len(sc),
        'calibration_model': {
            'coefficients': CALIB_COEF,
            'intercept': CALIB_INTERCEPT,
        },
        'metrics': {},
        'regime_breakdowns': {},
    }

    for mode in ['baseline', 'calibrated', 'market_iv']:
        m = results[mode]['metrics'].copy()
        m.pop('entry_costs_ps', None)  # Don't save raw list
        save_results['metrics'][mode] = m
        save_results['regime_breakdowns'][mode] = results[mode].get('regime_breakdown', {})

    results_path = OUTPUT_DIR / "backtest_results.json"
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    # Save trades for each mode
    for mode in ['baseline', 'calibrated', 'market_iv']:
        trades_path = OUTPUT_DIR / f"trades_{mode}.json"
        with open(trades_path, 'w') as f:
            json.dump(results[mode]['trades'], f, indent=2, default=str)

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            with mlflow.start_run(run_name=f"v10_calibrated_spread_bt_{datetime.now():%Y%m%d_%H%M}"):
                # Log config
                mlflow.log_param("initial_capital", INITIAL_CAPITAL)
                mlflow.log_param("moneyness_pct", MONEYNESS_PCT)
                mlflow.log_param("spread_pct", SPREAD_PCT)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("profit_target", PROFIT_TARGET_PCT)
                mlflow.log_param("haircut", HAIRCUT)
                mlflow.log_param("n_days", len(sc))
                mlflow.log_param("data_range", f"{sc.index[0].date()} to {sc.index[-1].date()}")

                for mode in ['baseline', 'calibrated', 'market_iv']:
                    m = results[mode]['metrics']
                    mlflow.log_metric(f"{mode}_sharpe", m['sharpe'])
                    mlflow.log_metric(f"{mode}_sortino", m['sortino'])
                    mlflow.log_metric(f"{mode}_pf", m['pf'])
                    mlflow.log_metric(f"{mode}_wr", m['wr'])
                    mlflow.log_metric(f"{mode}_mdd", m['mdd'])
                    mlflow.log_metric(f"{mode}_total_return", m['total_return'])
                    mlflow.log_metric(f"{mode}_n_trades", m['n_trades'])
                    mlflow.log_metric(f"{mode}_mean_entry_cost_ps", m['mean_entry_cost_ps'])
                    mlflow.log_metric(f"{mode}_mean_pnl", m['mean_pnl'])

                # Key comparison metrics
                if baseline_sharpe > 0:
                    mlflow.log_metric("calib_sharpe_retention_pct",
                                      calib_sharpe / baseline_sharpe * 100)
                    mlflow.log_metric("mktiv_sharpe_retention_pct",
                                      mktiv_sharpe / baseline_sharpe * 100)

                if baseline_costs and calib_costs:
                    mlflow.log_metric("cost_inflation_ratio",
                                      np.mean(calib_costs) / (np.mean(baseline_costs) + 1e-10))

                mlflow.log_artifact(str(results_path))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint("\nDone.")


if __name__ == '__main__':
    main()
