#!/usr/bin/env python3
"""
Sector Spreads V6 Paper Trading Engine
========================================

Validated V6 config: Sharpe 2.79, 5/5 gates across all stress tests.

Strategy:
  VIX < 20:  Bull call spreads (top-ranked) + bear put spreads (bottom-ranked) = pairs
  VIX >= 20: Bull call spreads only on top-ranked sectors
  GRU regime filter: score > 0.4 = elevated (required for entries)

Config:
  - Capital: $645 | DTE: 21 | Spread: 3% OTM moneyness: 2%
  - Rebalance: weekly (Friday close)
  - LGBM ranking with 17 features (dropped: vol_21d, vol_63d, sector_relative_vol_21d, maxdd_63d)
  - Hold to expiry (DTE 21)
  - $2.60/spread commission

Runs weekly on Fridays at 4:00 PM ET (via PM2 cron).
Logs trade recommendations to output/paper_engines/sector_spreads_v6/
Sends Discord alerts via webhook.

Usage:
    python sector_spreads_v6_paper.py              # normal run
    python sector_spreads_v6_paper.py --dry-run    # simulate without state changes
"""
import json
import logging
import os
import sys
import traceback
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──
BASE = Path(__file__).resolve().parents[2]  # scripts/paper_engines -> Lvl3Quant
sys.path.insert(0, str(BASE))

from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    exit_spread_value,
    compute_atr,
    COMMISSION_RT_SPREAD,
    DEFAULT_HAIRCUT,
)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# ── Output / State / Log paths ──
OUTPUT_DIR = BASE / 'output' / 'paper_engines' / 'sector_spreads_v6'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR = BASE / 'state'
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / 'sector_spreads_v6_paper_state.json'
TRADE_LOG = OUTPUT_DIR / 'trades.jsonl'
RECO_LOG = OUTPUT_DIR / 'recommendations.jsonl'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'engine.log'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger('sector_spreads_v6')

# ==================== DRY-RUN MODE ====================
DRY_RUN = '--dry-run' in sys.argv

# ==================== STRATEGY CONFIG (V6 Production) ====================
SECTORS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
EXTRA_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', 'SHY', 'HYG', 'GLD']
INITIAL_CAPITAL = 645.0
REBALANCE_INTERVAL = 5       # weekly (5 trading days)
SPREAD_PCT = 3.0              # spread width as % of strike
DTE = 21                      # hold to expiry
MONEYNESS_PCT = 2.0           # 2% OTM
MAX_POS_SIZE = 200            # max per trade
MAX_POS_PCT = 0.40            # max % of equity per trade
HAIRCUT = 0.15
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM   # $2.60 round trip

# VIX threshold — switches between pair mode and bull-only
VIX_THRESHOLD = 20.0

# High-VIX mode (bull spreads only): top-2 sectors
HIGH_VIX_TOP_K = 2
# Low-VIX mode (pairs): long top + short bottom
LOW_VIX_TOP_K = 2
LOW_VIX_BOTTOM_K = 2

# Regime thresholds
REGIME_BULL_THRESHOLD = 0.4

# Regime predictions file
REGIME_FILE = BASE / 'output' / 'regime_detector_v1' / 'regime_predictions_v1.npz'

# LGBM training lookback
LGBM_TRAIN_DAYS = 500

# ==================== V6 FEATURES (17 total — dropped vol_21d, vol_63d, sector_relative_vol_21d, maxdd_63d) ====================
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'sharpe_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y', 'up_capture',
    'trend_r2_63d', 'trend_slope_63d',
    # cross-asset (kept 2 of 3, dropped sector_relative_vol_21d)
    'sector_spy_beta_63d',
    'cross_sector_dispersion',
]
assert len(FEAT_COLS) == 17, f"Expected 17 features, got {len(FEAT_COLS)}"


# ==================== DISCORD WEBHOOK ====================

def send_discord_alert(message: str):
    """Send alert to Discord via webhook."""
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        log.warning("DISCORD_WEBHOOK_URL not set — alert not sent via webhook")
        return
    try:
        import urllib.request
        data = json.dumps({"content": message}).encode()
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=10)
        log.info("Discord alert sent successfully")
    except Exception as e:
        log.error(f"Failed to send Discord alert: {e}")


# ==================== REGIME MODEL ====================

def load_regime_predictions():
    """Load GRU regime predictions from pre-computed NPZ file."""
    if not REGIME_FILE.exists():
        log.warning("Regime file not found — will use VIX-based proxy")
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


# ==================== STATE MANAGEMENT ====================

def _default_state():
    return {
        'config_version': 'v6',
        'equity': INITIAL_CAPITAL,
        'open_positions': [],
        'closed_trades': [],
        'last_rebalance': None,
        'days_since_rebalance': 999,
        'total_trades': 0,
        'total_pnl': 0.0,
        'wins': 0,
        'losses': 0,
        'long_wins': 0,
        'long_losses': 0,
        'short_wins': 0,
        'short_losses': 0,
        'current_mode': None,
        'created': datetime.now().isoformat(),
    }


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


def save_state(state):
    if DRY_RUN:
        log.info("[DRY-RUN] State NOT saved")
        return
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(trade_record):
    if DRY_RUN:
        log.info(f"[DRY-RUN] Trade NOT logged: {trade_record.get('action')} "
                 f"{trade_record.get('ticker', 'N/A')}")
        return
    with open(TRADE_LOG, 'a') as f:
        f.write(json.dumps(trade_record, default=str) + '\n')


def log_recommendation(reco):
    """Log weekly trade recommendation to JSONL."""
    if DRY_RUN:
        return
    with open(RECO_LOG, 'a') as f:
        f.write(json.dumps(reco, default=str) + '\n')


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download sector ETF data using yfinance."""
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    log.info(f"Downloading data for {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, period='2y', progress=False)
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
    log.info(f"Data downloaded: {len(ix)} trading days, {len(sc.columns)} sectors")
    return close.loc[ix], sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]


# ==================== FEATURE ENGINEERING (V6: 17 features) ====================

def compute_features(px, sector_ticker, close_df):
    """Compute all 17 V6 features for a single sector ETF.
    Dropped from v4/v7: vol_21d, vol_63d, sector_relative_vol_21d, maxdd_63d."""
    from scipy import stats

    if len(px) < 260:
        return None

    f = {}

    # Returns at various lookbacks
    for lb, nm in [(5, 'ret_5d'), (10, 'ret_10d'), (21, 'ret_21d'),
                   (63, 'ret_63d'), (126, 'ret_126d'), (252, 'ret_252d')]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()

    # Sharpe 63d (kept)
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0

    # Pct of 52w high (kept)
    f['pct_52w_high'] = float(px.iloc[-1] / px.iloc[-252:].max())

    # Momentum acceleration (kept)
    f['mom_accel'] = f['ret_21d'] - f['ret_63d'] / 3

    # Pct positive months 12m (kept)
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5

    # Sortino 63d (kept)
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0

    # Calmar 1y (kept — uses maxdd internally but maxdd_63d feature itself is dropped)
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)

    # Up capture (kept)
    up_days = rets[rets > 0]
    f['up_capture'] = float(up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)) if len(up_days) > 10 else 1.0

    # Trend R2 and slope (kept)
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val ** 2
        f['trend_slope_63d'] = slope * 252
    else:
        f['trend_r2_63d'] = 0.0
        f['trend_slope_63d'] = 0.0

    # Cross-asset: sector-SPY beta 63d (kept)
    spy = close_df['SPY'].dropna() if 'SPY' in close_df.columns else None
    if spy is not None and len(spy) > 63:
        spy_ret = spy.pct_change().dropna()
        sector_px = close_df[sector_ticker].dropna() if sector_ticker in close_df.columns else None
        if sector_px is not None and len(sector_px) > 63:
            sec_ret = sector_px.pct_change().dropna()
            common = spy_ret.index.intersection(sec_ret.index)
            if len(common) > 63:
                sr = sec_ret.loc[common].iloc[-63:]
                mr = spy_ret.loc[common].iloc[-63:]
                cov = np.cov(sr.values, mr.values)
                f['sector_spy_beta_63d'] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
            else:
                f['sector_spy_beta_63d'] = 1.0
        else:
            f['sector_spy_beta_63d'] = 1.0
    else:
        f['sector_spy_beta_63d'] = 1.0

    # Cross-asset: cross-sector dispersion (kept)
    sector_cols = [c for c in SECTORS if c in close_df.columns]
    if len(sector_cols) > 3:
        sector_rets = close_df[sector_cols].pct_change()
        daily_disp = sector_rets.std(axis=1)
        if len(daily_disp) > 21:
            f['cross_sector_dispersion'] = float(daily_disp.rolling(21).mean().iloc[-1])
        else:
            f['cross_sector_dispersion'] = 0.01
    else:
        f['cross_sector_dispersion'] = 0.01

    return f


# ==================== LGBM RANKING (V6: 17 features, 500-day train) ====================

def run_lgbm_ranking(sc, close_df):
    """Train LGBM on recent 500 trading days to rank sectors.
    Returns dict of {ticker: score} where higher = better (long candidate)."""
    if not HAS_LGBM:
        log.warning("No LightGBM — using simple momentum ranking")
        rets_21d = sc.pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    # Build training data from trailing rebalance periods
    records = []
    n_days = min(LGBM_TRAIN_DAYS, len(sc))
    all_dates = sc.index[-n_days:]
    rebal_dates = all_dates[::REBALANCE_INTERVAL]

    for dt in rebal_dates[:-1]:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx + 1].dropna()
            feats = compute_features(px, tk, close_df.iloc[:idx + 1])
            if not feats:
                continue
            fi = min(idx + 14, len(sc) - 1)
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
    log.info(f"LGBM trained on {len(df)} samples, {len(FEAT_COLS)} features")

    # Predict current rankings
    current_feats = {}
    for tk in sc.columns:
        px = sc[tk].dropna()
        feats = compute_features(px, tk, close_df)
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


# ==================== SPREAD PRICING ====================

def price_bull_spread(S, spread_pct, dte, sh_tk, sl_tk, sc_tk, vix_val):
    """Price a bull call spread with 2% OTM moneyness.
    Returns (cost_dollars, max_profit_dollars, K1, K2, entry_cost_ps)."""
    K1 = round(S * (1 + MONEYNESS_PCT / 100))
    K2 = round(K1 * (1 + spread_pct / 100))

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
    """Price a bear put spread with 2% OTM moneyness.
    K1 = S * 0.98 (long put, higher strike), K2 = K1 * 0.97 (short put, lower).
    Returns (cost_dollars, max_profit_dollars, K1, K2, entry_cost_ps)."""
    K1 = round(S * (1 - MONEYNESS_PCT / 100))  # long put strike (higher)
    K2 = round(K1 * (1 - spread_pct / 100))     # short put strike (lower)
    if K2 >= K1:
        K2 = K1 - 1

    if sh_tk is not None and len(sh_tk) >= 14:
        atr = compute_atr(sh_tk, sl_tk, sc_tk, period=14)
    else:
        atr = S * 0.015

    # price_bear_put_spread expects K1 < K2 (K1=lower, K2=higher)
    entry_cost_ps, max_profit_ps = price_bear_put_spread(
        S=S, K1=K2, K2=K1, dte=dte, atr=atr, vix=vix_val, haircut=HAIRCUT
    )

    cost_dollars = entry_cost_ps * 100 + SPREAD_COMM
    max_profit_dollars = max_profit_ps * 100 - SPREAD_COMM
    # Return K1=lower (short put), K2=higher (long put) for exit calc
    return cost_dollars, max_profit_dollars, K2, K1, entry_cost_ps


# ==================== POSITION EXIT (hold to expiry only) ====================

def check_bull_exit(pos, current_price, days_held):
    """Bull call spread at expiry: intrinsic = max(S-K1,0) - max(S-K2,0)."""
    if days_held < DTE:
        return False, 0, 'hold'
    S = current_price
    K1, K2 = pos['K1'], pos['K2']
    entry_cost_ps = pos.get('entry_cost_ps', (pos['cost'] - SPREAD_COMM) / 100.0)
    intrinsic = max(S - K1, 0.0) - max(S - K2, 0.0)
    pnl = (intrinsic - entry_cost_ps) * 100 - SPREAD_COMM
    return True, pnl, 'expiry'


def check_bear_exit(pos, current_price, days_held):
    """Bear put spread at expiry: intrinsic = max(K2-S,0) - max(K1-S,0)."""
    if days_held < DTE:
        return False, 0, 'hold'
    S = current_price
    K1, K2 = pos['K1'], pos['K2']
    entry_cost_ps = pos.get('entry_cost_ps', (pos['cost'] - SPREAD_COMM) / 100.0)
    intrinsic = max(K2 - S, 0.0) - max(K1 - S, 0.0)
    pnl = (intrinsic - entry_cost_ps) * 100 - SPREAD_COMM
    return True, pnl, 'expiry'


# ==================== POSITION SIZING ====================

def compute_position_size(state, max_concurrent):
    """Position size scales with equity growth."""
    equity_ratio = state['equity'] / INITIAL_CAPITAL
    scaled_max = MAX_POS_SIZE * equity_ratio
    equity_cap = state['equity'] * MAX_POS_PCT
    return min(scaled_max, equity_cap, state['equity'] / max(max_concurrent, 2))


# ==================== MAIN WEEKLY RUN ====================

def run_weekly():
    """Main weekly run. Called every Friday at market close."""
    state = load_state()
    run_ts = datetime.now().isoformat()

    if DRY_RUN:
        log.info("=" * 60)
        log.info("  DRY-RUN MODE — no state changes will be persisted")
        log.info("=" * 60)

    log.info("=== Sector Spreads V6 Paper Engine ===")
    log.info(f"Equity: ${state['equity']:.2f} | Open: {len(state['open_positions'])} | "
             f"Trades: {state['total_trades']} | W/L: {state['wins']}/{state['losses']}")

    # Load regime predictions
    regime_series = load_regime_predictions()

    # Download latest data
    try:
        close_df, sc, sh, sl, spy, vix = download_data()
    except Exception as e:
        log.error(f"Data download failed: {e}")
        send_discord_alert(f"[V6 Paper] Data download failed: {e}")
        return

    today = sc.index[-1]
    current_vix = float(vix.iloc[-1])
    high_vix_mode = current_vix >= VIX_THRESHOLD

    # Get regime score
    rscore = get_regime_score(regime_series, today)
    if rscore is None:
        rscore = vix_regime_proxy(current_vix)
        regime_source = 'VIX proxy'
    else:
        regime_source = 'GRU model'

    regime_active = rscore > REGIME_BULL_THRESHOLD
    mode_label = 'HIGH-VIX (bull only)' if high_vix_mode else 'LOW-VIX (pairs)'

    log.info(f"Date: {today.date()} | VIX: {current_vix:.1f} | Mode: {mode_label}")
    log.info(f"Regime score: {rscore:.3f} ({regime_source}) | "
             f"Gate: {'ACTIVE' if regime_active else 'INACTIVE'}")

    # ==================== CHECK EXISTING POSITIONS FOR EXPIRY ====================
    positions_to_close = []
    for i, pos in enumerate(state['open_positions']):
        entry_date = pd.Timestamp(pos['entry_date'])
        days_held = len(sc.index[(sc.index > entry_date) & (sc.index <= today)])

        if pos['ticker'] not in sc.columns:
            continue

        current_price = float(sc[pos['ticker']].iloc[-1])

        if pos['mode'] == 'bull':
            should_exit, pnl, reason = check_bull_exit(pos, current_price, days_held)
        else:
            should_exit, pnl, reason = check_bear_exit(pos, current_price, days_held)

        if should_exit:
            positions_to_close.append((i, pnl, reason, days_held))
            log.info(f"  EXIT {pos['ticker']} {pos['mode']} spread: "
                     f"PnL ${pnl:.2f} ({reason}, held {days_held}d)")

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
        }
        state['closed_trades'].append(trade_record)
        log_trade(trade_record)

    # ==================== REGIME GATE ====================
    if not regime_active:
        log.info(f"Regime score {rscore:.3f} < {REGIME_BULL_THRESHOLD} — no new entries")

        reco = {
            'timestamp': run_ts,
            'date': str(today.date()),
            'vix': current_vix,
            'regime_score': round(rscore, 4),
            'regime_source': regime_source,
            'action': 'HOLD',
            'reason': 'regime_inactive',
            'equity': round(state['equity'], 2),
        }
        log_recommendation(reco)

        send_discord_alert(
            f"**Sector Spreads V6** — {today.date()}\n"
            f"VIX: {current_vix:.1f} | Regime: {rscore:.3f} (inactive)\n"
            f"No new trades — regime below threshold.\n"
            f"Equity: ${state['equity']:.2f} | Open: {len(state['open_positions'])}"
        )

        _print_summary(state)
        save_state(state)
        return

    # ==================== RUN LGBM RANKING ====================
    rankings = run_lgbm_ranking(sc, close_df)
    if not rankings:
        log.warning("No rankings available. Skipping.")
        save_state(state)
        return

    ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    log.info(f"Rankings: {', '.join(f'{t}={s:.3f}' for t, s in ranked)}")

    # ==================== DETERMINE TRADES ====================
    entries = []
    max_concurrent = HIGH_VIX_TOP_K if high_vix_mode else (LOW_VIX_TOP_K + LOW_VIX_BOTTOM_K)
    max_pos = compute_position_size(state, max_concurrent)

    if max_pos < 30:
        log.warning(f"Position size too small (${max_pos:.0f}). Skipping entries.")
        _print_summary(state)
        save_state(state)
        return

    # --- BULL SIDE (always): top-ranked sectors ---
    top_k = HIGH_VIX_TOP_K if high_vix_mode else LOW_VIX_TOP_K
    long_picks = [t for t, _ in ranked[:top_k]]
    log.info(f"Bull picks (top {top_k}): {long_picks}")

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

        position = {
            'ticker': tk, 'mode': 'bull',
            'entry_date': str(today.date()), 'entry_price': S,
            'K1': K1, 'K2': K2,
            'cost': round(cost, 2), 'entry_cost_ps': round(entry_cost_ps, 6),
            'max_profit': round(max_profit, 2),
            'vix_at_entry': current_vix, 'vix_mode': 'high_vix' if high_vix_mode else 'low_vix',
            'regime_score': round(rscore, 4),
            'lgbm_score': round(rankings.get(tk, 0), 4),
        }
        state['open_positions'].append(position)
        state['total_trades'] += 1
        entries.append(position)

        log_trade({
            'action': 'OPEN', 'date': str(today.date()),
            'ticker': tk, 'mode': 'bull',
            'vix_mode': 'high_vix' if high_vix_mode else 'low_vix',
            'entry_price': S, 'strikes': f"{K1}/{K2}",
            'cost': round(cost, 2), 'max_profit': round(max_profit, 2),
            'vix': current_vix, 'regime_score': round(rscore, 4),
            'lgbm_score': round(rankings.get(tk, 0), 4),
            'equity': round(state['equity'], 2),
        })
        log.info(f"  ENTER {tk} BULL call {K1}/{K2}: cost ${cost:.2f}, max ${max_profit:.2f}")

    # --- BEAR SIDE (low-VIX only): bottom-ranked sectors ---
    short_picks = []
    if not high_vix_mode:
        short_picks = [t for t, _ in ranked[-LOW_VIX_BOTTOM_K:]]
        log.info(f"Bear picks (bottom {LOW_VIX_BOTTOM_K}): {short_picks}")

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

            position = {
                'ticker': tk, 'mode': 'bear',
                'entry_date': str(today.date()), 'entry_price': S,
                'K1': K1, 'K2': K2,
                'cost': round(cost, 2), 'entry_cost_ps': round(entry_cost_ps, 6),
                'max_profit': round(max_profit, 2),
                'vix_at_entry': current_vix, 'vix_mode': 'low_vix',
                'regime_score': round(rscore, 4),
                'lgbm_score': round(rankings.get(tk, 0), 4),
            }
            state['open_positions'].append(position)
            state['total_trades'] += 1
            entries.append(position)

            log_trade({
                'action': 'OPEN', 'date': str(today.date()),
                'ticker': tk, 'mode': 'bear', 'vix_mode': 'low_vix',
                'entry_price': S, 'strikes': f"{K1}/{K2}",
                'cost': round(cost, 2), 'max_profit': round(max_profit, 2),
                'vix': current_vix, 'regime_score': round(rscore, 4),
                'lgbm_score': round(rankings.get(tk, 0), 4),
                'equity': round(state['equity'], 2),
            })
            log.info(f"  ENTER {tk} BEAR put {K1}/{K2}: cost ${cost:.2f}, max ${max_profit:.2f}")

    # ==================== LOG RECOMMENDATION ====================
    reco = {
        'timestamp': run_ts,
        'date': str(today.date()),
        'vix': current_vix,
        'vix_mode': 'high_vix' if high_vix_mode else 'low_vix',
        'regime_score': round(rscore, 4),
        'regime_source': regime_source,
        'rankings': {t: round(s, 4) for t, s in ranked},
        'long_picks': long_picks,
        'short_picks': short_picks if not high_vix_mode else [],
        'entries': [{
            'ticker': e['ticker'], 'mode': e['mode'],
            'strikes': f"{e['K1']}/{e['K2']}", 'cost': e['cost'],
        } for e in entries],
        'n_entries': len(entries),
        'equity': round(state['equity'], 2),
        'total_trades': state['total_trades'],
        'win_rate': round(state['wins'] / max(state['wins'] + state['losses'], 1) * 100, 1),
    }
    log_recommendation(reco)

    # ==================== DISCORD ALERT ====================
    total = state['wins'] + state['losses']
    wr = state['wins'] / total * 100 if total > 0 else 0

    if entries:
        bull_entries = [e for e in entries if e['mode'] == 'bull']
        bear_entries = [e for e in entries if e['mode'] == 'bear']
        entry_lines = []
        for e in bull_entries:
            entry_lines.append(f"  BULL {e['ticker']} {e['K1']}/{e['K2']} (${e['cost']:.0f})")
        for e in bear_entries:
            entry_lines.append(f"  BEAR {e['ticker']} {e['K1']}/{e['K2']} (${e['cost']:.0f})")
        entry_text = '\n'.join(entry_lines)

        alert_msg = (
            f"**Sector Spreads V6 — Weekly Signal** ({today.date()})\n"
            f"VIX: {current_vix:.1f} | Mode: {mode_label} | Regime: {rscore:.3f}\n"
            f"**New trades:**\n{entry_text}\n"
            f"Equity: ${state['equity']:.2f} | WR: {wr:.0f}% ({total} trades)"
        )
    else:
        alert_msg = (
            f"**Sector Spreads V6** — {today.date()}\n"
            f"VIX: {current_vix:.1f} | Mode: {mode_label} | Regime: {rscore:.3f}\n"
            f"No new entries this week.\n"
            f"Equity: ${state['equity']:.2f} | Open: {len(state['open_positions'])}"
        )

    send_discord_alert(alert_msg)

    # ==================== SUMMARY ====================
    _print_summary(state)
    save_state(state)
    log.info("State saved. Done.")


def _print_summary(state):
    """Print portfolio summary."""
    total = state['wins'] + state['losses']
    wr = state['wins'] / total * 100 if total > 0 else 0
    long_total = state['long_wins'] + state['long_losses']
    short_total = state['short_wins'] + state['short_losses']
    long_wr = state['long_wins'] / long_total * 100 if long_total > 0 else 0
    short_wr = state['short_wins'] / short_total * 100 if short_total > 0 else 0

    log.info(f"\n=== V6 Summary ===")
    log.info(f"Equity: ${state['equity']:.2f} (P&L: ${state['total_pnl']:.2f})")
    log.info(f"Mode: {state.get('current_mode', 'N/A')}")
    log.info(f"Trades: {state['total_trades']} | W: {state['wins']} L: {state['losses']} | WR: {wr:.1f}%")
    if long_total > 0:
        log.info(f"  Long:  W: {state['long_wins']} L: {state['long_losses']} | WR: {long_wr:.1f}%")
    if short_total > 0:
        log.info(f"  Short: W: {state['short_wins']} L: {state['short_losses']} | WR: {short_wr:.1f}%")

    bulls = [p for p in state['open_positions'] if p['mode'] == 'bull']
    bears = [p for p in state['open_positions'] if p['mode'] == 'bear']
    log.info(f"Open: {len(bulls)} bull + {len(bears)} bear = {len(state['open_positions'])} total")
    for p in state['open_positions']:
        log.info(f"  {p['ticker']} {p['mode']} {p['K1']}/{p['K2']} "
                 f"(entry {p['entry_date']}, cost ${p['cost']:.2f})")


if __name__ == '__main__':
    try:
        run_weekly()
    except Exception as e:
        log.error(f"V6 Paper Engine CRASHED: {e}")
        log.error(traceback.format_exc())
        send_discord_alert(f"**V6 Paper Engine CRASHED**: {e}")
        sys.exit(1)
