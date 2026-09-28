#!/usr/bin/env python3
"""
Sector Bull Call Spreads Paper Trading Engine (Production v6)
==============================================================

Paper engine for our #1 validated strategy — upgraded to production v6 config:
- Bull call spreads on top LGBM-ranked sectors when GRU regime_score > 0.4
- Hold to expiry (21 DTE), NO early exit or take-profit
- 21 features: 18 legacy + 3 cross-asset (sector_spy_beta_63d, sector_relative_vol_21d,
  cross_sector_dispersion)
- NO bear puts (net negative PnL at expiry — finding #69)
- NO VIX 25-30 skip (skipping hurts Sharpe — finding #70)
- NO flow features (HURT performance — finding #42)
- Low-vol overlay: when regime_score < 0.4, park in SPY/TLT/GLD risk parity
- Position size scales with equity growth
- HC #750: Multi-signal confluence (min 2 signals required)
- HC #750 R4: Audit trail — logs which signals confirmed each entry

Backtest (v6): Sharpe 2.44, Sortino 10.46, PF 5.05, WR 54.6%, MDD -8.8%
    v6 changes: 2% OTM moneyness (+19% Sharpe, -42% MaxDD). Cross-validated exp 173+174.
    v5 changes: weekly rebalance (5d), top 2 sectors (cross-validated, MLflow exp 168)

Runs once daily at market close (via PM2 cron).
Checks if today is a rebalance day (weekly).
If rebalance: downloads data, checks regime, runs LGBM, checks signals, enters paper trades.
If not rebalance: checks existing positions for expiry.
When regime_score < 0.4: allocates to risk parity overlay instead.

State persisted to JSON between runs.

Usage:
    python sector_spreads_paper.py              # normal daily run
    python sector_spreads_paper.py --dry-run    # simulate without state changes
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
    exit_spread_value,
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
STATE_PATH = STATE_DIR / 'sector_spreads_paper_state.json'
TRADE_LOG = LOG_DIR / 'sector_spreads_trades.jsonl'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'sector_spreads_paper.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ==================== DRY-RUN MODE ====================
DRY_RUN = '--dry-run' in sys.argv

# ==================== STRATEGY CONFIG (Production v4) ====================
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
EXTRA_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', 'SHY', 'HYG', 'GLD']
INITIAL_CAPITAL = 645.0
TOP_K = 2  # v5: concentrated on top 2 (was 3 in v4). Cross-validated: +108% Sharpe
REBALANCE_INTERVAL = 5  # v5: weekly (was 10/biweekly in v4). Cross-validated: dominant improvement
SPREAD_PCT = 3.0  # spread width as % of stock price
DTE = 21  # v4: 21 DTE, NOT 30
MONEYNESS_PCT = 2.0  # v6: 2% OTM. Cross-validated: +19% Sharpe, -42% MaxDD (exp 173+174)
MAX_POS_SIZE = 200  # max per trade
MAX_POS_PCT = 0.40  # max % of equity per trade
HAIRCUT = 0.15
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # $2.60 round trip

# Regime thresholds
REGIME_BULL_THRESHOLD = 0.4

# Risk parity overlay allocation (when regime_score < 0.4)
RISK_PARITY_ALLOC = {'SPY': 0.333, 'TLT': 0.333, 'GLD': 0.334}

# Regime predictions file
REGIME_FILE = BASE / 'output' / 'regime_detector_v1' / 'regime_predictions_v1.npz'

# ==================== FEATURES (v4: 21 total) ====================
LEGACY_FEATURES = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y', 'up_capture',
    'trend_r2_63d', 'trend_slope_63d',
]

CROSS_ASSET_FEATURES = [
    'sector_spy_beta_63d',
    'sector_relative_vol_21d',
    'cross_sector_dispersion',
]

FEAT_COLS = LEGACY_FEATURES + CROSS_ASSET_FEATURES  # 21 total


# ==================== REGIME MODEL ====================

def load_regime_predictions():
    """Load GRU regime predictions from pre-computed NPZ file.
    Returns a date-indexed pd.Series of regime scores, or None if unavailable."""
    if not REGIME_FILE.exists():
        log.warning(f"Regime file not found — will use VIX-based proxy")
        return None

    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data['dates'])
    scores = data['regime_scores']
    regime_series = pd.Series(scores, index=dates, name='regime_score')
    regime_series = regime_series[~regime_series.index.duplicated(keep='last')]
    log.info(f"Regime predictions loaded: {len(regime_series)} days "
             f"({regime_series.index[0].date()} to {regime_series.index[-1].date()})")
    log.info(f"  Mean score: {regime_series.mean():.3f}, "
             f"Days >0.4: {(regime_series > REGIME_BULL_THRESHOLD).sum()}")
    return regime_series


def get_regime_score(regime_series, dt):
    """Get regime score at a given date, with nearest-date fallback.
    Returns VIX-based proxy if regime_series is None."""
    if regime_series is None:
        return None  # caller handles VIX proxy
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method='ffill')]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return None


def vix_regime_proxy(vix_val):
    """Fallback: map VIX to a pseudo regime_score.
    VIX >= 25 -> ~0.7, VIX 20-25 -> ~0.5, VIX < 20 -> ~0.3"""
    if vix_val >= 25:
        return 0.7
    elif vix_val >= 20:
        return 0.4 + (vix_val - 20) * 0.06  # 0.4 at 20, 0.7 at 25
    else:
        return 0.15 + vix_val * 0.0125  # 0.15 at 0, ~0.4 at 20


# ==================== STATE MANAGEMENT ====================

def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        # Migrate v3 state to v4 if needed
        if 'risk_parity_positions' not in state:
            state['risk_parity_positions'] = []
        if 'config_version' not in state:
            state['config_version'] = 'v4'
        return state
    return {
        'config_version': 'v4',
        'equity': INITIAL_CAPITAL,
        'open_positions': [],
        'risk_parity_positions': [],
        'closed_trades': [],
        'last_rebalance': None,
        'days_since_rebalance': 999,
        'total_trades': 0,
        'total_pnl': 0,
        'wins': 0,
        'losses': 0,
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
        log.info(f"[DRY-RUN] Trade NOT logged: {trade_record.get('action')} {trade_record.get('ticker', 'N/A')}")
        return
    with open(TRADE_LOG, 'a') as f:
        f.write(json.dumps(trade_record, default=str) + '\n')


# ==================== DATA DOWNLOAD ====================

def download_data():
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    raw = yf.download(all_tickers, start='2024-01-01', progress=False)
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


# ==================== FEATURE ENGINEERING (v4: 21 features) ====================

def compute_legacy_features(px):
    """Compute the 18 legacy quality-momentum features for a single sector ETF."""
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
    up_days = rets[rets > 0]
    f['up_capture'] = float(up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)) if len(up_days) > 10 else 1.0

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


def compute_cross_asset_features(sector_ticker, close_df):
    """Compute the 3 validated cross-asset features (v4 addition)."""
    f = {}

    spy = close_df['SPY'].dropna() if 'SPY' in close_df.columns else None
    sector_px = close_df[sector_ticker].dropna() if sector_ticker in close_df.columns else None

    if spy is None or len(spy) < 63:
        return {k: 0.0 for k in CROSS_ASSET_FEATURES}

    spy_ret = spy.pct_change().dropna()

    # 1. Sector-SPY beta 63d
    if sector_px is not None and len(sector_px) > 63:
        sec_ret = sector_px.pct_change().dropna()
        common = spy_ret.index.intersection(sec_ret.index)
        if len(common) > 63:
            sr = sec_ret.loc[common].iloc[-63:]
            mr = spy_ret.loc[common].iloc[-63:]
            cov = np.cov(sr.values, mr.values)
            beta = cov[0, 1] / (cov[1, 1] + 1e-10)
            f['sector_spy_beta_63d'] = float(beta)
        else:
            f['sector_spy_beta_63d'] = 1.0
    else:
        f['sector_spy_beta_63d'] = 1.0

    # 2. Sector relative vol 21d
    if sector_px is not None and len(sector_px) > 21:
        sec_ret = sector_px.pct_change().dropna()
        if len(sec_ret) > 21 and len(spy_ret) > 21:
            sec_vol = sec_ret.iloc[-21:].std()
            spy_vol = spy_ret.iloc[-21:].std()
            f['sector_relative_vol_21d'] = float(sec_vol / (spy_vol + 1e-10))
        else:
            f['sector_relative_vol_21d'] = 1.0
    else:
        f['sector_relative_vol_21d'] = 1.0

    # 3. Cross-sector dispersion (rolling 21d stdev of cross-sectional sector returns)
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


def compute_all_features(tk, sc, close_df):
    """Compute all 21 features for a single sector ETF."""
    px = sc[tk].dropna()
    legacy = compute_legacy_features(px)
    if legacy is None:
        return None
    cross_asset = compute_cross_asset_features(tk, close_df)
    legacy.update(cross_asset)
    return legacy


# ==================== LGBM RANKING (v4: 21 features) ====================

def run_lgbm_ranking(sc, close_df):
    """Run walk-forward LGBM ranking using trailing data with 21 features."""
    if not HAS_LGBM:
        log.warning("No LightGBM — using simple momentum ranking")
        rets_21d = sc.pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    # Build training data from trailing rebalance periods
    records = []
    all_dates = sc.index[-300:]  # ~14 months
    rebal_dates = all_dates[::REBALANCE_INTERVAL]

    for dt in rebal_dates[:-1]:  # exclude last (current) for training
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx + 1].dropna()
            feats = compute_legacy_features(px)
            if not feats:
                continue
            # Cross-asset features using data up to this point
            ca = compute_cross_asset_features(tk, close_df.iloc[:idx + 1])
            feats.update(ca)
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

    # Train on all historical, predict current
    X_train = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
    y_train = df['rank_label'].values.astype(np.float32)

    m = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1
    )
    m.fit(X_train, y_train)

    # Predict current rankings using all 21 features
    current_feats = {}
    for tk in sc.columns:
        feats = compute_all_features(tk, sc, close_df)
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


# ==================== CONFLUENCE CHECK ====================

def confluence_check(tk, sc, spy, vix):
    """HC #750: Multi-signal confluence for bull call spreads. Min 2 independent signals required.
    Returns (passes, n_signals, signal_names).
    v4: bull-only, no bear mode."""
    signals = []
    px = sc[tk].dropna()
    if len(px) < 63:
        return False, 0, []

    # Signal 1: Momentum (21d positive)
    ret_21d = float(px.iloc[-1] / px.iloc[-21] - 1)
    if ret_21d > 0:
        signals.append('mom_21d_pos')

    # Signal 2: Trend (price vs 50-SMA)
    sma50 = px.rolling(50).mean()
    if not pd.isna(sma50.iloc[-1]) and px.iloc[-1] > sma50.iloc[-1]:
        signals.append('above_sma50')

    # Signal 3: Relative strength vs SPY
    if len(spy) > 21:
        common = px.index.intersection(spy.index)
        if len(common) > 21:
            rel = px.loc[common] / spy.loc[common]
            rel_ret = float(rel.iloc[-1] / rel.iloc[-21] - 1)
            if rel_ret > 0:
                signals.append('rel_str_pos')

    # Signal 4: RSI check (not overbought)
    rets = px.pct_change().dropna()
    if len(rets) > 14:
        gains = rets.clip(lower=0).rolling(14).mean()
        losses = (-rets.clip(upper=0)).rolling(14).mean()
        rs = gains / (losses + 1e-10)
        rsi = 100 - 100 / (1 + rs)
        r = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        if r < 80:
            signals.append('rsi_ok')

    # Signal 5: Momentum acceleration
    if len(px) > 63:
        ret_63d = float(px.iloc[-1] / px.iloc[-63] - 1)
        accel = ret_21d - ret_63d / 3
        if accel > 0:
            signals.append('mom_accel_pos')

    return len(signals) >= 2, len(signals), signals


# ==================== SPREAD PRICING (using standardized pricer) ====================

def price_spread(S, spread_pct, dte, sh_tk, sl_tk, sc_tk, vix_val):
    """Price a bull call spread using the standardized options_pricer.
    Returns (cost_dollars, max_profit_dollars, K1, K2)."""
    # v6: 2% OTM moneyness shift
    K1 = round(S * (1 + MONEYNESS_PCT / 100))
    K2 = round(K1 * (1 + spread_pct / 100))

    # Compute ATR from high/low/close data
    if sh_tk is not None and len(sh_tk) >= 14:
        atr = compute_atr(sh_tk, sl_tk, sc_tk, period=14)
    else:
        # Fallback: estimate ATR from volatility
        atr = S * 0.015  # ~1.5% daily range

    entry_cost_ps, max_profit_ps = price_bull_call_spread(
        S=S, K1=K1, K2=K2, dte=dte, atr=atr, vix=vix_val,
        haircut=HAIRCUT
    )

    cost_dollars = entry_cost_ps * 100 + SPREAD_COMM
    max_profit_dollars = max_profit_ps * 100 - SPREAD_COMM

    return cost_dollars, max_profit_dollars, K1, K2, entry_cost_ps


# ==================== POSITION EXIT (v4: hold to expiry only) ====================

def check_position_exit(pos, current_price, days_held):
    """v4: Hold to expiry ONLY (21 DTE). No early exit, no take-profit.
    Returns (should_exit, pnl, reason)."""
    S = current_price
    K1, K2 = pos['K1'], pos['K2']
    # v3->v4 migration: old positions may lack entry_cost_ps
    entry_cost_ps = pos.get('entry_cost_ps')
    if entry_cost_ps is None:
        # Reconstruct from total cost: cost_dollars = entry_cost_ps * 100 + SPREAD_COMM
        entry_cost_ps = (pos['cost'] - SPREAD_COMM) / 100.0

    if days_held >= DTE:
        # At expiry: intrinsic value only
        intrinsic = exit_spread_value(S, K1, K2, remaining_dte=0, original_dte=DTE,
                                       atr=0, vix=0)  # atr/vix unused at expiry
        pnl = (intrinsic - entry_cost_ps) * 100 - SPREAD_COMM
        return True, pnl, 'expiry'

    # Not at expiry yet — hold
    return False, 0, 'hold'


# ==================== RISK PARITY OVERLAY ====================

def enter_risk_parity(state, close_df, today):
    """When regime_score < 0.4, park capital in SPY/TLT/GLD risk parity.
    Returns updated state."""
    available = state['equity'] - sum(p['cost'] for p in state['open_positions'])
    if available < 50:
        log.info("Insufficient capital for risk parity overlay")
        return state

    # Check if already in risk parity
    if state['risk_parity_positions']:
        log.info("Risk parity positions already open — holding")
        return state

    for ticker, weight in RISK_PARITY_ALLOC.items():
        if ticker not in close_df.columns:
            log.warning(f"  Risk parity ticker {ticker} not in data — skipping")
            continue
        price = float(close_df[ticker].dropna().iloc[-1])
        alloc = available * weight
        shares = int(alloc / price)  # fractional not supported
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

        trade_record = {
            'action': 'OPEN_RP',
            'date': str(today.date()),
            'ticker': ticker,
            'type': 'risk_parity',
            'shares': shares,
            'entry_price': price,
            'cost': round(shares * price, 2),
        }
        log_trade(trade_record)

    return state


def exit_risk_parity(state, close_df, today, reason='regime_shift'):
    """Exit risk parity positions (when regime shifts back above 0.4)."""
    if not state['risk_parity_positions']:
        return state

    for pos in state['risk_parity_positions']:
        tk = pos['ticker']
        if tk not in close_df.columns:
            continue
        current_price = float(close_df[tk].dropna().iloc[-1])
        pnl = (current_price - pos['entry_price']) * pos['shares']
        state['equity'] += pnl
        state['total_pnl'] += pnl

        log.info(f"  EXIT RISK PARITY: {pos['shares']} shares {tk} @ ${current_price:.2f} "
                 f"(PnL: ${pnl:.2f}, reason: {reason})")

        trade_record = {
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
        }
        log_trade(trade_record)

    state['risk_parity_positions'] = []
    return state


# ==================== POSITION SIZING (v4: scales with equity) ====================

def compute_position_size(state):
    """v4: Position size scales with equity growth.
    Base max is $200, scales proportionally with equity / initial_capital."""
    equity_ratio = state['equity'] / INITIAL_CAPITAL
    scaled_max = MAX_POS_SIZE * equity_ratio
    # Cap at 40% of current equity
    equity_cap = state['equity'] * MAX_POS_PCT
    return min(scaled_max, equity_cap, state['equity'] / 3)


# ==================== MAIN DAILY RUN ====================

def run_daily():
    """Main daily run. Called once per trading day."""
    state = load_state()
    if DRY_RUN:
        log.info("=" * 60)
        log.info("  DRY-RUN MODE — no state changes will be persisted")
        log.info("=" * 60)

    log.info(f"=== Sector Spreads Paper Engine (v4) ===")
    log.info(f"Equity: ${state['equity']:.2f} | Open spreads: {len(state['open_positions'])} | "
             f"Open RP: {len(state['risk_parity_positions'])} | "
             f"Trades: {state['total_trades']} | W/L: {state['wins']}/{state['losses']}")

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

    # Get regime score
    rscore = get_regime_score(regime_series, today)
    if rscore is None:
        rscore = vix_regime_proxy(current_vix)
        log.info(f"Date: {today.date()} | VIX: {current_vix:.1f} | "
                 f"Regime score: {rscore:.3f} (VIX proxy)")
    else:
        log.info(f"Date: {today.date()} | VIX: {current_vix:.1f} | "
                 f"Regime score: {rscore:.3f} (GRU model)")

    regime_active = rscore > REGIME_BULL_THRESHOLD
    log.info(f"Regime gate: {'ACTIVE (bull spreads allowed)' if regime_active else 'INACTIVE (low-vol overlay)'}")

    # ==================== CHECK EXISTING SPREAD POSITIONS ====================
    positions_to_close = []
    for i, pos in enumerate(state['open_positions']):
        entry_date = pd.Timestamp(pos['entry_date'])
        days_held = len(sc.index[(sc.index > entry_date) & (sc.index <= today)])

        if pos['ticker'] not in sc.columns:
            continue

        current_price = float(sc[pos['ticker']].iloc[-1])
        should_exit, pnl, reason = check_position_exit(pos, current_price, days_held)

        if should_exit:
            positions_to_close.append((i, pnl, reason, days_held))
            log.info(f"  EXIT {pos['ticker']} bull spread: PnL ${pnl:.2f} ({reason}, held {days_held}d)")

    # Close positions (reverse order to preserve indices)
    for i, pnl, reason, days_held in reversed(positions_to_close):
        pos = state['open_positions'].pop(i)
        state['equity'] += pnl
        state['total_pnl'] += pnl
        if pnl > 0:
            state['wins'] += 1
        else:
            state['losses'] += 1

        trade_record = {
            'action': 'CLOSE',
            'date': str(today.date()),
            'ticker': pos['ticker'],
            'mode': 'bull',
            'entry_date': pos['entry_date'],
            'days_held': days_held,
            'cost': pos['cost'],
            'pnl': round(pnl, 2),
            'exit_reason': reason,
            'equity_after': round(state['equity'], 2),
            'signals_at_entry': pos.get('signals', []),
            'regime_score_at_entry': pos.get('regime_score', None),
        }
        state['closed_trades'].append(trade_record)
        log_trade(trade_record)

    # ==================== HANDLE REGIME TRANSITIONS ====================
    if regime_active and state['risk_parity_positions']:
        # Regime shifted back to active — exit risk parity
        log.info("Regime shifted to ACTIVE — exiting risk parity overlay")
        state = exit_risk_parity(state, close_df, today, reason='regime_shift_active')

    # ==================== CHECK FOR REBALANCE ====================
    state['days_since_rebalance'] = state.get('days_since_rebalance', 999) + 1

    if state['days_since_rebalance'] < REBALANCE_INTERVAL:
        log.info(f"Not rebalance day ({state['days_since_rebalance']}/{REBALANCE_INTERVAL}). "
                 f"Checking positions only.")
        save_state(state)
        return

    log.info(f"=== REBALANCE DAY ===")
    state['days_since_rebalance'] = 0
    state['last_rebalance'] = str(today.date())

    # ==================== LOW-VOL REGIME: RISK PARITY OVERLAY ====================
    if not regime_active:
        log.info(f"Regime score {rscore:.3f} < {REGIME_BULL_THRESHOLD} — "
                 f"LOW-VOL REGIME. Entering risk parity overlay.")
        state = enter_risk_parity(state, close_df, today)
        save_state(state)
        return

    # ==================== RUN LGBM RANKING (21 features) ====================
    rankings = run_lgbm_ranking(sc, close_df)
    if not rankings:
        log.warning("No rankings available. Skipping rebalance.")
        save_state(state)
        return

    # v4: Bull only — pick top-ranked sectors
    ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    picks = [t for t, _ in ranked[:TOP_K]]
    log.info(f"LGBM bull picks (top {TOP_K}): {picks}")
    log.info(f"Rankings: {', '.join(f'{t}={s:.3f}' for t, s in ranked)}")

    # ==================== ENTER NEW POSITIONS ====================
    max_pos = compute_position_size(state)
    if max_pos < 30:
        log.warning(f"Position size too small (${max_pos:.0f}). Skipping entries.")
        save_state(state)
        return

    entries = 0
    for tk in picks:
        if entries >= 3:
            break
        if tk not in sc.columns:
            continue

        # Check if already holding this ticker
        if any(p['ticker'] == tk for p in state['open_positions']):
            log.info(f"  SKIP {tk}: already holding")
            continue

        # HC #750: Confluence check (bull only)
        passes, n_signals, signal_names = confluence_check(tk, sc, spy, vix)
        if not passes:
            log.info(f"  SKIP {tk}: confluence FAIL ({n_signals} signals: {signal_names})")
            continue

        log.info(f"  PASS {tk}: confluence OK ({n_signals} signals: {signal_names})")

        # Price spread using standardized pricer
        S = float(sc[tk].iloc[-1])
        sh_tk = sh[tk] if tk in sh.columns else None
        sl_tk = sl[tk] if tk in sl.columns else None
        sc_tk = sc[tk] if tk in sc.columns else None
        cost, max_profit, K1, K2, entry_cost_ps = price_spread(
            S, SPREAD_PCT, DTE, sh_tk, sl_tk, sc_tk, current_vix
        )

        if cost <= 0 or cost > max_pos or cost > state['equity'] * MAX_POS_PCT:
            log.info(f"  SKIP {tk}: cost ${cost:.2f} exceeds limits "
                     f"(max_pos=${max_pos:.0f}, equity_cap=${state['equity'] * MAX_POS_PCT:.0f})")
            continue

        # ENTER paper trade
        position = {
            'ticker': tk,
            'mode': 'bull',
            'entry_date': str(today.date()),
            'entry_price': S,
            'K1': K1,
            'K2': K2,
            'cost': round(cost, 2),
            'entry_cost_ps': round(entry_cost_ps, 6),  # per-share for expiry calc
            'max_profit': round(max_profit, 2),
            'vix_at_entry': current_vix,
            'regime_score': round(rscore, 4),
            'lgbm_score': round(rankings.get(tk, 0), 4),
            'signals': signal_names,
            'n_signals': n_signals,
        }
        state['open_positions'].append(position)
        state['total_trades'] += 1
        entries += 1

        trade_record = {
            'action': 'OPEN',
            'date': str(today.date()),
            'ticker': tk,
            'mode': 'bull',
            'entry_price': S,
            'strikes': f"{K1}/{K2}",
            'cost': round(cost, 2),
            'max_profit': round(max_profit, 2),
            'vix': current_vix,
            'regime_score': round(rscore, 4),
            'lgbm_score': round(rankings.get(tk, 0), 4),
            'signals': signal_names,
            'equity': round(state['equity'], 2),
        }
        log_trade(trade_record)
        log.info(f"  ENTER {tk} bull spread {K1}/{K2}: cost ${cost:.2f}, "
                 f"max profit ${max_profit:.2f}, VIX {current_vix:.1f}, "
                 f"regime {rscore:.3f}")

    if entries == 0:
        log.info(f"No entries this rebalance (VIX={current_vix:.1f}, regime={rscore:.3f})")

    # Summary
    wr = state['wins'] / (state['wins'] + state['losses']) * 100 if (state['wins'] + state['losses']) > 0 else 0
    log.info(f"\n=== Summary ===")
    log.info(f"Equity: ${state['equity']:.2f} (P&L: ${state['total_pnl']:.2f})")
    log.info(f"Trades: {state['total_trades']} | W: {state['wins']} L: {state['losses']} | WR: {wr:.1f}%")
    log.info(f"Open spread positions: {len(state['open_positions'])}")
    for p in state['open_positions']:
        log.info(f"  {p['ticker']} bull {p['K1']}/{p['K2']} (entry {p['entry_date']}, "
                 f"cost ${p['cost']:.2f}, regime {p.get('regime_score', 'N/A')}, "
                 f"signals: {p['signals']})")
    if state['risk_parity_positions']:
        log.info(f"Risk parity positions: {len(state['risk_parity_positions'])}")
        for p in state['risk_parity_positions']:
            log.info(f"  {p['ticker']} {p['shares']} shares @ ${p['entry_price']:.2f}")

    save_state(state)
    log.info("State saved. Done.")


if __name__ == '__main__':
    run_daily()
