#!/usr/bin/env python3
"""
Sector Earnings Standalone Paper Trading Engine
=================================================

Earnings-only sector rotation strategy using 14 LGBM features:
  - 8 earnings features (from constituent earnings event detection)
  - 6 earnings x momentum interaction features

Strategy: rank 11 sector ETFs, select top 3 for bull call spreads,
bottom 3 for bear put spreads. VIX-adaptive: VIX>=20 bulls only.

Key parameters:
  - 14 LGBM features (earnings-only, no legacy momentum)
  - Weekly Friday rebalance, DTE=21, 3% OTM
  - Adaptive spread width: max($3, 3% of strike)
  - $645 starting capital, $200 max per trade
  - Commission: $2.60 per spread RT, 15% entry haircut
  - Walk-forward: 25-week sliding window train, predict current week

Backtest reference: Sharpe 2.85, 5/5 adversarial gates, complementary to production.

Usage:
    python sector_earnings_standalone_paper.py              # normal daily run
    python sector_earnings_standalone_paper.py --dry-run    # simulate without state changes
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
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# ── Standardized tools ──
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from research.tools.options_pricer import (
    price_bull_call_spread,
    price_bear_put_spread,
    exit_spread_value,
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
STATE_PATH = STATE_DIR / 'sector_earnings_standalone_paper_state.json'
TRADE_LOG = LOG_DIR / 'sector_earnings_standalone_trades.jsonl'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'sector_earnings_standalone_paper.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ==================== DRY-RUN MODE ====================
DRY_RUN = '--dry-run' in sys.argv

# ==================== STRATEGY CONFIG ====================
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
EXTRA_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', 'SHY', 'HYG', 'GLD']
INITIAL_CAPITAL = 645.0
REBALANCE_INTERVAL = 5   # weekly (5 trading days)
SPREAD_PCT = 3.0          # spread width as % of strike
MIN_SPREAD_WIDTH = 3.0    # minimum $3 spread width (adaptive)
DTE = 21                  # 21 DTE
MONEYNESS_PCT = 3.0       # 3% OTM
MAX_POS_SIZE = 200        # max per trade
MAX_POS_PCT = 0.40        # max % of equity per trade
HAIRCUT = 0.15
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # $2.60 round trip

# VIX threshold — switches between modes
VIX_THRESHOLD = 20.0

# Low-VIX mode (pair trades): top-3 long + bottom-3 short
TOP_K = 3
BOTTOM_K = 3

# Walk-forward training window
WF_TRAIN_PERIODS = 25

# ==================== SECTOR CONSTITUENTS ====================
SECTOR_CONSTITUENTS = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM', 'ADBE', 'CSCO', 'ACN', 'ORCL', 'IBM',
            'INTC', 'AMD', 'QCOM', 'TXN', 'AMAT', 'INTU', 'NOW', 'ADI', 'LRCX', 'SNPS'],
    'XLF': ['BRK-B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SPGI', 'BLK',
            'C', 'AXP', 'SCHW', 'CB', 'PGR', 'ICE', 'CME', 'AON', 'MET'],
    'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'PXD', 'OXY',
            'WMB', 'HES', 'DVN', 'HAL', 'FANG', 'BKR', 'TRGP', 'KMI', 'OKE', 'CTRA'],
    'XLV': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'AMGN',
            'BMY', 'ISRG', 'SYK', 'VRTX', 'GILD', 'MDT', 'REGN', 'CI', 'ELV', 'ZTS'],
    'XLI': ['GE', 'CAT', 'HON', 'UNP', 'UPS', 'RTX', 'BA', 'DE', 'LMT', 'ADP',
            'MMM', 'FDX', 'GD', 'NSC', 'NOC', 'WM', 'CSX', 'ITW', 'EMR', 'ETN'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG',
            'F', 'GM', 'ORLY', 'AZO', 'ROST', 'DHI', 'LEN', 'MAR', 'HLT', 'YUM'],
    'XLP': ['PG', 'PEP', 'KO', 'COST', 'WMT', 'PM', 'MDLZ', 'MO', 'CL', 'KMB',
            'GIS', 'SJM', 'HSY', 'STZ', 'KHC', 'TAP', 'CAG', 'CPB', 'HRL'],
    'XLU': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'ED', 'WEC',
            'AWK', 'DTE', 'AEE', 'CMS', 'PPL', 'FE', 'ETR', 'CEG', 'PEG', 'EVRG'],
    'XLB': ['LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'NUE', 'VMC', 'MLM', 'DOW',
            'DD', 'PPG', 'IFF', 'CE', 'ALB', 'EMN', 'PKG', 'IP', 'CF', 'MOS'],
    'XLRE': ['PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'DLR', 'AVB',
             'EQR', 'ARE', 'VTR', 'MAA', 'UDR', 'KIM', 'REG', 'HST', 'CPT', 'BXP'],
    'XLC': ['META', 'GOOGL', 'GOOG', 'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS', 'CHTR',
            'EA', 'TTWO', 'WBD', 'OMC', 'FOXA', 'FOX', 'PARA', 'LYV', 'MTCH'],
}

# Approximate market cap weights for top constituents
CONSTITUENT_WEIGHTS = {}
for _sector, _tickers in SECTOR_CONSTITUENTS.items():
    weights = {}
    for i, tk in enumerate(_tickers):
        if i < 3:
            weights[tk] = 3.0
        elif i < 7:
            weights[tk] = 2.0
        else:
            weights[tk] = 1.0
    total = sum(weights.values())
    CONSTITUENT_WEIGHTS[_sector] = {tk: w / total for tk, w in weights.items()}

# ==================== FEATURE DEFINITIONS (14 total) ====================
EARNINGS_FEATURES = [
    'earnings_pct_reporting_2w',
    'earnings_avg_surprise',
    'earnings_post_drift',
    'earnings_days_to_heavy_week',
    'earnings_recent_surprise_quality',
    'earnings_beat_rate_1m',
    'earnings_vol_impact',
    'sector_earnings_cycle_position',
]

INTERACTION_FEATURES = [
    'earn_density_x_sector_mom_21d',
    'earn_surprise_x_sector_vol',
    'earn_drift_x_sector_ret_63d',
    'earn_beat_x_sector_sharpe',
    'earn_cycle_x_sector_ret_5d',
    'earn_vol_impact_x_dispersion',
]

FEAT_COLS = EARNINGS_FEATURES + INTERACTION_FEATURES  # 14 total


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
        'config_version': 'earnings_standalone_v1',
        'equity': INITIAL_CAPITAL,
        'open_positions': [],
        'closed_trades': [],
        'last_rebalance': None,
        'days_since_rebalance': 999,
        'total_trades': 0,
        'total_pnl': 0,
        'wins': 0,
        'losses': 0,
        'long_wins': 0,
        'long_losses': 0,
        'short_wins': 0,
        'short_losses': 0,
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

def download_sector_data():
    """Download sector ETF + macro data."""
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    log.info(f"Downloading {len(all_tickers)} sector/macro tickers...")
    raw = yf.download(all_tickers, start='2009-01-01', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    volume = raw['Volume'] if mi else raw

    for df in [close, high, low, volume]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

    close = close.ffill()
    high = high.ffill()
    low = low.ffill()
    volume = volume.ffill()

    rename_map = {'^VIX': 'VIX', '^VIX3M': 'VIX3M'}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)
    volume = volume.rename(columns=rename_map)

    vc = 'VIX' if 'VIX' in close.columns else ('^VIX' if '^VIX' in close.columns else None)
    if vc is None:
        raise ValueError("VIX data not available")
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index)

    log.info(f"Sector data: {len(ix)} days ({ix[0].date()} to {ix[-1].date()})")
    return close.loc[ix], sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix], volume


def download_constituent_data():
    """Download constituent stock data for earnings event detection."""
    import yfinance as yf

    all_constituents = set()
    for tickers in SECTOR_CONSTITUENTS.values():
        all_constituents.update(tickers)
    all_constituents = sorted(list(all_constituents))
    log.info(f"Downloading {len(all_constituents)} constituent stocks...")

    chunk_size = 50
    all_close = {}
    all_volume = {}

    for i in range(0, len(all_constituents), chunk_size):
        chunk = all_constituents[i:i + chunk_size]
        log.info(f"  Chunk {i // chunk_size + 1}: {len(chunk)} tickers...")
        try:
            raw = yf.download(chunk, start='2009-01-01', progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                c = raw['Close']
                v = raw['Volume']
                if isinstance(c.columns, pd.MultiIndex):
                    c.columns = c.columns.get_level_values(-1)
                    v.columns = v.columns.get_level_values(-1)
            else:
                c = raw[['Close']]
                v = raw[['Volume']]

            for col in c.columns:
                all_close[col] = c[col].ffill()
                all_volume[col] = v[col].ffill()
        except Exception as e:
            log.warning(f"    Chunk download failed: {e}")
            continue

    const_close = pd.DataFrame(all_close)
    const_volume = pd.DataFrame(all_volume)
    log.info(f"  Constituent data: {len(const_close)} days, {len(const_close.columns)} tickers")
    return const_close, const_volume


# ==================== EARNINGS EVENT DETECTION ====================

def detect_earnings_events(const_close, const_volume, vol_mult=3.0, gap_thresh=0.03):
    """
    Detect earnings-like events using volume spike + gap proxy.
    Volume >= vol_mult * rolling 20d median AND abs gap > gap_thresh.
    Returns dict: {ticker: pd.Series of event dates with surprise magnitude}
    """
    log.info(f"Detecting earnings events (vol_mult={vol_mult}, gap_thresh={gap_thresh})...")

    earnings_events = {}
    total_events = 0

    for ticker in const_close.columns:
        c = const_close[ticker].dropna()
        v = const_volume[ticker].dropna() if ticker in const_volume.columns else None

        if v is None or len(c) < 60 or len(v) < 60:
            continue

        common = c.index.intersection(v.index)
        c = c.loc[common]
        v = v.loc[common]

        rets = c.pct_change()
        vol_median = v.rolling(20, min_periods=10).median()
        vol_spike = v > (vol_median * vol_mult)
        gap_condition = rets.abs() > gap_thresh
        events = vol_spike & gap_condition
        event_dates = events[events].index

        if len(event_dates) > 0:
            surprise = rets.loc[event_dates]
            earnings_events[ticker] = surprise
            total_events += len(event_dates)

    log.info(f"  Detected {total_events} earnings events across {len(earnings_events)} tickers")
    return earnings_events


# ==================== EARNINGS FEATURE COMPUTATION ====================

def build_earnings_features(sector_ticker, idx, close_df, earnings_events, const_close):
    """Compute all 8 earnings features for a sector at a given date index."""
    f = {}
    constituents = SECTOR_CONSTITUENTS.get(sector_ticker, [])
    weights = CONSTITUENT_WEIGHTS.get(sector_ticker, {})

    if not constituents:
        return {feat: 0.0 for feat in EARNINGS_FEATURES}

    current_date = close_df.index[idx]
    lookback_10d = close_df.index[max(0, idx - 10):idx + 1]
    lookback_21d = close_df.index[max(0, idx - 21):idx + 1]
    lookback_63d = close_df.index[max(0, idx - 63):idx + 1]

    # 1. earnings_pct_reporting_2w — % of sector constituents reporting in trailing 2 weeks
    n_reported_2w = 0
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_10d)
            if len(recent) > 0:
                n_reported_2w += 1
    f['earnings_pct_reporting_2w'] = n_reported_2w / len(constituents)

    # 2. earnings_avg_surprise — average earnings surprise over trailing 63d
    surprises = []
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_63d)
            if len(recent) > 0:
                surprises.extend(events.loc[recent].values)
    f['earnings_avg_surprise'] = float(np.mean(surprises)) if surprises else 0.0

    # 3. earnings_post_drift — post-earnings drift magnitude (5-day PEAD)
    drifts = []
    for tk in constituents:
        if tk in earnings_events and tk in const_close.columns:
            events = earnings_events[tk]
            recent_events = events.index.intersection(lookback_63d)
            for edt in recent_events:
                if edt in const_close.index:
                    edt_idx = const_close.index.get_loc(edt)
                    end_idx = min(edt_idx + 5, len(const_close) - 1)
                    if end_idx > edt_idx and end_idx <= idx:
                        drift = float(const_close[tk].iloc[end_idx] /
                                      const_close[tk].iloc[edt_idx] - 1)
                        drifts.append(drift)
    f['earnings_post_drift'] = float(np.mean(drifts)) if drifts else 0.0

    # 4. earnings_days_to_heavy_week — days until next heavy reporting week (seasonal proxy)
    all_event_weeks = []
    for tk in constituents:
        if tk in earnings_events:
            for edt in earnings_events[tk].index:
                all_event_weeks.append(edt.isocalendar()[1])

    if all_event_weeks:
        week_counts = pd.Series(all_event_weeks).value_counts()
        peak_weeks = week_counts.head(4).index.tolist()
        current_week = current_date.isocalendar()[1]
        min_dist = 52
        for pw in peak_weeks:
            dist = (pw - current_week) % 52
            min_dist = min(min_dist, dist)
        f['earnings_days_to_heavy_week'] = min_dist / 26.0
    else:
        f['earnings_days_to_heavy_week'] = 0.5

    # 5. earnings_recent_surprise_quality — cap-weighted surprise quality
    weighted_beats = 0.0
    total_weight = 0.0
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_63d)
            if len(recent) > 0:
                w = weights.get(tk, 1.0 / len(constituents))
                avg_surprise = float(events.loc[recent].mean())
                quality = avg_surprise if avg_surprise > 0 else avg_surprise * 2
                weighted_beats += quality * w
                total_weight += w
    f['earnings_recent_surprise_quality'] = weighted_beats / (total_weight + 1e-10)

    # 6. earnings_beat_rate_1m — % of sector beating estimates in last month
    n_beat = 0
    n_reported = 0
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_21d)
            if len(recent) > 0:
                n_reported += 1
                if events.loc[recent].mean() > 0:
                    n_beat += 1
    f['earnings_beat_rate_1m'] = n_beat / max(n_reported, 1)

    # 7. earnings_vol_impact — avg absolute return on earnings day
    abs_impacts = []
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_63d)
            if len(recent) > 0:
                abs_impacts.extend(np.abs(events.loc[recent].values))
    f['earnings_vol_impact'] = float(np.mean(abs_impacts)) if abs_impacts else 0.0

    # 8. sector_earnings_cycle_position — where in reporting season (0-1)
    n_reported_63d = 0
    for tk in constituents:
        if tk in earnings_events:
            events = earnings_events[tk]
            recent = events.index.intersection(lookback_63d)
            if len(recent) > 0:
                n_reported_63d += 1
    expected_reporters = len(constituents)
    f['sector_earnings_cycle_position'] = min(n_reported_63d / expected_reporters, 1.0)

    return f


def compute_interaction_features(earn_feats, sector_ticker, idx, close_df):
    """Compute 6 interaction features between earnings and sector momentum."""
    f = {}
    sector_px = close_df[sector_ticker].iloc[:idx + 1].dropna() if sector_ticker in close_df.columns else None

    if sector_px is None or len(sector_px) < 63:
        return {feat: 0.0 for feat in INTERACTION_FEATURES}

    rets = sector_px.pct_change().dropna()

    # Sector momentum/vol for interactions
    ret_5d = float(sector_px.iloc[-1] / sector_px.iloc[-5] - 1) if len(sector_px) > 5 else 0.0
    ret_21d = float(sector_px.iloc[-1] / sector_px.iloc[-21] - 1) if len(sector_px) > 21 else 0.0
    ret_63d = float(sector_px.iloc[-1] / sector_px.iloc[-63] - 1) if len(sector_px) > 63 else 0.0
    vol_21d = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    r63 = rets.iloc[-63:]
    sharpe_63d = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0

    # Cross-sector dispersion
    sector_cols = [c for c in SECTORS if c in close_df.columns]
    if len(sector_cols) > 3:
        sector_rets = close_df[sector_cols].iloc[:idx + 1].pct_change()
        daily_disp = sector_rets.std(axis=1)
        dispersion = float(daily_disp.rolling(21).mean().iloc[-1]) if len(daily_disp) > 21 else 0.01
    else:
        dispersion = 0.01

    density = earn_feats.get('earnings_pct_reporting_2w', 0)
    surprise = earn_feats.get('earnings_avg_surprise', 0)
    drift = earn_feats.get('earnings_post_drift', 0)
    beat_rate = earn_feats.get('earnings_beat_rate_1m', 0)
    cycle = earn_feats.get('sector_earnings_cycle_position', 0.5)
    vol_impact = earn_feats.get('earnings_vol_impact', 0)

    f['earn_density_x_sector_mom_21d'] = density * ret_21d
    f['earn_surprise_x_sector_vol'] = surprise * vol_21d
    f['earn_drift_x_sector_ret_63d'] = drift * ret_63d
    f['earn_beat_x_sector_sharpe'] = beat_rate * sharpe_63d
    f['earn_cycle_x_sector_ret_5d'] = cycle * ret_5d
    f['earn_vol_impact_x_dispersion'] = vol_impact * dispersion

    return f


def compute_all_features(tk, idx, close_df, earnings_events, const_close):
    """Compute all 14 features for a single sector ETF at a given date index."""
    earn_feats = build_earnings_features(tk, idx, close_df, earnings_events, const_close)
    interaction_feats = compute_interaction_features(earn_feats, tk, idx, close_df)
    earn_feats.update(interaction_feats)
    return earn_feats


# ==================== LGBM RANKING (14 earnings features) ====================

def run_lgbm_ranking(sc, close_df, earnings_events, const_close):
    """Run walk-forward LGBM ranking using 14 earnings features.
    Uses 25-week sliding window for training, predicts current week.
    Returns dict of {ticker: score} where higher = better (long candidates)."""
    if not HAS_LGBM:
        log.warning("No LightGBM -- using simple momentum ranking fallback")
        rets_21d = sc.pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    # Build training data from trailing rebalance periods
    records = []
    all_dates = sc.index[-300:]  # ~60 weeks of data
    rebal_dates = all_dates[::REBALANCE_INTERVAL]

    # Use last WF_TRAIN_PERIODS rebalance dates for training
    train_dates = rebal_dates[-(WF_TRAIN_PERIODS + 1):-1]

    for dt in train_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 63:
            continue
        for tk in sc.columns:
            feats = compute_all_features(tk, idx, close_df, earnings_events, const_close)
            if not feats:
                continue
            # Forward return over DTE period
            fi = min(idx + DTE, len(sc) - 1)
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

    # Log feature importances
    importances = dict(zip(FEAT_COLS, m.feature_importances_))
    top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    log.info(f"Top features: {', '.join(f'{n}={v}' for n, v in top_feats)}")

    # Predict current rankings
    current_idx = len(sc) - 1
    current_feats = {}
    for tk in sc.columns:
        feats = compute_all_features(tk, current_idx, close_df, earnings_events, const_close)
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
    """Price a bull call spread with 3% OTM moneyness and adaptive width.
    K1 = S * 1.03, K2 = K1 + max($3, K1 * 0.03).
    Returns (cost_dollars, max_profit_dollars, K1, K2, entry_cost_ps)."""
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
    """Price a bear put spread with 3% OTM moneyness and adaptive width.
    K1 = S * 0.97 (long put), K2 = K1 - max($3, K1 * 0.03).
    Returns (cost_dollars, max_profit_dollars, K1, K2, entry_cost_ps)."""
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

    # price_bear_put_spread expects K1 < K2 (K1=lower, K2=higher)
    entry_cost_ps, max_profit_ps = price_bear_put_spread(
        S=S, K1=K2, K2=K1, dte=dte, atr=atr, vix=vix_val, haircut=HAIRCUT
    )

    cost_dollars = entry_cost_ps * 100 + SPREAD_COMM
    max_profit_dollars = max_profit_ps * 100 - SPREAD_COMM

    return cost_dollars, max_profit_dollars, K2, K1, entry_cost_ps


# ==================== POSITION EXIT (hold to expiry) ====================

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


# ==================== REBALANCE DAY CHECK ====================

def is_friday(dt):
    """Check if date is a Friday (for weekly Friday rebalance)."""
    return dt.weekday() == 4


# ==================== MAIN DAILY RUN ====================

def run_daily():
    """Main daily run. Called at 4:30 PM ET on weekdays."""
    state = load_state()
    if DRY_RUN:
        log.info("=" * 60)
        log.info("  DRY-RUN MODE -- no state changes will be persisted")
        log.info("=" * 60)

    log.info("=== Sector Earnings Standalone Paper Engine ===")
    log.info(f"Equity: ${state['equity']:.2f} | Open spreads: {len(state['open_positions'])} | "
             f"Trades: {state['total_trades']} | W/L: {state['wins']}/{state['losses']}")

    # Download sector ETF data
    try:
        close_df, sc, sh, sl, spy, vix, volume = download_sector_data()
    except Exception as e:
        log.error(f"Sector data download failed: {e}")
        return

    # Download constituent data for earnings detection
    try:
        const_close, const_volume = download_constituent_data()
    except Exception as e:
        log.error(f"Constituent data download failed: {e}")
        return

    # Detect earnings events
    earnings_events = detect_earnings_events(const_close, const_volume)

    today = sc.index[-1]
    current_vix = float(vix.iloc[-1])
    high_vix_mode = current_vix >= VIX_THRESHOLD

    mode_str = 'HIGH-VIX (bulls only)' if high_vix_mode else 'LOW-VIX (bulls + bears)'
    log.info(f"Date: {today.date()} | VIX: {current_vix:.1f} | MODE: {mode_str}")

    # ==================== CHECK EXISTING POSITIONS ====================
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

    state['current_mode'] = 'high_vix' if high_vix_mode else 'low_vix'

    # ==================== CHECK FOR REBALANCE (Friday only) ====================
    state['days_since_rebalance'] = state.get('days_since_rebalance', 999) + 1

    if not is_friday(today):
        log.info(f"Not Friday ({today.strftime('%A')}). Checking positions only.")
        _print_summary(state)
        save_state(state)
        return

    log.info("=== REBALANCE DAY (Friday) ===")
    state['days_since_rebalance'] = 0
    state['last_rebalance'] = str(today.date())

    # Run LGBM ranking with earnings features
    rankings = run_lgbm_ranking(sc, close_df, earnings_events, const_close)
    if not rankings:
        log.warning("No rankings available. Skipping rebalance.")
        save_state(state)
        return

    ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    long_picks = [t for t, _ in ranked[:TOP_K]]
    short_picks = [t for t, _ in ranked[-BOTTOM_K:]]

    log.info(f"Rankings: {', '.join(f'{t}={s:.3f}' for t, s in ranked)}")
    log.info(f"LONG picks (top {TOP_K}): {long_picks}")
    if not high_vix_mode:
        log.info(f"SHORT picks (bottom {BOTTOM_K}): {short_picks}")

    max_concurrent = TOP_K + (BOTTOM_K if not high_vix_mode else 0)
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

        # Cost/Width filter: reject if entry cost > 50% of spread width
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
            'vix_mode': 'high_vix' if high_vix_mode else 'low_vix',
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
            'vix_mode': 'high_vix' if high_vix_mode else 'low_vix',
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

    # --- SHORT SIDE: Bear put spreads on bottom-ranked (low-VIX only) ---
    if not high_vix_mode:
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

            # Cost/Width filter
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
        log.info("No entries this rebalance")

    _print_summary(state)
    save_state(state)
    log.info("State saved. Done.")


def _print_summary(state):
    """Print summary of current portfolio state."""
    total = state['wins'] + state['losses']
    wr = state['wins'] / total * 100 if total > 0 else 0
    long_total = state['long_wins'] + state['long_losses']
    short_total = state['short_wins'] + state['short_losses']
    long_wr = state['long_wins'] / long_total * 100 if long_total > 0 else 0
    short_wr = state['short_wins'] / short_total * 100 if short_total > 0 else 0

    log.info(f"\n=== Summary ===")
    log.info(f"Equity: ${state['equity']:.2f} (P&L: ${state['total_pnl']:.2f})")
    log.info(f"Mode: {state.get('current_mode', 'N/A')}")
    log.info(f"Trades: {state['total_trades']} | W: {state['wins']} L: {state['losses']} | WR: {wr:.1f}%")
    log.info(f"  Long side:  W: {state['long_wins']} L: {state['long_losses']} | WR: {long_wr:.1f}%")
    log.info(f"  Short side: W: {state['short_wins']} L: {state['short_losses']} | WR: {short_wr:.1f}%")

    bulls = [p for p in state['open_positions'] if p['mode'] == 'bull']
    bears = [p for p in state['open_positions'] if p['mode'] == 'bear']
    log.info(f"Open positions: {len(bulls)} bull + {len(bears)} bear = {len(state['open_positions'])} total")
    for p in state['open_positions']:
        log.info(f"  {p['ticker']} {p['mode']} {p['K1']}/{p['K2']} "
                 f"(entry {p['entry_date']}, cost ${p['cost']:.2f}, "
                 f"LGBM {p.get('lgbm_score', 'N/A')}, via {p.get('vix_mode', 'N/A')})")


if __name__ == '__main__':
    run_daily()
