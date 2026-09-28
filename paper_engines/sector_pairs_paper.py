#!/usr/bin/env python3
"""
Sector Pair Trades Paper Trading Engine (HC #751)
===================================================

Paper engine for validated sector pair trade strategy:
- Long bull call spread on top-ranked sectors + Short bear put spread on bottom-ranked sectors
- VIX < 20 threshold (NOT GRU regime score)
- LGBM ranking with 21 features (18 legacy momentum/quality + 3 cross-asset)
- Top 3 long + Bottom 3 short from 11 sector ETFs
- Biweekly (10 trading day) rebalance when VIX < 20
- ATM, 3% wide, DTE=21, hold to expiry
- $200 max per trade, $645 starting capital, scale with equity
- Commission: $2.60 per spread round trip

Backtest: Sharpe 2.41, passed 5/5 adversarial gates.

Usage:
    python sector_pairs_paper.py              # normal daily run
    python sector_pairs_paper.py --dry-run    # simulate without state changes
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
STATE_PATH = STATE_DIR / 'sector_pairs_paper_state.json'
TRADE_LOG = LOG_DIR / 'sector_pairs_trades.jsonl'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'sector_pairs_paper.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ==================== DRY-RUN MODE ====================
DRY_RUN = '--dry-run' in sys.argv

# ==================== STRATEGY CONFIG (HC #751) ====================
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
EXTRA_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', 'SHY', 'HYG', 'GLD']
INITIAL_CAPITAL = 645.0
TOP_K = 3           # top 3 sectors for long (bull call spreads)
BOTTOM_K = 3        # bottom 3 sectors for short (bear put spreads)
REBALANCE_INTERVAL = 10  # biweekly (10 trading days)
SPREAD_PCT = 3.0    # spread width as % of stock price
DTE = 21            # hold to expiry
MAX_POS_SIZE = 200  # max per trade
MAX_POS_PCT = 0.40  # max % of equity per trade
HAIRCUT = 0.15
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # $2.60 round trip

# VIX threshold (NOT GRU regime — this strategy uses VIX directly)
VIX_THRESHOLD = 20.0

# ==================== FEATURES (21 total: 18 legacy + 3 cross-asset) ====================
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


# ==================== STATE MANAGEMENT ====================

def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        'config_version': 'v1_pairs',
        'equity': INITIAL_CAPITAL,
        'open_positions': [],   # both bull and bear spreads
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


# ==================== FEATURE ENGINEERING (21 features) ====================

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
    """Compute the 3 validated cross-asset features."""
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

    # 3. Cross-sector dispersion
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


# ==================== LGBM RANKING (21 features) ====================

def run_lgbm_ranking(sc, close_df):
    """Run walk-forward LGBM ranking using trailing data with 21 features.
    Returns dict of {ticker: score} where higher = better (long candidates)."""
    if not HAS_LGBM:
        log.warning("No LightGBM — using simple momentum ranking")
        rets_21d = sc.pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    # Build training data from trailing rebalance periods
    records = []
    all_dates = sc.index[-300:]  # ~14 months
    rebal_dates = all_dates[::REBALANCE_INTERVAL]

    for dt in rebal_dates[:-1]:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx + 1].dropna()
            feats = compute_legacy_features(px)
            if not feats:
                continue
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

    X_train = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
    y_train = df['rank_label'].values.astype(np.float32)

    m = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1
    )
    m.fit(X_train, y_train)

    # Predict current rankings
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


# ==================== SPREAD PRICING ====================

def price_bull_spread(S, spread_pct, dte, sh_tk, sl_tk, sc_tk, vix_val):
    """Price a bull call spread. Returns (cost_dollars, max_profit_dollars, K1, K2, entry_cost_ps)."""
    K1 = round(S)
    K2 = round(S * (1 + spread_pct / 100))

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
    """Price a bear put spread. Returns (cost_dollars, max_profit_dollars, K1, K2, entry_cost_ps).
    Bear put: buy put at K2 (ATM), sell put at K1 (lower). Profits when underlying falls."""
    K2 = round(S)  # ATM long put
    K1 = round(S * (1 - spread_pct / 100))  # 3% lower short put

    if K1 >= K2:
        K1 = K2 - 1  # ensure K1 < K2

    if sh_tk is not None and len(sh_tk) >= 14:
        atr = compute_atr(sh_tk, sl_tk, sc_tk, period=14)
    else:
        atr = S * 0.015

    entry_cost_ps, max_profit_ps = price_bear_put_spread(
        S=S, K1=K1, K2=K2, dte=dte, atr=atr, vix=vix_val, haircut=HAIRCUT
    )

    cost_dollars = entry_cost_ps * 100 + SPREAD_COMM
    max_profit_dollars = max_profit_ps * 100 - SPREAD_COMM

    return cost_dollars, max_profit_dollars, K1, K2, entry_cost_ps


# ==================== POSITION EXIT (hold to expiry only) ====================

def check_bull_exit(pos, current_price, days_held):
    """Bull call spread: hold to expiry only. At expiry, intrinsic = max(S-K1,0) - max(S-K2,0)."""
    if days_held < DTE:
        return False, 0, 'hold'

    S = current_price
    K1, K2 = pos['K1'], pos['K2']
    entry_cost_ps = pos.get('entry_cost_ps', (pos['cost'] - SPREAD_COMM) / 100.0)

    # At expiry: intrinsic value
    intrinsic = max(S - K1, 0.0) - max(S - K2, 0.0)
    pnl = (intrinsic - entry_cost_ps) * 100 - SPREAD_COMM
    return True, pnl, 'expiry'


def check_bear_exit(pos, current_price, days_held):
    """Bear put spread: hold to expiry only. At expiry, intrinsic = max(K2-S,0) - max(K1-S,0)."""
    if days_held < DTE:
        return False, 0, 'hold'

    S = current_price
    K1, K2 = pos['K1'], pos['K2']
    entry_cost_ps = pos.get('entry_cost_ps', (pos['cost'] - SPREAD_COMM) / 100.0)

    # At expiry: intrinsic value of bear put spread
    intrinsic = max(K2 - S, 0.0) - max(K1 - S, 0.0)
    pnl = (intrinsic - entry_cost_ps) * 100 - SPREAD_COMM
    return True, pnl, 'expiry'


# ==================== POSITION SIZING ====================

def compute_position_size(state):
    """Position size scales with equity growth. Base max $200."""
    equity_ratio = state['equity'] / INITIAL_CAPITAL
    scaled_max = MAX_POS_SIZE * equity_ratio
    equity_cap = state['equity'] * MAX_POS_PCT
    return min(scaled_max, equity_cap, state['equity'] / 6)  # /6 because up to 6 positions


# ==================== MAIN DAILY RUN ====================

def run_daily():
    """Main daily run. Called once per trading day at 4:30 PM ET."""
    state = load_state()
    if DRY_RUN:
        log.info("=" * 60)
        log.info("  DRY-RUN MODE — no state changes will be persisted")
        log.info("=" * 60)

    log.info(f"=== Sector Pairs Paper Engine (HC #751) ===")
    log.info(f"Equity: ${state['equity']:.2f} | Open positions: {len(state['open_positions'])} | "
             f"Trades: {state['total_trades']} | W/L: {state['wins']}/{state['losses']}")

    # Download latest data
    try:
        close_df, sc, sh, sl, spy, vix = download_data()
    except Exception as e:
        log.error(f"Data download failed: {e}")
        return

    today = sc.index[-1]
    current_vix = float(vix.iloc[-1])
    vix_ok = current_vix < VIX_THRESHOLD

    log.info(f"Date: {today.date()} | VIX: {current_vix:.1f} | "
             f"VIX gate: {'ACTIVE (VIX < 20, trades allowed)' if vix_ok else 'BLOCKED (VIX >= 20, no new trades)'}")

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
        else:  # bear
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

        # Track long/short separately
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

    # ==================== CHECK FOR REBALANCE ====================
    state['days_since_rebalance'] = state.get('days_since_rebalance', 999) + 1

    if state['days_since_rebalance'] < REBALANCE_INTERVAL:
        log.info(f"Not rebalance day ({state['days_since_rebalance']}/{REBALANCE_INTERVAL}). "
                 f"Checking positions only.")
        _print_summary(state)
        save_state(state)
        return

    log.info(f"=== REBALANCE DAY ===")
    state['days_since_rebalance'] = 0
    state['last_rebalance'] = str(today.date())

    # ==================== VIX GATE ====================
    if not vix_ok:
        log.info(f"VIX {current_vix:.1f} >= {VIX_THRESHOLD} — NO new trades. "
                 f"Existing positions held to expiry.")
        _print_summary(state)
        save_state(state)
        return

    # ==================== RUN LGBM RANKING (21 features) ====================
    rankings = run_lgbm_ranking(sc, close_df)
    if not rankings:
        log.warning("No rankings available. Skipping rebalance.")
        save_state(state)
        return

    ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    long_picks = [t for t, _ in ranked[:TOP_K]]
    short_picks = [t for t, _ in ranked[-BOTTOM_K:]]

    log.info(f"LGBM rankings: {', '.join(f'{t}={s:.3f}' for t, s in ranked)}")
    log.info(f"LONG picks (top {TOP_K}): {long_picks}")
    log.info(f"SHORT picks (bottom {BOTTOM_K}): {short_picks}")

    # ==================== ENTER NEW POSITIONS ====================
    max_pos = compute_position_size(state)
    if max_pos < 30:
        log.warning(f"Position size too small (${max_pos:.0f}). Skipping entries.")
        _print_summary(state)
        save_state(state)
        return

    entries = 0

    # --- LONG SIDE: Bull call spreads on top-ranked sectors ---
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
            log.info(f"  SKIP {tk} bull: cost ${cost:.2f} exceeds limits "
                     f"(max_pos=${max_pos:.0f}, equity_cap=${state['equity'] * MAX_POS_PCT:.0f})")
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
            'lgbm_score': round(rankings.get(tk, 0), 4),
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
            'lgbm_score': round(rankings.get(tk, 0), 4),
            'equity': round(state['equity'], 2),
        }
        log_trade(trade_record)
        log.info(f"  ENTER {tk} BULL call spread {K1}/{K2}: cost ${cost:.2f}, "
                 f"max profit ${max_profit:.2f}, VIX {current_vix:.1f}")

    # --- SHORT SIDE: Bear put spreads on bottom-ranked sectors ---
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
            log.info(f"  SKIP {tk} bear: cost ${cost:.2f} exceeds limits "
                     f"(max_pos=${max_pos:.0f}, equity_cap=${state['equity'] * MAX_POS_PCT:.0f})")
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
            'lgbm_score': round(rankings.get(tk, 0), 4),
        }
        state['open_positions'].append(position)
        state['total_trades'] += 1
        entries += 1

        trade_record = {
            'action': 'OPEN',
            'date': str(today.date()),
            'ticker': tk,
            'mode': 'bear',
            'entry_price': S,
            'strikes': f"{K1}/{K2}",
            'cost': round(cost, 2),
            'max_profit': round(max_profit, 2),
            'vix': current_vix,
            'lgbm_score': round(rankings.get(tk, 0), 4),
            'equity': round(state['equity'], 2),
        }
        log_trade(trade_record)
        log.info(f"  ENTER {tk} BEAR put spread {K1}/{K2}: cost ${cost:.2f}, "
                 f"max profit ${max_profit:.2f}, VIX {current_vix:.1f}")

    if entries == 0:
        log.info(f"No entries this rebalance (VIX={current_vix:.1f})")

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
    log.info(f"Trades: {state['total_trades']} | W: {state['wins']} L: {state['losses']} | WR: {wr:.1f}%")
    log.info(f"  Long side:  W: {state['long_wins']} L: {state['long_losses']} | WR: {long_wr:.1f}%")
    log.info(f"  Short side: W: {state['short_wins']} L: {state['short_losses']} | WR: {short_wr:.1f}%")

    bulls = [p for p in state['open_positions'] if p['mode'] == 'bull']
    bears = [p for p in state['open_positions'] if p['mode'] == 'bear']
    log.info(f"Open positions: {len(bulls)} bull + {len(bears)} bear = {len(state['open_positions'])} total")
    for p in state['open_positions']:
        log.info(f"  {p['ticker']} {p['mode']} {p['K1']}/{p['K2']} "
                 f"(entry {p['entry_date']}, cost ${p['cost']:.2f}, "
                 f"LGBM {p.get('lgbm_score', 'N/A')})")


if __name__ == '__main__':
    run_daily()
