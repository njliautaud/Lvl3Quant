#!/usr/bin/env python3
"""
Sector Equity Rotation Paper Trading Engine
=============================================

Monthly equity rotation: rank 11 sector ETFs using LGBM model, buy top-2.

Strategy validated in backtest: Sharpe 1.40, 4/5 gates pass, p=0.002.
This is the LIVE paper engine for tracking real-time performance.

How it works:
  - First trading day of each month: sell all, retrain LGBM, buy top-2 ETFs
  - Equal weight allocation across top-2 picks
  - No options, no spreads — just buy/sell ETF shares
  - $0 commission (Robinhood equity trades)
  - Daily mark-to-market of positions

Features: 17 momentum/quality features (same as V10 backtest that passed validation)
Universe: XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLV, XLY, XLU

Usage:
    python sector_equity_rotation_paper.py              # normal daily run
    python sector_equity_rotation_paper.py --dry-run    # simulate without state changes
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

STATE_DIR = BASE / 'state'
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / 'sector_equity_rotation_paper_state.json'

LOG_DIR = BASE / 'logs' / 'paper_engines'
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADE_LOG = LOG_DIR / 'sector_equity_rotation_trades.jsonl'
ENGINE_LOG = LOG_DIR / 'sector_equity_rotation.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(ENGINE_LOG),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ==================== DRY-RUN MODE ====================
DRY_RUN = '--dry-run' in sys.argv

# ==================== CONFIG ====================
CONFIG_VERSION = 'equity_rotation_top2_monthly'
INITIAL_CAPITAL = 645.0
TOP_K = 2
REBALANCE_FREQ = 'monthly'  # first trading day of month

SECTORS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLV', 'XLY', 'XLU']

# 17 LGBM momentum features (identical to validated V10 backtest)
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
]

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    log.warning("LightGBM not available. Will use simple momentum ranking fallback.")


# ==================== STATE MANAGEMENT ====================

def _default_state():
    return {
        'config_version': CONFIG_VERSION,
        'equity': INITIAL_CAPITAL,
        'initial_capital': INITIAL_CAPITAL,
        'positions': [],       # list of {ticker, shares, entry_price, entry_date}
        'trade_count': 0,
        'win_count': 0,
        'loss_count': 0,
        'total_pnl': 0.0,
        'last_rebalance': None,
        'rankings': {},        # latest LGBM rankings
        'last_update': None,
        'created': datetime.now().isoformat(),
    }


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        # Migration: ensure all keys exist
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
    state['last_update'] = datetime.now().isoformat()
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(record):
    if DRY_RUN:
        log.info(f"[DRY-RUN] Trade NOT logged: {record.get('action')} {record.get('ticker', 'N/A')}")
        return
    with open(TRADE_LOG, 'a') as f:
        f.write(json.dumps(record, default=str) + '\n')


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download sector ETF + SPY data via yfinance. Need ~400 days for features."""
    import yfinance as yf
    all_tickers = SECTORS + ['SPY']
    log.info(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start='2024-01-01', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    close = close.ffill()
    spy = close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    ix = sc.index.intersection(spy.index)
    return sc.loc[ix], spy.loc[ix]


# ==================== FEATURE ENGINEERING (17 features) ====================

def compute_features(px):
    """Compute the 17 momentum/quality features for a single sector ETF.
    Identical to the validated backtest (sector_ranking_equity_backtest_v1.py)."""
    if len(px) < 260:
        return None
    f = {}

    # Returns at multiple lookbacks
    for lb, nm in [(5, 'ret_5d'), (10, 'ret_10d'), (21, 'ret_21d'),
                   (63, 'ret_63d'), (126, 'ret_126d'), (252, 'ret_252d')]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0

    rets = px.pct_change().dropna()

    # Volatility
    f['vol_21d'] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2

    # Risk-adjusted
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0

    # Drawdown
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:] / pk63) - 1).min())

    # Relative position
    f['pct_52w_high'] = float(px.iloc[-1] / px.iloc[-252:].max())

    # Momentum acceleration
    f['mom_accel'] = f['ret_21d'] - f['ret_63d'] / 3

    # Monthly consistency
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5

    # Sortino
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0

    # Calmar
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)

    # Trend
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

def run_lgbm_ranking(sc):
    """Walk-forward LGBM ranking using 60-day sliding window of training samples.
    Identical logic to the validated backtest.

    Returns dict of {ticker: predicted_rank_score} where higher = better.
    """
    if not HAS_LGBM:
        log.warning("No LightGBM -- falling back to 21-day momentum ranking")
        rets_21d = sc.pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    idx_end = len(sc) - 1

    # Build training samples: go back ~400 trading days, sample every 20 days
    records = []
    start_i = max(260, idx_end - 400)
    all_idx = list(range(start_i, idx_end))
    rebal_idx = all_idx[::20]

    for i in rebal_idx[:-1]:
        for tk in sc.columns:
            px = sc[tk].iloc[:i + 1].dropna()
            feats = compute_features(px)
            if not feats:
                continue
            fi = min(i + 28, len(sc) - 1)
            feats.update({
                'date_idx': i,
                'ticker': tk,
                'fwd_ret': float(sc[tk].iloc[fi] / sc[tk].iloc[i] - 1)
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

    # Label: percentile rank of forward return within each date group
    df['rank_label'] = df.groupby('date_idx')['fwd_ret'].rank(pct=True)

    X_train = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
    y_train = df['rank_label'].values.astype(np.float32)

    m = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1
    )
    m.fit(X_train, y_train)

    # Predict current rankings for all sectors
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


# ==================== REBALANCE DETECTION ====================

def is_first_trading_day_of_month(dt, all_dates):
    """Check if dt is the first trading day of its month.
    Looks at all_dates to find the actual first trading day."""
    month_dates = [d for d in all_dates if d.year == dt.year and d.month == dt.month]
    if not month_dates:
        return False
    return dt == month_dates[0]


# ==================== CURRENT PRICES ====================

def get_current_prices(tickers):
    """Get current/latest prices for a list of tickers via yfinance."""
    import yfinance as yf
    data = yf.download(tickers, period='5d', progress=False)
    mi = isinstance(data.columns, pd.MultiIndex)
    close = data['Close'] if mi else data
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    prices = {}
    for tk in tickers:
        if tk in close.columns:
            val = close[tk].dropna()
            if len(val) > 0:
                prices[tk] = float(val.iloc[-1])
    return prices


# ==================== MAIN DAILY RUN ====================

def run_daily():
    """Main daily run. Designed to be called once per day at ~4:30 PM ET."""
    state = load_state()

    if DRY_RUN:
        log.info("=" * 60)
        log.info("  DRY-RUN MODE -- no state changes will be persisted")
        log.info("=" * 60)

    log.info("=== Sector Equity Rotation Paper Engine ===")
    log.info(f"Config: {CONFIG_VERSION} | Top-{TOP_K} | {REBALANCE_FREQ}")
    log.info(f"Equity: ${state['equity']:.2f} | Positions: {len(state['positions'])} | "
             f"Trades: {state['trade_count']} | W/L: {state['win_count']}/{state['loss_count']} | "
             f"PnL: ${state['total_pnl']:.2f}")

    # Download data (need history for LGBM training)
    try:
        sc, spy = download_data()
    except Exception as e:
        log.error(f"Data download failed: {e}")
        return

    today = sc.index[-1]
    today_dt = today.to_pydatetime() if hasattr(today, 'to_pydatetime') else today
    log.info(f"Date: {today.date()} | Data: {sc.index[0].date()} to {sc.index[-1].date()} ({len(sc)} days)")

    # ── Mark-to-market existing positions ──
    if state['positions']:
        log.info("--- Mark-to-Market ---")
        total_position_value = 0.0
        for pos in state['positions']:
            tk = pos['ticker']
            if tk in sc.columns:
                current_price = float(sc[tk].iloc[-1])
                position_value = pos['shares'] * current_price
                unrealized_pnl = pos['shares'] * (current_price - pos['entry_price'])
                total_position_value += position_value
                log.info(f"  {tk}: {pos['shares']:.4f} shares @ ${pos['entry_price']:.2f} -> "
                         f"${current_price:.2f} | Value: ${position_value:.2f} | "
                         f"Unrealized: ${unrealized_pnl:+.2f}")

        # Update equity = cash (realized PnL) + current position value
        # equity = initial_capital + total_realized_pnl + total_unrealized_pnl
        total_unrealized = sum(
            pos['shares'] * (float(sc[pos['ticker']].iloc[-1]) - pos['entry_price'])
            for pos in state['positions']
            if pos['ticker'] in sc.columns
        )
        state['equity'] = state['initial_capital'] + state['total_pnl'] + total_unrealized
        log.info(f"  Portfolio value: ${total_position_value:.2f} | "
                 f"Equity: ${state['equity']:.2f} (realized: ${state['total_pnl']:.2f}, "
                 f"unrealized: ${total_unrealized:+.2f})")
    else:
        log.info("No open positions.")

    # ── Check for rebalance ──
    all_dates = list(sc.index)
    is_rebalance = is_first_trading_day_of_month(today, all_dates)

    # Also rebalance if we have no positions and have never rebalanced
    if not state['positions'] and state['last_rebalance'] is None:
        is_rebalance = True
        log.info("First run with no positions -- forcing initial rebalance.")

    if not is_rebalance:
        log.info(f"Not a rebalance day (last rebalance: {state['last_rebalance']}). "
                 f"MTM update only.")
        save_state(state)
        return

    # ==================== REBALANCE ====================
    log.info("=" * 50)
    log.info("  REBALANCE DAY")
    log.info("=" * 50)

    # Step 1: Run LGBM ranking
    log.info("Training LGBM and ranking sectors...")
    rankings = run_lgbm_ranking(sc)
    if not rankings:
        log.error("No rankings produced. Skipping rebalance.")
        save_state(state)
        return

    ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    log.info("Rankings (best to worst):")
    for i, (tk, score) in enumerate(ranked):
        marker = " <-- BUY" if i < TOP_K else ""
        log.info(f"  {i+1:2d}. {tk}: {score:.4f}{marker}")

    # Store rankings in state
    state['rankings'] = {tk: round(score, 4) for tk, score in ranked}

    # Step 2: Close all existing positions
    if state['positions']:
        log.info("--- Closing all positions ---")
        for pos in state['positions']:
            tk = pos['ticker']
            if tk not in sc.columns:
                log.warning(f"  {tk}: no price data, skipping close")
                continue

            exit_price = float(sc[tk].iloc[-1])
            pnl = pos['shares'] * (exit_price - pos['entry_price'])
            is_win = pnl > 0

            state['total_pnl'] += pnl
            state['trade_count'] += 1
            if is_win:
                state['win_count'] += 1
            else:
                state['loss_count'] += 1

            log.info(f"  SELL {tk}: {pos['shares']:.4f} shares @ ${exit_price:.2f} "
                     f"(entry ${pos['entry_price']:.2f}, PnL: ${pnl:+.2f}, "
                     f"{'WIN' if is_win else 'LOSS'})")

            log_trade({
                'action': 'SELL',
                'date': str(today.date()),
                'ticker': tk,
                'shares': round(pos['shares'], 4),
                'entry_price': pos['entry_price'],
                'entry_date': pos['entry_date'],
                'exit_price': round(exit_price, 2),
                'pnl': round(pnl, 2),
                'result': 'WIN' if is_win else 'LOSS',
                'equity_after': round(state['initial_capital'] + state['total_pnl'], 2),
            })

        state['positions'] = []

    # Step 3: Buy top-K sectors
    # Available equity = initial_capital + total_realized_pnl (all positions now closed)
    available_equity = state['initial_capital'] + state['total_pnl']
    state['equity'] = available_equity
    alloc_per_pick = available_equity / TOP_K

    log.info(f"--- Opening new positions ---")
    log.info(f"Available equity: ${available_equity:.2f} | Per pick: ${alloc_per_pick:.2f}")

    top_picks = [tk for tk, _ in ranked[:TOP_K]]
    new_positions = []

    for tk in top_picks:
        if tk not in sc.columns:
            log.warning(f"  {tk}: no price data, skipping buy")
            continue

        current_price = float(sc[tk].iloc[-1])
        shares = alloc_per_pick / current_price

        if shares < 0.0001:
            log.warning(f"  {tk}: share count too small (${alloc_per_pick:.2f} / ${current_price:.2f})")
            continue

        position = {
            'ticker': tk,
            'shares': round(shares, 4),
            'entry_price': round(current_price, 2),
            'entry_date': str(today.date()),
        }
        new_positions.append(position)

        log.info(f"  BUY {tk}: {shares:.4f} shares @ ${current_price:.2f} "
                 f"(${alloc_per_pick:.2f}, LGBM score: {rankings[tk]:.4f})")

        log_trade({
            'action': 'BUY',
            'date': str(today.date()),
            'ticker': tk,
            'shares': round(shares, 4),
            'entry_price': round(current_price, 2),
            'allocation': round(alloc_per_pick, 2),
            'lgbm_score': round(rankings[tk], 4),
            'lgbm_rank': top_picks.index(tk) + 1,
            'equity': round(available_equity, 2),
        })

    state['positions'] = new_positions
    state['last_rebalance'] = str(today.date())

    # ── Summary ──
    _print_summary(state)
    save_state(state)
    log.info("State saved. Done.")


def _print_summary(state):
    """Print portfolio summary."""
    total_trades = state['trade_count']
    wr = state['win_count'] / total_trades * 100 if total_trades > 0 else 0

    log.info(f"\n=== Summary ===")
    log.info(f"Equity: ${state['equity']:.2f} | Initial: ${state['initial_capital']:.2f} | "
             f"Return: {(state['equity'] / state['initial_capital'] - 1) * 100:+.2f}%")
    log.info(f"Total PnL: ${state['total_pnl']:.2f}")
    log.info(f"Trades: {total_trades} | W: {state['win_count']} L: {state['loss_count']} | "
             f"WR: {wr:.1f}%")
    log.info(f"Last rebalance: {state['last_rebalance']}")

    if state['positions']:
        log.info(f"Open positions ({len(state['positions'])}):")
        for pos in state['positions']:
            log.info(f"  {pos['ticker']}: {pos['shares']:.4f} shares @ ${pos['entry_price']:.2f} "
                     f"(since {pos['entry_date']})")

    if state['rankings']:
        ranked = sorted(state['rankings'].items(), key=lambda x: x[1], reverse=True)
        log.info(f"Latest rankings: {', '.join(f'{tk}={s:.3f}' for tk, s in ranked[:5])} ...")


if __name__ == '__main__':
    run_daily()
