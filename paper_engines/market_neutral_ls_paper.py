#!/usr/bin/env python3
"""
Market-Neutral Long/Short Sector Rotation Paper Trading Engine
===============================================================

Monthly rebalance paper engine. Uses LightGBM to rank 11 sector ETFs
using 17 quality-momentum features. Longs top-3, shorts bottom-3.

STRATEGY (KB #285, #289 — VALIDATED, Sharpe 2.64, 4/5 audit):
  - LGBM ranks 11 sector ETFs using 17 quality-momentum features
  - Long top-3, Short bottom-3, equal-weight within each side
  - Monthly rebalance (~21 trading days)
  - $10,000 starting capital
  - Zero commission (ETF trades)
  - Pure equity (NO options)

Usage:
  python3 paper_engines/market_neutral_ls_paper.py
  python3 paper_engines/market_neutral_ls_paper.py --dry-run

PM2 cron: "30 16 * * 1-5"
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# --- Paths ---
ENGINE_DIR = Path(__file__).resolve().parent
LOG_DIR = ENGINE_DIR / 'logs'
LOG_DIR.mkdir(exist_ok=True)
STATE_DIR = ENGINE_DIR.parent / 'state'
STATE_DIR.mkdir(exist_ok=True)

STATE_PATH = STATE_DIR / 'market_neutral_ls_paper_state.json'
TRADE_LOG_PATH = LOG_DIR / 'market_neutral_ls_trades.jsonl'
LOG_FILE = LOG_DIR / 'market_neutral_ls_paper.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# --- Constants ---
CAPITAL_INITIAL = 10000.0
TOP_K = 3          # long top-3, short bottom-3
REBALANCE_DAYS = 21
DATA_START = '2024-01-01'
TRAIN_LOOKBACK = 300  # trailing days for LGBM training

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
BENCHMARK = 'SPY'

FEATURE_NAMES = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high',
    'mom_accel', 'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'up_capture', 'trend_slope_63d',
]

LGBM_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 5,
    'verbose': -1,
    'random_state': 42,
}


# --- Feature Engineering ---

def compute_features_at(close_series: pd.Series, benchmark_series: pd.Series, idx: int) -> dict:
    """Compute 17 features for a single ETF at a given index position.

    close_series and benchmark_series must be aligned and indexed 0..N-1.
    idx is the position at which to compute features (needs >=252 prior bars).
    """
    c = close_series.iloc[:idx + 1]
    b = benchmark_series.iloc[:idx + 1]

    if len(c) < 253:
        return None

    features = {}
    lr = np.log(c / c.shift(1)).dropna()

    # Momentum returns
    for d in [5, 10, 21, 63, 126, 252]:
        features[f'ret_{d}d'] = float(c.iloc[-1] / c.iloc[-1 - d] - 1) if len(c) > d else np.nan

    # Volatility
    features['vol_21d'] = float(lr.iloc[-21:].std() * np.sqrt(252)) if len(lr) >= 21 else np.nan
    features['vol_63d'] = float(lr.iloc[-63:].std() * np.sqrt(252)) if len(lr) >= 63 else np.nan

    # Sharpe 63d
    if len(lr) >= 63:
        r63 = lr.iloc[-63:]
        std63 = r63.std()
        features['sharpe_63d'] = float(r63.mean() / std63) if std63 > 0 else 0.0
    else:
        features['sharpe_63d'] = np.nan

    # Max drawdown 63d
    if len(c) >= 63:
        c63 = c.iloc[-63:]
        roll_max = c63.cummax()
        dd = (c63 - roll_max) / roll_max
        features['maxdd_63d'] = float(dd.min())
    else:
        features['maxdd_63d'] = np.nan

    # Pct of 52w high
    if len(c) >= 252:
        features['pct_52w_high'] = float(c.iloc[-1] / c.iloc[-252:].max())
    else:
        features['pct_52w_high'] = np.nan

    # Momentum acceleration: ret_126d now minus ret_126d 63 days ago
    if len(c) > 189:
        ret_126_now = c.iloc[-1] / c.iloc[-127] - 1
        ret_126_prior = c.iloc[-64] / c.iloc[-190] - 1
        features['mom_accel'] = float(ret_126_now - ret_126_prior)
    else:
        features['mom_accel'] = np.nan

    # Pct positive months in last 12 months (~252 days, sample monthly)
    if len(c) >= 252:
        monthly_rets = []
        for m in range(12):
            start_idx = -(m + 1) * 21 - 1
            end_idx = -m * 21 - 1 if m > 0 else -1
            if abs(start_idx) < len(c):
                mr = c.iloc[end_idx] / c.iloc[start_idx] - 1
                monthly_rets.append(mr)
        features['pct_pos_months_12m'] = float(sum(1 for r in monthly_rets if r > 0) / max(len(monthly_rets), 1))
    else:
        features['pct_pos_months_12m'] = np.nan

    # Sortino 63d
    if len(lr) >= 63:
        r63 = lr.iloc[-63:]
        downside = r63[r63 < 0]
        ds_std = downside.std() if len(downside) > 1 else 1e-8
        features['sortino_63d'] = float(r63.mean() / ds_std) if ds_std > 0 else 0.0
    else:
        features['sortino_63d'] = np.nan

    # Calmar 1y: annual return / max drawdown
    if len(c) >= 252:
        ann_ret = c.iloc[-1] / c.iloc[-252] - 1
        c252 = c.iloc[-252:]
        roll_max252 = c252.cummax()
        dd252 = ((c252 - roll_max252) / roll_max252).min()
        features['calmar_1y'] = float(ann_ret / abs(dd252)) if abs(dd252) > 1e-8 else 0.0
    else:
        features['calmar_1y'] = np.nan

    # Up capture vs benchmark (63d)
    if len(lr) >= 63 and len(b) > idx:
        blr = np.log(b / b.shift(1)).dropna()
        if len(blr) >= 63:
            b63 = blr.iloc[-63:]
            a63 = lr.iloc[-63:]
            up_mask = b63 > 0
            if up_mask.sum() > 5:
                features['up_capture'] = float(a63[up_mask].mean() / b63[up_mask].mean())
            else:
                features['up_capture'] = 1.0
        else:
            features['up_capture'] = np.nan
    else:
        features['up_capture'] = np.nan

    # Trend slope 63d (linear regression of log prices)
    if len(c) >= 63:
        y = np.log(c.iloc[-63:].values)
        x = np.arange(63)
        slope, _, _, _, _ = linregress(x, y)
        features['trend_slope_63d'] = float(slope)
    else:
        features['trend_slope_63d'] = np.nan

    return features


def compute_current_features(close_series: pd.Series, benchmark_series: pd.Series) -> dict:
    """Compute features at the latest bar."""
    return compute_features_at(close_series, benchmark_series, len(close_series) - 1)


# --- LGBM Training ---

def build_training_data(all_close: pd.DataFrame, benchmark_close: pd.Series, lookback: int = TRAIN_LOOKBACK):
    """Build training dataset from trailing `lookback` days.

    For each rebalance point (every 21 days going back), compute features
    for all sectors, and the forward 21-day return as target.
    """
    rows = []
    n = len(all_close)

    # Need at least 252 warmup + 21 forward
    start_idx = 252
    end_idx = n - 21  # need 21 days forward for target

    if end_idx <= start_idx:
        return None, None

    # Limit lookback
    earliest = max(start_idx, end_idx - lookback)

    # Sample at rebalance intervals (every 21 days)
    sample_indices = list(range(end_idx - 1, earliest - 1, -21))
    sample_indices.reverse()

    for idx in sample_indices:
        for ticker in SECTOR_ETFS:
            if ticker not in all_close.columns:
                continue
            cs = all_close[ticker].reset_index(drop=True)
            bs = benchmark_close.reset_index(drop=True)

            feats = compute_features_at(cs, bs, idx)
            if feats is None:
                continue
            if any(np.isnan(v) for v in feats.values()):
                continue

            # Forward 21-day return as target
            fwd_ret = float(all_close[ticker].iloc[idx + 21] / all_close[ticker].iloc[idx] - 1)
            row = {**feats, 'target': fwd_ret, 'ticker': ticker, 'idx': idx}
            rows.append(row)

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    X = df[FEATURE_NAMES].values
    y = df['target'].values
    return X, y


def train_lgbm(X, y):
    """Train LightGBM regressor."""
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y)
    return model


# --- State Management ---

def default_state():
    return {
        'config_version': 'market_neutral_ls_v1',
        'equity': CAPITAL_INITIAL,
        'long_positions': [],
        'short_positions': [],
        'last_rebalance': None,
        'days_since_rebalance': 999,
        'total_trades': 0,
        'total_pnl': 0.0,
        'wins': 0,
        'losses': 0,
        'created': datetime.now().isoformat(),
        'last_run': None,
        'rebalance_count': 0,
        'benchmark_start_price': None,
    }


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return default_state()


def save_state(state):
    state['last_run'] = datetime.now().isoformat()
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def log_trade(trade: dict):
    """Append a trade record to the JSONL log."""
    trade['timestamp'] = datetime.now().isoformat()
    with open(TRADE_LOG_PATH, 'a') as f:
        f.write(json.dumps(trade, default=str) + '\n')


# --- Mark-to-Market ---

def mark_to_market(state, prices: dict):
    """Compute unrealized PnL and current equity."""
    long_pnl = 0.0
    short_pnl = 0.0

    for pos in state.get('long_positions', []):
        ticker = pos['ticker']
        if ticker in prices:
            current = prices[ticker]
            pnl = (current - pos['entry_price']) * pos['shares']
            long_pnl += pnl

    for pos in state.get('short_positions', []):
        ticker = pos['ticker']
        if ticker in prices:
            current = prices[ticker]
            pnl = (pos['entry_price'] - current) * pos['shares']
            short_pnl += pnl

    total_unrealized = long_pnl + short_pnl
    # Equity = initial capital + realized PnL + unrealized PnL
    current_equity = CAPITAL_INITIAL + state.get('total_pnl', 0.0) + total_unrealized
    return current_equity, long_pnl, short_pnl


# --- Main ---

def main():
    import yfinance as yf

    parser = argparse.ArgumentParser(description='Market-Neutral L/S Sector Rotation Paper Engine')
    parser.add_argument('--dry-run', action='store_true', help='Compute signals but do not update state')
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Market-Neutral L/S Sector Rotation — Daily Run")
    log.info("=" * 60)

    if not HAS_LGBM:
        log.error("LightGBM not installed. pip install lightgbm")
        sys.exit(1)

    state = load_state()
    today = datetime.now()

    if today.weekday() >= 5:
        log.info(f"Weekend ({today.strftime('%A')}), skipping")
        save_state(state)
        return

    # Download data for all sectors + benchmark
    tickers = SECTOR_ETFS + [BENCHMARK]
    log.info(f"Downloading data for {len(tickers)} tickers from {DATA_START}...")

    try:
        raw = yf.download(tickers, start=DATA_START, progress=False, group_by='ticker')
    except Exception as e:
        log.error(f"yfinance download failed: {e}")
        return

    # Extract close prices into a clean DataFrame
    close_prices = pd.DataFrame()
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                col = raw[(ticker, 'Close')]
            else:
                col = raw['Close']
            close_prices[ticker] = col
        except Exception as e:
            log.warning(f"Could not extract {ticker}: {e}")

    close_prices = close_prices.dropna(how='all').ffill()
    log.info(f"Price data: {len(close_prices)} trading days, {close_prices.shape[1]} tickers")

    if len(close_prices) < 253:
        log.error(f"Insufficient data: {len(close_prices)} days, need 253+")
        return

    # Get current prices for mark-to-market
    current_prices = {}
    for ticker in SECTOR_ETFS:
        if ticker in close_prices.columns:
            current_prices[ticker] = float(close_prices[ticker].iloc[-1])

    # Store benchmark start price on first run
    if state.get('benchmark_start_price') is None and BENCHMARK in close_prices.columns:
        state['benchmark_start_price'] = float(close_prices[BENCHMARK].iloc[-1])

    # Increment days since rebalance
    state['days_since_rebalance'] = state.get('days_since_rebalance', 999) + 1

    # Check if rebalance is needed
    has_positions = bool(state.get('long_positions') or state.get('short_positions'))
    needs_rebalance = state['days_since_rebalance'] >= REBALANCE_DAYS or not has_positions

    if not needs_rebalance:
        # Mark-to-market only
        equity, long_pnl, short_pnl = mark_to_market(state, current_prices)
        state['equity'] = round(equity, 2)

        pct_return = (equity / CAPITAL_INITIAL - 1) * 100
        log.info(f"Day {state['days_since_rebalance']}/{REBALANCE_DAYS} — Not rebalance day")
        log.info(f"  Equity: ${equity:,.2f} ({pct_return:+.1f}%)")
        log.info(f"  Long PnL: ${long_pnl:+,.2f}  Short PnL: ${short_pnl:+,.2f}")

        long_tickers = [p['ticker'] for p in state.get('long_positions', [])]
        short_tickers = [p['ticker'] for p in state.get('short_positions', [])]
        log.info(f"  LONG:  {', '.join(long_tickers)}")
        log.info(f"  SHORT: {', '.join(short_tickers)}")

        if not args.dry_run:
            save_state(state)
        log.info("Done (mark-to-market only).")
        return

    # === REBALANCE DAY ===
    log.info("REBALANCE DAY — Training LGBM and scoring sectors...")

    # Build training data
    sector_close = close_prices[SECTOR_ETFS].copy()
    benchmark_close = close_prices[BENCHMARK].copy()

    X_train, y_train = build_training_data(sector_close, benchmark_close, lookback=TRAIN_LOOKBACK)

    if X_train is None or len(X_train) < 20:
        log.error(f"Insufficient training data ({0 if X_train is None else len(X_train)} samples)")
        save_state(state)
        return

    log.info(f"Training LGBM on {len(X_train)} samples...")
    model = train_lgbm(X_train, y_train)

    # Feature importance
    importances = model.feature_importances_
    top_feats = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)[:5]
    log.info(f"Top features: {', '.join(f'{n}={v}' for n, v in top_feats)}")

    # Score all sectors
    scores = {}
    for ticker in SECTOR_ETFS:
        if ticker not in sector_close.columns:
            continue
        cs = sector_close[ticker].reset_index(drop=True)
        bs = benchmark_close.reset_index(drop=True)

        feats = compute_current_features(cs, bs)
        if feats is None:
            log.warning(f"  {ticker}: insufficient data for features")
            continue
        if any(np.isnan(v) for v in feats.values()):
            log.warning(f"  {ticker}: NaN in features, skipping")
            continue

        X_pred = np.array([[feats[f] for f in FEATURE_NAMES]])
        pred = model.predict(X_pred)[0]
        scores[ticker] = {'predicted_fwd_ret': float(pred), 'features': {k: round(float(v), 6) for k, v in feats.items()}}

    if len(scores) < 2 * TOP_K:
        log.error(f"Only {len(scores)} sectors scored, need at least {2 * TOP_K}")
        save_state(state)
        return

    # Rank: top-3 LONG, bottom-3 SHORT
    ranked = sorted(scores.items(), key=lambda x: x[1]['predicted_fwd_ret'], reverse=True)
    long_tickers = [t for t, _ in ranked[:TOP_K]]
    short_tickers = [t for t, _ in ranked[-TOP_K:]]

    log.info("\nSector Rankings (predicted 21d fwd return):")
    for i, (ticker, data) in enumerate(ranked):
        side = "LONG" if ticker in long_tickers else ("SHORT" if ticker in short_tickers else "     ")
        log.info(f"  {i+1:2d}. {ticker:5s}  pred={data['predicted_fwd_ret']:+.4f}  [{side}]")

    if args.dry_run:
        log.info("\n[DRY RUN] Would rebalance to:")
        log.info(f"  LONG:  {', '.join(long_tickers)}")
        log.info(f"  SHORT: {', '.join(short_tickers)}")
        log.info("  No state changes made.")
        return

    # --- Close existing positions and realize PnL ---
    realized_pnl = 0.0

    for pos in state.get('long_positions', []):
        ticker = pos['ticker']
        current = current_prices.get(ticker, pos['entry_price'])
        pnl = (current - pos['entry_price']) * pos['shares']
        realized_pnl += pnl
        is_win = pnl > 0
        if is_win:
            state['wins'] = state.get('wins', 0) + 1
        else:
            state['losses'] = state.get('losses', 0) + 1
        state['total_trades'] = state.get('total_trades', 0) + 1

        log.info(f"  CLOSE LONG  {ticker}: entry={pos['entry_price']:.2f} exit={current:.2f} pnl=${pnl:+.2f}")
        log_trade({
            'action': 'CLOSE_LONG',
            'ticker': ticker,
            'shares': pos['shares'],
            'entry_price': pos['entry_price'],
            'exit_price': current,
            'pnl': round(pnl, 2),
            'entry_date': pos['entry_date'],
            'exit_date': str(today.date()),
        })

    for pos in state.get('short_positions', []):
        ticker = pos['ticker']
        current = current_prices.get(ticker, pos['entry_price'])
        pnl = (pos['entry_price'] - current) * pos['shares']
        realized_pnl += pnl
        is_win = pnl > 0
        if is_win:
            state['wins'] = state.get('wins', 0) + 1
        else:
            state['losses'] = state.get('losses', 0) + 1
        state['total_trades'] = state.get('total_trades', 0) + 1

        log.info(f"  CLOSE SHORT {ticker}: entry={pos['entry_price']:.2f} exit={current:.2f} pnl=${pnl:+.2f}")
        log_trade({
            'action': 'CLOSE_SHORT',
            'ticker': ticker,
            'shares': pos['shares'],
            'entry_price': pos['entry_price'],
            'exit_price': current,
            'pnl': round(pnl, 2),
            'entry_date': pos['entry_date'],
            'exit_date': str(today.date()),
        })

    state['total_pnl'] = round(state.get('total_pnl', 0.0) + realized_pnl, 2)

    # --- Open new positions ---
    # Equal weight: $10K / 6 positions = ~$1,666.67 per position
    updated_equity = CAPITAL_INITIAL + state['total_pnl']
    weight_per_position = updated_equity / (2 * TOP_K)

    new_long_positions = []
    for ticker in long_tickers:
        price = current_prices.get(ticker)
        if price is None or price <= 0:
            log.warning(f"  Cannot open LONG {ticker}: no price")
            continue
        shares = round(weight_per_position / price, 4)
        new_long_positions.append({
            'ticker': ticker,
            'shares': shares,
            'entry_date': str(today.date()),
            'entry_price': price,
        })
        state['total_trades'] = state.get('total_trades', 0) + 1
        log.info(f"  OPEN LONG   {ticker}: {shares:.4f} shares @ ${price:.2f} (${weight_per_position:,.2f})")
        log_trade({
            'action': 'OPEN_LONG',
            'ticker': ticker,
            'shares': shares,
            'entry_price': price,
            'entry_date': str(today.date()),
        })

    new_short_positions = []
    for ticker in short_tickers:
        price = current_prices.get(ticker)
        if price is None or price <= 0:
            log.warning(f"  Cannot open SHORT {ticker}: no price")
            continue
        shares = round(weight_per_position / price, 4)
        new_short_positions.append({
            'ticker': ticker,
            'shares': shares,
            'entry_date': str(today.date()),
            'entry_price': price,
        })
        state['total_trades'] = state.get('total_trades', 0) + 1
        log.info(f"  OPEN SHORT  {ticker}: {shares:.4f} shares @ ${price:.2f} (${weight_per_position:,.2f})")
        log_trade({
            'action': 'OPEN_SHORT',
            'ticker': ticker,
            'shares': shares,
            'entry_price': price,
            'entry_date': str(today.date()),
        })

    # Update state
    state['long_positions'] = new_long_positions
    state['short_positions'] = new_short_positions
    state['last_rebalance'] = str(today.date())
    state['days_since_rebalance'] = 0
    state['rebalance_count'] = state.get('rebalance_count', 0) + 1
    state['equity'] = round(updated_equity, 2)

    # Benchmark comparison
    if state.get('benchmark_start_price') and BENCHMARK in current_prices:
        bench_ret = (current_prices.get(BENCHMARK, close_prices[BENCHMARK].iloc[-1]) / state['benchmark_start_price'] - 1) * 100
    else:
        bench_ret = 0.0

    # Summary
    total_return_pct = (state['equity'] / CAPITAL_INITIAL - 1) * 100
    total_trades = state.get('total_trades', 0)
    wins = state.get('wins', 0)
    losses = state.get('losses', 0)
    win_rate = wins / max(wins + losses, 1) * 100

    log.info(f"\n{'='*40}")
    log.info(f"REBALANCE #{state['rebalance_count']} COMPLETE")
    log.info(f"{'='*40}")
    log.info(f"  Equity:       ${state['equity']:,.2f} ({total_return_pct:+.1f}%)")
    log.info(f"  Realized PnL: ${state['total_pnl']:+,.2f}")
    log.info(f"  This period:  ${realized_pnl:+,.2f}")
    log.info(f"  Trades: {total_trades}  W/L: {wins}/{losses}  WR: {win_rate:.0f}%")
    log.info(f"  SPY return:   {bench_ret:+.1f}%")
    log.info(f"  LONG:  {', '.join(long_tickers)}")
    log.info(f"  SHORT: {', '.join(short_tickers)}")
    log.info(f"  Next rebalance in {REBALANCE_DAYS} trading days")

    save_state(state)
    log.info("State saved. Done.")


if __name__ == '__main__':
    main()
