#!/usr/bin/env python3
"""
Factor ETF Rotation Paper Trading Engine
==========================================

Monthly rotation across factor ETFs using LGBM ranking + trailing stop.
Based on backtest variant D: Sharpe 1.132, CAGR 21.7%, $645→$4,986, 4/5 gates.

How it works:
  - Biweekly (every 14 trading days): retrain LGBM, buy top-2 factor ETFs
  - 15% trailing stop from peak equity
  - Equal weight allocation
  - $0 commission (Robinhood equity trades)
  - Daily mark-to-market

Features: 22 (17 momentum + 5 cross-asset) — enhanced per LGBM enhancement study
Universe: MTUM, VLUE, QUAL, SIZE, USMV, VTV, VUG, MOAT, COWZ, NOBL

Usage:
    python factor_etf_rotation_paper.py              # normal daily run
    python factor_etf_rotation_paper.py --dry-run    # simulate without state changes
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

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

STATE_DIR = BASE / 'state'
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / 'factor_etf_rotation_paper_state.json'

LOG_DIR = BASE / 'logs' / 'paper_engines'
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRADE_LOG = LOG_DIR / 'factor_etf_rotation_trades.jsonl'
ENGINE_LOG = LOG_DIR / 'factor_etf_rotation.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(ENGINE_LOG),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ==================== CONFIG ====================

FACTOR_ETFS = ['MTUM', 'VLUE', 'QUAL', 'SIZE', 'USMV', 'VTV', 'VUG', 'MOAT', 'COWZ', 'NOBL']
INITIAL_CAPITAL = 645.0
N_PICKS = 2
REBALANCE_DAYS = 14
TRAILING_STOP = 0.15

FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
    'corr_to_spy_63d', 'beta_to_spy_63d', 'rel_strength_vs_mean',
    'ret_21d_minus_spy', 'vol_ratio_21_63',
]

DRY_RUN = '--dry-run' in sys.argv


# ==================== STATE ====================

def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        'capital': INITIAL_CAPITAL,
        'positions': {},
        'peak_equity': INITIAL_CAPITAL,
        'last_rebalance': None,
        'trades': [],
        'created': datetime.now().isoformat(),
    }

def save_state(state):
    if DRY_RUN:
        log.info("[DRY RUN] Would save state")
        return
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def log_trade(trade):
    if DRY_RUN:
        log.info(f"[DRY RUN] Would log trade: {trade}")
        return
    with open(TRADE_LOG, 'a') as f:
        f.write(json.dumps(trade, default=str) + '\n')


# ==================== FEATURES ====================

def compute_features(px, spy_px=None, all_close=None):
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

    # Cross-asset features
    if all_close is not None:
        peer_rets = []
        for c in all_close.columns:
            if len(all_close[c].dropna()) > 21:
                peer_rets.append(float(all_close[c].iloc[-1] / all_close[c].iloc[-21] - 1))
        f['rel_strength_vs_mean'] = f['ret_21d'] - (np.mean(peer_rets) if peer_rets else 0)
    else:
        f['rel_strength_vs_mean'] = 0.0

    if spy_px is not None and len(spy_px) > 63 and len(px) >= 63:
        sp_rets = spy_px.pct_change().dropna().iloc[-63:]
        tk_rets = rets.iloc[-63:]
        common = sp_rets.index.intersection(tk_rets.index)
        if len(common) > 20:
            cov = np.cov(tk_rets.loc[common].values, sp_rets.loc[common].values)
            f['beta_to_spy_63d'] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
            f['corr_to_spy_63d'] = float(np.corrcoef(tk_rets.loc[common].values, sp_rets.loc[common].values)[0, 1])
        else:
            f['beta_to_spy_63d'] = 1.0
            f['corr_to_spy_63d'] = 0.5
        spy_21d = float(spy_px.iloc[-1] / spy_px.iloc[-21] - 1) if len(spy_px) > 21 else 0
        f['ret_21d_minus_spy'] = f['ret_21d'] - spy_21d
    else:
        f['beta_to_spy_63d'] = 1.0
        f['corr_to_spy_63d'] = 0.5
        f['ret_21d_minus_spy'] = 0.0

    f['vol_ratio_21_63'] = f['vol_21d'] / max(f['vol_63d'], 0.001)
    return f


# ==================== LGBM RANKING ====================

def get_rankings(fc, spy):
    """Train LGBM and rank factor ETFs."""
    import lightgbm as lgb

    idx_end = len(fc) - 1
    records = []
    start_i = max(260, idx_end - 500)
    for i in list(range(start_i, idx_end))[::20]:
        for tk in fc.columns:
            px = fc[tk].iloc[:i + 1].dropna()
            spy_px = spy.iloc[:i + 1]
            feats = compute_features(px, spy_px, fc.iloc[:i + 1])
            if not feats:
                continue
            fi = min(i + 28, len(fc) - 1)
            feats['fwd_ret'] = float(fc[tk].iloc[fi] / fc[tk].iloc[i] - 1)
            feats['date_idx'] = i
            records.append(feats)

    df = pd.DataFrame(records)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)

    if len(df) < 50:
        log.warning("Not enough training data for LGBM")
        return {}

    df['rank_label'] = df.groupby('date_idx')['fwd_ret'].rank(pct=True)
    X = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
    y = df['rank_label'].values.astype(np.float32)

    m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
    m.fit(X, y)

    current = {}
    for tk in fc.columns:
        feats = compute_features(fc[tk].dropna(), spy, fc)
        if feats:
            current[tk] = feats

    if not current:
        return {}

    pred_df = pd.DataFrame(current).T
    for c in FEAT_COLS:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    X_pred = np.nan_to_num(pred_df[FEAT_COLS].values.astype(np.float32))
    scores = m.predict(X_pred)

    ranked = sorted(zip(pred_df.index, scores), key=lambda x: x[1], reverse=True)
    return {tk: {'score': float(s), 'rank': i+1} for i, (tk, s) in enumerate(ranked)}


# ==================== MAIN ====================

def main():
    log.info("=" * 60)
    log.info("FACTOR ETF ROTATION PAPER ENGINE")
    log.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if DRY_RUN:
        log.info("[DRY RUN MODE]")
    log.info("=" * 60)

    # Load state
    state = load_state()

    # Download data
    import yfinance as yf
    all_tickers = FACTOR_ETFS + ['SPY', '^VIX']
    raw = yf.download(all_tickers, start='2020-01-01', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill().rename(columns={'^VIX': 'VIX'})

    spy = close['SPY'].dropna()
    fc = close[[c for c in FACTOR_ETFS if c in close.columns]].dropna(how='all')

    today = datetime.now().strftime('%Y-%m-%d')
    today_dt = pd.Timestamp(today)

    # Mark-to-market existing positions
    positions = state.get('positions', {})
    total_position_value = 0
    for tk, pos in positions.items():
        if tk in close.columns and today_dt in close.index:
            current_price = float(close[tk].loc[today_dt])
        elif tk in close.columns:
            current_price = float(close[tk].dropna().iloc[-1])
        else:
            current_price = pos.get('entry_price', 0)

        pos['current_price'] = current_price
        pos['current_value'] = pos['shares'] * current_price
        pos['pnl'] = pos['current_value'] - pos['shares'] * pos['entry_price']
        pos['pnl_pct'] = pos['pnl'] / (pos['shares'] * pos['entry_price']) * 100 if pos['entry_price'] > 0 else 0
        total_position_value += pos['current_value']

    cash = state['capital']
    equity = cash + total_position_value
    state['peak_equity'] = max(state.get('peak_equity', equity), equity)

    log.info(f"Equity: ${equity:.2f} (cash ${cash:.2f} + positions ${total_position_value:.2f})")
    log.info(f"Peak: ${state['peak_equity']:.2f}")
    for tk, pos in positions.items():
        log.info(f"  {tk}: {pos['shares']:.2f} shares @ ${pos['entry_price']:.2f} → ${pos['current_price']:.2f} ({pos['pnl_pct']:+.1f}%)")

    # Check trailing stop
    if state['peak_equity'] > 0:
        drawdown = 1 - equity / state['peak_equity']
        if drawdown > TRAILING_STOP and positions:
            log.info(f"TRAILING STOP HIT: {drawdown:.1%} drawdown > {TRAILING_STOP:.0%} threshold")
            for tk, pos in positions.items():
                trade = {
                    'date': today, 'action': 'SELL', 'ticker': tk,
                    'shares': pos['shares'], 'price': pos['current_price'],
                    'pnl': pos['pnl'], 'reason': 'trailing_stop',
                }
                log_trade(trade)
                log.info(f"  SELL {tk}: {pos['shares']:.2f} @ ${pos['current_price']:.2f} (PnL: ${pos['pnl']:.2f})")
                cash += pos['current_value']
            positions = {}
            state['positions'] = positions
            state['capital'] = cash
            state['peak_equity'] = cash  # Reset peak
            save_state(state)
            return

    # Check if rebalance needed
    last_rebal = state.get('last_rebalance')
    days_since = None
    if last_rebal:
        last_dt = pd.Timestamp(last_rebal)
        # Count trading days
        trading_days = fc.index[(fc.index > last_dt) & (fc.index <= fc.index[-1])]
        days_since = len(trading_days)

    should_rebalance = (last_rebal is None or (days_since is not None and days_since >= REBALANCE_DAYS))

    if not should_rebalance:
        log.info(f"No rebalance needed ({days_since} trading days since last, threshold {REBALANCE_DAYS})")
        state['positions'] = positions
        save_state(state)
        return

    log.info(f"REBALANCING (days since last: {days_since})")

    # Get rankings
    rankings = get_rankings(fc, spy)
    if not rankings:
        log.error("Could not compute rankings")
        save_state(state)
        return

    log.info("LGBM Rankings:")
    for tk, info in sorted(rankings.items(), key=lambda x: x[1]['rank']):
        log.info(f"  #{info['rank']}: {tk} (score {info['score']:.4f})")

    top_picks = sorted(rankings.keys(), key=lambda t: rankings[t]['score'], reverse=True)[:N_PICKS]
    log.info(f"Top {N_PICKS} picks: {top_picks}")

    # Sell existing positions
    for tk, pos in positions.items():
        trade = {
            'date': today, 'action': 'SELL', 'ticker': tk,
            'shares': pos['shares'], 'price': pos['current_price'],
            'pnl': pos['pnl'], 'reason': 'rebalance',
        }
        log_trade(trade)
        log.info(f"  SELL {tk}: {pos['shares']:.2f} @ ${pos['current_price']:.2f} (PnL: ${pos['pnl']:.2f})")
        cash += pos['current_value']

    # Buy new positions
    positions = {}
    per_position = cash / N_PICKS

    for tk in top_picks:
        if tk in close.columns:
            price = float(close[tk].dropna().iloc[-1])
            shares = per_position / price
            positions[tk] = {
                'shares': round(shares, 4),
                'entry_price': price,
                'entry_date': today,
                'current_price': price,
                'current_value': shares * price,
                'pnl': 0,
                'pnl_pct': 0,
            }
            trade = {
                'date': today, 'action': 'BUY', 'ticker': tk,
                'shares': round(shares, 4), 'price': price,
                'rank': rankings[tk]['rank'], 'score': rankings[tk]['score'],
            }
            log_trade(trade)
            log.info(f"  BUY {tk}: {shares:.2f} shares @ ${price:.2f} = ${shares * price:.2f}")
            cash -= shares * price

    state['positions'] = positions
    state['capital'] = round(cash, 2)
    state['last_rebalance'] = today
    state['peak_equity'] = equity
    save_state(state)

    log.info(f"Rebalance complete. Cash: ${cash:.2f}, Positions: {list(positions.keys())}")


if __name__ == '__main__':
    main()
