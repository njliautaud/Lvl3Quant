#!/usr/bin/env python3
"""
Momentum Options Paper Trading Engine (KB #281)
================================================

Single-leg options on sector ETFs using LGBM momentum ranking.
Level 2 account ($645): buy calls on top-2 bullish sectors, buy puts on bottom-2 bearish.

Key design:
  - ATM calls/puts (single leg, no spreads)
  - Quick exits: +30% TP, -25% SL, trailing stop, 5-day max hold
  - Weekly rebalance (every 5 trading days)
  - VIX > 15 filter (need vol for options to move)
  - Option pricing via simplified BS approximation
  - Commission: $0.65 per contract

Usage:
    python momentum_options_paper.py              # normal daily run
    python momentum_options_paper.py --dry-run    # simulate without state changes
"""
import json
import logging
import math
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: LightGBM not available. Using simple momentum ranking.")

# ── Paths ──
BASE = Path(__file__).resolve().parents[1]
LOG_DIR = Path(__file__).resolve().parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)
STATE_DIR = BASE / 'state'
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / 'momentum_options_paper_state.json'
TRADE_LOG = LOG_DIR / 'momentum_options_trades.jsonl'
LOG_FILE = LOG_DIR / 'momentum_options_paper.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ── DRY-RUN ──
DRY_RUN = '--dry-run' in sys.argv

# ==================== STRATEGY CONFIG ====================
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
EXTRA_TICKERS = ['SPY', '^VIX']
INITIAL_CAPITAL = 645.0
MAX_POS_COST = 150.0          # max $150 per trade
COMMISSION_PER_CONTRACT = 0.65
DTE_TARGET = 30               # buy ~30 DTE options for 2-5 day holds
REBALANCE_INTERVAL = 5        # weekly (5 trading days)
VIX_MIN = 15.0                # only trade when VIX > 15
TOP_K = 2                     # top-2 bullish sectors (calls)
BOTTOM_K = 2                  # bottom-2 bearish sectors (puts)

# Exit rules
TP_PCT = 0.30                 # +30% take profit
SL_PCT = -0.25                # -25% stop loss
TRAILING_ACTIVATE_PCT = 0.15  # activate trailing stop when up >15%
TRAILING_GIVEBACK_PCT = 0.50  # give back 50% of peak gain
MAX_HOLD_DAYS = 5             # close after 5 trading days

# LGBM features (same 17 as V10 — simplified subset for sector ranking)
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high',
    'mom_accel', 'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
]


# ==================== STATE MANAGEMENT ====================

def _default_state():
    return {
        'config_version': 'momentum_options_v1',
        'equity': INITIAL_CAPITAL,
        'cash': INITIAL_CAPITAL,
        'open_positions': [],
        'closed_trades': [],
        'last_rebalance': None,
        'days_since_rebalance': 999,
        'total_trades': 0,
        'total_pnl': 0.0,
        'wins': 0,
        'losses': 0,
        'call_wins': 0,
        'call_losses': 0,
        'put_wins': 0,
        'put_losses': 0,
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


def log_trade(record):
    if DRY_RUN:
        log.info(f"[DRY-RUN] Trade NOT logged: {record.get('action')} "
                 f"{record.get('ticker', 'N/A')} {record.get('option_type', '')}")
        return
    with open(TRADE_LOG, 'a') as f:
        f.write(json.dumps(record, default=str) + '\n')


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download sector ETF + VIX data via yfinance."""
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    raw = yf.download(all_tickers, start='2024-01-01', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    close = close.ffill()
    rename_map = {'^VIX': 'VIX'}
    close = close.rename(columns=rename_map)

    vc = 'VIX' if 'VIX' in close.columns else None
    if vc is None:
        raise ValueError("VIX data not available")

    vix = close[vc].dropna()
    spy = close['SPY'].dropna() if 'SPY' in close.columns else None
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')

    ix = sc.index.intersection(vix.index)
    if spy is not None:
        ix = ix.intersection(spy.index)

    return close.loc[ix], sc.loc[ix], vix.loc[ix]


# ==================== FEATURE ENGINEERING ====================

def compute_features(px):
    """Compute 17 momentum/quality features for a single sector ETF series."""
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


# ==================== LGBM RANKING ====================

def run_lgbm_ranking(sc):
    """LGBM walk-forward ranking using trailing data. Returns {ticker: score}."""
    if not HAS_LGBM:
        log.warning("No LightGBM -- using simple 21d momentum ranking")
        rets_21d = sc.pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    records = []
    all_dates = sc.index[-300:]
    rebal_dates = all_dates[::REBALANCE_INTERVAL]

    for dt in rebal_dates[:-1]:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx + 1].dropna()
            feats = compute_features(px)
            if not feats:
                continue
            fi = min(idx + 5, len(sc) - 1)  # 5-day forward return (matches hold horizon)
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
        log.warning(f"Not enough LGBM training data ({len(df)} rows). Fallback to momentum.")
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

    # Predict current
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


# ==================== OPTION PRICING (simplified BS) ====================

def estimate_option_price(S, option_type, vix_val, dte_days=30):
    """Estimate ATM option price using simplified BS approximation.

    ATM call/put ~= 0.4 * S * sqrt(DTE/365) * IV
    where IV = VIX/100 * 1.2 for sector ETFs (sector ETFs tend to have
    slightly higher IV than SPX).

    Returns price per share (multiply by 100 for contract price).
    """
    iv = (vix_val / 100.0) * 1.2
    t = dte_days / 365.0
    price_per_share = 0.4 * S * math.sqrt(t) * iv
    return max(price_per_share, 0.01)


def estimate_option_value(pos, current_underlying, days_held, vix_val):
    """Estimate current option value given underlying move and time decay.

    For ATM options:
      - Delta ~= 0.5 for calls, -0.5 for puts
      - Call value change = 0.5 * (S_current - S_entry)
      - Put value change = -0.5 * (S_current - S_entry)  [i.e., +0.5 * (S_entry - S_current)]
      - Theta decay: lose ~(entry_price / DTE_remaining) per day

    Returns estimated current option price per share.
    """
    entry_price_ps = pos['entry_price_option']  # per share
    S_entry = pos['entry_price_underlying']
    S_now = current_underlying
    dte_at_entry = pos['dte']
    dte_remaining = max(dte_at_entry - days_held, 1)

    # Delta P&L
    if pos['option_type'] == 'call':
        delta_pnl = 0.5 * (S_now - S_entry)
    else:  # put
        delta_pnl = 0.5 * (S_entry - S_now)

    # Theta decay: proportional to time passed
    # Option loses value as sqrt(time) decays, approximate linearly for short holds
    theta_decay = entry_price_ps * (days_held / dte_at_entry)

    current_ps = entry_price_ps + delta_pnl - theta_decay
    return max(current_ps, 0.01)  # option can't go below ~0


# ==================== POSITION MANAGEMENT ====================

def open_position(state, ticker, option_type, underlying_price, vix_val, today, lgbm_score):
    """Open a new single-leg option position."""
    option_price_ps = estimate_option_price(underlying_price, option_type, vix_val, DTE_TARGET)
    contract_cost = option_price_ps * 100 + COMMISSION_PER_CONTRACT

    if contract_cost > MAX_POS_COST:
        log.info(f"  SKIP {ticker} {option_type}: contract cost ${contract_cost:.2f} > ${MAX_POS_COST}")
        return state, False

    if contract_cost > state['cash']:
        log.info(f"  SKIP {ticker} {option_type}: insufficient cash "
                 f"(${state['cash']:.2f} < ${contract_cost:.2f})")
        return state, False

    strike = round(underlying_price)  # ATM

    position = {
        'ticker': ticker,
        'option_type': option_type,
        'entry_date': str(today.date()),
        'entry_price_option': round(option_price_ps, 4),
        'entry_price_underlying': round(underlying_price, 2),
        'strike': strike,
        'dte': DTE_TARGET,
        'cost': round(contract_cost, 2),
        'peak_value_ps': round(option_price_ps, 4),  # for trailing stop
        'trailing_active': False,
        'vix_at_entry': round(vix_val, 2),
        'lgbm_score': round(lgbm_score, 4),
    }
    state['open_positions'].append(position)
    state['cash'] -= contract_cost
    state['total_trades'] += 1

    log_trade({
        'action': 'OPEN',
        'date': str(today.date()),
        'ticker': ticker,
        'option_type': option_type,
        'strike': strike,
        'entry_price_option': round(option_price_ps, 4),
        'entry_price_underlying': round(underlying_price, 2),
        'cost': round(contract_cost, 2),
        'vix': round(vix_val, 2),
        'lgbm_score': round(lgbm_score, 4),
        'cash_after': round(state['cash'], 2),
        'equity': round(state['equity'], 2),
    })

    log.info(f"  ENTER {ticker} {option_type.upper()} strike={strike} "
             f"option=${option_price_ps:.2f}/sh (${contract_cost:.2f} total) "
             f"VIX={vix_val:.1f} LGBM={lgbm_score:.3f}")
    return state, True


def check_exits(state, sc, vix, today):
    """Check all open positions for exit conditions."""
    positions_to_close = []

    for i, pos in enumerate(state['open_positions']):
        tk = pos['ticker']
        if tk not in sc.columns:
            continue

        entry_date = pd.Timestamp(pos['entry_date'])
        days_held = len(sc.index[(sc.index > entry_date) & (sc.index <= today)])
        if days_held == 0:
            continue  # same day as entry

        current_underlying = float(sc[tk].iloc[-1])
        current_vix = float(vix.iloc[-1])
        current_ps = estimate_option_value(pos, current_underlying, days_held, current_vix)

        entry_ps = pos['entry_price_option']
        pct_change = (current_ps - entry_ps) / entry_ps

        # Update peak for trailing stop
        if current_ps > pos['peak_value_ps']:
            pos['peak_value_ps'] = round(current_ps, 4)

        # Activate trailing stop if up >15%
        if pct_change >= TRAILING_ACTIVATE_PCT:
            pos['trailing_active'] = True

        exit_reason = None

        # 1. Take profit: +30%
        if pct_change >= TP_PCT:
            exit_reason = 'take_profit'

        # 2. Stop loss: -25%
        elif pct_change <= SL_PCT:
            exit_reason = 'stop_loss'

        # 3. Trailing stop: gave back 50% of peak gain
        elif pos['trailing_active']:
            peak_ps = pos['peak_value_ps']
            peak_gain = peak_ps - entry_ps
            current_gain = current_ps - entry_ps
            if peak_gain > 0 and current_gain < peak_gain * (1 - TRAILING_GIVEBACK_PCT):
                exit_reason = 'trailing_stop'

        # 4. Max hold: 5 trading days
        elif days_held >= MAX_HOLD_DAYS:
            exit_reason = 'max_hold'

        if exit_reason:
            # P&L = (current_value - entry_cost) per share * 100 - commission
            pnl = (current_ps - entry_ps) * 100 - COMMISSION_PER_CONTRACT
            positions_to_close.append((i, pnl, exit_reason, days_held, current_ps, current_underlying))

    # Close positions (reverse order)
    for i, pnl, reason, days_held, exit_ps, exit_underlying in reversed(positions_to_close):
        pos = state['open_positions'].pop(i)
        state['equity'] += pnl
        state['cash'] += pos['cost'] + pnl  # return cost + pnl to cash
        state['total_pnl'] += pnl

        is_win = pnl > 0
        if is_win:
            state['wins'] += 1
        else:
            state['losses'] += 1

        if pos['option_type'] == 'call':
            if is_win:
                state['call_wins'] += 1
            else:
                state['call_losses'] += 1
        else:
            if is_win:
                state['put_wins'] += 1
            else:
                state['put_losses'] += 1

        pct_ret = pnl / pos['cost'] * 100

        log.info(f"  EXIT {pos['ticker']} {pos['option_type'].upper()}: "
                 f"PnL ${pnl:.2f} ({pct_ret:+.1f}%) reason={reason} held={days_held}d")

        trade_record = {
            'action': 'CLOSE',
            'date': str(today.date()),
            'ticker': pos['ticker'],
            'option_type': pos['option_type'],
            'strike': pos['strike'],
            'entry_date': pos['entry_date'],
            'days_held': days_held,
            'entry_price_option': pos['entry_price_option'],
            'exit_price_option': round(exit_ps, 4),
            'entry_price_underlying': pos['entry_price_underlying'],
            'exit_price_underlying': round(exit_underlying, 2),
            'cost': pos['cost'],
            'pnl': round(pnl, 2),
            'pct_return': round(pct_ret, 2),
            'exit_reason': reason,
            'equity_after': round(state['equity'], 2),
            'cash_after': round(state['cash'], 2),
        }
        state['closed_trades'].append(trade_record)
        log_trade(trade_record)

    return state


# ==================== MAIN DAILY RUN ====================

def run_daily():
    """Main daily run. Called once per trading day after market close."""
    state = load_state()

    if DRY_RUN:
        log.info("=" * 60)
        log.info("  DRY-RUN MODE -- no state changes will be persisted")
        log.info("=" * 60)

    log.info("=== Momentum Options Paper Engine (KB #281) ===")
    log.info(f"Equity: ${state['equity']:.2f} | Cash: ${state['cash']:.2f} | "
             f"Open: {len(state['open_positions'])} | "
             f"Trades: {state['total_trades']} | W/L: {state['wins']}/{state['losses']}")

    # Download data
    try:
        close_df, sc, vix = download_data()
    except Exception as e:
        log.error(f"Data download failed: {e}")
        return

    today = sc.index[-1]
    current_vix = float(vix.iloc[-1])
    log.info(f"Date: {today.date()} | VIX: {current_vix:.1f}")

    # ── Check exits on existing positions ──
    state = check_exits(state, sc, vix, today)

    # ── VIX filter ──
    if current_vix < VIX_MIN:
        log.info(f"VIX {current_vix:.1f} < {VIX_MIN} threshold. No new entries. Holding existing.")
        _print_summary(state)
        save_state(state)
        return

    # ── Rebalance check ──
    state['days_since_rebalance'] = state.get('days_since_rebalance', 999) + 1
    if state['days_since_rebalance'] < REBALANCE_INTERVAL:
        log.info(f"Not rebalance day ({state['days_since_rebalance']}/{REBALANCE_INTERVAL}). "
                 f"Exits only today.")
        _print_summary(state)
        save_state(state)
        return

    log.info("=== REBALANCE DAY ===")
    state['days_since_rebalance'] = 0
    state['last_rebalance'] = str(today.date())

    # ── Run LGBM ranking ──
    rankings = run_lgbm_ranking(sc)
    if not rankings:
        log.warning("No rankings available. Skipping rebalance.")
        _print_summary(state)
        save_state(state)
        return

    ranked = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    log.info(f"LGBM rankings: {', '.join(f'{t}={s:.3f}' for t, s in ranked)}")

    call_picks = [t for t, _ in ranked[:TOP_K]]
    put_picks = [t for t, _ in ranked[-BOTTOM_K:]]
    log.info(f"CALL picks (top {TOP_K}): {call_picks}")
    log.info(f"PUT picks (bottom {BOTTOM_K}): {put_picks}")

    # ── Enter calls on top sectors ──
    entries = 0
    for tk in call_picks:
        if tk not in sc.columns:
            continue
        if any(p['ticker'] == tk and p['option_type'] == 'call' for p in state['open_positions']):
            log.info(f"  SKIP {tk} call: already holding")
            continue
        S = float(sc[tk].iloc[-1])
        score = rankings.get(tk, 0)
        state, entered = open_position(state, tk, 'call', S, current_vix, today, score)
        if entered:
            entries += 1

    # ── Enter puts on bottom sectors ──
    for tk in put_picks:
        if tk not in sc.columns:
            continue
        if any(p['ticker'] == tk and p['option_type'] == 'put' for p in state['open_positions']):
            log.info(f"  SKIP {tk} put: already holding")
            continue
        S = float(sc[tk].iloc[-1])
        score = rankings.get(tk, 0)
        state, entered = open_position(state, tk, 'put', S, current_vix, today, score)
        if entered:
            entries += 1

    if entries == 0:
        log.info("No new entries this rebalance.")
    else:
        log.info(f"Opened {entries} new positions.")

    _print_summary(state)
    save_state(state)
    log.info("Done.")


def _print_summary(state):
    """Print portfolio summary."""
    total = state['wins'] + state['losses']
    wr = state['wins'] / total * 100 if total > 0 else 0
    call_total = state['call_wins'] + state['call_losses']
    put_total = state['put_wins'] + state['put_losses']
    call_wr = state['call_wins'] / call_total * 100 if call_total > 0 else 0
    put_wr = state['put_wins'] / put_total * 100 if put_total > 0 else 0

    log.info(f"\n=== Summary ===")
    log.info(f"Equity: ${state['equity']:.2f} | Cash: ${state['cash']:.2f} | "
             f"P&L: ${state['total_pnl']:.2f} ({state['total_pnl']/INITIAL_CAPITAL*100:+.1f}%)")
    log.info(f"Trades: {state['total_trades']} | W: {state['wins']} L: {state['losses']} | WR: {wr:.1f}%")
    log.info(f"  Calls: W: {state['call_wins']} L: {state['call_losses']} | WR: {call_wr:.1f}%")
    log.info(f"  Puts:  W: {state['put_wins']} L: {state['put_losses']} | WR: {put_wr:.1f}%")

    calls = [p for p in state['open_positions'] if p['option_type'] == 'call']
    puts = [p for p in state['open_positions'] if p['option_type'] == 'put']
    log.info(f"Open: {len(calls)} calls + {len(puts)} puts = {len(state['open_positions'])} total")
    for p in state['open_positions']:
        log.info(f"  {p['ticker']} {p['option_type'].upper()} strike={p['strike']} "
                 f"entry={p['entry_date']} cost=${p['cost']:.2f} "
                 f"LGBM={p.get('lgbm_score', 'N/A')} "
                 f"trailing={'ON' if p.get('trailing_active') else 'off'}")


if __name__ == '__main__':
    run_daily()
