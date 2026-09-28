#!/usr/bin/env python3
"""
PEAD ML Paper Trading Engine
==============================

ML-scored post-earnings announcement drift options paper trader.
Uses pre-trained LGBM model (Variant D) to predict which earnings gaps
will continue drifting, then paper-trades ATM options in the gap direction.

STRATEGY (Backtest: Sharpe 1.513, Sortino 9.23, PF 3.64, WR 52.4%):
  - Signal: Stock gaps 5%+ after earnings + ML confidence >= 60%
  - Filter: 5d momentum must align with gap direction
  - Entry: ATM option in gap direction, 14 DTE
  - Exit: +30% TP, -25% SL, 50% trailing giveback, 3-day max hold
  - Capital: $645 paper (mirrors agentic account)
  - Max $200 per position, max 2 concurrent

USAGE:
  python3 pead_ml_paper.py                # normal daily run
  python3 pead_ml_paper.py --check-now    # force check (skip weekday/hour guard)
  python3 pead_ml_paper.py --dry-run      # simulate without state changes
  python3 pead_ml_paper.py --summary      # print current state summary
  python3 pead_ml_paper.py --train        # retrain the ML model

CRON: Run at 9:40 AM ET weekdays (after 9:35 earnings gap alert)
"""

import json
import logging
import math
import os
import pickle
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──
BASE = Path(__file__).resolve().parents[1]
LOG_DIR = Path(__file__).resolve().parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)
STATE_DIR = BASE / 'state'
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / 'pead_ml_paper_state.json'
TRADE_LOG = LOG_DIR / 'pead_ml_trades.jsonl'
LOG_FILE = LOG_DIR / 'pead_ml_paper.log'
MODEL_DIR = BASE / 'models' / 'pead_ml'
MODEL_PATH = MODEL_DIR / 'pead_lgbm_latest.pkl'
META_PATH = MODEL_DIR / 'pead_model_meta.json'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ── Flags ──
DRY_RUN = '--dry-run' in sys.argv
CHECK_NOW = '--check-now' in sys.argv
SUMMARY_ONLY = '--summary' in sys.argv
RETRAIN = '--train' in sys.argv

# ==================== STRATEGY CONFIG ====================

STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI',
]

# Tickers with historically negative PEAD (avoid list from backtest)
AVOID_TICKERS = {'LYFT', 'JD', 'PINS', 'RBLX', 'SOFI'}

INITIAL_CAPITAL = 645.0
MAX_POSITION_COST = 200.0       # max $200 per trade (HC from adversarial validation)
MAX_CONCURRENT = 2              # max 2 concurrent trades
COMMISSION_PER_CONTRACT = 0.65  # RH options commission
DTE_TARGET = 14                 # 14 DTE for post-earnings
MIN_GAP_PCT = 5.0               # minimum 5% gap to trigger
CONFIDENCE_THRESHOLD = 0.60     # ML model confidence threshold
HOLD_DAYS_MAX = 3               # max 3 trading days hold
DRIFT_THRESHOLD = 0.03          # 3% drift target (model training param)

# Exit rules
TP_PCT = 0.30                   # +30% take profit on option
SL_PCT = -0.25                  # -25% stop loss on option
TRAILING_ACTIVATE_PCT = 0.15    # activate trailing stop at +15%
TRAILING_GIVEBACK_PCT = 0.50    # give back 50% of peak gain

# ML feature columns (must match scorer)
FEATURE_COLS = [
    'abs_gap', 'gap_direction', 'mom_5d', 'mom_10d', 'mom_21d',
    'vol_21d', 'rel_str_21d', 'vix', 'vol_ratio', 'rsi',
    'dist_52w_high', 'prev_gap', 'ticker_hash'
]


# ==================== BLACK-SCHOLES ====================

from scipy.stats import norm

RISK_FREE_RATE = 0.05

def bs_call(S, K, T, r, sigma):
    if T <= 1e-8: return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 1e-8: return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def option_price(S, K, T, r, sigma, opt_type='call'):
    return bs_call(S, K, T, r, sigma) if opt_type == 'call' else bs_put(S, K, T, r, sigma)

def estimate_current_option_price(entry_price, current_price, strike, opt_type,
                                   entry_iv, days_elapsed, dte_at_entry):
    """Estimate current option price given stock move and time decay."""
    remaining_dte = max(dte_at_entry - days_elapsed, 0.5)
    T = remaining_dte / 252.0
    # IV crush after earnings — assume 20% drop from entry IV
    current_iv = entry_iv * 0.85
    return option_price(current_price, strike, T, RISK_FREE_RATE, current_iv, opt_type)


# ==================== STATE MANAGEMENT ====================

def _default_state():
    return {
        'config_version': 'pead_ml_v1',
        'equity': INITIAL_CAPITAL,
        'cash': INITIAL_CAPITAL,
        'open_positions': [],
        'closed_trades': [],
        'last_scan': None,
        'total_trades': 0,
        'total_pnl': 0.0,
        'wins': 0,
        'losses': 0,
        'created': datetime.now().isoformat(),
    }


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        # Backfill any missing keys
        defaults = _default_state()
        for k, v in defaults.items():
            if k not in state:
                state[k] = v
        return state
    return _default_state()


def save_state(state):
    state['last_scan'] = datetime.now().isoformat()
    if not DRY_RUN:
        with open(STATE_PATH, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        log.info("State saved.")
    else:
        log.info("[DRY RUN] State NOT saved.")


def log_trade(trade_record):
    """Append trade to JSONL log."""
    if not DRY_RUN:
        with open(TRADE_LOG, 'a') as f:
            f.write(json.dumps(trade_record, default=str) + '\n')


# ==================== ML MODEL ====================

def load_model():
    """Load the pre-trained PEAD LGBM model."""
    if not MODEL_PATH.exists():
        log.error(f"No trained model found at {MODEL_PATH}. Run --train first.")
        return None
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)


def extract_features_live(ticker, df, spy_close=None, vix_close=None, gap_pct=None):
    """Extract ML features for a single ticker from current price data."""
    feats = {}

    feats['abs_gap'] = abs(gap_pct) if gap_pct is not None else 0.05
    feats['gap_direction'] = 1 if (gap_pct is not None and gap_pct > 0) else -1

    close = df['close'] if 'close' in df.columns else df['Close']

    # Momentum features
    for w in [5, 10, 21]:
        if len(close) >= w + 1:
            feats[f'mom_{w}d'] = float(close.iloc[-1] / close.iloc[-w-1] - 1)
        else:
            feats[f'mom_{w}d'] = 0

    # Volatility
    if len(close) >= 22:
        feats['vol_21d'] = float(close.iloc[-22:].pct_change().dropna().std() * np.sqrt(252))
    else:
        feats['vol_21d'] = 0.3

    # Relative strength vs SPY
    if spy_close is not None and len(close) >= 22:
        try:
            spy_recent = spy_close.iloc[-21:]
            stock_recent = close.iloc[-21:]
            feats['rel_str_21d'] = float(
                (stock_recent.iloc[-1] / stock_recent.iloc[0] - 1) -
                (spy_recent.iloc[-1] / spy_recent.iloc[0] - 1))
        except:
            feats['rel_str_21d'] = 0
    else:
        feats['rel_str_21d'] = 0

    # VIX
    feats['vix'] = float(vix_close.iloc[-1]) if vix_close is not None and len(vix_close) > 0 else 20

    # Volume ratio (recent vs average)
    vol_col = 'volume' if 'volume' in df.columns else 'Volume'
    if vol_col in df.columns and len(df) >= 22:
        feats['vol_ratio'] = float(df[vol_col].iloc[-5:].mean() / max(df[vol_col].iloc[-22:].mean(), 1))
    else:
        feats['vol_ratio'] = 1.0

    # RSI (14-day)
    if len(close) >= 15:
        rets = close.iloc[-15:].pct_change().dropna()
        gains = rets.clip(lower=0).mean()
        losses = (-rets).clip(lower=0).mean()
        feats['rsi'] = float(100 - 100 / (1 + gains/losses)) if losses > 0 else 100.0
    else:
        feats['rsi'] = 50

    # Distance from 52w high
    if len(close) >= 252:
        h52 = close.iloc[-252:].max()
        feats['dist_52w_high'] = float(close.iloc[-1] / h52 - 1)
    else:
        feats['dist_52w_high'] = 0

    feats['prev_gap'] = 0
    feats['ticker_hash'] = hash(ticker) % 100

    return feats


def score_earnings_gap(ticker, gap_pct, model, df, spy_close=None, vix_close=None):
    """
    Score a ticker's earnings gap using the ML model.
    Returns (confidence, momentum_aligned, trade_recommended, features).
    """
    feats = extract_features_live(ticker, df, spy_close, vix_close, gap_pct)

    X = pd.DataFrame([feats])[FEATURE_COLS]
    prob = float(model.predict_proba(X)[0][1])

    # Momentum filter (variant D requirement)
    momentum_aligned = True
    if gap_pct > 0 and feats['mom_5d'] < 0:
        momentum_aligned = False
    if gap_pct < 0 and feats['mom_5d'] > 0:
        momentum_aligned = False

    # Trade decision
    trade = (
        prob >= CONFIDENCE_THRESHOLD
        and momentum_aligned
        and abs(gap_pct) >= MIN_GAP_PCT / 100.0
        and ticker not in AVOID_TICKERS
    )

    return prob, momentum_aligned, trade, feats


# ==================== MARKET DATA ====================

def get_market_data():
    """Download recent prices for universe + SPY + VIX."""
    import yfinance as yf

    all_tickers = STOCK_UNIVERSE + ['SPY', '^VIX']
    data = {}

    for ticker in all_tickers:
        try:
            df = yf.download(ticker, period='1y', progress=False, auto_adjust=True)
            if len(df) < 30:
                continue
            # Normalize columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            data[ticker] = df
        except:
            continue

    return data


def check_earnings_gaps(data):
    """
    Check each stock for a recent earnings gap (within last 2 trading days).
    Returns list of (ticker, gap_pct, earnings_date) for stocks that gapped 5%+.
    """
    import yfinance as yf

    gaps = []
    today = datetime.now()

    for ticker in STOCK_UNIVERSE:
        if ticker not in data:
            continue

        try:
            stock = yf.Ticker(ticker)
            earnings = stock.get_earnings_dates(limit=5)
            if earnings is None or len(earnings) == 0:
                continue

            df = data[ticker]

            for earn_date in earnings.index:
                # Handle timezone
                if hasattr(earn_date, 'tzinfo') and earn_date.tzinfo is not None:
                    edate = pd.Timestamp(earn_date).tz_convert(None)
                else:
                    edate = pd.Timestamp(earn_date)

                days_ago = (today - edate).days

                # Only look at earnings from yesterday or today
                if days_ago < 0 or days_ago > 2:
                    continue

                # Find the gap
                loc = df.index.searchsorted(edate)
                if loc < 1 or loc >= len(df):
                    continue

                close_before = float(df['close'].iloc[loc - 1])
                open_after = float(df['open'].iloc[loc])

                if close_before <= 0:
                    continue

                gap_pct = (open_after / close_before - 1)

                if abs(gap_pct) >= MIN_GAP_PCT / 100.0:
                    gaps.append({
                        'ticker': ticker,
                        'gap_pct': gap_pct,
                        'close_before': close_before,
                        'open_after': open_after,
                        'earnings_date': str(edate.date()),
                        'current_price': float(df['close'].iloc[-1]),
                    })
                    log.info(f"  EARNINGS GAP: {ticker} {gap_pct*100:+.1f}% on {edate.date()}")
                break  # Only check most recent earnings per ticker

        except Exception as e:
            log.debug(f"  {ticker}: earnings check failed ({e})")
            continue

    return gaps


# ==================== POSITION MANAGEMENT ====================

def check_exits(state, data):
    """Check open positions for exit conditions."""
    if not state['open_positions']:
        return state

    remaining = []
    today = datetime.now()

    for pos in state['open_positions']:
        ticker = pos['ticker']
        entry_date = datetime.fromisoformat(pos['entry_date'])
        days_held = np.busday_count(entry_date.date(), today.date())

        if ticker not in data:
            remaining.append(pos)
            continue

        current_price = float(data[ticker]['close'].iloc[-1])

        # Estimate current option value
        current_opt_price = estimate_current_option_price(
            entry_price=pos['entry_stock_price'],
            current_price=current_price,
            strike=pos['strike'],
            opt_type=pos['option_type'],
            entry_iv=pos['entry_iv'],
            days_elapsed=days_held,
            dte_at_entry=DTE_TARGET,
        )
        current_contract_value = current_opt_price * 100
        entry_cost = pos['entry_premium'] * 100

        pnl_pct = (current_contract_value / entry_cost - 1) if entry_cost > 0 else 0
        peak_pnl = pos.get('peak_pnl_pct', pnl_pct)

        # Update peak
        if pnl_pct > peak_pnl:
            pos['peak_pnl_pct'] = pnl_pct
            peak_pnl = pnl_pct

        exit_reason = None

        # Check exit conditions
        if pnl_pct >= TP_PCT:
            exit_reason = 'TP'
        elif pnl_pct <= SL_PCT:
            exit_reason = 'SL'
        elif peak_pnl >= TRAILING_ACTIVATE_PCT and pnl_pct <= peak_pnl * (1 - TRAILING_GIVEBACK_PCT):
            exit_reason = 'TRAIL'
        elif days_held >= HOLD_DAYS_MAX:
            exit_reason = 'TIME'

        if exit_reason:
            pnl_dollar = current_contract_value - entry_cost - COMMISSION_PER_CONTRACT * 2

            trade_record = {
                **pos,
                'exit_date': today.isoformat(),
                'exit_stock_price': current_price,
                'exit_premium': round(current_opt_price, 2),
                'exit_contract_value': round(current_contract_value, 2),
                'pnl_pct': round(pnl_pct * 100, 2),
                'pnl_dollar': round(pnl_dollar, 2),
                'days_held': days_held,
                'exit_reason': exit_reason,
            }

            state['closed_trades'].append(trade_record)
            state['cash'] += entry_cost + pnl_dollar  # return capital + P&L
            state['equity'] += pnl_dollar
            state['total_trades'] += 1
            state['total_pnl'] += pnl_dollar

            if pnl_dollar > 0:
                state['wins'] += 1
            else:
                state['losses'] += 1

            log_trade(trade_record)

            direction = 'UP' if pos['gap_pct'] > 0 else 'DOWN'
            log.info(f"  CLOSED: {ticker} {pos['option_type'].upper()} ${pos['strike']} | "
                     f"Gap {pos['gap_pct']*100:+.1f}% {direction} | "
                     f"P&L {pnl_pct*100:+.1f}% (${pnl_dollar:+.2f}) | "
                     f"{days_held}d | {exit_reason}")
        else:
            pos['current_pnl_pct'] = round(pnl_pct * 100, 2)
            remaining.append(pos)

    state['open_positions'] = remaining
    return state


def open_positions(state, gaps, model, data):
    """Open new positions for qualifying earnings gaps."""
    if len(state['open_positions']) >= MAX_CONCURRENT:
        log.info(f"  Max concurrent positions ({MAX_CONCURRENT}) reached. Skipping new entries.")
        return state

    spy_close = data.get('SPY', {}).get('close') if 'SPY' in data else None
    if spy_close is None and 'SPY' in data:
        spy_close = data['SPY']['close']

    vix_close = data.get('^VIX', {}).get('close') if '^VIX' in data else None
    if vix_close is None and '^VIX' in data:
        vix_close = data['^VIX']['close']

    # Skip tickers we already hold
    held_tickers = {p['ticker'] for p in state['open_positions']}

    for gap_info in gaps:
        if len(state['open_positions']) >= MAX_CONCURRENT:
            break

        ticker = gap_info['ticker']
        if ticker in held_tickers:
            continue

        if ticker not in data:
            continue

        gap_pct = gap_info['gap_pct']

        # Score with ML model
        prob, momentum_aligned, trade_recommended, feats = score_earnings_gap(
            ticker, gap_pct, model, data[ticker], spy_close, vix_close
        )

        confidence_label = 'HIGH' if prob >= 0.70 else ('MEDIUM' if prob >= 0.60 else 'LOW')

        log.info(f"  ML SCORE: {ticker} | Confidence: {prob:.1%} ({confidence_label}) | "
                 f"Momentum: {'Aligned' if momentum_aligned else 'Misaligned'} | "
                 f"Trade: {'YES' if trade_recommended else 'NO'}")

        if not trade_recommended:
            continue

        # Price the option
        current_price = gap_info['current_price']
        opt_type = 'call' if gap_pct > 0 else 'put'
        strike = round(current_price)  # ATM
        entry_iv = feats['vol_21d'] * 0.80  # post-earnings IV crush estimate
        entry_iv = max(entry_iv, 0.15)
        T = DTE_TARGET / 252.0
        premium = option_price(current_price, strike, T, RISK_FREE_RATE, entry_iv, opt_type)
        contract_cost = premium * 100 + COMMISSION_PER_CONTRACT

        # Affordability check
        if contract_cost > MAX_POSITION_COST:
            log.info(f"  SKIP: {ticker} too expensive (${contract_cost:.0f} > ${MAX_POSITION_COST})")
            continue

        if contract_cost > state['cash']:
            log.info(f"  SKIP: {ticker} insufficient cash (${state['cash']:.0f} < ${contract_cost:.0f})")
            continue

        # Open position
        pos = {
            'ticker': ticker,
            'option_type': opt_type,
            'strike': strike,
            'dte': DTE_TARGET,
            'entry_date': datetime.now().isoformat(),
            'entry_stock_price': current_price,
            'entry_premium': round(premium, 2),
            'entry_contract_cost': round(contract_cost, 2),
            'entry_iv': round(entry_iv, 4),
            'gap_pct': round(gap_pct, 4),
            'ml_confidence': round(prob, 4),
            'confidence_label': confidence_label,
            'momentum_aligned': momentum_aligned,
            'earnings_date': gap_info['earnings_date'],
            'peak_pnl_pct': 0,
            'current_pnl_pct': 0,
            'features': {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
                        for k, v in feats.items()},
        }

        state['open_positions'].append(pos)
        state['cash'] -= contract_cost
        held_tickers.add(ticker)

        direction = 'CALL (gap up)' if gap_pct > 0 else 'PUT (gap down)'
        log.info(f"  OPENED: {ticker} {direction} ${strike} @ ${premium:.2f} | "
                 f"ML: {prob:.1%} | Gap: {gap_pct*100:+.1f}% | "
                 f"Cost: ${contract_cost:.2f}")

    return state


# ==================== SUMMARY ====================

def print_summary(state):
    """Print current state summary."""
    n_closed = state['total_trades']
    wr = state['wins'] / n_closed * 100 if n_closed > 0 else 0

    print(f"\n{'='*60}")
    print(f"  PEAD ML Paper Engine — Summary")
    print(f"{'='*60}")
    print(f"  Equity:     ${state['equity']:,.2f} ({(state['equity']/INITIAL_CAPITAL-1)*100:+.1f}%)")
    print(f"  Cash:       ${state['cash']:,.2f}")
    print(f"  Open:       {len(state['open_positions'])}")
    print(f"  Closed:     {n_closed} (W: {state['wins']}, L: {state['losses']}, WR: {wr:.0f}%)")
    print(f"  Total P&L:  ${state['total_pnl']:+,.2f}")
    print(f"  Last scan:  {state.get('last_scan', 'never')}")

    if state['open_positions']:
        print(f"\n  Open Positions:")
        for pos in state['open_positions']:
            days = np.busday_count(
                datetime.fromisoformat(pos['entry_date']).date(),
                datetime.now().date()
            )
            print(f"    {pos['ticker']} {pos['option_type'].upper()} ${pos['strike']} | "
                  f"ML: {pos['ml_confidence']:.0%} | Gap: {pos['gap_pct']*100:+.1f}% | "
                  f"Day {days}/{HOLD_DAYS_MAX} | P&L: {pos.get('current_pnl_pct', 0):+.1f}%")

    if state['closed_trades']:
        recent = state['closed_trades'][-5:]
        print(f"\n  Recent Trades:")
        for t in recent:
            print(f"    {t['ticker']} {t['option_type'].upper()} | "
                  f"P&L: {t['pnl_pct']:+.1f}% (${t['pnl_dollar']:+.2f}) | "
                  f"{t['days_held']}d | {t['exit_reason']}")

    print(f"{'='*60}\n")


# ==================== MAIN ====================

def main():
    log.info("=" * 50)
    log.info("PEAD ML Paper Engine — Daily Run")
    log.info("=" * 50)

    state = load_state()

    if SUMMARY_ONLY:
        print_summary(state)
        return

    if RETRAIN:
        # Import and run the trainer
        sys.path.insert(0, str(BASE / 'scripts' / 'growth_research'))
        from pead_ml_live_scorer import train_and_save_model
        train_and_save_model()
        return

    # Time guard (only run during market hours on weekdays unless --check-now)
    now = datetime.now()
    if not CHECK_NOW:
        if now.weekday() >= 5:
            log.info(f"Weekend ({now.strftime('%A')}), skipping.")
            save_state(state)
            return

    # Load ML model
    model = load_model()
    if model is None:
        log.error("No ML model available. Run with --train first.")
        return

    log.info(f"Model loaded. Equity: ${state['equity']:.2f}, "
             f"Open: {len(state['open_positions'])}, Cash: ${state['cash']:.2f}")

    # Get market data
    log.info("Fetching market data...")
    data = get_market_data()
    log.info(f"  Got data for {len(data)} tickers")

    # 1. Check exits first
    log.info("Checking exits...")
    state = check_exits(state, data)

    # 2. Scan for new earnings gaps
    log.info("Scanning for earnings gaps...")
    gaps = check_earnings_gaps(data)

    if gaps:
        log.info(f"  Found {len(gaps)} qualifying gaps (>= {MIN_GAP_PCT}%)")
        # Sort by gap size (stronger surprise = stronger drift)
        gaps.sort(key=lambda x: abs(x['gap_pct']), reverse=True)
        state = open_positions(state, gaps, model, data)
    else:
        log.info("  No qualifying earnings gaps today")

    # Save state
    save_state(state)

    # Print summary
    print_summary(state)

    log.info("Done.")


if __name__ == '__main__':
    main()
