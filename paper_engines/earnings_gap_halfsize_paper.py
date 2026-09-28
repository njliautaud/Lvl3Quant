#!/usr/bin/env python3
"""
Earnings Gap + 40-Day Hold + Half-Size in Bears (AnalystC_H1) Paper Engine
===========================================================================

FIRST strategy to pass ALL 5 validation gates across 80+ strategies tested.

STRATEGY RULES:
  - Scan ~50 large-cap stocks daily for >3% gap-ups at open vs previous close
  - If a stock gaps >3% (earnings beat proxy), BUY at the open price
  - Hold for 40 trading days, then sell at close
  - Position sizing: $500/position when SPY > 200-SMA (bull), $250 (bear)
  - Max 5 simultaneous positions
  - Track regime at entry for analytics

Usage:
  python3 paper_engines/earnings_gap_halfsize_paper.py
"""

import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Logging ──
LOG_DIR = Path(__file__).resolve().parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'earnings_gap_halfsize_paper.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── State ──
STATE_DIR = Path(__file__).resolve().parent.parent / 'state'
STATE_DIR.mkdir(exist_ok=True)
STATE_PATH = STATE_DIR / 'earnings_gap_halfsize_paper_state.json'

# ── Strategy Parameters ──
MAX_CONCURRENT = 5
HOLD_DAYS = 40           # trading days
MIN_GAP_PCT = 3.0        # >3% gap-up required
BULL_POSITION_SIZE = 500  # dollars per position in bull regime
BEAR_POSITION_SIZE = 250  # half-size in bear regime
SMA_PERIOD = 200          # SPY 200-day SMA for regime detection

UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'V', 'JPM', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK',
    'ABBV', 'PEP', 'COST', 'KO', 'AVGO', 'LLY', 'WMT', 'TMO', 'MCD',
    'CSCO', 'ACN', 'DHR', 'ABT', 'NEE', 'TXN', 'PM', 'UNP', 'RTX',
    'LOW', 'HON', 'AMGN', 'IBM', 'CAT', 'GS', 'BA', 'SBUX', 'GE',
    'MMM', 'DIS', 'INTC', 'NKE', 'CRM'
]


def load_state():
    """Load or initialize paper trading state."""
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        'positions': [],
        'closed_trades': [],
        'total_invested': 0.0,
        'total_realized_pnl': 0.0,
        'created': str(datetime.now()),
        'last_run': None,
        'strategy': 'AnalystC_H1: Earnings Gap + 40d Hold + Half-Size Bears',
    }


def save_state(state):
    """Persist state to disk."""
    state['last_run'] = str(datetime.now())
    with open(STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def get_spy_regime():
    """Determine bull/bear regime: SPY vs 200-SMA.

    Returns:
        tuple: (regime_str, spy_price, sma_200_value)
    """
    import yfinance as yf

    try:
        spy = yf.download('SPY', period='1y', progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)

        if len(spy) < SMA_PERIOD:
            log.warning(f"Not enough SPY data ({len(spy)} bars) for {SMA_PERIOD}-SMA")
            return 'BULL', 0.0, 0.0  # default to bull if insufficient data

        close = spy['Close']
        sma_200 = float(close.rolling(SMA_PERIOD).mean().iloc[-1])
        current_price = float(close.iloc[-1])

        regime = 'BULL' if current_price > sma_200 else 'BEAR'
        return regime, round(current_price, 2), round(sma_200, 2)

    except Exception as e:
        log.error(f"Error fetching SPY regime: {e}")
        return 'BULL', 0.0, 0.0  # default to bull on error


def scan_gap_ups(held_tickers, recently_closed_tickers):
    """Scan universe for >3% gap-ups today.

    A gap-up is defined as today's Open being >3% above yesterday's Close.

    Returns:
        list of dicts with signal info
    """
    import yfinance as yf

    skip = held_tickers | recently_closed_tickers
    candidates = []

    for ticker in UNIVERSE:
        if ticker in skip:
            continue

        try:
            # Get last 5 trading days to detect today's gap
            df = yf.download(ticker, period='5d', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if len(df) < 2:
                continue

            prev_close = float(df['Close'].iloc[-2])
            today_open = float(df['Open'].iloc[-1])
            today_close = float(df['Close'].iloc[-1])

            if prev_close <= 0:
                continue

            gap_pct = (today_open / prev_close - 1) * 100

            if gap_pct >= MIN_GAP_PCT:
                candidates.append({
                    'ticker': ticker,
                    'prev_close': round(prev_close, 2),
                    'open_price': round(today_open, 2),
                    'current_price': round(today_close, 2),
                    'gap_pct': round(gap_pct, 2),
                    'date': str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1]),
                })
                log.info(f"  GAP-UP: {ticker} +{gap_pct:.1f}% "
                         f"(prev close ${prev_close:.2f} -> open ${today_open:.2f})")

        except Exception as e:
            log.warning(f"  {ticker}: Error scanning ({e})")

    # Sort by gap magnitude (strongest signals first)
    candidates.sort(key=lambda x: x['gap_pct'], reverse=True)
    return candidates


def process_exits(state, today):
    """Close positions that have reached their 40-day hold period."""
    import yfinance as yf

    remaining = []
    for pos in state['positions']:
        entry_date = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
        bdays = int(np.busday_count(entry_date.date(), today.date()))

        if bdays >= HOLD_DAYS:
            # Get current price for exit
            try:
                df = yf.download(pos['ticker'], period='5d', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                exit_price = float(df['Close'].iloc[-1])
            except Exception:
                exit_price = pos['entry_price']

            # Calculate P&L
            shares = pos['shares']
            entry_cost = pos['entry_price'] * shares
            exit_value = exit_price * shares
            pnl_dollar = exit_value - entry_cost
            commission = entry_cost * 0.001  # 10 bps RT estimate
            net_pnl = pnl_dollar - commission
            return_pct = (exit_price / pos['entry_price'] - 1) * 100

            closed = {
                **pos,
                'exit_date': today.strftime('%Y-%m-%d'),
                'exit_price': round(exit_price, 2),
                'return_pct': round(return_pct, 2),
                'pnl_dollar': round(net_pnl, 2),
                'days_held': bdays,
            }
            state['closed_trades'].append(closed)
            state['total_realized_pnl'] += net_pnl

            log.info(f"CLOSED: {pos['ticker']} | {return_pct:+.1f}% | "
                     f"${net_pnl:+.2f} | Entry ${pos['entry_price']} -> Exit ${exit_price:.2f} | "
                     f"{bdays}d held | Regime at entry: {pos.get('regime', 'N/A')}")
        else:
            remaining.append(pos)

    state['positions'] = remaining


def open_positions(state, candidates, regime, today):
    """Open new paper positions for qualifying gap-up stocks."""
    slots = MAX_CONCURRENT - len(state['positions'])
    if slots <= 0:
        log.info(f"  No open slots ({MAX_CONCURRENT} positions full)")
        return

    position_size = BULL_POSITION_SIZE if regime == 'BULL' else BEAR_POSITION_SIZE

    for cand in candidates[:slots]:
        entry_price = cand['open_price']  # buy at open price (gap price)
        if entry_price <= 0:
            continue

        shares = position_size / entry_price  # fractional shares OK for paper
        target_exit = today + timedelta(days=int(HOLD_DAYS * 1.5))  # calendar day estimate

        pos = {
            'ticker': cand['ticker'],
            'entry_date': today.strftime('%Y-%m-%d'),
            'entry_price': entry_price,
            'shares': round(shares, 4),
            'position_size_dollar': position_size,
            'gap_pct': cand['gap_pct'],
            'prev_close': cand['prev_close'],
            'regime': regime,
            'target_exit_date': str(target_exit.date()),
        }
        state['positions'].append(pos)
        state['total_invested'] += position_size

        log.info(f"OPENED: {cand['ticker']} at ${entry_price} | "
                 f"Gap +{cand['gap_pct']:.1f}% | "
                 f"Size: ${position_size} ({regime}) | "
                 f"Shares: {shares:.4f} | "
                 f"Target exit: ~{HOLD_DAYS} trading days")


def get_current_prices(positions):
    """Fetch current prices for all held positions."""
    import yfinance as yf

    prices = {}
    tickers = [p['ticker'] for p in positions]
    if not tickers:
        return prices

    try:
        data = yf.download(tickers if len(tickers) > 1 else tickers[0],
                           period='1d', progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            if len(tickers) == 1:
                prices[tickers[0]] = float(data['Close'].iloc[-1])
            else:
                for t in tickers:
                    try:
                        prices[t] = float(data['Close'][t].iloc[-1])
                    except Exception:
                        pass
        else:
            if len(tickers) == 1:
                prices[tickers[0]] = float(data['Close'].iloc[-1])
    except Exception as e:
        log.warning(f"Error fetching current prices: {e}")

    return prices


def print_summary(state, regime, spy_price, sma_200, today):
    """Print a comprehensive summary of the paper portfolio."""
    # Closed trade stats
    closed = state['closed_trades']
    n_closed = len(closed)
    n_wins = sum(1 for t in closed if t.get('pnl_dollar', 0) > 0)
    wr = n_wins / n_closed * 100 if n_closed > 0 else 0
    total_realized = state['total_realized_pnl']

    # Get unrealized P&L
    current_prices = get_current_prices(state['positions'])
    total_unrealized = 0.0

    log.info("")
    log.info("=" * 60)
    log.info("  AnalystC_H1: Earnings Gap + 40d Hold + Half-Size Bears")
    log.info("=" * 60)
    log.info(f"  Regime: {regime} (SPY ${spy_price} vs 200-SMA ${sma_200})")
    log.info(f"  Position size: ${BULL_POSITION_SIZE if regime == 'BULL' else BEAR_POSITION_SIZE}"
             f" ({'full' if regime == 'BULL' else 'HALF'})")
    log.info(f"  Open positions: {len(state['positions'])}/{MAX_CONCURRENT}")
    log.info(f"  Closed trades: {n_closed} | WR: {wr:.0f}%")
    log.info(f"  Realized P&L: ${total_realized:+.2f}")
    log.info("")

    if state['positions']:
        log.info("  OPEN POSITIONS:")
        log.info(f"  {'Ticker':<8} {'Entry':>8} {'Current':>8} {'P&L%':>7} {'P&L$':>8} "
                 f"{'Day':>6} {'Gap%':>6} {'Regime':>6}")
        log.info(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")

        for pos in state['positions']:
            entry_date = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
            bdays = int(np.busday_count(entry_date.date(), today.date()))

            cur_price = current_prices.get(pos['ticker'], pos['entry_price'])
            pnl_pct = (cur_price / pos['entry_price'] - 1) * 100
            pnl_dollar = pos['shares'] * (cur_price - pos['entry_price'])
            total_unrealized += pnl_dollar

            log.info(f"  {pos['ticker']:<8} {pos['entry_price']:>8.2f} {cur_price:>8.2f} "
                     f"{pnl_pct:>+6.1f}% {pnl_dollar:>+7.2f} "
                     f"{bdays:>3}/{HOLD_DAYS:<2} {pos['gap_pct']:>+5.1f}% {pos.get('regime', '?'):>6}")

        log.info("")
        log.info(f"  Unrealized P&L: ${total_unrealized:+.2f}")

    log.info(f"  Total P&L (realized + unrealized): ${total_realized + total_unrealized:+.2f}")
    log.info("=" * 60)

    # Calculate and show Sharpe/Sortino if we have enough closed trades
    if n_closed >= 5:
        returns = [t.get('return_pct', 0) for t in closed]
        avg_ret = np.mean(returns)
        std_ret = np.std(returns)
        downside = np.std([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0.001

        # Annualize assuming ~9 trades/year (conservative)
        sharpe = (avg_ret / std_ret) if std_ret > 0 else 0
        sortino = (avg_ret / downside) if downside > 0 else 0
        pf_wins = sum(r for r in returns if r > 0)
        pf_losses = abs(sum(r for r in returns if r < 0))
        profit_factor = pf_wins / pf_losses if pf_losses > 0 else float('inf')

        log.info(f"  Performance (per-trade): Sharpe {sharpe:.2f} | Sortino {sortino:.2f} | "
                 f"PF {profit_factor:.2f} | Avg {avg_ret:+.1f}%")


def main():
    import yfinance as yf

    log.info("=" * 60)
    log.info("AnalystC_H1 Paper Engine — Daily Run")
    log.info("=" * 60)

    state = load_state()
    today = datetime.now()

    # Skip weekends
    if today.weekday() >= 5:
        log.info(f"Weekend ({today.strftime('%A')}), skipping")
        save_state(state)
        return

    # 1. Determine market regime
    regime, spy_price, sma_200 = get_spy_regime()
    log.info(f"Regime: {regime} | SPY ${spy_price} vs 200-SMA ${sma_200}")

    # 2. Process exits (40-day holds)
    process_exits(state, today)

    # 3. Scan for new gap-up signals
    if len(state['positions']) < MAX_CONCURRENT:
        log.info(f"\nScanning for >3% gap-ups ({MAX_CONCURRENT - len(state['positions'])} slots open)...")

        held_tickers = set(p['ticker'] for p in state['positions'])
        # Don't re-enter recently closed positions (within 45 days)
        recently_closed = set(
            t['ticker'] for t in state['closed_trades']
            if (today - datetime.strptime(t['exit_date'], '%Y-%m-%d')).days < 45
        )

        candidates = scan_gap_ups(held_tickers, recently_closed)

        if candidates:
            open_positions(state, candidates, regime, today)
        else:
            log.info("  No >3% gap-ups today")
    else:
        log.info(f"All {MAX_CONCURRENT} slots full, skipping scan")

    # 4. Print summary
    print_summary(state, regime, spy_price, sma_200, today)

    # 5. Save state
    save_state(state)
    log.info("State saved. Done.")


if __name__ == '__main__':
    main()
