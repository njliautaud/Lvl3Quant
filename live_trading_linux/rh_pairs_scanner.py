#!/usr/bin/env python3
"""
Pairs Trading Mean-Reversion Scanner for Robinhood Account
Run daily to check for z-score divergence signals across correlated stock pairs.

Strategy: Buy the underperformer when the pair's price ratio z-score drops below -2.0.
Exit when z-score reverts to -0.5. Based on pairs_meanrev_v1 backtest:
  Sharpe 1.32, WR 65.5%, PF 3.03, p=0.000, regime-agnostic (9/10 gates passed).

Account: Robinhood cash account, $440 buying power, long-only.
Position size: $150 max per trade.
"""

import yfinance as yf
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# ── Configuration ────────────────────────────────────────────────────────────

PAIRS = [
    ('GOOGL', 'META'),
    ('JPM', 'GS'),
    ('HD', 'LOW'),
    ('KO', 'PG'),
    ('JNJ', 'ABBV'),
    ('AAPL', 'MSFT'),
    ('BAC', 'MS'),
    ('MCD', 'SBUX'),
]

ZSCORE_LOOKBACK = 60       # days for rolling z-score
ENTRY_Z = -2.0             # buy underperformer when z <= this
WATCH_Z = -1.5             # approaching signal threshold
EXIT_Z = -0.5              # close when z reverts to this
DOWNLOAD_DAYS = 90         # days of price history to fetch
AVG_HOLD_DAYS = 26         # expected hold time from backtest
MAX_POSITION = 150.0       # dollars per trade
BUYING_POWER = 440.0       # total account buying power
MAX_CONCURRENT = 3         # max concurrent positions

STATE_DIR = Path('/home/jupiter/Lvl3Quant/live_trading_linux/rh_pairs_state')
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / 'scanner_state.json'
HISTORY_FILE = STATE_DIR / 'signal_history.jsonl'


# ── Core Functions ───────────────────────────────────────────────────────────

def download_prices(tickers: list[str], days: int = DOWNLOAD_DAYS) -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    end = datetime.now()
    start = end - timedelta(days=days + 10)  # buffer for weekends/holidays

    all_tickers = list(set(tickers))
    data = yf.download(all_tickers, start=start.strftime('%Y-%m-%d'),
                       end=end.strftime('%Y-%m-%d'), progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = all_tickers

    return prices.dropna()


def compute_pair_zscore(prices: pd.DataFrame, stock_a: str, stock_b: str,
                        lookback: int = ZSCORE_LOOKBACK) -> pd.Series:
    """
    Compute rolling z-score of the log price ratio (A/B).
    Negative z-score means A is underperforming B (ratio dropped).
    """
    ratio = np.log(prices[stock_a] / prices[stock_b])
    rolling_mean = ratio.rolling(window=lookback).mean()
    rolling_std = ratio.rolling(window=lookback).std()
    zscore = (ratio - rolling_mean) / rolling_std
    return zscore


def identify_underperformer(prices: pd.DataFrame, stock_a: str, stock_b: str,
                            zscore: float) -> tuple[str, str]:
    """
    When z-score < 0, stock_a is the underperformer (ratio A/B fell).
    When z-score > 0, stock_b is the underperformer.
    Returns (underperformer, outperformer).
    """
    if zscore < 0:
        return stock_a, stock_b
    else:
        return stock_b, stock_a


def load_state() -> dict:
    """Load scanner state (active positions, history)."""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'active_positions': {}, 'last_scan': None}


def save_state(state: dict):
    """Save scanner state."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)


def log_signal(signal: dict):
    """Append signal to history log."""
    with open(HISTORY_FILE, 'a') as f:
        f.write(json.dumps(signal, default=str) + '\n')


def compute_position_size(price: float, max_dollars: float = MAX_POSITION) -> int:
    """Compute number of shares to buy within position size limit."""
    if price <= 0:
        return 0
    shares = int(max_dollars / price)
    return max(shares, 0)


# ── Main Scanner ─────────────────────────────────────────────────────────────

def run_scanner():
    """Run the pairs trading scanner."""
    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')

    # Collect all unique tickers
    all_tickers = list(set(t for pair in PAIRS for t in pair))

    print(f"\nPAIRS TRADING SCANNER \u2014 {today_str}")
    print("\u2501" * 72)
    print(f"{'Pair':<14} {'Z-Score':>8}  {'Signal':<14} {'Buy?':<22} {'Price':>8}")
    print("\u2500" * 72)

    # Download prices
    try:
        prices = download_prices(all_tickers)
    except Exception as e:
        print(f"\nERROR: Failed to download prices: {e}")
        sys.exit(1)

    if len(prices) < ZSCORE_LOOKBACK + 5:
        print(f"\nERROR: Only {len(prices)} trading days available, need {ZSCORE_LOOKBACK + 5}+")
        sys.exit(1)

    # Load state
    state = load_state()
    active = state.get('active_positions', {})

    entry_signals = []
    watch_signals = []
    exit_signals = []
    results = []

    for stock_a, stock_b in PAIRS:
        pair_label = f"{stock_a}/{stock_b}"

        if stock_a not in prices.columns or stock_b not in prices.columns:
            dash = "\u2014"
            print(f"{pair_label:<14} {'N/A':>8}  {'NO DATA':<14} {dash:<22} {dash:>8}")
            continue

        # Compute z-score
        zscores = compute_pair_zscore(prices, stock_a, stock_b)
        current_z = zscores.iloc[-1]
        prev_z = zscores.iloc[-2] if len(zscores) > 1 else current_z

        # Current prices
        price_a = float(prices[stock_a].iloc[-1])
        price_b = float(prices[stock_b].iloc[-1])

        # Rolling correlation (60-day)
        corr = prices[stock_a].pct_change().rolling(60).corr(
            prices[stock_b].pct_change()
        ).iloc[-1]

        # Determine signal
        underperformer, outperformer = identify_underperformer(
            prices, stock_a, stock_b, current_z
        )
        under_price = price_a if underperformer == stock_a else price_b
        shares = compute_position_size(under_price)

        result = {
            'pair': pair_label,
            'z_score': round(float(current_z), 3),
            'prev_z': round(float(prev_z), 3),
            'correlation': round(float(corr), 3) if not np.isnan(corr) else None,
            'underperformer': underperformer,
            'under_price': round(under_price, 2),
            'shares': shares,
            'dollar_size': round(shares * under_price, 2),
            'timestamp': today.isoformat(),
        }

        # Check for active position exit
        if pair_label in active:
            pos = active[pair_label]
            # Use absolute z-score for exit check (reversion toward 0)
            if abs(current_z) <= abs(EXIT_Z):
                signal_str = "\U0001f7e2 EXIT"
                result['signal'] = 'EXIT'
                buy_str = f"Close {pos.get('stock', '?')}"
                exit_signals.append(result)
            else:
                days_held = (today - datetime.fromisoformat(pos['entry_date'])).days
                signal_str = f"\U0001f535 HELD d{days_held}"
                result['signal'] = 'HELD'
                buy_str = f"Holding {pos.get('stock', '?')}"
        elif current_z <= ENTRY_Z or current_z >= -ENTRY_Z:
            # Entry signal (either tail)
            signal_str = "\U0001f534 ENTRY"
            result['signal'] = 'ENTRY'
            buy_str = f"Buy {underperformer} x{shares}"
            entry_signals.append(result)
        elif current_z <= WATCH_Z or current_z >= -WATCH_Z:
            if abs(current_z) >= abs(WATCH_Z):
                signal_str = "\u26a0\ufe0f  WATCH"
                result['signal'] = 'WATCH'
                buy_str = f"({underperformer} approaching)"
                watch_signals.append(result)
            else:
                signal_str = "NEUTRAL"
                result['signal'] = 'NEUTRAL'
                buy_str = "\u2014"
        else:
            signal_str = "NEUTRAL"
            result['signal'] = 'NEUTRAL'
            buy_str = "\u2014"

        price_str = f"${under_price:.2f}"
        print(f"{pair_label:<14} {current_z:>+8.3f}  {signal_str:<14} {buy_str:<22} {price_str:>8}")
        results.append(result)

    print("\u2500" * 72)

    # Summary
    n_active = len(active)
    print(f"\nActive positions: {n_active}/{MAX_CONCURRENT}")
    print(f"Entry signals:    {len(entry_signals)}")
    print(f"Watch signals:    {len(watch_signals)}")
    print(f"Exit signals:     {len(exit_signals)}")

    # Detail on entry signals
    if entry_signals:
        print(f"\n{'=' * 72}")
        print("ENTRY SIGNAL DETAILS")
        print(f"{'=' * 72}")
        for sig in entry_signals:
            pair = sig['pair']
            stock = sig['underperformer']
            price = sig['under_price']
            shares = sig['shares']
            dollar = sig['dollar_size']
            z = sig['z_score']
            corr = sig.get('correlation', 'N/A')

            print(f"\n  Pair:            {pair}")
            print(f"  Z-Score:         {z:+.3f} (entry threshold: {ENTRY_Z})")
            print(f"  Correlation:     {corr}")
            print(f"  Buy:             {stock} @ ${price:.2f}")
            print(f"  Position:        {shares} shares = ${dollar:.2f}")
            print(f"  Expected hold:   ~{AVG_HOLD_DAYS} trading days")
            print(f"  Exit target:     z-score reverts to {EXIT_Z}")
            print(f"  Strategy:        Buy underperformer, wait for mean reversion")

            if n_active >= MAX_CONCURRENT:
                print(f"  WARNING:         At max concurrent positions ({MAX_CONCURRENT}). "
                      f"Wait for exit before entering.")

    # Detail on watch signals
    if watch_signals:
        print(f"\n{'=' * 72}")
        print("APPROACHING SIGNALS (watch list)")
        print(f"{'=' * 72}")
        for sig in watch_signals:
            z = sig['z_score']
            dist = abs(z) - abs(ENTRY_Z)
            print(f"  {sig['pair']:<14} z={z:+.3f}  "
                  f"({abs(dist):.3f} from entry)  "
                  f"-> would buy {sig['underperformer']} @ ${sig['under_price']:.2f}")

    # Detail on exit signals
    if exit_signals:
        print(f"\n{'=' * 72}")
        print("EXIT SIGNALS")
        print(f"{'=' * 72}")
        for sig in exit_signals:
            pos = active.get(sig['pair'], {})
            entry_price = pos.get('entry_price', 0)
            current_price = sig['under_price']
            if entry_price > 0:
                pnl_pct = (current_price / entry_price - 1) * 100
                print(f"  {sig['pair']:<14} z={sig['z_score']:+.3f} (reverted to exit zone)")
                print(f"    Entry: ${entry_price:.2f} -> Current: ${current_price:.2f} "
                      f"({pnl_pct:+.1f}%)")
            else:
                print(f"  {sig['pair']:<14} z={sig['z_score']:+.3f} (reverted to exit zone)")

            # Remove from active positions
            if sig['pair'] in active:
                del active[sig['pair']]

    # Record new entries into state
    for sig in entry_signals:
        if n_active < MAX_CONCURRENT and sig['pair'] not in active:
            active[sig['pair']] = {
                'stock': sig['underperformer'],
                'entry_price': sig['under_price'],
                'shares': sig['shares'],
                'dollar_size': sig['dollar_size'],
                'entry_z': sig['z_score'],
                'entry_date': today.isoformat(),
                'target_exit_z': EXIT_Z,
            }
            n_active += 1
            log_signal({**sig, 'action': 'ENTRY'})

    for sig in exit_signals:
        log_signal({**sig, 'action': 'EXIT'})

    # Save state
    state['active_positions'] = active
    state['last_scan'] = today.isoformat()
    state['last_results'] = results
    save_state(state)

    # Correlation health check
    print(f"\n{'=' * 72}")
    print("CORRELATION HEALTH (60-day rolling)")
    print(f"{'=' * 72}")
    low_corr = []
    for r in results:
        corr = r.get('correlation')
        pair = r['pair']
        if corr is not None:
            status = "OK" if corr >= 0.5 else "LOW" if corr >= 0.3 else "BROKEN"
            marker = "" if status == "OK" else " <-- " + status
            print(f"  {pair:<14} {corr:+.3f}{marker}")
            if corr < 0.3:
                low_corr.append(pair)
        else:
            print(f"  {pair:<14} N/A")

    if low_corr:
        print(f"\n  WARNING: {', '.join(low_corr)} have broken correlation (<0.3). "
              f"Do NOT trade these pairs until correlation recovers.")

    print(f"\nState saved to: {STATE_FILE}")
    print(f"Signal log:     {HISTORY_FILE}")

    return results


if __name__ == '__main__':
    run_scanner()
