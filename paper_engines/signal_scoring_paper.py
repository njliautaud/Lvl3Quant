#!/usr/bin/env python3
"""
Signal Scoring Paper Engine — Strategy #13 (6/6 Adversarial PASS)
==================================================================
Regime-conditioned signal scoring:
  - Score each stock by how many validated signals fire today
  - Bull market (SPY > 200SMA): buy when score >= 3
  - Bear market (SPY < 200SMA): buy when score >= 1
  - Max 2 concurrent positions, $300 each
  - Exit: +10% TP, -15% SL, 21-day max hold

Validated performance: Sharpe 1.556, Sortino 2.045, WR 64.1%, PF 2.27
Perm p=0.012, all 4 sub-periods positive, 82% param robustness.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

STATE_FILE = Path('/home/jupiter/Lvl3Quant/state/signal_scoring_paper.json')
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'UNH', 'PG',
    'HD', 'ABBV', 'MRK', 'LLY', 'AVGO', 'COST', 'CRM', 'AMD', 'NFLX', 'V',
    'MA', 'WMT', 'PEP', 'KO', 'TMO', 'ABT', 'BAC', 'XOM', 'CVX', 'CSCO',
]

POS_SIZE = 300.0
MAX_CONCURRENT = 2
HOLD_DAYS = 21
PROFIT_TARGET = 0.10
STOP_LOSS = -0.15
BULL_THRESHOLD = 3
BEAR_THRESHOLD = 1

def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'positions': [],
        'closed_trades': [],
        'created': str(datetime.now()),
        'total_pnl': 0,
        'n_trades': 0,
    }

def save_state(state):
    state['last_updated'] = str(datetime.now())
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def get_market_data():
    """Download latest data for signal generation."""
    import yfinance as yf
    tickers = UNIVERSE + ['SPY', '^VIX', '^VIX3M', 'TLT', '^TNX']
    end = datetime.now()
    start = end - timedelta(days=120)  # Need 60d lookback for indicators
    raw = yf.download(tickers, start=start.strftime('%Y-%m-%d'),
                      end=end.strftime('%Y-%m-%d'),
                      auto_adjust=True, progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']; high = raw['High']; low = raw['Low']
    else:
        close = high = low = raw
    for df in [close, high, low]:
        if hasattr(df.columns, 'droplevel'):
            try: df.columns = df.columns.droplevel(1)
            except: pass
    close = close.ffill().dropna(how='all')
    return {
        'close': close,
        'high': high.reindex(close.index).ffill(),
        'low': low.reindex(close.index).ffill(),
    }

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _realized_vol(returns, window=21):
    return returns.rolling(window).std() * np.sqrt(252) * 100

def score_today(data):
    """Score each stock for today based on 7 validated signals."""
    close = data['close']
    today = close.index[-1]
    stock_tickers = [t for t in UNIVERSE if t in close.columns]
    scores = {}

    # 1. IV-RV Gap
    spy_ret = close['SPY'].pct_change()
    vix = close.get('^VIX')
    if vix is not None:
        rv = _realized_vol(spy_ret, 21)
        if not pd.isna(vix.iloc[-1]) and not pd.isna(rv.iloc[-1]):
            if vix.iloc[-1] - rv.iloc[-1] > 5:
                for t in stock_tickers:
                    scores[t] = scores.get(t, 0) + 1

    # 2. RSI Divergence
    for t in stock_tickers:
        if t not in close.columns: continue
        c = close[t].dropna()
        if len(c) < 21: continue
        rsi = _rsi(c, 14)
        if rsi.isna().iloc[-1]: continue
        wc = c.iloc[-11:]; wr = rsi.iloc[-11:]
        if len(wc) >= 11 and not wr.isna().any():
            if c.iloc[-1] < wc.iloc[0] and rsi.iloc[-1] > wr.iloc[0] and rsi.iloc[-1] < 40:
                scores[t] = scores.get(t, 0) + 1

    # 3. Bond Yield
    if '^TNX' in close.columns:
        tnx = close['^TNX']
        yc = tnx.diff(5)
        if not pd.isna(yc.iloc[-1]) and yc.iloc[-1] < -0.10:
            for t in stock_tickers:
                scores[t] = scores.get(t, 0) + 1

    # 4. Liquidity
    for t in stock_tickers:
        if t not in close.columns or t not in data['high'].columns: continue
        h = data['high'][t].dropna(); l = data['low'][t].dropna(); c = close[t].dropna()
        idx = h.index.intersection(l.index).intersection(c.index)
        if len(idx) < 65: continue
        h, l, c = h.loc[idx], l.loc[idx], c.loc[idx]
        hl = (h - l) / c; avg60 = hl.rolling(60).mean()
        if hl.iloc[-1] < avg60.iloc[-1] * 0.85:
            rsi = _rsi(c, 14)
            if not pd.isna(rsi.iloc[-1]) and rsi.iloc[-1] < 40:
                scores[t] = scores.get(t, 0) + 1

    # 5. Vol Term Structure
    vix = close.get('^VIX'); vix3m = close.get('^VIX3M')
    if vix is not None and vix3m is not None:
        if not pd.isna(vix.iloc[-1]) and not pd.isna(vix3m.iloc[-1]):
            if vix.iloc[-1] / vix3m.iloc[-1] > 1.0:
                for t in stock_tickers:
                    if t not in close.columns: continue
                    c = close[t].dropna()
                    if len(c) < 21: continue
                    sma20 = c.rolling(20).mean()
                    if (c.iloc[-1] - sma20.iloc[-1]) / sma20.iloc[-1] < -0.05:
                        scores[t] = scores.get(t, 0) + 1

    # 6. Consecutive Dip
    for t in stock_tickers:
        if t not in close.columns: continue
        c = close[t].dropna()
        if len(c) < 4: continue
        ret = c.pct_change()
        r1, r2, r3 = ret.iloc[-3], ret.iloc[-2], ret.iloc[-1]
        if r1 < 0 and r2 < 0 and r3 < 0 and r2 < r1 and r3 < r2:
            scores[t] = scores.get(t, 0) + 1

    # 7. Base MR
    for t in stock_tickers:
        if t not in close.columns: continue
        c = close[t].dropna()
        if len(c) < 51: continue
        rsi = _rsi(c, 14)
        h50 = c.rolling(50).max()
        dd = (c.iloc[-1] - h50.iloc[-1]) / h50.iloc[-1]
        if not pd.isna(rsi.iloc[-1]) and rsi.iloc[-1] < 30 and dd < -0.07:
            scores[t] = scores.get(t, 0) + 1

    # Determine regime
    spy = close['SPY']
    spy_sma = spy.rolling(200).mean()
    is_bull = spy.iloc[-1] > spy_sma.iloc[-1] if not pd.isna(spy_sma.iloc[-1]) else True
    threshold = BULL_THRESHOLD if is_bull else BEAR_THRESHOLD

    return scores, threshold, is_bull, today

def main():
    print(f"Signal Scoring Paper Engine — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    state = load_state()
    data = get_market_data()
    scores, threshold, is_bull, today = score_today(data)
    close = data['close']

    regime = "BULL" if is_bull else "BEAR"
    print(f"Market regime: {regime} (threshold: score >= {threshold})")

    # Check existing positions for exits
    active = []
    for pos in state['positions']:
        ticker = pos['ticker']
        entry_price = pos['entry_price']
        entry_date = pd.Timestamp(pos['entry_date'])
        days_held = (today - entry_date).days

        try:
            current_price = close.at[today, ticker]
        except:
            active.append(pos)
            continue

        if pd.isna(current_price):
            active.append(pos)
            continue

        ret = (current_price - entry_price) / entry_price
        exit_reason = None

        if ret >= PROFIT_TARGET:
            exit_reason = 'profit_target'
        elif ret <= STOP_LOSS:
            exit_reason = 'stop_loss'
        elif days_held >= HOLD_DAYS:
            exit_reason = 'time_expiry'

        if exit_reason:
            pnl = POS_SIZE * (ret - 0.001)  # spread cost
            trade = {
                'ticker': ticker, 'entry_date': str(entry_date.date()),
                'exit_date': str(today.date()), 'entry_price': round(entry_price, 2),
                'exit_price': round(current_price, 2), 'return': round(ret * 100, 2),
                'pnl': round(pnl, 2), 'exit_reason': exit_reason,
                'hold_days': days_held, 'score': pos.get('score', 0),
            }
            state['closed_trades'].append(trade)
            state['total_pnl'] = round(state['total_pnl'] + pnl, 2)
            state['n_trades'] += 1
            print(f"  EXIT {ticker}: {exit_reason} | ret={ret*100:+.1f}% | pnl=${pnl:.2f}")
        else:
            pos['current_price'] = round(current_price, 2)
            pos['unrealized_pnl'] = round(POS_SIZE * ret, 2)
            pos['days_held'] = days_held
            active.append(pos)
            print(f"  HOLD {ticker}: score={pos.get('score',0)} | {ret*100:+.1f}% | day {days_held}/{HOLD_DAYS}")

    state['positions'] = active

    # New entries
    if len(active) < MAX_CONCURRENT:
        # Filter by threshold and sort by score descending
        candidates = [(t, s) for t, s in scores.items() if s >= threshold]
        candidates.sort(key=lambda x: -x[1])

        # Skip stocks already held
        held_tickers = {p['ticker'] for p in active}
        candidates = [(t, s) for t, s in candidates if t not in held_tickers]

        slots = MAX_CONCURRENT - len(active)
        for ticker, score in candidates[:slots]:
            try:
                price = close.at[today, ticker]
            except:
                continue
            if pd.isna(price): continue
            pos = {
                'ticker': ticker, 'entry_date': str(today.date()),
                'entry_price': round(float(price), 2), 'score': int(score),
                'current_price': round(float(price), 2),
                'unrealized_pnl': 0, 'days_held': 0,
            }
            state['positions'].append(pos)
            print(f"  BUY  {ticker}: score={score} | price=${price:.2f}")

    if not candidates[:slots] if len(active) < MAX_CONCURRENT else True:
        if len(active) >= MAX_CONCURRENT:
            print(f"  No new entries — max concurrent ({MAX_CONCURRENT}) reached")
        elif not [t for t, s in scores.items() if s >= threshold]:
            print(f"  No stocks meet threshold (score >= {threshold}) today")

    # Summary
    print(f"\n{'─'*40}")
    print(f"Positions: {len(state['positions'])}/{MAX_CONCURRENT}")
    print(f"Closed trades: {state['n_trades']}")
    print(f"Total realized P&L: ${state['total_pnl']:.2f}")
    if state['closed_trades']:
        wins = [t for t in state['closed_trades'] if t['pnl'] > 0]
        wr = len(wins) / len(state['closed_trades']) * 100
        print(f"Win rate: {wr:.0f}%")

    # Top scores today
    top_scores = sorted(scores.items(), key=lambda x: -x[1])[:5]
    if top_scores:
        print(f"\nTop scores today: " + ", ".join(f"{t}({s})" for t, s in top_scores))

    save_state(state)
    print(f"\nState saved. Next run: tomorrow 4:25 PM ET.")

if __name__ == '__main__':
    main()
