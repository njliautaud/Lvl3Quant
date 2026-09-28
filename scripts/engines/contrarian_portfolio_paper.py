#!/usr/bin/env python3
"""
Contrarian Portfolio Paper Engine v1

Deploys our proven signal portfolio as a live paper trader:
- 7 signal families (oversold_mfi, drop3_mfi, confluence_drop_vc, vol_climax, volcomp_mfi, drop3_highvol, oversold_highvol)
- Risk appetite positioning overlay (HC #740)
- Regime-aware position sizing (SPY > 200 SMA = 15 max, below = 7 max)
- Scans every 30 minutes during market hours (9:30-16:00 ET)
- Tracks positions with entry/exit dates, P&L
- Logs to JSON portfolio state file

Based on:
- HC #740: Full positioning ecosystem
- Items 771-776: Signal validation + portfolio backtesting
- Proven: Sharpe 0.8, CAGR 10.3%, WR 54%, 3,759 trades over 12.5 years
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')

PORTFOLIO_FILE = '/home/jupiter/Lvl3Quant/output/contrarian_portfolio_paper/portfolio.json'
LOG_FILE = '/home/jupiter/Lvl3Quant/output/contrarian_portfolio_paper/trades.log'
STATE_DIR = '/home/jupiter/Lvl3Quant/output/contrarian_portfolio_paper'
os.makedirs(STATE_DIR, exist_ok=True)

# S&P 500 tickers
SP500_CACHE = '/home/jupiter/Lvl3Quant/output/sp500_tickers.json'

INITIAL_CAPITAL = 100000
MAX_POSITIONS_BULL = 15
MAX_POSITIONS_BEAR = 7


def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def get_tickers():
    if os.path.exists(SP500_CACHE):
        with open(SP500_CACHE) as f:
            return json.load(f)
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = sorted(tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist())
        with open(SP500_CACHE, 'w') as f:
            json.dump(tickers, f)
        return tickers
    except:
        return ['AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA']


def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE) as f:
            return json.load(f)
    return {
        'cash': INITIAL_CAPITAL,
        'positions': [],
        'closed_trades': [],
        'created': datetime.now().isoformat(),
        'last_scan': None,
        'total_pnl': 0,
    }


def save_portfolio(pf):
    pf['last_updated'] = datetime.now().isoformat()
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(pf, f, indent=2, default=str)


def fetch_prices(tickers, period='5d'):
    """Fetch recent prices for signal detection."""
    all_data = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        try:
            data = yf.download(batch, period=period, group_by='ticker', threads=True, progress=False)
            if data is None or data.empty:
                continue
            # Handle MultiIndex columns from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    try:
                        # Try to extract ticker data from MultiIndex
                        if t in data.columns.get_level_values(1):
                            td = data.xs(t, level=1, axis=1).copy()
                        elif t in data.columns.get_level_values(0):
                            td = data[t].copy()
                        else:
                            continue
                        td = td.dropna(subset=['Close']) if 'Close' in td.columns else td.dropna()
                        if len(td) < 2:
                            continue
                        td['ticker'] = t
                        td.index.name = 'date'
                        all_data.append(td.reset_index())
                    except Exception:
                        pass
            else:
                # Single ticker or flat columns
                if len(batch) == 1:
                    td = data.copy()
                    td = td.dropna(subset=['Close']) if 'Close' in td.columns else td.dropna()
                    if len(td) >= 2:
                        td['ticker'] = batch[0]
                        td.index.name = 'date'
                        all_data.append(td.reset_index())
        except Exception:
            pass
    if not all_data:
        return pd.DataFrame()
    df = pd.concat(all_data, ignore_index=True)
    # Flatten any remaining MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) and c[1] == '' else (c[0] if isinstance(c, tuple) else c) for c in df.columns]
    return df


def fetch_longer_prices(tickers, period='1mo'):
    """Fetch 1 month for indicators that need lookback."""
    return fetch_prices(tickers, period=period)


def compute_signals(df):
    """Compute all signal indicators."""
    results = []
    for ticker, gdf in df.groupby('ticker'):
        g = gdf.sort_values('date').copy()
        if len(g) < 5:
            continue

        g['ret_1d'] = g['Close'].pct_change()

        # RSI-14 (approximate with available data)
        delta = g['Close'].diff()
        gain = delta.clip(lower=0).rolling(min(14, len(g)-1)).mean()
        loss = (-delta.clip(upper=0)).rolling(min(14, len(g)-1)).mean()
        rs = gain / loss.replace(0, np.nan)
        g['rsi'] = 100 - (100 / (1 + rs))

        # Volume ratio
        g['vol_avg'] = g['Volume'].rolling(min(20, len(g)-1)).mean()
        g['vol_ratio'] = g['Volume'] / g['vol_avg'].replace(0, np.nan)

        # MFI (simplified)
        tp = (g['High'] + g['Low'] + g['Close']) / 3
        rmf = tp * g['Volume']
        pos = rmf.where(tp > tp.shift(1), 0).rolling(min(14, len(g)-1)).sum()
        neg = rmf.where(tp < tp.shift(1), 0).rolling(min(14, len(g)-1)).sum()
        g['mfi'] = 100 - (100 / (1 + pos / neg.replace(0, np.nan)))

        # Vol percentile (approximate with short history)
        g['vol_20d'] = g['ret_1d'].rolling(min(20, len(g)-1)).std()
        # Can't compute proper percentile with 5d data — use threshold instead
        g['vol_low'] = g['vol_20d'] < g['vol_20d'].quantile(0.20) if len(g) > 10 else False

        results.append(g)

    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def detect_today_signals(df):
    """Detect signals on the most recent day."""
    if len(df) == 0:
        return []

    # Get latest date per ticker
    latest = df.groupby('ticker').tail(1).copy()
    signals = []

    for _, row in latest.iterrows():
        ticker = row['ticker']
        rsi = row.get('rsi', 50)
        mfi = row.get('mfi', 50)
        ret_1d = row.get('ret_1d', 0)
        vol_ratio = row.get('vol_ratio', 1)
        vol_low = row.get('vol_low', False)
        price = row.get('Close', 0)

        if pd.isna(rsi) or pd.isna(ret_1d):
            continue

        # Signal detection
        if rsi < 20 and mfi < 20:
            signals.append({'ticker': ticker, 'strategy': 'oversold_mfi', 'hold_days': 10, 'price': price,
                          'reason': f'RSI {rsi:.0f}, MFI {mfi:.0f}'})

        if ret_1d < -0.03 and mfi < 20:
            signals.append({'ticker': ticker, 'strategy': 'drop3_mfi', 'hold_days': 21, 'price': price,
                          'reason': f'Drop {ret_1d:.1%}, MFI {mfi:.0f}'})

        if ret_1d < -0.03 and vol_low:
            signals.append({'ticker': ticker, 'strategy': 'confluence_drop_vc', 'hold_days': 10, 'price': price,
                          'reason': f'Drop {ret_1d:.1%}, vol compressed'})

        if vol_ratio >= 2.0 and ret_1d < -0.02 and vol_low:
            signals.append({'ticker': ticker, 'strategy': 'vol_climax', 'hold_days': 5, 'price': price,
                          'reason': f'Vol spike {vol_ratio:.1f}x, drop {ret_1d:.1%}'})

        if vol_low and mfi < 20:
            signals.append({'ticker': ticker, 'strategy': 'volcomp_mfi', 'hold_days': 10, 'price': price,
                          'reason': f'Vol compressed, MFI {mfi:.0f}'})

        if ret_1d < -0.03 and vol_ratio > 1.5:
            signals.append({'ticker': ticker, 'strategy': 'drop3_highvol', 'hold_days': 10, 'price': price,
                          'reason': f'Drop {ret_1d:.1%}, vol {vol_ratio:.1f}x avg'})

        if rsi < 20 and vol_ratio > 1.5:
            signals.append({'ticker': ticker, 'strategy': 'oversold_highvol', 'hold_days': 21, 'price': price,
                          'reason': f'RSI {rsi:.0f}, vol {vol_ratio:.1f}x avg'})

    return signals


def get_regime():
    """Check SPY vs 200 SMA."""
    try:
        spy = yf.download('SPY', period='1y', progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
        spy['sma_200'] = spy['Close'].rolling(200).mean()
        latest = spy.iloc[-1]
        above = latest['Close'] > latest['sma_200']
        return 'bull' if above else 'bear', latest['Close'], latest['sma_200']
    except:
        return 'bull', 0, 0  # Default to bull if can't determine


def get_risk_appetite():
    """Quick risk appetite check."""
    try:
        data = yf.download(['SPY', 'SHY', 'BIL'], period='1mo', group_by='ticker', progress=False)
        spy_vol = data['SPY']['Volume'].rolling(5).mean().iloc[-1]
        shy_vol = data['SHY']['Volume'].rolling(5).mean().iloc[-1]
        bil_vol = data['BIL']['Volume'].rolling(5).mean().iloc[-1]
        ratio = spy_vol / (shy_vol + bil_vol)
        median = data['SPY']['Volume'].rolling(20).mean().iloc[-1] / (data['SHY']['Volume'].rolling(20).mean().iloc[-1] + data['BIL']['Volume'].rolling(20).mean().iloc[-1])
        z = (ratio - median) / max(abs(median) * 0.1, 0.001)
        return 'low' if z < -0.5 else 'high' if z > 0.5 else 'neutral'
    except:
        return 'neutral'


def check_exits(pf):
    """Check if any positions should exit."""
    today = datetime.now().strftime('%Y-%m-%d')
    exiting = []
    remaining = []

    for pos in pf['positions']:
        if pos['exit_date'] <= today:
            # Get current price
            try:
                data = yf.download(pos['ticker'], period='1d', progress=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                current_price = float(data['Close'].iloc[-1])
            except:
                current_price = pos['entry_price']  # Fallback

            pnl_pct = (current_price / pos['entry_price'] - 1) * 100
            pos['exit_price'] = current_price
            pos['pnl_pct'] = round(pnl_pct, 2)
            pos['exit_actual'] = today
            pf['closed_trades'].append(pos)
            pf['total_pnl'] += pnl_pct
            exiting.append(pos)
            log(f"  EXIT: {pos['ticker']} ({pos['strategy']}) — entry ${pos['entry_price']:.2f} → ${current_price:.2f} ({pnl_pct:+.1f}%)")
        else:
            remaining.append(pos)

    pf['positions'] = remaining
    return exiting


def has_earnings_within_days(ticker, days=5):
    """Check if ticker has earnings within N business days. HC #736: no earnings plays."""
    try:
        stock = yf.Ticker(ticker)
        cal = stock.calendar
        if cal and 'Earnings Date' in cal:
            earnings_dates = cal['Earnings Date']
            if isinstance(earnings_dates, list):
                for ed in earnings_dates:
                    delta = (ed - datetime.now().date()).days
                    if -1 <= delta <= days * 1.5:  # Include 1 day after (post-earnings drift)
                        return True, ed
            elif hasattr(earnings_dates, 'date'):
                delta = (earnings_dates - datetime.now().date()).days
                if -1 <= delta <= days * 1.5:
                    return True, earnings_dates
    except:
        pass
    return False, None


def enter_positions(pf, signals, regime, risk_appetite):
    """Enter new positions from signals."""
    max_pos = MAX_POSITIONS_BULL if regime == 'bull' else MAX_POSITIONS_BEAR
    slots = max_pos - len(pf['positions'])

    if slots <= 0 or not signals:
        return []

    active_tickers = {p['ticker'] for p in pf['positions']}

    # HC #736: Filter out tickers with earnings within 5 business days
    # Cache per-ticker to avoid repeated API calls
    earnings_cache = {}
    filtered_signals = []
    for s in signals:
        t = s['ticker']
        if t not in earnings_cache:
            earnings_cache[t] = has_earnings_within_days(t, days=5)
        has_earn, earn_date = earnings_cache[t]
        if has_earn:
            log(f"  SKIP: {t} — earnings {earn_date} within 5 days (HC #736)")
        else:
            filtered_signals.append(s)
    signals = filtered_signals

    # Prioritize: high-priority signals first (risk_low positioning)
    if risk_appetite == 'low':
        # Boost priority for all signals during low risk appetite
        for s in signals:
            s['priority'] = 2
    else:
        for s in signals:
            s['priority'] = 1

    # Deduplicate by ticker (keep highest priority)
    seen = {}
    for s in sorted(signals, key=lambda x: -x['priority']):
        if s['ticker'] not in seen and s['ticker'] not in active_tickers:
            seen[s['ticker']] = s

    new_entries = list(seen.values())[:slots]
    entered = []

    for sig in new_entries:
        today = datetime.now()
        # Calculate exit date (trading days approximation)
        exit_date = today + timedelta(days=int(sig['hold_days'] * 1.5))

        pos = {
            'ticker': sig['ticker'],
            'strategy': sig['strategy'],
            'entry_price': round(sig['price'], 2),
            'entry_date': today.strftime('%Y-%m-%d'),
            'exit_date': exit_date.strftime('%Y-%m-%d'),
            'hold_days': sig['hold_days'],
            'reason': sig['reason'],
            'regime': regime,
            'risk_appetite': risk_appetite,
        }
        pf['positions'].append(pos)
        entered.append(pos)
        log(f"  ENTRY: {sig['ticker']} ({sig['strategy']}) — ${sig['price']:.2f}, hold {sig['hold_days']}d, exit {exit_date.strftime('%Y-%m-%d')}")
        log(f"         Reason: {sig['reason']}")

    return entered


def portfolio_summary(pf):
    """Generate portfolio summary."""
    n_pos = len(pf['positions'])
    n_closed = len(pf['closed_trades'])

    if n_closed > 0:
        wins = sum(1 for t in pf['closed_trades'] if t.get('pnl_pct', 0) > 0)
        wr = wins / n_closed
        avg_pnl = np.mean([t.get('pnl_pct', 0) for t in pf['closed_trades']])
    else:
        wr = 0
        avg_pnl = 0

    # Current positions mark-to-market
    mtm_lines = []
    for pos in pf['positions']:
        try:
            data = yf.download(pos['ticker'], period='1d', progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            current = float(data['Close'].iloc[-1])
            pnl = (current / pos['entry_price'] - 1) * 100
            mtm_lines.append(f"    {pos['ticker']}: entry ${pos['entry_price']:.2f} → ${current:.2f} ({pnl:+.1f}%), exit {pos['exit_date']}")
        except:
            mtm_lines.append(f"    {pos['ticker']}: entry ${pos['entry_price']:.2f}, exit {pos['exit_date']}")

    summary = f"Active: {n_pos} positions | Closed: {n_closed} trades | WR: {wr:.0%} | Avg P&L: {avg_pnl:+.1f}%"
    if mtm_lines:
        summary += "\n" + "\n".join(mtm_lines)

    return summary


def run_scan():
    """Main scan cycle."""
    log("=" * 60)
    log("CONTRARIAN PORTFOLIO PAPER ENGINE — SCAN")
    log("=" * 60)

    # Load state
    pf = load_portfolio()

    # Check regime
    regime, spy_price, spy_sma = get_regime()
    risk_appetite = get_risk_appetite()
    log(f"Regime: {regime} (SPY ${spy_price:.2f} vs 200SMA ${spy_sma:.2f})")
    log(f"Risk appetite: {risk_appetite}")
    log(f"Max positions: {MAX_POSITIONS_BULL if regime == 'bull' else MAX_POSITIONS_BEAR}")
    log(f"Current positions: {len(pf['positions'])}")

    # Check exits
    exits = check_exits(pf)
    if exits:
        log(f"Exited {len(exits)} positions")

    # Fetch prices and detect signals
    tickers = get_tickers()
    log(f"Scanning {len(tickers)} stocks...")

    df = fetch_longer_prices(tickers, period='1mo')
    if len(df) == 0:
        log("ERROR: No price data available")
        save_portfolio(pf)
        return

    df = compute_signals(df)
    signals = detect_today_signals(df)
    log(f"Detected {len(signals)} signals across {len(set(s['ticker'] for s in signals))} stocks")

    # Signal breakdown
    strat_counts = {}
    for s in signals:
        strat_counts[s['strategy']] = strat_counts.get(s['strategy'], 0) + 1
    for strat, count in sorted(strat_counts.items()):
        log(f"  {strat}: {count} signals")

    # Enter new positions
    entries = enter_positions(pf, signals, regime, risk_appetite)
    if entries:
        log(f"Entered {len(entries)} new positions")
    else:
        log("No new entries (full or no signals)")

    # Summary
    log(f"\n{portfolio_summary(pf)}")

    # Save
    pf['last_scan'] = datetime.now().isoformat()
    save_portfolio(pf)
    log("Scan complete.\n")


if __name__ == '__main__':
    run_scan()
