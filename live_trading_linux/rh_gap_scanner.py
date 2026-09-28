#!/usr/bin/env python3
"""
Gap-and-Go Scanner for Robinhood Account
Runs at 9:35 AM ET to check for 3%+ gaps on high volume.
Based on momentum_breakout_v1 backtest (Sharpe 0.797, p=0.005, passes all gates).
"""
import yfinance as yf
import json
from datetime import datetime, timedelta
import os

UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
    'AMD', 'INTC', 'BA', 'DIS', 'SBUX', 'HD', 'LOW', 'MCD', 'NKE',
    'COST', 'WMT', 'JPM', 'GS', 'BAC', 'MS', 'JNJ', 'PG', 'KO',
    'UNH', 'ABBV', 'CRM', 'NOW'
]

GAP_THRESHOLD = 3.0  # 3%+ gap
VOLUME_MULTIPLIER = 2.0  # 2x avg volume

STATE_DIR = '/home/jupiter/Lvl3Quant/live_trading_linux/rh_gap_state'
os.makedirs(STATE_DIR, exist_ok=True)

def check_gaps():
    """Check for gap-and-go signals at market open."""
    today = datetime.now()
    signals = []
    
    for ticker in UNIVERSE:
        try:
            # Get last 25 days of data
            data = yf.download(ticker, period='25d', progress=False)
            if len(data) < 2:
                continue
            
            # Today's open vs yesterday's close
            today_open = data['Open'].iloc[-1]
            prev_close = data['Close'].iloc[-2]
            gap_pct = (today_open / prev_close - 1) * 100
            
            # Volume check: today's volume vs 20-day average
            avg_vol = data['Volume'].iloc[:-1].tail(20).mean()
            today_vol = data['Volume'].iloc[-1]
            vol_ratio = today_vol / avg_vol if avg_vol > 0 else 0
            
            if gap_pct >= GAP_THRESHOLD and vol_ratio >= VOLUME_MULTIPLIER:
                signals.append({
                    'ticker': ticker,
                    'gap_pct': round(gap_pct, 2),
                    'volume_ratio': round(vol_ratio, 2),
                    'open_price': round(float(today_open), 2),
                    'prev_close': round(float(prev_close), 2),
                    'timestamp': today.isoformat()
                })
                print(f"🟢 GAP-AND-GO SIGNAL: {ticker} +{gap_pct:.1f}% gap, {vol_ratio:.1f}x volume")
            
        except Exception as e:
            pass
    
    if not signals:
        print(f"No gap-and-go signals today ({today.strftime('%Y-%m-%d')})")
    
    # Save signals
    state_file = os.path.join(STATE_DIR, f'signals_{today.strftime("%Y%m%d")}.json')
    with open(state_file, 'w') as f:
        json.dump({'date': today.isoformat(), 'signals': signals}, f, indent=2)
    
    return signals

if __name__ == '__main__':
    signals = check_gaps()
    for s in signals:
        print(f"\n  {s['ticker']}: Gap {s['gap_pct']:+.1f}%, Vol {s['volume_ratio']:.1f}x avg")
        print(f"  Open: ${s['open_price']}, Prev Close: ${s['prev_close']}")
        print(f"  Strategy: Buy call debit spread, hold ~20 days")
        
        # Affordability check for $440 account
        price = s['open_price']
        if price < 100:
            print(f"  💰 AFFORDABLE — options under $100/spread likely")
        elif price < 200:
            print(f"  ⚠️ MODERATE — spreads $100-200, check pricing")
        else:
            print(f"  ❌ EXPENSIVE — stock at ${price}, spreads may exceed $200")
