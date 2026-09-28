#!/usr/bin/env python3
"""META reversal signal watcher — alerts when bounce conditions align."""
import yfinance as yf
import json as _json
import os
from datetime import datetime

class SafeEncoder(_json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bool):
            return int(obj)
        return super().default(obj)

def check_meta_reversal():
    meta = yf.download("META", period="60d", interval="1d", progress=False)
    if meta.empty:
        return None
    if hasattr(meta.columns, 'levels'):
        meta.columns = [c[0] if isinstance(c, tuple) else c for c in meta.columns]
    
    close = [float(x) for x in meta['Close'].values]
    volume = [float(x) for x in meta['Volume'].values]
    current = close[-1]
    prev = close[-2]
    
    deltas = [close[i] - close[i-1] for i in range(1, len(close))]
    gains = [max(d, 0) for d in deltas[-5:]]
    losses = [abs(min(d, 0)) for d in deltas[-5:]]
    avg_gain = sum(gains) / 5
    avg_loss = max(sum(losses) / 5, 0.001)
    rsi5 = 100 - (100 / (1 + avg_gain / avg_loss))
    
    avg_vol_20 = sum(volume[-21:-1]) / 20
    vol_ratio = volume[-1] / avg_vol_20
    green = current > prev
    big_green = (current - prev) / prev > 0.015
    vol_surge = vol_ratio > 1.5
    rsi_turning = 30 < rsi5 < 50 and green
    # Price level alerts
    hit_530 = current <= 530
    near_530 = current <= 540  # approaching the zone

    alert = (big_green and vol_surge) or (rsi_turning and green) or hit_530

    signals = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "price": round(current, 2),
        "rsi5": round(rsi5, 1),
        "green_day": green,
        "big_green": big_green,
        "volume_surge": vol_surge,
        "vol_ratio": round(vol_ratio, 2),
        "rsi_turning_up": rsi_turning,
        "hit_530_support": hit_530,
        "near_530": near_530,
        "alert": alert,
    }
    
    state_path = "/home/jupiter/Lvl3Quant/state/meta_reversal_watch.json"
    with open(state_path, 'w') as f:
        f.write(_json.dumps(signals, indent=2, default=str))
    
    if alert:
        alert_path = "/home/jupiter/Lvl3Quant/state/meta_alert.txt"
        msg = f"META reversal signal: price ${current:.2f}, RSI {rsi5:.0f}"
        if hit_530:
            msg = f"META HIT $530 SUPPORT! Price ${current:.2f}. Historical 80% bounce rate from this level. 38.2% Fib retracement zone. Consider buying for personal account. Stop-loss $515."
        elif big_green and vol_surge:
            msg += f", big green day on {vol_ratio:.1f}x avg volume"
        elif rsi_turning:
            msg += ", RSI turning up from oversold"
        if near_530 and not hit_530:
            msg += f" — APPROACHING $530 support zone (currently ${current:.2f})"
        with open(alert_path, 'w') as f:
            f.write(msg)
        signals["message"] = msg
    
    return signals

if __name__ == "__main__":
    result = check_meta_reversal()
    if result:
        print(_json.dumps(result, indent=2, default=str))
        if result["alert"]:
            print(f"\nALERT: {result.get('message','reversal signal!')}")
        else:
            print("\nNo reversal signal yet. Watching.")
