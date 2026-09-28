#!/usr/bin/env python3
"""
Next Play Scanner v1 — Real-time momentum/oversold/flow-divergence scanner
Identifies high-conviction option setups across sectors, indices, and top liquid names.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Universe ───────────────────────────────────────────────────────────────

SECTOR_ETFS = ["XLE", "XLF", "XLV", "XLY", "XLC", "XLK", "XLB", "XLI", "XLU", "XLRE", "XLP"]
INDEX_ETFS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "SLV"]

# Top 30 most liquid S&P 500 by volume/options OI
TOP30_STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "V", "UNH",
    "MA", "HD", "PG", "JNJ", "XOM", "CVX", "BAC", "WFC", "ABBV", "MRK",
    "PFE", "KO", "PEP", "COST", "CRM", "AMD", "NFLX", "DIS", "INTC", "BA"
]

ALL_SYMBOLS = SECTOR_ETFS + INDEX_ETFS + TOP30_STOCKS


# ─── Technical Indicators ──────────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_mfi(high, low, close, volume, period=14):
    typical_price = (high + low + close) / 3
    raw_mf = typical_price * volume
    delta = typical_price.diff()
    pos_mf = raw_mf.where(delta > 0, 0.0)
    neg_mf = raw_mf.where(delta <= 0, 0.0)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mf_ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mf_ratio))


def compute_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_bollinger(close, period=20, num_std=2):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    # Position: 0 = at lower band, 1 = at upper band
    bb_pos = (close - lower) / (upper - lower).replace(0, np.nan)
    return bb_pos, sma, upper, lower


# ─── Data Fetch ─────────────────────────────────────────────────────────────

def fetch_data(symbols, period="6mo"):
    """Fetch OHLCV data for all symbols."""
    print(f"Fetching data for {len(symbols)} symbols...")
    data = {}
    # Batch download
    try:
        raw = yf.download(symbols, period=period, group_by="ticker", progress=False, threads=True)
    except Exception as e:
        print(f"Batch download failed: {e}, falling back to individual")
        raw = None

    for sym in symbols:
        try:
            if raw is not None and len(symbols) > 1:
                df = raw[sym].dropna(how="all")
            else:
                df = raw if len(symbols) == 1 else None

            if df is None or len(df) < 60:
                df = yf.download(sym, period=period, progress=False)

            if df is not None and len(df) >= 60:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[sym] = df
            else:
                print(f"  SKIP {sym}: insufficient data ({len(df) if df is not None else 0} rows)")
        except Exception as e:
            print(f"  SKIP {sym}: {e}")

    print(f"Got data for {len(data)}/{len(symbols)} symbols")
    return data


# ─── Analysis ───────────────────────────────────────────────────────────────

def analyze_symbol(sym, df):
    """Compute all indicators for a single symbol."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    last_price = float(close.iloc[-1])

    # Momentum
    mom_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0
    mom_20d = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 20 else 0
    mom_60d = float((close.iloc[-1] / close.iloc[-61] - 1) * 100) if len(close) > 60 else 0

    # RSI
    rsi = compute_rsi(close, 14)
    rsi_val = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50

    # MFI
    mfi = compute_mfi(high, low, close, volume, 14)
    mfi_val = float(mfi.iloc[-1]) if not np.isnan(mfi.iloc[-1]) else 50

    # OBV slope (10d)
    obv = compute_obv(close, volume)
    if len(obv) >= 10:
        obv_recent = obv.iloc[-10:].values
        x = np.arange(10)
        slope = np.polyfit(x, obv_recent, 1)[0]
        obv_slope = float(slope / (abs(obv_recent.mean()) + 1e-9))  # normalized
    else:
        obv_slope = 0

    # Bollinger Band position
    bb_pos, sma20, bb_upper, bb_lower = compute_bollinger(close, 20, 2)
    bb_pos_val = float(bb_pos.iloc[-1]) if not np.isnan(bb_pos.iloc[-1]) else 0.5

    # Volume spike
    vol_avg_20 = float(volume.iloc[-21:-1].mean())
    vol_today = float(volume.iloc[-1])
    vol_spike = vol_today / vol_avg_20 if vol_avg_20 > 0 else 1.0

    # 52-week high/low distance
    high_52w = float(high.iloc[-252:].max()) if len(high) >= 252 else float(high.max())
    low_52w = float(low.iloc[-252:].min()) if len(low) >= 252 else float(low.min())
    dist_from_high = (last_price / high_52w - 1) * 100
    dist_from_low = (last_price / low_52w - 1) * 100

    # ATR as % of price
    atr = compute_atr(high, low, close, 14)
    atr_val = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0
    atr_pct = (atr_val / last_price) * 100

    # SMAs
    sma20_val = float(sma20.iloc[-1]) if not np.isnan(sma20.iloc[-1]) else last_price
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else last_price
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else last_price

    above_sma20 = last_price > sma20_val
    above_sma50 = last_price > sma50
    above_sma200 = last_price > sma200

    # Pullback from recent high (20d)
    recent_high = float(high.iloc[-20:].max())
    pullback_pct = (last_price / recent_high - 1) * 100

    return {
        "symbol": sym,
        "price": round(last_price, 2),
        "mom_5d": round(mom_5d, 2),
        "mom_20d": round(mom_20d, 2),
        "mom_60d": round(mom_60d, 2),
        "rsi": round(rsi_val, 1),
        "mfi": round(mfi_val, 1),
        "obv_slope_norm": round(obv_slope, 4),
        "bb_position": round(bb_pos_val, 3),
        "vol_spike": round(vol_spike, 2),
        "dist_52w_high_pct": round(dist_from_high, 2),
        "dist_52w_low_pct": round(dist_from_low, 2),
        "atr_pct": round(atr_pct, 2),
        "atr_dollars": round(atr_val, 2),
        "above_sma20": above_sma20,
        "above_sma50": above_sma50,
        "above_sma200": above_sma200,
        "pullback_from_20d_high_pct": round(pullback_pct, 2),
        "sma20": round(sma20_val, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2),
    }


# ─── Setup Detection ───────────────────────────────────────────────────────

def detect_setups(results):
    """Identify momentum continuation, oversold bounce, and flow divergence setups."""
    setups = []

    for r in results:
        sym = r["symbol"]
        price = r["price"]

        # ── A. MOMENTUM CONTINUATION ──
        # Above all SMAs, RSI 55-70, MFI >60, OBV rising, pulling back 1-3%
        mom_score = 0
        mom_reasons = []

        if r["above_sma20"] and r["above_sma50"] and r["above_sma200"]:
            mom_score += 2
            mom_reasons.append("Above all SMAs")

        if 55 <= r["rsi"] <= 72:
            mom_score += 2
            mom_reasons.append(f"RSI {r['rsi']} (sweet spot)")
        elif 50 <= r["rsi"] < 55:
            mom_score += 1
            mom_reasons.append(f"RSI {r['rsi']} (emerging)")

        if r["mfi"] > 60:
            mom_score += 2
            mom_reasons.append(f"MFI {r['mfi']} (strong inflow)")
        elif r["mfi"] > 50:
            mom_score += 1
            mom_reasons.append(f"MFI {r['mfi']} (positive flow)")

        if r["obv_slope_norm"] > 0.005:
            mom_score += 2
            mom_reasons.append("OBV rising (accumulation)")
        elif r["obv_slope_norm"] > 0:
            mom_score += 1
            mom_reasons.append("OBV slightly positive")

        if -3.5 <= r["pullback_from_20d_high_pct"] <= -0.5:
            mom_score += 2
            mom_reasons.append(f"Pullback {r['pullback_from_20d_high_pct']:.1f}% (entry dip)")
        elif -5 <= r["pullback_from_20d_high_pct"] < -0.5:
            mom_score += 1
            mom_reasons.append(f"Pullback {r['pullback_from_20d_high_pct']:.1f}%")

        if r["mom_20d"] > 3:
            mom_score += 1
            mom_reasons.append(f"20d momentum +{r['mom_20d']:.1f}%")

        if mom_score >= 6:
            confidence = "HIGH" if mom_score >= 9 else "MEDIUM" if mom_score >= 7 else "LOW"
            setup = build_setup(r, "MOMENTUM_CONTINUATION", mom_score, confidence, mom_reasons)
            setups.append(setup)

        # ── B. OVERSOLD BOUNCE ──
        # RSI <35, MFI <35, below 20 SMA, above 200 SMA, volume spike
        osb_score = 0
        osb_reasons = []

        if r["rsi"] < 30:
            osb_score += 3
            osb_reasons.append(f"RSI {r['rsi']} (deeply oversold)")
        elif r["rsi"] < 35:
            osb_score += 2
            osb_reasons.append(f"RSI {r['rsi']} (oversold)")
        elif r["rsi"] < 40:
            osb_score += 1
            osb_reasons.append(f"RSI {r['rsi']} (approaching oversold)")

        if r["mfi"] < 25:
            osb_score += 2
            osb_reasons.append(f"MFI {r['mfi']} (extreme selling pressure)")
        elif r["mfi"] < 35:
            osb_score += 1
            osb_reasons.append(f"MFI {r['mfi']} (selling pressure)")

        if not r["above_sma20"] and r["above_sma200"]:
            osb_score += 2
            osb_reasons.append("Below 20 SMA but above 200 SMA (dip, not breakdown)")
        elif not r["above_sma20"] and not r["above_sma200"]:
            # Penalize — broken trend
            osb_score -= 1
            osb_reasons.append("Below 200 SMA (broken trend, risky)")

        if r["vol_spike"] > 1.5:
            osb_score += 2
            osb_reasons.append(f"Volume spike {r['vol_spike']:.1f}x (capitulation)")
        elif r["vol_spike"] > 1.2:
            osb_score += 1
            osb_reasons.append(f"Volume elevated {r['vol_spike']:.1f}x")

        if r["bb_position"] < 0.1:
            osb_score += 2
            osb_reasons.append(f"At lower Bollinger Band ({r['bb_position']:.2f})")
        elif r["bb_position"] < 0.2:
            osb_score += 1
            osb_reasons.append(f"Near lower Bollinger Band ({r['bb_position']:.2f})")

        if r["dist_52w_high_pct"] < -15:
            osb_score += 1
            osb_reasons.append(f"{r['dist_52w_high_pct']:.1f}% from 52w high")

        if osb_score >= 5:
            confidence = "HIGH" if osb_score >= 8 else "MEDIUM" if osb_score >= 6 else "LOW"
            setup = build_setup(r, "OVERSOLD_BOUNCE", osb_score, confidence, osb_reasons)
            setups.append(setup)

        # ── C. FLOW DIVERGENCE ──
        # OBV rising while price flat/down, MFI turning up from low
        fd_score = 0
        fd_reasons = []

        # Price flat or down but OBV rising
        if r["mom_5d"] <= 0 and r["obv_slope_norm"] > 0.005:
            fd_score += 3
            fd_reasons.append(f"Price {r['mom_5d']:.1f}% 5d but OBV rising (divergence)")
        elif r["mom_5d"] <= 1 and r["obv_slope_norm"] > 0.01:
            fd_score += 2
            fd_reasons.append(f"OBV strongly rising despite flat price")
        elif r["mom_20d"] <= 0 and r["obv_slope_norm"] > 0.003:
            fd_score += 2
            fd_reasons.append(f"20d price flat/down but OBV positive")

        # MFI recovering from low
        if 30 < r["mfi"] < 50 and r["rsi"] < 45:
            fd_score += 2
            fd_reasons.append(f"MFI {r['mfi']} recovering while RSI still low")
        elif r["mfi"] < 30 and r["obv_slope_norm"] > 0:
            fd_score += 1
            fd_reasons.append(f"MFI low but OBV not confirming selloff")

        # Accumulation pattern: above 200 SMA (not broken)
        if r["above_sma200"]:
            fd_score += 1
            fd_reasons.append("Above 200 SMA (trend intact)")

        # Volume pattern
        if r["vol_spike"] > 1.3:
            fd_score += 1
            fd_reasons.append(f"Elevated volume {r['vol_spike']:.1f}x")

        if fd_score >= 4:
            confidence = "HIGH" if fd_score >= 7 else "MEDIUM" if fd_score >= 5 else "LOW"
            setup = build_setup(r, "FLOW_DIVERGENCE", fd_score, confidence, fd_reasons)
            setups.append(setup)

    return setups


def build_setup(r, setup_type, raw_score, confidence, reasons):
    """Build a setup dict with strike/expiry suggestions and risk/reward."""
    sym = r["symbol"]
    price = r["price"]
    atr = r["atr_dollars"]
    atr_pct = r["atr_pct"]

    # ── Strike & Expiry Logic ──
    if setup_type == "MOMENTUM_CONTINUATION":
        # Slightly ITM or ATM calls, 2-4 weeks
        strike = round(price - atr * 0.5, 0)  # slightly ITM
        expiry_days = 21  # 3 weeks
        direction = "CALL"
        # Estimate cost: ATM call ~3-5% of price for 3 weeks (rough)
        est_cost_per_contract = round(price * 0.035, 2)
        # R/R: target is continuation of 5-10% move, risk is pullback to SMA
        target_move_pct = min(abs(r["mom_20d"]) * 0.5, 8)  # expect half of recent momentum
        risk_pct = abs(r["pullback_from_20d_high_pct"]) + atr_pct
        play_description = f"Buy {sym} ${strike:.0f} calls, ~3 weeks out"

    elif setup_type == "OVERSOLD_BOUNCE":
        # ATM or slightly OTM calls, 3-4 weeks (give time for bounce)
        strike = round(price, 0)  # ATM
        expiry_days = 28  # 4 weeks
        direction = "CALL"
        est_cost_per_contract = round(price * 0.03, 2)
        target_move_pct = min(abs(price / r["sma20"] - 1) * 100, 10)  # bounce to 20 SMA
        risk_pct = atr_pct * 2  # wider stop for bounce plays
        play_description = f"Buy {sym} ${strike:.0f} calls, ~4 weeks out (bounce play)"

    elif setup_type == "FLOW_DIVERGENCE":
        # ATM calls, 4 weeks (needs time)
        strike = round(price, 0)
        expiry_days = 28
        direction = "CALL"
        est_cost_per_contract = round(price * 0.035, 2)
        target_move_pct = 5  # accumulation breakout
        risk_pct = atr_pct * 1.5
        play_description = f"Buy {sym} ${strike:.0f} calls, ~4 weeks out (accumulation)"

    else:
        return None

    # R/R ratio
    rr_ratio = round(target_move_pct / max(risk_pct, 0.5), 2)

    # Composite score: confidence_num * expected_return / risk
    conf_num = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}[confidence]
    composite = round(conf_num * target_move_pct / max(risk_pct, 0.5), 2)

    # Estimated number of contracts for ~$200 position
    contracts_for_200 = max(1, int(200 / (est_cost_per_contract * 100))) if est_cost_per_contract > 0 else 1

    return {
        "symbol": sym,
        "setup_type": setup_type,
        "direction": direction,
        "confidence": confidence,
        "raw_score": raw_score,
        "composite_rank_score": composite,
        "play": play_description,
        "suggested_strike": strike,
        "expiry_days": expiry_days,
        "est_cost_per_contract": est_cost_per_contract,
        "contracts_at_200_budget": contracts_for_200,
        "total_est_cost": round(est_cost_per_contract * 100 * contracts_for_200, 2),
        "target_move_pct": round(target_move_pct, 1),
        "risk_pct": round(risk_pct, 1),
        "risk_reward": rr_ratio,
        "reasons": reasons,
        "technicals": {
            "price": r["price"],
            "rsi": r["rsi"],
            "mfi": r["mfi"],
            "obv_slope": r["obv_slope_norm"],
            "bb_position": r["bb_position"],
            "vol_spike": r["vol_spike"],
            "mom_5d": r["mom_5d"],
            "mom_20d": r["mom_20d"],
            "mom_60d": r["mom_60d"],
            "dist_52w_high": r["dist_52w_high_pct"],
            "atr_pct": r["atr_pct"],
            "above_sma20": r["above_sma20"],
            "above_sma50": r["above_sma50"],
            "above_sma200": r["above_sma200"],
            "pullback": r["pullback_from_20d_high_pct"],
        },
    }


# ─── Output ─────────────────────────────────────────────────────────────────

def print_summary(setups, top_n=5):
    """Print clean summary of top setups."""
    print("\n" + "=" * 80)
    print(f"  NEXT PLAY SCANNER — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 80)

    if not setups:
        print("\n  No setups found meeting criteria. Market may be in transition.\n")
        return

    ranked = sorted(setups, key=lambda s: s["composite_rank_score"], reverse=True)[:top_n]

    for i, s in enumerate(ranked, 1):
        conf_emoji = {"HIGH": "***", "MEDIUM": "** ", "LOW": "*  "}[s["confidence"]]
        print(f"\n  #{i} [{conf_emoji}] {s['symbol']} — {s['setup_type'].replace('_', ' ').title()}")
        print(f"  {'─' * 70}")
        print(f"  Price: ${s['technicals']['price']:.2f}  |  RSI: {s['technicals']['rsi']}  |  MFI: {s['technicals']['mfi']}  |  BB: {s['technicals']['bb_position']:.2f}")
        print(f"  Momentum: 5d {s['technicals']['mom_5d']:+.1f}%  20d {s['technicals']['mom_20d']:+.1f}%  60d {s['technicals']['mom_60d']:+.1f}%")
        print(f"  From 52w high: {s['technicals']['dist_52w_high']:+.1f}%  |  Vol spike: {s['technicals']['vol_spike']:.1f}x  |  ATR: {s['technicals']['atr_pct']:.1f}%")
        print(f"  SMAs: {'above' if s['technicals']['above_sma20'] else 'BELOW'} 20  |  {'above' if s['technicals']['above_sma50'] else 'BELOW'} 50  |  {'above' if s['technicals']['above_sma200'] else 'BELOW'} 200")
        print(f"")
        print(f"  PLAY: {s['play']}")
        print(f"  Est cost: ~${s['est_cost_per_contract']:.2f}/contract  |  {s['contracts_at_200_budget']} contracts @ ~$200 budget = ~${s['total_est_cost']:.0f}")
        print(f"  Target: +{s['target_move_pct']:.1f}%  |  Risk: {s['risk_pct']:.1f}%  |  R/R: {s['risk_reward']:.1f}x")
        print(f"  Confidence: {s['confidence']} (score {s['raw_score']}, composite {s['composite_rank_score']:.1f})")
        print(f"")
        print(f"  WHY:")
        for reason in s["reasons"]:
            print(f"    - {reason}")

    # Summary of all setups by type
    print(f"\n{'=' * 80}")
    print(f"  ALL SETUPS FOUND: {len(setups)}")
    by_type = {}
    for s in setups:
        by_type.setdefault(s["setup_type"], []).append(s)
    for t, items in by_type.items():
        syms = [f"{s['symbol']}({s['confidence'][0]})" for s in sorted(items, key=lambda x: x['composite_rank_score'], reverse=True)]
        print(f"  {t}: {', '.join(syms)}")
    print("=" * 80)


def save_results(setups, results, output_path):
    """Save full results to JSON."""
    ranked = sorted(setups, key=lambda s: s["composite_rank_score"], reverse=True)

    output = {
        "scan_time": datetime.now().isoformat(),
        "symbols_scanned": len(results),
        "setups_found": len(setups),
        "top_setups": ranked[:10],
        "all_setups": ranked,
        "all_technicals": results,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("Next Play Scanner v1")
    print(f"Scan time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Universe: {len(ALL_SYMBOLS)} symbols ({len(SECTOR_ETFS)} sectors, {len(INDEX_ETFS)} indices, {len(TOP30_STOCKS)} stocks)")

    # Fetch data
    data = fetch_data(ALL_SYMBOLS, period="1y")

    # Analyze each symbol
    print("\nComputing indicators...")
    results = []
    for sym, df in data.items():
        try:
            r = analyze_symbol(sym, df)
            results.append(r)
        except Exception as e:
            print(f"  Analysis failed for {sym}: {e}")

    print(f"Analyzed {len(results)} symbols")

    # Detect setups
    setups = detect_setups(results)

    # Print summary
    print_summary(setups, top_n=5)

    # Save results
    output_path = "/home/jupiter/Lvl3Quant/research/findings/next_play_scan_results.json"
    save_results(setups, results, output_path)

    return setups


if __name__ == "__main__":
    setups = main()
