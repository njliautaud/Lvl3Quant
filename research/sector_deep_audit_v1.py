#!/usr/bin/env python3
"""
Sector Deep Audit v1 — Multi-timeframe momentum, flow, trend analysis
Covers all 11 SPDR sector ETFs + SPY, QQQ, IWM, GLD, TLT
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Universe ──────────────────────────────────────────────────────────
SECTORS = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "XLC"]
BROAD   = ["SPY", "QQQ", "IWM", "GLD", "TLT"]
ALL_TICKERS = SECTORS + BROAD

SECTOR_NAMES = {
    "XLE": "Energy", "XLF": "Financials", "XLK": "Technology", "XLV": "Healthcare",
    "XLI": "Industrials", "XLU": "Utilities", "XLP": "Cons Staples", "XLY": "Cons Disc",
    "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communication",
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000", "GLD": "Gold", "TLT": "20Y Treasury"
}

# ── Technical Indicator Functions ─────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_adx(high, low, close, period=14):
    """Average Directional Index."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, min_periods=period).mean()
    return adx, plus_di, minus_di

def compute_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    obv = (direction * volume).cumsum()
    return obv

def compute_mfi(high, low, close, volume, period=14):
    """Money Flow Index."""
    tp = (high + low + close) / 3
    mf = tp * volume
    delta = tp.diff()
    pos_mf = mf.where(delta > 0, 0.0)
    neg_mf = mf.where(delta < 0, 0.0)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()

# ── Data Pull ─────────────────────────────────────────────────────────

def pull_data():
    end = datetime.now()
    start = end - timedelta(days=400)  # extra buffer for 200d SMA
    print(f"Pulling data from {start.date()} to {end.date()} ...")

    data = {}
    tickers_str = " ".join(ALL_TICKERS)
    raw = yf.download(tickers_str, start=start, end=end, group_by="ticker", progress=False)

    for t in ALL_TICKERS:
        try:
            df = raw[t].dropna(subset=["Close"])
            if len(df) < 60:
                print(f"  WARN: {t} only has {len(df)} rows, skipping")
                continue
            data[t] = df
        except Exception as e:
            print(f"  ERROR pulling {t}: {e}")

    print(f"  Got data for {len(data)} tickers, latest date: {list(data.values())[0].index[-1].date()}")
    return data

# ── Analysis ──────────────────────────────────────────────────────────

def analyze_ticker(df, spy_returns=None):
    """Compute all metrics for a single ticker."""
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    v = df["Volume"]

    latest = c.iloc[-1]

    # Momentum: returns
    ret_5d  = (c.iloc[-1] / c.iloc[-6] - 1) * 100 if len(c) >= 6 else np.nan
    ret_20d = (c.iloc[-1] / c.iloc[-21] - 1) * 100 if len(c) >= 21 else np.nan
    ret_60d = (c.iloc[-1] / c.iloc[-61] - 1) * 100 if len(c) >= 61 else np.nan

    # Trend: SMA positions
    sma20  = c.rolling(20).mean().iloc[-1]
    sma50  = c.rolling(50).mean().iloc[-1]
    sma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan

    above_20  = latest > sma20
    above_50  = latest > sma50
    above_200 = latest > sma200 if not np.isnan(sma200) else None

    pct_from_20  = (latest / sma20 - 1) * 100
    pct_from_50  = (latest / sma50 - 1) * 100
    pct_from_200 = (latest / sma200 - 1) * 100 if not np.isnan(sma200) else np.nan

    # ADX
    adx_val, plus_di, minus_di = compute_adx(h, l, c, 14)
    adx_now = adx_val.iloc[-1]
    plus_di_now = plus_di.iloc[-1]
    minus_di_now = minus_di.iloc[-1]
    trend_direction = "bullish" if plus_di_now > minus_di_now else "bearish"

    # RSI
    rsi = compute_rsi(c, 14).iloc[-1]

    # OBV slope (10d linear regression slope, normalized)
    obv = compute_obv(c, v)
    if len(obv) >= 10:
        obv_recent = obv.iloc[-10:].values
        x = np.arange(10)
        slope = np.polyfit(x, obv_recent, 1)[0]
        obv_slope_norm = slope / (np.abs(obv_recent).mean() + 1e-9) * 100
    else:
        obv_slope_norm = np.nan

    # MFI
    mfi = compute_mfi(h, l, c, v, 14).iloc[-1]

    # Volume ratio: 5d avg / 20d avg
    vol_5  = v.iloc[-5:].mean()
    vol_20 = v.iloc[-20:].mean()
    vol_ratio = vol_5 / vol_20 if vol_20 > 0 else np.nan

    # Volatility
    daily_ret = c.pct_change()
    vol_20d = daily_ret.iloc[-20:].std() * np.sqrt(252) * 100  # annualized %

    atr = compute_atr(h, l, c, 14).iloc[-1]
    atr_pct = (atr / latest) * 100

    # Relative strength vs SPY
    if spy_returns is not None and len(spy_returns) >= 20:
        ticker_ret = daily_ret.reindex(spy_returns.index)
        # Rolling correlation
        corr_20d = ticker_ret.rolling(20).corr(spy_returns).iloc[-1]

        # 20d alpha (excess cumulative return vs SPY)
        t_ret_20d = (c.iloc[-1] / c.iloc[-21] - 1) * 100 if len(c) >= 21 else np.nan
        spy_ret_20d_val = (spy_returns.iloc[-20:] + 1).prod() - 1
        spy_ret_20d_pct = spy_ret_20d_val * 100
        alpha_20d = t_ret_20d - spy_ret_20d_pct if not np.isnan(t_ret_20d) else np.nan
    else:
        corr_20d = np.nan
        alpha_20d = np.nan

    # Composite scores
    # Momentum score: weighted combo of returns (higher = stronger)
    mom_score = 0.2 * _zscore_safe(ret_5d) + 0.4 * _zscore_safe(ret_20d) + 0.4 * _zscore_safe(ret_60d)

    # Flow score: OBV slope + MFI + volume ratio
    flow_score = 0.4 * _zscore_safe(obv_slope_norm) + 0.4 * _zscore_safe(mfi - 50) + 0.2 * _zscore_safe(vol_ratio - 1)

    # Trend score: ADX strength + SMA alignment
    sma_alignment = sum([above_20, above_50, above_200 if above_200 is not None else False]) / 3
    trend_score = 0.5 * _zscore_safe(adx_now - 20) + 0.5 * (sma_alignment * 2 - 1)

    return {
        "price": round(latest, 2),
        "ret_5d": round(ret_5d, 2) if not np.isnan(ret_5d) else None,
        "ret_20d": round(ret_20d, 2) if not np.isnan(ret_20d) else None,
        "ret_60d": round(ret_60d, 2) if not np.isnan(ret_60d) else None,
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "sma200": round(sma200, 2) if not np.isnan(sma200) else None,
        "above_20": above_20,
        "above_50": above_50,
        "above_200": above_200,
        "pct_from_20": round(pct_from_20, 2),
        "pct_from_50": round(pct_from_50, 2),
        "pct_from_200": round(pct_from_200, 2) if not np.isnan(pct_from_200) else None,
        "adx": round(adx_now, 1),
        "adx_trend": trend_direction,
        "plus_di": round(plus_di_now, 1),
        "minus_di": round(minus_di_now, 1),
        "rsi": round(rsi, 1),
        "obv_slope_10d": round(obv_slope_norm, 3),
        "mfi": round(mfi, 1),
        "vol_ratio_5_20": round(vol_ratio, 3) if not np.isnan(vol_ratio) else None,
        "realized_vol_20d": round(vol_20d, 1),
        "atr_pct": round(atr_pct, 2),
        "corr_spy_20d": round(corr_20d, 3) if not np.isnan(corr_20d) else None,
        "alpha_vs_spy_20d": round(alpha_20d, 2) if alpha_20d is not None and not np.isnan(alpha_20d) else None,
        "momentum_score": round(mom_score, 3),
        "flow_score": round(flow_score, 3),
        "trend_score": round(trend_score, 3),
    }

# Track raw values for cross-sectional z-scoring
_raw_values = {}

def _zscore_safe(val):
    """Simple normalization: clip to [-3, 3] range, scale roughly."""
    if val is None or np.isnan(val):
        return 0
    return np.clip(val / max(abs(val), 1), -3, 3)

def classify_setups(results):
    """Identify key setups: strong momentum, accumulation (flow divergence), exhaustion."""
    setups = {
        "strongest_momentum": [],
        "accumulation_divergence": [],
        "distribution_exhaustion": [],
        "oversold_bounce_candidates": [],
        "strong_trend_continuation": [],
    }

    for ticker, m in results.items():
        if ticker in BROAD:
            continue  # classify sectors only

        name = SECTOR_NAMES.get(ticker, ticker)
        composite = m["momentum_score"] + m["flow_score"] + m["trend_score"]

        entry = {"ticker": ticker, "name": name, "composite": round(composite, 3)}
        entry.update({k: m[k] for k in ["ret_5d", "ret_20d", "ret_60d", "rsi", "mfi",
                                          "obv_slope_10d", "adx", "adx_trend", "alpha_vs_spy_20d",
                                          "above_20", "above_50", "above_200"]})

        # Strongest momentum: top composite
        setups["strongest_momentum"].append(entry)

        # Accumulation: price weak (ret_20d < 0 or near flat) but flow strong (MFI > 55, OBV slope positive)
        if m["ret_20d"] is not None and m["ret_20d"] < 2 and m["mfi"] > 52 and m["obv_slope_10d"] > 0:
            setups["accumulation_divergence"].append(entry)

        # Exhaustion: price strong but flow fading
        if m["ret_20d"] is not None and m["ret_20d"] > 2 and (m["mfi"] < 48 or m["obv_slope_10d"] < 0):
            setups["distribution_exhaustion"].append(entry)

        # Oversold bounce
        if m["rsi"] < 35 and m["mfi"] > 40:
            setups["oversold_bounce_candidates"].append(entry)

        # Strong trend continuation: ADX > 25, above all SMAs, momentum positive
        if (m["adx"] > 25 and m["above_20"] and m["above_50"]
            and (m["above_200"] is True) and m["ret_20d"] is not None and m["ret_20d"] > 0):
            setups["strong_trend_continuation"].append(entry)

    # Sort
    setups["strongest_momentum"].sort(key=lambda x: x["composite"], reverse=True)
    setups["accumulation_divergence"].sort(key=lambda x: x.get("mfi", 0), reverse=True)
    setups["distribution_exhaustion"].sort(key=lambda x: x.get("ret_20d", 0), reverse=True)

    return setups

def print_summary(results, setups):
    """Print clean summary tables."""

    print("\n" + "=" * 120)
    print("SECTOR DEEP AUDIT v1 — Multi-Timeframe Analysis")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 120)

    # ── Main Table ──
    print("\n── FULL DASHBOARD ─────────────────────────────────────────────────────────────────────────")
    header = f"{'Ticker':<6} {'Name':<14} {'Price':>8} {'5d%':>7} {'20d%':>7} {'60d%':>7} {'RSI':>5} {'ADX':>5} {'Trend':>7} {'MFI':>5} {'OBV':>8} {'VolR':>6} {'RVol%':>6} {'aSPY':>7} {'MomS':>6} {'FlwS':>6} {'TrdS':>6} {'Comp':>7}"
    print(header)
    print("-" * len(header))

    # Sort by composite
    sorted_tickers = sorted(results.keys(),
                           key=lambda t: results[t]["momentum_score"] + results[t]["flow_score"] + results[t]["trend_score"],
                           reverse=True)

    for t in sorted_tickers:
        m = results[t]
        composite = m["momentum_score"] + m["flow_score"] + m["trend_score"]
        name = SECTOR_NAMES.get(t, t)[:14]

        def fmt(v, w=7, dec=1):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return " " * w
            return f"{v:>{w}.{dec}f}"

        sma_flags = ""
        if m["above_20"]: sma_flags += "2"
        if m["above_50"]: sma_flags += "5"
        if m["above_200"]: sma_flags += "H"
        sma_flags = sma_flags if sma_flags else "---"

        trend_char = "B" if m["adx_trend"] == "bullish" else "b"

        print(f"{t:<6} {name:<14} {m['price']:>8.2f} "
              f"{fmt(m['ret_5d'])} {fmt(m['ret_20d'])} {fmt(m['ret_60d'])} "
              f"{m['rsi']:>5.1f} {m['adx']:>5.1f} {trend_char+'/'+sma_flags:>7} "
              f"{m['mfi']:>5.1f} {m['obv_slope_10d']:>8.3f} "
              f"{fmt(m['vol_ratio_5_20'], 6, 2)} "
              f"{m['realized_vol_20d']:>6.1f} "
              f"{fmt(m['alpha_vs_spy_20d'])} "
              f"{m['momentum_score']:>6.3f} {m['flow_score']:>6.3f} {m['trend_score']:>6.3f} "
              f"{composite:>7.3f}")

    # ── Key Setups ──
    print("\n── TOP MOMENTUM SECTORS ────────────────────────────────────────────────────────────────")
    for i, s in enumerate(setups["strongest_momentum"][:5]):
        flags = []
        if s["above_20"]: flags.append(">SMA20")
        if s["above_50"]: flags.append(">SMA50")
        if s["above_200"]: flags.append(">SMA200")
        print(f"  {i+1}. {s['ticker']} ({s['name']}): composite={s['composite']:+.3f} | "
              f"5d={s['ret_5d']:+.1f}% 20d={s['ret_20d']:+.1f}% 60d={s['ret_60d']:+.1f}% | "
              f"RSI={s['rsi']:.0f} ADX={s['adx']:.0f}({s['adx_trend']}) | "
              f"alpha_SPY={s['alpha_vs_spy_20d']:+.1f}% | {' '.join(flags)}")

    if setups["accumulation_divergence"]:
        print("\n── ACCUMULATION (price weak, flow strong = smart money buying) ─────────────────────────")
        for s in setups["accumulation_divergence"]:
            print(f"  {s['ticker']} ({s['name']}): 20d={s['ret_20d']:+.1f}% but MFI={s['mfi']:.0f}, "
                  f"OBV_slope={s['obv_slope_10d']:+.3f} | RSI={s['rsi']:.0f}")
    else:
        print("\n── ACCUMULATION: None detected ──")

    if setups["distribution_exhaustion"]:
        print("\n── EXHAUSTION/DISTRIBUTION (price strong, flow fading = smart money selling) ────────────")
        for s in setups["distribution_exhaustion"]:
            print(f"  {s['ticker']} ({s['name']}): 20d={s['ret_20d']:+.1f}% but MFI={s['mfi']:.0f}, "
                  f"OBV_slope={s['obv_slope_10d']:+.3f} | RSI={s['rsi']:.0f}")
    else:
        print("\n── EXHAUSTION: None detected ──")

    if setups["strong_trend_continuation"]:
        print("\n── STRONG TREND CONTINUATION (ADX>25, all SMAs aligned, positive momentum) ─────────────")
        for s in setups["strong_trend_continuation"]:
            print(f"  {s['ticker']} ({s['name']}): ADX={s['adx']:.0f} 20d={s['ret_20d']:+.1f}% "
                  f"alpha_SPY={s['alpha_vs_spy_20d']:+.1f}%")

    if setups["oversold_bounce_candidates"]:
        print("\n── OVERSOLD BOUNCE CANDIDATES (RSI<35, MFI>40) ─────────────────────────────────────────")
        for s in setups["oversold_bounce_candidates"]:
            print(f"  {s['ticker']} ({s['name']}): RSI={s['rsi']:.0f}, MFI={s['mfi']:.0f}, "
                  f"20d={s['ret_20d']:+.1f}%")

    # ── Broad Market Context ──
    print("\n── BROAD MARKET CONTEXT ────────────────────────────────────────────────────────────────")
    for t in BROAD:
        if t not in results:
            continue
        m = results[t]
        name = SECTOR_NAMES[t]
        trend = "BULL" if m["above_20"] and m["above_50"] else ("BEAR" if not m["above_20"] and not m["above_50"] else "MIXED")
        print(f"  {t} ({name}): {m['price']:.2f} | 5d={m['ret_5d']:+.1f}% 20d={m['ret_20d']:+.1f}% 60d={m['ret_60d']:+.1f}% | "
              f"RSI={m['rsi']:.0f} ADX={m['adx']:.0f} MFI={m['mfi']:.0f} | Trend={trend} | Vol={m['realized_vol_20d']:.1f}%")

    # ── Inter-market signals ──
    print("\n── INTER-MARKET SIGNALS ────────────────────────────────────────────────────────────────")
    if "GLD" in results and "TLT" in results and "SPY" in results:
        gld = results["GLD"]
        tlt = results["TLT"]
        spy = results["SPY"]

        # Risk-on vs risk-off
        if spy["ret_20d"] > 0 and tlt["ret_20d"] < 0:
            print("  Risk-On regime: SPY up, TLT down (bonds selling off = growth confidence)")
        elif spy["ret_20d"] < 0 and tlt["ret_20d"] > 0:
            print("  Risk-Off regime: SPY down, TLT up (flight to safety)")
        elif spy["ret_20d"] > 0 and tlt["ret_20d"] > 0:
            print("  Everything rally: SPY up, TLT up (liquidity-driven, watch for reversal)")
        else:
            print("  Broad weakness: SPY down, TLT down (tightening / stagflation signal)")

        if gld["ret_20d"] > 3:
            print(f"  Gold surging ({gld['ret_20d']:+.1f}% 20d) — inflation/uncertainty hedge active")
        elif gld["ret_20d"] < -3:
            print(f"  Gold weak ({gld['ret_20d']:+.1f}% 20d) — real rates rising or risk-on dominance")

    print("\n" + "=" * 120)


def main():
    data = pull_data()

    # Get SPY returns for relative strength
    spy_returns = data["SPY"]["Close"].pct_change() if "SPY" in data else None

    results = {}
    for t in ALL_TICKERS:
        if t not in data:
            continue
        results[t] = analyze_ticker(data[t], spy_returns)

    # Cross-sectional z-score normalization for sector composite
    if len([t for t in SECTORS if t in results]) >= 5:
        for score_key in ["momentum_score", "flow_score", "trend_score"]:
            sector_vals = [results[t][score_key] for t in SECTORS if t in results]
            mu = np.mean(sector_vals)
            sigma = np.std(sector_vals) if np.std(sector_vals) > 0.01 else 1
            for t in SECTORS:
                if t in results:
                    results[t][score_key] = round((results[t][score_key] - mu) / sigma, 3)

    setups = classify_setups(results)
    print_summary(results, setups)

    # ── Save JSON ──
    output = {
        "timestamp": datetime.now().isoformat(),
        "metrics": results,
        "setups": setups,
    }

    out_path = Path("/home/jupiter/Lvl3Quant/research/findings/sector_deep_audit_v1_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy/bool types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=convert)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
