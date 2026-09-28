#!/usr/bin/env python3
"""
Consolidation Breakout Backtester
==================================
Academic basis: Stocks trading in narrow ranges then breaking out tend to
continue.  Low volatility precedes directional moves (Garman-Klass vol
compression).  Related to Sewell (2011) technical analysis survey.

6 Variants:
  A. Narrow Range 7 (NR7)
  B. Bollinger Squeeze
  C. Low Vol Breakout
  D. Volume Breakout
  E. Multi-Factor Breakout (composite score)
  F. Weekly Breakout Portfolio (top-3 from E)

Universe : 24 large-cap US equities
OOT      : Jan 2022 – Jul 2026
Capital  : $645, $0 commission, 0.02 % slippage
Regime   : Bull = SPY > 200-SMA, Bear = SPY < 200-SMA

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation p < 0.05 (1000 iters)
  3. Regime gap < 0.5
  4. MaxDD > -50 %
  5. >= 20 trades
"""

import os, sys, json, time, warnings, logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    sys.exit("pip install yfinance")

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output" / "consolidation_breakout"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT / "cache"
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_PATH = ROOT / "data" / "consolidation_breakout_results.json"

# ── constants ──────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD",
    "CRM", "ADBE", "NFLX", "AVGO", "COST", "PEP", "LLY", "UNH",
    "V", "MA", "JPM", "HD", "INTC", "MU", "QCOM", "PYPL",
]

START_DATE = "2021-01-01"   # need lookback before OOT
OOT_START  = "2022-01-01"
END_DATE   = "2026-07-30"
CAPITAL    = 645.0
COMMISSION = 0.0
SLIPPAGE   = 0.0002  # 0.02 %

# ── data download ──────────────────────────────────────────────────────

def download_data(tickers, start, end):
    """Download OHLCV for all tickers + SPY.  Cache to disk."""
    all_tickers = list(set(tickers + ["SPY"]))
    cache = CACHE_DIR / "price_data.pkl"
    if cache.exists():
        age_hrs = (time.time() - cache.stat().st_mtime) / 3600
        if age_hrs < 12:
            log.info("Loading cached price data")
            return pd.read_pickle(cache)

    log.info(f"Downloading {len(all_tickers)} tickers from yfinance …")
    data = {}
    for tk in all_tickers:
        try:
            df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=True)
            if df is not None and len(df) > 100:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[tk] = df
        except Exception as e:
            log.warning(f"  {tk}: {e}")
    pd.to_pickle(data, cache)
    log.info(f"  Downloaded {len(data)} tickers")
    return data


# ── indicators ─────────────────────────────────────────────────────────

def add_indicators(df):
    """Add all needed technical indicators to a single-ticker df."""
    df = df.copy()
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    # SMAs
    df["SMA50"]  = c.rolling(50).mean()
    df["SMA200"] = c.rolling(200).mean()

    # Daily range
    df["Range"] = h - l

    # NR7: is today's range the narrowest of last 7?
    df["NR7"] = df["Range"] == df["Range"].rolling(7).min()

    # Bollinger Bands (20, 2)
    df["BB_mid"]   = c.rolling(20).mean()
    df["BB_std"]   = c.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * df["BB_std"]
    df["BB_lower"] = df["BB_mid"] - 2 * df["BB_std"]
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
    df["BB_width_6m_low"] = df["BB_width"].rolling(126).min()

    # 10-day realized vol (close-to-close annualized)
    df["RealVol10"] = c.pct_change().rolling(10).std() * np.sqrt(252)
    df["RealVol10_p20"] = df["RealVol10"].rolling(60).quantile(0.20)

    # 20-day high / volume avg
    df["High20"] = h.rolling(20).max()
    df["High10"] = h.rolling(10).max()
    df["VolAvg20"] = v.rolling(20).mean()

    # Volume increasing 3 days in a row
    v_inc = v > v.shift(1)
    df["VolUp3"] = v_inc & v_inc.shift(1) & v_inc.shift(2)

    return df


# ── trade simulation ───────────────────────────────────────────────────

def simulate_trades(signals, data, hold_days, capital, label):
    """
    signals: list of (date, ticker, signal_name)
    Returns list of trade dicts.
    """
    trades = []
    for entry_date, ticker, sig_name in signals:
        df = data.get(ticker)
        if df is None:
            continue
        if entry_date not in df.index:
            continue
        idx = df.index.get_loc(entry_date)
        if idx + 1 >= len(df):
            continue

        # Buy at close of signal day
        entry_price = df["Close"].iloc[idx]
        entry_price *= (1 + SLIPPAGE)  # slippage on entry

        # Exit after hold_days or at end of data
        exit_idx = min(idx + hold_days, len(df) - 1)
        exit_price = df["Close"].iloc[exit_idx]
        exit_price *= (1 - SLIPPAGE)  # slippage on exit

        exit_date = df.index[exit_idx]

        # How many shares can we buy with equal allocation?
        # Use full capital for simplicity (each signal is independent)
        shares = int(capital / entry_price) if entry_price > 0 else 0
        if shares == 0:
            continue

        pnl = shares * (exit_price - entry_price) - COMMISSION
        ret = (exit_price - entry_price) / entry_price

        trades.append({
            "variant": label,
            "ticker": ticker,
            "entry_date": str(entry_date.date()) if hasattr(entry_date, 'date') else str(entry_date)[:10],
            "exit_date": str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date)[:10],
            "entry_price": round(float(entry_price), 4),
            "exit_price": round(float(exit_price), 4),
            "shares": shares,
            "pnl": round(float(pnl), 2),
            "ret": round(float(ret), 6),
            "signal": sig_name,
        })
    return trades


# ── variant signal generators ──────────────────────────────────────────

def variant_A_NR7(data, oot_start):
    """Narrow Range 7: NR7 + above 50-SMA. Buy at close, hold 5 days."""
    signals = []
    for tk in TICKERS:
        df = data.get(tk)
        if df is None:
            continue
        df = add_indicators(df)
        mask = (df.index >= oot_start) & df["NR7"] & (df["Close"] > df["SMA50"])
        for dt in df.index[mask]:
            signals.append((dt, tk, "NR7"))
    return simulate_trades(signals, data, hold_days=5, capital=CAPITAL, label="A_NR7")


def variant_B_bollinger_squeeze(data, oot_start):
    """Bollinger Squeeze: BB width < 6mo low, next day breaks upper band. Hold 10."""
    signals = []
    for tk in TICKERS:
        df = data.get(tk)
        if df is None:
            continue
        df = add_indicators(df)
        for i in range(1, len(df)):
            if df.index[i] < pd.Timestamp(oot_start):
                continue
            # Yesterday BB width hit 6-month low
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            if pd.isna(prev["BB_width"]) or pd.isna(prev["BB_width_6m_low"]):
                continue
            if prev["BB_width"] <= prev["BB_width_6m_low"] * 1.01:  # within 1%
                # Today price breaks above upper band
                if curr["Close"] > prev["BB_upper"]:
                    signals.append((df.index[i], tk, "BB_Squeeze"))
    return simulate_trades(signals, data, hold_days=10, capital=CAPITAL, label="B_BollingerSqueeze")


def variant_C_low_vol_breakout(data, oot_start):
    """Low vol + new 20-day high + above 200-SMA. Hold 10."""
    signals = []
    for tk in TICKERS:
        df = data.get(tk)
        if df is None:
            continue
        df = add_indicators(df)
        for i in range(len(df)):
            if df.index[i] < pd.Timestamp(oot_start):
                continue
            row = df.iloc[i]
            if pd.isna(row["RealVol10"]) or pd.isna(row["RealVol10_p20"]) or pd.isna(row["SMA200"]):
                continue
            if (row["RealVol10"] < row["RealVol10_p20"]
                and row["High"] >= row["High20"]
                and row["Close"] > row["SMA200"]):
                signals.append((df.index[i], tk, "LowVolBreakout"))
    return simulate_trades(signals, data, hold_days=10, capital=CAPITAL, label="C_LowVolBreakout")


def variant_D_volume_breakout(data, oot_start):
    """Price breaks 20-day high with vol > 2x avg. Above 200-SMA. Hold 10."""
    signals = []
    for tk in TICKERS:
        df = data.get(tk)
        if df is None:
            continue
        df = add_indicators(df)
        for i in range(len(df)):
            if df.index[i] < pd.Timestamp(oot_start):
                continue
            row = df.iloc[i]
            if pd.isna(row["VolAvg20"]) or pd.isna(row["SMA200"]) or pd.isna(row["High20"]):
                continue
            if (row["High"] >= row["High20"]
                and row["Volume"] > 2.0 * row["VolAvg20"]
                and row["Close"] > row["SMA200"]):
                signals.append((df.index[i], tk, "VolumeBreakout"))
    return simulate_trades(signals, data, hold_days=10, capital=CAPITAL, label="D_VolumeBreakout")


def variant_E_multi_factor(data, oot_start):
    """Multi-Factor: score >= 3 AND price breaks 10-day high. Hold 10."""
    signals = []
    for tk in TICKERS:
        df = data.get(tk)
        if df is None:
            continue
        df = add_indicators(df)
        for i in range(len(df)):
            if df.index[i] < pd.Timestamp(oot_start):
                continue
            row = df.iloc[i]
            if pd.isna(row["SMA50"]) or pd.isna(row["RealVol10_p20"]) or pd.isna(row["High10"]):
                continue
            score = 0
            if row.get("NR7", False):
                score += 1
            if not pd.isna(row["RealVol10"]) and row["RealVol10"] < row["RealVol10_p20"]:
                score += 1
            if row["Close"] > row["SMA50"]:
                score += 1
            if row.get("VolUp3", False):
                score += 1
            if score >= 3 and row["High"] >= row["High10"]:
                signals.append((df.index[i], tk, f"MultiFactor_s{score}"))
    return simulate_trades(signals, data, hold_days=10, capital=CAPITAL, label="E_MultiFactor")


def variant_F_weekly_portfolio(data, oot_start):
    """Each week, buy top 3 by multi-factor score. Equal weight, hold 5 days."""
    # Pre-compute scores for all tickers / dates
    scores = {}  # date -> [(score, ticker)]
    for tk in TICKERS:
        df = data.get(tk)
        if df is None:
            continue
        df = add_indicators(df)
        for i in range(len(df)):
            if df.index[i] < pd.Timestamp(oot_start):
                continue
            row = df.iloc[i]
            if pd.isna(row["SMA50"]) or pd.isna(row["RealVol10_p20"]) or pd.isna(row["High10"]):
                continue
            score = 0
            if row.get("NR7", False):
                score += 1
            if not pd.isna(row["RealVol10"]) and row["RealVol10"] < row["RealVol10_p20"]:
                score += 1
            if row["Close"] > row["SMA50"]:
                score += 1
            if row.get("VolUp3", False):
                score += 1
            dt = df.index[i]
            if dt not in scores:
                scores[dt] = []
            scores[dt].append((score, tk))

    # Group by week (Monday)
    all_dates = sorted(scores.keys())
    weekly_signals = []
    last_week = None
    for dt in all_dates:
        week_id = dt.isocalendar()[:2]
        if week_id == last_week:
            continue
        last_week = week_id
        # Pick top 3 by score on this date
        candidates = sorted(scores[dt], key=lambda x: -x[0])
        for score, tk in candidates[:3]:
            if score >= 1:  # at least some signal
                weekly_signals.append((dt, tk, f"WeeklyTop3_s{score}"))

    return simulate_trades(weekly_signals, data, hold_days=5,
                           capital=CAPITAL / 3, label="F_WeeklyPortfolio")


# ── regime classification ──────────────────────────────────────────────

def get_regime(spy_data, date):
    """Bull if SPY > 200-SMA, else Bear."""
    if date not in spy_data.index:
        # Find closest prior date
        prior = spy_data.index[spy_data.index <= date]
        if len(prior) == 0:
            return "bull"
        date = prior[-1]
    idx = spy_data.index.get_loc(date)
    if idx < 200:
        return "bull"
    sma200 = spy_data["Close"].iloc[max(0, idx-199):idx+1].mean()
    return "bull" if spy_data["Close"].iloc[idx] > sma200 else "bear"


# ── performance metrics ───────────────────────────────────────────────

def calc_metrics(trades, spy_data):
    """Calculate performance metrics for a list of trades."""
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0, "avg_ret": 0, "total_pnl": 0,
            "sharpe": 0, "sortino": 0, "profit_factor": 0, "max_dd_pct": 0,
            "regime_sharpe_bull": 0, "regime_sharpe_bear": 0, "regime_gap": 0,
            "perm_p": 1.0, "pass_5gate": False, "gate_details": {},
        }

    rets = np.array([t["ret"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])

    n = len(trades)
    wins = np.sum(rets > 0)
    wr = wins / n if n > 0 else 0
    avg_ret = np.mean(rets)
    total_pnl = np.sum(pnls)

    # Sharpe (annualized, assume ~50 trades/year as rough scale)
    if np.std(rets) > 0:
        trades_per_year = max(n / 4.5, 1)  # OOT is ~4.5 years
        sharpe = (np.mean(rets) / np.std(rets)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0

    # Sortino
    downside = rets[rets < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = (np.mean(rets) / np.std(downside)) * np.sqrt(max(n / 4.5, 1))
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0

    # Profit factor
    gross_profit = np.sum(pnls[pnls > 0])
    gross_loss = abs(np.sum(pnls[pnls < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)

    # Max drawdown (equity curve from trades sorted by exit date)
    sorted_trades = sorted(trades, key=lambda t: t["exit_date"])
    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0
    for t in sorted_trades:
        equity += t["pnl"]
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

    # Regime split
    bull_rets, bear_rets = [], []
    for t in trades:
        regime = get_regime(spy_data, pd.Timestamp(t["entry_date"]))
        if regime == "bull":
            bull_rets.append(t["ret"])
        else:
            bear_rets.append(t["ret"])

    def _sharpe(r):
        r = np.array(r)
        if len(r) < 2 or np.std(r) == 0:
            return 0
        return (np.mean(r) / np.std(r)) * np.sqrt(max(len(r) / 4.5, 1))

    s_bull = _sharpe(bull_rets)
    s_bear = _sharpe(bear_rets)
    denom = max(abs(s_bull), abs(s_bear), 0.001)
    regime_gap = abs(s_bull - s_bear) / denom

    # Permutation test (1000 iters)
    np.random.seed(42)
    obs_mean = np.mean(rets)
    n_perm = 1000
    count = 0
    for _ in range(n_perm):
        shuffled = rets * np.random.choice([-1, 1], size=len(rets))
        if np.mean(shuffled) >= obs_mean:
            count += 1
    perm_p = count / n_perm

    # 5-gate validation
    g1 = sharpe > 0.5
    g2 = perm_p < 0.05
    g3 = regime_gap < 0.5
    g4 = max_dd > -0.50
    g5 = n >= 20
    pass_all = g1 and g2 and g3 and g4 and g5

    return {
        "n_trades": n,
        "win_rate": round(wr, 4),
        "avg_ret": round(float(avg_ret), 6),
        "total_pnl": round(float(total_pnl), 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(min(pf, 999)), 3),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "regime_sharpe_bull": round(float(s_bull), 3),
        "regime_sharpe_bear": round(float(s_bear), 3),
        "regime_gap": round(float(regime_gap), 3),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
        "perm_p": round(float(perm_p), 4),
        "pass_5gate": pass_all,
        "gate_details": {
            "G1_sharpe_gt_0.5": g1,
            "G2_perm_p_lt_0.05": g2,
            "G3_regime_gap_lt_0.5": g3,
            "G4_maxDD_gt_neg50pct": g4,
            "G5_min_20_trades": g5,
        },
    }


# ── main ───────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Consolidation Breakout Backtest")
    log.info("=" * 60)

    # Download data
    data = download_data(TICKERS, START_DATE, END_DATE)
    spy = data.get("SPY")
    if spy is None:
        log.error("Could not download SPY — aborting.")
        return

    # Add indicators to all tickers once for caching
    data_with_ind = {}
    for tk, df in data.items():
        data_with_ind[tk] = add_indicators(df)

    # Run all variants
    variants = {
        "A_NR7":              lambda: variant_A_NR7(data, OOT_START),
        "B_BollingerSqueeze": lambda: variant_B_bollinger_squeeze(data, OOT_START),
        "C_LowVolBreakout":   lambda: variant_C_low_vol_breakout(data, OOT_START),
        "D_VolumeBreakout":   lambda: variant_D_volume_breakout(data, OOT_START),
        "E_MultiFactor":      lambda: variant_E_multi_factor(data, OOT_START),
        "F_WeeklyPortfolio":  lambda: variant_F_weekly_portfolio(data, OOT_START),
    }

    results = {}
    all_trades = {}

    for name, fn in variants.items():
        log.info(f"\n{'─'*40}")
        log.info(f"Running variant: {name}")
        trades = fn()
        metrics = calc_metrics(trades, spy)
        results[name] = metrics
        all_trades[name] = trades

        gate = "PASS" if metrics["pass_5gate"] else "FAIL"
        log.info(f"  Trades: {metrics['n_trades']}")
        log.info(f"  Win Rate: {metrics['win_rate']:.1%}")
        log.info(f"  Sharpe: {metrics['sharpe']:.3f}")
        log.info(f"  Sortino: {metrics['sortino']:.3f}")
        log.info(f"  PF: {metrics['profit_factor']:.2f}")
        log.info(f"  MaxDD: {metrics['max_dd_pct']:.1f}%")
        log.info(f"  Perm p: {metrics['perm_p']:.4f}")
        log.info(f"  Regime gap: {metrics['regime_gap']:.3f}")
        log.info(f"  5-Gate: {gate}")
        for gk, gv in metrics["gate_details"].items():
            log.info(f"    {gk}: {'PASS' if gv else 'FAIL'}")

    # Summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    passed = [k for k, v in results.items() if v["pass_5gate"]]
    failed = [k for k, v in results.items() if not v["pass_5gate"]]
    log.info(f"  Passed 5-gate: {passed if passed else 'NONE'}")
    log.info(f"  Failed 5-gate: {failed if failed else 'NONE'}")

    # Save results
    output = {
        "metadata": {
            "strategy": "Consolidation Breakout",
            "academic_basis": "Garman-Klass vol compression, Sewell (2011)",
            "universe": TICKERS,
            "oot_period": f"{OOT_START} to {END_DATE}",
            "capital": CAPITAL,
            "commission": COMMISSION,
            "slippage_pct": SLIPPAGE * 100,
            "run_timestamp": datetime.now().isoformat(),
        },
        "variant_results": results,
        "passed_variants": passed,
        "failed_variants": failed,
        "sample_trades": {k: v[:5] for k, v in all_trades.items()},  # first 5 per variant
    }

    # Convert numpy types for JSON serialization
    def json_safe(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bool):
            return obj
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=json_safe)
    log.info(f"\nResults saved to {RESULTS_PATH}")

    return output


if __name__ == "__main__":
    main()
