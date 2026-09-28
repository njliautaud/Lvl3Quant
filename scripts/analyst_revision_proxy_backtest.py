#!/usr/bin/env python3
"""
Analyst Revision Momentum Proxy Backtest
=========================================
Academic research: stocks with upward analyst estimate revisions outperform 2-6 months.
Proxy: post-earnings gap-up + continuation drift ≈ revision cycle signal.

6 Variants (A-F), 5-gate validation, OOT: Jan 2022 – Jul 2026.
Account: $645, slippage: 0.02% each way.
"""

import json, warnings, itertools
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "WFC", "GS", "MS",
    "JNJ", "PFE", "UNH", "ABBV", "MRK",
    "HD", "LOW", "MCD", "DIS", "NFLX",
    "CRM", "ADBE", "AMD", "INTC",
    "PYPL", "SQ", "UBER", "ABNB",
]

SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech",
    "META": "Tech", "NVDA": "Tech", "TSLA": "Tech", "CRM": "Tech",
    "ADBE": "Tech", "AMD": "Tech", "INTC": "Tech",
    "JPM": "Fin", "BAC": "Fin", "WFC": "Fin", "GS": "Fin", "MS": "Fin",
    "JNJ": "Health", "PFE": "Health", "UNH": "Health", "ABBV": "Health", "MRK": "Health",
    "HD": "Consumer", "LOW": "Consumer", "MCD": "Consumer", "DIS": "Consumer",
    "NFLX": "Consumer", "PYPL": "Fin", "SQ": "Fin",
    "UBER": "Tech", "ABNB": "Consumer",
}

SECTOR_ETFS = {
    "Tech": "XLK", "Fin": "XLF", "Health": "XLV", "Consumer": "XLY",
}

ACCOUNT = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
DATA_START = "2021-06-01"  # extra lookback for SMA200 / volume avg
PERMUTATION_ITERS = 1000
RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/analyst_revision_proxy_results.json"

np.random.seed(42)


# ── Data Download ───────────────────────────────────────────────────────────
def download_data():
    """Download OHLCV for universe + sector ETFs."""
    tickers = list(set(UNIVERSE + list(SECTOR_ETFS.values())))
    print(f"Downloading {len(tickers)} tickers from {DATA_START} to {OOT_END} ...")
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
            if df is not None and len(df) > 100:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
        except Exception as e:
            print(f"  WARN: {t} download failed: {e}")
    print(f"  Got data for {len(data)} tickers")
    return data


# ── Feature Helpers ─────────────────────────────────────────────────────────
def compute_features(df):
    """Add gap%, volume ratio, SMA200, RSI14 columns."""
    df = df.copy()
    df["gap_pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["vol_avg20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_avg20"]
    df["sma200"] = df["Close"].rolling(200).mean()
    # RSI 14
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - (100 / (1 + rs))
    # Daily return
    df["ret"] = df["Close"].pct_change()
    return df


def detect_gap_days(df, gap_thresh=0.03, vol_mult=2.0):
    """Find days where stock gapped up >gap_thresh on high volume."""
    mask = (df["gap_pct"] > gap_thresh) & (df["vol_ratio"] > vol_mult)
    return df.index[mask]


# ── Trade Simulation ────────────────────────────────────────────────────────
def simulate_trades(entry_dates, df, hold_days, account):
    """
    Given entry dates and a price df, simulate equal-weight trades.
    Returns list of trade dicts with pnl info.
    """
    trades = []
    oot_start = pd.Timestamp(OOT_START)
    for entry_date in entry_dates:
        if entry_date < oot_start:
            continue
        idx = df.index.get_loc(entry_date)
        if idx + hold_days >= len(df):
            continue
        entry_price = df["Open"].iloc[idx]  # buy at open
        if pd.isna(entry_price) or entry_price <= 0:
            continue
        exit_price = df["Close"].iloc[idx + hold_days]
        if pd.isna(exit_price):
            continue
        # Slippage both ways
        adj_entry = entry_price * (1 + SLIPPAGE_PCT)
        adj_exit = exit_price * (1 - SLIPPAGE_PCT)
        # Position size: full account
        shares = int(account / adj_entry)
        if shares < 1:
            continue
        pnl = (adj_exit - adj_entry) * shares
        ret = (adj_exit / adj_entry) - 1
        trades.append({
            "entry_date": str(entry_date.date()),
            "exit_date": str(df.index[idx + hold_days].date()),
            "entry_price": round(adj_entry, 4),
            "exit_price": round(adj_exit, 4),
            "shares": shares,
            "pnl": round(pnl, 2),
            "ret": round(ret, 6),
            "ticker": df.attrs.get("ticker", "?"),
        })
    return trades


# ── Variant Strategies ──────────────────────────────────────────────────────
def variant_a(data):
    """Post-gap drift: gap >3% on 2x vol, buy day+3, hold 20d."""
    all_trades = []
    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = compute_features(data[ticker])
        df.attrs["ticker"] = ticker
        gap_days = detect_gap_days(df)
        entry_dates = []
        for gd in gap_days:
            loc = df.index.get_loc(gd)
            if loc + 3 < len(df):
                entry_dates.append(df.index[loc + 3])
        all_trades.extend(simulate_trades(entry_dates, df, 20, ACCOUNT))
    return all_trades


def variant_b(data):
    """Continuation: gap >3% AND next 3 days all positive → buy day+4, hold 15d."""
    all_trades = []
    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = compute_features(data[ticker])
        df.attrs["ticker"] = ticker
        gap_days = detect_gap_days(df)
        entry_dates = []
        for gd in gap_days:
            loc = df.index.get_loc(gd)
            if loc + 4 >= len(df):
                continue
            # Check next 3 days are all positive
            next3 = df["ret"].iloc[loc + 1: loc + 4]
            if len(next3) == 3 and (next3 > 0).all():
                entry_dates.append(df.index[loc + 4])
        all_trades.extend(simulate_trades(entry_dates, df, 15, ACCOUNT))
    return all_trades


def variant_c(data):
    """Extended hold: same as A but hold 40d."""
    all_trades = []
    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = compute_features(data[ticker])
        df.attrs["ticker"] = ticker
        gap_days = detect_gap_days(df)
        entry_dates = []
        for gd in gap_days:
            loc = df.index.get_loc(gd)
            if loc + 3 < len(df):
                entry_dates.append(df.index[loc + 3])
        all_trades.extend(simulate_trades(entry_dates, df, 40, ACCOUNT))
    return all_trades


def variant_d(data):
    """Quality filter: gap >3% AND above SMA200 → buy day+3, hold 20d."""
    all_trades = []
    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = compute_features(data[ticker])
        df.attrs["ticker"] = ticker
        gap_days = detect_gap_days(df)
        entry_dates = []
        for gd in gap_days:
            loc = df.index.get_loc(gd)
            if loc + 3 >= len(df):
                continue
            if pd.notna(df["sma200"].iloc[loc]) and df["Close"].iloc[loc] > df["sma200"].iloc[loc]:
                entry_dates.append(df.index[loc + 3])
        all_trades.extend(simulate_trades(entry_dates, df, 20, ACCOUNT))
    return all_trades


def variant_e(data):
    """Sector momentum: 3+ stocks gap up same week → buy sector ETF 20d."""
    # Collect all gap events with week info
    gap_events = []
    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = compute_features(data[ticker])
        gap_days = detect_gap_days(df)
        for gd in gap_days:
            sector = SECTOR_MAP.get(ticker, None)
            if sector:
                gap_events.append({"date": gd, "week": gd.isocalendar()[1],
                                   "year": gd.year, "sector": sector, "ticker": ticker})

    if not gap_events:
        return []

    gap_df = pd.DataFrame(gap_events)
    # Group by year-week-sector, count unique tickers
    grouped = gap_df.groupby(["year", "week", "sector"])["ticker"].nunique().reset_index()
    grouped.columns = ["year", "week", "sector", "count"]
    sector_signals = grouped[grouped["count"] >= 3]

    all_trades = []
    for _, row in sector_signals.iterrows():
        etf = SECTOR_ETFS.get(row["sector"])
        if etf is None or etf not in data:
            continue
        etf_df = compute_features(data[etf])
        etf_df.attrs["ticker"] = etf
        # Find the Monday of that week as entry
        week_gaps = gap_df[(gap_df["year"] == row["year"]) &
                          (gap_df["week"] == row["week"]) &
                          (gap_df["sector"] == row["sector"])]
        last_gap_date = week_gaps["date"].max()
        loc = etf_df.index.get_loc(etf_df.index[etf_df.index >= last_gap_date][0]) if any(etf_df.index >= last_gap_date) else None
        if loc is None:
            continue
        if loc + 1 < len(etf_df):
            entry_date = etf_df.index[loc + 1]  # day after last gap
            all_trades.extend(simulate_trades([entry_date], etf_df, 20, ACCOUNT))
    return all_trades


def variant_f(data):
    """Reversal avoidance: gap >3% but RSI(14) <80 → buy day+3, hold 20d."""
    all_trades = []
    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = compute_features(data[ticker])
        df.attrs["ticker"] = ticker
        gap_days = detect_gap_days(df)
        entry_dates = []
        for gd in gap_days:
            loc = df.index.get_loc(gd)
            if loc + 3 >= len(df):
                continue
            if pd.notna(df["rsi14"].iloc[loc]) and df["rsi14"].iloc[loc] < 80:
                entry_dates.append(df.index[loc + 3])
        all_trades.extend(simulate_trades(entry_dates, df, 20, ACCOUNT))
    return all_trades


# ── Validation Framework ───────────────────────────────────────────────────
def calc_metrics(trades):
    """Calculate key metrics from trade list."""
    if not trades:
        return {"n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
                "total_pnl": 0, "max_dd_pct": 0, "avg_ret": 0, "med_ret": 0}

    rets = np.array([t["ret"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    n = len(rets)
    avg_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1e-9

    # Annualize assuming ~20 trades/year avg hold ~20 days
    sharpe = (avg_ret / std_ret) * np.sqrt(min(n, 252)) if std_ret > 0 else 0

    # Sortino
    downside = rets[rets < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg_ret / down_std) * np.sqrt(min(n, 252)) if down_std > 0 else 0

    # Profit factor
    gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Win rate
    wr = (rets > 0).sum() / n

    # Max drawdown on equity curve
    equity = ACCOUNT + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 3),
        "total_pnl": round(float(pnls.sum()), 2),
        "max_dd_pct": round(float(max_dd) * 100, 2),
        "avg_ret": round(float(avg_ret) * 100, 4),
        "med_ret": round(float(np.median(rets)) * 100, 4),
    }


def permutation_test(trades, n_iter=PERMUTATION_ITERS):
    """Permutation test: shuffle trade signs, compute p-value for observed Sharpe."""
    if len(trades) < 5:
        return 1.0
    rets = np.array([t["ret"] for t in trades])
    observed_mean = np.mean(rets)
    count_ge = 0
    for _ in range(n_iter):
        signs = np.random.choice([-1, 1], size=len(rets))
        shuffled = rets * signs
        if np.mean(shuffled) >= observed_mean:
            count_ge += 1
    return count_ge / n_iter


def regime_analysis(trades, spy_data):
    """Split trades into bull/bear regimes using SPY SMA200."""
    if not trades or spy_data is None:
        return 1.0, {}, {}
    spy = compute_features(spy_data)
    bull_trades, bear_trades = [], []
    for t in trades:
        ed = pd.Timestamp(t["entry_date"])
        if ed in spy.index:
            loc = spy.index.get_loc(ed)
        else:
            candidates = spy.index[spy.index <= ed]
            if len(candidates) == 0:
                continue
            loc = spy.index.get_loc(candidates[-1])
        if pd.notna(spy["sma200"].iloc[loc]) and spy["Close"].iloc[loc] > spy["sma200"].iloc[loc]:
            bull_trades.append(t)
        else:
            bear_trades.append(t)

    bull_m = calc_metrics(bull_trades)
    bear_m = calc_metrics(bear_trades)
    s_bull = bull_m["sharpe"]
    s_bear = bear_m["sharpe"]
    denom = max(abs(s_bull), abs(s_bear), 1e-9)
    regime_gap = abs(s_bull - s_bear) / denom
    return round(regime_gap, 3), bull_m, bear_m


def validate_5gate(trades, spy_data, label):
    """Run 5-gate validation. Returns dict with pass/fail for each gate."""
    metrics = calc_metrics(trades)
    perm_p = permutation_test(trades)
    regime_gap, bull_m, bear_m = regime_analysis(trades, spy_data)

    gates = {
        "G1_sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "G2_perm_p_lt_0.05": perm_p < 0.05,
        "G3_regime_gap_lt_0.5": regime_gap < 0.5,
        "G4_maxdd_gt_neg50": metrics["max_dd_pct"] > -50,
        "G5_min_20_trades": metrics["n_trades"] >= 20,
    }
    passed = sum(gates.values())

    print(f"\n{'='*60}")
    print(f"  VARIANT {label}")
    print(f"{'='*60}")
    print(f"  Trades: {metrics['n_trades']}  |  Win Rate: {metrics['wr']*100:.1f}%")
    print(f"  Total PnL: ${metrics['total_pnl']:.2f}  |  Avg Ret: {metrics['avg_ret']:.3f}%  |  Med Ret: {metrics['med_ret']:.3f}%")
    print(f"  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}  |  PF: {metrics['pf']:.3f}")
    print(f"  MaxDD: {metrics['max_dd_pct']:.2f}%")
    print(f"  Perm p-value: {perm_p:.4f}")
    print(f"  Regime gap: {regime_gap:.3f}  (Bull Sharpe: {bull_m.get('sharpe','N/A')}, Bear Sharpe: {bear_m.get('sharpe','N/A')})")
    print(f"  ──────────────────────────────────────")
    for g, v in gates.items():
        status = "PASS" if v else "FAIL"
        print(f"  {g}: {status}")
    print(f"  GATES PASSED: {passed}/5  {'→ VIABLE' if passed == 5 else '→ NOT VIABLE'}")

    return {
        "variant": label,
        "metrics": metrics,
        "perm_p": round(perm_p, 4),
        "regime_gap": regime_gap,
        "bull_metrics": bull_m,
        "bear_metrics": bear_m,
        "gates": {k: bool(v) for k, v in gates.items()},
        "gates_passed": passed,
        "viable": passed == 5,
        "trades": trades,
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  ANALYST REVISION MOMENTUM PROXY BACKTEST")
    print(f"  OOT: {OOT_START} to {OOT_END}  |  Account: ${ACCOUNT}")
    print("=" * 60)

    data = download_data()

    # Download SPY for regime analysis
    spy = yf.download("SPY", start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    variants = {
        "A (Post-Gap Drift, 20d)": variant_a,
        "B (Continuation Momentum, 15d)": variant_b,
        "C (Extended Hold, 40d)": variant_c,
        "D (Quality SMA200 Filter, 20d)": variant_d,
        "E (Sector ETF Momentum, 20d)": variant_e,
        "F (RSI<80 Reversal Avoidance, 20d)": variant_f,
    }

    results = {}
    for label, func in variants.items():
        trades = func(data)
        result = validate_5gate(trades, spy, label)
        # Don't store individual trades in JSON (too large)
        result_clean = {k: v for k, v in result.items() if k != "trades"}
        results[label] = result_clean

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Variant':<42} {'Trades':>6} {'Sharpe':>7} {'PF':>6} {'WR':>6} {'PnL':>10} {'Gates':>6}")
    print(f"  {'-'*42} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*10} {'-'*6}")
    viable_count = 0
    for label, r in results.items():
        m = r["metrics"]
        tag = " *" if r["viable"] else ""
        print(f"  {label:<42} {m['n_trades']:>6} {m['sharpe']:>7.3f} {m['pf']:>6.2f} {m['wr']*100:>5.1f}% ${m['total_pnl']:>9.2f} {r['gates_passed']:>3}/5{tag}")
        if r["viable"]:
            viable_count += 1

    print(f"\n  Viable strategies (5/5 gates): {viable_count}/{len(results)}")

    # ── Save ────────────────────────────────────────────────────────────
    output = {
        "strategy": "Analyst Revision Momentum Proxy",
        "run_date": datetime.now().isoformat(),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "account": ACCOUNT,
        "slippage_pct": SLIPPAGE_PCT,
        "universe_size": len(UNIVERSE),
        "permutation_iters": PERMUTATION_ITERS,
        "variants": results,
    }
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
