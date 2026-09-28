#!/usr/bin/env python3
"""
Adversarial validation for Signal E — Inside Day After Dip
6-test battery: reimplementation, inverse signal, random timing,
sub-period stability, top-3 ticker removal, parameter sensitivity.

Baseline from intraday_pattern_signals_v1:
  Sharpe=0.744, N=106, WR=60.4%, PF=1.636, regime_gap=0.477
"""

import json, os, sys, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────
TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","AVGO","ORCL","CRM",
    "JPM","BAC","WFC","GS","MS","UNH","JNJ","LLY","PFE","MRK",
    "HD","MCD","NKE","LOW","SBUX","XOM","CVX","COP","PG","KO"
]
START = "2020-01-01"
END   = "2026-07-01"
COST_RT_PCT = 0.001  # 0.1% round-trip
POS_SIZE    = 300.0
MAX_CONC    = 2
TP_PCT      = 0.10
SL_PCT      = -0.15
MAX_HOLD    = 21
SEED        = 42
N_PERMS     = 1000

OUTPUT_DIR  = Path("/home/jupiter/Lvl3Quant/output/growth_research/inside_day_dip_adversarial")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE = {"sharpe": 0.744, "n_trades": 106, "wr": 0.604, "pf": 1.636, "regime_gap": 0.477}


# ── DATA ────────────────────────────────────────────────────────────
def download_data():
    """Download OHLCV for all tickers."""
    frames = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
            if len(df) > 100:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                frames[t] = df
        except Exception:
            pass
    return frames


def generate_signals(frames, dip_pct=-0.02, sma_window=20, sma_dist=0.95):
    """
    Signal E: Inside Day After Dip
    - Previous day close dropped > dip_pct vs 2-days-ago close
    - Today is an inside day (high < prev high, low > prev low)
    - Close < SMA(sma_window) * sma_dist  (i.e., >5% below SMA)
    """
    signals = []
    for ticker, df in frames.items():
        df = df.copy()
        df["sma"] = df["Close"].rolling(sma_window).mean()
        df["prev_close"] = df["Close"].shift(1)
        df["close_2d"] = df["Close"].shift(2)
        df["prev_high"] = df["High"].shift(1)
        df["prev_low"] = df["Low"].shift(1)

        for i in range(2, len(df)):
            row = df.iloc[i]
            prev_ret = (df.iloc[i-1]["Close"] - df.iloc[i-2]["Close"]) / df.iloc[i-2]["Close"]
            if prev_ret >= dip_pct:
                continue
            # Inside day
            if row["High"] >= df.iloc[i-1]["High"] or row["Low"] <= df.iloc[i-1]["Low"]:
                continue
            # Below SMA
            if pd.isna(row["sma"]) or row["Close"] >= row["sma"] * sma_dist:
                continue
            signals.append({
                "ticker": ticker,
                "entry_date": df.index[i],
                "entry_price": row["Close"] * (1 + COST_RT_PCT / 2),
            })
    return pd.DataFrame(signals)


def run_backtest(signals_df, frames, tp=TP_PCT, sl=SL_PCT, max_hold=MAX_HOLD):
    """Run backtest with max_concurrent enforcement."""
    if signals_df.empty:
        return {"n": 0, "wr": 0, "pf": 0, "sharpe": 0, "sortino": 0, "mdd": 0, "total_pnl": 0}

    signals_df = signals_df.sort_values("entry_date").reset_index(drop=True)
    trades = []
    active = []  # list of (exit_date_limit, ticker)

    for _, sig in signals_df.iterrows():
        # Remove expired active positions
        active = [(ed, tk) for ed, tk in active if ed > sig["entry_date"]]
        if len(active) >= MAX_CONC:
            continue

        ticker = sig["ticker"]
        entry_price = sig["entry_price"]
        entry_date = sig["entry_date"]

        df = frames[ticker]
        entry_idx = df.index.get_loc(entry_date)

        exit_price = None
        exit_date = None
        for j in range(1, max_hold + 1):
            if entry_idx + j >= len(df):
                break
            day = df.iloc[entry_idx + j]
            # Check TP (using high)
            if (day["High"] - entry_price) / entry_price >= tp:
                exit_price = entry_price * (1 + tp) * (1 - COST_RT_PCT / 2)
                exit_date = df.index[entry_idx + j]
                break
            # Check SL (using low)
            if (day["Low"] - entry_price) / entry_price <= sl:
                exit_price = entry_price * (1 + sl) * (1 - COST_RT_PCT / 2)
                exit_date = df.index[entry_idx + j]
                break

        if exit_price is None:
            # Max hold exit
            last_idx = min(entry_idx + max_hold, len(df) - 1)
            exit_price = df.iloc[last_idx]["Close"] * (1 - COST_RT_PCT / 2)
            exit_date = df.index[last_idx]

        pnl = (exit_price - entry_price) / entry_price * POS_SIZE
        trades.append({
            "ticker": ticker,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "pnl": pnl,
            "ret": (exit_price - entry_price) / entry_price,
        })
        active.append((exit_date, ticker))

    if not trades:
        return {"n": 0, "wr": 0, "pf": 0, "sharpe": 0, "sortino": 0, "mdd": 0, "total_pnl": 0}

    tdf = pd.DataFrame(trades)
    n = len(tdf)
    wr = (tdf["pnl"] > 0).mean()
    wins = tdf.loc[tdf["pnl"] > 0, "pnl"].sum()
    losses = abs(tdf.loc[tdf["pnl"] < 0, "pnl"].sum())
    pf = wins / losses if losses > 0 else 999
    rets = tdf["ret"].values
    sharpe = rets.mean() / rets.std() * np.sqrt(252 / 10) if rets.std() > 0 else 0  # ~10 day avg hold
    downside = rets[rets < 0]
    sortino = rets.mean() / downside.std() * np.sqrt(252 / 10) if len(downside) > 0 and downside.std() > 0 else 0

    # MDD on cumulative PnL
    cum = np.cumsum(tdf["pnl"].values)
    peak = np.maximum.accumulate(cum + POS_SIZE)
    dd = (cum + POS_SIZE - peak) / peak
    mdd = dd.min()

    return {
        "n": n, "wr": round(wr, 3), "pf": round(pf, 3),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "mdd": round(mdd, 4), "total_pnl": round(tdf["pnl"].sum(), 2),
        "mean_ret_pct": round(rets.mean() * 100, 2),
        "trades": tdf,
    }


# ── TESTS ───────────────────────────────────────────────────────────
def T1_reimplementation(frames):
    """Independent reimplementation — does our code reproduce baseline?"""
    signals = generate_signals(frames)
    result = run_backtest(signals, frames)
    lo = BASELINE["sharpe"] * 0.70
    hi = BASELINE["sharpe"] * 1.30
    passed = lo <= result["sharpe"] <= hi
    return {
        "test": "T1_reimplementation",
        "passed": str(passed),
        "sharpe": result["sharpe"],
        "n_trades": result["n"],
        "baseline_sharpe": BASELINE["sharpe"],
        "range": f"[{lo:.3f}, {hi:.3f}]",
        "detail": f"Sharpe={result['sharpe']} vs baseline={BASELINE['sharpe']} => range [{lo:.3f},{hi:.3f}]"
    }, result


def T2_inverse_signal(frames):
    """Inverse signal — flipping entry logic should destroy performance."""
    # Inverse: buy when NOT an inside day after dip (random non-signal days)
    signals = generate_signals(frames)
    signal_dates = set()
    if not signals.empty:
        for _, row in signals.iterrows():
            signal_dates.add((row["ticker"], row["entry_date"]))

    inv_signals = []
    np.random.seed(SEED)
    for ticker, df in frames.items():
        for i in range(20, len(df)):
            if (ticker, df.index[i]) in signal_dates:
                continue
            # Sample ~same frequency as signal
            if np.random.random() < len(signals) / (len(frames) * len(df)):
                inv_signals.append({
                    "ticker": ticker,
                    "entry_date": df.index[i],
                    "entry_price": df.iloc[i]["Close"] * (1 + COST_RT_PCT / 2),
                })

    inv_df = pd.DataFrame(inv_signals) if inv_signals else pd.DataFrame()
    inv_result = run_backtest(inv_df, frames) if not inv_df.empty else {"sharpe": 0, "n": 0}
    fwd_result = run_backtest(signals, frames)

    ratio = inv_result["sharpe"] / fwd_result["sharpe"] if fwd_result["sharpe"] != 0 else 999
    passed = ratio < 0.50
    return {
        "test": "T2_inverse_signal",
        "passed": str(passed),
        "inverse_sharpe": inv_result["sharpe"],
        "forward_sharpe": fwd_result["sharpe"],
        "ratio": round(ratio, 3),
        "inv_n_trades": inv_result.get("n", 0),
        "detail": f"Inverse Sharpe={inv_result['sharpe']}, ratio={ratio:.3f} (need <0.50), N={inv_result.get('n',0)}"
    }


def T3_random_timing(frames, fwd_sharpe):
    """Random timing permutation test — signal must beat 95th pctile of random."""
    signals = generate_signals(frames)
    if signals.empty:
        return {"test": "T3_random_timing", "passed": False, "detail": "No signals"}

    np.random.seed(SEED)
    perm_sharpes = []
    all_dates_by_ticker = {}
    for t, df in frames.items():
        all_dates_by_ticker[t] = df.index[20:].tolist()

    n_signals = len(signals)
    for perm in range(N_PERMS):
        perm_sigs = []
        for _ in range(n_signals):
            t = np.random.choice(TICKERS)
            if t not in all_dates_by_ticker or not all_dates_by_ticker[t]:
                continue
            d = np.random.choice(all_dates_by_ticker[t])
            if t in frames:
                idx = frames[t].index.get_loc(d)
                perm_sigs.append({
                    "ticker": t,
                    "entry_date": d,
                    "entry_price": frames[t].iloc[idx]["Close"] * (1 + COST_RT_PCT / 2),
                })
        if perm_sigs:
            pdf = pd.DataFrame(perm_sigs)
            pr = run_backtest(pdf, frames)
            perm_sharpes.append(pr["sharpe"])

    perm_sharpes = np.array(perm_sharpes)
    p_val = (perm_sharpes >= fwd_sharpe).mean()
    passed = p_val < 0.05

    return {
        "test": "T3_random_timing",
        "passed": passed,
        "perm_p": round(p_val, 4),
        "actual_sharpe": fwd_sharpe,
        "perm_mean": round(perm_sharpes.mean(), 3),
        "perm_std": round(perm_sharpes.std(), 3),
        "perm_95th": round(np.percentile(perm_sharpes, 95), 3),
        "n_perms": N_PERMS,
        "detail": f"p={p_val:.4f} (need <0.05), actual={fwd_sharpe}, perm_mean={perm_sharpes.mean():.3f}"
    }


def T4_subperiod_stability(trades_df):
    """All 4 sub-periods must have positive returns."""
    if trades_df is None or trades_df.empty:
        return {"test": "T4_subperiod_stability", "passed": False, "detail": "No trades"}

    trades_df = trades_df.copy()
    trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])

    # Split into 4 roughly equal periods
    dates_sorted = trades_df["entry_date"].sort_values()
    n = len(dates_sorted)
    cuts = [0, n // 4, n // 2, 3 * n // 4, n]

    period_sharpes = []
    period_returns = []
    period_dates = []
    period_ns = []

    for i in range(4):
        mask = (trades_df["entry_date"] >= dates_sorted.iloc[cuts[i]]) & \
               (trades_df["entry_date"] < dates_sorted.iloc[min(cuts[i + 1], n - 1)] + pd.Timedelta(days=1))
        sub = trades_df[mask]
        if len(sub) == 0:
            period_sharpes.append(0)
            period_returns.append(0)
            period_ns.append(0)
            period_dates.append("empty")
            continue

        rets = sub["ret"].values
        s = rets.mean() / rets.std() * np.sqrt(252 / 10) if rets.std() > 0 else 0
        period_sharpes.append(round(s, 3))
        period_returns.append(round(rets.mean() * 100, 2))
        d0 = sub["entry_date"].min().strftime("%Y-%m")
        d1 = sub["entry_date"].max().strftime("%Y-%m")
        period_dates.append(f"{d0}..{d1}")
        period_ns.append(len(sub))

    all_positive = all(r > 0 for r in period_returns)

    return {
        "test": "T4_subperiod_stability",
        "passed": all_positive,
        "period_sharpes": period_sharpes,
        "period_returns": period_returns,
        "period_dates": period_dates,
        "period_ns": period_ns,
        "detail": f"Periods: {period_dates}\n       Returns%: {period_returns}, Sharpes: {period_sharpes}, all positive: {all_positive}"
    }


def T5_top3_ticker_removal(frames, baseline_sharpe):
    """Remove top 3 most traded tickers — Sharpe shouldn't drop >50%."""
    signals = generate_signals(frames)
    if signals.empty:
        return {"test": "T5_top3_ticker_removal", "passed": False, "detail": "No signals"}

    result = run_backtest(signals, frames)
    if "trades" not in result or result["trades"] is None:
        return {"test": "T5_top3_ticker_removal", "passed": False, "detail": "No trades"}

    counts = result["trades"]["ticker"].value_counts()
    top3 = counts.head(3).index.tolist()
    top3_counts = counts.head(3).values.tolist()

    reduced_signals = signals[~signals["ticker"].isin(top3)]
    reduced_result = run_backtest(reduced_signals, frames)

    drop = (reduced_result["sharpe"] - baseline_sharpe) / baseline_sharpe if baseline_sharpe != 0 else 0
    passed = drop > -0.50  # Less than 50% drop

    return {
        "test": "T5_top3_ticker_removal",
        "passed": str(passed),
        "top3_removed": top3,
        "top3_counts": [int(c) for c in top3_counts],
        "normal_sharpe": baseline_sharpe,
        "reduced_sharpe": reduced_result["sharpe"],
        "drop_pct": round(drop, 3),
        "reduced_n_trades": reduced_result["n"],
        "detail": f"Removed {top3}: Sharpe {baseline_sharpe} -> {reduced_result['sharpe']}, drop={drop*100:.1f}% (need <50%), N={reduced_result['n']}"
    }


def T6_parameter_sensitivity(frames):
    """Test across 150 parameter combinations — ≥80% should have Sharpe > 0.30."""
    dip_pcts = [-0.01, -0.015, -0.02, -0.025, -0.03, -0.035]
    sma_windows = [10, 15, 20, 25, 30]
    sma_dists = [0.92, 0.93, 0.94, 0.95, 0.96]

    results = []
    for dp in dip_pcts:
        for sw in sma_windows:
            for sd in sma_dists:
                sigs = generate_signals(frames, dip_pct=dp, sma_window=sw, sma_dist=sd)
                r = run_backtest(sigs, frames)
                results.append(r["sharpe"])

    results = np.array(results)
    above = (results > 0.30).sum()
    pct = above / len(results)
    passed = pct >= 0.80

    return {
        "test": "T6_parameter_sensitivity",
        "passed": passed,
        "total_combos": len(results),
        "above_030": int(above),
        "pct_above": round(pct, 2),
        "median_sharpe": round(np.median(results), 3),
        "mean_sharpe": round(np.mean(results), 3),
        "min_sharpe": round(results.min(), 3),
        "max_sharpe": round(results.max(), 3),
        "detail": f"{above}/{len(results)} ({pct*100:.1f}%) have Sharpe>0.30 (need >=80%), median={np.median(results):.3f}, mean={np.mean(results):.3f}"
    }


# ── MAIN ────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 60)
    print("ADVERSARIAL VALIDATION: Signal E — Inside Day After Dip")
    print("=" * 60)

    print("\n[1/7] Downloading data...")
    frames = download_data()
    print(f"  Got {len(frames)} tickers")

    print("\n[2/7] T1 — Reimplementation check...")
    t1_result, fwd = T1_reimplementation(frames)
    print(f"  {t1_result['detail']}")
    print(f"  PASS: {t1_result['passed']}")

    fwd_sharpe = fwd["sharpe"]
    fwd_trades = fwd.get("trades", None)

    print("\n[3/7] T2 — Inverse signal...")
    t2_result = T2_inverse_signal(frames)
    print(f"  {t2_result['detail']}")
    print(f"  PASS: {t2_result['passed']}")

    print("\n[4/7] T3 — Random timing permutation (1000 perms)...")
    t3_result = T3_random_timing(frames, fwd_sharpe)
    print(f"  {t3_result['detail']}")
    print(f"  PASS: {t3_result['passed']}")

    print("\n[5/7] T4 — Sub-period stability...")
    t4_result = T4_subperiod_stability(fwd_trades)
    print(f"  {t4_result['detail']}")
    print(f"  PASS: {t4_result['passed']}")

    print("\n[6/7] T5 — Top-3 ticker removal...")
    t5_result = T5_top3_ticker_removal(frames, fwd_sharpe)
    print(f"  {t5_result['detail']}")
    print(f"  PASS: {t5_result['passed']}")

    print("\n[7/7] T6 — Parameter sensitivity (150 combos)...")
    t6_result = T6_parameter_sensitivity(frames)
    print(f"  {t6_result['detail']}")
    print(f"  PASS: {t6_result['passed']}")

    # Summary
    tests = [t1_result, t2_result, t3_result, t4_result, t5_result, t6_result]
    n_passed = sum(1 for t in tests if str(t["passed"]).lower() == "true")

    elapsed = (time.time() - t0) / 60

    verdict = "VALIDATED" if n_passed >= 5 else "REJECTED"
    print(f"\n{'=' * 60}")
    print(f"VERDICT: {verdict} ({n_passed}/6 tests passed)")
    print(f"Elapsed: {elapsed:.1f} min")
    print(f"{'=' * 60}")

    # Clean trades from results for JSON serialization
    reimpl = {k: v for k, v in fwd.items() if k != "trades"}

    output = {
        "strategy": "Inside Day After Dip (Signal E)",
        "run_timestamp": datetime.now().isoformat(),
        "baseline": BASELINE,
        "reimplemented_metrics": reimpl,
        "tests": tests,
        "summary": {
            "n_passed": n_passed,
            "n_total": 6,
            "overall_verdict": verdict,
            "elapsed_min": round(elapsed, 1),
        }
    }

    out_path = OUTPUT_DIR / f"adversarial_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Also save as latest
    with open(OUTPUT_DIR / "latest_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)


if __name__ == "__main__":
    main()
