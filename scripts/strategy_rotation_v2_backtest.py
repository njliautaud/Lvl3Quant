#!/usr/bin/env python3
"""
Strategy Rotation v2 Backtest — ENHANCED variants of the validated Strategy Rotation A.

Base Strategy (Rotation A — Sharpe 2.13, regime gap 0.128):
  Bull (SPY > 200-SMA): Long QQQ
  Bear dip (SPY < 200-SMA, weekly drop > 3%): Buy SPY for 5-day contrarian hold
  VIX spike (VIX > 25, declining): VIX fade, buy SPY
  Otherwise: Cash

Variants:
  A) Baseline reproduction of Strategy Rotation A
  B) Leveraged bull — TQQQ (3x) at 33% position size during bull
  C) Multi-timeframe trend — 50/100/200 SMA consensus
  D) VIX-scaled sizing — inverse VIX percentile position sizing in bull
  E) Sector rotation within bull — QQQ vs XLK/XLC/XLY relative strength
  F) Dynamic contrarian — sliding VIX-based dip threshold, extended hold
"""

import json, sys, warnings, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────
OOT_START = "2022-01-01"
OOT_END   = "2026-07-28"
STARTING_CAPITAL = 645.0
PERM_ITERS = 1000
SLIPPAGE = 0.0002  # 0.02% per trade
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/strategy_rotation_v2_results.json")

# 5-gate thresholds
GATE_SHARPE   = 0.5
GATE_PERM_P   = 0.05
GATE_REGIME   = 0.50
GATE_MDD      = -0.50
GATE_TRADES   = 20


# ── HELPERS ─────────────────────────────────────────────────────────────
def fetch_data():
    """Download SPY, QQQ, TQQQ, XLK, XLC, XLY, ^VIX."""
    print("[1/6] Downloading market data via yfinance …")
    tickers = ["SPY", "QQQ", "TQQQ", "XLK", "XLC", "XLY", "^VIX"]
    # Enough history for 200-SMA lookback before OOT
    raw = yf.download(tickers, start="2021-01-01", end=OOT_END,
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    df = pd.DataFrame(index=close.index)
    for t in ["SPY", "QQQ", "TQQQ", "XLK", "XLC", "XLY"]:
        df[t] = close[t] if t in close.columns else np.nan
    df["VIX"] = close["^VIX"] if "^VIX" in close.columns else close.get("^GSPC", np.nan)
    df = df.dropna()
    print(f"    Got {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def compute_regime_signals(df):
    """Add regime indicator columns."""
    print("[2/6] Computing regime signals …")
    df = df.copy()
    df["SMA200"] = df["SPY"].rolling(200).mean()
    df["SMA100"] = df["SPY"].rolling(100).mean()
    df["SMA50"]  = df["SPY"].rolling(50).mean()
    df["SPY_ret5"]  = df["SPY"].pct_change(5)
    df["VIX_chg5"]  = df["VIX"].pct_change(5)
    df["bull"] = (df["SPY"] > df["SMA200"]).astype(int)

    # VIX percentile (rolling 252-day)
    df["VIX_pctile"] = df["VIX"].rolling(252).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # Relative strength (20-day returns) for sector rotation
    for t in ["QQQ", "XLK", "XLC", "XLY"]:
        df[f"{t}_rs20"] = df[t].pct_change(20)

    df = df.dropna()
    return df


def _daily_returns(prices):
    return prices.pct_change().fillna(0)


# ── VARIANT ROTATION LOGIC ─────────────────────────────────────────────

def variant_A(df):
    """Baseline: exact reproduction of Strategy Rotation A.
    VIX > 25 → vix_fade regime (execution checks declining separately),
    bull → QQQ, else → contrarian."""
    print("    Variant A (Baseline Reproduction)")
    labels = []
    for i in range(len(df)):
        if df["VIX"].iloc[i] > 25:
            labels.append("vix_fade")
        elif df["bull"].iloc[i]:
            labels.append("earnings_momentum")
        else:
            labels.append("contrarian")
    return pd.Series(labels, index=df.index)


def variant_B(df):
    """Leveraged bull: TQQQ at 33% position in bull regime.
    Same switching logic, but uses leveraged ETF with reduced allocation."""
    print("    Variant B (Leveraged Bull — TQQQ 33%)")
    labels = []
    for i in range(len(df)):
        if df["VIX"].iloc[i] > 25:
            labels.append("vix_fade")
        elif df["bull"].iloc[i]:
            labels.append("leveraged_bull")
        else:
            labels.append("contrarian")
    return pd.Series(labels, index=df.index)


def variant_C(df):
    """Multi-timeframe trend: 50/100/200 SMA consensus.
    All three above = full bull (QQQ 100%)
    Above 200 but below 50 or 100 = moderate (QQQ 50%)
    Below 200 = bear (contrarian)."""
    print("    Variant C (Multi-Timeframe Trend)")
    labels = []
    for i in range(len(df)):
        if df["VIX"].iloc[i] > 25:
            labels.append("vix_fade")
        elif (df["SPY"].iloc[i] > df["SMA200"].iloc[i] and
              df["SPY"].iloc[i] > df["SMA100"].iloc[i] and
              df["SPY"].iloc[i] > df["SMA50"].iloc[i]):
            labels.append("earnings_momentum")  # full bull
        elif df["SPY"].iloc[i] > df["SMA200"].iloc[i]:
            labels.append("moderate_bull")  # above 200 but below 50 or 100
        else:
            labels.append("contrarian")
    return pd.Series(labels, index=df.index)


def variant_D(df):
    """VIX-scaled sizing in bull mode.
    VIX < 15 → QQQ 100%. 15-20 → 75%. 20-25 → 50%. >25 → VIX fade."""
    print("    Variant D (VIX-Scaled Sizing)")
    labels = []
    for i in range(len(df)):
        vix = df["VIX"].iloc[i]
        if vix > 25:
            labels.append("vix_fade")
        elif df["bull"].iloc[i]:
            if vix < 15:
                labels.append("bull_100")
            elif vix < 20:
                labels.append("bull_75")
            elif vix <= 25:
                labels.append("bull_50")
            else:
                labels.append("vix_fade")
        else:
            labels.append("contrarian")
    return pd.Series(labels, index=df.index)


def variant_E(df):
    """Sector rotation within bull: choose strongest of QQQ/XLK/XLC/XLY
    by 20-day relative strength. Updates daily."""
    print("    Variant E (Sector Rotation)")
    labels = []
    for i in range(len(df)):
        if df["VIX"].iloc[i] > 25:
            labels.append("vix_fade")
        elif df["bull"].iloc[i]:
            # Find strongest sector
            strengths = {}
            for t in ["QQQ", "XLK", "XLC", "XLY"]:
                strengths[t] = df[f"{t}_rs20"].iloc[i]
            best = max(strengths, key=strengths.get)
            labels.append(f"sector_{best}")
        else:
            labels.append("contrarian")
    return pd.Series(labels, index=df.index)


def variant_F(df):
    """Dynamic contrarian: sliding VIX-based dip threshold.
    Buy SPY when weekly return < -1.5 × VIX/100.
    Higher VIX = require deeper dip. Hold 8d in high-vol bears."""
    print("    Variant F (Dynamic Contrarian)")
    labels = []
    for i in range(len(df)):
        if df["VIX"].iloc[i] > 25:
            labels.append("vix_fade")
        elif df["bull"].iloc[i]:
            labels.append("earnings_momentum")
        else:
            labels.append("dynamic_contrarian")
    return pd.Series(labels, index=df.index)


# ── EXECUTE VARIANT ─────────────────────────────────────────────────────

def execute_rotation(df, rotation_labels):
    """Given strategy labels per day, produce equity curve with slippage."""
    spy_ret = _daily_returns(df["SPY"])
    qqq_ret = _daily_returns(df["QQQ"])
    tqqq_ret = _daily_returns(df["TQQQ"])

    # Sector returns
    sector_rets = {}
    for t in ["QQQ", "XLK", "XLC", "XLY"]:
        sector_rets[t] = _daily_returns(df[t])

    daily_ret = pd.Series(0.0, index=df.index)
    n_trades = 0
    prev_label = None
    hold_remaining = 0

    for i in range(len(df)):
        label = rotation_labels.iloc[i]

        # Count regime switches as trades
        if label != prev_label:
            n_trades += 1
            # Apply slippage on trade
            if prev_label is not None:
                daily_ret.iloc[i] -= SLIPPAGE
            hold_remaining = 0

        prev_label_save = prev_label
        prev_label = label

        if label == "earnings_momentum":
            daily_ret.iloc[i] += qqq_ret.iloc[i]

        elif label == "leveraged_bull":
            # TQQQ at 33% position size
            daily_ret.iloc[i] += tqqq_ret.iloc[i] * 0.33

        elif label == "moderate_bull":
            # QQQ at 50% position
            daily_ret.iloc[i] += qqq_ret.iloc[i] * 0.50

        elif label.startswith("bull_"):
            # VIX-scaled sizing
            pct = int(label.split("_")[1]) / 100.0
            daily_ret.iloc[i] += qqq_ret.iloc[i] * pct

        elif label.startswith("sector_"):
            ticker = label.replace("sector_", "")
            daily_ret.iloc[i] += sector_rets[ticker].iloc[i]

        elif label == "contrarian":
            # Original: buy SPY on -3% weekly drop, hold 5 days
            if hold_remaining > 0:
                daily_ret.iloc[i] += spy_ret.iloc[i]
                hold_remaining -= 1
            elif df["SPY_ret5"].iloc[i] <= -0.03:
                daily_ret.iloc[i] += spy_ret.iloc[i]
                hold_remaining = 4

        elif label == "dynamic_contrarian":
            # Sliding threshold: weekly return < -1.5 × VIX/100
            vix = df["VIX"].iloc[i]
            threshold = -1.5 * vix / 100.0
            hold_days = 8 if vix > 25 else 5
            if hold_remaining > 0:
                daily_ret.iloc[i] += spy_ret.iloc[i]
                hold_remaining -= 1
            elif df["SPY_ret5"].iloc[i] <= threshold:
                daily_ret.iloc[i] += spy_ret.iloc[i]
                hold_remaining = hold_days - 1

        elif label == "vix_fade":
            # Buy SPY only when VIX > 25 AND declining (original logic)
            if df["VIX"].iloc[i] > 25 and df["VIX_chg5"].iloc[i] < 0:
                daily_ret.iloc[i] += spy_ret.iloc[i]

        elif label == "cash":
            pass

    equity = STARTING_CAPITAL * (1 + daily_ret).cumprod()
    return daily_ret, equity, n_trades


# ── METRICS ─────────────────────────────────────────────────────────────

def compute_metrics(daily_ret, equity, n_trades, df, rotation_labels):
    trading_days = daily_ret[daily_ret != 0]

    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (252 / len(daily_ret)) - 1
    vol = daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 1e-9
    sharpe = ann_ret / vol if vol > 0 else 0

    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252) if (daily_ret < 0).sum() > 0 else 1e-9
    sortino = ann_ret / downside

    wins = (trading_days > 0).sum()
    losses = (trading_days < 0).sum()
    wr = wins / (wins + losses) if (wins + losses) > 0 else 0
    avg_win = trading_days[trading_days > 0].mean() if wins > 0 else 0
    avg_loss = abs(trading_days[trading_days < 0].mean()) if losses > 0 else 1e-9
    pf = (avg_win * wins) / (avg_loss * losses) if (avg_loss * losses) > 0 else 999

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    mdd = dd.min()

    # Regime-stratified Sharpe
    bull_mask = df["bull"] == 1
    bear_mask = df["bull"] == 0

    def _sharpe_subset(rets):
        if len(rets) < 5 or rets.std() == 0:
            return 0.0
        return (rets.mean() / rets.std()) * np.sqrt(252)

    sharpe_bull = _sharpe_subset(daily_ret[bull_mask])
    sharpe_bear = _sharpe_subset(daily_ret[bear_mask])
    regime_gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 0.01)

    return {
        "total_return": round(float(total_ret), 4),
        "annual_return": round(float(ann_ret), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(pf), 3),
        "win_rate": round(float(wr), 3),
        "max_drawdown": round(float(mdd), 4),
        "n_trades": int(n_trades),
        "n_active_days": int((trading_days != 0).sum()) if len(trading_days) > 0 else 0,
        "sharpe_bull": round(float(sharpe_bull), 3),
        "sharpe_bear": round(float(sharpe_bear), 3),
        "regime_gap": round(float(regime_gap), 3),
        "final_equity": round(float(equity.iloc[-1]), 2),
    }


# ── PERMUTATION TEST ───────────────────────────────────────────────────

def permutation_test(df, rotation_labels, daily_ret_actual, observed_sharpe, n_iter=PERM_ITERS):
    """Shuffle regime assignments (bull/bear label), recompute strategy labels and Sharpe.
    Block-shuffle by week to preserve autocorrelation."""
    spy_ret = _daily_returns(df["SPY"]).values
    qqq_ret = _daily_returns(df["QQQ"]).values
    n_days = len(df)

    # Build week boundaries
    dates = df.index
    week_starts = [0]
    for i in range(1, n_days):
        if dates[i].isocalendar()[1] != dates[i-1].isocalendar()[1] or \
           dates[i].year != dates[i-1].year:
            week_starts.append(i)
    week_starts.append(n_days)
    n_weeks = len(week_starts) - 1

    # Get unique labels and map to ints
    unique_labels = rotation_labels.unique()
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    orig_label_arr = np.array([label_to_int[l] for l in rotation_labels])
    unique_ints = np.array([label_to_int[l] for l in unique_labels])

    # Pre-compute masks for execution
    vix_vals = df["VIX"].values
    vix_chg5 = df["VIX_chg5"].values
    spy_ret5 = df["SPY_ret5"].values
    bull_arr = df["bull"].values

    rng = np.random.RandomState(7)
    count_ge = 0

    for _ in range(n_iter):
        # Block-shuffle: assign random label per week
        shuffled = orig_label_arr.copy()
        rand_labels = rng.choice(unique_ints, size=n_weeks)
        for w in range(n_weeks):
            shuffled[week_starts[w]:week_starts[w+1]] = rand_labels[w]

        # Simplified execution: earnings_momentum → QQQ, contrarian → SPY on dip,
        # vix_fade → SPY, anything with "bull" → QQQ scaled, sector → QQQ proxy
        perm_daily = np.zeros(n_days)
        hold_rem = 0
        for i in range(n_days):
            lab_int = shuffled[i]
            lab_str = unique_labels[lab_int] if lab_int < len(unique_labels) else "cash"
            if lab_str == "earnings_momentum" or lab_str.startswith("bull_") or lab_str.startswith("sector_"):
                perm_daily[i] = qqq_ret[i]
            elif lab_str == "leveraged_bull":
                perm_daily[i] = qqq_ret[i]  # simplified
            elif lab_str == "moderate_bull":
                perm_daily[i] = qqq_ret[i] * 0.5
            elif lab_str in ("contrarian", "dynamic_contrarian"):
                if hold_rem > 0:
                    perm_daily[i] = spy_ret[i]
                    hold_rem -= 1
                elif spy_ret5[i] <= -0.03:
                    perm_daily[i] = spy_ret[i]
                    hold_rem = 4
            elif lab_str == "vix_fade":
                perm_daily[i] = spy_ret[i]

        mu = perm_daily.mean()
        std = perm_daily.std()
        if std < 1e-12:
            perm_sharpe = 0.0
        else:
            perm_sharpe = (mu / std) * np.sqrt(252)

        if perm_sharpe >= observed_sharpe:
            count_ge += 1

    return round(count_ge / n_iter, 4)


# ── 5-GATE CHECK ───────────────────────────────────────────────────────

def five_gate_check(m, perm_p):
    gates = {
        "sharpe_gt_0.5": m["sharpe"] > GATE_SHARPE,
        "perm_p_lt_0.05": perm_p < GATE_PERM_P,
        "regime_gap_lt_0.5": m["regime_gap"] < GATE_REGIME,
        "mdd_gt_neg50pct": m["max_drawdown"] > GATE_MDD,
        "trades_gte_20": m["n_trades"] >= GATE_TRADES,
    }
    return gates, all(gates.values())


# ── BENCHMARKS ──────────────────────────────────────────────────────────

def compute_benchmarks(df_oot):
    benchmarks = {}
    for ticker in ["SPY", "QQQ"]:
        rets = _daily_returns(df_oot[ticker])
        eq = STARTING_CAPITAL * (1 + rets).cumprod()
        total = eq.iloc[-1] / eq.iloc[0] - 1
        ann = (1 + total) ** (252 / len(rets)) - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        peak = eq.cummax()
        mdd = ((eq - peak) / peak).min()
        benchmarks[f"{ticker}_buyhold"] = {
            "sharpe": round(float(sharpe), 3),
            "total_return": round(float(total), 4),
            "max_drawdown": round(float(mdd), 4),
            "final_equity": round(float(eq.iloc[-1]), 2),
        }
    return benchmarks


# ── MAIN ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STRATEGY ROTATION v2 BACKTEST — ENHANCED VARIANTS")
    print(f"OOT: {OOT_START} → {OOT_END}  |  Capital: ${STARTING_CAPITAL}")
    print(f"Slippage: {SLIPPAGE:.2%} per trade")
    print("=" * 70)

    df = fetch_data()
    df = compute_regime_signals(df)

    # Trim to OOT period
    df_oot = df.loc[OOT_START:OOT_END].copy()
    print(f"    OOT period: {len(df_oot)} days ({df_oot.index[0].date()} → {df_oot.index[-1].date()})")

    # Benchmarks
    print("[3/6] Computing benchmarks …")
    benchmarks = compute_benchmarks(df_oot)
    for name, bm in benchmarks.items():
        print(f"    {name}: Sharpe={bm['sharpe']}, Return={bm['total_return']:.1%}, "
              f"MDD={bm['max_drawdown']:.1%}, Final=${bm['final_equity']}")

    # Run all variants
    print("[4/6] Running strategy rotation v2 variants …")
    variant_funcs = {
        "A_baseline": variant_A,
        "B_leveraged_bull": variant_B,
        "C_multi_timeframe": variant_C,
        "D_vix_scaled": variant_D,
        "E_sector_rotation": variant_E,
        "F_dynamic_contrarian": variant_F,
    }

    results = {"benchmarks": benchmarks, "variants": {}, "meta": {
        "oot_start": OOT_START, "oot_end": OOT_END,
        "starting_capital": STARTING_CAPITAL,
        "slippage": SLIPPAGE,
        "run_timestamp": dt.datetime.now().isoformat(),
        "description": "Strategy Rotation v2 — enhanced variants of validated Strategy Rotation A",
    }}

    variant_data = {}  # store for permutation tests

    for vname, vfunc in variant_funcs.items():
        rotation_labels = vfunc(df_oot)
        daily_ret, equity, n_trades = execute_rotation(df_oot, rotation_labels)
        metrics = compute_metrics(daily_ret, equity, n_trades, df_oot, rotation_labels)

        # Allocation breakdown
        alloc = rotation_labels.value_counts(normalize=True).to_dict()
        metrics["allocation_pct"] = {k: round(float(v), 3) for k, v in alloc.items()}

        print(f"    {vname}: Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
              f"PF={metrics['profit_factor']}, WR={metrics['win_rate']:.1%}, "
              f"MDD={metrics['max_drawdown']:.1%}, trades={metrics['n_trades']}, "
              f"final=${metrics['final_equity']}")
        print(f"      Bull Sharpe={metrics['sharpe_bull']}, Bear Sharpe={metrics['sharpe_bear']}, "
              f"Regime Gap={metrics['regime_gap']}")

        results["variants"][vname] = metrics
        variant_data[vname] = (rotation_labels, daily_ret, equity, n_trades)

    # Variant A sanity check
    a_sharpe = results["variants"]["A_baseline"]["sharpe"]
    print(f"\n    *** Variant A Sharpe: {a_sharpe} (target: ~2.13) ***")
    if abs(a_sharpe - 2.13) > 0.3:
        print(f"    WARNING: Variant A Sharpe ({a_sharpe}) deviates from target 2.13 by "
              f"{abs(a_sharpe - 2.13):.3f}. May differ due to extended OOT window or slippage.")

    # Permutation tests
    print(f"\n[5/6] Permutation tests ({PERM_ITERS} iterations each) …")
    for vname in variant_funcs:
        rotation_labels, daily_ret, equity, n_trades = variant_data[vname]
        observed_sharpe = results["variants"][vname]["sharpe"]
        perm_p = permutation_test(df_oot, rotation_labels, daily_ret, observed_sharpe)
        results["variants"][vname]["perm_p"] = perm_p

        gates, passed = five_gate_check(results["variants"][vname], perm_p)
        results["variants"][vname]["five_gate"] = {k: bool(v) for k, v in gates.items()}
        results["variants"][vname]["five_gate_pass"] = passed

        status = "PASS" if passed else "FAIL"
        print(f"    {vname}: perm_p={perm_p}, 5-gate={status}")
        failed = [k for k, v in gates.items() if not v]
        if failed:
            print(f"      Failed: {', '.join(failed)}")

    # Summary
    print(f"\n[6/6] Summary")
    print("=" * 70)

    # Print comparison table
    print(f"\n{'Variant':<25} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} "
          f"{'MDD':>8} {'Return':>8} {'Final$':>8} {'5-Gate':>7}")
    print("-" * 90)
    for vname, m in results["variants"].items():
        status = "PASS" if m["five_gate_pass"] else "FAIL"
        print(f"{vname:<25} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.2f} "
              f"{m['win_rate']:>5.1%} {m['max_drawdown']:>7.1%} {m['total_return']:>7.1%} "
              f"${m['final_equity']:>7.0f} {status:>7}")
    print("-" * 90)
    for bname, bm in benchmarks.items():
        print(f"{bname:<25} {bm['sharpe']:>7.3f} {'':>8} {'':>6} {'':>6} "
              f"{bm['max_drawdown']:>7.1%} {bm['total_return']:>7.1%} "
              f"${bm['final_equity']:>7.0f}")

    # Identify best
    passers = [v for v, d in results["variants"].items() if d["five_gate_pass"]]
    if passers:
        print(f"\nPASSED 5-gate: {', '.join(passers)}")
        best = max(passers, key=lambda v: results["variants"][v]["sharpe"])
        bm = results["variants"][best]
        print(f"Best variant: {best} (Sharpe={bm['sharpe']}, Sortino={bm['sortino']}, "
              f"regime_gap={bm['regime_gap']})")

        # Compare best to baseline
        baseline = results["variants"]["A_baseline"]
        delta_sharpe = bm["sharpe"] - baseline["sharpe"]
        print(f"vs Baseline A: Sharpe delta = {delta_sharpe:+.3f}")
    else:
        print("NO VARIANT passed all 5 gates.")
        best_sharpe = max(results["variants"].items(), key=lambda x: x[1]["sharpe"])
        print(f"Best by Sharpe: {best_sharpe[0]} (Sharpe={best_sharpe[1]['sharpe']})")

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
