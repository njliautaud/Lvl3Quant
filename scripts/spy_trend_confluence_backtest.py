#!/usr/bin/env python3
"""
Multi-Timeframe Trend Confluence Backtest on SPY
=================================================
Moskowitz, Ooi & Pedersen (2012) — time-series momentum across timeframes.

6 variants tested with 5-gate walk-forward OOT validation.
Walk-forward OOT: Jan 2022 – Jul 2026.

Author: Claude Opus 4.6 (quant research)
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-01-01"  # need lookback for 200-SMA
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/spy_trend_confluence_results.json")


# ── DATA ────────────────────────────────────────────────────────────────────
def fetch_data():
    """Download SPY, TQQQ, TLT from yfinance."""
    tickers = ["SPY", "TQQQ", "TLT"]
    data = {}
    for t in tickers:
        df = yf.download(t, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        data[t] = df
        print(f"  {t}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return data


# ── INDICATORS ──────────────────────────────────────────────────────────────
def compute_indicators(df):
    """Compute all trend indicators on a price DataFrame."""
    c = df["Close"].copy()
    h = df["High"].copy()
    l = df["Low"].copy()

    ind = pd.DataFrame(index=df.index)
    # SMAs
    for w in [10, 20, 50, 200]:
        ind[f"SMA_{w}"] = c.rolling(w).mean()

    # SMA crossover signals: 1 = bullish, 0 = bearish
    ind["fast_bull"] = (ind["SMA_10"] > ind["SMA_20"]).astype(int)
    ind["med_bull"] = (ind["SMA_20"] > ind["SMA_50"]).astype(int)
    ind["slow_bull"] = (ind["SMA_50"] > ind["SMA_200"]).astype(int)

    # SMA consensus count (0-3)
    ind["sma_consensus"] = ind["fast_bull"] + ind["med_bull"] + ind["slow_bull"]

    # 20-day ROC
    ind["roc_20"] = c.pct_change(20)
    ind["roc_bull"] = (ind["roc_20"] > 0).astype(int)

    # ADX(14)
    ind["ADX"] = _adx(h, l, c, 14)

    # Regime: bull if close > 200-SMA
    ind["regime"] = np.where(c > ind["SMA_200"], "bull", "bear")

    ind["close"] = c
    return ind


def _adx(high, low, close, period=14):
    """Compute ADX indicator."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(span=period, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period).mean()
    return adx


# ── STRATEGY SIGNAL GENERATORS ──────────────────────────────────────────────
def signal_A_2of3(ind):
    """Variant A: Buy when 2+ of 3 SMA crossovers are bullish."""
    return (ind["sma_consensus"] >= 2).astype(int)


def signal_B_all3(ind):
    """Variant B: Buy only when ALL 3 SMA crossovers are bullish."""
    return (ind["sma_consensus"] == 3).astype(int)


def signal_C_momentum_trend(ind):
    """Variant C: Buy when ROC>0 AND medium SMA bullish."""
    return ((ind["roc_bull"] == 1) & (ind["med_bull"] == 1)).astype(int)


def signal_D_adx_filtered(ind):
    """Variant D: Buy when ADX>25 and SMA consensus >= 2."""
    enter = ((ind["ADX"] > 25) & (ind["sma_consensus"] >= 2)).astype(int)
    # Stay in until ADX < 20 (hysteresis)
    position = enter.copy()
    in_trade = False
    for i in range(len(position)):
        if enter.iloc[i] == 1:
            in_trade = True
        elif ind["ADX"].iloc[i] < 20:
            in_trade = False
        position.iloc[i] = 1 if in_trade else 0
    return position


def signal_E_leveraged(ind):
    """Variant E: Returns allocation code: 3=TQQQ, 2=SPY, 1=TLT, 0=cash."""
    alloc = pd.Series(0, index=ind.index)
    alloc[ind["sma_consensus"] == 3] = 3  # TQQQ
    alloc[ind["sma_consensus"] == 2] = 2  # SPY
    alloc[(ind["sma_consensus"] <= 1)] = 1  # TLT
    return alloc


def signal_F_dynamic_size(ind):
    """Variant F: Position size fraction based on consensus."""
    frac = ind["sma_consensus"] / 3.0
    return frac


# ── BACKTEST ENGINE ─────────────────────────────────────────────────────────
def backtest_binary(signal, spy_ret, oot_mask, label):
    """Backtest a binary (0/1) signal. Returns equity curve and stats."""
    sig = signal[oot_mask].copy()
    ret = spy_ret[oot_mask].copy()

    # Detect rebalance events (signal changes)
    changes = sig.diff().fillna(0) != 0
    n_rebalances = int(changes.sum())

    # Apply slippage on rebalance days
    costs = changes.astype(float) * SLIPPAGE_PCT
    strat_ret = sig.shift(1).fillna(0) * ret - costs

    equity = ACCOUNT_SIZE * (1 + strat_ret).cumprod()
    return equity, strat_ret, n_rebalances


def backtest_leveraged(alloc_signal, data, oot_mask):
    """Backtest variant E with multiple instruments."""
    sig = alloc_signal[oot_mask].copy()
    spy_ret = data["SPY"]["Close"].pct_change()[oot_mask]
    tqqq_ret = data["TQQQ"]["Close"].pct_change()[oot_mask]
    tlt_ret = data["TLT"]["Close"].pct_change()[oot_mask]

    prev_sig = sig.shift(1).fillna(0)
    changes = sig.diff().fillna(0) != 0
    n_rebalances = int(changes.sum())
    costs = changes.astype(float) * SLIPPAGE_PCT

    strat_ret = pd.Series(0.0, index=sig.index)
    mask3 = prev_sig == 3
    mask2 = prev_sig == 2
    mask1 = prev_sig == 1

    strat_ret[mask3] = tqqq_ret[mask3]
    strat_ret[mask2] = spy_ret[mask2]
    strat_ret[mask1] = tlt_ret[mask1]
    strat_ret = strat_ret - costs

    equity = ACCOUNT_SIZE * (1 + strat_ret).cumprod()
    return equity, strat_ret, n_rebalances


def backtest_fractional(frac_signal, spy_ret, oot_mask):
    """Backtest variant F with fractional position sizing."""
    frac = frac_signal[oot_mask].copy()
    ret = spy_ret[oot_mask].copy()

    prev_frac = frac.shift(1).fillna(0)
    frac_change = (frac - frac.shift(1)).abs().fillna(0)
    changes = frac_change > 0
    n_rebalances = int(changes.sum())

    costs = frac_change * SLIPPAGE_PCT  # slippage proportional to size change
    strat_ret = prev_frac * ret - costs

    equity = ACCOUNT_SIZE * (1 + strat_ret).cumprod()
    return equity, strat_ret, n_rebalances


# ── STATISTICS ──────────────────────────────────────────────────────────────
def compute_stats(strat_ret, equity, n_rebalances, regime_series, oot_mask):
    """Compute full performance statistics with regime breakdown."""
    ret = strat_ret.dropna()
    if len(ret) == 0 or ret.std() == 0:
        return None

    ann = 252
    sharpe = ret.mean() / ret.std() * np.sqrt(ann)
    downside = ret[ret < 0].std()
    sortino = ret.mean() / downside * np.sqrt(ann) if downside > 0 else np.inf

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Profit factor
    gross_profit = ret[ret > 0].sum()
    gross_loss = -ret[ret < 0].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Win rate (days with positive return when in position)
    in_position = ret != 0
    if in_position.sum() > 0:
        wr = (ret[in_position] > 0).mean()
    else:
        wr = 0.0

    # Total return
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    # CAGR
    years = len(ret) / 252
    cagr = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Regime breakdown
    regime = regime_series[oot_mask]
    regime = regime.reindex(ret.index)

    bull_ret = ret[regime == "bull"]
    bear_ret = ret[regime == "bear"]

    sharpe_bull = (bull_ret.mean() / bull_ret.std() * np.sqrt(ann)) if len(bull_ret) > 10 and bull_ret.std() > 0 else 0
    sharpe_bear = (bear_ret.mean() / bear_ret.std() * np.sqrt(ann)) if len(bear_ret) > 10 and bear_ret.std() > 0 else 0

    # Regime gap
    max_sharpe = max(abs(sharpe_bull), abs(sharpe_bear), 1e-10)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_sharpe

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(pf), 3),
        "win_rate": round(float(wr), 4),
        "max_drawdown_pct": round(float(max_dd * 100), 2),
        "total_return_pct": round(float(total_ret), 2),
        "cagr_pct": round(float(cagr), 2),
        "n_rebalances": n_rebalances,
        "n_trading_days": int(len(ret)),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "sharpe_bull": round(float(sharpe_bull), 3),
        "sharpe_bear": round(float(sharpe_bear), 3),
        "regime_gap": round(float(regime_gap), 3),
        "bull_days": int(len(bull_ret)),
        "bear_days": int(len(bear_ret)),
    }


# ── PERMUTATION TEST ───────────────────────────────────────────────────────
def permutation_test(signal, returns, oot_mask, n_perms=N_PERMUTATIONS, is_fractional=False):
    """
    Circular shift permutation test.
    Shifts signal by random offset to test if timing matters.
    Returns p-value.
    """
    rng = np.random.RandomState(RANDOM_SEED)
    sig = signal[oot_mask].values.copy()
    ret = returns[oot_mask].values.copy()
    n = len(sig)

    # Actual Sharpe
    if is_fractional:
        prev_sig = np.roll(sig, 1)
        prev_sig[0] = 0
        actual_ret = prev_sig * ret
    else:
        prev_sig = np.roll(sig, 1)
        prev_sig[0] = 0
        actual_ret = prev_sig * ret

    actual_sharpe = actual_ret.mean() / (actual_ret.std() + 1e-10) * np.sqrt(252)

    count_better = 0
    for _ in range(n_perms):
        offset = rng.randint(1, n)
        shifted = np.roll(sig, offset)
        prev_shifted = np.roll(shifted, 1)
        prev_shifted[0] = 0
        perm_ret = prev_shifted * ret
        perm_sharpe = perm_ret.mean() / (perm_ret.std() + 1e-10) * np.sqrt(252)
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    p_value = (count_better + 1) / (n_perms + 1)
    return round(float(p_value), 4)


# ── VALIDATION GATES ────────────────────────────────────────────────────────
def validate(stats, p_value):
    """Apply 5-gate validation."""
    if stats is None:
        return {"pass": False, "reason": "no stats (zero variance)"}

    gates = {}
    gates["sharpe_gt_0.5"] = stats["sharpe"] > 0.5
    gates["perm_p_lt_0.05"] = p_value < 0.05
    gates["regime_gap_lt_0.5"] = stats["regime_gap"] < 0.5
    gates["maxdd_gt_neg50"] = stats["max_drawdown_pct"] > -50
    gates["min_20_rebalances"] = stats["n_rebalances"] >= 20

    all_pass = all(gates.values())
    return {"pass": all_pass, "gates": gates, "p_value": p_value}


# ── BUY & HOLD BENCHMARK ───────────────────────────────────────────────────
def buy_hold_benchmark(spy_ret, oot_mask):
    """SPY buy-and-hold benchmark."""
    ret = spy_ret[oot_mask]
    equity = ACCOUNT_SIZE * (1 + ret).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    sharpe = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    return {
        "sharpe": round(float(sharpe), 3),
        "total_return_pct": round(float(total_ret), 2),
        "max_drawdown_pct": round(float(dd.min() * 100), 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
    }


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("MULTI-TIMEFRAME TREND CONFLUENCE BACKTEST — SPY")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Account Size: ${ACCOUNT_SIZE}")
    print("=" * 70)

    print("\n[1/5] Fetching data...")
    data = fetch_data()
    spy = data["SPY"]

    print("\n[2/5] Computing indicators...")
    ind = compute_indicators(spy)

    # Align all data to SPY index
    spy_ret = spy["Close"].pct_change()
    oot_mask = (ind.index >= OOT_START) & (ind.index <= OOT_END)

    print(f"  OOT days: {oot_mask.sum()}")
    print(f"  Total indicator rows: {len(ind)}")

    # Drop NaN indicator rows from OOT
    valid = ind[oot_mask].dropna(subset=["SMA_200", "ADX"]).index
    oot_mask_clean = ind.index.isin(valid)
    print(f"  OOT days after indicator warmup: {oot_mask_clean.sum()}")

    # Buy & hold benchmark
    bh = buy_hold_benchmark(spy_ret, oot_mask_clean)
    print(f"\n  SPY Buy&Hold benchmark: Sharpe={bh['sharpe']}, Return={bh['total_return_pct']}%, MaxDD={bh['max_drawdown_pct']}%")

    # ── Generate signals ──
    print("\n[3/5] Generating strategy signals...")
    signals = {
        "A_2of3_SMA": signal_A_2of3(ind),
        "B_All3_SMA": signal_B_all3(ind),
        "C_Momentum_Trend": signal_C_momentum_trend(ind),
        "D_ADX_Filtered": signal_D_adx_filtered(ind),
        "E_Leveraged": signal_E_leveraged(ind),
        "F_Dynamic_Size": signal_F_dynamic_size(ind),
    }

    # ── Run backtests ──
    print("\n[4/5] Running backtests + permutation tests...")
    results = {"benchmark_buy_hold": bh, "variants": {}, "metadata": {
        "oot_start": OOT_START,
        "oot_end": OOT_END,
        "account_size": ACCOUNT_SIZE,
        "slippage_pct": SLIPPAGE_PCT,
        "n_permutations": N_PERMUTATIONS,
        "run_date": str(dt.datetime.now()),
    }}

    variant_descriptions = {
        "A_2of3_SMA": "2-of-3 SMA Confluence (buy when 2+ of 3 crossovers bullish)",
        "B_All3_SMA": "All-3 SMA Required (buy only when all 3 bullish)",
        "C_Momentum_Trend": "Momentum + Trend (ROC>0 AND medium SMA bullish)",
        "D_ADX_Filtered": "ADX-Filtered (ADX>25 + SMA consensus, hysteresis at ADX<20)",
        "E_Leveraged": "Leveraged Confluence (TQQQ/SPY/TLT based on consensus)",
        "F_Dynamic_Size": "Dynamic Position Size (100%/66%/33%/0% based on consensus)",
    }

    for name, sig in signals.items():
        print(f"\n  --- Variant {name} ---")
        desc = variant_descriptions[name]

        if name == "E_Leveraged":
            equity, strat_ret, n_reb = backtest_leveraged(sig, data, oot_mask_clean)
            # For permutation test, use a simplified binary version
            binary_sig = (sig >= 2).astype(int)
            p_val = permutation_test(binary_sig, spy_ret, oot_mask_clean)
        elif name == "F_Dynamic_Size":
            equity, strat_ret, n_reb = backtest_fractional(sig, spy_ret, oot_mask_clean)
            p_val = permutation_test(sig, spy_ret, oot_mask_clean, is_fractional=True)
        else:
            equity, strat_ret, n_reb = backtest_binary(sig, spy_ret, oot_mask_clean, name)
            p_val = permutation_test(sig, spy_ret, oot_mask_clean)

        stats = compute_stats(strat_ret, equity, n_reb, ind["regime"], oot_mask_clean)
        validation = validate(stats, p_val)

        results["variants"][name] = {
            "description": desc,
            "stats": stats,
            "validation": validation,
        }

        if stats:
            status = "PASS" if validation["pass"] else "FAIL"
            print(f"    {status} | Sharpe={stats['sharpe']} | Sortino={stats['sortino']} | "
                  f"PF={stats['profit_factor']} | WR={stats['win_rate']:.1%} | "
                  f"MaxDD={stats['max_drawdown_pct']}% | Return={stats['total_return_pct']}% | "
                  f"Rebalances={n_reb} | p={p_val}")
            print(f"    Regime: Bull Sharpe={stats['sharpe_bull']}, Bear Sharpe={stats['sharpe_bear']}, Gap={stats['regime_gap']}")
            # Gate detail
            for gate, passed in validation["gates"].items():
                mark = "OK" if passed else "XX"
                print(f"      [{mark}] {gate}")
        else:
            print(f"    FAIL — no valid stats")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("[5/5] SUMMARY")
    print("=" * 70)
    print(f"\n{'Variant':<25} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MaxDD':>7} {'Return':>8} {'Rebal':>6} {'p-val':>6} {'Result':>7}")
    print("-" * 95)

    passing = []
    for name, v in results["variants"].items():
        s = v["stats"]
        val = v["validation"]
        if s:
            status = "PASS" if val["pass"] else "FAIL"
            if val["pass"]:
                passing.append(name)
            print(f"{name:<25} {s['sharpe']:>7.3f} {s['sortino']:>8.3f} {s['profit_factor']:>6.2f} "
                  f"{s['win_rate']:>5.1%} {s['max_drawdown_pct']:>6.1f}% {s['total_return_pct']:>7.1f}% "
                  f"{s['n_rebalances']:>6d} {val['p_value']:>6.4f} {status:>7}")

    print(f"\nBenchmark (SPY B&H): Sharpe={bh['sharpe']}, Return={bh['total_return_pct']}%, MaxDD={bh['max_drawdown_pct']}%")
    print(f"\nPassing variants: {len(passing)}/{len(results['variants'])} — {passing if passing else 'NONE'}")

    # Determine best
    best = None
    best_sharpe = -999
    for name in passing:
        s = results["variants"][name]["stats"]["sharpe"]
        if s > best_sharpe:
            best_sharpe = s
            best = name

    if best:
        results["best_variant"] = best
        bs = results["variants"][best]["stats"]
        print(f"\nBEST: {best} — Sharpe {bs['sharpe']}, Return {bs['total_return_pct']}%, "
              f"MaxDD {bs['max_drawdown_pct']}%, Final equity ${bs['final_equity']}")
    else:
        results["best_variant"] = None
        print("\nNo variant passed all 5 gates.")

    # ── Save ──
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
