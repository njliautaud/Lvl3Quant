#!/usr/bin/env python3
"""
Gold Momentum with Macro Regime Filter — Backtest
==================================================
6 variants testing gold as a diversification engine (uncorrelated with equities).

Variants:
  A: GLD momentum — long when 50d return > 0, cash otherwise
  B: GLD trend — long when above 200-SMA, cash when below
  C: GLD dual filter — long only when BOTH 50d ret > 0 AND > 200-SMA
  D: Gold + VIX overlay — C plus VIX > 18
  E: Gold vs Bonds rotation — GLD if GLD 3mo ret > TLT 3mo ret, else TLT
  F: Gold momentum half-size — C but half position when SPY > 200-SMA

OOT: Jan 2022 – Jul 2026
Starting capital: $645
Cost: $0 commission, 0.02% slippage per trade

5-gate validation:
  1. Sharpe > 0.5
  2. Permutation test p < 0.05
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades
"""

import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")


def download_data():
    """Download GLD, SPY, TLT, ^VIX via yfinance."""
    import yfinance as yf

    tickers = {"GLD": "GLD", "SPY": "SPY", "TLT": "TLT", "VIX": "^VIX"}
    data = {}
    for name, ticker in tickers.items():
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start="2020-01-01", end="2026-07-29", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[name] = df["Close"].copy()
        data[f"{name}_open"] = df["Open"].copy()

    prices = pd.DataFrame({k: v for k, v in data.items()})
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.dropna()
    return prices


def apply_slippage(returns, signal, slippage_pct=0.0002):
    """Apply slippage on signal changes (trades)."""
    trades = signal.diff().abs()
    trades.iloc[0] = abs(signal.iloc[0])
    cost = trades * slippage_pct
    return returns - cost


def calc_metrics(returns, rf_annual=0.0):
    """Calculate Sharpe, Sortino, PF, WR, MaxDD, CAGR."""
    if len(returns) == 0 or returns.std() == 0:
        return {"sharpe": 0, "sortino": 0, "pf": 0, "wr": 0, "maxdd": 0, "cagr": 0, "total_ret": 0}

    ann = 252
    mean_r = returns.mean()
    sharpe = mean_r / returns.std() * np.sqrt(ann) if returns.std() > 0 else 0

    downside = returns[returns < 0].std()
    sortino = mean_r / downside * np.sqrt(ann) if downside > 0 else 0

    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    wr = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    cum = (1 + returns).cumprod()
    maxdd = (cum / cum.cummax() - 1).min()

    years = len(returns) / ann
    total_ret = cum.iloc[-1] - 1
    cagr = (cum.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 3),
        "maxdd": round(maxdd, 3),
        "cagr": round(cagr, 3),
        "total_ret": round(total_ret, 3),
    }


def count_trades(signal):
    """Count round-trip trades (signal changes)."""
    changes = signal.diff().abs()
    changes.iloc[0] = abs(signal.iloc[0])
    return int(changes.sum())


def regime_classify(spy_returns):
    """Classify each day as bull/bear/flat based on trailing 20d SPY return."""
    trailing = spy_returns.rolling(20).sum()
    regime = pd.Series("flat", index=spy_returns.index)
    regime[trailing > 0.02] = "bull"
    regime[trailing < -0.02] = "bear"
    return regime


def regime_gap(returns, regime):
    """Calculate regime gap: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|)."""
    bull_ret = returns[regime == "bull"]
    bear_ret = returns[regime == "bear"]

    def _sharpe(r):
        if len(r) < 10 or r.std() == 0:
            return 0
        return r.mean() / r.std() * np.sqrt(252)

    s_bull = _sharpe(bull_ret)
    s_bear = _sharpe(bear_ret)

    denom = max(abs(s_bull), abs(s_bear))
    if denom == 0:
        return 0.0
    return abs(s_bull - s_bear) / denom


def permutation_test(strategy_returns, benchmark_returns, n_perms=5000):
    """Permutation test: is strategy Sharpe significantly > benchmark Sharpe?"""
    obs_sharpe_diff = (
        strategy_returns.mean() / strategy_returns.std()
        - benchmark_returns.mean() / benchmark_returns.std()
    ) * np.sqrt(252)

    combined = np.concatenate([strategy_returns.values, benchmark_returns.values])
    n = len(strategy_returns)
    count = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        perm = rng.permutation(combined)
        s1, s2 = perm[:n], perm[n:]
        perm_diff = (s1.mean() / s1.std() - s2.mean() / s2.std()) * np.sqrt(252)
        if perm_diff >= obs_sharpe_diff:
            count += 1
    return count / n_perms


def correlation_with_spy(strategy_returns, spy_returns):
    """Daily return correlation with SPY."""
    aligned = pd.concat([strategy_returns, spy_returns], axis=1).dropna()
    if len(aligned) < 20:
        return 0
    return round(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]), 3)


def run_variant_a(prices, oot_start, oot_end):
    """A: GLD momentum — long when 50d return > 0."""
    gld = prices["GLD"]
    ret_50d = gld.pct_change(50)
    signal = (ret_50d > 0).astype(float)
    gld_ret = gld.pct_change()

    mask = (prices.index >= oot_start) & (prices.index <= oot_end)
    sig_oot = signal[mask].shift(1).fillna(0)
    ret_oot = gld_ret[mask]
    strat_ret = apply_slippage(sig_oot * ret_oot, sig_oot)
    return strat_ret, sig_oot


def run_variant_b(prices, oot_start, oot_end):
    """B: GLD trend — long when above 200-SMA."""
    gld = prices["GLD"]
    sma200 = gld.rolling(200).mean()
    signal = (gld > sma200).astype(float)
    gld_ret = gld.pct_change()

    mask = (prices.index >= oot_start) & (prices.index <= oot_end)
    sig_oot = signal[mask].shift(1).fillna(0)
    ret_oot = gld_ret[mask]
    strat_ret = apply_slippage(sig_oot * ret_oot, sig_oot)
    return strat_ret, sig_oot


def run_variant_c(prices, oot_start, oot_end):
    """C: GLD dual filter — 50d ret > 0 AND > 200-SMA."""
    gld = prices["GLD"]
    ret_50d = gld.pct_change(50)
    sma200 = gld.rolling(200).mean()
    signal = ((ret_50d > 0) & (gld > sma200)).astype(float)
    gld_ret = gld.pct_change()

    mask = (prices.index >= oot_start) & (prices.index <= oot_end)
    sig_oot = signal[mask].shift(1).fillna(0)
    ret_oot = gld_ret[mask]
    strat_ret = apply_slippage(sig_oot * ret_oot, sig_oot)
    return strat_ret, sig_oot


def run_variant_d(prices, oot_start, oot_end):
    """D: Gold + VIX overlay — C plus VIX > 18."""
    gld = prices["GLD"]
    ret_50d = gld.pct_change(50)
    sma200 = gld.rolling(200).mean()
    vix = prices["VIX"]
    signal = ((ret_50d > 0) & (gld > sma200) & (vix > 18)).astype(float)
    gld_ret = gld.pct_change()

    mask = (prices.index >= oot_start) & (prices.index <= oot_end)
    sig_oot = signal[mask].shift(1).fillna(0)
    ret_oot = gld_ret[mask]
    strat_ret = apply_slippage(sig_oot * ret_oot, sig_oot)
    return strat_ret, sig_oot


def run_variant_e(prices, oot_start, oot_end):
    """E: Gold vs Bonds rotation — GLD if 3mo ret > TLT 3mo ret, else TLT."""
    gld = prices["GLD"]
    tlt = prices["TLT"]
    gld_3m = gld.pct_change(63)
    tlt_3m = tlt.pct_change(63)

    # signal=1 → GLD, signal=0 → TLT
    in_gld = (gld_3m > tlt_3m).astype(float)
    gld_ret = gld.pct_change()
    tlt_ret = tlt.pct_change()

    mask = (prices.index >= oot_start) & (prices.index <= oot_end)
    sig_oot = in_gld[mask].shift(1).fillna(0)
    gld_r = gld_ret[mask]
    tlt_r = tlt_ret[mask]
    raw_ret = sig_oot * gld_r + (1 - sig_oot) * tlt_r
    strat_ret = apply_slippage(raw_ret, sig_oot)
    return strat_ret, sig_oot


def run_variant_f(prices, oot_start, oot_end):
    """F: Gold momentum half-size — C but half position when SPY > 200-SMA."""
    gld = prices["GLD"]
    spy = prices["SPY"]
    ret_50d = gld.pct_change(50)
    sma200_gld = gld.rolling(200).mean()
    sma200_spy = spy.rolling(200).mean()

    base_signal = ((ret_50d > 0) & (gld > sma200_gld)).astype(float)
    # Half size when SPY is above its 200-SMA (equity bull = less gold needed)
    spy_bull = (spy > sma200_spy).astype(float)
    signal = base_signal * (1 - 0.5 * spy_bull)  # 1.0 in bear, 0.5 in bull

    gld_ret = gld.pct_change()
    mask = (prices.index >= oot_start) & (prices.index <= oot_end)
    sig_oot = signal[mask].shift(1).fillna(0)
    ret_oot = gld_ret[mask]
    strat_ret = apply_slippage(sig_oot * ret_oot, sig_oot)
    return strat_ret, sig_oot


def main():
    oot_start = "2022-01-01"
    oot_end = "2026-07-29"
    starting_capital = 645

    print("=" * 70)
    print("GOLD MOMENTUM WITH MACRO REGIME FILTER — BACKTEST")
    print("=" * 70)

    prices = download_data()
    print(f"\nData range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Total rows: {len(prices)}")

    spy_ret = prices["SPY"].pct_change().dropna()
    regime = regime_classify(spy_ret)

    # Buy-and-hold benchmarks
    mask = (prices.index >= oot_start) & (prices.index <= oot_end)
    gld_bh_ret = prices["GLD"].pct_change()[mask]
    spy_bh_ret = prices["SPY"].pct_change()[mask]

    variants = {
        "A": ("GLD Momentum (50d ret > 0)", run_variant_a),
        "B": ("GLD Trend (> 200-SMA)", run_variant_b),
        "C": ("GLD Dual Filter (50d + 200-SMA)", run_variant_c),
        "D": ("Gold + VIX Overlay (VIX > 18)", run_variant_d),
        "E": ("Gold vs Bonds Rotation (GLD/TLT)", run_variant_e),
        "F": ("Gold Half-Size in Bull", run_variant_f),
    }

    results = {}
    print("\n" + "=" * 70)

    # Benchmarks
    gld_bh_metrics = calc_metrics(gld_bh_ret.dropna())
    spy_bh_metrics = calc_metrics(spy_bh_ret.dropna())
    gld_bh_corr = correlation_with_spy(gld_bh_ret.dropna(), spy_bh_ret.dropna())

    print(f"\n{'BENCHMARK':>35s} | Sharpe | Sortino |   PF  |  WR   | MaxDD  |  CAGR  | TotRet | Corr/SPY")
    print("-" * 110)
    print(
        f"{'GLD Buy&Hold':>35s} | {gld_bh_metrics['sharpe']:6.3f} | {gld_bh_metrics['sortino']:7.3f} | "
        f"{gld_bh_metrics['pf']:5.3f} | {gld_bh_metrics['wr']:5.3f} | {gld_bh_metrics['maxdd']:6.3f} | "
        f"{gld_bh_metrics['cagr']:6.3f} | {gld_bh_metrics['total_ret']:6.3f} | {gld_bh_corr:6.3f}"
    )
    print(
        f"{'SPY Buy&Hold':>35s} | {spy_bh_metrics['sharpe']:6.3f} | {spy_bh_metrics['sortino']:7.3f} | "
        f"{spy_bh_metrics['pf']:5.3f} | {spy_bh_metrics['wr']:5.3f} | {spy_bh_metrics['maxdd']:6.3f} | "
        f"{spy_bh_metrics['cagr']:6.3f} | {spy_bh_metrics['total_ret']:6.3f} |  1.000"
    )

    print(f"\n{'VARIANT':>35s} | Sharpe | Sortino |   PF  |  WR   | MaxDD  |  CAGR  | TotRet | Trades | Corr/SPY | RegGap")
    print("-" * 120)

    for key, (name, func) in variants.items():
        strat_ret, signal = func(prices, oot_start, oot_end)
        strat_ret = strat_ret.dropna()

        metrics = calc_metrics(strat_ret)
        n_trades = count_trades(signal)
        corr_spy = correlation_with_spy(strat_ret, spy_bh_ret.reindex(strat_ret.index))

        regime_oot = regime.reindex(strat_ret.index).fillna("flat")
        rgap = round(regime_gap(strat_ret, regime_oot), 3)

        # Permutation test vs GLD buy-and-hold
        gld_bh_aligned = gld_bh_ret.reindex(strat_ret.index).dropna()
        strat_aligned = strat_ret.reindex(gld_bh_aligned.index).dropna()
        if len(strat_aligned) > 50:
            p_val = permutation_test(strat_aligned, gld_bh_aligned)
        else:
            p_val = 1.0

        # Equity curve
        cum = (1 + strat_ret).cumprod()
        final_capital = round(starting_capital * cum.iloc[-1], 2)

        # Per-regime Sharpe
        bull_sharpe = calc_metrics(strat_ret[regime_oot == "bull"])["sharpe"]
        bear_sharpe = calc_metrics(strat_ret[regime_oot == "bear"])["sharpe"]
        flat_sharpe = calc_metrics(strat_ret[regime_oot == "flat"])["sharpe"]

        # 5-gate validation
        gates = {
            "sharpe_gt_05": metrics["sharpe"] > 0.5,
            "perm_p_lt_05": p_val < 0.05,
            "regime_gap_lt_05": rgap < 0.5,
            "maxdd_gt_neg50": metrics["maxdd"] > -0.50,
            "trades_gte_20": n_trades >= 20,
        }
        passed = sum(gates.values())

        print(
            f"{key}: {name:>29s} | {metrics['sharpe']:6.3f} | {metrics['sortino']:7.3f} | "
            f"{metrics['pf']:5.3f} | {metrics['wr']:5.3f} | {metrics['maxdd']:6.3f} | "
            f"{metrics['cagr']:6.3f} | {metrics['total_ret']:6.3f} | {n_trades:6d} | {corr_spy:8.3f} | {rgap:6.3f}"
        )

        results[key] = {
            "name": name,
            "metrics": metrics,
            "n_trades": n_trades,
            "correlation_with_spy": corr_spy,
            "regime_gap": rgap,
            "permutation_p_value": round(p_val, 4),
            "final_capital": final_capital,
            "per_regime_sharpe": {
                "bull": bull_sharpe,
                "bear": bear_sharpe,
                "flat": flat_sharpe,
            },
            "gates": gates,
            "gates_passed": f"{passed}/5",
            "all_gates_pass": passed == 5,
        }

    # Summary
    print("\n" + "=" * 70)
    print("5-GATE VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Variant':>10s} | Sharpe>0.5 | Perm p<0.05 | RegGap<0.5 | MaxDD>-50% | Trades>=20 | PASS")
    print("-" * 85)
    for key, r in results.items():
        g = r["gates"]
        icon = lambda x: "  YES " if x else "  NO  "
        all_pass = "*** PASS ***" if r["all_gates_pass"] else "    FAIL    "
        print(
            f"    {key}      |{icon(g['sharpe_gt_05'])}|{icon(g['perm_p_lt_05'])}|"
            f"{icon(g['regime_gap_lt_05'])}|{icon(g['maxdd_gt_neg50'])}|{icon(g['trades_gte_20'])}| {all_pass}"
        )

    # Correlation matrix
    print("\n" + "=" * 70)
    print("CORRELATION WITH SPY (diversification test)")
    print("=" * 70)
    for key, r in results.items():
        corr = r["correlation_with_spy"]
        assessment = "UNCORRELATED" if abs(corr) < 0.2 else ("LOW" if abs(corr) < 0.4 else "HIGH")
        print(f"  {key}: {r['name']:>35s} → corr = {corr:+.3f} ({assessment})")

    # Per-regime detail
    print("\n" + "=" * 70)
    print("PER-REGIME SHARPE (diversification thesis check)")
    print("=" * 70)
    print(f"{'Variant':>10s} | Bull Sharpe | Bear Sharpe | Flat Sharpe | Regime Gap | Assessment")
    print("-" * 90)
    for key, r in results.items():
        prs = r["per_regime_sharpe"]
        rgap = r["regime_gap"]
        if rgap < 0.3:
            assessment = "GREAT — works across regimes"
        elif rgap < 0.5:
            assessment = "OK — some regime dependency"
        else:
            assessment = "FAIL — regime-dependent"
        print(
            f"    {key}      | {prs['bull']:10.3f} | {prs['bear']:11.3f} | {prs['flat']:11.3f} | "
            f"{rgap:10.3f} | {assessment}"
        )

    # Final capital
    print("\n" + "=" * 70)
    print(f"FINAL CAPITAL (starting: ${starting_capital})")
    print("=" * 70)
    for key, r in results.items():
        print(f"  {key}: ${r['final_capital']:,.2f}  (total return: {r['metrics']['total_ret']*100:+.1f}%)")

    gld_bh_final = round(starting_capital * (1 + gld_bh_metrics["total_ret"]), 2)
    spy_bh_final = round(starting_capital * (1 + spy_bh_metrics["total_ret"]), 2)
    print(f"  GLD B&H: ${gld_bh_final:,.2f}  (total return: {gld_bh_metrics['total_ret']*100:+.1f}%)")
    print(f"  SPY B&H: ${spy_bh_final:,.2f}  (total return: {spy_bh_metrics['total_ret']*100:+.1f}%)")

    # Save results
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "oot_period": f"{oot_start} to {oot_end}",
        "starting_capital": starting_capital,
        "benchmarks": {
            "GLD_buy_hold": {**gld_bh_metrics, "final_capital": gld_bh_final, "corr_with_spy": gld_bh_corr},
            "SPY_buy_hold": {**spy_bh_metrics, "final_capital": spy_bh_final},
        },
        "variants": results,
    }

    out_path = "/home/jupiter/Lvl3Quant/data/gold_momentum_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
