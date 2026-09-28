#!/usr/bin/env python3
"""
Treasury / Fixed Income Momentum Backtest
==========================================
6 strategy variants, walk-forward OOT Jan 2022 – Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
Computes QQQ correlation for each variant.

Account: $645 Robinhood. $0 commission, 0.02% slippage.
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
ACCOUNT = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-01-01"  # extra lookback for indicators
PERM_ITERS = 1000
SEED = 42

TICKERS = ["TLT", "IEF", "SHY", "TIP", "TMF", "SPY", "QQQ", "GLD"]

# VIX needs special handling
VIX_TICKER = "^VIX"

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/treasury_momentum_results.json")


def fetch_data():
    """Download all required price data."""
    print("Fetching price data...")
    all_tickers = TICKERS + [VIX_TICKER]
    data = yf.download(all_tickers, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

    # Handle MultiIndex columns from yf.download
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    # Rename VIX column
    if "^VIX" in close.columns:
        close = close.rename(columns={"^VIX": "VIX"})

    close = close.ffill().dropna(how="all")
    print(f"  Data: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}, {len(close)} days")
    return close


def apply_slippage(price, direction="buy"):
    """Apply slippage to entry/exit price."""
    if direction == "buy":
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def compute_returns_from_signals(prices, signals, ticker="TLT"):
    """
    Given a price series and a signal series (1=long, 0=cash),
    compute daily returns accounting for slippage on transitions.
    Returns daily strategy returns series.
    """
    daily_ret = prices.pct_change().fillna(0)
    # Detect transitions for slippage
    signal_diff = signals.diff().fillna(0)
    # Entry: signal goes from 0 to 1 → buy slippage
    # Exit: signal goes from 1 to 0 → sell slippage
    slippage_cost = signal_diff.abs() * SLIPPAGE_PCT
    strat_ret = signals.shift(1).fillna(0) * daily_ret - slippage_cost
    return strat_ret


def count_trades(signals):
    """Count number of round-trip trades (entries)."""
    entries = ((signals == 1) & (signals.shift(1).fillna(0) == 0)).sum()
    return int(entries)


def compute_metrics(returns, name=""):
    """Compute Sharpe, Sortino, MaxDD, PF, WR, total return."""
    if len(returns) == 0 or returns.std() == 0:
        return {"sharpe": 0, "sortino": 0, "max_dd_pct": -100, "pf": 0, "wr": 0,
                "total_return_pct": 0, "annual_return_pct": 0, "n_days": 0}

    ann = 252
    mean_r = returns.mean()
    std_r = returns.std()
    sharpe = (mean_r / std_r) * np.sqrt(ann) if std_r > 0 else 0

    downside = returns[returns < 0].std()
    sortino = (mean_r / downside) * np.sqrt(ann) if downside > 0 else 0

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min() * 100

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else (99.0 if gains > 0 else 0)

    # Win rate (of trading days, not all days)
    trading_days = returns[returns != 0]
    wr = (trading_days > 0).mean() * 100 if len(trading_days) > 0 else 0

    total_ret = (cum.iloc[-1] - 1) * 100 if len(cum) > 0 else 0
    n_years = len(returns) / ann
    annual_ret = ((1 + total_ret / 100) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd, 2),
        "pf": round(pf, 3),
        "wr": round(wr, 1),
        "total_return_pct": round(total_ret, 2),
        "annual_return_pct": round(annual_ret, 2),
        "n_days": len(returns),
    }


def permutation_test(returns, signals, prices, n_iter=PERM_ITERS):
    """
    Shuffle entry dates to test if strategy Sharpe is significant.
    Returns p-value.
    """
    rng = np.random.RandomState(SEED)
    actual_sharpe = compute_metrics(returns)["sharpe"]
    daily_ret = prices.pct_change().fillna(0)

    count_better = 0
    for _ in range(n_iter):
        # Shuffle signal positions
        shuffled = signals.copy()
        shuffled_vals = shuffled.values.copy()
        rng.shuffle(shuffled_vals)
        shuffled = pd.Series(shuffled_vals, index=signals.index)
        perm_ret = shuffled.shift(1).fillna(0) * daily_ret
        perm_sharpe = compute_metrics(perm_ret)["sharpe"]
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return round((count_better + 1) / (n_iter + 1), 4)


def regime_analysis(returns, spy_close):
    """
    Split returns into bull/bear regimes based on SPY vs 200-SMA.
    Returns regime gap metric.
    """
    spy_sma200 = spy_close.rolling(200).mean()
    bull_mask = spy_close > spy_sma200
    # Align
    common = returns.index.intersection(bull_mask.index)
    bull_mask = bull_mask.reindex(common).fillna(False)
    ret_aligned = returns.reindex(common).fillna(0)

    bull_ret = ret_aligned[bull_mask]
    bear_ret = ret_aligned[~bull_mask]

    bull_metrics = compute_metrics(bull_ret)
    bear_metrics = compute_metrics(bear_ret)

    s_bull = bull_metrics["sharpe"]
    s_bear = bear_metrics["sharpe"]
    denom = max(abs(s_bull), abs(s_bear), 0.001)
    regime_gap = abs(s_bull - s_bear) / denom

    return {
        "bull_sharpe": s_bull,
        "bear_sharpe": s_bear,
        "regime_gap": round(regime_gap, 3),
        "bull_days": len(bull_ret),
        "bear_days": len(bear_ret),
    }


def qqq_correlation(strat_returns, qqq_returns):
    """Compute correlation of strategy daily returns with QQQ."""
    common = strat_returns.index.intersection(qqq_returns.index)
    s = strat_returns.reindex(common).fillna(0)
    q = qqq_returns.reindex(common).fillna(0)
    if len(common) < 20:
        return 0.0
    return round(s.corr(q), 4)


# ════════════════════════════════════════════════════════════════════════════
# STRATEGY VARIANTS
# ════════════════════════════════════════════════════════════════════════════

def strategy_a_tlt_trend(close, oot_mask):
    """A) TLT Trend Following: 20d SMA > 50d SMA → long TLT."""
    tlt = close["TLT"]
    sma20 = tlt.rolling(20).mean()
    sma50 = tlt.rolling(50).mean()
    signal = ((sma20 > sma50) & oot_mask).astype(float)
    ret = compute_returns_from_signals(tlt, signal)
    return ret[oot_mask], signal[oot_mask], tlt[oot_mask], "TLT"


def strategy_b_tlt_mean_rev(close, oot_mask):
    """B) TLT Mean Reversion: Buy when TLT drops >5% in 20d, hold 20 days."""
    tlt = close["TLT"]
    ret_20d = tlt.pct_change(20)
    daily_ret = tlt.pct_change().fillna(0)

    signal = pd.Series(0.0, index=tlt.index)
    hold_counter = 0

    for i in range(len(tlt)):
        if hold_counter > 0:
            signal.iloc[i] = 1.0
            hold_counter -= 1
        elif i >= 20 and ret_20d.iloc[i] < -0.05 and oot_mask.iloc[i]:
            signal.iloc[i] = 1.0
            hold_counter = 19  # hold for 20 total days

    ret = compute_returns_from_signals(tlt, signal)
    return ret[oot_mask], signal[oot_mask], tlt[oot_mask], "TLT"


def strategy_c_curve_steepener(close, oot_mask):
    """C) Curve Steepener: TLT vs IEF relative momentum → mean reversion."""
    tlt = close["TLT"]
    ief = close["IEF"]

    tlt_ret20 = tlt.pct_change(20)
    ief_ret20 = ief.pct_change(20)
    spread = tlt_ret20 - ief_ret20  # TLT outperformance over IEF

    # When TLT underperforms IEF by >2% → buy TLT (curve flattened, expect steepening)
    # When TLT outperforms IEF by >2% → buy IEF
    signal_tlt = ((spread < -0.02) & oot_mask).astype(float)
    signal_ief = ((spread > 0.02) & oot_mask).astype(float)

    ret_tlt = compute_returns_from_signals(tlt, signal_tlt)
    ret_ief = compute_returns_from_signals(ief, signal_ief)
    combined_ret = ret_tlt + ret_ief

    combined_signal = ((signal_tlt == 1) | (signal_ief == 1)).astype(float)
    # Use TLT as reference price for permutation test
    return combined_ret[oot_mask], combined_signal[oot_mask], tlt[oot_mask], "TLT/IEF"


def strategy_d_tips_momentum(close, oot_mask):
    """D) TIPS Momentum: Buy TIP when 20d mom positive AND GLD 20d mom positive."""
    tip = close["TIP"]
    gld = close["GLD"]

    tip_mom = tip.pct_change(20)
    gld_mom = gld.pct_change(20)

    signal = ((tip_mom > 0) & (gld_mom > 0) & oot_mask).astype(float)
    ret = compute_returns_from_signals(tip, signal)
    return ret[oot_mask], signal[oot_mask], tip[oot_mask], "TIP"


def strategy_e_bond_equity_rotation(close, oot_mask):
    """E) Bond-Equity Rotation: Weekly, hold TLT or SPY based on 20d Sharpe."""
    tlt = close["TLT"]
    spy = close["SPY"]

    tlt_ret = tlt.pct_change().fillna(0)
    spy_ret = spy.pct_change().fillna(0)

    # Rolling 20d Sharpe
    tlt_sharpe = tlt_ret.rolling(20).mean() / tlt_ret.rolling(20).std()
    spy_sharpe = spy_ret.rolling(20).mean() / spy_ret.rolling(20).std()

    # Weekly rebalance: hold whichever has higher rolling Sharpe
    # For simplicity, compute daily signal then only update on Fridays
    raw_signal_tlt = (tlt_sharpe > spy_sharpe).astype(float)

    # Only rebalance weekly (on Fridays)
    signal_tlt = pd.Series(np.nan, index=tlt.index)
    for i in range(len(tlt.index)):
        dt = tlt.index[i]
        if dt.weekday() == 4:  # Friday
            signal_tlt.iloc[i] = raw_signal_tlt.iloc[i]
    signal_tlt = signal_tlt.ffill().fillna(0)

    signal_spy = 1 - signal_tlt  # always in one or the other

    # Combined return
    ret = (signal_tlt.shift(1).fillna(0) * tlt_ret +
           signal_spy.shift(1).fillna(0) * spy_ret)

    # Slippage on transitions
    tlt_transitions = signal_tlt.diff().abs().fillna(0)
    ret -= tlt_transitions * SLIPPAGE_PCT

    # Signal for trade counting: any transition
    combined_signal = signal_tlt  # transitions in signal_tlt = transitions overall
    return ret[oot_mask], combined_signal[oot_mask], tlt[oot_mask], "TLT/SPY"


def strategy_f_tmf_leveraged(close, oot_mask):
    """F) TMF Leveraged: Buy TMF when VIX>25 AND TLT 5d momentum positive. Hold max 10 days."""
    tmf = close["TMF"]
    tlt = close["TLT"]
    vix = close["VIX"] if "VIX" in close.columns else None

    if vix is None:
        print("  WARNING: VIX data not available, skipping TMF strategy")
        empty = pd.Series(0.0, index=tmf[oot_mask].index)
        return empty, empty, tmf[oot_mask], "TMF"

    tlt_mom5 = tlt.pct_change(5)
    daily_ret = tmf.pct_change().fillna(0)

    signal = pd.Series(0.0, index=tmf.index)
    hold_counter = 0

    for i in range(len(tmf)):
        if hold_counter > 0:
            signal.iloc[i] = 1.0
            hold_counter -= 1
        elif (i >= 5 and oot_mask.iloc[i] and
              not pd.isna(vix.iloc[i]) and vix.iloc[i] > 25 and
              not pd.isna(tlt_mom5.iloc[i]) and tlt_mom5.iloc[i] > 0):
            signal.iloc[i] = 1.0
            hold_counter = 9  # hold for 10 total days

    ret = compute_returns_from_signals(tmf, signal)
    return ret[oot_mask], signal[oot_mask], tmf[oot_mask], "TMF"


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    close = fetch_data()

    # OOT mask
    oot_mask = pd.Series(False, index=close.index)
    oot_mask[(close.index >= OOT_START) & (close.index <= OOT_END)] = True

    qqq_daily_ret = close["QQQ"].pct_change().fillna(0)

    strategies = {
        "A_TLT_Trend": strategy_a_tlt_trend,
        "B_TLT_MeanRev": strategy_b_tlt_mean_rev,
        "C_Curve_Steepener": strategy_c_curve_steepener,
        "D_TIPS_Momentum": strategy_d_tips_momentum,
        "E_Bond_Equity_Rotation": strategy_e_bond_equity_rotation,
        "F_TMF_Leveraged": strategy_f_tmf_leveraged,
    }

    results = {}
    print(f"\n{'='*80}")
    print(f"TREASURY MOMENTUM BACKTEST — OOT: {OOT_START} to {OOT_END}")
    print(f"Account: ${ACCOUNT:.0f} | Slippage: {SLIPPAGE_PCT*100:.2f}%")
    print(f"{'='*80}\n")

    for name, func in strategies.items():
        print(f"─── {name} ───")
        try:
            ret, signal, ref_prices, instrument = func(close, oot_mask)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[name] = {"error": str(e)}
            continue

        n_trades = count_trades(signal)
        metrics = compute_metrics(ret, name)
        qqq_corr = qqq_correlation(ret, qqq_daily_ret)

        # Permutation test
        print(f"  Running {PERM_ITERS}-iteration permutation test...")
        perm_p = permutation_test(ret, signal, ref_prices)

        # Regime analysis
        regime = regime_analysis(ret, close["SPY"])

        # Final P&L
        final_equity = ACCOUNT * (1 + metrics["total_return_pct"] / 100)

        # 5-gate validation
        gates = {
            "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
            "perm_p_lt_0.05": perm_p < 0.05,
            "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
            "max_dd_gt_neg50": metrics["max_dd_pct"] > -50,
            "trades_gte_20": n_trades >= 20,
        }
        gates_passed = sum(gates.values())
        validated = gates_passed == 5

        result = {
            "instrument": instrument,
            "metrics": metrics,
            "n_trades": n_trades,
            "qqq_correlation": qqq_corr,
            "perm_p_value": perm_p,
            "regime": regime,
            "final_equity_usd": round(final_equity, 2),
            "gates": gates,
            "gates_passed": f"{gates_passed}/5",
            "VALIDATED": validated,
        }
        results[name] = result

        # Print summary
        m = metrics
        print(f"  Sharpe: {m['sharpe']:.3f} | Sortino: {m['sortino']:.3f} | PF: {m['pf']:.2f} | WR: {m['wr']:.1f}%")
        print(f"  Return: {m['total_return_pct']:.1f}% | MaxDD: {m['max_dd_pct']:.1f}% | Trades: {n_trades}")
        print(f"  QQQ Corr: {qqq_corr:.4f} | Perm p: {perm_p:.4f}")
        print(f"  Regime: Bull Sharpe={regime['bull_sharpe']:.3f}, Bear Sharpe={regime['bear_sharpe']:.3f}, Gap={regime['regime_gap']:.3f}")
        print(f"  Final equity: ${final_equity:.2f}")
        gates_str = " | ".join([f"{'PASS' if v else 'FAIL'}" for v in gates.values()])
        gate_names = list(gates.keys())
        print(f"  Gates: {' | '.join(gate_names)}")
        print(f"         {gates_str}")
        status = "VALIDATED" if validated else f"REJECTED ({gates_passed}/5)"
        print(f"  >>> {status}")
        print()

    # ── Summary table ──
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Strategy':<25} {'Sharpe':>7} {'Sortino':>8} {'Return%':>8} {'MaxDD%':>7} {'QQQ_r':>7} {'Perm_p':>7} {'Gates':>6} {'Status':>10}")
    print("-" * 95)
    for name, r in results.items():
        if "error" in r:
            print(f"{name:<25} ERROR: {r['error']}")
            continue
        m = r["metrics"]
        print(f"{name:<25} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['total_return_pct']:>8.1f} "
              f"{m['max_dd_pct']:>7.1f} {r['qqq_correlation']:>7.4f} {r['perm_p_value']:>7.4f} "
              f"{r['gates_passed']:>6} {'VALID' if r['VALIDATED'] else 'REJECT':>10}")
    print()

    # ── Correlation ranking ──
    print("QQQ CORRELATION RANKING (lowest = most diversifying):")
    corr_items = [(n, r["qqq_correlation"]) for n, r in results.items() if "error" not in r]
    corr_items.sort(key=lambda x: abs(x[1]))
    for rank, (name, corr) in enumerate(corr_items, 1):
        label = "EXCELLENT" if abs(corr) < 0.15 else ("GOOD" if abs(corr) < 0.30 else "MODERATE")
        print(f"  {rank}. {name}: {corr:.4f} ({label})")

    # ── Best uncorrelated + validated ──
    valid_uncorr = [(n, r) for n, r in results.items()
                    if "error" not in r and r["VALIDATED"] and abs(r["qqq_correlation"]) < 0.3]
    print(f"\nValidated + low QQQ correlation (<0.3): {len(valid_uncorr)} strategies")
    for n, r in sorted(valid_uncorr, key=lambda x: abs(x[1]["qqq_correlation"])):
        print(f"  {n}: Sharpe={r['metrics']['sharpe']:.3f}, QQQ_r={r['qqq_correlation']:.4f}, "
              f"Final=${r['final_equity_usd']:.2f}")

    # ── Save results ──
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "meta": {
                "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "oot_period": f"{OOT_START} to {OOT_END}",
                "account_usd": ACCOUNT,
                "slippage_pct": SLIPPAGE_PCT,
                "perm_iterations": PERM_ITERS,
                "purpose": "Find treasury strategies UNCORRELATED to QQQ Signal Agg A (Sharpe 2.38)",
            },
            "strategies": results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
