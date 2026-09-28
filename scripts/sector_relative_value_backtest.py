#!/usr/bin/env python3
"""
Sector Relative Value Backtest — 6 Variants Uncorrelated with Tech/QQQ
Walk-forward OOT: Jan 2022 – Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
START_DATE = "2021-06-01"  # extra lookback for indicators
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
PERM_ITERS = 1000
SEED = 42

TICKERS = [
    "XLE", "XLF", "XLV", "XLU", "XLI", "XLRE", "XLP", "XLB",
    "SPY", "QQQ", "TLT", "XLK"
]

np.random.seed(SEED)


def download_data():
    """Download all sector ETF data."""
    print("Downloading data...")
    data = yf.download(TICKERS, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    # Handle multi-level columns from yfinance
    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
    close = close.dropna()
    print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return close


def apply_slippage(price, direction):
    """Apply slippage to trade price. direction: 1=buy, -1=sell."""
    return price * (1 + direction * SLIPPAGE_PCT)


def compute_returns_from_signals(close_series, signals, account_size=ACCOUNT_SIZE):
    """
    Given a price series and a signal series (1=long, 0=cash, -1=short),
    compute daily returns with slippage applied on signal changes.
    Returns: daily_returns (Series), trade_count (int)
    """
    signals = signals.reindex(close_series.index).fillna(0)
    daily_ret = close_series.pct_change().fillna(0)

    # Detect signal changes for slippage
    signal_changes = signals.diff().fillna(0).abs()
    trade_count = int((signal_changes > 0).sum())

    # Strategy returns: position * daily_return - slippage on changes
    strat_ret = signals.shift(1).fillna(0) * daily_ret - signal_changes * SLIPPAGE_PCT

    return strat_ret, trade_count


def compute_portfolio_returns(weights_dict, close_df, signals_dict=None):
    """
    For multi-asset strategies. weights_dict: {ticker: weight}.
    If signals_dict provided, apply signals per ticker.
    Returns combined daily returns.
    """
    combined = pd.Series(0.0, index=close_df.index)
    total_trades = 0

    for ticker, weight in weights_dict.items():
        if ticker not in close_df.columns:
            continue
        daily_ret = close_df[ticker].pct_change().fillna(0)
        if signals_dict and ticker in signals_dict:
            sig = signals_dict[ticker].reindex(close_df.index).fillna(0)
            signal_changes = sig.diff().fillna(0).abs()
            total_trades += int((signal_changes > 0).sum())
            combined += weight * sig.shift(1).fillna(0) * daily_ret - weight * signal_changes * SLIPPAGE_PCT
        else:
            combined += weight * daily_ret

    return combined, total_trades


def sharpe_ratio(returns, ann=252):
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(ann))


def sortino_ratio(returns, ann=252):
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(ann))


def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def profit_factor(returns):
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float('inf') if gains > 0 else 0.0
    return float(gains / losses)


def win_rate(returns):
    nonzero = returns[returns != 0]
    if len(nonzero) == 0:
        return 0.0
    return float((nonzero > 0).sum() / len(nonzero))


def permutation_test(returns, signals, n_iter=PERM_ITERS):
    """Shuffle signal dates, compute Sharpe distribution."""
    actual_sharpe = sharpe_ratio(returns)
    count_better = 0
    sig_arr = signals.values.copy()
    ret_arr = returns.values.copy()

    for _ in range(n_iter):
        np.random.shuffle(sig_arr)
        # Rough: apply shuffled signals to returns
        perm_ret = pd.Series(sig_arr[:-1], index=returns.index[1:]).reindex(returns.index).fillna(0) * \
                   pd.Series(ret_arr, index=returns.index)
        perm_sharpe = sharpe_ratio(perm_ret)
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_iter


def regime_analysis(returns, spy_close):
    """Split returns by bull/bear regime (SPY vs 200-SMA)."""
    sma200 = spy_close.rolling(200).mean()
    bull_mask = spy_close > sma200
    bear_mask = spy_close <= sma200

    # Align
    bull_mask = bull_mask.reindex(returns.index).fillna(False)
    bear_mask = bear_mask.reindex(returns.index).fillna(False)

    bull_ret = returns[bull_mask]
    bear_ret = returns[bear_mask]

    sharpe_bull = sharpe_ratio(bull_ret) if len(bull_ret) > 20 else 0.0
    sharpe_bear = sharpe_ratio(bear_ret) if len(bear_ret) > 20 else 0.0

    denom = max(abs(sharpe_bull), abs(sharpe_bear), 0.001)
    regime_gap = abs(sharpe_bull - sharpe_bear) / denom

    return {
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_days": int(bull_mask.sum()),
        "bear_days": int(bear_mask.sum()),
    }


def correlation_with_qqq(strat_returns, qqq_returns):
    """Rolling and overall correlation with QQQ."""
    aligned = pd.DataFrame({"strat": strat_returns, "qqq": qqq_returns}).dropna()
    if len(aligned) < 20:
        return {"overall": 0.0, "rolling_60d_mean": 0.0}

    overall = float(aligned["strat"].corr(aligned["qqq"]))
    rolling = aligned["strat"].rolling(60).corr(aligned["qqq"]).dropna()

    return {
        "overall": round(overall, 3),
        "rolling_60d_mean": round(float(rolling.mean()), 3),
        "rolling_60d_max": round(float(rolling.max()), 3),
    }


def bear_2022_performance(returns):
    """Performance specifically in 2022 bear market."""
    mask = (returns.index >= "2022-01-01") & (returns.index <= "2022-12-31")
    bear_ret = returns[mask]
    if len(bear_ret) < 20:
        return {}
    cum_ret = float((1 + bear_ret).prod() - 1)
    return {
        "return_2022": round(cum_ret * 100, 2),
        "sharpe_2022": round(sharpe_ratio(bear_ret), 3),
        "maxdd_2022": round(max_drawdown(bear_ret) * 100, 2),
    }


def combined_portfolio_analysis(strat_returns, qqq_returns):
    """60% QQQ + 40% variant vs 100% QQQ."""
    combined = 0.6 * qqq_returns + 0.4 * strat_returns
    qqq_only = qqq_returns

    return {
        "combined_60_40": {
            "sharpe": round(sharpe_ratio(combined), 3),
            "sortino": round(sortino_ratio(combined), 3),
            "maxdd_pct": round(max_drawdown(combined) * 100, 2),
            "total_return_pct": round(float((1 + combined).prod() - 1) * 100, 2),
        },
        "qqq_100": {
            "sharpe": round(sharpe_ratio(qqq_only), 3),
            "sortino": round(sortino_ratio(qqq_only), 3),
            "maxdd_pct": round(max_drawdown(qqq_only) * 100, 2),
            "total_return_pct": round(float((1 + qqq_only).prod() - 1) * 100, 2),
        }
    }


def validate_5gate(metrics):
    """5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": metrics["perm_p"] < 0.05,
        "regime_gap_lt_0.5": metrics["regime"]["regime_gap"] < 0.5,
        "maxdd_gt_neg50": metrics["maxdd_pct"] > -50.0,
        "trades_gte_20": metrics["trade_count"] >= 20,
    }
    metrics["gates"] = gates
    metrics["gates_passed"] = sum(gates.values())
    metrics["all_gates_pass"] = all(gates.values())
    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY VARIANTS
# ═══════════════════════════════════════════════════════════════════════════

def variant_a_energy_utilities(close):
    """Energy-Utilities Spread: z-score of XLE/XLU ratio."""
    ratio = close["XLE"] / close["XLU"]
    zscore = (ratio - ratio.rolling(20).mean()) / ratio.rolling(20).std()

    signals = pd.Series(0, index=close.index)
    signals[zscore > 1.5] = 1   # Buy XLU (energy overextended)
    signals[zscore < -1.5] = -1  # Buy XLE (utilities overextended)

    # When signal=1, we're long XLU; when signal=-1, we're long XLE
    # Implement as: returns from XLU when sig=1, returns from XLE when sig=-1
    xlu_ret = close["XLU"].pct_change().fillna(0)
    xle_ret = close["XLE"].pct_change().fillna(0)

    sig_prev = signals.shift(1).fillna(0)
    strat_ret = pd.Series(0.0, index=close.index)
    strat_ret[sig_prev == 1] = xlu_ret[sig_prev == 1]
    strat_ret[sig_prev == -1] = xle_ret[sig_prev == -1]

    # Slippage on signal changes
    signal_changes = signals.diff().fillna(0).abs()
    strat_ret -= signal_changes * SLIPPAGE_PCT
    trade_count = int((signal_changes > 0).sum())

    return strat_ret, signals, trade_count


def variant_b_financials_rate(close):
    """Financials Rate Play: TLT movement drives XLF vs XLU."""
    tlt_20d_ret = close["TLT"].pct_change(20).fillna(0)

    signals = pd.Series(0, index=close.index)
    signals[tlt_20d_ret < -0.03] = 1   # Rates rising -> buy XLF
    signals[tlt_20d_ret > 0.03] = -1   # Rates falling -> buy XLU

    xlf_ret = close["XLF"].pct_change().fillna(0)
    xlu_ret = close["XLU"].pct_change().fillna(0)

    sig_prev = signals.shift(1).fillna(0)
    strat_ret = pd.Series(0.0, index=close.index)
    strat_ret[sig_prev == 1] = xlf_ret[sig_prev == 1]
    strat_ret[sig_prev == -1] = xlu_ret[sig_prev == -1]

    signal_changes = signals.diff().fillna(0).abs()
    strat_ret -= signal_changes * SLIPPAGE_PCT
    trade_count = int((signal_changes > 0).sum())

    return strat_ret, signals, trade_count


def variant_c_healthcare_momentum(close):
    """Healthcare Momentum: Buy XLV when its momentum > 0 AND SPY momentum < 0."""
    xlv_mom = close["XLV"].pct_change(20).fillna(0)
    spy_mom = close["SPY"].pct_change(20).fillna(0)

    signals = pd.Series(0, index=close.index)
    signals[(xlv_mom > 0) & (spy_mom < 0)] = 1  # Defensive outperformance

    xlv_ret = close["XLV"].pct_change().fillna(0)

    sig_prev = signals.shift(1).fillna(0)
    strat_ret = sig_prev * xlv_ret

    signal_changes = signals.diff().fillna(0).abs()
    strat_ret -= signal_changes * SLIPPAGE_PCT
    trade_count = int((signal_changes > 0).sum())

    return strat_ret, signals, trade_count


def variant_d_realestate_rate(close):
    """Real Estate Rate Sensitivity: Buy XLRE when TLT has positive 50d momentum."""
    tlt_mom = close["TLT"].pct_change(50).fillna(0)

    signals = pd.Series(0, index=close.index)
    signals[tlt_mom > 0] = 1  # Rates falling -> buy XLRE

    xlre_ret = close["XLRE"].pct_change().fillna(0)

    sig_prev = signals.shift(1).fillna(0)
    strat_ret = sig_prev * xlre_ret

    signal_changes = signals.diff().fillna(0).abs()
    strat_ret -= signal_changes * SLIPPAGE_PCT
    trade_count = int((signal_changes > 0).sum())

    return strat_ret, signals, trade_count


def variant_e_antitech_barbell(close):
    """Anti-Tech Barbell: Equal weight XLE + XLV + XLF, monthly rebalance."""
    components = ["XLE", "XLV", "XLF"]
    weight = 1.0 / len(components)

    # Monthly rebalance signal (1 on first trading day of month)
    months = close.index.to_period("M")
    rebal_mask = months != months.shift(1)  # This is a boolean for rebalance days

    # Always long, equal weight - rebalance monthly
    strat_ret = pd.Series(0.0, index=close.index)
    for comp in components:
        strat_ret += weight * close[comp].pct_change().fillna(0)

    # Slippage only on rebalance days (all 3 positions)
    strat_ret[rebal_mask] -= 3 * SLIPPAGE_PCT * weight

    # Signals are always 1 (always long)
    signals = pd.Series(1, index=close.index)
    trade_count = int(rebal_mask.sum()) * 3  # 3 trades per rebalance

    return strat_ret, signals, trade_count


def variant_f_sector_momentum_extech(close):
    """Sector Momentum ex-Tech: Buy top-2 sectors on 1-month momentum, hold 1 month."""
    sectors = ["XLE", "XLF", "XLV", "XLU", "XLI", "XLRE", "XLP", "XLB"]

    # Monthly rebalance
    months = close.index.to_period("M")
    rebal_mask = months != months.shift(1)
    rebal_dates = close.index[rebal_mask]

    # Build signals per sector
    signals = {s: pd.Series(0.0, index=close.index) for s in sectors}

    for i, rd in enumerate(rebal_dates):
        # Look back 21 trading days for momentum
        idx = close.index.get_loc(rd)
        if idx < 21:
            continue

        # Compute 1-month momentum for each sector
        mom = {}
        for s in sectors:
            mom[s] = float(close[s].iloc[idx] / close[s].iloc[idx - 21] - 1)

        # Rank and pick top 2
        ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
        top2 = [r[0] for r in ranked[:2]]

        # Set signals until next rebalance
        end_idx = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else close.index[-1]
        end_loc = close.index.get_loc(end_idx)

        for s in sectors:
            signals[s].iloc[idx:end_loc] = 0.5 if s in top2 else 0.0

    # Compute returns
    strat_ret = pd.Series(0.0, index=close.index)
    for s in sectors:
        daily_ret = close[s].pct_change().fillna(0)
        strat_ret += signals[s].shift(1).fillna(0) * daily_ret

    # Slippage on rebalance (assume 4 trades avg: sell 2, buy 2)
    for rd in rebal_dates:
        if rd in strat_ret.index:
            strat_ret.loc[rd] -= 4 * SLIPPAGE_PCT * 0.5

    # Aggregate signal for permutation test
    agg_signal = pd.Series(0.0, index=close.index)
    for s in sectors:
        agg_signal += signals[s]

    trade_count = int(len(rebal_dates)) * 4  # ~4 trades per rebalance

    return strat_ret, agg_signal, trade_count


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_variant(name, description, close, variant_func):
    """Run a single variant through full analysis pipeline."""
    print(f"\n{'='*60}")
    print(f"  {name}: {description}")
    print(f"{'='*60}")

    # Get OOT period
    oot_mask = (close.index >= OOT_START) & (close.index <= OOT_END)
    close_oot = close[oot_mask]

    # Run strategy on full data, then slice OOT
    strat_ret_full, signals_full, trade_count_full = variant_func(close)

    strat_ret = strat_ret_full[oot_mask]
    signals = signals_full[oot_mask]

    # Recount trades in OOT only
    signal_changes_oot = signals.diff().fillna(0).abs()
    trade_count = int((signal_changes_oot > 0).sum())
    if trade_count == 0:
        # For always-on strategies like E, count rebalances
        trade_count = trade_count_full

    # Core metrics
    sr = sharpe_ratio(strat_ret)
    sort = sortino_ratio(strat_ret)
    mdd = max_drawdown(strat_ret)
    pf = profit_factor(strat_ret)
    wr = win_rate(strat_ret)
    total_ret = float((1 + strat_ret).prod() - 1)

    print(f"  Sharpe:  {sr:.3f}")
    print(f"  Sortino: {sort:.3f}")
    print(f"  MaxDD:   {mdd*100:.1f}%")
    print(f"  PF:      {pf:.2f}")
    print(f"  WR:      {wr*100:.1f}%")
    print(f"  Return:  {total_ret*100:.1f}%")
    print(f"  Trades:  {trade_count}")

    # Permutation test
    print("  Running permutation test...")
    perm_p = permutation_test(strat_ret, signals)
    print(f"  Perm p:  {perm_p:.4f}")

    # QQQ correlation
    qqq_ret = close_oot["QQQ"].pct_change().fillna(0)
    corr = correlation_with_qqq(strat_ret, qqq_ret)
    print(f"  QQQ corr: {corr['overall']:.3f}")

    # Regime analysis
    regime = regime_analysis(strat_ret, close_oot["SPY"])
    print(f"  Bull Sharpe: {regime['sharpe_bull']:.3f}, Bear Sharpe: {regime['sharpe_bear']:.3f}, Gap: {regime['regime_gap']:.3f}")

    # 2022 bear market
    bear_2022 = bear_2022_performance(strat_ret)
    if bear_2022:
        print(f"  2022 return: {bear_2022['return_2022']:.1f}%, Sharpe: {bear_2022['sharpe_2022']:.3f}")

    # Combined portfolio
    combo = combined_portfolio_analysis(strat_ret, qqq_ret)
    print(f"  60/40 combo Sharpe: {combo['combined_60_40']['sharpe']:.3f} vs QQQ-only: {combo['qqq_100']['sharpe']:.3f}")

    metrics = {
        "name": name,
        "description": description,
        "sharpe": round(sr, 3),
        "sortino": round(sort, 3),
        "maxdd_pct": round(mdd * 100, 2),
        "profit_factor": round(pf, 3),
        "win_rate_pct": round(wr * 100, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "trade_count": trade_count,
        "perm_p": round(perm_p, 4),
        "qqq_correlation": corr,
        "regime": regime,
        "bear_2022": bear_2022,
        "combined_portfolio": combo,
        "account_size": ACCOUNT_SIZE,
        "oot_period": f"{OOT_START} to {OOT_END}",
    }

    metrics = validate_5gate(metrics)

    gate_str = "PASS" if metrics["all_gates_pass"] else "FAIL"
    print(f"  5-Gate: {gate_str} ({metrics['gates_passed']}/5)")
    for g, v in metrics["gates"].items():
        status = "OK" if v else "FAIL"
        print(f"    {status}: {g}")

    return metrics


def main():
    close = download_data()

    variants = [
        ("A) Energy-Utilities Spread", "Z-score of XLE/XLU ratio, mean-reversion", variant_a_energy_utilities),
        ("B) Financials Rate Play", "TLT movement drives XLF vs XLU rotation", variant_b_financials_rate),
        ("C) Healthcare Momentum", "Buy XLV when defensive outperforms in weak market", variant_c_healthcare_momentum),
        ("D) Real Estate Rate Sensitivity", "Buy XLRE when rates falling (TLT momentum)", variant_d_realestate_rate),
        ("E) Anti-Tech Barbell", "Equal weight XLE+XLV+XLF, monthly rebalance", variant_e_antitech_barbell),
        ("F) Sector Momentum ex-Tech", "Top-2 sector momentum, monthly rebalance", variant_f_sector_momentum_extech),
    ]

    results = {}
    for name, desc, func in variants:
        try:
            metrics = run_variant(name, desc, close, func)
            results[name] = metrics
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"name": name, "error": str(e)}

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY — SECTOR RELATIVE VALUE STRATEGIES")
    print("=" * 70)
    print(f"{'Variant':<35} {'Sharpe':>7} {'QQQ r':>7} {'MaxDD':>7} {'Perm p':>7} {'Gates':>6}")
    print("-" * 70)

    passing = []
    for name, m in results.items():
        if "error" in m:
            print(f"{name:<35} ERROR: {m['error']}")
            continue
        gate_str = f"{m['gates_passed']}/5"
        if m["all_gates_pass"]:
            gate_str += " *"
            passing.append(name)
        print(f"{name:<35} {m['sharpe']:>7.3f} {m['qqq_correlation']['overall']:>7.3f} {m['maxdd_pct']:>6.1f}% {m['perm_p']:>7.4f} {gate_str:>6}")

    print("-" * 70)

    # Diversification value
    print("\n  DIVERSIFICATION VALUE (QQQ correlation < 0.5 = useful)")
    for name, m in results.items():
        if "error" in m:
            continue
        corr = m["qqq_correlation"]["overall"]
        useful = "YES - good diversifier" if abs(corr) < 0.5 else "NO - too correlated"
        print(f"  {name:<35} r={corr:>6.3f}  {useful}")

    # 2022 bear performance
    print("\n  2022 BEAR MARKET PERFORMANCE")
    for name, m in results.items():
        if "error" in m or not m.get("bear_2022"):
            continue
        b = m["bear_2022"]
        print(f"  {name:<35} Return: {b['return_2022']:>6.1f}%  Sharpe: {b['sharpe_2022']:>6.3f}")

    # Combined portfolios
    print("\n  COMBINED PORTFOLIO: 60% QQQ + 40% Variant vs 100% QQQ")
    for name, m in results.items():
        if "error" in m:
            continue
        c = m["combined_portfolio"]
        improvement = c["combined_60_40"]["sharpe"] - c["qqq_100"]["sharpe"]
        print(f"  {name:<35} Combo Sharpe: {c['combined_60_40']['sharpe']:>6.3f}  QQQ: {c['qqq_100']['sharpe']:>6.3f}  Delta: {improvement:>+6.3f}")

    if passing:
        print(f"\n  STRATEGIES PASSING ALL 5 GATES: {', '.join(passing)}")
    else:
        print("\n  NO STRATEGIES PASSED ALL 5 GATES")

    # Save results
    output = {
        "run_date": datetime.now().isoformat(),
        "config": {
            "account_size": ACCOUNT_SIZE,
            "slippage_pct": SLIPPAGE_PCT,
            "oot_period": f"{OOT_START} to {OOT_END}",
            "perm_iterations": PERM_ITERS,
            "validation_gates": {
                "sharpe": "> 0.5",
                "perm_p": "< 0.05",
                "regime_gap": "< 0.5",
                "maxdd": "> -50%",
                "min_trades": ">= 20",
            }
        },
        "variants": results,
        "summary": {
            "passing_all_gates": passing,
            "best_diversifier": min(
                [(n, abs(m["qqq_correlation"]["overall"])) for n, m in results.items() if "error" not in m],
                key=lambda x: x[1]
            )[0] if results else None,
        }
    }

    output_path = Path("/home/jupiter/Lvl3Quant/data/sector_relative_value_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
