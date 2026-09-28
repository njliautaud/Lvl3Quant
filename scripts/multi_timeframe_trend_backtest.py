#!/usr/bin/env python3
"""
Multi-Timeframe Trend Confluence Backtest
==========================================
Enters trades only when multiple timeframes agree on direction.
Walk-forward OOT: Jan 2022 – Jul 2026.
5-gate validation: Sharpe >0.5, perm p<0.05, regime gap <0.5, MaxDD >-50%, >=20 trades.

6 Variants:
A) SPY Full Alignment (binary)
B) SPY Graduated (score-weighted)
C) SPY + QQQ Relative (best scorer)
D) Leveraged Confidence (TQQQ/QQQ/SPY/Cash)
E) Long-Short (SPY/SH extremes only)
F) Sector Rotation + Trend (momentum-based sector pick)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-06-01"  # need 200-SMA warmup
PERM_ITERS = 1000
SEED = 42

TICKERS = ["SPY", "QQQ", "TQQQ", "SH", "TLT", "IWM",
           "XLK", "XLE", "XLF", "XLV", "XLY", "XLC"]

SMA_WINDOWS = [5, 20, 50, 200]
SECTOR_ETFS = ["XLK", "XLE", "XLF", "XLV", "XLY", "XLC"]

OUT_PATH = Path("/home/jupiter/Lvl3Quant/data/multi_timeframe_trend_results.json")


# ── Data Download ───────────────────────────────────────────────────────
def download_data():
    print("Downloading price data...")
    data = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
            if len(df) > 200:
                data[t] = df["Close"].squeeze()
                print(f"  {t}: {len(df)} bars")
            else:
                print(f"  {t}: insufficient data ({len(df)} bars)")
        except Exception as e:
            print(f"  {t}: download failed - {e}")
    return pd.DataFrame(data).dropna(how="all")


# ── Trend Score ─────────────────────────────────────────────────────────
def compute_trend_scores(prices_df):
    """Compute trend score (0-4) for each ticker each day."""
    scores = {}
    for t in prices_df.columns:
        p = prices_df[t].dropna()
        score = pd.Series(0, index=p.index, dtype=int)
        for w in SMA_WINDOWS:
            sma = p.rolling(w).mean()
            score = score + (p > sma).astype(int)
        scores[t] = score
    return pd.DataFrame(scores)


def compute_momentum_20d(prices_df):
    """20-day momentum (return) for each ticker."""
    return prices_df.pct_change(20)


# ── Backtest Engine ─────────────────────────────────────────────────────
def backtest(prices_df, daily_holdings_func, scores_df, mom_df, label="strategy"):
    """
    Generic backtest engine.
    daily_holdings_func(date, scores, mom, prices) -> dict {ticker: weight}
    weights sum to <= 1.0, remainder is cash.
    """
    oot_mask = (prices_df.index >= OOT_START) & (prices_df.index <= OOT_END)
    dates = prices_df.index[oot_mask]
    if len(dates) < 20:
        return None

    capital = INITIAL_CAPITAL
    equity_curve = []
    prev_holdings = {}  # ticker -> weight
    trade_dates = []

    for i, date in enumerate(dates):
        # get target holdings
        s_row = scores_df.loc[date] if date in scores_df.index else None
        m_row = mom_df.loc[date] if date in mom_df.index else None
        p_row = prices_df.loc[date] if date in prices_df.index else None

        if s_row is None or p_row is None:
            equity_curve.append(capital)
            continue

        target = daily_holdings_func(date, s_row, m_row, p_row)
        # Clean NaN weights
        target = {k: v for k, v in target.items() if not np.isnan(v) and v > 0}

        # Apply slippage on rebalance
        changed = set(target.keys()) != set(prev_holdings.keys())
        if not changed:
            for k in target:
                if k in prev_holdings and abs(target[k] - prev_holdings.get(k, 0)) > 0.01:
                    changed = True
                    break

        if changed and i > 0:
            trade_dates.append(date)
            # slippage cost proportional to turnover
            turnover = 0
            all_tickers = set(list(target.keys()) + list(prev_holdings.keys()))
            for t in all_tickers:
                turnover += abs(target.get(t, 0) - prev_holdings.get(t, 0))
            slippage_cost = capital * turnover * SLIPPAGE_PCT
            capital -= slippage_cost

        # compute daily return
        if i > 0:
            day_ret = 0.0
            for t, w in prev_holdings.items():
                if t in prices_df.columns:
                    prev_price = prices_df[t].loc[dates[i - 1]] if dates[i - 1] in prices_df.index else None
                    curr_price = prices_df[t].loc[date] if date in prices_df.index else None
                    if prev_price is not None and curr_price is not None and prev_price > 0:
                        day_ret += w * (curr_price / prev_price - 1)
            capital *= (1 + day_ret)

        prev_holdings = target
        equity_curve.append(capital)

    return {
        "equity_curve": equity_curve,
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "trade_dates": [d.strftime("%Y-%m-%d") for d in trade_dates],
        "n_trades": len(trade_dates),
        "label": label,
    }


# ── Strategy Variants ──────────────────────────────────────────────────
def variant_a_full_alignment(date, scores, mom, prices):
    """Buy SPY when all 4 timeframes bullish, else cash."""
    spy_score = scores.get("SPY", 0) if isinstance(scores, dict) else scores.get("SPY", 0)
    if hasattr(spy_score, 'item'):
        spy_score = spy_score.item()
    if spy_score == 4:
        return {"SPY": 1.0}
    return {}


def variant_b_graduated(date, scores, mom, prices):
    """Score 4=100%, 3=75%, 2=50%, 1=25%, 0=0%."""
    spy_score = scores.get("SPY", 0)
    if hasattr(spy_score, 'item'):
        spy_score = spy_score.item()
    weight = spy_score / 4.0
    if weight > 0:
        return {"SPY": weight}
    return {}


def variant_c_relative(date, scores, mom, prices):
    """Hold whichever of SPY/QQQ has higher trend score. Tiebreak: 20d momentum."""
    spy_s = scores.get("SPY", 0)
    qqq_s = scores.get("QQQ", 0)
    if hasattr(spy_s, 'item'):
        spy_s = spy_s.item()
    if hasattr(qqq_s, 'item'):
        qqq_s = qqq_s.item()

    if spy_s == 0 and qqq_s == 0:
        return {}

    if qqq_s > spy_s:
        return {"QQQ": 1.0}
    elif spy_s > qqq_s:
        return {"SPY": 1.0}
    else:
        # tiebreak: 20d momentum
        spy_m = mom.get("SPY", 0) if mom is not None else 0
        qqq_m = mom.get("QQQ", 0) if mom is not None else 0
        if hasattr(spy_m, 'item'):
            spy_m = spy_m.item()
        if hasattr(qqq_m, 'item'):
            qqq_m = qqq_m.item()
        if np.isnan(spy_m):
            spy_m = 0
        if np.isnan(qqq_m):
            qqq_m = 0
        if qqq_m > spy_m:
            return {"QQQ": 1.0}
        return {"SPY": 1.0}


def variant_d_leveraged(date, scores, mom, prices):
    """Score 4=TQQQ, 3=QQQ, 2=SPY, 0-1=Cash."""
    spy_score = scores.get("SPY", 0)
    if hasattr(spy_score, 'item'):
        spy_score = spy_score.item()
    if spy_score == 4:
        return {"TQQQ": 1.0}
    elif spy_score == 3:
        return {"QQQ": 1.0}
    elif spy_score == 2:
        return {"SPY": 1.0}
    return {}


def variant_e_longshort(date, scores, mom, prices):
    """Score 4=SPY long, Score 0=SH (inverse), 1-3=cash."""
    spy_score = scores.get("SPY", 0)
    if hasattr(spy_score, 'item'):
        spy_score = spy_score.item()
    if spy_score == 4:
        return {"SPY": 1.0}
    elif spy_score == 0:
        return {"SH": 1.0}
    return {}


def variant_f_sector_rotation(date, scores, mom, prices):
    """
    SPY score >= 3: buy sector ETF with highest 20d momentum.
    SPY score <= 1: buy XLV or TLT (whichever has higher 20d mom).
    Score 2: cash.
    """
    spy_score = scores.get("SPY", 0)
    if hasattr(spy_score, 'item'):
        spy_score = spy_score.item()

    if spy_score >= 3:
        # pick best sector by 20d momentum
        best_t, best_m = None, -999
        for s in SECTOR_ETFS:
            m = mom.get(s, np.nan) if mom is not None else np.nan
            if hasattr(m, 'item'):
                m = m.item()
            if not np.isnan(m) and m > best_m:
                best_m = m
                best_t = s
        if best_t:
            return {best_t: 1.0}
        return {"SPY": 1.0}
    elif spy_score <= 1:
        # defensive
        xlv_m = mom.get("XLV", np.nan) if mom is not None else np.nan
        tlt_m = mom.get("TLT", np.nan) if mom is not None else np.nan
        if hasattr(xlv_m, 'item'):
            xlv_m = xlv_m.item()
        if hasattr(tlt_m, 'item'):
            tlt_m = tlt_m.item()
        if np.isnan(xlv_m):
            xlv_m = -999
        if np.isnan(tlt_m):
            tlt_m = -999
        if tlt_m > xlv_m:
            return {"TLT": 1.0}
        return {"XLV": 1.0}
    return {}


# ── Metrics ─────────────────────────────────────────────────────────────
def compute_metrics(result, prices_df):
    """Compute Sharpe, Sortino, MaxDD, PF, WR, regime analysis."""
    eq = np.array(result["equity_curve"])
    dates = pd.to_datetime(result["dates"])

    if len(eq) < 20:
        return None

    daily_rets = np.diff(eq) / eq[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]

    if len(daily_rets) < 20:
        return None

    # Sharpe (annualized)
    mu = np.mean(daily_rets)
    sigma = np.std(daily_rets, ddof=1)
    sharpe = (mu / sigma * np.sqrt(252)) if sigma > 0 else 0.0

    # Sortino
    downside = daily_rets[daily_rets < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mu / down_std * np.sqrt(252)) if down_std > 0 else 0.0

    # Max Drawdown
    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / running_max
    max_dd = np.min(drawdown)

    # Profit Factor & Win Rate (daily)
    wins = daily_rets[daily_rets > 0]
    losses = daily_rets[daily_rets < 0]
    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else 999.0
    wr = len(wins) / len(daily_rets) if len(daily_rets) > 0 else 0

    # Total return
    total_ret = (eq[-1] / eq[0] - 1) * 100

    # CAGR
    years = len(daily_rets) / 252
    cagr = ((eq[-1] / eq[0]) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Regime analysis (Bull = SPY > 200-SMA, Bear = SPY < 200-SMA)
    spy_close = prices_df["SPY"].reindex(dates).ffill()
    spy_200 = spy_close.rolling(200).mean()
    bull_mask = spy_close > spy_200
    bear_mask = spy_close <= spy_200

    # regime returns (aligned with daily_rets which is 1 shorter)
    bull_days = bull_mask.values[1:]
    bear_days = bear_mask.values[1:]

    bull_rets = daily_rets[bull_days[:len(daily_rets)]] if np.any(bull_days[:len(daily_rets)]) else np.array([0])
    bear_rets = daily_rets[bear_days[:len(daily_rets)]] if np.any(bear_days[:len(daily_rets)]) else np.array([0])

    bull_sharpe = (np.mean(bull_rets) / (np.std(bull_rets, ddof=1) + 1e-9)) * np.sqrt(252) if len(bull_rets) > 1 else 0
    bear_sharpe = (np.mean(bear_rets) / (np.std(bear_rets, ddof=1) + 1e-9)) * np.sqrt(252) if len(bear_rets) > 1 else 0

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd_pct": round(max_dd * 100, 2),
        "profit_factor": round(pf, 4),
        "win_rate": round(wr * 100, 2),
        "total_return_pct": round(total_ret, 2),
        "cagr_pct": round(cagr, 2),
        "final_equity": round(eq[-1], 2),
        "n_trades": result["n_trades"],
        "n_days": len(daily_rets),
        "bull_sharpe": round(bull_sharpe, 4),
        "bear_sharpe": round(bear_sharpe, 4),
        "regime_gap": round(regime_gap, 4),
        "bull_days": int(np.sum(bull_days[:len(daily_rets)])),
        "bear_days": int(np.sum(bear_days[:len(daily_rets)])),
    }


# ── Permutation Test ────────────────────────────────────────────────────
def permutation_test(daily_rets, n_iter=PERM_ITERS):
    """Shuffle daily returns, compute Sharpe each time. Return p-value."""
    rng = np.random.RandomState(SEED)
    actual_sharpe = np.mean(daily_rets) / (np.std(daily_rets, ddof=1) + 1e-9) * np.sqrt(252)

    count_ge = 0
    for _ in range(n_iter):
        shuffled = rng.permutation(daily_rets)
        s = np.mean(shuffled) / (np.std(shuffled, ddof=1) + 1e-9) * np.sqrt(252)
        if s >= actual_sharpe:
            count_ge += 1

    return round(count_ge / n_iter, 4)


# ── 5-Gate Validation ──────────────────────────────────────────────────
def validate_5gate(metrics, perm_p):
    """
    Gate 1: Sharpe > 0.5
    Gate 2: Permutation p < 0.05
    Gate 3: Regime gap < 0.5
    Gate 4: MaxDD > -50%
    Gate 5: >= 20 trades
    """
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "maxdd_gt_neg50": metrics["max_dd_pct"] > -50,
        "trades_ge_20": metrics["n_trades"] >= 20,
    }
    gates["all_pass"] = all(gates.values())
    return gates


# ── Buy & Hold Benchmark ───────────────────────────────────────────────
def buy_and_hold_benchmark(prices_df, ticker="SPY"):
    """Simple buy-and-hold benchmark."""
    oot_mask = (prices_df.index >= OOT_START) & (prices_df.index <= OOT_END)
    p = prices_df[ticker][oot_mask].dropna()
    if len(p) < 20:
        return None
    eq = INITIAL_CAPITAL * (p / p.iloc[0])
    daily_rets = eq.pct_change().dropna().values
    mu = np.mean(daily_rets)
    sigma = np.std(daily_rets, ddof=1)
    sharpe = mu / sigma * np.sqrt(252) if sigma > 0 else 0
    running_max = np.maximum.accumulate(eq.values)
    dd = (eq.values - running_max) / running_max
    max_dd = np.min(dd)
    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    return {
        "sharpe": round(sharpe, 4),
        "max_dd_pct": round(max_dd * 100, 2),
        "total_return_pct": round(total_ret, 2),
        "final_equity": round(eq.iloc[-1], 2),
    }


# ── Main ────────────────────────────────────────────────────────────────
def main():
    prices_df = download_data()
    if "SPY" not in prices_df.columns:
        print("FATAL: No SPY data.")
        return

    print("\nComputing trend scores and momentum...")
    scores_df = compute_trend_scores(prices_df)
    mom_df = compute_momentum_20d(prices_df)

    variants = {
        "A_Full_Alignment": variant_a_full_alignment,
        "B_Graduated": variant_b_graduated,
        "C_Relative_SPY_QQQ": variant_c_relative,
        "D_Leveraged_Confidence": variant_d_leveraged,
        "E_Long_Short": variant_e_longshort,
        "F_Sector_Rotation": variant_f_sector_rotation,
    }

    # Benchmark
    print("\n── Buy & Hold Benchmarks ──")
    bh_spy = buy_and_hold_benchmark(prices_df, "SPY")
    bh_qqq = buy_and_hold_benchmark(prices_df, "QQQ")
    print(f"  SPY B&H: Sharpe={bh_spy['sharpe']}, Return={bh_spy['total_return_pct']}%, MaxDD={bh_spy['max_dd_pct']}%, Final=${bh_spy['final_equity']}")
    print(f"  QQQ B&H: Sharpe={bh_qqq['sharpe']}, Return={bh_qqq['total_return_pct']}%, MaxDD={bh_qqq['max_dd_pct']}%, Final=${bh_qqq['final_equity']}")

    results = {
        "metadata": {
            "initial_capital": INITIAL_CAPITAL,
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "slippage_pct": SLIPPAGE_PCT,
            "perm_iterations": PERM_ITERS,
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "benchmarks": {
            "SPY_BuyHold": bh_spy,
            "QQQ_BuyHold": bh_qqq,
        },
        "variants": {},
    }

    print("\n── Running Variants ──")
    for name, func in variants.items():
        print(f"\n{'='*60}")
        print(f"  Variant {name}")
        print(f"{'='*60}")

        bt = backtest(prices_df, func, scores_df, mom_df, label=name)
        if bt is None:
            print(f"  SKIP: insufficient data")
            continue

        metrics = compute_metrics(bt, prices_df)
        if metrics is None:
            print(f"  SKIP: insufficient returns data")
            continue

        # Permutation test
        eq = np.array(bt["equity_curve"])
        daily_rets = np.diff(eq) / eq[:-1]
        daily_rets = daily_rets[np.isfinite(daily_rets)]
        print(f"  Running {PERM_ITERS} permutations...")
        perm_p = permutation_test(daily_rets)

        # 5-gate validation
        gates = validate_5gate(metrics, perm_p)

        print(f"  Sharpe:   {metrics['sharpe']}")
        print(f"  Sortino:  {metrics['sortino']}")
        print(f"  MaxDD:    {metrics['max_dd_pct']}%")
        print(f"  PF:       {metrics['profit_factor']}")
        print(f"  WR:       {metrics['win_rate']}%")
        print(f"  Return:   {metrics['total_return_pct']}%")
        print(f"  CAGR:     {metrics['cagr_pct']}%")
        print(f"  Final $:  ${metrics['final_equity']}")
        print(f"  Trades:   {metrics['n_trades']}")
        print(f"  Perm p:   {perm_p}")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']}, Bear Sharpe: {metrics['bear_sharpe']}")
        print(f"  Regime Gap:  {metrics['regime_gap']}")
        print(f"\n  5-Gate Validation:")
        for g, v in gates.items():
            status = "PASS" if v else "FAIL"
            print(f"    {g}: {status}")

        results["variants"][name] = {
            "metrics": metrics,
            "perm_p": perm_p,
            "gates": gates,
            "equity_start": round(eq[0], 2),
            "equity_end": round(eq[-1], 2),
        }

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<25} {'Sharpe':>8} {'Sortino':>8} {'MaxDD%':>8} {'Return%':>9} {'Final$':>8} {'5Gate':>6}")
    print("-" * 75)
    for name, v in results["variants"].items():
        m = v["metrics"]
        gate_str = "PASS" if v["gates"]["all_pass"] else "FAIL"
        print(f"{name:<25} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['max_dd_pct']:>7.1f}% {m['total_return_pct']:>8.1f}% {m['final_equity']:>8.2f} {gate_str:>6}")

    print(f"\n{'SPY B&H':<25} {bh_spy['sharpe']:>8.3f} {'':>8} {bh_spy['max_dd_pct']:>7.1f}% {bh_spy['total_return_pct']:>8.1f}% {bh_spy['final_equity']:>8.2f}")
    print(f"{'QQQ B&H':<25} {bh_qqq['sharpe']:>8.3f} {'':>8} {bh_qqq['max_dd_pct']:>7.1f}% {bh_qqq['total_return_pct']:>8.1f}% {bh_qqq['final_equity']:>8.2f}")

    # Save results (without equity curves for compactness)
    # Convert numpy types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    results = make_serializable(results)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
