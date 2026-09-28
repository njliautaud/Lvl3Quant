#!/usr/bin/env python3
"""
EARNINGS GUIDANCE QUALITY BACKTEST
===================================
Strategy: Use day-after-earnings price action as a proxy for guidance quality.
Companies that beat AND rally = good guidance. Beat but sell = bad guidance.

Walk-forward OOT: Jan 2022 – Jul 2026
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, ≥20 trades

Universe: 24 liquid names across mega-cap tech and high-beta growth.
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "COIN",
    "RBLX", "UBER", "LYFT", "DDOG", "TTD", "NET", "SHOP", "ROKU",
]
START_DATE = "2021-06-01"  # buffer for SMA warmup
END_DATE = "2026-07-30"
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"

ACCOUNT_SIZE = 669.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% on shares
COMMISSION = 0.0

GAP_THRESHOLD = 0.02  # 2% abs gap = proxy for earnings
STRONG_GAP = 0.03     # 3% for classification

N_PERMUTATIONS = 1000
np.random.seed(42)


# ── DATA DOWNLOAD ───────────────────────────────────────────────────────────
def download_data():
    """Download OHLCV for universe + SPY."""
    tickers = UNIVERSE + ["SPY"]
    print(f"Downloading {len(tickers)} tickers...")
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[t] = df
                print(f"  {t}: {len(df)} bars")
            else:
                print(f"  {t}: SKIP (only {len(df)} bars)")
        except Exception as e:
            print(f"  {t}: ERROR {e}")
    return data


# ── EARNINGS EVENT DETECTION ────────────────────────────────────────────────
def detect_earnings_events(price_data):
    """
    Detect earnings events via large overnight gaps (>2% absolute).
    Classify each event by gap direction + day-1 follow-through.
    """
    events = []
    for ticker, df in price_data.items():
        if ticker == "SPY":
            continue
        df = df.copy().sort_index()
        # Overnight gap = today's open vs yesterday's close
        df["prev_close"] = df["Close"].shift(1)
        df["gap_pct"] = (df["Open"] - df["prev_close"]) / df["prev_close"]
        df["day_return"] = (df["Close"] - df["Open"]) / df["Open"]  # intraday after gap

        for i in range(1, len(df)):
            row = df.iloc[i]
            gap = row["gap_pct"]
            if abs(gap) < GAP_THRESHOLD:
                continue
            date = df.index[i]
            if date < pd.Timestamp(OOT_START) or date > pd.Timestamp(OOT_END):
                continue

            day_ret = row["day_return"]
            open_price = row["Open"]
            close_price = row["Close"]
            follow_through = close_price > open_price  # day-1 close > gap open

            # Classify
            if gap > STRONG_GAP and follow_through:
                event_type = "strong_beat"
            elif gap > STRONG_GAP and not follow_through:
                event_type = "beat_but_sell"
            elif gap < -STRONG_GAP and not follow_through:
                event_type = "miss_and_dump"
            elif gap < -STRONG_GAP and follow_through:
                event_type = "miss_but_recover"
            else:
                event_type = "minor"  # between 2-3%, skip for now

            events.append({
                "ticker": ticker,
                "date": date,
                "gap_pct": gap,
                "day_return": day_ret,
                "event_type": event_type,
                "open_price": open_price,
                "close_price": close_price,
            })

    events_df = pd.DataFrame(events)
    if len(events_df) > 0:
        events_df = events_df.sort_values("date").reset_index(drop=True)
    print(f"\nDetected {len(events_df)} earnings-proxy events")
    if len(events_df) > 0:
        print("  Type distribution:")
        print(events_df["event_type"].value_counts().to_string())
    return events_df


# ── FORWARD RETURNS ─────────────────────────────────────────────────────────
def compute_forward_returns(events_df, price_data, hold_days):
    """Compute forward returns for each event over hold_days."""
    fwd_returns = []
    for _, ev in events_df.iterrows():
        ticker = ev["ticker"]
        date = ev["date"]
        df = price_data[ticker]
        idx = df.index.get_loc(date)
        exit_idx = min(idx + hold_days, len(df) - 1)
        if exit_idx <= idx:
            continue
        entry_price = ev["close_price"]  # enter at day-1 close
        exit_price = df.iloc[exit_idx]["Close"]
        raw_ret = (exit_price - entry_price) / entry_price
        net_ret = raw_ret - SLIPPAGE_PCT * 2  # entry + exit slippage
        fwd_returns.append({
            **ev.to_dict(),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "hold_days": hold_days,
            "raw_return": raw_ret,
            "net_return": net_ret,
            "exit_date": df.index[exit_idx],
        })
    return pd.DataFrame(fwd_returns)


# ── SPY REGIME ──────────────────────────────────────────────────────────────
def get_spy_regime(spy_df, date, sma_period=200):
    """Return 'bull' if SPY > SMA, else 'bear'."""
    spy = spy_df.copy()
    spy["SMA"] = spy["Close"].rolling(sma_period).mean()
    if date not in spy.index:
        # Find nearest prior date
        mask = spy.index <= date
        if mask.sum() == 0:
            return "unknown"
        date = spy.index[mask][-1]
    idx = spy.index.get_loc(date)
    if pd.isna(spy.iloc[idx]["SMA"]):
        return "unknown"
    return "bull" if spy.iloc[idx]["Close"] > spy.iloc[idx]["SMA"] else "bear"


def get_spy_above_sma50(spy_df, date):
    """Return True if SPY > 50-SMA on date."""
    spy = spy_df.copy()
    spy["SMA50"] = spy["Close"].rolling(50).mean()
    if date not in spy.index:
        mask = spy.index <= date
        if mask.sum() == 0:
            return False
        date = spy.index[mask][-1]
    idx = spy.index.get_loc(date)
    if pd.isna(spy.iloc[idx]["SMA50"]):
        return False
    return spy.iloc[idx]["Close"] > spy.iloc[idx]["SMA50"]


# ── BACKTEST ENGINE ─────────────────────────────────────────────────────────
def run_backtest(trades_df, spy_df, variant_name):
    """
    Run portfolio-level backtest with position sizing.
    Returns equity curve and trade-level stats.
    """
    if len(trades_df) == 0:
        return empty_result(variant_name)

    trades = trades_df.sort_values("date").reset_index(drop=True)
    n_trades = len(trades)

    # Position sizing: equal weight, max 3 concurrent
    # Simple approach: size = account / max_concurrent
    pos_size = ACCOUNT_SIZE / MAX_CONCURRENT

    # Trade-level P&L
    trades["pnl_dollar"] = trades["net_return"] * pos_size
    trades["pnl_pct"] = trades["net_return"]

    # Compute regime for each trade
    regimes = []
    for _, t in trades.iterrows():
        regimes.append(get_spy_regime(spy_df, t["date"]))
    trades["regime"] = regimes

    # Build daily equity curve
    all_dates = pd.date_range(OOT_START, OOT_END, freq="B")
    equity = pd.Series(ACCOUNT_SIZE, index=all_dates)
    daily_pnl = pd.Series(0.0, index=all_dates)

    for _, t in trades.iterrows():
        entry_date = t["date"]
        exit_date = t["exit_date"]
        hold = t["hold_days"]
        daily_ret = t["net_return"] / hold if hold > 0 else 0
        daily_dollar = daily_ret * pos_size
        mask = (all_dates >= entry_date) & (all_dates <= exit_date)
        daily_pnl[mask] += daily_dollar

    equity = ACCOUNT_SIZE + daily_pnl.cumsum()

    # Performance metrics
    daily_returns = daily_pnl / ACCOUNT_SIZE
    # Only count days where we had positions
    active_returns = daily_returns[daily_returns != 0]

    total_return = (equity.iloc[-1] - ACCOUNT_SIZE) / ACCOUNT_SIZE
    ann_factor = np.sqrt(252)

    if len(active_returns) > 1 and active_returns.std() > 0:
        sharpe = active_returns.mean() / active_returns.std() * ann_factor
        downside = active_returns[active_returns < 0]
        downside_std = downside.std() if len(downside) > 1 else active_returns.std()
        sortino = active_returns.mean() / downside_std * ann_factor if downside_std > 0 else 0
    else:
        sharpe = 0.0
        sortino = 0.0

    winners = trades[trades["net_return"] > 0]
    losers = trades[trades["net_return"] <= 0]
    win_rate = len(winners) / n_trades if n_trades > 0 else 0
    gross_profit = winners["pnl_dollar"].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers["pnl_dollar"].sum()) if len(losers) > 0 else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # Max drawdown
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Regime analysis
    bull_trades = trades[trades["regime"] == "bull"]
    bear_trades = trades[trades["regime"] == "bear"]

    def regime_sharpe(regime_trades):
        if len(regime_trades) < 3:
            return 0.0
        rets = regime_trades["net_return"]
        if rets.std() == 0:
            return 0.0
        return rets.mean() / rets.std() * ann_factor

    sharpe_bull = regime_sharpe(bull_trades)
    sharpe_bear = regime_sharpe(bear_trades)
    max_regime_sharpe = max(abs(sharpe_bull), abs(sharpe_bear), 0.001)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_regime_sharpe

    # Avg trade stats
    avg_return = trades["net_return"].mean()
    avg_win = winners["net_return"].mean() if len(winners) > 0 else 0
    avg_loss = losers["net_return"].mean() if len(losers) > 0 else 0

    return {
        "variant": variant_name,
        "n_trades": n_trades,
        "total_return_pct": round(total_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "avg_return_pct": round(avg_return * 100, 3),
        "avg_win_pct": round(avg_win * 100, 3),
        "avg_loss_pct": round(avg_loss * 100, 3),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "n_bull_trades": len(bull_trades),
        "n_bear_trades": len(bear_trades),
        "final_equity": round(equity.iloc[-1], 2),
        "trades_df": trades,
        "equity_curve": equity,
    }


def empty_result(variant_name):
    return {
        "variant": variant_name,
        "n_trades": 0,
        "total_return_pct": 0,
        "sharpe": 0,
        "sortino": 0,
        "profit_factor": 0,
        "win_rate": 0,
        "max_drawdown_pct": 0,
        "avg_return_pct": 0,
        "avg_win_pct": 0,
        "avg_loss_pct": 0,
        "sharpe_bull": 0,
        "sharpe_bear": 0,
        "regime_gap": 0,
        "n_bull_trades": 0,
        "n_bear_trades": 0,
        "final_equity": ACCOUNT_SIZE,
        "trades_df": pd.DataFrame(),
        "equity_curve": pd.Series(),
    }


# ── PERMUTATION TEST ───────────────────────────────────────────────────────
def permutation_test(trades_df, n_perms=N_PERMUTATIONS):
    """Shuffle trade selections to test if edge is real."""
    if len(trades_df) < 5:
        return 1.0
    actual_mean = trades_df["net_return"].mean()
    all_returns = trades_df["net_return"].values
    n = len(all_returns)
    count_better = 0
    for _ in range(n_perms):
        shuffled = np.random.choice(all_returns, size=n, replace=True)
        if shuffled.mean() >= actual_mean:
            count_better += 1
    return count_better / n_perms


# ── VARIANT DEFINITIONS ────────────────────────────────────────────────────
def build_variants(events_df, price_data, spy_df):
    """Build trade lists for each variant."""
    variants = {}

    # A) Strong Beat Momentum: buy after strong beat, hold 20 days
    strong_beats = events_df[events_df["event_type"] == "strong_beat"]
    trades_a = compute_forward_returns(strong_beats, price_data, hold_days=20)
    variants["A_StrongBeatMomentum"] = trades_a

    # B) Beat-But-Sell Contrarian: buy after beat-but-sell, hold 10 days
    beat_sell = events_df[events_df["event_type"] == "beat_but_sell"]
    trades_b = compute_forward_returns(beat_sell, price_data, hold_days=10)
    variants["B_BeatButSellContrarian"] = trades_b

    # C) Miss-But-Recover: buy after miss but recover, hold 20 days
    miss_recover = events_df[events_df["event_type"] == "miss_but_recover"]
    trades_c = compute_forward_returns(miss_recover, price_data, hold_days=20)
    variants["C_MissButRecover"] = trades_c

    # D) Combined Quality: strong beats + miss-but-recover, hold 20 days
    quality = events_df[events_df["event_type"].isin(["strong_beat", "miss_but_recover"])]
    trades_d = compute_forward_returns(quality, price_data, hold_days=20)
    variants["D_CombinedQuality"] = trades_d

    # E) Quality Momentum Long-Term: strong beats, hold 40 days
    trades_e = compute_forward_returns(strong_beats, price_data, hold_days=40)
    variants["E_QualityMomentumLT"] = trades_e

    # F) Regime-Adjusted Quality: only enter when SPY > 50-SMA
    quality_events = events_df[events_df["event_type"].isin(["strong_beat", "miss_but_recover"])]
    regime_filtered = quality_events[
        quality_events["date"].apply(lambda d: get_spy_above_sma50(spy_df, d))
    ]
    trades_f = compute_forward_returns(regime_filtered, price_data, hold_days=20)
    variants["F_RegimeAdjQuality"] = trades_f

    return variants


# ── 5-GATE VALIDATION ──────────────────────────────────────────────────────
def validate_5gate(result, perm_p):
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": result["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": result["regime_gap"] < 0.5,
        "max_dd_gt_neg50": result["max_drawdown_pct"] > -50,
        "min_20_trades": result["n_trades"] >= 20,
    }
    passed = sum(gates.values())
    return gates, passed


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("EARNINGS GUIDANCE QUALITY BACKTEST")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Account: ${ACCOUNT_SIZE}, Max concurrent: {MAX_CONCURRENT}")
    print("=" * 70)

    # 1. Download data
    price_data = download_data()
    if "SPY" not in price_data:
        print("ERROR: Could not download SPY data")
        sys.exit(1)
    spy_df = price_data["SPY"]

    # 2. Detect earnings events
    events_df = detect_earnings_events(price_data)
    if len(events_df) == 0:
        print("ERROR: No earnings events detected")
        sys.exit(1)

    # 3. Build variants
    print("\n" + "=" * 70)
    print("BUILDING VARIANTS")
    print("=" * 70)
    variants = build_variants(events_df, price_data, spy_df)

    # 4. Run backtests + validation
    results_all = []
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    for name, trades_df in variants.items():
        print(f"\n{'─' * 50}")
        print(f"VARIANT: {name}")
        print(f"{'─' * 50}")

        result = run_backtest(trades_df, spy_df, name)

        # Permutation test
        if len(trades_df) > 0:
            perm_p = permutation_test(trades_df)
        else:
            perm_p = 1.0
        result["perm_p_value"] = round(perm_p, 4)

        # 5-gate validation
        gates, n_passed = validate_5gate(result, perm_p)
        result["gates"] = gates
        result["gates_passed"] = n_passed
        result["passes_all_gates"] = n_passed == 5

        # Print results
        print(f"  Trades: {result['n_trades']}")
        print(f"  Total Return: {result['total_return_pct']}%")
        print(f"  Sharpe: {result['sharpe']}")
        print(f"  Sortino: {result['sortino']}")
        print(f"  Profit Factor: {result['profit_factor']}")
        print(f"  Win Rate: {result['win_rate']:.1%}")
        print(f"  Max DD: {result['max_drawdown_pct']}%")
        print(f"  Avg Return: {result['avg_return_pct']}%")
        print(f"  Avg Win: {result['avg_win_pct']}% | Avg Loss: {result['avg_loss_pct']}%")
        print(f"  Sharpe Bull: {result['sharpe_bull']} | Bear: {result['sharpe_bear']}")
        print(f"  Regime Gap: {result['regime_gap']}")
        print(f"  Perm p-value: {result['perm_p_value']}")
        print(f"  Final Equity: ${result['final_equity']}")
        print(f"  5-Gate: {n_passed}/5 {'✓ PASS' if result['passes_all_gates'] else '✗ FAIL'}")
        for gate_name, passed in gates.items():
            print(f"    {'✓' if passed else '✗'} {gate_name}")

        # Remove non-serializable fields for JSON
        result_clean = {k: v for k, v in result.items()
                       if k not in ("trades_df", "equity_curve")}
        results_all.append(result_clean)

    # 5. Event summary stats
    event_summary = {}
    for etype in ["strong_beat", "beat_but_sell", "miss_and_dump", "miss_but_recover", "minor"]:
        subset = events_df[events_df["event_type"] == etype]
        event_summary[etype] = {
            "count": len(subset),
            "avg_gap_pct": round(subset["gap_pct"].mean() * 100, 2) if len(subset) > 0 else 0,
            "avg_day_return_pct": round(subset["day_return"].mean() * 100, 2) if len(subset) > 0 else 0,
        }

    # 6. Top trades analysis for best variant
    best = max(results_all, key=lambda x: x["sharpe"])
    best_name = best["variant"]
    best_trades = variants[best_name]
    top_trades = []
    if len(best_trades) > 0:
        sorted_t = best_trades.sort_values("net_return", ascending=False)
        for _, t in sorted_t.head(5).iterrows():
            top_trades.append({
                "ticker": t["ticker"],
                "date": t["date"].strftime("%Y-%m-%d"),
                "event_type": t["event_type"],
                "gap_pct": round(t["gap_pct"] * 100, 2),
                "net_return_pct": round(t["net_return"] * 100, 2),
            })

    # 7. Save results
    output = {
        "metadata": {
            "strategy": "Earnings Guidance Quality",
            "oot_period": f"{OOT_START} to {OOT_END}",
            "account_size": ACCOUNT_SIZE,
            "universe_size": len(UNIVERSE),
            "total_events_detected": len(events_df),
            "slippage_pct": SLIPPAGE_PCT,
            "commission": COMMISSION,
            "n_permutations": N_PERMUTATIONS,
            "run_timestamp": datetime.now().isoformat(),
        },
        "event_summary": event_summary,
        "variants": results_all,
        "best_variant": best_name,
        "best_sharpe": best["sharpe"],
        "top_trades_best_variant": top_trades,
        "any_passes_all_gates": any(r["passes_all_gates"] for r in results_all),
    }

    out_path = Path("/home/jupiter/Lvl3Quant/data/guidance_quality_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # 8. Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Best variant: {best_name} (Sharpe: {best['sharpe']})")
    print(f"Any pass all 5 gates: {output['any_passes_all_gates']}")
    print(f"\nTotal earnings events: {len(events_df)}")
    print(f"Event breakdown: {json.dumps(event_summary, indent=2)}")

    # Variant comparison table
    print(f"\n{'Variant':<30} {'Trades':>6} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'Gates':>5}")
    print("-" * 70)
    for r in results_all:
        print(f"{r['variant']:<30} {r['n_trades']:>6} {r['sharpe']:>7.3f} "
              f"{r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} "
              f"{r['max_drawdown_pct']:>6.1f}% {r['gates_passed']:>3}/5")

    return output


if __name__ == "__main__":
    main()
