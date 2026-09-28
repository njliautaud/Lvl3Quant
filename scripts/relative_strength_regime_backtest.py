#!/usr/bin/env python3
"""
Relative Strength + Regime Filter Backtest
6 variants testing whether relative strength fixes regime dependency.

Walk-forward OOT: Jan 2022 to present
Initial capital: $645, Commission: $0 (Robinhood)
5 gates: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
OOT_START = "2022-01-01"
DATA_START = "2020-01-01"  # extra history for indicators
N_PERM = 1000

STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "SHOP", "SQ", "COIN", "SNOW", "DDOG", "NET",
    "RBLX", "PLTR", "UBER", "LYFT"
]

SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLY", "XLC", "XLI", "XLB", "XLRE", "XLU", "XLP"]

GATES = {
    "sharpe_min": 0.5,
    "perm_p_max": 0.05,
    "regime_gap_max": 0.5,
    "max_dd_floor": -0.50,
    "min_trades": 20,
}


# ── Data Download ───────────────────────────────────────────────────────────
def download_data():
    """Download all needed price data."""
    all_tickers = list(set(STOCKS + SECTOR_ETFS + ["SPY", "^VIX"]))
    print(f"Downloading {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=DATA_START, auto_adjust=True, progress=False)
    close = data["Close"].copy()
    # Some tickers may have NaN — forward fill then drop remaining
    close = close.ffill()
    print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")
    return close


# ── Regime ──────────────────────────────────────────────────────────────────
def compute_regime(spy_close):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    sma200 = spy_close.rolling(200).mean()
    regime = pd.Series("bull", index=spy_close.index)
    regime[spy_close < sma200] = "bear"
    return regime


# ── Relative Strength ──────────────────────────────────────────────────────
def relative_strength(close_df, spy_close, lookback=20):
    """Return relative strength: stock return - SPY return over lookback."""
    stock_ret = close_df.pct_change(lookback)
    spy_ret = spy_close.pct_change(lookback)
    rs = stock_ret.sub(spy_ret, axis=0)
    return rs


def rs_rank(rs_df):
    """Rank stocks by RS each day (1 = best)."""
    return rs_df.rank(axis=1, ascending=False, method="min")


# ── Backtest Engine ─────────────────────────────────────────────────────────
def run_backtest(close_df, spy_close, vix_close, regime, signals_func, hold_days, universe_tickers):
    """
    Generic backtest engine.
    signals_func(date, idx, close_df, spy_close, vix_close, regime, universe_tickers)
        -> list of (ticker, weight) or empty list
    Returns daily equity curve and trade log.
    """
    oot_mask = close_df.index >= OOT_START
    dates = close_df.index[oot_mask]

    equity = INITIAL_CAPITAL
    equity_curve = []
    trades = []

    # Active positions: list of {ticker, entry_date, entry_price, shares, exit_idx}
    positions = []

    for i, date in enumerate(dates):
        global_idx = close_df.index.get_loc(date)

        # Close expired positions
        new_positions = []
        for pos in positions:
            if i >= pos["exit_idx"]:
                exit_price = close_df.loc[date, pos["ticker"]]
                if pd.isna(exit_price):
                    exit_price = pos["entry_price"]  # no change if missing
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                equity += pnl
                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "entry_price": round(float(pos["entry_price"]), 2),
                    "exit_price": round(float(exit_price), 2),
                    "shares": round(float(pos["shares"]), 4),
                    "pnl": round(float(pnl), 2),
                    "return_pct": round(float((exit_price / pos["entry_price"] - 1) * 100), 2),
                })
            else:
                new_positions.append(pos)
        positions = new_positions

        # Mark-to-market for equity curve
        mtm = equity
        for pos in positions:
            current_price = close_df.loc[date, pos["ticker"]]
            if not pd.isna(current_price):
                mtm += (current_price - pos["entry_price"]) * pos["shares"]
        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "equity": round(float(mtm), 2)})

        # Generate new signals (only if no positions or weekly rebalance day)
        if len(positions) == 0:
            sigs = signals_func(date, global_idx, close_df, spy_close, vix_close, regime, universe_tickers)
            if sigs:
                # Allocate capital equally (adjusted by weight)
                total_weight = sum(w for _, w in sigs)
                if total_weight > 0:
                    for ticker, weight in sigs:
                        alloc = equity * (weight / total_weight)
                        price = close_df.loc[date, ticker]
                        if pd.isna(price) or price <= 0:
                            continue
                        shares = alloc / price
                        exit_idx = min(i + hold_days, len(dates) - 1)
                        positions.append({
                            "ticker": ticker,
                            "entry_date": date,
                            "entry_price": float(price),
                            "shares": float(shares),
                            "exit_idx": exit_idx,
                        })

    # Force close remaining positions at last date
    last_date = dates[-1]
    for pos in positions:
        exit_price = close_df.loc[last_date, pos["ticker"]]
        if pd.isna(exit_price):
            exit_price = pos["entry_price"]
        pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        equity += pnl
        trades.append({
            "ticker": pos["ticker"],
            "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
            "exit_date": last_date.strftime("%Y-%m-%d"),
            "entry_price": round(float(pos["entry_price"]), 2),
            "exit_price": round(float(exit_price), 2),
            "shares": round(float(pos["shares"]), 4),
            "pnl": round(float(pnl), 2),
            "return_pct": round(float((exit_price / pos["entry_price"] - 1) * 100), 2),
        })

    return equity_curve, trades


# ── Signal Functions for Each Variant ───────────────────────────────────────
def make_variant_a(rs_df, rank_df):
    """A: RS Leaders, no filter."""
    def signals(date, idx, close_df, spy_close, vix_close, regime, tickers):
        if idx < 20:
            return []
        row_rs = rs_df.loc[date, tickers].dropna()
        row_rank = rank_df.loc[date, tickers].dropna()
        # Top 3 with positive RS
        candidates = row_rs[row_rs > 0].sort_values(ascending=False).head(3)
        return [(t, 1.0) for t in candidates.index]
    return signals


def make_variant_b(rs_df, rank_df):
    """B: RS + Bear Market Hedge (cash in bear)."""
    def signals(date, idx, close_df, spy_close, vix_close, regime, tickers):
        if idx < 20:
            return []
        if regime.loc[date] == "bear":
            return []  # 100% cash in bear
        row_rs = rs_df.loc[date, tickers].dropna()
        candidates = row_rs[row_rs > 0].sort_values(ascending=False).head(3)
        return [(t, 1.0) for t in candidates.index]
    return signals


def make_variant_c(rs_df, rank_df):
    """C: RS + Inverse VIX Sizing."""
    def signals(date, idx, close_df, spy_close, vix_close, regime, tickers):
        if idx < 20:
            return []
        row_rs = rs_df.loc[date, tickers].dropna()
        candidates = row_rs[row_rs > 0].sort_values(ascending=False).head(3)
        if len(candidates) == 0:
            return []
        # VIX sizing
        vix_val = vix_close.loc[date] if date in vix_close.index and not pd.isna(vix_close.loc[date]) else 20
        if vix_val < 15:
            size_mult = 1.0
        elif vix_val <= 25:
            size_mult = 0.5
        else:
            size_mult = 0.25
        return [(t, size_mult) for t in candidates.index]
    return signals


def make_variant_d(rs_df_sectors, rank_df_sectors):
    """D: RS on Sector ETFs, top 2, hold 20 days."""
    def signals(date, idx, close_df, spy_close, vix_close, regime, tickers):
        if idx < 20:
            return []
        row_rs = rs_df_sectors.loc[date, tickers].dropna()
        candidates = row_rs[row_rs > 0].sort_values(ascending=False).head(2)
        return [(t, 1.0) for t in candidates.index]
    return signals


def make_variant_e(rs_df, rank_df):
    """E: RS Acceleration — buy stocks whose RS rank improved this week vs last."""
    def signals(date, idx, close_df, spy_close, vix_close, regime, tickers):
        if idx < 25:
            return []
        # Current vs 5 days ago rank
        dates_list = close_df.index
        prev_idx = max(0, idx - 5)
        prev_date = dates_list[prev_idx]
        cur_rank = rank_df.loc[date, tickers].dropna()
        prev_rank = rank_df.loc[prev_date, tickers].dropna()
        common = cur_rank.index.intersection(prev_rank.index)
        if len(common) == 0:
            return []
        # Improvement = lower rank number (better)
        improvement = prev_rank[common] - cur_rank[common]
        # Also must have positive RS
        cur_rs = rs_df.loc[date, tickers].dropna()
        improving = improvement[improvement > 0]
        # Intersect with positive RS
        valid = improving.index.intersection(cur_rs[cur_rs > 0].index)
        if len(valid) == 0:
            return []
        picks = improvement[valid].sort_values(ascending=False).head(3)
        return [(t, 1.0) for t in picks.index]
    return signals


def make_variant_f(rs_df_5d, rank_df_5d, rs_df_60d, rank_df_60d):
    """F: Dual RS — both 5d and 60d RS rank in top 5."""
    def signals(date, idx, close_df, spy_close, vix_close, regime, tickers):
        if idx < 60:
            return []
        rank_5 = rank_df_5d.loc[date, tickers].dropna()
        rank_60 = rank_df_60d.loc[date, tickers].dropna()
        common = rank_5.index.intersection(rank_60.index)
        if len(common) == 0:
            return []
        # Both in top 5
        top5_short = set(rank_5[common].nsmallest(5).index)
        top5_long = set(rank_60[common].nsmallest(5).index)
        agreement = top5_short & top5_long
        if not agreement:
            return []
        # Sort by combined rank
        combined = (rank_5[list(agreement)] + rank_60[list(agreement)]).sort_values()
        picks = combined.head(3)
        return [(t, 1.0) for t in picks.index]
    return signals


# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(equity_curve, trades, regime):
    """Compute Sharpe, Sortino, PF, WR, MaxDD, regime gap."""
    eq_df = pd.DataFrame(equity_curve)
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.set_index("date")

    daily_ret = eq_df["equity"].pct_change().dropna()

    if len(daily_ret) == 0 or daily_ret.std() == 0:
        return None

    ann_factor = np.sqrt(252)
    sharpe = float(daily_ret.mean() / daily_ret.std() * ann_factor)

    downside = daily_ret[daily_ret < 0]
    sortino = float(daily_ret.mean() / downside.std() * ann_factor) if len(downside) > 0 and downside.std() > 0 else 0.0

    # PF and WR from trades
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    pf = float(gross_profit / gross_loss)
    wr = float(len(wins) / len(trades) * 100) if trades else 0.0

    # MaxDD
    peak = eq_df["equity"].expanding().max()
    dd = (eq_df["equity"] - peak) / peak
    max_dd = float(dd.min())

    # Total return
    total_return = float((eq_df["equity"].iloc[-1] / INITIAL_CAPITAL - 1) * 100)

    # Regime-stratified Sharpe
    bull_dates = regime[regime == "bull"].index
    bear_dates = regime[regime == "bear"].index

    bull_ret = daily_ret[daily_ret.index.isin(bull_dates)]
    bear_ret = daily_ret[daily_ret.index.isin(bear_dates)]

    sharpe_bull = float(bull_ret.mean() / bull_ret.std() * ann_factor) if len(bull_ret) > 5 and bull_ret.std() > 0 else 0.0
    sharpe_bear = float(bear_ret.mean() / bear_ret.std() * ann_factor) if len(bear_ret) > 5 and bear_ret.std() > 0 else 0.0

    # Regime gap
    denom = max(abs(sharpe_bull), abs(sharpe_bear), 0.001)
    regime_gap = float(abs(sharpe_bull - sharpe_bear) / denom)

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate_pct": round(wr, 1),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "total_return_pct": round(total_return, 1),
        "n_trades": len(trades),
        "final_equity": round(float(eq_df["equity"].iloc[-1]), 2),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
    }


def permutation_test(equity_curve, trades, n_perm=N_PERM):
    """Permutation test on trade returns."""
    if len(trades) < 5:
        return 1.0

    returns = np.array([t["return_pct"] for t in trades])
    actual_mean = np.mean(returns)

    count_ge = 0
    for _ in range(n_perm):
        perm = returns.copy()
        np.random.shuffle(perm)
        # Randomly flip signs to test if direction matters
        signs = np.random.choice([-1, 1], size=len(perm))
        perm_mean = np.mean(perm * signs)
        if perm_mean >= actual_mean:
            count_ge += 1

    return round(float(count_ge / n_perm), 4)


def check_gates(metrics, perm_p):
    """Check all 5 gates."""
    passed = {}
    passed["sharpe"] = metrics["sharpe"] >= GATES["sharpe_min"]
    passed["perm_p"] = perm_p <= GATES["perm_p_max"]
    passed["regime_gap"] = metrics["regime_gap"] <= GATES["regime_gap_max"]
    passed["max_dd"] = metrics["max_drawdown_pct"] >= GATES["max_dd_floor"] * 100
    passed["min_trades"] = metrics["n_trades"] >= GATES["min_trades"]
    passed["all_passed"] = all(passed.values())
    return passed


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("RELATIVE STRENGTH + REGIME FILTER BACKTEST")
    print("=" * 70)

    close = download_data()
    spy_close = close["SPY"]
    vix_close = close["^VIX"] if "^VIX" in close.columns else close.get("VIX", pd.Series(dtype=float))

    regime = compute_regime(spy_close)

    # Precompute RS for stocks (20d)
    stock_close = close[STOCKS].copy()
    rs_20d = relative_strength(stock_close, spy_close, 20)
    rank_20d = rs_rank(rs_20d)

    # RS for sectors
    sector_close = close[[c for c in SECTOR_ETFS if c in close.columns]].copy()
    rs_sectors = relative_strength(sector_close, spy_close, 20)
    rank_sectors = rs_rank(rs_sectors)

    # RS 5d and 60d for variant F
    rs_5d = relative_strength(stock_close, spy_close, 5)
    rank_5d = rs_rank(rs_5d)
    rs_60d = relative_strength(stock_close, spy_close, 60)
    rank_60d = rs_rank(rs_60d)

    # Define variants
    variants = {
        "A_RS_Leaders": {
            "signals_func": make_variant_a(rs_20d, rank_20d),
            "hold_days": 5,
            "universe": STOCKS,
            "desc": "Top-3 relative strength, hold 5 days, no filter",
        },
        "B_RS_Bear_Hedge": {
            "signals_func": make_variant_b(rs_20d, rank_20d),
            "hold_days": 5,
            "universe": STOCKS,
            "desc": "Top-3 RS, cash in bear (SPY < 200-SMA)",
        },
        "C_RS_VIX_Sizing": {
            "signals_func": make_variant_c(rs_20d, rank_20d),
            "hold_days": 5,
            "universe": STOCKS,
            "desc": "Top-3 RS, inverse VIX sizing",
        },
        "D_RS_Sector_ETF": {
            "signals_func": make_variant_d(rs_sectors, rank_sectors),
            "hold_days": 20,
            "universe": [c for c in SECTOR_ETFS if c in close.columns],
            "desc": "Top-2 sector ETFs by RS, hold 20 days",
        },
        "E_RS_Acceleration": {
            "signals_func": make_variant_e(rs_20d, rank_20d),
            "hold_days": 10,
            "universe": STOCKS,
            "desc": "Improving RS rank + positive RS, hold 10 days",
        },
        "F_Dual_RS": {
            "signals_func": make_variant_f(rs_5d, rank_5d, rs_60d, rank_60d),
            "hold_days": 15,
            "universe": STOCKS,
            "desc": "5d + 60d RS agreement in top-5, hold 15 days",
        },
    }

    results = {}

    for name, cfg in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Running Variant {name}: {cfg['desc']}")
        print(f"{'─' * 60}")

        equity_curve, trades = run_backtest(
            close, spy_close, vix_close, regime,
            cfg["signals_func"], cfg["hold_days"], cfg["universe"]
        )

        metrics = compute_metrics(equity_curve, trades, regime)
        if metrics is None:
            print(f"  [SKIP] No valid returns")
            results[name] = {"status": "SKIP", "reason": "No valid returns"}
            continue

        perm_p = permutation_test(equity_curve, trades)
        gates = check_gates(metrics, perm_p)

        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Final Equity: ${metrics['final_equity']:.2f} (return: {metrics['total_return_pct']:.1f}%)")
        print(f"  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}")
        print(f"  PF: {metrics['profit_factor']:.3f}  |  WR: {metrics['win_rate_pct']:.1f}%")
        print(f"  MaxDD: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Regime — Bull Sharpe: {metrics['sharpe_bull']:.3f}  Bear Sharpe: {metrics['sharpe_bear']:.3f}  Gap: {metrics['regime_gap']:.3f}")
        print(f"  Permutation p-value: {perm_p:.4f}")

        gate_str = " | ".join([f"{k}:{'PASS' if v else 'FAIL'}" for k, v in gates.items()])
        status = "PASS ALL GATES" if gates["all_passed"] else "FAIL"
        print(f"  Gates: {gate_str}")
        print(f"  >>> {status} <<<")

        results[name] = {
            "description": cfg["desc"],
            "hold_days": cfg["hold_days"],
            "universe_size": len(cfg["universe"]),
            "metrics": metrics,
            "perm_p_value": perm_p,
            "gates": {k: v for k, v in gates.items()},
            "status": status,
            "sample_trades": trades[:5] if trades else [],
            "n_total_trades": len(trades),
        }

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Variant':<25} {'Sharpe':>8} {'Sortino':>8} {'PF':>8} {'WR%':>6} {'MaxDD%':>8} {'RegGap':>8} {'Perm-p':>8} {'Status':>12}")
    print("-" * 100)

    for name, r in results.items():
        if r.get("status") == "SKIP":
            print(f"{name:<25} {'SKIP':>60}")
            continue
        m = r["metrics"]
        print(f"{name:<25} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['profit_factor']:>8.3f} {m['win_rate_pct']:>6.1f} {m['max_drawdown_pct']:>8.1f} {m['regime_gap']:>8.3f} {r['perm_p_value']:>8.4f} {r['status']:>12}")

    # Save results
    output_path = Path("/home/jupiter/Lvl3Quant/data/relative_strength_regime_results.json")
    output = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "initial_capital": INITIAL_CAPITAL,
            "oot_start": OOT_START,
            "commission": 0,
            "n_permutations": N_PERM,
            "stock_universe": STOCKS,
            "sector_etf_universe": SECTOR_ETFS,
            "gates": GATES,
        },
        "variants": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    np.random.seed(42)
    main()
