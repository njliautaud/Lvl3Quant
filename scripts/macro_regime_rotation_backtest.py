#!/usr/bin/env python3
"""
Macro Regime-Based Asset Rotation Backtest
==========================================
Rotates between asset classes based on macro regime signals derived from ETF price trends.

Regimes (Growth x Inflation):
  - Goldilocks: SPY up + TLT stable/up
  - Reflation:  SPY up + GLD up
  - Stagflation: SPY down + GLD up
  - Deflation:  SPY down + GLD down

6 Variants (A-F) with 5-gate validation framework.
"""

import json
import datetime
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Download data ──────────────────────────────────────────────────────────────

def download_data():
    import yfinance as yf
    tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "SLV", "HYG", "XLU", "XLY", "XLE", "XLK", "^VIX"]
    data = yf.download(tickers, start="2005-01-01", end="2026-07-29", auto_adjust=True, progress=False)
    close = data["Close"].copy()
    close.columns = [c.replace("^", "") for c in close.columns]
    close = close.ffill().dropna(how="all")
    return close


# ── Regime classification ──────────────────────────────────────────────────────

def classify_regime(close, lookback=50):
    """Classify each day into one of 4 macro regimes."""
    spy_sma = close["SPY"].rolling(lookback).mean()
    gld_sma = close["GLD"].rolling(lookback).mean()
    tlt_sma = close["TLT"].rolling(lookback).mean()

    spy_up = close["SPY"] > spy_sma
    gld_up = close["GLD"] > gld_sma

    regime = pd.Series("Unknown", index=close.index)
    # Goldilocks: growth up, inflation low (SPY up, GLD not up → or TLT up)
    regime[spy_up & ~gld_up] = "Goldilocks"
    # Reflation: growth up, inflation up
    regime[spy_up & gld_up] = "Reflation"
    # Stagflation: growth down, inflation up
    regime[~spy_up & gld_up] = "Stagflation"
    # Deflation: growth down, inflation down
    regime[~spy_up & ~gld_up] = "Deflation"
    return regime


def get_vix_fear(close, threshold=20):
    """VIX > threshold = fear."""
    return close["VIX"] > threshold


# ── Helper: trend score (momentum) ────────────────────────────────────────────

def momentum_score(close, tickers, lookback=20):
    """Return rolling momentum (return over lookback) for each ticker."""
    scores = pd.DataFrame(index=close.index)
    for t in tickers:
        if t in close.columns:
            scores[t] = close[t].pct_change(lookback)
    return scores


# ── Strategy implementations ──────────────────────────────────────────────────

def strategy_A_simple_rotation(close, regime):
    """Simple regime rotation: fixed asset per regime."""
    mapping = {
        "Goldilocks": "QQQ",
        "Reflation": "XLE",
        "Stagflation": "GLD",
        "Deflation": "TLT",
    }
    position = regime.map(mapping)
    position[position == "Unknown"] = "SPY"  # fallback
    return position


def strategy_B_momentum_within_regime(close, regime):
    """Pick top momentum asset within regime bucket."""
    buckets = {
        "Goldilocks": ["QQQ", "XLK", "SPY"],
        "Reflation": ["XLE", "XLY", "IWM"],
        "Stagflation": ["GLD", "SLV", "XLU"],
        "Deflation": ["TLT", "XLU", "GLD"],
    }
    mom = momentum_score(close, close.columns, lookback=20)
    position = pd.Series("SPY", index=close.index)
    for date in close.index:
        r = regime.loc[date]
        if r in buckets:
            candidates = buckets[r]
            valid = [c for c in candidates if c in mom.columns and not np.isnan(mom.loc[date, c])]
            if valid:
                best = max(valid, key=lambda t: mom.loc[date, t])
                position.loc[date] = best
    return position


def strategy_C_binary_risk(close):
    """Binary risk-on/off: SPY above 50-SMA → QQQ, below → TLT."""
    spy_sma50 = close["SPY"].rolling(50).mean()
    position = pd.Series("TLT", index=close.index)
    position[close["SPY"] > spy_sma50] = "QQQ"
    return position


def strategy_D_quad_vix(close, regime):
    """Quad regime + VIX fear overlay. Stagflation + VIX>20 → cash."""
    mapping = {
        "Goldilocks": "QQQ",
        "Reflation": "XLE",
        "Stagflation": "GLD",
        "Deflation": "TLT",
    }
    vix_fear = get_vix_fear(close, 20)
    position = regime.map(mapping)
    position[position == "Unknown"] = "SPY"
    # Override: stagflation + high VIX → cash
    position[(regime == "Stagflation") & vix_fear] = "CASH"
    return position


def strategy_E_monthly_rebalance(close, regime):
    """Monthly regime check + rotation."""
    mapping = {
        "Goldilocks": "QQQ",
        "Reflation": "XLE",
        "Stagflation": "GLD",
        "Deflation": "TLT",
    }
    # Only change position on first trading day of each month
    position = pd.Series("SPY", index=close.index)
    month_starts = close.index.to_series().dt.to_period("M")
    current_pos = "SPY"
    last_month = None
    for date in close.index:
        m = month_starts.loc[date]
        if m != last_month:
            r = regime.loc[date]
            current_pos = mapping.get(r, "SPY")
            last_month = m
        position.loc[date] = current_pos
    return position


def strategy_F_contrarian_transition(close, regime):
    """On regime transition, buy the NEW regime's asset."""
    mapping = {
        "Goldilocks": "QQQ",
        "Reflation": "XLE",
        "Stagflation": "GLD",
        "Deflation": "TLT",
    }
    position = pd.Series("SPY", index=close.index)
    prev_regime = regime.shift(1)
    # Only trade on regime transitions; hold until next transition
    current_pos = "SPY"
    for i, date in enumerate(close.index):
        r = regime.iloc[i]
        pr = prev_regime.iloc[i] if i > 0 else None
        if r != pr and r in mapping:
            current_pos = mapping[r]
        position.iloc[i] = current_pos
    return position


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(close, position, initial_capital=645, slippage_pct=0.0002,
                 oot_start="2022-01-01", oot_end="2026-07-29"):
    """Run backtest on a position series. Returns metrics dict."""
    # Filter to OOT period
    mask = (close.index >= pd.Timestamp(oot_start)) & (close.index <= pd.Timestamp(oot_end))
    pos_oot = position[mask].copy()
    close_oot = close[mask].copy()

    if len(pos_oot) < 10:
        return None

    # Build daily returns
    daily_returns = pd.Series(0.0, index=close_oot.index)
    trades = 0
    prev_asset = None

    for i in range(1, len(close_oot)):
        date = close_oot.index[i]
        prev_date = close_oot.index[i - 1]
        asset = pos_oot.iloc[i - 1]  # position decided at prev close, held today

        if asset == "CASH" or asset not in close_oot.columns:
            daily_returns.iloc[i] = 0.0
        else:
            ret = (close_oot[asset].iloc[i] / close_oot[asset].iloc[i - 1]) - 1
            daily_returns.iloc[i] = ret

        # Count trades (position changes)
        if asset != prev_asset:
            trades += 1
            # Apply slippage on trade day
            daily_returns.iloc[i] -= slippage_pct
        prev_asset = asset

    # Compute equity curve
    equity = initial_capital * (1 + daily_returns).cumprod()

    # Metrics
    total_return = (equity.iloc[-1] / initial_capital) - 1
    ann_return = (1 + total_return) ** (252 / len(daily_returns)) - 1 if len(daily_returns) > 0 else 0
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    downside = daily_returns[daily_returns < 0].std() * np.sqrt(252)
    sortino = ann_return / downside if downside > 0 else 0

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Win rate (daily)
    winning_days = (daily_returns > 0).sum()
    total_days = (daily_returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0

    # Profit factor
    gross_profit = daily_returns[daily_returns > 0].sum()
    gross_loss = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Regime-stratified Sharpe
    regime_full = classify_regime(close, 50)
    regime_oot = regime_full[mask]
    regime_sharpes = {}
    for r in ["Goldilocks", "Reflation", "Stagflation", "Deflation"]:
        r_mask = regime_oot == r
        r_rets = daily_returns[r_mask]
        if len(r_rets) > 20:
            r_ann = r_rets.mean() * 252
            r_vol = r_rets.std() * np.sqrt(252)
            regime_sharpes[r] = round(r_ann / r_vol, 3) if r_vol > 0 else 0
        else:
            regime_sharpes[r] = None

    # Regime gap
    valid_sharpes = [v for v in regime_sharpes.values() if v is not None]
    if len(valid_sharpes) >= 2:
        regime_gap = (max(valid_sharpes) - min(valid_sharpes)) / max(abs(max(valid_sharpes)), abs(min(valid_sharpes)), 1e-9)
    else:
        regime_gap = 0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "ann_return_pct": round(ann_return * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate": round(win_rate, 3),
        "trades": trades,
        "total_days": len(daily_returns),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "regime_sharpes": regime_sharpes,
        "regime_gap": round(regime_gap, 3),
        "equity_curve_start": str(equity.index[0].date()),
        "equity_curve_end": str(equity.index[-1].date()),
    }


# ── Permutation test ─────────────────────────────────────────────────────────

def permutation_test(close, position, n_perms=1000, initial_capital=645, slippage_pct=0.0002,
                     oot_start="2022-01-01", oot_end="2026-07-29"):
    """Shuffle regime assignments, recompute Sharpe. Return p-value."""
    mask = (close.index >= pd.Timestamp(oot_start)) & (close.index <= pd.Timestamp(oot_end))
    pos_oot = position[mask].copy()
    close_oot = close[mask].copy()

    # Actual Sharpe
    actual = run_backtest(close, position, initial_capital, slippage_pct, oot_start, oot_end)
    if actual is None:
        return 1.0
    actual_sharpe = actual["sharpe"]

    # Get unique assets used
    assets_used = [a for a in pos_oot.unique() if a != "CASH" and a in close_oot.columns]
    if not assets_used:
        return 1.0

    count_better = 0
    for _ in range(n_perms):
        # Randomly assign assets from the used set
        shuffled = pos_oot.copy()
        shuffled[:] = np.random.choice(assets_used, size=len(shuffled))
        perm_result = run_backtest(close, _rebuild_full_position(position, shuffled, mask),
                                    initial_capital, slippage_pct, oot_start, oot_end)
        if perm_result and perm_result["sharpe"] >= actual_sharpe:
            count_better += 1

    return count_better / n_perms


def _rebuild_full_position(orig, oot_override, mask):
    """Replace OOT portion of position series."""
    result = orig.copy()
    result[mask] = oot_override.values
    return result


# ── 5-Gate validation ─────────────────────────────────────────────────────────

def validate_5gates(metrics, perm_p):
    """Apply 5-gate framework."""
    gates = {
        "G1_sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "G2_perm_p_lt_0.05": perm_p < 0.05,
        "G3_regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "G4_maxdd_gt_neg50": metrics["max_dd_pct"] > -50,
        "G5_trades_gte_20": metrics["trades"] >= 20,
    }
    gates["passed_all"] = all(gates.values())
    gates["perm_p_value"] = round(perm_p, 4)
    return gates


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MACRO REGIME-BASED ASSET ROTATION BACKTEST")
    print("=" * 70)

    print("\n[1/4] Downloading ETF data via yfinance...")
    close = download_data()
    print(f"  Data shape: {close.shape}, range: {close.index[0].date()} → {close.index[-1].date()}")
    print(f"  Tickers: {list(close.columns)}")

    print("\n[2/4] Classifying macro regimes...")
    regime = classify_regime(close, lookback=50)
    oot_mask = (close.index >= pd.Timestamp("2022-01-01")) & (close.index <= pd.Timestamp("2026-07-29"))
    regime_oot = regime[oot_mask]
    print(f"  OOT regime distribution:")
    for r, cnt in regime_oot.value_counts().items():
        print(f"    {r}: {cnt} days ({cnt/len(regime_oot)*100:.1f}%)")

    print("\n[3/4] Running 6 strategy variants...")
    strategies = {
        "A_simple_rotation": strategy_A_simple_rotation(close, regime),
        "B_momentum_in_regime": strategy_B_momentum_within_regime(close, regime),
        "C_binary_risk_onoff": strategy_C_binary_risk(close),
        "D_quad_vix_overlay": strategy_D_quad_vix(close, regime),
        "E_monthly_rebalance": strategy_E_monthly_rebalance(close, regime),
        "F_contrarian_transition": strategy_F_contrarian_transition(close, regime),
    }

    results = {}
    for name, pos in strategies.items():
        print(f"\n  --- Variant {name} ---")
        metrics = run_backtest(close, pos, initial_capital=645, slippage_pct=0.0002,
                               oot_start="2022-01-01", oot_end="2026-07-29")
        if metrics is None:
            print(f"    SKIP: insufficient data")
            results[name] = {"error": "insufficient data"}
            continue

        print(f"    Return: {metrics['total_return_pct']:.1f}%  Sharpe: {metrics['sharpe']:.3f}  "
              f"Sortino: {metrics['sortino']:.3f}  PF: {metrics['profit_factor']:.2f}  "
              f"WR: {metrics['win_rate']:.1%}  MaxDD: {metrics['max_dd_pct']:.1f}%  "
              f"Trades: {metrics['trades']}  Final: ${metrics['final_equity']:.2f}")
        print(f"    Regime Sharpes: {metrics['regime_sharpes']}")
        print(f"    Regime Gap: {metrics['regime_gap']:.3f}")

        # Permutation test
        print(f"    Running permutation test (1000 iters)...", end=" ", flush=True)
        perm_p = permutation_test(close, pos, n_perms=1000, initial_capital=645,
                                   slippage_pct=0.0002, oot_start="2022-01-01", oot_end="2026-07-29")
        print(f"p={perm_p:.4f}")

        gates = validate_5gates(metrics, perm_p)
        metrics["gates"] = gates
        results[name] = metrics

        gate_str = " | ".join([f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items()
                               if k not in ("passed_all", "perm_p_value")])
        print(f"    Gates: {gate_str}")
        print(f"    ALL GATES: {'✓ PASSED' if gates['passed_all'] else '✗ FAILED'}")

    # ── SPY buy-and-hold benchmark ──
    print("\n  --- Benchmark: SPY Buy-and-Hold ---")
    spy_pos = pd.Series("SPY", index=close.index)
    spy_metrics = run_backtest(close, spy_pos, initial_capital=645, slippage_pct=0.0002,
                                oot_start="2022-01-01", oot_end="2026-07-29")
    if spy_metrics:
        print(f"    Return: {spy_metrics['total_return_pct']:.1f}%  Sharpe: {spy_metrics['sharpe']:.3f}  "
              f"MaxDD: {spy_metrics['max_dd_pct']:.1f}%  Final: ${spy_metrics['final_equity']:.2f}")
        results["BENCHMARK_SPY"] = spy_metrics

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Variant':<30} {'Sharpe':>7} {'Sortino':>8} {'Return%':>8} {'MaxDD%':>7} {'PF':>6} {'Gates':>6}")
    print("-" * 70)
    for name, m in results.items():
        if "error" in m:
            print(f"{name:<30} {'ERROR':>7}")
            continue
        passed = m.get("gates", {}).get("passed_all", "N/A")
        gate_label = "PASS" if passed is True else ("FAIL" if passed is False else "N/A")
        print(f"{name:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['total_return_pct']:>7.1f}% "
              f"{m['max_dd_pct']:>6.1f}% {m['profit_factor']:>6.2f} {gate_label:>6}")

    # ── Save results ──
    output_path = "/home/jupiter/Lvl3Quant/data/macro_regime_rotation_results.json"
    # Convert any non-serializable types
    def clean_for_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_json(i) for i in obj]
        return obj

    save_data = {
        "generated": datetime.datetime.now().isoformat(),
        "oot_period": "2022-01-01 to 2026-07-29",
        "initial_capital": 645,
        "slippage_pct": 0.0002,
        "permutation_iters": 1000,
        "variants": clean_for_json(results),
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
