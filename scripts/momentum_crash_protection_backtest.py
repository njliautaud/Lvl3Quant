#!/usr/bin/env python3
"""
Momentum Crash Protection Backtest
===================================
6 variants of momentum investing with various crash-protection mechanisms.
Walk-forward OOT: Jan 2022 - Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.

Daniel & Moskowitz (2016) inspired crash protection.
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0

START_DATE = "2021-10-01"  # buffer for lookback
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"

GROWTH_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "PLTR", "SOFI", "HOOD", "UBER", "COIN",
]
SECTOR_ETFS = [
    "XLK", "XLC", "XLY", "XLF", "XLE", "XLV", "XLI", "XLP", "XLB", "XLRE", "XLU",
]
SPY = "SPY"
VIX = "^VIX"

N_PERM = 1000
RANDOM_SEED = 42

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/momentum_crash_protection_results.json")


# ── Data Download ──────────────────────────────────────────────────────────────
def download_data():
    """Download all required price data."""
    all_tickers = list(set(GROWTH_STOCKS + SECTOR_ETFS + [SPY]))
    print(f"Downloading {len(all_tickers)} tickers + VIX...")

    # Download equity/ETF data
    data = yf.download(all_tickers, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data

    # Download VIX
    vix_data = yf.download(VIX, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    vix_close = vix_data["Close"].squeeze() if isinstance(vix_data["Close"], pd.DataFrame) else vix_data["Close"]

    # Forward fill, drop weekends
    close = close.ffill().dropna(how="all")
    vix_close = vix_close.ffill()

    print(f"  Data range: {close.index[0].date()} to {close.index[-1].date()}")
    print(f"  Tickers with data: {close.shape[1]}")
    return close, vix_close


# ── Helpers ────────────────────────────────────────────────────────────────────
def monthly_rebalance_dates(idx, start, end):
    """Get last trading day of each month in range."""
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    monthly = idx[mask].to_series().groupby([idx[mask].year, idx[mask].month]).last()
    return monthly.values


def apply_slippage(price, direction="buy"):
    """Apply slippage to price."""
    if direction == "buy":
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def compute_momentum(prices_df, tickers, date, lookback_days=21):
    """Compute 1-month (21 trading day) momentum for given tickers."""
    loc = prices_df.index.get_loc(date)
    if loc < lookback_days:
        return {}
    mom = {}
    for t in tickers:
        if t not in prices_df.columns:
            continue
        cur = prices_df[t].iloc[loc]
        prev = prices_df[t].iloc[loc - lookback_days]
        if pd.notna(cur) and pd.notna(prev) and prev > 0:
            mom[t] = (cur / prev) - 1.0
    return mom


def compute_volatility(prices_df, tickers, date, lookback_days=21):
    """Compute realized volatility over lookback period."""
    loc = prices_df.index.get_loc(date)
    if loc < lookback_days:
        return {}
    vol = {}
    for t in tickers:
        if t not in prices_df.columns:
            continue
        segment = prices_df[t].iloc[max(0, loc - lookback_days):loc + 1]
        rets = segment.pct_change().dropna()
        if len(rets) > 5:
            vol[t] = rets.std() * np.sqrt(252)
    return vol


def spy_regime(prices_df, date):
    """Bull if SPY > 200-SMA, else Bear."""
    loc = prices_df.index.get_loc(date)
    if loc < 200:
        return "bull"  # not enough data, assume bull
    sma200 = prices_df[SPY].iloc[max(0, loc - 199):loc + 1].mean()
    return "bull" if prices_df[SPY].iloc[loc] > sma200 else "bear"


# ── Strategy Engines ───────────────────────────────────────────────────────────
def run_strategy(prices_df, vix_series, variant, account=ACCOUNT_SIZE):
    """
    Run a single strategy variant. Returns daily equity curve and trade log.
    """
    reb_dates = monthly_rebalance_dates(prices_df.index, OOT_START, OOT_END)
    oot_mask = (prices_df.index >= pd.Timestamp(OOT_START)) & (prices_df.index <= pd.Timestamp(OOT_END))
    oot_dates = prices_df.index[oot_mask]

    if variant in ("A", "B", "C", "D"):
        universe = SECTOR_ETFS
        top_n = 3
    else:
        universe = GROWTH_STOCKS
        top_n = 3

    equity = account
    peak_equity = equity
    cash_mode = False  # for VIX/DD stop variants
    positions = {}  # ticker -> {"shares": float, "entry_price": float}
    equity_curve = []
    trades = []
    regime_returns = {"bull": [], "bear": []}

    prev_equity = equity

    for date in oot_dates:
        day_str = str(date.date())
        regime = spy_regime(prices_df, date)

        # ── Check stop conditions ──
        if variant == "B" or variant == "F":
            # VIX stop
            if date in vix_series.index:
                vix_val = vix_series.loc[date] if not isinstance(vix_series.loc[date], pd.Series) else vix_series.loc[date].iloc[-1]
            else:
                # find nearest prior
                prior = vix_series.index[vix_series.index <= date]
                vix_val = vix_series.iloc[-1] if len(prior) == 0 else vix_series.loc[prior[-1]]
                if isinstance(vix_val, pd.Series):
                    vix_val = vix_val.iloc[-1]

            if not cash_mode and vix_val > 25:
                # liquidate
                for t, pos in list(positions.items()):
                    sell_price = apply_slippage(prices_df[t].loc[date], "sell")
                    pnl = (sell_price - pos["entry_price"]) * pos["shares"]
                    equity += pnl
                    trades.append({
                        "date": day_str, "ticker": t, "side": "sell",
                        "price": round(sell_price, 2), "pnl": round(pnl, 2),
                        "reason": f"VIX stop ({vix_val:.1f})"
                    })
                positions = {}
                cash_mode = True
            elif cash_mode and vix_val < 20:
                cash_mode = False

        if variant == "C":
            dd = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0
            if not cash_mode and dd < -0.05:
                for t, pos in list(positions.items()):
                    sell_price = apply_slippage(prices_df[t].loc[date], "sell")
                    pnl = (sell_price - pos["entry_price"]) * pos["shares"]
                    equity += pnl
                    trades.append({
                        "date": day_str, "ticker": t, "side": "sell",
                        "price": round(sell_price, 2), "pnl": round(pnl, 2),
                        "reason": "DD stop (>5%)"
                    })
                positions = {}
                cash_mode = True
            elif cash_mode and equity >= peak_equity:
                cash_mode = False

        if variant == "F":
            # Trailing stop: exit any position down >8% from entry
            for t, pos in list(positions.items()):
                cur_price = prices_df[t].loc[date]
                if pd.notna(cur_price) and cur_price < pos["entry_price"] * 0.92:
                    sell_price = apply_slippage(cur_price, "sell")
                    pnl = (sell_price - pos["entry_price"]) * pos["shares"]
                    equity += pnl
                    trades.append({
                        "date": day_str, "ticker": t, "side": "sell",
                        "price": round(sell_price, 2), "pnl": round(pnl, 2),
                        "reason": "trailing stop (-8%)"
                    })
                    del positions[t]

        # ── Rebalance on rebalance dates ──
        is_reb_day = date in reb_dates

        if is_reb_day and not cash_mode:
            mom = compute_momentum(prices_df, universe, date)
            if len(mom) < top_n:
                # not enough data, skip
                pass
            else:
                # Rank and select top N
                ranked = sorted(mom.items(), key=lambda x: x[1], reverse=True)
                new_picks = [t for t, _ in ranked[:top_n]]

                # Sell positions not in new picks
                for t in list(positions.keys()):
                    if t not in new_picks:
                        sell_price = apply_slippage(prices_df[t].loc[date], "sell")
                        pnl = (sell_price - positions[t]["entry_price"]) * positions[t]["shares"]
                        equity += pnl
                        trades.append({
                            "date": day_str, "ticker": t, "side": "sell",
                            "price": round(sell_price, 2), "pnl": round(pnl, 2),
                            "reason": "rebalance"
                        })
                        del positions[t]

                # Determine weights
                if variant == "D":
                    # Inverse volatility weighting
                    vols = compute_volatility(prices_df, new_picks, date)
                    if vols:
                        inv_vols = {t: 1.0 / max(v, 0.01) for t, v in vols.items() if t in new_picks}
                        total_inv = sum(inv_vols.values())
                        weights = {t: v / total_inv for t, v in inv_vols.items()}
                    else:
                        weights = {t: 1.0 / len(new_picks) for t in new_picks}
                else:
                    weights = {t: 1.0 / len(new_picks) for t in new_picks}

                # Buy new positions (use current equity for sizing)
                # First compute current portfolio value
                port_val = equity
                for t, pos in positions.items():
                    cur_p = prices_df[t].loc[date]
                    if pd.notna(cur_p):
                        port_val += (cur_p - pos["entry_price"]) * pos["shares"]

                for t in new_picks:
                    if t in positions:
                        continue  # already holding
                    buy_price = apply_slippage(prices_df[t].loc[date], "buy")
                    if pd.isna(buy_price) or buy_price <= 0:
                        continue
                    alloc = port_val * weights.get(t, 1.0 / len(new_picks))
                    shares = alloc / buy_price
                    if shares > 0:
                        positions[t] = {"shares": shares, "entry_price": buy_price}
                        trades.append({
                            "date": day_str, "ticker": t, "side": "buy",
                            "price": round(buy_price, 2), "shares": round(shares, 4),
                            "reason": "momentum pick"
                        })

        # ── Mark to market ──
        port_val = equity
        for t, pos in positions.items():
            cur_p = prices_df[t].loc[date]
            if pd.notna(cur_p):
                port_val += (cur_p - pos["entry_price"]) * pos["shares"]

        daily_ret = (port_val - prev_equity) / prev_equity if prev_equity > 0 else 0
        regime_returns[regime].append(daily_ret)

        peak_equity = max(peak_equity, port_val)
        equity_curve.append({"date": day_str, "equity": round(port_val, 2), "regime": regime})
        prev_equity = port_val

    # Final liquidation
    final_val = equity
    for t, pos in positions.items():
        last_date = oot_dates[-1]
        cur_p = prices_df[t].loc[last_date]
        if pd.notna(cur_p):
            final_val += (cur_p - pos["entry_price"]) * pos["shares"]

    return equity_curve, trades, regime_returns, final_val


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(equity_curve, trades, regime_returns):
    """Compute strategy performance metrics."""
    eq = pd.DataFrame(equity_curve)
    eq["equity"] = eq["equity"].astype(float)
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.set_index("date")

    daily_returns = eq["equity"].pct_change().dropna()

    if len(daily_returns) < 10:
        return None

    # Basic metrics
    total_return = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) - 1
    annual_factor = 252 / len(daily_returns)
    cagr = (1 + total_return) ** (annual_factor) - 1 if total_return > -1 else -1

    # Sharpe
    mean_ret = daily_returns.mean()
    std_ret = daily_returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-6
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    # Max drawdown
    cummax = eq["equity"].cummax()
    drawdown = (eq["equity"] - cummax) / cummax
    max_dd = drawdown.min()

    # Win rate from trades
    closed_trades = [t for t in trades if t["side"] == "sell"]
    n_trades = len(closed_trades)
    winning = [t for t in closed_trades if t.get("pnl", 0) > 0]
    win_rate = len(winning) / n_trades if n_trades > 0 else 0

    # Profit factor
    gross_profit = sum(t["pnl"] for t in closed_trades if t.get("pnl", 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in closed_trades if t.get("pnl", 0) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (99.0 if gross_profit > 0 else 0)

    # Regime analysis
    bull_rets = regime_returns.get("bull", [])
    bear_rets = regime_returns.get("bear", [])

    bull_sharpe = (np.mean(bull_rets) / np.std(bull_rets)) * np.sqrt(252) if len(bull_rets) > 20 and np.std(bull_rets) > 0 else 0
    bear_sharpe = (np.mean(bear_rets) / np.std(bear_rets)) * np.sqrt(252) if len(bear_rets) > 20 and np.std(bear_rets) > 0 else 0

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    return {
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "profit_factor": round(profit_factor, 3),
        "win_rate_pct": round(win_rate * 100, 1),
        "n_trades": n_trades,
        "final_equity": round(eq["equity"].iloc[-1], 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_days": len(bull_rets),
        "bear_days": len(bear_rets),
    }


# ── Permutation Test ───────────────────────────────────────────────────────────
def permutation_test(prices_df, vix_series, variant, actual_sharpe, n_perm=N_PERM):
    """
    Shuffle monthly momentum selections to test if strategy Sharpe is significant.
    """
    rng = np.random.RandomState(RANDOM_SEED)

    if variant in ("A", "B", "C", "D"):
        universe = SECTOR_ETFS
    else:
        universe = GROWTH_STOCKS
    top_n = 3

    reb_dates = monthly_rebalance_dates(prices_df.index, OOT_START, OOT_END)
    oot_mask = (prices_df.index >= pd.Timestamp(OOT_START)) & (prices_df.index <= pd.Timestamp(OOT_END))
    oot_dates = prices_df.index[oot_mask]

    perm_sharpes = []

    for _ in range(n_perm):
        equity = ACCOUNT_SIZE
        prev_eq = equity
        positions = {}
        daily_rets = []

        for date in oot_dates:
            is_reb = date in reb_dates

            if is_reb:
                # Sell all
                for t, pos in positions.items():
                    cur_p = prices_df[t].loc[date]
                    if pd.notna(cur_p):
                        equity += (cur_p - pos["entry_price"]) * pos["shares"]
                positions = {}

                # Random selection instead of momentum
                avail = [t for t in universe if t in prices_df.columns and pd.notna(prices_df[t].loc[date])]
                if len(avail) >= top_n:
                    picks = rng.choice(avail, size=top_n, replace=False)
                    alloc_each = equity / top_n
                    for t in picks:
                        bp = apply_slippage(prices_df[t].loc[date], "buy")
                        if bp > 0:
                            positions[t] = {"shares": alloc_each / bp, "entry_price": bp}

            # Mark to market
            port_val = equity
            for t, pos in positions.items():
                cur_p = prices_df[t].loc[date]
                if pd.notna(cur_p):
                    port_val += (cur_p - pos["entry_price"]) * pos["shares"]

            daily_ret = (port_val - prev_eq) / prev_eq if prev_eq > 0 else 0
            daily_rets.append(daily_ret)
            prev_eq = port_val

        rets = np.array(daily_rets)
        if rets.std() > 0:
            s = (rets.mean() / rets.std()) * np.sqrt(252)
        else:
            s = 0
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= actual_sharpe)
    return round(p_value, 4), round(np.mean(perm_sharpes), 3), round(np.std(perm_sharpes), 3)


# ── 5-Gate Validation ──────────────────────────────────────────────────────────
def validate_5gate(metrics, p_value):
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("MOMENTUM CRASH PROTECTION BACKTEST")
    print("Walk-Forward OOT: Jan 2022 - Jul 2026 | Account: $645")
    print("=" * 70)

    prices, vix = download_data()

    variants = {
        "A": "Simple Sector Momentum (top-3 ETFs, monthly)",
        "B": "Momentum + VIX Stop (>25 cash, <20 re-enter)",
        "C": "Momentum + Drawdown Stop (>5% DD = cash)",
        "D": "Dynamic Momentum (inverse-vol weighting)",
        "E": "Growth Stock Momentum (top-3 stocks)",
        "F": "Protected Growth (VIX stop + 8% trailing stop)",
    }

    results = {
        "strategy": "Momentum Crash Protection",
        "account_size": ACCOUNT_SIZE,
        "oot_period": f"{OOT_START} to {OOT_END}",
        "slippage_pct": SLIPPAGE_PCT,
        "commission": COMMISSION,
        "n_permutations": N_PERM,
        "timestamp": dt.datetime.now().isoformat(),
        "variants": {},
    }

    for var_key, var_name in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {var_key}: {var_name}")
        print(f"{'─' * 60}")

        eq_curve, trade_log, regime_rets, final_val = run_strategy(prices, vix, var_key)
        metrics = compute_metrics(eq_curve, trade_log, regime_rets)

        if metrics is None:
            print("  !! Insufficient data for metrics")
            results["variants"][var_key] = {"name": var_name, "error": "insufficient data"}
            continue

        print(f"  Final Equity: ${metrics['final_equity']:.2f}  |  Return: {metrics['total_return_pct']:.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}  |  MaxDD: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Win Rate: {metrics['win_rate_pct']:.1f}%  |  PF: {metrics['profit_factor']:.2f}  |  Trades: {metrics['n_trades']}")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']:.3f}  |  Bear Sharpe: {metrics['bear_sharpe']:.3f}  |  Regime Gap: {metrics['regime_gap']:.3f}")

        # Permutation test
        print(f"  Running {N_PERM} permutations...")
        p_val, perm_mean, perm_std = permutation_test(prices, vix, var_key, metrics["sharpe"])
        print(f"  Perm p-value: {p_val}  |  Perm Sharpe mean: {perm_mean} +/- {perm_std}")

        # 5-gate
        gates = validate_5gate(metrics, p_val)
        gate_status = "PASS" if gates["all_passed"] else "FAIL"
        failed = [k for k, v in gates.items() if not v and k != "all_passed"]
        print(f"  5-Gate: {gate_status}" + (f"  (failed: {', '.join(failed)})" if failed else ""))

        results["variants"][var_key] = {
            "name": var_name,
            "metrics": metrics,
            "permutation": {"p_value": p_val, "perm_sharpe_mean": perm_mean, "perm_sharpe_std": perm_std},
            "five_gate": gates,
            "gate_result": gate_status,
            "sample_trades": trade_log[:10],
            "equity_curve_endpoints": {
                "start": eq_curve[0] if eq_curve else None,
                "end": eq_curve[-1] if eq_curve else None,
            }
        }

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Variant':<10} {'Name':<45} {'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'Gate':>6}")
    print(f"{'─' * 83}")
    for var_key in variants:
        v = results["variants"][var_key]
        if "error" in v:
            print(f"  {var_key:<8} {v['name']:<45} {'ERROR':>7}")
            continue
        m = v["metrics"]
        print(f"  {var_key:<8} {v['name']:<45} {m['sharpe']:>7.3f} {m['total_return_pct']:>7.1f}% {m['max_drawdown_pct']:>6.1f}% {v['gate_result']:>6}")

    # Best variant
    passing = [(k, v) for k, v in results["variants"].items()
               if "metrics" in v and v["five_gate"].get("all_passed")]
    if passing:
        best_key, best = max(passing, key=lambda x: x[1]["metrics"]["sharpe"])
        results["recommendation"] = {
            "best_variant": best_key,
            "name": best["name"],
            "sharpe": best["metrics"]["sharpe"],
            "reason": "Highest Sharpe among 5-gate passing variants"
        }
        print(f"\n  RECOMMENDED: Variant {best_key} — {best['name']}")
        print(f"    Sharpe {best['metrics']['sharpe']:.3f} | Sortino {best['metrics']['sortino']:.3f} | "
              f"Return {best['metrics']['total_return_pct']:.1f}% | MaxDD {best['metrics']['max_drawdown_pct']:.1f}%")
    else:
        # Pick best even if no pass
        all_with_metrics = [(k, v) for k, v in results["variants"].items() if "metrics" in v]
        if all_with_metrics:
            best_key, best = max(all_with_metrics, key=lambda x: x[1]["metrics"]["sharpe"])
            results["recommendation"] = {
                "best_variant": best_key,
                "name": best["name"],
                "sharpe": best["metrics"]["sharpe"],
                "reason": "Highest Sharpe (none passed all 5 gates)"
            }
            print(f"\n  NO VARIANT PASSED ALL 5 GATES.")
            print(f"  Best available: Variant {best_key} — {best['name']}")
            print(f"    Sharpe {best['metrics']['sharpe']:.3f} | MaxDD {best['metrics']['max_drawdown_pct']:.1f}%")
            failed_gates = [k for k, v in best["five_gate"].items() if not v and k != "all_passed"]
            print(f"    Failed gates: {', '.join(failed_gates)}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
