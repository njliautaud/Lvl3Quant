#!/usr/bin/env python3
"""
Commodity-Equity Divergence Backtest
====================================
Exploits divergences between commodity and equity prices as mean-reversion signals.
When commodities and equities disconnect, trade the reversion or the trend beneficiary.

6 Variants:
  A) Oil-SPY Divergence     B) Gold-QQQ Divergence
  C) Commodity Index MR     D) Silver-Gold Ratio
  E) Agriculture Momentum   F) Multi-Commodity Score

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation p < 0.05 (1000 iter)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades

Regime: Bull = SPY > 200-SMA, Bear = SPY < 200-SMA
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TICKERS = ["DBC", "USO", "GLD", "SLV", "DBA", "SPY", "QQQ", "XLE", "XLB"]
START_DATE = "2006-01-01"
END_DATE = "2026-07-30"
OOT_START = "2022-01-01"
INITIAL_CAPITAL = 645.0
SLIPPAGE_BPS = 0.0002  # 0.02% each way
N_PERMUTATIONS = 1000
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/commodity_equity_divergence_results.json")


# ─── DATA DOWNLOAD ───────────────────────────────────────────────────────────
def download_data():
    print("Downloading data...")
    data = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[t] = df["Close"].copy()
                print(f"  {t}: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
            else:
                print(f"  {t}: insufficient data ({len(df)} rows)")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")
    prices = pd.DataFrame(data).ffill().dropna(how="all")
    print(f"Combined: {len(prices)} rows, {prices.columns.tolist()}\n")
    return prices


# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────
def backtest_strategy(prices, signal_func, oot_start=OOT_START,
                      initial_capital=INITIAL_CAPITAL, slippage=SLIPPAGE_BPS):
    """
    Generic event-driven backtest.
    signal_func(prices, i) -> list of (ticker, hold_days, direction) or empty.
    direction: +1 = long, -1 = short.
    """
    oot_mask = prices.index >= pd.Timestamp(oot_start)
    oot_prices = prices[oot_mask].copy()

    if len(oot_prices) < 50:
        return None

    capital = initial_capital
    equity_curve = [capital]
    dates = [oot_prices.index[0]]
    trades = []
    positions = {}  # ticker -> {entry_price, entry_date, exit_date, direction, notional}
    daily_returns = []
    daily_dates = []

    for i in range(1, len(oot_prices)):
        today = oot_prices.index[i]

        # Close expired positions
        closed = []
        for tk, pos in list(positions.items()):
            if today >= pos["exit_date"]:
                exit_price = oot_prices[tk].iloc[i]
                entry_price = pos["entry_price"]
                gross_ret = (exit_price / entry_price - 1) * pos["direction"]
                net_ret = gross_ret - 2 * slippage
                pnl = pos["notional"] * net_ret
                capital += pos["notional"] + pnl
                trades.append({
                    "ticker": tk,
                    "entry_date": str(pos["entry_date"].date()),
                    "exit_date": str(today.date()),
                    "direction": pos["direction"],
                    "gross_ret": gross_ret,
                    "net_ret": net_ret,
                    "pnl": pnl,
                })
                closed.append(tk)
        for tk in closed:
            del positions[tk]

        # Check for new signals
        full_idx = prices.index.get_loc(today)
        signals = signal_func(prices, full_idx)

        for sig in signals:
            tk, hold_days, direction = sig
            if tk in positions or capital <= 0:
                continue
            if tk not in oot_prices.columns:
                continue
            entry_price = oot_prices[tk].iloc[i]
            if pd.isna(entry_price) or entry_price <= 0:
                continue

            # Calculate exit date
            future_dates = oot_prices.index[i:]
            exit_idx = min(hold_days, len(future_dates) - 1)
            exit_date = future_dates[exit_idx]

            notional = capital  # full position
            positions[tk] = {
                "entry_price": entry_price,
                "entry_date": today,
                "exit_date": exit_date,
                "direction": direction,
                "notional": notional,
            }
            capital -= notional

        # Mark-to-market
        total = capital
        for tk, pos in positions.items():
            current = oot_prices[tk].iloc[i]
            if pd.isna(current):
                current = pos["entry_price"]
            mtm_ret = (current / pos["entry_price"] - 1) * pos["direction"]
            total += pos["notional"] * (1 + mtm_ret)

        daily_ret = (total / equity_curve[-1] - 1) if equity_curve[-1] > 0 else 0
        daily_returns.append(daily_ret)
        daily_dates.append(today)
        equity_curve.append(total)
        dates.append(today)

    # Force-close remaining
    for tk, pos in positions.items():
        exit_price = oot_prices[tk].iloc[-1]
        gross_ret = (exit_price / pos["entry_price"] - 1) * pos["direction"]
        net_ret = gross_ret - 2 * slippage
        pnl = pos["notional"] * net_ret
        trades.append({
            "ticker": tk,
            "entry_date": str(pos["entry_date"].date()),
            "exit_date": str(oot_prices.index[-1].date()),
            "direction": pos["direction"],
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "pnl": pnl,
        })

    return {
        "equity_curve": equity_curve,
        "dates": dates,
        "trades": trades,
        "daily_returns": np.array(daily_returns),
        "daily_dates": daily_dates,
    }


# ─── METRICS ─────────────────────────────────────────────────────────────────
def compute_metrics(result, prices):
    if result is None or len(result["trades"]) == 0:
        return None

    trades = result["trades"]
    rets = np.array([t["net_ret"] for t in trades])
    eq = np.array(result["equity_curve"])
    daily_rets = result["daily_returns"]

    n_trades = len(trades)
    win_rate = np.mean(rets > 0)
    total_ret = eq[-1] / eq[0] - 1

    # Sharpe (annualized from daily returns)
    if len(daily_rets) > 1 and np.std(daily_rets) > 0:
        sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino
    downside = daily_rets[daily_rets < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = np.mean(daily_rets) / np.std(downside) * np.sqrt(252)
    else:
        sortino = sharpe

    # Profit Factor
    gross_wins = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_losses = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Max Drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # CAGR
    n_years = len(daily_rets) / 252
    if n_years > 0 and eq[-1] > 0:
        cagr = (eq[-1] / eq[0]) ** (1 / n_years) - 1
    else:
        cagr = 0

    return {
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "total_return": round(total_ret, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "profit_factor": round(pf, 4),
        "max_drawdown": round(max_dd, 4),
        "cagr": round(cagr, 4),
        "final_equity": round(eq[-1], 2),
    }


def compute_regime_sharpes(result, prices):
    """Compute Sharpe in Bull (SPY > 200-SMA) and Bear (SPY < 200-SMA) regimes."""
    if result is None or len(result["daily_returns"]) < 20:
        return 0, 0, 1.0

    spy = prices["SPY"].copy()
    spy_sma200 = spy.rolling(200).mean()

    daily_dates = result["daily_dates"]
    daily_rets = result["daily_returns"]

    bull_rets = []
    bear_rets = []

    for dt, ret in zip(daily_dates, daily_rets):
        if dt in spy_sma200.index:
            sma_val = spy_sma200.loc[dt]
            spy_val = spy.loc[dt]
            if pd.notna(sma_val) and pd.notna(spy_val):
                if spy_val > sma_val:
                    bull_rets.append(ret)
                else:
                    bear_rets.append(ret)

    bull_rets = np.array(bull_rets)
    bear_rets = np.array(bear_rets)

    sharpe_bull = (np.mean(bull_rets) / np.std(bull_rets) * np.sqrt(252)) if len(bull_rets) > 5 and np.std(bull_rets) > 0 else 0
    sharpe_bear = (np.mean(bear_rets) / np.std(bear_rets) * np.sqrt(252)) if len(bear_rets) > 5 and np.std(bear_rets) > 0 else 0

    denom = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / denom if denom > 0 else 0

    return round(sharpe_bull, 4), round(sharpe_bear, 4), round(regime_gap, 4)


def compute_qqq_correlation(result, prices):
    """Correlation of strategy daily returns vs QQQ daily returns (when in position)."""
    if result is None or len(result["daily_returns"]) < 20:
        return 0

    qqq_rets = prices["QQQ"].pct_change()
    daily_dates = result["daily_dates"]
    daily_rets = result["daily_returns"]

    strat_list = []
    qqq_list = []

    for dt, ret in zip(daily_dates, daily_rets):
        if dt in qqq_rets.index and pd.notna(qqq_rets.loc[dt]):
            strat_list.append(ret)
            qqq_list.append(qqq_rets.loc[dt])

    if len(strat_list) < 20:
        return 0

    corr = np.corrcoef(strat_list, qqq_list)[0, 1]
    return round(corr, 4) if not np.isnan(corr) else 0


def permutation_test(result, prices, signal_func, n_perms=N_PERMUTATIONS):
    """Shuffle entry dates to get permutation p-value."""
    if result is None or len(result["trades"]) < 5:
        return 1.0

    actual_sharpe = compute_metrics(result, prices)["sharpe"]
    trade_dates = [t["entry_date"] for t in result["trades"]]

    oot_dates = prices.index[prices.index >= pd.Timestamp(OOT_START)]
    n_beats = 0

    for _ in range(n_perms):
        # Shuffle: pick random entry dates, same number of trades
        rand_indices = np.random.choice(len(oot_dates) - 25, size=len(trade_dates), replace=True)
        shuffled_rets = []

        for idx in rand_indices:
            entry_date = oot_dates[idx]
            exit_idx = min(idx + 20, len(oot_dates) - 1)
            exit_date = oot_dates[exit_idx]

            # Pick a random tradeable ticker from the actual trades
            t = result["trades"][np.random.randint(len(result["trades"]))]
            tk = t["ticker"]
            direction = t["direction"]

            if tk in prices.columns:
                entry_p = prices[tk].loc[entry_date] if entry_date in prices[tk].index else np.nan
                exit_p = prices[tk].loc[exit_date] if exit_date in prices[tk].index else np.nan
                if pd.notna(entry_p) and pd.notna(exit_p) and entry_p > 0:
                    ret = (exit_p / entry_p - 1) * direction - 2 * SLIPPAGE_BPS
                    shuffled_rets.append(ret)

        if len(shuffled_rets) > 1 and np.std(shuffled_rets) > 0:
            perm_sharpe = np.mean(shuffled_rets) / np.std(shuffled_rets) * np.sqrt(252 / 20)
            if perm_sharpe >= actual_sharpe:
                n_beats += 1

    return round(n_beats / n_perms, 4)


# ─── SIGNAL FUNCTIONS ────────────────────────────────────────────────────────

def signal_a_oil_spy_divergence(prices, i):
    """A) Oil-SPY Divergence: USO 20d ret >10% AND SPY 20d ret <0 -> buy XLE."""
    if i < 20:
        return []
    uso_ret = prices["USO"].iloc[i] / prices["USO"].iloc[i - 20] - 1
    spy_ret = prices["SPY"].iloc[i] / prices["SPY"].iloc[i - 20] - 1
    if uso_ret > 0.10 and spy_ret < 0:
        return [("XLE", 20, 1)]
    return []


def signal_b_gold_qqq_divergence(prices, i):
    """B) Gold-QQQ Divergence: GLD 20d ret >5% AND QQQ 20d ret <-5% -> buy GLD."""
    if i < 20:
        return []
    gld_ret = prices["GLD"].iloc[i] / prices["GLD"].iloc[i - 20] - 1
    qqq_ret = prices["QQQ"].iloc[i] / prices["QQQ"].iloc[i - 20] - 1
    if gld_ret > 0.05 and qqq_ret < -0.05:
        return [("GLD", 20, 1)]
    return []


def signal_c_commodity_index_mr(prices, i):
    """C) Commodity Index Mean Reversion: DBC/SPY ratio at 20d extreme -> trade reversion."""
    if i < 20:
        return []
    ratio = prices["DBC"].iloc[i - 20:i + 1] / prices["SPY"].iloc[i - 20:i + 1]
    current = ratio.iloc[-1]
    p90 = ratio.quantile(0.9)
    p10 = ratio.quantile(0.1)
    if current >= p90:
        # DBC rich vs SPY -> buy SPY (lagging)
        return [("SPY", 15, 1)]
    elif current <= p10:
        # DBC cheap vs SPY -> buy DBC (lagging)
        return [("DBC", 15, 1)]
    return []


def signal_d_silver_gold_ratio(prices, i):
    """D) Silver-Gold Ratio: SLV/GLD ratio extremes -> trade reversion."""
    if i < 20:
        return []
    ratio = prices["SLV"].iloc[i - 20:i + 1] / prices["GLD"].iloc[i - 20:i + 1]
    current = ratio.iloc[-1]
    p90 = ratio.quantile(0.9)
    p10 = ratio.quantile(0.1)
    if current <= p10:
        # Silver undervalued -> buy SLV
        return [("SLV", 20, 1)]
    elif current >= p90:
        # Silver overvalued -> sell SLV, buy GLD
        return [("GLD", 20, 1)]
    return []


def signal_e_agriculture_momentum(prices, i):
    """E) Agriculture Momentum: DBA positive 20d AND 5d momentum -> buy DBA."""
    if i < 20:
        return []
    mom_20 = prices["DBA"].iloc[i] / prices["DBA"].iloc[i - 20] - 1
    mom_5 = prices["DBA"].iloc[i] / prices["DBA"].iloc[i - 5] - 1
    if mom_20 > 0 and mom_5 > 0:
        return [("DBA", 20, 1)]
    return []


def signal_f_multi_commodity_score(prices, i):
    """F) Multi-Commodity Score: Score USO/GLD/DBA on 20d momentum. 2/3 pos -> buy DBC. 2/3 neg + SPY pos -> buy SPY."""
    if i < 20:
        return []
    uso_mom = prices["USO"].iloc[i] / prices["USO"].iloc[i - 20] - 1
    gld_mom = prices["GLD"].iloc[i] / prices["GLD"].iloc[i - 20] - 1
    dba_mom = prices["DBA"].iloc[i] / prices["DBA"].iloc[i - 20] - 1
    spy_mom = prices["SPY"].iloc[i] / prices["SPY"].iloc[i - 20] - 1

    pos_count = sum(1 for m in [uso_mom, gld_mom, dba_mom] if m > 0)
    neg_count = sum(1 for m in [uso_mom, gld_mom, dba_mom] if m < 0)

    if pos_count >= 2:
        return [("DBC", 20, 1)]
    elif neg_count >= 2 and spy_mom > 0:
        return [("SPY", 20, 1)]
    return []


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    prices = download_data()

    variants = {
        "A_Oil_SPY_Divergence": signal_a_oil_spy_divergence,
        "B_Gold_QQQ_Divergence": signal_b_gold_qqq_divergence,
        "C_Commodity_Index_MR": signal_c_commodity_index_mr,
        "D_Silver_Gold_Ratio": signal_d_silver_gold_ratio,
        "E_Agriculture_Momentum": signal_e_agriculture_momentum,
        "F_Multi_Commodity_Score": signal_f_multi_commodity_score,
    }

    all_results = {}

    for name, signal_func in variants.items():
        print(f"{'='*60}")
        print(f"Running {name}...")
        result = backtest_strategy(prices, signal_func)
        metrics = compute_metrics(result, prices)

        if metrics is None:
            print(f"  NO TRADES or insufficient data\n")
            all_results[name] = {"status": "NO_TRADES"}
            continue

        sharpe_bull, sharpe_bear, regime_gap = compute_regime_sharpes(result, prices)
        qqq_corr = compute_qqq_correlation(result, prices)

        print(f"  Trades: {metrics['n_trades']}, WR: {metrics['win_rate']:.1%}")
        print(f"  Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}, PF: {metrics['profit_factor']:.2f}")
        print(f"  MaxDD: {metrics['max_drawdown']:.1%}, CAGR: {metrics['cagr']:.1%}")
        print(f"  Final Equity: ${metrics['final_equity']:.2f}")
        print(f"  Regime: Bull Sharpe={sharpe_bull:.2f}, Bear Sharpe={sharpe_bear:.2f}, Gap={regime_gap:.2f}")
        print(f"  QQQ Correlation: {qqq_corr:.3f}")

        # Permutation test
        print(f"  Running permutation test ({N_PERMUTATIONS} iterations)...")
        perm_p = permutation_test(result, prices, signal_func)
        print(f"  Permutation p-value: {perm_p:.4f}")

        # 5-gate validation
        gate_1 = metrics["sharpe"] > 0.5
        gate_2 = perm_p < 0.05
        gate_3 = regime_gap < 0.5
        gate_4 = metrics["max_drawdown"] > -0.50
        gate_5 = metrics["n_trades"] >= 20

        passed = all([gate_1, gate_2, gate_3, gate_4, gate_5])
        gates = {
            "sharpe_gt_0.5": gate_1,
            "perm_p_lt_0.05": gate_2,
            "regime_gap_lt_0.5": gate_3,
            "maxdd_gt_neg50pct": gate_4,
            "min_20_trades": gate_5,
        }

        status = "PASS" if passed else "FAIL"
        gate_str = " | ".join(f"{'OK' if v else 'FAIL'}" for v in gates.values())
        print(f"  Gates: [{gate_str}] -> {status}\n")

        all_results[name] = {
            "status": status,
            "metrics": metrics,
            "sharpe_bull": sharpe_bull,
            "sharpe_bear": sharpe_bear,
            "regime_gap": regime_gap,
            "qqq_correlation": qqq_corr,
            "perm_p_value": perm_p,
            "gates": gates,
            "n_bull_days": None,  # filled below
            "n_bear_days": None,
            "sample_trades": [t for t in result["trades"][:5]],
        }

        # Count regime days
        spy_sma = prices["SPY"].rolling(200).mean()
        oot_dates = [d for d in result["daily_dates"]]
        bull_days = sum(1 for d in oot_dates if d in spy_sma.index and pd.notna(spy_sma.loc[d]) and prices["SPY"].loc[d] > spy_sma.loc[d])
        bear_days = len(oot_dates) - bull_days
        all_results[name]["n_bull_days"] = bull_days
        all_results[name]["n_bear_days"] = bear_days

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<30} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'QQQ_r':>7} {'Status':>7}")
    print("-" * 82)

    passing = []
    for name, r in all_results.items():
        if r["status"] in ("PASS", "FAIL"):
            m = r["metrics"]
            print(f"{name:<30} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {m['max_drawdown']:>6.1%} {r['qqq_correlation']:>7.3f} {r['status']:>7}")
            if r["status"] == "PASS":
                passing.append(name)
        else:
            print(f"{name:<30} {'--':>7} {'--':>8} {'--':>6} {'--':>6} {'--':>7} {'--':>7} NO_TRADE")

    print(f"\nPassing strategies: {len(passing)}/{len(variants)}")
    if passing:
        print(f"  Winners: {', '.join(passing)}")
        # Find lowest QQQ correlation among winners
        best_uncorr = min(passing, key=lambda n: abs(all_results[n]["qqq_correlation"]))
        print(f"  Most uncorrelated to QQQ: {best_uncorr} (r={all_results[best_uncorr]['qqq_correlation']:.3f})")

    # Save results
    output = {
        "metadata": {
            "strategy_family": "commodity_equity_divergence",
            "run_date": datetime.now().isoformat(),
            "oot_period": f"{OOT_START} to {END_DATE}",
            "initial_capital": INITIAL_CAPITAL,
            "slippage_bps": SLIPPAGE_BPS * 10000,
            "n_permutations": N_PERMUTATIONS,
            "regime_definition": "Bull=SPY>200SMA, Bear=SPY<200SMA",
        },
        "variants": {},
    }

    for name, r in all_results.items():
        entry = dict(r)
        if "sample_trades" in entry:
            for t in entry["sample_trades"]:
                for k in t:
                    if isinstance(t[k], (np.floating, np.integer)):
                        t[k] = float(t[k])
        output["variants"][name] = entry

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    output = convert(output)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
