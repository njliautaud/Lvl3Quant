#!/usr/bin/env python3
"""
VIX Regime Switching Backtest — Dual Signal D variants
Tests whether dynamically adjusting strategy parameters based on VIX level
improves the base Dual Signal D dip-buying strategy.

6 Variants tested across 20-stock quality universe, Jan 2022 - Jul 2026.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────

UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]

OOT_START = "2022-01-01"
OOT_END = "2026-07-31"
DATA_START = "2021-06-01"  # extra history for indicators

STARTING_CAPITAL = 645.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% each way

# VIX regime thresholds
VIX_LOW = 15
VIX_HIGH = 25

# ─── Data Download ───────────────────────────────────────────────────────────

def download_data():
    """Download price data for universe + VIX + SPY."""
    tickers = UNIVERSE + ["^VIX", "SPY"]
    print(f"Downloading {len(tickers)} tickers from {DATA_START} to {OOT_END}...")
    data = yf.download(tickers, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
    close = data["Close"]
    # Flatten columns if MultiIndex
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill()
    print(f"Downloaded {len(close)} trading days, {close.shape[1]} tickers")
    return close


def compute_indicators(close):
    """Compute RSI, consecutive red days, dip %, 200-SMA for SPY, VIX regime."""
    indicators = {}

    # SPY 200-SMA for bull/bear regime
    spy = close["SPY"]
    spy_sma200 = spy.rolling(200).mean()

    # VIX data and regime
    vix = close["^VIX"]
    vix_regime = pd.Series("Normal", index=vix.index)
    vix_regime[vix < VIX_LOW] = "Low"
    vix_regime[vix > VIX_HIGH] = "High"

    # VIX 5-day change for Variant F
    vix_5d_change = vix.pct_change(5)

    for ticker in UNIVERSE:
        if ticker not in close.columns:
            continue
        px = close[ticker]

        # RSI-14
        delta = px.diff()
        gain = delta.clip(lower=0)
        loss = (-delta.clip(upper=0))
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # Consecutive red days
        daily_ret = px.pct_change()
        red = (daily_ret < 0).astype(int)
        consec_red = red.copy() * 0
        for i in range(1, len(red)):
            if red.iloc[i] == 1:
                consec_red.iloc[i] = consec_red.iloc[i - 1] + 1
            else:
                consec_red.iloc[i] = 0

        # Dip from 20-day high
        high_20d = px.rolling(20).max()
        dip_pct = (px - high_20d) / high_20d * 100  # negative number

        indicators[ticker] = {
            "price": px,
            "rsi": rsi,
            "consec_red": consec_red,
            "dip_pct": dip_pct,
        }

    return indicators, spy, spy_sma200, vix, vix_regime, vix_5d_change


# ─── Variant Parameter Functions ─────────────────────────────────────────────

def get_params_A(vix_regime_today, vix_5d_chg):
    """Baseline: fixed parameters."""
    return {"hold": 10, "size": 200, "dip_thresh": -5.0, "entry_ok": True}


def get_params_B(vix_regime_today, vix_5d_chg):
    """Adaptive Hold."""
    hold_map = {"Low": 5, "Normal": 10, "High": 15}
    return {"hold": hold_map[vix_regime_today], "size": 200, "dip_thresh": -5.0, "entry_ok": True}


def get_params_C(vix_regime_today, vix_5d_chg):
    """Adaptive Dip Threshold."""
    dip_map = {"Low": -3.0, "Normal": -5.0, "High": -7.0}
    return {"hold": 10, "size": 200, "dip_thresh": dip_map[vix_regime_today], "entry_ok": True}


def get_params_D(vix_regime_today, vix_5d_chg):
    """Adaptive Position Size."""
    size_map = {"Low": 150, "Normal": 200, "High": 250}
    return {"hold": 10, "size": size_map[vix_regime_today], "dip_thresh": -5.0, "entry_ok": True}


def get_params_E(vix_regime_today, vix_5d_chg):
    """Full Adaptive: B+C+D combined."""
    hold_map = {"Low": 5, "Normal": 10, "High": 15}
    dip_map = {"Low": -3.0, "Normal": -5.0, "High": -7.0}
    size_map = {"Low": 150, "Normal": 200, "High": 250}
    return {
        "hold": hold_map[vix_regime_today],
        "size": size_map[vix_regime_today],
        "dip_thresh": dip_map[vix_regime_today],
        "entry_ok": True,
    }


def get_params_F(vix_regime_today, vix_5d_chg):
    """VIX Spike Entry: only enter when VIX spiked >20% in 5 days."""
    entry_ok = (vix_5d_chg > 0.20) if not np.isnan(vix_5d_chg) else False
    return {"hold": 10, "size": 200, "dip_thresh": -5.0, "entry_ok": bool(entry_ok)}


VARIANTS = {
    "A_Baseline": get_params_A,
    "B_Adaptive_Hold": get_params_B,
    "C_Adaptive_Dip": get_params_C,
    "D_Adaptive_Size": get_params_D,
    "E_Full_Adaptive": get_params_E,
    "F_VIX_Spike_Entry": get_params_F,
}


# ─── Backtest Engine ─────────────────────────────────────────────────────────

def run_backtest(variant_name, param_fn, indicators, spy, spy_sma200, vix, vix_regime, vix_5d_change, oot_dates):
    """Run a single variant backtest."""
    capital = STARTING_CAPITAL
    positions = []  # list of {ticker, entry_price, entry_date, hold_days, shares, target_hold}
    trades = []
    daily_equity = []

    for date in oot_dates:
        if date not in vix_regime.index:
            continue

        regime_today = vix_regime.loc[date]
        vix_5d = vix_5d_change.loc[date] if date in vix_5d_change.index else np.nan
        params = param_fn(regime_today, vix_5d)

        # Check for exits
        new_positions = []
        for pos in positions:
            pos["hold_days"] += 1
            ticker = pos["ticker"]
            if date in indicators[ticker]["price"].index:
                current_price = indicators[ticker]["price"].loc[date]
            else:
                new_positions.append(pos)
                continue

            if pos["hold_days"] >= pos["target_hold"]:
                # Exit with slippage
                exit_price = current_price * (1 - SLIPPAGE_PCT)
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                capital += exit_price * pos["shares"]

                # Determine if bull or bear at exit
                is_bull = spy.loc[date] > spy_sma200.loc[date] if date in spy_sma200.index else True

                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "shares": pos["shares"],
                    "pnl": round(pnl, 2),
                    "return_pct": round((exit_price / pos["entry_price"] - 1) * 100, 2),
                    "hold_days": pos["hold_days"],
                    "regime": "Bull" if is_bull else "Bear",
                    "vix_regime": regime_today,
                })
            else:
                new_positions.append(pos)
        positions = new_positions

        # Check for entries
        if params["entry_ok"] and len(positions) < MAX_CONCURRENT:
            for ticker in UNIVERSE:
                if len(positions) >= MAX_CONCURRENT:
                    break
                if ticker not in indicators:
                    continue
                if date not in indicators[ticker]["price"].index:
                    continue
                # Skip if already holding this ticker
                if any(p["ticker"] == ticker for p in positions):
                    continue

                rsi = indicators[ticker]["rsi"].loc[date]
                consec = indicators[ticker]["consec_red"].loc[date]
                dip = indicators[ticker]["dip_pct"].loc[date]

                if np.isnan(rsi) or np.isnan(dip):
                    continue

                # Dual Signal D: dip threshold, RSI < 35, 3+ consecutive red days
                if dip <= params["dip_thresh"] and rsi < 35 and consec >= 3:
                    entry_price = indicators[ticker]["price"].loc[date] * (1 + SLIPPAGE_PCT)
                    trade_size = min(params["size"], capital * 0.95)  # keep 5% buffer
                    if trade_size < 10:
                        continue
                    shares = trade_size / entry_price
                    cost = entry_price * shares
                    if cost > capital:
                        continue
                    capital -= cost
                    positions.append({
                        "ticker": ticker,
                        "entry_price": entry_price,
                        "entry_date": date,
                        "hold_days": 0,
                        "shares": shares,
                        "target_hold": params["hold"],
                    })

        # Daily equity
        pos_value = 0
        for pos in positions:
            ticker = pos["ticker"]
            if date in indicators[ticker]["price"].index:
                pos_value += indicators[ticker]["price"].loc[date] * pos["shares"]
        daily_equity.append({
            "date": date,
            "equity": capital + pos_value,
            "vix_regime": regime_today,
        })

    # Force-close any remaining positions at end
    if positions:
        last_date = oot_dates[-1]
        for pos in positions:
            ticker = pos["ticker"]
            if last_date in indicators[ticker]["price"].index:
                exit_price = indicators[ticker]["price"].loc[last_date] * (1 - SLIPPAGE_PCT)
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                capital += exit_price * pos["shares"]
                is_bull = spy.loc[last_date] > spy_sma200.loc[last_date] if last_date in spy_sma200.index else True
                regime_today = vix_regime.loc[last_date] if last_date in vix_regime.index else "Normal"
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": last_date.strftime("%Y-%m-%d"),
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "shares": pos["shares"],
                    "pnl": round(pnl, 2),
                    "return_pct": round((exit_price / pos["entry_price"] - 1) * 100, 2),
                    "hold_days": pos["hold_days"],
                    "regime": "Bull" if is_bull else "Bear",
                    "vix_regime": regime_today,
                })

    return trades, daily_equity


# ─── Analytics ────────────────────────────────────────────────────────────────

def compute_metrics(trades, daily_equity, starting_capital):
    """Compute performance metrics from trade list and equity curve."""
    if not trades:
        return {
            "sharpe": 0, "sortino": 0, "win_rate": 0, "profit_factor": 0,
            "max_drawdown_pct": 0, "total_return_pct": 0, "num_trades": 0,
        }

    eq = pd.DataFrame(daily_equity)
    eq.set_index("date", inplace=True)

    # Daily returns
    eq["returns"] = eq["equity"].pct_change().fillna(0)

    # Sharpe (annualized, 252 trading days)
    mean_ret = eq["returns"].mean()
    std_ret = eq["returns"].std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    # Sortino
    downside = eq["returns"][eq["returns"] < 0]
    downside_std = downside.std() if len(downside) > 0 else 0
    sortino = (mean_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Win rate
    pnls = [t["pnl"] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls) * 100 if pnls else 0

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)

    # Max drawdown
    eq["cum_max"] = eq["equity"].cummax()
    eq["drawdown"] = (eq["equity"] - eq["cum_max"]) / eq["cum_max"] * 100
    max_dd = eq["drawdown"].min()

    # Total return
    final_equity = eq["equity"].iloc[-1] if len(eq) > 0 else starting_capital
    total_return = (final_equity / starting_capital - 1) * 100

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "total_return_pct": round(total_return, 2),
        "num_trades": len(trades),
        "final_equity": round(final_equity, 2),
    }


def regime_analysis(trades):
    """Compute per-regime (Bull vs Bear) metrics."""
    result = {}
    for regime in ["Bull", "Bear"]:
        rt = [t for t in trades if t["regime"] == regime]
        if not rt:
            result[regime] = {"sharpe": 0, "win_rate": 0, "num_trades": 0, "avg_return_pct": 0}
            continue
        rets = [t["return_pct"] for t in rt]
        mean_r = np.mean(rets)
        std_r = np.std(rets) if len(rets) > 1 else 0.001
        # Approximate trade-level Sharpe
        sharpe = mean_r / std_r * np.sqrt(len(rets)) if std_r > 0 else 0
        wins = sum(1 for r in rets if r > 0)
        result[regime] = {
            "sharpe": round(sharpe, 3),
            "win_rate": round(wins / len(rets) * 100, 1),
            "num_trades": len(rt),
            "avg_return_pct": round(mean_r, 2),
        }
    return result


def vix_regime_analysis(trades):
    """Compute per-VIX-regime breakdown."""
    result = {}
    for vr in ["Low", "Normal", "High"]:
        rt = [t for t in trades if t.get("vix_regime") == vr]
        if not rt:
            result[vr] = {"num_trades": 0, "win_rate": 0, "avg_return_pct": 0, "total_pnl": 0}
            continue
        rets = [t["return_pct"] for t in rt]
        wins = sum(1 for r in rets if r > 0)
        result[vr] = {
            "num_trades": len(rt),
            "win_rate": round(wins / len(rets) * 100, 1),
            "avg_return_pct": round(np.mean(rets), 2),
            "total_pnl": round(sum(t["pnl"] for t in rt), 2),
        }
    return result


def permutation_test(trades, n_perm=1000):
    """Permutation test: shuffle trade returns, compute p-value for observed mean."""
    if len(trades) < 5:
        return 1.0
    rets = np.array([t["return_pct"] for t in trades])
    observed_mean = np.mean(rets)

    count_ge = 0
    for _ in range(n_perm):
        shuffled = rets * np.random.choice([-1, 1], size=len(rets))
        if np.mean(shuffled) >= observed_mean:
            count_ge += 1

    return round(count_ge / n_perm, 4)


def five_gate_validation(metrics, perm_p, regime_data):
    """Apply 5-gate validation."""
    bull_sharpe = regime_data.get("Bull", {}).get("sharpe", 0)
    bear_sharpe = regime_data.get("Bear", {}).get("sharpe", 0)
    max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_sharpe if max_sharpe > 0 else 0

    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime_gap < 0.5,
        "mdd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "trades_gte_20": metrics["num_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    gates["regime_gap"] = round(regime_gap, 3)
    return gates


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("VIX REGIME SWITCHING BACKTEST — Dual Signal D Variants")
    print("=" * 70)

    # Download data
    close = download_data()

    # Compute indicators
    indicators, spy, spy_sma200, vix, vix_regime, vix_5d_change = compute_indicators(close)

    # OOT date range
    oot_dates = close.loc[OOT_START:OOT_END].index.tolist()
    print(f"\nOOT period: {oot_dates[0].strftime('%Y-%m-%d')} to {oot_dates[-1].strftime('%Y-%m-%d')}")
    print(f"Total OOT days: {len(oot_dates)}")

    # VIX regime distribution
    vix_in_oot = vix_regime.loc[oot_dates[0]:oot_dates[-1]]
    for r in ["Low", "Normal", "High"]:
        cnt = (vix_in_oot == r).sum()
        print(f"  VIX {r}: {cnt} days ({cnt/len(vix_in_oot)*100:.1f}%)")

    results = {}

    for vname, param_fn in VARIANTS.items():
        print(f"\n{'─'*60}")
        print(f"Running Variant {vname}...")
        trades, daily_equity = run_backtest(
            vname, param_fn, indicators, spy, spy_sma200, vix, vix_regime, vix_5d_change, oot_dates
        )

        metrics = compute_metrics(trades, daily_equity, STARTING_CAPITAL)
        regimes = regime_analysis(trades)
        vix_regimes = vix_regime_analysis(trades)
        perm_p = permutation_test(trades)
        gates = five_gate_validation(metrics, perm_p, regimes)

        print(f"  Trades: {metrics['num_trades']} | Sharpe: {metrics['sharpe']} | "
              f"Sortino: {metrics['sortino']} | WR: {metrics['win_rate']}% | "
              f"PF: {metrics['profit_factor']} | MDD: {metrics['max_drawdown_pct']}% | "
              f"Return: {metrics['total_return_pct']}%")
        print(f"  Perm p-value: {perm_p} | Gates passed: {sum(v for k,v in gates.items() if k != 'all_passed' and k != 'regime_gap')}/5")

        if trades:
            print(f"  VIX regime breakdown:")
            for vr in ["Low", "Normal", "High"]:
                vrd = vix_regimes.get(vr, {})
                print(f"    {vr}: {vrd.get('num_trades',0)} trades, "
                      f"WR={vrd.get('win_rate',0)}%, "
                      f"avg_ret={vrd.get('avg_return_pct',0)}%, "
                      f"PnL=${vrd.get('total_pnl',0):.2f}")

        results[vname] = {
            "metrics": metrics,
            "regime_analysis": regimes,
            "vix_regime_breakdown": vix_regimes,
            "permutation_p_value": perm_p,
            "five_gate_validation": gates,
            "sample_trades": trades[:5] if trades else [],
        }

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Variant':<22} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'PF':>6} {'MDD%':>7} {'Ret%':>8} {'Gates':>5}")
    print("-" * 80)
    for vname, r in results.items():
        m = r["metrics"]
        g = r["five_gate_validation"]
        passed = sum(v for k, v in g.items() if k not in ("all_passed", "regime_gap"))
        print(f"{vname:<22} {m['num_trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['win_rate']:>5.1f}% {m['profit_factor']:>6.2f} {m['max_drawdown_pct']:>6.2f}% "
              f"{m['total_return_pct']:>7.2f}% {passed:>3}/5")

    # Best variant
    best = max(results.items(), key=lambda x: x[1]["metrics"]["sharpe"])
    print(f"\nBest variant by Sharpe: {best[0]} (Sharpe={best[1]['metrics']['sharpe']})")

    # Save results
    output = {
        "backtest": "VIX Regime Switching — Dual Signal D",
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "oot_period": f"{oot_dates[0].strftime('%Y-%m-%d')} to {oot_dates[-1].strftime('%Y-%m-%d')}",
        "starting_capital": STARTING_CAPITAL,
        "universe": UNIVERSE,
        "slippage_pct": SLIPPAGE_PCT,
        "vix_thresholds": {"low": VIX_LOW, "high": VIX_HIGH},
        "variants": results,
        "best_variant": best[0],
    }

    out_path = Path("/home/jupiter/Lvl3Quant/data/vix_regime_switch_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
