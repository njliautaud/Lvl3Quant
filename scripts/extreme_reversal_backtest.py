#!/usr/bin/env python3
"""
Extreme Reversal After Big Moves — Growth Stock Backtest
=========================================================
Tests 6 variants of buying extreme drops with quality/trend filters.
Walk-forward OOT: Jan 2022 – Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "CRM", "NFLX",
    "SHOP", "DDOG", "SNOW", "UBER", "COIN", "PLTR", "SQ", "ROKU", "SNAP", "PINS",
    "NET", "CRWD", "ZS", "PANW", "MDB"
]
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% per trade (each side)
START_DATE = "2020-01-01"
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
MAX_POSITIONS = 3
PERM_ITERS = 1000
np.random.seed(42)


# ── Data Download ───────────────────────────────────────────────────────────
def download_data():
    """Download daily OHLCV for universe + SPY."""
    tickers = UNIVERSE + ["SPY"]
    print(f"Downloading {len(tickers)} tickers...")
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=OOT_END, progress=False, auto_adjust=True)
            if len(df) > 200:
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                data[ticker] = df
                print(f"  {ticker}: {len(df)} bars")
            else:
                print(f"  {ticker}: SKIPPED (only {len(df)} bars)")
        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")
    return data


# ── Feature Computation ────────────────────────────────────────────────────
def compute_features(data):
    """Add SMA, RSI, volume features to all dataframes."""
    for ticker, df in data.items():
        df["ret"] = df["Close"].pct_change()
        df["sma200"] = df["Close"].rolling(200).mean()
        df["sma50"] = df["Close"].rolling(50).mean()
        df["vol_ma20"] = df["Volume"].rolling(20).mean()
        df["vol_ratio"] = df["Volume"] / df["vol_ma20"]

        # RSI(14)
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi14"] = 100 - (100 / (1 + rs))

        # Multi-day returns
        df["ret_5d"] = df["Close"].pct_change(5)

        # Intraday range
        df["intraday_drop"] = (df["Low"] - df["Open"]) / df["Open"]
        df["intraday_recovery"] = (df["Close"] - df["Open"]) / df["Open"]

        # Consecutive red days
        df["red_day"] = (df["Close"] < df["Close"].shift(1)).astype(int)
        df["consec_red"] = 0
        consec = 0
        red_vals = df["red_day"].values
        consec_arr = np.zeros(len(df), dtype=int)
        for i in range(len(df)):
            if red_vals[i] == 1:
                consec += 1
            else:
                consec = 0
            consec_arr[i] = consec
        df["consec_red"] = consec_arr

        # Green day (close > open)
        df["green_day"] = df["Close"] > df["Open"]

        data[ticker] = df
    return data


# ── Signal Generators ───────────────────────────────────────────────────────
def signal_A_big_drop_trend(df):
    """Buy when stock drops >5% in a day AND above 200-SMA."""
    cond = (df["ret"] < -0.05) & (df["Close"] > df["sma200"])
    return cond


def signal_B_multiday_crash_recovery(df):
    """Buy when stock drops >10% over 5 days, then shows green day, above 200-SMA."""
    cond = (
        (df["ret_5d"] < -0.10) &
        df["green_day"] &
        (df["Close"] > df["sma200"])
    )
    return cond


def signal_C_volume_capitulation(df):
    """Buy when stock drops >3% on volume > 2x 20-day avg AND above 200-SMA."""
    cond = (
        (df["ret"] < -0.03) &
        (df["vol_ratio"] > 2.0) &
        (df["Close"] > df["sma200"])
    )
    return cond


def signal_D_post_earnings_drop(df):
    """Buy when stock drops >5% on >3x normal volume (earnings proxy) AND above 200-SMA."""
    cond = (
        (df["ret"] < -0.05) &
        (df["vol_ratio"] > 3.0) &
        (df["Close"] > df["sma200"])
    )
    return cond


def signal_E_consecutive_red(df):
    """Buy after 5 consecutive red days IF above 200-SMA AND RSI(14) < 40."""
    cond = (
        (df["consec_red"] >= 5) &
        (df["Close"] > df["sma200"]) &
        (df["rsi14"] < 40)
    )
    return cond


def signal_F_intraday_recovery(df):
    """Buy when stock drops >5% intraday but closes within 2% of open (hammer). Above 50-SMA."""
    cond = (
        (df["intraday_drop"] < -0.05) &
        (df["intraday_recovery"].abs() < 0.02) &
        (df["Close"] > df["sma50"])
    )
    return cond


VARIANTS = {
    "A_BigDrop_Trend": {"signal_fn": signal_A_big_drop_trend, "hold_days": 10, "profit_target": 0.05, "max_pos": 3},
    "B_MultiDay_Crash": {"signal_fn": signal_B_multiday_crash_recovery, "hold_days": 10, "profit_target": None, "max_pos": 3},
    "C_Vol_Capitulation": {"signal_fn": signal_C_volume_capitulation, "hold_days": 10, "profit_target": None, "max_pos": 3},
    "D_PostEarnings_Drop": {"signal_fn": signal_D_post_earnings_drop, "hold_days": 20, "profit_target": None, "max_pos": 3},
    "E_Consec_Red": {"signal_fn": signal_E_consecutive_red, "hold_days": 10, "profit_target": None, "max_pos": 3},
    "F_Intraday_Recovery": {"signal_fn": signal_F_intraday_recovery, "hold_days": 10, "profit_target": None, "max_pos": 3},
}


# ── Backtest Engine ────────────────────────────────────────────────────────
def run_backtest(data, spy_df, variant_name, variant_config, oot_start=OOT_START, oot_end=OOT_END):
    """
    Run a single variant backtest.
    Returns: trades list, equity curve, stats dict
    """
    signal_fn = variant_config["signal_fn"]
    hold_days = variant_config["hold_days"]
    profit_target = variant_config["profit_target"]
    max_pos = variant_config["max_pos"]

    # Collect all signal dates across universe
    all_signals = []
    all_drop_days = []  # For permutation test: all days meeting the DROP criterion (without trend filter)

    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = data[ticker]
        mask = (df.index >= pd.Timestamp(oot_start)) & (df.index <= pd.Timestamp(oot_end))
        df_oot = df[mask]
        if len(df_oot) == 0:
            continue

        signals = signal_fn(df_oot)
        signal_dates = df_oot.index[signals]
        for dt in signal_dates:
            all_signals.append((dt, ticker))

        # Drop days for permutation: days with the price drop component only (no trend filter)
        # This ensures we shuffle among legitimate drop days
        if "BigDrop" in variant_name or "PostEarnings" in variant_name:
            drop_mask = df_oot["ret"] < -0.05
        elif "MultiDay" in variant_name:
            drop_mask = df_oot["ret_5d"] < -0.10
        elif "Vol_Capitulation" in variant_name:
            drop_mask = (df_oot["ret"] < -0.03) & (df_oot["vol_ratio"] > 2.0)
        elif "Consec_Red" in variant_name:
            drop_mask = df_oot["consec_red"] >= 5
        elif "Intraday" in variant_name:
            drop_mask = df_oot["intraday_drop"] < -0.05
        else:
            drop_mask = df_oot["ret"] < -0.03

        drop_dates = df_oot.index[drop_mask]
        for dt in drop_dates:
            all_drop_days.append((dt, ticker))

    # Sort signals by date
    all_signals.sort(key=lambda x: x[0])

    # Simulate trades
    trades = []
    positions = []  # list of (ticker, entry_date, entry_price, exit_date_limit)
    equity = ACCOUNT_SIZE
    equity_curve = [(pd.Timestamp(oot_start), equity)]

    # Build date index
    spy_oot = spy_df[(spy_df.index >= pd.Timestamp(oot_start)) & (spy_df.index <= pd.Timestamp(oot_end))]
    trading_days = spy_oot.index.tolist()

    for day_idx, today in enumerate(trading_days):
        # Check exits for existing positions
        new_positions = []
        for pos in positions:
            ticker, entry_date, entry_price, exit_day_limit = pos
            df = data[ticker]
            if today not in df.index:
                new_positions.append(pos)
                continue

            days_held = (today - entry_date).days
            current_price = df.loc[today, "Close"]
            ret = (current_price / entry_price) - 1.0

            # Exit conditions
            exit_now = False
            if days_held >= exit_day_limit:
                exit_now = True
            elif profit_target is not None and ret >= profit_target:
                exit_now = True

            if exit_now:
                # Apply slippage
                net_ret = ret - 2 * SLIPPAGE_PCT  # entry + exit slippage
                pos_size = ACCOUNT_SIZE / max_pos  # equal weight
                pnl = pos_size * net_ret
                equity += pnl

                # Determine regime
                spy_close = spy_df.loc[entry_date, "Close"] if entry_date in spy_df.index else np.nan
                spy_sma200 = spy_df.loc[entry_date, "sma200"] if entry_date in spy_df.index else np.nan
                regime = "bull" if (not np.isnan(spy_close) and not np.isnan(spy_sma200) and spy_close > spy_sma200) else "bear"

                trades.append({
                    "ticker": ticker,
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(today.date()),
                    "entry_price": round(float(entry_price), 2),
                    "exit_price": round(float(current_price), 2),
                    "return_pct": round(float(ret * 100), 2),
                    "net_return_pct": round(float(net_ret * 100), 2),
                    "pnl": round(float(pnl), 2),
                    "days_held": int(days_held),
                    "regime": regime,
                })
                equity_curve.append((today, equity))
            else:
                new_positions.append(pos)
        positions = new_positions

        # Check for new entries
        if len(positions) < max_pos:
            todays_signals = [(dt, tk) for dt, tk in all_signals if dt == today]
            for _, ticker in todays_signals:
                if len(positions) >= max_pos:
                    break
                # Don't double up on same ticker
                if any(p[0] == ticker for p in positions):
                    continue
                df = data[ticker]
                if today not in df.index:
                    continue
                entry_price = df.loc[today, "Close"]  # buy at close
                positions.append((ticker, today, entry_price, hold_days))

    # Force close remaining positions on last day
    last_day = trading_days[-1] if trading_days else pd.Timestamp(oot_end)
    for pos in positions:
        ticker, entry_date, entry_price, _ = pos
        df = data[ticker]
        if last_day in df.index:
            current_price = df.loc[last_day, "Close"]
        else:
            # find last available price
            avail = df[df.index <= last_day]
            if len(avail) == 0:
                continue
            current_price = avail["Close"].iloc[-1]
            last_day = avail.index[-1]

        ret = (current_price / entry_price) - 1.0
        net_ret = ret - 2 * SLIPPAGE_PCT
        pos_size = ACCOUNT_SIZE / max_pos
        pnl = pos_size * net_ret
        equity += pnl

        spy_close = spy_df.loc[entry_date, "Close"] if entry_date in spy_df.index else np.nan
        spy_sma200 = spy_df.loc[entry_date, "sma200"] if entry_date in spy_df.index else np.nan
        regime = "bull" if (not np.isnan(spy_close) and not np.isnan(spy_sma200) and spy_close > spy_sma200) else "bear"

        trades.append({
            "ticker": ticker,
            "entry_date": str(entry_date.date()),
            "exit_date": str(last_day.date() if hasattr(last_day, 'date') else last_day),
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(current_price), 2),
            "return_pct": round(float(ret * 100), 2),
            "net_return_pct": round(float(net_ret * 100), 2),
            "pnl": round(float(pnl), 2),
            "days_held": int((last_day - entry_date).days),
            "regime": regime,
        })
        equity_curve.append((last_day, equity))

    return trades, equity_curve, all_drop_days


def compute_stats(trades, equity_curve):
    """Compute performance statistics from trades."""
    if len(trades) == 0:
        return {
            "n_trades": 0, "sharpe": 0, "sortino": 0, "win_rate": 0,
            "profit_factor": 0, "total_return_pct": 0, "max_dd_pct": 0,
            "avg_return_pct": 0, "avg_days_held": 0,
        }

    returns = np.array([t["net_return_pct"] / 100 for t in trades])
    n = len(returns)
    win_rate = np.mean(returns > 0)
    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-6

    # Annualize: assume ~25 trades/year as scaling (or use actual)
    years = max(0.5, (pd.Timestamp(OOT_END) - pd.Timestamp(OOT_START)).days / 365.25)
    trades_per_year = n / years
    annual_ret = avg_ret * trades_per_year
    annual_vol = std_ret * np.sqrt(trades_per_year)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_vol = np.std(downside, ddof=1) * np.sqrt(trades_per_year) if len(downside) > 1 else 1e-6
    sortino = annual_ret / downside_vol if downside_vol > 0 else 0

    # Profit factor
    gross_win = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown from equity curve
    eq_vals = [e[1] for e in equity_curve]
    peak = eq_vals[0]
    max_dd = 0
    for v in eq_vals:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    total_return = (eq_vals[-1] / eq_vals[0] - 1) * 100 if eq_vals[0] > 0 else 0

    return {
        "n_trades": n,
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "win_rate": round(float(win_rate * 100), 1),
        "profit_factor": round(float(profit_factor), 2),
        "total_return_pct": round(float(total_return), 2),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "avg_return_pct": round(float(avg_ret * 100), 2),
        "avg_days_held": round(float(np.mean([t["days_held"] for t in trades])), 1),
        "final_equity": round(float(eq_vals[-1]), 2),
        "trades_per_year": round(float(trades_per_year), 1),
    }


def regime_analysis(trades):
    """Split stats by bull/bear regime."""
    bull_trades = [t for t in trades if t["regime"] == "bull"]
    bear_trades = [t for t in trades if t["regime"] == "bear"]

    def regime_sharpe(trade_list):
        if len(trade_list) < 3:
            return 0.0
        rets = np.array([t["net_return_pct"] / 100 for t in trade_list])
        years = max(0.5, (pd.Timestamp(OOT_END) - pd.Timestamp(OOT_START)).days / 365.25)
        tpy = len(rets) / years
        avg = np.mean(rets)
        std = np.std(rets, ddof=1)
        ann_ret = avg * tpy
        ann_vol = std * np.sqrt(tpy) if std > 0 else 1e-6
        return ann_ret / ann_vol

    sharpe_bull = regime_sharpe(bull_trades)
    sharpe_bear = regime_sharpe(bear_trades)
    max_abs = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs if max_abs > 0 else 0

    return {
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
        "sharpe_bull": round(float(sharpe_bull), 3),
        "sharpe_bear": round(float(sharpe_bear), 3),
        "regime_gap": round(float(regime_gap), 3),
    }


def permutation_test(trades, all_drop_days, data, spy_df, variant_config, n_iters=PERM_ITERS):
    """
    Shuffle which drop days get traded (not random calendar days).
    This tests whether the specific entry criteria add value vs randomly trading any drop day.
    """
    if len(trades) < 5 or len(all_drop_days) < len(trades):
        return {"p_value": 1.0, "actual_sharpe": 0, "perm_mean_sharpe": 0}

    actual_stats = compute_stats(trades, [(pd.Timestamp(OOT_START), ACCOUNT_SIZE)])
    actual_sharpe = actual_stats["sharpe"]
    n_trades_actual = len(trades)

    hold_days = variant_config["hold_days"]
    profit_target = variant_config["profit_target"]
    max_pos = variant_config["max_pos"]

    perm_sharpes = []

    for _ in range(n_iters):
        # Randomly sample same number of trades from all drop days
        if len(all_drop_days) < n_trades_actual:
            sampled = all_drop_days[:]
        else:
            indices = np.random.choice(len(all_drop_days), size=n_trades_actual, replace=False)
            sampled = [all_drop_days[i] for i in indices]

        perm_returns = []
        for dt, ticker in sampled:
            if ticker not in data:
                continue
            df = data[ticker]
            if dt not in df.index:
                continue
            entry_price = df.loc[dt, "Close"]
            # Find exit
            future = df[df.index > dt].head(hold_days)
            if len(future) == 0:
                continue

            if profit_target is not None:
                # Check for early exit on profit target
                exited = False
                for exit_dt in future.index:
                    ret = (df.loc[exit_dt, "Close"] / entry_price) - 1.0
                    if ret >= profit_target:
                        net_ret = ret - 2 * SLIPPAGE_PCT
                        perm_returns.append(net_ret)
                        exited = True
                        break
                if not exited:
                    last = future.index[-1]
                    ret = (df.loc[last, "Close"] / entry_price) - 1.0
                    net_ret = ret - 2 * SLIPPAGE_PCT
                    perm_returns.append(net_ret)
            else:
                last = future.index[-1]
                ret = (df.loc[last, "Close"] / entry_price) - 1.0
                net_ret = ret - 2 * SLIPPAGE_PCT
                perm_returns.append(net_ret)

        if len(perm_returns) < 3:
            continue

        rets = np.array(perm_returns)
        years = max(0.5, (pd.Timestamp(OOT_END) - pd.Timestamp(OOT_START)).days / 365.25)
        tpy = len(rets) / years
        avg = np.mean(rets)
        std = np.std(rets, ddof=1)
        ann_ret = avg * tpy
        ann_vol = std * np.sqrt(tpy) if std > 0 else 1e-6
        perm_sharpe = ann_ret / ann_vol
        perm_sharpes.append(perm_sharpe)

    if len(perm_sharpes) == 0:
        return {"p_value": 1.0, "actual_sharpe": actual_sharpe, "perm_mean_sharpe": 0}

    p_value = np.mean(np.array(perm_sharpes) >= actual_sharpe)
    return {
        "p_value": round(float(p_value), 4),
        "actual_sharpe": round(float(actual_sharpe), 3),
        "perm_mean_sharpe": round(float(np.mean(perm_sharpes)), 3),
        "perm_std_sharpe": round(float(np.std(perm_sharpes)), 3),
    }


# ── 5-Gate Validation ──────────────────────────────────────────────────────
def validate_5_gates(stats, regime, perm):
    """Apply 5-gate validation."""
    gates = {}
    gates["1_sharpe_gt_0.5"] = {"pass": stats["sharpe"] > 0.5, "value": stats["sharpe"]}
    gates["2_perm_p_lt_0.05"] = {"pass": perm["p_value"] < 0.05, "value": perm["p_value"]}
    gates["3_regime_gap_lt_0.5"] = {"pass": regime["regime_gap"] < 0.5, "value": regime["regime_gap"]}
    gates["4_maxdd_gt_neg50"] = {"pass": stats["max_dd_pct"] > -50.0, "value": stats["max_dd_pct"]}
    gates["5_min_20_trades"] = {"pass": stats["n_trades"] >= 20, "value": stats["n_trades"]}
    gates["gates_passed"] = sum(1 for g in gates.values() if isinstance(g, dict) and g.get("pass", False))
    gates["all_passed"] = gates["gates_passed"] == 5
    return gates


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("EXTREME REVERSAL AFTER BIG MOVES — GROWTH STOCK BACKTEST")
    print("=" * 80)
    print(f"Universe: {len(UNIVERSE)} growth stocks")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Account: ${ACCOUNT_SIZE}")
    print()

    # Download data
    data = download_data()
    spy_df = data.pop("SPY")
    compute_features({"SPY": spy_df})
    spy_df = spy_df  # re-assign after feature computation
    data = compute_features(data)

    print(f"\nLoaded {len(data)} stocks successfully.\n")

    # Run all variants
    results = {}
    for vname, vconfig in VARIANTS.items():
        print(f"\n{'─' * 60}")
        print(f"  Variant: {vname}")
        print(f"{'─' * 60}")

        trades, equity_curve, all_drop_days = run_backtest(data, spy_df, vname, vconfig)
        stats = compute_stats(trades, equity_curve)
        regime = regime_analysis(trades)

        print(f"  Trades: {stats['n_trades']}, WR: {stats['win_rate']}%, "
              f"Sharpe: {stats['sharpe']}, Sortino: {stats['sortino']}")
        print(f"  Total Return: {stats['total_return_pct']}%, MaxDD: {stats['max_dd_pct']}%")
        print(f"  Regime: Bull={regime['bull_trades']} (Sharpe {regime['sharpe_bull']}), "
              f"Bear={regime['bear_trades']} (Sharpe {regime['sharpe_bear']}), Gap={regime['regime_gap']}")

        # Permutation test
        print(f"  Running permutation test ({PERM_ITERS} iterations)...")
        perm = permutation_test(trades, all_drop_days, data, spy_df, vconfig)
        print(f"  Perm p-value: {perm['p_value']} (actual Sharpe {perm['actual_sharpe']} vs perm mean {perm['perm_mean_sharpe']})")

        # 5-gate validation
        gates = validate_5_gates(stats, regime, perm)
        passed = gates["gates_passed"]
        print(f"  Gates: {passed}/5 passed {'*** VALIDATED ***' if gates['all_passed'] else ''}")

        results[vname] = {
            "stats": stats,
            "regime": regime,
            "permutation": perm,
            "gates": gates,
            "trades": trades,
        }

    # ── Summary Table ───────────────────────────────────────────────────────
    print("\n\n" + "=" * 120)
    print("SUMMARY TABLE — ALL VARIANTS")
    print("=" * 120)
    header = f"{'Variant':<25} {'Trades':>6} {'WR%':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'TotRet%':>8} {'MaxDD%':>8} {'PermP':>7} {'RGap':>6} {'Gates':>6}"
    print(header)
    print("-" * 120)

    for vname, res in results.items():
        s = res["stats"]
        r = res["regime"]
        p = res["permutation"]
        g = res["gates"]
        status = "PASS" if g["all_passed"] else f"{g['gates_passed']}/5"
        print(f"{vname:<25} {s['n_trades']:>6} {s['win_rate']:>5.1f}% {s['sharpe']:>7.3f} {s['sortino']:>8.3f} "
              f"{s['profit_factor']:>6.2f} {s['total_return_pct']:>7.2f}% {s['max_dd_pct']:>7.2f}% "
              f"{p['p_value']:>7.4f} {r['regime_gap']:>6.3f} {status:>6}")

    # ── Gate Detail ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("5-GATE VALIDATION DETAIL")
    print("=" * 80)
    for vname, res in results.items():
        g = res["gates"]
        print(f"\n{vname}:")
        for gname, gval in g.items():
            if isinstance(gval, dict):
                status = "PASS" if gval["pass"] else "FAIL"
                print(f"  {gname}: {status} (value={gval['value']})")
        gp = g["gates_passed"]
        print(f"  => {'ALL GATES PASSED' if g['all_passed'] else f'{gp}/5 gates passed'}")

    # ── Best Variant Analysis ───────────────────────────────────────────────
    best_name = max(results.keys(), key=lambda k: results[k]["stats"]["sharpe"])
    best = results[best_name]
    print(f"\n\n{'=' * 80}")
    print(f"BEST VARIANT: {best_name}")
    print(f"{'=' * 80}")
    print(f"  Sharpe: {best['stats']['sharpe']}")
    print(f"  Sortino: {best['stats']['sortino']}")
    print(f"  Win Rate: {best['stats']['win_rate']}%")
    print(f"  Profit Factor: {best['stats']['profit_factor']}")
    print(f"  Total Return: {best['stats']['total_return_pct']}%")
    print(f"  Max Drawdown: {best['stats']['max_dd_pct']}%")
    print(f"  Trades: {best['stats']['n_trades']}")
    print(f"  Avg Days Held: {best['stats']['avg_days_held']}")
    print(f"  Permutation p: {best['permutation']['p_value']}")
    print(f"  Regime Gap: {best['regime']['regime_gap']}")
    print(f"  Gates: {best['gates']['gates_passed']}/5")

    if best["stats"]["n_trades"] > 0:
        print(f"\n  Top 5 trades:")
        sorted_trades = sorted(best["trades"], key=lambda t: t["net_return_pct"], reverse=True)[:5]
        for t in sorted_trades:
            print(f"    {t['ticker']} {t['entry_date']}: +{t['net_return_pct']:.1f}% ({t['days_held']}d, {t['regime']})")

        print(f"\n  Bottom 5 trades:")
        worst_trades = sorted(best["trades"], key=lambda t: t["net_return_pct"])[:5]
        for t in worst_trades:
            print(f"    {t['ticker']} {t['entry_date']}: {t['net_return_pct']:.1f}% ({t['days_held']}d, {t['regime']})")

    # ── Comparison with RSI B baseline ──────────────────────────────────────
    print(f"\n\n{'=' * 80}")
    print("COMPARISON WITH RSI B BASELINE (Sharpe 1.63, 5/6 adversarial gates)")
    print("=" * 80)
    for vname, res in results.items():
        s = res["stats"]
        comparison = "BETTER" if s["sharpe"] > 1.63 else "WORSE" if s["sharpe"] < 1.63 else "EQUAL"
        complementary = "YES" if (s["sharpe"] > 0.5 and res["gates"]["gates_passed"] >= 4) else "NO"
        print(f"  {vname}: Sharpe {s['sharpe']} ({comparison} than RSI B), Complementary: {complementary}")

    # ── Save Results ────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "strategy": "Extreme Reversal After Big Moves",
            "universe": UNIVERSE,
            "oot_period": f"{OOT_START} to {OOT_END}",
            "account_size": ACCOUNT_SIZE,
            "slippage_pct": SLIPPAGE_PCT,
            "run_date": str(datetime.now()),
            "n_permutations": PERM_ITERS,
        },
        "variants": {},
    }
    for vname, res in results.items():
        output["variants"][vname] = {
            "stats": res["stats"],
            "regime": res["regime"],
            "permutation": res["permutation"],
            "gates": res["gates"],
            "n_trades": res["stats"]["n_trades"],
            "trades": res["trades"],
        }

    output_path = Path("/home/jupiter/Lvl3Quant/data/extreme_reversal_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
