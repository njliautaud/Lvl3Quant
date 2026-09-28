#!/usr/bin/env python3
"""
RSI Extreme Bounce Backtest
===========================
Tests contrarian mean-reversion at index level using extreme RSI readings.
6 variants tested with 5-gate validation framework.

Starting capital: $645
OOT period: 2022-01-01 to 2026-07-30
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
START_CAPITAL = 645.0
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
SLIPPAGE_PCT = 0.0002  # 0.02% per trade (applied on entry and exit)
PERMUTATION_ITERS = 1000
SMA_PERIOD = 200  # for regime classification

# Download buffer for RSI / SMA warm-up
DOWNLOAD_START = "2020-06-01"


def download_data():
    """Download QQQ, SPY, TQQQ, ^VIX with buffer for indicator warm-up."""
    tickers = {"QQQ": "QQQ", "SPY": "SPY", "TQQQ": "TQQQ", "VIX": "^VIX"}
    data = {}
    for name, ticker in tickers.items():
        df = yf.download(ticker, start=DOWNLOAD_START, end=OOT_END, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        data[name] = df
    return data


def compute_rsi(series, period):
    """Standard RSI calculation."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_indicators(data):
    """Add RSI and regime indicators to price data."""
    qqq = data["QQQ"].copy()
    spy = data["SPY"].copy()
    vix = data["VIX"].copy()

    qqq["RSI_14"] = compute_rsi(qqq["Close"], 14)
    qqq["RSI_2"] = compute_rsi(qqq["Close"], 2)
    spy["SMA_200"] = spy["Close"].rolling(SMA_PERIOD).mean()
    spy["regime_bull"] = (spy["Close"] > spy["SMA_200"]).astype(int)

    # Align VIX to QQQ index
    vix_close = vix["Close"].reindex(qqq.index, method="ffill")

    combined = pd.DataFrame(index=qqq.index)
    combined["qqq_close"] = qqq["Close"]
    combined["qqq_open"] = qqq["Open"]
    combined["qqq_ret"] = qqq["Close"].pct_change()
    combined["spy_close"] = spy["Close"].reindex(qqq.index, method="ffill")
    combined["spy_ret"] = spy["Close"].reindex(qqq.index, method="ffill").pct_change()
    combined["tqqq_close"] = data["TQQQ"]["Close"].reindex(qqq.index, method="ffill")
    combined["tqqq_open"] = data["TQQQ"]["Open"].reindex(qqq.index, method="ffill")
    combined["tqqq_ret"] = data["TQQQ"]["Close"].reindex(qqq.index, method="ffill").pct_change()
    combined["vix"] = vix_close
    combined["rsi_14"] = qqq["RSI_14"]
    combined["rsi_2"] = qqq["RSI_2"]
    combined["spy_sma200"] = spy["SMA_200"].reindex(qqq.index, method="ffill")
    combined["regime_bull"] = spy["regime_bull"].reindex(qqq.index, method="ffill")

    return combined


def generate_signals(df, variant):
    """Generate entry signals for each variant. Returns boolean Series (signal on day T)."""
    if variant == "A":
        return df["rsi_14"] < 25
    elif variant == "B":
        return df["rsi_14"] < 20
    elif variant == "C":
        return df["rsi_2"] < 10
    elif variant == "D":
        return (df["rsi_14"] < 25) & (df["vix"] > 25)
    elif variant == "E":
        return df["rsi_14"] < 25
    elif variant == "F":
        return df["rsi_14"] < 25
    else:
        raise ValueError(f"Unknown variant: {variant}")


def get_variant_config(variant):
    """Return (instrument, hold_days, description, leverage_type)."""
    configs = {
        "A": ("QQQ", 5,  "QQQ RSI(14)<25, hold 5d", "long"),
        "B": ("QQQ", 10, "QQQ RSI(14)<20, hold 10d", "long"),
        "C": ("QQQ", 3,  "QQQ RSI(2)<10, hold 3d (Connors)", "long"),
        "D": ("QQQ", 10, "QQQ RSI(14)<25 + VIX>25, hold 10d", "long"),
        "E": ("TQQQ", 5, "TQQQ RSI(14)<25 on QQQ, hold 5d (3x)", "long"),
        "F": ("QQQ", 10, "QQQ RSI(14)<25, options proxy 3x/20% cap, hold 10d", "options_proxy"),
    }
    return configs[variant]


def run_backtest(df, variant, start_capital=START_CAPITAL):
    """
    Run backtest for a single variant.
    Signal on day T → buy at day T+1 open (shifted to avoid look-ahead).
    Hold for N trading days, sell at day T+1+N close.
    """
    instrument, hold_days, desc, lev_type = get_variant_config(variant)

    # Filter to OOT period
    oot = df.loc[OOT_START:OOT_END].copy()

    # Generate raw signals and shift by 1 to avoid look-ahead
    raw_signals = generate_signals(df, variant)
    signals = raw_signals.shift(1).reindex(oot.index).fillna(False).astype(bool)

    # Determine price columns
    if instrument == "TQQQ":
        entry_price_col = "tqqq_open"
        exit_price_col = "tqqq_close"  # actually we'll use close of exit day
        ret_col = "tqqq_ret"
    else:
        entry_price_col = "qqq_open"
        exit_price_col = "qqq_close"
        ret_col = "qqq_ret"

    # Build trades
    trades = []
    dates = oot.index.tolist()
    i = 0
    while i < len(dates):
        if signals.iloc[i]:
            entry_date = dates[i]
            entry_price = oot.loc[entry_date, entry_price_col]

            # Exit after hold_days trading days
            exit_idx = min(i + hold_days, len(dates) - 1)
            exit_date = dates[exit_idx]
            exit_price = oot.loc[exit_date, exit_price_col]

            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
                i += 1
                continue

            # Raw return
            raw_ret = (exit_price / entry_price) - 1

            # Apply slippage (entry + exit)
            net_ret = raw_ret - 2 * SLIPPAGE_PCT

            # Options proxy: 3x leverage with 20% max loss cap
            if lev_type == "options_proxy":
                leveraged_ret = net_ret * 3.0
                leveraged_ret = max(leveraged_ret, -0.20)  # 20% max loss cap
                net_ret = leveraged_ret

            regime = oot.loc[entry_date, "regime_bull"]
            bear_2022 = entry_date.year == 2022

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "raw_ret": raw_ret,
                "net_ret": net_ret,
                "regime_bull": regime,
                "bear_2022": bear_2022,
            })

            # Skip ahead past hold period (no overlapping trades)
            i = exit_idx + 1
        else:
            i += 1

    # Build daily equity curve
    equity = start_capital
    equity_curve = []
    daily_returns = []
    in_trade = False
    trade_idx = 0

    for date in dates:
        if trade_idx < len(trades) and date == trades[trade_idx]["entry_date"]:
            in_trade = True

        if in_trade and trade_idx < len(trades):
            t = trades[trade_idx]
            if date == t["exit_date"]:
                equity *= (1 + t["net_ret"])
                in_trade = False
                trade_idx += 1
                daily_returns.append(t["net_ret"] / max(1, (dates.index(t["exit_date"]) - dates.index(t["entry_date"]))))
            elif date > t["entry_date"] and date < t["exit_date"]:
                # Approximate intra-trade daily return
                daily_returns.append(0)  # simplified; actual PnL accrues at exit
            else:
                daily_returns.append(0)
        else:
            daily_returns.append(0)

        equity_curve.append(equity)

    # More accurate daily returns: distribute trade return across hold period
    daily_rets = pd.Series(0.0, index=oot.index)
    for t in trades:
        entry_idx = dates.index(t["entry_date"])
        exit_idx = dates.index(t["exit_date"])
        n_days = exit_idx - entry_idx
        if n_days > 0:
            # Use daily returns of the instrument during hold
            if instrument == "TQQQ":
                hold_rets = oot.iloc[entry_idx+1:exit_idx+1]["tqqq_ret"]
            else:
                hold_rets = oot.iloc[entry_idx+1:exit_idx+1]["qqq_ret"]

            if lev_type == "options_proxy":
                hold_rets = hold_rets * 3.0
                # Apply daily max loss tracking (simplified)

            for d in hold_rets.index:
                daily_rets.loc[d] = hold_rets.loc[d] if not pd.isna(hold_rets.loc[d]) else 0

    # Rebuild equity curve from daily returns
    equity_series = (1 + daily_rets).cumprod() * start_capital

    # For options proxy, cap cumulative loss at 20% per trade
    if lev_type == "options_proxy":
        equity_val = start_capital
        eq_list = [equity_val]
        for t in trades:
            entry_idx = dates.index(t["entry_date"])
            exit_idx = dates.index(t["exit_date"])
            trade_start_eq = equity_val
            for j in range(entry_idx + 1, exit_idx + 1):
                if j < len(dates):
                    r = oot.iloc[j]["qqq_ret"] * 3.0 if not pd.isna(oot.iloc[j]["qqq_ret"]) else 0
                    equity_val *= (1 + r)
                    # Cap loss at 20% of trade entry
                    if equity_val < trade_start_eq * 0.80:
                        equity_val = trade_start_eq * 0.80
                        break
        # Recalculate final equity from trades
        equity_val = start_capital
        for t in trades:
            equity_val *= (1 + t["net_ret"])

    # Final equity from trades (most accurate)
    final_equity = start_capital
    for t in trades:
        final_equity *= (1 + t["net_ret"])

    return trades, daily_rets, final_equity, equity_series


def compute_metrics(trades, daily_rets, final_equity, start_capital, qqq_daily_rets, variant):
    """Compute all required metrics for a variant."""
    instrument, hold_days, desc, lev_type = get_variant_config(variant)

    n_trades = len(trades)
    if n_trades == 0:
        return {"variant": variant, "description": desc, "n_trades": 0, "error": "No trades"}

    trade_rets = [t["net_ret"] for t in trades]
    wins = [r for r in trade_rets if r > 0]
    losses = [r for r in trade_rets if r <= 0]

    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")

    total_return = (final_equity / start_capital) - 1

    # Time period for annualization
    n_days = len(daily_rets)
    n_years = n_days / 252

    cagr = (final_equity / start_capital) ** (1 / n_years) - 1 if n_years > 0 else 0
    annual_vol = daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
    sharpe = cagr / annual_vol if annual_vol > 0 else 0

    # Sortino
    downside = daily_rets[daily_rets < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    sortino = cagr / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    equity_curve = (1 + daily_rets).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Correlation with QQQ buy-and-hold
    aligned_qqq = qqq_daily_rets.reindex(daily_rets.index).fillna(0)
    if daily_rets.std() > 0 and aligned_qqq.std() > 0:
        corr = daily_rets.corr(aligned_qqq)
    else:
        corr = 0

    # Bear 2022 performance
    bear_trades = [t for t in trades if t["bear_2022"]]
    bear_ret = 0
    if bear_trades:
        bear_eq = 1.0
        for t in bear_trades:
            bear_eq *= (1 + t["net_ret"])
        bear_ret = bear_eq - 1

    # Regime-stratified Sharpe
    bull_trades = [t for t in trades if t["regime_bull"] == 1]
    bear_regime_trades = [t for t in trades if t["regime_bull"] == 0]

    def regime_sharpe(regime_trades):
        if len(regime_trades) < 2:
            return 0
        rets = [t["net_ret"] for t in regime_trades]
        mean_r = np.mean(rets)
        std_r = np.std(rets, ddof=1)
        if std_r == 0:
            return 0
        # Annualize: assume average hold is hold_days
        ann_factor = 252 / hold_days
        return (mean_r * ann_factor) / (std_r * np.sqrt(ann_factor))

    sharpe_bull = regime_sharpe(bull_trades)
    sharpe_bear = regime_sharpe(bear_regime_trades)

    max_abs = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs if max_abs > 0 else 0

    return {
        "variant": variant,
        "description": desc,
        "n_trades": n_trades,
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "annual_vol_pct": round(annual_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_win_pct": round(avg_win * 100, 2),
        "avg_loss_pct": round(avg_loss * 100, 2),
        "corr_qqq_bnh": round(corr, 3),
        "bear_2022_return_pct": round(bear_ret * 100, 2),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "n_bull_trades": len(bull_trades),
        "n_bear_trades": len(bear_regime_trades),
        "final_equity": round(final_equity, 2),
    }


def permutation_test(df, variant, observed_sharpe, n_iter=PERMUTATION_ITERS):
    """
    Permutation test: randomize entry dates, compute Sharpe each time.
    p-value = fraction of random Sharpes >= observed Sharpe.
    """
    instrument, hold_days, desc, lev_type = get_variant_config(variant)
    oot = df.loc[OOT_START:OOT_END].copy()
    dates = oot.index.tolist()

    # Count actual signals to know how many random entries to generate
    raw_signals = generate_signals(df, variant)
    signals = raw_signals.shift(1).reindex(oot.index).fillna(False).astype(bool)
    n_signals = signals.sum()

    if n_signals == 0:
        return 1.0

    random_sharpes = []
    rng = np.random.RandomState(42)

    for _ in range(n_iter):
        # Random entry dates (same count as real signals)
        random_entries = sorted(rng.choice(len(dates) - hold_days - 1, size=min(n_signals, len(dates) // (hold_days + 1)), replace=False))

        # Remove overlapping
        filtered = [random_entries[0]]
        for idx in random_entries[1:]:
            if idx > filtered[-1] + hold_days:
                filtered.append(idx)

        if len(filtered) == 0:
            random_sharpes.append(0)
            continue

        trade_rets = []
        for idx in filtered:
            entry_date = dates[idx]
            exit_idx = min(idx + hold_days, len(dates) - 1)
            exit_date = dates[exit_idx]

            if instrument == "TQQQ":
                ep = oot.loc[entry_date, "tqqq_open"]
                xp = oot.loc[exit_date, "tqqq_close"]
            else:
                ep = oot.loc[entry_date, "qqq_open"]
                xp = oot.loc[exit_date, "qqq_close"]

            if pd.isna(ep) or pd.isna(xp) or ep <= 0:
                continue

            ret = (xp / ep) - 1 - 2 * SLIPPAGE_PCT
            if lev_type == "options_proxy":
                ret = max(ret * 3.0, -0.20)
            trade_rets.append(ret)

        if len(trade_rets) < 2:
            random_sharpes.append(0)
            continue

        mean_r = np.mean(trade_rets)
        std_r = np.std(trade_rets, ddof=1)
        if std_r > 0:
            ann_factor = 252 / hold_days
            s = (mean_r * ann_factor) / (std_r * np.sqrt(ann_factor))
            random_sharpes.append(s)
        else:
            random_sharpes.append(0)

    p_value = np.mean([1 if s >= observed_sharpe else 0 for s in random_sharpes])
    return p_value


def validate_gates(metrics, p_value):
    """Apply 5-gate validation framework."""
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates["G1_sharpe_gt_0.5"] = metrics.get("sharpe", 0) > 0.5

    # Gate 2: Permutation p < 0.05
    gates["G2_perm_p_lt_0.05"] = p_value < 0.05

    # Gate 3: Regime gap < 0.5
    gates["G3_regime_gap_lt_0.5"] = metrics.get("regime_gap", 1) < 0.5

    # Gate 4: Max drawdown > -50% (i.e., not worse than -50%)
    gates["G4_maxdd_gt_neg50"] = metrics.get("max_drawdown_pct", -100) > -50

    # Gate 5: Trades >= 20
    gates["G5_trades_gte_20"] = metrics.get("n_trades", 0) >= 20

    gates["p_value"] = round(p_value, 4)
    gates["all_pass"] = all(v for k, v in gates.items() if k.startswith("G"))

    return gates


def qqq_benchmark(df):
    """QQQ buy-and-hold benchmark over OOT period."""
    oot = df.loc[OOT_START:OOT_END]
    first_close = oot["qqq_close"].dropna().iloc[0]
    last_close = oot["qqq_close"].dropna().iloc[-1]
    total_ret = (last_close / first_close) - 1
    n_years = len(oot) / 252
    cagr = (1 + total_ret) ** (1 / n_years) - 1
    daily_rets = oot["qqq_ret"].fillna(0)
    vol = daily_rets.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    eq = (1 + daily_rets).cumprod()
    max_dd = ((eq - eq.cummax()) / eq.cummax()).min()
    return {
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
    }


def main():
    print("=" * 80)
    print("RSI EXTREME BOUNCE BACKTEST")
    print(f"Capital: ${START_CAPITAL} | OOT: {OOT_START} to {OOT_END}")
    print(f"Slippage: {SLIPPAGE_PCT*100:.2f}% per trade | Permutations: {PERMUTATION_ITERS}")
    print("=" * 80)

    print("\n[1/4] Downloading data...")
    data = download_data()
    print(f"  QQQ: {len(data['QQQ'])} rows, TQQQ: {len(data['TQQQ'])} rows")

    print("\n[2/4] Building indicators...")
    df = build_indicators(data)
    oot = df.loc[OOT_START:OOT_END]
    print(f"  OOT period: {oot.index[0].date()} to {oot.index[-1].date()} ({len(oot)} trading days)")

    # QQQ benchmark
    bench = qqq_benchmark(df)
    print(f"\n  QQQ Buy-and-Hold Benchmark:")
    print(f"    Total Return: {bench['total_return_pct']:.1f}%  CAGR: {bench['cagr_pct']:.1f}%  Sharpe: {bench['sharpe']:.3f}  MaxDD: {bench['max_drawdown_pct']:.1f}%")

    qqq_daily = oot["qqq_ret"].fillna(0)

    variants = ["A", "B", "C", "D", "E", "F"]
    all_results = {}

    print("\n[3/4] Running backtests + permutation tests...")
    for v in variants:
        cfg = get_variant_config(v)
        print(f"\n  Variant {v}: {cfg[2]}")

        trades, daily_rets, final_equity, equity_series = run_backtest(df, v)
        metrics = compute_metrics(trades, daily_rets, final_equity, START_CAPITAL, qqq_daily, v)

        if metrics.get("error"):
            print(f"    ERROR: {metrics['error']}")
            all_results[v] = {"metrics": metrics, "gates": {"all_pass": False, "error": "No trades"}}
            continue

        print(f"    Trades: {metrics['n_trades']} | Return: {metrics['total_return_pct']:.1f}% | Sharpe: {metrics['sharpe']:.3f} | WR: {metrics['win_rate']:.0f}%")

        # Permutation test (use trade-level Sharpe for comparison)
        observed_sharpe = metrics["sharpe"]
        print(f"    Running permutation test ({PERMUTATION_ITERS} iterations)...", end=" ", flush=True)
        p_value = permutation_test(df, v, observed_sharpe)
        print(f"p={p_value:.4f}")

        gates = validate_gates(metrics, p_value)
        print(f"    Gates: ", end="")
        for k, v_gate in gates.items():
            if k.startswith("G"):
                status = "PASS" if v_gate else "FAIL"
                print(f"{k}={status}", end=" ")
        print(f"| ALL={'PASS' if gates['all_pass'] else 'FAIL'}")

        all_results[v] = {
            "metrics": metrics,
            "gates": gates,
            "trades": [
                {
                    "entry": t["entry_date"].strftime("%Y-%m-%d"),
                    "exit": t["exit_date"].strftime("%Y-%m-%d"),
                    "ret_pct": round(t["net_ret"] * 100, 2),
                    "regime": "bull" if t["regime_bull"] else "bear",
                }
                for t in trades
            ],
        }

    # ── SUMMARY TABLE ──────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)

    header = f"{'Var':<4} {'Description':<45} {'#Tr':>4} {'TotRet%':>8} {'CAGR%':>7} {'Vol%':>6} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Calmar':>7} {'WR%':>5} {'PF':>5} {'RGap':>5} {'p-val':>6} {'Pass':>5}"
    print(header)
    print("-" * 120)

    for v in variants:
        r = all_results[v]
        m = r["metrics"]
        g = r["gates"]

        if m.get("error"):
            print(f"  {v:<4} {m.get('description','?'):<45} {'NO TRADES':>8}")
            continue

        pass_str = "YES" if g.get("all_pass") else "NO"
        p_val = g.get("p_value", 1.0)

        print(f"  {v:<4} {m['description']:<45} {m['n_trades']:>4} {m['total_return_pct']:>8.1f} {m['cagr_pct']:>7.1f} {m['annual_vol_pct']:>6.1f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['max_drawdown_pct']:>7.1f} {m['calmar']:>7.3f} {m['win_rate']:>5.0f} {m['profit_factor']:>5.2f} {m['regime_gap']:>5.3f} {p_val:>6.3f} {pass_str:>5}")

    # Gate detail
    print("\n" + "-" * 80)
    print("GATE DETAIL")
    print("-" * 80)
    print(f"{'Var':<4} {'G1:Sharpe>0.5':>14} {'G2:Perm<0.05':>14} {'G3:RGap<0.5':>14} {'G4:DD>-50%':>14} {'G5:Trades>=20':>14}")

    for v in variants:
        g = all_results[v].get("gates", {})
        if "error" in g:
            print(f"  {v:<4} {'N/A':>14} {'N/A':>14} {'N/A':>14} {'N/A':>14} {'N/A':>14}")
            continue

        def gs(key):
            return "PASS" if g.get(key, False) else "FAIL"

        print(f"  {v:<4} {gs('G1_sharpe_gt_0.5'):>14} {gs('G2_perm_p_lt_0.05'):>14} {gs('G3_regime_gap_lt_0.5'):>14} {gs('G4_maxdd_gt_neg50'):>14} {gs('G5_trades_gte_20'):>14}")

    # Regime detail
    print("\n" + "-" * 80)
    print("REGIME STRATIFICATION")
    print("-" * 80)
    print(f"{'Var':<4} {'Bull Trades':>12} {'Bull Sharpe':>12} {'Bear Trades':>12} {'Bear Sharpe':>12} {'Regime Gap':>12} {'2022 Ret%':>10}")

    for v in variants:
        m = all_results[v]["metrics"]
        if m.get("error"):
            continue
        print(f"  {v:<4} {m['n_bull_trades']:>12} {m['sharpe_bull']:>12.3f} {m['n_bear_trades']:>12} {m['sharpe_bear']:>12.3f} {m['regime_gap']:>12.3f} {m['bear_2022_return_pct']:>10.1f}")

    # Benchmark comparison
    print("\n" + "-" * 80)
    print("QQQ BUY-AND-HOLD BENCHMARK")
    print("-" * 80)
    print(f"  Total Return: {bench['total_return_pct']:.1f}%  |  CAGR: {bench['cagr_pct']:.1f}%  |  Sharpe: {bench['sharpe']:.3f}  |  MaxDD: {bench['max_drawdown_pct']:.1f}%")

    # ── SAVE RESULTS ───────────────────────────────────────────────────────
    print("\n[4/4] Saving results...")
    output = {
        "metadata": {
            "script": "rsi_extreme_bounce_backtest.py",
            "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start_capital": START_CAPITAL,
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "slippage_pct": SLIPPAGE_PCT,
            "permutation_iters": PERMUTATION_ITERS,
        },
        "benchmark_qqq_bnh": bench,
        "variants": {},
    }

    for v in variants:
        r = all_results[v]
        output["variants"][v] = {
            "metrics": r["metrics"],
            "gates": r["gates"],
            "trade_count": len(r.get("trades", [])),
            "trades": r.get("trades", []),
        }

    output_path = "/home/jupiter/Lvl3Quant/data/rsi_extreme_bounce_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Results saved to {output_path}")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
