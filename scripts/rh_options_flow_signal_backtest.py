#!/usr/bin/env python3
"""
Robinhood Options Flow Signal Backtest
======================================
Uses high relative volume + price patterns as proxy for informed options flow.
6 variants tested against 5-gate validation framework.

Universe: 30 high-options-flow stocks, 2022-01-01 to 2026-07-28
Account: $645, max 3 concurrent positions, 0.02% slippage each way
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "MU", "NFLX",
    "CRM", "ADBE", "SOFI", "HOOD", "COIN", "PLTR", "SNAP", "PINS", "RBLX", "UBER",
    "ABNB", "SQ", "PYPL", "RIVN", "MARA", "RIOT", "DIS", "BA", "JPM", "GS"
]

# Sector mapping for variant E
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "Tech", "META": "Tech",
    "NVDA": "Semis", "AMD": "Semis", "MU": "Semis",
    "TSLA": "EV/Auto", "RIVN": "EV/Auto",
    "NFLX": "Media", "DIS": "Media", "SNAP": "Media", "PINS": "Media", "RBLX": "Media",
    "CRM": "SaaS", "ADBE": "SaaS", "PLTR": "SaaS",
    "SOFI": "Fintech", "HOOD": "Fintech", "COIN": "Fintech", "SQ": "Fintech", "PYPL": "Fintech",
    "UBER": "Platform", "ABNB": "Platform",
    "MARA": "Crypto", "RIOT": "Crypto",
    "BA": "Industrial", "JPM": "Finance", "GS": "Finance",
}

SECTOR_ETFS = {
    "Tech": "XLK", "Semis": "SMH", "EV/Auto": "CARZ", "Media": "XLC",
    "SaaS": "IGV", "Fintech": "FINX", "Platform": "XLK", "Crypto": "BITO",
    "Industrial": "XLI", "Finance": "XLF",
}

START_DATE = "2022-01-01"
END_DATE = "2026-07-28"
INITIAL_CAPITAL = 645.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% each way
N_PERMUTATIONS = 1000

# ============================================================
# DATA DOWNLOAD
# ============================================================
def download_data():
    """Download all required price data."""
    print("Downloading price data...")

    all_tickers = UNIVERSE + ["SPY", "^VIX"]
    # Add sector ETFs
    for etf in SECTOR_ETFS.values():
        if etf not in all_tickers:
            all_tickers.append(etf)

    data = {}
    # Download in batch
    raw = yf.download(all_tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    for ticker in all_tickers:
        try:
            if len(all_tickers) > 1:
                df = pd.DataFrame({
                    "Open": raw["Open"][ticker],
                    "High": raw["High"][ticker],
                    "Low": raw["Low"][ticker],
                    "Close": raw["Close"][ticker],
                    "Volume": raw["Volume"][ticker],
                })
            else:
                df = raw.copy()
            df = df.dropna(subset=["Close"])
            if len(df) > 50:
                data[ticker] = df
        except Exception:
            pass

    print(f"  Downloaded {len(data)} tickers, {len(data.get('SPY', []))} trading days")
    return data


def compute_features(data):
    """Compute all features needed for signal generation."""
    features = {}

    spy = data.get("SPY")
    if spy is None:
        raise ValueError("SPY data missing")

    spy_sma200 = spy["Close"].rolling(200).mean()
    vix = data.get("^VIX")

    for ticker in UNIVERSE:
        if ticker not in data:
            continue
        df = data[ticker].copy()

        # Volume features
        df["vol_sma20"] = df["Volume"].rolling(20).mean()
        df["rel_volume"] = df["Volume"] / df["vol_sma20"]

        # Price features
        df["high_5d"] = df["High"].rolling(5).max()
        df["low_5d"] = df["Low"].rolling(5).min()
        df["range_5d_pct"] = (df["high_5d"] - df["low_5d"]) / df["Close"] * 100
        df["high_20d"] = df["High"].rolling(20).max()
        df["ret_5d"] = df["Close"].pct_change(5) * 100

        # Align SPY regime
        df["spy_above_200sma"] = spy_sma200.reindex(df.index).apply(
            lambda x: True if pd.notna(x) else np.nan
        )
        # Actually compare SPY close to its SMA200
        spy_close_aligned = spy["Close"].reindex(df.index)
        spy_sma_aligned = spy_sma200.reindex(df.index)
        df["spy_above_200sma"] = spy_close_aligned > spy_sma_aligned

        # VIX
        if vix is not None:
            df["vix"] = vix["Close"].reindex(df.index)
        else:
            df["vix"] = 20.0  # default

        # Sector
        df["sector"] = SECTOR_MAP.get(ticker, "Other")

        features[ticker] = df.dropna(subset=["vol_sma20", "range_5d_pct", "high_20d", "ret_5d"])

    return features, spy, spy_sma200


# ============================================================
# SIGNAL GENERATORS
# ============================================================
def signal_A(features):
    """Volume spike + low range: vol >2x AND 5d range <3% -> hold 10 days."""
    signals = []
    for ticker, df in features.items():
        mask = (df["rel_volume"] > 2.0) & (df["range_5d_pct"] < 3.0)
        for date in df.index[mask]:
            signals.append({"ticker": ticker, "date": date, "hold_days": 10})
    return signals


def signal_B(features):
    """Volume spike + breakout: vol >2x AND close > 20d high -> hold 5 days."""
    signals = []
    for ticker, df in features.items():
        # Price breaks above previous day's 20-day high (shift to avoid lookahead)
        prev_high20 = df["high_20d"].shift(1)
        mask = (df["rel_volume"] > 2.0) & (df["Close"] > prev_high20)
        for date in df.index[mask]:
            signals.append({"ticker": ticker, "date": date, "hold_days": 5})
    return signals


def signal_C(features):
    """Volume surge + momentum: vol >3x AND 5d return >2% -> hold 5 days."""
    signals = []
    for ticker, df in features.items():
        mask = (df["rel_volume"] > 3.0) & (df["ret_5d"] > 2.0)
        for date in df.index[mask]:
            signals.append({"ticker": ticker, "date": date, "hold_days": 5})
    return signals


def signal_D(features):
    """Contrarian flow: vol >2x AND stock down >3% in 5 days -> hold 10 days."""
    signals = []
    for ticker, df in features.items():
        mask = (df["rel_volume"] > 2.0) & (df["ret_5d"] < -3.0)
        for date in df.index[mask]:
            signals.append({"ticker": ticker, "date": date, "hold_days": 10})
    return signals


def signal_E(features, data):
    """Sector flow rotation: 3+ stocks in sector with vol >2x -> buy sector ETF for 10 days."""
    signals = []
    # Group by date and sector
    date_sector_counts = defaultdict(lambda: defaultdict(int))
    for ticker, df in features.items():
        sector = SECTOR_MAP.get(ticker, "Other")
        high_vol_dates = df.index[df["rel_volume"] > 2.0]
        for date in high_vol_dates:
            date_sector_counts[date][sector] += 1

    for date, sectors in date_sector_counts.items():
        for sector, count in sectors.items():
            if count >= 3:
                etf = SECTOR_ETFS.get(sector)
                if etf and etf in data:
                    signals.append({"ticker": etf, "date": date, "hold_days": 10})
    return signals


def signal_F(features):
    """Combined: vol >2x AND (breakout OR dip >3%) AND VIX <25 -> hold 10 days."""
    signals = []
    for ticker, df in features.items():
        prev_high20 = df["high_20d"].shift(1)
        breakout = df["Close"] > prev_high20
        dip = df["ret_5d"] < -3.0
        mask = (df["rel_volume"] > 2.0) & (breakout | dip) & (df["vix"] < 25)
        for date in df.index[mask]:
            signals.append({"ticker": ticker, "date": date, "hold_days": 10})
    return signals


# ============================================================
# BACKTEST ENGINE
# ============================================================
def backtest(signals, data, initial_capital=INITIAL_CAPITAL, max_concurrent=MAX_CONCURRENT,
             slippage=SLIPPAGE_PCT):
    """
    Run backtest with position limits and slippage.
    Returns trade list and equity curve.
    """
    if not signals:
        return [], pd.Series(dtype=float), []

    # Sort signals by date
    signals = sorted(signals, key=lambda x: x["date"])

    capital = initial_capital
    positions = []  # list of {ticker, entry_date, entry_price, exit_date, shares}
    trades = []     # completed trades
    equity_curve = {}

    # Build a date index from SPY
    spy = data.get("SPY")
    if spy is None:
        return [], pd.Series(dtype=float), []
    all_dates = spy.index.sort_values()

    # For each signal, find entry (next day open) and exit
    pending_signals = []
    for sig in signals:
        entry_idx = all_dates.searchsorted(sig["date"]) + 1  # next trading day
        if entry_idx >= len(all_dates):
            continue
        entry_date = all_dates[entry_idx]

        exit_idx = entry_idx + sig["hold_days"]
        if exit_idx >= len(all_dates):
            exit_idx = len(all_dates) - 1
        exit_date = all_dates[exit_idx]

        pending_signals.append({
            "ticker": sig["ticker"],
            "entry_date": entry_date,
            "exit_date": exit_date,
        })

    # Sort by entry date
    pending_signals.sort(key=lambda x: x["entry_date"])

    # Simulate
    active_positions = []
    capital = initial_capital
    sig_idx = 0

    for date in all_dates:
        # Close expired positions
        still_active = []
        for pos in active_positions:
            if date >= pos["exit_date"]:
                # Exit at close price on exit date
                ticker_data = data.get(pos["ticker"])
                if ticker_data is not None and date in ticker_data.index:
                    exit_price = ticker_data.loc[date, "Close"] * (1 - slippage)
                    pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                    capital += exit_price * pos["shares"]
                    trades.append({
                        "ticker": pos["ticker"],
                        "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                        "exit_date": date.strftime("%Y-%m-%d"),
                        "entry_price": round(pos["entry_price"], 4),
                        "exit_price": round(exit_price, 4),
                        "shares": pos["shares"],
                        "pnl": round(pnl, 2),
                        "ret": round(pnl / (pos["entry_price"] * pos["shares"]) * 100, 4),
                    })
                else:
                    # Can't exit, carry forward
                    still_active.append(pos)
                    continue
            else:
                still_active.append(pos)
        active_positions = still_active

        # Open new positions
        while sig_idx < len(pending_signals):
            sig = pending_signals[sig_idx]
            if sig["entry_date"] > date:
                break
            if sig["entry_date"] == date and len(active_positions) < max_concurrent:
                ticker_data = data.get(sig["ticker"])
                if ticker_data is not None and date in ticker_data.index:
                    entry_price = ticker_data.loc[date, "Open"] * (1 + slippage)
                    pos_size = capital / max_concurrent  # equal weight
                    if pos_size > 10 and entry_price > 0:
                        shares = int(pos_size / entry_price)
                        if shares > 0:
                            cost = entry_price * shares
                            capital -= cost
                            active_positions.append({
                                "ticker": sig["ticker"],
                                "entry_date": date,
                                "exit_date": sig["exit_date"],
                                "entry_price": entry_price,
                                "shares": shares,
                            })
            sig_idx += 1

        # Mark to market
        mtm = capital
        for pos in active_positions:
            ticker_data = data.get(pos["ticker"])
            if ticker_data is not None and date in ticker_data.index:
                mtm += ticker_data.loc[date, "Close"] * pos["shares"]
            else:
                mtm += pos["entry_price"] * pos["shares"]
        equity_curve[date] = mtm

    eq = pd.Series(equity_curve).sort_index()
    return trades, eq, pending_signals


# ============================================================
# METRICS & VALIDATION
# ============================================================
def compute_metrics(trades, eq, spy_data, spy_sma200):
    """Compute all performance metrics."""
    if not trades or len(eq) < 2:
        return {
            "n_trades": 0, "sharpe": 0, "sortino": 0, "profit_factor": 0,
            "win_rate": 0, "max_dd": -100, "total_return": 0,
            "avg_ret": 0, "regime_gap": 999,
        }

    # Returns
    rets = eq.pct_change().dropna()
    daily_ret = rets.mean()
    daily_std = rets.std()
    downside_std = rets[rets < 0].std()

    sharpe = (daily_ret / daily_std * np.sqrt(252)) if daily_std > 0 else 0
    sortino = (daily_ret / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Trade stats
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    peak = eq.expanding().max()
    dd = (eq - peak) / peak * 100
    max_dd = dd.min()

    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    avg_ret = np.mean([t["ret"] for t in trades])

    # Regime analysis (bull vs bear based on SPY vs 200 SMA)
    bull_trades = []
    bear_trades = []
    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        if entry in spy_sma200.index and entry in spy_data.index:
            spy_close = spy_data.loc[entry, "Close"] if entry in spy_data.index else None
            sma_val = spy_sma200.loc[entry] if entry in spy_sma200.index else None
            if spy_close is not None and sma_val is not None and pd.notna(sma_val):
                if spy_close > sma_val:
                    bull_trades.append(t["ret"])
                else:
                    bear_trades.append(t["ret"])

    bull_sharpe = 0
    bear_sharpe = 0
    if len(bull_trades) > 5:
        bull_sharpe = np.mean(bull_trades) / (np.std(bull_trades) + 1e-9) * np.sqrt(252 / 7.5)
    if len(bear_trades) > 5:
        bear_sharpe = np.mean(bear_trades) / (np.std(bear_trades) + 1e-9) * np.sqrt(252 / 7.5)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "n_trades": len(trades),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate, 1),
        "max_dd": round(max_dd, 1),
        "total_return": round(total_return, 1),
        "avg_ret": round(avg_ret, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "n_bull_trades": len(bull_trades),
        "n_bear_trades": len(bear_trades),
    }


def permutation_test(trades, eq, n_perms=N_PERMUTATIONS):
    """Shuffle trade returns to test if Sharpe is significant."""
    if not trades or len(eq) < 10:
        return 1.0

    trade_rets = np.array([t["ret"] for t in trades])
    actual_sharpe = np.mean(trade_rets) / (np.std(trade_rets) + 1e-9)

    count_better = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        shuffled = trade_rets.copy()
        # Randomly flip signs (null: no directional edge)
        signs = rng.choice([-1, 1], size=len(shuffled))
        shuffled = shuffled * signs
        perm_sharpe = np.mean(shuffled) / (np.std(shuffled) + 1e-9)
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_perms


def validate_5gate(metrics, perm_p):
    """Apply 5-gate validation framework."""
    gates = {
        "G1_Sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "G2_Perm_p_lt_0.05": perm_p < 0.05,
        "G3_Regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "G4_MaxDD_gt_neg50": metrics["max_dd"] > -50,
        "G5_Trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["ALL_PASS"] = all(gates.values())
    return gates


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("ROBINHOOD OPTIONS FLOW SIGNAL BACKTEST")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Capital: ${INITIAL_CAPITAL} | Max concurrent: {MAX_CONCURRENT} | Slippage: {SLIPPAGE_PCT*100}%")
    print("=" * 70)

    # Download data
    data = download_data()

    # Compute features
    features, spy_data, spy_sma200 = compute_features(data)
    print(f"  Computed features for {len(features)} stocks\n")

    # Generate signals for each variant
    variant_signals = {
        "A_vol_spike_low_range": signal_A(features),
        "B_vol_spike_breakout": signal_B(features),
        "C_vol_surge_momentum": signal_C(features),
        "D_contrarian_flow": signal_D(features),
        "E_sector_rotation": signal_E(features, data),
        "F_combined_signal": signal_F(features),
    }

    results = {}

    for name, signals in variant_signals.items():
        print(f"\n{'─' * 60}")
        print(f"VARIANT: {name}")
        print(f"  Raw signals: {len(signals)}")

        if len(signals) == 0:
            print("  NO SIGNALS - SKIP")
            results[name] = {
                "metrics": {"n_trades": 0, "sharpe": 0},
                "gates": {"ALL_PASS": False},
                "status": "NO_SIGNALS"
            }
            continue

        # Run backtest
        trades, eq, _ = backtest(signals, data)
        print(f"  Executed trades: {len(trades)}")

        if len(trades) == 0:
            print("  NO TRADES EXECUTED - SKIP")
            results[name] = {
                "metrics": {"n_trades": 0, "sharpe": 0},
                "gates": {"ALL_PASS": False},
                "status": "NO_TRADES"
            }
            continue

        # Compute metrics
        metrics = compute_metrics(trades, eq, spy_data, spy_sma200)

        # Permutation test
        perm_p = permutation_test(trades, eq)
        metrics["perm_p"] = round(perm_p, 4)

        # Validate
        gates = validate_5gate(metrics, perm_p)

        # Print results
        print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f}")
        print(f"  PF: {metrics['profit_factor']:.2f} | WR: {metrics['win_rate']:.1f}%")
        print(f"  Total Return: {metrics['total_return']:.1f}% | MaxDD: {metrics['max_dd']:.1f}%")
        print(f"  Avg Trade: {metrics['avg_ret']:.3f}% | Trades: {metrics['n_trades']}")
        print(f"  Regime: Bull Sharpe={metrics['bull_sharpe']:.3f} ({metrics['n_bull_trades']} trades) "
              f"| Bear Sharpe={metrics['bear_sharpe']:.3f} ({metrics['n_bear_trades']} trades)")
        print(f"  Regime Gap: {metrics['regime_gap']:.3f} | Perm p-value: {perm_p:.4f}")
        print()

        # Gates
        status = "PASS" if gates["ALL_PASS"] else "FAIL"
        print(f"  5-GATE VALIDATION: {'*** PASS ***' if gates['ALL_PASS'] else 'FAIL'}")
        for gate_name, passed in gates.items():
            if gate_name != "ALL_PASS":
                mark = "PASS" if passed else "FAIL"
                print(f"    [{mark}] {gate_name}")

        # Top trades
        top_winners = sorted(trades, key=lambda x: x["pnl"], reverse=True)[:3]
        top_losers = sorted(trades, key=lambda x: x["pnl"])[:3]
        print(f"\n  Top 3 winners:")
        for t in top_winners:
            print(f"    {t['ticker']} {t['entry_date']}->{t['exit_date']}: ${t['pnl']:.2f} ({t['ret']:.2f}%)")
        print(f"  Top 3 losers:")
        for t in top_losers:
            print(f"    {t['ticker']} {t['entry_date']}->{t['exit_date']}: ${t['pnl']:.2f} ({t['ret']:.2f}%)")

        results[name] = {
            "metrics": metrics,
            "gates": gates,
            "status": status,
            "n_raw_signals": len(signals),
            "sample_trades": trades[:10],
            "final_equity": round(eq.iloc[-1], 2) if len(eq) > 0 else INITIAL_CAPITAL,
        }

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY — ALL VARIANTS")
    print("=" * 70)
    print(f"{'Variant':<30} {'Sharpe':>7} {'PF':>6} {'WR%':>6} {'DD%':>6} {'Ret%':>7} {'Trades':>6} {'Pass?':>6}")
    print("-" * 70)

    for name, res in results.items():
        m = res.get("metrics", {})
        passed = "YES" if res.get("gates", {}).get("ALL_PASS", False) else "NO"
        print(f"{name:<30} {m.get('sharpe',0):>7.3f} {m.get('profit_factor',0):>6.2f} "
              f"{m.get('win_rate',0):>5.1f}% {m.get('max_dd',0):>5.1f}% "
              f"{m.get('total_return',0):>6.1f}% {m.get('n_trades',0):>6d} {passed:>6}")

    passing = [n for n, r in results.items() if r.get("gates", {}).get("ALL_PASS", False)]
    print(f"\nPassing variants: {len(passing)}/{len(results)}")
    if passing:
        print(f"  Winners: {', '.join(passing)}")
    else:
        print("  No variants passed all 5 gates.")

    # Best variant by Sharpe
    best = max(results.items(), key=lambda x: x[1].get("metrics", {}).get("sharpe", -999))
    print(f"\nBest by Sharpe: {best[0]} (Sharpe={best[1]['metrics'].get('sharpe', 0):.3f})")

    # Save results
    output_path = "/home/jupiter/Lvl3Quant/data/rh_options_flow_signal_results.json"

    # Make JSON serializable
    save_data = {}
    for name, res in results.items():
        save_res = {}
        for k, v in res.items():
            if isinstance(v, dict):
                save_res[k] = {kk: (bool(vv) if isinstance(vv, (np.bool_,)) else
                                    float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                               for kk, vv in v.items()}
            elif isinstance(v, list):
                save_res[k] = v
            else:
                save_res[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        save_data[name] = save_res

    save_data["_meta"] = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "period": f"{START_DATE} to {END_DATE}",
        "initial_capital": INITIAL_CAPITAL,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_pct": SLIPPAGE_PCT,
        "n_permutations": N_PERMUTATIONS,
        "universe_size": len(UNIVERSE),
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
    print("DONE.")


if __name__ == "__main__":
    main()
