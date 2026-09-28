#!/usr/bin/env python3
"""
Insider Buying + Price Momentum Combo Backtest
===============================================
Proxy: volume-confirmed breakout as a stand-in for informed accumulation.

6 Variants tested through a 5-gate validation framework.
"""

import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    "JPM", "BAC", "WFC", "JNJ", "PFE", "UNH", "HD", "MCD",
    "DIS", "NFLX", "CRM", "ADBE", "AMD", "INTC", "PYPL", "SQ",
    "SHOP", "COIN", "UBER", "ABNB", "RIVN", "SNAP", "PINS",
]

# Rough GICS sector mapping for variant D
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Comm", "AMZN": "ConsDisc",
    "META": "Comm", "NVDA": "Tech", "TSLA": "ConsDisc", "JPM": "Fin",
    "BAC": "Fin", "WFC": "Fin", "JNJ": "Health", "PFE": "Health",
    "UNH": "Health", "HD": "ConsDisc", "MCD": "ConsDisc", "DIS": "Comm",
    "NFLX": "Comm", "CRM": "Tech", "ADBE": "Tech", "AMD": "Tech",
    "INTC": "Tech", "PYPL": "Fin", "SQ": "Fin", "SHOP": "Tech",
    "COIN": "Fin", "UBER": "Tech", "ABNB": "ConsDisc", "RIVN": "ConsDisc",
    "SNAP": "Comm", "PINS": "Comm",
}

ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
HOLD_DAYS = 20
MAX_POSITIONS = 5
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
DATA_START = "2021-01-01"  # extra lookback for indicators
N_PERMUTATIONS = 1000
SEED = 42

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/insider_buying_momentum_results.json")


# ── DATA DOWNLOAD ───────────────────────────────────────────────────────────
def download_data():
    """Download price + volume data for all tickers + SPY + ^VIX."""
    all_tickers = TICKERS + ["SPY", "^VIX"]
    print(f"Downloading data for {len(all_tickers)} symbols ...")
    data = {}
    # Download in batches to avoid rate limits
    batch_size = 10
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {', '.join(batch)}")
        for ticker in batch:
            try:
                df = yf.download(ticker, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
                if len(df) > 50:
                    # Flatten multi-level columns if needed
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    data[ticker] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            except Exception as e:
                print(f"    WARN: {ticker} download failed: {e}")
        time.sleep(0.5)
    print(f"  Got data for {len(data)} symbols")
    return data


# ── INDICATOR COMPUTATION ───────────────────────────────────────────────────
def compute_indicators(data: dict) -> dict:
    """Add technical indicators needed by all variants."""
    enriched = {}
    for ticker, df in data.items():
        df = df.copy()
        df["ret_1d"] = df["Close"].pct_change()
        df["high_20d"] = df["High"].rolling(20).max()
        df["vol_avg_20d"] = df["Volume"].rolling(20).mean()
        df["ret_60d"] = df["Close"].pct_change(60)
        df["sma_50d"] = df["Close"].rolling(50).mean()
        # slope of 50-day SMA over last 10 days (for variant F)
        df["sma_50d_slope"] = df["sma_50d"].diff(10)
        enriched[ticker] = df.dropna(subset=["high_20d", "vol_avg_20d"])
    return enriched


def compute_spy_regime(spy_df: pd.DataFrame) -> pd.Series:
    """Bull if SPY > 200-SMA, else bear."""
    sma200 = spy_df["Close"].rolling(200).mean()
    regime = (spy_df["Close"] > sma200).astype(int)  # 1=bull, 0=bear
    regime.name = "regime"
    return regime


def compute_sector_ranks(data: dict, date: pd.Timestamp) -> set:
    """Return set of sectors that are in top-3 by 20-day return on given date."""
    sector_rets = {}
    for ticker, df in data.items():
        if ticker in ("SPY", "^VIX"):
            continue
        sector = SECTOR_MAP.get(ticker)
        if sector is None:
            continue
        if date in df.index:
            idx = df.index.get_loc(date)
            if idx >= 20:
                r = df["Close"].iloc[idx] / df["Close"].iloc[idx - 20] - 1
                sector_rets.setdefault(sector, []).append(r)
    # Average return per sector
    avg = {s: np.mean(v) for s, v in sector_rets.items()}
    if len(avg) < 3:
        return set(avg.keys())
    top3 = sorted(avg, key=avg.get, reverse=True)[:3]
    return set(top3)


# ── SIGNAL GENERATORS ──────────────────────────────────────────────────────
def signal_base(df: pd.DataFrame, date, idx: int) -> bool:
    """Variant A: 20-day breakout + 1.5x volume."""
    if idx < 1:
        return False
    close = df["Close"].iloc[idx]
    prev_high20 = df["high_20d"].iloc[idx - 1]  # yesterday's 20-day high
    vol = df["Volume"].iloc[idx]
    vol_avg = df["vol_avg_20d"].iloc[idx]
    return close > prev_high20 and vol > 1.5 * vol_avg


def signal_momentum(df: pd.DataFrame, date, idx: int) -> bool:
    """Variant B: base + 60-day return > 0."""
    if not signal_base(df, date, idx):
        return False
    return df["ret_60d"].iloc[idx] > 0


def signal_pullback(df: pd.DataFrame, date, idx: int) -> bool:
    """Variant C: within 5% of 20-day high + volume spike."""
    if idx < 1:
        return False
    close = df["Close"].iloc[idx]
    high20 = df["high_20d"].iloc[idx]
    vol = df["Volume"].iloc[idx]
    vol_avg = df["vol_avg_20d"].iloc[idx]
    within_5pct = close >= high20 * 0.95
    vol_spike = vol > 1.5 * vol_avg
    not_breakout = close <= high20  # NOT above, just near
    return within_5pct and not_breakout and vol_spike


def signal_sector_rotation(df: pd.DataFrame, date, idx: int, top_sectors: set, ticker: str) -> bool:
    """Variant D: base + ticker's sector must be in top-3."""
    if not signal_base(df, date, idx):
        return False
    sector = SECTOR_MAP.get(ticker)
    return sector in top_sectors


def signal_vix_filter(df: pd.DataFrame, date, idx: int, vix_level: float) -> bool:
    """Variant E: base + VIX < 25."""
    if not signal_base(df, date, idx):
        return False
    return vix_level < 25


def signal_multi_tf(df: pd.DataFrame, date, idx: int) -> bool:
    """Variant F: 20-day breakout + 50-day SMA slope positive."""
    if not signal_base(df, date, idx):
        return False
    slope = df["sma_50d_slope"].iloc[idx]
    return slope > 0


# ── BACKTESTER ──────────────────────────────────────────────────────────────
class BacktestEngine:
    def __init__(self, account_size: float, max_positions: int, hold_days: int, slippage: float):
        self.account_size = account_size
        self.max_positions = max_positions
        self.hold_days = hold_days
        self.slippage = slippage

    def run(self, data: dict, signal_func, spy_regime: pd.Series,
            vix_data: pd.DataFrame = None, variant_name: str = "base") -> dict:
        """Run backtest for a given signal function. Returns trade list + equity curve."""
        # Build common date index (OOT only)
        all_dates = set()
        for df in data.values():
            if df is not None:
                all_dates.update(df.index)
        all_dates = sorted([d for d in all_dates if str(d)[:10] >= OOT_START and str(d)[:10] <= OOT_END])
        if not all_dates:
            return {"trades": [], "equity": []}

        trades = []
        open_positions = []  # list of dicts: {ticker, entry_date, entry_price, exit_date_target, shares}
        equity = self.account_size
        equity_curve = []

        for date in all_dates:
            # Close positions that have reached hold period
            still_open = []
            for pos in open_positions:
                if date >= pos["exit_date_target"]:
                    # Exit
                    ticker_df = data.get(pos["ticker"])
                    if ticker_df is not None and date in ticker_df.index:
                        exit_price = ticker_df.loc[date, "Close"] * (1 - self.slippage)
                    else:
                        # Find nearest date
                        exit_price = pos["entry_price"]  # fallback
                    pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                    equity += pnl
                    trades.append({
                        "ticker": pos["ticker"],
                        "entry_date": str(pos["entry_date"])[:10],
                        "exit_date": str(date)[:10],
                        "entry_price": pos["entry_price"],
                        "exit_price": exit_price,
                        "shares": pos["shares"],
                        "pnl": pnl,
                        "ret": (exit_price / pos["entry_price"]) - 1,
                        "regime": "bull" if (date in spy_regime.index and spy_regime.loc[date] == 1) else "bear",
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

            # Generate signals for each ticker
            if len(open_positions) < self.max_positions:
                # Pre-compute sector ranks for variant D (once per day)
                top_sectors = None
                if variant_name == "D":
                    top_sectors = compute_sector_ranks(data, date)

                vix_level = 20.0  # default
                if vix_data is not None and date in vix_data.index:
                    vix_level = vix_data.loc[date, "Close"]

                candidates = []
                held_tickers = {p["ticker"] for p in open_positions}
                for ticker in TICKERS:
                    if ticker in held_tickers:
                        continue
                    df = data.get(ticker)
                    if df is None or date not in df.index:
                        continue
                    idx = df.index.get_loc(date)

                    # Call appropriate signal
                    triggered = False
                    if variant_name == "A":
                        triggered = signal_base(df, date, idx)
                    elif variant_name == "B":
                        triggered = signal_momentum(df, date, idx)
                    elif variant_name == "C":
                        triggered = signal_pullback(df, date, idx)
                    elif variant_name == "D":
                        triggered = signal_sector_rotation(df, date, idx, top_sectors, ticker)
                    elif variant_name == "E":
                        triggered = signal_vix_filter(df, date, idx, vix_level)
                    elif variant_name == "F":
                        triggered = signal_multi_tf(df, date, idx)

                    if triggered:
                        candidates.append((ticker, idx))

                # Sort by volume ratio (strongest signal first)
                def vol_ratio(c):
                    t, i = c
                    df = data[t]
                    return df["Volume"].iloc[i] / df["vol_avg_20d"].iloc[i]
                candidates.sort(key=vol_ratio, reverse=True)

                slots = self.max_positions - len(open_positions)
                for ticker, idx in candidates[:slots]:
                    df = data[ticker]
                    entry_price = df["Close"].iloc[idx] * (1 + self.slippage)
                    # Equal weight: allocate equity / max_positions
                    alloc = equity / self.max_positions
                    if alloc < 1:
                        continue
                    shares = alloc / entry_price
                    # Find exit date target
                    future_dates = [d for d in all_dates if d > date]
                    if len(future_dates) >= self.hold_days:
                        exit_target = future_dates[self.hold_days - 1]
                    elif future_dates:
                        exit_target = future_dates[-1]
                    else:
                        continue
                    open_positions.append({
                        "ticker": ticker,
                        "entry_date": date,
                        "entry_price": entry_price,
                        "exit_date_target": exit_target,
                        "shares": shares,
                    })

            # Mark-to-market equity
            mtm = equity
            for pos in open_positions:
                df = data.get(pos["ticker"])
                if df is not None and date in df.index:
                    current = df.loc[date, "Close"]
                    mtm += (current - pos["entry_price"]) * pos["shares"]
            equity_curve.append({"date": str(date)[:10], "equity": mtm})

        # Force close remaining positions at last date
        last_date = all_dates[-1] if all_dates else None
        for pos in open_positions:
            df = data.get(pos["ticker"])
            if df is not None and last_date in df.index:
                exit_price = df.loc[last_date, "Close"] * (1 - self.slippage)
            else:
                exit_price = pos["entry_price"]
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            equity += pnl
            regime_val = "bull"
            if last_date in spy_regime.index and spy_regime.loc[last_date] == 0:
                regime_val = "bear"
            trades.append({
                "ticker": pos["ticker"],
                "entry_date": str(pos["entry_date"])[:10],
                "exit_date": str(last_date)[:10],
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "shares": pos["shares"],
                "pnl": pnl,
                "ret": (exit_price / pos["entry_price"]) - 1,
                "regime": regime_val,
            })

        return {"trades": trades, "equity_curve": equity_curve}


# ── VALIDATION GATES ────────────────────────────────────────────────────────
def compute_metrics(trades: list, equity_curve: list) -> dict:
    """Compute Sharpe, MaxDD, trade stats."""
    if not trades:
        return {"sharpe": 0, "sortino": 0, "max_dd": 0, "n_trades": 0,
                "win_rate": 0, "profit_factor": 0, "total_return": 0,
                "avg_ret": 0, "med_ret": 0}

    rets = [t["ret"] for t in trades]
    n = len(rets)
    avg = np.mean(rets)
    std = np.std(rets, ddof=1) if n > 1 else 1e-9
    sharpe = (avg / std) * np.sqrt(252 / HOLD_DAYS) if std > 1e-9 else 0

    downside = np.std([r for r in rets if r < 0], ddof=1) if any(r < 0 for r in rets) else 1e-9
    sortino = (avg / downside) * np.sqrt(252 / HOLD_DAYS) if downside > 1e-9 else 0

    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    win_rate = len(wins) / n if n > 0 else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 1e-9 else 999

    # MaxDD from equity curve
    if equity_curve:
        eqs = [e["equity"] for e in equity_curve]
        peak = eqs[0]
        max_dd = 0
        for eq in eqs:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd
    else:
        max_dd = 0

    total_ret = (equity_curve[-1]["equity"] / ACCOUNT_SIZE - 1) if equity_curve else 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd, 4),
        "n_trades": n,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 3),
        "total_return": round(total_ret, 4),
        "avg_ret": round(avg, 5),
        "med_ret": round(float(np.median(rets)), 5),
    }


def permutation_test(trades: list, data: dict, n_perms: int = N_PERMUTATIONS) -> float:
    """Shuffle entry dates, recompute avg return. Return p-value."""
    if len(trades) < 5:
        return 1.0

    rng = np.random.RandomState(SEED)
    observed_avg = np.mean([t["ret"] for t in trades])

    # Pool of all possible 20-day returns across all tickers in OOT
    pool = []
    for ticker in TICKERS:
        df = data.get(ticker)
        if df is None:
            continue
        oot = df[(df.index >= OOT_START) & (df.index <= OOT_END)]
        closes = oot["Close"].values
        for i in range(len(closes) - HOLD_DAYS):
            r = closes[i + HOLD_DAYS] / closes[i] - 1
            pool.append(r)

    if len(pool) < len(trades):
        return 1.0

    pool = np.array(pool)
    n_trades = len(trades)
    count_ge = 0
    for _ in range(n_perms):
        sample = rng.choice(pool, size=n_trades, replace=True)
        if np.mean(sample) >= observed_avg:
            count_ge += 1

    return count_ge / n_perms


def regime_gap(trades: list) -> float:
    """Compute |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|)."""
    bull_rets = [t["ret"] for t in trades if t["regime"] == "bull"]
    bear_rets = [t["ret"] for t in trades if t["regime"] == "bear"]

    def sharpe_from_rets(rets):
        if len(rets) < 2:
            return 0
        avg = np.mean(rets)
        std = np.std(rets, ddof=1)
        return (avg / std) * np.sqrt(252 / HOLD_DAYS) if std > 1e-9 else 0

    s_bull = sharpe_from_rets(bull_rets)
    s_bear = sharpe_from_rets(bear_rets)
    denom = max(abs(s_bull), abs(s_bear))
    if denom < 1e-9:
        return 0
    return abs(s_bull - s_bear) / denom


def validate(trades, equity_curve, data, variant_name):
    """Run 5-gate validation. Returns dict with pass/fail per gate + metrics."""
    metrics = compute_metrics(trades, equity_curve)

    print(f"  Running permutation test ({N_PERMUTATIONS} iters) ...")
    p_val = permutation_test(trades, data) if metrics["n_trades"] >= 20 else 1.0
    r_gap = regime_gap(trades) if metrics["n_trades"] >= 5 else 1.0

    gates = {
        "G1_sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "G2_perm_p_lt_0.05": p_val < 0.05,
        "G3_regime_gap_lt_0.5": r_gap < 0.5,
        "G4_maxdd_gt_neg50pct": metrics["max_dd"] > -0.50,
        "G5_min_20_trades": metrics["n_trades"] >= 20,
    }
    all_pass = all(gates.values())

    return {
        "variant": variant_name,
        "metrics": metrics,
        "p_value": round(p_val, 4),
        "regime_gap": round(r_gap, 4),
        "gates": gates,
        "all_gates_pass": all_pass,
        "n_bull_trades": len([t for t in trades if t["regime"] == "bull"]),
        "n_bear_trades": len([t for t in trades if t["regime"] == "bear"]),
    }


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("INSIDER BUYING + MOMENTUM COMBO BACKTEST")
    print("Proxy: Volume-confirmed breakout strategies")
    print(f"OOT: {OOT_START} to {OOT_END} | Account: ${ACCOUNT_SIZE}")
    print("=" * 70)

    # 1. Download data
    data = download_data()
    if "SPY" not in data:
        print("ERROR: SPY data missing, cannot compute regime")
        return

    # 2. Compute indicators
    print("\nComputing indicators ...")
    data = compute_indicators(data)

    # SPY regime
    spy_regime = compute_spy_regime(data["SPY"])

    # VIX data
    vix_data = data.get("^VIX")

    # 3. Run backtests
    engine = BacktestEngine(ACCOUNT_SIZE, MAX_POSITIONS, HOLD_DAYS, SLIPPAGE_PCT)
    variants = ["A", "B", "C", "D", "E", "F"]
    variant_names = {
        "A": "Base: 20d breakout + 1.5x volume",
        "B": "Momentum: + 60d return > 0",
        "C": "Pullback: within 5% of 20d high + vol spike",
        "D": "Sector rotation: top-3 sectors only",
        "E": "VIX filter: only when VIX < 25",
        "F": "Multi-TF: + 50d SMA slope positive",
    }

    results = {}
    for v in variants:
        print(f"\n{'─' * 60}")
        print(f"VARIANT {v}: {variant_names[v]}")
        print(f"{'─' * 60}")

        bt = engine.run(data, None, spy_regime, vix_data, variant_name=v)
        trades = bt["trades"]
        equity_curve = bt["equity_curve"]

        print(f"  Trades: {len(trades)}")
        if equity_curve:
            final_eq = equity_curve[-1]["equity"]
            print(f"  Final equity: ${final_eq:.2f} (return: {(final_eq / ACCOUNT_SIZE - 1) * 100:.1f}%)")

        result = validate(trades, equity_curve, data, v)
        result["description"] = variant_names[v]

        # Print gate results
        print(f"\n  GATE RESULTS:")
        for gate, passed in result["gates"].items():
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {gate}")
        print(f"  Sharpe: {result['metrics']['sharpe']} | Sortino: {result['metrics']['sortino']}")
        print(f"  WR: {result['metrics']['win_rate']:.1%} | PF: {result['metrics']['profit_factor']}")
        print(f"  MaxDD: {result['metrics']['max_dd']:.2%} | Perm p: {result['p_value']}")
        print(f"  Regime gap: {result['regime_gap']:.3f} | Bull/Bear trades: {result['n_bull_trades']}/{result['n_bear_trades']}")
        verdict = "ALL GATES PASS" if result["all_gates_pass"] else "FAILED"
        print(f"  >>> VERDICT: {verdict}")

        results[v] = result

    # 4. Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Var':<4} {'Description':<42} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'#Tr':>5} {'Pass':>5}")
    print("-" * 70)
    for v in variants:
        r = results[v]
        m = r["metrics"]
        p = "YES" if r["all_gates_pass"] else "NO"
        print(f"  {v:<3} {variant_names[v]:<42} {m['sharpe']:>6.2f} {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {m['n_trades']:>4} {p:>5}")

    # 5. Save results
    # Convert for JSON serialization
    output = {
        "metadata": {
            "strategy": "Insider Buying + Momentum Combo (Volume-Confirmed Breakout Proxy)",
            "oot_period": f"{OOT_START} to {OOT_END}",
            "account_size": ACCOUNT_SIZE,
            "slippage_pct": SLIPPAGE_PCT,
            "hold_days": HOLD_DAYS,
            "max_positions": MAX_POSITIONS,
            "n_tickers": len(TICKERS),
            "n_permutations": N_PERMUTATIONS,
            "run_timestamp": datetime.now().isoformat(),
        },
        "variants": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")
    print("DONE.")


if __name__ == "__main__":
    main()
