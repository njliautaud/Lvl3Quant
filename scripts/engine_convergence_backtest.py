#!/usr/bin/env python3
"""
Paper Engine Convergence Backtest
=================================
Simulates 6 independent paper trading engines and trades on their convergence.
Walk-forward OOT: Jan 2022 – Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.

6 Variants:
  A) Any-3 Convergence
  B) Any-4 Convergence
  C) Momentum+Reversion Combo
  D) Top-1 Convergence (highest agreement)
  E) Options on Convergence (>=4 agree, buy ATM calls)
  F) Regime-Filtered (A but only when VIX<20)
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
SECTORS = ["XLK", "XLF", "XLV", "XLE", "XLC", "XLY", "XLP", "XLU", "XLI", "XLB", "XLRE"]
BENCHMARKS = ["SPY", "RSP", "^VIX"]
ALL_TICKERS = SECTORS + BENCHMARKS

ACCOUNT_SIZE = 645.0
MAX_POSITIONS = 3
SLIPPAGE_PCT = 0.0002  # 0.02%
OPTION_PREMIUM_PCT = 0.02
OPTION_COMMISSION = 0.65
OPTION_SPREAD_HAIRCUT = 0.05

DATA_START = "2021-01-01"
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"

N_PERMUTATIONS = 1000
REBALANCE_DOW = 4  # Friday

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/engine_convergence_results.json")


# ── DATA DOWNLOAD ───────────────────────────────────────────────────────────
def download_data():
    print("Downloading price data...")
    data = {}
    for t in ALL_TICKERS:
        try:
            df = yf.download(t, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
            if len(df) > 100:
                data[t] = df["Close"].squeeze()
                print(f"  {t}: {len(df)} bars")
        except Exception as e:
            print(f"  {t}: ERROR {e}")
    return pd.DataFrame(data).ffill().dropna()


# ── ENGINE SIGNALS ──────────────────────────────────────────────────────────
def compute_engine_signals(prices: pd.DataFrame, sectors: list) -> dict:
    """Return dict of engine_name -> DataFrame[sectors] with 1/0 buy signals."""
    spy = prices["SPY"]
    rsp = prices.get("RSP", spy)

    engines = {}

    # 1. MOMENTUM: 20d return > 5%, top-3 only
    ret20 = prices[sectors].pct_change(20)
    mom_base = (ret20 > 0.05).astype(int)
    mom_ranked = ret20.rank(axis=1, ascending=False)
    engines["momentum"] = mom_base.where(mom_ranked <= 3, 0).fillna(0).astype(int)

    # 2. MEAN-REVERSION: 5d return < -3% AND above 200-SMA
    ret5 = prices[sectors].pct_change(5)
    sma200 = prices[sectors].rolling(200).mean()
    engines["mean_reversion"] = ((ret5 < -0.03) & (prices[sectors] > sma200)).astype(int).fillna(0)

    # 3. TREND (golden cross): price > 50-SMA AND 50-SMA > 200-SMA
    sma50 = prices[sectors].rolling(50).mean()
    engines["trend"] = ((prices[sectors] > sma50) & (sma50 > sma200)).astype(int).fillna(0)

    # 4. RELATIVE STRENGTH: 20d RS ratio > 50d RS ratio
    rs = prices[sectors].div(spy, axis=0)
    engines["rel_strength"] = (rs.rolling(20).mean() > rs.rolling(50).mean()).astype(int).fillna(0)

    # 5. VOL-ADJUSTED: rolling 20d annualized Sharpe > 0.5
    daily_ret = prices[sectors].pct_change()
    rm = daily_ret.rolling(20).mean()
    rs_std = daily_ret.rolling(20).std().replace(0, np.nan)
    engines["vol_adjusted"] = ((rm / rs_std * np.sqrt(252)) > 0.5).astype(int).fillna(0)

    # 6. BREADTH: RSP/SPY trending up (broadcast to all sectors)
    br = rsp / spy
    breadth_up = (br.rolling(20).mean() > br.rolling(50).mean()).astype(int).fillna(0)
    engines["breadth"] = pd.DataFrame(
        np.tile(breadth_up.values.reshape(-1, 1), (1, len(sectors))),
        index=prices.index, columns=sectors,
    )

    return engines


def compute_convergence(engines: dict, sectors: list) -> pd.DataFrame:
    """Sum of engine signals per sector per day."""
    conv = pd.DataFrame(0, index=list(engines.values())[0].index, columns=sectors)
    for sig in engines.values():
        conv = conv.add(sig[sectors].fillna(0), fill_value=0)
    return conv.astype(int)


# ── SIGNAL SELECTION LOGIC ──────────────────────────────────────────────────
def get_target_sectors(conv_row, engines, date, vix_val, variant, sectors):
    """Return list of sectors to hold for this variant."""
    if variant == "A":
        return [s for s in sectors if conv_row.get(s, 0) >= 3]
    elif variant == "B":
        return [s for s in sectors if conv_row.get(s, 0) >= 4]
    elif variant == "C":
        mom = engines["momentum"].loc[date] if date in engines["momentum"].index else pd.Series(dtype=float)
        mr = engines["mean_reversion"].loc[date] if date in engines["mean_reversion"].index else pd.Series(dtype=float)
        return [s for s in sectors if mom.get(s, 0) == 1 and mr.get(s, 0) == 1]
    elif variant == "D":
        if conv_row.max() >= 2:
            return [conv_row[sectors].idxmax()]
        return []
    elif variant == "E":
        return [s for s in sectors if conv_row.get(s, 0) >= 4]
    elif variant == "F":
        if vix_val < 20:
            return [s for s in sectors if conv_row.get(s, 0) >= 3]
        return []
    return []


# ── BACKTEST ENGINE (SHARES) ───────────────────────────────────────────────
def run_backtest_shares(
    prices, convergence, engines, vix, variant, sectors,
):
    """Portfolio-based backtest for share variants (A-D, F)."""
    oot_mask = (prices.index >= OOT_START) & (prices.index <= OOT_END)
    dates = prices.index[oot_mask]
    rebal_dates = [d for d in dates if d.dayofweek == REBALANCE_DOW]

    if not rebal_dates:
        return {"trades": [], "equity_curve": [], "final_equity": ACCOUNT_SIZE}

    cash = ACCOUNT_SIZE
    positions = {}  # sector -> {shares, entry_price, entry_date}
    equity_curve = []
    trades = []

    for date in rebal_dates:
        if date not in convergence.index:
            # Mark to market
            port_val = cash + sum(
                pos["shares"] * prices[s].loc[date]
                for s, pos in positions.items()
                if s in prices.columns and date in prices[s].index
            )
            equity_curve.append({"date": str(date.date()), "equity": round(port_val, 2)})
            continue

        conv_row = convergence.loc[date]
        vix_val = vix.loc[date] if date in vix.index else 25.0

        target = get_target_sectors(conv_row, engines, date, vix_val, variant, sectors)

        # Cap to MAX_POSITIONS by convergence score
        if len(target) > MAX_POSITIONS:
            target.sort(key=lambda s: -conv_row.get(s, 0))
            target = target[:MAX_POSITIONS]

        target_set = set(target)

        # CLOSE positions not in target
        for s in list(positions.keys()):
            if s not in target_set:
                pos = positions.pop(s)
                exit_p = prices[s].loc[date] * (1 - SLIPPAGE_PCT)
                pnl = (exit_p - pos["entry_price"]) * pos["shares"]
                cash += pos["shares"] * exit_p
                trades.append({
                    "sector": s,
                    "entry_date": str(pos["entry_date"]),
                    "exit_date": str(date.date()),
                    "entry_price": round(pos["entry_price"], 4),
                    "exit_price": round(exit_p, 4),
                    "shares": pos["shares"],
                    "pnl": round(pnl, 2),
                })

        # Compute total portfolio value for sizing new positions
        port_val = cash + sum(
            pos["shares"] * prices[s].loc[date]
            for s, pos in positions.items()
            if s in prices.columns and date in prices[s].index
        )

        # OPEN new positions
        new_sectors = [s for s in target if s not in positions and s in prices.columns and date in prices[s].index]
        n_total = len(positions) + len(new_sectors)
        if n_total > 0 and new_sectors:
            alloc = port_val / n_total
            for s in new_sectors:
                entry_p = prices[s].loc[date] * (1 + SLIPPAGE_PCT)
                shares = int(alloc / entry_p)
                if shares < 1 or shares * entry_p > cash:
                    shares = int(cash / entry_p)
                if shares < 1:
                    continue
                cost = shares * entry_p
                cash -= cost
                positions[s] = {"shares": shares, "entry_price": entry_p, "entry_date": date.date()}

        # Mark to market
        port_val = cash + sum(
            pos["shares"] * prices[s].loc[date]
            for s, pos in positions.items()
            if s in prices.columns and date in prices[s].index
        )
        equity_curve.append({"date": str(date.date()), "equity": round(port_val, 2)})

    # Close remaining at last date
    last_date = dates[-1]
    for s in list(positions.keys()):
        pos = positions.pop(s)
        exit_p = prices[s].iloc[-1] * (1 - SLIPPAGE_PCT) if s in prices.columns else pos["entry_price"]
        pnl = (exit_p - pos["entry_price"]) * pos["shares"]
        cash += pos["shares"] * exit_p
        trades.append({
            "sector": s,
            "entry_date": str(pos["entry_date"]),
            "exit_date": str(last_date.date()),
            "entry_price": round(pos["entry_price"], 4),
            "exit_price": round(exit_p, 4),
            "shares": pos["shares"],
            "pnl": round(pnl, 2),
        })

    final_eq = cash
    return {"trades": trades, "equity_curve": equity_curve, "final_equity": round(final_eq, 2)}


# ── BACKTEST ENGINE (OPTIONS — VARIANT E) ──────────────────────────────────
def run_backtest_options(
    prices, convergence, engines, vix, sectors,
):
    """Options backtest: buy ATM calls when >=4 engines agree, 2-week expiry model."""
    oot_mask = (prices.index >= OOT_START) & (prices.index <= OOT_END)
    dates = prices.index[oot_mask]
    rebal_dates = [d for d in dates if d.dayofweek == REBALANCE_DOW]

    if not rebal_dates:
        return {"trades": [], "equity_curve": [], "final_equity": ACCOUNT_SIZE}

    cash = ACCOUNT_SIZE
    equity_curve = []
    trades = []
    # Options settle in 2 weeks, so we track open option positions
    open_options = []  # list of {sector, entry_date, expiry_date, strike, n_contracts, cost}

    for idx, date in enumerate(rebal_dates):
        # Settle expired options
        for opt in list(open_options):
            if date >= pd.Timestamp(opt["expiry_date"]):
                s = opt["sector"]
                if s in prices.columns and date in prices[s].index:
                    curr_price = prices[s].loc[date]
                else:
                    curr_price = opt["strike"]
                intrinsic = max(0, curr_price - opt["strike"])
                proceeds = intrinsic * 100 * opt["n_contracts"]
                if proceeds > 0:
                    proceeds -= OPTION_COMMISSION  # exit commission
                pnl = proceeds - opt["cost"]
                cash += proceeds
                trades.append({
                    "sector": s,
                    "entry_date": opt["entry_date"],
                    "exit_date": str(date.date()),
                    "strike": round(opt["strike"], 2),
                    "n_contracts": opt["n_contracts"],
                    "cost": round(opt["cost"], 2),
                    "proceeds": round(proceeds, 2),
                    "pnl": round(pnl, 2),
                    "type": "call_option",
                })
                open_options.remove(opt)

        if date not in convergence.index:
            equity_curve.append({"date": str(date.date()), "equity": round(cash, 2)})
            continue

        conv_row = convergence.loc[date]
        target = [s for s in sectors if conv_row.get(s, 0) >= 4]
        if len(target) > MAX_POSITIONS:
            target.sort(key=lambda s: -conv_row.get(s, 0))
            target = target[:MAX_POSITIONS]

        # Don't re-enter if already have open option on same sector
        active_sectors = {opt["sector"] for opt in open_options}
        target = [s for s in target if s not in active_sectors]

        n_new = len(target)
        if n_new > 0:
            alloc = cash / (n_new + len(open_options) + 1)  # keep some cash
            for s in target:
                if s not in prices.columns or date not in prices[s].index:
                    continue
                underlying = prices[s].loc[date]
                premium = underlying * OPTION_PREMIUM_PCT * (1 + OPTION_SPREAD_HAIRCUT)
                cost_per_contract = premium * 100 + OPTION_COMMISSION
                n_contracts = max(1, int(alloc / cost_per_contract))
                total_cost = n_contracts * cost_per_contract
                if total_cost > cash:
                    n_contracts = max(1, int(cash / cost_per_contract))
                    total_cost = n_contracts * cost_per_contract
                if total_cost > cash:
                    continue

                cash -= total_cost
                # Expiry = 2 weeks out
                expiry = date + pd.Timedelta(days=14)
                open_options.append({
                    "sector": s,
                    "entry_date": str(date.date()),
                    "expiry_date": expiry,
                    "strike": underlying,
                    "n_contracts": n_contracts,
                    "cost": total_cost,
                })

        # Mark to market (approximate: option value = max(0, curr - strike) + time_value)
        option_mtm = 0
        for opt in open_options:
            s = opt["sector"]
            if s in prices.columns and date in prices[s].index:
                curr = prices[s].loc[date]
                intrinsic = max(0, curr - opt["strike"])
                days_left = (pd.Timestamp(opt["expiry_date"]) - date).days
                time_val = opt["cost"] / (opt["n_contracts"] * 100) * (days_left / 14) * 0.5
                option_mtm += (intrinsic + max(0, time_val)) * 100 * opt["n_contracts"]

        equity_curve.append({"date": str(date.date()), "equity": round(cash + option_mtm, 2)})

    # Settle remaining open options at last price
    last_date = dates[-1]
    for opt in open_options:
        s = opt["sector"]
        curr_price = prices[s].iloc[-1] if s in prices.columns else opt["strike"]
        intrinsic = max(0, curr_price - opt["strike"])
        proceeds = intrinsic * 100 * opt["n_contracts"]
        if proceeds > 0:
            proceeds -= OPTION_COMMISSION
        pnl = proceeds - opt["cost"]
        cash += proceeds
        trades.append({
            "sector": s,
            "entry_date": opt["entry_date"],
            "exit_date": str(last_date.date()),
            "strike": round(opt["strike"], 2),
            "n_contracts": opt["n_contracts"],
            "cost": round(opt["cost"], 2),
            "proceeds": round(proceeds, 2),
            "pnl": round(pnl, 2),
            "type": "call_option",
        })

    return {"trades": trades, "equity_curve": equity_curve, "final_equity": round(cash, 2)}


# ── RUN VARIANT ─────────────────────────────────────────────────────────────
def run_variant(prices, convergence, engines, vix, variant, sectors):
    if variant == "E":
        return run_backtest_options(prices, convergence, engines, vix, sectors)
    else:
        return run_backtest_shares(prices, convergence, engines, vix, variant, sectors)


# ── METRICS ─────────────────────────────────────────────────────────────────
def compute_metrics(result: dict, spy: pd.Series) -> dict:
    ec = result["equity_curve"]
    if len(ec) < 2:
        return {"error": "Not enough data points"}

    eq_series = pd.Series(
        [e["equity"] for e in ec],
        index=pd.to_datetime([e["date"] for e in ec]),
    )

    returns = eq_series.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], 0)
    if len(returns) < 5:
        return {"error": "Not enough returns"}

    periods_per_year = 52
    ann_ret = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(periods_per_year) if len(downside) > 2 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 1e-8 else 0

    cumret = (1 + returns).cumprod()
    running_max = cumret.cummax()
    drawdown = (cumret - running_max) / running_max
    max_dd = drawdown.min()

    trades = result["trades"]
    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(wins) / n_trades
        total_win = sum(t["pnl"] for t in wins)
        total_loss = sum(abs(t["pnl"]) for t in losses)
        profit_factor = total_win / total_loss if total_loss > 0 else float("inf")
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0
        total_pnl = sum(t["pnl"] for t in trades)
    else:
        win_rate = avg_win = avg_loss = total_pnl = 0
        profit_factor = 0

    # Regime analysis
    spy_sma200 = spy.rolling(200).mean()
    bull_rets, bear_rets = [], []
    for date, ret in returns.items():
        if date in spy.index and date in spy_sma200.index:
            (bull_rets if spy.loc[date] > spy_sma200.loc[date] else bear_rets).append(ret)

    def _sharpe(rets):
        if len(rets) < 5:
            return 0
        a = np.array(rets)
        return (a.mean() * periods_per_year) / (a.std() * np.sqrt(periods_per_year)) if a.std() > 0 else 0

    bull_sharpe = _sharpe(bull_rets)
    bear_sharpe = _sharpe(bear_rets)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)

    return {
        "n_trades": n_trades,
        "total_pnl": round(total_pnl, 2),
        "final_equity": result["final_equity"],
        "total_return_pct": round((result["final_equity"] / ACCOUNT_SIZE - 1) * 100, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "n_bull_weeks": len(bull_rets),
        "n_bear_weeks": len(bear_rets),
    }


# ── PERMUTATION TEST ────────────────────────────────────────────────────────
def permutation_test(prices, engines, vix, spy, variant, sectors, actual_sharpe, n_perms=N_PERMUTATIONS):
    """Shuffle sector labels to test if convergence signal is real."""
    print(f"  Running {n_perms} permutations for variant {variant}...")
    rng = np.random.RandomState(42)
    count_above = 0

    for i in range(n_perms):
        perm = rng.permutation(len(sectors))
        shuffled_sectors = [sectors[j] for j in perm]

        shuffled_engines = {}
        for name, sig in engines.items():
            s = sig[sectors].copy()
            s.columns = shuffled_sectors
            s = s[sectors]
            shuffled_engines[name] = s

        shuffled_conv = compute_convergence(shuffled_engines, sectors)
        result = run_variant(prices, shuffled_conv, shuffled_engines, vix, variant, sectors)
        metrics = compute_metrics(result, spy)

        if isinstance(metrics.get("sharpe"), (int, float)) and metrics["sharpe"] >= actual_sharpe:
            count_above += 1

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n_perms} done (p so far: {count_above/(i+1):.4f})")

    return count_above / n_perms


# ── 5-GATE VALIDATION ──────────────────────────────────────────────────────
def validate_5gate(metrics: dict, p_value: float) -> dict:
    gates = {
        "sharpe_gt_0.5": metrics.get("sharpe", 0) > 0.5,
        "perm_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": metrics.get("regime_gap", 1) < 0.5,
        "max_dd_gt_neg50": metrics.get("max_drawdown_pct", -100) > -50,
        "n_trades_gte_20": metrics.get("n_trades", 0) >= 20,
    }
    gates["all_pass"] = all(gates.values())
    gates["p_value"] = round(p_value, 4)
    return gates


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    prices = download_data()
    sectors = [s for s in SECTORS if s in prices.columns]
    spy = prices["SPY"]
    vix = prices["^VIX"] if "^VIX" in prices.columns else pd.Series(20, index=prices.index)

    print(f"\nSectors available: {sectors}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"OOT window: {OOT_START} to {OOT_END}")

    engines = compute_engine_signals(prices, sectors)
    convergence = compute_convergence(engines, sectors)

    # Engine signal stats in OOT
    oot_mask = (convergence.index >= OOT_START) & (convergence.index <= OOT_END)
    print("\n── Engine Signal Stats (OOT avg signals/day per sector) ──")
    for name, sig in engines.items():
        avg = sig.loc[oot_mask, sectors].mean().mean()
        print(f"  {name}: {avg:.3f}")

    conv_oot = convergence.loc[oot_mask]
    print(f"\n  Convergence mean across sectors: {conv_oot.mean().mean():.2f}")
    print(f"  Days with >=3 convergence on any sector: {(conv_oot.max(axis=1) >= 3).sum()}")
    print(f"  Days with >=4 convergence on any sector: {(conv_oot.max(axis=1) >= 4).sum()}")

    VARIANT_DESC = {
        "A": "Any-3 Convergence (>=3/6 engines agree)",
        "B": "Any-4 Convergence (>=4/6 engines agree)",
        "C": "Momentum+MeanReversion Combo (both agree)",
        "D": "Top-1 Convergence (highest agreement sector only)",
        "E": "Options on Convergence (>=4 agree, buy ATM calls)",
        "F": "Regime-Filtered (A + VIX<20)",
    }

    variants = list(VARIANT_DESC.keys())
    all_results = {}

    for v in variants:
        print(f"\n{'='*60}")
        print(f"VARIANT {v}: {VARIANT_DESC[v]}")
        print(f"{'='*60}")

        result = run_variant(prices, convergence, engines, vix, v, sectors)
        metrics = compute_metrics(result, spy)

        if "error" in metrics:
            print(f"  ERROR: {metrics['error']}")
            all_results[f"variant_{v}"] = {"description": VARIANT_DESC[v], "metrics": metrics, "validation": {"all_pass": False}}
            continue

        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Final equity: ${metrics['final_equity']:.2f} (started ${ACCOUNT_SIZE})")
        print(f"  Total return: {metrics['total_return_pct']:.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}")
        print(f"  Sortino: {metrics['sortino']:.3f}")
        print(f"  Max DD: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']:.3f} ({metrics['n_bull_weeks']} weeks)")
        print(f"  Bear Sharpe: {metrics['bear_sharpe']:.3f} ({metrics['n_bear_weeks']} weeks)")
        print(f"  Regime Gap: {metrics['regime_gap']:.3f}")

        # Permutation test
        actual_sharpe = metrics["sharpe"]
        if metrics["n_trades"] >= 10:
            p_value = permutation_test(prices, engines, vix, spy, v, sectors, actual_sharpe)
        else:
            p_value = 1.0
            print("  Skipping permutation test (too few trades)")

        gates = validate_5gate(metrics, p_value)
        print(f"\n  5-Gate Validation:")
        for gate, passed in gates.items():
            if gate in ("all_pass", "p_value"):
                continue
            print(f"    {gate}: {'PASS' if passed else 'FAIL'}")
        print(f"    p_value: {gates['p_value']}")
        print(f"    >>> {'ALL GATES PASS' if gates['all_pass'] else 'FAILED'} <<<")

        all_results[f"variant_{v}"] = {
            "description": VARIANT_DESC[v],
            "metrics": metrics,
            "validation": gates,
            "sample_trades": result["trades"][:5],
            "equity_curve_monthly": result["equity_curve"][::4],  # ~monthly
        }

    # ── Summary Table ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Var':<4} {'Sharpe':>7} {'Sort':>7} {'Ret%':>7} {'MaxDD%':>7} {'WR':>6} {'PF':>6} {'#Tr':>5} {'Pass':>5}"
    print(header)
    print("-" * len(header))
    for v in variants:
        r = all_results.get(f"variant_{v}", {})
        m = r.get("metrics", {})
        g = r.get("validation", {})
        if "error" in m:
            print(f"  {v:<3} ERROR: {m['error']}")
            continue
        pf = m.get("profit_factor", 0)
        pf_s = f"{pf:.2f}" if isinstance(pf, (int, float)) else str(pf)
        print(
            f"  {v:<3} {m.get('sharpe',0):>7.3f} {m.get('sortino',0):>7.3f} "
            f"{m.get('total_return_pct',0):>6.1f}% {m.get('max_drawdown_pct',0):>6.1f}% "
            f"{m.get('win_rate',0):>5.1%} {pf_s:>6} {m.get('n_trades',0):>5} "
            f"{'YES' if g.get('all_pass') else 'NO':>5}"
        )

    # ── Save ──
    output = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "account_size": ACCOUNT_SIZE,
            "oot_window": f"{OOT_START} to {OOT_END}",
            "sectors": sectors,
            "n_engines": 6,
            "engine_names": list(engines.keys()),
            "n_permutations": N_PERMUTATIONS,
            "rebalance": "weekly (Friday)",
            "costs": {"slippage_pct": SLIPPAGE_PCT, "commission_shares": 0, "option_commission": OPTION_COMMISSION},
        },
        "results": all_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_PATH}")
    return output


if __name__ == "__main__":
    main()
