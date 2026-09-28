#!/usr/bin/env python3
"""
Enhanced Rotation Strategy Backtest
====================================
Tests 6 variants against the validated baseline (GLD/TLT/UUP vol-adj RS, Sharpe 2.02).

Variants:
  A) Expanded Safe Haven Rotation (8 assets)
  B) Weekly Rebalance (GLD/TLT/UUP)
  C) Dual Momentum Rotation (absolute + relative)
  D) Adaptive Lookback (best trailing hit rate)
  E) Risk Parity Weighted (top 2, inverse-vol)
  F) Regime-Adaptive Rotation (growth in bull, safe in bear)

5-gate validation per variant.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

def log(msg):
    print(msg, flush=True)

# ─── CONFIG ──────────────────────────────────────────────────────────────────

START = "2022-01-01"
END = "2026-07-30"
CAPITAL = 669.0
SLIPPAGE_PCT = 0.0002  # 0.02%
N_PERMS = 1000
SEED = 42

# Kill switch: VIX > 20 AND SPY below 50-SMA → cash
KILL_VIX_THRESH = 20
KILL_SMA_LEN = 50

# Validation gates
SHARPE_GATE = 0.5
PERM_P_GATE = 0.05
REGIME_GAP_GATE = 0.50
MAX_DD_GATE = -0.20
MIN_TRADES = 15

# Asset universes
BASELINE_ASSETS = ["GLD", "TLT", "UUP"]
EXPANDED_ASSETS = ["GLD", "TLT", "UUP", "SHY", "AGG", "TIP", "IEF", "BIL"]
REGIME_GROWTH = ["QQQ", "XLK"]
CASH_PROXY = "BIL"

ALL_TICKERS = list(set(BASELINE_ASSETS + EXPANDED_ASSETS + REGIME_GROWTH + ["SPY", "^VIX", CASH_PROXY]))


def download_data():
    """Download all needed price data."""
    log("Downloading price data...")
    data = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data
    close.columns = [c.strip() if isinstance(c, str) else c for c in close.columns]
    close = close.ffill().dropna(how="all")
    log(f"  Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
    return close


def vol_adjusted_rs(close, assets, lookback_days=63):
    """Vol-adjusted relative strength: return / volatility over lookback."""
    ret = close[assets].pct_change()
    mom = ret.rolling(lookback_days).mean() * 252
    vol = ret.rolling(lookback_days).std() * np.sqrt(252)
    rs = mom / vol.replace(0, np.nan)
    return rs


def get_kill_switch(close):
    """Kill switch: VIX > 20 AND SPY below 50-SMA."""
    vix = close.get("^VIX")
    spy = close.get("SPY")
    if vix is None or spy is None:
        return pd.Series(False, index=close.index)
    spy_sma50 = spy.rolling(KILL_SMA_LEN).mean()
    kill = (vix > KILL_VIX_THRESH) & (spy < spy_sma50)
    return kill.fillna(False)


def get_regime(close):
    """Bull = SPY above 200-SMA."""
    spy = close["SPY"]
    sma200 = spy.rolling(200).mean()
    return (spy > sma200).fillna(False)


def get_monthly_dates(index):
    return index.to_series().groupby([index.year, index.month]).last()


def get_weekly_dates(index):
    return index.to_series().groupby([index.year, index.isocalendar().week]).last()


def simulate_rotation(close, signals_df):
    """
    Fast vectorized rotation simulation.
    signals_df: columns = asset tickers, values = portfolio weight.
    Index = rebalance dates.
    Returns: equity Series, trade count.
    """
    all_assets = signals_df.columns.tolist()
    # Filter to assets that exist in close
    valid = [a for a in all_assets if a in close.columns]
    if not valid:
        return pd.Series(CAPITAL, index=close.index), 0

    returns = close[valid].pct_change().fillna(0)
    dates = returns.index

    # Expand signals to daily
    daily_w = signals_df.reindex(columns=valid, fill_value=0).reindex(dates).ffill().fillna(0)

    # Portfolio daily returns
    port_ret = (daily_w * returns).sum(axis=1)

    # Detect rebalances (weight changes)
    w_diff = daily_w.diff().abs().sum(axis=1)
    w_diff.iloc[0] = daily_w.iloc[0].abs().sum()
    rebal_mask = w_diff > 1e-8

    # Slippage cost as fraction of equity
    trade_cost_frac = w_diff * SLIPPAGE_PCT

    # Build equity
    equity = np.zeros(len(dates))
    equity[0] = CAPITAL
    for i in range(1, len(dates)):
        equity[i] = equity[i-1] * (1 + port_ret.iloc[i]) - equity[i-1] * trade_cost_frac.iloc[i]

    n_trades = int(rebal_mask.sum())
    return pd.Series(equity, index=dates), n_trades


def simulate_rotation_fast_returns(returns_matrix, signal_indices, rebal_dates_idx, n_assets):
    """
    Ultra-fast permutation simulation. Returns Sharpe ratio.
    returns_matrix: np array (T, n_assets)
    signal_indices: array of asset index per rebalance date (len = n_rebal)
    rebal_dates_idx: array of integer indices into returns_matrix for each rebalance
    """
    T = returns_matrix.shape[0]
    weights = np.zeros(n_assets)
    equity = CAPITAL
    daily_rets = np.zeros(T)

    rebal_set = set()
    rebal_map = {}
    for i, ridx in enumerate(rebal_dates_idx):
        rebal_set.add(ridx)
        rebal_map[ridx] = i

    for t in range(1, T):
        if t in rebal_set:
            ri = rebal_map[t]
            weights[:] = 0
            ai = signal_indices[ri]
            if ai >= 0:
                weights[ai] = 1.0

        port_ret = np.dot(weights, returns_matrix[t])
        daily_rets[t] = port_ret
        equity = equity * (1 + port_ret)

    # Sharpe
    dr = daily_rets[1:]
    if dr.std() == 0:
        return 0.0
    return (dr.mean() * 252) / (dr.std() * np.sqrt(252))


# ─── STRATEGY IMPLEMENTATIONS ───────────────────────────────────────────────

def strategy_baseline(close, kill):
    assets = BASELINE_ASSETS
    rs = vol_adjusted_rs(close, assets, 63)
    rebal_dates = get_monthly_dates(close.index)
    signals = pd.DataFrame(0.0, index=rebal_dates.values, columns=assets)
    for dt in rebal_dates.values:
        if kill.loc[dt]:
            continue
        scores = rs.loc[dt, assets]
        if scores.isna().all():
            continue
        signals.loc[dt, scores.idxmax()] = 1.0
    return signals


def strategy_A_expanded(close, kill):
    assets = [a for a in EXPANDED_ASSETS if a in close.columns]
    rs = vol_adjusted_rs(close, assets, 63)
    rebal_dates = get_monthly_dates(close.index)
    signals = pd.DataFrame(0.0, index=rebal_dates.values, columns=assets)
    for dt in rebal_dates.values:
        if kill.loc[dt]:
            continue
        scores = rs.loc[dt, assets]
        if scores.isna().all():
            continue
        signals.loc[dt, scores.idxmax()] = 1.0
    return signals


def strategy_B_weekly(close, kill):
    assets = BASELINE_ASSETS
    rs = vol_adjusted_rs(close, assets, 63)
    rebal_dates = get_weekly_dates(close.index)
    signals = pd.DataFrame(0.0, index=rebal_dates.values, columns=assets)
    for dt in rebal_dates.values:
        if kill.loc[dt]:
            continue
        scores = rs.loc[dt, assets]
        if scores.isna().all():
            continue
        signals.loc[dt, scores.idxmax()] = 1.0
    return signals


def strategy_C_dual_momentum(close, kill):
    assets = BASELINE_ASSETS
    rs = vol_adjusted_rs(close, assets, 63)
    ret_3m = close[assets].pct_change(63)
    cash = CASH_PROXY if CASH_PROXY in close.columns else None
    all_cols = assets + ([cash] if cash and cash not in assets else [])
    rebal_dates = get_monthly_dates(close.index)
    signals = pd.DataFrame(0.0, index=rebal_dates.values, columns=all_cols)
    for dt in rebal_dates.values:
        if kill.loc[dt]:
            continue
        scores = rs.loc[dt, assets]
        if scores.isna().all():
            continue
        top = scores.idxmax()
        abs_ret = ret_3m.loc[dt, top] if not pd.isna(ret_3m.loc[dt, top]) else 0
        if abs_ret > 0:
            signals.loc[dt, top] = 1.0
        elif cash and cash in signals.columns:
            signals.loc[dt, cash] = 1.0
    return signals


def strategy_D_adaptive_lookback(close, kill):
    assets = BASELINE_ASSETS
    lookbacks = [21, 42, 63, 126]
    rebal_dates = get_monthly_dates(close.index)
    rs_dict = {lb: vol_adjusted_rs(close, assets, lb) for lb in lookbacks}
    signals = pd.DataFrame(0.0, index=rebal_dates.values, columns=assets)

    for i, dt in enumerate(rebal_dates.values):
        if kill.loc[dt]:
            continue
        best_lb = 63
        if i >= 12:
            hit_rates = {}
            for lb in lookbacks:
                hits = 0
                total = 0
                for j in range(max(0, i-12), i):
                    past_dt = rebal_dates.values[j]
                    scores = rs_dict[lb].loc[past_dt, assets]
                    if scores.isna().all():
                        continue
                    top_pick = scores.idxmax()
                    if j + 1 < len(rebal_dates):
                        next_dt = rebal_dates.values[j+1]
                        if past_dt in close.index and next_dt in close.index:
                            fwd_ret = (close.loc[next_dt, top_pick] / close.loc[past_dt, top_pick]) - 1
                            hits += int(fwd_ret > 0)
                            total += 1
                if total > 0:
                    hit_rates[lb] = hits / total
            if hit_rates:
                best_lb = max(hit_rates, key=hit_rates.get)

        scores = rs_dict[best_lb].loc[dt, assets]
        if scores.isna().all():
            continue
        signals.loc[dt, scores.idxmax()] = 1.0
    return signals


def strategy_E_risk_parity(close, kill):
    assets = BASELINE_ASSETS
    rs = vol_adjusted_rs(close, assets, 63)
    ret = close[assets].pct_change()
    rebal_dates = get_monthly_dates(close.index)
    signals = pd.DataFrame(0.0, index=rebal_dates.values, columns=assets)

    for dt in rebal_dates.values:
        if kill.loc[dt]:
            continue
        scores = rs.loc[dt, assets]
        if scores.isna().all():
            continue
        ranked = scores.dropna().sort_values(ascending=False)
        if len(ranked) < 2:
            signals.loc[dt, ranked.index[0]] = 1.0
            continue
        top2 = ranked.index[:2]
        vol20 = ret.loc[:dt, top2].tail(20).std() * np.sqrt(252)
        inv_vol = 1.0 / vol20.replace(0, np.nan)
        inv_vol = inv_vol.dropna()
        if len(inv_vol) == 0:
            signals.loc[dt, top2[0]] = 1.0
            continue
        weights = inv_vol / inv_vol.sum()
        for asset in weights.index:
            signals.loc[dt, asset] = weights[asset]
    return signals


def strategy_F_regime_adaptive(close, kill):
    safe = BASELINE_ASSETS
    growth = [a for a in REGIME_GROWTH if a in close.columns]
    bull_regime = get_regime(close)
    rebal_dates = get_monthly_dates(close.index)
    all_assets = list(set(safe + growth))
    signals = pd.DataFrame(0.0, index=rebal_dates.values, columns=all_assets)

    for dt in rebal_dates.values:
        if kill.loc[dt]:
            continue
        universe = safe + growth if bull_regime.loc[dt] else safe
        rs = vol_adjusted_rs(close, universe, 63)
        scores = rs.loc[dt, universe]
        if scores.isna().all():
            continue
        signals.loc[dt, scores.idxmax()] = 1.0
    return signals


# ─── METRICS & VALIDATION ───────────────────────────────────────────────────

def compute_metrics(equity):
    rets = equity.pct_change().dropna()
    if len(rets) < 10:
        return {}
    mean_r = rets.mean() * 252
    std_r = rets.std() * np.sqrt(252)
    sharpe = mean_r / std_r if std_r > 0 else 0
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = mean_r / downside if downside > 0 else 0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    monthly = rets.resample("ME").sum()
    wins = monthly[monthly > 0]
    losses = monthly[monthly < 0]
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 999
    wr = len(wins) / len(monthly) if len(monthly) > 0 else 0
    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "profit_factor": round(pf, 2),
        "win_rate": round(wr * 100, 1),
        "final_equity": round(equity.iloc[-1], 2),
        "total_return_pct": round((equity.iloc[-1] / equity.iloc[0] - 1) * 100, 2),
    }


def regime_stratified_sharpe(equity, close):
    bull = get_regime(close)
    rets = equity.pct_change().dropna()
    common = rets.index.intersection(bull.index)
    rets = rets.loc[common]
    bull = bull.loc[common]
    def _sharpe(r):
        if len(r) < 5:
            return 0
        return (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0
    s_bull = _sharpe(rets[bull])
    s_bear = _sharpe(rets[~bull])
    denom = max(abs(s_bull), abs(s_bear))
    gap = abs(s_bull - s_bear) / denom if denom > 0 else 0
    return round(s_bull, 3), round(s_bear, 3), round(gap, 3)


def permutation_test(close, kill, strategy_fn, observed_sharpe, n_perms=N_PERMS):
    """
    Fast permutation test: shuffle which asset gets selected each rebalance.
    Uses vectorized returns for speed.
    """
    rng = np.random.RandomState(SEED)
    orig_signals = strategy_fn(close, kill)
    assets = orig_signals.columns.tolist()
    valid_assets = [a for a in assets if a in close.columns]
    if not valid_assets:
        return 1.0

    returns = close[valid_assets].pct_change().fillna(0)
    dates = returns.index
    returns_np = returns.values  # (T, n_assets)
    n_assets = len(valid_assets)

    rebal_dates = orig_signals.index
    # Map rebalance dates to integer indices in the returns array
    date_to_idx = {d: i for i, d in enumerate(dates)}
    rebal_idx = np.array([date_to_idx.get(d, -1) for d in rebal_dates])
    rebal_idx = rebal_idx[rebal_idx >= 0]
    n_rebal = len(rebal_idx)

    # Check which rebalance dates are killed
    kill_mask = np.array([kill.loc[rebal_dates[i]] if rebal_dates[i] in kill.index else False
                          for i in range(len(rebal_dates))])
    kill_mask = kill_mask[:n_rebal]

    count_beat = 0
    T = len(dates)

    for p in range(n_perms):
        # Random asset selection per rebalance
        rand_picks = rng.randint(0, n_assets, size=n_rebal)
        rand_picks[kill_mask] = -1  # no position when killed

        # Simulate
        weights = np.zeros(n_assets)
        daily_rets = np.zeros(T)

        cur_rebal = 0
        for t in range(1, T):
            # Check if we hit a rebalance date
            if cur_rebal < n_rebal and t >= rebal_idx[cur_rebal]:
                while cur_rebal < n_rebal and rebal_idx[cur_rebal] <= t:
                    weights[:] = 0
                    ai = rand_picks[cur_rebal]
                    if ai >= 0:
                        weights[ai] = 1.0
                    cur_rebal += 1

            daily_rets[t] = np.dot(weights, returns_np[t])

        dr = daily_rets[1:]
        std = dr.std()
        if std > 0:
            rand_sharpe = (dr.mean() * 252) / (std * np.sqrt(252))
            if rand_sharpe >= observed_sharpe:
                count_beat += 1

    return round(count_beat / n_perms, 4)


def validate_5gate(metrics, p_value, regime_gap, n_trades):
    gates = {
        "sharpe_pass": metrics.get("sharpe", 0) > SHARPE_GATE,
        "perm_p_pass": p_value < PERM_P_GATE,
        "regime_gap_pass": regime_gap < REGIME_GAP_GATE,
        "max_dd_pass": metrics.get("max_dd", -100) > MAX_DD_GATE * 100,
        "trades_pass": n_trades >= MIN_TRADES,
    }
    gates["all_pass"] = all(gates.values())
    return gates


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    close = download_data()
    kill = get_kill_switch(close)

    strategies = {
        "Baseline (GLD/TLT/UUP monthly)": strategy_baseline,
        "A) Expanded Safe Haven (8 assets)": strategy_A_expanded,
        "B) Weekly Rebalance (GLD/TLT/UUP)": strategy_B_weekly,
        "C) Dual Momentum (abs+rel)": strategy_C_dual_momentum,
        "D) Adaptive Lookback": strategy_D_adaptive_lookback,
        "E) Risk Parity (top 2)": strategy_E_risk_parity,
        "F) Regime-Adaptive (growth+safe)": strategy_F_regime_adaptive,
    }

    results = {}

    for name, strat_fn in strategies.items():
        log(f"\n{'='*60}")
        log(f"  {name}")
        log(f"{'='*60}")

        signals = strat_fn(close, kill)
        equity, n_trades = simulate_rotation(close, signals)
        metrics = compute_metrics(equity)

        log(f"  Sharpe: {metrics.get('sharpe', 'N/A')}")
        log(f"  Sortino: {metrics.get('sortino', 'N/A')}")
        log(f"  CAGR: {metrics.get('cagr', 'N/A')}%")
        log(f"  Max DD: {metrics.get('max_dd', 'N/A')}%")
        log(f"  PF: {metrics.get('profit_factor', 'N/A')}")
        log(f"  WR: {metrics.get('win_rate', 'N/A')}%")
        log(f"  Final Equity: ${metrics.get('final_equity', 'N/A')}")
        log(f"  Total Return: {metrics.get('total_return_pct', 'N/A')}%")
        log(f"  Trades: {n_trades}")

        s_bull, s_bear, gap = regime_stratified_sharpe(equity, close)
        log(f"  Bull Sharpe: {s_bull}, Bear Sharpe: {s_bear}, Gap: {gap}")

        log(f"  Running permutation test ({N_PERMS} shuffles)...")
        p_val = permutation_test(close, kill, strat_fn, metrics.get("sharpe", 0))
        log(f"  Permutation p-value: {p_val}")

        gates = validate_5gate(metrics, p_val, gap, n_trades)
        log(f"  5-Gate: {'PASS' if gates['all_pass'] else 'FAIL'}")
        for g, v in gates.items():
            if g != "all_pass":
                log(f"    {g}: {'PASS' if v else 'FAIL'}")

        results[name] = {
            "metrics": metrics,
            "regime": {"bull_sharpe": s_bull, "bear_sharpe": s_bear, "gap": gap},
            "permutation_p": p_val,
            "n_trades": n_trades,
            "gates": gates,
        }

    # ─── COMPARISON TABLE ────────────────────────────────────────────────────
    log(f"\n\n{'='*100}")
    log("  COMPARISON TABLE — All Variants vs Baseline")
    log(f"{'='*100}")

    header = f"{'Strategy':<40} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>7} {'PF':>6} {'WR%':>6} {'Trades':>7} {'5-Gate':>7}"
    log(header)
    log("-" * 100)

    baseline_sharpe = results.get("Baseline (GLD/TLT/UUP monthly)", {}).get("metrics", {}).get("sharpe", 0)

    for name, res in results.items():
        m = res["metrics"]
        gate_str = "PASS" if res["gates"]["all_pass"] else "FAIL"
        row = f"{name:<40} {m.get('sharpe',0):>7.3f} {m.get('sortino',0):>8.3f} {m.get('cagr',0):>6.1f}% {m.get('max_dd',0):>6.1f}% {m.get('profit_factor',0):>6.2f} {m.get('win_rate',0):>5.1f}% {res['n_trades']:>7} {gate_str:>7}"
        log(row)

    log(f"\nBaseline Sharpe for comparison: {baseline_sharpe}")
    log("\nVariants that BEAT baseline:")
    any_beat = False
    for name, res in results.items():
        if "Baseline" in name:
            continue
        s = res["metrics"].get("sharpe", 0)
        if s > baseline_sharpe:
            log(f"  {name}: Sharpe {s:.3f} (baseline {baseline_sharpe:.3f}, +{s - baseline_sharpe:.3f})")
            any_beat = True
    if not any_beat:
        log("  None — baseline remains strongest")

    log("\nVariants that PASS all 5 gates:")
    any_pass = False
    for name, res in results.items():
        if res["gates"]["all_pass"]:
            log(f"  {name}: Sharpe {res['metrics']['sharpe']:.3f}")
            any_pass = True
    if not any_pass:
        log("  None passed all 5 gates")

    # ─── SAVE RESULTS ────────────────────────────────────────────────────────
    output_path = Path("/home/jupiter/Lvl3Quant/data/enhanced_rotation_results.json")
    output = {
        "run_date": datetime.now().isoformat(),
        "period": f"{START} to {END}",
        "capital": CAPITAL,
        "slippage_pct": SLIPPAGE_PCT,
        "kill_switch": f"VIX>{KILL_VIX_THRESH} AND SPY<50SMA",
        "validation_gates": {
            "sharpe": f">{SHARPE_GATE}",
            "perm_p": f"<{PERM_P_GATE}",
            "regime_gap": f"<{REGIME_GAP_GATE}",
            "max_dd": f">{MAX_DD_GATE*100}%",
            "min_trades": f">={MIN_TRADES}",
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
