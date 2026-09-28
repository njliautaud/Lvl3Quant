#!/usr/bin/env python3
"""
Multi-Asset Trend Following Backtest — CTA-Style
==================================================
Goal: Find return streams UNCORRELATED to long-QQQ (our champion strategy).
Key metric: QQQ_CORRELATION — lower is better.

6 Variants:
A) Classic Dual Momentum
B) Bond-Heavy Trend (TLT/IEF/TIP)
C) Gold Momentum
D) Multi-Asset Risk Parity Trend
E) Volatility Premium Capture
F) Currency Carry + Trend

Period: Jan 2022 – Jul 2026
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── CONSTANTS ───────────────────────────────────────────────────────────
START = "2022-01-01"
END = "2026-07-30"
RF_ANNUAL = 0.045
RF_DAILY = RF_ANNUAL / 252
SLIPPAGE_BPS = 0.0002  # 0.02% per trade
N_PERMUTATIONS = 1000
SEED = 42

# 5-gate thresholds
GATE_SHARPE = 0.5
GATE_PERM_P = 0.05
GATE_REGIME_GAP = 0.5
GATE_MAX_DD = -0.50
GATE_MIN_TRADES = 20


def download_data(tickers: list) -> pd.DataFrame:
    """Download adjusted close prices for all tickers."""
    print(f"Downloading: {', '.join(tickers)}")
    all_tickers = list(set(tickers + ["QQQ", "SPY"]))  # always need QQQ + SPY
    data = yf.download(all_tickers, start=START, end=END, auto_adjust=True, progress=False)
    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data
    close = close.ffill().dropna(how="all")
    return close


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def momentum_return(series: pd.Series, lookback: int) -> pd.Series:
    return series.pct_change(lookback)


def compute_metrics(returns: pd.Series, qqq_returns: pd.Series, spy_prices: pd.Series, name: str) -> dict:
    """Compute all required metrics for a strategy."""
    # Clean
    rets = returns.dropna()
    if len(rets) < 20:
        return {"name": name, "error": "insufficient data"}

    # Basic stats
    total_return = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # Sharpe (excess over risk-free)
    excess = rets - RF_DAILY
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    # Sortino
    downside = excess[excess < 0]
    downside_std = downside.std() if len(downside) > 0 else 1e-10
    sortino = excess.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

    # Max drawdown
    cum = (1 + rets).cumprod()
    rolling_max = cum.cummax()
    dd = (cum - rolling_max) / rolling_max
    max_dd = dd.min()

    # Win rate and profit factor
    winning = rets[rets > 0]
    losing = rets[rets < 0]
    wr = len(winning) / len(rets) if len(rets) > 0 else 0
    gross_profit = winning.sum() if len(winning) > 0 else 0
    gross_loss = abs(losing.sum()) if len(losing) > 0 else 1e-10
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    # Number of trades (signal changes)
    # We'll count this from the positions passed separately

    # QQQ correlation — THE KEY METRIC
    aligned = pd.concat([rets, qqq_returns], axis=1, keys=["strat", "qqq"]).dropna()
    if len(aligned) > 10:
        qqq_corr = aligned["strat"].corr(aligned["qqq"])
    else:
        qqq_corr = np.nan

    # Regime analysis: bull/bear using SPY 200-SMA
    spy_sma200 = sma(spy_prices, 200)
    bull_mask = spy_prices > spy_sma200
    # Align with returns
    bull_aligned = bull_mask.reindex(rets.index).ffill().fillna(True)

    bull_rets = rets[bull_aligned]
    bear_rets = rets[~bull_aligned]

    bull_sharpe = (bull_rets.mean() - RF_DAILY) / bull_rets.std() * np.sqrt(252) if len(bull_rets) > 20 and bull_rets.std() > 0 else 0
    bear_sharpe = (bear_rets.mean() - RF_DAILY) / bear_rets.std() * np.sqrt(252) if len(bear_rets) > 20 and bear_rets.std() > 0 else 0

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.01)

    return {
        "name": name,
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 4),
        "total_return": round(total_return, 4),
        "qqq_correlation": round(qqq_corr, 4),
        "bull_sharpe": round(bull_sharpe, 4),
        "bear_sharpe": round(bear_sharpe, 4),
        "regime_gap": round(regime_gap, 4),
        "n_days": len(rets),
        "ann_vol": round(rets.std() * np.sqrt(252), 4),
    }


def count_trades(positions: pd.Series) -> int:
    """Count number of position changes."""
    changes = positions.diff().fillna(0)
    return int((changes != 0).sum())


def apply_slippage(returns: pd.Series, positions: pd.Series) -> pd.Series:
    """Apply slippage cost on position changes."""
    changes = positions.diff().fillna(0)
    trade_mask = changes != 0
    slippage = pd.Series(0.0, index=returns.index)
    slippage[trade_mask] = SLIPPAGE_BPS
    return returns - slippage


def permutation_test(returns: pd.Series, n_perms: int = N_PERMUTATIONS) -> float:
    """Permutation test: shuffle returns, compute fraction with Sharpe >= observed."""
    rng = np.random.RandomState(SEED)
    rets = returns.dropna().values
    obs_sharpe = (rets.mean() - RF_DAILY) / rets.std() * np.sqrt(252) if rets.std() > 0 else 0

    count_better = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(rets)
        shuf_sharpe = (shuffled.mean() - RF_DAILY) / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
        if shuf_sharpe >= obs_sharpe:
            count_better += 1

    return count_better / n_perms


def five_gate_validation(metrics: dict, perm_p: float, n_trades: int) -> dict:
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gate": metrics.get("sharpe", 0) >= GATE_SHARPE,
        "perm_gate": perm_p < GATE_PERM_P,
        "regime_gate": metrics.get("regime_gap", 1) < GATE_REGIME_GAP,
        "maxdd_gate": metrics.get("max_dd", -1) > GATE_MAX_DD,
        "trades_gate": n_trades >= GATE_MIN_TRADES,
    }
    gates["all_pass"] = all(gates.values())
    gates["perm_p"] = round(perm_p, 4)
    gates["n_trades"] = n_trades
    return gates


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════

def strategy_a_dual_momentum(prices: pd.DataFrame) -> pd.Series:
    """A) Classic Dual Momentum — 12mo momentum, top 2 if above 200-SMA, else SHY."""
    tickers = ["SPY", "TLT", "GLD", "EFA", "VNQ"]
    available = [t for t in tickers if t in prices.columns]

    # 252-day momentum (12 months)
    mom = pd.DataFrame()
    above_sma = pd.DataFrame()
    for t in available:
        mom[t] = momentum_return(prices[t], 252)
        above_sma[t] = prices[t] > sma(prices[t], 200)

    # Daily returns for each asset
    asset_rets = prices[available].pct_change()
    shy_rets = prices["SHY"].pct_change() if "SHY" in prices.columns else pd.Series(RF_DAILY, index=prices.index)

    # For each day: rank by momentum, pick top 2 that are above 200-SMA
    positions = pd.DataFrame(0.0, index=prices.index, columns=available)

    for i in range(253, len(prices)):
        date = prices.index[i]
        prev = prices.index[i - 1]  # LAG 1 DAY — signal from yesterday

        day_mom = mom.loc[prev]
        day_above = above_sma.loc[prev]

        # Filter to above SMA and sort by momentum
        eligible = day_mom[day_above].dropna().sort_values(ascending=False)

        top2 = eligible.head(2).index.tolist()
        if len(top2) > 0:
            weight = 0.5 if len(top2) == 2 else 1.0
            for t in top2:
                positions.loc[date, t] = weight

    # Compute portfolio return
    port_rets = (positions.shift(0) * asset_rets).sum(axis=1)  # positions already lagged
    # Where fully in cash, earn SHY
    cash_weight = 1.0 - positions.sum(axis=1)
    port_rets = port_rets + cash_weight * shy_rets

    # Apply slippage
    total_pos = positions.sum(axis=1)
    port_rets = apply_slippage(port_rets, total_pos)

    return port_rets, positions


def strategy_b_bond_trend(prices: pd.DataFrame) -> pd.Series:
    """B) Bond-Heavy Trend — TLT/IEF/TIP using 50/200 SMA crossover."""
    tickers = ["TLT", "IEF", "TIP"]
    available = [t for t in tickers if t in prices.columns]

    asset_rets = prices[available].pct_change()
    shy_rets = prices["SHY"].pct_change() if "SHY" in prices.columns else pd.Series(RF_DAILY, index=prices.index)

    positions = pd.DataFrame(0.0, index=prices.index, columns=available)
    weight = 1.0 / len(available)

    for t in available:
        sma50 = sma(prices[t], 50)
        sma200 = sma(prices[t], 200)
        # Golden cross: 50 > 200 = long, else cash
        signal = (sma50 > sma200).shift(1)  # LAG 1 DAY
        positions[t] = signal.astype(float) * weight

    port_rets = (positions * asset_rets).sum(axis=1)
    cash_weight = 1.0 - positions.sum(axis=1)
    port_rets = port_rets + cash_weight * shy_rets

    total_pos = positions.sum(axis=1)
    port_rets = apply_slippage(port_rets, total_pos)

    return port_rets, positions


def strategy_c_gold_momentum(prices: pd.DataFrame) -> pd.Series:
    """C) Gold Momentum — GLD long when 20d return > 0 AND above 50-SMA."""
    gld = prices["GLD"]
    gld_rets = gld.pct_change()
    shy_rets = prices["SHY"].pct_change() if "SHY" in prices.columns else pd.Series(RF_DAILY, index=prices.index)

    ret_20d = momentum_return(gld, 20)
    sma50 = sma(gld, 50)

    # Signal: 20d return > 0 AND price > 50-SMA
    signal = ((ret_20d > 0) & (gld > sma50)).shift(1).astype(float)  # LAG 1 DAY

    port_rets = signal * gld_rets + (1 - signal) * shy_rets
    port_rets = apply_slippage(port_rets, signal)

    return port_rets, signal


def strategy_d_risk_parity_trend(prices: pd.DataFrame) -> pd.Series:
    """D) Multi-Asset Risk Parity Trend — Equal risk budget, long if above 200-SMA."""
    tickers = ["SPY", "TLT", "GLD"]
    # DBC might not be available via yfinance for full period, use what we have
    if "DBC" in prices.columns:
        tickers.append("DBC")

    available = [t for t in tickers if t in prices.columns]
    asset_rets = prices[available].pct_change()
    shy_rets = prices["SHY"].pct_change() if "SHY" in prices.columns else pd.Series(RF_DAILY, index=prices.index)

    # Risk parity weights: inverse volatility (60-day rolling)
    rolling_vol = asset_rets.rolling(60, min_periods=30).std()
    inv_vol = 1.0 / rolling_vol.replace(0, np.nan)
    risk_parity_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(0)

    # Trend filter: only long if above 200-SMA
    positions = pd.DataFrame(0.0, index=prices.index, columns=available)
    for t in available:
        above = (prices[t] > sma(prices[t], 200)).shift(1).astype(float)  # LAG 1 DAY
        positions[t] = risk_parity_weights[t] * above

    port_rets = (positions * asset_rets).sum(axis=1)
    cash_weight = 1.0 - positions.sum(axis=1).clip(0, 1)
    port_rets = port_rets + cash_weight * shy_rets

    total_pos = positions.sum(axis=1)
    port_rets = apply_slippage(port_rets, total_pos)

    return port_rets, positions


def strategy_e_vol_premium(prices: pd.DataFrame) -> pd.Series:
    """E) Volatility Premium Capture — proxy contango using 20d vs 60d realized vol on SPY."""
    spy = prices["SPY"]
    spy_rets = spy.pct_change()
    shy_rets = prices["SHY"].pct_change() if "SHY" in prices.columns else pd.Series(RF_DAILY, index=prices.index)

    # Proxy for VIX contango: short-term vol < long-term vol = contango = sell vol
    vol_20d = spy_rets.rolling(20).std() * np.sqrt(252)
    vol_60d = spy_rets.rolling(60).std() * np.sqrt(252)

    # Use SVXY if available, else construct inverse-vol proxy from SPY
    # When contango (short vol < long vol), being long equities captures vol premium
    # We'll use SPY as proxy since SVXY has limited history
    if "SVXY" in prices.columns:
        target_rets = prices["SVXY"].pct_change()
    else:
        # Proxy: leveraged SPY exposure when contango, cash when backwardation
        target_rets = spy_rets * 1.5  # ~1.5x exposure as vol premium proxy

    contango = (vol_20d < vol_60d).shift(1).astype(float)  # LAG 1 DAY

    port_rets = contango * target_rets + (1 - contango) * shy_rets
    port_rets = apply_slippage(port_rets, contango)

    return port_rets, contango


def strategy_f_currency_carry(prices: pd.DataFrame) -> pd.Series:
    """F) Currency Carry + Trend — UUP trend + EEM momentum."""
    shy_rets = prices["SHY"].pct_change() if "SHY" in prices.columns else pd.Series(RF_DAILY, index=prices.index)

    positions = pd.DataFrame(0.0, index=prices.index, columns=["UUP", "EEM"])

    # UUP: long when above 50-SMA, short (via FXE) when below
    if "UUP" in prices.columns:
        uup_sma50 = sma(prices["UUP"], 50)
        uup_signal = (prices["UUP"] > uup_sma50).shift(1)  # LAG 1 DAY
        # +0.5 when above, -0.5 when below (using FXE as inverse proxy)
        positions["UUP"] = uup_signal.map({True: 0.5, False: -0.5}).fillna(0)

    # EEM: momentum — long when 20d return > 0 AND above 50-SMA
    if "EEM" in prices.columns:
        eem_mom = momentum_return(prices["EEM"], 20)
        eem_sma50 = sma(prices["EEM"], 50)
        eem_signal = ((eem_mom > 0) & (prices["EEM"] > eem_sma50)).shift(1).astype(float)
        positions["EEM"] = eem_signal * 0.5

    # Returns
    all_rets = pd.DataFrame(index=prices.index)
    for t in ["UUP", "EEM"]:
        if t in prices.columns:
            all_rets[t] = prices[t].pct_change()
        else:
            all_rets[t] = 0.0

    # For short UUP, use FXE returns (inverse dollar)
    if "FXE" in prices.columns and "UUP" in prices.columns:
        # When short UUP (positions < 0), use FXE long returns
        short_mask = positions["UUP"] < 0
        uup_rets = all_rets["UUP"].copy()
        fxe_rets = prices["FXE"].pct_change()
        # Short UUP ~ long FXE
        effective_uup_rets = uup_rets.copy()
        effective_uup_rets[short_mask] = fxe_rets[short_mask]
        positions.loc[short_mask, "UUP"] = 0.5  # Make positive, using FXE returns
        all_rets["UUP"] = effective_uup_rets

    port_rets = (positions * all_rets).sum(axis=1)
    cash_weight = 1.0 - positions.abs().sum(axis=1).clip(0, 1)
    port_rets = port_rets + cash_weight.clip(0) * shy_rets

    total_pos = positions.abs().sum(axis=1)
    port_rets = apply_slippage(port_rets, total_pos)

    return port_rets, positions


# ═══════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("MULTI-ASSET TREND FOLLOWING BACKTEST — CTA STYLE")
    print(f"Period: {START} to {END}")
    print(f"Risk-free rate: {RF_ANNUAL*100:.1f}% annualized")
    print("=" * 70)

    # Download all needed tickers
    all_tickers = [
        "SPY", "QQQ", "TLT", "IEF", "TIP", "GLD", "EFA", "VNQ",
        "SHY", "DBC", "SVXY", "UUP", "FXE", "EEM"
    ]
    prices = download_data(all_tickers)
    print(f"\nData range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Available tickers: {list(prices.columns)}")

    # QQQ buy-and-hold returns for correlation
    qqq_rets = prices["QQQ"].pct_change().dropna()
    spy_prices = prices["SPY"]

    # Run all strategies
    strategies = {
        "A_Dual_Momentum": strategy_a_dual_momentum,
        "B_Bond_Trend": strategy_b_bond_trend,
        "C_Gold_Momentum": strategy_c_gold_momentum,
        "D_Risk_Parity_Trend": strategy_d_risk_parity_trend,
        "E_Vol_Premium": strategy_e_vol_premium,
        "F_Currency_Carry": strategy_f_currency_carry,
    }

    results = {}

    for name, func in strategies.items():
        print(f"\n{'─' * 50}")
        print(f"Running: {name}")
        try:
            rets, pos = func(prices)

            # Compute metrics
            metrics = compute_metrics(rets, qqq_rets, spy_prices, name)

            # Count trades
            if isinstance(pos, pd.DataFrame):
                n_trades = sum(count_trades(pos[c]) for c in pos.columns)
            else:
                n_trades = count_trades(pos)
            metrics["n_trades"] = n_trades

            # Permutation test
            print(f"  Running {N_PERMUTATIONS} permutations...")
            perm_p = permutation_test(rets, N_PERMUTATIONS)

            # 5-gate validation
            gates = five_gate_validation(metrics, perm_p, n_trades)
            metrics["gates"] = gates

            results[name] = metrics

            print(f"  Sharpe:  {metrics['sharpe']:+.3f}")
            print(f"  Sortino: {metrics['sortino']:+.3f}")
            print(f"  CAGR:    {metrics['cagr']*100:+.1f}%")
            print(f"  MaxDD:   {metrics['max_dd']*100:.1f}%")
            print(f"  WR:      {metrics['win_rate']*100:.1f}%")
            print(f"  PF:      {metrics['profit_factor']:.2f}")
            print(f"  Trades:  {n_trades}")
            print(f"  QQQ_CORR: {metrics['qqq_correlation']:+.3f}  *** KEY METRIC ***")
            print(f"  Bull Sharpe: {metrics['bull_sharpe']:+.3f} | Bear Sharpe: {metrics['bear_sharpe']:+.3f}")
            print(f"  Regime Gap: {metrics['regime_gap']:.3f}")
            print(f"  Perm p: {perm_p:.4f}")
            print(f"  5-Gate: {'PASS' if gates['all_pass'] else 'FAIL'} — {gates}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[name] = {"name": name, "error": str(e)}

    # ── SUMMARY ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — RANKED BY QQQ CORRELATION (LOWER = MORE VALUABLE)")
    print("=" * 70)

    valid = {k: v for k, v in results.items() if "error" not in v}
    ranked = sorted(valid.items(), key=lambda x: abs(x[1].get("qqq_correlation", 1)))

    print(f"\n{'Strategy':<25} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'QQQ_CORR':>9} {'5-Gate':>7}")
    print("-" * 70)
    for name, m in ranked:
        gate_str = "PASS" if m.get("gates", {}).get("all_pass", False) else "FAIL"
        print(f"{name:<25} {m['sharpe']:>+7.3f} {m['sortino']:>+8.3f} {m['cagr']*100:>+6.1f}% {m['max_dd']*100:>6.1f}% {m['qqq_correlation']:>+9.3f} {gate_str:>7}")

    # Best diversifier
    if ranked:
        best = ranked[0]
        print(f"\nBEST DIVERSIFIER: {best[0]}")
        print(f"  QQQ Correlation: {best[1]['qqq_correlation']:+.3f}")
        print(f"  Sharpe: {best[1]['sharpe']:+.3f}")

        # Portfolio analysis: 50/50 QQQ + best diversifier
        print(f"\n  Hypothetical 50/50 blend with QQQ buy-and-hold:")
        qqq_sharpe = (qqq_rets.mean() - RF_DAILY) / qqq_rets.std() * np.sqrt(252) if qqq_rets.std() > 0 else 0
        print(f"  QQQ standalone Sharpe: {qqq_sharpe:+.3f}")

    # Save results
    output = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "period": f"{START} to {END}",
            "risk_free_rate": RF_ANNUAL,
            "slippage_bps": SLIPPAGE_BPS * 10000,
            "n_permutations": N_PERMUTATIONS,
            "purpose": "Find CTA-style strategies uncorrelated to long-QQQ",
        },
        "strategies": results,
        "ranking_by_diversification": [
            {"rank": i + 1, "strategy": name, "qqq_correlation": m["qqq_correlation"], "sharpe": m["sharpe"]}
            for i, (name, m) in enumerate(ranked)
        ],
    }

    out_path = Path("/home/jupiter/Lvl3Quant/data/multi_asset_trend_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("DONE")


if __name__ == "__main__":
    main()
