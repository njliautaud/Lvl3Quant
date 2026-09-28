#!/usr/bin/env python3
"""
Crypto Momentum Signal Backtest — 6 Variants
Uses BTC/ETH/COIN price action as leading signals for equity trades.
Walk-forward OOT: Jan 2022 – Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
"""

import json
import warnings
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-01-01"  # extra lookback for indicators
PERM_ITERATIONS = 1000
SEED = 42

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/crypto_momentum_signal_results.json")

# ─── DATA DOWNLOAD ────────────────────────────────────────────────────────────
def download_data():
    tickers = ["BTC-USD", "ETH-USD", "SPY", "QQQ", "COIN", "MSTR"]
    print(f"Downloading {tickers} from {DATA_START} to {OOT_END}...")
    raw = yf.download(tickers, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw[("Close",)]
    close.columns = [c if isinstance(c, str) else c for c in close.columns]

    # Forward fill crypto (trades 24/7, but yfinance gives daily)
    close = close.ffill()

    print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")
    print(f"Non-null counts:\n{close.count()}")
    return close


# ─── REGIME ───────────────────────────────────────────────────────────────────
def compute_regime(spy_close: pd.Series) -> pd.Series:
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    sma200 = spy_close.rolling(200).mean()
    regime = pd.Series("bull", index=spy_close.index)
    regime[spy_close < sma200] = "bear"
    return regime


# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────
def backtest(signal: pd.Series, target_returns: pd.Series, regime: pd.Series,
             oot_start: str = OOT_START) -> dict:
    """
    signal: daily Series with values in {0, 0.5, 1} representing position sizing.
            Signal on day t is applied to day t+1 return (no lookahead).
    target_returns: daily returns of the traded instrument.
    regime: 'bull'/'bear' Series aligned to same index.
    Returns dict of metrics.
    """
    # Align all series
    common = signal.index.intersection(target_returns.index).intersection(regime.index)
    common = common.sort_values()

    sig = signal.reindex(common).fillna(0)
    ret = target_returns.reindex(common).fillna(0)
    reg = regime.reindex(common)

    # OOT filter
    oot_mask = common >= pd.Timestamp(oot_start)
    sig = sig[oot_mask]
    ret = ret[oot_mask]
    reg = reg[oot_mask]

    # Strategy returns: signal on day t → trade on day t+1
    # Shift signal by 1 to avoid lookahead
    pos = sig.shift(1).fillna(0)

    # Apply slippage on position changes
    pos_change = pos.diff().abs().fillna(0)
    slippage = pos_change * SLIPPAGE_PCT

    strat_ret = pos * ret - slippage

    # Count trades (position changes from 0 to nonzero or nonzero to 0)
    pos_binary = (pos > 0).astype(int)
    trade_entries = (pos_binary.diff().abs() > 0).sum()
    n_trades = int(trade_entries)

    # Equity curve
    equity = ACCOUNT_SIZE * (1 + strat_ret).cumprod()

    # Metrics
    ann_factor = 252
    mean_ret = strat_ret.mean() * ann_factor
    std_ret = strat_ret.std() * np.sqrt(ann_factor)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0

    downside = strat_ret[strat_ret < 0].std() * np.sqrt(ann_factor)
    sortino = mean_ret / downside if downside > 0 else 0

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Win rate
    daily_trades = strat_ret[pos > 0]
    win_rate = (daily_trades > 0).mean() if len(daily_trades) > 0 else 0

    # Profit factor
    gross_profit = strat_ret[strat_ret > 0].sum()
    gross_loss = abs(strat_ret[strat_ret < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Regime analysis
    bull_ret = strat_ret[reg == "bull"]
    bear_ret = strat_ret[reg == "bear"]

    sharpe_bull = (bull_ret.mean() * ann_factor) / (bull_ret.std() * np.sqrt(ann_factor)) if len(bull_ret) > 10 and bull_ret.std() > 0 else 0
    sharpe_bear = (bear_ret.mean() * ann_factor) / (bear_ret.std() * np.sqrt(ann_factor)) if len(bear_ret) > 10 and bear_ret.std() > 0 else 0

    regime_gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 0.01)

    # Final equity
    final_equity = equity.iloc[-1] if len(equity) > 0 else ACCOUNT_SIZE
    total_return = (final_equity / ACCOUNT_SIZE - 1) * 100

    # Days in market
    pct_in_market = (pos > 0).mean() * 100

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(win_rate, 4),
        "max_dd": round(max_dd, 4),
        "n_trades": n_trades,
        "total_return_pct": round(total_return, 2),
        "final_equity": round(final_equity, 2),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "pct_in_market": round(pct_in_market, 1),
        "strat_returns": strat_ret,  # for permutation test
        "positions": pos,
        "equity_curve": equity,
    }


# ─── PERMUTATION TEST ────────────────────────────────────────────────────────
def permutation_test(result: dict, n_iter: int = PERM_ITERATIONS) -> float:
    """Shuffle trade entry dates, compute null Sharpe distribution, return p-value."""
    rng = np.random.RandomState(SEED)
    actual_sharpe = result["sharpe"]
    strat_ret = result["strat_returns"]
    pos = result["positions"]

    # Get the actual returns of the target (strat_ret / pos where pos > 0)
    target_ret = strat_ret.copy()
    # We'll shuffle the position series
    pos_vals = pos.values.copy()

    null_sharpes = []
    for _ in range(n_iter):
        rng.shuffle(pos_vals)
        shuffled_pos = pd.Series(pos_vals, index=pos.index)
        # Recompute returns with shuffled positions
        # We need the underlying target returns
        underlying = strat_ret / pos
        underlying = underlying.replace([np.inf, -np.inf], 0).fillna(0)

        shuf_ret = shuffled_pos * underlying
        shuf_mean = shuf_ret.mean() * 252
        shuf_std = shuf_ret.std() * np.sqrt(252)
        shuf_sharpe = shuf_mean / shuf_std if shuf_std > 0 else 0
        null_sharpes.append(shuf_sharpe)

    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= actual_sharpe).mean()
    return round(float(p_value), 4)


# ─── SIGNAL GENERATORS ───────────────────────────────────────────────────────

def signal_A_btc_momentum_qqq(close: pd.DataFrame) -> pd.Series:
    """BTC 10-day momentum > 5% → buy QQQ, < -5% → cash."""
    btc_mom = close["BTC-USD"].pct_change(10)
    signal = pd.Series(0.0, index=close.index)
    signal[btc_mom > 0.05] = 1.0
    signal[btc_mom < -0.05] = 0.0
    # Between -5% and +5%: hold previous position (use ffill logic)
    mask_between = (btc_mom >= -0.05) & (btc_mom <= 0.05)
    signal[mask_between] = np.nan
    signal = signal.ffill().fillna(0)
    return signal


def signal_B_btc_weekend(close: pd.DataFrame) -> pd.Series:
    """BTC weekend return > 2% → buy QQQ on Monday. Otherwise cash."""
    btc = close["BTC-USD"].copy()
    signal = pd.Series(0.0, index=close.index)

    for i in range(1, len(close)):
        dt = close.index[i]
        # Monday = 0
        if dt.dayofweek == 0:
            # Find previous Friday
            prev_fri_idx = i - 1
            while prev_fri_idx >= 0 and close.index[prev_fri_idx].dayofweek > 4:
                prev_fri_idx -= 1
            if prev_fri_idx < 0:
                continue

            fri_date = close.index[prev_fri_idx]
            # BTC weekend return: Monday BTC price vs Friday BTC price
            if pd.notna(btc.iloc[i]) and pd.notna(btc.iloc[prev_fri_idx]) and btc.iloc[prev_fri_idx] > 0:
                weekend_ret = (btc.iloc[i] - btc.iloc[prev_fri_idx]) / btc.iloc[prev_fri_idx]
                if weekend_ret > 0.02:
                    signal.iloc[i] = 1.0
                    # Hold for the week (next 4 trading days)
                    for j in range(i+1, min(i+5, len(close))):
                        if close.index[j].dayofweek < 5:
                            signal.iloc[j] = 1.0

    return signal


def signal_C_btc_equity_divergence(close: pd.DataFrame) -> pd.Series:
    """BTC up >10% 20d but QQQ flat/negative → buy QQQ (catch-up).
       BTC down >10% 20d but QQQ positive → cash (delayed selloff)."""
    btc_mom = close["BTC-USD"].pct_change(20)
    qqq_mom = close["QQQ"].pct_change(20)

    signal = pd.Series(0.0, index=close.index)

    # Bullish divergence: BTC up, QQQ lagging → buy
    bull_div = (btc_mom > 0.10) & (qqq_mom < 0.0)
    signal[bull_div] = 1.0

    # Bearish divergence: BTC down, QQQ still up → cash
    bear_div = (btc_mom < -0.10) & (qqq_mom > 0.0)
    signal[bear_div] = 0.0

    # Neutral: hold previous
    neutral = ~bull_div & ~bear_div
    signal[neutral] = np.nan
    signal = signal.ffill().fillna(0)

    return signal


def signal_D_crypto_fear_greed(close: pd.DataFrame) -> pd.Series:
    """BTC realized vol (10-day) < 30th pctl → calm/greedy → buy QQQ.
       BTC RV > 70th pctl → panic → cash."""
    btc_ret = close["BTC-USD"].pct_change()
    btc_rv = btc_ret.rolling(10).std() * np.sqrt(252)

    # Expanding percentile (use all history up to that point)
    rv_pctl = btc_rv.expanding(min_periods=60).apply(
        lambda x: (x.iloc[-1] <= x).mean() if len(x) > 0 else 0.5, raw=False
    )

    signal = pd.Series(0.0, index=close.index)
    signal[rv_pctl < 0.30] = 1.0   # calm → buy
    signal[rv_pctl > 0.70] = 0.0   # panic → cash
    # In between: hold
    between = (rv_pctl >= 0.30) & (rv_pctl <= 0.70)
    signal[between] = np.nan
    signal = signal.ffill().fillna(0)

    return signal


def signal_E_coin_momentum(close: pd.DataFrame) -> pd.Series:
    """COIN 5-day momentum positive → buy QQQ. Negative → cash.
       COIN only listed since April 2021."""
    coin_mom = close["COIN"].pct_change(5)
    signal = pd.Series(0.0, index=close.index)
    signal[coin_mom > 0] = 1.0
    signal[coin_mom <= 0] = 0.0
    # Where COIN data not available, stay cash
    signal[close["COIN"].isna()] = 0.0
    return signal


def signal_F_multi_crypto(close: pd.DataFrame) -> pd.Series:
    """Both BTC and ETH 10-day positive → full position.
       Only one positive → half position. Both negative → cash."""
    btc_mom = close["BTC-USD"].pct_change(10)
    eth_mom = close["ETH-USD"].pct_change(10)

    signal = pd.Series(0.0, index=close.index)

    both_pos = (btc_mom > 0) & (eth_mom > 0)
    one_pos = ((btc_mom > 0) & (eth_mom <= 0)) | ((btc_mom <= 0) & (eth_mom > 0))
    both_neg = (btc_mom <= 0) & (eth_mom <= 0)

    signal[both_pos] = 1.0
    signal[one_pos] = 0.5
    signal[both_neg] = 0.0

    return signal


# ─── VALIDATION GATES ─────────────────────────────────────────────────────────
def validate(result: dict, p_value: float) -> dict:
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": result["sharpe"] > 0.5,
        "perm_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": result["regime_gap"] < 0.5,
        "maxdd_gt_neg50": result["max_dd"] > -0.50,
        "trades_gte_20": result["n_trades"] >= 20,
    }
    gates["all_pass"] = all(gates.values())
    return gates


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("CRYPTO MOMENTUM SIGNAL BACKTEST — 6 Variants")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Account: ${ACCOUNT_SIZE}")
    print("=" * 80)

    close = download_data()

    qqq_ret = close["QQQ"].pct_change()
    regime = compute_regime(close["SPY"])

    # Buy-and-hold benchmark
    oot_mask = close.index >= pd.Timestamp(OOT_START)
    qqq_oot = close["QQQ"][oot_mask]
    bnh_return = (qqq_oot.iloc[-1] / qqq_oot.iloc[0] - 1) * 100 if len(qqq_oot) > 1 else 0
    bnh_equity = ACCOUNT_SIZE * (1 + qqq_ret[oot_mask]).cumprod()
    bnh_sharpe = (qqq_ret[oot_mask].mean() * 252) / (qqq_ret[oot_mask].std() * np.sqrt(252)) if qqq_ret[oot_mask].std() > 0 else 0
    bnh_peak = bnh_equity.cummax()
    bnh_dd = ((bnh_equity - bnh_peak) / bnh_peak).min()

    print(f"\nBenchmark (QQQ B&H): Return={bnh_return:.1f}%, Sharpe={bnh_sharpe:.3f}, MaxDD={bnh_dd:.3f}")

    # Define variants
    variants = {
        "A_BTC_Momentum_QQQ": {
            "func": signal_A_btc_momentum_qqq,
            "desc": "BTC 10d momentum >5% → buy QQQ, <-5% → cash",
        },
        "B_BTC_Weekend_Signal": {
            "func": signal_B_btc_weekend,
            "desc": "BTC weekend gain >2% → buy QQQ Mon-Fri",
        },
        "C_BTC_Equity_Divergence": {
            "func": signal_C_btc_equity_divergence,
            "desc": "BTC up >10% 20d, QQQ flat → buy catch-up",
        },
        "D_Crypto_Fear_Greed": {
            "func": signal_D_crypto_fear_greed,
            "desc": "BTC low vol → calm → buy QQQ; high vol → cash",
        },
        "E_COIN_Momentum": {
            "func": signal_E_coin_momentum,
            "desc": "COIN 5d momentum positive → buy QQQ",
        },
        "F_Multi_Crypto": {
            "func": signal_F_multi_crypto,
            "desc": "BTC+ETH both positive → full; one → half; none → cash",
        },
    }

    results = {}

    for name, spec in variants.items():
        print(f"\n{'─'*60}")
        print(f"VARIANT {name}: {spec['desc']}")
        print(f"{'─'*60}")

        # Generate signal
        signal = spec["func"](close)

        # Run backtest
        result = backtest(signal, qqq_ret, regime)

        # Permutation test
        print(f"  Running permutation test ({PERM_ITERATIONS} iterations)...")
        p_value = permutation_test(result)

        # Validation gates
        gates = validate(result, p_value)

        # Print results
        print(f"  Sharpe:        {result['sharpe']:>8.3f}  {'PASS' if gates['sharpe_gt_0.5'] else 'FAIL'}")
        print(f"  Sortino:       {result['sortino']:>8.3f}")
        print(f"  Profit Factor: {result['profit_factor']:>8.3f}")
        print(f"  Win Rate:      {result['win_rate']:>8.1%}")
        print(f"  Max DD:        {result['max_dd']:>8.1%}  {'PASS' if gates['maxdd_gt_neg50'] else 'FAIL'}")
        print(f"  N Trades:      {result['n_trades']:>8d}  {'PASS' if gates['trades_gte_20'] else 'FAIL'}")
        print(f"  Total Return:  {result['total_return_pct']:>7.1f}%")
        print(f"  Final Equity:  ${result['final_equity']:>8.2f}")
        print(f"  % In Market:   {result['pct_in_market']:>7.1f}%")
        print(f"  Perm p-value:  {p_value:>8.4f}  {'PASS' if gates['perm_p_lt_0.05'] else 'FAIL'}")
        print(f"  Sharpe Bull:   {result['sharpe_bull']:>8.3f}")
        print(f"  Sharpe Bear:   {result['sharpe_bear']:>8.3f}")
        print(f"  Regime Gap:    {result['regime_gap']:>8.3f}  {'PASS' if gates['regime_gap_lt_0.5'] else 'FAIL'}")
        print(f"  ══ ALL GATES:  {'*** PASS ***' if gates['all_pass'] else 'FAIL'}")

        # Store (without non-serializable data)
        results[name] = {
            "description": spec["desc"],
            "sharpe": result["sharpe"],
            "sortino": result["sortino"],
            "profit_factor": result["profit_factor"],
            "win_rate": result["win_rate"],
            "max_dd": result["max_dd"],
            "n_trades": result["n_trades"],
            "total_return_pct": result["total_return_pct"],
            "final_equity": result["final_equity"],
            "pct_in_market": result["pct_in_market"],
            "perm_p_value": p_value,
            "sharpe_bull": result["sharpe_bull"],
            "sharpe_bear": result["sharpe_bear"],
            "regime_gap": result["regime_gap"],
            "gates": gates,
        }

    # ─── SUMMARY ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY — ALL VARIANTS")
    print("=" * 80)

    survivors = []
    for name, r in results.items():
        status = "SURVIVOR" if r["gates"]["all_pass"] else "KILLED"
        if r["gates"]["all_pass"]:
            survivors.append(name)
        gates_passed = sum(1 for k, v in r["gates"].items() if k != "all_pass" and v)
        print(f"  {name:<30s} Sharpe={r['sharpe']:>6.3f}  DD={r['max_dd']:>7.1%}  "
              f"Trades={r['n_trades']:>3d}  p={r['perm_p_value']:.3f}  "
              f"RGap={r['regime_gap']:.3f}  Gates={gates_passed}/5  [{status}]")

    print(f"\nSurvivors: {len(survivors)}/6")
    if survivors:
        print(f"  → {', '.join(survivors)}")
    else:
        print("  → None survived all 5 gates.")

    print(f"\nBenchmark QQQ B&H: Sharpe={bnh_sharpe:.3f}, Return={bnh_return:.1f}%, MaxDD={bnh_dd:.1%}")

    # ─── SAVE ─────────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "strategy_family": "crypto_momentum_signals",
            "oot_period": f"{OOT_START} to {OOT_END}",
            "account_size": ACCOUNT_SIZE,
            "slippage_pct": SLIPPAGE_PCT,
            "perm_iterations": PERM_ITERATIONS,
            "run_timestamp": datetime.now().isoformat(),
            "benchmark_qqq_sharpe": round(bnh_sharpe, 3),
            "benchmark_qqq_return_pct": round(bnh_return, 1),
            "benchmark_qqq_maxdd": round(bnh_dd, 4),
        },
        "variants": results,
        "survivors": survivors,
        "total_variants": 6,
        "total_survivors": len(survivors),
    }

    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=convert)

    print(f"\nResults saved to {RESULTS_PATH}")
    return output


if __name__ == "__main__":
    main()
