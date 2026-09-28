#!/usr/bin/env python3
"""
HEDGED ROTATION V1 — Does hedging market beta improve sector rotation returns?

We have a validated LGBM sector rotation model (Sharpe 1.87-2.96) that ranks 11 sector
ETFs. This experiment tests whether hedging out (or timing) the beta exposure improves
risk-adjusted returns.

Sector universe: XLK, XLF, XLE, XLV, XLI, XLY, XLC, XLP, XLU, XLRE, XLB
Benchmark: SPY
Starting capital: $645 (Robinhood fractional shares, $0 commission)
OOT period: Jan 2022 – Jul 2026 (walk-forward monthly rebalance)
Ranking signal: 21-day momentum return rank (proxy for LGBM — conservative)

6 VARIANTS TESTED:
  A) Long/Short Rotation — Long top-2, short bottom-2 (monthly rebal)
  B) SPY-Hedged Long — Long top-3, short equal-weight SPY hedge
  C) Dynamic Hedge — Long top-3, hedge ratio = VIX/20 using SPY short
  D) Timing the Beta — Long top-3 when VIX<20, 100% cash when VIX>=20
  E) Regime-Adaptive L/S — Long top-2/short bottom-2 when VIX<20, cash when VIX>=20
  F) Correlation-Timed Hedge — Long top-3, full SPY hedge when 60d avg corr>0.8

5-GATE VALIDATION (HC #428 R1):
  G1: Sharpe > 0.5
  G2: Permutation test p < 0.05 (1000 shuffles)
  G3: Beat random selection baseline
  G4: Regime balance |S_green - S_red|/max < 0.50
  G5: Max drawdown > -50%

Regime classification: SPY daily close-to-close (green = up, red = down)
MLflow logging: localhost:5000
"""
from __future__ import annotations
import json, os, sys, time, warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = ROOT / "output/hedged_rotation_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLC", "XLP", "XLU", "XLRE", "XLB"]
BENCHMARK = "SPY"
ALL_TICKERS = SECTORS + [BENCHMARK]

INITIAL_CAPITAL = 645.0
COMMISSION = 0.0  # Robinhood: $0 for shares

OOT_START = "2022-01-01"
OOT_END = "2026-07-25"
LOOKBACK_MOM = 21  # 21-day momentum for ranking
REBAL_FREQ = "ME"  # Monthly end rebalance

N_PERMUTATIONS = 1000
RANDOM_SEED = 42

# 5-Gate thresholds
GATES = dict(sharpe=0.5, perm_p=0.05, mdd=-0.50, regime_gap=0.50)

# ── Data ────────────────────────────────────────────────────────────────────

def download_data() -> pd.DataFrame:
    """Download daily prices for all tickers + VIX via yfinance."""
    import yfinance as yf

    tickers_dl = ALL_TICKERS + ["^VIX"]
    print(f"Downloading {len(tickers_dl)} tickers from yfinance...")
    # Need data from well before OOT_START for lookback
    start = "2021-01-01"
    raw = yf.download(tickers_dl, start=start, end=OOT_END, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns: (field, ticker)
    close = raw["Close"].copy()
    # Rename ^VIX
    close = close.rename(columns={"^VIX": "VIX"})
    close = close.dropna(how="all")
    close = close.ffill()
    print(f"  Loaded {len(close)} trading days, {close.columns.tolist()}")
    return close


def compute_rankings(close: pd.DataFrame) -> pd.DataFrame:
    """Rank sectors by 21-day momentum return (ascending rank, 1=worst, 11=best)."""
    sector_close = close[SECTORS]
    mom = sector_close.pct_change(LOOKBACK_MOM)
    # Rank: higher return = higher rank number
    ranks = mom.rank(axis=1, ascending=True)
    return ranks


def classify_regime(spy_returns: pd.Series) -> pd.Series:
    """Green = positive day, Red = negative day."""
    regime = pd.Series("flat", index=spy_returns.index)
    regime[spy_returns > 0.001] = "green"
    regime[spy_returns < -0.001] = "red"
    return regime


# ── Portfolio simulation helpers ────────────────────────────────────────────

def simulate_portfolio(weights_series: dict[str, pd.DataFrame],
                       close: pd.DataFrame,
                       capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    """Given a dict of {ticker: weight_series}, simulate daily returns.

    weights_series: dict mapping ticker -> pd.Series of target portfolio weight on rebal dates.
    Between rebal dates, positions drift with prices (buy-and-hold between rebals).

    Returns: pd.DataFrame with columns [date, portfolio_value, daily_return].
    """
    # Get the full date range
    dates = close.loc[OOT_START:OOT_END].index
    if len(dates) == 0:
        raise ValueError("No dates in OOT range")

    # Build daily return matrix for all tickers we hold
    daily_ret = close.pct_change().loc[dates]

    # Determine rebalance dates (month-ends within OOT)
    rebal_dates = close.loc[OOT_START:OOT_END].resample(REBAL_FREQ).last().index
    # Also add the first OOT date
    rebal_dates = rebal_dates.union(pd.DatetimeIndex([dates[0]]))
    rebal_dates = rebal_dates.sort_values()

    portfolio_value = capital
    values = []
    # Current position weights (drift between rebals)
    current_weights = {}

    for i, dt in enumerate(dates):
        if dt in rebal_dates:
            # Rebalance: set target weights for this date
            new_w = {}
            for ticker, ws in weights_series.items():
                if dt in ws.index:
                    new_w[ticker] = ws.loc[dt]
                elif len(ws.loc[:dt]) > 0:
                    new_w[ticker] = ws.loc[:dt].iloc[-1]
                else:
                    new_w[ticker] = 0.0
            current_weights = new_w

        # Daily portfolio return from current weights
        port_ret = 0.0
        for ticker, w in current_weights.items():
            if ticker in daily_ret.columns and not np.isnan(daily_ret.loc[dt, ticker]):
                port_ret += w * daily_ret.loc[dt, ticker]

        portfolio_value *= (1 + port_ret)
        values.append({"date": dt, "value": portfolio_value, "daily_return": port_ret})

    df = pd.DataFrame(values)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


# ── Strategy definitions ────────────────────────────────────────────────────

def strategy_A_long_short(ranks: pd.DataFrame, close: pd.DataFrame) -> dict:
    """Long top-2 ranked sectors, short bottom-2 ranked sectors. Monthly rebal."""
    rebal_dates = ranks.loc[OOT_START:OOT_END].resample(REBAL_FREQ).last().index
    weights = {t: pd.Series(dtype=float) for t in SECTORS}

    for dt in rebal_dates:
        if dt not in ranks.index:
            continue
        r = ranks.loc[dt]
        if r.isna().all():
            continue
        top2 = r.nlargest(2).index.tolist()
        bot2 = r.nsmallest(2).index.tolist()
        for t in SECTORS:
            if t in top2:
                weights[t].loc[dt] = 0.25  # 25% each long (50% total long)
            elif t in bot2:
                weights[t].loc[dt] = -0.25  # 25% each short (50% total short)
            else:
                weights[t].loc[dt] = 0.0

    # Convert to proper Series
    return {t: pd.Series(weights[t], dtype=float) for t in SECTORS}


def strategy_B_spy_hedged(ranks: pd.DataFrame, close: pd.DataFrame) -> dict:
    """Long top-3 sectors equal weight, short equal-weight SPY hedge."""
    rebal_dates = ranks.loc[OOT_START:OOT_END].resample(REBAL_FREQ).last().index
    weights = {t: pd.Series(dtype=float) for t in SECTORS + [BENCHMARK]}

    for dt in rebal_dates:
        if dt not in ranks.index:
            continue
        r = ranks.loc[dt]
        if r.isna().all():
            continue
        top3 = r.nlargest(3).index.tolist()
        for t in SECTORS:
            if t in top3:
                weights[t].loc[dt] = 1.0 / 3.0  # ~33% each
            else:
                weights[t].loc[dt] = 0.0
        # SPY hedge: short 100% to offset the long exposure
        weights[BENCHMARK].loc[dt] = -1.0

    return {t: pd.Series(weights[t], dtype=float) for t in SECTORS + [BENCHMARK]}


def strategy_C_dynamic_hedge(ranks: pd.DataFrame, close: pd.DataFrame) -> dict:
    """Long top-3 sectors. Hedge ratio = VIX/20 using SPY short."""
    rebal_dates = ranks.loc[OOT_START:OOT_END].resample(REBAL_FREQ).last().index
    weights = {t: pd.Series(dtype=float) for t in SECTORS + [BENCHMARK]}

    for dt in rebal_dates:
        if dt not in ranks.index:
            continue
        r = ranks.loc[dt]
        if r.isna().all():
            continue
        top3 = r.nlargest(3).index.tolist()
        for t in SECTORS:
            if t in top3:
                weights[t].loc[dt] = 1.0 / 3.0
            else:
                weights[t].loc[dt] = 0.0
        # Dynamic hedge: SPY short = -1.0 * (VIX/20)
        vix_val = close.loc[:dt, "VIX"].iloc[-1] if "VIX" in close.columns else 20.0
        if np.isnan(vix_val):
            vix_val = 20.0
        hedge_ratio = min(vix_val / 20.0, 2.0)  # Cap at 200% hedge
        weights[BENCHMARK].loc[dt] = -hedge_ratio

    return {t: pd.Series(weights[t], dtype=float) for t in SECTORS + [BENCHMARK]}


def strategy_D_timing_beta(ranks: pd.DataFrame, close: pd.DataFrame) -> dict:
    """Long top-3 sectors when VIX<20, 100% cash when VIX>=20."""
    rebal_dates = ranks.loc[OOT_START:OOT_END].resample(REBAL_FREQ).last().index
    weights = {t: pd.Series(dtype=float) for t in SECTORS}

    for dt in rebal_dates:
        if dt not in ranks.index:
            continue
        r = ranks.loc[dt]
        if r.isna().all():
            continue
        vix_val = close.loc[:dt, "VIX"].iloc[-1] if "VIX" in close.columns else 20.0
        if np.isnan(vix_val):
            vix_val = 20.0

        top3 = r.nlargest(3).index.tolist()
        for t in SECTORS:
            if vix_val < 20 and t in top3:
                weights[t].loc[dt] = 1.0 / 3.0
            else:
                weights[t].loc[dt] = 0.0

    return {t: pd.Series(weights[t], dtype=float) for t in SECTORS}


def strategy_E_regime_adaptive(ranks: pd.DataFrame, close: pd.DataFrame) -> dict:
    """Long top-2 / short bottom-2 when VIX<20. Cash when VIX>=20."""
    rebal_dates = ranks.loc[OOT_START:OOT_END].resample(REBAL_FREQ).last().index
    weights = {t: pd.Series(dtype=float) for t in SECTORS}

    for dt in rebal_dates:
        if dt not in ranks.index:
            continue
        r = ranks.loc[dt]
        if r.isna().all():
            continue
        vix_val = close.loc[:dt, "VIX"].iloc[-1] if "VIX" in close.columns else 20.0
        if np.isnan(vix_val):
            vix_val = 20.0

        top2 = r.nlargest(2).index.tolist()
        bot2 = r.nsmallest(2).index.tolist()
        for t in SECTORS:
            if vix_val < 20:
                if t in top2:
                    weights[t].loc[dt] = 0.25
                elif t in bot2:
                    weights[t].loc[dt] = -0.25
                else:
                    weights[t].loc[dt] = 0.0
            else:
                weights[t].loc[dt] = 0.0

    return {t: pd.Series(weights[t], dtype=float) for t in SECTORS}


def strategy_F_correlation_hedge(ranks: pd.DataFrame, close: pd.DataFrame) -> dict:
    """Long top-3 sectors. Full SPY hedge when 60d avg sector-SPY correlation > 0.8."""
    rebal_dates = ranks.loc[OOT_START:OOT_END].resample(REBAL_FREQ).last().index
    weights = {t: pd.Series(dtype=float) for t in SECTORS + [BENCHMARK]}

    spy_ret = close[BENCHMARK].pct_change()
    sector_ret = close[SECTORS].pct_change()

    for dt in rebal_dates:
        if dt not in ranks.index:
            continue
        r = ranks.loc[dt]
        if r.isna().all():
            continue
        top3 = r.nlargest(3).index.tolist()
        for t in SECTORS:
            if t in top3:
                weights[t].loc[dt] = 1.0 / 3.0
            else:
                weights[t].loc[dt] = 0.0

        # Compute 60d rolling average correlation between sectors and SPY
        lookback_end = dt
        lookback_start = dt - pd.Timedelta(days=90)  # ~60 trading days
        spy_slice = spy_ret.loc[lookback_start:lookback_end].dropna()
        sec_slice = sector_ret.loc[lookback_start:lookback_end].dropna()
        if len(spy_slice) < 30:
            weights[BENCHMARK].loc[dt] = 0.0
            continue
        corrs = sec_slice.corrwith(spy_slice)
        avg_corr = corrs.mean()
        if avg_corr > 0.8:
            weights[BENCHMARK].loc[dt] = -1.0  # Full hedge
        else:
            weights[BENCHMARK].loc[dt] = 0.0

    return {t: pd.Series(weights[t], dtype=float) for t in SECTORS + [BENCHMARK]}


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(pf: pd.DataFrame, regime: pd.Series) -> dict:
    """Compute risk-adjusted metrics for a portfolio DataFrame."""
    rets = pf["daily_return"].values
    vals = pf["value"].values

    # Sharpe (annualized)
    mu = np.mean(rets)
    sd = np.std(rets, ddof=1) if len(rets) > 1 else 1e-9
    sharpe = (mu / sd * np.sqrt(252)) if sd > 1e-9 else 0.0

    # Sortino (annualized)
    neg = rets[rets < 0]
    downside_sd = np.sqrt(np.mean(neg ** 2)) if len(neg) > 0 else 1e-9
    sortino = (mu / downside_sd * np.sqrt(252)) if downside_sd > 1e-9 else 0.0

    # Profit factor
    gross_gains = np.sum(rets[rets > 0])
    gross_losses = np.abs(np.sum(rets[rets < 0]))
    pf_ratio = gross_gains / gross_losses if gross_losses > 1e-9 else float("inf")

    # Win rate
    wr = np.mean(rets > 0) if len(rets) > 0 else 0.0

    # Max drawdown
    cummax = np.maximum.accumulate(vals)
    drawdown = (vals - cummax) / cummax
    mdd = float(np.min(drawdown))

    # CAGR
    n_years = len(rets) / 252
    total_ret = vals[-1] / vals[0] if vals[0] > 0 else 0
    cagr = (total_ret ** (1 / n_years) - 1) if n_years > 0 and total_ret > 0 else 0.0

    # Final value
    final_val = float(vals[-1])

    # Regime-stratified Sharpe (HC #428 R1)
    aligned_regime = regime.reindex(pf.index).dropna()
    common = pf.index.intersection(aligned_regime.index)
    green_rets = rets[np.isin(pf.index, common[aligned_regime[common] == "green"])]
    red_rets = rets[np.isin(pf.index, common[aligned_regime[common] == "red"])]

    def _sharpe(r):
        if len(r) < 5:
            return 0.0
        m = np.mean(r)
        s = np.std(r, ddof=1)
        return (m / s * np.sqrt(252)) if s > 1e-9 else 0.0

    sharpe_green = _sharpe(green_rets)
    sharpe_red = _sharpe(red_rets)

    denom = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / denom if denom > 1e-9 else 0.0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf_ratio, 3),
        "win_rate": round(wr, 4),
        "cagr": round(cagr, 4),
        "mdd": round(mdd, 4),
        "final_value": round(final_val, 2),
        "total_return_pct": round((final_val / INITIAL_CAPITAL - 1) * 100, 2),
        "n_days": len(rets),
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(regime_gap, 4),
        "n_green_days": len(green_rets),
        "n_red_days": len(red_rets),
    }


# ── Permutation test ───────────────────────────────────────────────────────

def permutation_test(pf: pd.DataFrame, n_perms: int = N_PERMUTATIONS) -> float:
    """Shuffle daily returns, compute Sharpe distribution. Return p-value."""
    rng = np.random.RandomState(RANDOM_SEED)
    rets = pf["daily_return"].values.copy()
    real_sharpe = np.mean(rets) / (np.std(rets, ddof=1) + 1e-9) * np.sqrt(252)

    count_ge = 0
    for _ in range(n_perms):
        shuffled = rng.permutation(rets)
        s = np.mean(shuffled) / (np.std(shuffled, ddof=1) + 1e-9) * np.sqrt(252)
        if s >= real_sharpe:
            count_ge += 1
    return count_ge / n_perms


def random_selection_sharpe(close: pd.DataFrame, n_runs: int = 200) -> float:
    """Baseline: random sector selection (same structure as top-3 long)."""
    rng = np.random.RandomState(RANDOM_SEED)
    sharpes = []
    dates = close.loc[OOT_START:OOT_END].index
    daily_ret = close[SECTORS].pct_change().loc[dates]

    for _ in range(n_runs):
        rebal_dates = close.loc[OOT_START:OOT_END].resample(REBAL_FREQ).last().index
        port_rets = []
        current_picks = rng.choice(SECTORS, 3, replace=False)
        pick_idx = 0

        for dt in dates:
            if dt in rebal_dates:
                current_picks = rng.choice(SECTORS, 3, replace=False)
            r = daily_ret.loc[dt, current_picks].mean()
            if not np.isnan(r):
                port_rets.append(r)

        if len(port_rets) > 10:
            arr = np.array(port_rets)
            s = np.mean(arr) / (np.std(arr, ddof=1) + 1e-9) * np.sqrt(252)
            sharpes.append(s)

    return float(np.median(sharpes))


# ── 5-Gate validation ───────────────────────────────────────────────────────

def five_gate_validation(metrics: dict, perm_p: float, random_sharpe: float) -> dict:
    """Apply 5-gate validation. Returns dict with pass/fail per gate."""
    gates = {
        "G1_sharpe_gt_0.5": metrics["sharpe"] > GATES["sharpe"],
        "G2_perm_p_lt_0.05": perm_p < GATES["perm_p"],
        "G3_beat_random": metrics["sharpe"] > random_sharpe,
        "G4_regime_balanced": metrics["regime_gap"] < GATES["regime_gap"],
        "G5_mdd_gt_neg50pct": metrics["mdd"] > GATES["mdd"],
    }
    gates["all_pass"] = all(gates.values())
    return gates


# ── MLflow logging ──────────────────────────────────────────────────────────

def log_to_mlflow(variant: str, metrics: dict, gates: dict, perm_p: float):
    """Best-effort MLflow logging."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("hedged_rotation_v1")
        with mlflow.start_run(run_name=f"variant_{variant}"):
            mlflow.log_param("variant", variant)
            mlflow.log_param("initial_capital", INITIAL_CAPITAL)
            mlflow.log_param("oot_start", OOT_START)
            mlflow.log_param("oot_end", OOT_END)
            mlflow.log_param("lookback_momentum", LOOKBACK_MOM)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            mlflow.log_metric("perm_p_value", perm_p)
            for k, v in gates.items():
                mlflow.log_metric(f"gate_{k}", int(v))
            print(f"  [MLflow] Logged variant {variant}")
    except Exception as e:
        print(f"  [MLflow] Warning: {e}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 80)
    print("HEDGED ROTATION V1 — Beta Hedging Analysis")
    print("=" * 80)

    # 1. Download data
    close = download_data()

    # 2. Compute rankings
    ranks = compute_rankings(close)
    print(f"\nRankings computed: {ranks.loc[OOT_START:OOT_END].shape[0]} OOT days")

    # 3. Regime classification
    spy_ret = close[BENCHMARK].pct_change()
    regime = classify_regime(spy_ret)
    n_green = (regime == "green").sum()
    n_red = (regime == "red").sum()
    print(f"Regime: {n_green} green, {n_red} red, {(regime == 'flat').sum()} flat days")

    # 4. Random baseline
    print("\nComputing random selection baseline (200 runs)...")
    random_sharpe = random_selection_sharpe(close)
    print(f"  Random baseline Sharpe: {random_sharpe:.3f}")

    # 5. Run all strategies
    strategies = {
        "A_LongShort": strategy_A_long_short,
        "B_SPYHedged": strategy_B_spy_hedged,
        "C_DynamicHedge": strategy_C_dynamic_hedge,
        "D_TimingBeta": strategy_D_timing_beta,
        "E_RegimeAdaptive": strategy_E_regime_adaptive,
        "F_CorrHedge": strategy_F_correlation_hedge,
    }

    all_results = {}
    print("\n" + "=" * 80)
    print("RUNNING 6 STRATEGY VARIANTS")
    print("=" * 80)

    for name, strat_fn in strategies.items():
        print(f"\n{'─' * 60}")
        print(f"Strategy {name}")
        print(f"{'─' * 60}")

        # Generate weights
        weights = strat_fn(ranks, close)

        # Simulate
        pf = simulate_portfolio(weights, close)

        # Metrics
        metrics = compute_metrics(pf, regime)

        # Permutation test
        print("  Running permutation test (1000 shuffles)...")
        perm_p = permutation_test(pf)

        # 5-gate validation
        gates = five_gate_validation(metrics, perm_p, random_sharpe)

        # Print results
        print(f"  Sharpe:        {metrics['sharpe']:>8.3f}")
        print(f"  Sortino:       {metrics['sortino']:>8.3f}")
        print(f"  Profit Factor: {metrics['profit_factor']:>8.3f}")
        print(f"  Win Rate:      {metrics['win_rate']:>8.1%}")
        print(f"  CAGR:          {metrics['cagr']:>8.2%}")
        print(f"  MDD:           {metrics['mdd']:>8.2%}")
        print(f"  Final Value:   ${metrics['final_value']:>8.2f} (from ${INITIAL_CAPITAL})")
        print(f"  Total Return:  {metrics['total_return_pct']:>8.2f}%")
        print(f"  Sharpe (green): {metrics['sharpe_green']:>7.3f}  Sharpe (red): {metrics['sharpe_red']:>7.3f}  Gap: {metrics['regime_gap']:.4f}")
        print(f"  Perm p-value:  {perm_p:.4f}")
        print(f"  Random Sharpe: {random_sharpe:.3f}")
        print(f"\n  5-Gate Validation:")
        for g, v in gates.items():
            status = "PASS" if v else "FAIL"
            print(f"    {g}: {status}")

        # MLflow
        log_to_mlflow(name, metrics, gates, perm_p)

        # Save per-variant equity curve
        pf.to_csv(OUT_DIR / f"equity_{name}.csv")

        all_results[name] = {
            "metrics": metrics,
            "perm_p": round(perm_p, 4),
            "random_sharpe": round(random_sharpe, 3),
            "gates": gates,
        }

    # 6. Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    hdr = f"{'Variant':<22} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'CAGR':>7} {'MDD':>7} {'Final$':>8} {'RegGap':>7} {'PermP':>6} {'5-Gate':>7}"
    print(hdr)
    print("-" * len(hdr))
    for name, res in all_results.items():
        m = res["metrics"]
        g = res["gates"]
        verdict = "PASS" if g["all_pass"] else "FAIL"
        print(f"{name:<22} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.2f} "
              f"{m['win_rate']:>6.1%} {m['cagr']:>7.2%} {m['mdd']:>7.2%} "
              f"${m['final_value']:>7.2f} {m['regime_gap']:>7.4f} {res['perm_p']:>6.4f} {verdict:>7}")

    # 7. SPY buy-and-hold benchmark
    print(f"\n{'─' * 60}")
    print("BENCHMARK: SPY Buy-and-Hold")
    spy_oot = close[BENCHMARK].loc[OOT_START:OOT_END].dropna()
    spy_ret_oot = spy_oot.pct_change().dropna()
    spy_vals = INITIAL_CAPITAL * (1 + spy_ret_oot).cumprod()
    spy_sharpe = float(spy_ret_oot.mean() / spy_ret_oot.std() * np.sqrt(252))
    spy_final = float(spy_vals.iloc[-1])
    spy_mdd = float(((spy_vals - spy_vals.cummax()) / spy_vals.cummax()).min())
    print(f"  SPY Sharpe: {spy_sharpe:.3f}, Final: ${spy_final:.2f}, MDD: {spy_mdd:.2%}")

    # 8. Save all results
    out_path = OUT_DIR / "results_summary.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "initial_capital": INITIAL_CAPITAL,
                "oot_start": OOT_START,
                "oot_end": OOT_END,
                "sectors": SECTORS,
                "lookback_momentum": LOOKBACK_MOM,
            },
            "spy_benchmark": {
                "sharpe": round(spy_sharpe, 3),
                "final_value": round(spy_final, 2),
                "mdd": round(spy_mdd, 4),
            },
            "random_baseline_sharpe": round(random_sharpe, 3),
            "variants": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # 9. Headline verdicts
    print("\n" + "=" * 80)
    print("HEADLINE VERDICTS")
    print("=" * 80)
    passing = [n for n, r in all_results.items() if r["gates"]["all_pass"]]
    failing = [n for n, r in all_results.items() if not r["gates"]["all_pass"]]
    if passing:
        print(f"  PASSED 5-gate: {', '.join(passing)}")
    if failing:
        print(f"  FAILED 5-gate: {', '.join(failing)}")

    best = max(all_results.items(), key=lambda x: x[1]["metrics"]["sharpe"])
    print(f"  Best risk-adjusted: {best[0]} (Sharpe {best[1]['metrics']['sharpe']:.3f})")
    print(f"  Does hedging help? Compare A (L/S) vs D (timing) vs raw rotation (random baseline)")


if __name__ == "__main__":
    main()
