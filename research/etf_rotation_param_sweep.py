"""
ETF Rotation v3 — Parameter Sensitivity Sweep

Sweeps train_window × hold_period grid (5×5 = 25 configs) to validate
whether 252d/21d is robust or fragile.

Walk-forward SLIDING window only (HC #0).
Anti-concentration rules applied in all configs (max streak 3, min 3 sectors).
5 bps round-trip transaction costs per leg.

Optimization: uses Ridge regression (fast) for the sweep since we're comparing
relative performance across configs, not absolute alpha. The ranking of configs
is stable between Ridge and LGBM — confirmed in v3 development.

Usage:
    python3 research/etf_rotation_param_sweep.py
"""
from __future__ import annotations

import json
import time
import warnings
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = ROOT / "output/etf_rotation_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
UNIVERSE = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
BENCH = "SPY"
K = 3  # top-K sectors to hold
TXN_COST_BPS = 5.0
TRADING_DAYS_YR = 252

# Anti-concentration (HC #670 R2)
HOLD_DECAY = {0: 1.0, 1: 1.0, 2: 0.6, 3: 0.0}
MAX_STREAK = 3

# Sweep grid
TRAIN_WINDOWS = [126, 189, 252, 378, 504]  # trading days
HOLD_PERIODS = [10, 15, 21, 42, 63]  # trading days


# ---------------------------------------------------------------------------
# DATA DOWNLOAD
# ---------------------------------------------------------------------------
def download_data() -> pd.DataFrame:
    """Download ~10+ years of daily data for sector ETFs + SPY via yfinance."""
    import yfinance as yf

    tickers = UNIVERSE + [BENCH]
    start = "2014-01-01"
    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    print(f"[sweep] Downloading {len(tickers)} tickers from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        raise RuntimeError("yfinance returned empty data")

    rows = []
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw[("Close", t)].dropna()
            else:
                closes = raw["Close"].dropna()
            for dt, px in closes.items():
                rows.append({"ticker": t, "date": pd.Timestamp(dt).normalize(),
                             "close": float(px)})
        except Exception as e:
            print(f"  Warning: {t} failed: {e}")
            continue

    df = pd.DataFrame(rows)
    df = df.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last").reset_index(drop=True)
    print(f"[sweep] Downloaded {len(df)} rows, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------
FEATURES = [
    "ret_20d", "ret_60d", "rel_strength_spy", "momentum_cross_20_60",
    "rs_rank_among_sectors", "rs_acceleration_10d", "rs_acceleration_20d",
    "ret_20d_chg_10d", "cross_sector_dispersion", "rank_change_10d",
]


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build rotation-quality features for all ETFs."""
    spy = prices[prices["ticker"] == BENCH].sort_values("date").set_index("date")["close"]
    spy_r20 = spy.pct_change(20)
    spy_r60 = spy.pct_change(60)

    rows = []
    for t in UNIVERSE:
        s = prices[prices["ticker"] == t].sort_values("date").copy()
        if s.empty:
            continue
        s = s.set_index("date")
        s["ret_1d"] = s["close"].pct_change()
        s["ret_20d"] = s["close"].pct_change(20)
        s["ret_60d"] = s["close"].pct_change(60)
        s["sma20"] = s["close"].rolling(20, min_periods=10).mean()
        s["sma60"] = s["close"].rolling(60, min_periods=30).mean()
        s["momentum_cross_20_60"] = (s["sma20"] / s["sma60"]) - 1.0
        s["rel_strength_spy"] = s["ret_60d"] - spy_r60.reindex(s.index)

        # Rotation-timing features
        rs = s["ret_20d"] - spy_r20.reindex(s.index)
        s["rs_acceleration_10d"] = rs.diff(10)
        s["rs_acceleration_20d"] = rs.diff(20)
        s["ret_20d_chg_10d"] = s["ret_20d"].diff(10)

        s["ticker"] = t
        rows.append(s.reset_index())

    panel = pd.concat(rows, ignore_index=True)

    # Cross-sectional features
    panel["rs_rank_among_sectors"] = panel.groupby("date")["ret_20d"].rank(pct=True)
    panel["rank_change_10d"] = panel.groupby("ticker")["rs_rank_among_sectors"].diff(10)

    # Cross-sector dispersion
    disp = panel.groupby("date")["ret_20d"].std().rename("cross_sector_dispersion")
    panel = panel.merge(disp.reset_index(), on="date", how="left")

    return panel


# ---------------------------------------------------------------------------
# MODEL (Ridge — fast for sweep)
# ---------------------------------------------------------------------------
def _fit_ridge(X_tr, y_tr, X_oot, alpha=1.0):
    """Ridge regression — O(p^2 * n) complexity, instant for our feature set."""
    X_mean = X_tr.mean(axis=0)
    y_mean = y_tr.mean()
    Xc = X_tr - X_mean
    yc = y_tr - y_mean
    XtX = Xc.T @ Xc
    Xty = Xc.T @ yc
    try:
        beta = np.linalg.solve(XtX + alpha * np.eye(X_tr.shape[1]), Xty)
        intercept = y_mean - X_mean @ beta
        return X_oot @ beta + intercept
    except np.linalg.LinAlgError:
        return np.zeros(len(X_oot))


def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def _xs_zscore_full(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Cross-sectional z-score the full panel once (pre-computation)."""
    out = panel.copy()
    for f in feats:
        if f not in out.columns:
            out[f] = 0.0
            continue
        x = pd.to_numeric(out[f], errors="coerce").astype(float)
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        x = x.clip(lower=lo, upper=hi)
        out[f] = x
        mu = out.groupby("date")[f].transform("mean")
        sd = out.groupby("date")[f].transform("std")
        z = (x - mu) / sd.replace(0.0, np.nan)
        out[f] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# SINGLE CONFIG BACKTEST
# ---------------------------------------------------------------------------
def run_single_config(panel_z: pd.DataFrame, feats: list[str],
                      train_window: int, hold_period: int) -> dict:
    """
    Run walk-forward backtest for one (train_window, hold_period) config.
    Uses Ridge regression for speed. Panel is pre-z-scored.
    """
    # Build forward returns for this hold_period
    p = panel_z.sort_values(["ticker", "date"]).copy()
    p["y_fwd"] = p.groupby("ticker")["close"].shift(-hold_period) / p["close"] - 1.0

    # Get all unique dates
    all_dates = sorted(p["date"].unique())
    n_dates = len(all_dates)

    # Need enough data
    if n_dates < train_window + hold_period + 80:
        return {"error": "insufficient data"}

    # Walk-forward: rebalance every hold_period days
    start_idx = 70 + train_window
    rebal_indices = list(range(start_idx, n_dates - hold_period, hold_period))

    daily_returns = []
    hold_streak = defaultdict(int)
    prev_picks = set()
    n_rebalances = 0
    turnover_list = []

    for ri in rebal_indices:
        rebal_date = all_dates[ri]
        train_start_date = all_dates[ri - train_window]

        # Train data (already z-scored)
        train = p[(p["date"] >= train_start_date) & (p["date"] < rebal_date)]
        train = train.dropna(subset=["y_fwd"])

        if len(train) < 100:
            hold_end = min(ri + hold_period, n_dates)
            for hi in range(ri, hold_end):
                daily_returns.append((all_dates[hi], 0.0))
            continue

        # Snapshot for scoring
        snap = p[p["date"] == rebal_date]
        if len(snap) < K:
            hold_end = min(ri + hold_period, n_dates)
            for hi in range(ri, hold_end):
                daily_returns.append((all_dates[hi], 0.0))
            continue

        X_tr = train[feats].values
        y_tr = train["y_fwd"].values
        X_snap = snap[feats].values

        # Ridge predict
        scores = _fit_ridge(X_tr, y_tr, X_snap)

        snap_scored = snap.copy()
        snap_scored["score"] = scores

        # Anti-concentration: hold decay
        snap_scored["consec_holds"] = snap_scored["ticker"].map(
            lambda e: hold_streak.get(e, 0))
        snap_scored["adj_score"] = snap_scored.apply(
            lambda row: (
                -999.0 if row["consec_holds"] >= MAX_STREAK
                else row["score"] * HOLD_DECAY.get(
                    min(int(row["consec_holds"]), MAX_STREAK), 0.0)
            ), axis=1)

        # Pick top-K
        top = snap_scored.nlargest(K, "adj_score")
        picks = set(top["ticker"].tolist())

        # Update hold streaks
        new_streak = defaultdict(int)
        for etf in UNIVERSE:
            if etf in picks:
                new_streak[etf] = hold_streak.get(etf, 0) + 1
            else:
                new_streak[etf] = 0
        hold_streak = new_streak

        # Turnover
        if prev_picks:
            changed = len(picks - prev_picks) + len(prev_picks - picks)
            total = len(picks) + len(prev_picks)
            turnover_list.append(changed / total if total > 0 else 0)
        prev_picks = picks
        n_rebalances += 1

        # Transaction cost (on rebal day)
        n_legs = len(picks)
        tc = n_legs * (TXN_COST_BPS / 10000.0)

        # Hold period returns: equal-weight picked sectors
        hold_end = min(ri + hold_period, n_dates)
        for hi in range(ri, hold_end):
            d = all_dates[hi]
            day_rets = []
            for t in picks:
                mask = (p["ticker"] == t) & (p["date"] == d)
                r_vals = p.loc[mask, "ret_1d"]
                if not r_vals.empty:
                    r = float(r_vals.iloc[0])
                    if np.isfinite(r):
                        day_rets.append(r)

            port_ret = np.mean(day_rets) if day_rets else 0.0

            # TC on first day only
            if hi == ri:
                port_ret -= tc

            daily_returns.append((d, port_ret))

    if not daily_returns:
        return {"error": "no returns generated"}

    # Aggregate (take last for overlapping days)
    df = pd.DataFrame(daily_returns, columns=["date", "ret"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.groupby("date")["ret"].sum().sort_index()

    # Compute metrics
    metrics = _compute_metrics(df)
    metrics["n_rebalances"] = n_rebalances
    metrics["avg_turnover"] = float(np.mean(turnover_list)) if turnover_list else 0.0
    metrics["train_window"] = train_window
    metrics["hold_period"] = hold_period

    return metrics


def _compute_metrics(rets: pd.Series) -> dict:
    """Compute Sharpe, Sortino, CAGR, MaxDD from daily returns."""
    rets = rets.dropna()
    if len(rets) < 20:
        return {"sharpe": np.nan, "sortino": np.nan, "cagr": np.nan,
                "max_dd": np.nan}

    mu = float(rets.mean())
    sd = float(rets.std(ddof=1))
    sharpe = (mu / sd * np.sqrt(TRADING_DAYS_YR)) if sd > 0 else 0.0

    downside = rets[rets < 0]
    ds_std = float(downside.std(ddof=1)) if len(downside) > 5 else sd
    sortino = (mu / ds_std * np.sqrt(TRADING_DAYS_YR)) if ds_std > 0 else 0.0

    cum = (1 + rets).cumprod()
    n_years = len(rets) / TRADING_DAYS_YR
    if n_years > 0 and cum.iloc[-1] > 0:
        cagr = (cum.iloc[-1]) ** (1 / n_years) - 1.0
    else:
        cagr = 0.0

    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = float(dd.min())

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2),
    }


# ---------------------------------------------------------------------------
# MAIN SWEEP
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 70)
    print("ETF ROTATION v3 — PARAMETER SENSITIVITY SWEEP")
    print("=" * 70)
    print(f"Grid: train_windows={TRAIN_WINDOWS} x hold_periods={HOLD_PERIODS}")
    print(f"Total configs: {len(TRAIN_WINDOWS) * len(HOLD_PERIODS)}")
    print(f"Universe: {len(UNIVERSE)} sector ETFs, K={K}, cost={TXN_COST_BPS}bps")
    print(f"Anti-concentration: max_streak={MAX_STREAK}, hold_decay={HOLD_DECAY}")
    print(f"Model: Ridge regression (fast sweep mode)")
    print()

    # Download data
    prices = download_data()

    # Build features
    print("[sweep] Building features...")
    panel = build_features(prices)
    print(f"[sweep] Panel: {len(panel)} rows, "
          f"{panel['date'].min().date()} -> {panel['date'].max().date()}")

    # Pre-z-score entire panel (the expensive step, do it ONCE)
    feats = [f for f in FEATURES if f in panel.columns]
    print(f"[sweep] Pre-computing z-scores for {len(feats)} features...")
    panel_z = _xs_zscore_full(panel, feats)
    print(f"[sweep] Z-score done.")

    # Run sweep
    results = []
    total = len(TRAIN_WINDOWS) * len(HOLD_PERIODS)
    for i, (tw, hp) in enumerate(product(TRAIN_WINDOWS, HOLD_PERIODS)):
        t1 = time.time()
        print(f"\n[sweep] Config {i+1}/{total}: train={tw}d, hold={hp}d ...", end=" ")
        r = run_single_config(panel_z, feats, tw, hp)
        dt = time.time() - t1
        if "error" in r:
            print(f"ERROR: {r['error']} ({dt:.1f}s)")
            r.update({"train_window": tw, "hold_period": hp,
                      "sharpe": np.nan, "sortino": np.nan,
                      "cagr": np.nan, "max_dd": np.nan,
                      "n_rebalances": 0, "avg_turnover": 0.0})
        else:
            print(f"Sharpe={r['sharpe']:.3f} Sortino={r['sortino']:.3f} "
                  f"CAGR={r['cagr']:.1f}% MaxDD={r['max_dd']:.1f}% "
                  f"Rebals={r['n_rebalances']} Turn={r['avg_turnover']:.0%} ({dt:.1f}s)")
        results.append(r)

    # SPY benchmark
    spy_prices = prices[prices["ticker"] == BENCH].sort_values("date").set_index("date")["close"]
    spy_ret = spy_prices.pct_change().dropna()
    # Align to conservative start (after max train window warmup)
    panel_dates = sorted(panel["date"].unique())
    start_idx = max(TRAIN_WINDOWS) + 70
    if start_idx < len(panel_dates):
        bench_start = panel_dates[start_idx]
        spy_ret = spy_ret[spy_ret.index >= bench_start]
    spy_metrics = _compute_metrics(spy_ret)

    # Results table
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("sharpe", ascending=False).reset_index(drop=True)

    elapsed = time.time() - t0
    print("\n\n" + "=" * 80)
    print("SWEEP RESULTS — SORTED BY SHARPE")
    print("=" * 80)
    print(f"\n{'Rank':<5} {'Train':<7} {'Hold':<6} {'Sharpe':<8} {'Sortino':<9} "
          f"{'CAGR%':<8} {'MaxDD%':<8} {'Rebals':<8} {'Turnover':<10}")
    print("-" * 80)

    for idx, row in df_results.iterrows():
        rank = idx + 1
        marker = " <-- CURRENT" if (row["train_window"] == 252 and row["hold_period"] == 21) else ""
        print(f"{rank:<5} {int(row['train_window']):<7} {int(row['hold_period']):<6} "
              f"{row['sharpe']:<8.3f} {row['sortino']:<9.3f} "
              f"{row['cagr']:<8.1f} {row['max_dd']:<8.1f} "
              f"{int(row['n_rebalances']):<8} {row['avg_turnover']:<10.1%}{marker}")

    print("-" * 80)
    print(f"{'SPY':<5} {'B&H':<7} {'':<6} {spy_metrics['sharpe']:<8.3f} "
          f"{spy_metrics['sortino']:<9.3f} {spy_metrics['cagr']:<8.1f} "
          f"{spy_metrics['max_dd']:<8.1f}")

    # Robustness analysis
    best_sharpe = df_results["sharpe"].max()
    current = df_results[(df_results["train_window"] == 252) &
                         (df_results["hold_period"] == 21)]

    print("\n\n" + "=" * 70)
    print("ROBUSTNESS ANALYSIS")
    print("=" * 70)

    gap = np.nan
    if not current.empty:
        current_sharpe = float(current.iloc[0]["sharpe"])
        gap = best_sharpe - current_sharpe
        current_rank = int(current.index[0]) + 1
        print(f"\n  Current config (252d/21d): Sharpe = {current_sharpe:.3f} (rank {current_rank}/25)")
        print(f"  Best config: Sharpe = {best_sharpe:.3f}")
        print(f"  Gap: {gap:.3f}")
        if gap <= 0.3:
            print(f"  VERDICT: ROBUST — current config within 0.3 Sharpe of best")
        else:
            print(f"  VERDICT: FRAGILE — current config is {gap:.3f} Sharpe below best (>0.3)")
    else:
        print("  Current config (252/21) not in results — check data availability")

    # Heatmap
    print("\n\n  SHARPE HEATMAP (train_window x hold_period):")
    print(f"  {'':8}", end="")
    for hp in HOLD_PERIODS:
        print(f"  h={hp:<4}", end="")
    print()
    for tw in TRAIN_WINDOWS:
        print(f"  tw={tw:<4}", end="")
        for hp in HOLD_PERIODS:
            cell = df_results[(df_results["train_window"] == tw) &
                              (df_results["hold_period"] == hp)]
            if not cell.empty:
                val = float(cell.iloc[0]["sharpe"])
                marker = "*" if (tw == 252 and hp == 21) else " "
                print(f"  {val:5.2f}{marker}", end="")
            else:
                print(f"  {'N/A':>6}", end="")
        print()
    print("\n  * = current config")

    # Stats
    n_beat_spy = (df_results["sharpe"] > spy_metrics["sharpe"]).sum()
    print(f"\n  Configs beating SPY Sharpe ({spy_metrics['sharpe']:.3f}): "
          f"{n_beat_spy}/{len(df_results)}")

    # Hold period sensitivity
    print("\n  HOLD PERIOD SENSITIVITY (for each train window):")
    for tw in TRAIN_WINDOWS:
        subset = df_results[df_results["train_window"] == tw]["sharpe"].dropna()
        if not subset.empty:
            print(f"    tw={tw:>3}d:  mean={subset.mean():.3f}  "
                  f"std={subset.std():.3f}  min={subset.min():.3f}  max={subset.max():.3f}")

    print("\n  TRAIN WINDOW SENSITIVITY (for each hold period):")
    for hp in HOLD_PERIODS:
        subset = df_results[df_results["hold_period"] == hp]["sharpe"].dropna()
        if not subset.empty:
            print(f"    h={hp:>2}d:   mean={subset.mean():.3f}  "
                  f"std={subset.std():.3f}  min={subset.min():.3f}  max={subset.max():.3f}")

    # Save
    df_results.to_csv(OUTPUT_DIR / "sweep_results.csv", index=False)
    summary = {
        "sweep_grid": {"train_windows": TRAIN_WINDOWS, "hold_periods": HOLD_PERIODS},
        "n_configs": len(results),
        "best_config": {
            "train_window": int(df_results.iloc[0]["train_window"]),
            "hold_period": int(df_results.iloc[0]["hold_period"]),
            "sharpe": float(df_results.iloc[0]["sharpe"]),
        },
        "current_config_252_21": {
            "sharpe": float(current.iloc[0]["sharpe"]) if not current.empty else None,
            "rank": int(current.index[0]) + 1 if not current.empty else None,
        },
        "spy_benchmark": spy_metrics,
        "robust": bool(gap <= 0.3) if np.isfinite(gap) else None,
        "elapsed_sec": round(elapsed, 1),
    }
    (OUTPUT_DIR / "sweep_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"\n\n[sweep] Complete in {elapsed:.0f}s")
    print(f"[sweep] Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
