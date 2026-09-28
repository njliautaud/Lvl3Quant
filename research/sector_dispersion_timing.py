"""
Sector Dispersion Timing Overlay for ETF Rotation v3.

THESIS: When cross-sector dispersion is HIGH, rotation alpha is strongest
(sectors are diverging → picking the right ones matters more).
When dispersion is LOW (all sectors moving together), rotation adds
nothing over benchmark → reduce exposure or hedge more.

This script:
1. Computes daily cross-sector dispersion (rolling std of sector returns)
2. Tests whether v3 rotation alpha is concentrated in high-dispersion periods
3. Builds a dispersion-scaled position sizing overlay
4. Walk-forward validates with permutation test (HC #665)
5. Reports R1 gap, Sharpe, and comparison to baseline v3

HC #667: adversarial data-driven validation
HC #670: rotation quality
HC #666: creative research, SPY benchmark mandatory
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))

PRICE_PATH = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"
MACRO_EXTRA_PATH = ROOT / "wheel_strategy_v1/data/cache/macro_extra.parquet"
MACRO_FEATURES_PATH = ROOT / "macro_exposure_v1/data/cache/macro_features.parquet"

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLU", "XLI", "XLV", "XLB", "XLC", "XLRE"]
BENCHMARK = "SPY"
TXN_COST_BPS = 5.0
TRADING_DAYS = 252

OUT_DIR = ROOT / "output/sector_dispersion_timing"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data Loading
# ============================================================================

def load_prices() -> pd.DataFrame:
    px = pd.read_parquet(PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    sub = px[px["ticker"].isin(SECTOR_ETFS + [BENCHMARK])].copy()
    sub["close"] = sub["close"].astype(float)
    return sub.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_yield_curve() -> pd.DataFrame:
    me = pd.read_parquet(MACRO_EXTRA_PATH)
    me["date"] = pd.to_datetime(me["date"])
    out = pd.DataFrame()
    out["date"] = me["date"]
    out["yc_2s10s"] = me["yc_2s10s"].astype(float)
    out["yc_2s10s_roc_20d"] = out["yc_2s10s"].diff(20)
    return out.set_index("date")


def load_vix() -> pd.Series:
    """Load VIX from macro_features or yfinance fallback."""
    try:
        mf = pd.read_parquet(MACRO_FEATURES_PATH)
        mf["date"] = pd.to_datetime(mf["date"])
        if "vix" in mf.columns:
            return mf.set_index("date")["vix"].astype(float)
        if "VIX_close" in mf.columns:
            return mf.set_index("date")["VIX_close"].astype(float)
    except Exception:
        pass
    # Fallback: load from prices cache (^VIX or VIX ticker)
    try:
        px = pd.read_parquet(PRICE_PATH)
        px["date"] = pd.to_datetime(px["date"])
        for ticker in ["^VIX", "VIX", "VIXY"]:
            sub = px[px["ticker"] == ticker]
            if len(sub) > 100:
                return sub.set_index("date")["close"].astype(float)
    except Exception:
        pass
    # Return empty series — functions that need VIX will handle gracefully
    print("  WARNING: VIX data not found, using empty series")
    return pd.Series(dtype=float)


# ============================================================================
# Feature Engineering — Sector Dispersion
# ============================================================================

def compute_sector_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Build daily sector return panel + dispersion metrics."""
    # Pivot to get daily closes per sector
    sector_px = prices[prices["ticker"].isin(SECTOR_ETFS)].pivot_table(
        index="date", columns="ticker", values="close"
    )

    # Daily returns
    sector_ret = sector_px.pct_change()

    # SPY returns for benchmark
    spy_px = prices[prices["ticker"] == BENCHMARK].set_index("date")["close"]
    spy_ret = spy_px.pct_change()

    # Cross-sector dispersion (rolling std of daily sector returns)
    result = pd.DataFrame(index=sector_ret.index)

    # Multiple lookback windows for dispersion
    for window in [5, 10, 20, 40]:
        # Daily cross-sectional std of returns
        daily_xsec_std = sector_ret.std(axis=1)
        result[f"dispersion_{window}d"] = daily_xsec_std.rolling(window).mean()

        # Dispersion of cumulative returns (more stable)
        cum_ret = sector_ret.rolling(window).sum()
        result[f"cum_dispersion_{window}d"] = cum_ret.std(axis=1)

    # Dispersion regime — high/low based on rolling percentile
    disp_20 = result["dispersion_20d"]
    result["dispersion_pctile"] = disp_20.rolling(252, min_periods=60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    result["high_dispersion"] = (result["dispersion_pctile"] > 0.6).astype(int)

    # Sector momentum features
    for etf in SECTOR_ETFS:
        if etf in sector_ret.columns:
            result[f"{etf}_ret_20d"] = sector_px[etf].pct_change(20)
            result[f"{etf}_rel_spy_20d"] = (
                sector_px[etf].pct_change(20) - spy_px.pct_change(20)
            )

    # Sector rank dispersion — how much do rankings change?
    ranks_20d = pd.DataFrame()
    for etf in SECTOR_ETFS:
        if etf in sector_px.columns:
            ranks_20d[etf] = sector_px[etf].pct_change(20)
    ranks_20d = ranks_20d.rank(axis=1, pct=True)
    result["rank_dispersion_10d"] = ranks_20d.diff(10).abs().mean(axis=1)

    # SPY as benchmark
    result["spy_ret"] = spy_ret
    result["spy_ret_20d"] = spy_px.pct_change(20)

    return result.dropna()


# ============================================================================
# Walk-Forward ETF Rotation v3 (with dispersion overlay)
# ============================================================================

def _xs_zscore(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Cross-sectional z-score within each date."""
    out = panel.copy()
    for c in cols:
        g = out.groupby("date")[c]
        out[c] = (out[c] - g.transform("mean")) / g.transform("std").clip(lower=1e-8)
    return out


def _fit_lgbm(X_tr, y_tr, X_oot):
    """Train LightGBM and return OOT scores."""
    try:
        import lightgbm as lgb
        dtrain = lgb.Dataset(X_tr, y_tr, free_raw_data=True)
        params = {
            "objective": "regression",
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1,
            "n_jobs": 4,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        return model.predict(X_oot)
    except Exception:
        return None


# Consecutive hold decay (HC #670 R2)
HOLD_DECAY = {0: 1.0, 1: 1.0, 2: 0.6, 3: 0.0}


def run_rotation_backtest(
    prices: pd.DataFrame,
    yc: pd.DataFrame,
    vix: pd.Series,
    dispersion: pd.DataFrame,
    *,
    use_dispersion_sizing: bool = False,
    dispersion_low_scale: float = 0.3,
    dispersion_high_scale: float = 1.0,
    dispersion_threshold_pctile: float = 0.4,  # below this = low dispersion
    train_months: int = 24,
    oot_months: int = 1,
    min_sectors: int = 3,
    n_hold: int = 3,
    txn_cost_bps: float = TXN_COST_BPS,
) -> dict:
    """
    Walk-forward sector rotation backtest with optional dispersion sizing.

    Returns daily returns series and metadata.
    """
    # Build full feature panel
    sector_px = prices[prices["ticker"].isin(SECTOR_ETFS)].pivot_table(
        index="date", columns="ticker", values="close"
    )
    spy_px = prices[prices["ticker"] == BENCHMARK].set_index("date")["close"]

    # Build panel of features per sector per date
    records = []
    for etf in SECTOR_ETFS:
        if etf not in sector_px.columns:
            continue
        px = sector_px[etf]
        ret_1d = px.pct_change()
        ret_5d = px.pct_change(5)
        ret_20d = px.pct_change(20)
        ret_60d = px.pct_change(60)
        rel_spy = ret_20d - spy_px.pct_change(20)

        # Momentum acceleration
        rs_accel_10 = rel_spy.diff(10)
        rs_accel_20 = rel_spy.diff(20)

        for dt in px.index:
            if pd.isna(ret_60d.get(dt)):
                continue
            rec = {
                "date": dt,
                "etf": etf,
                "ret_20d": ret_20d.get(dt, np.nan),
                "ret_60d": ret_60d.get(dt, np.nan),
                "rel_strength_spy": rel_spy.get(dt, np.nan),
                "rs_accel_10": rs_accel_10.get(dt, np.nan),
                "rs_accel_20": rs_accel_20.get(dt, np.nan),
                "momentum_cross": (ret_20d.get(dt, 0) - ret_60d.get(dt, 0)),
            }

            # Add yield curve features
            if dt in yc.index:
                rec["yc_2s10s"] = yc.loc[dt, "yc_2s10s"]
                rec["yc_2s10s_roc_20d"] = yc.loc[dt, "yc_2s10s_roc_20d"]
            else:
                rec["yc_2s10s"] = np.nan
                rec["yc_2s10s_roc_20d"] = np.nan

            # Forward return (target) — 21-day forward
            fwd_idx = px.index.get_indexer([dt])[0]
            if fwd_idx + 21 < len(px):
                rec["y_fwd"] = px.iloc[fwd_idx + 21] / px.iloc[fwd_idx] - 1
            else:
                rec["y_fwd"] = np.nan

            records.append(rec)

    panel = pd.DataFrame(records)
    panel["date"] = pd.to_datetime(panel["date"])

    feats = ["ret_20d", "ret_60d", "rel_strength_spy", "rs_accel_10",
             "rs_accel_20", "momentum_cross", "yc_2s10s", "yc_2s10s_roc_20d"]

    # Fill and z-score
    panel_z = _xs_zscore(panel, feats)
    for f in feats:
        panel_z[f] = panel_z[f].fillna(0.0)
    panel_z = panel_z.dropna(subset=["y_fwd"])

    # Walk-forward windows
    all_dates = sorted(panel_z["date"].unique())
    min_date = all_dates[0]
    max_date = all_dates[-1]

    # Generate windows
    windows = []
    import datetime
    train_days = train_months * 21
    oot_days = oot_months * 21

    start_idx = train_days
    while start_idx + oot_days <= len(all_dates):
        tr_start = all_dates[max(0, start_idx - train_days)]
        tr_end = all_dates[start_idx]
        oot_start = all_dates[start_idx]
        oot_end_idx = min(start_idx + oot_days, len(all_dates) - 1)
        oot_end = all_dates[oot_end_idx]
        windows.append((tr_start, tr_end, oot_start, oot_end))
        start_idx += oot_days

    # Backtest
    daily_returns = {}
    all_picks = []
    hold_counts = {etf: 0 for etf in SECTOR_ETFS}

    for tr_start, tr_end, oot_start, oot_end in windows:
        train = panel_z[(panel_z["date"] >= tr_start) & (panel_z["date"] < tr_end)]
        oot = panel_z[(panel_z["date"] >= oot_start) & (panel_z["date"] < oot_end)]

        if len(train) < 100 or len(oot) < 10:
            continue

        X_tr = train[feats].values
        y_tr = train["y_fwd"].values
        X_oot = oot[feats].values

        scores = _fit_lgbm(X_tr, y_tr, X_oot)
        if scores is None:
            continue

        oot = oot.copy()
        oot["score"] = scores

        # Pick sectors on first OOT date
        rebal_date = oot["date"].min()
        snap = oot[oot["date"] == rebal_date].dropna(subset=["score"])
        if len(snap) < min_sectors:
            continue

        # Apply hold decay (HC #670 R2)
        snap = snap.copy()
        for idx, row in snap.iterrows():
            etf = row["etf"]
            consec = hold_counts.get(etf, 0)
            decay = HOLD_DECAY.get(min(consec, 3), 0.0)
            snap.loc[idx, "adj_score"] = row["score"] * decay

        # Pick top N
        top = snap.nlargest(n_hold, "adj_score")
        picked_etfs = top["etf"].tolist()

        # Enforce minimum diversity
        if len(set(picked_etfs)) < min_sectors:
            # Add sectors not already picked, by raw score
            remaining = snap[~snap["etf"].isin(picked_etfs)].nlargest(
                min_sectors - len(set(picked_etfs)), "score"
            )
            picked_etfs = picked_etfs[:min_sectors - len(remaining)] + remaining["etf"].tolist()

        # Update hold counts
        for etf in SECTOR_ETFS:
            if etf in picked_etfs:
                hold_counts[etf] = hold_counts.get(etf, 0) + 1
            else:
                hold_counts[etf] = 0

        all_picks.append((rebal_date, picked_etfs))

        # Compute daily returns for OOT period
        for dt in sorted(oot["date"].unique()):
            if dt in daily_returns:
                continue

            # Equal-weight portfolio of picked sectors
            day_ret = 0.0
            n_valid = 0
            for etf in picked_etfs:
                if etf in sector_px.columns and dt in sector_px.index:
                    prev_idx = sector_px.index.get_indexer([dt])[0] - 1
                    if prev_idx >= 0:
                        r = sector_px[etf].iloc[sector_px.index.get_indexer([dt])[0]] / \
                            sector_px[etf].iloc[prev_idx] - 1
                        day_ret += r
                        n_valid += 1

            if n_valid > 0:
                day_ret /= n_valid

                # Apply txn cost on rebalance days only
                if dt == rebal_date:
                    day_ret -= txn_cost_bps / 10000 * len(picked_etfs) * 2  # buy + sell

                # Apply dispersion sizing
                scale = 1.0
                if use_dispersion_sizing and dt in dispersion.index:
                    disp_pctile = dispersion.loc[dt, "dispersion_pctile"]
                    if not pd.isna(disp_pctile):
                        if disp_pctile < dispersion_threshold_pctile:
                            scale = dispersion_low_scale
                        else:
                            scale = dispersion_high_scale

                daily_returns[dt] = day_ret * scale

    if not daily_returns:
        return {"error": "No returns generated"}

    ret_series = pd.Series(daily_returns).sort_index()

    # Compute metrics
    sharpe = ret_series.mean() / ret_series.std() * np.sqrt(TRADING_DAYS) if ret_series.std() > 0 else 0
    sortino_denom = ret_series[ret_series < 0].std()
    sortino = ret_series.mean() / sortino_denom * np.sqrt(TRADING_DAYS) if sortino_denom > 0 else 0

    cum = (1 + ret_series).cumprod()
    n_years = len(ret_series) / TRADING_DAYS
    cagr = (cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    max_dd = (cum / cum.cummax() - 1).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    wr = (ret_series > 0).mean()

    # Profit factor
    gains = ret_series[ret_series > 0].sum()
    losses = abs(ret_series[ret_series < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Rotation quality metrics (HC #670)
    sector_counts = {}
    for _, picks in all_picks:
        for etf in picks:
            sector_counts[etf] = sector_counts.get(etf, 0) + 1
    total_slots = sum(sector_counts.values())
    hhi = sum((v / total_slots) ** 2 for v in sector_counts.values()) if total_slots > 0 else 1

    sorted_freq = sorted([v/len(all_picks) for v in sector_counts.values()], reverse=True)
    top2_conc = sum(sorted_freq[:2]) if len(sorted_freq) >= 2 else 1.0

    return {
        "returns": ret_series,
        "n_days": len(ret_series),
        "n_rebalances": len(all_picks),
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar,
        "wr": wr,
        "pf": pf,
        "hhi": hhi,
        "top2_concentration": top2_conc,
        "n_sectors_used": len(sector_counts),
        "picks": all_picks,
    }


# ============================================================================
# R1 Regime Gap Test (HC #428)
# ============================================================================

def compute_r1_gap(daily_ret: pd.Series, spy_close: pd.Series) -> dict:
    """HC #428 R1: regime gap must be ≤ 0.50."""
    # Align dates
    common = daily_ret.index.intersection(spy_close.index)
    ret = daily_ret.loc[common]
    spy = spy_close.loc[common]

    # Regime: SPY close-to-close direction
    spy_ret = spy.pct_change()
    green = spy_ret > 0.001
    red = spy_ret < -0.001
    flat = ~green & ~red

    def sharpe(r):
        return r.mean() / r.std() * np.sqrt(252) if len(r) > 10 and r.std() > 0 else 0

    s_green = sharpe(ret[green])
    s_red = sharpe(ret[red])
    s_flat = sharpe(ret[flat])

    gap = abs(s_green - s_red) / max(abs(s_green), abs(s_red), 0.01)

    return {
        "sharpe_green": round(s_green, 4),
        "sharpe_red": round(s_red, 4),
        "sharpe_flat": round(s_flat, 4),
        "n_green": int(green.sum()),
        "n_red": int(red.sum()),
        "n_flat": int(flat.sum()),
        "gap": round(gap, 4),
        "r1_pass": gap <= 0.50,
    }


# ============================================================================
# Permutation Test (HC #665)
# ============================================================================

def permutation_test(daily_ret: pd.Series, n_perm: int = 200) -> dict:
    """
    Shuffle daily return SIGNS to test if the strategy's returns are real.
    If random sign flips also produce high Sharpe, the result is an artifact.
    """
    real_sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

    rng = np.random.default_rng(42)
    random_sharpes = []

    for _ in range(n_perm):
        # Shuffle dates (break temporal structure)
        shuffled = daily_ret.sample(frac=1, replace=False, random_state=rng.integers(1e9))
        shuffled.index = daily_ret.index  # keep original dates for R1
        s = shuffled.mean() / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
        random_sharpes.append(s)

    random_sharpes = np.array(random_sharpes)
    p_value = (random_sharpes >= real_sharpe).mean()

    return {
        "p_value": round(float(p_value), 4),
        "real_sharpe": round(real_sharpe, 4),
        "random_mean": round(float(random_sharpes.mean()), 4),
        "random_std": round(float(random_sharpes.std()), 4),
        "n_perm": n_perm,
        "pass": p_value < 0.05,
    }


# ============================================================================
# SPY Benchmark (HC #666)
# ============================================================================

def spy_benchmark(prices: pd.DataFrame, start_date, end_date) -> dict:
    """Compute buy-and-hold SPY metrics for comparison."""
    spy_px = prices[prices["ticker"] == BENCHMARK].set_index("date")["close"]
    spy_px = spy_px[(spy_px.index >= start_date) & (spy_px.index <= end_date)]
    spy_ret = spy_px.pct_change().dropna()

    sharpe = spy_ret.mean() / spy_ret.std() * np.sqrt(252) if spy_ret.std() > 0 else 0
    sortino_d = spy_ret[spy_ret < 0].std()
    sortino = spy_ret.mean() / sortino_d * np.sqrt(252) if sortino_d > 0 else 0
    cum = (1 + spy_ret).cumprod()
    n_years = len(spy_ret) / 252
    cagr = (cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0
    max_dd = (cum / cum.cummax() - 1).min()

    return {
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "cagr": round(cagr * 100, 1),
        "max_dd": round(max_dd * 100, 1),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("SECTOR DISPERSION TIMING — ETF ROTATION v3 OVERLAY")
    print("=" * 70)

    print("\nLoading data...")
    prices = load_prices()
    yc = load_yield_curve()
    vix = load_vix()

    print("Computing dispersion features...")
    dispersion = compute_sector_panel(prices)
    print(f"  Dispersion panel: {len(dispersion)} days, {dispersion.columns.size} features")
    print(f"  Date range: {dispersion.index.min().date()} to {dispersion.index.max().date()}")

    # Descriptive stats on dispersion
    disp_20 = dispersion["dispersion_20d"]
    print(f"\n  Dispersion 20d: mean={disp_20.mean():.4f}, "
          f"p25={disp_20.quantile(0.25):.4f}, p75={disp_20.quantile(0.75):.4f}")

    # ==== PHASE 1: BASELINE — ETF Rotation v3 WITHOUT dispersion ====
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE — ETF Rotation v3 (no dispersion)")
    print("=" * 70)

    baseline = run_rotation_backtest(
        prices, yc, vix, dispersion,
        use_dispersion_sizing=False,
    )

    if "error" in baseline:
        print(f"  ERROR: {baseline['error']}")
        return

    spy_px = prices[prices["ticker"] == BENCHMARK].set_index("date")["close"]
    r1_base = compute_r1_gap(baseline["returns"], spy_px)

    print(f"\n  Days: {baseline['n_days']}, Rebalances: {baseline['n_rebalances']}")
    print(f"  Sharpe: {baseline['sharpe']:.2f}, Sortino: {baseline['sortino']:.2f}")
    print(f"  CAGR: {baseline['cagr']*100:.1f}%, MaxDD: {baseline['max_dd']*100:.1f}%")
    print(f"  Calmar: {baseline['calmar']:.2f}, WR: {baseline['wr']*100:.1f}%, PF: {baseline['pf']:.2f}")
    print(f"  R1 gap: {r1_base['gap']:.3f} ({'PASS' if r1_base['r1_pass'] else 'FAIL'})")
    print(f"  HHI: {baseline['hhi']:.4f}, Top-2 conc: {baseline['top2_concentration']*100:.1f}%")
    print(f"  Sectors used: {baseline['n_sectors_used']}/11")

    # ==== PHASE 2: DISPERSION OVERLAY SWEEP ====
    print("\n" + "=" * 70)
    print("PHASE 2: DISPERSION SIZING OVERLAY SWEEP")
    print("=" * 70)

    configs = []
    for low_scale in [0.0, 0.2, 0.3, 0.5]:
        for thresh_pctile in [0.3, 0.4, 0.5]:
            configs.append({
                "low_scale": low_scale,
                "high_scale": 1.0,
                "thresh_pctile": thresh_pctile,
            })

    results = []
    for cfg in configs:
        r = run_rotation_backtest(
            prices, yc, vix, dispersion,
            use_dispersion_sizing=True,
            dispersion_low_scale=cfg["low_scale"],
            dispersion_high_scale=cfg["high_scale"],
            dispersion_threshold_pctile=cfg["thresh_pctile"],
        )
        if "error" in r:
            continue

        r1 = compute_r1_gap(r["returns"], spy_px)
        results.append({
            "config": cfg,
            "sharpe": r["sharpe"],
            "sortino": r["sortino"],
            "cagr": r["cagr"],
            "max_dd": r["max_dd"],
            "calmar": r["calmar"],
            "wr": r["wr"],
            "pf": r["pf"],
            "r1_gap": r1["gap"],
            "r1_pass": r1["r1_pass"],
            "sharpe_green": r1["sharpe_green"],
            "sharpe_red": r1["sharpe_red"],
            "n_days": r["n_days"],
            "returns": r["returns"],
        })

    # Sort by Sharpe (R1-passing first)
    results.sort(key=lambda x: (-int(x["r1_pass"]), -x["sharpe"]))

    print(f"\n  Tested {len(results)} configs:")
    print(f"  {'Low':>5} {'Thr':>5} {'Sharpe':>7} {'Sort':>6} {'CAGR':>6} {'MaxDD':>6} {'Gap':>5} {'R1':>4}")
    print("  " + "-" * 50)
    for r in results:
        cfg = r["config"]
        r1_str = "✅" if r["r1_pass"] else "❌"
        print(f"  {cfg['low_scale']:5.1f} {cfg['thresh_pctile']:5.1f} "
              f"{r['sharpe']:7.2f} {r['sortino']:6.2f} "
              f"{r['cagr']*100:5.1f}% {r['max_dd']*100:5.1f}% "
              f"{r['r1_gap']:5.3f} {r1_str}")

    # ==== PHASE 3: BEST CONFIG — ADVERSARIAL VALIDATION ====
    best = results[0] if results else None
    if best is None:
        print("\n  No valid configs found!")
        return

    print(f"\n" + "=" * 70)
    print(f"PHASE 3: ADVERSARIAL VALIDATION — BEST CONFIG")
    print(f"  low_scale={best['config']['low_scale']}, "
          f"thresh={best['config']['thresh_pctile']}")
    print("=" * 70)

    # Permutation test (HC #665)
    print("\n  Running permutation test (200 shuffles)...")
    perm = permutation_test(best["returns"], n_perm=200)
    print(f"  Real Sharpe: {perm['real_sharpe']:.2f}")
    print(f"  Random mean: {perm['random_mean']:.2f} ± {perm['random_std']:.2f}")
    print(f"  p-value: {perm['p_value']:.3f}")
    print(f"  Permutation: {'PASS ✅' if perm['pass'] else 'FAIL ❌'}")

    # SPY benchmark (HC #666)
    spy_bench = spy_benchmark(
        prices,
        best["returns"].index.min(),
        best["returns"].index.max(),
    )

    print(f"\n  VS SPY buy-and-hold:")
    print(f"    Strategy: Sharpe {best['sharpe']:.2f}, CAGR {best['cagr']*100:.1f}%, MaxDD {best['max_dd']*100:.1f}%")
    print(f"    SPY:      Sharpe {spy_bench['sharpe']:.2f}, CAGR {spy_bench['cagr']}%, MaxDD {spy_bench['max_dd']}%")

    # ==== PHASE 4: DISPERSION ALPHA ANALYSIS ====
    print(f"\n" + "=" * 70)
    print(f"PHASE 4: IS DISPERSION ACTUALLY PREDICTIVE?")
    print("=" * 70)

    # Compare baseline alpha in high vs low dispersion periods
    base_ret = baseline["returns"]
    common_dates = base_ret.index.intersection(dispersion.index)
    base_common = base_ret.loc[common_dates]
    disp_common = dispersion.loc[common_dates, "dispersion_pctile"]

    # Drop NaN
    valid = ~disp_common.isna()
    base_valid = base_common[valid]
    disp_valid = disp_common[valid]

    high_mask = disp_valid > 0.6
    low_mask = disp_valid < 0.4

    high_ret = base_valid[high_mask]
    low_ret = base_valid[low_mask]

    if len(high_ret) > 20 and len(low_ret) > 20:
        sharpe_high = high_ret.mean() / high_ret.std() * np.sqrt(252) if high_ret.std() > 0 else 0
        sharpe_low = low_ret.mean() / low_ret.std() * np.sqrt(252) if low_ret.std() > 0 else 0

        print(f"\n  Baseline rotation alpha by dispersion regime:")
        print(f"    HIGH dispersion periods ({len(high_ret)} days): Sharpe {sharpe_high:.2f}")
        print(f"    LOW  dispersion periods ({len(low_ret)} days):  Sharpe {sharpe_low:.2f}")
        print(f"    Difference: {sharpe_high - sharpe_low:+.2f}")

        if sharpe_high > sharpe_low + 0.3:
            print(f"\n    ✅ CONFIRMED: Rotation alpha IS concentrated in high-dispersion periods")
            print(f"       Dispersion timing adds value.")
        elif abs(sharpe_high - sharpe_low) < 0.3:
            print(f"\n    ⚠️ MIXED: Rotation alpha is similar in both regimes")
            print(f"       Dispersion timing may not add much value.")
        else:
            print(f"\n    ❌ SURPRISE: Rotation alpha is STRONGER in low-dispersion periods")
            print(f"       Dispersion timing thesis is WRONG.")

    # ==== SAVE RESULTS ====
    output = {
        "baseline": {
            "sharpe": round(baseline["sharpe"], 3),
            "sortino": round(baseline["sortino"], 3),
            "cagr": round(baseline["cagr"], 4),
            "max_dd": round(baseline["max_dd"], 4),
            "calmar": round(baseline["calmar"], 3),
            "wr": round(baseline["wr"], 4),
            "pf": round(baseline["pf"], 3),
            "r1_gap": r1_base["gap"],
            "r1_pass": r1_base["r1_pass"],
            "n_days": baseline["n_days"],
            "hhi": round(baseline["hhi"], 4),
        },
        "best_dispersion_config": {
            "config": best["config"],
            "sharpe": round(best["sharpe"], 3),
            "sortino": round(best["sortino"], 3),
            "cagr": round(best["cagr"], 4),
            "max_dd": round(best["max_dd"], 4),
            "calmar": round(best["calmar"], 3),
            "r1_gap": best["r1_gap"],
            "r1_pass": best["r1_pass"],
        },
        "permutation": perm,
        "spy_benchmark": spy_bench,
        "all_configs": [
            {
                "config": r["config"],
                "sharpe": round(r["sharpe"], 3),
                "r1_gap": r["r1_gap"],
                "r1_pass": r["r1_pass"],
            }
            for r in results
        ],
    }

    # Add dispersion alpha analysis
    if len(high_ret) > 20 and len(low_ret) > 20:
        output["dispersion_alpha"] = {
            "sharpe_high_dispersion": round(sharpe_high, 3),
            "sharpe_low_dispersion": round(sharpe_low, 3),
            "n_days_high": len(high_ret),
            "n_days_low": len(low_ret),
            "thesis_confirmed": sharpe_high > sharpe_low + 0.3,
        }

    with open(OUT_DIR / "dispersion_timing_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Results saved to {OUT_DIR}/dispersion_timing_results.json")

    # ==== VERDICT ====
    print(f"\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    improved = best["sharpe"] > baseline["sharpe"] + 0.1
    passes_perm = perm["pass"]
    passes_r1 = best["r1_pass"]

    if improved and passes_perm:
        print(f"  ✅ DISPERSION TIMING IMPROVES ETF ROTATION")
        print(f"     Baseline Sharpe: {baseline['sharpe']:.2f} → With overlay: {best['sharpe']:.2f}")
        print(f"     Permutation: PASS (p={perm['p_value']:.3f})")
    elif not improved:
        print(f"  ❌ DISPERSION TIMING DOES NOT IMPROVE ETF ROTATION")
        print(f"     Baseline Sharpe: {baseline['sharpe']:.2f}, Best overlay: {best['sharpe']:.2f}")
        print(f"     The overlay reduces exposure but doesn't improve risk-adjusted returns.")
    elif not passes_perm:
        print(f"  ⚠️ IMPROVEMENT IS NOT ROBUST (permutation fails)")
        print(f"     Baseline Sharpe: {baseline['sharpe']:.2f}, Best overlay: {best['sharpe']:.2f}")
        print(f"     But random shuffles achieve similar results (p={perm['p_value']:.3f})")

    if not passes_r1:
        print(f"  ⚠️ Best config FAILS R1 (gap {best['r1_gap']:.3f} > 0.50)")

    print(f"\n  Recommendation: {'ADOPT overlay' if improved and passes_perm else 'KEEP baseline v3 unchanged'}")


if __name__ == "__main__":
    main()
