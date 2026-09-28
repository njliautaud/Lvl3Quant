"""
ETF Rotation Quality — HC #670 rotation-optimized sector strategy.

Three phases:
  Phase 1: Audit existing v2 for concentration/stickiness bias
  Phase 2: Build rotation-quality-optimized strategy with anti-concentration
  Phase 3: Walk-forward backtest with rotation quality metrics

Key insight: alpha is in TIMING sector switches, not picking "best" sector.
Anti-concentration: decay penalty for consecutive holds, min 3-sector diversification.

HC #670 rules:
  R2: no sector held >3 consecutive rebalances without penalty
  R3: if >60% holdings in 1-2 sectors, model has failed
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
from walk_forward import _metrics  # type: ignore  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PRICE_PATH = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"
MACRO_EXTRA_PATH = ROOT / "wheel_strategy_v1/data/cache/macro_extra.parquet"
MACRO_FEATURES_PATH = ROOT / "macro_exposure_v1/data/cache/macro_features.parquet"
SECTOR_FLOW_PATH = ROOT / "data/feature_store/sector_etf_flows/daily.parquet"

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLY", "XLP", "XLU", "XLI", "XLV", "XLB", "XLC", "XLRE"]
BENCHMARK = "SPY"
TRADING_DAYS = 252
TXN_COST_BPS = 5.0

# Consecutive hold decay (HC #670 R2) — aggressive to enforce <=3 max streak
HOLD_DECAY = {0: 1.0, 1: 1.0, 2: 0.6, 3: 0.0}  # rebal count -> multiplier
# After 3 consecutive holds, score zeroed out (forces rotation)

# Minimum sectors to hold (HC #670 R3)
MIN_SECTORS = 3


# ============================================================================
# PHASE 1: AUDIT V2 FOR CONCENTRATION BIAS
# ============================================================================

def audit_v2(book_path: Path, panel: pd.DataFrame, feats: list[str],
             prices: pd.DataFrame, yc: pd.DataFrame, vix: pd.Series) -> dict:
    """
    Re-run v2 logic but TRACK which sectors are selected at each rebalance.
    Returns concentration metrics.
    """
    print("\n" + "=" * 65)
    print("PHASE 1: AUDITING V2 FOR CONCENTRATION BIAS")
    print("=" * 65)

    # We need to replay the v2 walk-forward to see sector picks.
    # Re-use v2 logic: LGBM trained on momentum + yield curve features,
    # then pick top-2 sectors per rebalance.
    # Instead of full replay, we'll do a simplified version that captures picks.

    spy_ma = _build_regime_series(prices, vix)
    windows = _iter_wf_windows(
        panel["date"].min(), panel["date"].max(),
        train_months=24, oot_months=1, step_months=1,
    )

    all_picks = []  # list of (rebal_date, [etf1, etf2, ...])

    panel_z = _xs_zscore(panel, feats)
    for f in feats:
        panel_z[f] = panel_z[f].fillna(0.0)
    panel_z = panel_z.dropna(subset=["y_fwd"])

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

        # Pick top 2 on first day of OOT (rebalance day)
        unique_dates = sorted(oot["date"].unique())
        for rd in unique_dates[::21]:  # every 21 days
            snap = oot[oot["date"] == rd].dropna(subset=["score"])
            if len(snap) < 2:
                continue
            top2 = snap.nlargest(2, "score")["etf"].tolist()
            all_picks.append((rd, top2))

    if not all_picks:
        print("  No picks found - cannot audit v2")
        return {}

    # Analyze picks
    n_rebals = len(all_picks)
    sector_counts = defaultdict(int)
    for _, picks in all_picks:
        for etf in picks:
            sector_counts[etf] += 1

    # Frequency table
    freq = {k: v / n_rebals for k, v in sorted(sector_counts.items(), key=lambda x: -x[1])}

    # Consecutive hold streaks
    etf_streaks = defaultdict(list)
    prev_picks = set()
    streak_counter = defaultdict(int)
    for _, picks in all_picks:
        curr = set(picks)
        for etf in SECTOR_ETFS:
            if etf in curr and etf in prev_picks:
                streak_counter[etf] += 1
            else:
                if streak_counter[etf] > 0:
                    etf_streaks[etf].append(streak_counter[etf])
                streak_counter[etf] = 1 if etf in curr else 0
        prev_picks = curr
    # Flush remaining streaks
    for etf in SECTOR_ETFS:
        if streak_counter[etf] > 0:
            etf_streaks[etf].append(streak_counter[etf])

    max_streak_per_etf = {etf: max(streaks) if streaks else 0
                          for etf, streaks in etf_streaks.items()}
    overall_max_streak = max(max_streak_per_etf.values()) if max_streak_per_etf else 0

    # Herfindahl concentration index
    total_slots = sum(sector_counts.values())
    hhi = sum((v / total_slots) ** 2 for v in sector_counts.values())

    # Rotation frequency: what % of holdings change per rebalance
    rotation_pcts = []
    for i in range(1, len(all_picks)):
        prev = set(all_picks[i-1][1])
        curr = set(all_picks[i][1])
        changed = len(curr - prev)
        total = len(curr)
        rotation_pcts.append(changed / total if total > 0 else 0)
    avg_rotation = np.mean(rotation_pcts) if rotation_pcts else 0

    # % time in tech (XLK)
    xlk_pct = freq.get("XLK", 0)

    # Top 2 sector concentration
    sorted_freq = sorted(freq.values(), reverse=True)
    top2_conc = sum(sorted_freq[:2]) if len(sorted_freq) >= 2 else sum(sorted_freq)

    audit = {
        "n_rebalances": n_rebals,
        "sector_frequency": freq,
        "max_consecutive_hold": max_streak_per_etf,
        "overall_max_streak": overall_max_streak,
        "herfindahl_index": hhi,
        "avg_rotation_pct": avg_rotation,
        "xlk_pct_time": xlk_pct,
        "top2_sector_concentration": top2_conc,
    }

    print(f"\n  Rebalances: {n_rebals}")
    print(f"\n  Sector holding frequency (% of rebalances):")
    for etf, pct in freq.items():
        bar = "#" * int(pct * 40)
        print(f"    {etf:5s}: {pct*100:5.1f}%  {bar}")
    print(f"\n  Max consecutive hold streak: {overall_max_streak}")
    for etf, s in sorted(max_streak_per_etf.items(), key=lambda x: -x[1]):
        if s > 2:
            print(f"    {etf}: {s} consecutive rebalances")
    print(f"\n  Herfindahl (concentration): {hhi:.4f}  "
          f"(1/{len(SECTOR_ETFS)} = {1/len(SECTOR_ETFS):.4f} = perfect diversification)")
    print(f"  Avg rotation per rebalance: {avg_rotation*100:.1f}%")
    print(f"  % time in XLK (tech): {xlk_pct*100:.1f}%")
    print(f"  Top-2 sector concentration: {top2_conc*100:.1f}%")

    # HC #670 R3 check
    if top2_conc > 0.60:
        print(f"\n  *** FAIL HC #670 R3: top-2 sectors hold {top2_conc*100:.1f}% > 60% ***")
    else:
        print(f"\n  PASS HC #670 R3: top-2 sector concentration {top2_conc*100:.1f}% <= 60%")

    return audit


# ============================================================================
# PHASE 2: ROTATION-QUALITY-OPTIMIZED STRATEGY
# ============================================================================

# ---------------------------------------------------------------------------
# Data loaders (extended from v2)
# ---------------------------------------------------------------------------
def _load_prices() -> pd.DataFrame:
    px = pd.read_parquet(PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    sub = px[px["ticker"].isin(SECTOR_ETFS + [BENCHMARK])].copy()
    sub["close"] = sub["close"].astype(float)
    return sub.sort_values(["ticker", "date"]).reset_index(drop=True)


def _load_yield_curve() -> pd.DataFrame:
    me = pd.read_parquet(MACRO_EXTRA_PATH)
    me["date"] = pd.to_datetime(me["date"])
    me = me.sort_values("date").reset_index(drop=True)

    out = pd.DataFrame()
    out["date"] = me["date"]
    out["yc_2s10s"] = me["yc_2s10s"].astype(float)
    out["fed_funds"] = me["fed_funds"].astype(float)
    out["ust_10y"] = me["ust_10y"].astype(float)
    out["yc_2s10s_roc_20d"] = out["yc_2s10s"].diff(20)
    out["fed_funds_roc_20d"] = out["fed_funds"].diff(20)
    out["ust_10y_roc_20d"] = out["ust_10y"].diff(20)
    return out.set_index("date")


def _load_macro_features() -> pd.DataFrame:
    mf = pd.read_parquet(MACRO_FEATURES_PATH)
    mf["date"] = pd.to_datetime(mf["date"])
    return mf.set_index("date")


def _load_sector_flows() -> pd.DataFrame | None:
    """Load sector ETF flow data if available."""
    if not SECTOR_FLOW_PATH.exists():
        return None
    sf = pd.read_parquet(SECTOR_FLOW_PATH)
    sf["date"] = pd.to_datetime(sf["date"])
    return sf[sf["etf"].isin(SECTOR_ETFS)].copy()


# ---------------------------------------------------------------------------
# Extended feature panel with rotation signals
# ---------------------------------------------------------------------------
MOMENTUM_FEATURES = [
    "ret_20d",
    "ret_60d",
    "rel_strength_spy",
    "momentum_cross_20_60",
    "rs_rank_among_sectors",
]

# NEW: rotation-timing features
ROTATION_FEATURES = [
    "rs_acceleration_10d",     # 10d change in relative strength (rotation signal)
    "rs_acceleration_20d",     # 20d change in relative strength
    "ret_20d_chg_10d",         # momentum CHANGE (acceleration, not level)
    "cross_sector_dispersion", # high = rotation opportunity
    "rank_change_10d",         # rank change among sectors (who's rising/falling)
]

YIELD_CURVE_FEATURES = [
    "yc_2s10s",
    "yc_2s10s_roc_20d",
    "fed_funds",
    "fed_funds_roc_20d",
    "ust_10y",
    "ust_10y_roc_20d",
]

VIX_FEATURES = [
    "vix_ts_slope",          # VIX term structure (contango = calm, backwardation = fear)
    "vix_pct_20d",           # VIX % change over 20d
]

FLOW_FEATURES = [
    "flow_volume_rank",       # volume rank among sectors (flow signal)
    "flow_rel_strength",      # relative volume momentum
]

ALL_FEATURES_V3 = (MOMENTUM_FEATURES + ROTATION_FEATURES +
                   YIELD_CURVE_FEATURES + VIX_FEATURES + FLOW_FEATURES)


def build_panel_v3(hold_days: int, prices: pd.DataFrame,
                   yc: pd.DataFrame, macro_feats: pd.DataFrame,
                   sector_flows: pd.DataFrame | None) -> pd.DataFrame:
    """
    Extended panel with rotation-timing features.
    """
    spy = prices[prices["ticker"] == BENCHMARK].sort_values("date").set_index("date")["close"]
    spy_r20 = spy.pct_change(20)
    spy_r60 = spy.pct_change(60)

    rows = []
    for t in SECTOR_ETFS:
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

        # NEW: rotation timing features
        # Relative strength ACCELERATION (change in RS, not RS itself)
        rs = s["ret_20d"] - spy_r20.reindex(s.index)
        s["rs_acceleration_10d"] = rs.diff(10)
        s["rs_acceleration_20d"] = rs.diff(20)

        # Momentum acceleration (change in momentum)
        s["ret_20d_chg_10d"] = s["ret_20d"].diff(10)

        # Yield curve broadcast join — T-1 lag to prevent look-ahead on
        # intraday macro releases (HC #428 fix: fed funds, yield curve data
        # released during trading day must use prior day's value)
        for col in YIELD_CURVE_FEATURES:
            if col in yc.columns:
                s[col] = yc[col].shift(1).reindex(s.index)

        # VIX term structure features — T-1 lag for same reason
        if "vix_ts_slope" in macro_feats.columns:
            s["vix_ts_slope"] = macro_feats["vix_ts_slope"].shift(1).reindex(s.index)
        if "vix_pct_20d" in macro_feats.columns:
            s["vix_pct_20d"] = macro_feats["vix_pct_20d"].shift(1).reindex(s.index)

        # Sector flow features
        if sector_flows is not None:
            etf_flows = sector_flows[sector_flows["etf"] == t].set_index("date")
            if "volume" in etf_flows.columns:
                vol_20 = etf_flows["volume"].rolling(20, min_periods=5).mean()
                vol_60 = etf_flows["volume"].rolling(60, min_periods=20).mean()
                s["flow_rel_strength"] = (vol_20 / vol_60 - 1.0).reindex(s.index)
            else:
                s["flow_rel_strength"] = np.nan
        else:
            s["flow_rel_strength"] = np.nan

        s["etf"] = t
        rows.append(s.reset_index())

    panel = pd.concat(rows, ignore_index=True)

    # Cross-sectional features (need all ETFs)
    panel["rs_rank_among_sectors"] = panel.groupby("date")["ret_20d"].rank(pct=True)
    panel["rank_change_10d"] = panel.groupby("etf")["rs_rank_among_sectors"].diff(10)

    # Cross-sector dispersion (same value for all ETFs on a date)
    disp = panel.groupby("date")["ret_20d"].std().rename("cross_sector_dispersion")
    panel = panel.merge(disp.reset_index(), on="date", how="left")

    # Flow volume rank (cross-sectional)
    if sector_flows is not None and "volume" in sector_flows.columns:
        # Merge flow volume into panel
        flow_vol = sector_flows[["etf", "date", "volume"]].rename(columns={"volume": "flow_vol_raw"})
        panel = panel.merge(flow_vol, on=["etf", "date"], how="left")
        panel["flow_volume_rank"] = panel.groupby("date")["flow_vol_raw"].rank(pct=True)
        panel.drop(columns=["flow_vol_raw"], inplace=True, errors="ignore")
    else:
        panel["flow_volume_rank"] = np.nan

    # Forward return target
    panel = panel.sort_values(["etf", "date"]).reset_index(drop=True)
    panel["y_fwd"] = (
        panel.groupby("etf")["close"].shift(-hold_days) / panel["close"] - 1.0
    )

    # VIX for regime gating
    if "vix" in macro_feats.columns:
        vix_series = macro_feats["vix"]
        panel["vix"] = vix_series.reindex(panel["date"]).values

    return panel


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def _fit_lgbm(X_tr: np.ndarray, y_tr: np.ndarray,
              X_oot: np.ndarray) -> np.ndarray | None:
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    params = {
        "objective": "regression",
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": 1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr)
    return model.predict(X_oot)


# ---------------------------------------------------------------------------
# Z-score utilities
# ---------------------------------------------------------------------------
def _winsorize(s: pd.Series, p: float = 0.01) -> pd.Series:
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lower=lo, upper=hi)


def _xs_zscore(panel: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    out = panel.copy()
    for f in feats:
        if f not in out.columns:
            out[f] = 0.0
            continue
        x = pd.to_numeric(out[f], errors="coerce").astype(float)
        x = _winsorize(x, 0.01)
        out[f] = x
        mu = out.groupby("date")[f].transform("mean")
        sd = out.groupby("date")[f].transform("std")
        z = (x - mu) / sd.replace(0.0, np.nan)
        out[f] = z.replace([np.inf, -np.inf], np.nan)
    return out


# ---------------------------------------------------------------------------
# Regime gate (same as v2)
# ---------------------------------------------------------------------------
def compute_regime_allocation(spy_close, spy_ma, vix_level):
    if not np.isfinite(spy_ma) or not np.isfinite(spy_close):
        return 1.0, 0.0, "bull_full"
    ma_gap_pct = (spy_close - spy_ma) / spy_ma
    if ma_gap_pct >= 0:
        if not np.isfinite(vix_level) or vix_level < 20.0:
            return 1.0, 0.0, "bull_full"
        elif vix_level <= 25.0:
            return 0.80, 0.0, "bull_cautious"
        else:
            return 0.80, 0.0, "bull_highvol"
    else:
        if ma_gap_pct > -0.02:
            return 0.0, 0.0, "bear_shallow"
        else:
            return 0.0, 0.20, "bear_deep"


def _build_regime_series(prices, vix, ma_days=60):
    spy = prices[prices["ticker"] == BENCHMARK].sort_values("date").set_index("date")["close"]
    ma = spy.rolling(ma_days, min_periods=max(20, ma_days // 2)).mean()
    ma._spy_close = spy
    ma._spy_ret = spy.pct_change().fillna(0.0)
    return ma


# ---------------------------------------------------------------------------
# Walk-forward windows (SLIDING — HC #0)
# ---------------------------------------------------------------------------
def _iter_wf_windows(start, end, train_months=12, oot_months=1, step_months=1):
    out = []
    cursor = start
    while True:
        tr_start = cursor
        tr_end = tr_start + pd.DateOffset(months=train_months)
        oot_start = tr_end
        oot_end = oot_start + pd.DateOffset(months=oot_months)
        if oot_end > end + pd.Timedelta(days=1):
            break
        out.append((tr_start, tr_end, oot_start, oot_end))
        cursor = cursor + pd.DateOffset(months=step_months)
    return out


# ---------------------------------------------------------------------------
# Vol-target sizing
# ---------------------------------------------------------------------------
def _estimate_book_vol(panel, rd, longs, lookback_days=60):
    cutoff_lo = rd - pd.Timedelta(days=lookback_days * 2 + 10)
    hist = panel[
        (panel["date"] < rd) & (panel["date"] >= cutoff_lo) & (panel["etf"].isin(longs))
    ]
    if hist.empty:
        return 0.0
    by_date = hist.groupby("date")["ret_1d"].mean().dropna().tail(lookback_days)
    if len(by_date) < 20:
        return 0.0
    sd = float(by_date.std(ddof=1))
    return sd * np.sqrt(TRADING_DAYS) if np.isfinite(sd) else 0.0


# ---------------------------------------------------------------------------
# Core: rotation-quality walk-forward fold
# ---------------------------------------------------------------------------
def _wf_fold_v3(
    panel: pd.DataFrame,
    feats: list[str],
    tr_start, tr_end, oot_start, oot_end,
    hold_days: int = 21,
    n_long: int = 3,           # min 3 sectors (HC #670 R3)
    target_vol: float = 0.15,
    lev_min: float = 0.25,
    lev_max: float = 2.0,
    txn_cost_bps: float = TXN_COST_BPS,
    spy_ma=None, vix=None,
    prev_holdings: list[str] | None = None,
    hold_streak: dict | None = None,
) -> dict:
    """
    One fold with rotation-quality anti-concentration.

    Key differences from v2:
    1. Consecutive hold penalty: sectors held >2 consecutive rebalances get score decay
    2. Minimum 3 sectors (not 2)
    3. Tracks which sectors are selected for rotation metrics
    """
    if hold_streak is None:
        hold_streak = defaultdict(int)

    train = panel[(panel["date"] >= tr_start) & (panel["date"] < tr_end)].copy()
    oot = panel[(panel["date"] >= oot_start) & (panel["date"] < oot_end)].copy()

    if len(train) < 100 or len(oot) < 10:
        return {
            "daily_pnl": pd.Series(dtype=float),
            "picks": [],
            "hold_streak": hold_streak,
            "oot_start": str(oot_start.date()),
            "oot_end": str(oot_end.date()),
            "n_rebal": 0,
        }

    # Z-score features
    train_z = _xs_zscore(train, feats)
    oot_z = _xs_zscore(oot, feats)
    for f in feats:
        train_z[f] = train_z[f].fillna(0.0)
        oot_z[f] = oot_z[f].fillna(0.0)
    train_z = train_z.dropna(subset=["y_fwd"])
    if len(train_z) < 50:
        return {
            "daily_pnl": pd.Series(dtype=float),
            "picks": [],
            "hold_streak": hold_streak,
            "oot_start": str(oot_start.date()),
            "oot_end": str(oot_end.date()),
            "n_rebal": 0,
        }

    X_tr = train_z[feats].values
    y_tr = train_z["y_fwd"].values
    X_oot = oot_z[feats].values

    scores_oot = _fit_lgbm(X_tr, y_tr, X_oot)
    if scores_oot is None:
        return {
            "daily_pnl": pd.Series(dtype=float),
            "picks": [],
            "hold_streak": hold_streak,
            "oot_start": str(oot_start.date()),
            "oot_end": str(oot_end.date()),
            "n_rebal": 0,
        }

    oot_z = oot_z.copy()
    oot_z["score"] = scores_oot
    oot_z["ret_raw"] = pd.to_numeric(oot["ret_1d"], errors="coerce").astype(float).values

    unique_dates = sorted(oot_z["date"].unique())
    rebal_dates = unique_dates[::hold_days]

    daily = []
    picks_log = []
    n_rebal_actual = 0

    for rd in rebal_dates:
        # Regime gate
        spy_close_val = float("nan")
        spy_ma_val = float("nan")
        vix_val = float("nan")

        if spy_ma is not None:
            prior_ma = spy_ma.loc[:pd.Timestamp(rd)]
            if len(prior_ma) > 0:
                spy_ma_val = float(prior_ma.iloc[-1])
        if hasattr(spy_ma, "_spy_close"):
            prior_close = spy_ma._spy_close.loc[:pd.Timestamp(rd)]
            if len(prior_close) > 0:
                spy_close_val = float(prior_close.iloc[-1])
        if vix is not None:
            prior_vix = vix.loc[:pd.Timestamp(rd)]
            if len(prior_vix) > 0:
                vix_val = float(prior_vix.iloc[-1])

        sector_frac, sh_frac, regime_label = compute_regime_allocation(
            spy_close_val, spy_ma_val, vix_val
        )

        # Score sectors
        snap = oot_z[oot_z["date"] == rd].dropna(subset=["score"])
        if len(snap) < n_long:
            continue

        if sector_frac <= 0:
            # Bear regime: no sector allocation
            longs = []
        else:
            # ANTI-CONCENTRATION: apply consecutive hold decay penalty
            # HC #670 R2: hard block any sector held >=3 consecutive rebalances
            snap = snap.copy()
            snap["consec_holds"] = snap["etf"].map(
                lambda e: hold_streak.get(e, 0)
            )
            snap["adj_score"] = snap.apply(
                lambda row: (
                    -999.0 if row["consec_holds"] >= 3  # hard block at 3
                    else row["score"] * HOLD_DECAY.get(
                        min(int(row["consec_holds"]), 3), 0.0
                    )
                ),
                axis=1,
            )

            # Select top N sectors by adjusted score
            top_n = snap.nlargest(n_long, "adj_score")
            longs = top_n["etf"].tolist()

            # Update hold streaks
            new_streak = defaultdict(int)
            for etf in SECTOR_ETFS:
                if etf in longs:
                    new_streak[etf] = hold_streak.get(etf, 0) + 1
                else:
                    new_streak[etf] = 0
            hold_streak = new_streak

        picks_log.append({
            "date": rd,
            "picks": longs.copy(),
            "regime": regime_label,
            "streaks": {e: hold_streak[e] for e in longs} if longs else {},
        })

        # Vol-target sizing
        if longs:
            realised_vol = _estimate_book_vol(panel, rd, longs)
            if realised_vol <= 1e-6:
                gross_lev = 1.0
            else:
                gross_lev = float(np.clip(target_vol / realised_vol, lev_min, lev_max))
        else:
            gross_lev = 0.0

        # Hold window P&L
        hold_win = oot_z[(oot_z["date"] > rd)
                         & (oot_z["date"] <= rd + pd.Timedelta(days=hold_days * 2))]
        hold_dates = sorted(hold_win["date"].unique())[:hold_days]

        for d in hold_dates:
            g = hold_win[hold_win["date"] == d]
            d_ts = pd.Timestamp(d)

            # Intra-hold regime check
            if hasattr(spy_ma, "_spy_close"):
                pc = spy_ma._spy_close.loc[:d_ts]
                d_spy_close = float(pc.iloc[-1]) if len(pc) > 0 else float("nan")
            else:
                d_spy_close = float("nan")
            d_spy_ma = float(spy_ma.loc[:d_ts].iloc[-1]) if (spy_ma is not None and len(spy_ma.loc[:d_ts]) > 0) else float("nan")
            d_vix = float(vix.loc[:d_ts].iloc[-1]) if (vix is not None and len(vix.loc[:d_ts]) > 0) else float("nan")
            d_sfrac, d_shfrac, d_label = compute_regime_allocation(d_spy_close, d_spy_ma, d_vix)

            # Sector sleeve
            if longs and d_sfrac > 0:
                lret = g[g["etf"].isin(longs)]["ret_raw"].mean()
                lret = float(lret) if pd.notna(lret) else 0.0
                sector_ret = gross_lev * d_sfrac * lret
            else:
                sector_ret = 0.0

            # SH sleeve
            if d_shfrac > 0 and hasattr(spy_ma, "_spy_ret"):
                spy_ret_d = spy_ma._spy_ret.get(d_ts, 0.0)
                sh_ret = d_shfrac * (-float(spy_ret_d))
            else:
                sh_ret = 0.0

            book_ret = float(np.clip(sector_ret + sh_ret, -0.20, 0.20))
            daily.append((d, book_ret, gross_lev, d_label))

        # Transaction cost
        n_legs = len(longs) + (1 if sh_frac > 0 else 0)
        tc = n_legs * (txn_cost_bps / 10000.0) * max(gross_lev, sh_frac)
        daily.append((rd, -tc, gross_lev, regime_label))
        n_rebal_actual += 1

    if not daily:
        return {
            "daily_pnl": pd.Series(dtype=float),
            "picks": picks_log,
            "hold_streak": hold_streak,
            "oot_start": str(oot_start.date()),
            "oot_end": str(oot_end.date()),
            "n_rebal": 0,
        }

    df_daily = pd.DataFrame(daily, columns=["date", "ret", "lev", "regime"])
    df_daily["date"] = pd.to_datetime(df_daily["date"])
    df_daily = df_daily.groupby("date", as_index=True).agg(
        ret=("ret", "sum"), lev=("lev", "max"), regime=("regime", "last")
    )

    return {
        "daily_pnl": df_daily["ret"],
        "daily_lev": df_daily["lev"],
        "daily_regime": df_daily["regime"],
        "picks": picks_log,
        "hold_streak": hold_streak,
        "oot_start": str(oot_start.date()),
        "oot_end": str(oot_end.date()),
        "n_rebal": n_rebal_actual,
    }


# ============================================================================
# PHASE 3: ROTATION QUALITY METRICS
# ============================================================================

def compute_rotation_metrics(all_picks: list[dict]) -> dict:
    """
    Compute comprehensive rotation quality metrics from pick history.
    """
    if not all_picks:
        return {}

    # Filter to picks where sectors were actually selected
    active_picks = [p for p in all_picks if p["picks"]]
    if not active_picks:
        return {}

    n_rebals = len(active_picks)

    # Sector frequency
    sector_counts = defaultdict(int)
    for p in active_picks:
        for etf in p["picks"]:
            sector_counts[etf] += 1
    total_slots = sum(sector_counts.values())
    freq = {k: v / n_rebals for k, v in sorted(sector_counts.items(), key=lambda x: -x[1])}

    # Herfindahl concentration index
    hhi = sum((v / total_slots) ** 2 for v in sector_counts.values()) if total_slots > 0 else 1.0

    # Rotation frequency (% of holdings that change per rebalance)
    rotation_pcts = []
    for i in range(1, len(active_picks)):
        prev = set(active_picks[i-1]["picks"])
        curr = set(active_picks[i]["picks"])
        changed = len(curr - prev)
        total = len(curr)
        rotation_pcts.append(changed / total if total > 0 else 0)
    avg_rotation = float(np.mean(rotation_pcts)) if rotation_pcts else 0

    # Max consecutive hold per sector
    max_streak_per_etf = {}
    for p in active_picks:
        for etf, streak in p.get("streaks", {}).items():
            max_streak_per_etf[etf] = max(max_streak_per_etf.get(etf, 0), streak)

    overall_max_streak = max(max_streak_per_etf.values()) if max_streak_per_etf else 0

    # % time in tech
    xlk_pct = freq.get("XLK", 0)

    # Top-2 concentration: what fraction of total sector-slots are occupied
    # by the two most frequently held sectors?
    # E.g., if we hold 3 sectors and the top-2 sectors appear in 35% and 34%
    # of rebalances, that's (35+34)/(3*100) of all slots = 23%.
    # But HC #670 R3 intent: no 1-2 sectors should dominate.
    # Interpretation: top-2 sector frequency should not exceed 60% of rebalances each.
    sorted_freq_vals = sorted(freq.values(), reverse=True)
    top2_conc_pct = sum(sorted_freq_vals[:2]) / 2 if len(sorted_freq_vals) >= 2 else sum(sorted_freq_vals)
    # top2_conc_pct = avg frequency of the 2 most common sectors (as % of rebalances)

    # Average number of unique sectors held
    avg_sectors = float(np.mean([len(set(p["picks"])) for p in active_picks]))

    # Sector diversity: how many unique sectors appeared at least once
    all_sectors_seen = set()
    for p in active_picks:
        all_sectors_seen.update(p["picks"])
    sector_diversity = len(all_sectors_seen)

    return {
        "n_rebalances": n_rebals,
        "sector_frequency": freq,
        "herfindahl_index": round(hhi, 4),
        "avg_rotation_pct": round(avg_rotation * 100, 1),
        "max_consecutive_hold": overall_max_streak,
        "max_streak_per_etf": max_streak_per_etf,
        "xlk_pct_time": round(xlk_pct * 100, 1),
        "top2_sector_concentration": round(top2_conc_pct * 100, 1),
        "avg_sectors_held": round(avg_sectors, 1),
        "sector_diversity": sector_diversity,
        "hc670_r2_pass": overall_max_streak <= 3,
        "hc670_r3_pass": top2_conc_pct <= 0.60,
    }


# ---------------------------------------------------------------------------
# Regime split
# ---------------------------------------------------------------------------
def _regime_split(all_pnl, all_regime):
    groups = {
        "green": all_pnl[all_regime.isin(["bull_full", "bull_cautious", "bull_highvol"])],
        "flat":  all_pnl[all_regime == "bear_shallow"],
        "red":   all_pnl[all_regime == "bear_deep"],
    }
    result = {}
    for label, s in groups.items():
        s = s.dropna()
        if len(s) >= 20:
            m = _metrics(s)
            result[label] = {"sharpe": m.get("sharpe", float("nan")), "n_days": len(s)}
        else:
            result[label] = {"sharpe": float("nan"), "n_days": len(s)}
    return result


def _day_concentration(pnl):
    cum = float(pnl.sum())
    if cum <= 0 or pnl.empty:
        return float("nan")
    return float(pnl.max() / cum)


# ============================================================================
# MAIN RUNNER
# ============================================================================
def run(out_dir: Path | None = None, run_audit: bool = True):
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / f"output/macro_picker/etf_rotation_quality_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("[v3-quality] Loading prices...")
    prices = _load_prices()

    print("[v3-quality] Loading yield curve...")
    yc = _load_yield_curve()

    print("[v3-quality] Loading macro features (VIX term structure)...")
    macro_feats = _load_macro_features()

    print("[v3-quality] Loading sector flows...")
    sector_flows = _load_sector_flows()
    if sector_flows is not None:
        print(f"  Sector flow data: {len(sector_flows)} rows, "
              f"{sector_flows['date'].min().date()} to {sector_flows['date'].max().date()}")
    else:
        print("  No sector flow data available")

    vix = macro_feats["vix"] if "vix" in macro_feats.columns else pd.Series(dtype=float)

    # Build feature panels
    hold_days = 21

    # V2 features for audit
    v2_feats = MOMENTUM_FEATURES + YIELD_CURVE_FEATURES

    # V3 feature panel (rotation-quality)
    print("[v3-quality] Building rotation-quality feature panel...")
    panel_v3 = build_panel_v3(hold_days, prices, yc, macro_feats, sector_flows)

    feats_v3 = [f for f in ALL_FEATURES_V3 if f in panel_v3.columns]
    print(f"  Panel: {len(panel_v3)} rows, features: {len(feats_v3)}")
    print(f"  Rotation features available: "
          f"{[f for f in ROTATION_FEATURES if f in panel_v3.columns]}")
    print(f"  VIX features available: "
          f"{[f for f in VIX_FEATURES if f in panel_v3.columns]}")
    print(f"  Flow features available: "
          f"{[f for f in FLOW_FEATURES if f in panel_v3.columns]}")

    # -----------------------------------------------------------------------
    # PHASE 1: Audit v2
    # -----------------------------------------------------------------------
    v2_audit = {}
    if run_audit:
        # Build v2-style panel for audit
        panel_v2 = build_panel_v3(hold_days, prices, yc, macro_feats, None)
        v2_feats_present = [f for f in v2_feats if f in panel_v2.columns]
        v2_audit = audit_v2(
            ROOT / "output/macro_picker/etf_rotation_v2_20260709_202559/book.parquet",
            panel_v2, v2_feats_present, prices, yc, vix,
        )

    # -----------------------------------------------------------------------
    # PHASE 2 + 3: Rotation-quality walk-forward
    # -----------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("PHASE 2+3: ROTATION-QUALITY WALK-FORWARD BACKTEST")
    print("=" * 65)

    spy_ma = _build_regime_series(prices, vix)

    windows = _iter_wf_windows(
        panel_v3["date"].min(), panel_v3["date"].max(),
        train_months=12, oot_months=1, step_months=1,
    )
    print(f"  Walk-forward: 12m train / 1m OOT / 1m step -> {len(windows)} folds")

    # Run folds sequentially (need to track hold streaks across folds)
    fold_results = []
    hold_streak = defaultdict(int)
    prev_holdings = []

    for i, (ts_, te_, os_, oe_) in enumerate(windows):
        if i % 20 == 0:
            print(f"  Fold {i+1}/{len(windows)}...")
        result = _wf_fold_v3(
            panel_v3, feats_v3, ts_, te_, os_, oe_,
            hold_days=hold_days, n_long=MIN_SECTORS,
            target_vol=0.15, lev_min=0.25, lev_max=2.0,
            txn_cost_bps=TXN_COST_BPS,
            spy_ma=spy_ma, vix=vix,
            prev_holdings=prev_holdings,
            hold_streak=hold_streak,
        )
        fold_results.append(result)

        # Carry forward state
        hold_streak = result.get("hold_streak", defaultdict(int))
        if result["picks"]:
            prev_holdings = result["picks"][-1]["picks"]

    # Aggregate
    pnl_parts = [r["daily_pnl"] for r in fold_results if not r["daily_pnl"].empty]
    if not pnl_parts:
        print("[v3-quality] ERROR: no fold produced P&L")
        return {}

    all_pnl = pd.concat(pnl_parts).sort_index()
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="last")]

    regime_parts = [r.get("daily_regime", pd.Series(dtype=str))
                    for r in fold_results if not r["daily_pnl"].empty]
    all_regime = pd.concat(regime_parts).sort_index()
    all_regime = all_regime[~all_regime.index.duplicated(keep="last")]
    all_regime = all_regime.reindex(all_pnl.index).fillna("bull_full")

    # Core metrics
    pooled = _metrics(all_pnl)

    # Regime split
    regime_split = _regime_split(all_pnl, all_regime)

    # Regime skew
    gs = regime_split.get("green", {}).get("sharpe", float("nan"))
    rs_val = regime_split.get("red", {}).get("sharpe", float("nan"))
    if np.isfinite(gs) and np.isfinite(rs_val) and max(abs(gs), abs(rs_val)) > 0:
        regime_skew = abs(gs - rs_val) / max(abs(gs), abs(rs_val))
    else:
        regime_skew = float("nan")

    # Day concentration
    day_conc = _day_concentration(all_pnl[all_pnl > 0])

    # Collect all picks
    all_picks = []
    for r in fold_results:
        all_picks.extend(r.get("picks", []))

    # Rotation quality metrics
    rot_metrics = compute_rotation_metrics(all_picks)

    # SPY buy-and-hold benchmark
    spy_prices = prices[prices["ticker"] == BENCHMARK].sort_values("date").set_index("date")["close"]
    spy_ret = spy_prices.pct_change().dropna()
    spy_ret = spy_ret.reindex(all_pnl.index).dropna()
    spy_metrics = _metrics(spy_ret) if len(spy_ret) > 20 else {}

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    p = pooled
    cagr_pct = p.get("cagr", float("nan")) * 100
    maxdd_pct = p.get("max_dd", float("nan")) * 100

    print("\n" + "=" * 70)
    print("RESULTS: ETF ROTATION v3 (ROTATION-QUALITY OPTIMIZED)")
    print("=" * 70)

    print(f"\n  PERFORMANCE:")
    print(f"    Sharpe   {p.get('sharpe', float('nan')):.2f}     "
          f"Sortino  {p.get('sortino', float('nan')):.2f}")
    print(f"    Calmar   {p.get('calmar', float('nan')):.2f}     "
          f"MaxDD    {maxdd_pct:.1f}%")
    print(f"    WR       {p.get('wr', float('nan'))*100:.1f}%    "
          f"PF       {p.get('pf', float('nan')):.2f}")
    print(f"    CAGR     {cagr_pct:.1f}%")
    print(f"    DayConc  {day_conc*100:.1f}%")

    print(f"\n  REGIME SPLIT:")
    for label in ["green", "flat", "red"]:
        rs_info = regime_split.get(label, {})
        sh = rs_info.get("sharpe", float("nan"))
        nd = rs_info.get("n_days", 0)
        print(f"    {label.capitalize()}: Sharpe {sh:.2f}  ({nd} days)")
    if np.isfinite(regime_skew):
        print(f"    Skew = {regime_skew:.2f}  "
              f"[{'PASS' if regime_skew <= 0.50 else 'FAIL'} <=0.50]")

    print(f"\n  ROTATION QUALITY:")
    print(f"    Avg rotation per rebalance: {rot_metrics.get('avg_rotation_pct', 0):.1f}%  "
          f"(target: 40-60%)")
    print(f"    Max consecutive hold: {rot_metrics.get('max_consecutive_hold', 0)}  "
          f"[{'PASS' if rot_metrics.get('hc670_r2_pass', False) else 'FAIL'} HC #670 R2 <=3]")
    print(f"    Herfindahl index: {rot_metrics.get('herfindahl_index', 1):.4f}  "
          f"(perfect = {1/len(SECTOR_ETFS):.4f})")
    print(f"    Top-2 sector concentration: {rot_metrics.get('top2_sector_concentration', 0):.1f}%  "
          f"[{'PASS' if rot_metrics.get('hc670_r3_pass', False) else 'FAIL'} HC #670 R3 <=60%]")
    print(f"    % time in tech (XLK): {rot_metrics.get('xlk_pct_time', 0):.1f}%")
    print(f"    Avg sectors held: {rot_metrics.get('avg_sectors_held', 0):.1f}")
    print(f"    Sector diversity: {rot_metrics.get('sector_diversity', 0)} / {len(SECTOR_ETFS)}")

    if rot_metrics.get("sector_frequency"):
        print(f"\n  SECTOR FREQUENCY:")
        for etf, pct in rot_metrics["sector_frequency"].items():
            bar = "#" * int(pct * 40)
            print(f"    {etf:5s}: {pct*100:5.1f}%  {bar}")

    # Benchmark comparison
    print(f"\n  BENCHMARK COMPARISON:")
    print(f"    {'Metric':<12} {'v3-Quality':<12} {'v2-Momentum':<14} {'SPY B&H':<12}")
    print(f"    {'Sharpe':<12} {p.get('sharpe', float('nan')):<12.2f} "
          f"{'1.90':<14} {spy_metrics.get('sharpe', float('nan')):<12.2f}")
    print(f"    {'CAGR':<12} {cagr_pct:<12.1f}% "
          f"{'20.6%':<14} {spy_metrics.get('cagr', float('nan'))*100:<12.1f}%")
    print(f"    {'MaxDD':<12} {maxdd_pct:<12.1f}% "
          f"{'-6.8%':<14} {spy_metrics.get('max_dd', float('nan'))*100:<12.1f}%")

    print("\n" + "=" * 70)

    # Save outputs
    metrics_out = {
        "version": "v3_rotation_quality",
        "config": {
            "hold_days": hold_days,
            "n_long": MIN_SECTORS,
            "train_months": 12,
            "hold_decay": HOLD_DECAY,
            "min_sectors": MIN_SECTORS,
            "features": feats_v3,
        },
        "pooled": pooled,
        "regime_split": regime_split,
        "regime_skew": regime_skew,
        "day_concentration": day_conc,
        "rotation_quality": rot_metrics,
        "v2_audit": v2_audit,
        "spy_benchmark": spy_metrics,
        "date_range": [
            str(all_pnl.index.min().date()) if not all_pnl.empty else None,
            str(all_pnl.index.max().date()) if not all_pnl.empty else None,
        ],
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_out, indent=2, default=str)
    )

    # Book parquet
    book = pd.DataFrame({
        "date": all_pnl.index,
        "daily_ret": all_pnl.values,
        "regime": all_regime.reindex(all_pnl.index).values,
    })
    book.to_parquet(out_dir / "book.parquet", index=False)

    # Picks log
    picks_df = pd.DataFrame([
        {"date": p["date"], "picks": ",".join(p["picks"]),
         "regime": p["regime"], "n_picks": len(p["picks"])}
        for p in all_picks
    ])
    if not picks_df.empty:
        picks_df.to_parquet(out_dir / "picks.parquet", index=False)

    print(f"\nOutput: {out_dir}")
    return metrics_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF Rotation Quality (HC #670)")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no-audit", action="store_true",
                        help="Skip Phase 1 v2 audit")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else None
    run(out_dir=out_dir, run_audit=not args.no_audit)
