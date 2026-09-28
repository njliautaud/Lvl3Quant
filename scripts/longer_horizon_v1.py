#!/usr/bin/env python3
"""HC #539 R1 follow-up — Longer-horizon execution test on Feb-Apr 2026 ES.

GOAL: Test whether ANY tradeable strategy exists at retail-grade execution at
longer hold horizons {1m, 5m, 15m, 30m, 1h, 1d} given that microstructure
direction (1s/5s/30s) is settled as non-tradeable.

Three signal families on the same 32-day OOT window (2026-02-23 → 2026-04-14):

  Family A — v3.4.2 30s pred extended to longer holds.
    For each day, take top {1%, 5%, 10%, 20%} |pred_log_ret_30s| signals.
    Map each sample's position (within the day's sample array) to a proportional
    time-of-day inside the RTH window, snap to the corresponding minute bar,
    and compute PnL from minute-bar close-to-close at hold horizons
    {1m, 5m, 15m, 30m}. Test BOTH predicted-side and inverted side (HC #537
    found 30s head sign-anti-predictive at top-conf).

  Family B — Classical minute-bar signals (built from minute bars in
    data/processed/mbo_minute_bars_v1/YYYYMMDD.parquet):
      B1: Momentum (sign of N-min return)         {5m,15m,30m}
      B2: Mean-reversion (z-score over rolling 60m, fade if |z|>2)
      B3: Volatility breakout (close > prev_high + k*ATR)
    Hold horizons: {15m, 30m, 60m, EOD}. Top {25%, 50%} magnitude.

  Family C — Day-classifier driven swing.
    Use cross_asset_day_classifier_v1 LOO predictions (p_profitable_combo)
    and VIX_change_5d single-feature champion (AUC=0.848) to pick days.
    Entry: 30 min after open. Exit: {60m, 120m, EOD}. Direction: positive
    expected_return signal from in-sample correlation of feature with
    day_mean_net_realized.

COST MODEL (CANONICAL HC #74):
  ES market RT = 1.376 ticks  (commission + 1 tick spread crossing)
  ES passive RT = 0.376 ticks (commission only — REPORTED FOR INFO ONLY;
                               headline gates use market exec per HC #539 R3
                               because a rigorous queue/adverse-selection
                               simulation is not implemented here.)

METRICS (HC #69 — risk-adjusted primary):
  per-day Sharpe (annualized), Sortino, PF, WR
  N trades, N days, mean daily $/contract
  Regime gap (green/red/flat per SPX close-to-close ±0.10%)
  Day concentration

GATES (HC #428 R1, slightly relaxed for fewer trades/day at long horizons):
  per-day Sharpe > 1.5
  per-day PF > 1.4
  per-day WR > 55%
  regime gap ≤ 0.50
  day-conc ≤ 0.70
  N days ≥ 20
  N trades ≥ 50

OUTPUTS:
  output/longer_horizon_v1/results_table.csv
  output/longer_horizon_v1/per_day_breakdown.csv
  output/longer_horizon_v1/survivors.txt
  output/longer_horizon_v1/verdict.md
  MLflow experiment 'longer_horizon_v1'
"""
from __future__ import annotations
import os, sys, json, glob, math, time, traceback
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
OOT_NPZ_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
MINUTE_BARS_DIR = ROOT / "data/processed/mbo_minute_bars_v1"
CLASSIFIER_DIR = ROOT / "output/cross_asset_day_classifier_v1"
OUT_DIR = ROOT / "output/longer_horizon_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# OOT window
OOT_DATES = [
    "20260223","20260224","20260225","20260226","20260227",
    "20260302","20260303","20260304","20260305","20260306",
    "20260309","20260310","20260311","20260312","20260313",
    "20260316","20260317","20260318","20260319",
    "20260401","20260402","20260403","20260406","20260407",
    "20260408","20260409","20260410","20260413","20260414",
]
# Note: oot_npz also has 20260301 (Sun), 20260308 (Sun), 20260315 (Sun), 20260405 (Sun), 20260412 (Sun)
# We exclude weekends. Above is 29 weekday dates.

# Costs
ES_TICK_VALUE = 12.50
ES_MARKET_RT_TICKS = 1.376
ES_PASSIVE_RT_TICKS = 0.376
ES_MARKET_RT_USD = ES_MARKET_RT_TICKS * ES_TICK_VALUE  # $17.20
ES_PASSIVE_RT_USD = ES_PASSIVE_RT_TICKS * ES_TICK_VALUE  # $4.70
PASSIVE_FILL_DISCOUNT = 0.50  # conservative adverse-selection discount

# Prices in minute_bars are in TICKS (1 unit = 0.25 ES index points).
# So PnL_in_ticks = (close_exit - close_entry)  (already in ticks)
# PnL_in_dollars = ticks * ES_TICK_VALUE

# RTH window
RTH_START_MIN = 13*60 + 30   # 13:30 UTC = 9:30 ET
RTH_END_MIN   = 21*60        # 21:00 UTC = 17:00 ET (file is actually 13:30-21:00 = 450min)
# We'll use only 13:30-20:00 UTC = 9:30-16:00 ET for the strict RTH window
RTH_END_MIN_STRICT = 20*60   # 20:00 UTC = 16:00 ET

# Gates (HC #428 R1)
GATE_SHARPE = 1.5
GATE_PF = 1.4
GATE_WR = 0.55
GATE_REGIME_GAP = 0.50
GATE_DAY_CONC = 0.70
GATE_N_DAYS = 20
GATE_N_TRADES = 50

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def load_minute_bars(date_str: str) -> pd.DataFrame | None:
    fp = MINUTE_BARS_DIR / f"{date_str}.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    df = df.sort_values("ts_minute").reset_index(drop=True)
    # Filter to strict RTH 13:30-20:00 UTC (9:30-16:00 ET)
    ts = df["ts_minute"].dt.tz_convert("UTC")
    minute_of_day = ts.dt.hour * 60 + ts.dt.minute
    df = df.loc[(minute_of_day >= RTH_START_MIN) & (minute_of_day < RTH_END_MIN_STRICT)].reset_index(drop=True)
    df["minute_of_day"] = ts.loc[df.index].dt.hour * 60 + ts.loc[df.index].dt.minute  # may be misaligned; recompute below
    df["minute_of_day"] = df["ts_minute"].dt.tz_convert("UTC").dt.hour * 60 + df["ts_minute"].dt.tz_convert("UTC").dt.minute
    return df


def annualized_sharpe(daily_pnls: np.ndarray) -> float:
    if len(daily_pnls) < 2:
        return 0.0
    sd = daily_pnls.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(daily_pnls.mean() / sd * np.sqrt(252))


def sortino(daily_pnls: np.ndarray) -> float:
    if len(daily_pnls) < 2:
        return 0.0
    neg = daily_pnls[daily_pnls < 0]
    if len(neg) == 0:
        return 999.0
    dd = neg.std(ddof=1)
    if dd == 0:
        return 999.0
    return float(daily_pnls.mean() / dd * np.sqrt(252))


def profit_factor(trade_pnls: np.ndarray) -> float:
    pos = trade_pnls[trade_pnls > 0].sum()
    neg = -trade_pnls[trade_pnls < 0].sum()
    if neg == 0:
        return 999.0 if pos > 0 else 0.0
    return float(pos / neg)


def max_drawdown(daily_pnls: np.ndarray) -> float:
    if len(daily_pnls) == 0:
        return 0.0
    cum = daily_pnls.cumsum()
    peak = np.maximum.accumulate(cum)
    return float((peak - cum).max())


def day_classify(daily_pnls_dict: dict[str, float], spx_change_dict: dict[str, float] | None = None) -> dict[str, str]:
    """Classify each date as green/red/flat based on SPX close-to-close ±0.10%."""
    # We don't have SPX directly, but we have ES close-to-close which is a very close proxy.
    if spx_change_dict is None:
        return {}
    out = {}
    for d, ch in spx_change_dict.items():
        if ch > 0.001:
            out[d] = "green"
        elif ch < -0.001:
            out[d] = "red"
        else:
            out[d] = "flat"
    return out


def regime_gap(per_day_pnls: dict[str, float], regimes: dict[str, str]) -> float:
    g = [p for d, p in per_day_pnls.items() if regimes.get(d) == "green"]
    r = [p for d, p in per_day_pnls.items() if regimes.get(d) == "red"]
    if len(g) < 2 or len(r) < 2:
        return 0.0
    g_arr = np.asarray(g)
    r_arr = np.asarray(r)
    sg = g_arr.mean() / g_arr.std(ddof=1) * np.sqrt(252) if g_arr.std(ddof=1) > 0 else 0.0
    sr = r_arr.mean() / r_arr.std(ddof=1) * np.sqrt(252) if r_arr.std(ddof=1) > 0 else 0.0
    denom = max(abs(sg), abs(sr))
    if denom == 0:
        return 0.0
    return float(abs(sg - sr) / denom)


def day_concentration(per_day_pnls: dict[str, float]) -> float:
    vals = np.asarray(list(per_day_pnls.values()))
    if vals.sum() <= 0:
        return 1.0
    abs_vals = np.abs(vals)
    top1 = abs_vals.max()
    return float(top1 / abs_vals.sum())


# -----------------------------------------------------------------------------
# Build SPX-proxy day classification from minute bars (ES close-to-close)
# -----------------------------------------------------------------------------

def build_day_classification() -> dict[str, str]:
    """ES close-to-close percent change → green/red/flat with ±0.10% threshold."""
    closes = {}
    for d in OOT_DATES:
        mb = load_minute_bars(d)
        if mb is None or len(mb) == 0:
            continue
        closes[d] = (mb.iloc[0]["close"], mb.iloc[-1]["close"])
    # Compute day return as last-close minus first-open of THIS day (intraday convention)
    regimes = {}
    for d, (o, c) in closes.items():
        ret = (c - o) / o  # using tick units, ratio is fine
        if ret > 0.001:
            regimes[d] = "green"
        elif ret < -0.001:
            regimes[d] = "red"
        else:
            regimes[d] = "flat"
    return regimes


# -----------------------------------------------------------------------------
# Family A — v3.4.2 30s pred extended to longer holds
# -----------------------------------------------------------------------------

def family_A_signals():
    """For each OOT date: load preds, sample top X% |pred_log_ret_30s|,
    map to minute index by proportional position within RTH, compute hold-PnL
    at horizons h in minutes from minute-bar close prices.

    Returns a per-trade DataFrame with cols:
      date, side, horizon_m, pct, entry_minute, entry_close, exit_close, pnl_ticks
    """
    rows = []
    for d in OOT_DATES:
        f = OOT_NPZ_DIR / f"oot_{d}.npz"
        if not f.exists():
            continue
        mb = load_minute_bars(d)
        if mb is None or len(mb) < 30:
            continue
        npz = np.load(f, allow_pickle=True)
        pred = npz["pred_log_ret_30s"]
        n_pred = len(pred)
        if n_pred < 100:
            continue
        n_min = len(mb)  # number of minute bars in strict RTH

        # Map sample index → minute index proportionally
        sample_idx = np.arange(n_pred)
        # Each sample's RTH-fraction:
        rth_frac = sample_idx / max(n_pred - 1, 1)
        # Map to entry minute_idx (0 .. n_min-1)
        entry_min_idx = (rth_frac * (n_min - 1)).astype(int)

        abs_pred = np.abs(pred)
        # For each top-X%:
        for pct in [0.01, 0.05, 0.10, 0.20]:
            k = int(n_pred * pct)
            if k < 10:
                continue
            # threshold at top-pct
            thresh = np.partition(abs_pred, n_pred - k)[n_pred - k]
            mask = abs_pred >= thresh
            sel_idx = np.where(mask)[0]
            # Deduplicate entries by minute (one trade per minute max)
            chosen_min = entry_min_idx[sel_idx]
            seen = {}
            for s_i, m_i in zip(sel_idx, chosen_min):
                if m_i not in seen:
                    seen[m_i] = s_i
            for m_i, s_i in seen.items():
                sgn = 1 if pred[s_i] > 0 else -1  # predicted side
                entry_close = float(mb.iloc[m_i]["close"])
                # For each horizon h, compute exit minute and exit close
                for h_m in [1, 5, 15, 30]:
                    exit_idx = m_i + h_m
                    if exit_idx >= n_min:
                        continue
                    exit_close = float(mb.iloc[exit_idx]["close"])
                    gross_ticks = (exit_close - entry_close) * sgn  # prices already in ticks
                    rows.append({
                        "family": "A",
                        "date": d,
                        "signal": "v342_30s",
                        "side_label": "predicted",
                        "horizon_m": h_m,
                        "pct": pct,
                        "entry_minute": m_i,
                        "entry_close": entry_close,
                        "exit_close": exit_close,
                        "gross_ticks": gross_ticks,
                        "pred_sign": sgn,
                    })
                    # Inverted side (HC #537 anti-predictive test)
                    rows.append({
                        "family": "A_inv",
                        "date": d,
                        "signal": "v342_30s_INV",
                        "side_label": "inverted",
                        "horizon_m": h_m,
                        "pct": pct,
                        "entry_minute": m_i,
                        "entry_close": entry_close,
                        "exit_close": exit_close,
                        "gross_ticks": -gross_ticks,
                        "pred_sign": -sgn,
                    })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Family B — Classical minute-bar signals
# -----------------------------------------------------------------------------

def family_B_signals():
    """B1: Momentum sign of N-min return; B2: Mean reversion z-score; B3: ATR breakout."""
    rows = []
    for d in OOT_DATES:
        mb = load_minute_bars(d)
        if mb is None or len(mb) < 70:
            continue
        n = len(mb)
        close = mb["close"].values.astype(np.float64)
        high = mb["high"].values.astype(np.float64)
        low = mb["low"].values.astype(np.float64)

        # --- B1: momentum sign (5m, 15m, 30m return signal) ---
        for lookback_m in [5, 15, 30]:
            sig = np.full(n, np.nan)
            for i in range(lookback_m, n):
                sig[i] = close[i] - close[i - lookback_m]
            # Each minute that has a valid signal → ranked across the day
            valid_idx = np.where(~np.isnan(sig))[0]
            if len(valid_idx) < 50:
                continue
            mags = np.abs(sig[valid_idx])
            for top_pct in [0.25, 0.50]:
                k = int(len(valid_idx) * top_pct)
                if k < 5:
                    continue
                thresh = np.partition(mags, len(mags) - k)[len(mags) - k]
                sel_mask = mags >= thresh
                sel_minutes = valid_idx[sel_mask]
                sel_signs = np.sign(sig[sel_minutes])
                for m_i, sgn in zip(sel_minutes, sel_signs):
                    if sgn == 0:
                        continue
                    for h_m in [15, 30, 60, "EOD"]:
                        if h_m == "EOD":
                            exit_idx = n - 1
                        else:
                            exit_idx = m_i + h_m
                            if exit_idx >= n:
                                continue
                        if exit_idx <= m_i:
                            continue
                        gross_ticks = (close[exit_idx] - close[m_i]) * sgn
                        rows.append({
                            "family": "B1",
                            "date": d,
                            "signal": f"B1_mom_{lookback_m}m",
                            "side_label": "momentum",
                            "horizon_m": h_m if h_m != "EOD" else 9999,
                            "pct": top_pct,
                            "entry_minute": int(m_i),
                            "entry_close": float(close[m_i]),
                            "exit_close": float(close[exit_idx]),
                            "gross_ticks": float(gross_ticks),
                            "pred_sign": int(sgn),
                        })

        # --- B2: Mean reversion z-score (over rolling 60m of 1-min returns) ---
        rets = np.diff(close, prepend=close[0])
        # rolling 60m mean, std
        win = 60
        cum = np.concatenate([[0], np.cumsum(rets)])
        roll_mean = (cum[win:] - cum[:-win]) / win  # len n-win+1
        # rolling std via cumsum of squares
        sq = rets ** 2
        cum_sq = np.concatenate([[0], np.cumsum(sq)])
        roll_var = (cum_sq[win:] - cum_sq[:-win]) / win - roll_mean ** 2
        roll_std = np.sqrt(np.maximum(roll_var, 1e-12))
        # roll arrays index i corresponds to bar (win-1+i) for ret[win-1+i] z = (ret[i] - mean)/std
        # We'll compute z at bar index i = win-1 + j for j in 0..len(roll_mean)-1
        z_arr = np.full(n, np.nan)
        for j in range(len(roll_mean)):
            i = win - 1 + j
            z_arr[i] = (rets[i] - roll_mean[j]) / roll_std[j]
        valid_idx = np.where((~np.isnan(z_arr)) & (np.abs(z_arr) > 2.0))[0]
        if len(valid_idx) >= 10:
            mags = np.abs(z_arr[valid_idx])
            for top_pct in [0.25, 0.50, 1.0]:
                k = max(int(len(valid_idx) * top_pct), 1)
                if k < 1:
                    continue
                thresh = np.partition(mags, len(mags) - k)[len(mags) - k] if k < len(mags) else mags.min()
                sel_mask = mags >= thresh
                sel_minutes = valid_idx[sel_mask]
                # Mean revert: fade the move → side = -sign(z)
                sel_signs = -np.sign(z_arr[sel_minutes])
                for m_i, sgn in zip(sel_minutes, sel_signs):
                    if sgn == 0:
                        continue
                    for h_m in [15, 30, 60, "EOD"]:
                        if h_m == "EOD":
                            exit_idx = n - 1
                        else:
                            exit_idx = m_i + h_m
                            if exit_idx >= n:
                                continue
                        if exit_idx <= m_i:
                            continue
                        gross_ticks = (close[exit_idx] - close[m_i]) * sgn
                        rows.append({
                            "family": "B2",
                            "date": d,
                            "signal": "B2_meanrev_z2",
                            "side_label": "meanrev",
                            "horizon_m": h_m if h_m != "EOD" else 9999,
                            "pct": top_pct,
                            "entry_minute": int(m_i),
                            "entry_close": float(close[m_i]),
                            "exit_close": float(close[exit_idx]),
                            "gross_ticks": float(gross_ticks),
                            "pred_sign": int(sgn),
                        })

        # --- B3: Volatility breakout (close > prev_high_30 + k*ATR) ---
        atr_win = 30
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ])
        atr = np.full(n, np.nan)
        for i in range(atr_win, n):
            atr[i] = tr[i - atr_win + 1:i + 1].mean()
        for k_mult in [0.5, 1.0]:
            sig = np.zeros(n)
            for i in range(atr_win + 1, n):
                prev_high = high[i - atr_win:i].max()
                prev_low = low[i - atr_win:i].min()
                up_thresh = prev_high + k_mult * atr[i]
                dn_thresh = prev_low - k_mult * atr[i]
                if close[i] > up_thresh:
                    sig[i] = 1
                elif close[i] < dn_thresh:
                    sig[i] = -1
            valid_idx = np.where(sig != 0)[0]
            if len(valid_idx) < 5:
                continue
            for m_i in valid_idx:
                sgn = int(sig[m_i])
                for h_m in [15, 30, 60, "EOD"]:
                    if h_m == "EOD":
                        exit_idx = n - 1
                    else:
                        exit_idx = m_i + h_m
                        if exit_idx >= n:
                            continue
                    if exit_idx <= m_i:
                        continue
                    gross_ticks = (close[exit_idx] - close[m_i]) * sgn
                    rows.append({
                        "family": "B3",
                        "date": d,
                        "signal": f"B3_atr_k{k_mult}",
                        "side_label": "breakout",
                        "horizon_m": h_m if h_m != "EOD" else 9999,
                        "pct": 1.0,
                        "entry_minute": int(m_i),
                        "entry_close": float(close[m_i]),
                        "exit_close": float(close[exit_idx]),
                        "gross_ticks": float(gross_ticks),
                        "pred_sign": int(sgn),
                    })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Family C — Day-classifier driven swing
# -----------------------------------------------------------------------------

def family_C_signals():
    """Use cross_asset_day_classifier_v1: VIX_change_5d (desc → low VIX_change = profitable)
    and combined_features.parquet for day-level direction expectation."""
    feat_path = CLASSIFIER_DIR / "combined_features.parquet"
    if not feat_path.exists():
        return pd.DataFrame()
    feats = pd.read_parquet(feat_path)
    feats["date"] = feats["date"].astype(str)

    # We use VIX_change_5d (the AUC=0.848 champion). "desc" means LOW VIX_change_5d → profitable.
    # The classifier predicts "profitable day" but doesn't give DIRECTION.
    # For direction, use the existing in-sample correlation: from the labels, profitable
    # days had positive day_mean_net_realized → trade LONG on classified-profitable days.
    # Cross-check: the classifier was built on label_profitable from an ES strategy that
    # has a long bias historically. We'll trade LONG on top-K classifier days.

    # K-rank: top 5, 10 days by VIX_change_5d ascending (lowest VIX rise = best signal)
    rows = []
    feats_in_oot = feats[feats["date"].isin(OOT_DATES)].copy()
    feats_in_oot = feats_in_oot.sort_values("VIX_change_5d", ascending=True).reset_index(drop=True)
    n_avail = len(feats_in_oot)

    for top_k in [5, 10, 15]:
        if top_k > n_avail:
            continue
        winners = set(feats_in_oot.head(top_k)["date"].tolist())
        for d in winners:
            mb = load_minute_bars(d)
            if mb is None or len(mb) < 90:
                continue
            close = mb["close"].values.astype(np.float64)
            # Entry: 30 min after RTH open (i.e. minute index 30, since file already starts at 13:30 UTC)
            entry_idx = 30
            if entry_idx >= len(mb):
                continue
            entry_close = float(close[entry_idx])
            # Direction: LONG (classifier expects up-day)
            sgn = 1
            for h_m in [60, 120, "EOD"]:
                if h_m == "EOD":
                    exit_idx = len(mb) - 1
                else:
                    exit_idx = entry_idx + h_m
                    if exit_idx >= len(mb):
                        continue
                exit_close = float(close[exit_idx])
                gross_ticks = (exit_close - entry_close) * sgn
                rows.append({
                    "family": "C",
                    "date": d,
                    "signal": f"C_dayclf_top{top_k}",
                    "side_label": "long",
                    "horizon_m": h_m if h_m != "EOD" else 9999,
                    "pct": top_k / max(n_avail, 1),
                    "entry_minute": entry_idx,
                    "entry_close": entry_close,
                    "exit_close": exit_close,
                    "gross_ticks": float(gross_ticks),
                    "pred_sign": sgn,
                })
        # Also test SHORT version (in case the signal is anti-predictive)
        for d in winners:
            mb = load_minute_bars(d)
            if mb is None or len(mb) < 90:
                continue
            close = mb["close"].values.astype(np.float64)
            entry_idx = 30
            entry_close = float(close[entry_idx])
            sgn = -1
            for h_m in [60, 120, "EOD"]:
                if h_m == "EOD":
                    exit_idx = len(mb) - 1
                else:
                    exit_idx = entry_idx + h_m
                    if exit_idx >= len(mb):
                        continue
                exit_close = float(close[exit_idx])
                gross_ticks = (exit_close - entry_close) * sgn
                rows.append({
                    "family": "C_short",
                    "date": d,
                    "signal": f"C_dayclf_top{top_k}_SHORT",
                    "side_label": "short",
                    "horizon_m": h_m if h_m != "EOD" else 9999,
                    "pct": top_k / max(n_avail, 1),
                    "entry_minute": entry_idx,
                    "entry_close": entry_close,
                    "exit_close": exit_close,
                    "gross_ticks": float(gross_ticks),
                    "pred_sign": sgn,
                })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Aggregate per-cell metrics
# -----------------------------------------------------------------------------

def aggregate_cells(trades: pd.DataFrame, regimes: dict[str, str], cost_ticks: float = ES_MARKET_RT_TICKS) -> pd.DataFrame:
    """Group by (family, signal, side_label, horizon_m, pct), compute metrics."""
    if len(trades) == 0:
        return pd.DataFrame()
    out_rows = []
    grp_cols = ["family", "signal", "side_label", "horizon_m", "pct"]
    for keys, sub in trades.groupby(grp_cols):
        sub = sub.copy()
        sub["net_ticks"] = sub["gross_ticks"] - cost_ticks
        sub["pnl_usd"] = sub["net_ticks"] * ES_TICK_VALUE
        n_trades = len(sub)
        # Per-day aggregation
        per_day = sub.groupby("date")["pnl_usd"].sum()
        per_day_arr = per_day.values
        n_days = len(per_day)
        if n_days < 2 or n_trades < 5:
            continue
        sharpe = annualized_sharpe(per_day_arr)
        sortino_v = sortino(per_day_arr)
        pf = profit_factor(sub["pnl_usd"].values)
        wr_trade = float((sub["pnl_usd"] > 0).mean())
        wr_day = float((per_day_arr > 0).mean())
        mean_daily_usd = float(per_day_arr.mean())
        mdd = max_drawdown(per_day_arr)
        per_day_dict = per_day.to_dict()
        reg_gap = regime_gap(per_day_dict, regimes)
        day_conc = day_concentration(per_day_dict)
        # Regime stratification
        greens = [v for k, v in per_day_dict.items() if regimes.get(k) == "green"]
        reds   = [v for k, v in per_day_dict.items() if regimes.get(k) == "red"]
        flats  = [v for k, v in per_day_dict.items() if regimes.get(k) == "flat"]
        passes = (
            sharpe > GATE_SHARPE
            and pf > GATE_PF
            and wr_day > GATE_WR
            and reg_gap <= GATE_REGIME_GAP
            and day_conc <= GATE_DAY_CONC
            and n_days >= GATE_N_DAYS
            and n_trades >= GATE_N_TRADES
        )
        family, signal, side_label, horizon_m, pct = keys
        out_rows.append({
            "family": family,
            "signal": signal,
            "side_label": side_label,
            "horizon_m": horizon_m,
            "pct": pct,
            "cost_model": "ES_market_RT",
            "cost_ticks": cost_ticks,
            "n_trades": n_trades,
            "n_days": n_days,
            "trades_per_day": n_trades / n_days,
            "mean_daily_usd": mean_daily_usd,
            "sharpe": sharpe,
            "sortino": sortino_v,
            "pf": pf,
            "wr_day": wr_day,
            "wr_trade": wr_trade,
            "mdd_usd": mdd,
            "day_conc": day_conc,
            "regime_gap": reg_gap,
            "n_green": len(greens),
            "n_red": len(reds),
            "n_flat": len(flats),
            "mean_daily_green_usd": float(np.mean(greens)) if greens else 0.0,
            "mean_daily_red_usd": float(np.mean(reds)) if reds else 0.0,
            "passes_gates": bool(passes),
        })
    return pd.DataFrame(out_rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    t0 = time.time()
    print(f"[longer_horizon_v1] starting at {datetime.now()}")
    sys.stdout.flush()

    # MLflow logging (optional)
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("longer_horizon_v1")
        ml_run = mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow_active = True
    except Exception as e:
        print(f"[mlflow] not available, continuing without: {e}")
        mlflow_active = False

    # Day regime classification
    regimes = build_day_classification()
    print(f"[regimes] {len(regimes)} days classified: "
          f"green={sum(1 for v in regimes.values() if v=='green')}, "
          f"red={sum(1 for v in regimes.values() if v=='red')}, "
          f"flat={sum(1 for v in regimes.values() if v=='flat')}")

    # Family A
    print("[Family A] generating signals from v3.4.2 30s pred...")
    sys.stdout.flush()
    trades_A = family_A_signals()
    print(f"[Family A] {len(trades_A)} trade-rows generated (incl. both predicted & inverted)")

    # Family B
    print("[Family B] generating classical minute-bar signals...")
    sys.stdout.flush()
    trades_B = family_B_signals()
    print(f"[Family B] {len(trades_B)} trade-rows generated")

    # Family C
    print("[Family C] generating day-classifier driven swing...")
    sys.stdout.flush()
    trades_C = family_C_signals()
    print(f"[Family C] {len(trades_C)} trade-rows generated")

    all_trades = pd.concat([trades_A, trades_B, trades_C], ignore_index=True)
    print(f"[combined] {len(all_trades)} trades total")

    # Aggregate cells (ES market RT cost = headline)
    print("[aggregate] computing cell metrics for ES market RT cost...")
    cells_market = aggregate_cells(all_trades, regimes, cost_ticks=ES_MARKET_RT_TICKS)
    cells_market["cost_label"] = "ES_market_RT"

    # Also compute ES passive RT as informational only
    print("[aggregate] computing informational ES passive RT cells (50% adverse-selection discount)...")
    # For passive: apply discount to gross before subtracting commission
    if len(all_trades) > 0:
        trades_passive = all_trades.copy()
        trades_passive["gross_ticks"] = trades_passive["gross_ticks"] * PASSIVE_FILL_DISCOUNT
        cells_passive = aggregate_cells(trades_passive, regimes, cost_ticks=ES_PASSIVE_RT_TICKS)
        cells_passive["cost_label"] = "ES_passive_RT_50pct_discount"
    else:
        cells_passive = pd.DataFrame()

    cells = pd.concat([cells_market, cells_passive], ignore_index=True)

    # Sort by sharpe within cost label
    cells = cells.sort_values(["cost_label", "sharpe"], ascending=[True, False]).reset_index(drop=True)

    # Save
    cells_path = OUT_DIR / "results_table.csv"
    cells.to_csv(cells_path, index=False)
    print(f"[saved] {cells_path}  ({len(cells)} cells)")

    # Per-day breakdown for survivors + top cells
    per_day_rows = []
    headline_cells = cells[cells["cost_label"] == "ES_market_RT"]
    top_cells_for_breakdown = headline_cells.head(20)
    for _, c in top_cells_for_breakdown.iterrows():
        sub = all_trades[
            (all_trades["family"] == c["family"]) &
            (all_trades["signal"] == c["signal"]) &
            (all_trades["side_label"] == c["side_label"]) &
            (all_trades["horizon_m"] == c["horizon_m"]) &
            (all_trades["pct"] == c["pct"])
        ].copy()
        sub["net_ticks"] = sub["gross_ticks"] - ES_MARKET_RT_TICKS
        sub["pnl_usd"] = sub["net_ticks"] * ES_TICK_VALUE
        per_day = sub.groupby("date")["pnl_usd"].agg(["sum", "count"]).reset_index()
        for _, r in per_day.iterrows():
            per_day_rows.append({
                "family": c["family"], "signal": c["signal"],
                "side_label": c["side_label"], "horizon_m": c["horizon_m"], "pct": c["pct"],
                "date": r["date"], "regime": regimes.get(r["date"], "unknown"),
                "n_trades": int(r["count"]), "pnl_usd": float(r["sum"]),
            })
    per_day_df = pd.DataFrame(per_day_rows)
    per_day_path = OUT_DIR / "per_day_breakdown.csv"
    per_day_df.to_csv(per_day_path, index=False)
    print(f"[saved] {per_day_path}  ({len(per_day_df)} rows)")

    # Survivors
    survivors_market = headline_cells[headline_cells["passes_gates"]].copy()
    surv_path = OUT_DIR / "survivors.txt"
    with open(surv_path, "w") as f:
        f.write("# Cells passing ALL HC #428 R1 gates (ES market RT cost)\n")
        f.write(f"# Gates: Sharpe>{GATE_SHARPE} | PF>{GATE_PF} | WR_day>{GATE_WR} "
                f"| regime_gap<={GATE_REGIME_GAP} | day_conc<={GATE_DAY_CONC} "
                f"| n_days>={GATE_N_DAYS} | n_trades>={GATE_N_TRADES}\n\n")
        if len(survivors_market) == 0:
            f.write("NO SURVIVORS — no cell passes all gates at ES market exec.\n")
        else:
            f.write(survivors_market.to_string(index=False))
    print(f"[saved] {surv_path}  ({len(survivors_market)} survivors)")

    # Verdict
    write_verdict(cells, all_trades, regimes)

    if mlflow_active:
        try:
            mlflow.log_metric("n_trades_total", len(all_trades))
            mlflow.log_metric("n_cells_market", len(headline_cells))
            mlflow.log_metric("n_survivors_market", len(survivors_market))
            mlflow.log_artifact(str(cells_path))
            mlflow.log_artifact(str(surv_path))
            mlflow.log_artifact(str(OUT_DIR / "verdict.md"))
            mlflow.end_run()
        except Exception as e:
            print(f"[mlflow] log failed: {e}")

    print(f"[done] elapsed {time.time()-t0:.1f}s")


def write_verdict(cells: pd.DataFrame, all_trades: pd.DataFrame, regimes: dict[str, str]):
    headline = cells[cells["cost_label"] == "ES_market_RT"]
    informational = cells[cells["cost_label"] == "ES_passive_RT_50pct_discount"]
    top5 = headline.sort_values("sharpe", ascending=False).head(5)
    survivors = headline[headline["passes_gates"]]

    n_per_family = all_trades.groupby("family")["date"].agg(["count", "nunique"])

    lines = []
    lines.append("# longer_horizon_v1 — VERDICT\n")
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    lines.append("\n## (a) WHAT RAN — N trades / N days per family\n\n")
    lines.append("| family | n_trade_rows | n_days |\n|---|---|---|\n")
    for fam, row in n_per_family.iterrows():
        lines.append(f"| {fam} | {row['count']} | {row['nunique']} |\n")

    lines.append("\n## (b) TOP 5 CELLS by per-day Sharpe (ES market RT cost — HEADLINE)\n\n")
    if len(top5) == 0:
        lines.append("(no cells qualified)\n")
    else:
        cols = ["family","signal","side_label","horizon_m","pct","n_trades","n_days",
                "trades_per_day","mean_daily_usd","sharpe","sortino","pf","wr_day",
                "day_conc","regime_gap","passes_gates"]
        lines.append(top5[cols].to_string(index=False))
        lines.append("\n")

    lines.append("\n## (c) CELLS PASSING ALL GATES (per family separately)\n\n")
    if len(survivors) == 0:
        lines.append("NONE — no cell passes the HC #428 R1 gates at ES market RT execution cost in this window.\n")
    else:
        for fam in sorted(survivors["family"].unique()):
            fam_surv = survivors[survivors["family"] == fam]
            lines.append(f"### Family {fam}: {len(fam_surv)} survivor(s)\n\n")
            cols = ["signal","side_label","horizon_m","pct","n_trades","n_days",
                    "trades_per_day","mean_daily_usd","sharpe","sortino","pf","wr_day",
                    "day_conc","regime_gap"]
            lines.append(fam_surv[cols].to_string(index=False))
            lines.append("\n\n")

    lines.append("\n## (d) VERDICT — does longer-horizon execution work?\n\n")
    if len(survivors) > 0:
        lines.append(f"**YES — {len(survivors)} cell(s) pass all gates.** ")
        winners = survivors.sort_values("sharpe", ascending=False)
        best = winners.iloc[0]
        lines.append(f"Winning family: **{best['family']}** "
                     f"(signal={best['signal']}, side={best['side_label']}, "
                     f"horizon={best['horizon_m']}m, pct={best['pct']:.2f}, "
                     f"Sharpe={best['sharpe']:.2f}, $/day={best['mean_daily_usd']:.2f}).\n")
    else:
        lines.append("**NO — no cell passes the HC #428 R1 gates at retail-grade execution (ES market RT) in this window.**\n\n")
        # Find which came closest
        if len(headline) > 0:
            best_sharpe = headline.sort_values("sharpe", ascending=False).iloc[0]
            best_mean = headline.sort_values("mean_daily_usd", ascending=False).iloc[0]
            lines.append(f"Best Sharpe achieved: {best_sharpe['sharpe']:.2f} "
                         f"({best_sharpe['family']} / {best_sharpe['signal']} / "
                         f"horizon={best_sharpe['horizon_m']}m / side={best_sharpe['side_label']} / "
                         f"pct={best_sharpe['pct']:.2f}; "
                         f"$/day={best_sharpe['mean_daily_usd']:.2f}, "
                         f"PF={best_sharpe['pf']:.2f}, WR_day={best_sharpe['wr_day']:.2f}).\n\n")
            lines.append(f"Best $/day: {best_mean['mean_daily_usd']:.2f} "
                         f"({best_mean['family']} / {best_mean['signal']} / "
                         f"horizon={best_mean['horizon_m']}m / side={best_mean['side_label']}).\n\n")
            # Diagnose closest family by family
            for fam in sorted(headline["family"].unique()):
                fam_cells = headline[headline["family"] == fam]
                if len(fam_cells) == 0: continue
                best_fam = fam_cells.sort_values("sharpe", ascending=False).iloc[0]
                blockers = []
                if best_fam["sharpe"] <= GATE_SHARPE: blockers.append(f"Sharpe={best_fam['sharpe']:.2f}≤{GATE_SHARPE}")
                if best_fam["pf"] <= GATE_PF: blockers.append(f"PF={best_fam['pf']:.2f}≤{GATE_PF}")
                if best_fam["wr_day"] <= GATE_WR: blockers.append(f"WR_day={best_fam['wr_day']:.2f}≤{GATE_WR}")
                if best_fam["regime_gap"] > GATE_REGIME_GAP: blockers.append(f"regime_gap={best_fam['regime_gap']:.2f}>{GATE_REGIME_GAP}")
                if best_fam["day_conc"] > GATE_DAY_CONC: blockers.append(f"day_conc={best_fam['day_conc']:.2f}>{GATE_DAY_CONC}")
                if best_fam["n_days"] < GATE_N_DAYS: blockers.append(f"n_days={best_fam['n_days']}<{GATE_N_DAYS}")
                if best_fam["n_trades"] < GATE_N_TRADES: blockers.append(f"n_trades={best_fam['n_trades']}<{GATE_N_TRADES}")
                lines.append(f"- **Family {fam}** closest cell ({best_fam['signal']}/h={best_fam['horizon_m']}m/{best_fam['side_label']}): "
                             f"blocked by {', '.join(blockers) if blockers else 'NOTHING (PASSES?)'}.\n")
        lines.append("\n## (e/f) RECOMMENDED CONFIG / NEXT LANE\n\n")
        lines.append("Since nothing survives at retail execution costs in this 32-day Feb-Apr 2026 window, "
                     "the microstructure-research lane should shift to:\n\n")
        lines.append("1. **DIFFERENT SAMPLE WINDOW** — Try 2025 H2 (Aug-Dec) or 2024 high-vol regimes "
                     "where signal-to-noise may be materially different.\n")
        lines.append("2. **DIFFERENT ASSET CLASS** — ES at retail cost (~1 tick spread + $4.70 RT comm) "
                     "leaves only ~0.6 ticks/trade for net edge AFTER costs; even a 70%-WR/1.5-tick-edge "
                     "system clears just ~0.2 net ticks. Consider: NQ (lower relative cost % of move), "
                     "CL (wider ranges per session), or ZN/ZB (lower volatility but tight book).\n")
        lines.append("3. **PROFESSIONAL EXECUTION** — passive queue with rebates and adverse-selection-aware "
                     "ordering could shift the calculus, but only if a serious queue simulator is built.\n")

    lines.append("\n## (informational) PASSIVE EXEC TOP 5 — for context (NOT used for headline gates)\n\n")
    if len(informational) == 0:
        lines.append("(passive cells unavailable)\n")
    else:
        top5_p = informational.sort_values("sharpe", ascending=False).head(5)
        cols = ["family","signal","side_label","horizon_m","pct","n_trades","n_days",
                "mean_daily_usd","sharpe","pf","wr_day","day_conc","regime_gap","passes_gates"]
        lines.append(top5_p[cols].to_string(index=False))
        lines.append("\n")
        lines.append("\nNote: passive numbers apply a 50% gross-PnL discount to account for adverse selection "
                     "on filled-only-when-favorable; this is a coarse proxy, not a rigorous queue simulation.\n")

    with open(OUT_DIR / "verdict.md", "w") as f:
        f.writelines(lines)
    print(f"[saved] {OUT_DIR/'verdict.md'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
