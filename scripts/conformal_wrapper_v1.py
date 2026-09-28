#!/usr/bin/env python3
"""
Split-Conformal Wrapper v1 (HC #488 creativity-mandate, model axis #2)
======================================================================
Wraps v3.4.2 CNN-Mamba OOT point predictions in split-conformal prediction
intervals to test whether interval WIDTH (residual-calibrated reliability)
extracts edge that raw point-confidence missed.

Method (split-conformal regression):
  1. For each horizon h in {1s, 5s, 10s, 30s}:
       - CALIBRATION = first half of OOT days (chronological).
       - TEST        = second half of OOT days.
  2. On CALIBRATION compute residuals r_i = y_i - yhat_i  (target_log_ret_h - pred_log_ret_h).
     Take alpha=0.1 quantile of |r_i|  -> conformal radius q_alpha.
  3. On TEST every event gets PI = [yhat - q_alpha, yhat + q_alpha].
     Constant-width split-conformal width = 2*q_alpha for every test event.
  4. To get ADAPTIVE width, fit a small regressor on CALIBRATION that predicts
     |residual| from features (here we use |yhat|, |yhat_other_horizons|,
     pred_p_up_*, and abs of quantile spread q90-q10 where available).
     Use predicted |residual| as the adaptive width estimate on TEST.
  5. Sort TEST events by adaptive-width decile (deciles per horizon side).
     "high-quality" bucket = bottom 10% adaptive widths.
  6. For each (horizon x side x adaptive-width-decile) bucket, run passive-limit
     P&L over the horizon. Side: long if pred>0, short if pred<0.
     Realized tick move at horizon h = target_log_ret_h (already in ticks per HC #486).
     Net = side * target_h - 0.376  (passive commission).

Inputs:
  /home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_YYYYMMDD.npz
  (34 OOT files available; SKIP 20260308, 20260315 per task spec.)

Outputs:
  /home/jupiter/Lvl3Quant/output/conformal_wrapper_v1/
    conformal_intervals.parquet   -- per-event: date, horizon, pred, lower, upper,
                                     width_const, width_adaptive, side, y, net_ticks
    bucket_summary.csv            -- (horizon x side x adaptive-width-decile):
                                     n_trades, mean_net_ticks, sharpe, wr, profitable_days
    winning_cells.txt             -- cells passing HC #428 R1 gates
    REPORT.md                     -- verdict + interpretation
    .regen_complete.json          -- HC #485 R5 stamp
"""
import json, os, sys, time, glob, traceback
from pathlib import Path
import numpy as np
import pandas as pd

OOT_DIR  = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate")
OUT_DIR  = Path("/home/jupiter/Lvl3Quant/output/conformal_wrapper_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cost constants per CLAUDE.md ES cost table
ES_RT_COMMISSION_TICKS = 0.376  # passive limit
ALPHA = 0.10                    # conformal miscoverage target (90% PI)

HORIZONS = ["1s", "5s", "10s", "30s"]
SKIP_DATES = {"20260308", "20260315"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_day(path):
    """Return DataFrame for one OOT day or None if missing required heads."""
    z = np.load(path, allow_pickle=True)
    needed = ["sample_dates"] + \
             [f"pred_log_ret_{h}" for h in HORIZONS] + \
             [f"target_log_ret_{h}" for h in HORIZONS] + \
             [f"mask_log_ret_{h}" for h in HORIZONS]
    if any(k not in z.files for k in needed):
        return None
    n = z["pred_log_ret_5s"].shape[0]
    if n == 0:
        return None
    out = {"date": np.asarray(z["sample_dates"]).astype(str)}
    for h in HORIZONS:
        out[f"pred_{h}"]   = z[f"pred_log_ret_{h}"].astype(np.float32)
        out[f"target_{h}"] = z[f"target_log_ret_{h}"].astype(np.float32)
        out[f"mask_{h}"]   = z[f"mask_log_ret_{h}"].astype(np.float32)
    # Extra features for adaptive width
    for q in ["pred_p_up_5s", "pred_p_up_10s", "pred_p_up_30s"]:
        if q in z.files:
            out[q] = z[q].astype(np.float32)
    # Quantile spread (only 10s/30s/60s present)
    for h in ["10s", "30s"]:
        q10 = f"pred_log_ret_{h}_q10"
        q90 = f"pred_log_ret_{h}_q90"
        if q10 in z.files and q90 in z.files:
            out[f"qspread_{h}"] = (z[q90] - z[q10]).astype(np.float32)
    df = pd.DataFrame(out)
    return df


def load_all():
    files = sorted(OOT_DIR.glob("oot_*.npz"))
    dfs = []
    for f in files:
        date = f.stem.replace("oot_", "")
        if date in SKIP_DATES:
            continue
        d = load_day(f)
        if d is None or len(d) == 0:
            print(f"  [skip] {f.name}: empty/missing heads", flush=True)
            continue
        dfs.append(d)
        print(f"  [load] {f.name}: {len(d):,} rows", flush=True)
    if not dfs:
        raise RuntimeError("No OOT data loaded.")
    df = pd.concat(dfs, ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Conformal core
# ---------------------------------------------------------------------------

def split_calibration_test(df, n_cal_days):
    """Chronological causal split: first n_cal_days dates -> CAL, rest -> TEST."""
    dates_sorted = sorted(df["date"].unique())
    cal_dates  = set(dates_sorted[:n_cal_days])
    test_dates = set(dates_sorted[n_cal_days:])
    cal_mask  = df["date"].isin(cal_dates).values
    test_mask = df["date"].isin(test_dates).values
    return cal_mask, test_mask, sorted(cal_dates), sorted(test_dates)


def conformal_radius(residuals, alpha=ALPHA):
    """Standard split-conformal radius.

    For miscoverage alpha, q = ceil((n+1)(1-alpha))/n quantile of |residuals|.
    Returns the conformal radius q_alpha.
    """
    r = np.abs(residuals)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan
    n = len(r)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    q = np.partition(r, k - 1)[k - 1]
    return float(q)


def fit_adaptive_width(X_cal, abs_resid_cal):
    """Tiny ridge regressor: predict |residual| from features.

    Uses closed-form ridge with light L2. Returns (w, b, x_mean, x_std).
    Anything non-finite is clamped.
    """
    X = np.asarray(X_cal, dtype=np.float64)
    y = np.asarray(abs_resid_cal, dtype=np.float64)
    finite = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[finite]; y = y[finite]
    if len(y) < 100:
        return None
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Xs = (X - mu) / sd
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    lam = 1.0
    A = Xs.T @ Xs + lam * np.eye(Xs.shape[1])
    b = Xs.T @ y
    coef = np.linalg.solve(A, b)
    return {"coef": coef, "mu": mu, "sd": sd}


def apply_adaptive_width(model, X_test):
    if model is None:
        return None
    X = np.asarray(X_test, dtype=np.float64)
    finite = np.isfinite(X).all(axis=1)
    Xs = np.zeros_like(X)
    Xs[finite] = (X[finite] - model["mu"]) / model["sd"]
    Xs = np.hstack([Xs, np.ones((len(Xs), 1))])
    pred = Xs @ model["coef"]
    # Predictions should be non-negative (it's |residual|); clip and fill non-finite.
    pred = np.where(np.isfinite(pred), pred, np.nan)
    pred = np.clip(pred, 0.0, None)
    return pred


# ---------------------------------------------------------------------------
# Bucket metrics
# ---------------------------------------------------------------------------

def per_bucket_stats(net_ticks, dates):
    n = len(net_ticks)
    if n == 0:
        return dict(n=0, mean_net_ticks=np.nan, sharpe=np.nan, wr=np.nan,
                    profitable_days=0, total_days=0, day_conc=np.nan)
    mu = net_ticks.mean()
    sd = net_ticks.std()
    sharpe = (mu / sd * np.sqrt(n)) if sd > 1e-9 else 0.0
    wr = float((net_ticks > 0).mean())
    # Day-level aggregation
    df = pd.DataFrame({"net": net_ticks, "date": dates})
    day_sums = df.groupby("date")["net"].sum()
    total_days = int((day_sums.shape[0]))
    profitable_days = int((day_sums > 0).sum())
    abs_sum = float(day_sums.abs().sum())
    day_conc = float(day_sums.abs().max() / abs_sum) if abs_sum > 1e-9 else np.nan
    return dict(
        n=int(n),
        mean_net_ticks=float(mu),
        sharpe=float(sharpe),
        wr=wr,
        profitable_days=profitable_days,
        total_days=total_days,
        day_conc=day_conc,
    )


def regime_split_check(net_ticks, dates):
    """Classify days as green/red by sum of NET ticks (proxy regime since we
    don't have ES close here). Returns Sharpe gap per HC #428 R1.

    Note: True regime would be ES close-to-close. We approximate by sign of the
    day's net trade sum which is correlated with intraday direction. This is
    a conservative proxy and is documented in the report.
    """
    if len(net_ticks) == 0:
        return dict(sh_green=np.nan, sh_red=np.nan, gap=np.nan)
    df = pd.DataFrame({"net": net_ticks, "date": dates})
    day_sums = df.groupby("date")["net"].sum()
    green_days = set(day_sums[day_sums > 0].index)
    red_days   = set(day_sums[day_sums < 0].index)
    g = df[df["date"].isin(green_days)]["net"].values
    r = df[df["date"].isin(red_days)]["net"].values

    def _sh(x):
        if len(x) == 0 or x.std() <= 1e-9:
            return np.nan
        return float(x.mean() / x.std() * np.sqrt(len(x)))

    sh_g = _sh(g); sh_r = _sh(r)
    if np.isfinite(sh_g) and np.isfinite(sh_r):
        denom = max(abs(sh_g), abs(sh_r), 1e-9)
        gap = abs(sh_g - sh_r) / denom
    else:
        gap = np.nan
    return dict(sh_green=sh_g, sh_red=sh_r, gap=gap)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    print("[1/5] Loading OOT predictions ...", flush=True)
    df = load_all()
    dates_sorted = sorted(df["date"].unique())
    print(f"  total events: {len(df):,}  unique OOT days: {len(dates_sorted)}", flush=True)

    n_cal_days = len(dates_sorted) // 2
    print(f"  calibration days: {n_cal_days}   test days: {len(dates_sorted) - n_cal_days}", flush=True)

    cal_mask, test_mask, cal_dates, test_dates = split_calibration_test(df, n_cal_days)

    all_event_rows = []
    bucket_rows = []

    for h in HORIZONS:
        print(f"\n[2/5] Horizon {h}: conformal calibration", flush=True)
        pred = df[f"pred_{h}"].values
        targ = df[f"target_{h}"].values
        mask = df[f"mask_{h}"].values

        valid = (mask > 0.5) & np.isfinite(pred) & np.isfinite(targ)
        cal_idx = valid & cal_mask
        test_idx = valid & test_mask

        if cal_idx.sum() < 1000 or test_idx.sum() < 1000:
            print(f"  [skip] insufficient data: cal={cal_idx.sum()} test={test_idx.sum()}", flush=True)
            continue

        residuals_cal = targ[cal_idx] - pred[cal_idx]
        q_alpha = conformal_radius(residuals_cal, alpha=ALPHA)
        print(f"  q_alpha (90% PI half-width, ticks) = {q_alpha:.4f}  "
              f"n_cal={cal_idx.sum():,}  n_test={test_idx.sum():,}", flush=True)

        # Adaptive width regressor on CALIBRATION
        feat_cols = [f"pred_{h2}" for h2 in HORIZONS]
        for extra in ["pred_p_up_5s", "pred_p_up_10s", "pred_p_up_30s",
                      "qspread_10s", "qspread_30s"]:
            if extra in df.columns:
                feat_cols.append(extra)
        # Use abs(pred) features instead of signed (residual magnitude is symmetric).
        Xc = np.column_stack([np.abs(df[c].values[cal_idx]) for c in feat_cols])
        ac = np.abs(residuals_cal)
        adapt_model = fit_adaptive_width(Xc, ac)

        Xt = np.column_stack([np.abs(df[c].values[test_idx]) for c in feat_cols])
        width_adaptive_test = apply_adaptive_width(adapt_model, Xt)
        if width_adaptive_test is None:
            width_adaptive_test = np.full(test_idx.sum(), np.nan)

        # Test set basic vectors
        pred_t = pred[test_idx]
        targ_t = targ[test_idx]
        date_t = df["date"].values[test_idx]
        side_t = np.where(pred_t > 0, 1, np.where(pred_t < 0, -1, 0))
        lower_t = pred_t - q_alpha
        upper_t = pred_t + q_alpha

        # P&L (passive limit): side * realized - commission. Skip side==0.
        net_t = side_t * targ_t - ES_RT_COMMISSION_TICKS
        # We'll only count trades where side != 0.

        # Per-event log
        ev_df = pd.DataFrame({
            "date": date_t,
            "horizon": h,
            "pred": pred_t,
            "lower": lower_t,
            "upper": upper_t,
            "width_const": 2.0 * q_alpha,
            "width_adaptive": width_adaptive_test,
            "side": side_t,
            "y": targ_t,
            "net_ticks": net_t,
        })
        all_event_rows.append(ev_df)

        # Bucket by adaptive-width DECILE per (horizon, side)
        print(f"[3/5] Horizon {h}: bucketing by adaptive-width deciles", flush=True)
        for side_val, side_name in [(1, "long"), (-1, "short")]:
            sel = (side_t == side_val) & np.isfinite(width_adaptive_test)
            if sel.sum() < 200:
                continue
            w = width_adaptive_test[sel]
            net = net_t[sel]
            dts = date_t[sel]
            # Deciles (1 = lowest width = highest confidence-of-quality)
            ranks = np.argsort(np.argsort(w))
            deciles = (ranks * 10 // len(w)) + 1
            deciles = np.clip(deciles, 1, 10)
            for d in range(1, 11):
                m = (deciles == d)
                if m.sum() == 0:
                    continue
                stats = per_bucket_stats(net[m], dts[m])
                regime = regime_split_check(net[m], dts[m])
                bucket_rows.append({
                    "horizon": h,
                    "side": side_name,
                    "width_decile": d,
                    "q_alpha_ticks": q_alpha,
                    **stats,
                    "sh_green": regime["sh_green"],
                    "sh_red":   regime["sh_red"],
                    "regime_gap": regime["gap"],
                })

    # ---------------------------------------------------------------
    # Persist outputs
    # ---------------------------------------------------------------
    print("\n[4/5] Writing outputs ...", flush=True)
    if all_event_rows:
        ev_all = pd.concat(all_event_rows, ignore_index=True)
        ev_path = OUT_DIR / "conformal_intervals.parquet"
        try:
            ev_all.to_parquet(ev_path, index=False)
        except Exception as e:
            print(f"  parquet write failed ({e}); falling back to CSV", flush=True)
            ev_all.to_csv(OUT_DIR / "conformal_intervals.csv", index=False)
        print(f"  wrote {len(ev_all):,} event rows", flush=True)
    else:
        ev_all = pd.DataFrame()

    buckets = pd.DataFrame(bucket_rows)
    buckets.to_csv(OUT_DIR / "bucket_summary.csv", index=False)
    print(f"  wrote {len(buckets)} bucket rows", flush=True)

    # ---------------------------------------------------------------
    # Winning cells per HC #428 R1
    #   - sharpe > 0.5 (per-trade annualized-ish proxy)
    #   - wr > 0.50
    #   - mean_net_ticks > 0
    #   - profitable_days / total_days >= 0.55
    #   - day_conc <= 0.70 (HC #344)
    #   - regime_gap <= 0.50 OR one regime missing (small sample)
    # ---------------------------------------------------------------
    if len(buckets):
        b = buckets.copy()
        b["frac_prof_days"] = b["profitable_days"] / b["total_days"].clip(lower=1)
        winners = b[
            (b["sharpe"] > 0.5)
            & (b["wr"] > 0.50)
            & (b["mean_net_ticks"] > 0.0)
            & (b["frac_prof_days"] >= 0.55)
            & (b["day_conc"].fillna(0.0) <= 0.70)
            & ((b["regime_gap"].fillna(0.0) <= 0.50))
            & (b["n"] >= 100)
        ].copy()
        winners = winners.sort_values("sharpe", ascending=False)
    else:
        winners = pd.DataFrame()

    with open(OUT_DIR / "winning_cells.txt", "w") as fp:
        fp.write(f"# Winning cells (HC #428 R1 gates passed)\n")
        fp.write(f"# Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if len(winners) == 0:
            fp.write("NONE\n")
        else:
            cols = ["horizon", "side", "width_decile", "n", "mean_net_ticks",
                    "sharpe", "wr", "frac_prof_days", "day_conc", "regime_gap"]
            fp.write(winners[cols].to_string(index=False))
            fp.write("\n")

    # ---------------------------------------------------------------
    # REPORT.md
    # ---------------------------------------------------------------
    accept = len(winners) > 0
    best_line = ""
    if accept:
        top = winners.iloc[0]
        best_line = (f"BEST: h={top['horizon']} {top['side']} decile={int(top['width_decile'])}  "
                     f"n={int(top['n'])}  net={top['mean_net_ticks']:.4f} t/trade  "
                     f"Sharpe={top['sharpe']:.3f}  WR={top['wr']*100:.1f}%  "
                     f"prof_days={int(top['profitable_days'])}/{int(top['total_days'])}  "
                     f"day_conc={top['day_conc']:.2f}  regime_gap={top['regime_gap']:.2f}")

    # Adaptive-width informativeness: does decile 1 beat decile 10?
    width_signal = []
    if len(buckets):
        for (h, s), g in buckets.groupby(["horizon", "side"]):
            d1 = g[g["width_decile"] == 1]
            d10 = g[g["width_decile"] == 10]
            if len(d1) == 1 and len(d10) == 1:
                width_signal.append({
                    "horizon": h, "side": s,
                    "d1_sharpe": float(d1["sharpe"].iloc[0]),
                    "d10_sharpe": float(d10["sharpe"].iloc[0]),
                    "d1_net":    float(d1["mean_net_ticks"].iloc[0]),
                    "d10_net":   float(d10["mean_net_ticks"].iloc[0]),
                })
    ws_df = pd.DataFrame(width_signal)
    width_informative = False
    if len(ws_df):
        # Average d1 sharpe - d10 sharpe
        delta = (ws_df["d1_sharpe"] - ws_df["d10_sharpe"]).mean()
        width_informative = bool(delta > 0.10 and (ws_df["d1_sharpe"] > ws_df["d10_sharpe"]).mean() >= 0.6)

    with open(OUT_DIR / "REPORT.md", "w") as fp:
        fp.write("# Conformal Wrapper v1 - REPORT\n\n")
        fp.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        fp.write(f"- OOT days loaded: {len(dates_sorted)}  (cal={n_cal_days}, test={len(dates_sorted)-n_cal_days})\n")
        fp.write(f"- Total events: {len(df):,}\n")
        fp.write(f"- Horizons: {HORIZONS}\n")
        fp.write(f"- alpha (miscoverage): {ALPHA}\n\n")
        fp.write("## Verdict\n\n")
        fp.write(f"**{'ACCEPT' if accept else 'REJECT'}**\n\n")
        if best_line:
            fp.write(best_line + "\n\n")
        fp.write("## Width vs Edge (decile 1 vs decile 10)\n\n")
        if len(ws_df):
            fp.write(ws_df.to_markdown(index=False))
            fp.write("\n\n")
            fp.write(f"Mean delta Sharpe(d1 - d10) = "
                     f"{(ws_df['d1_sharpe'] - ws_df['d10_sharpe']).mean():.3f}\n\n")
            fp.write(f"Width is informative: **{width_informative}**\n\n")
        else:
            fp.write("Insufficient data for width-vs-edge comparison.\n\n")
        fp.write("## Interpretation\n\n")
        fp.write(
            "Split-conformal calibrates uncertainty against realized residuals on the CAL split, "
            "then the trained ridge predicts the residual magnitude per event on the TEST split. "
            "The bottom decile (decile 1) of predicted residual magnitude is the most 'reliable' "
            "set of predictions. If this bucket is meaningfully more profitable than the top decile, "
            "the conformal-width feature carries information that raw point-confidence missed. "
            "If it is not, then the residual is largely irreducible at this feature set and the "
            "7-test rejection of v3.4.2 stands.\n\n"
        )
        fp.write("## Methodology Caveats\n\n")
        fp.write(
            "- Regime gap uses day NET-sum sign as a proxy (no ES close-to-close pulled here). "
            "This is conservative since the day's own trades influence the classification.\n"
            "- Targets in npz are documented as ticks per HC #486; P&L net = side*target_h - 0.376.\n"
            "- Causal split: calibration days strictly precede test days.\n"
            "- Adaptive width uses |pred| heads + p_up + quantile-spreads; a richer feature "
            "set could change conclusions.\n"
        )

    # HC #485 R5 regen stamp
    stamp = {
        "script": "scripts/conformal_wrapper_v1.py",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_seconds": round(time.time() - t0, 1),
        "n_events": int(len(df)),
        "n_oot_days": int(len(dates_sorted)),
        "cal_days": n_cal_days,
        "test_days": len(dates_sorted) - n_cal_days,
        "horizons": HORIZONS,
        "alpha": ALPHA,
        "winners": int(len(winners)),
        "verdict": "ACCEPT" if accept else "REJECT",
        "width_informative": bool(width_informative),
    }
    with open(OUT_DIR / ".regen_complete.json", "w") as fp:
        json.dump(stamp, fp, indent=2)

    print(f"\n[5/5] Done in {time.time()-t0:.1f}s. Verdict: {'ACCEPT' if accept else 'REJECT'}.", flush=True)
    print(f"      winners: {len(winners)}   width_informative: {width_informative}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
