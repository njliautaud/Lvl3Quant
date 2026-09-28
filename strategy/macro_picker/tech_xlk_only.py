"""
Tech (A) — XLK-only momentum + SPY regime-gated long-only.

HC #581 R2(a). Replicates the leader's shape (hold21, longonly, txn 5bps, SPY-MA
bull-only regime filter, sliding 24m train / 6m OOT / 3m step) but the asset
universe is reduced to {XLK} only.

Trade rule per rebalance date:
  - If SPY > MA(60d) → momentum score = blended ret_20d + ret_60d + rel_strength_spy
                       on XLK. If score > 0, enter XLK long; else stay flat.
  - If SPY ≤ MA(60d) → stay flat for the cycle.
  - Vol-target sizing on XLK's prior-60d realised vol → target 15% annual.
  - 5bps round-trip txn cost on every rebalance that changes position.

WF: sliding 24m / 6m / 3m, same as the leader (HC #0 — NEVER expanding).

Outputs to /home/jupiter/Lvl3Quant/output/macro_picker/tech_xlk_only_<TS>/:
  book.parquet, report.json, rebal_picks.parquet, regime_stratification.csv
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
from walk_forward import _metrics  # type: ignore

ETF_FLOWS_PATH = ROOT / "data/feature_store/sector_etf_flows/daily.parquet"
SPY_PRICE_PATH = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"

TRADING_DAYS = 252
TXN_COST_BPS = 5.0
TARGET_VOL = 0.15
LEV_MIN, LEV_MAX = 0.25, 2.0
HOLD_DAYS = 21
REGIME_MA_DAYS = 60

# Leader WF range
WF_START = pd.Timestamp("2021-06-07")
WF_END = pd.Timestamp("2026-02-27")  # matches leader date_range upper bound
TRAIN_MONTHS, OOT_MONTHS, STEP_MONTHS = 24, 6, 3


def _load_xlk() -> pd.DataFrame:
    df = pd.read_parquet(ETF_FLOWS_PATH)
    df["date"] = pd.to_datetime(df["date"])
    xlk = df[df["etf"] == "XLK"].sort_values("date").reset_index(drop=True)
    xlk = xlk[(xlk["date"] >= WF_START) & (xlk["date"] <= WF_END)].copy()
    return xlk


def _load_spy_regime(ma_days: int = REGIME_MA_DAYS) -> pd.Series:
    px = pd.read_parquet(SPY_PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    ma = spy.rolling(ma_days, min_periods=max(20, ma_days // 2)).mean()
    return pd.Series(np.where(spy > ma, "bull", "bear"), index=spy.index, name="reg").dropna()


def _load_spy_daily_ret() -> pd.Series:
    px = pd.read_parquet(SPY_PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    return spy.pct_change().dropna()


def _estimate_xlk_vol(xlk: pd.DataFrame, rd: pd.Timestamp, lookback: int = 60) -> float:
    cutoff = rd - pd.Timedelta(days=lookback * 2 + 10)
    hist = xlk[(xlk["date"] < rd) & (xlk["date"] >= cutoff)]
    hist = hist["ret_1d"].dropna().tail(lookback)
    if len(hist) < 20:
        return 0.0
    sd = float(hist.std(ddof=1))
    if not np.isfinite(sd):
        return 0.0
    return sd * np.sqrt(TRADING_DAYS)


def _iter_wf(start, end, train_m=TRAIN_MONTHS, oot_m=OOT_MONTHS, step_m=STEP_MONTHS):
    out, cursor = [], start
    while True:
        tr_s = cursor
        tr_e = tr_s + pd.DateOffset(months=train_m)
        os_ = tr_e
        oe = os_ + pd.DateOffset(months=oot_m)
        if oe > end + pd.Timedelta(days=1):
            break
        out.append((tr_s, tr_e, os_, oe))
        cursor = cursor + pd.DateOffset(months=step_m)
    return out


def _momentum_score(row: pd.Series, blend: dict) -> float:
    """Sum of z-style blend; here we use the leader's three momentum features
    directly (already same scale ≈ pct returns / rel strength). Train-side
    quantile thresholds are learned per fold."""
    s = 0.0
    for f, w in blend.items():
        v = row.get(f)
        if v is None or not np.isfinite(v):
            continue
        s += w * float(v)
    return s


def _learn_threshold(train_df: pd.DataFrame, hold_days: int) -> tuple[float, dict]:
    """On train window, learn the score threshold that maximises forward-N
    return given the long-only-XLK choice. We use a simple equal-weight blend
    of {ret_20d, ret_60d, rel_strength_spy} and learn a quantile threshold."""
    blend = {"ret_20d": 1.0, "ret_60d": 1.0, "rel_strength_spy": 1.0}
    train_df = train_df.copy()
    train_df["close"] = train_df["close"].astype(float)
    train_df["y_fwd"] = train_df["close"].shift(-hold_days) / train_df["close"] - 1.0
    train_df["score"] = train_df.apply(lambda r: _momentum_score(r, blend), axis=1)
    sub = train_df.dropna(subset=["score", "y_fwd"])
    if sub.empty:
        return 0.0, blend
    # Pick threshold = train-median score (matches leader's "above median" no-trade floor)
    thr = float(sub["score"].median())
    return thr, blend


def run_wf(xlk: pd.DataFrame, regime: pd.Series) -> tuple[pd.DataFrame, list]:
    """Run sliding WF; return per-day book DataFrame + list of rebal picks."""
    windows = _iter_wf(xlk["date"].min(), xlk["date"].max())
    daily_rows = []
    rebal_rows = []

    for (tr_s, tr_e, os_, oe) in windows:
        train = xlk[(xlk["date"] >= tr_s) & (xlk["date"] < tr_e)].copy()
        oot = xlk[(xlk["date"] >= os_) & (xlk["date"] < oe)].copy()
        if len(train) < 100 or len(oot) < 20:
            continue
        thr, blend = _learn_threshold(train, HOLD_DAYS)

        # rebal dates inside OOT every HOLD_DAYS
        oot_dates = sorted(oot["date"].unique())
        rebal_dates = oot_dates[::HOLD_DAYS]

        # Pre-compute score on OOT
        oot["score"] = oot.apply(lambda r: _momentum_score(r, blend), axis=1)

        prev_position = 0.0  # 0 or +1 (long)
        for rd in rebal_dates:
            # Regime gate: bull-only
            reg = regime.get(pd.Timestamp(rd))
            if reg is None:
                prior = regime.loc[:pd.Timestamp(rd)]
                reg = prior.iloc[-1] if len(prior) else "bear"

            snap = oot[oot["date"] == rd]
            if snap.empty:
                continue
            score = float(snap["score"].iloc[0]) if pd.notna(snap["score"].iloc[0]) else float("-inf")

            position = 1.0 if (reg == "bull" and score > thr) else 0.0

            # Vol-target sizing on XLK
            xv = _estimate_xlk_vol(xlk, pd.Timestamp(rd))
            if xv <= 1e-6 or position == 0:
                gross_lev = 1.0 if position > 0 else 0.0
            else:
                gross_lev = float(np.clip(TARGET_VOL / xv, LEV_MIN, LEV_MAX))

            rebal_rows.append({
                "rebal_date": pd.Timestamp(rd),
                "regime": reg,
                "score": score,
                "threshold": thr,
                "position": position,
                "gross_lev": gross_lev,
                "fold_oot_start": str(os_.date()),
            })

            # Hold window: daily returns over next HOLD_DAYS
            hold_win = oot[(oot["date"] > rd) & (oot["date"] <= rd + pd.Timedelta(days=int(HOLD_DAYS * 1.6)))]
            hold_win = hold_win.head(HOLD_DAYS)
            for _, d_row in hold_win.iterrows():
                d = d_row["date"]
                # Intra-hold regime check
                rg = regime.get(pd.Timestamp(d))
                if rg is None:
                    prior = regime.loc[:pd.Timestamp(d)]
                    rg = prior.iloc[-1] if len(prior) else "bear"
                ret_1d = float(d_row["ret_1d"]) if pd.notna(d_row["ret_1d"]) else 0.0
                if position == 0 or rg != "bull":
                    day_ret = 0.0
                else:
                    day_ret = gross_lev * ret_1d
                    day_ret = float(np.clip(day_ret, -0.20, 0.20))
                daily_rows.append({"date": d, "ret": day_ret, "lev": gross_lev})

            # Txn cost on rebal day proportional to position change
            pos_change = abs(position - prev_position)
            if pos_change > 0:
                tc = (TXN_COST_BPS / 10000.0) * gross_lev * pos_change
                daily_rows.append({"date": pd.Timestamp(rd), "ret": -tc, "lev": gross_lev})
            prev_position = position

    if not daily_rows:
        return pd.DataFrame(columns=["date", "daily_ret", "gross_lev"]), rebal_rows

    df = pd.DataFrame(daily_rows)
    df["date"] = pd.to_datetime(df["date"])
    book = df.groupby("date", as_index=False).agg(daily_ret=("ret", "sum"), gross_lev=("lev", "max"))
    book = book.sort_values("date").reset_index(drop=True)
    return book, rebal_rows


def regime_stratification(book: pd.DataFrame, spy_ret: pd.Series, thresh: float = 0.001) -> pd.DataFrame:
    """Classify each trading day green/red/flat by SPY close-to-close return,
    then compute Sharpe stratified per regime."""
    b = book.set_index("date")["daily_ret"]
    spy = spy_ret.reindex(b.index).fillna(0.0)
    cls = pd.Series("flat", index=b.index, dtype=object)
    cls[spy > thresh] = "green"
    cls[spy < -thresh] = "red"

    rows = []
    for r in ["green", "red", "flat"]:
        sub = b[cls == r].dropna()
        if len(sub) < 5:
            rows.append({"regime": r, "n_days": int(len(sub)), "mean_ret": float("nan"),
                         "sharpe": float("nan")})
            continue
        sd = sub.std(ddof=1)
        sh = float(sub.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")
        rows.append({"regime": r, "n_days": int(len(sub)),
                     "mean_ret": float(sub.mean()), "sharpe": sh})
    return pd.DataFrame(rows)


def deploy_verdict(m: dict, strat_df: pd.DataFrame, book: pd.DataFrame) -> dict:
    """Apply HC #428 + #581 deploy gates."""
    calmar = m.get("calmar", float("nan"))
    sharpe = m.get("sharpe", float("nan"))
    n_days = int(book["daily_ret"].dropna().shape[0])
    # Regime balance check
    g = strat_df[strat_df["regime"] == "green"]["sharpe"].iloc[0] if not strat_df.empty else float("nan")
    r = strat_df[strat_df["regime"] == "red"]["sharpe"].iloc[0] if not strat_df.empty else float("nan")
    if np.isfinite(g) and np.isfinite(r) and max(abs(g), abs(r)) > 0:
        regime_imbalance = abs(g - r) / max(abs(g), abs(r))
        regime_ok = regime_imbalance <= 0.50
    else:
        regime_imbalance = float("nan")
        regime_ok = False
    # Day-conc
    abs_pnl = book["daily_ret"].abs()
    if abs_pnl.sum() > 0:
        day_conc = float(abs_pnl.max() / abs_pnl.sum())
    else:
        day_conc = float("nan")
    day_conc_ok = np.isfinite(day_conc) and day_conc <= 0.70
    oot_days_ok = n_days >= 40
    calmar_ok = np.isfinite(calmar) and calmar >= 1.5
    sharpe_ok = np.isfinite(sharpe) and sharpe >= 1.0
    PASS = bool(calmar_ok and sharpe_ok and regime_ok and day_conc_ok and oot_days_ok)
    return {
        "calmar_ge_1_5": bool(calmar_ok),
        "sharpe_ge_1_0": bool(sharpe_ok),
        "regime_imbalance": float(regime_imbalance) if np.isfinite(regime_imbalance) else None,
        "regime_balance_ok": bool(regime_ok),
        "day_concentration": float(day_conc) if np.isfinite(day_conc) else None,
        "day_concentration_ok": bool(day_conc_ok),
        "n_oot_days": n_days,
        "oot_days_ge_40": bool(oot_days_ok),
        "PASSES_DEPLOY_GATES": PASS,
    }


def main():
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / f"output/macro_picker/tech_xlk_only_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    xlk = _load_xlk()
    print(f"[tech_xlk] XLK rows {len(xlk)} {xlk['date'].min().date()} → {xlk['date'].max().date()}")
    regime = _load_spy_regime()
    spy_ret = _load_spy_daily_ret()

    book, rebal_rows = run_wf(xlk, regime)
    book.to_parquet(out_dir / "book.parquet", index=False)
    pd.DataFrame(rebal_rows).to_parquet(out_dir / "rebal_picks.parquet", index=False)

    s = book.set_index("date")["daily_ret"]
    m = _metrics(s) if not s.empty else {}
    strat_df = regime_stratification(book, spy_ret)
    strat_df.to_csv(out_dir / "regime_stratification.csv", index=False)

    gates = deploy_verdict(m, strat_df, book)

    report = {
        "strategy": "tech_xlk_only",
        "config": {
            "universe": ["XLK"],
            "hold_days": HOLD_DAYS,
            "longonly": True,
            "txn_cost_bps": TXN_COST_BPS,
            "target_vol": TARGET_VOL,
            "lev_clip": [LEV_MIN, LEV_MAX],
            "regime_filter": f"SPY > MA{REGIME_MA_DAYS}d (bull-only)",
            "wf_train_months": TRAIN_MONTHS,
            "wf_oot_months": OOT_MONTHS,
            "wf_step_months": STEP_MONTHS,
            "wf_window_type": "SLIDING (HC #0)",
        },
        "date_range": [str(book["date"].min().date()) if not book.empty else None,
                       str(book["date"].max().date()) if not book.empty else None],
        "headline": {
            "cagr_pct": (m.get("cagr") or 0) * 100,
            "sharpe": m.get("sharpe"),
            "sortino": m.get("sortino"),
            "calmar": m.get("calmar"),
            "max_dd_pct": (m.get("max_dd") or 0) * 100,
            "pf": m.get("pf"),
            "wr_pct": (m.get("wr") or 0) * 100,
            "n_oot_days": int(s.dropna().shape[0]) if not s.empty else 0,
        },
        "regime_stratification": strat_df.to_dict(orient="records"),
        "deploy_gates": gates,
        "n_rebalances": len(rebal_rows),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))

    print(f"[tech_xlk] wrote {out_dir}")
    print(f"[tech_xlk] CAGR={report['headline']['cagr_pct']:.1f}% "
          f"Sharpe={report['headline']['sharpe']:.2f} "
          f"Calmar={report['headline']['calmar']:.2f} "
          f"MaxDD={report['headline']['max_dd_pct']:.1f}% "
          f"PASS={gates['PASSES_DEPLOY_GATES']}")
    return out_dir, report


if __name__ == "__main__":
    main()
