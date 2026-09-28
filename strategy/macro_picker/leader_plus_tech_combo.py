"""
Blended portfolio sweep: LEADER (ETF sector rotation hold21 longonly regime-gated)
                       + TECH SUB-INDUSTRY ROTATION (top-2 momentum hold21 regime-gated)

For each w_tech in {0.00..1.00 step 0.10}:
    blended_daily = (1 - w_tech) * leader_ret + w_tech * tech_ret
Compute CAGR, Sharpe, Sortino, Calmar, MaxDD, PF, WR, day-concentration.

Picks:
  * Best Calmar  s.t. Sharpe >= 1.0 and MaxDD >= -15%
  * Best Sharpe  s.t. MaxDD >= -10%   (Sharpe-pref point)

Inputs are pre-computed book.parquet files (date, daily_ret, gross_lev).
Date alignment: INTERSECTION of the two date indexes.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
LEADER_BOOK = ROOT / "output/macro_picker/etf_rotation_regime_20260608_164726_hold21_longonly/book.parquet"
TECH_BOOK = ROOT / "output/macro_picker/tech_sub_industry_rotation_20260609_072759/book.parquet"

OUT_DIR = ROOT / f"output/macro_picker/leader_plus_tech_{time.strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = [round(0.10 * i, 2) for i in range(0, 11)]  # 0.00 .. 1.00


def metrics(daily: pd.Series, label: str = "") -> dict:
    """Standard risk-adjusted metrics + day-concentration."""
    daily = daily.dropna().sort_index()
    if daily.empty:
        return {"label": label, "n_days": 0}
    eq = (1.0 + daily).cumprod()
    mu = daily.mean()
    sd = daily.std()
    sharpe = (mu / sd) * np.sqrt(252) if sd > 0 else 0.0
    down = daily[daily < 0].std()
    sortino = (mu / down) * np.sqrt(252) if down and down > 0 else 0.0
    n_cal_days = max(1, (daily.index[-1] - daily.index[0]).days)
    cagr = float(eq.iloc[-1]) ** (365.25 / n_cal_days) - 1.0
    roll = eq.cummax()
    dd = (eq - roll) / roll
    mdd = float(dd.min())
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    wr = float((daily > 0).mean())
    gains = daily[daily > 0].sum()
    losses = -daily[daily < 0].sum()
    pf = float(gains / losses) if losses > 0 else 0.0
    # Day concentration: top single-day positive return / sum of positive returns
    pos = daily[daily > 0]
    day_conc = float(pos.max() / pos.sum()) if pos.sum() > 0 else 0.0
    return {
        "label": label,
        "n_days": int(len(daily)),
        "cagr_pct": round(cagr * 100, 3),
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "calmar": round(float(calmar), 4),
        "max_dd_pct": round(mdd * 100, 3),
        "pf": round(float(pf), 4),
        "wr": round(float(wr), 4),
        "day_conc": round(float(day_conc), 4),
    }


def load_book(path: Path, label: str) -> pd.Series:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["daily_ret"].astype(float).sort_index()
    s.name = label
    return s


def main() -> None:
    leader = load_book(LEADER_BOOK, "leader")
    tech = load_book(TECH_BOOK, "tech")

    aligned = pd.concat([leader, tech], axis=1, join="inner").dropna()
    n_overlap = len(aligned)
    if n_overlap < 20:
        raise SystemExit(f"Date intersection too small: {n_overlap} days")

    corr = float(aligned["leader"].corr(aligned["tech"]))
    overlap_start = aligned.index[0].date().isoformat()
    overlap_end = aligned.index[-1].date().isoformat()

    # Standalone reference (on intersection window for apples-to-apples)
    rows = []
    for w in WEIGHTS:
        blended = (1.0 - w) * aligned["leader"] + w * aligned["tech"]
        m = metrics(blended, label=f"w_tech={w:.2f}")
        m["w_tech"] = w
        m["w_leader"] = round(1.0 - w, 2)
        rows.append(m)

    sweep = pd.DataFrame(rows)
    # Reorder cols
    front = ["w_tech", "w_leader", "n_days",
             "cagr_pct", "sharpe", "sortino", "calmar",
             "max_dd_pct", "pf", "wr", "day_conc"]
    sweep = sweep[front + [c for c in sweep.columns if c not in front and c != "label"]]
    sweep_csv = OUT_DIR / "weight_sweep.csv"
    sweep.to_csv(sweep_csv, index=False)

    # Pick winners
    # Best Calmar with Sharpe >= 1.0 and MaxDD >= -15%
    calmar_cands = sweep[(sweep["sharpe"] >= 1.0) & (sweep["max_dd_pct"] >= -15.0)]
    best_calmar_row = (calmar_cands.sort_values("calmar", ascending=False).iloc[0]
                       if not calmar_cands.empty else None)

    # Best Sharpe with MaxDD >= -10%
    sharpe_cands = sweep[sweep["max_dd_pct"] >= -10.0]
    best_sharpe_row = (sharpe_cands.sort_values("sharpe", ascending=False).iloc[0]
                       if not sharpe_cands.empty else None)

    # Build & save best-calmar blended book
    if best_calmar_row is not None:
        w = float(best_calmar_row["w_tech"])
        blended = (1.0 - w) * aligned["leader"] + w * aligned["tech"]
        bbook = pd.DataFrame({
            "date": blended.index,
            "daily_ret": blended.values,
            "equity": (1.0 + blended).cumprod().values,
            "w_tech": w,
            "w_leader": 1.0 - w,
        })
        bbook.to_parquet(OUT_DIR / "best_calmar_book.parquet", index=False)

    report = {
        "leader_book": str(LEADER_BOOK),
        "tech_book": str(TECH_BOOK),
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "overlap_n_days": n_overlap,
        "leader_tech_return_corr": round(corr, 4),
        "weights_tested": WEIGHTS,
        "constraints": {
            "calmar_pick": "sharpe >= 1.0 AND max_dd_pct >= -15.0",
            "sharpe_pick": "max_dd_pct >= -10.0",
        },
        "best_calmar": (None if best_calmar_row is None
                        else best_calmar_row.to_dict()),
        "best_sharpe": (None if best_sharpe_row is None
                        else best_sharpe_row.to_dict()),
        "full_sweep": sweep.to_dict(orient="records"),
    }
    with open(OUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"OUT_DIR: {OUT_DIR}")
    print(f"overlap: {overlap_start} -> {overlap_end}  ({n_overlap} days)")
    print(f"leader<->tech daily-return correlation: {corr:.4f}")
    print()
    print(sweep.to_string(index=False))
    print()
    if best_calmar_row is not None:
        print("BEST CALMAR PICK (Sharpe>=1.0, MaxDD>=-15%):")
        print(best_calmar_row.to_string())
    if best_sharpe_row is not None:
        print()
        print("BEST SHARPE PICK (MaxDD>=-10%):")
        print(best_sharpe_row.to_string())


if __name__ == "__main__":
    main()
