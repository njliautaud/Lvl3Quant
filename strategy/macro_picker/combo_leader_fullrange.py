"""
Combo evaluation aligned on LEADER FULL DATE RANGE (419 days).
On dates where utilities sleeve has no return, contribute 0.0 (cash).
This preserves leader's headline Sharpe ~1.9 / Calmar ~3.2 baseline as the comparator.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "strategy/macro_picker"))

LEADER_BOOK = ROOT / "output/macro_picker/etf_rotation_regime_20260608_164726_hold21_longonly/book.parquet"
COMBO_DIR_PREV = ROOT / "output/macro_picker/combo_leader_utilsleeve_20260608_223923"
OUT = ROOT / f"output/macro_picker/combo_fullrange_{time.strftime('%Y%m%d_%H%M%S')}"
OUT.mkdir(parents=True, exist_ok=True)


def metrics(daily, label=""):
    daily = daily.dropna()
    if daily.empty:
        return {"label": label, "n": 0}
    eq = (1 + daily).cumprod()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() else 0.0
    down = daily[daily < 0].std()
    sortino = daily.mean() / down * np.sqrt(252) if down else 0.0
    n_days = max(1, (daily.index[-1] - daily.index[0]).days)
    cagr = eq.iloc[-1] ** (365.25 / n_days) - 1
    roll = eq.cummax()
    dd = (eq - roll) / roll
    mdd = float(dd.min())
    calmar = cagr / abs(mdd) if mdd else 0.0
    wr = float((daily > 0).mean())
    pf = (daily[daily > 0].sum() / -daily[daily < 0].sum()) if (daily < 0).any() else 0.0
    return {"label": label, "sharpe": float(sharpe), "sortino": float(sortino),
            "cagr": float(cagr), "max_dd": mdd, "calmar": float(calmar),
            "wr": wr, "pf": float(pf), "n": int(len(daily))}


def main():
    lb = pd.read_parquet(LEADER_BOOK)
    lb["date"] = pd.to_datetime(lb["date"])
    L = lb.set_index("date")["daily_ret"].sort_index()
    print(f"leader: {len(L)} days")

    daily = pd.read_parquet(COMBO_DIR_PREV / "daily.parquet")
    U_raw = daily["utilities"].copy()
    # Reindex utilities onto leader range; missing -> 0 (cash sleeve on those days)
    U = U_raw.reindex(L.index).fillna(0.0)

    rows = [metrics(L, "leader_full_419d"), metrics(U, "utilities_fullrange_zerofill")]
    gates = []
    for w_u in [0.05, 0.10, 0.15, 0.20]:
        combo = (1 - w_u) * L + w_u * U
        m = metrics(combo, f"combo_w_util_{w_u:.2f}")
        rows.append(m)
        gates.append({
            "w_util": w_u,
            "sharpe": m["sharpe"], "calmar": m["calmar"],
            "max_dd": m["max_dd"], "sortino": m["sortino"], "pf": m["pf"],
            "pass_sharpe_ge_1_5": m["sharpe"] >= 1.5,
            "pass_calmar_ge_1_5": m["calmar"] >= 1.5,
            "pass_dd_gt_neg_15": m["max_dd"] > -0.15,
        })
        gates[-1]["DEPLOY_PASS"] = all([gates[-1]["pass_sharpe_ge_1_5"],
                                        gates[-1]["pass_calmar_ge_1_5"],
                                        gates[-1]["pass_dd_gt_neg_15"]])

    df = pd.DataFrame(rows)
    print("\nSummary (leader full 419-day window, utility missing -> cash 0):")
    print(df[["label", "sharpe", "sortino", "cagr", "max_dd", "calmar", "pf", "wr", "n"]]
          .to_string(index=False))

    g = pd.DataFrame(gates)
    print("\nDeploy gates:")
    print(g.to_string(index=False))

    df.to_csv(OUT / "summary.csv", index=False)
    g.to_csv(OUT / "deploy_gates.csv", index=False)
    with open(OUT / "report.json", "w") as f:
        json.dump({"summary": rows, "gates": gates}, f, indent=2)
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
