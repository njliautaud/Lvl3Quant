"""
Combo test: 0.9 * leader (ETF rotation hold21 longonly + intra-hold regime gate)
          + 0.1 * utilities-sector-only v9 sub-industry tilt picker

Goal: check whether the small "utilities sleeve" lifts Calmar past 1.5 deploy gate
while preserving leader's Sharpe ~1.9.

Outputs: per-day combined returns, headline Sharpe/Sortino/CAGR/MaxDD/Calmar/PF/WR.
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
PANEL_V2 = ROOT / "data/feature_store/master_panel/master_panel_v2.parquet"
OUT_DIR = ROOT / f"output/macro_picker/combo_leader_utilsleeve_{time.strftime('%Y%m%d_%H%M%S')}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def metrics(daily: pd.Series, label: str = "") -> dict:
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


def build_utilities_series(hold_days: int = 10, K: int = 10, lam: float = 1.0) -> pd.Series:
    """Re-run v9 logic restricted to Utilities sector. Returns per-date Series."""
    import sector_picker_v9_subind_tilt as v9
    panel = pd.read_parquet(PANEL_V2)
    panel["date"] = pd.to_datetime(panel["date"])
    feats = [c for c in panel.columns if c not in (
        "ticker", "date", "sector", "ret", "y_fwd", "open", "high", "low",
        "close", "volume", "log_ret",
    )]
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["y_fwd"] = (panel.groupby("ticker")["ret"]
                      .rolling(hold_days).sum().shift(-hold_days)
                      .reset_index(level=0, drop=True))
    sp = panel[panel["sector"] == "Utilities"].copy()
    if sp.empty:
        raise RuntimeError("No Utilities rows in panel")
    res = v9.portfolio_for_sector_v9(sp, feats, K=K, hold_days=hold_days, lam=lam)
    s = res["daily_pnl"].copy()
    # De-overlap: groupby date sum (multiple folds may write to same date)
    s = s.groupby(level=0).sum()
    s = s.sort_index()
    return s


def main():
    print("[combo] loading leader book...")
    lb = pd.read_parquet(LEADER_BOOK)
    lb["date"] = pd.to_datetime(lb["date"])
    leader = lb.set_index("date")["daily_ret"].sort_index()
    print(f"  leader: {len(leader)} days, {leader.index.min().date()} -> {leader.index.max().date()}")

    print("[combo] building utilities-only v9 picker series...")
    util = build_utilities_series(hold_days=10, K=10, lam=1.0)
    print(f"  utilities: {len(util)} days, {util.index.min().date()} -> {util.index.max().date()}")

    # Align on leader date range (leader is the deploy candidate; we evaluate combo
    # over the SAME window leader is validated on -- 2023-06-07 to 2026-02-27).
    common = leader.index.intersection(util.index)
    print(f"  intersect on leader range: {len(common)} days")
    L = leader.reindex(common).fillna(0.0)
    U = util.reindex(common).fillna(0.0)

    # Headline standalone metrics on the SAME aligned window
    m_leader = metrics(L, "leader_aligned")
    m_util = metrics(U, "utilities_aligned")

    # Combos: vary utilities weight
    rows = [m_leader, m_util]
    daily_records = {"leader": L, "utilities": U}
    for w_u in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        combo = (1 - w_u) * L + w_u * U
        m = metrics(combo, f"combo_leader{1-w_u:.2f}_util{w_u:.2f}")
        rows.append(m)
        daily_records[f"combo_w_util_{w_u:.2f}"] = combo

    # Also: combo using utilities FULL period (only at dates where leader has obs)
    df_summary = pd.DataFrame(rows)
    print("\n[combo] summary:")
    print(df_summary[["label", "sharpe", "sortino", "cagr", "max_dd", "calmar", "pf", "wr", "n"]]
          .to_string(index=False))

    # Save
    df_summary.to_csv(OUT_DIR / "summary.csv", index=False)
    daily_df = pd.DataFrame(daily_records)
    daily_df.to_parquet(OUT_DIR / "daily.parquet")

    # Deploy-gate evaluation per combo
    gates = []
    for w_u in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        combo = (1 - w_u) * L + w_u * U
        m = metrics(combo)
        gate = {
            "w_util": w_u,
            "sharpe": m.get("sharpe", 0),
            "calmar": m.get("calmar", 0),
            "max_dd": m.get("max_dd", 0),
            "sortino": m.get("sortino", 0),
            "pass_sharpe_ge_1_5": m.get("sharpe", 0) >= 1.5,
            "pass_calmar_ge_1_5": m.get("calmar", 0) >= 1.5,
            "pass_dd_gt_neg_15": m.get("max_dd", 0) > -0.15,
        }
        gate["DEPLOY_PASS"] = all([gate["pass_sharpe_ge_1_5"],
                                    gate["pass_calmar_ge_1_5"],
                                    gate["pass_dd_gt_neg_15"]])
        gates.append(gate)
    gate_df = pd.DataFrame(gates)
    print("\n[combo] deploy gates:")
    print(gate_df.to_string(index=False))
    gate_df.to_csv(OUT_DIR / "deploy_gates.csv", index=False)

    with open(OUT_DIR / "report.json", "w") as f:
        json.dump({
            "leader_book": str(LEADER_BOOK),
            "utilities_config": {"K": 10, "hold_days": 10, "lam": 1.0, "sector": "Utilities"},
            "common_dates_n": len(common),
            "date_range": [str(common.min().date()), str(common.max().date())],
            "summary_rows": rows,
            "deploy_gates": gates,
        }, f, indent=2)

    print(f"\n[combo] outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
