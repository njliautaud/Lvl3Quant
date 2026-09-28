"""
Capacity analysis for leader (ETF rotation hold21 longonly + intra-hold regime gate).

Method:
  1. Replay the leader's pick logic to extract per-rebal (date, longs[], gross_lev).
  2. For each rebal, compute participation rate per ETF at AUM in {1M, 10M, 50M, 100M, 500M, 1B}.
     Per-ETF dollar trade = AUM * gross_lev / n_long.
     ADV per ETF on rebal day = trailing 20-day mean of close * volume.
     Participation = dollar_trade / ADV.
  3. Apply sqrt market-impact model: impact_bps = k * sqrt(participation).
     Use k=10 (standard liquid-ETF prior; bp impact at 1% participation = 10*sqrt(0.01)*10000bps = bug).
     Standard parameterization: impact_bps = k * sqrt(participation_pct) where participation_pct in [0,100].
     Equivalent: impact_bps = k * sqrt(participation) * 10 if participation is a decimal fraction.
     We use: impact_bps_per_side = 10 * sqrt(100 * participation_decimal).
     Two-sided (entry + exit) round-trip: impact_rt_bps = 2 * impact_bps_per_side.
  4. Subtract incremental impact (above 5bps baseline already in book) from daily_ret on rebal days.
  5. Recompute pooled Sharpe/Sortino/Calmar/MaxDD/PF/WR.
  6. Decompose: fraction of total impact from the 3 smallest-ADV ETFs in the holdings.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "strategy/macro_picker"))
import etf_rotation_v1 as etf  # noqa: E402

PANEL_V2 = ROOT / "data/feature_store/master_panel/master_panel_v2.parquet"
LEADER = ROOT / "output/macro_picker/etf_rotation_regime_20260608_164726_hold21_longonly"
OUT = ROOT / f"output/macro_picker/capacity_{time.strftime('%Y%m%d_%H%M%S')}"
OUT.mkdir(parents=True, exist_ok=True)

# Strategy config (must match leader)
HOLD_DAYS = 21
N_LONG = 2
N_SHORT = 0
ALLOW_SHORT = False
TARGET_VOL = 0.15
LEV_MIN = 0.25
LEV_MAX = 2.0
TRAIN_M, OOT_M, STEP_M = 24, 6, 3
BASE_TXN_BPS = 5.0
REGIME_MA_DAYS = 60

UNIVERSE = ["XLK","XLF","XLE","XLY","XLP","XLU","XLI","XLV","XLB","XLC","XLRE"]


def build_etf_panel() -> pd.DataFrame:
    print("[capacity] loading master panel...")
    panel = pd.read_parquet(PANEL_V2)
    panel = panel[panel["ticker"].isin(UNIVERSE)].copy()
    panel = panel.rename(columns={"ticker": "etf"})
    panel["date"] = pd.to_datetime(panel["date"])
    panel["ret_1d"] = panel["ret"].astype(float)
    # ADV: trailing 20-day mean of close*volume
    panel["dvol"] = panel["close"] * panel["volume"]
    panel = panel.sort_values(["etf","date"]).reset_index(drop=True)
    panel["adv_20d"] = panel.groupby("etf")["dvol"].transform(
        lambda s: s.rolling(20, min_periods=5).mean().shift(1)  # PIT: don't use today
    )
    return panel


def replay_picks() -> pd.DataFrame:
    """Re-run the leader's WF logic, capture per-rebal (date, longs, gross_lev)."""
    print("[capacity] replaying leader picks per rebal...")
    panel, feats = etf.build_panel(HOLD_DAYS)
    print(f"  panel rows={len(panel)}, etfs={panel['etf'].nunique()}, feats={len(feats)}")
    # ret_1d derived from close
    panel = panel.sort_values(["etf","date"]).reset_index(drop=True)
    panel["ret_1d"] = panel.groupby("etf")["close"].pct_change()
    # SPY regime classifier
    regime_filter = etf._load_spy_regime(REGIME_MA_DAYS)

    # WF windows
    start = panel["date"].min()
    end = panel["date"].max()
    windows = etf._iter_wf_windows(start, end, TRAIN_M, OOT_M, STEP_M)

    rebal_records = []  # list of dicts: date, longs, gross_lev
    for tr_s, tr_e, oot_s, oot_e in windows:
        train = panel[(panel["date"] >= tr_s) & (panel["date"] < tr_e)].copy()
        oot = panel[(panel["date"] >= oot_s) & (panel["date"] < oot_e)].copy()
        if len(train) < 200 or len(oot) < 20:
            continue
        train_z = etf._xs_zscore(train, feats)
        oot_z = etf._xs_zscore(oot, feats)
        for f in feats:
            train_z[f] = train_z[f].fillna(0.0)
            oot_z[f] = oot_z[f].fillna(0.0)
        train_z = train_z.dropna(subset=["y_fwd"])
        if train_z.empty:
            continue
        X_tr = train_z[feats].values
        y_tr = train_z["y_fwd"].values
        coef, intercept, alpha = etf._fit_ridge(X_tr, y_tr)
        if coef is None:
            continue
        X_oot = oot_z[feats].values
        oot_z["score"] = X_oot @ coef + intercept
        oot_z["ret_raw"] = pd.to_numeric(oot["ret_1d"], errors="coerce").astype(float).values

        unique_dates = sorted(oot_z["date"].unique())
        rebal_dates = unique_dates[::HOLD_DAYS]
        for rd in rebal_dates:
            rg = regime_filter.get(pd.Timestamp(rd))
            if rg is None:
                prior = regime_filter.loc[:pd.Timestamp(rd)]
                rg = prior.iloc[-1] if len(prior) else "bull"
            if rg != "bull":
                continue
            snap = oot_z[oot_z["date"] == rd].dropna(subset=["score"])
            if len(snap) < N_LONG:
                continue
            med = snap["score"].median()
            top_score = snap["score"].max()
            if top_score <= med:
                continue
            longs = snap.nlargest(N_LONG, "score")["etf"].tolist()
            realised_vol = etf._estimate_book_vol(panel, rd, longs, [])
            if realised_vol <= 1e-6:
                gl = 1.0
            else:
                gl = float(np.clip(TARGET_VOL / realised_vol, LEV_MIN, LEV_MAX))
            rebal_records.append({
                "date": pd.Timestamp(rd),
                "longs": longs,
                "gross_lev": gl,
            })
    return pd.DataFrame(rebal_records)


def adv_lookup(etf_panel: pd.DataFrame) -> dict:
    """date -> {etf: adv_20d}"""
    g = etf_panel[["date","etf","adv_20d"]].dropna()
    out = {}
    for (d, e), v in g.set_index(["date","etf"])["adv_20d"].items():
        out.setdefault(d, {})[e] = v
    return out


def compute_impact_drag(rebals: pd.DataFrame, adv_map: dict,
                        aum_usd: float, k: float = 10.0) -> dict:
    """Return per-rebal incremental cost (in bps, round-trip) above baseline.
    Sqrt model: per-side impact_bps = k * sqrt(participation_pct),
    where participation_pct = 100 * dollar_trade / ADV.
    Round-trip = 2 * per_side (entry + exit at next rebal).
    """
    rebal_costs = []
    impact_share_by_etf = {}
    for _, r in rebals.iterrows():
        per_etf_dollar = aum_usd * r["gross_lev"] / N_LONG
        date_advs = adv_map.get(r["date"], {})
        leg_impacts = []
        for e in r["longs"]:
            adv = date_advs.get(e)
            if adv is None or adv <= 0 or not np.isfinite(adv):
                # fallback: median ADV across all dates for this ETF
                adv = 0
            if adv > 0:
                participation_pct = 100.0 * per_etf_dollar / adv
                impact_bps_side = k * np.sqrt(participation_pct)
            else:
                impact_bps_side = 100.0  # heavy penalty if no ADV
            rt_bps = 2.0 * impact_bps_side
            leg_impacts.append((e, rt_bps))
            impact_share_by_etf.setdefault(e, []).append(rt_bps)
        # Average impact across the n_long legs (each contributes 50% of portfolio)
        avg_rt = float(np.mean([x[1] for x in leg_impacts])) if leg_impacts else 0.0
        rebal_costs.append({
            "date": r["date"],
            "gross_lev": r["gross_lev"],
            "longs": r["longs"],
            "avg_impact_rt_bps": avg_rt,
            "per_leg_bps": leg_impacts,
        })
    return {"rebal_costs": rebal_costs,
            "impact_share_by_etf": impact_share_by_etf}


def apply_impact_to_book(book: pd.DataFrame, rebal_costs: list,
                         base_bps: float = BASE_TXN_BPS) -> pd.Series:
    """Subtract incremental impact (above base) on rebal days, scaled by leverage."""
    b = book.copy()
    b["date"] = pd.to_datetime(b["date"])
    b = b.sort_values("date").set_index("date")
    daily = b["daily_ret"].copy()
    for rc in rebal_costs:
        d = rc["date"]
        delta_bps = max(0.0, rc["avg_impact_rt_bps"] - base_bps)
        if delta_bps <= 0:
            continue
        # cost applied on rebal day; scaled by gross_lev (matches base implementation)
        # base script applies cost = (bps/10000)*gross_lev; we mimic the same shape
        delta = (delta_bps / 10000.0) * rc["gross_lev"]
        if d in daily.index:
            daily.loc[d] = daily.loc[d] - delta
    return daily


def metrics(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 5:
        return {"n": 0}
    mu = s.mean(); sd = s.std(ddof=1)
    sharpe = mu / sd * np.sqrt(252) if sd > 0 else 0.0
    down = s[s < 0].std(ddof=1)
    sortino = mu / down * np.sqrt(252) if down else 0.0
    eq = (1 + s).cumprod()
    n_yr = max(len(s)/252.0, 1e-6)
    cagr = eq.iloc[-1] ** (1/n_yr) - 1
    dd = (eq - eq.cummax()) / eq.cummax()
    mdd = float(dd.min())
    calmar = cagr/abs(mdd) if mdd else 0.0
    wr = float((s>0).mean())
    pf = (s[s>0].sum() / -s[s<0].sum()) if (s<0).any() else 0.0
    return {"sharpe": float(sharpe), "sortino": float(sortino),
            "cagr": float(cagr), "max_dd": mdd, "calmar": float(calmar),
            "pf": float(pf), "wr": wr, "n": int(len(s))}


def main():
    panel_full = build_etf_panel()

    # Replay picks
    rebals = replay_picks()
    print(f"[capacity] {len(rebals)} rebal events")
    rebals.to_parquet(OUT / "rebal_picks.parquet")
    # Count holdings frequency
    from collections import Counter
    pick_counter = Counter()
    for ll in rebals["longs"]:
        for e in ll:
            pick_counter[e] += 1
    print("\nPick frequency:")
    for e, c in pick_counter.most_common():
        print(f"  {e}: {c}")

    # Build ADV map
    adv_map = adv_lookup(panel_full)

    # Load leader book
    book = pd.read_parquet(LEADER / "book.parquet")

    # HC #580 R3: anchor to real net worth — $20K → $50K → $100K → $250K (% returns primary).
    AUM_LEVELS = [20e3, 50e3, 100e3, 250e3]
    aum_labels = ["$20K","$50K","$100K","$250K"]

    rows = []
    per_etf_share = []
    for aum, lbl in zip(AUM_LEVELS, aum_labels):
        impact = compute_impact_drag(rebals, adv_map, aum, k=10.0)
        daily_aum = apply_impact_to_book(book, impact["rebal_costs"])
        m = metrics(daily_aum)
        # Mean incremental impact rt bps across rebals
        mean_rt_bps = float(np.mean([rc["avg_impact_rt_bps"] for rc in impact["rebal_costs"]]))
        m["aum"] = lbl
        m["aum_usd"] = aum
        m["mean_impact_rt_bps"] = mean_rt_bps
        rows.append(m)
        # ETF-level impact share (sum of bps contributions)
        share_sum = {e: sum(vals) for e, vals in impact["impact_share_by_etf"].items()}
        total = sum(share_sum.values())
        share_pct = {e: 100.0*v/total for e,v in share_sum.items()} if total else {}
        per_etf_share.append({"aum": lbl, "share_pct": share_pct})

    df = pd.DataFrame(rows)[["aum","mean_impact_rt_bps","sharpe","sortino","calmar","max_dd","cagr","pf","wr","n"]]
    print("\n[capacity] AUM ladder:")
    print(df.to_string(index=False))
    df.to_csv(OUT / "capacity_ladder.csv", index=False)

    # ETF-level decomposition at largest AUM
    print("\n[capacity] Per-ETF impact share at $1B AUM:")
    big = per_etf_share[-1]["share_pct"]
    sorted_share = sorted(big.items(), key=lambda x: -x[1])
    for e, pct in sorted_share:
        print(f"  {e}: {pct:.1f}%  (pick count: {pick_counter.get(e,0)})")

    # Smallest 3 ETFs by ADV (taken from latest 60d window)
    adv_by_etf = {}
    for e in UNIVERSE:
        s = panel_full[panel_full["etf"]==e].sort_values("date").tail(60)
        adv_by_etf[e] = float(s["dvol"].mean()) if not s.empty else 0
    sorted_adv = sorted(adv_by_etf.items(), key=lambda x: x[1])
    smallest3 = [e for e, _ in sorted_adv[:3]]
    print(f"\nSmallest-3 ADV ETFs: {smallest3} (ADVs: {[adv_by_etf[e]/1e6 for e in smallest3]} M$)")

    # Impact contribution of smallest 3
    small3_decomp = []
    for sh, lbl in zip(per_etf_share, aum_labels):
        share = sh["share_pct"]
        small3_pct = sum(share.get(e, 0) for e in smallest3)
        small3_decomp.append({"aum": lbl, "smallest3_impact_pct": small3_pct,
                              "smallest3": smallest3})
    print("\nSmallest-3 ETFs share of total impact:")
    for r in small3_decomp:
        print(f"  {r['aum']}: {r['smallest3_impact_pct']:.1f}%")

    # Sharpe-=1.0 capacity breakpoint
    sharpe_arr = df["sharpe"].values
    aum_arr = np.array([r["aum_usd"] for r in rows])
    cap_at_sharpe_1 = None
    for i in range(len(aum_arr)-1):
        if sharpe_arr[i] >= 1.0 >= sharpe_arr[i+1]:
            # log-linear interp on AUM
            x0, x1 = np.log10(aum_arr[i]), np.log10(aum_arr[i+1])
            y0, y1 = sharpe_arr[i], sharpe_arr[i+1]
            interp_log = x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0)
            cap_at_sharpe_1 = 10 ** interp_log
            break
    print(f"\nCAPACITY (Sharpe=1.0): ${cap_at_sharpe_1/1e6:.0f}M" if cap_at_sharpe_1 else
          "\nCAPACITY: Sharpe>=1.0 at all tested AUM (uncapped within range)")

    with open(OUT / "report.json", "w") as f:
        json.dump({
            "config": {"hold_days": HOLD_DAYS, "n_long": N_LONG,
                       "model_k": 10.0, "base_bps": BASE_TXN_BPS,
                       "universe": UNIVERSE},
            "n_rebals": len(rebals),
            "pick_frequency": dict(pick_counter),
            "ladder": rows,
            "smallest_3_etfs_by_adv": smallest3,
            "smallest_3_adv_usd_m": {e: adv_by_etf[e]/1e6 for e in smallest3},
            "smallest_3_impact_share_pct": small3_decomp,
            "per_etf_share_at_1B": sorted_share,
            "capacity_at_sharpe_1": cap_at_sharpe_1,
        }, f, indent=2, default=str)
    print(f"\n[capacity] outputs -> {OUT}")


if __name__ == "__main__":
    main()
