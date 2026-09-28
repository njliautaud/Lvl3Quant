"""
BPS GA Hedge Overlay — Can rolling beta hedge make BPS pass R1?

BPS GA: Sharpe 2.76, gap 0.681 (FAIL R1). Alpha is REAL (permutation p=0.000).
V5 and ETF v3 both went from failing R1 to passing with rolling beta hedge.
Same approach here: rolling 60d beta-scaled SPY hedge.

HC #667: adversarial must produce data-driven numbers.
HC #670: rotation quality matters.
HC #428: R1 gate = regime-agnostic OOT validation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))

# BPS equity curve (15% BA haircut — realistic)
BPS_EQ_PATH = ROOT / "output/bps_full_stack_v2/eq_BA_15pct.parquet"
OUT_DIR = ROOT / "output/bps_hedge_overlay"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# SPY prices
PRICE_CACHE = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"


def load_spy_daily() -> pd.Series:
    """Load SPY daily close prices."""
    try:
        prices = pd.read_parquet(PRICE_CACHE)
        spy = prices[prices["ticker"] == "SPY"][["date", "close"]].copy()
        spy["date"] = pd.to_datetime(spy["date"])
        spy = spy.sort_values("date").set_index("date")["close"]
        return spy
    except Exception:
        # Fallback: yfinance
        import yfinance as yf
        spy = yf.download("SPY", start="2018-01-01", auto_adjust=True, progress=False)
        return spy["Close"]


def compute_r1_gap(daily_ret: pd.Series, spy_close: pd.Series) -> dict:
    """
    HC #428 R1: regime-agnostic validation.
    Green/Red days based on SPY daily close-to-close (±0.5% threshold).
    """
    spy_ret = spy_close.pct_change()

    # Align
    common = daily_ret.index.intersection(spy_ret.index)
    dr = daily_ret.loc[common]
    sr = spy_ret.loc[common]

    green = dr[sr > 0.005]
    red = dr[sr < -0.005]
    flat = dr[(sr >= -0.005) & (sr <= 0.005)]

    def sharpe(s):
        if len(s) < 10:
            return float('nan')
        return s.mean() / s.std() * np.sqrt(252) if s.std() > 0 else 0.0

    s_green = sharpe(green)
    s_red = sharpe(red)

    # R1 gap: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|)
    denom = max(abs(s_green), abs(s_red))
    gap = abs(s_green - s_red) / denom if denom > 0 else 0.0

    return {
        "sharpe_green": round(s_green, 4),
        "sharpe_red": round(s_red, 4),
        "sharpe_flat": round(sharpe(flat), 4),
        "n_green": len(green),
        "n_red": len(red),
        "n_flat": len(flat),
        "gap": round(gap, 4),
        "r1_pass": gap <= 0.50,
    }


def metrics(daily_ret: pd.Series) -> dict:
    """Standard performance metrics."""
    if len(daily_ret) < 20:
        return {}

    ann = np.sqrt(252)
    mu = daily_ret.mean()
    sigma = daily_ret.std()
    sharpe = mu / sigma * ann if sigma > 0 else 0.0

    # Sortino
    neg = daily_ret[daily_ret < 0]
    downside = np.sqrt((neg ** 2).mean()) if len(neg) > 0 else 1e-9
    sortino = mu / downside * ann

    # CAGR
    total = (1 + daily_ret).prod()
    years = len(daily_ret) / 252
    cagr = total ** (1 / years) - 1 if years > 0 else 0.0

    # Max DD
    cum = (1 + daily_ret).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0.0

    # PF and WR
    wins = daily_ret[daily_ret > 0].sum()
    losses = abs(daily_ret[daily_ret < 0].sum())
    pf = wins / losses if losses > 0 else float('inf')
    wr = (daily_ret > 0).mean()

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "calmar": round(calmar, 4),
        "pf": round(pf, 4),
        "wr": round(wr, 4),
        "n_days": len(daily_ret),
    }


def apply_beta_hedge(bps_ret: pd.Series, spy_ret: pd.Series,
                     window: int, scale: float, cost_bps: float = 5.0) -> pd.Series:
    """
    Apply rolling beta-scaled SPY hedge to BPS daily returns.

    On each day:
    1. Compute rolling beta of BPS vs SPY over `window` days
    2. Hedge return = bps_ret - beta * scale * spy_ret - daily_cost
    3. Daily cost accounts for hedge rebalancing (~5 bps per rebal)
    """
    common = bps_ret.index.intersection(spy_ret.index)
    bps = bps_ret.loc[common]
    spy = spy_ret.loc[common]

    # Rolling beta
    cov = bps.rolling(window, min_periods=max(20, window // 2)).cov(spy)
    var = spy.rolling(window, min_periods=max(20, window // 2)).var()
    beta = (cov / var).clip(0, 3.0)  # clip to reasonable range

    # Hedged return: subtract beta * scale * SPY return
    # Cost: small rebalancing cost per day the hedge changes
    beta_change = beta.diff().abs()
    daily_cost = beta_change * cost_bps / 10000  # cost proportional to hedge change

    hedged = bps - beta * scale * spy - daily_cost.fillna(0)

    return hedged.dropna()


def run():
    print("=" * 70)
    print("BPS GA HEDGE OVERLAY — R1 REGIME GATE FIX")
    print("=" * 70)

    # Load BPS equity curve
    eq = pd.read_parquet(BPS_EQ_PATH)
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date").set_index("date")
    bps_ret = eq["equity"].pct_change().dropna()
    print(f"BPS daily returns: {len(bps_ret)} days, {bps_ret.index.min().date()} to {bps_ret.index.max().date()}")

    # Load SPY
    spy_close = load_spy_daily()
    spy_ret = spy_close.pct_change().dropna()
    print(f"SPY daily returns: {len(spy_ret)} days")

    # Baseline (unhedged)
    m_base = metrics(bps_ret)
    r1_base = compute_r1_gap(bps_ret, spy_close)
    print(f"\n--- BASELINE (UNHEDGED) ---")
    print(f"  Sharpe: {m_base['sharpe']}, Sortino: {m_base['sortino']}")
    print(f"  CAGR: {m_base['cagr']*100:.1f}%, MaxDD: {m_base['max_dd']*100:.1f}%")
    print(f"  R1 gap: {r1_base['gap']} ({'PASS' if r1_base['r1_pass'] else 'FAIL'})")
    print(f"  Green Sharpe: {r1_base['sharpe_green']}, Red Sharpe: {r1_base['sharpe_red']}")

    # Sweep hedge parameters
    results = []
    windows = [40, 60, 90, 120]
    scales = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3]

    print(f"\n--- HEDGE SWEEP ({len(windows)} windows × {len(scales)} scales = {len(windows)*len(scales)} configs) ---")

    for w in windows:
        for s in scales:
            hedged = apply_beta_hedge(bps_ret, spy_ret, window=w, scale=s)
            m = metrics(hedged)
            r1 = compute_r1_gap(hedged, spy_close)

            r = {
                "window": w,
                "scale": s,
                **m,
                **{f"r1_{k}": v for k, v in r1.items()},
            }
            results.append(r)

            flag = " ✅ PASS" if r1["r1_pass"] else ""
            print(f"  w={w:3d} s={s:.2f}: Sharpe={m.get('sharpe',0):.2f} "
                  f"gap={r1['gap']:.3f} "
                  f"green={r1['sharpe_green']:.2f} red={r1['sharpe_red']:.2f}"
                  f"{flag}")

    # Find best R1-passing config
    passing = [r for r in results if r.get("r1_r1_pass")]

    print(f"\n{'='*70}")
    if not passing:
        print("NO CONFIGS PASS R1 — hedge overlay insufficient for BPS")
        verdict = "FAIL"
        best = max(results, key=lambda r: r.get("sharpe", -99))
        print(f"\nClosest config: w={best['window']} s={best['scale']}")
        print(f"  Sharpe: {best['sharpe']}, gap: {best['r1_gap']}")
    else:
        print(f"{len(passing)}/{len(results)} CONFIGS PASS R1!")

        # Best by Sharpe among passing
        best_sharpe = max(passing, key=lambda r: r.get("sharpe", -99))
        best_calmar = max(passing, key=lambda r: r.get("calmar", -99))
        best_gap = min(passing, key=lambda r: r.get("r1_gap", 99))

        print(f"\nBEST SHARPE (R1-passing): w={best_sharpe['window']} s={best_sharpe['scale']}")
        print(f"  Sharpe: {best_sharpe['sharpe']}, Sortino: {best_sharpe['sortino']}")
        print(f"  CAGR: {best_sharpe['cagr']*100:.1f}%, MaxDD: {best_sharpe['max_dd']*100:.1f}%")
        print(f"  Calmar: {best_sharpe['calmar']:.2f}, PF: {best_sharpe['pf']:.2f}, WR: {best_sharpe['wr']:.1%}")
        print(f"  R1 gap: {best_sharpe['r1_gap']:.3f} (green={best_sharpe['r1_sharpe_green']:.2f}, red={best_sharpe['r1_sharpe_red']:.2f})")

        print(f"\nBEST GAP (closest to regime-neutral): w={best_gap['window']} s={best_gap['scale']}")
        print(f"  Sharpe: {best_gap['sharpe']}, gap: {best_gap['r1_gap']:.3f}")

        verdict = "PASS"

    # Save results
    output = {
        "verdict": verdict,
        "baseline": {**m_base, **{f"r1_{k}": v for k, v in r1_base.items()}},
        "sweep_results": results,
        "n_passing": len(passing),
        "n_total": len(results),
        "best_config": best_sharpe if passing else None,
    }

    (OUT_DIR / "hedge_overlay_results.json").write_text(
        json.dumps(output, indent=2, default=str)
    )

    print(f"\nSaved: {OUT_DIR}/hedge_overlay_results.json")

    # Permutation test on best config if it passes
    if passing:
        print(f"\n--- PERMUTATION TEST (200 shuffles) ---")
        best = best_sharpe
        hedged_real = apply_beta_hedge(bps_ret, spy_ret, window=best["window"], scale=best["scale"])
        real_sharpe = metrics(hedged_real)["sharpe"]

        n_perm = 200
        perm_sharpes = []
        for i in range(n_perm):
            shuffled = bps_ret.copy()
            shuffled.values[:] = np.random.permutation(shuffled.values)
            hedged_shuf = apply_beta_hedge(shuffled, spy_ret, window=best["window"], scale=best["scale"])
            m_shuf = metrics(hedged_shuf)
            perm_sharpes.append(m_shuf.get("sharpe", 0))

            if (i + 1) % 50 == 0:
                p_so_far = sum(1 for s in perm_sharpes if s >= real_sharpe) / len(perm_sharpes)
                print(f"  {i+1}/{n_perm}: p={p_so_far:.4f} (real={real_sharpe:.2f} vs random mean={np.mean(perm_sharpes):.2f})")

        p_value = sum(1 for s in perm_sharpes if s >= real_sharpe) / n_perm
        print(f"\n  PERMUTATION RESULT: p={p_value:.4f}")
        print(f"  Real Sharpe: {real_sharpe:.2f} vs random mean: {np.mean(perm_sharpes):.2f}")
        print(f"  Verdict: {'PASS — edge is REAL' if p_value < 0.05 else 'FAIL — edge may be artifact'}")

        perm_output = {
            "p_value": p_value,
            "real_sharpe": real_sharpe,
            "random_mean_sharpe": round(np.mean(perm_sharpes), 4),
            "random_std_sharpe": round(np.std(perm_sharpes), 4),
            "n_permutations": n_perm,
            "pass": p_value < 0.05,
        }
        output["permutation_test"] = perm_output

        (OUT_DIR / "hedge_overlay_results.json").write_text(
            json.dumps(output, indent=2, default=str)
        )

    print(f"\n{'='*70}")
    print("DONE")
    return output


if __name__ == "__main__":
    run()
