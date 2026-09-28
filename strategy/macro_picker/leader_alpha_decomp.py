"""
Leader ETF rotation — alpha decomposition vs SPY benchmark.

The "leader" config (etf_rotation_v1.py: hold21, longonly, intra-hold regime
gate) failed HC #428 R1 regime gate. Question: does it have REAL alpha after
stripping market beta, or is it just a long-equity proxy?

Method:
  1. Load leader daily PnL from cached book.parquet (419 OOT days).
  2. Load SPY daily total-return on the same dates.
  3. Univariate OLS: leader_ret = alpha + beta * SPY_ret + eps
     - Newey-West HAC SE (5 lags) for t-stat on alpha.
     - Annualised alpha, R^2, Sharpe of residuals (pure-alpha series).
     - Information Ratio = annualised alpha / annualised sigma(residuals).
  4. Buy-and-hold comparison on same window:
     - SPY B&H: CAGR, Sharpe, Sortino, Calmar, MaxDD.
     - Leader: CAGR, Sharpe, Sortino, Calmar, MaxDD.
     - Excess: Leader - SPY on CAGR & Sharpe.
  5. Capital efficiency: mean gross exposure, Sharpe/mean_exposure.

Inputs (hardcoded to the regime-gated hold21 longonly run):
  output/macro_picker/etf_rotation_regime_20260608_164726_hold21_longonly/
    book.parquet  (date, daily_ret, gross_lev)
SPY prices:
  wheel_strategy_v1/data/cache/prices_v2.parquet

Output:
  output/macro_picker/leader_alpha_decomp_<TS>/results.json
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")

LEADER_BOOK = ROOT / "output/macro_picker/etf_rotation_regime_20260608_164726_hold21_longonly/book.parquet"
SPY_PRICE_PATH = ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet"

TRADING_DAYS = 252


# -----------------------------------------------------------------------------
# OLS w/ Newey-West HAC
# -----------------------------------------------------------------------------
def ols_newey_west(y: np.ndarray, x: np.ndarray, lags: int = 5) -> dict:
    """OLS of y on [1, x]. Returns alpha (intercept), beta, residuals,
    HAC standard errors w/ Bartlett kernel (Newey-West) at given lags.
    """
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    # OLS via normal equations
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    coef = XtX_inv @ X.T @ y
    alpha = float(coef[0])
    beta = float(coef[1])
    resid = y - X @ coef

    # Newey-West HAC covariance: S = sum_{t} u_t u_t' (X_t X_t')
    # Build meat: omega = sum_{l=0..lags} w_l * (Gamma_l + Gamma_l')
    # Bartlett weights w_l = 1 - l/(lags+1).
    u = resid
    # Compute Gamma_l = (1/n) sum_{t=l+1..n} (u_t X_t) (u_{t-l} X_{t-l})'
    omega = np.zeros((X.shape[1], X.shape[1]))
    Xu = X * u[:, None]  # n x k
    # Gamma_0
    Gamma0 = (Xu.T @ Xu) / n
    omega += Gamma0
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        Gl = (Xu[l:].T @ Xu[:-l]) / n
        omega += w * (Gl + Gl.T)
    # NW covariance:  n * XtX_inv @ omega @ XtX_inv
    nw_cov = n * XtX_inv @ omega @ XtX_inv
    se_alpha = float(np.sqrt(nw_cov[0, 0]))
    se_beta = float(np.sqrt(nw_cov[1, 1]))
    t_alpha = alpha / se_alpha if se_alpha > 0 else float("nan")
    t_beta = beta / se_beta if se_beta > 0 else float("nan")

    # R^2
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "alpha_daily": alpha,
        "beta": beta,
        "se_alpha_nw": se_alpha,
        "se_beta_nw": se_beta,
        "t_alpha_nw": t_alpha,
        "t_beta_nw": t_beta,
        "r2": r2,
        "residuals": resid,
        "n_obs": n,
        "nw_lags": lags,
    }


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def perf_metrics(daily_ret: pd.Series) -> dict:
    """CAGR / Sharpe / Sortino / Calmar / MaxDD from a daily return series."""
    r = daily_ret.dropna()
    if len(r) < 2:
        return {}
    n = len(r)
    yrs = n / TRADING_DAYS
    eq = (1.0 + r).cumprod()
    final = float(eq.iloc[-1])
    cagr = final ** (1.0 / yrs) - 1.0 if yrs > 0 else float("nan")
    sd = float(r.std(ddof=1))
    sharpe = (r.mean() / sd) * np.sqrt(TRADING_DAYS) if sd > 0 else float("nan")
    downside = r[r < 0]
    sd_down = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (r.mean() / sd_down) * np.sqrt(TRADING_DAYS) if sd_down > 0 else float("nan")
    # MaxDD
    peak = eq.cummax()
    dd = (eq / peak - 1.0)
    max_dd = float(dd.min())
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "max_dd": max_dd,
        "n_days": int(n),
        "years": float(yrs),
        "total_return": float(final - 1.0),
    }


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_leader_book(path: Path) -> pd.DataFrame:
    b = pd.read_parquet(path)
    b["date"] = pd.to_datetime(b["date"])
    b = b.sort_values("date").reset_index(drop=True)
    return b


def load_spy_daily_ret() -> pd.Series:
    px = pd.read_parquet(SPY_PRICE_PATH)
    px["date"] = pd.to_datetime(px["date"])
    spy = px[px["ticker"] == "SPY"].sort_values("date").set_index("date")["close"]
    return spy.pct_change().dropna().rename("spy_ret")


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run(out_dir: Path, nw_lags: int = 5) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[alpha_decomp] loading leader book from {LEADER_BOOK.name}")
    leader = load_leader_book(LEADER_BOOK)
    print(f"[alpha_decomp] leader rows={len(leader)} "
          f"range={leader['date'].min().date()}->{leader['date'].max().date()}")

    spy_ret = load_spy_daily_ret()
    print(f"[alpha_decomp] spy returns rows={len(spy_ret)}")

    # Align on leader's date index
    merged = (leader.set_index("date")[["daily_ret", "gross_lev"]]
              .join(spy_ret, how="inner"))
    merged = merged.dropna(subset=["daily_ret", "spy_ret"])
    print(f"[alpha_decomp] aligned rows={len(merged)}")

    y = merged["daily_ret"].values.astype(float)  # leader daily ret
    x = merged["spy_ret"].values.astype(float)    # SPY daily ret

    # 1) OLS with NW HAC
    fit = ols_newey_west(y, x, lags=nw_lags)
    alpha_daily = fit["alpha_daily"]
    beta = fit["beta"]
    resid = fit["residuals"]
    resid_s = pd.Series(resid, index=merged.index, name="residual")

    alpha_annualised = alpha_daily * TRADING_DAYS
    resid_sd_daily = float(resid_s.std(ddof=1))
    resid_sd_annual = resid_sd_daily * np.sqrt(TRADING_DAYS)
    sharpe_residuals = (resid_s.mean() / resid_sd_daily) * np.sqrt(TRADING_DAYS) \
        if resid_sd_daily > 0 else float("nan")
    info_ratio = alpha_annualised / resid_sd_annual if resid_sd_annual > 0 else float("nan")

    # 2) Buy-and-hold perf on same window
    leader_perf = perf_metrics(merged["daily_ret"])
    spy_perf = perf_metrics(merged["spy_ret"])
    excess = {
        "cagr_excess": leader_perf["cagr"] - spy_perf["cagr"],
        "sharpe_excess": leader_perf["sharpe"] - spy_perf["sharpe"],
        "sortino_excess": leader_perf["sortino"] - spy_perf["sortino"],
        "maxdd_excess": leader_perf["max_dd"] - spy_perf["max_dd"],
    }

    # 3) Capital efficiency
    lev = merged["gross_lev"].astype(float)
    # mean exposure over all days (including 0-exposure regime-cash days)
    mean_exposure_all = float(lev.mean())
    # mean exposure conditional on being deployed (>0)
    deployed = lev[lev > 0]
    mean_exposure_when_in = float(deployed.mean()) if len(deployed) else float("nan")
    days_in_market = int((lev > 0).sum())
    pct_in_market = days_in_market / len(lev)
    sharpe_per_unit_exposure_all = leader_perf["sharpe"] / mean_exposure_all \
        if mean_exposure_all > 0 else float("nan")
    sharpe_per_unit_exposure_in = leader_perf["sharpe"] / mean_exposure_when_in \
        if mean_exposure_when_in and mean_exposure_when_in > 0 else float("nan")

    # Assemble results
    results = {
        "window": {
            "date_start": str(merged.index.min().date()),
            "date_end": str(merged.index.max().date()),
            "n_days": int(len(merged)),
            "years": float(len(merged) / TRADING_DAYS),
        },
        "ols_capm_style": {
            "alpha_daily": alpha_daily,
            "alpha_annualised": float(alpha_annualised),
            "alpha_annualised_pct": float(alpha_annualised * 100),
            "beta": beta,
            "se_alpha_nw5": fit["se_alpha_nw"],
            "se_beta_nw5": fit["se_beta_nw"],
            "t_alpha_nw5": fit["t_alpha_nw"],
            "t_beta_nw5": fit["t_beta_nw"],
            "r2": fit["r2"],
            "nw_lags": fit["nw_lags"],
            "n_obs": fit["n_obs"],
        },
        "pure_alpha_series": {
            "sharpe_residuals": float(sharpe_residuals),
            "residual_sd_daily": resid_sd_daily,
            "residual_sd_annual": resid_sd_annual,
            "information_ratio": float(info_ratio),
        },
        "buy_and_hold_compare": {
            "leader": leader_perf,
            "spy": spy_perf,
            "excess": excess,
        },
        "capital_efficiency": {
            "mean_exposure_all_days": mean_exposure_all,
            "mean_exposure_when_deployed": mean_exposure_when_in,
            "pct_days_in_market": float(pct_in_market),
            "days_in_market": days_in_market,
            "sharpe_per_unit_exposure_all_days": float(sharpe_per_unit_exposure_all),
            "sharpe_per_unit_exposure_when_deployed": float(sharpe_per_unit_exposure_in),
        },
        "verdict_inputs": {
            "alpha_significant_at_5pct_t_ge_1p96": bool(abs(fit["t_alpha_nw"]) >= 1.96),
            "information_ratio_ge_0p5": bool(info_ratio >= 0.5),
            "sharpe_residuals_positive": bool(sharpe_residuals > 0),
            "leader_beats_spy_sharpe": bool(leader_perf["sharpe"] > spy_perf["sharpe"]),
            "leader_beats_spy_cagr": bool(leader_perf["cagr"] > spy_perf["cagr"]),
        },
    }

    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"[alpha_decomp] wrote {out_path}")

    # Concise console summary
    print()
    print("=" * 70)
    print("ALPHA DECOMP SUMMARY")
    print("=" * 70)
    print(f"Window:           {results['window']['date_start']} -> "
          f"{results['window']['date_end']}  ({results['window']['n_days']} days)")
    print(f"Alpha (annual):   {alpha_annualised*100:+.2f}%   "
          f"t-stat (NW5): {fit['t_alpha_nw']:+.2f}")
    print(f"Beta to SPY:      {beta:+.3f}   R^2: {fit['r2']:.3f}")
    print(f"Info Ratio:       {info_ratio:+.2f}   "
          f"Sharpe(resid): {sharpe_residuals:+.2f}")
    print(f"Leader Sharpe:    {leader_perf['sharpe']:+.2f}   "
          f"CAGR: {leader_perf['cagr']*100:+.2f}%   "
          f"MaxDD: {leader_perf['max_dd']*100:.1f}%")
    print(f"SPY    Sharpe:    {spy_perf['sharpe']:+.2f}   "
          f"CAGR: {spy_perf['cagr']*100:+.2f}%   "
          f"MaxDD: {spy_perf['max_dd']*100:.1f}%")
    print(f"Excess Sharpe:    {excess['sharpe_excess']:+.2f}   "
          f"Excess CAGR: {excess['cagr_excess']*100:+.2f}%")
    print(f"Mean Exposure:    {mean_exposure_all:.2f} (all)   "
          f"{mean_exposure_when_in:.2f} (when deployed)   "
          f"in-mkt: {pct_in_market*100:.1f}%")
    print(f"Sharpe/Exposure:  {sharpe_per_unit_exposure_all:+.2f} (all)   "
          f"{sharpe_per_unit_exposure_in:+.2f} (when deployed)")
    print("=" * 70)

    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leader alpha decomp vs SPY")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--nw-lags", type=int, default=5)
    return p.parse_args()


def main():
    args = _parse_args()
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / f"output/macro_picker/leader_alpha_decomp_{ts}"
    else:
        out_dir = Path(args.out)
    run(out_dir, nw_lags=args.nw_lags)


if __name__ == "__main__":
    main()
