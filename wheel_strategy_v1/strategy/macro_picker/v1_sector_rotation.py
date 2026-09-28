"""
v1_sector_rotation.py — HC #558 R3 macro stock-picker v1.

Strategy (intentionally simple, honest first):

  Long basket = top-3 sectors by 60-day relative-strength vs SPY × positive
  dollar-volume-flow z-score (proxy for fund inflows). Within each chosen
  sector, hold equal-weight the 5 highest-momentum names from our 70-ticker
  universe that belong to that sector. Rebalance monthly.

  Optional short basket = bottom-2 sectors by same criteria, top 5 worst-
  momentum names each. Off by default in v1.

  Macro gate = stand down (100% cash) when regime_overlay.risk_off=True OR
  vix > 35. Re-enter on next rebalance after gate clears.

Backtest:
  - 2020-01-01 → 2025-12-31, starting $100k.
  - Honest costs: 0.5 bps slippage one-way + 0.5 bps commission per
    one-way trade (≈ $5 per $100k turnover). No borrow cost in v1 (long-only).
  - Benchmarks: SPY 1.0×, SPY 1.5× margin (DFF + 1.5% broker spread),
    SPY 2.0× margin.
  - Regime-bucket diagnostics (HC #557 R2): green / red / flat day classifier
    on SPY close-to-close; stratified Sharpe + monthly return per bucket.

Verdict logic (HC #557 + HC #558 R5):
  - WIN if Sharpe_picker > Sharpe_marginSPY_1.5x AND CAGR_picker > CAGR_marginSPY_1.5x.
  - HALF-WIN if Sharpe wins but CAGR loses (acceptable per HC #557, but flagged).
  - LOSE if both lose.

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python3 -m strategy.macro_picker.v1_sector_rotation
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1")
CACHE = ROOT / "data" / "cache"
OUT_DIR = ROOT / "results" / "macro_picker_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2020-01-01")
END = pd.Timestamp("2025-12-31")
STARTING_CASH = 100_000.0

# Costs — one-way, in bps of notional traded
SLIPPAGE_BPS = 0.5
COMMISSION_BPS = 0.5
ONE_WAY_COST_BPS = SLIPPAGE_BPS + COMMISSION_BPS

TRADING_DAYS = 252
ANN = np.sqrt(TRADING_DAYS)

# Sector map from our universe sectors → SPDR ETFs
# Universe uses Yahoo sector taxonomy.
UNIVERSE_TO_ETF: Dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Communication Services": "XLC",
    # Utilities, Real Estate, Materials — empty in our universe but mapped for completeness
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
}

# Rebalance config
LOOKBACK_RS = 60       # days for sector relative-strength
LOOKBACK_FLOW = 20     # days for dollar-volume flow z-score
LOOKBACK_NAME_MOM = 60 # days for stock momentum within sector
TOP_N_SECTORS = 3
NAMES_PER_SECTOR = 5
VIX_GATE = 35.0


# -------------------- Helpers --------------------

def load_inputs() -> dict:
    prices = pd.read_parquet(CACHE / "prices.parquet")
    universe = pd.read_parquet(CACHE / "universe.parquet")
    macro = pd.read_parquet(CACHE / "macro.parquet")
    regime = pd.read_parquet(CACHE / "regime_overlay.parquet")
    etfs = pd.read_parquet(CACHE / "sector_etfs.parquet")
    return {
        "prices": prices, "universe": universe, "macro": macro,
        "regime": regime, "etfs": etfs,
    }


def make_pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """ticker × date → wide on `col`."""
    p = df.pivot(index="date", columns="ticker", values=col)
    p.index = pd.to_datetime(p.index)
    return p.sort_index()


def zscore(s: pd.Series, win: int) -> pd.Series:
    mu = s.rolling(win).mean()
    sd = s.rolling(win).std()
    return (s - mu) / sd.where(sd > 0)


def risk_metrics(daily_ret: pd.Series, equity: pd.Series) -> dict:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return {"cagr": float("nan"), "sharpe": float("nan"), "sortino": float("nan"), "max_dd": float("nan")}
    yrs = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else float("nan")
    std = daily_ret.std()
    sharpe = daily_ret.mean() / std * ANN if std > 0 else float("nan")
    dn = daily_ret[daily_ret < 0].std()
    sortino = daily_ret.mean() / dn * ANN if dn and dn > 0 else float("nan")
    rm = equity.cummax()
    dd = (equity - rm) / rm
    return {"cagr": float(cagr), "sharpe": float(sharpe),
            "sortino": float(sortino), "max_dd": float(dd.min())}


def regime_classify(spy_close: pd.Series, thresh_sigma: float = 0.5) -> pd.Series:
    ret = spy_close.pct_change().dropna()
    sd = ret.std()
    cut = thresh_sigma * sd
    out = pd.Series("flat", index=ret.index, dtype=object)
    out[ret > cut] = "green"
    out[ret < -cut] = "red"
    return out


def stratified_sharpe(daily_ret: pd.Series, regime: pd.Series) -> Dict[str, float]:
    out = {}
    for b in ("green", "red", "flat"):
        days = regime[regime == b].index
        sub = daily_ret.reindex(days).dropna()
        if len(sub) < 5 or sub.std() == 0:
            out[b] = float("nan")
        else:
            out[b] = float(sub.mean() / sub.std() * ANN)
    return out


def monthly_return_by_regime(daily_ret: pd.Series, regime: pd.Series) -> Dict[str, float]:
    df = pd.DataFrame({"ret": daily_ret, "regime": regime.reindex(daily_ret.index, method="ffill")})
    out = {}
    for b in ("green", "red", "flat"):
        sub = df[df["regime"] == b]["ret"]
        if len(sub) == 0:
            out[b] = 0.0
            continue
        monthly = (1 + sub).resample("M").apply(lambda x: x.prod() - 1)
        out[b] = float(monthly.mean()) if len(monthly) else 0.0
    return out


# -------------------- Strategy core --------------------

def select_sectors(etfs_close: pd.DataFrame, etfs_dvol: pd.DataFrame,
                   spy_close: pd.Series, date: pd.Timestamp,
                   sector_etf_list: List[str]) -> List[str]:
    """Return top-N sectors by relative-strength × positive flow z."""
    if date not in etfs_close.index:
        # use most recent available
        try:
            date = etfs_close.index[etfs_close.index.searchsorted(date) - 1]
        except Exception:
            return []
    # Relative-strength vs SPY: ETF_ret_60d - SPY_ret_60d
    win = LOOKBACK_RS
    if etfs_close.index.get_loc(date) < win:
        return []
    rs = {}
    flow = {}
    for tk in sector_etf_list:
        if tk not in etfs_close.columns:
            continue
        ret_etf = etfs_close[tk].loc[:date].pct_change(win).iloc[-1]
        ret_spy = spy_close.loc[:date].pct_change(win).iloc[-1]
        if pd.isna(ret_etf) or pd.isna(ret_spy):
            continue
        rs[tk] = ret_etf - ret_spy
        # flow z-score
        dvol_series = etfs_dvol[tk].loc[:date]
        z = zscore(dvol_series, LOOKBACK_FLOW).iloc[-1] if len(dvol_series) > LOOKBACK_FLOW else 0.0
        flow[tk] = float(z) if not pd.isna(z) else 0.0

    # Combined score: rs is the primary, flow is a soft tilt
    score = {tk: rs[tk] + 0.10 * max(0.0, flow.get(tk, 0.0)) for tk in rs}
    ordered = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    return [tk for tk, _ in ordered[:TOP_N_SECTORS]]


def select_names_in_sector(prices_close: pd.DataFrame, universe: pd.DataFrame,
                           etf_ticker: str, date: pd.Timestamp,
                           etf_to_sector: Dict[str, str]) -> List[str]:
    """Top-NAMES_PER_SECTOR names in that sector by 60-day momentum."""
    sector_name = etf_to_sector.get(etf_ticker)
    if sector_name is None:
        return []
    sector_tickers = universe[universe["sector"] == sector_name]["ticker"].tolist()
    sector_tickers = [t for t in sector_tickers if t in prices_close.columns]
    if not sector_tickers:
        return []
    win = LOOKBACK_NAME_MOM
    if prices_close.index.get_loc(date) < win:
        return []
    px = prices_close.loc[:date, sector_tickers]
    mom = px.pct_change(win).iloc[-1].dropna()
    if mom.empty:
        return []
    ordered = mom.sort_values(ascending=False)
    return ordered.head(NAMES_PER_SECTOR).index.tolist()


def macro_gate_open(date: pd.Timestamp, macro_df: pd.DataFrame,
                    regime_df: pd.DataFrame) -> bool:
    """True = trade. False = 100% cash."""
    m = macro_df.set_index("date")
    r = regime_df.set_index("date")
    m_row = m.loc[:date].iloc[-1] if len(m.loc[:date]) else None
    r_row = r.loc[:date].iloc[-1] if len(r.loc[:date]) else None
    if r_row is not None and bool(r_row.get("risk_off", False)):
        return False
    if m_row is not None and float(m_row.get("vix", 0.0)) > VIX_GATE:
        return False
    return True


def backtest_sector_rotation(data: dict) -> dict:
    prices = data["prices"]
    universe = data["universe"]
    macro = data["macro"]
    regime = data["regime"]
    etfs = data["etfs"]

    prices_close = make_pivot(prices, "close")
    prices_close = prices_close[(prices_close.index >= START) & (prices_close.index <= END)]
    etfs_close = make_pivot(etfs, "close")
    etfs_dvol = make_pivot(etfs, "dollar_volume")
    etfs_close = etfs_close[(etfs_close.index >= START - pd.Timedelta(days=120)) & (etfs_close.index <= END)]
    etfs_dvol = etfs_dvol[(etfs_dvol.index >= START - pd.Timedelta(days=120)) & (etfs_dvol.index <= END)]

    # Filter sector ETFs that actually have data
    sector_etf_list = [e for e in UNIVERSE_TO_ETF.values() if e in etfs_close.columns]
    etf_to_sector = {v: k for k, v in UNIVERSE_TO_ETF.items()}

    if "SPY" not in etfs_close.columns:
        raise SystemExit("SPY missing from sector_etfs.parquet — re-run ingest_sector_etfs.py")
    spy_close = etfs_close["SPY"]

    # Trading dates = union of price + ETF dates within window
    dates = prices_close.index.intersection(etfs_close.index)
    dates = dates[(dates >= START) & (dates <= END)]

    # Monthly rebalance dates: first trading day of each month
    rebal_dates = pd.Series(dates).groupby(pd.Series(dates).dt.to_period("M")).first().tolist()

    weights = {tk: 0.0 for tk in prices_close.columns}
    equity = [STARTING_CASH]
    eq_dates = [dates[0]]
    holdings_log = []
    last_rebal_date = None

    for i, d in enumerate(dates[1:], start=1):
        d_prev = dates[i - 1]

        # Mark current weights forward using prev→d returns
        rets = (prices_close.loc[d] / prices_close.loc[d_prev] - 1.0).fillna(0.0)
        port_ret = sum(weights[tk] * rets[tk] for tk in weights if tk in rets.index)
        equity.append(equity[-1] * (1 + port_ret))
        eq_dates.append(d)

        # Rebalance?
        if d in rebal_dates and d != dates[0]:
            gate = macro_gate_open(d, macro, regime)
            if not gate:
                target = {tk: 0.0 for tk in weights}
                rationale = "macro_gate_closed"
            else:
                chosen_etfs = select_sectors(etfs_close, etfs_dvol, spy_close, d, sector_etf_list)
                chosen_names = []
                for etf in chosen_etfs:
                    chosen_names.extend(select_names_in_sector(
                        prices_close, universe, etf, d, etf_to_sector))
                if not chosen_names:
                    target = {tk: 0.0 for tk in weights}
                    rationale = "no_names_selected"
                else:
                    w = 1.0 / len(chosen_names)
                    target = {tk: 0.0 for tk in weights}
                    for tk in chosen_names:
                        target[tk] = w
                    rationale = f"long {len(chosen_names)} names in sectors {chosen_etfs}"

            # Apply turnover cost
            turnover = sum(abs(target[tk] - weights[tk]) for tk in weights)
            cost = turnover * (ONE_WAY_COST_BPS / 1e4)
            equity[-1] *= (1 - cost)
            weights = target
            last_rebal_date = d
            holdings_log.append({"date": d, "rationale": rationale,
                                 "holdings": [tk for tk, w in target.items() if w > 0]})

    eq_series = pd.Series(equity, index=pd.DatetimeIndex(eq_dates), name="equity")
    daily_ret = eq_series.pct_change().dropna()
    metrics = risk_metrics(daily_ret, eq_series)

    return {
        "equity": eq_series,
        "daily_ret": daily_ret,
        "metrics": metrics,
        "holdings_log": holdings_log,
        "spy_close": spy_close.loc[eq_series.index[0]:eq_series.index[-1]],
    }


# -------------------- SPY benchmarks --------------------

def spy_benchmarks(spy_close: pd.Series, macro_df: pd.DataFrame,
                   leverages=(1.0, 1.5, 2.0)) -> Dict[float, dict]:
    """Build levered SPY equity curves with broker-margin interest cost."""
    m = macro_df.set_index("date").sort_index()
    # short rate proxy: fed_funds if present in macro_extra, else 0.05 flat
    try:
        macx = pd.read_parquet(CACHE / "macro_extra.parquet").set_index("date").sort_index()
        rate = macx["fed_funds"].reindex(spy_close.index, method="ffill") / 100.0
        rate = rate.fillna(0.05)
    except Exception:
        rate = pd.Series(0.05, index=spy_close.index)
    BROKER_SPREAD = 0.015
    spy_ret = spy_close.pct_change().fillna(0.0)
    out = {}
    for lev in leverages:
        # Levered daily return: lev * spy_ret - (lev - 1) * (rate + spread) / 252
        borrow_cost = (lev - 1) * (rate + BROKER_SPREAD) / TRADING_DAYS
        lev_ret = lev * spy_ret - borrow_cost
        eq = (1 + lev_ret).cumprod() * STARTING_CASH
        eq.index = spy_close.index
        out[lev] = {
            "equity": eq, "daily_ret": lev_ret.iloc[1:],
            "metrics": risk_metrics(lev_ret.iloc[1:], eq),
        }
    return out


# -------------------- Verdict --------------------

def verdict(picker_metrics: dict, spy15_metrics: dict,
            regime_monthly: Dict[str, float],
            strat_sharpe: Dict[str, float]) -> str:
    reasons = []
    # HC #558 R5: picker must beat 1.5x SPY on BOTH Sharpe AND CAGR
    sh_win = picker_metrics["sharpe"] > spy15_metrics["sharpe"]
    cagr_win = picker_metrics["cagr"] > spy15_metrics["cagr"]
    if not sh_win:
        reasons.append(f"Sharpe {picker_metrics['sharpe']:.2f} <= margin-SPY 1.5x {spy15_metrics['sharpe']:.2f}")
    if not cagr_win:
        reasons.append(f"CAGR {picker_metrics['cagr']:.1%} <= margin-SPY 1.5x {spy15_metrics['cagr']:.1%}")
    # HC #557 R2 regime gate
    vals = [v for v in regime_monthly.values()]
    if min(vals) < 0:
        reasons.append(f"monthly return negative in some regime: {regime_monthly}")
    sh_vals = [v for v in strat_sharpe.values() if not np.isnan(v)]
    if len(sh_vals) >= 2:
        worst = min(sh_vals, key=abs)
        best = max(sh_vals, key=abs)
        if abs(best) > 0 and abs(worst) / abs(best) < 0.50:
            reasons.append(f"regime sharpe asymmetric worst/best={abs(worst)/abs(best):.2f} < 0.50")
    if not reasons:
        return "PICKER WINS"
    if sh_win and not cagr_win and not any("regime" in r or "monthly" in r for r in reasons):
        return "PICKER HALF-WIN (Sharpe beats margin-SPY but CAGR loses; acceptable per HC #557, flagged)"
    return "PICKER LOSES — reasons: " + "; ".join(reasons)


# -------------------- Main --------------------

def main():
    print("[picker] loading inputs")
    data = load_inputs()
    print(f"[picker] sector ETFs available: {sorted(data['etfs']['ticker'].unique())[:20]}...")

    print("[picker] running sector-rotation backtest 2020-01-01..2025-12-31")
    res = backtest_sector_rotation(data)
    print(f"[picker] metrics: {res['metrics']}")

    print("[picker] computing SPY 1.0×/1.5×/2.0× benchmarks")
    spy_b = spy_benchmarks(res["spy_close"], data["macro"])
    spy10 = spy_b[1.0]["metrics"]
    spy15 = spy_b[1.5]["metrics"]
    spy20 = spy_b[2.0]["metrics"]

    print("[picker] regime diagnostics")
    regime = regime_classify(res["spy_close"])
    monthly_by_regime = monthly_return_by_regime(res["daily_ret"], regime)
    strat_sh = stratified_sharpe(res["daily_ret"], regime)

    v = verdict(res["metrics"], spy15, monthly_by_regime, strat_sh)

    # Persist
    res["equity"].to_frame().to_parquet(OUT_DIR / "equity_picker_v1.parquet")
    pd.DataFrame(res["holdings_log"]).to_parquet(OUT_DIR / "holdings_log.parquet")
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({
            "picker": res["metrics"],
            "spy_1.0x": spy10, "spy_1.5x": spy15, "spy_2.0x": spy20,
            "monthly_return_by_regime": monthly_by_regime,
            "stratified_sharpe": strat_sh,
            "verdict": v,
        }, f, indent=2, default=str)

    report = f"""# Macro Picker v1 — Sector Rotation

**Verdict: {v}**

Window: 2020-01-01 → 2025-12-31, $100k starting. Monthly rebalance. Long-only.
Top-3 sectors by 60d relative-strength + flow z-score; 5 highest-momentum names per sector.
Macro gate: stand down if regime_overlay.risk_off OR VIX > 35.

## Headline

| Strategy | CAGR | Sharpe | Sortino | MaxDD |
|---|---|---|---|---|
| Picker v1 | {res['metrics']['cagr']:.1%} | {res['metrics']['sharpe']:.2f} | {res['metrics']['sortino']:.2f} | {res['metrics']['max_dd']:.1%} |
| SPY 1.0× | {spy10['cagr']:.1%} | {spy10['sharpe']:.2f} | {spy10['sortino']:.2f} | {spy10['max_dd']:.1%} |
| SPY 1.5× margin | {spy15['cagr']:.1%} | {spy15['sharpe']:.2f} | {spy15['sortino']:.2f} | {spy15['max_dd']:.1%} |
| SPY 2.0× margin | {spy20['cagr']:.1%} | {spy20['sharpe']:.2f} | {spy20['sortino']:.2f} | {spy20['max_dd']:.1%} |

## Regime Diagnostics (HC #557 R2)

Mean monthly return by regime:
- green months: {monthly_by_regime['green']:.2%}
- red months:   {monthly_by_regime['red']:.2%}
- flat months:  {monthly_by_regime['flat']:.2%}

Stratified Sharpe by regime:
- green: {strat_sh['green']:.2f}
- red:   {strat_sh['red']:.2f}
- flat:  {strat_sh['flat']:.2f}

## Notes
- Costs: {ONE_WAY_COST_BPS} bps one-way per trade (slippage + commission).
- No shorts in v1. No borrow cost.
- Top SaaS→AI/Semis rotation 2023-2024 should show up in holdings_log (look for XLK + SMH appearing).
"""
    (OUT_DIR / "report.md").write_text(report)
    print(f"\n[picker] {v}")
    print(f"[picker] Report: {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
