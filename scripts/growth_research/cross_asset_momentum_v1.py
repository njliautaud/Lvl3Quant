#!/usr/bin/env python3
"""
Cross-Asset Time-Series Momentum v1
=====================================
GENUINELY NEW — only superficially touched (CTA Trend Sharpe 0.93 was equity-only).

True CTA/systematic macro: trend following SIMULTANEOUSLY across equities,
bonds, commodities, currencies, and real estate using ETFs.

Variants:
  A: Equal-weight, 12-1 month momentum
  B: Inverse-vol weight, 12-1 month momentum
  C: Inverse-vol, multi-speed (avg of 1m, 3m, 12m)
  D: With trend filter (SMA 200d)
  E: Multi-speed + trend filter
  F: Long/short

Capital: $10,000 for portfolio management track
"""
import json, sys, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs); sys.stdout.flush()

sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.adversarial_validator import validate_trades

BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "cross_asset_momentum_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "SPY": {"class": "equity"}, "QQQ": {"class": "equity"},
    "IWM": {"class": "equity"}, "EFA": {"class": "equity"}, "EEM": {"class": "equity"},
    "TLT": {"class": "bond"}, "IEF": {"class": "bond"},
    "TIP": {"class": "bond"}, "LQD": {"class": "bond"}, "HYG": {"class": "bond"},
    "GLD": {"class": "commodity"}, "SLV": {"class": "commodity"},
    "DBA": {"class": "commodity"}, "USO": {"class": "commodity"},
    "UUP": {"class": "currency"}, "FXE": {"class": "currency"},
    "VNQ": {"class": "real_estate"}, "VNQI": {"class": "real_estate"},
}
CAP = 10000.0

def fetch_data():
    import yfinance as yf
    tickers = list(ASSETS.keys())
    fprint(f"Fetching data for {len(tickers)} cross-asset ETFs...")
    data = yf.download(tickers, period="10y", auto_adjust=True, progress=False)
    prices = data["Close"].dropna(axis=1, how="all")
    fprint(f"  Got {len(prices)} days, {len(prices.columns)} assets")
    return prices

def compute_momentum(prices, lookback_months=12, skip_months=1):
    lookback_days = lookback_months * 21
    skip_days = skip_months * 21
    if len(prices) < lookback_days + skip_days:
        return pd.Series(0, index=prices.columns)
    recent = prices.iloc[-skip_days - 1] if skip_days > 0 else prices.iloc[-1]
    past = prices.iloc[-lookback_days]
    return (recent / past) - 1.0

def compute_volatility(returns, lookback=63):
    return returns.iloc[-lookback:].std() * np.sqrt(252)

def run_variant(prices, variant_name, momentum_windows=None, weighting="equal",
                use_trend_filter=False, long_short=False):
    fprint(f"\n{'='*60}")
    fprint(f"Variant {variant_name}")
    if momentum_windows is None: momentum_windows = [12]
    returns = prices.pct_change().dropna()
    monthly_dates = returns.resample("M").last().index
    start_month = 12
    if start_month >= len(monthly_dates):
        fprint("  Not enough data!"); return {"variant": variant_name, "sharpe": 0, "trades": 0, "passed": False}

    portfolio_returns = []; trade_records = []
    for m_idx in range(start_month, len(monthly_dates)):
        month_end = monthly_dates[m_idx]
        hist_prices = prices[prices.index <= month_end]
        hist_returns = returns[returns.index <= month_end]
        if len(hist_returns) < 252: continue
        available = [c for c in prices.columns if hist_prices[c].notna().iloc[-252:].sum() > 200]
        if len(available) < 5: continue

        signals = pd.Series(0.0, index=available)
        for window in momentum_windows:
            signals += compute_momentum(hist_prices[available], lookback_months=window, skip_months=1) / len(momentum_windows)

        if use_trend_filter:
            sma_200 = hist_prices[available].iloc[-200:].mean()
            above_sma = hist_prices[available].iloc[-1] > sma_200
            signals = signals * above_sma.astype(float)

        if long_short: weights = np.sign(signals)
        else: weights = (signals > 0).astype(float)

        if weighting == "inverse_vol" and weights.sum() != 0:
            vols = pd.Series(index=available, dtype=float)
            for ticker in available:
                vols[ticker] = max(compute_volatility(hist_returns[ticker]), 0.01)
            inv_vol = 1.0 / vols
            weights = weights * inv_vol
            if weights.abs().sum() > 0: weights = weights / weights.abs().sum()
        elif weights.sum() > 0: weights = weights / weights.abs().sum()

        if m_idx + 1 < len(monthly_dates):
            next_month_end = monthly_dates[m_idx + 1]
            next_mask = (returns.index > month_end) & (returns.index <= next_month_end)
            next_returns = returns[next_mask][available]
            if len(next_returns) > 0:
                daily_port_ret = (next_returns * weights[available]).sum(axis=1)
                portfolio_returns.extend(daily_port_ret.values.tolist())
                monthly_ret = (1 + daily_port_ret).prod() - 1
                trade_records.append({
                    "entry_date": month_end.strftime("%Y-%m-%d"), "exit_date": next_month_end.strftime("%Y-%m-%d"),
                    "pnl": monthly_ret * CAP, "pnl_pct": monthly_ret * 100, "direction": "long",
                    "n_assets": int((weights != 0).sum()),
                })

    if len(portfolio_returns) == 0:
        fprint("  No returns!"); return {"variant": variant_name, "sharpe": 0, "trades": 0, "passed": False}

    port_ret = np.array(portfolio_returns)
    annual_sharpe = np.mean(port_ret) / np.std(port_ret) * np.sqrt(252) if np.std(port_ret) > 0 else 0
    downside = port_ret[port_ret < 0]
    sortino = np.mean(port_ret) / np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) > 0 else annual_sharpe
    total_ret = np.prod(1 + port_ret) - 1
    n_years = len(port_ret) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    eq_curve = np.cumprod(1 + port_ret)
    peak = np.maximum.accumulate(eq_curve)
    max_dd = np.min((eq_curve - peak) / peak)
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    monthly_pnls = [t["pnl"] for t in trade_records]
    wins = [p for p in monthly_pnls if p > 0]
    losses = [p for p in monthly_pnls if p <= 0]
    wr = len(wins) / len(monthly_pnls) * 100 if monthly_pnls else 0
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")
    final_equity = CAP * (1 + total_ret)

    fprint(f"  Sharpe: {annual_sharpe:.3f} | Sortino: {sortino:.3f} | CAGR: {cagr:.1%}")
    fprint(f"  MaxDD: {max_dd:.1%} | Calmar: {calmar:.2f} | WR: {wr:.1f}% | PF: {pf:.2f}")
    fprint(f"  Final: ${final_equity:,.0f} ({n_years:.1f}yr)")

    gates_passed = 0
    try:
        val_result = validate_trades(trade_records, CAP)
        gates_passed = val_result.get("gates_passed", 0)
        fprint(f"  Gates: {gates_passed}/5")
    except Exception as e: fprint(f"  Validation error: {e}")

    return {"variant": variant_name, "sharpe": round(annual_sharpe, 3), "sortino": round(sortino, 3),
            "cagr": round(cagr * 100, 1), "total_return": round(total_ret * 100, 1),
            "max_dd": round(max_dd * 100, 1), "calmar": round(calmar, 2), "wr": round(wr, 1),
            "pf": round(pf, 2), "trades": len(trade_records), "final_equity": round(final_equity, 0),
            "n_years": round(n_years, 1), "gates_passed": gates_passed}

def main():
    t0 = time.time()
    prices = fetch_data()
    results = []
    results.append(run_variant(prices, "A_EqualWeight_12m", momentum_windows=[12], weighting="equal"))
    results.append(run_variant(prices, "B_InvVol_12m", momentum_windows=[12], weighting="inverse_vol"))
    results.append(run_variant(prices, "C_MultiSpeed", momentum_windows=[1, 3, 12], weighting="inverse_vol"))
    results.append(run_variant(prices, "D_TrendFilter", momentum_windows=[12], weighting="inverse_vol", use_trend_filter=True))
    results.append(run_variant(prices, "E_MultiSpeed_Trend", momentum_windows=[1, 3, 12], weighting="inverse_vol", use_trend_filter=True))
    results.append(run_variant(prices, "F_LongShort", momentum_windows=[12], weighting="inverse_vol", long_short=True))

    elapsed = time.time() - t0
    fprint(f"\n{'='*60}")
    fprint(f"CROSS-ASSET MOMENTUM v1 — COMPLETE ({elapsed:.0f}s)")
    fprint(f"\n{'Variant':<25} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>7} {'MDD':>7} {'WR':>6} {'PF':>6} {'Gates':>6}")
    fprint("-" * 80)
    for r in results:
        fprint(f"{r['variant']:<25} {r['sharpe']:>8.3f} {r.get('sortino',0):>8.3f} {r['cagr']:>6.1f}% {r['max_dd']:>6.1f}% {r['wr']:>5.1f}% {r['pf']:>5.2f} {r['gates_passed']:>5}/5")

    if "SPY" in prices.columns:
        spy_ret = prices["SPY"].pct_change().dropna()
        spy_sharpe = spy_ret.mean() / spy_ret.std() * np.sqrt(252)
        fprint(f"\nSPY benchmark: Sharpe {spy_sharpe:.3f}")

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"\nSaved to {OUTPUT_DIR}/results.json")

if __name__ == "__main__":
    main()
