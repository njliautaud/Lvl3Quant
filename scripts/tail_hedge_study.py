#!/usr/bin/env python3
"""
Tail Hedge Study for Wheel Strategies
=======================================
HC #662 R2: Stress-test worst-case drawdowns.

Both CSP and BPS lose badly in crashes (Sharpe -6 when VIX>35).
The March 2026 -22.6% single-day crash dominated the MaxDD.

Question: Would a simple tail hedge improve risk-adjusted returns?

Approach: Simulate buying SPY put protection when VIX crosses
trigger thresholds. Cost the hedge realistically (BS pricing).
Measure impact on Sharpe/Calmar/MaxDD.

Hedging strategies tested:
  A) Constant allocation: always spend X% of NAV on 1-month OTM puts
  B) VIX-triggered: buy protection only when VIX > threshold
  C) Trailing DD trigger: buy protection when trailing 5d return < -3%
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, "/home/jupiter/Lvl3Quant/output/wheel_higher_returns_study")
from higher_returns_study import run_baseline_csp, load_data

OUTPUT = "/home/jupiter/Lvl3Quant/output/tail_hedge_study"
os.makedirs(OUTPUT, exist_ok=True)


def bs_put_price(S, K, T, sigma, r=0.04):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def compute_metrics(eq_series, starting_cash=100_000.0):
    rets = eq_series.pct_change().dropna()
    total_years = len(rets) / 252
    total_ret = (eq_series.iloc[-1] / starting_cash) - 1
    cagr = (1 + total_ret) ** (1/max(total_years, 0.01)) - 1
    sharpe = rets.mean() / max(rets.std(), 1e-9) * np.sqrt(252)
    downside = rets[rets < 0].std()
    sortino = rets.mean() / max(downside, 1e-9) * np.sqrt(252) if downside > 0 else 0
    peak = eq_series.cummax()
    max_dd = ((eq_series - peak) / peak).min()
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else 0
    worst_day = rets.min()
    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr_pct": round(cagr*100, 2),
        "max_dd_pct": round(max_dd*100, 2),
        "calmar": round(calmar, 3),
        "worst_day_pct": round(worst_day*100, 3),
    }


def simulate_hedge(base_eq, spy_data, vix_data,
                   hedge_type="constant",
                   cost_pct=0.005,     # Annual cost as % of NAV
                   vix_trigger=25,     # For VIX-triggered
                   dd_trigger=-0.03,   # For DD-triggered
                   dd_lookback=5,
                   put_otm_pct=0.05,   # 5% OTM puts
                   put_dte=21):        # 21-day puts
    """
    Simulate adding tail hedge to existing equity curve.

    Logic: On hedge days, we own SPY puts worth X% of NAV.
    The puts pay off when SPY drops significantly.
    Cost is the premium spent. Payoff is max(K-SPY, 0) at expiry.

    Simplification: model as daily cost + daily payoff from
    a rolling portfolio of OTM puts.
    """
    dates = base_eq["date"].values
    base_equity = base_eq["equity"].values.copy()
    n = len(dates)

    # Align SPY and VIX — normalize date keys to date objects
    spy_by_date = {pd.Timestamp(k).date(): v for k, v in
                   spy_data.set_index("date")["close"].to_dict().items()}
    vix_by_date = {pd.Timestamp(k).date(): v for k, v in
                   vix_data.set_index("date")["vix"].to_dict().items()}

    hedged_equity = base_equity.copy()
    total_hedge_cost = 0
    total_hedge_payoff = 0
    hedge_days = 0

    for i in range(1, n):
        dt = dates[i]
        dt_prev = dates[i-1]

        dt_key = pd.Timestamp(dt).date()
        dt_prev_key = pd.Timestamp(dt_prev).date()
        spy_today = spy_by_date.get(dt_key)
        spy_yesterday = spy_by_date.get(dt_prev_key)
        vix_today = vix_by_date.get(dt_key)

        if spy_today is None or spy_yesterday is None or vix_today is None:
            continue

        # Determine if we should be hedged today
        should_hedge = False

        if hedge_type == "constant":
            should_hedge = True
        elif hedge_type == "vix_triggered":
            should_hedge = (vix_today > vix_trigger)
        elif hedge_type == "dd_triggered":
            if i >= dd_lookback:
                trailing_ret = (hedged_equity[i-1] - hedged_equity[i-dd_lookback]) / hedged_equity[i-dd_lookback]
                should_hedge = (trailing_ret < dd_trigger)

        if should_hedge:
            hedge_days += 1
            nav = hedged_equity[i-1]

            # Daily cost of rolling put protection
            # Annual cost_pct → daily cost
            daily_cost = nav * cost_pct / 252

            # Put payoff: if SPY drops, the put makes money
            # Model put delta as roughly -0.10 for 5% OTM
            spy_return = (spy_today - spy_yesterday) / spy_yesterday

            # If SPY drops more than OTM distance, put kicks in
            # Approximate: payoff = max(0, -spy_return - otm_pct/put_dte*252) * nav * leverage
            # Simpler model: put notional = nav * (cost_pct * 252 / put_premium_pct)

            # Use BS to price the put
            sigma_spy = vix_today / 100  # VIX ≈ annualized vol
            T = put_dte / 252
            K = spy_yesterday * (1 - put_otm_pct)  # OTM strike

            put_price_yesterday = bs_put_price(spy_yesterday, K, T, sigma_spy)
            put_price_today = bs_put_price(spy_today, K, T - 1/252, sigma_spy)

            if put_price_yesterday > 0:
                # How many puts can we buy with our hedge budget?
                n_puts = daily_cost * put_dte / max(put_price_yesterday * 100, 1)
                put_pnl = (put_price_today - put_price_yesterday) * 100 * n_puts
            else:
                put_pnl = 0

            total_hedge_cost += daily_cost
            total_hedge_payoff += max(put_pnl, -daily_cost)  # Floor at cost

            hedged_equity[i] = base_equity[i] - daily_cost + max(put_pnl, 0)
        else:
            hedged_equity[i] = base_equity[i]

    return pd.Series(hedged_equity), {
        "hedge_days": hedge_days,
        "hedge_pct_days": round(hedge_days / n * 100, 1),
        "total_cost": round(total_hedge_cost, 2),
        "total_payoff": round(total_hedge_payoff, 2),
        "net_cost": round(total_hedge_cost - total_hedge_payoff, 2),
    }


def main():
    print("Loading data...")
    prices, iv, macro, fund, universe, earnings = load_data()

    # Run V5 baseline
    print("\nRunning V5 CSP baseline...")
    csp = run_baseline_csp(prices, iv, macro, fund, universe, earnings,
                           dte_override=10, label="V5 Baseline")
    base_eq = csp["equity_curve"]
    base_metrics = compute_metrics(base_eq["equity"])
    print(f"  Baseline: Sharpe={base_metrics['sharpe']}, MaxDD={base_metrics['max_dd_pct']}%, "
          f"Worst day={base_metrics['worst_day_pct']}%")

    # Get SPY and VIX data
    # SPY not in the 230-ticker universe — load from separate parquet
    spy_data = pd.read_parquet("/home/jupiter/Lvl3Quant/data/spy_daily.parquet")
    spy_data["date"] = pd.to_datetime(spy_data["date"])
    vix_data = macro[["date", "vix"]].dropna().copy()

    results = {"baseline": base_metrics}

    # ── Test hedge configurations ──
    configs = [
        # (name, type, cost_pct, vix_trigger, dd_trigger, dd_lookback)
        ("constant_0.5%",    "constant",      0.005, None, None, None),
        ("constant_1.0%",    "constant",      0.010, None, None, None),
        ("constant_2.0%",    "constant",      0.020, None, None, None),
        ("vix_20_0.5%",      "vix_triggered", 0.005, 20,   None, None),
        ("vix_25_0.5%",      "vix_triggered", 0.005, 25,   None, None),
        ("vix_25_1.0%",      "vix_triggered", 0.010, 25,   None, None),
        ("vix_30_1.0%",      "vix_triggered", 0.010, 30,   None, None),
        ("dd_3%_5d_0.5%",    "dd_triggered",  0.005, None, -0.03, 5),
        ("dd_5%_3d_0.5%",    "dd_triggered",  0.005, None, -0.05, 3),
        ("dd_5%_3d_1.0%",    "dd_triggered",  0.010, None, -0.05, 3),
    ]

    print(f"\n{'Config':25s} {'Sharpe':>7s} {'MaxDD':>8s} {'Calmar':>7s} {'Worst':>8s} {'Hedge%':>7s} {'NetCost':>10s}")
    print("-" * 80)
    print(f"{'baseline':25s} {base_metrics['sharpe']:7.3f} {base_metrics['max_dd_pct']:7.2f}% "
          f"{base_metrics['calmar']:7.3f} {base_metrics['worst_day_pct']:7.3f}% {'N/A':>7s} {'$0':>10s}")

    for name, htype, cost, vix_t, dd_t, dd_lb in configs:
        kwargs = {"hedge_type": htype, "cost_pct": cost}
        if vix_t is not None:
            kwargs["vix_trigger"] = vix_t
        if dd_t is not None:
            kwargs["dd_trigger"] = dd_t
        if dd_lb is not None:
            kwargs["dd_lookback"] = dd_lb

        hedged_eq, hedge_info = simulate_hedge(base_eq, spy_data, vix_data, **kwargs)
        metrics = compute_metrics(hedged_eq)

        results[name] = {**metrics, **hedge_info}

        print(f"{name:25s} {metrics['sharpe']:7.3f} {metrics['max_dd_pct']:7.2f}% "
              f"{metrics['calmar']:7.3f} {metrics['worst_day_pct']:7.3f}% "
              f"{hedge_info['hedge_pct_days']:6.1f}% ${hedge_info['net_cost']:>9,.0f}")

    # ── Find best config ──
    print(f"\n{'='*60}")
    # Best = highest Calmar improvement over baseline
    best_name = None
    best_calmar_improvement = 0

    for name, data in results.items():
        if name == "baseline":
            continue
        if data["calmar"] > base_metrics["calmar"] + best_calmar_improvement:
            best_calmar_improvement = data["calmar"] - base_metrics["calmar"]
            best_name = name

    if best_name and best_calmar_improvement > 0:
        best = results[best_name]
        print(f"BEST HEDGE: {best_name}")
        print(f"  Calmar: {base_metrics['calmar']} → {best['calmar']} "
              f"(+{best_calmar_improvement:.3f})")
        print(f"  MaxDD:  {base_metrics['max_dd_pct']}% → {best['max_dd_pct']}%")
        print(f"  Sharpe: {base_metrics['sharpe']} → {best['sharpe']}")
        print(f"  Worst day: {base_metrics['worst_day_pct']}% → {best['worst_day_pct']}%")
        print(f"  Cost: hedge active {best['hedge_pct_days']}% of days, net ${best['net_cost']:,.0f}")
    else:
        print("NO HEDGE IMPROVES CALMAR — tail hedging is not worth the cost for this strategy.")
        print("The bear gate + DD trigger already handle crash periods adequately.")

    # Save
    with open(f"{OUTPUT}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUTPUT}/results.json")


if __name__ == "__main__":
    main()
