#!/usr/bin/env python3
"""
BPS IV Skew Sensitivity — How Much Does Flat-IV Assumption Matter?
==================================================================

HC #664 R4 gap: "real option chain data (not BS modeled)"

Current BPS backtest uses flat ATM IV from historical data. Real option markets have:
1. SKEW: OTM puts have higher IV than ATM (typically 2-8 vol points for 30-delta)
2. TERM STRUCTURE: Shorter DTE options have higher IV in normal markets
3. SURFACE DYNAMICS: Skew steepens in down markets, flattens in up markets

This matters for BPS because:
- Short leg (30-delta put) gets MORE premium in reality due to skew (good for us)
- Long leg (further OTM put) gets EVEN MORE premium due to deeper skew (bad — costs more)
- Net effect is empirical — this study quantifies it

We test BPS performance under:
1. Flat IV (current baseline)
2. Mild skew (+3 vol pts at 30-delta, +5 at 20-delta)
3. Moderate skew (+5 vol pts at 30-delta, +8 at 20-delta)
4. Steep skew (+8 vol pts at 30-delta, +12 at 20-delta, typical high-VIX)
5. Dynamic skew (skew scales with VIX: steeper when VIX > 25)
"""

import sys
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))
from higher_returns_study import (
    load_data, _Phi, _ndtri, strike_from_delta,
    compute_metrics, COST_PER_CONTRACT, SLIPPAGE_FRAC, SLIPPAGE_MIN
)

OUTPUT = ROOT / "output" / "bps_iv_skew_sensitivity"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ── IV Skew Model ──

def skew_adjusted_iv(atm_sigma, delta, skew_params):
    """
    Adjust ATM IV for skew based on delta.

    Skew model: IV(delta) = ATM_IV + skew_slope * (0.50 - abs(delta))
    For puts, delta is negative (e.g., -0.30), so abs(delta)=0.30
    Skew adds IV as we move further OTM (lower abs(delta)).

    skew_params: dict with 'slope' (vol pts per 0.10 delta step from ATM)
    """
    delta_dist = 0.50 - abs(delta)  # Distance from ATM (0 at ATM, 0.40 at 10-delta)
    skew_add = skew_params.get("slope", 0) * delta_dist / 0.10  # per 0.10 delta step
    return atm_sigma + skew_add


def bs_price_skewed(S, K, T, atm_sigma, skew_params, r=0.04, kind="put"):
    """BS price with skew-adjusted IV."""
    if T <= 0:
        if kind == "put":
            return max(K - S, 0.0)
        return max(S - K, 0.0)

    # Approximate delta for this strike to determine skew adjustment
    d1 = (math.log(S / K) + (r + 0.5 * atm_sigma**2) * T) / (atm_sigma * math.sqrt(T))
    approx_delta = _Phi(d1) - 1.0 if kind == "put" else _Phi(d1)

    # Get skew-adjusted IV
    adj_sigma = skew_adjusted_iv(atm_sigma, approx_delta, skew_params)
    adj_sigma = max(adj_sigma, 0.01)  # Floor at 1%

    # Reprice with adjusted IV
    d1 = (math.log(S / K) + (r + 0.5 * adj_sigma**2) * T) / (adj_sigma * math.sqrt(T))
    d2 = d1 - adj_sigma * math.sqrt(T)

    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
    return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)


def trade_cost_fn(premium, contracts):
    """Cost to open/close one leg."""
    slip = max(SLIPPAGE_MIN, SLIPPAGE_FRAC * premium) * 100 * contracts
    comm = COST_PER_CONTRACT * contracts
    return slip + comm


def run_bps_with_skew(prices, iv, macro, fund, universe, earnings,
                      skew_params, label,
                      starting_cash=100_000.0,
                      spread_width=10.0, put_delta=0.30,
                      dte_target=7, profit_take=0.50,
                      margin_cap=0.30, max_concurrent=30,
                      per_name_pct=0.025, vix_gate=35.0,
                      dd_lookback=3, dd_threshold=-0.05):
    """Run BPS backtest with skew-adjusted IV."""

    prices_df = prices.copy()
    iv_df = iv.copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()

    macro_by_date = macro.set_index("date").to_dict("index")
    sector_of = dict(zip(fund["ticker"], fund.get("sector", pd.Series(["Unknown"]*len(fund)))))

    earnings_set = {}
    for _, row in earnings.iterrows():
        tk = row["ticker"]
        ed = pd.Timestamp(row["earnings_date"])
        earnings_set.setdefault(tk, set()).add(ed)

    spy_sma50 = {}
    spy = prices_df[prices_df["ticker"] == "SPY"].sort_values("date")
    if len(spy) > 0:
        spy["sma50"] = spy["close"].rolling(50).mean()
        for _, row in spy.iterrows():
            spy_sma50[row["date"]] = (row["close"], row["sma50"] if pd.notna(row["sma50"]) else 0)

    all_dates = sorted(prices_df["date"].unique())

    cash = starting_cash
    positions = {}
    equity_curve = []
    ledger = []
    dd_trigger_active = 0

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        m = macro_by_date.get(dt, {})
        vix = m.get("vix", float("nan"))

        # Dynamic skew: scale with VIX
        active_skew = dict(skew_params)
        if skew_params.get("dynamic", False) and not np.isnan(vix):
            vix_scale = max(0.5, min(2.0, vix / 20.0))  # 1.0 at VIX=20
            active_skew["slope"] = skew_params["slope"] * vix_scale

        # ── Update positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            if T_days <= 0:
                # Expiry settlement
                short_itm = S < pos["short_strike"]
                long_itm = S < pos["long_strike"]
                close_cost = COST_PER_CONTRACT * 2 * pos["contracts"]

                if not short_itm:
                    realized = pos["net_credit"] - close_cost
                    cash -= close_cost
                elif short_itm and not long_itm:
                    loss = (pos["short_strike"] - S) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost
                else:
                    loss = (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost

                ledger.append({"date": dt, "ticker": tk, "pnl": realized})
                to_remove.append(tk)
            else:
                # Profit take with skew-adjusted pricing
                short_val = bs_price_skewed(S, pos["short_strike"], T, sigma_atm, active_skew, kind="put")
                long_val = bs_price_skewed(S, pos["long_strike"], T, sigma_atm, active_skew, kind="put")
                spread_val = (short_val - long_val) * 100 * pos["contracts"]
                initial_credit = pos["net_credit"]
                cost_close = spread_val + trade_cost_fn(short_val, pos["contracts"]) + trade_cost_fn(long_val, pos["contracts"])
                captured = (initial_credit - cost_close) / max(initial_credit, 1e-6)
                if captured >= profit_take:
                    realized = initial_credit - cost_close
                    cash -= cost_close
                    ledger.append({"date": dt, "ticker": tk, "pnl": realized})
                    to_remove.append(tk)

        for tk in to_remove:
            del positions[tk]

        # ── MTM equity ──
        equity = cash
        for tk, pos in positions.items():
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T = max((pos["expiry"] - dt).days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            short_val = bs_price_skewed(S, pos["short_strike"], T, sigma_atm, active_skew, kind="put")
            long_val = bs_price_skewed(S, pos["long_strike"], T, sigma_atm, active_skew, kind="put")
            equity -= (short_val - long_val) * 100 * pos["contracts"]

        equity_curve.append({"date": dt, "equity": equity})

        # ── Drawdown trigger ──
        if dd_lookback > 0 and len(equity_curve) >= dd_lookback + 1:
            lb_eq = equity_curve[-(dd_lookback + 1)]["equity"]
            trailing_ret = (equity - lb_eq) / max(lb_eq, 1)
            if trailing_ret < dd_threshold:
                dd_trigger_active = dd_lookback

        if dd_trigger_active > 0:
            dd_trigger_active -= 1
            continue

        # ── Gates ──
        if not np.isnan(vix) and vix > vix_gate:
            continue
        if dt in spy_sma50:
            spy_close, spy_sma = spy_sma50[dt]
            if spy_sma > 0 and spy_close < spy_sma:
                continue
        if len(positions) >= max_concurrent:
            continue

        current_margin = sum(
            (p["short_strike"] - p["long_strike"]) * 100 * p["contracts"]
            for p in positions.values()
        )
        if current_margin >= margin_cap * equity:
            continue

        # ── Select candidates ──
        candidates = []
        for tk in universe:
            if tk in positions or tk == "SPY":
                continue
            S = date_px.get(tk)
            if S is None or np.isnan(S) or S < 10 or S > 500:
                continue
            sigma = date_sigma.get(tk)
            if sigma is None or np.isnan(sigma) or sigma <= 0:
                continue

            if tk in earnings_set:
                e_dates = earnings_set[tk]
                check = pd.Timestamp(dt)
                expiry = check + pd.Timedelta(days=dte_target)
                if any(check - pd.Timedelta(days=2) <= pd.Timestamp(ed) <= expiry + pd.Timedelta(days=2)
                       for ed in e_dates):
                    continue

            candidates.append((tk, S, sigma))

        # Sort by IV rank (prefer high IV tickers)
        candidates.sort(key=lambda x: -x[2])

        slots = min(max_concurrent - len(positions), max(1, max_concurrent // 5))
        remaining_margin = margin_cap * equity - current_margin

        for tk, S, sigma in candidates[:slots]:
            T = dte_target / 365.0
            K_short = strike_from_delta(S, T, sigma, put_delta, kind="put")
            K_long = K_short - spread_width
            if K_long <= 0 or K_short <= 0:
                continue

            # Price with skew
            prem_short = bs_price_skewed(S, K_short, T, sigma, active_skew, kind="put")
            prem_long = bs_price_skewed(S, K_long, T, sigma, active_skew, kind="put")
            net_prem = prem_short - prem_long

            if net_prem <= 0.05:
                continue

            margin_per_ct = spread_width * 100
            max_alloc = per_name_pct * equity
            n_cts = max(1, int(max_alloc // margin_per_ct))
            if margin_per_ct * n_cts > remaining_margin:
                n_cts = max(1, int(remaining_margin // margin_per_ct))
            if n_cts < 1:
                continue

            net_credit = net_prem * 100 * n_cts
            open_costs = trade_cost_fn(prem_short, n_cts) + trade_cost_fn(prem_long, n_cts)
            net_credit -= open_costs
            if net_credit <= 0:
                continue

            cash += net_credit
            remaining_margin -= margin_per_ct * n_cts
            positions[tk] = {
                "short_strike": K_short,
                "long_strike": K_long,
                "contracts": n_cts,
                "net_credit": net_credit,
                "open_date": dt,
                "expiry": dt + pd.Timedelta(days=dte_target),
                "open_sigma": sigma,
            }

    # Compute metrics
    eq_df = pd.DataFrame(equity_curve)
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.set_index("date")

    daily_returns = eq_df["equity"].pct_change().dropna()
    n_days = len(daily_returns)
    years = n_days / 252

    total_return = eq_df["equity"].iloc[-1] / starting_cash - 1
    cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1

    ann_ret = daily_returns.mean() * 252
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg_vol = daily_returns[daily_returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / neg_vol if neg_vol > 0 else 0

    cum = (1 + daily_returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    trade_pnls = [t["pnl"] for t in ledger]
    n_trades = len(trade_pnls)
    n_wins = sum(1 for p in trade_pnls if p > 0)
    wr = n_wins / n_trades if n_trades > 0 else 0
    gross_win = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")

    return {
        "label": label,
        "skew_slope_vol_pts": skew_params.get("slope", 0) * 100,  # in vol points
        "dynamic_skew": skew_params.get("dynamic", False),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "cagr_pct": round(cagr * 100, 1),
        "max_dd_pct": round(max_dd * 100, 1),
        "win_rate_pct": round(wr * 100, 1),
        "profit_factor": round(pf, 2),
        "n_trades": n_trades,
        "final_equity": round(eq_df["equity"].iloc[-1], 2),
    }


def main():
    print("=" * 70)
    print("BPS IV SKEW SENSITIVITY STUDY")
    print("HC #664 R4 — How much does flat-IV assumption matter?")
    print("=" * 70)

    prices, iv, macro, fund, universe_data, earnings = load_data()

    cache_dir = ROOT / "wheel_strategy_v1" / "data" / "cache"
    tickers = set()
    for pf in ["prices.parquet", "prices_expanded.parquet", "prices_v3_expansion.parquet"]:
        path = cache_dir / pf
        if path.exists():
            df = pd.read_parquet(path)
            if "ticker" in df.columns:
                tickers |= set(df["ticker"].unique())
    tickers -= {"SPY", "^VIX", "VIX"}
    universe = sorted(tickers)
    print(f"Universe: {len(universe)} tickers")

    # Define skew scenarios
    # slope is in decimal (0.01 = 1 vol point per 0.10 delta step from ATM)
    skew_scenarios = [
        {"name": "Flat (baseline)", "params": {"slope": 0.0}},
        {"name": "Mild skew (+3 vol at 30d)", "params": {"slope": 0.015}},  # ~3 vol pts at 30-delta
        {"name": "Moderate skew (+5 vol at 30d)", "params": {"slope": 0.025}},  # ~5 vol pts
        {"name": "Steep skew (+8 vol at 30d)", "params": {"slope": 0.04}},  # ~8 vol pts (high VIX)
        {"name": "Dynamic skew (VIX-scaled)", "params": {"slope": 0.025, "dynamic": True}},
    ]

    results = {}
    for scenario in skew_scenarios:
        name = scenario["name"]
        params = scenario["params"]
        print(f"\n{'-'*50}")
        print(f"Testing: {name}")
        print(f"{'-'*50}")

        r = run_bps_with_skew(
            prices, iv, macro, fund, universe, earnings,
            skew_params=params, label=name,
        )
        print(f"  Sharpe={r['sharpe']}, CAGR={r['cagr_pct']}%, MaxDD={r['max_dd_pct']}%, WR={r['win_rate_pct']}%, PF={r['profit_factor']}, Trades={r['n_trades']}")
        results[name] = r

    # Also test conservative config with moderate skew
    print(f"\n{'-'*50}")
    print(f"Testing: Conservative + Moderate Skew")
    print(f"{'-'*50}")
    r = run_bps_with_skew(
        prices, iv, macro, fund, universe, earnings,
        skew_params={"slope": 0.025},
        label="Conservative + Moderate Skew",
        put_delta=0.25, spread_width=15.0,
        margin_cap=0.15, max_concurrent=15,
        per_name_pct=0.02, profit_take=0.40,
    )
    print(f"  Sharpe={r['sharpe']}, CAGR={r['cagr_pct']}%, MaxDD={r['max_dd_pct']}%, WR={r['win_rate_pct']}%, PF={r['profit_factor']}, Trades={r['n_trades']}")
    results["Conservative + Moderate Skew"] = r

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — IV SKEW SENSITIVITY")
    print("=" * 70)
    print(f"\n{'Config':<40} {'Sharpe':>7} {'CAGR%':>8} {'MaxDD%':>8} {'WR%':>6} {'PF':>6} {'Trades':>7}")
    print("-" * 85)
    for name, r in results.items():
        print(f"{r['label']:<40} {r['sharpe']:>7.2f} {r['cagr_pct']:>7.1f}% {r['max_dd_pct']:>7.1f}% {r['win_rate_pct']:>5.1f}% {r['profit_factor']:>5.2f} {r['n_trades']:>7}")

    # Impact analysis
    baseline = results.get("Flat (baseline)", {})
    if baseline:
        print("\n" + "=" * 70)
        print("SKEW IMPACT vs FLAT BASELINE")
        print("=" * 70)
        b_sharpe = baseline["sharpe"]
        for name, r in results.items():
            if name == "Flat (baseline)":
                continue
            delta_sharpe = r["sharpe"] - b_sharpe
            pct_change = (delta_sharpe / abs(b_sharpe)) * 100 if b_sharpe != 0 else 0
            print(f"  {name:<40} Sharpe Δ={delta_sharpe:+.2f} ({pct_change:+.0f}%)")

    # Save
    with open(OUTPUT / "skew_sensitivity_results.json", "w") as f:
        json.dump({
            "generated": pd.Timestamp.now().isoformat(),
            "description": "Impact of IV skew on BPS performance. Skew adds IV to OTM puts (higher premium for both short and long legs). Net effect depends on spread width.",
            "skew_model": "IV(delta) = ATM_IV + slope * (0.50 - abs(delta)) / 0.10. Slope in decimal (0.01 = 1 vol pt per 10-delta step).",
            "results": results,
        }, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT}/skew_sensitivity_results.json")
    return results


if __name__ == "__main__":
    main()
