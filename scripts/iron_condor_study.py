#!/usr/bin/env python3
"""
Iron Condor Study — Market-Neutral Premium Capture (HC #662 R4)
================================================================
BPS is structurally bullish (bear Sharpe = -0.28). Iron condors sell
BOTH sides: bull put spread (downside protection) + bear call spread
(upside protection). Should be more market-neutral.

Tests whether combining both legs improves bear-regime performance
while maintaining overall edge.

Also tests: ratio variants (heavier put side vs balanced) and
regime-conditional switching (IC only in low/normal VIX, BPS-only
in high VIX when puts are richer).
"""
import sys
import json
import time
import math
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))
from higher_returns_study import (
    load_data, bs_price, bs_delta, strike_from_delta, trade_cost,
    compute_metrics, COST_PER_CONTRACT, _Phi
)

OUTPUT = ROOT / "output" / "iron_condor_study"
OUTPUT.mkdir(parents=True, exist_ok=True)


def run_iron_condor(prices, iv, macro, fund, universe, earnings,
                     spread_width=10.0, starting_cash=100_000.0,
                     put_delta=0.25, call_delta=0.25,
                     dte_target=7, profit_take=0.50,
                     margin_cap=0.30, max_concurrent=50,
                     vix_gate=40.0, per_name_pct=0.04,
                     put_weight=1.0, call_weight=1.0,  # ratio: 1.0/1.0 = balanced
                     regime_switch=False,  # if True, skip call side when VIX > 25
                     label="Iron Condor"):
    """
    Iron condor: sell OTM put spread + sell OTM call spread on same ticker.
    Max risk = spread_width (only one side can be ITM at expiry).
    Margin = spread_width * 100 * contracts (same as BPS — only one side at risk).
    """
    prices_df = prices.copy()
    iv_df = iv.copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    iv_rank_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()
        iv_rank_by_date[d] = g.set_index("ticker")["iv_rank"].to_dict()

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
    positions = {}  # ticker -> {put_short, put_long, call_short, call_long, contracts, ...}
    equity_curve = []
    ledger = []

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})
        m = macro_by_date.get(dt, {})
        vix = m.get("vix", float("nan"))

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
                close_cost = COST_PER_CONTRACT * 4 * pos["contracts"]  # 4 legs

                # Put side
                put_short_itm = S < pos["put_short"]
                put_long_itm = S < pos["put_long"]
                put_loss = 0
                if put_short_itm and not put_long_itm:
                    put_loss = (pos["put_short"] - S) * 100 * pos["contracts"]
                elif put_short_itm and put_long_itm:
                    put_loss = (pos["put_short"] - pos["put_long"]) * 100 * pos["contracts"]

                # Call side
                call_short_itm = S > pos["call_short"]
                call_long_itm = S > pos["call_long"]
                call_loss = 0
                if call_short_itm and not call_long_itm:
                    call_loss = (S - pos["call_short"]) * 100 * pos["contracts"]
                elif call_short_itm and call_long_itm:
                    call_loss = (pos["call_long"] - pos["call_short"]) * 100 * pos["contracts"]

                total_loss = put_loss + call_loss
                realized = pos["net_credit"] - total_loss - close_cost
                cash -= total_loss + close_cost

                ledger.append({"date": dt, "ticker": tk, "kind": "IC_expire",
                              "pnl": realized, "put_loss": put_loss, "call_loss": call_loss})
                to_remove.append(tk)
            else:
                # Profit take: mark all 4 legs
                put_short_val = bs_price(S, pos["put_short"], T, sigma_atm, kind="put")
                put_long_val = bs_price(S, pos["put_long"], T, sigma_atm, kind="put")
                call_short_val = bs_price(S, pos["call_short"], T, sigma_atm, kind="call")
                call_long_val = bs_price(S, pos["call_long"], T, sigma_atm, kind="call")

                spread_val = ((put_short_val - put_long_val) + (call_short_val - call_long_val)) * 100 * pos["contracts"]
                close_costs = sum(trade_cost(v, pos["contracts"]) for v in
                                 [put_short_val, put_long_val, call_short_val, call_long_val])
                cost_to_close = spread_val + close_costs

                captured = (pos["net_credit"] - cost_to_close) / max(pos["net_credit"], 1e-6)
                if captured >= profit_take:
                    realized = pos["net_credit"] - cost_to_close
                    cash -= cost_to_close
                    ledger.append({"date": dt, "ticker": tk, "kind": "IC_pt", "pnl": realized,
                                  "put_loss": 0, "call_loss": 0})
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
            put_short_val = bs_price(S, pos["put_short"], T, sigma_atm, kind="put")
            put_long_val = bs_price(S, pos["put_long"], T, sigma_atm, kind="put")
            call_short_val = bs_price(S, pos["call_short"], T, sigma_atm, kind="call")
            call_long_val = bs_price(S, pos["call_long"], T, sigma_atm, kind="call")
            equity -= ((put_short_val - put_long_val) + (call_short_val - call_long_val)) * 100 * pos["contracts"]

        equity_curve.append({"date": dt, "equity": equity})

        # ── Gates ──
        if not np.isnan(vix) and vix > vix_gate:
            continue
        # No bear gate for iron condors — they're market neutral
        if len(positions) >= max_concurrent:
            continue

        current_margin = sum(
            spread_width * 100 * p["contracts"]
            for p in positions.values()
        )
        if current_margin >= margin_cap * equity:
            continue

        # ── Select candidates ──
        candidates = []
        for tk, S in date_px.items():
            if tk == "__date__" or tk in positions:
                continue
            if S is None or np.isnan(S) or S < 10 or S > 500:
                continue
            sigma = date_sigma.get(tk)
            if sigma is None or np.isnan(sigma) or sigma <= 0:
                continue
            iv_rk = date_iv_rank.get(tk, 0.5)

            expiry_date = dt + pd.Timedelta(days=dte_target)
            tk_earnings = earnings_set.get(tk, set())
            near_earnings = any(abs((ed - expiry_date).days) <= 2 for ed in tk_earnings)
            if near_earnings:
                continue

            candidates.append((tk, S, sigma, iv_rk))

        candidates.sort(key=lambda r: r[3], reverse=True)

        slots = min(max_concurrent - len(positions), max(1, max_concurrent // 5))
        remaining_margin = margin_cap * equity - current_margin

        # Regime switch: skip call side in high VIX (puts are richer, calls risky)
        sell_calls = True
        if regime_switch and not np.isnan(vix) and vix > 25:
            sell_calls = False

        for tk, S, sigma, iv_rk in candidates[:slots]:
            T = dte_target / 365.0

            # Put side
            K_put_short = strike_from_delta(S, T, sigma, put_delta, kind="put")
            K_put_long = K_put_short - spread_width
            if K_put_long <= 0:
                continue

            prem_put_short = bs_price(S, K_put_short, T, sigma, kind="put")
            prem_put_long = bs_price(S, K_put_long, T, sigma, kind="put")
            put_credit = (prem_put_short - prem_put_long) * put_weight

            # Call side
            if sell_calls:
                K_call_short = strike_from_delta(S, T, sigma, call_delta, kind="call")
                K_call_long = K_call_short + spread_width
                prem_call_short = bs_price(S, K_call_short, T, sigma, kind="call")
                prem_call_long = bs_price(S, K_call_long, T, sigma, kind="call")
                call_credit = (prem_call_short - prem_call_long) * call_weight
            else:
                K_call_short = S * 2  # far OTM, won't trigger
                K_call_long = K_call_short + spread_width
                call_credit = 0

            total_credit_per_share = put_credit + call_credit
            if total_credit_per_share <= 0.05:
                continue

            # Margin: max(put spread, call spread) = spread_width per contract
            margin_per_contract = spread_width * 100
            max_alloc = per_name_pct * equity
            n_contracts = max(1, int(max_alloc // margin_per_contract))
            if margin_per_contract * n_contracts > remaining_margin:
                n_contracts = max(1, int(remaining_margin // margin_per_contract))
            if n_contracts < 1:
                continue

            net_credit = total_credit_per_share * 100 * n_contracts
            n_legs = 4 if sell_calls else 2
            open_costs = COST_PER_CONTRACT * n_legs * n_contracts  # simplified
            if sell_calls:
                open_costs += (trade_cost(prem_put_short, n_contracts) +
                              trade_cost(prem_put_long, n_contracts) +
                              trade_cost(prem_call_short, n_contracts) +
                              trade_cost(prem_call_long, n_contracts))
            else:
                open_costs += (trade_cost(prem_put_short, n_contracts) +
                              trade_cost(prem_put_long, n_contracts))
            net_credit -= open_costs
            if net_credit <= 0:
                continue

            cash += net_credit
            positions[tk] = {
                "put_short": K_put_short,
                "put_long": K_put_long,
                "call_short": K_call_short,
                "call_long": K_call_long,
                "contracts": n_contracts,
                "net_credit": net_credit,
                "open_date": dt,
                "expiry": dt + pd.Timedelta(days=dte_target),
                "open_sigma": sigma,
                "has_call": sell_calls,
            }
            remaining_margin -= margin_per_contract * n_contracts

            if len(positions) >= max_concurrent:
                break

    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty:
        return {"label": label, "error": "no equity curve", "metrics": {}}

    metrics = compute_metrics(eq_df, starting_cash, label)
    metrics["n_trades"] = len(ledger)

    # Loss attribution
    ledger_df = pd.DataFrame(ledger) if ledger else pd.DataFrame()
    if not ledger_df.empty and "put_loss" in ledger_df.columns:
        metrics["total_put_losses"] = round(ledger_df["put_loss"].sum(), 0)
        metrics["total_call_losses"] = round(ledger_df["call_loss"].sum(), 0)

    return {"metrics": metrics, "equity_curve": eq_df, "ledger": ledger_df}


def compute_regime_metrics(equity_df, macro):
    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date")
    eq["ret"] = eq["equity"].pct_change()
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    vix_by_date = macro.set_index("date")["vix"].to_dict() if "vix" in macro.columns else {}
    results = {}
    for regime, vix_range in [("bull", (0, 18)), ("correction", (18, 25)), ("bear", (25, 200))]:
        regime_dates = {d for d, v in vix_by_date.items()
                       if not np.isnan(v) and vix_range[0] <= v < vix_range[1]}
        regime_rets = eq[eq["date"].isin(regime_dates)]["ret"].dropna()
        if len(regime_rets) > 10:
            sharpe = regime_rets.mean() / max(regime_rets.std(), 1e-10) * np.sqrt(252)
        else:
            sharpe = float("nan")
        results[f"{regime}_sharpe"] = round(sharpe, 3)
        results[f"{regime}_days"] = len(regime_rets)
    bull_s = results.get("bull_sharpe", 0)
    bear_s = results.get("bear_sharpe", 0)
    denom = max(abs(bull_s), abs(bear_s), 0.01)
    results["regime_gap"] = round(abs(bull_s - bear_s) / denom, 3)
    return results


def main():
    t0 = time.time()
    print("=" * 60)
    print("IRON CONDOR STUDY — Market-Neutral Premium Capture")
    print("HC #662 R4: Better risk controls for bear regime")
    print("=" * 60)

    prices, iv, macro, fund, universe, earnings = load_data()
    print(f"\nUniverse: {prices['ticker'].nunique()} tickers")

    configs = [
        # (label, put_delta, call_delta, put_wt, call_wt, regime_switch, margin, dte, pt)
        ("IC Balanced 25d", 0.25, 0.25, 1.0, 1.0, False, 0.30, 7, 0.50),
        ("IC Balanced 20d", 0.20, 0.20, 1.0, 1.0, False, 0.30, 7, 0.50),
        ("IC Balanced 30d", 0.30, 0.30, 1.0, 1.0, False, 0.30, 7, 0.50),
        ("IC Put-Heavy (2:1)", 0.30, 0.15, 1.0, 0.5, False, 0.30, 7, 0.50),
        ("IC Regime-Switch", 0.25, 0.25, 1.0, 1.0, True, 0.30, 7, 0.50),
        ("IC 65% PT", 0.25, 0.25, 1.0, 1.0, False, 0.30, 7, 0.65),
        ("IC 14-DTE", 0.25, 0.25, 1.0, 1.0, False, 0.30, 14, 0.50),
        ("BPS-only reference", 0.30, 0.01, 1.0, 0.0, False, 0.30, 7, 0.65),
    ]

    all_results = {}
    for label, pd_, cd, pw, cw, rs, margin, dte, pt in configs:
        print(f"\n=== {label} ===")
        result = run_iron_condor(
            prices, iv, macro, fund, universe, earnings,
            spread_width=10.0, put_delta=pd_, call_delta=cd,
            put_weight=pw, call_weight=cw,
            regime_switch=rs, margin_cap=margin,
            dte_target=dte, profit_take=pt,
            max_concurrent=50, per_name_pct=0.04,
            label=label,
        )
        m = result["metrics"]
        regime = compute_regime_metrics(result["equity_curve"], macro)
        print(f"  CAGR: {m.get('cagr_pct')}%  Sharpe: {m.get('sharpe')}  "
              f"Sortino: {m.get('sortino')}  MaxDD: {m.get('max_dd_pct')}%")
        print(f"  Regime: Bull={regime['bull_sharpe']} | Corr={regime.get('correction_sharpe','?')} "
              f"| Bear={regime['bear_sharpe']} | Gap={regime['regime_gap']}")
        if "total_put_losses" in m:
            print(f"  Put losses: ${m['total_put_losses']:,.0f}  Call losses: ${m['total_call_losses']:,.0f}")

        all_results[label] = {"metrics": m, "regime": regime}

    # Permutation test on best iron condor config
    best_label = max(all_results.keys(), key=lambda k: all_results[k]["metrics"].get("sharpe", 0))
    best_sharpe = all_results[best_label]["metrics"]["sharpe"]
    print(f"\n=== Permutation Test on '{best_label}' (Sharpe={best_sharpe}) ===")

    # Find the config params for best
    best_cfg = None
    for label, pd_, cd, pw, cw, rs, margin, dte, pt in configs:
        if label == best_label:
            best_cfg = (pd_, cd, pw, cw, rs, margin, dte, pt)
            break

    perm_sharpes = []
    for i in range(15):
        iv_shuf = iv.copy()
        for d in iv_shuf["date"].unique():
            mask = iv_shuf["date"] == d
            vals = iv_shuf.loc[mask, "iv_rank"].values.copy()
            np.random.shuffle(vals)
            iv_shuf.loc[mask, "iv_rank"] = vals
        perm_r = run_iron_condor(
            prices, iv_shuf, macro, fund, universe, earnings,
            spread_width=10.0, put_delta=best_cfg[0], call_delta=best_cfg[1],
            put_weight=best_cfg[2], call_weight=best_cfg[3],
            regime_switch=best_cfg[4], margin_cap=best_cfg[5],
            dte_target=best_cfg[6], profit_take=best_cfg[7],
            max_concurrent=50, per_name_pct=0.04,
            label=f"Perm {i+1}",
        )
        ps = perm_r["metrics"].get("sharpe", 0)
        perm_sharpes.append(ps)
        if (i+1) % 5 == 0:
            print(f"  Perm {i+1}/15: Sharpe={ps:.2f}")

    p_val = sum(1 for s in perm_sharpes if s >= best_sharpe) / len(perm_sharpes)
    perm = {
        "config": best_label,
        "real_sharpe": best_sharpe,
        "perm_mean": round(float(np.mean(perm_sharpes)), 3),
        "p_value": round(p_val, 4),
        "verdict": "PASS" if p_val < 0.05 else "FAIL",
    }
    print(f"\n  Permutation: p={p_val:.3f}, verdict={perm['verdict']}")

    # Save
    summary = {
        "generated": pd.Timestamp.now().isoformat(),
        "configs": all_results,
        "permutation_test": perm,
        "reference": {"bps_7dte_sharpe": 4.48, "bps_7dte_bear_sharpe": -0.275},
    }
    with open(OUTPUT / "iron_condor_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} min")
    print(f"{'='*60}")

    # Comparison table
    print(f"\n{'Config':<25} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Bull':>8} {'Bear':>8} {'Gap':>6}")
    print("-" * 95)
    for label, data in all_results.items():
        m = data["metrics"]
        r = data["regime"]
        print(f"{label:<25} {m.get('cagr_pct','?'):>7}% {m.get('sharpe','?'):>8} "
              f"{m.get('sortino','?'):>8} {m.get('max_dd_pct','?'):>7}% "
              f"{r.get('bull_sharpe','?'):>8} {r.get('bear_sharpe','?'):>8} {r.get('regime_gap','?'):>6}")


if __name__ == "__main__":
    main()
