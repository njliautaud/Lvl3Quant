#!/usr/bin/env python3
"""
BPS Combined Improvements — Expanded Universe + Drawdown Trigger
=================================================================
Tests the COMBINATION of:
1. Expanded universe (HC #660) — more diverse tickers
2. Drawdown trigger (HC #662 R2) — halt new positions after 3-day -3% drawdown

Also tests a new idea: sector concentration limits (no more than N% from one sector)
to reduce sector-clustering risk.
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))
from higher_returns_study import (
    load_data, bs_price, bs_delta, strike_from_delta, trade_cost,
    compute_metrics, COST_PER_CONTRACT
)

OUTPUT = ROOT / "output" / "bps_combined_improvements"
OUTPUT.mkdir(parents=True, exist_ok=True)


def run_bps_with_dd_trigger(prices, iv, macro, fund, universe, earnings,
                             spread_width=10.0, starting_cash=100_000.0,
                             put_delta=0.30, dte_target=7,
                             profit_take=0.65, margin_cap=0.30,
                             max_concurrent=60, vix_gate=35.0,
                             per_name_pct=0.05,
                             # Drawdown trigger params
                             dd_lookback=3, dd_threshold=-0.03,
                             dd_reduction=0.75,
                             # Sector concentration limit
                             sector_max_pct=0.25,  # max 25% in one sector
                             label="BPS Combined"):
    """BPS with drawdown trigger and optional sector limits."""
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

    # Earnings
    earnings_set = {}
    for _, row in earnings.iterrows():
        tk = row["ticker"]
        ed = pd.Timestamp(row["earnings_date"])
        earnings_set.setdefault(tk, set()).add(ed)

    # SPY SMA50
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
    dd_trigger_active = 0  # days remaining in drawdown halt

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

                ledger.append({"date": dt, "ticker": tk, "sector": sector_of.get(tk, "Unknown"),
                              "kind": "BPS_expire", "pnl": realized})
                to_remove.append(tk)
            else:
                short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
                long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
                spread_val = (short_val - long_val) * 100 * pos["contracts"]
                initial_credit = pos["net_credit"]
                current_cost_to_close = spread_val + trade_cost(short_val, pos["contracts"]) + trade_cost(long_val, pos["contracts"])
                captured = (initial_credit - current_cost_to_close) / max(initial_credit, 1e-6)
                if captured >= profit_take:
                    realized = initial_credit - current_cost_to_close
                    cash -= current_cost_to_close
                    ledger.append({"date": dt, "ticker": tk, "sector": sector_of.get(tk, "Unknown"),
                                  "kind": "BPS_pt", "pnl": realized})
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
            short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
            long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
            equity -= (short_val - long_val) * 100 * pos["contracts"]

        equity_curve.append({"date": dt, "equity": equity})

        # ── DRAWDOWN TRIGGER CHECK ──
        if dd_lookback > 0 and len(equity_curve) >= dd_lookback + 1:
            lookback_equity = equity_curve[-(dd_lookback+1)]["equity"]
            trailing_ret = (equity - lookback_equity) / max(lookback_equity, 1)
            if trailing_ret < dd_threshold:
                dd_trigger_active = dd_lookback  # halt for N days

        if dd_trigger_active > 0:
            dd_trigger_active -= 1
            continue  # skip opening new positions

        # ── Standard gates ──
        if not np.isnan(vix) and vix > vix_gate:
            continue
        if dt in spy_sma50:
            spy_close, spy_sma = spy_sma50[dt]
            if spy_sma > 0 and spy_close < spy_sma:
                continue

        if len(positions) >= max_concurrent:
            continue

        # Effective margin cap (reduced during drawdown recovery)
        eff_margin_cap = margin_cap

        current_margin = sum(
            (p["short_strike"] - p["long_strike"]) * 100 * p["contracts"]
            for p in positions.values()
        )
        if current_margin >= eff_margin_cap * equity:
            continue

        # ── Sector concentration check ──
        sector_margin = {}
        for tk, pos in positions.items():
            sec = sector_of.get(tk, "Unknown")
            m_pos = (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"]
            sector_margin[sec] = sector_margin.get(sec, 0) + m_pos

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

            # Sector concentration check
            if sector_max_pct is not None:
                sec = sector_of.get(tk, "Unknown")
                sec_alloc = sector_margin.get(sec, 0)
                if sec_alloc / max(equity, 1) >= sector_max_pct:
                    continue

            candidates.append((tk, S, sigma, iv_rk))

        candidates.sort(key=lambda r: r[3], reverse=True)

        slots = min(max_concurrent - len(positions), max(1, max_concurrent // 5))
        remaining_margin = eff_margin_cap * equity - current_margin

        for tk, S, sigma, iv_rk in candidates[:slots]:
            T = dte_target / 365.0
            K_short = strike_from_delta(S, T, sigma, put_delta, kind="put")
            K_long = K_short - spread_width
            if K_long <= 0 or K_short <= 0:
                continue

            prem_short = bs_price(S, K_short, T, sigma, kind="put")
            prem_long = bs_price(S, K_long, T, sigma, kind="put")
            net_prem_per_share = prem_short - prem_long
            if net_prem_per_share <= 0.05:
                continue

            margin_per_contract = spread_width * 100
            max_alloc = per_name_pct * equity
            n_contracts = max(1, int(max_alloc // margin_per_contract))
            if margin_per_contract * n_contracts > remaining_margin:
                n_contracts = max(1, int(remaining_margin // margin_per_contract))
            if n_contracts < 1:
                continue

            net_credit = net_prem_per_share * 100 * n_contracts
            open_costs = trade_cost(prem_short, n_contracts) + trade_cost(prem_long, n_contracts)
            net_credit -= open_costs
            if net_credit <= 0:
                continue

            cash += net_credit
            positions[tk] = {
                "short_strike": K_short,
                "long_strike": K_long,
                "contracts": n_contracts,
                "net_credit": net_credit,
                "open_date": dt,
                "expiry": dt + pd.Timedelta(days=dte_target),
                "open_sigma": sigma,
            }
            remaining_margin -= margin_per_contract * n_contracts

            # Update sector margin tracking
            sec = sector_of.get(tk, "Unknown")
            sector_margin[sec] = sector_margin.get(sec, 0) + margin_per_contract * n_contracts

            if len(positions) >= max_concurrent:
                break

    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty:
        return {"label": label, "error": "no equity curve"}

    metrics = compute_metrics(eq_df, starting_cash, label)
    metrics["n_trades"] = len(ledger)

    return {
        "metrics": metrics,
        "equity_curve": eq_df,
        "ledger": pd.DataFrame(ledger) if ledger else pd.DataFrame(),
    }


def compute_regime_metrics(equity_df, macro):
    eq = equity_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date")
    eq["ret"] = eq["equity"].pct_change()
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    if "vix" in macro.columns:
        vix_by_date = macro.set_index("date")["vix"].to_dict()
    else:
        vix_by_date = {}
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
    print("BPS COMBINED IMPROVEMENTS STUDY")
    print("Expanded Universe + Drawdown Trigger + Sector Limits")
    print("=" * 60)

    prices, iv, macro, fund, universe, earnings = load_data()
    n_tickers = prices["ticker"].nunique()
    print(f"\nUniverse: {n_tickers} tickers")

    configs = [
        # (label, dd_trigger, sector_limit, margin)
        ("Baseline (no DD, no sector)", 0, None, 0.30),
        ("DD trigger only", -0.03, None, 0.30),
        ("Sector 25% only", 0, 0.25, 0.30),
        ("DD + Sector 25%", -0.03, 0.25, 0.30),
        ("DD + Sector 20%", -0.03, 0.20, 0.30),
        ("DD + Sector 25% + 20% margin", -0.03, 0.25, 0.20),
        ("DD + Sector 25% + 40% margin", -0.03, 0.25, 0.40),
    ]

    all_results = {}
    for label, dd_thresh, sec_max, margin in configs:
        print(f"\n=== {label} ===")
        result = run_bps_with_dd_trigger(
            prices, iv, macro, fund, universe, earnings,
            spread_width=10.0, dte_target=7, max_concurrent=60,
            margin_cap=margin, per_name_pct=0.05,
            dd_lookback=3, dd_threshold=dd_thresh if dd_thresh != 0 else -999,
            dd_reduction=0.75,
            sector_max_pct=sec_max,
            label=label,
        )
        m = result["metrics"]
        regime = compute_regime_metrics(result["equity_curve"], macro)
        print(f"  CAGR: {m.get('cagr_pct')}%  Sharpe: {m.get('sharpe')}  "
              f"Sortino: {m.get('sortino')}  MaxDD: {m.get('max_dd_pct')}%  "
              f"PF: {m.get('profit_factor')}")
        print(f"  Regime: Bull={regime['bull_sharpe']} | Correction={regime['correction_sharpe']} "
              f"| Bear={regime['bear_sharpe']} | Gap={regime['regime_gap']}")

        safe_label = label.replace(" ", "_").replace("/", "_").replace("%", "pct").replace("+", "and")
        result["equity_curve"].to_parquet(OUTPUT / f"eq_{safe_label}.parquet")

        all_results[label] = {
            "metrics": m,
            "regime": regime,
            "n_trades": m.get("n_trades", 0),
        }

    # ── Permutation test on best config ──
    print("\n=== Permutation Test (DD + Sector 25%) ===")
    real_sharpe = all_results["DD + Sector 25%"]["metrics"].get("sharpe", 0)
    perm_sharpes = []
    for i in range(15):
        iv_shuffled = iv.copy()
        for date_val in iv_shuffled["date"].unique():
            mask = iv_shuffled["date"] == date_val
            vals = iv_shuffled.loc[mask, "iv_rank"].values.copy()
            np.random.shuffle(vals)
            iv_shuffled.loc[mask, "iv_rank"] = vals
        perm_r = run_bps_with_dd_trigger(
            prices, iv_shuffled, macro, fund, universe, earnings,
            spread_width=10.0, dte_target=7, max_concurrent=60,
            margin_cap=0.30, per_name_pct=0.05,
            dd_lookback=3, dd_threshold=-0.03,
            dd_reduction=0.75, sector_max_pct=0.25,
            label=f"Perm {i+1}",
        )
        ps = perm_r["metrics"].get("sharpe", 0)
        perm_sharpes.append(ps)
        if (i+1) % 5 == 0:
            print(f"  Perm {i+1}/15: Sharpe={ps:.2f}")

    p_value = sum(1 for s in perm_sharpes if s >= real_sharpe) / len(perm_sharpes)
    perm_summary = {
        "real_sharpe": real_sharpe,
        "perm_sharpe_mean": round(float(np.mean(perm_sharpes)), 3),
        "perm_sharpe_std": round(float(np.std(perm_sharpes)), 3),
        "p_value": round(p_value, 4),
        "verdict": "PASS" if p_value < 0.05 else "FAIL",
    }
    print(f"  Permutation: p={p_value:.3f}, verdict={perm_summary['verdict']}")

    # ── Save ──
    summary = {
        "generated": pd.Timestamp.now().isoformat(),
        "universe_size": n_tickers,
        "configs": all_results,
        "permutation_test": perm_summary,
        "reference": {
            "original_230_bps10_weekly": {"sharpe": 4.09, "cagr": 154.0, "max_dd": -30.4},
            "expanded_no_trigger": {"sharpe": 4.48, "cagr": 212.77, "max_dd": -30.51},
        }
    }
    with open(OUTPUT / "combined_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed/60:.1f} minutes")
    print(f"{'='*60}")

    # Comparison table
    print(f"\n{'Config':<35} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Bear':>8}")
    print("-" * 85)
    for label, data in all_results.items():
        m = data["metrics"]
        r = data["regime"]
        print(f"{label:<35} {m.get('cagr_pct','?'):>7}% {m.get('sharpe','?'):>8} "
              f"{m.get('sortino','?'):>8} {m.get('max_dd_pct','?'):>7}% {r.get('bear_sharpe','?'):>8}")
    print(f"{'Original 230 (reference)':<35} {'154.0':>7}% {'4.09':>8} {'4.65':>8} {'-30.4':>7}% {'0.12':>8}")


if __name__ == "__main__":
    main()
