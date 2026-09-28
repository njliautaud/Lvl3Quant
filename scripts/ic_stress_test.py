#!/usr/bin/env python3
"""
Iron Condor Comprehensive Stress Test
=======================================
Runs the IC backtest across ALL regimes (2019-2026) with:
1. Multiple wing widths ($5, $7, $10, $15 fixed; 3%, 5%, 7%, 10% proportional)
2. VIX stratification (VIX<18, 18-25, 25-30, 30-40, 40+)
3. HC #428 R1 regime gap validation (green/red/flat day Sharpe)
4. SPY buy-and-hold comparison
5. Tail event analysis (COVID, Volmageddon proxies, tariff shocks)

Uses the existing iron_condor_study infrastructure.
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
    COST_PER_CONTRACT, _Phi
)

OUTPUT = ROOT / "output" / "ic_stress_test"
OUTPUT.mkdir(parents=True, exist_ok=True)


def compute_metrics_full(eq_df, starting_cash):
    """Compute comprehensive performance metrics."""
    eq = eq_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date").reset_index(drop=True)
    eq["ret"] = eq["equity"].pct_change()

    rets = eq["ret"].dropna()
    if len(rets) < 10:
        return {"error": "insufficient data"}

    years = len(rets) / 252
    total_ret = (eq["equity"].iloc[-1] / starting_cash) - 1
    cagr = (1 + total_ret) ** (1 / max(years, 0.1)) - 1

    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / max(ann_vol, 1e-10)

    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = ann_ret / max(downside, 1e-10)

    # Max drawdown
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Profit factor
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / max(losses, 1e-10)

    # Win rate (daily)
    wr = (rets > 0).mean()

    # Calmar
    calmar = cagr / max(abs(max_dd), 1e-10)

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr_pct": round(cagr * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "win_rate_pct": round(wr * 100, 1),
        "profit_factor": round(pf, 3),
        "total_return_pct": round(total_ret * 100, 2),
        "final_equity": round(eq["equity"].iloc[-1], 2),
        "years": round(years, 2),
        "n_days": len(rets),
    }


def run_ic_backtest(prices_df, iv_df, macro, fund, universe, earnings,
                    spread_width=10.0, starting_cash=100_000.0,
                    put_delta=0.25, call_delta=0.25,
                    dte_target=7, profit_take=0.65,
                    margin_cap=0.30, max_concurrent=50,
                    vix_gate=40.0, per_name_pct=0.04,
                    width_mode="fixed",  # "fixed" or "pct"
                    width_pct=0.05,      # used if width_mode=="pct"
                    stop_loss_mult=2.0,
                    label="IC"):
    """
    Run iron condor backtest. Supports fixed $ width or % of stock price.
    """
    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    iv_rank_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()
        iv_rank_by_date[d] = g.set_index("ticker")["iv_rank"].to_dict()

    macro_by_date = macro.set_index("date").to_dict("index")

    earnings_set = {}
    for _, row in earnings.iterrows():
        tk = row["ticker"]
        ed = pd.Timestamp(row["earnings_date"])
        earnings_set.setdefault(tk, set()).add(ed)

    all_dates = sorted(prices_df["date"].unique())

    cash = starting_cash
    positions = {}
    equity_curve = []
    ledger = []
    daily_vix = []

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})
        m = macro_by_date.get(dt, {})
        vix = m.get("vix", float("nan"))
        daily_vix.append({"date": dt, "vix": vix})

        # ── Update positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            sw = pos["spread_width"]

            if T_days <= 0:
                # Expiry settlement
                close_cost = COST_PER_CONTRACT * 4 * pos["contracts"]
                put_loss = 0
                if S < pos["put_short"] and S >= pos["put_long"]:
                    put_loss = (pos["put_short"] - S) * 100 * pos["contracts"]
                elif S < pos["put_long"]:
                    put_loss = sw * 100 * pos["contracts"]

                call_loss = 0
                if S > pos["call_short"] and S <= pos["call_long"]:
                    call_loss = (S - pos["call_short"]) * 100 * pos["contracts"]
                elif S > pos["call_long"]:
                    call_loss = sw * 100 * pos["contracts"]

                total_loss = put_loss + call_loss
                realized = pos["net_credit"] - total_loss - close_cost
                cash -= total_loss + close_cost

                side = "otm" if total_loss == 0 else ("put" if put_loss > 0 and call_loss == 0
                        else ("call" if call_loss > 0 and put_loss == 0 else "both"))
                ledger.append({"date": dt, "ticker": tk, "kind": f"expire_{side}",
                              "pnl": realized, "put_loss": put_loss, "call_loss": call_loss,
                              "vix": vix})
                to_remove.append(tk)
            else:
                # Profit take / stop loss
                put_short_val = bs_price(S, pos["put_short"], T, sigma_atm, kind="put")
                put_long_val = bs_price(S, pos["put_long"], T, sigma_atm, kind="put")
                call_short_val = bs_price(S, pos["call_short"], T, sigma_atm, kind="call")
                call_long_val = bs_price(S, pos["call_long"], T, sigma_atm, kind="call")

                spread_val = ((put_short_val - put_long_val) + (call_short_val - call_long_val)) * 100 * pos["contracts"]
                close_costs = sum(trade_cost(v, pos["contracts"]) for v in
                                 [put_short_val, put_long_val, call_short_val, call_long_val])
                cost_to_close = spread_val + close_costs

                if pos["net_credit"] > 0:
                    captured = (pos["net_credit"] - cost_to_close) / pos["net_credit"]

                    if captured >= profit_take:
                        realized = pos["net_credit"] - cost_to_close
                        cash -= cost_to_close
                        ledger.append({"date": dt, "ticker": tk, "kind": "pt",
                                      "pnl": realized, "put_loss": 0, "call_loss": 0, "vix": vix})
                        to_remove.append(tk)
                    elif captured <= -stop_loss_mult:
                        realized = pos["net_credit"] - cost_to_close
                        cash -= cost_to_close
                        ledger.append({"date": dt, "ticker": tk, "kind": "sl",
                                      "pnl": realized, "put_loss": 0, "call_loss": 0, "vix": vix})
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
        if len(positions) >= max_concurrent:
            continue

        current_margin = sum(
            p["spread_width"] * 100 * p["contracts"]
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

        for tk, S, sigma, iv_rk in candidates[:slots]:
            T = dte_target / 365.0

            # Determine wing width
            if width_mode == "pct":
                sw = round(S * width_pct, 0)
                sw = max(sw, 2.0)  # minimum $2 width
            else:
                sw = spread_width

            # Put side
            K_put_short = strike_from_delta(S, T, sigma, put_delta, kind="put")
            K_put_long = K_put_short - sw
            if K_put_long <= 0:
                continue

            prem_put_short = bs_price(S, K_put_short, T, sigma, kind="put")
            prem_put_long = bs_price(S, K_put_long, T, sigma, kind="put")
            put_credit = prem_put_short - prem_put_long

            # Call side
            K_call_short = strike_from_delta(S, T, sigma, call_delta, kind="call")
            K_call_long = K_call_short + sw

            # Minimum gap between short strikes
            min_gap = max(3.0, S * 0.04)
            if K_call_short - K_put_short < min_gap:
                continue

            prem_call_short = bs_price(S, K_call_short, T, sigma, kind="call")
            prem_call_long = bs_price(S, K_call_long, T, sigma, kind="call")
            call_credit = prem_call_short - prem_call_long

            total_credit_per_share = put_credit + call_credit
            if total_credit_per_share <= 0.05:
                continue

            margin_per_contract = sw * 100
            max_alloc = per_name_pct * equity
            n_contracts = max(1, int(max_alloc // margin_per_contract))
            if margin_per_contract * n_contracts > remaining_margin:
                n_contracts = max(1, int(remaining_margin // margin_per_contract))
            if n_contracts < 1:
                continue

            net_credit = total_credit_per_share * 100 * n_contracts
            open_costs = COST_PER_CONTRACT * 4 * n_contracts
            open_costs += (trade_cost(prem_put_short, n_contracts) +
                          trade_cost(prem_put_long, n_contracts) +
                          trade_cost(prem_call_short, n_contracts) +
                          trade_cost(prem_call_long, n_contracts))
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
                "spread_width": sw,
            }
            remaining_margin -= margin_per_contract * n_contracts

            if len(positions) >= max_concurrent:
                break

    eq_df = pd.DataFrame(equity_curve)
    ledger_df = pd.DataFrame(ledger) if ledger else pd.DataFrame()
    return eq_df, ledger_df


def compute_regime_stratified(eq_df, macro, prices_df):
    """
    HC #428 R1: Green/Red/Flat day classification based on SPY close-to-close.
    Green: SPY > +0.5 sigma, Red: SPY < -0.5 sigma, Flat: within +/- 0.5 sigma.
    """
    eq = eq_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date")
    eq["ret"] = eq["equity"].pct_change()

    # SPY daily returns
    spy = prices_df[prices_df["ticker"] == "SPY"][["date", "close"]].copy()
    spy = spy.sort_values("date")
    spy["spy_ret"] = spy["close"].pct_change()
    spy_sigma = spy["spy_ret"].std()

    spy["regime"] = "flat"
    spy.loc[spy["spy_ret"] > 0.5 * spy_sigma, "regime"] = "green"
    spy.loc[spy["spy_ret"] < -0.5 * spy_sigma, "regime"] = "red"

    spy_regime = spy.set_index("date")["regime"].to_dict()
    eq["regime"] = eq["date"].map(spy_regime)

    results = {}
    for regime in ["green", "red", "flat"]:
        r = eq[eq["regime"] == regime]["ret"].dropna()
        if len(r) > 10:
            sharpe = r.mean() / max(r.std(), 1e-10) * np.sqrt(252)
            sortino_d = r[r < 0].std() * np.sqrt(252)
            sortino = r.mean() * 252 / max(sortino_d, 1e-10)
        else:
            sharpe = float("nan")
            sortino = float("nan")
        results[f"{regime}_sharpe"] = round(sharpe, 3)
        results[f"{regime}_sortino"] = round(sortino, 3)
        results[f"{regime}_days"] = len(r)

    # HC #428 R1 gap: |green - red| / max(|green|, |red|)
    g_s = results.get("green_sharpe", 0)
    r_s = results.get("red_sharpe", 0)
    denom = max(abs(g_s), abs(r_s), 0.01)
    results["regime_gap"] = round(abs(g_s - r_s) / denom, 3)
    results["passes_r1"] = results["regime_gap"] <= 0.50

    return results


def compute_vix_stratified(eq_df, macro):
    """Stratify returns by VIX bucket."""
    eq = eq_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date")
    eq["ret"] = eq["equity"].pct_change()

    macro_c = macro.copy()
    macro_c["date"] = pd.to_datetime(macro_c["date"])
    vix_map = macro_c.set_index("date")["vix"].to_dict()
    eq["vix"] = eq["date"].map(vix_map)

    buckets = [
        ("VIX<18", 0, 18),
        ("VIX 18-25", 18, 25),
        ("VIX 25-30", 25, 30),
        ("VIX 30-40", 30, 40),
        ("VIX 40+", 40, 200),
    ]

    results = {}
    for name, lo, hi in buckets:
        mask = (eq["vix"] >= lo) & (eq["vix"] < hi)
        r = eq.loc[mask, "ret"].dropna()
        if len(r) > 5:
            sharpe = r.mean() / max(r.std(), 1e-10) * np.sqrt(252)
            avg_ret = r.mean() * 252
            max_dd_day = r.min()
        else:
            sharpe = float("nan")
            avg_ret = float("nan")
            max_dd_day = float("nan")
        results[name] = {
            "sharpe": round(sharpe, 3) if not np.isnan(sharpe) else None,
            "ann_ret_pct": round(avg_ret * 100, 2) if not np.isnan(avg_ret) else None,
            "worst_day_pct": round(max_dd_day * 100, 2) if not np.isnan(max_dd_day) else None,
            "n_days": int(mask.sum()),
        }

    return results


def compute_tail_events(eq_df, ledger_df):
    """Analyze behavior during known tail events."""
    eq = eq_df.copy()
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date").set_index("date")

    events = {
        "COVID_Crash": ("2020-02-20", "2020-03-23"),
        "COVID_Recovery": ("2020-03-24", "2020-06-08"),
        "2022_Rate_Hike_Start": ("2022-01-03", "2022-03-14"),
        "2022_Jun_Bear": ("2022-06-01", "2022-06-30"),
        "2022_Q4_Bear": ("2022-09-15", "2022-12-30"),
        "2023_SVB": ("2023-03-08", "2023-03-17"),
        "2024_Aug_VIX_Spike": ("2024-07-31", "2024-08-09"),
        "2025_Tariff_Shock": ("2025-04-02", "2025-04-11"),
    }

    results = {}
    for name, (start, end) in events.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        window = eq[(eq.index >= start_dt) & (eq.index <= end_dt)]
        if len(window) < 2:
            results[name] = {"return_pct": None, "max_dd_pct": None, "days": 0}
            continue

        ret = (window["equity"].iloc[-1] / window["equity"].iloc[0]) - 1
        cum = window["equity"] / window["equity"].iloc[0]
        peak = cum.cummax()
        dd = ((cum - peak) / peak).min()

        results[name] = {
            "return_pct": round(ret * 100, 2),
            "max_dd_pct": round(dd * 100, 2),
            "days": len(window),
        }

    # Analyze trades where both wings blown
    if not ledger_df.empty and "kind" in ledger_df.columns:
        both_blown = ledger_df[ledger_df["kind"] == "expire_both"]
        results["both_wings_blown"] = {
            "count": len(both_blown),
            "total_pnl": round(both_blown["pnl"].sum(), 2) if len(both_blown) > 0 else 0,
            "avg_loss": round(both_blown["pnl"].mean(), 2) if len(both_blown) > 0 else 0,
        }

    return results


def spy_buyhold(prices_df, starting_cash=100_000.0):
    """SPY buy-and-hold benchmark."""
    spy = prices_df[prices_df["ticker"] == "SPY"][["date", "close"]].copy()
    spy = spy.sort_values("date").reset_index(drop=True)
    if spy.empty:
        return {}
    spy["equity"] = starting_cash * spy["close"] / spy["close"].iloc[0]
    return compute_metrics_full(spy.rename(columns={"equity": "equity"})[["date", "equity"]], starting_cash)


def main():
    t0 = time.time()
    print("=" * 70)
    print("IRON CONDOR COMPREHENSIVE STRESS TEST")
    print("HC #428 R1 Regime Validation + VIX Stress + Wing Width Sweep")
    print("=" * 70)

    prices, iv, macro, fund, universe, earnings = load_data()

    n_tickers = prices["ticker"].nunique()
    date_range = f"{prices['date'].min().strftime('%Y-%m-%d')} to {prices['date'].max().strftime('%Y-%m-%d')}"
    n_days = prices["date"].nunique()
    print(f"\nUniverse: {n_tickers} tickers, {n_days} days ({date_range})")

    # SPY benchmark
    print("\n--- SPY Buy & Hold Benchmark ---")
    spy_metrics = spy_buyhold(prices)
    print(f"  SPY: CAGR={spy_metrics.get('cagr_pct')}%, Sharpe={spy_metrics.get('sharpe')}, "
          f"MaxDD={spy_metrics.get('max_dd_pct')}%")

    # ══════════════════════════════════════════════════════════
    # TEST 1: Current paper-engine config (baseline)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 1: PAPER ENGINE CONFIG (25-delta, $10 wings, 7 DTE, 50% PT)")
    print("=" * 70)

    eq_baseline, ledger_baseline = run_ic_backtest(
        prices, iv, macro, fund, universe, earnings,
        spread_width=10.0, put_delta=0.25, call_delta=0.25,
        dte_target=7, profit_take=0.50, vix_gate=40.0,
        stop_loss_mult=1.0, label="Paper Engine Config"
    )
    baseline_metrics = compute_metrics_full(eq_baseline, 100_000.0)
    baseline_regime = compute_regime_stratified(eq_baseline, macro, prices)
    baseline_vix = compute_vix_stratified(eq_baseline, macro)
    baseline_tails = compute_tail_events(eq_baseline, ledger_baseline)

    print(f"  Sharpe: {baseline_metrics['sharpe']}")
    print(f"  Sortino: {baseline_metrics['sortino']}")
    print(f"  CAGR: {baseline_metrics['cagr_pct']}%")
    print(f"  MaxDD: {baseline_metrics['max_dd_pct']}%")
    print(f"  WR: {baseline_metrics['win_rate_pct']}%")
    print(f"  Regime gap: {baseline_regime['regime_gap']} (need <= 0.50)")
    print(f"  Green Sharpe: {baseline_regime['green_sharpe']}")
    print(f"  Red Sharpe: {baseline_regime['red_sharpe']}")
    print(f"  Passes R1: {baseline_regime['passes_r1']}")

    # Trade-level stats
    if not ledger_baseline.empty:
        avg_pnl = ledger_baseline["pnl"].mean()
        n_trades = len(ledger_baseline)
        wins = (ledger_baseline["pnl"] > 0).sum()
        wr_trade = wins / n_trades * 100
        print(f"  Trades: {n_trades}, WR(trade): {wr_trade:.1f}%, Avg PnL: ${avg_pnl:.2f}")

    # ══════════════════════════════════════════════════════════
    # TEST 2: Best config from prior study (65% PT)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 2: BEST PRIOR CONFIG (25-delta, $10 wings, 7 DTE, 65% PT)")
    print("=" * 70)

    eq_best, ledger_best = run_ic_backtest(
        prices, iv, macro, fund, universe, earnings,
        spread_width=10.0, put_delta=0.25, call_delta=0.25,
        dte_target=7, profit_take=0.65, vix_gate=40.0,
        stop_loss_mult=2.0, label="IC 65% PT"
    )
    best_metrics = compute_metrics_full(eq_best, 100_000.0)
    best_regime = compute_regime_stratified(eq_best, macro, prices)
    best_vix = compute_vix_stratified(eq_best, macro)
    best_tails = compute_tail_events(eq_best, ledger_best)

    print(f"  Sharpe: {best_metrics['sharpe']}")
    print(f"  Sortino: {best_metrics['sortino']}")
    print(f"  CAGR: {best_metrics['cagr_pct']}%")
    print(f"  MaxDD: {best_metrics['max_dd_pct']}%")
    print(f"  Regime gap: {best_regime['regime_gap']} (need <= 0.50)")
    print(f"  Passes R1: {best_regime['passes_r1']}")

    # ══════════════════════════════════════════════════════════
    # TEST 3: WING WIDTH SWEEP (Fixed $)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 3: WING WIDTH SWEEP (Fixed $ widths)")
    print("=" * 70)

    wing_results = {}
    for width in [5.0, 7.0, 10.0, 15.0, 20.0]:
        print(f"\n  --- Wing width: ${width:.0f} ---")
        eq, ledger = run_ic_backtest(
            prices, iv, macro, fund, universe, earnings,
            spread_width=width, put_delta=0.25, call_delta=0.25,
            dte_target=7, profit_take=0.65, vix_gate=40.0,
            stop_loss_mult=2.0, width_mode="fixed",
            label=f"IC ${width:.0f} wings"
        )
        m = compute_metrics_full(eq, 100_000.0)
        r = compute_regime_stratified(eq, macro, prices)
        v = compute_vix_stratified(eq, macro)
        t = compute_tail_events(eq, ledger)

        wing_results[f"${width:.0f}_fixed"] = {
            "metrics": m, "regime": r, "vix_strat": v, "tails": t
        }
        print(f"    Sharpe={m['sharpe']}, CAGR={m['cagr_pct']}%, MaxDD={m['max_dd_pct']}%")
        print(f"    Regime gap={r['regime_gap']}, Passes R1={r['passes_r1']}")
        if v.get("VIX 30-40", {}).get("sharpe") is not None:
            print(f"    VIX 30-40 Sharpe: {v['VIX 30-40']['sharpe']}")
        if v.get("VIX 40+", {}).get("sharpe") is not None:
            print(f"    VIX 40+ Sharpe: {v['VIX 40+']['sharpe']}")

    # ══════════════════════════════════════════════════════════
    # TEST 4: PROPORTIONAL WING WIDTH SWEEP
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 4: PROPORTIONAL WING WIDTH SWEEP (% of stock price)")
    print("=" * 70)

    for pct in [0.03, 0.05, 0.07, 0.10]:
        print(f"\n  --- Wing width: {pct*100:.0f}% of stock price ---")
        eq, ledger = run_ic_backtest(
            prices, iv, macro, fund, universe, earnings,
            spread_width=10.0,  # fallback, overridden by pct
            put_delta=0.25, call_delta=0.25,
            dte_target=7, profit_take=0.65, vix_gate=40.0,
            stop_loss_mult=2.0, width_mode="pct", width_pct=pct,
            label=f"IC {pct*100:.0f}% wings"
        )
        m = compute_metrics_full(eq, 100_000.0)
        r = compute_regime_stratified(eq, macro, prices)
        v = compute_vix_stratified(eq, macro)
        t = compute_tail_events(eq, ledger)

        wing_results[f"{pct*100:.0f}%_pct"] = {
            "metrics": m, "regime": r, "vix_strat": v, "tails": t
        }
        print(f"    Sharpe={m['sharpe']}, CAGR={m['cagr_pct']}%, MaxDD={m['max_dd_pct']}%")
        print(f"    Regime gap={r['regime_gap']}, Passes R1={r['passes_r1']}")

    # ══════════════════════════════════════════════════════════
    # TEST 5: NO VIX GATE (pure stress test)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 5: NO VIX GATE (trades through ALL vol regimes)")
    print("=" * 70)

    eq_novix, ledger_novix = run_ic_backtest(
        prices, iv, macro, fund, universe, earnings,
        spread_width=10.0, put_delta=0.25, call_delta=0.25,
        dte_target=7, profit_take=0.65, vix_gate=999.0,  # effectively no gate
        stop_loss_mult=2.0, label="IC No VIX Gate"
    )
    novix_metrics = compute_metrics_full(eq_novix, 100_000.0)
    novix_regime = compute_regime_stratified(eq_novix, macro, prices)
    novix_vix = compute_vix_stratified(eq_novix, macro)
    novix_tails = compute_tail_events(eq_novix, ledger_novix)

    print(f"  Sharpe: {novix_metrics['sharpe']}")
    print(f"  MaxDD: {novix_metrics['max_dd_pct']}%")
    print(f"  Regime gap: {novix_regime['regime_gap']}")
    print(f"  VIX>30 Sharpe: {novix_vix.get('VIX 30-40', {}).get('sharpe')}")
    print(f"  VIX>40 Sharpe: {novix_vix.get('VIX 40+', {}).get('sharpe')}")

    # Blown wings analysis
    if not ledger_novix.empty:
        both = ledger_novix[ledger_novix["kind"] == "expire_both"]
        print(f"  Both wings blown: {len(both)} trades, total loss: ${both['pnl'].sum():,.0f}")

    # ══════════════════════════════════════════════════════════
    # COMPILE FINAL REPORT
    # ══════════════════════════════════════════════════════════
    elapsed = time.time() - t0

    report = {
        "generated": pd.Timestamp.now().isoformat(),
        "data_range": date_range,
        "n_tickers": n_tickers,
        "n_days": n_days,
        "elapsed_minutes": round(elapsed / 60, 1),

        "spy_benchmark": spy_metrics,

        "paper_engine_config": {
            "params": {"put_delta": 0.25, "call_delta": 0.25, "spread_width": 10.0,
                      "dte": 7, "profit_take": 0.50, "vix_gate": 40.0, "stop_loss": 1.0},
            "metrics": baseline_metrics,
            "regime": baseline_regime,
            "vix_stratified": baseline_vix,
            "tail_events": baseline_tails,
        },

        "best_prior_config": {
            "params": {"put_delta": 0.25, "call_delta": 0.25, "spread_width": 10.0,
                      "dte": 7, "profit_take": 0.65, "vix_gate": 40.0, "stop_loss": 2.0},
            "metrics": best_metrics,
            "regime": best_regime,
            "vix_stratified": best_vix,
            "tail_events": best_tails,
        },

        "wing_width_sweep": wing_results,

        "no_vix_gate_stress": {
            "metrics": novix_metrics,
            "regime": novix_regime,
            "vix_stratified": novix_vix,
            "tail_events": novix_tails,
        },

        "hc428_r1_validation": {
            "threshold": 0.50,
            "paper_engine_gap": baseline_regime["regime_gap"],
            "paper_engine_passes": baseline_regime["passes_r1"],
            "best_config_gap": best_regime["regime_gap"],
            "best_config_passes": best_regime["passes_r1"],
            "any_wing_width_passes": any(
                v["regime"]["passes_r1"] for v in wing_results.values()
            ),
        },

        "risk_flag_analysis": {
            "expected_5d_move_exceeds_wings": True,
            "explanation": "The paper engine uses $10 fixed wings on 25-delta short strikes. "
                          "For a $100 stock with 50% IV, 5-day expected move is ~$8.20 (8.2%). "
                          "Wing width is $10 (10%), so wings ARE wider than expected move. "
                          "But for high-IV names (>80% IV), 5-day expected move can exceed wings. "
                          "Credit/width ratio of 38.9% is aggressive but consistent with 25-delta selection.",
            "recommendation": "Use proportional wings (7-10% of stock price) instead of fixed $10 "
                            "to maintain consistent risk/reward across price levels."
        },
    }

    # Save
    with open(OUTPUT / "ic_stress_test_results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save equity curves
    eq_baseline.to_parquet(OUTPUT / "eq_paper_config.parquet")
    eq_best.to_parquet(OUTPUT / "eq_best_config.parquet")
    eq_novix.to_parquet(OUTPUT / "eq_no_vix_gate.parquet")

    # Print summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Config':<25} {'Sharpe':>8} {'Sortino':>8} {'CAGR%':>8} {'MaxDD%':>8} {'Gap':>6} {'R1?':>5}")
    print("-" * 75)
    print(f"{'SPY Buy&Hold':<25} {spy_metrics.get('sharpe','?'):>8} {spy_metrics.get('sortino','?'):>8} "
          f"{spy_metrics.get('cagr_pct','?'):>7}% {spy_metrics.get('max_dd_pct','?'):>7}% {'n/a':>6} {'n/a':>5}")
    print(f"{'Paper Engine (50%PT)':<25} {baseline_metrics['sharpe']:>8} {baseline_metrics['sortino']:>8} "
          f"{baseline_metrics['cagr_pct']:>7}% {baseline_metrics['max_dd_pct']:>7}% "
          f"{baseline_regime['regime_gap']:>6} {'YES' if baseline_regime['passes_r1'] else 'NO':>5}")
    print(f"{'Best (65%PT)':<25} {best_metrics['sharpe']:>8} {best_metrics['sortino']:>8} "
          f"{best_metrics['cagr_pct']:>7}% {best_metrics['max_dd_pct']:>7}% "
          f"{best_regime['regime_gap']:>6} {'YES' if best_regime['passes_r1'] else 'NO':>5}")
    print(f"{'No VIX Gate':<25} {novix_metrics['sharpe']:>8} {novix_metrics['sortino']:>8} "
          f"{novix_metrics['cagr_pct']:>7}% {novix_metrics['max_dd_pct']:>7}% "
          f"{novix_regime['regime_gap']:>6} {'YES' if novix_regime['passes_r1'] else 'NO':>5}")

    print(f"\n{'Wing Width':<25} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'Gap':>6} {'R1?':>5} {'VIX30-40':>10}")
    print("-" * 75)
    for wname, wdata in sorted(wing_results.items()):
        m = wdata["metrics"]
        r = wdata["regime"]
        v30 = wdata["vix_strat"].get("VIX 30-40", {}).get("sharpe", "n/a")
        print(f"{wname:<25} {m['sharpe']:>8} {m['cagr_pct']:>7}% {m['max_dd_pct']:>7}% "
              f"{r['regime_gap']:>6} {'YES' if r['passes_r1'] else 'NO':>5} {str(v30):>10}")

    print(f"\n{'VIX Bucket':<15} {'Paper Sharpe':>14} {'Best Sharpe':>14} {'NoGate Sharpe':>14} {'Days':>6}")
    print("-" * 65)
    for bucket in ["VIX<18", "VIX 18-25", "VIX 25-30", "VIX 30-40", "VIX 40+"]:
        bs = baseline_vix.get(bucket, {}).get("sharpe", "n/a")
        be = best_vix.get(bucket, {}).get("sharpe", "n/a")
        nv = novix_vix.get(bucket, {}).get("sharpe", "n/a")
        nd = baseline_vix.get(bucket, {}).get("n_days", 0)
        print(f"{bucket:<15} {str(bs):>14} {str(be):>14} {str(nv):>14} {nd:>6}")

    print(f"\nTail Events (Paper Engine Config):")
    print(f"{'Event':<25} {'Return%':>10} {'MaxDD%':>10}")
    print("-" * 50)
    for event, data in baseline_tails.items():
        if event == "both_wings_blown":
            continue
        if data.get("return_pct") is not None:
            print(f"{event:<25} {data['return_pct']:>9}% {data['max_dd_pct']:>9}%")

    if "both_wings_blown" in baseline_tails:
        bwb = baseline_tails["both_wings_blown"]
        print(f"\nBoth Wings Blown: {bwb['count']} trades, avg loss: ${bwb['avg_loss']:.0f}")

    print(f"\nElapsed: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {OUTPUT}/ic_stress_test_results.json")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    any_passes = (baseline_regime["passes_r1"] or best_regime["passes_r1"] or
                  any(v["regime"]["passes_r1"] for v in wing_results.values()))
    if any_passes:
        print("AT LEAST ONE CONFIG PASSES HC #428 R1")
        passing = []
        if baseline_regime["passes_r1"]:
            passing.append(f"Paper Engine (gap={baseline_regime['regime_gap']})")
        if best_regime["passes_r1"]:
            passing.append(f"Best 65%PT (gap={best_regime['regime_gap']})")
        for wname, wdata in wing_results.items():
            if wdata["regime"]["passes_r1"]:
                passing.append(f"Wing {wname} (gap={wdata['regime']['regime_gap']})")
        print(f"  Passing configs: {', '.join(passing)}")
    else:
        print("NO CONFIG PASSES HC #428 R1 (regime gap > 0.50 for ALL)")
        print("  Iron condor is structurally regime-asymmetric on BS-priced multi-stock universe.")
        print("  The paper-engine 11/11 WR is too few trades to be statistically meaningful.")

    # Compare vs SPY
    if baseline_metrics["sharpe"] > spy_metrics.get("sharpe", 0):
        print(f"\n  IC outperforms SPY on Sharpe: {baseline_metrics['sharpe']} vs {spy_metrics.get('sharpe')}")
    else:
        print(f"\n  IC UNDERPERFORMS SPY on Sharpe: {baseline_metrics['sharpe']} vs {spy_metrics.get('sharpe')}")


if __name__ == "__main__":
    main()
