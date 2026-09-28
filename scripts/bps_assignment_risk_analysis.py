#!/usr/bin/env python3
"""
BPS Assignment Risk & Early Close Analysis
===========================================

HC #664 R4 gap: assignment risk modeling, fill quality testing

Current BPS backtest holds all positions to expiry if profit-take isn't hit.
In practice, smart traders close positions 1 DTE to avoid pin risk and assignment.

This study answers:
  1. What fraction of BPS trades hit profit-take vs go to expiry?
  2. Of those that expire, what fraction are OTM / pin-risk / breached / max-loss?
  3. What does 1-DTE buyback cost vs holding through expiry?
  4. How often do overnight gaps cause breach events?
  5. Distribution of actual losses when assigned
  6. Optimal close-before-expiry timing (1 DTE, 2 DTE, 50% loss stop)

Output: output/bps_assignment_risk/assignment_risk_report.json
"""

import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))
from higher_returns_study import (
    load_data, bs_price, bs_delta, strike_from_delta, trade_cost,
    compute_metrics, COST_PER_CONTRACT
)

OUTPUT = ROOT / "output" / "bps_assignment_risk"
OUTPUT.mkdir(parents=True, exist_ok=True)


def run_bps_with_trade_tracking(prices, iv, macro, fund, universe, earnings,
                                 spread_width=15.0, put_delta=0.25,
                                 dte_target=10, profit_take=0.40,
                                 margin_cap=0.15, max_concurrent=40,
                                 per_name_pct=0.03, starting_cash=100_000.0,
                                 close_before_expiry_days=0,  # 0 = hold to expiry, 1 = close 1 DTE
                                 loss_stop=None,  # e.g. 0.50 = close if lost 50% of max risk
                                 label="BPS"):
    """
    Enhanced BPS backtest that tracks every position's lifecycle in detail.
    Returns detailed trade records for assignment risk analysis.
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

    # Universe is a DataFrame with 'ticker' column
    if isinstance(universe, pd.DataFrame):
        universe_tickers = universe["ticker"].tolist()
    else:
        universe_tickers = list(universe)

    all_dates = sorted(prices_df["date"].unique())

    cash = starting_cash
    positions = {}
    equity_curve = []
    detailed_trades = []  # Every trade with full lifecycle info

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})
        m = macro_by_date.get(dt, {})
        vix = m.get("vix", float("nan")) if isinstance(m, dict) else float("nan")

        # ── Update positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            # Track min price during life (for gap risk analysis)
            if S < pos.get("min_price", float("inf")):
                pos["min_price"] = S
                pos["min_price_date"] = dt

            # Track price history for overnight gap analysis
            if "price_history" not in pos:
                pos["price_history"] = []
            pos["price_history"].append({"date": dt, "price": S})

            # ── Close before expiry? ──
            if close_before_expiry_days > 0 and T_days == close_before_expiry_days:
                short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
                long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
                cost_to_close = (short_val - long_val) * 100 * pos["contracts"]
                close_fees = trade_cost(short_val, pos["contracts"]) + trade_cost(long_val, pos["contracts"])

                realized = pos["net_credit"] - cost_to_close - close_fees
                cash -= cost_to_close + close_fees

                # Determine position status at close
                distance_to_short = (S - pos["short_strike"]) / pos["short_strike"]
                status = "safe" if distance_to_short > 0.02 else ("pin_risk" if distance_to_short > -0.02 else "breached")

                detailed_trades.append({
                    "open_date": pos["open_date"],
                    "close_date": dt,
                    "ticker": tk,
                    "exit_type": f"early_close_{close_before_expiry_days}DTE",
                    "short_strike": pos["short_strike"],
                    "long_strike": pos["long_strike"],
                    "open_price": pos.get("open_stock_price", 0),
                    "close_price": S,
                    "min_price": pos.get("min_price", S),
                    "net_credit": pos["net_credit"],
                    "realized_pnl": realized,
                    "contracts": pos["contracts"],
                    "distance_to_short_pct": distance_to_short * 100,
                    "status_at_close": status,
                    "spread_width": pos["short_strike"] - pos["long_strike"],
                    "max_loss": (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"],
                    "days_held": (dt - pos["open_date"]).days,
                })
                to_remove.append(tk)
                continue

            # ── Loss stop? ──
            if loss_stop is not None and T_days > 0:
                short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
                long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
                spread_val = (short_val - long_val) * 100 * pos["contracts"]
                max_possible_loss = (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"]
                unrealized_loss = spread_val - pos["net_credit"]

                if unrealized_loss > 0 and unrealized_loss / max_possible_loss >= loss_stop:
                    close_fees = trade_cost(short_val, pos["contracts"]) + trade_cost(long_val, pos["contracts"])
                    realized = pos["net_credit"] - spread_val - close_fees
                    cash -= spread_val + close_fees

                    detailed_trades.append({
                        "open_date": pos["open_date"],
                        "close_date": dt,
                        "ticker": tk,
                        "exit_type": "loss_stop",
                        "short_strike": pos["short_strike"],
                        "long_strike": pos["long_strike"],
                        "open_price": pos.get("open_stock_price", 0),
                        "close_price": S,
                        "min_price": pos.get("min_price", S),
                        "net_credit": pos["net_credit"],
                        "realized_pnl": realized,
                        "contracts": pos["contracts"],
                        "distance_to_short_pct": ((S - pos["short_strike"]) / pos["short_strike"]) * 100,
                        "status_at_close": "loss_stop",
                        "spread_width": pos["short_strike"] - pos["long_strike"],
                        "max_loss": max_possible_loss,
                        "days_held": (dt - pos["open_date"]).days,
                    })
                    to_remove.append(tk)
                    continue

            if T_days <= 0:
                # Expiry settlement
                short_itm = S < pos["short_strike"]
                long_itm = S < pos["long_strike"]
                close_cost = COST_PER_CONTRACT * 2 * pos["contracts"]

                distance_to_short = (S - pos["short_strike"]) / pos["short_strike"]
                breach_depth = 0.0

                if not short_itm:
                    realized = pos["net_credit"] - close_cost
                    cash -= close_cost
                    status = "OTM_safe" if distance_to_short > 0.02 else "OTM_pin_risk"
                elif short_itm and not long_itm:
                    loss = (pos["short_strike"] - S) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost
                    status = "partial_breach"
                    breach_depth = (pos["short_strike"] - S) / (pos["short_strike"] - pos["long_strike"])
                else:
                    loss = (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost
                    status = "full_breach"
                    breach_depth = 1.0

                # Check for overnight gap into breach
                gap_breach = False
                if len(pos.get("price_history", [])) >= 2:
                    ph = pos["price_history"]
                    prev_close = ph[-2]["price"] if len(ph) >= 2 else S
                    overnight_move = (S - prev_close) / prev_close
                    if prev_close >= pos["short_strike"] and S < pos["short_strike"]:
                        gap_breach = True
                else:
                    overnight_move = 0.0

                detailed_trades.append({
                    "open_date": pos["open_date"],
                    "close_date": dt,
                    "ticker": tk,
                    "exit_type": "expiry",
                    "short_strike": pos["short_strike"],
                    "long_strike": pos["long_strike"],
                    "open_price": pos.get("open_stock_price", 0),
                    "close_price": S,
                    "min_price": pos.get("min_price", S),
                    "net_credit": pos["net_credit"],
                    "realized_pnl": realized,
                    "contracts": pos["contracts"],
                    "distance_to_short_pct": distance_to_short * 100,
                    "status_at_close": status,
                    "breach_depth": breach_depth,
                    "gap_breach": gap_breach,
                    "overnight_move_pct": overnight_move * 100 if overnight_move else 0,
                    "spread_width": pos["short_strike"] - pos["long_strike"],
                    "max_loss": (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"],
                    "days_held": (dt - pos["open_date"]).days,
                })
                to_remove.append(tk)
            else:
                # Profit take
                short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
                long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
                spread_val = (short_val - long_val) * 100 * pos["contracts"]

                initial_credit = pos["net_credit"]
                cost_to_close = spread_val + trade_cost(short_val, pos["contracts"]) + trade_cost(long_val, pos["contracts"])

                captured = (initial_credit - cost_to_close) / max(initial_credit, 1e-6)
                if captured >= profit_take:
                    realized = initial_credit - cost_to_close
                    cash -= cost_to_close

                    detailed_trades.append({
                        "open_date": pos["open_date"],
                        "close_date": dt,
                        "ticker": tk,
                        "exit_type": "profit_take",
                        "short_strike": pos["short_strike"],
                        "long_strike": pos["long_strike"],
                        "open_price": pos.get("open_stock_price", 0),
                        "close_price": S,
                        "min_price": pos.get("min_price", S),
                        "net_credit": pos["net_credit"],
                        "realized_pnl": realized,
                        "contracts": pos["contracts"],
                        "distance_to_short_pct": ((S - pos["short_strike"]) / pos["short_strike"]) * 100,
                        "status_at_close": "profit_take",
                        "spread_width": pos["short_strike"] - pos["long_strike"],
                        "max_loss": (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"],
                        "days_held": (dt - pos["open_date"]).days,
                    })
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

        # ── Open new positions (same logic as base) ──
        try:
            vix_val = float(vix)
        except (TypeError, ValueError):
            vix_val = float("nan")
        if not np.isnan(vix_val) and vix_val > 35:
            continue

        current_margin = sum(
            (p["short_strike"] - p["long_strike"]) * 100 * p["contracts"]
            for p in positions.values()
        )
        max_margin = margin_cap * equity
        remaining_margin = max_margin - current_margin
        slots = max_concurrent - len(positions)

        if slots <= 0 or remaining_margin <= 0:
            continue

        # Rank candidates by IV rank
        candidates = []
        for tk in universe_tickers:
            if tk in positions:
                continue
            S = date_px.get(tk)
            sigma = date_sigma.get(tk)
            iv_rk = date_iv_rank.get(tk, 0)
            if S is None or sigma is None or np.isnan(S) or np.isnan(sigma):
                continue
            if sigma < 0.05:
                continue
            candidates.append((tk, S, sigma, iv_rk))

        candidates.sort(key=lambda x: -x[3])

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
                "open_stock_price": S,
                "min_price": S,
                "min_price_date": dt,
                "price_history": [{"date": dt, "price": S}],
            }
            remaining_margin -= margin_per_contract * n_contracts

            if len(positions) >= max_concurrent:
                break

    eq_df = pd.DataFrame(equity_curve)
    metrics = compute_metrics(eq_df, starting_cash, label) if not eq_df.empty else {}
    metrics["n_trades"] = len(detailed_trades)

    return {
        "metrics": metrics,
        "equity_curve": eq_df,
        "trades": pd.DataFrame(detailed_trades),
        "label": label,
    }


def analyze_assignment_risk(trades_df):
    """Comprehensive assignment risk analysis."""
    results = {}

    if trades_df.empty:
        return {"error": "no trades"}

    # 1. Exit type distribution
    exit_counts = trades_df["exit_type"].value_counts().to_dict()
    total = len(trades_df)
    exit_pcts = {k: round(v / total * 100, 1) for k, v in exit_counts.items()}
    results["exit_distribution"] = {"counts": exit_counts, "percentages": exit_pcts}

    # 2. Expiry-specific analysis
    expiry_trades = trades_df[trades_df["exit_type"] == "expiry"]
    if len(expiry_trades) > 0:
        status_counts = expiry_trades["status_at_close"].value_counts().to_dict()
        exp_total = len(expiry_trades)
        status_pcts = {k: round(v / exp_total * 100, 1) for k, v in status_counts.items()}
        results["expiry_status"] = {"counts": status_counts, "percentages": status_pcts}

        # Breach analysis
        breached = expiry_trades[expiry_trades["status_at_close"].isin(["partial_breach", "full_breach"])]
        if len(breached) > 0:
            results["breach_analysis"] = {
                "total_breached": len(breached),
                "pct_of_all_trades": round(len(breached) / total * 100, 2),
                "pct_of_expiry_trades": round(len(breached) / exp_total * 100, 2),
                "avg_loss_per_breach": round(breached["realized_pnl"].mean(), 2),
                "median_loss_per_breach": round(breached["realized_pnl"].median(), 2),
                "worst_loss": round(breached["realized_pnl"].min(), 2),
                "avg_breach_depth": round(breached["breach_depth"].mean(), 3),
                "partial_breach_count": int((breached["status_at_close"] == "partial_breach").sum()),
                "full_breach_count": int((breached["status_at_close"] == "full_breach").sum()),
                "avg_partial_loss_vs_max": None,
            }

            partial = breached[breached["status_at_close"] == "partial_breach"]
            if len(partial) > 0:
                avg_partial_loss_ratio = (-partial["realized_pnl"] / partial["max_loss"]).mean()
                results["breach_analysis"]["avg_partial_loss_vs_max"] = round(avg_partial_loss_ratio, 3)

        # Pin risk (within 2% of short strike at expiry)
        pin_risk = expiry_trades[expiry_trades["status_at_close"] == "OTM_pin_risk"]
        results["pin_risk"] = {
            "count": len(pin_risk),
            "pct_of_expiry": round(len(pin_risk) / exp_total * 100, 2) if exp_total > 0 else 0,
            "avg_distance_pct": round(pin_risk["distance_to_short_pct"].mean(), 2) if len(pin_risk) > 0 else 0,
        }

        # Gap breach events
        gap_breaches = expiry_trades[expiry_trades.get("gap_breach", pd.Series(False, index=expiry_trades.index)) == True]
        results["gap_risk"] = {
            "gap_breach_events": len(gap_breaches),
            "pct_of_breaches": round(len(gap_breaches) / max(len(breached), 1) * 100, 1) if len(breached) > 0 else 0,
        }
    else:
        results["expiry_status"] = {"note": "all trades closed before expiry"}

    # 3. P&L distribution by exit type
    pnl_by_exit = {}
    for exit_type in trades_df["exit_type"].unique():
        subset = trades_df[trades_df["exit_type"] == exit_type]
        pnl_by_exit[exit_type] = {
            "count": len(subset),
            "total_pnl": round(subset["realized_pnl"].sum(), 2),
            "avg_pnl": round(subset["realized_pnl"].mean(), 2),
            "win_rate": round((subset["realized_pnl"] > 0).mean() * 100, 1),
            "avg_days_held": round(subset["days_held"].mean(), 1),
        }
    results["pnl_by_exit_type"] = pnl_by_exit

    # 4. Risk/reward summary
    winners = trades_df[trades_df["realized_pnl"] > 0]
    losers = trades_df[trades_df["realized_pnl"] <= 0]
    results["risk_reward"] = {
        "avg_win": round(winners["realized_pnl"].mean(), 2) if len(winners) > 0 else 0,
        "avg_loss": round(losers["realized_pnl"].mean(), 2) if len(losers) > 0 else 0,
        "win_loss_ratio": round(abs(winners["realized_pnl"].mean() / losers["realized_pnl"].mean()), 2) if len(losers) > 0 and losers["realized_pnl"].mean() != 0 else float("inf"),
        "max_single_loss": round(trades_df["realized_pnl"].min(), 2),
        "max_single_win": round(trades_df["realized_pnl"].max(), 2),
        "total_pnl": round(trades_df["realized_pnl"].sum(), 2),
    }

    # 5. Credit vs max-risk ratio
    credit_to_risk = trades_df["net_credit"] / trades_df["max_loss"]
    results["credit_risk_ratio"] = {
        "avg_credit_to_max_risk": round(credit_to_risk.mean(), 3),
        "median_credit_to_max_risk": round(credit_to_risk.median(), 3),
        "min_credit_to_max_risk": round(credit_to_risk.min(), 3),
    }

    return results


def main():
    t0 = time.time()
    print("Loading data...")
    prices, iv, macro, fund, universe, earnings = load_data()
    print(f"Data loaded in {time.time()-t0:.1f}s")

    all_results = {}

    # ── Config 1: Conservative baseline (hold to expiry) ──
    print("\n=== Config 1: Conservative BPS — Hold to Expiry ===")
    r1 = run_bps_with_trade_tracking(
        prices, iv, macro, fund, universe, earnings,
        spread_width=15.0, put_delta=0.25, dte_target=10,
        profit_take=0.40, margin_cap=0.15, max_concurrent=40,
        per_name_pct=0.03, close_before_expiry_days=0,
        label="Conservative_HoldToExpiry"
    )
    print(f"  Sharpe: {r1['metrics'].get('sharpe')}  MaxDD: {r1['metrics'].get('max_dd_pct')}%  Trades: {r1['metrics'].get('n_trades')}")
    a1 = analyze_assignment_risk(r1["trades"])
    all_results["hold_to_expiry"] = {
        "metrics": r1["metrics"],
        "assignment_risk": a1,
    }

    # ── Config 2: Same but close 1 DTE ──
    print("\n=== Config 2: Conservative BPS — Close 1 DTE ===")
    r2 = run_bps_with_trade_tracking(
        prices, iv, macro, fund, universe, earnings,
        spread_width=15.0, put_delta=0.25, dte_target=10,
        profit_take=0.40, margin_cap=0.15, max_concurrent=40,
        per_name_pct=0.03, close_before_expiry_days=1,
        label="Conservative_Close1DTE"
    )
    print(f"  Sharpe: {r2['metrics'].get('sharpe')}  MaxDD: {r2['metrics'].get('max_dd_pct')}%  Trades: {r2['metrics'].get('n_trades')}")
    a2 = analyze_assignment_risk(r2["trades"])
    all_results["close_1dte"] = {
        "metrics": r2["metrics"],
        "assignment_risk": a2,
    }

    # ── Config 3: Same but close 2 DTE ──
    print("\n=== Config 3: Conservative BPS — Close 2 DTE ===")
    r3 = run_bps_with_trade_tracking(
        prices, iv, macro, fund, universe, earnings,
        spread_width=15.0, put_delta=0.25, dte_target=10,
        profit_take=0.40, margin_cap=0.15, max_concurrent=40,
        per_name_pct=0.03, close_before_expiry_days=2,
        label="Conservative_Close2DTE"
    )
    print(f"  Sharpe: {r3['metrics'].get('sharpe')}  MaxDD: {r3['metrics'].get('max_dd_pct')}%  Trades: {r3['metrics'].get('n_trades')}")
    a3 = analyze_assignment_risk(r3["trades"])
    all_results["close_2dte"] = {
        "metrics": r3["metrics"],
        "assignment_risk": a3,
    }

    # ── Config 4: Hold to expiry + 50% loss stop ──
    print("\n=== Config 4: Conservative BPS — Hold to Expiry + 50% Loss Stop ===")
    r4 = run_bps_with_trade_tracking(
        prices, iv, macro, fund, universe, earnings,
        spread_width=15.0, put_delta=0.25, dte_target=10,
        profit_take=0.40, margin_cap=0.15, max_concurrent=40,
        per_name_pct=0.03, close_before_expiry_days=0, loss_stop=0.50,
        label="Conservative_LossStop50pct"
    )
    print(f"  Sharpe: {r4['metrics'].get('sharpe')}  MaxDD: {r4['metrics'].get('max_dd_pct')}%  Trades: {r4['metrics'].get('n_trades')}")
    a4 = analyze_assignment_risk(r4["trades"])
    all_results["loss_stop_50pct"] = {
        "metrics": r4["metrics"],
        "assignment_risk": a4,
    }

    # ── Config 5: Close 1 DTE + 50% loss stop ──
    print("\n=== Config 5: Conservative BPS — Close 1 DTE + 50% Loss Stop ===")
    r5 = run_bps_with_trade_tracking(
        prices, iv, macro, fund, universe, earnings,
        spread_width=15.0, put_delta=0.25, dte_target=10,
        profit_take=0.40, margin_cap=0.15, max_concurrent=40,
        per_name_pct=0.03, close_before_expiry_days=1, loss_stop=0.50,
        label="Conservative_Close1DTE_LossStop50"
    )
    print(f"  Sharpe: {r5['metrics'].get('sharpe')}  MaxDD: {r5['metrics'].get('max_dd_pct')}%  Trades: {r5['metrics'].get('n_trades')}")
    a5 = analyze_assignment_risk(r5["trades"])
    all_results["close_1dte_lossstop50"] = {
        "metrics": r5["metrics"],
        "assignment_risk": a5,
    }

    # ── Summary comparison ──
    print("\n" + "="*80)
    print("ASSIGNMENT RISK COMPARISON")
    print("="*80)
    print(f"{'Config':<35} {'Sharpe':>7} {'MaxDD':>7} {'WR':>6} {'PF':>5} {'Trades':>7}")
    print("-"*80)
    for key, data in all_results.items():
        m = data["metrics"]
        print(f"{key:<35} {m.get('sharpe', 'N/A'):>7} {m.get('max_dd_pct', 'N/A'):>7}% {m.get('win_rate_pct', 'N/A'):>5}% {m.get('profit_factor', 'N/A'):>5} {m.get('n_trades', 0):>7}")

    print("\n" + "="*80)
    print("BREACH RATES")
    print("="*80)
    for key, data in all_results.items():
        ar = data["assignment_risk"]
        breach = ar.get("breach_analysis", {})
        print(f"\n{key}:")
        print(f"  Exit distribution: {ar.get('exit_distribution', {}).get('percentages', {})}")
        if breach:
            print(f"  Breach rate: {breach.get('pct_of_all_trades', 0):.1f}% of all trades")
            print(f"  Avg loss per breach: ${breach.get('avg_loss_per_breach', 0):.0f}")
            print(f"  Partial vs full: {breach.get('partial_breach_count', 0)} partial, {breach.get('full_breach_count', 0)} full")
        pin = ar.get("pin_risk", {})
        if pin:
            print(f"  Pin risk (OTM but <2% from strike): {pin.get('count', 0)} trades ({pin.get('pct_of_expiry', 0):.1f}% of expiry)")

    # ── Save ──
    # Convert non-serializable types
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_dict(v) for v in d]
        else:
            return make_serializable(d)

    report = clean_dict(all_results)
    with open(OUTPUT / "assignment_risk_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save detailed trade-level data for the baseline
    r1["trades"].to_parquet(OUTPUT / "trades_hold_to_expiry.parquet", index=False)
    r2["trades"].to_parquet(OUTPUT / "trades_close_1dte.parquet", index=False)

    print(f"\n✓ Report saved. Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
