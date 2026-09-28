#!/usr/bin/env python3
"""
BPS Dynamic Delta Study
========================

Tests whether dynamically adjusting put delta based on VIX regime improves
risk-adjusted returns. Intuition: in high-vol environments, use a lower delta
(more OTM) to reduce breach probability, even though premiums are richer.

Configs tested:
  A) Fixed delta=30 (0.30) — baseline
  B) Fixed delta=25 (0.25) — more conservative
  C) Fixed delta=20 (0.20) — very conservative
  D) Dynamic: delta=30 when VIX<18, delta=25 when VIX 18-25, delta=20 when VIX>25
  E) Dynamic aggressive: delta=35 when VIX<15, delta=30 when VIX 15-22, delta=25 when VIX>22
  F) Dynamic w/ spread width: delta=30/$15 when VIX<20, delta=25/$20 when VIX 20-28

All configs use:
  - 70-ticker universe
  - 5% BA cost
  - VIX-scaled sizing (linear scale down from VIX=15)
  - 2% CB (portfolio circuit breaker, 1-day freeze)
  - Earnings filter (skip trades with earnings within 7 days)
  - VIX hard cutoff at 30

Output: output/bps_dynamic_delta/
"""

import sys, json, time, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import timedelta

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "output" / "wheel_higher_returns_study"))

from higher_returns_study import (
    load_data, bs_price, bs_delta, strike_from_delta, trade_cost,
    COST_PER_CONTRACT
)

OUTPUT = ROOT / "output" / "bps_dynamic_delta"
OUTPUT.mkdir(parents=True, exist_ok=True)

BA_COST_FRAC = 0.05
STARTING_CAPITAL = 100_000


# ═══════════════════════════════════════════════════════════════════
# Earnings Filter
# ═══════════════════════════════════════════════════════════════════

def load_earnings():
    """Load earnings dates for filtering."""
    try:
        earnings = pd.read_parquet(ROOT / "wheel_strategy_v1" / "data" / "cache" / "earnings_dates.parquet")
        earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"]).dt.tz_localize(None)
        return earnings
    except Exception:
        return pd.DataFrame(columns=["ticker", "earnings_date"])


def build_earnings_lookup(earnings_df):
    """Build per-ticker sorted earnings date arrays for fast lookup."""
    lookup = {}
    for ticker in earnings_df["ticker"].unique():
        dates = earnings_df[earnings_df["ticker"] == ticker]["earnings_date"].sort_values().values
        if len(dates) > 0:
            lookup[ticker] = dates
    return lookup


def has_earnings_within(ticker, open_date, dte_target, earnings_lookup, buffer_days=7):
    """Check if ticker has earnings within buffer_days of the holding period."""
    if ticker not in earnings_lookup:
        return False
    earn_dates = earnings_lookup[ticker]
    hold_start = np.datetime64(open_date) - np.timedelta64(buffer_days, 'D')
    hold_end = np.datetime64(open_date) + np.timedelta64(dte_target + buffer_days, 'D')
    mask = (earn_dates >= hold_start) & (earn_dates <= hold_end)
    return mask.any()


# ═══════════════════════════════════════════════════════════════════
# Delta Selection Functions
# ═══════════════════════════════════════════════════════════════════

def fixed_delta(vix, delta_val, spread_width_val):
    """Fixed delta regardless of VIX."""
    return delta_val, spread_width_val


def dynamic_conservative(vix):
    """D) delta=30 when VIX<18, delta=25 when VIX 18-25, delta=20 when VIX>25."""
    if vix < 18:
        return 0.30, 15.0
    elif vix < 25:
        return 0.25, 15.0
    else:
        return 0.20, 15.0


def dynamic_aggressive(vix):
    """E) delta=35 when VIX<15, delta=30 when VIX 15-22, delta=25 when VIX>22."""
    if vix < 15:
        return 0.35, 15.0
    elif vix < 22:
        return 0.30, 15.0
    else:
        return 0.25, 15.0


def dynamic_with_spread(vix):
    """F) delta=30/$15 when VIX<20, delta=25/$20 when VIX 20-28."""
    if vix < 20:
        return 0.30, 15.0
    elif vix < 28:
        return 0.25, 20.0
    else:
        return 0.20, 20.0  # most conservative for VIX >= 28


# ═══════════════════════════════════════════════════════════════════
# BPS Backtest Engine (with dynamic delta + all defenses)
# ═══════════════════════════════════════════════════════════════════

def run_bps_dynamic(prices, iv, macro, fund, universe, earnings_lookup,
                    delta_fn, label="test",
                    dte_target=10, profit_take=0.65,
                    margin_cap=0.25, max_concurrent=40,
                    per_name_pct=0.03,
                    # Defenses
                    vix_scale=True, vix_base=15.0,
                    vix_hard_cutoff=30,
                    portfolio_cb_threshold=-0.02, portfolio_cb_freeze_days=1,
                    earnings_buffer_days=7):
    """
    Full BPS backtest with dynamic delta selection per trade.

    delta_fn: callable(vix) -> (delta, spread_width)
    """
    print(f"  Running: {label}...")

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

    if isinstance(universe, pd.DataFrame):
        universe_tickers = universe["ticker"].tolist()
    else:
        universe_tickers = list(universe)

    all_dates = sorted(prices_df["date"].unique())

    cash = STARTING_CAPITAL
    positions = {}
    equity_curve = []
    detailed_trades = []

    # Circuit breaker state
    frozen_until = None
    cb_trigger_count = 0

    # Tracking
    n_earnings_blocked = 0
    n_vix_blocked = 0
    n_cb_frozen = 0
    trades_by_delta = defaultdict(int)
    trades_by_vix_regime = defaultdict(lambda: {"opened": 0, "breached": 0, "pnl": 0.0})

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})
        m_data = macro_by_date.get(dt, {})
        vix = m_data.get("vix", float("nan")) if isinstance(m_data, dict) else float("nan")

        try:
            vix_val = float(vix)
        except (TypeError, ValueError):
            vix_val = float("nan")

        # ── Update/close positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            # Track min price
            if S < pos.get("min_price", float("inf")):
                pos["min_price"] = S

            # Close 1 DTE (early close)
            if T_days == 1:
                short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
                long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
                cost_to_close = (short_val - long_val) * 100 * pos["contracts"]
                close_fees = trade_cost(short_val, pos["contracts"]) + trade_cost(long_val, pos["contracts"])

                realized = pos["net_credit"] - cost_to_close - close_fees
                cash -= cost_to_close + close_fees

                distance_to_short = (S - pos["short_strike"]) / pos["short_strike"]
                status = "safe" if distance_to_short > 0.02 else ("pin_risk" if distance_to_short > -0.02 else "breached")

                # Determine VIX regime at open
                vix_regime = _vix_regime(pos.get("open_vix", 15))
                trades_by_vix_regime[vix_regime]["pnl"] += realized * pos.get("vix_scalar", 1.0)
                if status == "breached":
                    trades_by_vix_regime[vix_regime]["breached"] += 1

                detailed_trades.append({
                    "open_date": pos["open_date"],
                    "close_date": dt,
                    "ticker": tk,
                    "exit_type": "early_close_1DTE",
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
                    "put_delta_used": pos.get("put_delta_used", 0.30),
                    "open_vix": pos.get("open_vix", np.nan),
                    "vix_scalar": pos.get("vix_scalar", 1.0),
                })
                to_remove.append(tk)
                continue

            # Expiry
            if T_days <= 0:
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

                vix_regime = _vix_regime(pos.get("open_vix", 15))
                trades_by_vix_regime[vix_regime]["pnl"] += realized * pos.get("vix_scalar", 1.0)
                if status in ("partial_breach", "full_breach"):
                    trades_by_vix_regime[vix_regime]["breached"] += 1

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
                    "spread_width": pos["short_strike"] - pos["long_strike"],
                    "max_loss": (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"],
                    "days_held": (dt - pos["open_date"]).days,
                    "breach_depth": breach_depth,
                    "put_delta_used": pos.get("put_delta_used", 0.30),
                    "open_vix": pos.get("open_vix", np.nan),
                    "vix_scalar": pos.get("vix_scalar", 1.0),
                })
                to_remove.append(tk)
                continue

            # Profit take check
            short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
            long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
            spread_val = (short_val - long_val) * 100 * pos["contracts"]
            initial_credit = pos["net_credit"]
            cost_to_close = spread_val + trade_cost(short_val, pos["contracts"]) + trade_cost(long_val, pos["contracts"])

            captured = (initial_credit - cost_to_close) / max(initial_credit, 1e-6)
            if captured >= profit_take:
                realized = initial_credit - cost_to_close
                cash -= cost_to_close

                vix_regime = _vix_regime(pos.get("open_vix", 15))
                trades_by_vix_regime[vix_regime]["pnl"] += realized * pos.get("vix_scalar", 1.0)

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
                    "put_delta_used": pos.get("put_delta_used", 0.30),
                    "open_vix": pos.get("open_vix", np.nan),
                    "vix_scalar": pos.get("vix_scalar", 1.0),
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

        # ── Circuit Breaker check ──
        if portfolio_cb_threshold is not None and len(equity_curve) >= 2:
            prev_eq = equity_curve[-2]["equity"]
            if prev_eq > 0:
                daily_ret = (equity - prev_eq) / prev_eq
                if daily_ret < portfolio_cb_threshold:
                    freeze_end = dt + pd.Timedelta(days=portfolio_cb_freeze_days)
                    if frozen_until is None or freeze_end > frozen_until:
                        frozen_until = freeze_end
                        cb_trigger_count += 1

        # ── VIX hard cutoff ──
        if not np.isnan(vix_val) and vix_val > vix_hard_cutoff:
            n_vix_blocked += 1  # count days blocked
            continue

        # ── CB frozen? ──
        if frozen_until is not None and dt <= frozen_until:
            n_cb_frozen += 1
            continue

        # ── VIX sizing scalar ──
        if vix_scale and not np.isnan(vix_val):
            vix_scalar = max(0.0, 1.0 - (vix_val - vix_base) / 30.0) if vix_val > vix_base else 1.0
        else:
            vix_scalar = 1.0

        # ── Get delta for this VIX level ──
        if np.isnan(vix_val):
            put_delta, spread_width = 0.30, 15.0
        else:
            put_delta, spread_width = delta_fn(vix_val)

        # ── Open new positions ──
        current_margin = sum(
            (p["short_strike"] - p["long_strike"]) * 100 * p["contracts"]
            for p in positions.values()
        )
        max_margin = margin_cap * equity
        remaining_margin = max_margin - current_margin
        slots = max_concurrent - len(positions)

        if slots <= 0 or remaining_margin <= 0:
            continue

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
            # Earnings filter
            if has_earnings_within(tk, dt, dte_target, earnings_lookup, earnings_buffer_days):
                n_earnings_blocked += 1
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

            # Apply VIX sizing scalar
            n_contracts = max(1, int(n_contracts * vix_scalar))

            if margin_per_contract * n_contracts > remaining_margin:
                n_contracts = max(1, int(remaining_margin // margin_per_contract))
            if n_contracts < 1:
                continue

            net_credit = net_prem_per_share * 100 * n_contracts
            open_costs = trade_cost(prem_short, n_contracts) + trade_cost(prem_long, n_contracts)

            # Apply BA cost on open
            ba_open = net_prem_per_share * 100 * n_contracts * BA_COST_FRAC * 2
            net_credit -= open_costs + ba_open

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
                "put_delta_used": put_delta,
                "open_vix": vix_val,
                "vix_scalar": vix_scalar,
            }
            remaining_margin -= margin_per_contract * n_contracts

            # Track delta usage
            delta_bucket = f"d{int(put_delta * 100)}"
            trades_by_delta[delta_bucket] += 1

            vix_regime = _vix_regime(vix_val)
            trades_by_vix_regime[vix_regime]["opened"] += 1

            if len(positions) >= max_concurrent:
                break

    # ── Compute metrics ──
    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty or len(eq_df) < 30:
        return {"label": label, "error": "insufficient data"}

    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.sort_values("date").reset_index(drop=True)

    trades_df = pd.DataFrame(detailed_trades)

    # Apply BA cost on close for profit_take and early_close trades
    if not trades_df.empty:
        closed_early = trades_df["exit_type"].isin(["profit_take", "early_close_1DTE"])
        ba_close = pd.Series(0.0, index=trades_df.index)
        ba_close[closed_early] = trades_df.loc[closed_early, "net_credit"].abs() * BA_COST_FRAC * 2
        trades_df["ba_close_cost"] = ba_close
        trades_df["honest_pnl"] = trades_df["realized_pnl"] - ba_close

        # Scale by VIX scalar
        trades_df["scaled_pnl"] = trades_df["honest_pnl"] * trades_df["vix_scalar"]
    else:
        trades_df["honest_pnl"] = []
        trades_df["scaled_pnl"] = []

    # Build daily P&L from equity curve
    eq_df["ret"] = eq_df["equity"].pct_change()
    rets = eq_df["ret"].dropna()

    total_days = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days
    total_years = max(total_days / 365.25, 0.01)
    total_return = eq_df["equity"].iloc[-1] / STARTING_CAPITAL
    cagr = (total_return ** (1 / total_years)) - 1 if total_return > 0 else -1.0

    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0

    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * np.sqrt(252)) if len(downside) > 5 and downside.std() > 0 else 0.0

    eq_df["peak"] = eq_df["equity"].cummax()
    eq_df["dd"] = (eq_df["equity"] - eq_df["peak"]) / eq_df["peak"]
    max_dd = float(eq_df["dd"].min())
    max_dd_date = eq_df.loc[eq_df["dd"].idxmin(), "date"]

    wr = float(len(rets[rets > 0]) / len(rets) * 100) if len(rets) > 0 else 0.0

    wins = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = float(wins / losses) if losses > 0 else float("inf")

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # Per-trade metrics
    n_trades = len(trades_df)
    if n_trades > 0:
        trade_wr = float((trades_df["honest_pnl"] > 0).sum() / n_trades * 100)
        avg_credit = float(trades_df["net_credit"].mean())
        breach_trades = trades_df[trades_df["status_at_close"].isin(
            ["breached", "partial_breach", "full_breach"])]
        breach_rate = float(len(breach_trades) / n_trades * 100)
    else:
        trade_wr = 0.0
        avg_credit = 0.0
        breach_rate = 0.0

    # Breach rate per VIX regime
    regime_breach = {}
    if n_trades > 0 and "open_vix" in trades_df.columns:
        for rname, lo, hi in [("low", 0, 15), ("normal", 15, 25), ("high", 25, 35), ("crisis", 35, 200)]:
            mask = (trades_df["open_vix"] >= lo) & (trades_df["open_vix"] < hi)
            subset = trades_df[mask]
            if len(subset) > 0:
                n_breach = len(subset[subset["status_at_close"].isin(
                    ["breached", "partial_breach", "full_breach"])])
                regime_breach[rname] = {
                    "n_trades": len(subset),
                    "n_breached": n_breach,
                    "breach_rate_pct": round(n_breach / len(subset) * 100, 2),
                    "avg_credit": round(float(subset["net_credit"].mean()), 2),
                    "avg_pnl": round(float(subset["honest_pnl"].mean()), 2),
                    "trade_wr_pct": round(float((subset["honest_pnl"] > 0).sum() / len(subset) * 100), 1),
                }

    # Delta distribution
    delta_dist = {}
    if n_trades > 0 and "put_delta_used" in trades_df.columns:
        for d_val in trades_df["put_delta_used"].unique():
            d_mask = trades_df["put_delta_used"] == d_val
            d_sub = trades_df[d_mask]
            delta_dist[f"d{int(d_val*100)}"] = {
                "n_trades": len(d_sub),
                "avg_credit": round(float(d_sub["net_credit"].mean()), 2),
                "breach_rate_pct": round(
                    float(len(d_sub[d_sub["status_at_close"].isin(
                        ["breached", "partial_breach", "full_breach"])]) / len(d_sub) * 100), 2),
                "trade_wr_pct": round(float((d_sub["honest_pnl"] > 0).sum() / len(d_sub) * 100), 1),
            }

    metrics = {
        "label": label,
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "max_dd_date": str(max_dd_date.date()) if hasattr(max_dd_date, 'date') else str(max_dd_date),
        "calmar": round(calmar, 2),
        "daily_wr_pct": round(wr, 1),
        "profit_factor": round(pf, 2),
        "final_equity": round(float(eq_df["equity"].iloc[-1]), 2),
        "n_trades": n_trades,
        "trade_wr_pct": round(trade_wr, 1),
        "avg_premium_collected": round(avg_credit, 2),
        "breach_rate_pct": round(breach_rate, 2),
        "n_earnings_blocked": n_earnings_blocked,
        "n_vix_hard_blocked_days": n_vix_blocked,
        "n_cb_frozen_days": n_cb_frozen,
        "cb_triggers": cb_trigger_count,
        "regime_breach": regime_breach,
        "delta_distribution": delta_dist,
        "delta_usage": dict(trades_by_delta),
    }

    # Save equity curve
    eq_df[["date", "equity"]].to_parquet(OUTPUT / f"eq_{label}.parquet", index=False)

    return {
        "metrics": metrics,
        "trades_df": trades_df,
        "equity_df": eq_df,
    }


def _vix_regime(vix):
    if vix < 15:
        return "low"
    elif vix < 25:
        return "normal"
    elif vix < 35:
        return "high"
    else:
        return "crisis"


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 70)
    print("BPS DYNAMIC DELTA STUDY")
    print("Does VIX-adaptive delta selection improve risk-adjusted returns?")
    print("=" * 70)

    # Load data
    prices, iv, macro, fund, universe, earnings_raw = load_data()
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])

    # Load earnings for filter
    earnings_df = load_earnings()
    earnings_lookup = build_earnings_lookup(earnings_df)
    print(f"  Earnings lookup: {len(earnings_lookup)} tickers with earnings data")

    # Universe info
    if isinstance(universe, pd.DataFrame):
        n_tickers = len(universe)
    else:
        n_tickers = len(universe)
    print(f"  Universe: {n_tickers} tickers")

    # Define configs
    configs = {
        "A_fixed_d30": lambda vix: (0.30, 15.0),
        "B_fixed_d25": lambda vix: (0.25, 15.0),
        "C_fixed_d20": lambda vix: (0.20, 15.0),
        "D_dynamic_conservative": dynamic_conservative,
        "E_dynamic_aggressive": dynamic_aggressive,
        "F_dynamic_spread_adj": dynamic_with_spread,
    }

    all_results = {}

    for name, delta_fn in configs.items():
        result = run_bps_dynamic(
            prices, iv, macro, fund, universe, earnings_lookup,
            delta_fn=delta_fn, label=name,
            dte_target=10, profit_take=0.65,
            margin_cap=0.25, max_concurrent=40,
            per_name_pct=0.03,
            vix_scale=True, vix_base=15.0,
            vix_hard_cutoff=30,
            portfolio_cb_threshold=-0.02, portfolio_cb_freeze_days=1,
            earnings_buffer_days=7,
        )
        all_results[name] = result

        if "error" in result:
            print(f"  {name}: ERROR - {result['error']}")
            continue

        m = result["metrics"]
        print(f"\n  {'─'*60}")
        print(f"  {name}")
        print(f"  {'─'*60}")
        print(f"    CAGR:          {m['cagr_pct']:>8.1f}%")
        print(f"    Sharpe:        {m['sharpe']:>8.2f}")
        print(f"    Sortino:       {m['sortino']:>8.2f}")
        print(f"    Max DD:        {m['max_dd_pct']:>8.1f}%  (on {m['max_dd_date']})")
        print(f"    Calmar:        {m['calmar']:>8.2f}")
        print(f"    Daily WR:      {m['daily_wr_pct']:>8.1f}%")
        print(f"    PF:            {m['profit_factor']:>8.2f}")
        print(f"    Trade WR:      {m['trade_wr_pct']:>8.1f}%")
        print(f"    N Trades:      {m['n_trades']:>8d}")
        print(f"    Avg Premium:   ${m['avg_premium_collected']:>8.2f}")
        print(f"    Breach Rate:   {m['breach_rate_pct']:>8.2f}%")
        print(f"    CB Triggers:   {m['cb_triggers']:>8d}")

    # ═══════════════════════════════════════════════════════════════
    # Comparison Table
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 130)
    print("COMPARISON TABLE — ALL CONFIGS")
    print("=" * 130)
    header = (f"{'Config':<28} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} "
              f"{'Calmar':>7} {'PF':>6} {'TradeWR':>8} {'BreachR':>8} "
              f"{'AvgPrem':>9} {'Trades':>7}")
    print(header)
    print("-" * 130)

    for name in configs:
        if "error" in all_results[name]:
            print(f"{name:<28}  (ERROR)")
            continue
        m = all_results[name]["metrics"]
        print(f"{name:<28} {m['cagr_pct']:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd_pct']:>7.1f}% {m['calmar']:>7.2f} {m['profit_factor']:>6.2f} "
              f"{m['trade_wr_pct']:>7.1f}% {m['breach_rate_pct']:>7.2f}% "
              f"${m['avg_premium_collected']:>8.2f} {m['n_trades']:>7d}")

    # ═══════════════════════════════════════════════════════════════
    # Breach Rate by VIX Regime
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 130)
    print("BREACH RATE BY VIX REGIME")
    print("=" * 130)
    for name in configs:
        if "error" in all_results[name]:
            continue
        m = all_results[name]["metrics"]
        print(f"\n  {name}:")
        for regime in ["low", "normal", "high"]:
            rb = m["regime_breach"].get(regime, {})
            if rb:
                print(f"    {regime:<10} N={rb['n_trades']:>5}  Breach={rb['breach_rate_pct']:>5.1f}%  "
                      f"AvgCredit=${rb['avg_credit']:>7.2f}  AvgPnL=${rb['avg_pnl']:>8.2f}  "
                      f"WR={rb['trade_wr_pct']:>5.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # Delta Distribution (for dynamic configs)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("DELTA DISTRIBUTION (dynamic configs)")
    print("=" * 100)
    for name in ["D_dynamic_conservative", "E_dynamic_aggressive", "F_dynamic_spread_adj"]:
        if "error" in all_results[name]:
            continue
        m = all_results[name]["metrics"]
        print(f"\n  {name}:")
        print(f"    Delta usage: {m.get('delta_usage', {})}")
        for dk, dv in m.get("delta_distribution", {}).items():
            print(f"    {dk}: N={dv['n_trades']:>5}  AvgCredit=${dv['avg_credit']:>7.2f}  "
                  f"BreachRate={dv['breach_rate_pct']:>5.1f}%  WR={dv['trade_wr_pct']:>5.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # Premium vs. Breach Tradeoff Analysis
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("PREMIUM vs BREACH TRADEOFF")
    print("Does lower delta's reduced breach rate compensate for reduced premium?")
    print("=" * 100)
    baseline = all_results.get("A_fixed_d30", {})
    if "error" not in baseline:
        bm = baseline["metrics"]
        for name in configs:
            if name == "A_fixed_d30" or "error" in all_results[name]:
                continue
            m = all_results[name]["metrics"]
            delta_sharpe = m["sharpe"] - bm["sharpe"]
            delta_dd = m["max_dd_pct"] - bm["max_dd_pct"]
            delta_breach = m["breach_rate_pct"] - bm["breach_rate_pct"]
            delta_prem = m["avg_premium_collected"] - bm["avg_premium_collected"]
            print(f"  {name} vs baseline:")
            print(f"    Sharpe:  {delta_sharpe:>+.2f}  |  MaxDD: {delta_dd:>+.1f}pp  |  "
                  f"Breach: {delta_breach:>+.2f}pp  |  AvgPrem: ${delta_prem:>+.2f}")

    # ═══════════════════════════════════════════════════════════════
    # Best Config Selection
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("BEST CONFIG (Sortino * (1 - |MaxDD|/100))")
    print("=" * 70)
    scores = {}
    for name in configs:
        if "error" in all_results[name]:
            continue
        m = all_results[name]["metrics"]
        dd_penalty = 1.0 - abs(m["max_dd_pct"]) / 100.0
        score = m["sortino"] * dd_penalty
        scores[name] = round(score, 3)
        print(f"  {name:<28} Score={score:.3f}  (Sortino={m['sortino']:.2f} * DD_pen={dd_penalty:.3f})")

    if scores:
        best = max(scores, key=scores.get)
        print(f"\n  >>> BEST: {best}  (Score={scores[best]:.3f})")
    else:
        best = "none"

    # ═══════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════
    save_data = {
        "generated": pd.Timestamp.now().isoformat(),
        "ba_cost_frac": BA_COST_FRAC,
        "starting_capital": STARTING_CAPITAL,
        "configs": {},
        "scores": scores,
        "best_config": best,
        "methodology": {
            "dte_target": 10,
            "profit_take": 0.65,
            "margin_cap": 0.25,
            "vix_hard_cutoff": 30,
            "vix_scale_base": 15,
            "portfolio_cb": "-2% threshold, 1-day freeze",
            "earnings_filter": "skip if earnings within 7 days",
            "ba_cost": "5% of premium, both legs, open+close",
        },
    }
    for name in configs:
        if "error" in all_results[name]:
            save_data["configs"][name] = {"error": all_results[name].get("error", "unknown")}
        else:
            save_data["configs"][name] = all_results[name]["metrics"]

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    save_data = convert(save_data)
    with open(OUTPUT / "dynamic_delta_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Save trades for each config
    for name in configs:
        if "error" in all_results[name]:
            continue
        trades_df = all_results[name]["trades_df"]
        if not trades_df.empty:
            trades_df.to_parquet(OUTPUT / f"trades_{name}.parquet", index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Results saved to {OUTPUT}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
