#!/usr/bin/env python3
"""
wheel_v3_dd_protection.py — Drawdown protection overlays for v3 wheel engine.

Tests whether adding drawdown-aware position sizing can reduce bear-year losses
without destroying bull-year returns.

Overlays tested (on top of validated 2d-earnings-buffer config):
  O1 - Equity curve brake: if equity < (1-X%) of N-day peak → scale=0.5
  O2 - Vol-regime scaling: high realized vol → scale down exposure
  O3 - Combined halt: equity brake + vol → scale=0 (no new CSPs)
  O4 - Trailing stop: if portfolio DD > Y% → halt new CSPs until recovery

All overlays affect ONLY new CSP opens. Existing positions managed normally.
"""
import sys
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

ROOT = Path("/home/jupiter/Lvl3Quant")
OUT_DIR = ROOT / "output" / "wheel_v3_dd_protection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from wheel_universe_v3_expand import load_all_data, compute_metrics, regime_analysis
from wheel_earnings_filter import download_earnings_dates, build_earnings_lookup

logging.basicConfig(
    format='%(asctime)s [DD-PROT] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger('DD-PROT')


def run_portfolio_v3_with_overlay(
    prices_df, spy_regime, sector_map,
    starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
    put_delta=0.30, dte_target=14, profit_take=0.65,
    bear_mode="liq_csp_only", max_assignments_5d=3,
    max_share_positions=5, loss_cut_pct=-0.15,
    min_price=10.0, max_price=500.0,
    earnings_lookup=None, earnings_buffer_days=2,
    # Overlay parameters
    overlay_type="none",
    eq_brake_lookback=60,    # N days for equity peak lookback
    eq_brake_threshold=0.05, # X% drawdown threshold
    eq_brake_scale=0.5,      # Scale factor when braking
    vol_percentile_high=0.70,# Vol percentile for scaling
    vol_percentile_halt=0.90,# Vol percentile for halt
    vol_scale_high=0.50,     # Scale at high vol
    vol_scale_halt=0.25,     # Scale at very high vol
    trailing_stop_pct=0.10,  # Portfolio DD level to halt
    trailing_recovery_pct=0.03, # Recovery from trough to resume
):
    """
    Portfolio wheel with drawdown protection overlays.

    overlay_type: "none", "eq_brake", "vol_regime", "combined", "trailing_stop"
    """
    import math
    from collections import deque
    from dataclasses import dataclass

    MARGIN_REQ_PCT = 0.20
    COST_PER_CONTRACT = 0.65
    SLIPPAGE_FRAC = 0.025
    SLIPPAGE_MIN = 0.03
    VIX_MAX = 35.0
    RISK_FREE = 0.04
    CALL_DELTA = 0.30

    def _Phi(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if kind == "put":
            return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
        return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)

    def find_strike(S, sigma, T, delta_target, kind="put"):
        if T <= 0 or sigma <= 0:
            return S
        if kind == "put":
            lo, hi = S * 0.3, S * 1.0
        else:
            lo, hi = S * 1.0, S * 2.0
        for _ in range(60):
            K = (lo + hi) / 2
            d1 = (math.log(S / K) + (RISK_FREE + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T) + 1e-9)
            if kind == "put":
                delta_abs = _Phi(-d1)
            else:
                delta_abs = _Phi(d1)
            if delta_abs > delta_target:
                if kind == "put":
                    hi = K
                else:
                    lo = K
            else:
                if kind == "put":
                    lo = K
                else:
                    hi = K
        return round(K * 2) / 2

    # Filter tickers
    valid_tickers = set()
    for t, grp in prices_df.groupby("ticker"):
        median_price = grp["close"].median()
        if min_price <= median_price <= max_price:
            valid_tickers.add(t)

    prices_df = prices_df[prices_df["ticker"].isin(valid_tickers)].copy()
    tickers_available = sorted(prices_df["ticker"].unique())

    # Build date-indexed lookups
    ticker_data = {}
    for t in tickers_available:
        tdf = prices_df[prices_df["ticker"] == t].set_index("date").sort_index()
        ticker_data[t] = tdf

    spy_map = {}
    if not spy_regime.empty:
        for _, row in spy_regime.iterrows():
            spy_map[pd.Timestamp(row["date"])] = int(row["bear"])

    all_dates = sorted(pd.Timestamp(d) for d in prices_df["date"].unique())

    # Pre-compute SPY 20d realized vol percentiles (expanding, point-in-time)
    spy_vol_map = {}
    if "SPY" in ticker_data:
        spy_df = ticker_data["SPY"]
        if "sigma" in spy_df.columns:
            spy_vols = spy_df["sigma"].dropna()
            # Expanding percentile (only use past data)
            for i, (dt, vol) in enumerate(spy_vols.items()):
                if i < 252:  # Need at least 1yr history
                    spy_vol_map[dt] = 0.5  # neutral
                else:
                    past = spy_vols.iloc[:i+1]
                    pct = (past < vol).sum() / len(past)
                    spy_vol_map[dt] = pct

    # State
    cash = float(starting_cash)
    csp_positions = {}
    share_positions = {}
    assignment_dates = deque()

    trades = []
    daily_equity = []
    overlay_log = []  # Track when overlays activate

    # Trailing stop state
    halted = False
    trough_equity = starting_cash

    @dataclass
    class CSPPos:
        ticker: str
        strike: float
        premium: float
        entry_date: object
        expiry_date: object
        margin_held: float

    @dataclass
    class SharePos:
        ticker: str
        shares: int
        cost_basis: float
        entry_date: object
        cc_strike: float = 0.0
        cc_premium: float = 0.0
        cc_expiry: object = None
        has_cc: bool = False

    import random

    for di, date in enumerate(all_dates):
        is_bear = spy_map.get(date, 0) == 1

        # Mark to market
        nav = cash
        for pos in csp_positions.values():
            nav += pos.margin_held
        for t, pos in share_positions.items():
            if t in ticker_data and date in ticker_data[t].index:
                px = ticker_data[t].loc[date]["close"]
                nav += pos.shares * px
            else:
                nav += pos.shares * pos.cost_basis

        daily_equity.append({"date": date, "equity": nav})

        # === COMPUTE OVERLAY SCALE ===
        scale = 1.0
        overlay_reason = ""

        if overlay_type == "eq_brake" or overlay_type == "combined":
            # Equity curve brake
            if len(daily_equity) > eq_brake_lookback:
                recent_peak = max(e["equity"] for e in daily_equity[-eq_brake_lookback:])
                if nav < recent_peak * (1 - eq_brake_threshold):
                    scale = min(scale, eq_brake_scale)
                    overlay_reason += f"eq_brake({nav:.0f}<{recent_peak*(1-eq_brake_threshold):.0f}) "

        if overlay_type == "vol_regime" or overlay_type == "combined":
            # Vol-regime scaling
            vol_pct = spy_vol_map.get(date, 0.5)
            if vol_pct >= vol_percentile_halt:
                scale = min(scale, vol_scale_halt)
                overlay_reason += f"vol_halt(p={vol_pct:.2f}) "
            elif vol_pct >= vol_percentile_high:
                scale = min(scale, vol_scale_high)
                overlay_reason += f"vol_high(p={vol_pct:.2f}) "

        if overlay_type == "trailing_stop":
            # Portfolio trailing stop
            if len(daily_equity) > 1:
                all_eq = [e["equity"] for e in daily_equity]
                peak = max(all_eq)
                dd = (nav - peak) / peak

                if halted:
                    # Check recovery
                    recovery = (nav - trough_equity) / trough_equity if trough_equity > 0 else 0
                    if recovery >= trailing_recovery_pct:
                        halted = False
                        overlay_reason += f"trailing_resume(rec={recovery:.1%}) "
                    else:
                        scale = 0.0
                        trough_equity = min(trough_equity, nav)
                        overlay_reason += f"trailing_halted(dd={dd:.1%}) "
                elif dd < -trailing_stop_pct:
                    halted = True
                    trough_equity = nav
                    scale = 0.0
                    overlay_reason += f"trailing_triggered(dd={dd:.1%}) "

        if overlay_reason:
            overlay_log.append({"date": date, "scale": scale, "reason": overlay_reason.strip(), "nav": nav})

        # Process CSP expirations (unchanged)
        expired_csps = [t for t, pos in csp_positions.items() if date >= pos.expiry_date]
        for t in expired_csps:
            pos = csp_positions[t]
            if t not in ticker_data or date not in ticker_data[t].index:
                cash += pos.margin_held
                trades.append({"date": date, "ticker": t, "action": "csp_expired_otm",
                              "pnl": pos.premium * 100})
                del csp_positions[t]
                continue

            px = ticker_data[t].loc[date]["close"]

            if px <= pos.strike:
                while assignment_dates and (date - assignment_dates[0]).days > 5:
                    assignment_dates.popleft()

                if (len(assignment_dates) >= max_assignments_5d or
                    len(share_positions) >= max_share_positions):
                    intrinsic = (pos.strike - px) * 100
                    loss = intrinsic - pos.premium * 100 + COST_PER_CONTRACT
                    cash += pos.margin_held
                    cash -= loss
                    trades.append({"date": date, "ticker": t, "action": "assignment_refused",
                                  "pnl": -(loss / 100)})
                    del csp_positions[t]
                    continue

                share_cost = pos.strike * 100 + COST_PER_CONTRACT
                cash += pos.margin_held
                cash -= share_cost
                share_positions[t] = SharePos(
                    ticker=t, shares=100,
                    cost_basis=pos.strike - pos.premium,
                    entry_date=date,
                )
                assignment_dates.append(date)
                trades.append({"date": date, "ticker": t, "action": "assigned"})
                del csp_positions[t]
            else:
                cash += pos.margin_held
                trades.append({"date": date, "ticker": t, "action": "csp_expired_otm",
                              "pnl": pos.premium})
                del csp_positions[t]

        # Process CC expirations (unchanged)
        for t in list(share_positions.keys()):
            pos = share_positions[t]
            if not pos.has_cc or pos.cc_expiry is None or date < pos.cc_expiry:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                pos.has_cc = False
                continue
            px = ticker_data[t].loc[date]["close"]
            if px >= pos.cc_strike:
                proceeds = pos.cc_strike * 100 - COST_PER_CONTRACT
                cash += proceeds
                pnl = (pos.cc_strike - pos.cost_basis) * 100 + pos.cc_premium * 100
                trades.append({"date": date, "ticker": t, "action": "called_away", "pnl": pnl / 100})
                del share_positions[t]
            else:
                pos.cost_basis -= pos.cc_premium
                pos.has_cc = False
                trades.append({"date": date, "ticker": t, "action": "cc_expired_otm"})

        # Loss-cut on shares (unchanged)
        for t in list(share_positions.keys()):
            pos = share_positions[t]
            if pos.has_cc:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                continue
            px = ticker_data[t].loc[date]["close"]
            pnl_pct = (px - pos.cost_basis) / pos.cost_basis if pos.cost_basis > 0 else 0
            if pnl_pct <= loss_cut_pct:
                proceeds = px * 100 - COST_PER_CONTRACT
                cash += proceeds
                realized = (px - pos.cost_basis) * 100
                trades.append({"date": date, "ticker": t, "action": "loss_cut", "pnl": realized / 100})
                del share_positions[t]

        # Profit-take on CSPs (unchanged)
        for t in list(csp_positions.keys()):
            pos = csp_positions[t]
            if t not in ticker_data or date not in ticker_data[t].index:
                continue
            row = ticker_data[t].loc[date]
            px = row["close"]
            sigma = row.get("sigma", 0.3) if isinstance(row, pd.Series) else 0.3
            dte_remain = (pos.expiry_date - date).days
            if dte_remain <= 0:
                continue
            current_val = bs_price(px, pos.strike, dte_remain / 365, sigma)
            if current_val <= pos.premium * (1 - profit_take):
                buyback = current_val * 100 + COST_PER_CONTRACT
                cash += pos.margin_held
                cash -= buyback
                profit = (pos.premium - current_val) * 100 - 2 * COST_PER_CONTRACT
                trades.append({"date": date, "ticker": t, "action": "profit_take", "pnl": profit / 100})
                del csp_positions[t]

        # Bear protection (unchanged)
        if is_bear and bear_mode == "liq_csp_only":
            for t in list(csp_positions.keys()):
                pos = csp_positions[t]
                if t not in ticker_data or date not in ticker_data[t].index:
                    continue
                row = ticker_data[t].loc[date]
                px = row["close"]
                sigma = row.get("sigma", 0.3) if isinstance(row, pd.Series) else 0.3
                dte_remain = (pos.expiry_date - date).days
                if dte_remain <= 0:
                    continue
                current_val = bs_price(px, pos.strike, dte_remain / 365, sigma)
                buyback = current_val * 100 + COST_PER_CONTRACT
                cash += pos.margin_held
                cash -= buyback
                pnl = (pos.premium - current_val) * 100 - 2 * COST_PER_CONTRACT
                trades.append({"date": date, "ticker": t, "action": "bear_close", "pnl": pnl / 100})
                del csp_positions[t]

        # Write CCs on shares (unchanged)
        for t in list(share_positions.keys()):
            pos = share_positions[t]
            if pos.has_cc:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                continue
            row = ticker_data[t].loc[date]
            px = row["close"]
            sigma = row.get("sigma", 0.3) if isinstance(row, pd.Series) else 0.3
            if pd.isna(sigma) or sigma < 0.05:
                continue
            T = dte_target / 365
            K = find_strike(px, sigma, T, CALL_DELTA, kind="call")
            premium = bs_price(px, K, T, sigma, kind="call")
            premium = max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)
            if premium < 0.10:
                continue
            expiry = date + pd.Timedelta(days=dte_target)
            pos.has_cc = True
            pos.cc_strike = K
            pos.cc_premium = premium
            pos.cc_expiry = expiry
            cash += premium * 100 - COST_PER_CONTRACT
            trades.append({"date": date, "ticker": t, "action": "sell_cc", "premium": premium})

        # === OPEN NEW CSPS (with overlay scaling) ===
        if is_bear and bear_mode != "none":
            continue

        if scale <= 0:
            continue  # Overlay says halt

        current_margin_used = sum(p.margin_held for p in csp_positions.values())
        # Scale the margin cap by overlay factor
        effective_margin_cap = margin_cap * scale
        available_margin = nav * effective_margin_cap - current_margin_used
        per_name_limit = nav * per_name_pct * scale

        if available_margin <= 0:
            continue

        rng = random.Random(di)
        candidates = list(tickers_available)
        rng.shuffle(candidates)

        for t in candidates:
            if t in csp_positions or t in share_positions:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                continue

            # Earnings avoidance
            if earnings_lookup is not None and t in earnings_lookup:
                _ed = earnings_lookup[t]
                _ws = np.datetime64(date) - np.timedelta64(earnings_buffer_days, 'D')
                _we = np.datetime64(date) + np.timedelta64(dte_target + earnings_buffer_days, 'D')
                _idx_s = np.searchsorted(_ed, _ws, side='left')
                _idx_e = np.searchsorted(_ed, _we, side='right')
                if _idx_e > _idx_s:
                    continue

            row = ticker_data[t].loc[date]
            px = row["close"]
            sigma = row.get("sigma", 0.3) if isinstance(row, pd.Series) else 0.3
            vix = row.get("vix", 20.0) if isinstance(row, pd.Series) else 20.0

            if vix > VIX_MAX:
                continue
            if pd.isna(sigma) or sigma < 0.05:
                continue

            notional = px * 100
            margin_req = notional * MARGIN_REQ_PCT

            if margin_req > per_name_limit:
                continue
            if margin_req > available_margin:
                continue
            if cash < margin_req:
                continue

            T = dte_target / 365
            K = find_strike(px, sigma, T, put_delta, kind="put")
            premium = bs_price(px, K, T, sigma)
            premium = max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)

            if premium < 0.10:
                continue

            expiry = date + pd.Timedelta(days=dte_target)
            cash -= margin_req
            cash += premium * 100 - COST_PER_CONTRACT

            csp_positions[t] = CSPPos(
                ticker=t, strike=K, premium=premium,
                entry_date=date, expiry_date=expiry,
                margin_held=margin_req,
            )
            available_margin -= margin_req
            trades.append({"date": date, "ticker": t, "action": "sell_csp",
                          "strike": K, "premium": premium, "margin": margin_req})

    return daily_equity, trades, overlay_log


def per_year_analysis(daily_equity):
    """Break down performance by calendar year."""
    eq = pd.DataFrame(daily_equity)
    eq["year"] = eq["date"].dt.year
    eq["return"] = eq["equity"].pct_change()

    results = {}
    for year, grp in eq.groupby("year"):
        if len(grp) < 20:
            continue
        rets = grp["return"].dropna()
        mu = rets.mean() * 252
        std = rets.std() * np.sqrt(252)
        sharpe = mu / std if std > 0 else 0

        start_eq = grp["equity"].iloc[0]
        end_eq = grp["equity"].iloc[-1]
        yr_ret = (end_eq / start_eq - 1) * 100

        # Max DD within year
        peak = grp["equity"].cummax()
        dd = (grp["equity"] / peak - 1).min() * 100

        results[year] = {
            "return_pct": round(yr_ret, 1),
            "sharpe": round(sharpe, 2),
            "max_dd_pct": round(dd, 1),
            "n_days": len(grp),
        }
    return results


def main():
    log.info("=" * 60)
    log.info("WHEEL V3 DRAWDOWN PROTECTION OVERLAY TEST")
    log.info("=" * 60)

    # Load data
    log.info("Loading data...")
    prices, spy_regime, sector_map = load_all_data()

    # Load earnings
    log.info("Loading earnings dates...")
    earnings_raw = download_earnings_dates(prices["ticker"].unique())
    earnings_lookup = build_earnings_lookup(earnings_raw)
    log.info(f"Earnings data for {len(earnings_lookup)} tickers")

    # Base config (validated winner: 30-delta, 2d earnings buffer)
    base_params = dict(
        starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
        put_delta=0.30, dte_target=14, profit_take=0.65,
        bear_mode="liq_csp_only", max_assignments_5d=3,
        max_share_positions=5, loss_cut_pct=-0.15,
        min_price=10.0, max_price=500.0,
        earnings_lookup=earnings_lookup, earnings_buffer_days=2,
    )

    # Test configurations
    overlays = [
        {"name": "BASELINE (no overlay)", "overlay_type": "none"},

        # Equity curve brake variants
        {"name": "EQ_BRAKE 60d/5%/0.5x", "overlay_type": "eq_brake",
         "eq_brake_lookback": 60, "eq_brake_threshold": 0.05, "eq_brake_scale": 0.5},
        {"name": "EQ_BRAKE 60d/3%/0.5x", "overlay_type": "eq_brake",
         "eq_brake_lookback": 60, "eq_brake_threshold": 0.03, "eq_brake_scale": 0.5},
        {"name": "EQ_BRAKE 20d/5%/0.5x", "overlay_type": "eq_brake",
         "eq_brake_lookback": 20, "eq_brake_threshold": 0.05, "eq_brake_scale": 0.5},

        # Vol regime scaling
        {"name": "VOL_REGIME 70/90 pct", "overlay_type": "vol_regime",
         "vol_percentile_high": 0.70, "vol_percentile_halt": 0.90,
         "vol_scale_high": 0.50, "vol_scale_halt": 0.25},
        {"name": "VOL_REGIME 60/80 pct", "overlay_type": "vol_regime",
         "vol_percentile_high": 0.60, "vol_percentile_halt": 0.80,
         "vol_scale_high": 0.50, "vol_scale_halt": 0.0},

        # Combined
        {"name": "COMBINED 60d/5%+vol70/90", "overlay_type": "combined",
         "eq_brake_lookback": 60, "eq_brake_threshold": 0.05, "eq_brake_scale": 0.5,
         "vol_percentile_high": 0.70, "vol_percentile_halt": 0.90,
         "vol_scale_high": 0.50, "vol_scale_halt": 0.25},

        # Trailing stop variants
        {"name": "TRAILING_STOP 10%/3%rec", "overlay_type": "trailing_stop",
         "trailing_stop_pct": 0.10, "trailing_recovery_pct": 0.03},
        {"name": "TRAILING_STOP 15%/5%rec", "overlay_type": "trailing_stop",
         "trailing_stop_pct": 0.15, "trailing_recovery_pct": 0.05},
        {"name": "TRAILING_STOP 8%/2%rec", "overlay_type": "trailing_stop",
         "trailing_stop_pct": 0.08, "trailing_recovery_pct": 0.02},
    ]

    all_results = []

    for cfg in overlays:
        name = cfg.pop("name")
        log.info(f"\n{'='*50}")
        log.info(f"Running: {name}")
        log.info(f"  Overlay params: {cfg}")

        t0 = time.time()
        equity, trades_list, overlay_activations = run_portfolio_v3_with_overlay(
            prices, spy_regime, sector_map, **base_params, **cfg
        )
        elapsed = time.time() - t0

        metrics = compute_metrics(equity, starting_cash=100_000)
        regime = regime_analysis(equity, spy_regime)
        yearly = per_year_analysis(equity)

        n_activations = len(overlay_activations)
        n_halt_days = sum(1 for a in overlay_activations if a["scale"] == 0)
        n_scaled_days = sum(1 for a in overlay_activations if 0 < a["scale"] < 1)

        result = {
            "name": name,
            "metrics": metrics,
            "regime": regime,
            "yearly": yearly,
            "n_overlay_activations": n_activations,
            "n_halt_days": n_halt_days,
            "n_scaled_days": n_scaled_days,
            "n_trades": len([t for t in trades_list if t["action"] == "sell_csp"]),
            "elapsed_s": round(elapsed, 1),
        }
        all_results.append(result)

        log.info(f"  CAGR: {metrics.get('cagr_pct', '?')}% | Sharpe: {metrics.get('sharpe', '?')} | "
                 f"Sortino: {metrics.get('sortino', '?')} | MaxDD: {metrics.get('max_dd_pct', '?')}% | "
                 f"Calmar: {metrics.get('calmar', '?')}")
        log.info(f"  Overlay active: {n_activations} days ({n_halt_days} halted, {n_scaled_days} scaled)")
        log.info(f"  CSP trades: {result['n_trades']} | Elapsed: {elapsed:.1f}s")

        # Per-year breakdown
        for yr, yr_data in sorted(yearly.items()):
            marker = "***" if yr_data["return_pct"] < 0 else ""
            log.info(f"    {yr}: {yr_data['return_pct']:+.1f}% (Sharpe {yr_data['sharpe']:.2f}, "
                     f"DD {yr_data['max_dd_pct']:.1f}%){marker}")

        # Save equity curve
        eq_df = pd.DataFrame(equity)
        eq_df.to_parquet(OUT_DIR / f"equity_{name.replace(' ', '_').replace('/', '_')}.parquet", index=False)

        # Restore name for next iteration record
        cfg["name"] = name

    # === SUMMARY TABLE ===
    log.info("\n" + "=" * 80)
    log.info("SUMMARY COMPARISON")
    log.info("=" * 80)

    header = f"{'Config':<35} {'CAGR':>6} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Calmar':>7} {'PF':>5} {'Trades':>7} {'Halts':>6}"
    log.info(header)
    log.info("-" * len(header))

    for r in all_results:
        m = r["metrics"]
        log.info(f"{r['name']:<35} {m.get('cagr_pct','?'):>5.1f}% {m.get('sharpe','?'):>7.2f} "
                 f"{m.get('sortino','?'):>8.2f} {m.get('max_dd_pct','?'):>6.1f}% "
                 f"{m.get('calmar','?'):>7.2f} {m.get('profit_factor','?'):>5.2f} "
                 f"{r['n_trades']:>7} {r['n_halt_days']:>6}")

    # === BEAR YEAR COMPARISON ===
    log.info("\n" + "=" * 80)
    log.info("BEAR YEAR PERFORMANCE (years where baseline loses money)")
    log.info("=" * 80)

    baseline_yearly = all_results[0]["yearly"]
    bear_years = [yr for yr, d in baseline_yearly.items() if d["return_pct"] < 0]

    if bear_years:
        header2 = f"{'Config':<35} " + " ".join(f"{yr:>12}" for yr in sorted(bear_years))
        log.info(header2)
        log.info("-" * len(header2))

        for r in all_results:
            yr_strs = []
            for yr in sorted(bear_years):
                if yr in r["yearly"]:
                    yr_strs.append(f"{r['yearly'][yr]['return_pct']:>+10.1f}%")
                else:
                    yr_strs.append(f"{'N/A':>11}")
            log.info(f"{r['name']:<35} " + " ".join(yr_strs))

    # === BEST OVERLAY SELECTION ===
    log.info("\n" + "=" * 80)
    log.info("VERDICT")
    log.info("=" * 80)

    baseline = all_results[0]["metrics"]
    best_idx = 0
    best_score = 0

    for i, r in enumerate(all_results[1:], 1):
        m = r["metrics"]
        # Score: improvement in Sharpe + improvement in Calmar + reduction in MaxDD
        sharpe_imp = (m.get("sharpe", 0) - baseline.get("sharpe", 0)) / max(baseline.get("sharpe", 1), 0.01)
        calmar_imp = (m.get("calmar", 0) - baseline.get("calmar", 0)) / max(baseline.get("calmar", 1), 0.01)
        dd_imp = (abs(baseline.get("max_dd_pct", 0)) - abs(m.get("max_dd_pct", 0))) / max(abs(baseline.get("max_dd_pct", 1)), 0.01)
        cagr_loss = (m.get("cagr_pct", 0) - baseline.get("cagr_pct", 0)) / max(baseline.get("cagr_pct", 1), 0.01)

        # Penalize if CAGR drops too much (>30% relative)
        score = sharpe_imp + calmar_imp + dd_imp
        if cagr_loss < -0.30:
            score -= 1.0  # Heavy penalty

        log.info(f"  {r['name']}: score={score:.3f} (Sharpe Δ={sharpe_imp:+.2f}, Calmar Δ={calmar_imp:+.2f}, "
                 f"DD Δ={dd_imp:+.2f}, CAGR Δ={cagr_loss:+.2f})")

        if score > best_score:
            best_score = score
            best_idx = i

    if best_score > 0:
        best = all_results[best_idx]
        log.info(f"\n  WINNER: {best['name']}")
        log.info(f"  Sharpe: {best['metrics']['sharpe']} (vs baseline {baseline['sharpe']})")
        log.info(f"  MaxDD: {best['metrics']['max_dd_pct']}% (vs baseline {baseline['max_dd_pct']}%)")
        log.info(f"  CAGR: {best['metrics']['cagr_pct']}% (vs baseline {baseline['cagr_pct']}%)")
    else:
        log.info("\n  NO OVERLAY IMPROVES ON BASELINE. Bear-year losses are structural.")

    # Save full results
    with open(OUT_DIR / "dd_protection_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    log.info(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
