"""
wheel_dd_overlay_v1.py — ADDITIVE risk-overlay test for the canonical v8_WF
wheel config (Tier2_Balanced_FW). Follows wheel_v8_robustness_v1 verdict
ROBUST with caveat: realistic forward MaxDD ~-15% (4 of 11 parameter
neighbors roughly double the baseline -8.4% DD).

PRE-REGISTERED DESIGN (written before any cell is run — 2026-06-11):

Question: can an overlay that scales NEW-POSITION exposure only (no change
to any strategy dial — caveat 2 of the robustness run forbids retuning
roll/PT/DTE) pull the realistic worst-case drawdown down materially
without costing >10% Sharpe?

Overlays (all signals point-in-time, computed from data <= decision day t;
existing positions are ALWAYS managed by unchanged v8_WF rules — overlays
gate/scale NEW short puts only):

  O1 equity-curve brake (grid N x X):
     if strategy MTM equity at t < (1 - X) * max(equity over trailing N
     trading days incl t)  ->  scale = 0.5, else 1.0.
     Grid: N in {20, 60}, X in {3%, 5%}  ->  4 variants.
  O2 vol-regime scaling:
     p_t = expanding percentile (past-only, >=252 prior obs, SPY history
     from 2015) of SPY 20d realized vol at t.
     scale = 1.00 if p_t < 0.70; 0.50 if 0.70 <= p_t < 0.90; 0.25 else.
  O3 crisis circuit-breaker:
     scale = 0 (halt NEW positions) if SPY close_t < 200d MA_t AND
     p_t > 0.80; else 1.0.

Exposure scaling implementation (only active when scale < 1; at scale == 1
the code path is byte-identical to backtest.wheel_engine.run_wheel, which
the (baseline_cfg, no-overlay) replication cell verifies):
  - effective max concurrent names = max(1, floor(max_names * scale))
    (new opens blocked while book is above the scaled cap; existing
    positions untouched),
  - per-name allocation cap 0.15 * equity * scale, with NO 1-contract
    floor while scaled (name skipped if 0 contracts fit),
  - scale == 0 -> no new CSPs at all that day.

Configs evaluated (5): the v8_WF baseline + the 4 high-DD parameter
neighbors from wheel_v8_robustness_v1 (dte_24_36 -15.6%, roll_3 -16.0%,
pt_078 -14.1%, ivr_030 -13.8%) — the overlay must cap THEIR DD too.

Cells: 5 configs x 7 overlays (none + 4xO1 + O2 + O3) = 35 (cap 40).
Combos only if singles show promise (would be a follow-up phase).

PRE-REGISTERED VERDICT RULES (per overlay, realized-cash basis like the
robustness run):
  - replication: (baseline, none) realized Sharpe within +/-0.05 of 1.4915
    else harness void. Neighbor none-cells must match robustness DDs
    within +/-0.01.
  - sharpe_cost = 1 - Sharpe(baseline+overlay)/Sharpe(baseline,none)
  - worst_neighbor_dd = min realized MaxDD across the 4 neighbors w/ overlay
  - SUCCESS  : worst_neighbor_dd > -0.10 AND sharpe_cost < 0.10 AND
               regime_gap(baseline+overlay) <= 0.50
  - PARTIAL  : worst_neighbor_dd improved >= 25% vs -0.160 (i.e. > -0.120)
               AND sharpe_cost < 0.10 AND gap <= 0.50
  - FAIL     : otherwise. Honest negative is acceptable.

SLIDING-WINDOW note (HC #0): no parameter is selected by this harness for
deployment without the pre-registered grid above; evaluation windows are
fixed calendar strata of the same 2020-2025 simulation. Skew calibration
inside is the leak-free walk-forward schedule (v8_WF).

Usage (from /home/jupiter/Lvl3Quant/wheel_strategy_v1):
    python3 -m strategy.wheel_dd_overlay_v1 \
        --out results/wheel_dd_overlay_v1 --workers 4
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

START = "2020-01-01"
END = "2025-12-31"
CAPITAL = 100_000.0

CRISIS_WINDOWS = {
    "covid_2020": ("2020-02-19", "2020-04-30"),
    "bear_2022": ("2022-01-03", "2022-10-14"),
    "unwind_aug2024": ("2024-07-15", "2024-08-15"),
}

# Robustness-run realized MaxDDs we must replicate (tolerance 0.01) and
# protect against:
ROBUSTNESS_NONE_DD = {
    "baseline": -0.08350859213087147,
    "dte_24_36": -0.1555774581657864,
    "roll_3": -0.16001770797531634,
    "pt_078": -0.14062860750669648,
    "ivr_030": -0.1381404795509511,
}
BASELINE_SHARPE_REF = 1.491544910957763

# ---------------- pre-registered grid ----------------
# configs: (name, cfg_overrides, iv_rank_floor_override)
CONFIGS = [
    ("baseline",  {},                              None),
    ("dte_24_36", {"dte_min": 24, "dte_max": 36},  None),
    ("roll_3",    {"roll_dte_trigger": 3},         None),
    ("pt_078",    {"profit_take_pct": 0.78},       None),
    ("ivr_030",   {},                              0.30),
]

# overlays: (name, spec-dict or None)
OVERLAYS = [
    ("none",      None),
    ("o1_n20_x3", {"kind": "o1", "N": 20, "X": 0.03}),
    ("o1_n20_x5", {"kind": "o1", "N": 20, "X": 0.05}),
    ("o1_n60_x3", {"kind": "o1", "N": 60, "X": 0.03}),
    ("o1_n60_x5", {"kind": "o1", "N": 60, "X": 0.05}),
    ("o2_vol",    {"kind": "o2"}),
    ("o3_cb",     {"kind": "o3"}),
    # combo phase (run via --overlays combo_* only if singles show promise)
    ("combo_o1n20x3_o3", {"kind": "combo", "parts": [
        {"kind": "o1", "N": 20, "X": 0.03}, {"kind": "o3"}]}),
    ("combo_o1n20x5_o3", {"kind": "combo", "parts": [
        {"kind": "o1", "N": 20, "X": 0.05}, {"kind": "o3"}]}),
]
DEFAULT_OVERLAYS = ["none", "o1_n20_x3", "o1_n20_x5", "o1_n60_x3",
                    "o1_n60_x5", "o2_vol", "o3_cb"]


# ---------------- SPY point-in-time signals ----------------

def build_spy_signals(spy_close: pd.Series) -> pd.DataFrame:
    """vol_pct = expanding past-only percentile of 20d realized vol;
    below_200ma = close_t < 200d MA_t. All values use data <= t only."""
    s = spy_close.sort_index()
    ret = s.pct_change()
    vol = ret.rolling(20).std() * np.sqrt(252)
    v = vol.values
    pct = np.full(len(v), np.nan)
    hist = []
    for i, x in enumerate(v):
        if np.isfinite(x):
            if len(hist) >= 252:
                pct[i] = np.mean(np.array(hist) <= x)
            hist.append(x)
    ma200 = s.rolling(200).mean()
    return pd.DataFrame({
        "vol_pct": pct,
        "below_200ma": (s < ma200).values,
    }, index=s.index)


def overlay_scale(spec: dict | None, dt, equity_hist: list,
                  sig: "pd.DataFrame | None") -> float:
    """Exposure scale in [0, 1] for NEW positions on day dt.
    equity_hist: engine's (date, mtm_equity) list INCLUDING today's mark."""
    if spec is None:
        return 1.0
    kind = spec["kind"]
    if kind == "combo":
        sc = 1.0
        for part in spec["parts"]:
            sc = min(sc, overlay_scale(part, dt, equity_hist, sig))
        return sc
    if kind == "o1":
        N, X = spec["N"], spec["X"]
        eqs = [e for _, e in equity_hist[-N:]]
        if len(eqs) >= 2:
            peak = max(eqs)
            if peak > 0 and eqs[-1] < (1.0 - X) * peak:
                return 0.5
        return 1.0
    # O2 / O3 need SPY signals; missing signal -> no scaling (fail open,
    # matches burn-in period before 252 obs which is all pre-2017 anyway)
    if sig is None or dt not in sig.index:
        return 1.0
    row = sig.loc[dt]
    p = row["vol_pct"]
    if kind == "o2":
        if not np.isfinite(p):
            return 1.0
        if p >= 0.90:
            return 0.25
        if p >= 0.70:
            return 0.50
        return 1.0
    if kind == "o3":
        if np.isfinite(p) and p > 0.80 and bool(row["below_200ma"]):
            return 0.0
        return 1.0
    raise ValueError(f"unknown overlay kind {kind}")


# ---------------- overlay-aware engine ----------------
# Copy of backtest.wheel_engine.run_wheel (2026-06-11 state) with the
# overlay hook in step 3. All helpers are imported from wheel_engine so
# pricing/slippage/skew behavior is shared. At scale == 1.0 the open logic
# is identical to the original; (baseline, none) replication cell verifies.

def run_wheel_overlay(cfg, prices, iv, macro, fundamentals, universe,
                      starting_cash=100_000.0, start=None, end=None,
                      overlay_spec: dict | None = None,
                      spy_signals: "pd.DataFrame | None" = None) -> dict:
    from backtest import wheel_engine as we
    from backtest.wheel_engine import (
        Position, TradeLedgerEntry, bs_price, _sigma_at_strike,
        _strike_from_delta_skew, _slippage_per_share, _equity_mtm,
        _select_target_dte, _sector_exposure, WheelState)
    from backtest.costs import per_contract_cost, share_taf_cost

    prices = prices.copy(); iv = iv.copy(); macro = macro.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    iv["date"] = pd.to_datetime(iv["date"])
    macro["date"] = pd.to_datetime(macro["date"])
    if start is not None:
        s = pd.Timestamp(start)
        prices = prices[prices["date"] >= s]
        iv = iv[iv["date"] >= s]
        macro = macro[macro["date"] >= s]
    if end is not None:
        e = pd.Timestamp(end)
        prices = prices[prices["date"] <= e]
        iv = iv[iv["date"] <= e]
        macro = macro[macro["date"] <= e]

    px_by_date = {d: g.set_index("ticker")["close"].to_dict()
                  for d, g in prices.groupby("date")}
    sigma_by_date = {d: g.set_index("ticker")["sigma"].to_dict()
                     for d, g in iv.groupby("date")}
    iv_rank_by_date = {d: g.set_index("ticker")["iv_rank"].to_dict()
                       for d, g in iv.groupby("date")}
    macro_by_date = macro.set_index("date").to_dict("index")

    sector_of = dict(zip(universe["ticker"],
                         universe.get("sector", pd.Series(["Unknown"] * len(universe)))))
    fund_score_of = dict(zip(fundamentals["ticker"],
                             fundamentals.get("fund_score", pd.Series([50.0] * len(fundamentals)))))

    all_dates = sorted(prices["date"].unique())
    state = WheelState(cash=starting_cash)

    assignment_count = 0
    called_away_count = 0
    csp_opened = 0
    cc_opened = 0
    scale_log = []  # (date, scale) for diagnostics

    for di, dt in enumerate(all_dates):
        if we.USE_SKEW and we._iv_skew_mod is not None and \
                getattr(we._iv_skew_mod, "SCHEDULE", None):
            we._iv_skew_mod.set_asof(dt)
        date_px = px_by_date.get(dt, {})
        date_px["__date__"] = dt
        date_sigma = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})
        lag = max(int(getattr(cfg, "macro_lag_days", 0) or 0), 0)
        if lag > 0:
            m = macro_by_date.get(all_dates[di - lag], {}) if di >= lag else {}
        else:
            m = macro_by_date.get(dt, {})
        vix = m.get("vix", float("nan"))
        naaim = m.get("naaim", float("nan"))

        # -- 1) Update / close existing positions (UNCHANGED v8_WF rules)
        to_remove = []
        for tk, p in list(state.positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (p.expiry - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, p.open_sigma) or p.open_sigma or 0.20

            if p.state == "short_put":
                if T_days <= 0:
                    if S < p.strike:
                        cost = p.strike * 100 * p.contracts
                        state.cash -= cost
                        assignment_count += 1
                        p.state = "long_shares"
                        p.share_cost_basis = p.strike - p.open_price
                    else:
                        realized = p.open_price * 100 * p.contracts - per_contract_cost() * p.contracts
                        state.ledger.append(TradeLedgerEntry(
                            open_date=p.open_date, close_date=dt, ticker=tk, kind="CSP",
                            strike=p.strike, contracts=p.contracts,
                            dte_open=(p.expiry - p.open_date).days,
                            delta_open=p.target_delta,
                            premium_received=p.open_price * 100 * p.contracts,
                            premium_closed_at=0.0, realized_pnl=realized,
                            assigned=False, called_away=False,
                        ))
                        to_remove.append(tk)
                else:
                    sigma_k = _sigma_at_strike(sigma_atm, S, p.strike, T, ticker=tk)
                    cur = bs_price(S, p.strike, T, sigma_k, r=cfg.r, kind="put")
                    captured = (p.open_price - cur) / max(p.open_price, 1e-6)
                    if captured >= cfg.profit_take_pct or T_days <= cfg.roll_dte_trigger:
                        slip = _slippage_per_share(cur) * 100 * p.contracts
                        cost = cur * 100 * p.contracts + per_contract_cost() * p.contracts + slip
                        realized = p.open_price * 100 * p.contracts - cost
                        state.cash -= (cur * 100 * p.contracts
                                       + per_contract_cost() * p.contracts + slip)
                        state.ledger.append(TradeLedgerEntry(
                            open_date=p.open_date, close_date=dt, ticker=tk, kind="CSP",
                            strike=p.strike, contracts=p.contracts,
                            dte_open=(p.expiry - p.open_date).days,
                            delta_open=p.target_delta,
                            premium_received=p.open_price * 100 * p.contracts,
                            premium_closed_at=cur * 100 * p.contracts,
                            realized_pnl=realized,
                            assigned=False, called_away=False,
                        ))
                        to_remove.append(tk)

            elif p.state == "short_call":
                if T_days <= 0:
                    if S > p.strike:
                        proceeds = p.strike * 100 * p.contracts
                        state.cash += proceeds - share_taf_cost(proceeds)
                        called_away_count += 1
                        realized_call = p.open_price * 100 * p.contracts - per_contract_cost() * p.contracts
                        realized_shares = (p.strike - p.share_cost_basis) * 100 * p.contracts
                        state.ledger.append(TradeLedgerEntry(
                            open_date=p.open_date, close_date=dt, ticker=tk, kind="CC",
                            strike=p.strike, contracts=p.contracts,
                            dte_open=(p.expiry - p.open_date).days,
                            delta_open=p.target_delta,
                            premium_received=p.open_price * 100 * p.contracts,
                            premium_closed_at=0.0,
                            realized_pnl=realized_call + realized_shares,
                            assigned=False, called_away=True,
                        ))
                        to_remove.append(tk)
                    else:
                        realized = p.open_price * 100 * p.contracts - per_contract_cost() * p.contracts
                        state.ledger.append(TradeLedgerEntry(
                            open_date=p.open_date, close_date=dt, ticker=tk, kind="CC",
                            strike=p.strike, contracts=p.contracts,
                            dte_open=(p.expiry - p.open_date).days,
                            delta_open=p.target_delta,
                            premium_received=p.open_price * 100 * p.contracts,
                            premium_closed_at=0.0, realized_pnl=realized,
                            assigned=False, called_away=False,
                        ))
                        p.state = "long_shares"
                        p.open_price = 0.0
                        p.strike = 0.0
                        p.expiry = dt
                else:
                    sigma_k = _sigma_at_strike(sigma_atm, S, p.strike, T, ticker=tk)
                    cur = bs_price(S, p.strike, T, sigma_k, r=cfg.r, kind="call")
                    captured = (p.open_price - cur) / max(p.open_price, 1e-6)
                    if captured >= cfg.profit_take_pct or T_days <= cfg.roll_dte_trigger:
                        slip = _slippage_per_share(cur) * 100 * p.contracts
                        cost = cur * 100 * p.contracts + per_contract_cost() * p.contracts + slip
                        realized = p.open_price * 100 * p.contracts - cost
                        state.cash -= cost
                        state.ledger.append(TradeLedgerEntry(
                            open_date=p.open_date, close_date=dt, ticker=tk, kind="CC",
                            strike=p.strike, contracts=p.contracts,
                            dte_open=(p.expiry - p.open_date).days,
                            delta_open=p.target_delta,
                            premium_received=p.open_price * 100 * p.contracts,
                            premium_closed_at=cur * 100 * p.contracts,
                            realized_pnl=realized,
                            assigned=False, called_away=False,
                        ))
                        p.state = "long_shares"
                        p.open_price = 0.0
                        p.strike = 0.0
                        p.expiry = dt

        for tk in to_remove:
            del state.positions[tk]

        # -- 1b) Share stop-loss (off by default in v8_WF; kept for parity)
        stop_pct = float(getattr(cfg, "share_stop_loss_pct", 0.0) or 0.0)
        if stop_pct > 0:
            for tk, p in list(state.positions.items()):
                if p.state not in ("long_shares", "short_call"):
                    continue
                S = date_px.get(tk)
                if S is None or np.isnan(S) or p.share_cost_basis <= 0:
                    continue
                if S >= p.share_cost_basis * (1.0 - stop_pct):
                    continue
                if p.state == "short_call":
                    T_days = (p.expiry - dt).days
                    T = max(T_days, 0) / 365.0
                    sigma_atm = date_sigma.get(tk, p.open_sigma) or p.open_sigma or 0.20
                    sigma_k = _sigma_at_strike(sigma_atm, S, p.strike, T, ticker=tk)
                    cur = bs_price(S, p.strike, T, sigma_k, r=cfg.r, kind="call")
                    slip = _slippage_per_share(cur) * 100 * p.contracts
                    cc_buyback = cur * 100 * p.contracts + per_contract_cost() * p.contracts + slip
                    state.cash -= cc_buyback
                    state.ledger.append(TradeLedgerEntry(
                        open_date=p.open_date, close_date=dt, ticker=tk, kind="CC",
                        strike=p.strike, contracts=p.contracts,
                        dte_open=(p.expiry - p.open_date).days,
                        delta_open=p.target_delta,
                        premium_received=p.open_price * 100 * p.contracts,
                        premium_closed_at=cur * 100 * p.contracts,
                        realized_pnl=p.open_price * 100 * p.contracts - cc_buyback,
                        assigned=False, called_away=False,
                        exit_reason="forced_close", sector=p.sector,
                    ))
                proceeds = S * 100 * p.contracts
                state.cash += proceeds - share_taf_cost(proceeds)
                state.ledger.append(TradeLedgerEntry(
                    open_date=p.open_date, close_date=dt, ticker=tk, kind="SHARES",
                    strike=0.0, contracts=p.contracts, dte_open=0, delta_open=0.0,
                    premium_received=0.0, premium_closed_at=0.0,
                    realized_pnl=(S - p.share_cost_basis) * 100 * p.contracts
                                 - share_taf_cost(proceeds),
                    assigned=False, called_away=False,
                    exit_reason="stop_loss", sector=p.sector,
                ))
                del state.positions[tk]

        # -- 2) Re-sell CCs on assigned shares (UNCHANGED)
        for tk, p in list(state.positions.items()):
            if p.state == "long_shares" and (p.expiry <= dt or p.strike == 0.0):
                S = date_px.get(tk)
                sigma = date_sigma.get(tk, 0.25) or 0.25
                if S is None or np.isnan(S) or sigma <= 0:
                    continue
                target_dte = _select_target_dte(cfg)
                T = target_dte / 365.0
                K, sigma_k = _strike_from_delta_skew(
                    S, T, sigma, cfg.call_delta_target, r=cfg.r, kind="call",
                    ticker=tk)
                premium = bs_price(S, K, T, sigma_k, r=cfg.r, kind="call")
                if premium <= 0:
                    continue
                slip = _slippage_per_share(premium) * 100 * p.contracts
                state.cash += premium * 100 * p.contracts - per_contract_cost() * p.contracts - slip
                p.state = "short_call"
                p.strike = K
                p.open_price = premium
                p.open_date = dt
                p.expiry = dt + pd.Timedelta(days=target_dte)
                p.open_underlying = S
                p.open_sigma = sigma
                p.target_delta = cfg.call_delta_target
                cc_opened += 1

        # -- 3) New CSPs — OVERLAY HOOK lives here
        equity = _equity_mtm(state, date_px, date_sigma, cfg.r)
        if np.isnan(equity) or equity <= 0:
            equity = state.cash
        state.equity_curve.append((dt, equity))

        scale = overlay_scale(overlay_spec, dt, state.equity_curve, spy_signals)
        scale_log.append((dt, scale))
        if scale <= 0.0:
            continue  # circuit breaker: no NEW positions today
        eff_max = (cfg.max_concurrent_names if scale >= 1.0
                   else max(1, int(np.floor(cfg.max_concurrent_names * scale))))

        if not np.isnan(vix) and vix > cfg.vix_max_gate:
            continue
        if not np.isnan(naaim) and naaim < cfg.naaim_min_gate:
            continue
        if len(state.positions) >= eff_max:
            continue

        cap_pct = float(getattr(cfg, "max_assigned_notional_pct", 1.0) or 1.0)
        if cap_pct < 1.0:
            assigned_mv = 0.0
            for p in state.positions.values():
                if p.state in ("long_shares", "short_call"):
                    Sp = date_px.get(p.ticker)
                    if Sp is not None and not np.isnan(Sp):
                        assigned_mv += Sp * 100 * p.contracts
            if assigned_mv > cap_pct * max(equity, 1.0):
                continue

        candidates = []
        for tk, S in date_px.items():
            if tk == "__date__":
                continue
            if tk in state.positions:
                continue
            if S is None or np.isnan(S):
                continue
            fs = fund_score_of.get(tk, 50.0)
            if fs < cfg.fund_score_floor:
                continue
            sigma = date_sigma.get(tk)
            if sigma is None or np.isnan(sigma) or sigma <= 0:
                continue
            iv_rk = date_iv_rank.get(tk, 0.5)
            sector = sector_of.get(tk, "Unknown")
            if _sector_exposure(state, sector, equity) > cfg.sector_cap_pct:
                continue
            candidates.append((tk, S, sigma, iv_rk, sector, fs))

        candidates.sort(key=lambda r: (r[3], r[5]), reverse=True)

        slots = eff_max - len(state.positions)
        slots = min(slots, max(1, cfg.max_concurrent_names // 5))

        for tk, S, sigma, iv_rk, sector, fs in candidates[:slots]:
            target_dte = _select_target_dte(cfg)
            T = target_dte / 365.0
            K, sigma_k = _strike_from_delta_skew(
                S, T, sigma, cfg.put_delta_target, r=cfg.r, kind="put",
                ticker=tk)
            premium = bs_price(S, K, T, sigma_k, r=cfg.r, kind="put")
            if premium <= 0:
                continue
            if K is None or not np.isfinite(K) or K <= 0:
                continue
            if not np.isfinite(premium) or not np.isfinite(equity) or equity <= 0:
                continue
            # exposure scaling: allocation cap scaled; no 1-contract floor
            # while scaled (skip instead) — at scale==1 identical to engine
            max_alloc = 0.15 * equity * scale
            if scale >= 1.0:
                n_contracts = max(1, int(max_alloc // (K * 100)))
            else:
                n_contracts = int(max_alloc // (K * 100))
                if n_contracts < 1:
                    continue
            secure_needed = K * 100 * n_contracts
            if secure_needed > state.cash:
                n_contracts = int(state.cash // (K * 100))
                if n_contracts < 1:
                    continue
            if (K * 100 * n_contracts) / max(equity, 1) > 0.15:
                n_contracts = max(1, int(0.15 * equity // (K * 100)))
                if n_contracts < 1:
                    continue
            if (_sector_exposure(state, sector, equity) + (K * 100 * n_contracts) / max(equity, 1)) > cfg.sector_cap_pct:
                continue
            slip = _slippage_per_share(premium) * 100 * n_contracts
            credit = premium * 100 * n_contracts - per_contract_cost() * n_contracts - slip
            state.cash += credit
            state.positions[tk] = Position(
                ticker=tk, sector=sector, state="short_put",
                open_date=dt, expiry=dt + pd.Timedelta(days=target_dte),
                strike=K, contracts=n_contracts, open_price=premium,
                open_underlying=S, open_sigma=sigma,
                target_delta=cfg.put_delta_target,
                profit_take_pct=cfg.profit_take_pct,
                roll_dte_trigger=cfg.roll_dte_trigger,
            )
            csp_opened += 1
            if len(state.positions) >= eff_max:
                break

    if all_dates:
        last_dt = all_dates[-1]
        last_px = px_by_date.get(last_dt, {})
        last_px["__date__"] = last_dt
        last_sigma = sigma_by_date.get(last_dt, {})
        from backtest.wheel_engine import _equity_mtm as _emtm
        final_eq = _emtm(state, last_px, last_sigma, cfg.r)
        state.equity_curve.append((last_dt, final_eq))

    eq_df = pd.DataFrame(state.equity_curve, columns=["date", "equity"]).drop_duplicates("date", keep="last")
    led_df = pd.DataFrame([le.__dict__ for le in state.ledger])
    sc_df = pd.DataFrame(scale_log, columns=["date", "scale"])

    return {
        "equity_curve": eq_df,
        "ledger": led_df,
        "scale_curve": sc_df,
        "csp_opened": csp_opened,
        "cc_opened": cc_opened,
        "assignment_count": assignment_count,
        "called_away_count": called_away_count,
        "final_cash": state.cash,
        "final_equity": eq_df["equity"].iloc[-1] if not eq_df.empty else state.cash,
        "starting_cash": starting_cash,
    }


# ---------------- per-process worker state ----------------
_W = {}


def _worker_init():
    from strategy import iv_skew as _ivs
    from strategy.tier_runner import _load_inputs, _load_spy_close
    from strategy.regime_overlay import build_regime, apply_regime_gate

    _ivs.load_calibration_walkforward()
    data = _load_inputs(modeled=False, smoke=False, real_iv=True)
    regime = build_regime()
    data["macro"] = apply_regime_gate(data["macro"], regime,
                                      vix_force_gate=999.0)
    _W["data"] = data
    spy = _load_spy_close(data)
    _W["spy_close"] = spy
    _W["spy_signals"] = build_spy_signals(spy) if spy is not None else None


def _window_metrics(req: pd.Series, label: str) -> dict:
    from strategy.tier_runner import _sharpe, _sortino, _max_dd
    if len(req) < 3:
        return {f"{label}_sharpe": float("nan"),
                f"{label}_ret_pct": float("nan"),
                f"{label}_max_dd": float("nan")}
    ret = req.pct_change().fillna(0.0)
    return {
        f"{label}_sharpe": _sharpe(ret),
        f"{label}_sortino": _sortino(ret),
        f"{label}_ret_pct": float(req.iloc[-1] / req.iloc[0] - 1.0) * 100.0,
        f"{label}_max_dd": _max_dd(req),
    }


def _concentrations(led: pd.DataFrame) -> dict:
    out = {"day_concentration": float("nan"),
           "ticker_concentration": float("nan")}
    if led is None or led.empty:
        return out
    pos = led[led["realized_pnl"] > 0]
    tot_pos = pos["realized_pnl"].sum()
    if tot_pos > 0:
        by_day = pos.groupby("close_date")["realized_pnl"].sum()
        out["day_concentration"] = float(by_day.max() / tot_pos)
        by_tk = pos.groupby("ticker")["realized_pnl"].sum()
        out["ticker_concentration"] = float(by_tk.max() / tot_pos)
    return out


def run_cell(cell) -> dict:
    cfg_name, overrides, ivr_override, ov_name, ov_spec = cell
    t0 = time.time()

    from strategy.tiers import balanced_tier, _full_wheel
    from strategy.tier_runner import (
        compute_metrics, _apply_iv_rank_floor, _load_spy_close,
        _realized_cash_curve)

    data = _W["data"]
    tier = _full_wheel(balanced_tier())          # Tier2_Balanced_FW
    cfg = replace(tier.wheel_cfg, **overrides)
    ivr = tier.iv_rank_floor if ivr_override is None else ivr_override

    tickers = tier.universe_filter(
        data["universe"], data["fundamentals"], data["iv"], data["prices"])
    px = data["prices"][data["prices"]["ticker"].isin(tickers)].copy()
    iv = data["iv"][data["iv"]["ticker"].isin(tickers)].copy()
    iv = _apply_iv_rank_floor(iv, ivr)

    result = run_wheel_overlay(
        cfg=cfg, prices=px, iv=iv, macro=data["macro"],
        fundamentals=data["fundamentals"], universe=data["universe"],
        starting_cash=CAPITAL, start=START, end=END,
        overlay_spec=ov_spec, spy_signals=_W["spy_signals"])

    m = compute_metrics(result, CAPITAL, spy_close=_W["spy_close"])

    eq_df = result["equity_curve"].sort_values("date").reset_index(drop=True)
    led = result["ledger"]
    dates = pd.DatetimeIndex(pd.to_datetime(eq_df["date"]))
    req = _realized_cash_curve(led, CAPITAL, dates)
    req.index = dates

    m["realized_calmar"] = (m["realized_cagr"] / abs(m["realized_max_dd"])
                            if m.get("realized_max_dd") else float("nan"))
    m.update(_concentrations(led))

    sc = result["scale_curve"]
    m["pct_days_scaled"] = float((sc["scale"] < 1.0).mean()) if len(sc) else 0.0
    m["pct_days_halted"] = float((sc["scale"] <= 0.0).mean()) if len(sc) else 0.0

    strat = {}
    for yr in range(2020, 2026):
        sub = req[(req.index >= f"{yr}-01-01") & (req.index <= f"{yr}-12-31")]
        strat.update(_window_metrics(sub, f"y{yr}"))
    for label, (s, e) in CRISIS_WINDOWS.items():
        sub = req[(req.index >= s) & (req.index <= e)]
        strat.update(_window_metrics(sub, label))

    return {
        "cell": f"{cfg_name}__{ov_name}",
        "config": cfg_name, "overlay": ov_name,
        "overrides": overrides, "iv_rank_floor": ivr,
        "n_universe": len(tickers),
        "metrics": m, "strata": strat,
        "equity": eq_df, "ledger": led, "scale_curve": sc,
        "runtime_s": time.time() - t0,
    }


# ---------------- verdict logic (pre-registered) ----------------

NEIGHBORS = ["dte_24_36", "roll_3", "pt_078", "ivr_030"]
WORST_NONE_DD = -0.160  # roll_3 from robustness run


def build_verdict(df: pd.DataFrame) -> dict:
    def get(cfg, ov, col):
        sub = df[(df.config == cfg) & (df.overlay == ov)]
        return float(sub.iloc[0][col]) if len(sub) else float("nan")

    base_sh = get("baseline", "none", "realized_sharpe")
    replication_ok = bool(abs(base_sh - BASELINE_SHARPE_REF) <= 0.05)
    neighbor_repl = {}
    for cfg, ref_dd in ROBUSTNESS_NONE_DD.items():
        dd = get(cfg, "none", "realized_max_dd")
        neighbor_repl[cfg] = {"dd": dd, "ref": ref_dd,
                              "ok": bool(np.isfinite(dd) and abs(dd - ref_dd) <= 0.01)}

    overlays = [o for o in df.overlay.unique() if o != "none"]
    per_overlay = {}
    for ov in overlays:
        b_sh_ov = get("baseline", ov, "realized_sharpe")
        sharpe_cost = 1.0 - (b_sh_ov / base_sh) if base_sh else float("nan")
        gap = get("baseline", ov, "regime_gap")
        n_dds = [get(c, ov, "realized_max_dd") for c in NEIGHBORS]
        n_dds = [d for d in n_dds if np.isfinite(d)]
        worst_n_dd = min(n_dds) if n_dds else float("nan")
        b_dd = get("baseline", ov, "realized_max_dd")
        success = (np.isfinite(worst_n_dd) and worst_n_dd > -0.10
                   and sharpe_cost < 0.10
                   and np.isfinite(gap) and gap <= 0.50)
        partial = (not success and np.isfinite(worst_n_dd)
                   and worst_n_dd > -0.120 and sharpe_cost < 0.10
                   and np.isfinite(gap) and gap <= 0.50)
        per_overlay[ov] = {
            "baseline_sharpe": b_sh_ov,
            "baseline_cagr": get("baseline", ov, "realized_cagr"),
            "baseline_max_dd": b_dd,
            "sharpe_cost_pct": sharpe_cost * 100.0,
            "regime_gap_baseline": gap,
            "worst_neighbor_dd": worst_n_dd,
            "neighbor_dds": {c: get(c, ov, "realized_max_dd") for c in NEIGHBORS},
            "verdict": ("SUCCESS" if success else
                        "PARTIAL" if partial else "FAIL"),
        }
    return {
        "baseline_realized_sharpe": base_sh,
        "baseline_replication_ok": replication_ok,
        "neighbor_replication": neighbor_repl,
        "worst_neighbor_dd_no_overlay": WORST_NONE_DD,
        "per_overlay": per_overlay,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/wheel_dd_overlay_v1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overlays", default=None,
                    help="comma list of overlay names (default the 7 singles)")
    ap.add_argument("--configs", default=None,
                    help="comma list of config names (default all 5)")
    args = ap.parse_args()
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ov_names = (args.overlays.split(",") if args.overlays else DEFAULT_OVERLAYS)
    ov_map = dict(OVERLAYS)
    cfg_names = (args.configs.split(",") if args.configs
                 else [c[0] for c in CONFIGS])
    cfg_map = {c[0]: c for c in CONFIGS}

    cells = []
    for cn in cfg_names:
        _, overrides, ivr = cfg_map[cn]
        for on in ov_names:
            cells.append((cn, overrides, ivr, on, ov_map[on]))
    assert len(cells) <= 40, f"cell cap exceeded: {len(cells)}"

    print(f"[dd_overlay] running {len(cells)} cells, {args.workers} workers")
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_worker_init) as pool:
        futs = {pool.submit(run_cell, c): c for c in cells}
        for f in as_completed(futs):
            r = f.result()
            results[r["cell"]] = r
            m = r["metrics"]
            print(f"[dd_overlay] {r['cell']:28s} done {r['runtime_s']:5.0f}s  "
                  f"rSharpe={m['realized_sharpe']:.2f} "
                  f"rCAGR={m['realized_cagr']*100:5.1f}% "
                  f"rDD={m['realized_max_dd']*100:5.1f}% "
                  f"gap={m.get('regime_gap', float('nan')):.2f} "
                  f"scaled={m['pct_days_scaled']*100:.0f}%d", flush=True)
            r["equity"].to_parquet(out / f"equity_{r['cell']}.parquet", index=False)
            r["ledger"].to_parquet(out / f"ledger_{r['cell']}.parquet", index=False)
            r["scale_curve"].to_parquet(out / f"scale_{r['cell']}.parquet", index=False)

    rows = []
    for name, r in results.items():
        rows.append({"cell": name, "config": r["config"], "overlay": r["overlay"],
                     "iv_rank_floor": r["iv_rank_floor"],
                     "n_universe": r["n_universe"],
                     **r["metrics"], **r["strata"],
                     "runtime_s": r["runtime_s"]})
    df = pd.DataFrame(rows).sort_values(["config", "overlay"]).reset_index(drop=True)
    df.to_csv(out / "overlay_results.csv", index=False)
    df.to_parquet(out / "overlay_results.parquet", index=False)

    verdict = build_verdict(df)
    summary = {
        "experiment": "wheel_dd_overlay_v1",
        "config_tested": "Tier2_Balanced_FW (v8_WF) + additive exposure overlays",
        "window": [START, END], "capital": CAPITAL,
        "cells_run": sorted(results.keys()),
        "preregistered_rules": {
            "success_if": ("worst neighbor realized MaxDD > -10% AND baseline "
                           "Sharpe cost < 10% AND baseline regime gap <= 0.50"),
            "partial_if": "worst neighbor DD > -12% with same Sharpe/gap gates",
            "baseline_replication_tolerance": 0.05,
        },
        **verdict,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(verdict, indent=2, default=str))

    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("wheel_dd_overlay_v1")
        for _, row in df.iterrows():
            with mlflow.start_run(run_name=row["cell"]):
                mlflow.log_params({"config": row["config"],
                                   "overlay": row["overlay"],
                                   "iv_rank_floor": row["iv_rank_floor"],
                                   "base": "Tier2_Balanced_FW_v8WF"})
                for k in ["realized_sharpe", "realized_sortino", "realized_cagr",
                          "realized_max_dd", "realized_calmar", "pf", "wr",
                          "sharpe", "sortino", "cagr", "max_dd",
                          "regime_gap", "regime_green_sharpe", "regime_red_sharpe",
                          "day_concentration", "ticker_concentration",
                          "pct_days_scaled", "pct_days_halted",
                          "covid_2020_max_dd", "bear_2022_max_dd",
                          "unwind_aug2024_max_dd"]:
                    v = row.get(k)
                    try:
                        if v is not None and np.isfinite(float(v)):
                            mlflow.log_metric(k, float(v))
                    except (TypeError, ValueError):
                        pass
        with mlflow.start_run(run_name="SUMMARY"):
            verds = {ov: d["verdict"] for ov, d in verdict["per_overlay"].items()}
            mlflow.set_tag("verdicts", json.dumps(verds))
            mlflow.log_artifact(str(out / "summary.json"))
            mlflow.log_artifact(str(out / "overlay_results.csv"))
        print("[dd_overlay] MLflow logged -> wheel_dd_overlay_v1")
    except Exception as e:
        print(f"[dd_overlay] MLflow logging failed (non-fatal): {e}",
              file=sys.stderr)

    print(f"[dd_overlay] DONE -> {out}")


if __name__ == "__main__":
    main()
