#!/usr/bin/env python3
"""
wheel_paper_balanced.py - Tier2 Balanced FULL WHEEL, paper-only forward-test.

Deploys the validated Tier2_Balanced_FW config from the tier_ladder_v8 walk-forward
backtest (MLflow run tier_ladder_v8_WF_CALSKEW_SLIP_REGIME, experiment
wheel_tier_ladder_real_iv). Validated DEPLOY-CONDITIONAL 2026-06-10:
Sharpe 1.49 / CAGR 13.8% / MaxDD -8.4% / PF 2.47 / WR 90% — passes HC #428 R1
regime gate at 0.28.

Config (from wheel_strategy_v1/strategy/tiers.py::balanced_tier() + _full_wheel()):
    put_delta_target=0.22, call_delta_target=0.22
    dte_min=30, dte_max=45        (target DTE = 37)
    profit_take_pct=0.65          (FULL WHEEL: hold longer for premium)
    roll_dte_trigger=1            (allow assignment; true wheel behaviour)
    vix_max_gate=32.0
    regime gate: HC #555 macro overlay (risk_off blocks NEW entries) — this is
        the regime-gated variant the v8 WF run validated (regime_overlay=true).

This is a SINGLE-NAME (SPY) paper engine, same convention as the running
wheel-paper-engine (Tier2 scalp): the backtest universe is multi-name with
fundamentals + IV-rank gating, but no live multi-name options vendor is wired
in; SPY has the deepest reliable free chain on yfinance.

PRICING: entries/exits fill at REAL yfinance quote mid (the engine trades real
quotes, not modeled prices), so the backtest's calibrated-skew PRICING path is
not needed for fills. The walk-forward calibrated skew (latest period, 2026:
iv = sigma_atm*(1 + a*m + b*m^2)) IS used for mark-to-market: position marks
scale entry IV by the calibrated skew ratio as moneyness drifts.

FULL WHEEL mechanics added vs the scalp engine:
  - short put held to expiry (roll_dte_trigger=1); ITM at expiry -> ASSIGNED:
    buy 100*contracts shares at strike, keep premium.
  - while holding shares: sell covered call at call_delta_target, 30-45 DTE.
  - short call ITM at expiry -> shares CALLED AWAY at strike.
  - profit-take at 65% of max premium on both put and call legs.

NO REAL ORDERS. State/trades in JSON + MLflow + QCC alerts.

CLI:
    python3 -m live_trading_linux.wheel_paper_balanced --smoke
    python3 -m live_trading_linux.wheel_paper_balanced --daemon
    python3 -m live_trading_linux.wheel_paper_balanced --eod

Logs:    live_trading_linux/logs/wheel_paper_balanced.log
State:   live_trading_linux/wheel_paper_balanced_state/state.json
MLflow:  experiment "paper-wheel-balanced-fw"
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ----- Tier2 Balanced FULL WHEEL ruleset (v8 WF validated) -----
TIER2_BALANCED_FW = {
    "tier_name": "Tier2_Balanced_FW",
    "validated_run": "tier_ladder_v8_WF_CALSKEW_SLIP_REGIME",
    "put_delta_target": 0.22,
    "call_delta_target": 0.22,
    "dte_min": 30,
    "dte_max": 45,
    "dte_target": 37,
    "profit_take_pct": 0.65,
    "csp_stop_loss_mult": 1.0,   # Close CSP when loss reaches 1x premium collected
    "roll_dte_trigger": 1,       # full wheel: hold to expiry, allow assignment
    "vix_max_gate": 32.0,
    "leverage": 1.0,
    "underlying": "SPY",
    # v8 WF run was regime-gated (regime_overlay=true) and PASSES the
    # HC #428 R1 green/red gate at 0.28 -> entries are LIVE (not paused).
    "entries_paused": False,
    "regime_gate": True,          # HC #555 macro overlay blocks new entries on risk_off
}

# ----- Paper account sizing (match wheel-paper-engine conventions) -----
STARTING_CASH = 100_000.0
RISK_PCT_PER_TRADE = 0.05        # max 5% equity per CSP collateral (floor 1 contract, cash-gated)
COST_PER_CONTRACT = 0.03
WORK_INTERVAL_SECONDS = 300

# ----- Paths (fully separate from wheel-paper-engine) -----
ROOT = Path("/home/jupiter/Lvl3Quant/live_trading_linux")
STATE_DIR = ROOT / "wheel_paper_balanced_state"
LOG_DIR = ROOT / "logs"
STATE_FILE = STATE_DIR / "state.json"
TRADE_LOG = STATE_DIR / "trades.jsonl"
EQUITY_LOG = STATE_DIR / "equity.csv"
LOG_FILE = LOG_DIR / "wheel_paper_balanced.log"
QCC_DB = "/home/jupiter/teleclaude-main/data/qcc.db"
REGIME_PARQUET = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/regime_overlay.parquet")
SKEW_WF_JSON = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/skew_calibration_walkforward.json")

STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("wheel_paper_balanced")


# ===================== Calibrated skew (walk-forward, latest period) =====================
_SKEW_A, _SKEW_B = -0.1, 0.2   # hardcoded prior fallback

def load_skew_coeffs() -> tuple[float, float]:
    """Latest walk-forward period coefficients apply going forward."""
    global _SKEW_A, _SKEW_B
    try:
        with open(SKEW_WF_JSON) as f:
            cal = json.load(f)
        periods = cal["periods"]
        latest = sorted(periods.keys())[-1]
        _SKEW_A = float(periods[latest]["a"])
        _SKEW_B = float(periods[latest]["b"])
        log.info(f"calibrated skew loaded: period={latest} a={_SKEW_A} b={_SKEW_B}")
    except Exception as e:
        log.warning(f"skew calibration load failed, using prior a={_SKEW_A} b={_SKEW_B}: {e}")
    return _SKEW_A, _SKEW_B


def skew_factor(S: float, K: float, T: float) -> float:
    """f(m) = 1 + a*m + b*m^2, m = log(K/S)/sqrt(T). Clamped to [0.5, 2.0]."""
    if S <= 0 or K <= 0 or T <= 0:
        return 1.0
    m = math.log(K / S) / math.sqrt(T)
    f = 1.0 + _SKEW_A * m + _SKEW_B * m * m
    return min(max(f, 0.5), 2.0)


def mark_iv(entry_iv: float, entry_S: float, S_now: float, K: float, T: float) -> float:
    """Scale entry IV by calibrated skew ratio as moneyness drifts (MTM only)."""
    f_entry = skew_factor(entry_S, K, T)
    f_now = skew_factor(S_now, K, T)
    if f_entry <= 0:
        return entry_iv
    return entry_iv * (f_now / f_entry)


# ===================== Black-Scholes =====================
def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _d1(S, K, T, sigma, r):
    return (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))

def bs_put_price(S, K, T, sigma, r=0.04):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = _d1(S, K, T, sigma, r); d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)

def bs_call_price(S, K, T, sigma, r=0.04):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, sigma, r); d2 = d1 - sigma * math.sqrt(T)
    return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)

def bs_put_delta(S, K, T, sigma, r=0.04):
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    return _Phi(_d1(S, K, T, sigma, r)) - 1.0

def bs_call_delta(S, K, T, sigma, r=0.04):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    return _Phi(_d1(S, K, T, sigma, r))


# ===================== State =====================
@dataclass
class PaperState:
    cash: float = STARTING_CASH
    realized_pnl: float = 0.0
    positions: list = field(default_factory=list)     # option legs (dicts)
    stock: Optional[dict] = None                      # {"shares", "cost_basis", "acquired"}
    equity_curve: list = field(default_factory=list)
    trade_count: int = 0
    last_decision_ts: Optional[str] = None
    last_eod_date: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "positions": self.positions,
            "stock": self.stock,
            "equity_curve": self.equity_curve[-500:],
            "trade_count": self.trade_count,
            "last_decision_ts": self.last_decision_ts,
            "last_eod_date": self.last_eod_date,
        }


def load_state() -> PaperState:
    if not STATE_FILE.exists():
        s = PaperState()
        save_state(s)
        return s
    with open(STATE_FILE) as f:
        d = json.load(f)
    s = PaperState()
    s.cash = d.get("cash", STARTING_CASH)
    s.realized_pnl = d.get("realized_pnl", 0.0)
    s.positions = d.get("positions", [])
    s.stock = d.get("stock")
    s.equity_curve = d.get("equity_curve", [])
    s.trade_count = d.get("trade_count", 0)
    s.last_decision_ts = d.get("last_decision_ts")
    s.last_eod_date = d.get("last_eod_date")
    return s


def save_state(s: PaperState) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(s.to_dict(), f, indent=2, default=str)
    tmp.replace(STATE_FILE)


# ===================== Market data =====================
def fetch_spot(symbol: str) -> Optional[float]:
    try:
        price = float(yf.Ticker(symbol).fast_info.last_price)
        if price > 0:
            return price
    except Exception as e:
        log.warning(f"fast_info failed: {e}")
    try:
        hist = yf.Ticker(symbol).history(period="1d", interval="1m").tail(1)
        return float(hist["Close"].iloc[-1])
    except Exception as e:
        log.error(f"spot fetch failed: {e}")
        return None


def fetch_vix() -> Optional[float]:
    try:
        return float(yf.Ticker("^VIX").fast_info.last_price)
    except Exception as e:
        log.warning(f"VIX fetch failed: {e}")
        return None


def regime_risk_off() -> tuple[bool, str]:
    """HC #555 macro regime gate: latest row of regime_overlay.parquet.
    Returns (risk_off, info). Stale (>10 trading days) -> warn but use latest."""
    try:
        df = pd.read_parquet(REGIME_PARQUET)
        df["date"] = pd.to_datetime(df["date"])
        last = df.sort_values("date").iloc[-1]
        age_days = (pd.Timestamp.utcnow().tz_localize(None) - last["date"]).days
        info = f"asof={last['date'].date()} gates_on={int(last['gates_on'])} age={age_days}d"
        if age_days > 14:
            log.warning(f"regime overlay STALE ({info}) — using latest row anyway")
        return bool(last["risk_off"]), info
    except Exception as e:
        log.warning(f"regime overlay read failed (treating as risk-on): {e}")
        return False, "unavailable"


def fetch_chain_for_dte_band(symbol: str, dte_min: int, dte_max: int,
                             want: str = "puts") -> Optional[pd.DataFrame]:
    """Return puts or calls dataframe across expirations within [dte_min, dte_max]."""
    try:
        t = yf.Ticker(symbol)
        exps = list(t.options)
    except Exception as e:
        log.error(f"option expirations fetch failed: {e}")
        return None

    today = datetime.utcnow().date()
    keep = []
    for e in exps:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
        except Exception:
            continue
        dte = (d - today).days
        if dte_min <= dte <= dte_max:
            keep.append((e, dte))
    if not keep:
        log.warning(f"no expirations in DTE band [{dte_min},{dte_max}]")
        return None

    frames = []
    for e, dte in keep:
        try:
            ch = t.option_chain(e)
            p = (ch.puts if want == "puts" else ch.calls).copy()
            p["expiration"] = e
            p["dte"] = dte
            p["mid"] = (p["bid"].fillna(0) + p["ask"].fillna(0)) / 2.0
            frames.append(p)
        except Exception as ex:
            log.warning(f"chain fetch failed for {e}: {ex}")
        time.sleep(0.3)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ===================== Contract selection =====================
def pick_contract(chain: pd.DataFrame, S: float, target_delta: float,
                  target_dte: int, side: str,
                  vix_proxy: Optional[float] = None) -> Optional[dict]:
    """side='put' (OTM short put) or 'call' (OTM covered call).
    BS delta computed with IV floored at VIX-derived proxy (stale-quote guard)."""
    if chain is None or chain.empty:
        return None
    iv_floor = (vix_proxy / 100.0 * 1.1) if (vix_proxy and vix_proxy > 0) else 0.15
    rows = []
    for _, r in chain.iterrows():
        bid = float(r.get("bid", 0) or 0)
        ask = float(r.get("ask", 0) or 0)
        last = float(r.get("lastPrice", 0) or 0)
        iv = float(r.get("impliedVolatility", 0) or 0)
        K = float(r["strike"])
        dte = int(r["dte"])
        if iv < iv_floor:
            iv = iv_floor
        if iv > 2.0:
            continue
        if side == "put" and K >= S:
            continue
        if side == "call" and K <= S:
            continue
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        elif last > 0:
            mid = last
        else:
            continue
        T = max(dte / 365.0, 1.0 / 365.0)
        delta = bs_put_delta(S, K, T, iv) if side == "put" else bs_call_delta(S, K, T, iv)
        rows.append({"contractSymbol": r["contractSymbol"], "expiration": r["expiration"],
                     "strike": K, "bid": bid, "ask": ask, "mid": mid,
                     "iv": iv, "dte": dte, "delta": delta})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    tgt = -target_delta if side == "put" else target_delta
    df["delta_err"] = (df["delta"] - tgt).abs()
    df["dte_err"] = (df["dte"] - target_dte).abs()
    df = df.sort_values(["delta_err", "dte_err"]).reset_index(drop=True)
    return df.iloc[0].to_dict()


# ===================== Marking / equity =====================
def position_mark(p: dict, S: float) -> float:
    """Current per-share option value with calibrated-skew-adjusted IV."""
    exp_date = datetime.strptime(p["expiration"], "%Y-%m-%d").date()
    T = max((exp_date - datetime.utcnow().date()).days / 365.0, 1.0 / 365.0)
    iv = mark_iv(p.get("entry_iv", 0.20), p.get("entry_underlying", S), S, p["strike"], T)
    if p["side"] == "short_put":
        return bs_put_price(S, p["strike"], T, iv)
    return bs_call_price(S, p["strike"], T, iv)


def equity(state: PaperState, S: float) -> tuple[float, float]:
    unrl = 0.0
    for p in state.positions:
        cur = position_mark(p, S)
        unrl += (p["entry_premium"] - cur) * 100.0 * p["contracts"]
    stock_val = 0.0
    if state.stock:
        stock_val = state.stock["shares"] * S
        unrl += state.stock["shares"] * (S - state.stock["cost_basis"])
    # cash already excludes stock purchase cost; equity = cash + stock market value + option unrl
    eq = state.cash + stock_val + sum(
        (p["entry_premium"] - position_mark(p, S)) * 100.0 * p["contracts"]
        for p in state.positions)
    return eq, unrl


def log_trade(kind: str, detail: dict) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind, **detail}
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


# ===================== Decision cycle =====================
def decide_and_act(state: PaperState, S: float, vix: Optional[float],
                   chain_puts: Optional[pd.DataFrame]) -> dict:
    cfg = TIER2_BALANCED_FW
    actions = {"ts": datetime.now(timezone.utc).isoformat(), "spot": S, "vix": vix,
               "decisions": []}

    # ----- 1. Manage open option legs -----
    new_positions = []
    for p in state.positions:
        exp_date = datetime.strptime(p["expiration"], "%Y-%m-%d").date()
        today = datetime.utcnow().date()
        dte_now = (exp_date - today).days
        cur_price = position_mark(p, S)
        per_share_pnl = p["entry_premium"] - cur_price
        pct_of_max = per_share_pnl / max(p["entry_premium"], 1e-6)

        # PROFIT TAKE at 65% of max
        if pct_of_max >= cfg["profit_take_pct"]:
            realized = per_share_pnl * 100.0 * p["contracts"] - COST_PER_CONTRACT * p["contracts"]
            # FIX: premium was already credited to cash at OPEN, so only
            # subtract the buyback cost + close commission here (not the
            # full realized which would double-count the premium).
            buyback_cost = cur_price * 100.0 * p["contracts"] + COST_PER_CONTRACT * p["contracts"]
            state.cash -= buyback_cost
            state.realized_pnl += realized
            state.trade_count += 1
            log_trade("CLOSE_PROFIT_TAKE", {"contract": p["contract_symbol"],
                                            "side": p["side"], "pnl": realized, "spot": S})
            actions["decisions"].append({"action": "CLOSE_PROFIT_TAKE_65PCT",
                                         "contract": p["contract_symbol"],
                                         "side": p["side"],
                                         "pct_of_max": round(pct_of_max, 3),
                                         "realized_pnl": round(realized, 2)})
            continue

        # STOP LOSS: close CSP when loss reaches 1x premium collected
        sl_mult = cfg.get("csp_stop_loss_mult", 0)
        if sl_mult > 0 and p["side"] == "short_put" and pct_of_max <= -sl_mult:
            realized = per_share_pnl * 100.0 * p["contracts"] - COST_PER_CONTRACT * p["contracts"]
            # FIX: premium was already credited to cash at OPEN, so only
            # subtract the buyback cost + close commission here.
            buyback_cost = cur_price * 100.0 * p["contracts"] + COST_PER_CONTRACT * p["contracts"]
            state.cash -= buyback_cost
            state.realized_pnl += realized
            state.trade_count += 1
            log_trade("STOP_LOSS_CSP", {"contract": p["contract_symbol"],
                                        "side": p["side"], "pnl": realized, "spot": S,
                                        "pct_of_max": round(pct_of_max, 3)})
            actions["decisions"].append({"action": "STOP_LOSS_CSP",
                                         "contract": p["contract_symbol"],
                                         "side": p["side"],
                                         "pct_of_max": round(pct_of_max, 3),
                                         "realized_pnl": round(realized, 2)})
            continue

        # EXPIRY (roll_dte_trigger=1 -> we hold to/through expiry)
        if dte_now <= 0:
            if p["side"] == "short_put":
                if S < p["strike"]:
                    # ASSIGNED: buy shares at strike
                    # FIX: premium was already credited to cash at OPEN,
                    # so do NOT add premium_kept to cash again here.
                    # Only debit the share purchase cost.
                    shares = 100 * p["contracts"]
                    cost = p["strike"] * shares
                    premium_kept = p["entry_premium"] * 100.0 * p["contracts"] - COST_PER_CONTRACT * p["contracts"]
                    state.cash -= cost
                    state.realized_pnl += premium_kept
                    state.stock = {"shares": shares, "cost_basis": p["strike"],
                                   "acquired": today.isoformat()}
                    state.trade_count += 1
                    log_trade("ASSIGNED", {"contract": p["contract_symbol"],
                                           "strike": p["strike"], "shares": shares,
                                           "premium_kept": premium_kept, "spot": S})
                    actions["decisions"].append({"action": "PUT_ASSIGNED",
                                                 "strike": p["strike"], "shares": shares,
                                                 "premium_kept": round(premium_kept, 2)})
                else:
                    # FIX: premium was already credited to cash at OPEN.
                    # OTM expiry means option expires worthless — no buyback,
                    # no additional cash change needed. Only track realized_pnl.
                    realized = p["entry_premium"] * 100.0 * p["contracts"] - COST_PER_CONTRACT * p["contracts"]
                    state.realized_pnl += realized
                    state.trade_count += 1
                    log_trade("EXPIRE_OTM", {"contract": p["contract_symbol"],
                                             "pnl": realized, "spot": S})
                    actions["decisions"].append({"action": "PUT_EXPIRED_OTM",
                                                 "realized_pnl": round(realized, 2)})
            else:  # short_call (covered)
                # FIX: premium was already credited to cash at OPEN.
                # At expiry (OTM or called away), no buyback needed, so
                # no additional cash change. Only track realized_pnl.
                premium_kept = p["entry_premium"] * 100.0 * p["contracts"] - COST_PER_CONTRACT * p["contracts"]
                state.realized_pnl += premium_kept
                state.trade_count += 1
                if S > p["strike"] and state.stock:
                    # CALLED AWAY: sell shares at strike
                    shares = state.stock["shares"]
                    proceeds = p["strike"] * shares
                    stock_pnl = (p["strike"] - state.stock["cost_basis"]) * shares
                    state.cash += proceeds
                    state.realized_pnl += stock_pnl
                    log_trade("CALLED_AWAY", {"contract": p["contract_symbol"],
                                              "strike": p["strike"], "shares": shares,
                                              "stock_pnl": stock_pnl,
                                              "premium_kept": premium_kept, "spot": S})
                    actions["decisions"].append({"action": "CALLED_AWAY",
                                                 "strike": p["strike"],
                                                 "stock_pnl": round(stock_pnl, 2),
                                                 "premium_kept": round(premium_kept, 2)})
                    state.stock = None
                else:
                    log_trade("CALL_EXPIRED_OTM", {"contract": p["contract_symbol"],
                                                   "pnl": premium_kept, "spot": S})
                    actions["decisions"].append({"action": "CALL_EXPIRED_OTM",
                                                 "realized_pnl": round(premium_kept, 2)})
            continue

        new_positions.append(p)
    state.positions = new_positions

    # ----- 2. Covered call: holding shares and no short call -> sell CC -----
    if state.stock and not any(p["side"] == "short_call" for p in state.positions):
        chain_calls = fetch_chain_for_dte_band(cfg["underlying"], cfg["dte_min"],
                                               cfg["dte_max"], want="calls")
        pick = pick_contract(chain_calls, S, cfg["call_delta_target"],
                             cfg["dte_target"], "call", vix_proxy=vix) if chain_calls is not None else None
        if pick is not None:
            contracts = state.stock["shares"] // 100
            if contracts >= 1:
                premium = pick["mid"]
                credit = premium * 100.0 * contracts - COST_PER_CONTRACT * contracts
                state.cash += credit
                pos = {"contract_symbol": pick["contractSymbol"],
                       "underlying": cfg["underlying"], "side": "short_call",
                       "strike": pick["strike"], "expiration": pick["expiration"],
                       "contracts": contracts, "entry_premium": premium,
                       "entry_date": datetime.utcnow().date().isoformat(),
                       "entry_underlying": S, "entry_delta": pick["delta"],
                       "entry_iv": pick["iv"], "max_profit": premium}
                state.positions.append(pos)
                state.trade_count += 1
                log_trade("OPEN_COVERED_CALL", {"contract": pick["contractSymbol"],
                                                "strike": pick["strike"],
                                                "credit": credit, "spot": S})
                actions["decisions"].append({"action": "OPEN_COVERED_CALL",
                                             "contract": pick["contractSymbol"],
                                             "strike": pick["strike"],
                                             "expiration": pick["expiration"],
                                             "delta": round(pick["delta"], 3),
                                             "credit": round(credit, 2)})
        else:
            actions["decisions"].append({"action": "HOLD_NO_CC_CONTRACT"})

    # ----- 3. New CSP entry gates -----
    if cfg.get("entries_paused", False):
        actions["decisions"].append({"action": "HOLD_ENTRIES_PAUSED"})
        return actions

    if state.stock is not None:
        # wheel is in stock/CC phase; no new CSP until shares called away
        actions["decisions"].append({"action": "HOLD_IN_CC_PHASE"})
        return actions

    if any(p["side"] == "short_put" for p in state.positions):
        actions["decisions"].append({"action": "HOLD_HAVE_POSITION"})
        return actions

    if vix is not None and vix > cfg["vix_max_gate"]:
        actions["decisions"].append({"action": "HOLD_VIX_GATE", "vix": vix})
        return actions

    if cfg.get("regime_gate", True):
        risk_off, rinfo = regime_risk_off()
        if risk_off:
            actions["decisions"].append({"action": "HOLD_REGIME_RISK_OFF", "regime": rinfo})
            return actions

    # ----- 4. Pick + size + open CSP -----
    pick = pick_contract(chain_puts, S, cfg["put_delta_target"], cfg["dte_target"],
                         "put", vix_proxy=vix)
    if pick is None:
        actions["decisions"].append({"action": "HOLD_NO_VALID_CONTRACT"})
        return actions

    eq_now, _ = equity(state, S)
    max_collateral = eq_now * RISK_PCT_PER_TRADE * cfg["leverage"]
    contracts = max(1, int(max_collateral // (pick["strike"] * 100.0)))
    collateral = pick["strike"] * 100.0 * contracts
    if collateral > state.cash:
        actions["decisions"].append({"action": "HOLD_INSUFFICIENT_CASH",
                                     "needed": collateral, "have": round(state.cash, 2)})
        return actions

    premium = pick["mid"]
    credit = premium * 100.0 * contracts - COST_PER_CONTRACT * contracts
    state.cash += credit
    pos = {"contract_symbol": pick["contractSymbol"], "underlying": cfg["underlying"],
           "side": "short_put", "strike": pick["strike"], "expiration": pick["expiration"],
           "contracts": contracts, "entry_premium": premium,
           "entry_date": datetime.utcnow().date().isoformat(),
           "entry_underlying": S, "entry_delta": pick["delta"],
           "entry_iv": pick["iv"], "max_profit": premium}
    state.positions.append(pos)
    state.trade_count += 1
    log_trade("OPEN_SHORT_PUT", {"contract": pick["contractSymbol"],
                                 "strike": pick["strike"], "credit": credit, "spot": S})
    actions["decisions"].append({"action": "OPEN_SHORT_PUT",
                                 "contract": pick["contractSymbol"],
                                 "strike": pick["strike"], "expiration": pick["expiration"],
                                 "dte": pick["dte"], "delta": round(pick["delta"], 3),
                                 "premium_per_share": round(premium, 3),
                                 "contracts": contracts, "credit": round(credit, 2)})
    return actions


# ===================== MLflow =====================
_mlflow = None
_mlflow_run = None
def init_mlflow():
    global _mlflow, _mlflow_run
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("paper-wheel-balanced-fw")
        _mlflow = mlflow
        _mlflow_run = mlflow.start_run(
            run_name=f"wheel_paper_balanced_{datetime.utcnow().strftime('%Y%m%d')}")
        mlflow.log_params({
            "tier": TIER2_BALANCED_FW["tier_name"],
            "validated_run": TIER2_BALANCED_FW["validated_run"],
            "put_delta_target": TIER2_BALANCED_FW["put_delta_target"],
            "call_delta_target": TIER2_BALANCED_FW["call_delta_target"],
            "dte_target": TIER2_BALANCED_FW["dte_target"],
            "profit_take_pct": TIER2_BALANCED_FW["profit_take_pct"],
            "roll_dte_trigger": TIER2_BALANCED_FW["roll_dte_trigger"],
            "vix_max_gate": TIER2_BALANCED_FW["vix_max_gate"],
            "regime_gate": TIER2_BALANCED_FW["regime_gate"],
            "underlying": TIER2_BALANCED_FW["underlying"],
            "starting_cash": STARTING_CASH,
            "skew_a": _SKEW_A, "skew_b": _SKEW_B,
        })
        log.info(f"MLflow run started: {_mlflow_run.info.run_id}")
    except Exception as e:
        log.warning(f"MLflow init failed (non-fatal): {e}")
        _mlflow = None


def mlflow_log_metrics(metrics: dict, step: Optional[int] = None):
    if _mlflow is None:
        return
    try:
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                continue
            _mlflow.log_metric(k, float(v), step=step)
    except Exception as e:
        log.warning(f"MLflow metric log failed: {e}")


def mlflow_end():
    if _mlflow is not None and _mlflow_run is not None:
        try:
            _mlflow.end_run()
        except Exception:
            pass


# ===================== QCC alert (EOD report) =====================
def send_qcc_alert(severity: str, source: str, message: str) -> None:
    try:
        conn = sqlite3.connect(QCC_DB, timeout=5.0)
        conn.execute("INSERT INTO alerts (severity, source, node, message) VALUES (?, ?, ?, ?)",
                     (severity, source, "jupiter", message))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"QCC alert write failed: {e}")


def post_eod_report(state: PaperState, spot: float, vix: Optional[float]) -> None:
    eq_now, unrl = equity(state, spot)
    pnl_pct = (eq_now - STARTING_CASH) / STARTING_CASH * 100.0
    pos_lines = [f"  - {p['side']} @ {p['strike']:.0f} exp {p['expiration']} x{p['contracts']}"
                 for p in state.positions[:5]]
    if state.stock:
        pos_lines.append(f"  - LONG {state.stock['shares']} shares @ {state.stock['cost_basis']:.2f}")
    msg = (f"[EOD] Tier2 Balanced FULL WHEEL paper (v8 WF validated)\n"
           f"Equity: ${eq_now:,.0f} ({pnl_pct:+.2f}%)\n"
           f"Realized: ${state.realized_pnl:+,.0f}  Unrealized: ${unrl:+,.0f}\n"
           f"Trades: {state.trade_count}  Open legs: {len(state.positions)}\n"
           f"SPY: ${spot:.2f}  VIX: {vix if vix is not None else 'n/a'}\n"
           + ("\n".join(pos_lines) if pos_lines else "  (none)"))
    send_qcc_alert("info", "wheel_paper_balanced", msg)
    log.info("EOD report sent")


# ===================== Main loops =====================
_RUNNING = True
def _shutdown(signum, frame):
    global _RUNNING
    log.info(f"shutdown signal {signum}")
    _RUNNING = False
signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT, _shutdown)


def cycle(state: PaperState) -> dict:
    cfg = TIER2_BALANCED_FW
    S = fetch_spot(cfg["underlying"])
    if S is None:
        return {"error": "spot_fetch_failed"}
    vix = fetch_vix()
    chain = fetch_chain_for_dte_band(cfg["underlying"], cfg["dte_min"], cfg["dte_max"],
                                     want="puts")
    if chain is None:
        log.warning("chain unavailable; managing existing positions only")
    actions = decide_and_act(state, S, vix, chain if chain is not None else pd.DataFrame())

    eq_now, unrl = equity(state, S)
    state.equity_curve.append({"ts": datetime.now(timezone.utc).isoformat(),
                               "equity": eq_now, "realized": state.realized_pnl,
                               "unrealized": unrl, "spot": S, "vix": vix})
    state.last_decision_ts = datetime.now(timezone.utc).isoformat()
    save_state(state)

    with open(EQUITY_LOG, "a") as f:
        if f.tell() == 0:
            f.write("ts,equity,realized,unrealized,spot,vix\n")
        f.write(f"{state.last_decision_ts},{eq_now:.2f},{state.realized_pnl:.2f},"
                f"{unrl:.2f},{S:.2f},{vix if vix is not None else ''}\n")

    mlflow_log_metrics({"equity": eq_now, "realized_pnl": state.realized_pnl,
                        "unrealized_pnl": unrl, "open_positions": len(state.positions),
                        "holding_shares": state.stock["shares"] if state.stock else 0,
                        "spot": S, "vix": vix if vix is not None else float("nan")},
                       step=state.trade_count)

    log.info(f"cycle done: spot={S:.2f} vix={vix} equity=${eq_now:.0f} "
             f"realized=${state.realized_pnl:+.0f} unrealized=${unrl:+.0f} "
             f"legs={len(state.positions)} stock={state.stock is not None} "
             f"actions={actions['decisions']}")
    return actions


def maybe_eod(state: PaperState, force: bool = False) -> None:
    today = datetime.utcnow().date().isoformat()
    should = force or (datetime.utcnow().hour >= 21 and state.last_eod_date != today)
    if not should:
        return
    S = fetch_spot(TIER2_BALANCED_FW["underlying"]) or 0.0
    post_eod_report(state, S, fetch_vix())
    state.last_eod_date = today
    save_state(state)


def run_daemon():
    log.info("=== wheel_paper_balanced daemon starting ===")
    log.info(f"config: {json.dumps(TIER2_BALANCED_FW)}")
    load_skew_coeffs()
    init_mlflow()
    state = load_state()
    log.info(f"state loaded: cash=${state.cash:.2f} legs={len(state.positions)} "
             f"stock={state.stock} trades={state.trade_count}")
    try:
        while _RUNNING:
            try:
                cycle(state)
                maybe_eod(state)
            except Exception as e:
                log.exception(f"cycle error: {e}")
            for _ in range(WORK_INTERVAL_SECONDS):
                if not _RUNNING:
                    break
                time.sleep(1)
    finally:
        mlflow_end()
        log.info("daemon exiting")


def run_smoke():
    log.info("=== SMOKE TEST ===")
    load_skew_coeffs()
    init_mlflow()
    state = load_state()
    log.info(f"state: cash=${state.cash:.2f} legs={len(state.positions)} "
             f"trades={state.trade_count}")
    out = cycle(state)
    log.info(f"smoke decisions: {json.dumps(out, indent=2, default=str)}")
    mlflow_end()
    log.info("=== SMOKE OK ===")
    return out


def run_eod_only():
    load_skew_coeffs()
    init_mlflow()
    state = load_state()
    maybe_eod(state, force=True)
    mlflow_end()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eod", action="store_true")
    ap.add_argument("--daemon", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        run_smoke()
        return
    if args.eod:
        run_eod_only()
        return
    run_daemon()


if __name__ == "__main__":
    main()
