#!/usr/bin/env python3
"""
wheel_paper_engine.py - Tier2 Balanced Scalp wheel, paper-only forward-test.

Implements the Tier2 Balanced Scalp ruleset extracted from
wheel_strategy_v1/strategy/tiers.py::balanced_tier():

    put_delta_target=0.22, call_delta_target=0.22
    dte_min=30, dte_max=45        (target DTE = 37)
    profit_take_pct=0.50          (SCALP variant: close at 50% of max profit)
    roll_dte_trigger=10           (roll when DTE <= 10)
    vix_max_gate=32.0
    naaim_min_gate=-50.0          (not gated in paper engine - free data not always available)
    fund_score_floor=45.0         (SPY-only paper run skips this)
    iv_rank_floor=0.25            (skipped in single-name SPY paper mode)

This is a SINGLE-NAME (SPY) paper engine. The full backtest uses ~329 names with
fundamentals + IV-rank gating; for the live forward-test we use SPY-only because:
  - The wheel backtest used Dolt historical chains + BS modeled IV. No live
    multi-name options vendor is wired into this codebase.
  - SPY has the deepest free option chain on yfinance with reliable
    bid/ask/IV/delta and near-zero stale-quote risk.
  - SPY paper-test still validates the ruleset (entry delta, profit-take, roll).

NO REAL ORDERS are submitted. State and trades are tracked in JSON +
MLflow + the QCC alerts table for EOD reporting.

CLI:
    python3 -m live_trading_linux.wheel_paper_engine --smoke       # one-shot smoke
    python3 -m live_trading_linux.wheel_paper_engine               # daemon loop
    python3 -m live_trading_linux.wheel_paper_engine --eod         # force EOD report

Logs:    live_trading_linux/logs/wheel_paper_engine.log
State:   live_trading_linux/wheel_paper_state/state.json
MLflow:  experiment "paper-wheel-tier2-scalp"
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
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ----- Tier2 Balanced Scalp ruleset (from wheel_strategy_v1/strategy/tiers.py) -----
TIER2_BALANCED_SCALP = {
    "tier_name": "Tier2_Balanced_Scalp",
    "put_delta_target": 0.22,
    "call_delta_target": 0.22,
    "dte_min": 30,
    "dte_max": 45,
    "dte_target": 37,
    "profit_take_pct": 0.50,
    "csp_stop_loss_mult": 1.0,   # Close CSP when loss reaches 1x premium collected
    "roll_dte_trigger": 10,
    "vix_max_gate": 32.0,
    "leverage": 1.0,            # 1.0x sizing per stress-test recommendation
    "underlying": "SPY",
    # HC #583 R2 (2026-06-09): wheel fails HC #428 R1 green/red regime gate.
    # Entries paused until regime-gated variant validates. Existing positions
    # still manage/exit normally. Flip to False after the regime-gated backtest
    # passes the deploy gates and a `regime_gate` block is added below.
    "entries_paused": True,
}

# ----- Paper account sizing -----
STARTING_CASH = 100_000.0       # paper capital (USD)
RISK_PCT_PER_TRADE = 0.05       # max 5% of equity per single CSP collateral
COST_PER_CONTRACT = 0.03        # retail regulatory fee (Schwab/Robinhood)
QUOTE_POLL_SECONDS = 60         # min poll interval per task constraint
WORK_INTERVAL_SECONDS = 300     # full decision-cycle every 5 min

# ----- Paths -----
ROOT = Path("/home/jupiter/Lvl3Quant/live_trading_linux")
STATE_DIR = ROOT / "wheel_paper_state"
LOG_DIR = ROOT / "logs"
STATE_FILE = STATE_DIR / "state.json"
TRADE_LOG = STATE_DIR / "trades.jsonl"
EQUITY_LOG = STATE_DIR / "equity.csv"
LOG_FILE = LOG_DIR / "wheel_paper_engine.log"
QCC_DB = "/home/jupiter/teleclaude-main/data/qcc.db"

STATE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ----- Logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("wheel_paper")


# ===================== Black-Scholes helpers =====================
SQRT_2PI = math.sqrt(2 * math.pi)

def _Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_put_price(S: float, K: float, T: float, sigma: float, r: float = 0.04) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)

def bs_put_delta(S: float, K: float, T: float, sigma: float, r: float = 0.04) -> float:
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return _Phi(d1) - 1.0


# ===================== State =====================
@dataclass
class Position:
    contract_symbol: str
    underlying: str
    side: str                # "short_put"
    strike: float
    expiration: str          # YYYY-MM-DD
    contracts: int
    entry_premium: float     # per share, credit received
    entry_date: str
    entry_underlying: float
    entry_delta: float
    max_profit: float        # = entry_premium (for short put)

    def collateral(self) -> float:
        return self.strike * 100.0 * self.contracts


@dataclass
class PaperState:
    cash: float = STARTING_CASH
    realized_pnl: float = 0.0
    positions: list = field(default_factory=list)   # list[Position dict]
    equity_curve: list = field(default_factory=list)
    trade_count: int = 0
    last_decision_ts: Optional[str] = None
    last_eod_date: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "positions": self.positions,
            "equity_curve": self.equity_curve[-500:],   # cap
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
        t = yf.Ticker(symbol)
        # fast_info is the fastest path
        fi = t.fast_info
        price = float(fi.last_price)
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
        v = yf.Ticker("^VIX").fast_info.last_price
        return float(v)
    except Exception as e:
        log.warning(f"VIX fetch failed: {e}")
        return None


def fetch_chain_for_dte_band(symbol: str, dte_min: int, dte_max: int) -> Optional[pd.DataFrame]:
    """Return puts dataframe across expirations within [dte_min, dte_max].
    Columns: contractSymbol, expiration, strike, bid, ask, mid, impliedVolatility, dte"""
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
        log.warning(f"no expirations in DTE band [{dte_min},{dte_max}]; available={exps[:8]}")
        return None

    frames = []
    for e, dte in keep:
        try:
            ch = t.option_chain(e)
            p = ch.puts.copy()
            p["expiration"] = e
            p["dte"] = dte
            p["mid"] = (p["bid"].fillna(0) + p["ask"].fillna(0)) / 2.0
            frames.append(p)
        except Exception as ex:
            log.warning(f"chain fetch failed for {e}: {ex}")
        time.sleep(0.3)  # politeness

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ===================== Strategy =====================
def pick_short_put(chain: pd.DataFrame, S: float, target_delta: float,
                   target_dte: int, vix_proxy: Optional[float] = None) -> Optional[dict]:
    """Choose the put contract whose computed BS delta is closest to -target_delta
    AND whose DTE is closest to target_dte.
    yfinance pre-market IVs are often stale/wrong for OTM puts; we floor IV
    at vix_proxy (VIX/100 * 1.1 skew-bump) when the contract-level IV is
    implausibly low."""
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
        # Pre-market yfinance often returns absurd IV (e.g. 0.03 for SPY puts).
        # Floor at iv_floor (VIX-derived) to keep delta math meaningful.
        if iv < iv_floor:
            iv = iv_floor
        if iv > 2.0:
            continue
        if K >= S:           # we want OTM puts
            continue
        # Pre-market/after-hours: bid/ask often 0 but lastPrice + IV available.
        # Use mid when both sides quoted, otherwise lastPrice as a fallback.
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        elif last > 0:
            mid = last
        else:
            continue
        T = max(dte / 365.0, 1.0 / 365.0)
        delta = bs_put_delta(S, K, T, iv)
        rows.append({
            "contractSymbol": r["contractSymbol"],
            "expiration": r["expiration"],
            "strike": K,
            "bid": bid, "ask": ask, "mid": mid,
            "iv": iv, "dte": dte, "delta": delta,
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["delta_err"] = (df["delta"] - (-target_delta)).abs()
    df["dte_err"] = (df["dte"] - target_dte).abs()
    # rank: minimize delta-error then dte-error
    df = df.sort_values(["delta_err", "dte_err"]).reset_index(drop=True)
    return df.iloc[0].to_dict()


def equity(state: PaperState, S: float) -> tuple[float, float]:
    """Return (equity, unrealized_pnl). For short puts: unrealized = (entry_premium - current_mid) * 100 * contracts."""
    unrl = 0.0
    for p in state.positions:
        T_now = max((datetime.strptime(p["expiration"], "%Y-%m-%d").date() - datetime.utcnow().date()).days / 365.0,
                    1.0 / 365.0)
        # mark current value with last-known IV (stored at entry); good enough for paper MTM
        cur_price = bs_put_price(S, p["strike"], T_now, p.get("entry_iv", 0.20))
        per_share_pnl = (p["entry_premium"] - cur_price)
        unrl += per_share_pnl * 100.0 * p["contracts"]
    return state.cash + unrl, unrl


def decide_and_act(state: PaperState, S: float, vix: Optional[float],
                   chain: pd.DataFrame) -> dict:
    """One decision cycle. Returns a summary dict of what was done."""
    cfg = TIER2_BALANCED_SCALP
    actions = {"ts": datetime.now(timezone.utc).isoformat(), "spot": S, "vix": vix,
               "decisions": []}

    # ----- 1. Manage open positions -----
    new_positions = []
    for p in state.positions:
        exp_date = datetime.strptime(p["expiration"], "%Y-%m-%d").date()
        today = datetime.utcnow().date()
        dte_now = (exp_date - today).days
        T_now = max(dte_now / 365.0, 1.0 / 365.0)
        cur_price = bs_put_price(S, p["strike"], T_now, p.get("entry_iv", 0.20))
        per_share_pnl = (p["entry_premium"] - cur_price)
        pct_of_max = per_share_pnl / max(p["entry_premium"], 1e-6)

        # SCALP CLOSE: 50% of max profit hit
        if pct_of_max >= cfg["profit_take_pct"]:
            realized = per_share_pnl * 100.0 * p["contracts"] - COST_PER_CONTRACT * p["contracts"]
            state.cash += realized
            state.realized_pnl += realized
            state.trade_count += 1
            log_trade("CLOSE_SCALP", p, S, cur_price, realized)
            actions["decisions"].append({
                "action": "CLOSE_SCALP_50PCT",
                "contract": p["contract_symbol"],
                "pct_of_max": round(pct_of_max, 3),
                "realized_pnl": round(realized, 2),
            })
            continue

        # STOP LOSS: close CSP when loss reaches 1x premium collected
        sl_mult = cfg.get("csp_stop_loss_mult", 0)
        if sl_mult > 0 and pct_of_max <= -sl_mult:
            realized = per_share_pnl * 100.0 * p["contracts"] - COST_PER_CONTRACT * p["contracts"]
            state.cash += realized
            state.realized_pnl += realized
            state.trade_count += 1
            log_trade("STOP_LOSS_CSP", p, S, cur_price, realized)
            actions["decisions"].append({
                "action": "STOP_LOSS_CSP",
                "contract": p["contract_symbol"],
                "pct_of_max": round(pct_of_max, 3),
                "realized_pnl": round(realized, 2),
            })
            continue

        # ROLL TRIGGER: DTE <= 10 and ITM/tested
        if dte_now <= cfg["roll_dte_trigger"]:
            tested = S <= p["strike"] * 1.005   # within 0.5% of strike or below
            if tested:
                # close current and open new at target DTE
                realized = per_share_pnl * 100.0 * p["contracts"] - COST_PER_CONTRACT * p["contracts"]
                state.cash += realized
                state.realized_pnl += realized
                state.trade_count += 1
                log_trade("ROLL_CLOSE", p, S, cur_price, realized)
                actions["decisions"].append({
                    "action": "ROLL_TESTED",
                    "contract": p["contract_symbol"],
                    "dte_now": dte_now,
                    "realized_pnl": round(realized, 2),
                })
                continue  # next cycle will open a fresh one
            else:
                # Expire OTM path: just keep it; if dte_now == 0 and OTM, close it for full premium
                if dte_now <= 0:
                    realized = p["entry_premium"] * 100.0 * p["contracts"] - COST_PER_CONTRACT * p["contracts"]
                    state.cash += realized
                    state.realized_pnl += realized
                    state.trade_count += 1
                    log_trade("EXPIRE_OTM", p, S, 0.0, realized)
                    actions["decisions"].append({
                        "action": "EXPIRE_OTM",
                        "contract": p["contract_symbol"],
                        "realized_pnl": round(realized, 2),
                    })
                    continue

        new_positions.append(p)
    state.positions = new_positions

    # ----- 2. Entry gates -----
    # HC #583 R2: entries paused after regime-gate failure (2026-06-09).
    # Existing positions still manage / exit normally; only NEW entries are blocked.
    # Flip cfg["entries_paused"] to False after regime-gated variant validates.
    if cfg.get("entries_paused", True):
        actions["decisions"].append({"action": "HOLD_ENTRIES_PAUSED",
                                     "reason": "HC #583 — wheel fails green/red regime gate; "
                                               "regime-gated variant under test"})
        return actions

    if vix is not None and vix > cfg["vix_max_gate"]:
        actions["decisions"].append({"action": "HOLD_VIX_GATE", "vix": vix,
                                     "gate": cfg["vix_max_gate"]})
        return actions

    # Already have a SPY short put? Don't open another (single-name single-position
    # for the paper engine; multi-name complexity needs a real chain feed).
    if any(p["underlying"] == cfg["underlying"] and p["side"] == "short_put"
           for p in state.positions):
        actions["decisions"].append({"action": "HOLD_HAVE_POSITION",
                                     "underlying": cfg["underlying"]})
        return actions

    # ----- 3. Pick a contract -----
    pick = pick_short_put(chain, S, cfg["put_delta_target"], cfg["dte_target"],
                          vix_proxy=vix)
    if pick is None:
        actions["decisions"].append({"action": "HOLD_NO_VALID_CONTRACT"})
        return actions

    # ----- 4. Size -----
    eq_now, _ = equity(state, S)
    max_collateral = eq_now * RISK_PCT_PER_TRADE * cfg["leverage"]
    contracts = max(1, int(max_collateral // (pick["strike"] * 100.0)))
    collateral = pick["strike"] * 100.0 * contracts

    if collateral > state.cash:
        actions["decisions"].append({"action": "HOLD_INSUFFICIENT_CASH",
                                     "needed": collateral, "have": state.cash})
        return actions

    # ----- 5. Open (paper fill at mid) -----
    premium = pick["mid"]
    credit = premium * 100.0 * contracts - COST_PER_CONTRACT * contracts
    state.cash += credit

    new_pos = {
        "contract_symbol": pick["contractSymbol"],
        "underlying": cfg["underlying"],
        "side": "short_put",
        "strike": pick["strike"],
        "expiration": pick["expiration"],
        "contracts": contracts,
        "entry_premium": premium,
        "entry_date": datetime.utcnow().date().isoformat(),
        "entry_underlying": S,
        "entry_delta": pick["delta"],
        "entry_iv": pick["iv"],
        "max_profit": premium,
    }
    state.positions.append(new_pos)
    state.trade_count += 1
    log_trade("OPEN_SHORT_PUT", new_pos, S, premium, credit)

    actions["decisions"].append({
        "action": "OPEN_SHORT_PUT",
        "contract": pick["contractSymbol"],
        "strike": pick["strike"],
        "expiration": pick["expiration"],
        "dte": pick["dte"],
        "delta": round(pick["delta"], 3),
        "premium_per_share": round(premium, 3),
        "contracts": contracts,
        "credit": round(credit, 2),
    })
    return actions


def log_trade(kind: str, pos: dict, spot: float, mark: float, pnl: float) -> None:
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "contract": pos["contract_symbol"],
        "underlying": pos["underlying"],
        "strike": pos["strike"],
        "expiration": pos["expiration"],
        "contracts": pos["contracts"],
        "entry_premium": pos["entry_premium"],
        "spot": spot,
        "mark": mark,
        "pnl": pnl,
    }
    with open(TRADE_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ===================== MLflow =====================
_mlflow = None
_mlflow_run = None
def init_mlflow():
    global _mlflow, _mlflow_run
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("paper-wheel-tier2-scalp")
        _mlflow = mlflow
        run_name = f"wheel_paper_{datetime.utcnow().strftime('%Y%m%d')}"
        _mlflow_run = mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "tier": TIER2_BALANCED_SCALP["tier_name"],
            "put_delta_target": TIER2_BALANCED_SCALP["put_delta_target"],
            "dte_target": TIER2_BALANCED_SCALP["dte_target"],
            "profit_take_pct": TIER2_BALANCED_SCALP["profit_take_pct"],
            "roll_dte_trigger": TIER2_BALANCED_SCALP["roll_dte_trigger"],
            "vix_max_gate": TIER2_BALANCED_SCALP["vix_max_gate"],
            "leverage": TIER2_BALANCED_SCALP["leverage"],
            "underlying": TIER2_BALANCED_SCALP["underlying"],
            "starting_cash": STARTING_CASH,
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
    global _mlflow_run
    if _mlflow is None or _mlflow_run is None:
        return
    try:
        _mlflow.end_run()
    except Exception:
        pass


# ===================== QCC alert (EOD report) =====================
def send_qcc_alert(severity: str, source: str, message: str) -> None:
    try:
        conn = sqlite3.connect(QCC_DB, timeout=5.0)
        conn.execute(
            "INSERT INTO alerts (severity, source, node, message) VALUES (?, ?, ?, ?)",
            (severity, source, "jupiter", message),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"QCC alert write failed: {e}")


def post_eod_report(state: PaperState, spot: float, vix: Optional[float]) -> None:
    eq_now, unrl = equity(state, spot)
    pnl_total = eq_now - STARTING_CASH
    pnl_pct = pnl_total / STARTING_CASH * 100.0

    n_pos = len(state.positions)
    pos_lines = []
    for p in state.positions[:5]:
        pos_lines.append(
            f"  - SPY put @ {p['strike']:.0f} exp {p['expiration']} "
            f"x{p['contracts']} entry {p['entry_premium']:.2f}"
        )
    pos_str = "\n".join(pos_lines) if pos_lines else "  (none)"

    msg = (
        f"[EOD] Tier2 Balanced Scalp paper wheel\n"
        f"Equity: ${eq_now:,.0f} ({pnl_pct:+.2f}%)\n"
        f"Realized: ${state.realized_pnl:+,.0f}  Unrealized: ${unrl:+,.0f}\n"
        f"Trades: {state.trade_count}  Open positions: {n_pos}\n"
        f"SPY: ${spot:.2f}  VIX: {vix if vix is not None else 'n/a'}\n"
        f"{pos_str}"
    )
    send_qcc_alert("info", "wheel_paper_engine", msg)
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
    """One full decision cycle. Returns the action summary."""
    cfg = TIER2_BALANCED_SCALP
    S = fetch_spot(cfg["underlying"])
    if S is None:
        return {"error": "spot_fetch_failed"}
    vix = fetch_vix()
    chain = fetch_chain_for_dte_band(cfg["underlying"], cfg["dte_min"], cfg["dte_max"])
    if chain is None:
        log.warning("chain unavailable; managing existing positions only")
        actions = decide_and_act(state, S, vix, pd.DataFrame())
    else:
        actions = decide_and_act(state, S, vix, chain)

    eq_now, unrl = equity(state, S)
    state.equity_curve.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "equity": eq_now, "realized": state.realized_pnl, "unrealized": unrl,
        "spot": S, "vix": vix,
    })
    state.last_decision_ts = datetime.now(timezone.utc).isoformat()
    save_state(state)

    # log equity row
    with open(EQUITY_LOG, "a") as f:
        if f.tell() == 0:
            f.write("ts,equity,realized,unrealized,spot,vix\n")
        f.write(f"{state.last_decision_ts},{eq_now:.2f},{state.realized_pnl:.2f},"
                f"{unrl:.2f},{S:.2f},{vix if vix is not None else ''}\n")

    mlflow_log_metrics({
        "equity": eq_now,
        "realized_pnl": state.realized_pnl,
        "unrealized_pnl": unrl,
        "open_positions": len(state.positions),
        "spot": S,
        "vix": vix if vix is not None else float("nan"),
    }, step=state.trade_count)

    log.info(f"cycle done: spot={S:.2f} vix={vix} equity=${eq_now:.0f} "
             f"realized=${state.realized_pnl:+.0f} unrealized=${unrl:+.0f} "
             f"open={len(state.positions)} actions={actions['decisions']}")
    return actions


def maybe_eod(state: PaperState, force: bool = False) -> None:
    """Send EOD report once per UTC date after ~21:00 UTC (post-US close)."""
    today = datetime.utcnow().date().isoformat()
    now_utc_hour = datetime.utcnow().hour
    should = force or (now_utc_hour >= 21 and state.last_eod_date != today)
    if not should:
        return
    cfg = TIER2_BALANCED_SCALP
    S = fetch_spot(cfg["underlying"]) or 0.0
    vix = fetch_vix()
    post_eod_report(state, S, vix)
    state.last_eod_date = today
    save_state(state)


def run_daemon():
    log.info("=== wheel_paper_engine daemon starting ===")
    log.info(f"config: {json.dumps(TIER2_BALANCED_SCALP)}")
    init_mlflow()
    state = load_state()
    log.info(f"state loaded: cash=${state.cash:.2f} positions={len(state.positions)} "
             f"trades={state.trade_count}")
    try:
        while _RUNNING:
            try:
                cycle(state)
                maybe_eod(state)
            except Exception as e:
                log.exception(f"cycle error: {e}")
            # sleep
            for _ in range(WORK_INTERVAL_SECONDS):
                if not _RUNNING:
                    break
                time.sleep(1)
    finally:
        mlflow_end()
        log.info("daemon exiting")


def run_smoke():
    """One-shot: load, fetch, decide, log, exit. Used to validate plumbing."""
    log.info("=== SMOKE TEST ===")
    init_mlflow()
    state = load_state()
    log.info(f"state: cash=${state.cash:.2f} positions={len(state.positions)} "
             f"trades={state.trade_count}")
    out = cycle(state)
    log.info(f"smoke decisions: {json.dumps(out, indent=2)}")
    # don't post EOD on smoke
    mlflow_end()
    log.info("=== SMOKE OK ===")
    return out


def run_eod_only():
    """Force-emit the EOD report. Useful for cron."""
    init_mlflow()
    state = load_state()
    maybe_eod(state, force=True)
    mlflow_end()


# ===================== CLI =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one-shot smoke test")
    ap.add_argument("--eod", action="store_true", help="emit EOD report and exit")
    ap.add_argument("--daemon", action="store_true", help="run daemon loop")
    args = ap.parse_args()

    if args.smoke:
        run_smoke()
        return
    if args.eod:
        run_eod_only()
        return
    # default = daemon
    run_daemon()


if __name__ == "__main__":
    main()
