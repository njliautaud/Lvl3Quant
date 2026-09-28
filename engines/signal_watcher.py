#!/usr/bin/env python3
"""
Always-On Signal Watcher — HC #712
Monitors market conditions and fires alerts for validated trading signals.
Runs via PM2, checks every 5min during market hours, 30min outside.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytz

ET = pytz.timezone("US/Eastern")

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "signal_watcher_state.json"
ALERTS_FILE = STATE_DIR / "pending_alerts.json"
GATE_STATE_FILE = ROOT / "output" / "growth_research" / "daily_signals" / "gate_state.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SignalWatcher] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "signal_watcher.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("signal_watcher")

# ── Alert cooldown ──────────────────────────────────────────────────────────
COOLDOWN_SECONDS = 4 * 3600  # 4 hours

# ── VIX thresholds (mapped to VIXY proxy levels) ──────────────────────────
VIX_ELEVATED = 25.0
VIX_PANIC = 30.0

# ── Confluence gate thresholds ──────────────────────────────────────────────
CONFLUENCE_ENTRY = 2.5
CONFLUENCE_EXIT = 2.0

# ── Breadth threshold ──────────────────────────────────────────────────────
BREADTH_COLLAPSE_PCT = 30.0

# ── Sector ETFs for breadth ────────────────────────────────────────────────
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLB", "XLRE"]

# ── Protection overlay signals ─────────────────────────────────────────────
# VIX < 20, SPY > 50SMA, credit OK (LQD/HYG stable), breadth > 50%


# ═══════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Corrupt state file, starting fresh")
    return _default_state()


def _default_state() -> dict:
    return {
        "vix_regime": "NORMAL",          # NORMAL / ELEVATED / PANIC
        "confluence_score": None,
        "confluence_regime": None,        # UPRO / SPY / GLD
        "breadth_pct": None,
        "breadth_collapsed": False,
        "credit_stressed": False,
        "protection_status": "UNKNOWN",   # ALL_GREEN / WARNING
        "last_alert_times": {},           # alert_key -> ISO timestamp
        "last_check": None,
        "last_data_fetch": None,
        "version": "signal-watcher-v1",
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


# ═══════════════════════════════════════════════════════════════════════════
# ALERT DELIVERY
# ═══════════════════════════════════════════════════════════════════════════

def emit_alert(title: str, body: str, conviction: str = "NORMAL", state: dict = None):
    """
    Write alert to pending_alerts.json and stdout.
    conviction: NORMAL, HIGH
    """
    now = datetime.now(ET)
    alert_key = title.replace(" ", "_").lower()

    # ── Cooldown check ──
    if state and alert_key in state.get("last_alert_times", {}):
        last_time_str = state["last_alert_times"][alert_key]
        try:
            last_time = datetime.fromisoformat(last_time_str)
            if last_time.tzinfo is None:
                last_time = ET.localize(last_time)
            elapsed = (now - last_time).total_seconds()
            if elapsed < COOLDOWN_SECONDS:
                remaining = (COOLDOWN_SECONDS - elapsed) / 60
                log.info(f"Alert '{title}' suppressed (cooldown, {remaining:.0f}m remaining)")
                return
        except (ValueError, TypeError):
            pass

    # ── Build alert ──
    alert = {
        "timestamp": now.isoformat(),
        "title": title,
        "body": body,
        "conviction": conviction,
        "source": "signal_watcher",
    }

    # ── Write to pending_alerts.json ──
    try:
        existing = []
        if ALERTS_FILE.exists():
            try:
                existing = json.loads(ALERTS_FILE.read_text())
                if not isinstance(existing, list):
                    existing = []
            except (json.JSONDecodeError, OSError):
                existing = []
        existing.append(alert)
        # Keep only last 100 alerts
        if len(existing) > 100:
            existing = existing[-100:]
        ALERTS_FILE.write_text(json.dumps(existing, indent=2))
    except OSError as e:
        log.error(f"Failed to write alert file: {e}")

    # ── Print to stdout (PM2 logs) ──
    marker = "*** HIGH CONVICTION ***" if conviction == "HIGH" else "---"
    print(f"\n{marker} SIGNAL ALERT: {title} {marker}")
    print(f"  {body}")
    print(f"  Time: {now.strftime('%Y-%m-%d %H:%M ET')}")
    print()

    # ── Update cooldown ──
    if state is not None:
        if "last_alert_times" not in state:
            state["last_alert_times"] = {}
        state["last_alert_times"][alert_key] = now.isoformat()

    log.info(f"Alert fired: [{conviction}] {title}")


# ═══════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════

def fetch_market_data() -> dict | None:
    """
    Fetch all required market data via yfinance.
    Returns dict with DataFrames or None on failure.
    """
    try:
        import yfinance as yf
        import pandas as pd

        # All tickers we need
        all_tickers = ["SPY", "^VIX", "VIXY", "LQD", "HYG"] + SECTOR_ETFS
        log.info(f"Fetching data for {len(all_tickers)} tickers...")

        data = yf.download(
            all_tickers,
            period="300d",
            auto_adjust=True,
            threads=True,
            progress=False,
        )

        if data.empty:
            log.error("yfinance returned empty data")
            return None

        # Extract closes
        if isinstance(data.columns, pd.MultiIndex):
            closes = data["Close"]
        else:
            closes = data

        # Clean up multi-level columns if present
        if hasattr(closes.columns, "droplevel"):
            try:
                closes.columns = closes.columns.droplevel(1)
            except Exception:
                pass

        closes = closes.dropna(how="all")

        # Verify we have SPY at minimum
        if "SPY" not in closes.columns or closes["SPY"].dropna().empty:
            log.error("No SPY data available")
            return None

        return {"closes": closes}

    except Exception as e:
        log.error(f"Data fetch failed: {e}")
        log.debug(traceback.format_exc())
        return None


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════

def _safe_last(series):
    """Get last non-NaN value from a series, or None."""
    if series is None:
        return None
    clean = series.dropna()
    if clean.empty:
        return None
    val = clean.iloc[-1]
    if hasattr(val, "item"):
        val = val.item()
    return float(val) if not np.isnan(val) else None


def check_vix_signals(closes, state: dict) -> list[dict]:
    """Check VIX/VIXY panic signals (Strategy 1A)."""
    alerts = []

    # Try VIX directly first, fall back to VIXY
    vix_level = None
    if "^VIX" in closes.columns:
        vix_level = _safe_last(closes["^VIX"])

    vixy_level = None
    if "VIXY" in closes.columns:
        vixy_level = _safe_last(closes["VIXY"])

    if vix_level is None and vixy_level is None:
        log.warning("No VIX or VIXY data available")
        return alerts

    # Use VIX directly if available, otherwise estimate from VIXY
    # VIXY roughly tracks VIX but at different scale; use VIX when possible
    effective_vix = vix_level if vix_level is not None else None

    old_regime = state.get("vix_regime", "NORMAL")
    new_regime = "NORMAL"

    if effective_vix is not None:
        if effective_vix >= VIX_PANIC:
            new_regime = "PANIC"
        elif effective_vix >= VIX_ELEVATED:
            new_regime = "ELEVATED"

    # Only alert on regime changes
    if new_regime != old_regime:
        if new_regime == "ELEVATED" and old_regime == "NORMAL":
            alerts.append({
                "title": "VIX ELEVATED",
                "body": (
                    f"VIX crossed above {VIX_ELEVATED:.0f} (current: {effective_vix:.1f}). "
                    f"Market stress rising. Monitor for panic buy opportunity."
                ),
                "conviction": "NORMAL",
            })
        elif new_regime == "PANIC":
            alerts.append({
                "title": "PANIC BUY SIGNAL",
                "body": (
                    f"VIX crossed above {VIX_PANIC:.0f} (current: {effective_vix:.1f}). "
                    f"HIGH CONVICTION: Buy SPY. Hold 5-20 days. "
                    f"Historical WR 59-68%, PF 1.50-2.62. "
                    f"Strategy 1A validated (p=0.035)."
                ),
                "conviction": "HIGH",
            })
        # HC #747: VIX options put spread opportunity alert
        if new_regime == "ELEVATED" and old_regime == "NORMAL":
            alerts.append({
                "title": "VIX OPTIONS: PUT SPREAD OPPORTUNITY",
                "body": (
                    f"VIX spiked above 25 (current: {effective_vix:.1f}). "
                    f"VIX put spreads (buy 25P / sell 20P, 30-45 DTE) have "
                    f"Sharpe 1.09, 66% WR — VIX reverts to <20 within 30 days 68% of the time. "
                    f"Main account play, ~$200-300 per spread."
                ),
                "conviction": "HIGH",
            })
        elif new_regime == "NORMAL" and old_regime in ("ELEVATED", "PANIC"):
            alerts.append({
                "title": "VIX NORMALIZED",
                "body": f"VIX dropped back below {VIX_ELEVATED:.0f} (current: {effective_vix:.1f}). Stress subsiding.",
                "conviction": "NORMAL",
            })

    state["vix_regime"] = new_regime
    state["vix_level"] = effective_vix
    return alerts


def compute_confluence_score(spy_close) -> float:
    """
    Compute 3-timeframe confluence score (0-3).
    SHORT: 5d momentum > 0 (+0.5), 10d RSI > 50 (+0.5)
    MEDIUM: 20d > 50d MA (+0.5), 21d vol < 15% (+0.5)
    LONG: 200d MA slope > 0 (+0.5), 63d vol trend declining (+0.5)
    """
    spy = spy_close.dropna()
    if len(spy) < 200:
        return 0.0

    score = 0.0

    # SHORT: 5d momentum
    mom_5d = float(spy.iloc[-1] / spy.iloc[-6] - 1) if len(spy) > 5 else 0.0
    if mom_5d > 0:
        score += 0.5

    # SHORT: 10d RSI > 50
    delta = spy.diff()
    gain = delta.clip(lower=0).rolling(10).mean()
    loss = (-delta.clip(upper=0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = _safe_last(rsi)
    if rsi_val is not None and rsi_val > 50:
        score += 0.5

    # MEDIUM: 20d > 50d MA
    ma20 = spy.rolling(20).mean()
    ma50 = spy.rolling(50).mean()
    ma20_last = _safe_last(ma20)
    ma50_last = _safe_last(ma50)
    if ma20_last is not None and ma50_last is not None and ma20_last > ma50_last:
        score += 0.5

    # MEDIUM: 21d annualized vol < 15%
    ret = spy.pct_change()
    vol_21d = ret.rolling(21).std() * np.sqrt(252)
    vol_val = _safe_last(vol_21d)
    if vol_val is not None and vol_val < 0.15:
        score += 0.5

    # LONG: 200d MA slope > 0
    ma200 = spy.rolling(200).mean()
    if len(ma200.dropna()) >= 5:
        slope = float(ma200.dropna().iloc[-1] - ma200.dropna().iloc[-5])
        if slope > 0:
            score += 0.5

    # LONG: 63d vol trend declining
    vol_63d = ret.rolling(63).std() * np.sqrt(252)
    if len(vol_63d.dropna()) >= 21:
        vol_recent = float(vol_63d.dropna().iloc[-1])
        vol_earlier = float(vol_63d.dropna().iloc[-21])
        if vol_recent < vol_earlier:
            score += 0.5

    return round(score, 1)


def check_confluence_signals(closes, state: dict) -> list[dict]:
    """Check gameplan v3 confluence regime changes."""
    alerts = []

    if "SPY" not in closes.columns:
        return alerts

    spy_close = closes["SPY"].dropna()
    if len(spy_close) < 200:
        log.warning("Insufficient SPY data for confluence calculation")
        return alerts

    score = compute_confluence_score(spy_close)
    state["confluence_score"] = score

    old_regime = state.get("confluence_regime")

    # Also read gate_state.json for current holding info
    gate_info = ""
    try:
        if GATE_STATE_FILE.exists():
            gate = json.loads(GATE_STATE_FILE.read_text())
            in_upro = gate.get("in_upro", False)
            gate_info = f" Gate state: {'UPRO' if in_upro else 'SPY'}."
    except (json.JSONDecodeError, OSError):
        pass

    # Determine regime
    new_regime = old_regime
    if score >= CONFLUENCE_ENTRY:
        new_regime = "UPRO"
    elif score < CONFLUENCE_EXIT:
        new_regime = "SPY"
    # Between 2.0-2.5: keep current (hysteresis)

    if new_regime != old_regime and old_regime is not None:
        alerts.append({
            "title": f"REGIME CHANGE: {old_regime} -> {new_regime}",
            "body": (
                f"Confluence score moved to {score:.1f} (entry={CONFLUENCE_ENTRY}, exit={CONFLUENCE_EXIT}). "
                f"New regime: {new_regime}.{gate_info}"
            ),
            "conviction": "NORMAL",
        })

    state["confluence_regime"] = new_regime
    return alerts


def check_breadth_signals(closes, state: dict) -> list[dict]:
    """Check breadth collapse (% of sectors above 50d SMA)."""
    alerts = []

    available = [etf for etf in SECTOR_ETFS if etf in closes.columns]
    if len(available) < 6:
        log.warning(f"Only {len(available)} sector ETFs available, need at least 6")
        return alerts

    above_50sma = 0
    total = 0
    for etf in available:
        series = closes[etf].dropna()
        if len(series) < 50:
            continue
        total += 1
        sma50 = series.rolling(50).mean()
        current = _safe_last(series)
        sma_val = _safe_last(sma50)
        if current is not None and sma_val is not None and current > sma_val:
            above_50sma += 1

    if total == 0:
        return alerts

    breadth_pct = (above_50sma / total) * 100
    state["breadth_pct"] = round(breadth_pct, 1)

    old_collapsed = state.get("breadth_collapsed", False)
    new_collapsed = breadth_pct < BREADTH_COLLAPSE_PCT

    if new_collapsed and not old_collapsed:
        alerts.append({
            "title": "BREADTH COLLAPSE",
            "body": (
                f"Only {breadth_pct:.0f}% of sector ETFs above 50d SMA "
                f"({above_50sma}/{total} sectors). "
                f"PANIC BUY CONFIRMATION (Strategy 1A, p=0.000)."
            ),
            "conviction": "HIGH",
        })
    elif not new_collapsed and old_collapsed:
        alerts.append({
            "title": "BREADTH RECOVERY",
            "body": f"Breadth recovered to {breadth_pct:.0f}% of sectors above 50d SMA.",
            "conviction": "NORMAL",
        })

    state["breadth_collapsed"] = new_collapsed
    return alerts


def check_credit_signals(closes, state: dict) -> list[dict]:
    """Check credit stress (LQD/HYG + VIX)."""
    alerts = []

    if "HYG" not in closes.columns:
        log.warning("No HYG data for credit stress check")
        return alerts

    hyg = closes["HYG"].dropna()
    if len(hyg) < 6:
        return alerts

    # HYG 5-day return
    hyg_5d_ret = float(hyg.iloc[-1] / hyg.iloc[-6] - 1) if len(hyg) > 5 else 0.0

    # Check VIX level
    vix_level = state.get("vix_level")

    old_stressed = state.get("credit_stressed", False)
    new_stressed = hyg_5d_ret < -0.02 and vix_level is not None and vix_level > VIX_ELEVATED

    if new_stressed and not old_stressed:
        alerts.append({
            "title": "CREDIT STRESS SIGNAL",
            "body": (
                f"HYG dropped {hyg_5d_ret*100:.1f}% over 5 days while VIX at {vix_level:.1f}. "
                f"Credit stress confirms panic signal (Strategy 1A, p=0.005)."
            ),
            "conviction": "HIGH",
        })
    elif not new_stressed and old_stressed:
        alerts.append({
            "title": "CREDIT STRESS EASED",
            "body": f"HYG stabilized (5d: {hyg_5d_ret*100:.1f}%). Credit conditions improving.",
            "conviction": "NORMAL",
        })

    state["credit_stressed"] = new_stressed
    return alerts


def check_protection_overlay(closes, state: dict) -> list[dict]:
    """Check 4-signal protection overlay status."""
    alerts = []

    vix_level = state.get("vix_level")
    breadth_pct = state.get("breadth_pct")

    signals = {}

    # Signal 1: VIX < 20
    signals["vix_calm"] = vix_level is not None and vix_level < 20

    # Signal 2: SPY > 50 SMA
    if "SPY" in closes.columns:
        spy = closes["SPY"].dropna()
        if len(spy) >= 50:
            spy_last = _safe_last(spy)
            spy_sma50 = _safe_last(spy.rolling(50).mean())
            signals["spy_above_50sma"] = (
                spy_last is not None and spy_sma50 is not None and spy_last > spy_sma50
            )
        else:
            signals["spy_above_50sma"] = None
    else:
        signals["spy_above_50sma"] = None

    # Signal 3: Credit OK (HYG not dropping)
    if "HYG" in closes.columns:
        hyg = closes["HYG"].dropna()
        if len(hyg) > 5:
            hyg_5d = float(hyg.iloc[-1] / hyg.iloc[-6] - 1)
            signals["credit_ok"] = hyg_5d > -0.01
        else:
            signals["credit_ok"] = None
    else:
        signals["credit_ok"] = None

    # Signal 4: Breadth > 50%
    signals["breadth_ok"] = breadth_pct is not None and breadth_pct > 50

    # Determine status
    green_count = sum(1 for v in signals.values() if v is True)
    total_known = sum(1 for v in signals.values() if v is not None)

    if total_known == 0:
        new_status = "UNKNOWN"
    elif green_count == total_known:
        new_status = "ALL_GREEN"
    else:
        new_status = "WARNING"

    old_status = state.get("protection_status", "UNKNOWN")

    if new_status != old_status and old_status != "UNKNOWN":
        if new_status == "ALL_GREEN":
            alerts.append({
                "title": "PROTECTION: ALL GREEN",
                "body": (
                    f"All protection signals green ({green_count}/{total_known}). "
                    f"Full risk-on conditions."
                ),
                "conviction": "NORMAL",
            })
        elif new_status == "WARNING" and old_status == "ALL_GREEN":
            failing = [k for k, v in signals.items() if v is False]
            alerts.append({
                "title": "PROTECTION: WARNING",
                "body": (
                    f"Protection overlay degraded ({green_count}/{total_known} green). "
                    f"Failing: {', '.join(failing)}. Consider reducing exposure."
                ),
                "conviction": "NORMAL",
            })

    state["protection_status"] = new_status
    state["protection_signals"] = {k: v for k, v in signals.items()}
    return alerts


def check_high_conviction_combo(state: dict) -> list[dict]:
    """Check if multiple panic signals align = HIGH CONVICTION."""
    alerts = []

    vix_panic = state.get("vix_regime") == "PANIC"
    breadth_collapse = state.get("breadth_collapsed", False)
    credit_stress = state.get("credit_stressed", False)

    active_count = sum([vix_panic, breadth_collapse, credit_stress])

    if active_count >= 2:
        parts = []
        if vix_panic:
            parts.append(f"VIX>{VIX_PANIC:.0f}")
        if breadth_collapse:
            parts.append("Breadth collapsed")
        if credit_stress:
            parts.append("Credit stress")

        alerts.append({
            "title": "MULTI-SIGNAL PANIC BUY",
            "body": (
                f"HIGH CONVICTION trade: {active_count}/3 panic signals active "
                f"({', '.join(parts)}). "
                f"Buy SPY. Hold 5-20 days. Historical edge is strongest when multiple signals confirm."
            ),
            "conviction": "HIGH",
        })

    return alerts


# ═══════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_check_cycle(state: dict) -> dict:
    """Run one full check cycle. Returns updated state."""
    now = datetime.now(ET)
    log.info(f"Starting check cycle at {now.strftime('%Y-%m-%d %H:%M:%S ET')}")

    # Fetch data
    data = fetch_market_data()
    if data is None:
        log.error("Data fetch failed, skipping cycle")
        state["last_check"] = now.isoformat()
        state["last_check_status"] = "FETCH_FAILED"
        return state

    closes = data["closes"]
    state["last_data_fetch"] = now.isoformat()

    # Run all signal checks
    all_alerts = []

    try:
        all_alerts.extend(check_vix_signals(closes, state))
    except Exception as e:
        log.error(f"VIX check failed: {e}")
        log.debug(traceback.format_exc())

    try:
        all_alerts.extend(check_confluence_signals(closes, state))
    except Exception as e:
        log.error(f"Confluence check failed: {e}")
        log.debug(traceback.format_exc())

    try:
        all_alerts.extend(check_breadth_signals(closes, state))
    except Exception as e:
        log.error(f"Breadth check failed: {e}")
        log.debug(traceback.format_exc())

    try:
        all_alerts.extend(check_credit_signals(closes, state))
    except Exception as e:
        log.error(f"Credit check failed: {e}")
        log.debug(traceback.format_exc())

    try:
        all_alerts.extend(check_protection_overlay(closes, state))
    except Exception as e:
        log.error(f"Protection overlay check failed: {e}")
        log.debug(traceback.format_exc())

    # Check high-conviction combo (uses state from above checks)
    try:
        all_alerts.extend(check_high_conviction_combo(state))
    except Exception as e:
        log.error(f"High conviction combo check failed: {e}")

    # Emit alerts
    for alert in all_alerts:
        emit_alert(alert["title"], alert["body"], alert.get("conviction", "NORMAL"), state)

    # Update state
    state["last_check"] = now.isoformat()
    state["last_check_status"] = "OK"
    state["alerts_fired_this_cycle"] = len(all_alerts)

    # Log summary
    log.info(
        f"Cycle complete: VIX={state.get('vix_level', '?')}, "
        f"confluence={state.get('confluence_score', '?')}, "
        f"breadth={state.get('breadth_pct', '?')}%, "
        f"protection={state.get('protection_status', '?')}, "
        f"alerts={len(all_alerts)}"
    )

    return state


def get_sleep_seconds() -> int:
    """Return sleep duration: 5min during market hours, 30min outside."""
    now = datetime.now(ET)
    hour = now.hour
    minute = now.minute
    weekday = now.weekday()  # 0=Mon, 6=Sun

    # Weekend
    if weekday >= 5:
        return 30 * 60

    # Market hours: 9:30 - 16:00 ET
    market_open = hour > 9 or (hour == 9 and minute >= 30)
    market_close = hour < 16

    if market_open and market_close:
        return 5 * 60  # 5 minutes
    else:
        return 30 * 60  # 30 minutes


def main():
    parser = argparse.ArgumentParser(description="Signal Watcher — HC #712")
    parser.add_argument("--test", action="store_true", help="Run one check cycle and exit")
    parser.add_argument("--status", action="store_true", help="Print current state and exit")
    args = parser.parse_args()

    state = load_state()

    if args.status:
        print(json.dumps(state, indent=2))
        return

    if args.test:
        log.info("=== TEST MODE: Running single check cycle ===")
        state = run_check_cycle(state)
        save_state(state)
        print("\n=== Current State ===")
        print(json.dumps(state, indent=2))
        log.info("=== TEST MODE complete ===")
        return

    # ── Persistent loop ──
    log.info("Signal Watcher starting in persistent mode")
    log.info(f"State file: {STATE_FILE}")
    log.info(f"Alerts file: {ALERTS_FILE}")

    while True:
        try:
            state = run_check_cycle(state)
            save_state(state)
        except Exception as e:
            log.error(f"Check cycle crashed: {e}")
            log.error(traceback.format_exc())
            # Don't crash the process, just wait and retry
            state["last_check_status"] = f"CRASH: {str(e)[:100]}"
            save_state(state)

        sleep_sec = get_sleep_seconds()
        log.info(f"Sleeping {sleep_sec // 60}m until next check")
        time.sleep(sleep_sec)


if __name__ == "__main__":
    main()
