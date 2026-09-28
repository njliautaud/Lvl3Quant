"""
Unified Signal Watcher — monitors multiple signal families and alerts on changes.

Designed for PM2 cron invocation:
  - Every 30 minutes during market hours (9:30-16:00 ET)
  - Every 2 hours overnight

Signal families:
  1A. VIX Panic Reversal (highest conviction, 2-6x/year)
  1B. Gameplan v3 Regime Change (confluence-gated UPRO/SPY/GLD)
  1C. CTA Trend Signals (commodity/macro trend monitor)
  1D. Protection Overlay Status (4-signal risk dashboard)

Usage:
  python3 unified_signal_watcher.py              # run all checks
  python3 unified_signal_watcher.py --status      # print current state
  python3 unified_signal_watcher.py --force       # ignore cooldowns, alert anyway
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# VIX spike action engine — generates specific strike/sizing recommendations
try:
    from engines.vix_spike_action import generate_vix_spike_playbook
except ImportError:
    try:
        from vix_spike_action import generate_vix_spike_playbook
    except ImportError:
        generate_vix_spike_playbook = None
import pytz
import yfinance as yf

# ─── Paths ───────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "signal_watcher_state.json"
PENDING_ALERTS_FILE = STATE_DIR / "pending_discord_alerts.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

ET = pytz.timezone("US/Eastern")

# ─── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SIG-WATCH] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "signal_watcher.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("signal_watcher")

# ─── Constants ───────────────────────────────────────────────────────────────

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
CTA_ETFS = ["GLD", "SLV", "USO", "UNG", "DBA", "COPX", "UUP", "TLT", "EEM"]

# Minimum hours between repeated alerts of the same type
ALERT_COOLDOWN_HOURS = 4.0

# ─── State Management ───────────────────────────────────────────────────────


def _default_state() -> dict:
    """Return a fresh default state."""
    return {
        "last_run": None,
        "last_alerts": {},          # alert_type -> ISO timestamp
        "regime": {
            "in_upro": False,
            "confluence_score": 0.0,
            "vol_tier": "SPY",
        },
        "cta": {
            "uptrend_tickers": [],
            "last_monday_report": None,
        },
        "protection": {
            "vix_ok": True,
            "spy_above_50sma": True,
            "credit_healthy": True,
            "breadth_ok": True,
        },
        "vix_panic": {
            "last_peak_vix": None,
            "signal_active": False,
        },
    }


def load_state() -> dict:
    """Load persisted state, merging with defaults for any missing keys."""
    defaults = _default_state()
    if STATE_FILE.exists():
        try:
            saved = json.loads(STATE_FILE.read_text())
            # Deep-merge: ensure all top-level keys and nested dicts exist
            for key, default_val in defaults.items():
                if key not in saved:
                    saved[key] = default_val
                elif isinstance(default_val, dict) and isinstance(saved.get(key), dict):
                    for sub_key, sub_val in default_val.items():
                        if sub_key not in saved[key]:
                            saved[key][sub_key] = sub_val
            return saved
        except (json.JSONDecodeError, OSError):
            log.warning("Corrupt state file, starting fresh")
    return defaults


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_state(state: dict) -> None:
    """Persist state to disk."""
    state["last_run"] = datetime.now(ET).isoformat()
    STATE_FILE.write_text(json.dumps(state, indent=2, cls=_NumpyEncoder))


def queue_discord_alerts(alerts: list[str]) -> None:
    """Write alerts to pending file for Discord pickup by Claude's cycle."""
    if not alerts:
        return
    # Load existing pending alerts
    pending = []
    if PENDING_ALERTS_FILE.exists():
        try:
            pending = json.loads(PENDING_ALERTS_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            pending = []

    now = datetime.now(ET).isoformat()
    for alert_text in alerts:
        # Determine severity and channel
        is_high_conviction = any(kw in alert_text.upper() for kw in [
            "VIX PANIC", "HIGH CONVICTION", "ACTIONABLE", "BUY SIGNAL"
        ])
        channel = "alerts" if is_high_conviction else "system-status"

        # Strip the [DISCORD_ALERT] prefix for cleaner message
        clean_text = alert_text.replace("[DISCORD_ALERT] ", "").strip()

        pending.append({
            "timestamp": now,
            "channel": channel,
            "message": clean_text,
            "high_conviction": is_high_conviction,
            "sent": False,
        })

    PENDING_ALERTS_FILE.write_text(json.dumps(pending, indent=2))
    log.info(f"Queued {len(alerts)} alert(s) for Discord delivery")


def can_alert(state: dict, alert_type: str, force: bool = False) -> bool:
    """Check cooldown — returns True if we can send this alert type."""
    if force:
        return True
    last = state["last_alerts"].get(alert_type)
    if last is None:
        return True
    try:
        last_dt = datetime.fromisoformat(last)
        if last_dt.tzinfo is None:
            last_dt = ET.localize(last_dt)
        elapsed = (datetime.now(ET) - last_dt).total_seconds() / 3600
        return elapsed >= ALERT_COOLDOWN_HOURS
    except (ValueError, TypeError):
        return True


def mark_alerted(state: dict, alert_type: str) -> None:
    """Record that we just sent an alert of this type."""
    state["last_alerts"][alert_type] = datetime.now(ET).isoformat()


# ─── Data Fetching ───────────────────────────────────────────────────────────


def fetch_prices(tickers: list[str], period: str = "120d") -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for a list of tickers. Returns {ticker: df}."""
    results = {}
    try:
        data = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        if data.empty:
            log.error("yfinance returned empty dataframe")
            return results

        if len(tickers) == 1:
            # Single ticker: data columns are just OHLCV
            results[tickers[0]] = data.dropna()
        else:
            # Multi-ticker: MultiIndex columns (field, ticker)
            for t in tickers:
                try:
                    df = data.xs(t, axis=1, level=1).dropna()
                    if not df.empty:
                        results[t] = df
                except (KeyError, TypeError):
                    log.warning(f"No data for {t}")
    except Exception as e:
        log.error(f"yfinance download failed: {e}")
    return results


def safe_series(data: dict[str, pd.DataFrame], ticker: str, col: str = "Close") -> pd.Series | None:
    """Extract a close series safely."""
    df = data.get(ticker)
    if df is None or df.empty:
        return None
    if col in df.columns:
        return df[col].dropna()
    return None


# ─── Signal 1A: VIX Panic Reversal ──────────────────────────────────────────


def check_vix_panic(data: dict[str, pd.DataFrame], state: dict, force: bool) -> list[str]:
    """
    VIX Panic Reversal (highest conviction signal, fires 2-6x/year).

    Conditions:
      - VIX (via VIXY) peaked >= 25 in last 10 trading days AND now < 22
      - Breadth: sector ETFs above 200d SMA. < 30% = breadth stress
      - Credit: HYG 5d return < -3% = credit stress

    Confluence score 0-3. Score >= 2 = HIGH CONVICTION.
    """
    alerts = []

    vixy = safe_series(data, "VIXY")
    if vixy is None or len(vixy) < 20:
        log.warning("VIX panic check: insufficient VIXY data")
        return alerts

    # VIXY is an ETF, not VIX itself. We use it as a proxy.
    # For the "VIX >= 25" check, we look at VIXY's relative behavior:
    # peak in last 10 days vs current level (percentage drop from peak)
    recent_10d = vixy.iloc[-10:]
    peak_val = recent_10d.max()
    current_val = vixy.iloc[-1]

    # VIXY doesn't map 1:1 to VIX levels, so we check for a significant
    # spike and reversal: peak was >= 20% above 60d mean AND current is
    # back within 10% of mean
    mean_60d = vixy.iloc[-60:].mean() if len(vixy) >= 60 else vixy.mean()
    peak_elevated = (peak_val / mean_60d - 1) >= 0.20  # peak was 20%+ above mean
    current_reverted = (current_val / mean_60d - 1) < 0.10  # back near mean

    vix_reversal = peak_elevated and current_reverted

    # Breadth check: sectors above 200d SMA
    sectors_above_200 = 0
    sectors_checked = 0
    for etf in SECTOR_ETFS:
        s = safe_series(data, etf)
        if s is not None and len(s) >= 200:
            sectors_checked += 1
            sma200 = s.iloc[-200:].mean()
            if s.iloc[-1] > sma200:
                sectors_above_200 += 1

    breadth_pct = sectors_above_200 / max(sectors_checked, 1)
    breadth_stress = breadth_pct < 0.30

    # Credit check: HYG 5d return
    hyg = safe_series(data, "HYG")
    credit_stress = False
    hyg_5d_ret = None
    if hyg is not None and len(hyg) >= 6:
        hyg_5d_ret = (hyg.iloc[-1] / hyg.iloc[-6] - 1) * 100
        credit_stress = hyg_5d_ret < -3.0

    # Confluence score
    score = 0
    reasons = []
    if vix_reversal:
        score += 1
        reasons.append(f"VIX spike reversed (VIXY peak {peak_val:.2f} -> {current_val:.2f}, mean {mean_60d:.2f})")
    if breadth_stress:
        score += 1
        reasons.append(f"Breadth stress ({sectors_above_200}/{sectors_checked} sectors above 200d SMA = {breadth_pct:.0%})")
    if credit_stress:
        score += 1
        reasons.append(f"Credit stress (HYG 5d return: {hyg_5d_ret:.1f}%)")

    # Update state
    state["vix_panic"]["last_peak_vix"] = float(peak_val)
    was_active = state["vix_panic"]["signal_active"]
    state["vix_panic"]["signal_active"] = score >= 2

    log.info(f"VIX Panic: score={score}/3, vix_reversal={vix_reversal}, "
             f"breadth={breadth_pct:.0%}, credit_stress={credit_stress}")

    if score >= 2 and can_alert(state, "vix_panic", force):
        alert = (
            "[DISCORD_ALERT] HIGH CONVICTION TRADE SIGNAL\n"
            f"VIX Panic Reversal — Confluence {score}/3\n"
            f"Action: Buy SPY or UPRO\n"
            f"Hold: 10-20 days\n"
            f"Historical: 68% win rate, Sharpe 1.38, PF 2.62\n"
            f"Why now: {'; '.join(reasons)}"
        )
        alerts.append(alert)
        mark_alerted(state, "vix_panic")
    elif was_active and score < 2:
        log.info("VIX panic signal deactivated (score dropped below threshold)")

    return alerts


# ─── Signal 1B: Gameplan v3 Regime Change ────────────────────────────────────


def compute_rsi(series: pd.Series, window: int = 10) -> float:
    """Compute RSI for the last `window` periods."""
    delta = series.diff().iloc[-window:]
    gain = delta.where(delta > 0, 0.0).mean()
    loss = -delta.where(delta < 0, 0.0).mean()
    if loss == 0:
        return 100.0
    rs = gain / loss
    return 100.0 - (100.0 / (1.0 + rs))


def check_regime_change(data: dict[str, pd.DataFrame], state: dict, force: bool) -> list[str]:
    """
    Gameplan v3 regime change detection.

    3-timeframe confluence score (0-3):
      Short: 5d momentum > 0 (+0.5), 10d RSI > 50 (+0.5)
      Medium: 20d > 50d MA (+0.5), 21d vol < 15% (+0.5)
      Long: 200d MA slope (20d change) > 0 (+0.5), 63d vol trend < 0 (+0.5)

    Hysteresis: entry gate >= 2.5, exit gate < 2.0
    Vol tiers: <15% UPRO, 15-30% SPY, >30% GLD
    """
    alerts = []

    spy = safe_series(data, "SPY")
    if spy is None or len(spy) < 210:
        log.warning("Regime check: insufficient SPY data")
        return alerts

    # --- Compute indicators ---
    # Short-term
    mom_5d = spy.iloc[-1] / spy.iloc[-6] - 1 if len(spy) >= 6 else 0
    rsi_10d = compute_rsi(spy, 10)

    # Medium-term
    sma20 = spy.iloc[-20:].mean()
    sma50 = spy.iloc[-50:].mean()

    # 21d realized vol (annualized)
    returns_21d = spy.pct_change().iloc[-21:]
    vol_21d = returns_21d.std() * np.sqrt(252) * 100  # annualized %

    # Long-term
    sma200_now = spy.iloc[-200:].mean()
    sma200_20ago = spy.iloc[-220:-20].mean() if len(spy) >= 220 else spy.iloc[-200:].mean()
    sma200_slope = sma200_now - sma200_20ago

    # 63d vol trend (current 21d vol vs 63d vol)
    returns_63d = spy.pct_change().iloc[-63:]
    vol_63d = returns_63d.std() * np.sqrt(252) * 100
    vol_trend = vol_21d - vol_63d  # negative = vol compressing (good)

    # --- Confluence score ---
    score = 0.0
    score_details = []

    if mom_5d > 0:
        score += 0.5
        score_details.append(f"5d momentum +{mom_5d:.2%}")
    else:
        score_details.append(f"5d momentum {mom_5d:.2%} (bearish)")

    if rsi_10d > 50:
        score += 0.5
        score_details.append(f"10d RSI {rsi_10d:.1f} (bullish)")
    else:
        score_details.append(f"10d RSI {rsi_10d:.1f} (bearish)")

    if sma20 > sma50:
        score += 0.5
        score_details.append("20d > 50d MA (bullish)")
    else:
        score_details.append("20d < 50d MA (bearish)")

    if vol_21d < 15:
        score += 0.5
        score_details.append(f"21d vol {vol_21d:.1f}% (low)")
    else:
        score_details.append(f"21d vol {vol_21d:.1f}% (elevated)")

    if sma200_slope > 0:
        score += 0.5
        score_details.append("200d MA slope positive")
    else:
        score_details.append("200d MA slope negative")

    if vol_trend < 0:
        score += 0.5
        score_details.append(f"Vol compressing ({vol_trend:+.1f}%)")
    else:
        score_details.append(f"Vol expanding ({vol_trend:+.1f}%)")

    # --- Vol tier ---
    if vol_21d < 15:
        vol_tier = "UPRO"
    elif vol_21d < 30:
        vol_tier = "SPY"
    else:
        vol_tier = "GLD"

    # --- Hysteresis logic ---
    prev_score = state["regime"].get("confluence_score", 0)
    was_in_upro = state["regime"].get("in_upro", False)

    if was_in_upro:
        in_upro = score >= 2.0  # exit gate
    else:
        in_upro = score >= 2.5  # entry gate

    # Override: vol tier must allow UPRO
    if vol_tier != "UPRO":
        in_upro = False

    # Detect regime change
    prev_tier = state["regime"].get("vol_tier", "SPY")
    regime_changed = (in_upro != was_in_upro) or (vol_tier != prev_tier)

    # Update state
    state["regime"]["confluence_score"] = score
    state["regime"]["in_upro"] = in_upro
    state["regime"]["vol_tier"] = vol_tier

    log.info(f"Regime: score={score:.1f}/3.0, vol={vol_21d:.1f}%, tier={vol_tier}, "
             f"in_upro={in_upro}, 20/200 cross={'above' if sma20 > sma200_now else 'below'}")

    if regime_changed and can_alert(state, "regime_change", force):
        if in_upro:
            new_pos = "UPRO"
        else:
            new_pos = vol_tier

        old_pos = "UPRO" if was_in_upro else prev_tier

        # Build reason string from score changes
        bearish_reasons = [d for d in score_details if "bearish" in d or "elevated" in d or "negative" in d or "expanding" in d]
        reason_str = "; ".join(bearish_reasons[:3]) if bearish_reasons else "Score shift"

        alert = (
            f"[DISCORD_ALERT] Regime Change: {old_pos} -> {new_pos}\n"
            f"Confluence score: {prev_score:.1f} -> {score:.1f}\n"
            f"Vol tier: {vol_tier} (21d vol: {vol_21d:.1f}%)\n"
            f"Reason: {reason_str}"
        )
        alerts.append(alert)
        mark_alerted(state, "regime_change")

    return alerts


# ─── Signal 1C: CTA Trend Signals ───────────────────────────────────────────


def check_cta_trends(data: dict[str, pd.DataFrame], state: dict, force: bool) -> list[str]:
    """
    CTA Trend Monitor: check 9 ETFs vs their 50d SMA.
    Alert when uptrend count changes by +-2 or more.
    Weekly Monday: list all assets above/below SMA50.
    """
    alerts = []

    uptrend_now = []
    downtrend_now = []

    for etf in CTA_ETFS:
        s = safe_series(data, etf)
        if s is None or len(s) < 50:
            continue
        sma50 = s.iloc[-50:].mean()
        if s.iloc[-1] > sma50:
            uptrend_now.append(etf)
        else:
            downtrend_now.append(etf)

    prev_uptrend = set(state["cta"].get("uptrend_tickers", []))
    curr_uptrend = set(uptrend_now)

    # Detect significant change
    newly_up = curr_uptrend - prev_uptrend
    newly_down = prev_uptrend - curr_uptrend
    total_change = len(newly_up) + len(newly_down)

    state["cta"]["uptrend_tickers"] = list(curr_uptrend)

    log.info(f"CTA Trends: {len(uptrend_now)}/{len(uptrend_now)+len(downtrend_now)} above SMA50 "
             f"(changes: +{len(newly_up)} -{len(newly_down)})")

    # Alert on significant change (+-2 or more)
    if total_change >= 2 and can_alert(state, "cta_shift", force):
        parts = []
        if newly_up:
            parts.append(f"Turned bullish: {', '.join(sorted(newly_up))}")
        if newly_down:
            parts.append(f"Turned bearish: {', '.join(sorted(newly_down))}")

        alert = (
            f"[DISCORD_ALERT] CTA Trend Shift ({total_change} assets changed)\n"
            f"Above SMA50: {', '.join(sorted(uptrend_now)) or 'none'}\n"
            f"Below SMA50: {', '.join(sorted(downtrend_now)) or 'none'}\n"
            f"{'; '.join(parts)}"
        )
        alerts.append(alert)
        mark_alerted(state, "cta_shift")

    # Monday weekly summary
    now_et = datetime.now(ET)
    last_monday = state["cta"].get("last_monday_report")
    is_monday = now_et.weekday() == 0

    if is_monday:
        already_reported = False
        if last_monday:
            try:
                last_dt = datetime.fromisoformat(last_monday)
                if last_dt.tzinfo is None:
                    last_dt = ET.localize(last_dt)
                already_reported = (now_et - last_dt).days < 1
            except (ValueError, TypeError):
                pass

        if not already_reported:
            alert = (
                f"[DISCORD_ALERT] Weekly CTA Trend Report (Monday)\n"
                f"Above SMA50 ({len(uptrend_now)}): {', '.join(sorted(uptrend_now)) or 'none'}\n"
                f"Below SMA50 ({len(downtrend_now)}): {', '.join(sorted(downtrend_now)) or 'none'}"
            )
            alerts.append(alert)
            state["cta"]["last_monday_report"] = now_et.isoformat()

    return alerts


# ─── Signal 1D: Protection Overlay ──────────────────────────────────────────


def check_protection_overlay(data: dict[str, pd.DataFrame], state: dict, force: bool) -> list[str]:
    """
    Protection Overlay: 4 binary signals.
      1. VIX < 20 (via VIXY relative to its mean)
      2. SPY > 50d SMA
      3. Credit healthy (LQD/HYG ratio not elevated)
      4. Breadth OK (IWM vs SPY 21d relative diff > -3%)

    Alert when any signal flips, especially from all-green to warning.
    """
    alerts = []

    # Signal 1: VIX calm (VIXY not elevated)
    vixy = safe_series(data, "VIXY")
    vix_ok = True
    if vixy is not None and len(vixy) >= 60:
        mean_60d = vixy.iloc[-60:].mean()
        vix_ok = vixy.iloc[-1] < mean_60d * 1.15  # within 15% of 60d mean

    # Signal 2: SPY above 50d SMA
    spy = safe_series(data, "SPY")
    spy_above_50sma = True
    if spy is not None and len(spy) >= 50:
        sma50 = spy.iloc[-50:].mean()
        spy_above_50sma = spy.iloc[-1] > sma50

    # Signal 3: Credit healthy (LQD/HYG ratio)
    lqd = safe_series(data, "LQD")
    hyg = safe_series(data, "HYG")
    credit_healthy = True
    if lqd is not None and hyg is not None and len(lqd) >= 60 and len(hyg) >= 60:
        # Align on common dates
        ratio = lqd / hyg
        ratio = ratio.dropna()
        if len(ratio) >= 60:
            ratio_now = ratio.iloc[-1]
            ratio_mean = ratio.iloc[-60:].mean()
            ratio_std = ratio.iloc[-60:].std()
            # Elevated ratio = flight to quality = credit stress
            credit_healthy = ratio_now < ratio_mean + 1.5 * ratio_std

    # Signal 4: Breadth OK (IWM vs SPY 21d relative performance)
    iwm = safe_series(data, "IWM")
    breadth_ok = True
    if iwm is not None and spy is not None and len(iwm) >= 22 and len(spy) >= 22:
        iwm_21d = (iwm.iloc[-1] / iwm.iloc[-22] - 1) * 100
        spy_21d = (spy.iloc[-1] / spy.iloc[-22] - 1) * 100
        breadth_diff = iwm_21d - spy_21d
        breadth_ok = breadth_diff > -3.0

    # Previous state (normalize string bools from legacy state files)
    prev = {}
    for k, v in state["protection"].items():
        if isinstance(v, str):
            prev[k] = v.lower() == "true"
        else:
            prev[k] = bool(v)

    # Detect flips (cast to Python bool for clean JSON serialization)
    signals = {
        "vix_ok": bool(vix_ok),
        "spy_above_50sma": bool(spy_above_50sma),
        "credit_healthy": bool(credit_healthy),
        "breadth_ok": bool(breadth_ok),
    }

    signal_labels = {
        "vix_ok": "VIX calm",
        "spy_above_50sma": "SPY > 50d SMA",
        "credit_healthy": "Credit healthy",
        "breadth_ok": "Breadth OK (IWM vs SPY)",
    }

    flipped = []
    for key, val in signals.items():
        prev_val = prev.get(key, True)
        if val != prev_val:
            direction = "OK" if val else "WARNING"
            flipped.append(f"{signal_labels[key]}: {direction}")

    # Count greens
    green_count = sum(bool(v) for v in signals.values())
    prev_green_count = sum(bool(prev.get(k, True)) for k in signals)

    # Update state
    state["protection"] = signals

    log.info(f"Protection: {green_count}/4 green "
             f"(VIX={'OK' if vix_ok else 'WARN'}, SPY_SMA={'OK' if spy_above_50sma else 'WARN'}, "
             f"Credit={'OK' if credit_healthy else 'WARN'}, Breadth={'OK' if breadth_ok else 'WARN'})")

    if flipped and can_alert(state, "protection_flip", force):
        severity = "WARNING" if green_count < prev_green_count else "IMPROVEMENT"

        status_lines = []
        for key, label in signal_labels.items():
            icon = "green" if signals[key] else "RED"
            status_lines.append(f"  {label}: {icon}")

        alert = (
            f"[DISCORD_ALERT] Protection Overlay: {severity}\n"
            f"Signals: {green_count}/4 green\n"
            f"Changed: {'; '.join(flipped)}\n"
            + "\n".join(status_lines)
        )
        alerts.append(alert)
        mark_alerted(state, "protection_flip")

    return alerts


# ─── Signal 1E: VMR Regime ───────────────────────────────────────────────────


def check_vmr_regime(data: dict[str, pd.DataFrame], state: dict, force: bool) -> list[str]:
    """
    VMR (Vol Mean Reversion) regime check.
    Uses VIX to determine 5 regimes, alerts on transitions.
    VALIDATED strategy: Sharpe 1.516, perm p=0.000.
    """
    alerts = []

    # Get VIX data (try ^VIX first, fall back to VIXY proxy)
    vix_series = safe_series(data, "^VIX")
    if vix_series is None:
        vix_series = safe_series(data, "VIX")
    if vix_series is None:
        # Approximate VIX from VIXY
        vixy = safe_series(data, "VIXY")
        if vixy is not None and len(vixy) >= 10:
            # VIXY tracks VIX short-term futures, rough proxy
            vix_level = vixy.iloc[-1]
            vix_ma10 = vixy.iloc[-10:].mean()
            vix_peak20 = vixy.iloc[-20:].max() if len(vixy) >= 20 else vixy.max()
        else:
            log.warning("VMR: No VIX data available")
            return alerts
    else:
        vix_level = float(vix_series.iloc[-1])
        vix_ma10 = float(vix_series.iloc[-10:].mean()) if len(vix_series) >= 10 else vix_level
        vix_peak20 = float(vix_series.iloc[-20:].max()) if len(vix_series) >= 20 else vix_level

    vix_declining = vix_level < vix_ma10

    # Determine regime
    if vix_level < 15 and vix_declining:
        regime = "UPRO"
        reason = f"VIX {vix_level:.1f} < 15, declining"
    elif vix_level > 20 and vix_level < vix_peak20 * 0.85 and vix_declining:
        regime = "UPRO_MR"
        reason = f"VIX {vix_level:.1f} mean-reverting (peak {vix_peak20:.1f})"
    elif vix_level > 25 and not vix_declining:
        regime = "DEFENSIVE"
        reason = f"VIX {vix_level:.1f} > 25, rising"
    elif vix_level > 20 and not vix_declining:
        regime = "CAUTIOUS"
        reason = f"VIX {vix_level:.1f} > 20, rising"
    else:
        regime = "SPY"
        reason = f"VIX {vix_level:.1f}, default"

    # Check for regime change
    if "vmr" not in state:
        state["vmr"] = {}
    prev_regime = state["vmr"].get("regime", "SPY")

    state["vmr"]["regime"] = regime
    state["vmr"]["vix"] = round(vix_level, 2)
    state["vmr"]["vix_ma10"] = round(vix_ma10, 2)
    state["vmr"]["vix_peak20"] = round(vix_peak20, 2)

    log.info(f"VMR: regime={regime}, VIX={vix_level:.1f}, MA10={vix_ma10:.1f}, peak20={vix_peak20:.1f}")

    if regime != prev_regime and can_alert(state, "vmr_regime", force):
        is_upro = regime in ("UPRO", "UPRO_MR")
        is_defensive = regime in ("DEFENSIVE", "CAUTIOUS")

        if is_upro:
            alloc = "UPRO (leveraged long)"
        elif regime == "DEFENSIVE":
            alloc = "50% GLD + 50% TLT (defensive)"
        elif regime == "CAUTIOUS":
            alloc = "50% SPY + 50% TLT (cautious)"
        else:
            alloc = "SPY (neutral)"

        alert = (
            f"[DISCORD_ALERT] VMR Regime Change: {prev_regime} → {regime}\n"
            f"Reason: {reason}\n"
            f"Allocation: {alloc}\n"
            f"VIX: {vix_level:.1f} (MA10: {vix_ma10:.1f}, Peak20: {vix_peak20:.1f})"
        )

        # High conviction if entering UPRO_MR (mean reversion entry)
        if regime == "UPRO_MR":
            alert = alert.replace("[DISCORD_ALERT]", "[DISCORD_ALERT] HIGH CONVICTION")

        alerts.append(alert)
        mark_alerted(state, "vmr_regime")

    return alerts


# ─── Signal 1F: VIX Spike Puts Alert ─────────────────────────────────────────


def check_vix_spike_puts(data: dict[str, pd.DataFrame], state: dict, force: bool) -> list[str]:
    """
    VIX Spike Puts Alert — fires when VIX crosses 30 (or 35/40 for heavier sizing).
    Per HC #714 R4: VIX puts are the preferred instrument (83% WR, Sharpe 1.76).
    This triggers the 2-6x/year opportunistic play.
    """
    alerts = []

    vix_series = safe_series(data, "^VIX")
    if vix_series is None:
        vix_series = safe_series(data, "VIX")
    if vix_series is None:
        return alerts

    vix_now = float(vix_series.iloc[-1])
    vix_prev = float(vix_series.iloc[-2]) if len(vix_series) >= 2 else vix_now

    if "vix_spike_puts" not in state:
        state["vix_spike_puts"] = {"last_level": None, "alerted_30": False, "alerted_35": False, "alerted_40": False}

    spike_state = state["vix_spike_puts"]

    # Reset alerts if VIX drops back below 25 (spike resolved)
    if vix_now < 25:
        if spike_state.get("alerted_30") or spike_state.get("alerted_35") or spike_state.get("alerted_40"):
            spike_state["alerted_30"] = False
            spike_state["alerted_35"] = False
            spike_state["alerted_40"] = False
            log.info("VIX spike puts: alerts reset (VIX < 25)")

    # Level 1: VIX crosses 30
    if vix_now >= 30 and not spike_state.get("alerted_30") and can_alert(state, "vix_spike_30", force):
        # Generate full actionable playbook with specific strikes
        if generate_vix_spike_playbook is not None:
            try:
                playbook = generate_vix_spike_playbook(vix_now, allocation=5000.0)
                alert = f"[DISCORD_ALERT] {playbook}"
            except Exception as e:
                log.error(f"VIX spike playbook generation failed: {e}")
                alert = (
                    f"[DISCORD_ALERT] 🚨 VIX SPIKE — BUY VIX PUTS\n"
                    f"VIX just hit {vix_now:.1f} (crossed 30 threshold)\n"
                    f"Action: Buy UVXY puts, 30-45 DTE, ATM or slightly OTM\n"
                    f"Historical: 83% win rate, avg +137% return per event\n"
                    f"(Strike calculator failed — check manually)"
                )
        else:
            alert = (
                f"[DISCORD_ALERT] 🚨 VIX SPIKE — BUY VIX PUTS\n"
                f"VIX just hit {vix_now:.1f} (crossed 30 threshold)\n"
                f"Action: Buy UVXY puts, 30-45 DTE, ATM or slightly OTM\n"
                f"Historical: 83% win rate, avg +137% return per event"
            )
        alerts.append(alert)
        spike_state["alerted_30"] = True
        mark_alerted(state, "vix_spike_30")
        log.info(f"VIX spike puts: Level 1 alert with playbook (VIX={vix_now:.1f})")

    # Level 2: VIX crosses 35 — size up
    if vix_now >= 35 and not spike_state.get("alerted_35") and can_alert(state, "vix_spike_35", force):
        if generate_vix_spike_playbook is not None:
            try:
                playbook = generate_vix_spike_playbook(vix_now, allocation=5000.0)
                alert = f"[DISCORD_ALERT] {playbook}"
            except Exception as e:
                log.error(f"VIX spike playbook generation failed: {e}")
                alert = (
                    f"[DISCORD_ALERT] 🚨🚨 VIX SPIKE ESCALATION — ADD TO VIX PUTS\n"
                    f"VIX now at {vix_now:.1f} (crossed 35)\n"
                    f"Action: Add second tranche of UVXY puts"
                )
        else:
            alert = (
                f"[DISCORD_ALERT] 🚨🚨 VIX SPIKE ESCALATION — ADD TO VIX PUTS\n"
                f"VIX now at {vix_now:.1f} (crossed 35)\n"
                f"Action: Add second tranche of UVXY puts"
            )
        alerts.append(alert)
        spike_state["alerted_35"] = True
        mark_alerted(state, "vix_spike_35")
        log.info(f"VIX spike puts: Level 2 alert with playbook (VIX={vix_now:.1f})")

    # Level 3: VIX crosses 40 — extreme, caution
    if vix_now >= 40 and not spike_state.get("alerted_40") and can_alert(state, "vix_spike_40", force):
        if generate_vix_spike_playbook is not None:
            try:
                playbook = generate_vix_spike_playbook(vix_now, allocation=5000.0)
                alert = f"[DISCORD_ALERT] {playbook}"
            except Exception as e:
                log.error(f"VIX spike playbook generation failed: {e}")
                alert = (
                    f"[DISCORD_ALERT] 🚨🚨🚨 EXTREME VIX SPIKE — {vix_now:.1f}\n"
                    f"Rare territory. Hold existing puts. Don't over-stage."
                )
        else:
            alert = (
                f"[DISCORD_ALERT] 🚨🚨🚨 EXTREME VIX SPIKE — {vix_now:.1f}\n"
                f"Rare territory. Hold existing puts. Don't over-stage."
            )
        alerts.append(alert)
        spike_state["alerted_40"] = True
        mark_alerted(state, "vix_spike_40")
        log.info(f"VIX spike puts: Level 3 alert with playbook (VIX={vix_now:.1f})")

    spike_state["last_level"] = round(vix_now, 2)
    state["vix_spike_puts"] = spike_state

    return alerts


# ─── Orchestrator ────────────────────────────────────────────────────────────


def run_all_checks(force: bool = False) -> list[str]:
    """Run all signal checks. Returns list of alert strings."""
    state = load_state()
    all_alerts: list[str] = []

    # Fetch all needed data in one batch
    all_tickers = (
        ["SPY", "VIXY", "HYG", "LQD", "IWM", "^VIX"]
        + SECTOR_ETFS
        + CTA_ETFS
    )
    # Deduplicate while preserving order
    seen = set()
    unique_tickers = []
    for t in all_tickers:
        if t not in seen:
            seen.add(t)
            unique_tickers.append(t)

    log.info(f"Fetching data for {len(unique_tickers)} tickers...")
    data = fetch_prices(unique_tickers, period="250d")  # need 200d+ for SMA200

    if not data:
        log.error("No data fetched — aborting all checks")
        save_state(state)
        return ["[DISCORD_ALERT] Signal watcher: data fetch failed entirely. Check yfinance."]

    log.info(f"Got data for {len(data)}/{len(unique_tickers)} tickers")

    # Run each signal family with error isolation
    checks = [
        ("VIX Panic Reversal", check_vix_panic),
        ("Regime Change", check_regime_change),
        ("CTA Trends", check_cta_trends),
        ("Protection Overlay", check_protection_overlay),
        ("VMR Regime", check_vmr_regime),
        ("VIX Spike Puts", check_vix_spike_puts),
    ]

    for name, fn in checks:
        try:
            alerts = fn(data, state, force)
            all_alerts.extend(alerts)
        except Exception as e:
            log.error(f"{name} check failed: {e}\n{traceback.format_exc()}")

    save_state(state)
    return all_alerts


def print_status() -> None:
    """Print current signal state without running checks."""
    state = load_state()

    print("\n=== Unified Signal Watcher — Current State ===\n")
    print(f"Last run: {state.get('last_run', 'never')}\n")

    # VIX Panic
    vp = state.get("vix_panic", {})
    print(f"VIX Panic Reversal:")
    print(f"  Signal active: {vp.get('signal_active', False)}")
    print(f"  Last peak VIXY: {vp.get('last_peak_vix', 'N/A')}")
    print()

    # Regime
    rg = state.get("regime", {})
    print(f"Regime:")
    print(f"  In UPRO: {rg.get('in_upro', False)}")
    print(f"  Confluence score: {rg.get('confluence_score', 0):.1f}/3.0")
    print(f"  Vol tier: {rg.get('vol_tier', 'SPY')}")
    print()

    # CTA
    cta = state.get("cta", {})
    up = cta.get("uptrend_tickers", [])
    all_cta = set(CTA_ETFS)
    down = sorted(all_cta - set(up))
    print(f"CTA Trends:")
    print(f"  Above SMA50 ({len(up)}): {', '.join(sorted(up)) or 'none'}")
    print(f"  Below SMA50 ({len(down)}): {', '.join(down) or 'none'}")
    print()

    # Protection
    prot = state.get("protection", {})
    labels = {
        "vix_ok": "VIX calm",
        "spy_above_50sma": "SPY > 50d SMA",
        "credit_healthy": "Credit healthy",
        "breadth_ok": "Breadth OK",
    }
    green = sum(prot.get(k, True) for k in labels)
    print(f"Protection Overlay: {green}/4 green")
    for key, label in labels.items():
        status = "OK" if prot.get(key, True) else "WARNING"
        print(f"  {label}: {status}")
    print()

    # Last alerts
    la = state.get("last_alerts", {})
    if la:
        print("Last alerts:")
        for atype, atime in sorted(la.items()):
            print(f"  {atype}: {atime}")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    import time as _time

    parser = argparse.ArgumentParser(description="Unified Signal Watcher")
    parser.add_argument("--status", action="store_true", help="Print current state")
    parser.add_argument("--force", action="store_true", help="Ignore cooldowns")
    parser.add_argument("--once", action="store_true", help="Run once and exit (no loop)")
    parser.add_argument("--interval", type=int, default=300, help="Seconds between checks (default 300 = 5min)")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    while True:
        log.info("=" * 60)
        log.info("Unified Signal Watcher — starting check")
        log.info("=" * 60)

        try:
            alerts = run_all_checks(force=args.force)

            if alerts:
                log.info(f"Generated {len(alerts)} alert(s)")
                for alert in alerts:
                    print()
                    print(alert)
                    print()
                # Queue for Discord delivery
                queue_discord_alerts(alerts)
            else:
                log.info("No alerts — all signals stable")
        except Exception as e:
            log.error(f"Check failed: {e}")

        log.info("Check complete")

        if args.once:
            break

        log.info(f"Sleeping {args.interval}s until next check...")
        _time.sleep(args.interval)


if __name__ == "__main__":
    main()
