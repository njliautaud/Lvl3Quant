#!/usr/bin/env python3
"""
Expanded Signal Alerter — Persistent Monitoring with Discord Alerts
===================================================================
Monitors ALL asymmetric signals (original 7 + expanded set) and fires
alerts when signals cross critical thresholds or regime changes occur.

Designed to run via PM2 — checks every 5 minutes during market hours,
every 30 minutes outside. Sends Discord alerts on state changes only
(no spam — cooldown + dedup built in).

Signal Categories:
  A. Fear/Greed (original 7 from scorecard)
  B. Cross-Asset Lead/Lag (credit, bonds, gold)
  C. Rotation Signals (sector, factor, quality)
  D. Stock-Level Distress/Recovery
  E. Volatility Regime
  F. Breadth & Momentum

Author: Claude (Head of Quant)
Date: 2026-07-22
"""
from __future__ import annotations

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

ROOT = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "expanded_alerter_state.json"
ALERT_HISTORY = STATE_DIR / "expanded_alert_history.jsonl"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ExpandedAlerter] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "expanded_alerter.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("expanded_alerter")

# ── Alert config ──────────────────────────────────────────────────────────
COOLDOWN_SECONDS = 4 * 3600  # Don't re-alert same signal within 4 hours
MARKET_CHECK_INTERVAL = 300   # 5 minutes during market hours
OFFHOURS_CHECK_INTERVAL = 1800  # 30 minutes outside market hours

# ── Discord webhook ──────────────────────────────────────────────────────
WEBHOOK_FILE = ROOT.parent / "teleclaude-main" / "API_KEYS.md"


def get_discord_webhook() -> str | None:
    """Extract Discord webhook URL from API_KEYS.md."""
    try:
        if WEBHOOK_FILE.exists():
            text = WEBHOOK_FILE.read_text()
            for line in text.split('\n'):
                if 'discord' in line.lower() and 'webhook' in line.lower() and 'http' in line:
                    # Extract URL
                    import re
                    urls = re.findall(r'https://discord\.com/api/webhooks/\S+', line)
                    if urls:
                        return urls[0].strip('`').strip()
    except Exception:
        pass
    return None


def send_discord_alert(message: str, webhook_url: str | None = None):
    """Send alert to Discord via webhook."""
    if not webhook_url:
        webhook_url = get_discord_webhook()
    if not webhook_url:
        log.warning("No Discord webhook configured — alert not sent")
        return

    try:
        import urllib.request
        data = json.dumps({"content": message}).encode()
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=10)
        log.info(f"Discord alert sent: {message[:80]}...")
    except Exception as e:
        log.error(f"Failed to send Discord alert: {e}")


# ── State management ─────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Corrupt state file, starting fresh")
    return {
        "last_check": None,
        "active_signals": {},
        "alert_cooldowns": {},
        "regime": "UNKNOWN",
        "last_readings": {},
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def log_alert(alert: dict):
    """Append alert to history file."""
    with open(ALERT_HISTORY, "a") as f:
        f.write(json.dumps(alert, default=str) + "\n")


# ── Data fetching ────────────────────────────────────────────────────────

def fetch_data() -> dict:
    """Fetch all needed market data. Returns dict of ticker -> price series."""
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed")
        return {}

    tickers = {
        'indices': ['SPY', 'QQQ', 'IWM', 'DIA'],
        'bonds': ['TLT', 'IEF', 'SHY', 'HYG', 'LQD', 'JNK'],
        'commodities': ['GLD', 'SLV', 'USO'],
        'intl': ['EEM', 'EFA'],
        'sectors': ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB'],
        'vix': ['^VIX', '^VIX3M'],
        'factors': ['MTUM', 'VLUE', 'USMV'],
    }
    all_tickers = []
    for group in tickers.values():
        all_tickers.extend(group)

    import warnings
    warnings.filterwarnings('ignore')

    log.info(f"Fetching {len(all_tickers)} tickers...")
    try:
        raw = yf.download(all_tickers, period='1y', progress=False, auto_adjust=True, group_by='ticker')
    except Exception as e:
        log.error(f"Download failed: {e}")
        return {}

    data = {}
    for t in all_tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t in raw.columns.get_level_values(0):
                    series = raw[(t, 'Close')].dropna()
                    name = t.replace('^', '')
                    data[name] = series
            else:
                data[t] = raw['Close'].dropna()
        except Exception:
            pass

    return data

import pandas as pd


# ── Signal computation ───────────────────────────────────────────────────

def compute_all_signals(data: dict) -> dict:
    """Compute all signal readings from price data. Returns dict of signal_name -> reading dict."""
    readings = {}

    prices = pd.DataFrame(data)
    returns = prices.pct_change()

    # A1. VIX Term Structure
    if 'VIX' in prices and 'VIX3M' in prices:
        ratio = prices['VIX'].iloc[-1] / prices['VIX3M'].iloc[-1]
        readings['vix_term_structure'] = {
            'value': round(ratio, 4),
            'threshold_type': 'above',
            'alert_threshold': 1.05,  # backwardation
            'warning_threshold': 0.95,
            'current_state': 'BACKWARDATION' if ratio > 1.05 else 'CONTANGO' if ratio < 0.95 else 'FLAT',
            'description': f"VIX/VIX3M = {ratio:.3f}",
        }

    # A2. VIX Level
    if 'VIX' in prices:
        vix = prices['VIX'].iloc[-1]
        vix_20d = prices['VIX'].iloc[-20:].mean()
        vix_pctile = (prices['VIX'] < vix).mean() * 100
        readings['vix_level'] = {
            'value': round(vix, 2),
            'threshold_type': 'above',
            'alert_threshold': 25,
            'warning_threshold': 20,
            'current_state': 'PANIC' if vix > 30 else 'ELEVATED' if vix > 25 else 'NORMAL' if vix > 15 else 'COMPLACENT',
            'description': f"VIX = {vix:.1f} (20d avg: {vix_20d:.1f}, pctile: {vix_pctile:.0f}%)",
        }

        # A3. VIX Crush (falling from elevated)
        if len(prices['VIX']) >= 10:
            vix_5d_ago = prices['VIX'].iloc[-6]
            crush = (vix_5d_ago - vix) / vix_5d_ago * 100
            readings['vix_crush'] = {
                'value': round(crush, 2),
                'threshold_type': 'above',
                'alert_threshold': 15,  # VIX dropped 15%+ in 5 days
                'warning_threshold': 10,
                'current_state': 'CRUSHING' if crush > 15 else 'DECLINING' if crush > 5 else 'STABLE',
                'description': f"VIX 5d change: {crush:+.1f}% ({vix_5d_ago:.1f} → {vix:.1f})",
            }

    # B1. Credit Spread (HYG vs LQD)
    if 'HYG' in returns and 'LQD' in returns:
        credit_21d = (returns['HYG'] - returns['LQD']).rolling(21).sum().iloc[-1] * 100
        credit_z = (credit_21d - (returns['HYG'] - returns['LQD']).rolling(21).sum().mean() * 100) / \
                   ((returns['HYG'] - returns['LQD']).rolling(21).sum().std() * 100) if \
                   (returns['HYG'] - returns['LQD']).rolling(21).sum().std() > 0 else 0
        readings['credit_spread'] = {
            'value': round(credit_21d, 3),
            'z_score': round(credit_z, 2),
            'threshold_type': 'below',
            'alert_threshold': -2.0,  # z-score
            'warning_threshold': -1.5,
            'current_state': 'STRESS' if credit_z < -2 else 'WIDENING' if credit_z < -1 else 'NORMAL',
            'description': f"HYG-LQD 21d: {credit_21d:+.3f}% (z={credit_z:+.2f})",
        }

    # B2. Bond-Equity Correlation
    if 'SPY' in returns and 'TLT' in returns:
        corr_63 = returns['SPY'].rolling(63).corr(returns['TLT']).iloc[-1]
        corr_series = returns['SPY'].rolling(63).corr(returns['TLT']).dropna()
        corr_pctile = (corr_series < corr_63).mean() * 100
        readings['bond_equity_corr'] = {
            'value': round(corr_63, 4),
            'percentile': round(corr_pctile, 1),
            'threshold_type': 'above',
            'alert_threshold': 0.3,  # Strong positive = stocks & bonds moving together (unusual)
            'warning_threshold': 0.15,
            'current_state': 'POSITIVE_CORR' if corr_63 > 0.3 else 'MILD_POSITIVE' if corr_63 > 0 else 'NEGATIVE_CORR',
            'description': f"SPY-TLT 63d corr = {corr_63:.3f} (pctile: {corr_pctile:.0f}%)",
        }

    # B3. Gold Stress Signal
    if 'GLD' in prices and 'SPY' in prices:
        gld_rel_21 = (prices['GLD'].pct_change(21).iloc[-1] - prices['SPY'].pct_change(21).iloc[-1]) * 100
        readings['gold_stress'] = {
            'value': round(gld_rel_21, 3),
            'threshold_type': 'above',
            'alert_threshold': 5.0,  # Gold massively outperforming stocks
            'warning_threshold': 3.0,
            'current_state': 'FLIGHT_TO_GOLD' if gld_rel_21 > 5 else 'GOLD_LEADING' if gld_rel_21 > 2 else 'NORMAL',
            'description': f"GLD-SPY 21d relative: {gld_rel_21:+.2f}%",
        }

    # C1. Sector Rotation Breadth
    sector_list = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
    avail_sectors = [s for s in sector_list if s in prices]
    if len(avail_sectors) >= 5 and 'SPY' in prices:
        improving = 0
        for sec in avail_sectors:
            rs_now = prices[sec].pct_change(21).iloc[-1] - prices['SPY'].pct_change(21).iloc[-1]
            rs_prev = prices[sec].pct_change(21).iloc[-22] - prices['SPY'].pct_change(21).iloc[-22] if len(prices) > 22 else 0
            if rs_now > rs_prev:
                improving += 1
        rot_breadth = improving / len(avail_sectors)
        readings['rotation_breadth'] = {
            'value': round(rot_breadth, 3),
            'threshold_type': 'threshold_zone',
            'alert_low': 0.2,  # Very few sectors improving
            'alert_high': 0.8,  # Most sectors improving
            'current_state': 'NARROW_LEADERSHIP' if rot_breadth < 0.3 else 'BROAD_PARTICIPATION' if rot_breadth > 0.7 else 'NORMAL',
            'description': f"Sector rotation breadth: {improving}/{len(avail_sectors)} improving RS",
        }

    # C2. Risk Appetite (XLY/XLP)
    if 'XLY' in prices and 'XLP' in prices:
        risk_ratio = np.log(prices['XLY'].iloc[-1] / prices['XLP'].iloc[-1])
        risk_63d_avg = np.log(prices['XLY'] / prices['XLP']).rolling(63).mean().iloc[-1]
        risk_dev = risk_ratio - risk_63d_avg
        readings['risk_appetite'] = {
            'value': round(risk_dev, 4),
            'threshold_type': 'threshold_zone',
            'alert_low': -0.05,
            'alert_high': 0.05,
            'current_state': 'RISK_OFF' if risk_dev < -0.05 else 'RISK_ON' if risk_dev > 0.05 else 'NEUTRAL',
            'description': f"XLY/XLP deviation from 63d mean: {risk_dev:+.4f}",
        }

    # D1. Market Breadth (% above 50 SMA)
    if len(avail_sectors) >= 5:
        above_50 = sum(1 for s in avail_sectors if prices[s].iloc[-1] > prices[s].iloc[-50:].mean())
        breadth_pct = above_50 / len(avail_sectors)
        readings['market_breadth'] = {
            'value': round(breadth_pct, 3),
            'threshold_type': 'below',
            'alert_threshold': 0.3,  # Less than 30% above 50 SMA
            'warning_threshold': 0.4,
            'current_state': 'COLLAPSE' if breadth_pct < 0.2 else 'WEAK' if breadth_pct < 0.4 else 'HEALTHY' if breadth_pct > 0.7 else 'MIXED',
            'description': f"Sector breadth: {above_50}/{len(avail_sectors)} above 50d SMA ({breadth_pct:.0%})",
        }

    # D2. Multi-Asset Momentum
    mom_assets = [a for a in ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EEM', 'HYG', 'XLE', 'XLK', 'XLF'] if a in prices]
    if len(mom_assets) >= 5:
        pos_1m = sum(1 for a in mom_assets if prices[a].pct_change(21).iloc[-1] > 0)
        mom_breadth = pos_1m / len(mom_assets)
        readings['momentum_breadth'] = {
            'value': round(mom_breadth, 3),
            'threshold_type': 'threshold_zone',
            'alert_low': 0.3,
            'alert_high': 0.8,
            'current_state': 'BEARISH_BREADTH' if mom_breadth < 0.3 else 'BULLISH_BREADTH' if mom_breadth > 0.8 else 'MIXED',
            'description': f"1m momentum breadth: {pos_1m}/{len(mom_assets)} positive ({mom_breadth:.0%})",
        }

    # E1. SPY Drawdown
    if 'SPY' in prices:
        spy_max = prices['SPY'].iloc[-252:].max()
        spy_dd = (prices['SPY'].iloc[-1] / spy_max - 1) * 100
        readings['spy_drawdown'] = {
            'value': round(spy_dd, 2),
            'threshold_type': 'below',
            'alert_threshold': -10,  # 10%+ drawdown = correction
            'warning_threshold': -5,
            'current_state': 'BEAR' if spy_dd < -20 else 'CORRECTION' if spy_dd < -10 else 'PULLBACK' if spy_dd < -5 else 'HEALTHY',
            'description': f"SPY drawdown from 52w high: {spy_dd:+.1f}%",
        }

    # E2. Implied vs Realized Vol
    if 'VIX' in prices and 'SPY' in returns:
        rv21 = returns['SPY'].iloc[-21:].std() * np.sqrt(252) * 100
        iv_rv = prices['VIX'].iloc[-1] / rv21 if rv21 > 0 else 0
        readings['iv_rv_spread'] = {
            'value': round(iv_rv, 3),
            'threshold_type': 'above',
            'alert_threshold': 1.8,  # Implied way above realized = fear premium
            'warning_threshold': 1.5,
            'current_state': 'HIGH_FEAR_PREMIUM' if iv_rv > 1.8 else 'ELEVATED' if iv_rv > 1.3 else 'NORMAL',
            'description': f"VIX/RV21 = {iv_rv:.2f} (VIX={prices['VIX'].iloc[-1]:.1f}, RV21={rv21:.1f}%)",
        }

    # F1. International Divergence
    if 'EEM' in prices and 'SPY' in prices:
        eem_rel = (prices['EEM'].pct_change(21).iloc[-1] - prices['SPY'].pct_change(21).iloc[-1]) * 100
        readings['em_relative'] = {
            'value': round(eem_rel, 2),
            'threshold_type': 'threshold_zone',
            'alert_low': -5,
            'alert_high': 5,
            'current_state': 'EM_UNDERPERFORM' if eem_rel < -5 else 'EM_OUTPERFORM' if eem_rel > 5 else 'NORMAL',
            'description': f"EEM-SPY 21d relative: {eem_rel:+.1f}%",
        }

    # F2. Small-Cap Spread
    if 'IWM' in prices and 'SPY' in prices:
        iwm_rel = (prices['IWM'].pct_change(63).iloc[-1] - prices['SPY'].pct_change(63).iloc[-1]) * 100
        readings['smallcap_spread'] = {
            'value': round(iwm_rel, 2),
            'threshold_type': 'threshold_zone',
            'alert_low': -10,
            'alert_high': 10,
            'current_state': 'SMALLS_LAGGING' if iwm_rel < -5 else 'SMALLS_LEADING' if iwm_rel > 5 else 'NORMAL',
            'description': f"IWM-SPY 63d relative: {iwm_rel:+.1f}%",
        }

    # F3. Yield Curve Momentum
    if 'TLT' in prices and 'SHY' in prices:
        yc = np.log(prices['TLT'].iloc[-1] / prices['SHY'].iloc[-1])
        yc_21d_ago = np.log(prices['TLT'].iloc[-22] / prices['SHY'].iloc[-22]) if len(prices) > 22 else yc
        yc_change = (yc - yc_21d_ago) * 100
        readings['yield_curve_move'] = {
            'value': round(yc_change, 3),
            'threshold_type': 'threshold_zone',
            'alert_low': -3,
            'alert_high': 3,
            'current_state': 'FLATTENING' if yc_change < -2 else 'STEEPENING' if yc_change > 2 else 'STABLE',
            'description': f"Yield curve 21d move: {yc_change:+.2f}%",
        }

    return readings


# ── Alert logic ──────────────────────────────────────────────────────────

def check_for_alerts(readings: dict, state: dict) -> list[dict]:
    """Compare new readings against thresholds and previous state. Return list of alerts."""
    alerts = []
    now = datetime.now(ET)
    cooldowns = state.get('alert_cooldowns', {})
    prev_readings = state.get('last_readings', {})

    for name, reading in readings.items():
        # Check cooldown
        if name in cooldowns:
            cooldown_until = datetime.fromisoformat(cooldowns[name])
            if now < cooldown_until.replace(tzinfo=ET) if cooldown_until.tzinfo is None else now.replace(tzinfo=None) < cooldown_until.replace(tzinfo=None):
                continue

        prev = prev_readings.get(name, {})
        prev_state = prev.get('current_state', 'UNKNOWN')
        new_state = reading.get('current_state', 'UNKNOWN')

        # Alert on STATE CHANGE only (not every tick)
        if prev_state != new_state and prev_state != 'UNKNOWN':
            severity = 'INFO'
            # Determine severity based on signal type
            if 'PANIC' in new_state or 'COLLAPSE' in new_state or 'BEAR' in new_state or 'STRESS' in new_state:
                severity = 'CRITICAL'
            elif 'ELEVATED' in new_state or 'WEAK' in new_state or 'FLIGHT' in new_state:
                severity = 'WARNING'
            elif 'COMPLACENT' in new_state or 'HEALTHY' in new_state or 'BULLISH' in new_state:
                severity = 'POSITIVE'

            alerts.append({
                'signal': name,
                'severity': severity,
                'prev_state': prev_state,
                'new_state': new_state,
                'description': reading['description'],
                'timestamp': now.isoformat(),
            })

            # Set cooldown
            cooldowns[name] = (now + timedelta(seconds=COOLDOWN_SECONDS)).isoformat()

    state['alert_cooldowns'] = cooldowns
    return alerts


def format_alert_message(alerts: list[dict]) -> str:
    """Format alerts into a Discord-friendly message."""
    if not alerts:
        return ""

    severity_emoji = {
        'CRITICAL': '🔴',
        'WARNING': '🟡',
        'POSITIVE': '🟢',
        'INFO': '🔵',
    }

    lines = ["**Signal Alert — Regime Change Detected**\n"]
    for alert in sorted(alerts, key=lambda a: {'CRITICAL': 0, 'WARNING': 1, 'POSITIVE': 2, 'INFO': 3}.get(a['severity'], 4)):
        emoji = severity_emoji.get(alert['severity'], '⚪')
        lines.append(f"{emoji} **{alert['signal']}**: {alert['prev_state']} → {alert['new_state']}")
        lines.append(f"   {alert['description']}")

    return '\n'.join(lines)


# ── Market hours check ───────────────────────────────────────────────────

def is_market_hours() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:  # Weekend
        return False
    market_open = now.replace(hour=9, minute=30, second=0)
    market_close = now.replace(hour=16, minute=0, second=0)
    return market_open <= now <= market_close


# ── Main loop ────────────────────────────────────────────────────────────

def run_once():
    """Run a single check cycle."""
    log.info("Starting check cycle...")
    state = load_state()

    try:
        data = fetch_data()
        if not data:
            log.error("No data fetched, skipping cycle")
            return

        readings = compute_all_signals(data)
        if not readings:
            log.error("No signals computed, skipping cycle")
            return

        log.info(f"Computed {len(readings)} signal readings")

        # Check for alerts
        alerts = check_for_alerts(readings, state)

        if alerts:
            log.info(f"Found {len(alerts)} alerts!")
            msg = format_alert_message(alerts)
            send_discord_alert(msg)
            for alert in alerts:
                log_alert(alert)
        else:
            log.info("No state changes detected")

        # Update state
        state['last_check'] = datetime.now(ET).isoformat()
        state['last_readings'] = readings
        state['active_signals'] = {name: r['current_state'] for name, r in readings.items()}

        # Determine overall regime
        critical_count = sum(1 for r in readings.values()
                            if r.get('current_state', '') in ['PANIC', 'COLLAPSE', 'BEAR', 'STRESS', 'BACKWARDATION'])
        warning_count = sum(1 for r in readings.values()
                           if r.get('current_state', '') in ['ELEVATED', 'WEAK', 'FLIGHT_TO_GOLD', 'RISK_OFF'])

        if critical_count >= 2:
            state['regime'] = 'FEAR'
        elif critical_count >= 1 or warning_count >= 3:
            state['regime'] = 'CAUTIOUS'
        elif warning_count >= 1:
            state['regime'] = 'MIXED'
        else:
            state['regime'] = 'CALM'

        save_state(state)
        log.info(f"Cycle complete. Regime: {state['regime']}. {len(alerts)} alerts fired.")

    except Exception as e:
        log.error(f"Error in check cycle: {e}\n{traceback.format_exc()}")


def main():
    """Main loop — runs continuously."""
    log.info("Expanded Signal Alerter starting...")

    # Parse args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()

    if args.once:
        run_once()
        return

    while True:
        try:
            run_once()
        except Exception as e:
            log.error(f"Main loop error: {e}\n{traceback.format_exc()}")

        interval = MARKET_CHECK_INTERVAL if is_market_hours() else OFFHOURS_CHECK_INTERVAL
        log.info(f"Sleeping {interval}s ({'market hours' if is_market_hours() else 'off-hours'})")
        time.sleep(interval)


if __name__ == "__main__":
    main()
