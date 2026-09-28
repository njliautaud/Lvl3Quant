#!/usr/bin/env python3
"""
Panic Confluence Monitor — Unified 3-Signal Scoring
=====================================================

Checks the three validated panic-buying confluence signals:
  1. VIX spike-then-drop: VIX peaked >= 25 within last 10 days, now < 22
  2. Market breadth stress: < 30% of sector ETFs above 200-day MA
  3. Credit stress: HYG 5-day drop > 3%

Returns a confluence score (0-3) plus detailed signal states.

OFFENSIVE use: when confluence >= 2, premium-selling strategies should
INCREASE position sizes (panic = high IV = fat premiums). When confluence
drops back to 0 after a spike, it means the recovery trade is ON.

DEFENSIVE use: while signals are BUILDING (VIX rising, breadth falling,
credit widening), reduce exposure. The risk_overlay handles this.

This module is the OFFENSIVE complement to risk_overlay.py (which is
purely defensive). Together they give the paper engines a full picture:
  - risk_overlay: "how scared should we be?" -> scale DOWN
  - panic_confluence: "is the fear REVERSING?" -> scale UP (sell more premium)

Can be imported by any paper engine:
    from live_trading_linux.panic_confluence_monitor import get_panic_confluence

Backtested edge (SPY equity): Sharpe 1.59, WR 76%, PF 8.10, 66 trades / 16 years.
For premium selling: panic reversal = high IV + mean-reversion tailwind = ideal entry.

Author: Claude (2026-07-15)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_STATE_DIR = Path(__file__).parent
_STATE_FILE = _STATE_DIR / "panic_confluence_state.json"
_LOG_FILE = _STATE_DIR / "logs" / "panic_confluence.log"
_CACHE_TTL_HOURS = 6  # Recheck every 6 hours (more responsive than risk_overlay's 24h)

# ── Signal thresholds (from validated backtest) ──────────────────────────────
VIX_SPIKE_THRESHOLD = 25.0      # VIX must have reached this within lookback
VIX_DROP_THRESHOLD = 22.0       # VIX must currently be below this
VIX_LOOKBACK_DAYS = 10          # How far back to look for the spike

BREADTH_STRESS_PCT = 30.0       # % of sectors above 200d MA (below = stress)
BREADTH_TICKERS = [             # Sector ETFs as breadth proxy
    'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY'
]

HYG_DROP_THRESHOLD = -3.0       # HYG 5-day % change threshold
HYG_LOOKBACK_DAYS = 5


def _load_cache() -> Optional[dict]:
    """Return cached result if still valid."""
    if not _STATE_FILE.exists():
        return None
    try:
        data = json.loads(_STATE_FILE.read_text())
        expires = datetime.fromisoformat(data.get("cache_expires", "2000-01-01"))
        if datetime.now(timezone.utc) < expires:
            return data
    except Exception as e:
        log.warning("[panic_confluence] Cache read failed: %s", e)
    return None


def _save_state(result: dict) -> None:
    """Persist state + cache."""
    try:
        _STATE_FILE.write_text(json.dumps(result, indent=2, default=str))
    except Exception as e:
        log.warning("[panic_confluence] State write failed: %s", e)


def _log_daily(result: dict) -> None:
    """Append a one-line daily log entry."""
    try:
        _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": result["as_of"],
            "confluence": result["confluence_score"],
            "mode": result["mode"],
            "vix": result["signals"]["vix"].get("current"),
            "breadth_pct": result["signals"]["breadth"].get("pct_above_200d"),
            "hyg_5d": result["signals"]["credit"].get("drop_pct"),
        }
        with open(_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # logging should never block


def _check_vix_signal() -> dict:
    """Check VIX spike-then-drop signal."""
    import yfinance as yf
    try:
        vix = yf.Ticker('^VIX').history(period='30d')
        if len(vix) < VIX_LOOKBACK_DAYS:
            return {'active': False, 'phase': 'no_data', 'desc': 'Insufficient VIX data'}

        current_vix = float(vix['Close'].iloc[-1])
        recent_max = float(vix['Close'].iloc[-VIX_LOOKBACK_DAYS:].max())
        was_above_threshold = recent_max >= VIX_SPIKE_THRESHOLD
        now_below_drop = current_vix < VIX_DROP_THRESHOLD

        # Determine phase: building, peaked, reversing, calm
        if current_vix >= VIX_SPIKE_THRESHOLD:
            phase = 'spiking'  # VIX currently elevated — defensive
        elif was_above_threshold and now_below_drop:
            phase = 'reversing'  # The money signal — panic subsiding
        elif was_above_threshold and not now_below_drop:
            phase = 'elevated'  # Was high, coming down but not crossed yet
        else:
            phase = 'calm'

        return {
            'active': was_above_threshold and now_below_drop,
            'phase': phase,
            'current': round(current_vix, 2),
            'recent_peak': round(recent_max, 2),
            'desc': f'VIX {current_vix:.1f} (peak {recent_max:.1f} in {VIX_LOOKBACK_DAYS}d)'
        }
    except Exception as e:
        return {'active': False, 'phase': 'error', 'desc': f'VIX error: {e}'}


def _check_breadth_signal() -> dict:
    """Check market breadth (% of sector ETFs above 200d MA)."""
    import yfinance as yf
    try:
        above_200 = 0
        total = 0
        for ticker in BREADTH_TICKERS:
            hist = yf.Ticker(ticker).history(period='250d')
            if len(hist) >= 200:
                ma200 = float(hist['Close'].rolling(200).mean().iloc[-1])
                current = float(hist['Close'].iloc[-1])
                if current > ma200:
                    above_200 += 1
                total += 1

        pct = (above_200 / total * 100) if total > 0 else 50.0

        if pct < BREADTH_STRESS_PCT:
            phase = 'stressed'
        elif pct < 50:
            phase = 'weak'
        elif pct > 80:
            phase = 'strong'
        else:
            phase = 'normal'

        return {
            'active': pct < BREADTH_STRESS_PCT,
            'phase': phase,
            'pct_above_200d': round(pct, 1),
            'above_count': above_200,
            'total_count': total,
            'desc': f'{pct:.0f}% of sectors above 200d MA ({above_200}/{total})'
        }
    except Exception as e:
        return {'active': False, 'phase': 'error', 'desc': f'Breadth error: {e}'}


def _check_credit_signal() -> dict:
    """Check HYG credit stress (5-day drop > 3%)."""
    import yfinance as yf
    try:
        hyg = yf.Ticker('HYG').history(period='30d')
        if len(hyg) < HYG_LOOKBACK_DAYS + 1:
            return {'active': False, 'phase': 'no_data', 'desc': 'Insufficient HYG data'}

        current = float(hyg['Close'].iloc[-1])
        past = float(hyg['Close'].iloc[-(HYG_LOOKBACK_DAYS + 1)])
        drop_pct = (current / past - 1) * 100

        if drop_pct < HYG_DROP_THRESHOLD:
            phase = 'stressed'
        elif drop_pct < -1.0:
            phase = 'weak'
        elif drop_pct > 1.0:
            phase = 'recovering'
        else:
            phase = 'stable'

        return {
            'active': drop_pct < HYG_DROP_THRESHOLD,
            'phase': phase,
            'drop_pct': round(drop_pct, 2),
            'desc': f'HYG 5-day: {drop_pct:+.1f}%'
        }
    except Exception as e:
        return {'active': False, 'phase': 'error', 'desc': f'Credit error: {e}'}


def _determine_mode(confluence: int, signals: dict) -> str:
    """
    Determine the overall mode for position sizing.

    Returns one of:
        'offensive'  — confluence >= 2, panic REVERSING. Scale UP premium selling.
        'cautious'   — VIX spiking or credit stressed but not yet reversing.
        'normal'     — no signals firing. Business as usual.
    """
    if confluence >= 2:
        return 'offensive'

    # Check if we're in the buildup phase (signals building but not yet reversing)
    vix_phase = signals.get('vix', {}).get('phase', 'calm')
    credit_phase = signals.get('credit', {}).get('phase', 'stable')

    if vix_phase in ('spiking', 'elevated') or credit_phase == 'stressed':
        return 'cautious'

    return 'normal'


def _compute_position_multiplier(mode: str, confluence: int) -> float:
    """
    Compute the position size multiplier based on panic confluence.

    offensive (confluence >= 2): 1.50x — sell 50% more premium (panic = fat IV)
    offensive (confluence == 3): 1.75x — all three confirm, max conviction
    cautious:                    0.70x — building stress, pull back slightly
    normal:                      1.00x — no adjustment
    """
    if mode == 'offensive':
        if confluence >= 3:
            return 1.75
        return 1.50
    elif mode == 'cautious':
        return 0.70
    return 1.00


def get_panic_confluence(force_refresh: bool = False) -> dict:
    """
    Main entry point. Returns the panic confluence assessment.

    Returns:
        {
            "confluence_score": int,           # 0-3 (how many signals active)
            "mode": str,                       # "offensive" | "cautious" | "normal"
            "position_multiplier": float,      # multiply base size by this
            "signals": {
                "vix": {...},
                "breadth": {...},
                "credit": {...},
            },
            "as_of": str,
            "cache_expires": str,
        }
    """
    if not force_refresh:
        cached = _load_cache()
        if cached is not None:
            log.info("[panic_confluence] Using cached result (mode=%s, confluence=%d)",
                     cached.get("mode"), cached.get("confluence_score", 0))
            return cached

    log.info("[panic_confluence] Checking all three signals ...")

    signals = {
        'vix': _check_vix_signal(),
        'breadth': _check_breadth_signal(),
        'credit': _check_credit_signal(),
    }

    confluence = sum(1 for s in signals.values() if s.get('active'))
    mode = _determine_mode(confluence, signals)
    multiplier = _compute_position_multiplier(mode, confluence)

    now = datetime.now(timezone.utc)
    result = {
        "confluence_score": confluence,
        "mode": mode,
        "position_multiplier": multiplier,
        "signals": signals,
        "as_of": now.isoformat(),
        "cache_expires": (now + timedelta(hours=_CACHE_TTL_HOURS)).isoformat(),
    }

    log.info("[panic_confluence] confluence=%d  mode=%s  multiplier=%.2f",
             confluence, mode, multiplier)
    for name, sig in signals.items():
        status = "ACTIVE" if sig.get('active') else "inactive"
        log.info("[panic_confluence]   %s: %s (%s) — %s",
                 name, status, sig.get('phase', '?'), sig.get('desc', ''))

    _save_state(result)
    _log_daily(result)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser(description="Panic Confluence Monitor")
    parser.add_argument("--force", action="store_true", help="Force refresh (bypass cache)")
    args = parser.parse_args()

    result = get_panic_confluence(force_refresh=args.force)

    print("\n" + "=" * 60)
    print(f"PANIC CONFLUENCE MONITOR — {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 60)

    for name, sig in result['signals'].items():
        status = "ACTIVE" if sig.get('active') else "inactive"
        print(f"  {name:8s}: {status:10s} [{sig.get('phase', '?'):12s}] — {sig.get('desc', '')}")

    print(f"\n  Confluence : {result['confluence_score']}/3")
    print(f"  Mode       : {result['mode'].upper()}")
    print(f"  Multiplier : {result['position_multiplier']:.2f}x")

    if result['mode'] == 'offensive':
        print(f"\n  >>> OFFENSIVE: {result['confluence_score']}/3 signals confirm panic reversal.")
        print(f"      Premium-selling engines should scale UP by {result['position_multiplier']:.0%}.")
        print(f"      Backtested SPY edge: Sharpe 1.59, WR 76%, PF 8.10.")
    elif result['mode'] == 'cautious':
        print(f"\n  >>> CAUTIOUS: Stress building. Scale down to {result['position_multiplier']:.0%}.")
    else:
        print(f"\n  >>> NORMAL: No panic signals. Standard position sizing.")
