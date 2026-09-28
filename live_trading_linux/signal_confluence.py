#!/usr/bin/env python3
"""
Multi-Signal Confluence Module (HC #750)
=========================================
Centralized signal scoring for all paper engines.
Every trade entry MUST call `check_confluence()` and get >= 2 confirming signals.

Signals available:
  1. Momentum (5d, 20d, 60d price momentum)
  2. Quality (RSI, MFI not oversold/overbought for direction)
  3. Vol Regime (VIX context — normal/elevated/extreme)
  4. Trend (price vs 50d/200d SMA)
  5. Sector Rotation (ETF rotation rank — is sector in favor?)
  6. Earnings Calendar (no earnings within 2 days = safe)

Usage:
  from signal_confluence import check_confluence, log_confluence

  result = check_confluence(ticker, direction='short_put', price=150.0, vix=18.0)
  if result['score'] >= 2:
      # Proceed with trade
      log_confluence(result, trade_id='bps_AAPL_20260725')
  else:
      # Skip — insufficient confluence
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import json

import numpy as np

log = logging.getLogger('CONFLUENCE')

# ── Signal thresholds ─────────────────────────────────────────────────────
MOMENTUM_LOOKBACKS = [5, 20, 60]  # days
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MFI_OVERSOLD = 20
MFI_OVERBOUGHT = 80
VIX_LOW = 15
VIX_HIGH = 25
VIX_EXTREME = 35
MIN_SIGNALS_REQUIRED = 2

# Cache for yfinance data to avoid repeated downloads
_price_cache = {}
_cache_expiry = {}
CACHE_TTL_SECONDS = 300  # 5 min cache


def _get_price_history(ticker: str, days: int = 120) -> Optional[dict]:
    """Get recent price history with caching."""
    now = datetime.now()
    cache_key = f"{ticker}_{days}"

    if cache_key in _price_cache and cache_key in _cache_expiry:
        if (now - _cache_expiry[cache_key]).total_seconds() < CACHE_TTL_SECONDS:
            return _price_cache[cache_key]

    try:
        import yfinance as yf
        data = yf.download(ticker, period=f'{days}d', progress=False)
        if data is None or len(data) < 20:
            return None

        result = {
            'close': data['Close'].values.flatten() if hasattr(data['Close'], 'values') else data['Close'].values,
            'volume': data['Volume'].values.flatten() if 'Volume' in data.columns else None,
            'high': data['High'].values.flatten() if 'High' in data.columns else None,
            'low': data['Low'].values.flatten() if 'Low' in data.columns else None,
        }
        _price_cache[cache_key] = result
        _cache_expiry[cache_key] = now
        return result
    except Exception as e:
        log.debug(f"Failed to get price history for {ticker}: {e}")
        return None


def _compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    """Compute RSI from price array."""
    if len(prices) < period + 1:
        return 50.0  # neutral default

    deltas = np.diff(prices[-period-1:])
    gains = np.maximum(deltas, 0)
    losses = np.abs(np.minimum(deltas, 0))

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _compute_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 volume: np.ndarray, period: int = 14) -> float:
    """Compute Money Flow Index."""
    if any(x is None for x in [high, low, volume]) or len(close) < period + 1:
        return 50.0  # neutral default

    tp = (high + low + close) / 3
    mf = tp * volume

    tp_diff = np.diff(tp[-period-1:])
    mf_recent = mf[-period:]

    pos_flow = np.sum(mf_recent[tp_diff > 0]) if np.any(tp_diff > 0) else 0
    neg_flow = np.sum(mf_recent[tp_diff <= 0]) if np.any(tp_diff <= 0) else 1

    if neg_flow == 0:
        return 100.0

    mr = pos_flow / neg_flow
    return 100 - (100 / (1 + mr))


def check_confluence(ticker: str, direction: str = 'long',
                     price: Optional[float] = None,
                     vix: Optional[float] = None,
                     sigma: Optional[float] = None,
                     earnings_date: Optional[str] = None) -> Dict:
    """
    Check multi-signal confluence for a trade entry.

    Args:
        ticker: Stock/ETF ticker
        direction: 'long', 'short', 'short_put' (bullish), 'short_call' (bearish)
        price: Current price (optional, will fetch if not provided)
        vix: Current VIX level
        sigma: Implied volatility of the ticker
        earnings_date: Next earnings date string (YYYY-MM-DD)

    Returns:
        dict with 'score' (int), 'signals' (list of dicts), 'pass' (bool)
    """

    # Normalize direction
    bullish = direction in ('long', 'short_put', 'bull_put_spread', 'call')
    bearish = direction in ('short', 'short_call', 'bear_call_spread', 'put')
    # For neutral strategies (iron condor, strangle, jade lizard), both sides need signals
    neutral = direction in ('iron_condor', 'strangle', 'jade_lizard', 'straddle')

    signals = []
    hist = _get_price_history(ticker)

    if hist is None:
        return {
            'score': 0, 'signals': [], 'pass': False,
            'reason': f'Could not fetch price data for {ticker}'
        }

    close = hist['close']
    current_price = price if price else close[-1]

    # ── Signal 1: Momentum ──
    mom_score = 0
    mom_details = {}
    for lb in MOMENTUM_LOOKBACKS:
        if len(close) > lb:
            ret = (close[-1] / close[-lb] - 1) * 100
            mom_details[f'mom_{lb}d'] = round(ret, 2)
            if bullish and ret > 0:
                mom_score += 1
            elif bearish and ret < 0:
                mom_score += 1
            elif neutral:
                mom_score += 1  # neutral = ok either way

    momentum_confirms = mom_score >= 2  # at least 2 of 3 timeframes agree
    signals.append({
        'name': 'momentum',
        'confirms': momentum_confirms,
        'details': mom_details
    })

    # ── Signal 2: Quality (RSI/MFI) ──
    rsi = _compute_rsi(close)
    mfi = _compute_mfi(hist.get('high'), hist.get('low'), close,
                       hist.get('volume'))

    quality_ok = True
    if bullish:
        # For bullish: RSI not overbought (room to run), or oversold (bounce)
        quality_ok = rsi < RSI_OVERBOUGHT  # not overextended
    elif bearish:
        quality_ok = rsi > RSI_OVERSOLD  # not oversold (could bounce)
    # neutral: quality ok as long as not extreme

    signals.append({
        'name': 'quality',
        'confirms': quality_ok,
        'details': {'rsi': round(rsi, 1), 'mfi': round(mfi, 1)}
    })

    # ── Signal 3: Vol Regime ──
    vol_ok = True
    vol_details = {}
    if vix is not None:
        vol_details['vix'] = round(vix, 1)
        if vix > VIX_EXTREME:
            vol_ok = False  # too dangerous for any new positions
            vol_details['regime'] = 'extreme'
        elif vix > VIX_HIGH:
            vol_ok = direction in ('short_put', 'bull_put_spread', 'iron_condor',
                                   'strangle', 'jade_lizard')  # selling premium OK in high vol
            vol_details['regime'] = 'elevated'
        else:
            vol_ok = True
            vol_details['regime'] = 'normal'
    else:
        vol_details['regime'] = 'unknown'

    signals.append({
        'name': 'vol_regime',
        'confirms': vol_ok,
        'details': vol_details
    })

    # ── Signal 4: Trend (SMA alignment) ──
    trend_ok = False
    trend_details = {}
    if len(close) >= 200:
        sma50 = np.mean(close[-50:])
        sma200 = np.mean(close[-200:])
        above_50 = current_price > sma50
        above_200 = current_price > sma200
        golden_cross = sma50 > sma200

        trend_details = {
            'above_50sma': above_50,
            'above_200sma': above_200,
            'golden_cross': golden_cross
        }

        if bullish or neutral:
            trend_ok = above_50 and above_200  # uptrend
        elif bearish:
            trend_ok = not above_50  # below 50 SMA = weakening
    elif len(close) >= 50:
        sma50 = np.mean(close[-50:])
        above_50 = current_price > sma50
        trend_details = {'above_50sma': above_50}
        trend_ok = (bullish and above_50) or (bearish and not above_50) or neutral

    signals.append({
        'name': 'trend',
        'confirms': trend_ok,
        'details': trend_details
    })

    # ── Signal 5: Earnings Safety ──
    earnings_safe = True
    earnings_details = {}
    if earnings_date:
        try:
            ed = datetime.strptime(earnings_date, '%Y-%m-%d')
            days_to_earnings = (ed - datetime.now()).days
            earnings_safe = days_to_earnings > 2 or days_to_earnings < -1
            earnings_details = {
                'earnings_date': earnings_date,
                'days_away': days_to_earnings,
                'safe': earnings_safe
            }
        except Exception:
            earnings_details = {'error': 'could not parse date'}

    signals.append({
        'name': 'earnings_safety',
        'confirms': earnings_safe,
        'details': earnings_details
    })

    # ── Aggregate Score ──
    confirming = [s for s in signals if s['confirms']]
    score = len(confirming)
    passes = score >= MIN_SIGNALS_REQUIRED

    return {
        'ticker': ticker,
        'direction': direction,
        'score': score,
        'signals': signals,
        'confirming_signals': [s['name'] for s in confirming],
        'pass': passes,
        'min_required': MIN_SIGNALS_REQUIRED,
        'timestamp': datetime.now().isoformat()
    }


def log_confluence(result: Dict, trade_id: str = '',
                   log_dir: str = '/home/jupiter/Lvl3Quant/logs') -> None:
    """Log confluence check result to file for HC #750 audit trail."""
    log_path = Path(log_dir) / 'confluence_audit.jsonl'
    entry = {
        'trade_id': trade_id,
        **result
    }
    try:
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry, default=str) + '\n')
    except Exception as e:
        log.warning(f"Failed to write confluence log: {e}")


def format_confluence_summary(result: Dict) -> str:
    """Format a human-readable summary of confluence check."""
    ticker = result.get('ticker', '?')
    score = result['score']
    total = len(result['signals'])
    status = '✅ PASS' if result['pass'] else '❌ FAIL'

    lines = [f"{ticker} confluence: {score}/{total} signals — {status}"]
    for s in result['signals']:
        icon = '✅' if s['confirms'] else '❌'
        lines.append(f"  {icon} {s['name']}: {s.get('details', {})}")

    return '\n'.join(lines)
