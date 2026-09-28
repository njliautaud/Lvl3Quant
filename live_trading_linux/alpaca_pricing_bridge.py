"""
Alpaca Pricing Bridge — Real Market Quotes with BS Fallback
============================================================

Wraps alpaca_options_pricing.py to provide a drop-in replacement for bs_price()
calls in paper trading engines. Uses real Alpaca quotes during market hours,
falls back to Black-Scholes when market is closed or API is unavailable.

All comparisons are logged to a JSONL file for tracking real vs BS price gaps.

Usage in paper engines:
    from live_trading_linux.alpaca_pricing_bridge import get_premium, init_bridge

    # At startup:
    init_bridge(engine_name='wheel-v5')

    # Replace: premium = bs_price(S, K, T, sigma, kind='put')
    # With:    premium = get_premium(ticker, S, K, T, sigma, expiry_date, kind='put')
"""

import json
import logging
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger(__name__)

# ── Paths ──
ROOT = Path("/home/jupiter/Lvl3Quant")
COMPARISON_LOG = ROOT / "logs" / "alpaca_vs_bs_pricing.jsonl"
COMPARISON_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── Module state ──
_engine_name = "unknown"
_alpaca_available = False
_alpaca_module = None
_stats = {'alpaca_hits': 0, 'bs_fallbacks': 0, 'errors': 0}


def init_bridge(engine_name: str = "unknown"):
    """
    Initialize the pricing bridge. Call once at engine startup.
    Tests Alpaca import availability but does NOT require market to be open.
    """
    global _engine_name, _alpaca_available, _alpaca_module
    _engine_name = engine_name

    try:
        try:
            from live_trading_linux import alpaca_options_pricing as aop
        except ImportError:
            # Fallback: direct import when running from live_trading_linux dir
            import alpaca_options_pricing as aop
        _alpaca_module = aop
        _alpaca_available = True
        logger.info(f"[PRICING-BRIDGE] Alpaca options pricing loaded for {engine_name}")
    except ImportError as e:
        _alpaca_available = False
        logger.warning(f"[PRICING-BRIDGE] Alpaca SDK not available, BS-only mode: {e}")
    except Exception as e:
        _alpaca_available = False
        logger.warning(f"[PRICING-BRIDGE] Alpaca init failed, BS-only mode: {e}")


def _log_comparison(
    ticker: str, strike: float, expiry: Optional[date], kind: str,
    bs_price_val: float, alpaca_mid: float, source: str,
    alpaca_bid: float = 0, alpaca_ask: float = 0,
    alpaca_iv: float = 0, alpaca_delta: float = 0,
    bs_sigma: float = 0,
):
    """Log real vs BS price comparison to JSONL for analysis."""
    try:
        if bs_price_val > 0 and alpaca_mid > 0:
            ratio = alpaca_mid / bs_price_val
            diff_pct = (alpaca_mid - bs_price_val) / bs_price_val * 100
        else:
            ratio = 0
            diff_pct = 0

        record = {
            'ts': datetime.utcnow().isoformat(),
            'engine': _engine_name,
            'ticker': ticker,
            'strike': strike,
            'expiry': expiry.isoformat() if expiry else None,
            'kind': kind,
            'bs_price': round(bs_price_val, 4),
            'alpaca_mid': round(alpaca_mid, 4),
            'alpaca_bid': round(alpaca_bid, 4),
            'alpaca_ask': round(alpaca_ask, 4),
            'alpaca_iv': round(alpaca_iv, 4),
            'alpaca_delta': round(alpaca_delta, 4),
            'bs_sigma': round(bs_sigma, 4),
            'ratio': round(ratio, 4),
            'diff_pct': round(diff_pct, 2),
            'source_used': source,
        }

        with open(COMPARISON_LOG, 'a') as f:
            f.write(json.dumps(record) + '\n')
    except Exception as e:
        logger.debug(f"[PRICING-BRIDGE] Log write failed: {e}")


def get_premium(
    ticker: str,
    S: float,
    K: float,
    T: float,
    sigma: float,
    expiry_date: Optional[date],
    kind: str = "put",
    bs_price_fn: Optional[Callable] = None,
) -> float:
    """
    Get option premium — tries Alpaca real quotes first, falls back to BS.

    Args:
        ticker: Underlying symbol (e.g., 'AAPL')
        S: Current underlying price
        K: Strike price
        T: Time to expiry in years
        sigma: Realized vol (used for BS fallback)
        expiry_date: Expiration date (needed for Alpaca lookup)
        kind: 'put' or 'call'
        bs_price_fn: The engine's local bs_price function for fallback

    Returns:
        Premium per share (same units as bs_price returns)
    """
    global _stats

    # HC #722 FIX: Compute intrinsic value floor.
    # No option can trade below intrinsic in real markets (arbitrage).
    # Alpaca/BS can return theoretical < intrinsic for deep ITM options.
    if kind == "call":
        intrinsic = max(S - K, 0)
    else:  # put
        intrinsic = max(K - S, 0)

    # Always compute BS as baseline (needed for fallback AND comparison logging)
    bs_val = 0.0
    if bs_price_fn is not None:
        try:
            bs_val = bs_price_fn(S, K, T, sigma, kind=kind)
        except Exception:
            bs_val = 0.0
    # Enforce intrinsic floor on BS
    bs_val = max(bs_val, intrinsic)

    # Try Alpaca if available and we have an expiry date
    if _alpaca_available and _alpaca_module and expiry_date is not None:
        try:
            quote = _alpaca_module.get_option_quote(ticker, K, expiry_date, kind)
            alpaca_mid = quote.get('mid', 0)

            if alpaca_mid > 0:
                # HC #722 FIX: Enforce intrinsic floor on Alpaca pricing
                if alpaca_mid < intrinsic:
                    logger.warning(
                        f"[PRICING-BRIDGE] {ticker} {kind} K={K}: Alpaca mid "
                        f"${alpaca_mid:.3f} < intrinsic ${intrinsic:.3f} — using intrinsic"
                    )
                    alpaca_mid = intrinsic
                _stats['alpaca_hits'] += 1
                _log_comparison(
                    ticker=ticker, strike=K, expiry=expiry_date, kind=kind,
                    bs_price_val=bs_val, alpaca_mid=alpaca_mid,
                    source='alpaca',
                    alpaca_bid=quote.get('bid', 0),
                    alpaca_ask=quote.get('ask', 0),
                    alpaca_iv=quote.get('iv', 0),
                    alpaca_delta=quote.get('delta', 0),
                    bs_sigma=sigma,
                )
                return alpaca_mid

            # Mid was 0 — likely market closed or no data. Fall through to BS.
            logger.debug(f"[PRICING-BRIDGE] Alpaca mid=0 for {ticker} {K} {kind}, using BS")

        except Exception as e:
            _stats['errors'] += 1
            logger.debug(f"[PRICING-BRIDGE] Alpaca quote failed for {ticker}: {e}")

    # Fallback to BS
    _stats['bs_fallbacks'] += 1
    if bs_val > 0:
        _log_comparison(
            ticker=ticker, strike=K, expiry=expiry_date, kind=kind,
            bs_price_val=bs_val, alpaca_mid=0,
            source='bs_fallback',
            bs_sigma=sigma,
        )
        return bs_val

    # Last resort: if bs_price_fn was None, return 0
    return 0.0


def get_premium_pair(
    ticker: str,
    S: float,
    K_short: float,
    K_long: float,
    T: float,
    sigma: float,
    expiry_date: Optional[date],
    kind: str = "put",
    bs_price_fn: Optional[Callable] = None,
) -> tuple:
    """
    Get premiums for a spread (short + long strikes). Convenience wrapper.

    Returns:
        (short_premium, long_premium)
    """
    short_prem = get_premium(ticker, S, K_short, T, sigma, expiry_date, kind, bs_price_fn)
    long_prem = get_premium(ticker, S, K_long, T, sigma, expiry_date, kind, bs_price_fn)
    return short_prem, long_prem


def get_bridge_stats() -> Dict[str, Any]:
    """Return current session stats for the bridge."""
    return {
        'engine': _engine_name,
        'alpaca_available': _alpaca_available,
        'alpaca_hits': _stats['alpaca_hits'],
        'bs_fallbacks': _stats['bs_fallbacks'],
        'errors': _stats['errors'],
        'hit_rate': (
            _stats['alpaca_hits'] / max(1, _stats['alpaca_hits'] + _stats['bs_fallbacks'])
        ),
    }


def log_session_summary():
    """Log a summary of pricing sources used this session."""
    stats = get_bridge_stats()
    total = stats['alpaca_hits'] + stats['bs_fallbacks']
    if total > 0:
        logger.info(
            f"[PRICING-BRIDGE] Session summary for {stats['engine']}: "
            f"{stats['alpaca_hits']} Alpaca quotes, {stats['bs_fallbacks']} BS fallbacks "
            f"({stats['hit_rate']:.0%} real pricing rate), {stats['errors']} errors"
        )
