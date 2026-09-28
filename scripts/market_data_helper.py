#!/usr/bin/env python3
"""
Market Data Helper for Finance Quant MCP Server.
Uses yfinance to fetch market data and returns JSON to stdout.

Usage:
  python3 market_data_helper.py snapshot [TICKERS...]
  python3 market_data_helper.py vix
  python3 market_data_helper.py trends [TICKERS...]
"""

import sys
import json
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import yfinance as yf
import numpy as np


def safe_float(val, decimals=4):
    """Convert to float safely, handle NaN/None."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, decimals)
    except (TypeError, ValueError):
        return None


def compute_rsi(prices, period=14):
    """Compute RSI from a price series."""
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def get_snapshot(tickers):
    """Get current price snapshot with technicals for given tickers."""
    if not tickers:
        tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', '^VIX', 'DX-Y.NYB']

    results = {}
    end = datetime.now()
    start = end - timedelta(days=60)

    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(start=start, end=end)
            if hist.empty or len(hist) < 2:
                results[ticker] = {'error': 'No data available'}
                continue

            closes = hist['Close'].values
            volumes = hist['Volume'].values
            current = safe_float(closes[-1], 2)
            prev = safe_float(closes[-2], 2)

            # Daily change
            daily_change_pct = safe_float(((closes[-1] / closes[-2]) - 1) * 100, 2) if prev else None

            # 20d SMA
            sma20 = safe_float(np.mean(closes[-20:]), 2) if len(closes) >= 20 else None
            above_sma20 = current > sma20 if (current and sma20) else None

            # RSI(14)
            rsi = compute_rsi(closes, 14)

            # Volume vs 20d avg
            vol_current = safe_float(volumes[-1], 0) if len(volumes) > 0 else None
            vol_avg20 = safe_float(np.mean(volumes[-20:]), 0) if len(volumes) >= 20 else None
            vol_ratio = safe_float(volumes[-1] / np.mean(volumes[-20:]), 2) if (len(volumes) >= 20 and np.mean(volumes[-20:]) > 0) else None

            # Display name
            display = ticker.replace('^VIX', 'VIX').replace('DX-Y.NYB', 'DXY')

            results[display] = {
                'price': current,
                'daily_change_pct': daily_change_pct,
                'sma20': sma20,
                'above_sma20': above_sma20,
                'rsi14': rsi,
                'volume': vol_current,
                'volume_avg20': vol_avg20,
                'volume_ratio': vol_ratio,
            }
        except Exception as e:
            results[ticker] = {'error': str(e)}

    return results


def get_vix_dashboard():
    """Get VIX analysis with term structure and percentile ranks."""
    result = {}

    try:
        # VIX current
        vix = yf.Ticker('^VIX')
        vix_hist = vix.history(period='3y')
        if vix_hist.empty:
            return {'error': 'Could not fetch VIX data'}

        current_vix = safe_float(vix_hist['Close'].iloc[-1], 2)
        result['vix_current'] = current_vix

        # Percentile ranks
        closes_1y = vix_hist['Close'].values[-252:] if len(vix_hist) >= 252 else vix_hist['Close'].values
        closes_3y = vix_hist['Close'].values

        pct_1y = safe_float(np.percentile(closes_1y, [np.sum(closes_1y <= current_vix) / len(closes_1y) * 100])[0], 1)
        pct_rank_1y = safe_float(np.sum(closes_1y <= current_vix) / len(closes_1y) * 100, 1)
        pct_rank_3y = safe_float(np.sum(closes_3y <= current_vix) / len(closes_3y) * 100, 1)

        result['percentile_rank_1y'] = pct_rank_1y
        result['percentile_rank_3y'] = pct_rank_3y

        # VIX 1y stats
        result['vix_1y_mean'] = safe_float(np.mean(closes_1y), 2)
        result['vix_1y_min'] = safe_float(np.min(closes_1y), 2)
        result['vix_1y_max'] = safe_float(np.max(closes_1y), 2)

        # Term structure: VIX vs VIX3M
        try:
            vix3m = yf.Ticker('^VIX3M')
            vix3m_hist = vix3m.history(period='5d')
            if not vix3m_hist.empty:
                current_vix3m = safe_float(vix3m_hist['Close'].iloc[-1], 2)
                result['vix3m'] = current_vix3m
                if current_vix3m and current_vix:
                    ratio = safe_float(current_vix / current_vix3m, 4)
                    result['vix_vix3m_ratio'] = ratio
                    result['term_structure'] = 'BACKWARDATION (risk elevated)' if ratio > 1.0 else 'CONTANGO (normal)'
        except Exception:
            result['vix3m'] = 'unavailable'

        # VVIX (vol of vol)
        try:
            vvix = yf.Ticker('^VVIX')
            vvix_hist = vvix.history(period='5d')
            if not vvix_hist.empty:
                result['vvix'] = safe_float(vvix_hist['Close'].iloc[-1], 2)
        except Exception:
            result['vvix'] = 'unavailable'

        # Flags
        flags = []
        if current_vix and current_vix >= 35:
            flags.append('EXTREME VOLATILITY (VIX >= 35)')
        elif current_vix and current_vix >= 25:
            flags.append('SPIKE ALERT (VIX >= 25)')
        if current_vix and current_vix <= 13:
            flags.append('COMPLACENCY WARNING (VIX <= 13)')
        result['flags'] = flags

    except Exception as e:
        result['error'] = str(e)

    return result


def get_trend_signals(tickers):
    """Compute trend direction, momentum, and relative strength for assets."""
    if not tickers:
        tickers = ['SPY', 'QQQ', 'GLD', 'TLT', 'EEM', 'HYG', 'VNQ', 'XLE']

    end = datetime.now()
    start = end - timedelta(days=400)  # Need ~13 months for 12m-1m momentum

    results = {}
    spy_data = None

    # Fetch SPY first for relative strength
    try:
        spy_tk = yf.Ticker('SPY')
        spy_hist = spy_tk.history(start=start, end=end)
        if not spy_hist.empty and len(spy_hist) >= 22:
            spy_data = spy_hist['Close'].values
    except Exception:
        pass

    bullish = 0
    bearish = 0

    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(start=start, end=end)
            if hist.empty or len(hist) < 50:
                results[ticker] = {'error': 'Insufficient data'}
                continue

            closes = hist['Close'].values
            current = safe_float(closes[-1], 2)

            # Trend: above/below 50d SMA
            sma50 = safe_float(np.mean(closes[-50:]), 2)
            trend = 'BULLISH' if current > sma50 else 'BEARISH'

            if trend == 'BULLISH':
                bullish += 1
            else:
                bearish += 1

            # Momentum: 12m return - 1m return (standard momentum factor)
            ret_12m = safe_float(((closes[-1] / closes[-252]) - 1) * 100, 2) if len(closes) >= 252 else None
            ret_1m = safe_float(((closes[-1] / closes[-22]) - 1) * 100, 2) if len(closes) >= 22 else None
            momentum = safe_float(ret_12m - ret_1m, 2) if (ret_12m is not None and ret_1m is not None) else None

            # Relative strength vs SPY (ratio of 3m returns)
            rel_strength = None
            if spy_data is not None and len(closes) >= 63 and len(spy_data) >= 63:
                asset_ret = (closes[-1] / closes[-63]) - 1
                spy_ret = (spy_data[-1] / spy_data[-63]) - 1
                if spy_ret != 0:
                    rel_strength = safe_float(asset_ret / spy_ret, 3)

            results[ticker] = {
                'price': current,
                'sma50': sma50,
                'trend': trend,
                'return_12m_pct': ret_12m,
                'return_1m_pct': ret_1m,
                'momentum_score': momentum,
                'rel_strength_vs_spy': rel_strength,
            }
        except Exception as e:
            results[ticker] = {'error': str(e)}

    # Overall assessment
    total = bullish + bearish
    assessment = 'NEUTRAL'
    if total > 0:
        bull_pct = bullish / total
        if bull_pct >= 0.7:
            assessment = 'RISK-ON (strong breadth)'
        elif bull_pct >= 0.5:
            assessment = 'LEANING RISK-ON'
        elif bull_pct >= 0.3:
            assessment = 'LEANING RISK-OFF'
        else:
            assessment = 'RISK-OFF (weak breadth)'

    return {
        'assets': results,
        'summary': {
            'bullish_count': bullish,
            'bearish_count': bearish,
            'assessment': assessment,
        },
        'generated_at': datetime.now().isoformat(),
    }


def main():
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: market_data_helper.py <command> [args...]'}))
        sys.exit(1)

    cmd = sys.argv[1]
    tickers = sys.argv[2:] if len(sys.argv) > 2 else None

    try:
        if cmd == 'snapshot':
            result = get_snapshot(tickers)
        elif cmd == 'vix':
            result = get_vix_dashboard()
        elif cmd == 'trends':
            result = get_trend_signals(tickers)
        else:
            result = {'error': f'Unknown command: {cmd}'}
    except Exception as e:
        result = {'error': f'Fatal: {str(e)}'}

    print(json.dumps(result, default=str))


if __name__ == '__main__':
    main()
