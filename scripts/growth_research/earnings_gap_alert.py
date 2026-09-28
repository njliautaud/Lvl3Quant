#!/usr/bin/env python3
"""
Earnings Gap Alert System
=========================
Checks post-earnings gaps at market open and evaluates option trades.
Designed for the agentic account ($645, Level 2, high-growth mode).

Strategy: Post-Earnings Announcement Drift (PEAD)
- If stock gaps 5%+ after earnings, buy ATM option in gap direction
- Post-earnings IV is crushed → cheaper entry
- Hold 1-3 days riding continued momentum drift
- +30% TP, -25% SL, 50% trailing giveback

Runs via cron at 9:35 AM ET on weekdays during earnings season.
Also callable manually: python3 earnings_gap_alert.py [--check-now]

Outputs actionable trade recommendations to stdout and state/earnings_gap_signals.json
"""

import sys
import os
import json
import argparse
from datetime import datetime, timedelta

# Path setup
for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STATE_DIR = os.path.join(LVL3_ROOT, 'state')
os.makedirs(STATE_DIR, exist_ok=True)

# Target universe — stocks we've backtested in earnings momentum v1
EARNINGS_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI',
]

# Backtest performance by ticker (from earnings_momentum_v1 variant A)
TICKER_HISTORY = {
    'LI':   {'trades': 7,  'wr': 85.7, 'avg_pnl': 154.31, 'avg_gap': 7.7},
    'SHOP': {'trades': 1,  'wr': 100.0, 'avg_pnl': 550.96, 'avg_gap': 18.6},
    'PYPL': {'trades': 4,  'wr': 50.0, 'avg_pnl': 55.80, 'avg_gap': 10.3},
    'UBER': {'trades': 5,  'wr': 60.0, 'avg_pnl': 34.38, 'avg_gap': 9.1},
    'U':    {'trades': 2,  'wr': 50.0, 'avg_pnl': 84.54, 'avg_gap': 12.2},
    'PLTR': {'trades': 5,  'wr': 80.0, 'avg_pnl': 21.94, 'avg_gap': 14.6},
    'BABA': {'trades': 1,  'wr': 100.0, 'avg_pnl': 96.74, 'avg_gap': 8.7},
    'NIO':  {'trades': 5,  'wr': 60.0, 'avg_pnl': 14.97, 'avg_gap': 7.3},
    'RKLB': {'trades': 1,  'wr': 100.0, 'avg_pnl': 68.69, 'avg_gap': 6.7},
    'XPEV': {'trades': 5,  'wr': 40.0, 'avg_pnl': 5.39, 'avg_gap': 8.8},
    # Negative tickers — be cautious
    'LYFT': {'trades': 1,  'wr': 0.0, 'avg_pnl': -21.71, 'avg_gap': 15.7},
    'JD':   {'trades': 2,  'wr': 0.0, 'avg_pnl': -42.90, 'avg_gap': 11.3},
    'PINS': {'trades': 1,  'wr': 0.0, 'avg_pnl': -94.01, 'avg_gap': 10.9},
    'RBLX': {'trades': 4,  'wr': 25.0, 'avg_pnl': -44.14, 'avg_gap': 19.0},
    'SOFI': {'trades': 9,  'wr': 33.3, 'avg_pnl': -24.35, 'avg_gap': 11.0},
}


def get_gap_and_prices():
    """
    Get current prices and compute gaps vs previous close.
    Returns list of dicts with gap info for reporting stocks.
    """
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed")
        return []

    # Get today's reporting stocks from the saved earnings calendar
    # For now, hardcode this week's schedule (will be updated by cron)
    today = datetime.now().strftime('%Y-%m-%d')

    # This week's earnings from our universe
    schedule = {
        '2026-07-28': {'am': ['PYPL'], 'pm': ['ENPH']},
        '2026-07-29': {'am': ['SOFI'], 'pm': ['MSFT', 'META', 'HOOD', 'CMG', 'ARM', 'ALGN']},
        '2026-07-30': {'am': [], 'pm': ['AAPL', 'AMZN', 'COIN', 'RIVN', 'DXCM', 'FSLR', 'RBLX']},
        '2026-07-31': {'am': [], 'pm': []},
    }

    # Determine which stocks reported since last close
    # AM reporters today + PM reporters yesterday
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    if datetime.now().weekday() == 0:  # Monday
        yesterday = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')

    check_tickers = []
    if today in schedule:
        check_tickers.extend(schedule[today].get('am', []))
    if yesterday in schedule:
        check_tickers.extend(schedule[yesterday].get('pm', []))

    if not check_tickers:
        print(f"No earnings reporters to check today ({today})")
        return []

    print(f"Checking gaps for: {check_tickers}")

    # Get prices
    gaps = []
    for ticker in check_tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='5d')
            if len(hist) < 2:
                continue

            prev_close = hist['Close'].iloc[-2]
            current_open = hist['Open'].iloc[-1]
            current_price = hist['Close'].iloc[-1]

            gap_pct = (current_open / prev_close - 1) * 100

            # Get 5d momentum before earnings
            if len(hist) >= 5:
                momentum_5d = (hist['Close'].iloc[-2] / hist['Close'].iloc[0] - 1) * 100
            else:
                momentum_5d = 0

            gaps.append({
                'ticker': ticker,
                'prev_close': round(prev_close, 2),
                'open': round(current_open, 2),
                'current': round(current_price, 2),
                'gap_pct': round(gap_pct, 1),
                'abs_gap': round(abs(gap_pct), 1),
                'direction': 'UP' if gap_pct > 0 else 'DOWN',
                'momentum_5d': round(momentum_5d, 1),
                'momentum_aligned': (gap_pct > 0 and momentum_5d > 0) or (gap_pct < 0 and momentum_5d < 0),
                'in_backtest': ticker in TICKER_HISTORY,
                'backtest_wr': TICKER_HISTORY.get(ticker, {}).get('wr', 'N/A'),
                'backtest_avg_pnl': TICKER_HISTORY.get(ticker, {}).get('avg_pnl', 'N/A'),
            })
        except Exception as e:
            print(f"  Error getting {ticker}: {e}")

    return gaps


def get_ml_score(ticker):
    """Get ML confidence score from PEAD ML Live Scorer."""
    try:
        sys.path.insert(0, os.path.join(LVL3_ROOT, 'scripts', 'growth_research'))
        from pead_ml_live_scorer import score_ticker, load_model
        model = load_model()
        if model is None:
            return None
        result = score_ticker(ticker, model)
        if 'error' in result:
            return None
        return result
    except Exception as e:
        print(f"  ML scorer error for {ticker}: {e}")
        return None


def evaluate_trade(gap_info, account_equity=645.0, max_position=200.0):
    """
    Evaluate whether a post-gap trade meets our criteria.
    Uses ML model (PEAD ML Predictor v1 Variant D) for confidence scoring.
    Returns trade recommendation dict or None.
    """
    ticker = gap_info['ticker']
    abs_gap = gap_info['abs_gap']
    direction = gap_info['direction']

    # Gate 1: Gap must be >= 5%
    if abs_gap < 5.0:
        return None

    # Gate 2: Backtest history check
    bt = TICKER_HISTORY.get(ticker, {})
    if bt and bt.get('wr', 50) < 30:
        # This ticker has negative backtest history
        return {
            'ticker': ticker,
            'action': 'SKIP',
            'reason': f'Negative backtest history (WR={bt["wr"]}%, avg PnL=${bt["avg_pnl"]:.0f})',
            'gap_pct': gap_info['gap_pct'],
        }

    # Gate 3: ML Confidence Score (PEAD ML Predictor v1 Variant D)
    ml_result = get_ml_score(ticker)
    ml_confidence = ml_result['ml_confidence'] if ml_result else None
    ml_momentum = ml_result['momentum_aligned'] if ml_result else gap_info['momentum_aligned']

    # Gate 4: Momentum alignment (variant D filter)
    momentum_aligned = gap_info['momentum_aligned'] and ml_momentum

    # Estimate option cost
    stock_price = gap_info['current']
    strike = round(stock_price)

    # Use ML estimate if available, else rough estimate
    if ml_result and ml_result.get('est_contract_cost'):
        est_contract_cost = ml_result['est_contract_cost']
        est_premium = ml_result['est_premium']
    else:
        est_premium_pct = 0.04 if abs_gap < 10 else 0.05
        est_premium = stock_price * est_premium_pct
        est_contract_cost = est_premium * 100

    affordable = est_contract_cost <= max_position

    # Confidence scoring — ML-enhanced
    confidence = 0
    reasons = []

    # ML confidence is primary signal
    if ml_confidence is not None:
        if ml_confidence >= 0.70:
            confidence += 2.5
            reasons.append(f'HIGH ML confidence ({ml_confidence:.0%})')
        elif ml_confidence >= 0.60:
            confidence += 1.5
            reasons.append(f'MEDIUM ML confidence ({ml_confidence:.0%})')
        else:
            reasons.append(f'LOW ML confidence ({ml_confidence:.0%}) — caution')
    else:
        reasons.append('ML scorer unavailable — using heuristics')

    if abs_gap >= 10:
        confidence += 1
        reasons.append(f'Large gap ({gap_info["gap_pct"]:+.1f}%)')
    elif abs_gap >= 5:
        confidence += 0.5
        reasons.append(f'Moderate gap ({gap_info["gap_pct"]:+.1f}%)')

    if momentum_aligned:
        confidence += 1
        reasons.append('Momentum aligned')

    if bt and bt.get('wr', 0) >= 60:
        confidence += 1
        reasons.append(f'Strong backtest (WR={bt["wr"]}%)')
    elif bt and bt.get('wr', 0) >= 50:
        confidence += 0.5
        reasons.append(f'OK backtest (WR={bt["wr"]}%)')

    if not affordable:
        reasons.append(f'⚠️ Option too expensive (~${est_contract_cost:.0f})')

    opt_type = 'CALL' if direction == 'UP' else 'PUT'

    # Decision: BUY requires ML confidence >= 0.60 OR heuristic confidence >= 2
    should_buy = (
        (ml_confidence is not None and ml_confidence >= 0.60 and momentum_aligned and affordable) or
        (ml_confidence is None and confidence >= 2 and affordable)
    )

    return {
        'ticker': ticker,
        'action': 'BUY' if should_buy else 'WATCH',
        'option_type': opt_type,
        'strike_est': strike,
        'est_cost': round(est_contract_cost),
        'confidence': confidence,
        'ml_confidence': ml_confidence,
        'reasons': reasons,
        'gap_pct': gap_info['gap_pct'],
        'momentum_aligned': momentum_aligned,
        'affordable': affordable,
        'trade_plan': {
            'entry': f'Buy {opt_type} ${strike} ~14 DTE at market open',
            'tp': '+30% premium gain',
            'sl': '-25% premium loss',
            'trailing': '50% giveback from peak',
            'time_stop': '3 trading days',
            'max_cost': f'${min(est_contract_cost, max_position):.0f}',
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-now', action='store_true', help='Run gap check immediately')
    args = parser.parse_args()

    print("=" * 60)
    print("  EARNINGS GAP ALERT — AGENTIC ACCOUNT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    print("=" * 60)

    gaps = get_gap_and_prices()

    if not gaps:
        print("\nNo gaps to evaluate.")
        return

    print(f"\nFound {len(gaps)} stocks to evaluate:")

    recommendations = []
    for gap in sorted(gaps, key=lambda x: abs(x['gap_pct']), reverse=True):
        print(f"\n  {gap['ticker']}: {gap['gap_pct']:+.1f}% gap "
              f"(${gap['prev_close']} → ${gap['open']}), "
              f"now ${gap['current']}")

        rec = evaluate_trade(gap)
        if rec:
            recommendations.append(rec)
            action = rec['action']
            if action == 'BUY':
                print(f"    ✅ TRADE: {rec['trade_plan']['entry']}")
                print(f"       Cost: ~${rec['est_cost']}, Confidence: {rec['confidence']}/4")
                print(f"       Reasons: {', '.join(rec['reasons'])}")
            elif action == 'WATCH':
                print(f"    👀 WATCH: {', '.join(rec['reasons'])}")
            elif action == 'SKIP':
                print(f"    ❌ SKIP: {rec['reason']}")
        else:
            print(f"    ⏭️ Gap < 5%, no trade")

    # Save to state
    output = {
        'timestamp': datetime.now().isoformat(),
        'gaps': gaps,
        'recommendations': recommendations,
        'actionable': [r for r in recommendations if r['action'] == 'BUY'],
    }

    output_path = os.path.join(STATE_DIR, 'earnings_gap_signals.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Summary
    buys = [r for r in recommendations if r['action'] == 'BUY']
    if buys:
        print(f"\n{'=' * 60}")
        print(f"  🚨 {len(buys)} ACTIONABLE TRADE(S):")
        for b in buys:
            print(f"    {b['ticker']}: {b['trade_plan']['entry']}")
            print(f"    Cost: ~${b['est_cost']}, TP: {b['trade_plan']['tp']}, SL: {b['trade_plan']['sl']}")
        print(f"{'=' * 60}")
    else:
        print(f"\n  No actionable trades today (gaps < 5% or too expensive)")

    return output


if __name__ == '__main__':
    main()
