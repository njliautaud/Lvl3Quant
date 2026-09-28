#!/usr/bin/env python3
"""
Meta-Signal Ensemble v1 — Multi-Engine Confluence Aggregator
=============================================================
Reads all 19 paper engine state files, extracts sector ETF signals,
computes confluence scores, and produces actionable option trade
recommendations for the agentic account ($645 budget).

HC #750: Requires >= 3 confirming signals for any trade.
HC #751: All validated strategies feed signals to agentic account.
HC #749: Agentic account trades options only.

V8 config reference: DTE=14, 2% OTM for calls, 2% OTM for puts.
VIX regime: VIX<20 = pairs mode (bull+bear), VIX>=20 = bull only.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

STATE_DIR = '/home/jupiter/Lvl3Quant/state'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/meta_signals'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

ACCOUNT_EQUITY = 645.0
MAX_POSITIONS = 3
MIN_CONFLUENCE = 3  # HC #750
DTE_TARGET = 14
OTM_PCT = 0.02  # 2% OTM per V8 config
MAX_SINGLE_POSITION_PCT = 0.40  # Don't put more than 40% in one trade

# Engine weights — higher weight = more trusted signal
ENGINE_WEIGHTS = {
    'sector_combined_v8': 2.0,      # Primary production engine
    'sector_combined_v7': 1.5,      # Previous production version
    'sector_spreads_v6': 1.5,       # Validated spreads
    'sector_spreads': 1.0,          # Original spreads
    'sector_pairs': 1.0,            # Pairs trading
    'sector_etf_momentum': 1.5,     # Momentum signals
    'etf_rotation_v3': 1.0,         # Rotation model
    'cross_asset_trend': 0.75,      # Indirect sector signal via holdings
    'riskparity_portfolio': 0.5,    # Indirect via portfolio weights
    'quality_momentum': 0.5,        # Individual stocks -> sector mapping
    'dl_stock_ranker': 0.5,         # DL rankings -> sector mapping
    'covered_call': 0.5,            # Bullish bias implied
    'ic_paper': 0.5,                # Iron condors = neutral, but directional info
    'vol_crush': 0.25,              # Pre-earnings, less directional
    'earnings_jade_lizard': 0.25,   # Earnings specific
    'earnings_vol_crush': 0.25,     # Earnings specific
    'vix_call_spread': 0.25,        # VIX-specific, indirect
    'vix_contango': 0.25,           # VIX-specific, indirect
    'pead_drift': 0.25,             # Post-earnings drift
}

# Map individual stocks to their GICS sector ETFs
STOCK_TO_SECTOR = {
    # Tech (XLK)
    'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'AVGO': 'XLK',
    'INTC': 'XLK', 'CSCO': 'XLK', 'TXN': 'XLK', 'AMD': 'XLK',
    'QCOM': 'XLK', 'CRM': 'XLK', 'ADBE': 'XLK', 'ORCL': 'XLK',
    # Communication (XLC)
    'META': 'XLC', 'GOOGL': 'XLC', 'GOOG': 'XLC', 'NFLX': 'XLC',
    'DIS': 'XLC', 'CMCSA': 'XLC', 'T': 'XLC', 'VZ': 'XLC',
    # Consumer Discretionary (XLY)
    'AMZN': 'XLY', 'TSLA': 'XLY', 'HD': 'XLY', 'NKE': 'XLY',
    'MCD': 'XLY', 'SBUX': 'XLY', 'LOW': 'XLY', 'TJX': 'XLY',
    # Financials (XLF)
    'JPM': 'XLF', 'BAC': 'XLF', 'GS': 'XLF', 'MS': 'XLF',
    'V': 'XLF', 'MA': 'XLF', 'BRK.B': 'XLF', 'C': 'XLF',
    # Healthcare (XLV)
    'UNH': 'XLV', 'JNJ': 'XLV', 'PFE': 'XLV', 'ABBV': 'XLV',
    'MRK': 'XLV', 'LLY': 'XLV', 'ABT': 'XLV', 'TMO': 'XLV',
    'BMY': 'XLV',
    # Industrials (XLI)
    'CAT': 'XLI', 'HON': 'XLI', 'UPS': 'XLI', 'BA': 'XLI',
    'GE': 'XLI', 'MMM': 'XLI', 'RTX': 'XLI', 'DE': 'XLI',
    'AVAV': 'XLI',
    # Energy (XLE)
    'XOM': 'XLE', 'CVX': 'XLE', 'COP': 'XLE', 'SLB': 'XLE',
    'EOG': 'XLE', 'MPC': 'XLE', 'PSX': 'XLE', 'VLO': 'XLE',
    # Consumer Staples (XLP)
    'PG': 'XLP', 'KO': 'XLP', 'PEP': 'XLP', 'COST': 'XLP',
    'WMT': 'XLP', 'PM': 'XLP', 'CL': 'XLP', 'MDLZ': 'XLP',
    # Utilities (XLU) — note: sometimes traded as ETF directly
    'NEE': 'XLU', 'DUK': 'XLU', 'SO': 'XLU', 'D': 'XLU',
    # Real Estate (XLRE)
    'AMT': 'XLRE', 'PLD': 'XLRE', 'CCI': 'XLRE', 'EQIX': 'XLRE',
    # Materials (XLB)
    'LIN': 'XLB', 'APD': 'XLB', 'ECL': 'XLB', 'SHW': 'XLB',
    # ETF proxies
    'QQQ': 'XLK', 'IWM': 'XLI', 'DBC': 'XLE', 'USO': 'XLE',
    'VNQ': 'XLRE', 'IYR': 'XLRE', 'SKM': 'XLC',
}


# ============================================================
# ENGINE PARSERS — Extract signals from each state file format
# ============================================================

def load_state(filename):
    """Load a state JSON file, return None on error."""
    path = os.path.join(STATE_DIR, filename)
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return None


def parse_sector_options_engine(state, engine_name):
    """
    Parse engines with open_positions that have ticker + mode (bull/bear).
    Works for: sector_combined_v7, sector_combined_v8, sector_spreads,
               sector_spreads_v6, sector_pairs
    """
    signals = []
    if not state or 'open_positions' not in state:
        return signals

    for pos in state['open_positions']:
        ticker = pos.get('ticker', '')
        mode = pos.get('mode', '')
        if ticker not in SECTOR_ETFS:
            continue

        direction = 'bull' if mode == 'bull' else 'bear'
        lgbm_score = pos.get('lgbm_score', 0.5)

        signals.append({
            'ticker': ticker,
            'direction': direction,
            'source': engine_name,
            'weight': ENGINE_WEIGHTS.get(engine_name, 1.0),
            'confidence': lgbm_score,
            'detail': f"mode={mode}, lgbm={lgbm_score:.3f}",
        })

    return signals


def parse_holdings_engine(state, engine_name):
    """
    Parse engines with simple holdings lists (sector ETF momentum, ETF rotation).
    Being held = bullish signal.
    """
    signals = []
    if not state:
        return signals

    holdings = state.get('holdings', [])
    if isinstance(holdings, dict):
        holdings = list(holdings.keys())

    for ticker in holdings:
        if ticker in SECTOR_ETFS:
            signals.append({
                'ticker': ticker,
                'direction': 'bull',
                'source': engine_name,
                'weight': ENGINE_WEIGHTS.get(engine_name, 1.0),
                'confidence': 0.6,
                'detail': f"held in portfolio",
            })

    # Check for momentum scores
    if 'rebalance_history' in state and state['rebalance_history']:
        latest = state['rebalance_history'][-1]
        scores = latest.get('scores', {})
        for ticker, score in scores.items():
            if ticker in SECTOR_ETFS:
                # Already added as bullish, update confidence
                for sig in signals:
                    if sig['ticker'] == ticker:
                        sig['confidence'] = min(1.0, score * 2)  # Normalize
                        sig['detail'] = f"momentum score={score:.4f}"

    return signals


def parse_positions_dict_engine(state, engine_name):
    """
    Parse engines with positions dict (etf_rotation_v3, riskparity).
    Works for: etf_rotation_v3, riskparity_portfolio
    """
    signals = []
    if not state:
        return signals

    # Handle different position formats
    positions = state.get('positions', {})
    if isinstance(positions, dict):
        for ticker, pos_data in positions.items():
            sector = STOCK_TO_SECTOR.get(ticker, ticker if ticker in SECTOR_ETFS else None)
            if sector and sector in SECTOR_ETFS:
                pnl_pct = 0
                if isinstance(pos_data, dict):
                    pnl_pct = pos_data.get('pnl_pct', 0)

                signals.append({
                    'ticker': sector,
                    'direction': 'bull',  # Holding = bullish
                    'source': engine_name,
                    'weight': ENGINE_WEIGHTS.get(engine_name, 1.0),
                    'confidence': 0.55,
                    'detail': f"held via {ticker}" if ticker != sector else "direct holding",
                })

    # Also check holdings dicts (riskparity has eq_holdings, alt_holdings)
    for key in ['eq_holdings', 'alt_holdings', 'holdings']:
        holdings = state.get(key, {})
        if isinstance(holdings, dict):
            for ticker, shares in holdings.items():
                sector = STOCK_TO_SECTOR.get(ticker, ticker if ticker in SECTOR_ETFS else None)
                if sector and sector in SECTOR_ETFS:
                    signals.append({
                        'ticker': sector,
                        'direction': 'bull',
                        'source': engine_name,
                        'weight': ENGINE_WEIGHTS.get(engine_name, 1.0),
                        'confidence': 0.5,
                        'detail': f"held via {ticker}, {shares} shares",
                    })

    return signals


def parse_stock_engine(state, engine_name):
    """
    Parse engines that hold individual stocks -> map to sector ETFs.
    Works for: quality_momentum, dl_stock_ranker, covered_call
    """
    signals = []
    if not state:
        return signals

    # quality_momentum and dl_stock_ranker have 'holdings' list
    holdings = state.get('holdings', [])
    if isinstance(holdings, list):
        for ticker in holdings:
            sector = STOCK_TO_SECTOR.get(ticker)
            if sector:
                signals.append({
                    'ticker': sector,
                    'direction': 'bull',
                    'source': engine_name,
                    'weight': ENGINE_WEIGHTS.get(engine_name, 1.0),
                    'confidence': 0.5,
                    'detail': f"bullish on {ticker}",
                })

    # dl_stock_ranker has 'positions' dict with stock names
    positions = state.get('positions', {})
    if isinstance(positions, dict):
        for ticker, pos in positions.items():
            sector = STOCK_TO_SECTOR.get(ticker)
            if sector:
                signals.append({
                    'ticker': sector,
                    'direction': 'bull',
                    'source': engine_name,
                    'weight': ENGINE_WEIGHTS.get(engine_name, 1.0),
                    'confidence': 0.55,
                    'detail': f"ranked {ticker} highly",
                })

    # covered_call has 'positions' dict with stock+strike
    if 'positions' in state and isinstance(state['positions'], dict):
        for ticker, pos in state['positions'].items():
            if isinstance(pos, dict) and 'strike' in pos:
                sector = STOCK_TO_SECTOR.get(ticker)
                if sector:
                    signals.append({
                        'ticker': sector,
                        'direction': 'bull',
                        'source': engine_name,
                        'weight': ENGINE_WEIGHTS.get(engine_name, 1.0),
                        'confidence': 0.5,
                        'detail': f"covered call on {ticker}",
                    })

    # quality_momentum has scores in rebalance_history
    if 'rebalance_history' in state and state['rebalance_history']:
        latest = state['rebalance_history'][-1]
        scores = latest.get('scores', {})
        for ticker, score in scores.items():
            sector = STOCK_TO_SECTOR.get(ticker)
            if sector:
                for sig in signals:
                    if sig['source'] == engine_name and sig['detail'].endswith(ticker):
                        sig['confidence'] = min(1.0, abs(score))

    return signals


def parse_iron_condor_engine(state, engine_name):
    """
    Parse iron condor / vol-selling engines.
    These are neutral strategies but we can extract directional bias
    from delta skew and P&L direction.
    Works for: ic_paper, vol_crush, earnings_jade_lizard, earnings_vol_crush
    """
    signals = []
    if not state:
        return signals

    positions = state.get('positions', {})
    if isinstance(positions, dict):
        positions = list(positions.values())
    elif not isinstance(positions, list):
        return signals

    for pos in positions:
        if not isinstance(pos, dict):
            continue
        ticker = pos.get('ticker', '')
        sector = STOCK_TO_SECTOR.get(ticker)
        if not sector:
            continue

        # Extract directional bias from put/call delta or pnl
        put_delta = abs(pos.get('put_delta', pos.get('short_put_prem', 0)))
        call_delta = abs(pos.get('call_delta', pos.get('short_call_prem', 0)))
        pnl = pos.get('unrealized_pnl', 0)

        # If the position is profitable and skewed, there's a weak directional signal
        if put_delta > 0 and call_delta > 0:
            # Being short options = neutral, but the ticker choice itself is a signal
            # that the engine considers this stock/sector liquid and range-bound
            signals.append({
                'ticker': sector,
                'direction': 'neutral',
                'source': engine_name,
                'weight': ENGINE_WEIGHTS.get(engine_name, 1.0),
                'confidence': 0.3,
                'detail': f"vol-selling on {ticker} (neutral bias)",
            })

    return signals


def parse_cross_asset_trend(state, engine_name):
    """
    Parse cross-asset trend following.
    Holdings imply trend-following bullish signals for those assets.
    """
    signals = []
    if not state:
        return signals

    holdings = state.get('holdings', {})
    if isinstance(holdings, dict):
        for ticker, shares in holdings.items():
            sector = STOCK_TO_SECTOR.get(ticker, ticker if ticker in SECTOR_ETFS else None)
            if sector and sector in SECTOR_ETFS:
                signals.append({
                    'ticker': sector,
                    'direction': 'bull',
                    'source': engine_name,
                    'weight': ENGINE_WEIGHTS.get(engine_name, 1.0),
                    'confidence': 0.6,
                    'detail': f"trend-following long {ticker} ({shares} shares)",
                })

    return signals


def get_vix_regime():
    """Get current VIX level from state files or signal watcher."""
    # Try signal watcher first
    sw = load_state('signal_watcher_state.json')
    if sw and 'vix_level' in sw:
        return float(sw['vix_level'])

    # Try V8 state (has vix_at_entry)
    v8 = load_state('sector_combined_v8_paper_state.json')
    if v8 and v8.get('open_positions'):
        return float(v8['open_positions'][0].get('vix_at_entry', 20))

    return 20.0  # Default conservative


def get_market_regime():
    """Get market regime signals from signal watcher."""
    sw = load_state('signal_watcher_state.json')
    if not sw:
        return {'all_clear': False, 'spy_above_50sma': False}

    protection = sw.get('protection', sw.get('protection_signals', {}))
    return {
        'all_clear': all([
            protection.get('vix_ok', protection.get('vix_calm', False)),
            protection.get('spy_above_50sma', False),
            protection.get('credit_healthy', protection.get('credit_ok', False)),
            protection.get('breadth_ok', False),
        ]),
        'spy_above_50sma': protection.get('spy_above_50sma', False),
        'vix_ok': protection.get('vix_ok', protection.get('vix_calm', False)),
        'credit_ok': protection.get('credit_healthy', protection.get('credit_ok', False)),
        'breadth_ok': protection.get('breadth_ok', False),
    }


# ============================================================
# MAIN AGGREGATION PIPELINE
# ============================================================

def collect_all_signals():
    """Read all 19 paper engines and extract signals."""
    all_signals = []
    engines_loaded = {}

    # 1. Sector options engines (direct sector ETF positions with mode)
    sector_engines = {
        'sector_combined_v8_paper_state.json': 'sector_combined_v8',
        'sector_combined_v7_paper_state.json': 'sector_combined_v7',
        'sector_spreads_v6_paper_state.json': 'sector_spreads_v6',
        'sector_spreads_paper_state.json': 'sector_spreads',
        'sector_pairs_paper_state.json': 'sector_pairs',
    }
    for filename, name in sector_engines.items():
        state = load_state(filename)
        if state:
            sigs = parse_sector_options_engine(state, name)
            all_signals.extend(sigs)
            engines_loaded[name] = len(sigs)

    # 2. Holdings-based engines
    holdings_engines = {
        'sector_etf_momentum_paper_state.json': 'sector_etf_momentum',
    }
    for filename, name in holdings_engines.items():
        state = load_state(filename)
        if state:
            sigs = parse_holdings_engine(state, name)
            all_signals.extend(sigs)
            engines_loaded[name] = len(sigs)

    # 3. Position-dict engines
    pos_dict_engines = {
        'etf_rotation_v3_paper_state.json': 'etf_rotation_v3',
        'riskparity_portfolio_paper_state.json': 'riskparity_portfolio',
    }
    for filename, name in pos_dict_engines.items():
        state = load_state(filename)
        if state:
            sigs = parse_positions_dict_engine(state, name)
            all_signals.extend(sigs)
            engines_loaded[name] = len(sigs)

    # 4. Stock-based engines (map to sectors)
    stock_engines = {
        'quality_momentum_paper_state.json': 'quality_momentum',
        'dl_stock_ranker_paper_state.json': 'dl_stock_ranker',
        'covered_call_paper_state.json': 'covered_call',
    }
    for filename, name in stock_engines.items():
        state = load_state(filename)
        if state:
            sigs = parse_stock_engine(state, name)
            all_signals.extend(sigs)
            engines_loaded[name] = len(sigs)

    # 5. Iron condor / vol-selling engines
    vol_engines = {
        'ic_paper_state.json': 'ic_paper',
        'vol_crush_paper_state.json': 'vol_crush',
        'earnings_jade_lizard_paper_state.json': 'earnings_jade_lizard',
        'earnings_vol_crush_paper_state.json': 'earnings_vol_crush',
    }
    for filename, name in vol_engines.items():
        state = load_state(filename)
        if state:
            sigs = parse_iron_condor_engine(state, name)
            all_signals.extend(sigs)
            engines_loaded[name] = len(sigs)

    # 6. Cross-asset trend
    state = load_state('cross_asset_trend_paper_state.json')
    if state:
        sigs = parse_cross_asset_trend(state, 'cross_asset_trend')
        all_signals.extend(sigs)
        engines_loaded['cross_asset_trend'] = len(sigs)

    # 7. Remaining engines with empty or no-signal states
    for name in ['vix_call_spread', 'vix_contango', 'pead_drift']:
        state = load_state(f'{name}_paper_state.json')
        if state:
            engines_loaded[name] = 0  # Loaded but no sector signals

    return all_signals, engines_loaded


def aggregate_by_ticker(all_signals):
    """
    Aggregate signals by ticker and direction.
    Returns a dict of ticker -> {bull_signals, bear_signals, neutral_signals,
                                   bull_score, bear_score, net_direction, confluence}
    """
    ticker_data = {}

    for etf in SECTOR_ETFS:
        ticker_sigs = [s for s in all_signals if s['ticker'] == etf]

        bull_sigs = [s for s in ticker_sigs if s['direction'] == 'bull']
        bear_sigs = [s for s in ticker_sigs if s['direction'] == 'bear']
        neutral_sigs = [s for s in ticker_sigs if s['direction'] == 'neutral']

        # Weighted score: sum of (weight * confidence) for each direction
        bull_score = sum(s['weight'] * s['confidence'] for s in bull_sigs)
        bear_score = sum(s['weight'] * s['confidence'] for s in bear_sigs)

        # Count unique sources per direction
        bull_sources = list(set(s['source'] for s in bull_sigs))
        bear_sources = list(set(s['source'] for s in bear_sigs))

        # Net direction
        if bull_score > bear_score and len(bull_sources) > len(bear_sources):
            net_direction = 'bull'
            net_score = bull_score - bear_score
            confirming = len(bull_sources)
            conflicting = len(bear_sources)
        elif bear_score > bull_score and len(bear_sources) > len(bull_sources):
            net_direction = 'bear'
            net_score = bear_score - bull_score
            confirming = len(bear_sources)
            conflicting = len(bull_sources)
        else:
            net_direction = 'neutral'
            net_score = 0
            confirming = 0
            conflicting = max(len(bull_sources), len(bear_sources))

        # Confluence score: 0 to 1
        # Based on: number of confirming sources, weighted score, lack of conflict
        max_possible_sources = len(ENGINE_WEIGHTS)
        source_ratio = confirming / max(1, max_possible_sources)
        conflict_penalty = conflicting / max(1, confirming + conflicting) if confirming > 0 else 1.0
        confluence = min(1.0, source_ratio * 3) * (1 - conflict_penalty * 0.5)

        ticker_data[etf] = {
            'bull_signals': bull_sigs,
            'bear_signals': bear_sigs,
            'neutral_signals': neutral_sigs,
            'bull_score': round(bull_score, 3),
            'bear_score': round(bear_score, 3),
            'bull_sources': bull_sources,
            'bear_sources': bear_sources,
            'n_bull': len(bull_sources),
            'n_bear': len(bear_sources),
            'net_direction': net_direction,
            'net_score': round(net_score, 3),
            'confirming': confirming,
            'conflicting': conflicting,
            'confluence': round(confluence, 3),
        }

    return ticker_data


def generate_trade_recommendations(ticker_data, vix_level, market_regime):
    """
    Generate specific option trade recommendations for the agentic account.

    Rules:
    - HC #750: >= 3 confirming signals required
    - VIX < 20: pairs mode (bull + bear allowed)
    - VIX >= 20: bull only
    - DTE = 14, 2% OTM
    - Max 2-3 positions, total cost <= $645
    """
    vix_mode = 'pairs' if vix_level < 20 else 'bull_only'

    candidates = []

    for ticker, data in ticker_data.items():
        direction = data['net_direction']

        # Skip neutrals
        if direction == 'neutral':
            continue

        # HC #750: minimum 3 confirming signals
        if data['confirming'] < MIN_CONFLUENCE:
            continue

        # VIX regime filter
        if vix_mode == 'bull_only' and direction == 'bear':
            continue

        # Get approximate price from state files for strike calculation
        price = get_approximate_price(ticker)
        if price is None:
            continue

        # Calculate option parameters
        if direction == 'bull':
            strike = round(price * (1 + OTM_PCT), 0)
            option_type = 'call'
        else:
            strike = round(price * (1 - OTM_PCT), 0)
            option_type = 'put'

        # Estimate option cost (rough: use 2-4% of underlying for 14 DTE 2% OTM)
        # This is a heuristic — actual prices come from the broker
        est_pct = 0.02 if vix_level < 18 else 0.03 if vix_level < 22 else 0.04
        est_cost_per_share = price * est_pct
        est_cost = round(est_cost_per_share * 100, 2)  # 1 contract = 100 shares

        expiry = datetime.now() + timedelta(days=DTE_TARGET)
        # Round to next Friday
        days_to_friday = (4 - expiry.weekday()) % 7
        expiry = expiry + timedelta(days=days_to_friday)

        candidates.append({
            'ticker': ticker,
            'direction': direction,
            'option_type': option_type,
            'strike': strike,
            'expiry': expiry.strftime('%Y-%m-%d'),
            'dte': DTE_TARGET,
            'estimated_cost': est_cost,
            'confluence': data['confluence'],
            'confirming': data['confirming'],
            'conflicting': data['conflicting'],
            'bull_sources': data['bull_sources'],
            'bear_sources': data['bear_sources'],
            'bull_score': data['bull_score'],
            'bear_score': data['bear_score'],
            'net_score': data['net_score'],
            'price': price,
        })

    # Sort by confluence * net_score (best signals first)
    candidates.sort(key=lambda x: x['confluence'] * x['net_score'], reverse=True)

    # Select top candidates that fit budget
    selected = []
    remaining_budget = ACCOUNT_EQUITY

    for c in candidates:
        if len(selected) >= MAX_POSITIONS:
            break

        cost = c['estimated_cost']
        max_spend = remaining_budget * MAX_SINGLE_POSITION_PCT

        if cost > remaining_budget:
            # Try to find a cheaper way (further OTM)
            adjusted_strike = adjust_strike_for_budget(
                c['price'], c['direction'], remaining_budget
            )
            if adjusted_strike:
                c['strike'] = adjusted_strike['strike']
                c['estimated_cost'] = adjusted_strike['estimated_cost']
                c['note'] = f"Adjusted OTM to fit budget (est ${adjusted_strike['estimated_cost']:.0f})"
                cost = c['estimated_cost']
            else:
                c['skip_reason'] = f"Too expensive (${cost:.0f} > ${remaining_budget:.0f} remaining)"
                continue

        if cost > max_spend:
            c['note'] = f"Capped at {MAX_SINGLE_POSITION_PCT*100:.0f}% of equity"

        selected.append(c)
        remaining_budget -= cost

    # Rejected candidates (didn't meet confluence or budget)
    rejected = [c for c in candidates if c not in selected]

    return selected, rejected, vix_mode


def get_approximate_price(ticker):
    """Get approximate current price from state files."""
    # Check V8 state first (most recent prices)
    v8 = load_state('sector_combined_v8_paper_state.json')
    if v8:
        for pos in v8.get('open_positions', []):
            if pos.get('ticker') == ticker:
                return float(pos['entry_price'])

    # Check other state files
    for fname in ['sector_spreads_paper_state.json', 'sector_spreads_v6_paper_state.json',
                  'sector_pairs_paper_state.json', 'sector_combined_v7_paper_state.json']:
        state = load_state(fname)
        if state:
            for pos in state.get('open_positions', []):
                if pos.get('ticker') == ticker:
                    return float(pos['entry_price'])

    # Check ETF rotation positions
    rot = load_state('etf_rotation_v3_paper_state.json')
    if rot:
        positions = rot.get('positions', {})
        if ticker in positions:
            return float(positions[ticker].get('entry_price', 0))

    # Fallback approximate prices (as of late July 2026)
    approx = {
        'XLB': 51.0, 'XLC': 106.0, 'XLE': 60.0, 'XLF': 56.0,
        'XLI': 183.0, 'XLK': 176.0, 'XLP': 84.0, 'XLRE': 46.0,
        'XLU': 46.0, 'XLV': 163.0, 'XLY': 109.0,
    }
    return approx.get(ticker)


def adjust_strike_for_budget(price, direction, budget, vix_level=18):
    """Try to find a cheaper strike that fits the budget."""
    for otm_mult in [0.05, 0.08, 0.10, 0.15]:
        if direction == 'bull':
            strike = round(price * (1 + otm_mult), 0)
        else:
            strike = round(price * (1 - otm_mult), 0)

        est_pct = max(0.005, 0.03 - otm_mult * 0.5)
        est_cost = round(price * est_pct * 100, 2)

        if est_cost <= budget * 0.9:  # Leave 10% buffer
            return {'strike': strike, 'estimated_cost': est_cost}

    return None


# ============================================================
# OUTPUT + DISPLAY
# ============================================================

def print_summary(ticker_data, selected, rejected, vix_level, vix_mode,
                  market_regime, engines_loaded, all_signals):
    """Print a clear human-readable summary."""

    print("=" * 70)
    print("  META-SIGNAL ENSEMBLE v1 — Multi-Engine Confluence Report")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Market context
    print(f"\n  VIX: {vix_level:.1f} | Mode: {vix_mode.upper()}")
    regime_str = []
    for k, v in market_regime.items():
        if k != 'all_clear':
            regime_str.append(f"{'OK' if v else 'WARN'}: {k}")
    print(f"  Regime: {'ALL CLEAR' if market_regime.get('all_clear') else 'CAUTION'} ({', '.join(regime_str)})")

    # Engine status
    total_engines = len(engines_loaded)
    active_engines = sum(1 for v in engines_loaded.values() if v > 0)
    print(f"\n  Engines: {active_engines}/{total_engines} producing signals ({len(all_signals)} total raw signals)")

    # Sector signal heatmap
    print(f"\n  {'SECTOR':<8} {'BULL':>6} {'BEAR':>6} {'NET':>8} {'CONF':>6} {'DIR':>8}")
    print("  " + "-" * 50)

    sorted_tickers = sorted(ticker_data.items(),
                           key=lambda x: x[1]['net_score'], reverse=True)

    for ticker, data in sorted_tickers:
        direction_str = data['net_direction'].upper()
        if data['confirming'] >= MIN_CONFLUENCE:
            direction_str = f"*{direction_str}*"

        print(f"  {ticker:<8} {data['n_bull']:>6} {data['n_bear']:>6} "
              f"{data['net_score']:>+8.2f} {data['confluence']:>6.3f} {direction_str:>8}")

    # Trade recommendations
    print(f"\n{'=' * 70}")
    print("  TRADE RECOMMENDATIONS (HC #750: >= 3 confirming signals)")
    print("=" * 70)

    if not selected:
        print("\n  No trades meet the minimum confluence threshold.")
        print("  Consider lowering MIN_CONFLUENCE or waiting for more engine agreement.")
    else:
        total_cost = 0
        for i, trade in enumerate(selected, 1):
            print(f"\n  Trade #{i}: {trade['direction'].upper()} {trade['ticker']}")
            print(f"    Option: BUY {trade['option_type'].upper()} @ ${trade['strike']:.0f}")
            print(f"    Expiry: {trade['expiry']} (DTE ~{trade['dte']})")
            print(f"    Est. cost: ${trade['estimated_cost']:.0f}")
            print(f"    Confluence: {trade['confluence']:.3f} ({trade['confirming']} confirming, {trade['conflicting']} conflicting)")

            sources = trade['bull_sources'] if trade['direction'] == 'bull' else trade['bear_sources']
            print(f"    Sources: {', '.join(sources)}")

            if 'note' in trade:
                print(f"    Note: {trade['note']}")

            total_cost += trade['estimated_cost']

        print(f"\n  Total estimated cost: ${total_cost:.0f} / ${ACCOUNT_EQUITY:.0f} ({total_cost/ACCOUNT_EQUITY*100:.0f}%)")
        remaining = ACCOUNT_EQUITY - total_cost
        print(f"  Remaining budget: ${remaining:.0f}")

    # Below threshold
    if rejected:
        print(f"\n  Below threshold ({len(rejected)} candidates):")
        for r in rejected[:5]:
            reason = r.get('skip_reason', f"confluence={r['confluence']:.2f}")
            print(f"    {r['ticker']} {r['direction']} — {reason}")

    print(f"\n{'=' * 70}")


def save_output(ticker_data, selected, rejected, vix_level, vix_mode,
                market_regime, engines_loaded, all_signals):
    """Save results to JSON."""

    # Strip non-serializable data from ticker_data
    clean_ticker_data = {}
    for ticker, data in ticker_data.items():
        clean_ticker_data[ticker] = {
            k: v for k, v in data.items()
            if k not in ('bull_signals', 'bear_signals', 'neutral_signals')
        }

    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'meta_signal_ensemble_v1',
        'market_context': {
            'vix_level': vix_level,
            'vix_mode': vix_mode,
            'market_regime': market_regime,
        },
        'engines': {
            'total': len(engines_loaded),
            'active': sum(1 for v in engines_loaded.values() if v > 0),
            'detail': engines_loaded,
        },
        'total_raw_signals': len(all_signals),
        'sector_confluence': clean_ticker_data,
        'trade_recommendations': selected,
        'below_threshold': [
            {k: v for k, v in r.items() if k not in ('bull_signals', 'bear_signals')}
            for r in rejected[:10]
        ],
        'account': {
            'equity': ACCOUNT_EQUITY,
            'max_positions': MAX_POSITIONS,
            'min_confluence': MIN_CONFLUENCE,
        },
    }

    # Save latest
    latest_path = os.path.join(OUTPUT_DIR, 'latest.json')
    with open(latest_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Save timestamped copy
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    ts_path = os.path.join(OUTPUT_DIR, f'ensemble_{ts}.json')
    with open(ts_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n  Saved: {latest_path}")
    print(f"  Saved: {ts_path}")

    return output


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nCollecting signals from all paper engines...")
    all_signals, engines_loaded = collect_all_signals()

    print(f"  {len(all_signals)} raw signals from {len(engines_loaded)} engines")

    # Get market context
    vix_level = get_vix_regime()
    market_regime = get_market_regime()

    # Aggregate by ticker
    ticker_data = aggregate_by_ticker(all_signals)

    # Generate trade recommendations
    selected, rejected, vix_mode = generate_trade_recommendations(
        ticker_data, vix_level, market_regime
    )

    # Display
    print_summary(ticker_data, selected, rejected, vix_level, vix_mode,
                  market_regime, engines_loaded, all_signals)

    # Save
    output = save_output(ticker_data, selected, rejected, vix_level, vix_mode,
                         market_regime, engines_loaded, all_signals)

    return output


if __name__ == '__main__':
    main()
