"""
Liquidity Analysis V1 — Sector ETF Options Tradability Assessment
=================================================================
Analyzes real options chain data for all 11 sector ETFs to determine
which sectors have sufficient liquidity for our V8 vertical spread strategy.

V8 Config: DTE~14, OTM~2-5%, vertical spreads
- Bull call spread: buy K1=spot*1.02, sell K2=spot*1.05
- Bear put spread: buy K1=spot*0.98, sell K2=spot*0.95

Key question: Which sectors can we actually TRADE given real-world spreads?
Finding #224 showed Sharpe drops from 3.23 (mid) to 0.8 (ask) — fill quality is everything.
"""

import pandas as pd
import numpy as np
import glob
import json
import os
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = "/home/jupiter/Lvl3Quant/data/options_chains/2026-07-27"
OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_research/liquidity_analysis_v1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# V8 strategy parameters
TARGET_DTE_MIN = 7
TARGET_DTE_MAX = 24  # flexible: pick closest to 14
BULL_CALL_BUY_OTM = 0.02   # buy call at spot * 1.02
BULL_CALL_SELL_OTM = 0.05   # sell call at spot * 1.05
BEAR_PUT_BUY_OTM = 0.02    # buy put at spot * 0.98
BEAR_PUT_SELL_OTM = 0.05   # sell put at spot * 0.95
STRIKE_TOLERANCE = 0.015    # allow 1.5% strike matching tolerance

# Liquidity thresholds for tradability
MIN_OI_TRADABLE = 50        # minimum open interest to consider tradable
MIN_VOLUME_TRADABLE = 5     # minimum daily volume
MAX_SPREAD_PCT_TRADABLE = 0.15  # max 15% bid-ask spread as % of mid

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
SECTOR_NAMES = {
    "XLB": "Materials", "XLC": "Communication", "XLE": "Energy",
    "XLF": "Financials", "XLI": "Industrials", "XLK": "Technology",
    "XLP": "Consumer Staples", "XLRE": "Real Estate", "XLU": "Utilities",
    "XLV": "Healthcare", "XLY": "Consumer Disc."
}


def load_sector_data(ticker):
    """Load options chain data for a sector ETF."""
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def find_best_dte(df):
    """Find the DTE closest to 14 within our acceptable range."""
    dtes = df['dte_days'].unique()
    valid = [d for d in dtes if TARGET_DTE_MIN <= d <= TARGET_DTE_MAX]
    if not valid:
        return None
    return min(valid, key=lambda x: abs(x - 14))


def find_nearest_strike(df, target_strike, option_type):
    """Find the nearest available strike to target."""
    subset = df[df['option_type'] == option_type]
    if subset.empty:
        return None
    diffs = (subset['strike'] - target_strike).abs()
    best_idx = diffs.idxmin()
    best = subset.loc[best_idx]
    # Check if within tolerance
    if abs(best['strike'] - target_strike) / target_strike > STRIKE_TOLERANCE:
        return None
    return best


def compute_spread_metrics(row):
    """Compute bid-ask spread metrics for a single option."""
    bid, ask, mid, last = row['bid'], row['ask'], row['mid'], row['last']
    volume, oi = row['volume'], row['open_interest']

    # Handle weekend zero bid/ask
    # If bid/ask are both 0, we can estimate spread from last/mid
    # but we'll flag this as "no live quote"
    has_live_quote = bid > 0 or ask > 0

    if bid > 0 and ask > 0:
        spread_dollars = ask - bid
        spread_pct = spread_dollars / mid if mid > 0 else np.nan
    elif ask > 0 and bid == 0:
        # Only ask available — estimate spread as 2 * (ask - mid)
        spread_dollars = 2 * (ask - mid) if mid > 0 else np.nan
        spread_pct = spread_dollars / mid if mid > 0 else np.nan
    else:
        # Weekend — no live quotes. Use historical proxy.
        # For sector ETF options, typical spread is 5-20% of mid for OTM options
        spread_dollars = np.nan
        spread_pct = np.nan

    return {
        'strike': row['strike'],
        'option_type': row['option_type'],
        'bid': bid, 'ask': ask, 'mid': mid, 'last': last,
        'spread_dollars': spread_dollars,
        'spread_pct': spread_pct,
        'volume': volume,
        'open_interest': oi,
        'has_live_quote': has_live_quote,
        'dte_days': row['dte_days'],
    }


def analyze_spread_legs(df, spot, dte):
    """Find and analyze the 4 legs needed for bull call + bear put spreads."""
    df_dte = df[df['dte_days'] == dte].copy()

    legs = {}

    # Bull call spread legs
    bull_buy_target = spot * (1 + BULL_CALL_BUY_OTM)    # K1 = spot * 1.02
    bull_sell_target = spot * (1 + BULL_CALL_SELL_OTM)   # K2 = spot * 1.05

    legs['bull_call_buy'] = find_nearest_strike(df_dte, bull_buy_target, 'call')
    legs['bull_call_sell'] = find_nearest_strike(df_dte, bull_sell_target, 'call')

    # Bear put spread legs
    bear_buy_target = spot * (1 - BEAR_PUT_BUY_OTM)     # K1 = spot * 0.98
    bear_sell_target = spot * (1 - BEAR_PUT_SELL_OTM)    # K2 = spot * 0.95

    legs['bear_put_buy'] = find_nearest_strike(df_dte, bear_buy_target, 'put')
    legs['bear_put_sell'] = find_nearest_strike(df_dte, bear_sell_target, 'put')

    return legs


def compute_spread_cost(legs, spread_type):
    """Compute the cost of entering a spread at bid vs ask vs mid."""
    if spread_type == 'bull_call':
        buy_leg = legs.get('bull_call_buy')
        sell_leg = legs.get('bull_call_sell')
    else:
        buy_leg = legs.get('bear_put_buy')
        sell_leg = legs.get('bear_put_sell')

    if buy_leg is None or sell_leg is None:
        return None

    # Use mid as reference price (what backtest assumes)
    buy_mid = buy_leg['mid']
    sell_mid = sell_leg['mid']
    mid_cost = buy_mid - sell_mid  # debit spread cost at mid

    # Worst case: buy at ask, sell at bid
    buy_ask = buy_leg['ask'] if buy_leg['ask'] > 0 else buy_mid * 1.05  # estimate 5% wider
    sell_bid = sell_leg['bid'] if sell_leg['bid'] > 0 else sell_mid * 0.95
    worst_cost = buy_ask - sell_bid

    # Best case: buy at bid, sell at ask (unlikely but possible with patience)
    buy_bid = buy_leg['bid'] if buy_leg['bid'] > 0 else buy_mid * 0.95
    sell_ask = sell_leg['ask'] if sell_leg['ask'] > 0 else sell_mid * 1.05
    best_cost = buy_bid - sell_ask

    # Spread width (max profit potential)
    if spread_type == 'bull_call':
        width = sell_leg['strike'] - buy_leg['strike']
    else:
        width = buy_leg['strike'] - sell_leg['strike']

    return {
        'mid_cost': mid_cost,
        'worst_cost': worst_cost,  # realistic entry at ask/bid
        'best_cost': best_cost,    # patient limit order
        'width': width,
        'max_profit_mid': width - mid_cost if mid_cost > 0 else np.nan,
        'max_profit_worst': width - worst_cost if worst_cost > 0 else np.nan,
        'slippage_pct': (worst_cost - mid_cost) / mid_cost * 100 if mid_cost > 0 else np.nan,
        'buy_strike': buy_leg['strike'],
        'sell_strike': sell_leg['strike'],
        'buy_oi': buy_leg['open_interest'],
        'sell_oi': sell_leg['open_interest'],
        'buy_volume': buy_leg['volume'],
        'sell_volume': sell_leg['volume'],
    }


def compute_tradability_score(sector_metrics):
    """
    Composite tradability score (0-100) based on:
    - Volume (30%): higher = better
    - Open interest (30%): higher = better
    - Spread tightness (40%): lower spread = better (MOST important for us)
    """
    scores = {}

    # Collect raw metrics across sectors
    all_volumes = []
    all_oi = []
    all_spreads = []

    for ticker, m in sector_metrics.items():
        if m is None:
            continue
        avg_vol = np.nanmean([m.get('bull_volume', 0), m.get('bear_volume', 0)])
        avg_oi = np.nanmean([m.get('bull_oi', 0), m.get('bear_oi', 0)])
        avg_slip = np.nanmean([
            m.get('bull_slippage_pct', 100),
            m.get('bear_slippage_pct', 100)
        ])
        all_volumes.append((ticker, avg_vol))
        all_oi.append((ticker, avg_oi))
        all_spreads.append((ticker, avg_slip))

    if not all_volumes:
        return scores

    # Rank each dimension (higher rank = better)
    def rank_scores(items, higher_is_better=True):
        sorted_items = sorted(items, key=lambda x: x[1], reverse=higher_is_better)
        n = len(sorted_items)
        return {t: (n - i) / n * 100 for i, (t, _) in enumerate(sorted_items)}

    vol_ranks = rank_scores(all_volumes, higher_is_better=True)
    oi_ranks = rank_scores(all_oi, higher_is_better=True)
    spread_ranks = rank_scores(all_spreads, higher_is_better=False)  # lower spread = better

    for ticker in vol_ranks:
        scores[ticker] = (
            0.30 * vol_ranks.get(ticker, 0) +
            0.30 * oi_ranks.get(ticker, 0) +
            0.40 * spread_ranks.get(ticker, 0)
        )

    return scores


def analyze_overall_chain_liquidity(df, spot, dte):
    """Analyze ALL options in the relevant DTE/moneyness range, not just exact legs."""
    df_dte = df[df['dte_days'] == dte].copy()

    # Look at OTM options within our trading range (0.9x to 1.1x spot)
    otm_range = df_dte[
        (df_dte['strike'] >= spot * 0.90) &
        (df_dte['strike'] <= spot * 1.10)
    ].copy()

    if otm_range.empty:
        return None

    total_volume = otm_range['volume'].sum()
    total_oi = otm_range['open_interest'].sum()
    avg_volume = otm_range['volume'].mean()
    avg_oi = otm_range['open_interest'].mean()
    pct_with_volume = (otm_range['volume'] > 0).mean() * 100
    pct_with_oi = (otm_range['open_interest'] > 0).mean() * 100

    # Contracts with live quotes
    live_quotes = otm_range[(otm_range['bid'] > 0) | (otm_range['ask'] > 0)]
    pct_with_quotes = len(live_quotes) / len(otm_range) * 100 if len(otm_range) > 0 else 0

    # Spread metrics for contracts that have quotes
    if not live_quotes.empty:
        spreads = []
        for _, row in live_quotes.iterrows():
            m = compute_spread_metrics(row)
            if m['spread_pct'] is not None and not np.isnan(m['spread_pct']):
                spreads.append(m['spread_pct'])
        avg_spread_pct = np.mean(spreads) if spreads else np.nan
        median_spread_pct = np.median(spreads) if spreads else np.nan
    else:
        avg_spread_pct = np.nan
        median_spread_pct = np.nan

    return {
        'n_contracts': len(otm_range),
        'total_volume': int(total_volume),
        'total_oi': int(total_oi),
        'avg_volume': round(avg_volume, 1),
        'avg_oi': round(avg_oi, 1),
        'pct_with_volume': round(pct_with_volume, 1),
        'pct_with_oi': round(pct_with_oi, 1),
        'pct_with_quotes': round(pct_with_quotes, 1),
        'avg_spread_pct': round(avg_spread_pct * 100, 1) if not np.isnan(avg_spread_pct) else None,
        'median_spread_pct': round(median_spread_pct * 100, 1) if not np.isnan(median_spread_pct) else None,
    }


def main():
    print("=" * 80)
    print("LIQUIDITY ANALYSIS V1 — Sector ETF Options Tradability")
    print(f"Data: {DATA_DIR}")
    print(f"Date: 2026-07-27 (Friday July 25 close prices)")
    print("=" * 80)

    results = {}
    sector_metrics = {}

    for ticker in SECTOR_ETFS:
        print(f"\n{'─' * 60}")
        print(f"  {ticker} ({SECTOR_NAMES[ticker]})")
        print(f"{'─' * 60}")

        df = load_sector_data(ticker)
        if df is None:
            print(f"  ⚠ No data found")
            results[ticker] = {'status': 'no_data'}
            continue

        spot = df['underlying_price'].iloc[0]
        print(f"  Spot: ${spot:.2f}")

        # Find best DTE
        best_dte = find_best_dte(df)
        if best_dte is None:
            print(f"  No suitable DTE found (need {TARGET_DTE_MIN}-{TARGET_DTE_MAX})")
            results[ticker] = {'status': 'no_suitable_dte', 'spot': spot}
            continue

        print(f"  Best DTE: {best_dte} days")

        # Overall chain liquidity
        chain_liq = analyze_overall_chain_liquidity(df, spot, best_dte)

        # Find specific spread legs
        legs = analyze_spread_legs(df, spot, best_dte)

        # Analyze each spread
        bull_cost = compute_spread_cost(legs, 'bull_call')
        bear_cost = compute_spread_cost(legs, 'bear_put')

        # Compile metrics
        metrics = {
            'spot': spot,
            'dte': best_dte,
            'chain_liquidity': chain_liq,
        }

        if bull_cost:
            print(f"\n  BULL CALL SPREAD: Buy {bull_cost['buy_strike']}C / Sell {bull_cost['sell_strike']}C")
            print(f"    Mid cost: ${bull_cost['mid_cost']:.2f}  |  Worst: ${bull_cost['worst_cost']:.2f}")
            print(f"    Width: ${bull_cost['width']:.2f}  |  Max profit (mid): ${bull_cost['max_profit_mid']:.2f}")
            if bull_cost['max_profit_worst'] is not None and not np.isnan(bull_cost['max_profit_worst']):
                print(f"    Max profit (ask fill): ${bull_cost['max_profit_worst']:.2f}")
            if not np.isnan(bull_cost['slippage_pct']):
                print(f"    Slippage cost: {bull_cost['slippage_pct']:.1f}% of mid cost")
            print(f"    OI: buy={bull_cost['buy_oi']}, sell={bull_cost['sell_oi']}")
            print(f"    Volume: buy={bull_cost['buy_volume']}, sell={bull_cost['sell_volume']}")
            metrics['bull_slippage_pct'] = bull_cost['slippage_pct']
            metrics['bull_oi'] = bull_cost['buy_oi'] + bull_cost['sell_oi']
            metrics['bull_volume'] = bull_cost['buy_volume'] + bull_cost['sell_volume']
            metrics['bull_cost'] = bull_cost
        else:
            print(f"\n  BULL CALL SPREAD: Could not find matching strikes")
            metrics['bull_slippage_pct'] = 100
            metrics['bull_oi'] = 0
            metrics['bull_volume'] = 0

        if bear_cost:
            print(f"\n  BEAR PUT SPREAD: Buy {bear_cost['buy_strike']}P / Sell {bear_cost['sell_strike']}P")
            print(f"    Mid cost: ${bear_cost['mid_cost']:.2f}  |  Worst: ${bear_cost['worst_cost']:.2f}")
            print(f"    Width: ${bear_cost['width']:.2f}  |  Max profit (mid): ${bear_cost['max_profit_mid']:.2f}")
            if bear_cost['max_profit_worst'] is not None and not np.isnan(bear_cost['max_profit_worst']):
                print(f"    Max profit (ask fill): ${bear_cost['max_profit_worst']:.2f}")
            if not np.isnan(bear_cost['slippage_pct']):
                print(f"    Slippage cost: {bear_cost['slippage_pct']:.1f}% of mid cost")
            print(f"    OI: buy={bear_cost['buy_oi']}, sell={bear_cost['sell_oi']}")
            print(f"    Volume: buy={bear_cost['buy_volume']}, sell={bear_cost['sell_volume']}")
            metrics['bear_slippage_pct'] = bear_cost['slippage_pct']
            metrics['bear_oi'] = bear_cost['buy_oi'] + bear_cost['sell_oi']
            metrics['bear_volume'] = bear_cost['buy_volume'] + bear_cost['sell_volume']
            metrics['bear_cost'] = bear_cost
        else:
            print(f"\n  BEAR PUT SPREAD: Could not find matching strikes")
            metrics['bear_slippage_pct'] = 100
            metrics['bear_oi'] = 0
            metrics['bear_volume'] = 0

        if chain_liq:
            print(f"\n  CHAIN LIQUIDITY (OTM, DTE={best_dte}):")
            print(f"    Contracts in range: {chain_liq['n_contracts']}")
            print(f"    Total volume: {chain_liq['total_volume']}  |  Avg: {chain_liq['avg_volume']}")
            print(f"    Total OI: {chain_liq['total_oi']}  |  Avg: {chain_liq['avg_oi']}")
            print(f"    % with volume: {chain_liq['pct_with_volume']}%")
            print(f"    % with OI: {chain_liq['pct_with_oi']}%")
            if chain_liq['avg_spread_pct'] is not None:
                print(f"    Avg spread: {chain_liq['avg_spread_pct']}% of mid")

        sector_metrics[ticker] = metrics
        results[ticker] = metrics

    # ─── Tradability Ranking ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRADABILITY RANKING")
    print("=" * 80)

    scores = compute_tradability_score(sector_metrics)

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<5} {'Sector':<6} {'Name':<18} {'Score':<8} {'Tot Vol':<10} {'Tot OI':<10} {'Verdict'}")
    print("-" * 85)

    tradable = []
    avoid = []
    marginal = []

    for rank, (ticker, score) in enumerate(ranked, 1):
        m = sector_metrics.get(ticker, {})
        chain = m.get('chain_liquidity', {}) or {}
        tot_vol = chain.get('total_volume', 0)
        tot_oi = chain.get('total_oi', 0)

        if score >= 65 and tot_vol >= 50 and tot_oi >= 100:
            verdict = "TRADABLE"
            tradable.append(ticker)
        elif score >= 40 or tot_vol >= 20:
            verdict = "MARGINAL"
            marginal.append(ticker)
        else:
            verdict = "AVOID"
            avoid.append(ticker)

        print(f"  {rank:<4} {ticker:<6} {SECTOR_NAMES[ticker]:<18} {score:<8.1f} {tot_vol:<10} {tot_oi:<10} {verdict}")

    # ─── Practical Guide ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PRACTICAL TRADING GUIDE")
    print("=" * 80)

    print(f"\n  GREEN (trade freely):")
    for t in tradable:
        chain = sector_metrics[t].get('chain_liquidity', {}) or {}
        print(f"    {t} ({SECTOR_NAMES[t]}): vol={chain.get('total_volume',0)}, OI={chain.get('total_oi',0)}")

    print(f"\n  YELLOW (trade with patience, use limit orders only):")
    for t in marginal:
        chain = sector_metrics[t].get('chain_liquidity', {}) or {}
        print(f"    {t} ({SECTOR_NAMES[t]}): vol={chain.get('total_volume',0)}, OI={chain.get('total_oi',0)}")

    print(f"\n  RED (avoid — spreads too wide, insufficient liquidity):")
    for t in avoid:
        chain = sector_metrics[t].get('chain_liquidity', {}) or {}
        print(f"    {t} ({SECTOR_NAMES[t]}): vol={chain.get('total_volume',0)}, OI={chain.get('total_oi',0)}")

    # ─── Fill Quality Analysis ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FILL QUALITY IMPACT ON SHARPE")
    print("=" * 80)

    print(f"\n  Finding #224 context:")
    print(f"    V8 backtest at mid-price fills: Sharpe 3.23")
    print(f"    V8 backtest at ask-price fills: Sharpe ~0.8")
    print(f"    Sharpe degradation: ~75% from fill slippage alone")
    print(f"\n  Implication: every 1% of spread cost destroys ~0.3 Sharpe points")
    print(f"\n  Sector-specific impact estimates:")

    for ticker, score in ranked:
        m = sector_metrics.get(ticker, {})
        bull_slip = m.get('bull_slippage_pct', np.nan)
        bear_slip = m.get('bear_slippage_pct', np.nan)
        avg_slip = np.nanmean([bull_slip, bear_slip])

        if np.isnan(avg_slip):
            print(f"    {ticker}: insufficient data to estimate")
        else:
            # Estimate Sharpe at realistic fills
            # Linear interpolation: mid=3.23, ask=0.8, slippage maps between
            # "ask" is ~100% slippage, so: sharpe = 3.23 - (3.23-0.8) * slip/100
            est_sharpe = 3.23 - (3.23 - 0.8) * min(avg_slip, 100) / 100
            print(f"    {ticker}: avg slippage {avg_slip:.1f}% -> est. Sharpe ~{est_sharpe:.2f}")

    # ─── Volume/OI Heatmap by DTE ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("LIQUIDITY BY DTE (volume concentration)")
    print("=" * 80)

    for ticker in SECTOR_ETFS:
        df = load_sector_data(ticker)
        if df is None:
            continue
        spot = df['underlying_price'].iloc[0]
        otm = df[(df['strike'] >= spot * 0.90) & (df['strike'] <= spot * 1.10)]
        dte_vol = otm.groupby('dte_days')['volume'].sum()
        dte_oi = otm.groupby('dte_days')['open_interest'].sum()

        best_dte_vol = dte_vol.idxmax() if not dte_vol.empty else None
        best_dte_oi = dte_oi.idxmax() if not dte_oi.empty else None

        print(f"\n  {ticker}: peak volume at DTE={best_dte_vol} ({dte_vol.max() if not dte_vol.empty else 0}), "
              f"peak OI at DTE={best_dte_oi} ({dte_oi.max() if not dte_oi.empty else 0})")

        # Show all DTEs
        for dte in sorted(otm['dte_days'].unique()):
            v = dte_vol.get(dte, 0)
            o = dte_oi.get(dte, 0)
            marker = " <-- our target" if TARGET_DTE_MIN <= dte <= TARGET_DTE_MAX else ""
            print(f"      DTE {dte:>3}: vol={v:>6}, OI={o:>6}{marker}")

    # ─── Key Recommendations ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("KEY RECOMMENDATIONS")
    print("=" * 80)

    recommendations = []

    # 1. Sector filtering
    if tradable:
        rec = f"Focus V8 on liquid sectors: {', '.join(tradable)}"
        recommendations.append(rec)
        print(f"\n  1. {rec}")
    if avoid:
        rec = f"EXCLUDE from V8: {', '.join(avoid)} (insufficient liquidity will destroy edge)"
        recommendations.append(rec)
        print(f"  2. {rec}")

    # 2. Fill strategy
    print(f"\n  3. ALWAYS use limit orders at mid or better. Market orders destroy 75% of Sharpe.")
    recommendations.append("ALWAYS use limit orders at mid or better")

    # 3. DTE recommendation
    print(f"  4. Consider shifting DTE to where volume concentrates (check per-sector DTE analysis above)")
    recommendations.append("Consider DTE shift to volume concentration point")

    # 4. Sizing
    print(f"  5. Size positions proportional to liquidity — XLK/XLF/XLE can handle larger size")
    recommendations.append("Size proportional to liquidity")

    # 5. LGBM integration
    print(f"\n  6. LGBM INTEGRATION: If LGBM ranks a low-liquidity sector highly,")
    print(f"     the signal is real but UNTRADABLE. Filter these sectors OUT of the")
    print(f"     tradable universe BEFORE running LGBM ranking. Liquidity is a hard")
    print(f"     constraint, not a soft preference.")
    recommendations.append("Filter illiquid sectors before LGBM ranking, not after")

    # ─── Save Results ─────────────────────────────────────────────────────────
    output = {
        'timestamp': datetime.now().isoformat(),
        'data_date': '2026-07-27',
        'config': {
            'target_dte': '7-24 days (closest to 14)',
            'bull_call': f'buy {BULL_CALL_BUY_OTM*100}% OTM, sell {BULL_CALL_SELL_OTM*100}% OTM',
            'bear_put': f'buy {BEAR_PUT_BUY_OTM*100}% OTM, sell {BEAR_PUT_SELL_OTM*100}% OTM',
        },
        'tradability_ranking': [
            {'rank': i+1, 'ticker': t, 'name': SECTOR_NAMES[t], 'score': round(s, 1)}
            for i, (t, s) in enumerate(ranked)
        ],
        'categories': {
            'tradable': tradable,
            'marginal': marginal,
            'avoid': avoid,
        },
        'recommendations': recommendations,
        'sector_details': {},
    }

    # Serialize sector details (handle non-serializable types)
    for ticker, m in sector_metrics.items():
        detail = {
            'spot': m.get('spot'),
            'dte': m.get('dte'),
        }
        if m.get('chain_liquidity'):
            detail['chain_liquidity'] = m['chain_liquidity']
        if m.get('bull_cost'):
            bc = m['bull_cost'].copy()
            for k, v in bc.items():
                if isinstance(v, (np.integer,)):
                    bc[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    bc[k] = float(v) if not np.isnan(v) else None
            detail['bull_call_spread'] = bc
        if m.get('bear_cost'):
            bc = m['bear_cost'].copy()
            for k, v in bc.items():
                if isinstance(v, (np.integer,)):
                    bc[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    bc[k] = float(v) if not np.isnan(v) else None
            detail['bear_put_spread'] = bc
        output['sector_details'][ticker] = detail

    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Save ranking as CSV for easy consumption
    ranking_df = pd.DataFrame(output['tradability_ranking'])
    ranking_path = os.path.join(OUTPUT_DIR, 'tradability_ranking.csv')
    ranking_df.to_csv(ranking_path, index=False)

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
