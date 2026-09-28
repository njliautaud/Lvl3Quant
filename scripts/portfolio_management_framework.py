#!/usr/bin/env python3
"""
Portfolio Management Framework — Live Signal Generator
=======================================================
Unified daily portfolio allocation system for larger accounts (Track 2).

Computes signals and recommended allocations across 5 portfolio models:
  1. Equal Weight Top-K (momentum ranked)
  2. Risk Parity (inverse-vol)
  3. Momentum + Risk Parity (hybrid)
  4. Regime-Adaptive (bull/bear/transition)
  5. Core-Satellite (60/40 core-satellite)

Signal Modules:
  - 17-feature momentum scoring
  - Inverse-vol risk parity weights
  - SPY 200-SMA regime detection
  - Cross-asset signals (bond/equity, VIX, yield curve, credit)
  - Mean reversion overlay (oversold bounce candidates)

Usage:
  python3 scripts/portfolio_management_framework.py [--capital 50000] [--top-k 5]
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════

STATE_DIR = '/home/jupiter/Lvl3Quant/state'
STATE_FILE = os.path.join(STATE_DIR, 'portfolio_framework.json')
os.makedirs(STATE_DIR, exist_ok=True)

# Asset Universe
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
FACTOR_ETFS = ['MTUM', 'VLUE', 'QUAL', 'USMV']
BOND_ETFS = ['TLT', 'IEF', 'SHY', 'AGG']
COMMODITY_ETFS = ['GLD', 'USO']
INTL_ETFS = ['EFA', 'EEM']
BROAD_ETFS = ['SPY', 'QQQ', 'IWM']

ALL_ETFS = SECTOR_ETFS + FACTOR_ETFS + BOND_ETFS + COMMODITY_ETFS + INTL_ETFS + BROAD_ETFS

# Cross-asset signal tickers (not in portfolio but used for signals)
SIGNAL_TICKERS = ['^VIX', 'HYG', 'LQD']

DEFENSIVE_ETFS = {'TLT', 'IEF', 'SHY', 'AGG', 'GLD', 'XLU', 'XLP', 'USMV'}
CORE_ETFS = ['SPY', 'AGG']


# ════════════════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ════════════════════════════════════════════════════════════════════

def download_data(lookback_days=504):
    """Download price data for all ETFs + signal tickers."""
    start = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    all_tickers = ALL_ETFS + SIGNAL_TICKERS

    print(f"Downloading {len(all_tickers)} tickers from {start}...")
    sys.stdout.flush()

    raw = yf.download(all_tickers, start=start, progress=False)
    if hasattr(raw.index, 'tz') and raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    close = close.ffill().dropna(how='all')

    # Rename VIX column
    if '^VIX' in close.columns:
        close = close.rename(columns={'^VIX': 'VIX'})

    print(f"  {len(close)} trading days, {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")
    return close


# ════════════════════════════════════════════════════════════════════
# SIGNAL MODULE 1: MOMENTUM RANKING (17 features)
# ════════════════════════════════════════════════════════════════════

def compute_momentum_scores(close, etfs):
    """
    17-feature momentum composite score for each ETF.
    Features span multiple timeframes and signal types.
    Returns DataFrame with scores and ranks.
    """
    returns = close[etfs].pct_change()
    scores = pd.DataFrame(index=etfs, columns=['score'], dtype=float)
    feature_dict = {}

    for etf in etfs:
        if etf not in close.columns or close[etf].isna().all():
            scores.loc[etf, 'score'] = np.nan
            continue

        px = close[etf].dropna()
        if len(px) < 252:
            scores.loc[etf, 'score'] = np.nan
            continue

        ret = px.pct_change()
        features = {}

        # Price momentum (5 features)
        features['ret_5d'] = (px.iloc[-1] / px.iloc[-6] - 1) if len(px) >= 6 else 0
        features['ret_21d'] = (px.iloc[-1] / px.iloc[-22] - 1) if len(px) >= 22 else 0
        features['ret_63d'] = (px.iloc[-1] / px.iloc[-64] - 1) if len(px) >= 64 else 0
        features['ret_126d'] = (px.iloc[-1] / px.iloc[-127] - 1) if len(px) >= 127 else 0
        features['ret_252d'] = (px.iloc[-1] / px.iloc[-253] - 1) if len(px) >= 253 else 0

        # Risk-adjusted momentum (3 features)
        vol_21 = ret.iloc[-21:].std() * np.sqrt(252)
        vol_63 = ret.iloc[-63:].std() * np.sqrt(252)
        features['sharpe_21d'] = (features['ret_21d'] * 252 / 21) / vol_21 if vol_21 > 0 else 0
        features['sharpe_63d'] = (features['ret_63d'] * 252 / 63) / vol_63 if vol_63 > 0 else 0
        features['ret_vol_ratio'] = features['ret_126d'] / vol_63 if vol_63 > 0 else 0

        # Trend features (4 features)
        sma_50 = px.iloc[-50:].mean()
        sma_200 = px.iloc[-200:].mean() if len(px) >= 200 else px.mean()
        features['above_sma50'] = 1.0 if px.iloc[-1] > sma_50 else 0.0
        features['above_sma200'] = 1.0 if px.iloc[-1] > sma_200 else 0.0
        features['sma50_slope'] = (sma_50 / px.iloc[-55:-5].mean() - 1) if len(px) >= 55 else 0
        features['dist_from_high'] = px.iloc[-1] / px.iloc[-252:].max() - 1

        # Mean reversion features (3 features)
        features['rsi_14'] = _compute_rsi(px, 14)
        bb_mid = px.iloc[-20:].mean()
        bb_std = px.iloc[-20:].std()
        features['bb_position'] = (px.iloc[-1] - bb_mid) / (2 * bb_std) if bb_std > 0 else 0
        features['z_score_21d'] = (px.iloc[-1] - bb_mid) / bb_std if bb_std > 0 else 0

        # Volume/volatility features (2 features)
        features['vol_ratio'] = vol_21 / vol_63 if vol_63 > 0 else 1.0
        features['vol_21d'] = vol_21

        feature_dict[etf] = features

        # Composite score: weighted sum emphasizing medium-term momentum
        weights = {
            'ret_5d': 0.03, 'ret_21d': 0.08, 'ret_63d': 0.15,
            'ret_126d': 0.12, 'ret_252d': 0.07,
            'sharpe_21d': 0.05, 'sharpe_63d': 0.10, 'ret_vol_ratio': 0.08,
            'above_sma50': 0.05, 'above_sma200': 0.05, 'sma50_slope': 0.05,
            'dist_from_high': 0.05,
            'rsi_14': -0.02,  # penalize overbought
            'bb_position': -0.02,
            'z_score_21d': -0.02,
            'vol_ratio': -0.03,  # penalize rising vol
            'vol_21d': -0.05,  # penalize high vol
        }

        # Normalize features to z-scores within reasonable bounds
        raw_score = sum(features[k] * weights.get(k, 0) for k in features)
        scores.loc[etf, 'score'] = raw_score

    # Rank (1 = best)
    scores['rank'] = scores['score'].rank(ascending=False).astype(int)
    scores = scores.sort_values('rank')

    return scores, feature_dict


def _compute_rsi(prices, period=14):
    """Standard RSI calculation."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.iloc[-period:].mean()
    avg_loss = loss.iloc[-period:].mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ════════════════════════════════════════════════════════════════════
# SIGNAL MODULE 2: RISK PARITY WEIGHTS
# ════════════════════════════════════════════════════════════════════

def compute_risk_parity_weights(close, etfs, vol_window=21):
    """Inverse-volatility weighting. Lower vol -> higher weight."""
    returns = close[etfs].pct_change()
    vols = returns.iloc[-vol_window:].std() * np.sqrt(252)

    # Remove NaN/zero vol
    vols = vols.replace(0, np.nan).dropna()
    inv_vol = 1.0 / vols
    weights = inv_vol / inv_vol.sum()

    return weights.to_dict(), vols.to_dict()


# ════════════════════════════════════════════════════════════════════
# SIGNAL MODULE 3: REGIME DETECTION
# ════════════════════════════════════════════════════════════════════

def detect_regime(close):
    """
    Bull/Bear/Transition based on SPY vs 200-SMA.
    Returns regime string and details.
    """
    spy = close['SPY'].dropna()
    if len(spy) < 200:
        return 'unknown', {}

    sma_200 = spy.iloc[-200:].mean()
    current = spy.iloc[-1]
    pct_from_sma = (current / sma_200 - 1) * 100

    if pct_from_sma > 2.0:
        regime = 'bull'
    elif pct_from_sma < -2.0:
        regime = 'bear'
    else:
        regime = 'transition'

    # Additional regime context
    sma_50 = spy.iloc[-50:].mean()
    ret_21d = (spy.iloc[-1] / spy.iloc[-22] - 1) * 100 if len(spy) >= 22 else 0

    details = {
        'spy_price': round(float(current), 2),
        'sma_200': round(float(sma_200), 2),
        'sma_50': round(float(sma_50), 2),
        'pct_from_sma200': round(pct_from_sma, 2),
        'spy_21d_return_pct': round(float(ret_21d), 2),
        'golden_cross': bool(sma_50 > sma_200),
    }

    return regime, details


# ════════════════════════════════════════════════════════════════════
# SIGNAL MODULE 4: CROSS-ASSET SIGNALS
# ════════════════════════════════════════════════════════════════════

def compute_cross_asset_signals(close):
    """Bond/equity ratio, VIX level, yield curve, credit spread proxy."""
    signals = {}

    # Bond/Equity ratio (TLT/SPY)
    if 'TLT' in close.columns and 'SPY' in close.columns:
        ratio = close['TLT'] / close['SPY']
        current = ratio.iloc[-1]
        avg_63 = ratio.iloc[-63:].mean()
        signals['tlt_spy_ratio'] = round(float(current), 4)
        signals['tlt_spy_z'] = round(float((current - avg_63) / ratio.iloc[-63:].std()), 2) if ratio.iloc[-63:].std() > 0 else 0
        signals['bonds_favored'] = bool(current > avg_63)

    # VIX level
    if 'VIX' in close.columns:
        vix = float(close['VIX'].iloc[-1])
        vix_avg = float(close['VIX'].iloc[-63:].mean())
        signals['vix'] = round(vix, 2)
        signals['vix_63d_avg'] = round(vix_avg, 2)
        if vix > 30:
            signals['vix_regime'] = 'high_fear'
        elif vix > 20:
            signals['vix_regime'] = 'elevated'
        else:
            signals['vix_regime'] = 'calm'

    # Yield curve proxy (TLT/IEF — duration spread)
    if 'TLT' in close.columns and 'IEF' in close.columns:
        yc = close['TLT'] / close['IEF']
        signals['yield_curve_ratio'] = round(float(yc.iloc[-1]), 4)
        signals['yield_curve_21d_chg'] = round(float(yc.iloc[-1] / yc.iloc[-22] - 1) * 100, 2) if len(yc) >= 22 else 0

    # Credit spread proxy (HYG/LQD)
    if 'HYG' in close.columns and 'LQD' in close.columns:
        credit = close['HYG'] / close['LQD']
        current_c = credit.iloc[-1]
        avg_c = credit.iloc[-63:].mean()
        signals['credit_ratio'] = round(float(current_c), 4)
        signals['credit_z'] = round(float((current_c - avg_c) / credit.iloc[-63:].std()), 2) if credit.iloc[-63:].std() > 0 else 0
        signals['credit_stress'] = bool(current_c < avg_c)

    return signals


# ════════════════════════════════════════════════════════════════════
# SIGNAL MODULE 5: MEAN REVERSION OVERLAY
# ════════════════════════════════════════════════════════════════════

def compute_mean_reversion_flags(close, etfs, threshold=-0.10):
    """Flag ETFs with 21d return < threshold as potential bounce candidates."""
    flags = {}
    for etf in etfs:
        if etf not in close.columns or len(close[etf].dropna()) < 22:
            continue
        px = close[etf].dropna()
        ret_21d = px.iloc[-1] / px.iloc[-22] - 1
        if ret_21d < threshold:
            flags[etf] = {
                'return_21d_pct': round(float(ret_21d * 100), 2),
                'bounce_candidate': True,
                'rsi_14': round(_compute_rsi(px, 14), 1),
            }
    return flags


# ════════════════════════════════════════════════════════════════════
# PORTFOLIO MODELS
# ════════════════════════════════════════════════════════════════════

def model_equal_weight_topk(momentum_scores, k=5):
    """Top K by momentum, equal weight."""
    top_k = momentum_scores.head(k).index.tolist()
    weight = round(1.0 / k, 4)
    return {etf: weight for etf in top_k}


def model_risk_parity(rp_weights):
    """All ETFs, inverse-vol weighted."""
    return {k: round(v, 4) for k, v in rp_weights.items()}


def model_momentum_risk_parity(momentum_scores, close, k=5, vol_window=21):
    """Top K by momentum, then risk-parity weight among them."""
    top_k = momentum_scores.head(k).index.tolist()
    available = [e for e in top_k if e in close.columns]
    if not available:
        return {}
    weights, _ = compute_risk_parity_weights(close, available, vol_window)
    return {k: round(v, 4) for k, v in weights.items()}


def model_regime_adaptive(momentum_scores, rp_weights, regime, k=5):
    """
    Bull  -> top-5 momentum, equal weight
    Bear  -> risk parity among defensive ETFs + top-2 momentum defensives
    Trans -> 50/50 blend of bull and bear allocations
    """
    if regime == 'bull':
        top_k = momentum_scores.head(k).index.tolist()
        w = round(1.0 / k, 4)
        return {etf: w for etf in top_k}

    elif regime == 'bear':
        # Defensive: bonds, gold, XLU, XLP, USMV
        defensive = [e for e in momentum_scores.index if e in DEFENSIVE_ETFS]
        if not defensive:
            defensive = list(DEFENSIVE_ETFS)[:5]
        # Take top 5 defensive by momentum
        def_scores = momentum_scores.loc[momentum_scores.index.isin(defensive)]
        top_def = def_scores.head(min(5, len(def_scores))).index.tolist()
        if not top_def:
            return rp_weights
        w = round(1.0 / len(top_def), 4)
        return {etf: w for etf in top_def}

    else:  # transition
        # Blend: 50% top-5 momentum, 50% defensive
        bull_alloc = model_regime_adaptive(momentum_scores, rp_weights, 'bull', k)
        bear_alloc = model_regime_adaptive(momentum_scores, rp_weights, 'bear', k)
        blended = {}
        all_keys = set(list(bull_alloc.keys()) + list(bear_alloc.keys()))
        for key in all_keys:
            blended[key] = round(
                0.5 * bull_alloc.get(key, 0) + 0.5 * bear_alloc.get(key, 0), 4
            )
        return blended


def model_core_satellite(momentum_scores, close, vol_window=21):
    """60% core (SPY/AGG risk parity), 40% satellite (top-3 momentum sectors)."""
    # Core: SPY + AGG, risk-parity weighted, scaled to 60%
    core_available = [e for e in CORE_ETFS if e in close.columns]
    if core_available:
        core_w, _ = compute_risk_parity_weights(close, core_available, vol_window)
        core_alloc = {k: round(v * 0.60, 4) for k, v in core_w.items()}
    else:
        core_alloc = {'SPY': 0.60}

    # Satellite: top 3 sector ETFs by momentum, equal weight, scaled to 40%
    sector_scores = momentum_scores.loc[momentum_scores.index.isin(SECTOR_ETFS)]
    top3 = sector_scores.head(3).index.tolist()
    sat_w = round(0.40 / max(len(top3), 1), 4)
    sat_alloc = {etf: sat_w for etf in top3}

    alloc = {**core_alloc, **sat_alloc}
    return alloc


# ════════════════════════════════════════════════════════════════════
# RISK METRICS
# ════════════════════════════════════════════════════════════════════

def compute_portfolio_risk(allocations, close, vol_window=63):
    """Compute expected vol, diversification ratio, max concentration."""
    etfs = [e for e in allocations if e in close.columns]
    if not etfs:
        return {}

    weights = np.array([allocations[e] for e in etfs])
    returns = close[etfs].pct_change().iloc[-vol_window:]
    cov = returns.cov() * 252
    vols = returns.std() * np.sqrt(252)

    port_var = float(weights @ cov.values @ weights)
    port_vol = np.sqrt(port_var) if port_var > 0 else 0

    # Diversification ratio = weighted avg vol / portfolio vol
    weighted_vol = float(weights @ vols.values)
    div_ratio = weighted_vol / port_vol if port_vol > 0 else 1.0

    return {
        'expected_annual_vol_pct': round(port_vol * 100, 2),
        'diversification_ratio': round(div_ratio, 2),
        'max_concentration_pct': round(float(max(weights)) * 100, 1),
        'num_holdings': len(etfs),
    }


# ════════════════════════════════════════════════════════════════════
# SPY BENCHMARK
# ════════════════════════════════════════════════════════════════════

def compute_spy_benchmark(close):
    """SPY trailing stats for comparison."""
    spy = close['SPY'].dropna()
    ret = spy.pct_change().dropna()

    stats = {}
    for period, days in [('1m', 21), ('3m', 63), ('6m', 126), ('1y', 252)]:
        if len(ret) >= days:
            r = ret.iloc[-days:]
            total_ret = (spy.iloc[-1] / spy.iloc[-days - 1] - 1) * 100
            ann_vol = r.std() * np.sqrt(252) * 100
            sharpe = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0
            stats[period] = {
                'return_pct': round(float(total_ret), 2),
                'ann_vol_pct': round(float(ann_vol), 2),
                'sharpe': round(float(sharpe), 2),
            }

    return stats


# ════════════════════════════════════════════════════════════════════
# OUTPUT
# ════════════════════════════════════════════════════════════════════

def print_summary(regime, regime_details, momentum_scores, feature_dict,
                  rp_weights, cross_signals, mr_flags,
                  portfolios, risk_metrics, spy_bench, capital):
    """Print clean terminal summary."""
    print("\n" + "=" * 70)
    print(f"PORTFOLIO FRAMEWORK — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Capital: ${capital:,.0f}")
    print("=" * 70)

    # Regime
    r_emoji = {'bull': 'BULL', 'bear': 'BEAR', 'transition': 'TRANSITION', 'unknown': '???'}
    print(f"\nREGIME: {r_emoji.get(regime, regime).upper()}")
    print(f"  SPY: ${regime_details.get('spy_price', '?')} | "
          f"200-SMA: ${regime_details.get('sma_200', '?')} | "
          f"Distance: {regime_details.get('pct_from_sma200', '?')}%")
    if regime_details.get('golden_cross'):
        print("  Golden Cross: YES (50-SMA > 200-SMA)")

    # Cross-asset signals
    print(f"\nCROSS-ASSET SIGNALS:")
    if 'vix' in cross_signals:
        print(f"  VIX: {cross_signals['vix']} ({cross_signals.get('vix_regime', '?')}) | "
              f"63d avg: {cross_signals.get('vix_63d_avg', '?')}")
    if 'tlt_spy_z' in cross_signals:
        bond_dir = "Bonds favored" if cross_signals.get('bonds_favored') else "Equities favored"
        print(f"  TLT/SPY z-score: {cross_signals['tlt_spy_z']} ({bond_dir})")
    if 'credit_stress' in cross_signals:
        credit_status = "STRESS" if cross_signals['credit_stress'] else "Normal"
        print(f"  Credit (HYG/LQD): z={cross_signals.get('credit_z', '?')} ({credit_status})")

    # Momentum rankings
    print(f"\nMOMENTUM RANKINGS (top 15):")
    print(f"  {'Rank':>4}  {'ETF':<6}  {'Score':>8}  {'5d':>7}  {'21d':>7}  {'63d':>7}  {'RSI':>5}")
    print(f"  {'----':>4}  {'---':<6}  {'-----':>8}  {'---':>7}  {'---':>7}  {'---':>7}  {'---':>5}")
    for i, (etf, row) in enumerate(momentum_scores.head(15).iterrows()):
        feats = feature_dict.get(etf, {})
        print(f"  {int(row['rank']):>4}  {etf:<6}  {row['score']:>8.4f}  "
              f"{feats.get('ret_5d', 0)*100:>6.1f}%  "
              f"{feats.get('ret_21d', 0)*100:>6.1f}%  "
              f"{feats.get('ret_63d', 0)*100:>6.1f}%  "
              f"{feats.get('rsi_14', 0):>5.0f}")

    # Mean reversion flags
    if mr_flags:
        print(f"\nMEAN REVERSION FLAGS (21d return < -10%):")
        for etf, info in mr_flags.items():
            print(f"  {etf}: {info['return_21d_pct']:.1f}% (RSI {info['rsi_14']:.0f}) — bounce candidate")
    else:
        print(f"\nNo mean reversion flags (no ETF down >10% in 21d).")

    # Portfolio models
    print(f"\n{'='*70}")
    print("RECOMMENDED ALLOCATIONS")
    print(f"{'='*70}")

    for name, alloc in portfolios.items():
        risk = risk_metrics.get(name, {})
        print(f"\n--- {name} ---")
        sorted_alloc = sorted(alloc.items(), key=lambda x: -x[1])
        for etf, w in sorted_alloc:
            dollars = w * capital
            print(f"  {etf:<6}  {w*100:>5.1f}%  ${dollars:>10,.0f}")
        total_w = sum(alloc.values())
        print(f"  {'TOTAL':<6}  {total_w*100:>5.1f}%  ${total_w*capital:>10,.0f}")
        if risk:
            print(f"  Vol: {risk.get('expected_annual_vol_pct', '?')}% | "
                  f"Div ratio: {risk.get('diversification_ratio', '?')} | "
                  f"Max conc: {risk.get('max_concentration_pct', '?')}% | "
                  f"Holdings: {risk.get('num_holdings', '?')}")

    # SPY benchmark
    print(f"\nSPY BUY-AND-HOLD BENCHMARK:")
    for period, stats in spy_bench.items():
        print(f"  {period:>3}: {stats['return_pct']:>+6.1f}% return | "
              f"{stats['ann_vol_pct']:>5.1f}% vol | "
              f"Sharpe {stats['sharpe']:>5.2f}")

    print(f"\n{'='*70}")
    print(f"State saved to: {STATE_FILE}")


def build_state_json(regime, regime_details, momentum_scores, feature_dict,
                     rp_weights, vols, cross_signals, mr_flags,
                     portfolios, risk_metrics, spy_bench, capital):
    """Build JSON state for persistence."""
    # ETF signals
    etf_signals = {}
    for etf in momentum_scores.index:
        etf_signals[etf] = {
            'momentum_score': round(float(momentum_scores.loc[etf, 'score']), 6) if not pd.isna(momentum_scores.loc[etf, 'score']) else None,
            'momentum_rank': int(momentum_scores.loc[etf, 'rank']),
            'features': {k: round(float(v), 6) for k, v in feature_dict.get(etf, {}).items()},
            'risk_parity_weight': round(float(rp_weights.get(etf, 0)), 6),
            'annualized_vol': round(float(vols.get(etf, 0)), 4) if etf in vols else None,
            'mean_reversion_flag': etf in mr_flags,
        }

    # Portfolio allocations with dollar amounts
    portfolio_allocs = {}
    for name, alloc in portfolios.items():
        portfolio_allocs[name] = {
            'weights': {k: round(v, 4) for k, v in alloc.items()},
            'dollars': {k: round(v * capital, 2) for k, v in alloc.items()},
            'risk_metrics': risk_metrics.get(name, {}),
        }

    state = {
        'timestamp': datetime.now().isoformat(),
        'capital': capital,
        'regime': regime,
        'regime_details': regime_details,
        'cross_asset_signals': cross_signals,
        'mean_reversion_flags': mr_flags,
        'etf_signals': etf_signals,
        'portfolios': portfolio_allocs,
        'spy_benchmark': spy_bench,
    }

    return state


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Portfolio Management Framework')
    parser.add_argument('--capital', type=float, default=50000, help='Total capital ($)')
    parser.add_argument('--top-k', type=int, default=5, help='Top K ETFs for momentum models')
    parser.add_argument('--vol-window', type=int, default=21, help='Volatility lookback (days)')
    parser.add_argument('--json-only', action='store_true', help='Output JSON only, no terminal summary')
    args = parser.parse_args()

    # 1. Download data
    close = download_data()

    # Filter to ETFs actually available in data
    available_etfs = [e for e in ALL_ETFS if e in close.columns and not close[e].isna().all()]
    print(f"  {len(available_etfs)}/{len(ALL_ETFS)} ETFs available")

    # 2. Compute signals
    print("\nComputing signals...")

    momentum_scores, feature_dict = compute_momentum_scores(close, available_etfs)
    momentum_scores = momentum_scores.dropna(subset=['score'])

    rp_weights, vols = compute_risk_parity_weights(close, available_etfs, args.vol_window)

    regime, regime_details = detect_regime(close)

    cross_signals = compute_cross_asset_signals(close)

    mr_flags = compute_mean_reversion_flags(close, available_etfs)

    # 3. Build portfolio models
    print("Building portfolio models...")

    portfolios = {}
    portfolios['Equal Weight Top-K'] = model_equal_weight_topk(momentum_scores, args.top_k)
    portfolios['Risk Parity'] = model_risk_parity(rp_weights)
    portfolios['Momentum + Risk Parity'] = model_momentum_risk_parity(
        momentum_scores, close, args.top_k, args.vol_window
    )
    portfolios['Regime-Adaptive'] = model_regime_adaptive(
        momentum_scores, rp_weights, regime, args.top_k
    )
    portfolios['Core-Satellite'] = model_core_satellite(
        momentum_scores, close, args.vol_window
    )

    # 4. Risk metrics for each portfolio
    risk_metrics = {}
    for name, alloc in portfolios.items():
        risk_metrics[name] = compute_portfolio_risk(alloc, close)

    # 5. SPY benchmark
    spy_bench = compute_spy_benchmark(close)

    # 6. Output
    state = build_state_json(
        regime, regime_details, momentum_scores, feature_dict,
        rp_weights, vols, cross_signals, mr_flags,
        portfolios, risk_metrics, spy_bench, args.capital
    )

    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

    if args.json_only:
        print(json.dumps(state, indent=2, default=str))
    else:
        print_summary(
            regime, regime_details, momentum_scores, feature_dict,
            rp_weights, cross_signals, mr_flags,
            portfolios, risk_metrics, spy_bench, args.capital
        )

    print("\nDone.")
    return state


if __name__ == '__main__':
    main()
