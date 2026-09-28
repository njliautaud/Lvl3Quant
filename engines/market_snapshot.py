#!/usr/bin/env python3
"""
Market Data Snapshot — Persistent Market Context for Claude
============================================================
Runs every 5 minutes during market hours (9:30-16:00 ET weekdays).
Fetches live market data, computes key indicators, saves to a well-known
JSON file that Claude can read anytime for instant market context.

Designed to replace the need for a full MCP server — same data, simpler.

Output: /home/jupiter/Lvl3Quant/state/market_snapshot.json

Contents:
  - Prices: SPY, QQQ, UPRO, TQQQ, GLD, TLT, IWM, HYG, VIXY, UUP, USO, SLV, EEM
  - Volatility: 5d, 21d, 63d realized vol (SPY)
  - Trend: SPY vs SMA20/50/200, momentum 5d/21d
  - VIX proxy: VIXY level + 5d SMA + trend
  - Credit: HYG/LQD spread proxy
  - Breadth: IWM vs SPY relative strength
  - Gameplan v3: confluence score (0-3) + gate state
  - CTA: assets above/below SMA50
  - Protection overlay: 4-signal status
  - Regime: current allocation recommendation

HC #712: Part of the Finance Quant infrastructure mandate.
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")

ET = pytz.timezone("US/Eastern")
STATE_DIR = Path("/home/jupiter/Lvl3Quant/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = STATE_DIR / "market_snapshot.json"
LOG_DIR = Path("/home/jupiter/Lvl3Quant/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MKT-SNAP] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "market_snapshot.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("market_snapshot")

TICKERS = ['SPY', 'QQQ', 'UPRO', 'TQQQ', 'GLD', 'TLT', 'IWM', 'HYG',
           'LQD', 'VIXY', 'UUP', 'USO', 'SLV', 'EEM', 'COPX', 'UNG', 'DBA']

SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

CTA_UNIVERSE = ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'COPX', 'UUP', 'TLT', 'EEM']


def fetch_data():
    import yfinance as yf
    all_tickers = list(set(TICKERS + SECTOR_ETFS))
    data = yf.download(all_tickers, period="300d", auto_adjust=True,
                       threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except:
            pass
    return closes.dropna(how='all')


def safe_float(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), 4)


def compute_snapshot(closes):
    now = datetime.now(ET)
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    latest_date = closes.index[-1].strftime('%Y-%m-%d')

    snap = {
        'timestamp': now.isoformat(),
        'market_date': latest_date,
        'market_open': 9 <= now.hour < 16 and now.weekday() < 5,
    }

    # ── PRICES ──
    prices = {}
    daily_chg = {}
    weekly_chg = {}
    monthly_chg = {}
    for t in TICKERS:
        if t in closes.columns:
            p = closes[t].iloc[-1]
            prices[t] = safe_float(p)
            d = closes[t].pct_change().iloc[-1]
            daily_chg[t] = safe_float(d * 100) if not np.isnan(d) else None
            w = closes[t].pct_change(5).iloc[-1]
            weekly_chg[t] = safe_float(w * 100) if not np.isnan(w) else None
            m = closes[t].pct_change(21).iloc[-1]
            monthly_chg[t] = safe_float(m * 100) if not np.isnan(m) else None

    snap['prices'] = prices
    snap['daily_change_pct'] = daily_chg
    snap['weekly_change_pct'] = weekly_chg
    snap['monthly_change_pct'] = monthly_chg

    # ── VOLATILITY ──
    vol_5d = spy_ret.rolling(5).std().iloc[-1] * np.sqrt(252) * 100
    vol_21d = spy_ret.rolling(21).std().iloc[-1] * np.sqrt(252) * 100
    vol_63d = spy_ret.rolling(63).std().iloc[-1] * np.sqrt(252) * 100
    snap['volatility'] = {
        'vol_5d': safe_float(vol_5d),
        'vol_21d': safe_float(vol_21d),
        'vol_63d': safe_float(vol_63d),
        'vol_regime': 'LOW' if vol_21d < 15 else ('MEDIUM' if vol_21d < 30 else 'HIGH'),
    }

    # ── TREND (SPY) ──
    sma20 = spy.rolling(20).mean().iloc[-1]
    sma50 = spy.rolling(50).mean().iloc[-1]
    sma200 = spy.rolling(200).mean().iloc[-1]
    mom_5d = spy.pct_change(5).iloc[-1]
    mom_21d = spy.pct_change(21).iloc[-1]

    snap['trend'] = {
        'spy_price': safe_float(spy.iloc[-1]),
        'sma20': safe_float(sma20),
        'sma50': safe_float(sma50),
        'sma200': safe_float(sma200),
        'above_sma20': bool(spy.iloc[-1] > sma20) if not np.isnan(sma20) else None,
        'above_sma50': bool(spy.iloc[-1] > sma50) if not np.isnan(sma50) else None,
        'above_sma200': bool(spy.iloc[-1] > sma200) if not np.isnan(sma200) else None,
        'sma20_above_sma200': bool(sma20 > sma200) if not (np.isnan(sma20) or np.isnan(sma200)) else None,
        'momentum_5d_pct': safe_float(mom_5d * 100),
        'momentum_21d_pct': safe_float(mom_21d * 100),
        'trend_direction': 'UP' if (mom_5d > 0 and mom_21d > 0) else ('DOWN' if (mom_5d < 0 and mom_21d < 0) else 'MIXED'),
    }

    # ── VIX PROXY (VIXY) ──
    if 'VIXY' in closes.columns:
        vixy = closes['VIXY']
        vixy_sma5 = vixy.rolling(5).mean().iloc[-1]
        vixy_sma10 = vixy.rolling(10).mean().iloc[-1]
        snap['vix'] = {
            'vixy_price': safe_float(vixy.iloc[-1]),
            'vixy_sma5': safe_float(vixy_sma5),
            'vixy_rising': bool(vixy.iloc[-1] > vixy_sma5) if not np.isnan(vixy_sma5) else None,
            'vixy_1d_chg': safe_float(vixy.pct_change().iloc[-1] * 100),
            'vixy_5d_chg': safe_float(vixy.pct_change(5).iloc[-1] * 100),
            'fear_level': 'ELEVATED' if (not np.isnan(vixy_sma5) and vixy.iloc[-1] > vixy_sma5 * 1.1) else 'NORMAL',
        }

    # ── CREDIT ──
    if 'HYG' in closes.columns and 'LQD' in closes.columns:
        hyg = closes['HYG']
        lqd = closes['LQD']
        credit_ratio = lqd / hyg
        credit_sma20 = credit_ratio.rolling(20).mean().iloc[-1]
        hyg_5d = hyg.pct_change(5).iloc[-1]
        snap['credit'] = {
            'hyg_price': safe_float(hyg.iloc[-1]),
            'hyg_5d_chg_pct': safe_float(hyg_5d * 100),
            'credit_stressed': bool(hyg_5d < -0.03) if not np.isnan(hyg_5d) else False,
            'lqd_hyg_ratio': safe_float(credit_ratio.iloc[-1]),
            'ratio_vs_sma20': 'ELEVATED' if (not np.isnan(credit_sma20) and credit_ratio.iloc[-1] > credit_sma20) else 'NORMAL',
        }

    # ── BREADTH ──
    if 'IWM' in closes.columns:
        iwm_vs_spy = closes['IWM'].pct_change(21).iloc[-1] - spy.pct_change(21).iloc[-1]
        snap['breadth'] = {
            'iwm_vs_spy_21d': safe_float(iwm_vs_spy * 100),
            'small_caps_leading': bool(iwm_vs_spy > 0) if not np.isnan(iwm_vs_spy) else None,
            'breadth_healthy': bool(iwm_vs_spy > -0.03) if not np.isnan(iwm_vs_spy) else True,
        }

    # Sector breadth: % above 200d SMA
    sectors_above = 0
    sectors_total = 0
    for s in SECTOR_ETFS:
        if s in closes.columns:
            s200 = closes[s].rolling(200).mean().iloc[-1]
            if not np.isnan(s200):
                sectors_total += 1
                if closes[s].iloc[-1] > s200:
                    sectors_above += 1
    if sectors_total > 0:
        breadth_pct = sectors_above / sectors_total * 100
        snap['sector_breadth'] = {
            'pct_above_200sma': safe_float(breadth_pct),
            'count': f"{sectors_above}/{sectors_total}",
            'stressed': breadth_pct < 30,
        }

    # ── GAMEPLAN v3 CONFLUENCE SCORE ──
    score = 0.0
    components = {}

    # Short: 5d momentum
    if not np.isnan(mom_5d) and mom_5d > 0:
        score += 0.5
        components['mom_5d'] = 'PASS'
    else:
        components['mom_5d'] = 'FAIL'

    # Short: 10d RSI
    delta = spy_ret.copy()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = gain / loss_s.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).iloc[-1]
    if not np.isnan(rsi) and rsi > 50:
        score += 0.5
        components['rsi_10'] = f'PASS ({rsi:.1f})'
    else:
        components['rsi_10'] = f'FAIL ({rsi:.1f})' if not np.isnan(rsi) else 'FAIL'

    # Medium: 20/50 MA
    if not np.isnan(sma20) and not np.isnan(sma50) and sma20 > sma50:
        score += 0.5
        components['ma_20_50'] = 'PASS'
    else:
        components['ma_20_50'] = 'FAIL'

    # Medium: vol < 15%
    if not np.isnan(vol_21d) and vol_21d < 15:
        score += 0.5
        components['vol_low'] = f'PASS ({vol_21d:.1f}%)'
    else:
        components['vol_low'] = f'FAIL ({vol_21d:.1f}%)' if not np.isnan(vol_21d) else 'FAIL'

    # Long: 200d slope
    sma200_series = spy.rolling(200).mean()
    slope = sma200_series.pct_change(20).iloc[-1]
    if not np.isnan(slope) and slope > 0:
        score += 0.5
        components['sma200_slope'] = f'PASS ({slope*100:.3f}%)'
    else:
        components['sma200_slope'] = f'FAIL ({slope*100:.3f}%)' if not np.isnan(slope) else 'FAIL'

    # Long: 63d vol trend
    vol_63d_series = spy_ret.rolling(63).std() * np.sqrt(252) * 100
    vol_trend = (vol_63d_series - vol_63d_series.rolling(21).mean()).iloc[-1]
    if not np.isnan(vol_trend) and vol_trend < 0:
        score += 0.5
        components['vol_trend'] = f'PASS ({vol_trend:.2f})'
    else:
        components['vol_trend'] = f'FAIL ({vol_trend:.2f})' if not np.isnan(vol_trend) else 'FAIL'

    # Load gate state
    gate_state_file = STATE_DIR / "signal_watcher_state.json"
    try:
        gs = json.loads(gate_state_file.read_text())
        in_upro = gs.get('gameplan_in_upro', False)
    except:
        in_upro = False

    gate_threshold = 2.0 if in_upro else 2.5
    confluence_allows = score >= gate_threshold

    snap['confluence'] = {
        'score': safe_float(score),
        'max_score': 3.0,
        'components': components,
        'gate_state': 'IN_UPRO' if in_upro else 'OUT',
        'gate_threshold': gate_threshold,
        'gate_allows_upro': confluence_allows,
    }

    # ── PROTECTION OVERLAY (4 signals) ──
    prot_signals = {}
    prot_count = 0

    # 1. Vol < 20%
    if not np.isnan(vol_21d) and vol_21d < 20:
        prot_signals['vol_under_20'] = True
        prot_count += 1
    else:
        prot_signals['vol_under_20'] = False

    # 2. SPY > 50SMA
    if not np.isnan(sma50) and spy.iloc[-1] > sma50:
        prot_signals['spy_above_sma50'] = True
        prot_count += 1
    else:
        prot_signals['spy_above_sma50'] = False

    # 3. Credit not stressed
    if snap.get('credit', {}).get('credit_stressed', False):
        prot_signals['credit_healthy'] = False
    else:
        prot_signals['credit_healthy'] = True
        prot_count += 1

    # 4. Breadth OK
    if snap.get('breadth', {}).get('breadth_healthy', True):
        prot_signals['breadth_ok'] = True
        prot_count += 1
    else:
        prot_signals['breadth_ok'] = False

    snap['protection'] = {
        'signals': prot_signals,
        'count': f"{prot_count}/4",
        'all_green': prot_count == 4,
        'status': 'ALL CLEAR' if prot_count == 4 else ('WARNING' if prot_count >= 3 else 'DANGER'),
    }

    # ── CTA TREND STATUS ──
    cta_above = {}
    for t in CTA_UNIVERSE:
        if t in closes.columns:
            s50 = closes[t].rolling(50).mean().iloc[-1]
            if not np.isnan(s50):
                cta_above[t] = bool(closes[t].iloc[-1] > s50)

    above_count = sum(1 for v in cta_above.values() if v)
    snap['cta'] = {
        'assets_above_sma50': cta_above,
        'count_above': above_count,
        'count_total': len(cta_above),
        'trend_strength': 'STRONG' if above_count >= 6 else ('MODERATE' if above_count >= 3 else 'WEAK'),
    }

    # ── VIX PANIC SIGNALS ──
    # Check if VIX (VIXY) peaked >= 25 recently
    if 'VIXY' in closes.columns:
        vixy = closes['VIXY']
        # VIXY doesn't directly equal VIX, but we use it as proxy
        # For VIX-equivalent, we use realized vol or check if VIXY is spiking
        # The actual VIX signal uses vol_21d as a proxy
        vix_proxy = vol_21d  # 21d realized vol as VIX-like measure
        recent_max_vol = spy_ret.rolling(21).std().rolling(10).max().iloc[-1] * np.sqrt(252) * 100

        snap['panic_signals'] = {
            'vix_proxy': safe_float(vix_proxy),
            'recent_peak_vol': safe_float(recent_max_vol),
            'vol_peaked_above_25': bool(recent_max_vol > 25) if not np.isnan(recent_max_vol) else False,
            'vol_now_below_22': bool(vix_proxy < 22) if not np.isnan(vix_proxy) else False,
            'panic_reversal_active': bool(recent_max_vol > 25 and vix_proxy < 22) if not (np.isnan(recent_max_vol) or np.isnan(vix_proxy)) else False,
            'sector_breadth_stressed': snap.get('sector_breadth', {}).get('stressed', False),
            'credit_stressed': snap.get('credit', {}).get('credit_stressed', False),
        }

        panic_score = 0
        if snap['panic_signals']['panic_reversal_active']:
            panic_score += 1
        if snap['panic_signals']['sector_breadth_stressed']:
            panic_score += 1
        if snap['panic_signals']['credit_stressed']:
            panic_score += 1
        snap['panic_signals']['confluence_score'] = panic_score
        snap['panic_signals']['high_conviction'] = panic_score >= 2

    # ── FINAL REGIME RECOMMENDATION ──
    if vol_21d > 30:
        regime = 'GLD'
        reason = f'Crisis vol ({vol_21d:.1f}% > 30%)'
    elif vol_21d > 15 or (not np.isnan(sma20) and not np.isnan(sma200) and sma20 < sma200):
        regime = 'SPY'
        if not np.isnan(sma20) and not np.isnan(sma200) and sma20 < sma200:
            reason = '20/200 MA bearish crossover'
        else:
            reason = f'Medium vol ({vol_21d:.1f}% > 15%)'
    elif not confluence_allows:
        regime = 'SPY'
        reason = f'Confluence gate blocked (score {score:.1f} < {gate_threshold})'
    else:
        regime = 'UPRO'
        reason = f'All clear: vol {vol_21d:.1f}%, confluence {score:.1f}'

    snap['regime'] = {
        'allocation': regime,
        'reason': reason,
        'gameplan_version': 'v3',
    }

    return snap


def main():
    now = datetime.now(ET)

    # Skip weekends
    if now.weekday() >= 5:
        log.info("Weekend — skipping")
        return

    log.info("Fetching market data...")
    try:
        closes = fetch_data()
    except Exception as e:
        log.error(f"Data fetch failed: {e}")
        return

    log.info("Computing snapshot...")
    snap = compute_snapshot(closes)

    # Save
    OUTPUT_FILE.write_text(json.dumps(snap, indent=2, default=str))
    log.info(f"Snapshot saved. Regime: {snap['regime']['allocation']} | "
             f"Confluence: {snap['confluence']['score']}/3 | "
             f"Vol: {snap['volatility']['vol_21d']}% | "
             f"Protection: {snap['protection']['count']}")


if __name__ == '__main__':
    main()
