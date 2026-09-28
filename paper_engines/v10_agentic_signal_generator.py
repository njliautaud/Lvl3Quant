#!/usr/bin/env python3
"""
V10 Agentic Signal Generator — Single-Leg Options Signals for Robinhood
========================================================================

Adapts the V10 sector-ranking strategy (LGBM 17-feature momentum model)
to single-leg options trades compatible with Robinhood Option Level 2.

Level 2 allows: long calls, long puts, covered calls, cash-secured puts.
This script generates long call / long put signals only (no spreads).

Account context:
  - $645 cash, conservative $100-150 per trade
  - Monthly rebalance cadence (first trading day of month)
  - Generates SIGNALS for Claude to review — does NOT place orders

Cron: 30 16 1-7 * 1-5  (4:30 PM on first Mon-Fri of each month)

Usage:
    python v10_agentic_signal_generator.py              # generate signals
    python v10_agentic_signal_generator.py --dry-run    # print only, don't save
"""
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ──
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from research.tools.options_pricer import (
    bs_call_price,
    bs_put_price,
    estimate_iv,
    compute_atr,
    RISK_FREE_RATE,
    DEFAULT_HAIRCUT,
)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("WARNING: LightGBM not available. Falling back to momentum ranking.")

LOG_DIR = Path(__file__).resolve().parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)
STATE_DIR = BASE / 'state'
STATE_DIR.mkdir(exist_ok=True)
SIGNAL_PATH = STATE_DIR / 'v10_agentic_signals.json'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'v10_agentic_signal_generator.log'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

DRY_RUN = '--dry-run' in sys.argv

# ==================== STRATEGY CONFIG ====================

SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
EXTRA_TICKERS = ['SPY', '^VIX', '^VIX3M', 'TLT', 'SHY', 'HYG', 'GLD']

FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
]

# Account and position sizing
ACCOUNT_CASH = 645.0
MAX_TRADE_BUDGET = 150.0      # max premium per trade
MIN_TRADE_BUDGET = 50.0       # below this, skip (not worth commissions)
TARGET_TRADE_BUDGET = 120.0   # ideal spend per trade

# Option parameters
OTM_PCT = 0.03                # 3% OTM (conservative for single-leg)
TARGET_DTE = 28               # ~1 month
HAIRCUT = 0.15                # bid-ask haircut on BS price

# VIX regime
VIX_THRESHOLD = 20.0
REGIME_BULL_THRESHOLD = 0.4   # min regime score for high-VIX bullish entries

# Position counts
HIGH_VIX_TOP_K = 2            # long calls on top-2 only in high VIX
LOW_VIX_TOP_K = 2             # long calls on top-2
LOW_VIX_BOTTOM_K = 2          # long puts on bottom-2


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download sector ETF + macro data via yfinance."""
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    log.info(f"Downloading data for {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start='2024-01-01', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw

    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)

    close = close.ffill()
    high = high.ffill()
    low = low.ffill()

    rename_map = {'^VIX': 'VIX', '^VIX3M': 'VIX3M'}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)

    vc = 'VIX' if 'VIX' in close.columns else ('^VIX' if '^VIX' in close.columns else None)
    if vc is None:
        raise ValueError("VIX data not available")
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index)
    return close.loc[ix], sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]


# ==================== FEATURE ENGINEERING (17 momentum features) ====================

def compute_features(px):
    """Compute the 17 momentum features for a single sector ETF."""
    from scipy import stats
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, 'ret_5d'), (10, 'ret_10d'), (21, 'ret_21d'),
                   (63, 'ret_63d'), (126, 'ret_126d'), (252, 'ret_252d')]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:] / pk63) - 1).min())
    f['pct_52w_high'] = float(px.iloc[-1] / px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d'] / 3

    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)

    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val ** 2
        f['trend_slope_63d'] = slope * 252
    else:
        f['trend_r2_63d'] = 0.0
        f['trend_slope_63d'] = 0.0

    return f


# ==================== LGBM RANKING ====================

def run_lgbm_ranking(sc, close_df):
    """Walk-forward LGBM ranking using trailing data with 17 momentum features."""
    if not HAS_LGBM:
        log.warning("No LightGBM -- using simple momentum ranking")
        rets_21d = sc.pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    records = []
    all_dates = sc.index[-400:]
    rebal_dates = all_dates[::20]

    for dt in rebal_dates[:-1]:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx + 1].dropna()
            feats = compute_features(px)
            if not feats:
                continue
            fi = min(idx + 28, len(sc) - 1)
            feats.update({
                'date': dt, 'ticker': tk,
                'fwd_ret': float(sc[tk].iloc[fi] / sc[tk].iloc[idx] - 1)
            })
            records.append(feats)

    df = pd.DataFrame(records)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)

    if len(df) < 50:
        log.warning(f"Not enough training data ({len(df)} rows). Falling back to momentum.")
        rets_21d = sc.pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)

    X_train = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
    y_train = df['rank_label'].values.astype(np.float32)

    m = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1
    )
    m.fit(X_train, y_train)

    current_feats = {}
    for tk in sc.columns:
        px = sc[tk].dropna()
        feats = compute_features(px)
        if feats:
            current_feats[tk] = feats

    if not current_feats:
        return {}

    pred_df = pd.DataFrame(current_feats).T
    for c in FEAT_COLS:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    X_pred = np.nan_to_num(pred_df[FEAT_COLS].values.astype(np.float32))
    scores = m.predict(X_pred)

    return dict(zip(pred_df.index, scores))


# ==================== VIX REGIME ====================

def classify_regime(vix_val):
    """Classify VIX regime and return regime context."""
    if vix_val >= 30:
        return 'extreme_high', 0.8
    elif vix_val >= VIX_THRESHOLD:
        return 'high', 0.4 + (vix_val - 20) * 0.02
    else:
        return 'low', 0.15 + vix_val * 0.0125


# ==================== OPTION PRICING ====================

def estimate_option_premium(spot, strike, dte, is_call, high_series, low_series, close_series, vix_val):
    """
    Estimate single-leg option premium using BS with ATR-based IV.
    Returns premium per share (multiply by 100 for per-contract cost).
    """
    T = dte / 365.0
    if high_series is not None and len(high_series) >= 14:
        atr = compute_atr(high_series, low_series, close_series, period=14)
    else:
        atr = spot * 0.015

    sigma = estimate_iv(atr, spot, vix_val)

    if is_call:
        fair = bs_call_price(spot, strike, T, RISK_FREE_RATE, sigma)
    else:
        fair = bs_put_price(spot, strike, T, RISK_FREE_RATE, sigma)

    # Apply haircut — we're BUYING, so we pay more than mid
    premium = fair * (1 + HAIRCUT)
    return premium, sigma


def find_target_expiry(dte_target=28):
    """
    Find the closest STANDARD MONTHLY options expiry (3rd Friday) near
    dte_target DTE from today.

    RULES:
      1. Only consider 3rd-Friday monthlies (highest liquidity).
      2. Pick the one closest to dte_target that is >= 7 DTE.
      3. If within 21-45 DTE range, prefer that; otherwise pick closest monthly.
      4. NEVER return a non-Friday / non-standard date.
    """
    today = datetime.now().date()
    candidates = []

    for month_offset in range(0, 5):
        year = today.year
        month = today.month + month_offset
        if month > 12:
            month -= 12
            year += 1
        # Find 3rd Friday using calendar module
        import calendar as _cal
        cal = _cal.monthcalendar(year, month)
        fridays = [week[_cal.FRIDAY] for week in cal if week[_cal.FRIDAY] != 0]
        third_friday = date(year, month, fridays[2])
        dte = (third_friday - today).days
        if dte >= 7:
            candidates.append((third_friday, dte))

    if not candidates:
        # Emergency fallback: next Friday >= 21 DTE (should never happen)
        start = today + timedelta(days=21)
        days_to_friday = (4 - start.weekday()) % 7
        if days_to_friday == 0 and start.weekday() != 4:
            days_to_friday = 7
        fallback = start + timedelta(days=days_to_friday)
        return fallback, (fallback - today).days

    # Pick monthly closest to dte_target
    candidates.sort(key=lambda x: abs(x[1] - dte_target))
    best = candidates[0][0]
    return best, (best - today).days


# ==================== SIGNAL GENERATION ====================

def generate_signals(scores, sc, sh, sl, vix_val, spy):
    """
    Generate single-leg option signals based on LGBM rankings and VIX regime.

    Returns list of signal dicts ready for JSON output.
    """
    regime, regime_score = classify_regime(vix_val)
    log.info(f"VIX={vix_val:.1f}, regime={regime}, regime_score={regime_score:.2f}")

    # Sort sectors by LGBM score
    sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_sectors = sorted_sectors[:LOW_VIX_TOP_K]
    bottom_sectors = sorted_sectors[-LOW_VIX_BOTTOM_K:]

    # Score spread = confidence indicator
    if len(sorted_sectors) >= 4:
        top_avg = np.mean([s[1] for s in sorted_sectors[:3]])
        bot_avg = np.mean([s[1] for s in sorted_sectors[-3:]])
        score_spread = top_avg - bot_avg
    else:
        score_spread = 0.0

    # Confidence levels
    if score_spread > 0.3:
        confidence = 'high'
    elif score_spread > 0.15:
        confidence = 'medium'
    else:
        confidence = 'low'

    # Find expiry
    expiry_date, actual_dte = find_target_expiry(TARGET_DTE)
    log.info(f"Target expiry: {expiry_date} ({actual_dte} DTE)")

    signals = []

    # Determine which signals to generate based on regime
    if regime in ('high', 'extreme_high'):
        # HIGH VIX: only long calls on top-ranked sectors if regime supports it
        if regime_score >= REGIME_BULL_THRESHOLD:
            for tk, score in top_sectors[:HIGH_VIX_TOP_K]:
                sig = _build_call_signal(
                    tk, score, sc, sh, sl, vix_val,
                    expiry_date, actual_dte, confidence, regime
                )
                if sig:
                    signals.append(sig)
        else:
            log.info(f"Regime score {regime_score:.2f} < {REGIME_BULL_THRESHOLD} — "
                     f"no bullish entries in high-VIX mode")
            # In extreme high VIX with low regime score, consider protective puts
            # but skip for now (capital preservation)
    else:
        # LOW VIX: long calls on top-2, long puts on bottom-2
        for tk, score in top_sectors:
            sig = _build_call_signal(
                tk, score, sc, sh, sl, vix_val,
                expiry_date, actual_dte, confidence, regime
            )
            if sig:
                signals.append(sig)

        for tk, score in bottom_sectors:
            sig = _build_put_signal(
                tk, score, sc, sh, sl, vix_val,
                expiry_date, actual_dte, confidence, regime
            )
            if sig:
                signals.append(sig)

    # Budget check — ensure total doesn't exceed account
    total_budget = sum(s['max_budget'] for s in signals)
    if total_budget > ACCOUNT_CASH * 0.85:  # keep 15% cash buffer
        log.warning(f"Total budget ${total_budget:.0f} exceeds 85% of account "
                    f"(${ACCOUNT_CASH * 0.85:.0f}). Trimming to fit.")
        available = ACCOUNT_CASH * 0.85
        for sig in signals:
            sig['max_budget'] = min(sig['max_budget'], available / len(signals))

    return signals, {
        'regime': regime,
        'regime_score': regime_score,
        'vix': vix_val,
        'score_spread': score_spread,
        'confidence': confidence,
        'expiry': str(expiry_date),
        'dte': actual_dte,
        'rankings': {tk: round(s, 4) for tk, s in sorted_sectors},
    }


def _build_call_signal(tk, score, sc, sh, sl, vix_val, expiry_date, dte, confidence, regime):
    """Build a long call signal for a top-ranked sector."""
    spot = float(sc[tk].iloc[-1])
    strike = round(spot * (1 + OTM_PCT), 0)  # round to nearest dollar

    h = sh[tk] if tk in sh.columns else None
    l = sl[tk] if tk in sl.columns else None
    c = sc[tk] if tk in sc.columns else None

    premium_ps, iv = estimate_option_premium(spot, strike, dte, True, h, l, c, vix_val)
    cost_per_contract = premium_ps * 100  # 100 shares per contract

    if cost_per_contract < MIN_TRADE_BUDGET:
        log.info(f"{tk} call: premium ${cost_per_contract:.0f} below minimum ${MIN_TRADE_BUDGET}")
        # Still include — cheap options can be fine
    if cost_per_contract > MAX_TRADE_BUDGET:
        log.warning(f"{tk} call: premium ${cost_per_contract:.0f} exceeds max budget "
                    f"${MAX_TRADE_BUDGET}. Flagging as over-budget.")

    # How many contracts can we afford?
    max_contracts = max(1, int(TARGET_TRADE_BUDGET / cost_per_contract)) if cost_per_contract > 0 else 0
    actual_budget = min(cost_per_contract * max_contracts, MAX_TRADE_BUDGET)

    return {
        'ticker': tk,
        'direction': 'call',
        'action': 'BUY_CALL',
        'spot': round(spot, 2),
        'strike': int(strike),
        'expiry': str(expiry_date),
        'dte': dte,
        'est_premium_per_share': round(premium_ps, 2),
        'est_cost_per_contract': round(cost_per_contract, 2),
        'suggested_contracts': max_contracts,
        'max_budget': round(actual_budget, 2),
        'implied_vol': round(iv * 100, 1),
        'lgbm_score': round(score, 4),
        'confidence': confidence,
        'regime': regime,
        'otm_pct': round((strike / spot - 1) * 100, 1),
        'affordable': cost_per_contract <= MAX_TRADE_BUDGET,
        'rationale': f"Top-ranked sector (score {score:.3f}), "
                     f"{'high' if regime != 'low' else 'low'}-VIX bullish signal"
    }


def _build_put_signal(tk, score, sc, sh, sl, vix_val, expiry_date, dte, confidence, regime):
    """Build a long put signal for a bottom-ranked sector."""
    spot = float(sc[tk].iloc[-1])
    strike = round(spot * (1 - OTM_PCT), 0)  # OTM put = below spot

    h = sh[tk] if tk in sh.columns else None
    l = sl[tk] if tk in sl.columns else None
    c = sc[tk] if tk in sc.columns else None

    premium_ps, iv = estimate_option_premium(spot, strike, dte, False, h, l, c, vix_val)
    cost_per_contract = premium_ps * 100

    if cost_per_contract > MAX_TRADE_BUDGET:
        log.warning(f"{tk} put: premium ${cost_per_contract:.0f} exceeds max budget "
                    f"${MAX_TRADE_BUDGET}. Flagging as over-budget.")

    max_contracts = max(1, int(TARGET_TRADE_BUDGET / cost_per_contract)) if cost_per_contract > 0 else 0
    actual_budget = min(cost_per_contract * max_contracts, MAX_TRADE_BUDGET)

    return {
        'ticker': tk,
        'direction': 'put',
        'action': 'BUY_PUT',
        'spot': round(spot, 2),
        'strike': int(strike),
        'expiry': str(expiry_date),
        'dte': dte,
        'est_premium_per_share': round(premium_ps, 2),
        'est_cost_per_contract': round(cost_per_contract, 2),
        'suggested_contracts': max_contracts,
        'max_budget': round(actual_budget, 2),
        'implied_vol': round(iv * 100, 1),
        'lgbm_score': round(score, 4),
        'confidence': confidence,
        'regime': regime,
        'otm_pct': round((1 - strike / spot) * 100, 1),
        'affordable': cost_per_contract <= MAX_TRADE_BUDGET,
        'rationale': f"Bottom-ranked sector (score {score:.3f}), "
                     f"bearish signal in low-VIX regime"
    }


# ==================== OUTPUT ====================

def format_discord_summary(signals, context):
    """Format a clean Discord-friendly summary (plain English, no paths per HC #433)."""
    lines = []
    lines.append(f"**V10 Agentic Signals — {datetime.now().strftime('%b %d, %Y')}**")
    lines.append(f"VIX: {context['vix']:.1f} ({context['regime']} regime) | "
                 f"Confidence: {context['confidence']} | "
                 f"Score spread: {context['score_spread']:.3f}")
    lines.append("")

    if not signals:
        lines.append("No signals generated this month (regime filter or data issue).")
        return '\n'.join(lines)

    lines.append(f"**{len(signals)} signal(s) for review:**")
    total_budget = 0
    for s in signals:
        emoji = 'CALL' if s['direction'] == 'call' else 'PUT'
        affordable_tag = '' if s['affordable'] else ' [OVER BUDGET]'
        lines.append(
            f"  {emoji} {s['ticker']} ${s['strike']} {s['expiry']} "
            f"(~${s['est_cost_per_contract']:.0f}/contract, "
            f"{s['suggested_contracts']}x, "
            f"IV {s['implied_vol']}%){affordable_tag}"
        )
        total_budget += s['max_budget']

    lines.append("")
    lines.append(f"Total estimated cost: ${total_budget:.0f} / ${ACCOUNT_CASH:.0f} available")

    # Rankings summary
    rankings = context['rankings']
    sorted_r = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    top3 = ', '.join(f"{t}({s:.3f})" for t, s in sorted_r[:3])
    bot3 = ', '.join(f"{t}({s:.3f})" for t, s in sorted_r[-3:])
    lines.append(f"Top 3: {top3}")
    lines.append(f"Bottom 3: {bot3}")

    return '\n'.join(lines)


def print_summary(signals, context):
    """Print clean stdout summary."""
    print("\n" + "=" * 70)
    print(f"  V10 AGENTIC SIGNAL GENERATOR — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    print(f"\n  VIX:           {context['vix']:.1f}")
    print(f"  Regime:        {context['regime']} (score {context['regime_score']:.2f})")
    print(f"  Confidence:    {context['confidence']}")
    print(f"  Score spread:  {context['score_spread']:.3f}")
    print(f"  Target expiry: {context['expiry']} ({context['dte']} DTE)")

    print(f"\n  LGBM Rankings (high = bullish):")
    sorted_r = sorted(context['rankings'].items(), key=lambda x: x[1], reverse=True)
    for i, (tk, s) in enumerate(sorted_r, 1):
        bar = '#' * int(s * 40)
        print(f"    {i:2d}. {tk:5s}  {s:.4f}  {bar}")

    if not signals:
        print("\n  NO SIGNALS — regime filter blocked entries or insufficient data.")
    else:
        print(f"\n  SIGNALS ({len(signals)}):")
        print(f"  {'Ticker':<7} {'Type':<6} {'Strike':<8} {'Expiry':<12} "
              f"{'Premium':<10} {'Qty':<4} {'Budget':<8} {'IV':<7} {'OK?':<5}")
        print("  " + "-" * 70)
        total = 0
        for s in signals:
            ok = 'YES' if s['affordable'] else 'NO'
            print(f"  {s['ticker']:<7} {s['direction'].upper():<6} "
                  f"${s['strike']:<7} {s['expiry']:<12} "
                  f"${s['est_cost_per_contract']:<9.0f} {s['suggested_contracts']:<4} "
                  f"${s['max_budget']:<7.0f} {s['implied_vol']:.0f}%{'':>4} {ok:<5}")
            total += s['max_budget']
        print("  " + "-" * 68)
        print(f"  Total budget: ${total:.0f} / ${ACCOUNT_CASH:.0f} "
              f"({total / ACCOUNT_CASH * 100:.0f}% of account)")

    print("\n" + "=" * 70)


# ==================== MAIN ====================

def main():
    log.info("V10 Agentic Signal Generator starting...")

    # Download data
    try:
        close_df, sc, sh, sl, spy, vix = download_data()
    except Exception as e:
        log.error(f"Data download failed: {e}")
        return None, None

    vix_val = float(vix.iloc[-1])
    log.info(f"Latest VIX: {vix_val:.1f}")
    log.info(f"Data range: {sc.index[0].date()} to {sc.index[-1].date()} "
             f"({len(sc)} trading days)")

    # Run LGBM ranking
    log.info("Running LGBM sector ranking...")
    scores = run_lgbm_ranking(sc, close_df)
    if not scores:
        log.error("LGBM ranking returned no scores")
        return None, None

    log.info(f"Scores: {json.dumps({k: round(v, 4) for k, v in sorted(scores.items(), key=lambda x: -x[1])})}")

    # Generate signals
    signals, context = generate_signals(scores, sc, sh, sl, vix_val, spy)

    # Print summary
    print_summary(signals, context)

    # Discord-friendly summary
    discord_msg = format_discord_summary(signals, context)
    context['discord_summary'] = discord_msg
    log.info(f"Discord summary:\n{discord_msg}")

    # Save signals
    output = {
        'generated_at': datetime.now().isoformat(),
        'context': context,
        'signals': signals,
        'account': {
            'cash': ACCOUNT_CASH,
            'option_level': 2,
            'restriction': 'single-leg only (long calls, long puts)',
        },
    }

    if not DRY_RUN:
        with open(SIGNAL_PATH, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        log.info(f"Signals saved to {SIGNAL_PATH}")
    else:
        log.info("[DRY-RUN] Signals NOT saved")

    return signals, context


if __name__ == '__main__':
    main()
