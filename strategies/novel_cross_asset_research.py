#!/usr/bin/env python3
"""
Novel Cross-Asset Signal Research — 4 Strategies
Walk-forward backtest: 2019-01-01 to 2026-08-27
Sliding 12-month train / 1-month OOS, 10bps round-trip cost
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import json
import os

# ─── CONFIG ───────────────────────────────────────────────────────────────────

START = '2018-01-01'  # extra year for warmup
END = '2026-08-27'
TRAIN_MONTHS = 12
OOS_MONTHS = 1
COST_BPS = 10  # 10bps round-trip
RF_ANNUAL = 0.04  # risk-free rate for Sharpe/Sortino

TICKERS = [
    'SPY', 'HYG', 'LQD',  # credit
    'USO', 'GLD', 'COPX',  # commodities
    'XLE', 'XLU', 'XLP', 'XLI', 'XLB', 'XLK', 'XLV',  # sectors
    'TLT', 'SHY',  # duration
    'UUP',  # dollar (DXY proxy)
    '^VIX',  # VIX
]


def fetch_data():
    """Download all tickers, return adjusted close DataFrame."""
    print("Fetching data from yfinance...")
    raw = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close']
    else:
        prices = raw
    prices.columns = [c.replace('^', '') for c in prices.columns]
    prices = prices.ffill().dropna(how='all')
    print(f"  Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")
    return prices


def calc_metrics(returns, name, spy_returns=None):
    """Calculate CAGR, Sharpe, Sortino, MaxDD from daily return series."""
    if len(returns) < 20:
        return {'name': name, 'error': 'insufficient data'}

    cum = (1 + returns).cumprod()
    years = len(returns) / 252
    cagr = (cum.iloc[-1] ** (1 / years)) - 1 if years > 0 else 0

    rf_daily = (1 + RF_ANNUAL) ** (1/252) - 1
    excess = returns - rf_daily
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 0 else 0

    downside = excess[excess < 0]
    sortino = np.sqrt(252) * excess.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0

    drawdown = cum / cum.cummax() - 1
    max_dd = drawdown.min()

    # Win rate (daily)
    wr = (returns > 0).sum() / len(returns)

    result = {
        'name': name,
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd': round(max_dd * 100, 2),
        'win_rate': round(wr * 100, 1),
        'n_days': len(returns),
        'years': round(years, 2),
    }

    if spy_returns is not None and len(spy_returns) == len(returns):
        spy_cum = (1 + spy_returns).cumprod()
        spy_years = len(spy_returns) / 252
        spy_cagr = (spy_cum.iloc[-1] ** (1 / spy_years)) - 1 if spy_years > 0 else 0
        result['spy_cagr'] = round(spy_cagr * 100, 2)
        result['excess_cagr'] = round((cagr - spy_cagr) * 100, 2)

    return result


def regime_stability(returns, spy_returns):
    """Check performance in up vs down SPY months. Returns regime Sharpe ratio."""
    if len(returns) < 60:
        return {'up_sharpe': 0, 'down_sharpe': 0, 'ratio': 0}

    # Monthly resample
    monthly_ret = (1 + returns).resample('ME').prod() - 1
    monthly_spy = (1 + spy_returns).resample('ME').prod() - 1

    # Align
    common = monthly_ret.index.intersection(monthly_spy.index)
    monthly_ret = monthly_ret.loc[common]
    monthly_spy = monthly_spy.loc[common]

    up_months = monthly_spy > 0
    down_months = monthly_spy <= 0

    rf_monthly = (1 + RF_ANNUAL) ** (1/12) - 1

    up_excess = monthly_ret[up_months] - rf_monthly
    down_excess = monthly_ret[down_months] - rf_monthly

    up_sharpe = np.sqrt(12) * up_excess.mean() / up_excess.std() if len(up_excess) > 3 and up_excess.std() > 0 else 0
    down_sharpe = np.sqrt(12) * down_excess.mean() / down_excess.std() if len(down_excess) > 3 and down_excess.std() > 0 else 0

    return {
        'up_market_sharpe': round(up_sharpe, 3),
        'down_market_sharpe': round(down_sharpe, 3),
        'regime_ratio': round(abs(up_sharpe - down_sharpe) / max(abs(up_sharpe), abs(down_sharpe), 0.001), 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: CREDIT STRESS ARBITRAGE
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_credit_stress(prices):
    """
    Credit stress vs equity vol divergence.
    - HYG/LQD ratio drop (credit stress) without VIX spike → buy SPY (credit overreacting)
    - VIX spike without credit stress → hedge/sell SPY (equity overreacting)
    Walk-forward: train on 12m to calibrate thresholds, OOS on next month.
    """
    print("\n" + "="*80)
    print("STRATEGY 1: CREDIT STRESS ARBITRAGE")
    print("="*80)

    # Build signals
    hyg = prices.get('HYG')
    lqd = prices.get('LQD')
    vix = prices.get('VIX')
    spy = prices.get('SPY')

    if hyg is None or lqd is None or vix is None or spy is None:
        print("  Missing data for credit stress strategy")
        return None, None

    # Credit spread proxy: HYG/LQD ratio (lower = wider spreads = more stress)
    credit_ratio = hyg / lqd
    credit_ratio_z = (credit_ratio - credit_ratio.rolling(60).mean()) / credit_ratio.rolling(60).std()

    # VIX z-score
    vix_z = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()

    spy_ret = spy.pct_change()

    # Walk-forward
    all_dates = spy_ret.dropna().index
    # Start OOS from 2019-01-01 onward
    oos_start = pd.Timestamp('2019-01-01')

    positions = pd.Series(0.0, index=all_dates)

    # Monthly walk-forward
    month_starts = pd.date_range(start=oos_start, end=all_dates[-1], freq='MS')

    for ms in month_starts:
        # Train window: 12 months before this month
        train_end = ms - timedelta(days=1)
        train_start = ms - timedelta(days=365)

        train_mask = (all_dates >= train_start) & (all_dates <= train_end)
        oos_end = ms + pd.offsets.MonthEnd(1)
        oos_mask = (all_dates >= ms) & (all_dates <= oos_end)

        if train_mask.sum() < 60 or oos_mask.sum() < 5:
            continue

        # In training: find optimal z-score thresholds
        # We use training data to calibrate what "divergence" looks like
        train_credit_z = credit_ratio_z.reindex(all_dates[train_mask]).dropna()
        train_vix_z = vix_z.reindex(all_dates[train_mask]).dropna()
        train_ret = spy_ret.reindex(all_dates[train_mask]).dropna()

        common_train = train_credit_z.index.intersection(train_vix_z.index).intersection(train_ret.index)
        if len(common_train) < 60:
            continue

        tc_z = train_credit_z.loc[common_train]
        tv_z = train_vix_z.loc[common_train]
        tr = train_ret.loc[common_train]

        # Calibrate: use percentiles from training data
        credit_stress_thresh = tc_z.quantile(0.2)  # bottom 20% = credit stressed
        vix_calm_thresh = tv_z.quantile(0.6)       # below 60th pctile = VIX not spiking
        vix_spike_thresh = tv_z.quantile(0.8)      # top 20% = VIX spiking
        credit_calm_thresh = tc_z.quantile(0.5)     # above median = credit calm

        # Apply to OOS
        oos_dates = all_dates[oos_mask]
        for d in oos_dates:
            if d not in credit_ratio_z.index or d not in vix_z.index:
                continue
            cz = credit_ratio_z.loc[d]
            vz = vix_z.loc[d]

            if pd.isna(cz) or pd.isna(vz):
                continue

            # Signal 1: Credit panicking, equity calm → BUY (credit overreacting)
            if cz < credit_stress_thresh and vz < vix_calm_thresh:
                positions.loc[d] = 1.0
            # Signal 2: Equity panicking, credit calm → SHORT/HEDGE
            elif vz > vix_spike_thresh and cz > credit_calm_thresh:
                positions.loc[d] = -0.5
            else:
                positions.loc[d] = 0.0

    # Apply positions with 1-day lag (trade next day)
    positions = positions.shift(1).fillna(0)

    # Align positions and returns
    common_idx = positions.index.intersection(spy_ret.dropna().index)
    positions = positions.reindex(common_idx)
    spy_ret_aligned = spy_ret.reindex(common_idx).fillna(0)

    # Compute returns with costs
    trades = positions.diff().abs().fillna(0)
    cost = trades * COST_BPS / 10000
    strat_ret = positions * spy_ret_aligned - cost

    # Only OOS period
    oos_mask_final = strat_ret.index >= oos_start
    strat_ret_oos = strat_ret[oos_mask_final].dropna()
    spy_ret_oos = spy_ret.reindex(strat_ret_oos.index).fillna(0)

    # Exposure stats
    pos_oos = positions[positions.index >= oos_start]
    active = (pos_oos != 0).sum()
    total = len(pos_oos)
    long_pct = (pos_oos > 0).sum() / total * 100 if total > 0 else 0
    short_pct = (pos_oos < 0).sum() / total * 100 if total > 0 else 0

    print(f"  Exposure: {active}/{total} days ({active/total*100:.1f}%)")
    print(f"  Long: {long_pct:.1f}%, Short: {short_pct:.1f}%, Flat: {100-long_pct-short_pct:.1f}%")
    print(f"  Turnover: {trades[oos_mask_final].sum():.0f} round-trips equivalent")

    metrics = calc_metrics(strat_ret_oos, 'Credit Stress Arbitrage', spy_ret_oos)
    regime = regime_stability(strat_ret_oos, spy_ret_oos)
    metrics.update(regime)

    return metrics, strat_ret_oos


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: COMMODITY MOMENTUM → SECTOR ROTATION
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_commodity_sector_rotation(prices):
    """
    Multi-commodity momentum drives sector allocation.
    - Oil up → overweight XLE
    - Gold up → overweight defensives (XLU, XLP)
    - Copper up → overweight cyclicals (XLI, XLB)
    Monthly rebalance, walk-forward.
    """
    print("\n" + "="*80)
    print("STRATEGY 2: COMMODITY MOMENTUM → SECTOR ROTATION")
    print("="*80)

    spy = prices.get('SPY')
    uso = prices.get('USO')
    gld = prices.get('GLD')
    copx = prices.get('COPX')

    sectors = {}
    for s in ['XLE', 'XLU', 'XLP', 'XLI', 'XLB', 'XLK', 'XLV']:
        if s in prices.columns:
            sectors[s] = prices[s]

    if uso is None or gld is None or copx is None or len(sectors) < 5:
        print("  Missing data")
        return None, None

    spy_ret = spy.pct_change()
    sector_rets = pd.DataFrame({s: v.pct_change() for s, v in sectors.items()})

    all_dates = spy_ret.dropna().index
    oos_start = pd.Timestamp('2019-01-01')
    month_starts = pd.date_range(start=oos_start, end=all_dates[-1], freq='MS')

    # Equal-weight baseline
    equal_weights = {s: 1.0/len(sectors) for s in sectors}

    portfolio_ret = pd.Series(0.0, index=all_dates)
    prev_weights = equal_weights.copy()

    for i, ms in enumerate(month_starts):
        # Lookback for momentum
        lookback_20d = ms - timedelta(days=30)
        lookback_60d = ms - timedelta(days=90)

        # Train window for calibration
        train_start = ms - timedelta(days=365)
        train_end = ms - timedelta(days=1)

        oos_end = ms + pd.offsets.MonthEnd(1)
        oos_mask = (all_dates >= ms) & (all_dates <= oos_end)

        if oos_mask.sum() < 5:
            continue

        # Compute commodity momentum (20-day return)
        def safe_ret(series, start, end):
            sub = series.loc[start:end].dropna()
            if len(sub) < 5:
                return 0.0
            return sub.iloc[-1] / sub.iloc[0] - 1

        oil_mom = safe_ret(uso, lookback_20d, train_end)
        gold_mom = safe_ret(gld, lookback_20d, train_end)
        copper_mom = safe_ret(copx, lookback_20d, train_end)

        # Also compute 60-day for trend confirmation
        oil_trend = safe_ret(uso, lookback_60d, train_end)
        gold_trend = safe_ret(gld, lookback_60d, train_end)
        copper_trend = safe_ret(copx, lookback_60d, train_end)

        # Build weights — start from equal weight, tilt based on commodity signals
        weights = equal_weights.copy()

        # Oil momentum → XLE tilt
        if oil_mom > 0.02 and oil_trend > 0:
            weights['XLE'] = weights.get('XLE', 0) + 0.10
        elif oil_mom < -0.02 and oil_trend < 0:
            weights['XLE'] = max(weights.get('XLE', 0) - 0.08, 0.02)

        # Gold momentum → defensive tilt
        if gold_mom > 0.02 and gold_trend > 0:
            for d in ['XLU', 'XLP']:
                if d in weights:
                    weights[d] += 0.06
            for c in ['XLI', 'XLB']:
                if c in weights:
                    weights[c] = max(weights[c] - 0.04, 0.02)

        # Copper momentum → cyclical tilt
        if copper_mom > 0.03 and copper_trend > 0:
            for c in ['XLI', 'XLB']:
                if c in weights:
                    weights[c] += 0.06
            for d in ['XLU', 'XLP']:
                if d in weights:
                    weights[d] = max(weights[d] - 0.03, 0.02)
        elif copper_mom < -0.03 and copper_trend < 0:
            for d in ['XLU', 'XLP', 'XLV']:
                if d in weights:
                    weights[d] += 0.04

        # Normalize weights to sum to 1
        total_w = sum(weights.values())
        weights = {k: v/total_w for k, v in weights.items()}

        # Apply to OOS days
        oos_dates = all_dates[oos_mask]
        for d in oos_dates:
            daily_ret = 0.0
            for s, w in weights.items():
                if d in sector_rets.index and s in sector_rets.columns:
                    r = sector_rets.loc[d, s]
                    if not pd.isna(r):
                        daily_ret += w * r
            portfolio_ret.loc[d] = daily_ret

        # Turnover cost at rebalance
        turnover = sum(abs(weights.get(s, 0) - prev_weights.get(s, 0)) for s in set(list(weights.keys()) + list(prev_weights.keys())))
        if len(oos_dates) > 0:
            portfolio_ret.loc[oos_dates[0]] -= turnover * COST_BPS / 10000

        prev_weights = weights.copy()

    # OOS only
    oos_final = portfolio_ret.index >= oos_start
    strat_ret = portfolio_ret[oos_final].dropna()
    spy_ret_oos = spy_ret.reindex(strat_ret.index).fillna(0)

    metrics = calc_metrics(strat_ret, 'Commodity → Sector Rotation', spy_ret_oos)
    regime = regime_stability(strat_ret, spy_ret_oos)
    metrics.update(regime)

    return metrics, strat_ret


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: YIELD CURVE SHAPE → DURATION/SECTOR TRADE
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_yield_curve_rotation(prices):
    """
    TLT/SHY ratio as yield curve shape proxy.
    Steepening → cyclicals. Flattening → defensives.
    Monthly rotation, walk-forward calibration.
    """
    print("\n" + "="*80)
    print("STRATEGY 3: YIELD CURVE SHAPE → SECTOR ROTATION")
    print("="*80)

    tlt = prices.get('TLT')
    shy = prices.get('SHY')
    spy = prices.get('SPY')

    if tlt is None or shy is None:
        print("  Missing TLT/SHY data")
        return None, None

    # Cyclical and defensive baskets
    cyclical_tickers = ['XLI', 'XLB', 'XLE', 'XLK']
    defensive_tickers = ['XLU', 'XLP', 'XLV']

    cyclicals = {t: prices[t] for t in cyclical_tickers if t in prices.columns}
    defensives = {t: prices[t] for t in defensive_tickers if t in prices.columns}

    if not cyclicals or not defensives:
        print("  Missing sector data")
        return None, None

    # Compute returns
    spy_ret = spy.pct_change()
    cyc_rets = pd.DataFrame({t: v.pct_change() for t, v in cyclicals.items()})
    def_rets = pd.DataFrame({t: v.pct_change() for t, v in defensives.items()})

    # Curve shape proxy: TLT/SHY ratio change
    curve_ratio = tlt / shy
    curve_change_20d = curve_ratio.pct_change(20)  # 20-day change
    curve_change_60d = curve_ratio.pct_change(60)  # 60-day trend

    all_dates = spy_ret.dropna().index
    oos_start = pd.Timestamp('2019-01-01')
    month_starts = pd.date_range(start=oos_start, end=all_dates[-1], freq='MS')

    portfolio_ret = pd.Series(0.0, index=all_dates)

    for ms in month_starts:
        train_start = ms - timedelta(days=365)
        train_end = ms - timedelta(days=1)
        oos_end = ms + pd.offsets.MonthEnd(1)
        oos_mask = (all_dates >= ms) & (all_dates <= oos_end)

        if oos_mask.sum() < 5:
            continue

        # Training: find threshold for "meaningful" curve change
        train_curve = curve_change_20d.loc[train_start:train_end].dropna()
        if len(train_curve) < 30:
            continue

        steep_thresh = train_curve.quantile(0.7)   # top 30% = steepening
        flat_thresh = train_curve.quantile(0.3)     # bottom 30% = flattening

        # Current curve signal (at month start)
        if ms not in curve_change_20d.index:
            # Find nearest
            nearest = curve_change_20d.index[curve_change_20d.index <= ms]
            if len(nearest) == 0:
                continue
            ms_signal = nearest[-1]
        else:
            ms_signal = ms

        cc20 = curve_change_20d.get(ms_signal, 0)
        cc60 = curve_change_60d.get(ms_signal, 0) if ms_signal in curve_change_60d.index else 0

        if pd.isna(cc20):
            cc20 = 0
        if pd.isna(cc60):
            cc60 = 0

        # Determine allocation
        if cc20 > steep_thresh and cc60 > 0:
            # Steepening + uptrend → overweight cyclicals
            cyc_weight = 0.7
            def_weight = 0.3
        elif cc20 < flat_thresh and cc60 < 0:
            # Flattening + downtrend → overweight defensives
            cyc_weight = 0.3
            def_weight = 0.7
        else:
            # Neutral
            cyc_weight = 0.5
            def_weight = 0.5

        # Equal weight within each basket
        n_cyc = len(cyclicals)
        n_def = len(defensives)

        oos_dates = all_dates[oos_mask]
        for d in oos_dates:
            daily = 0.0
            for t in cyclicals:
                if d in cyc_rets.index and t in cyc_rets.columns:
                    r = cyc_rets.loc[d, t]
                    if not pd.isna(r):
                        daily += (cyc_weight / n_cyc) * r
            for t in defensives:
                if d in def_rets.index and t in def_rets.columns:
                    r = def_rets.loc[d, t]
                    if not pd.isna(r):
                        daily += (def_weight / n_def) * r
            portfolio_ret.loc[d] = daily

    # Minimal rebalance cost (monthly, small tilts)
    # Approximate: ~15% turnover per month
    n_months = len(month_starts)
    total_cost = n_months * 0.15 * COST_BPS / 10000

    oos_final = portfolio_ret.index >= oos_start
    strat_ret = portfolio_ret[oos_final].dropna()
    # Spread cost evenly
    if len(strat_ret) > 0:
        strat_ret -= total_cost / len(strat_ret)

    spy_ret_oos = spy_ret.reindex(strat_ret.index).fillna(0)

    metrics = calc_metrics(strat_ret, 'Yield Curve Rotation', spy_ret_oos)
    regime = regime_stability(strat_ret, spy_ret_oos)
    metrics.update(regime)

    return metrics, strat_ret


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: DOLLAR REGIME × SECTOR TILT
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_dollar_regime(prices):
    """
    USD strength → favor domestic-revenue sectors (XLU, XLP, XLV).
    USD weakness → favor international-revenue sectors (XLK, XLI, XLB).
    Only trade when DXY trend is clear (>2% move in 30 days).
    """
    print("\n" + "="*80)
    print("STRATEGY 4: DOLLAR REGIME × SECTOR TILT")
    print("="*80)

    uup = prices.get('UUP')  # Dollar bull ETF as DXY proxy
    spy = prices.get('SPY')

    if uup is None:
        print("  Missing UUP (dollar proxy) data")
        return None, None

    # Domestic-revenue sectors (less FX exposure)
    domestic = ['XLU', 'XLP', 'XLV']
    # International-revenue sectors (more FX exposure, benefit from weak USD)
    international = ['XLK', 'XLI', 'XLB', 'XLE']

    dom_prices = {t: prices[t] for t in domestic if t in prices.columns}
    intl_prices = {t: prices[t] for t in international if t in prices.columns}

    spy_ret = spy.pct_change()
    dom_rets = pd.DataFrame({t: v.pct_change() for t, v in dom_prices.items()})
    intl_rets = pd.DataFrame({t: v.pct_change() for t, v in intl_prices.items()})

    # Dollar momentum
    uup_ret_30d = uup.pct_change(30)
    uup_ret_60d = uup.pct_change(60)

    all_dates = spy_ret.dropna().index
    oos_start = pd.Timestamp('2019-01-01')
    month_starts = pd.date_range(start=oos_start, end=all_dates[-1], freq='MS')

    portfolio_ret = pd.Series(0.0, index=all_dates)

    for ms in month_starts:
        train_end = ms - timedelta(days=1)
        oos_end = ms + pd.offsets.MonthEnd(1)
        oos_mask = (all_dates >= ms) & (all_dates <= oos_end)

        if oos_mask.sum() < 5:
            continue

        # Current dollar signal
        if ms not in uup_ret_30d.index:
            nearest = uup_ret_30d.index[uup_ret_30d.index <= ms]
            if len(nearest) == 0:
                continue
            ms_signal = nearest[-1]
        else:
            ms_signal = ms

        dxy_30 = uup_ret_30d.get(ms_signal, 0)
        dxy_60 = uup_ret_60d.get(ms_signal, 0) if ms_signal in uup_ret_60d.index else 0

        if pd.isna(dxy_30):
            dxy_30 = 0
        if pd.isna(dxy_60):
            dxy_60 = 0

        # Threshold: only trade when trend is clear
        clear_trend = abs(dxy_30) > 0.02  # >2% move in 30 days
        trend_confirmed = (dxy_30 > 0 and dxy_60 > 0) or (dxy_30 < 0 and dxy_60 < 0)

        n_dom = len(dom_prices)
        n_intl = len(intl_prices)

        if clear_trend and trend_confirmed and dxy_30 > 0:
            # Strong dollar → favor domestic
            dom_w = 0.65
            intl_w = 0.35
        elif clear_trend and trend_confirmed and dxy_30 < 0:
            # Weak dollar → favor international
            dom_w = 0.35
            intl_w = 0.65
        else:
            # No clear trend → equal weight
            dom_w = 0.5
            intl_w = 0.5

        oos_dates = all_dates[oos_mask]
        for d in oos_dates:
            daily = 0.0
            for t in dom_prices:
                if d in dom_rets.index and t in dom_rets.columns:
                    r = dom_rets.loc[d, t]
                    if not pd.isna(r):
                        daily += (dom_w / n_dom) * r
            for t in intl_prices:
                if d in intl_rets.index and t in intl_rets.columns:
                    r = intl_rets.loc[d, t]
                    if not pd.isna(r):
                        daily += (intl_w / n_intl) * r
            portfolio_ret.loc[d] = daily

    # Rebalance cost
    n_months = len(month_starts)
    total_cost = n_months * 0.12 * COST_BPS / 10000

    oos_final = portfolio_ret.index >= oos_start
    strat_ret = portfolio_ret[oos_final].dropna()
    if len(strat_ret) > 0:
        strat_ret -= total_cost / len(strat_ret)

    spy_ret_oos = spy_ret.reindex(strat_ret.index).fillna(0)

    metrics = calc_metrics(strat_ret, 'Dollar Regime Rotation', spy_ret_oos)
    regime = regime_stability(strat_ret, spy_ret_oos)
    metrics.update(regime)

    return metrics, strat_ret


# ═══════════════════════════════════════════════════════════════════════════════
# CORRELATION & ENSEMBLE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def ensemble_analysis(returns_dict, spy_ret):
    """Analyze correlation between strategies and equal-weight ensemble."""
    print("\n" + "="*80)
    print("ENSEMBLE & CORRELATION ANALYSIS")
    print("="*80)

    # Align all return series
    valid = {k: v for k, v in returns_dict.items() if v is not None and len(v) > 100}
    if len(valid) < 2:
        print("  Not enough strategies to analyze")
        return None

    combined = pd.DataFrame(valid)
    combined = combined.dropna()

    if len(combined) < 100:
        print("  Insufficient overlapping data")
        return None

    # Correlation matrix
    corr = combined.corr()
    print("\n  Strategy Correlation Matrix:")
    print(corr.round(3).to_string())

    # Equal-weight ensemble
    ensemble_ret = combined.mean(axis=1)
    spy_aligned = spy_ret.reindex(ensemble_ret.index).fillna(0)

    ensemble_metrics = calc_metrics(ensemble_ret, 'Equal-Weight Ensemble', spy_aligned)
    regime = regime_stability(ensemble_ret, spy_aligned)
    ensemble_metrics.update(regime)

    # Correlation of each strategy with SPY
    print("\n  Correlation with SPY:")
    for name, ret in valid.items():
        aligned = pd.DataFrame({'strat': ret, 'spy': spy_ret}).dropna()
        if len(aligned) > 20:
            c = aligned['strat'].corr(aligned['spy'])
            print(f"    {name}: {c:.3f}")

    return ensemble_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*80)
    print("NOVEL CROSS-ASSET SIGNAL RESEARCH")
    print(f"Walk-Forward: 12-month train / 1-month OOS, 2019-01 to 2026-08")
    print(f"Cost assumption: {COST_BPS}bps round-trip")
    print("="*80)

    prices = fetch_data()

    # Run all 4 strategies
    results = {}
    returns = {}

    m1, r1 = strategy_credit_stress(prices)
    if m1: results['credit_stress'] = m1
    if r1 is not None: returns['Credit Stress'] = r1

    m2, r2 = strategy_commodity_sector_rotation(prices)
    if m2: results['commodity_rotation'] = m2
    if r2 is not None: returns['Commodity Rotation'] = r2

    m3, r3 = strategy_yield_curve_rotation(prices)
    if m3: results['yield_curve'] = m3
    if r3 is not None: returns['Yield Curve'] = r3

    m4, r4 = strategy_dollar_regime(prices)
    if m4: results['dollar_regime'] = m4
    if r4 is not None: returns['Dollar Regime'] = r4

    # SPY benchmark
    spy_ret = prices['SPY'].pct_change().dropna()
    spy_oos = spy_ret[spy_ret.index >= '2019-01-01']
    results['spy_benchmark'] = calc_metrics(spy_oos, 'SPY Buy & Hold')

    # Ensemble
    ensemble_m = ensemble_analysis(returns, spy_ret)
    if ensemble_m:
        results['ensemble'] = ensemble_m

    # ── FINAL REPORT ──────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)

    header = f"{'Strategy':<30} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'WR%':>6} {'Regime':>8}"
    print(header)
    print("-" * 80)

    for key in ['credit_stress', 'commodity_rotation', 'yield_curve', 'dollar_regime', 'ensemble', 'spy_benchmark']:
        if key in results:
            r = results[key]
            regime_r = r.get('regime_ratio', 0)
            regime_str = f"{regime_r:.2f}" if regime_r else "N/A"
            print(f"{r['name']:<30} {r.get('cagr', 0):>7.2f} {r.get('sharpe', 0):>7.3f} {r.get('sortino', 0):>8.3f} {r.get('max_dd', 0):>7.2f} {r.get('win_rate', 0):>6.1f} {regime_str:>8}")

    print("\n" + "-"*80)
    print("REGIME STABILITY (|Sharpe_up - Sharpe_down| / max, lower = better, <0.50 = pass):")
    for key in ['credit_stress', 'commodity_rotation', 'yield_curve', 'dollar_regime', 'ensemble']:
        if key in results:
            r = results[key]
            up_s = r.get('up_market_sharpe', 0)
            dn_s = r.get('down_market_sharpe', 0)
            ratio = r.get('regime_ratio', 0)
            status = "PASS" if ratio < 0.50 else "FAIL"
            print(f"  {r['name']:<28} Up={up_s:>7.3f}  Down={dn_s:>7.3f}  Ratio={ratio:.3f}  [{status}]")

    # Excess returns vs SPY
    spy_cagr = results.get('spy_benchmark', {}).get('cagr', 0)
    print(f"\nSPY CAGR: {spy_cagr:.2f}%")
    print("Excess CAGR vs SPY:")
    for key in ['credit_stress', 'commodity_rotation', 'yield_curve', 'dollar_regime', 'ensemble']:
        if key in results:
            r = results[key]
            excess = r.get('excess_cagr', r.get('cagr', 0) - spy_cagr)
            print(f"  {r['name']:<28} {excess:>+7.2f}%")

    # Save results
    output_path = '/home/jupiter/Lvl3Quant/strategies/novel_cross_asset_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Honest assessment
    print("\n" + "="*80)
    print("HONEST ASSESSMENT")
    print("="*80)

    winners = []
    losers = []
    for key in ['credit_stress', 'commodity_rotation', 'yield_curve', 'dollar_regime']:
        if key in results:
            r = results[key]
            sharpe = r.get('sharpe', 0)
            regime = r.get('regime_ratio', 1)
            if sharpe > 0.3 and regime < 0.50:
                winners.append(r['name'])
            elif sharpe < 0:
                losers.append(r['name'])

    if winners:
        print(f"  PROMISING (Sharpe>0.3, regime-stable): {', '.join(winners)}")
    if losers:
        print(f"  NO EDGE (negative Sharpe): {', '.join(losers)}")

    not_classified = [results[k]['name'] for k in ['credit_stress', 'commodity_rotation', 'yield_curve', 'dollar_regime']
                      if k in results and results[k]['name'] not in winners and results[k]['name'] not in losers]
    if not_classified:
        print(f"  MARGINAL (weak or regime-dependent): {', '.join(not_classified)}")

    print("\n  NOTE: These are SECTOR ROTATION strategies (long-only tilts), not absolute return.")
    print("  Edge comes from RELATIVE outperformance vs equal-weight sector allocation.")
    print("  Low Sharpe may still indicate real tilt-timing skill if excess CAGR is positive.")


if __name__ == '__main__':
    main()
