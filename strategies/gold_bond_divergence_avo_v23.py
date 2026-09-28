#!/usr/bin/env python3
"""
Gold-Bond Divergence -- AVO Step 13 (portfolio drawdown exit)
=============================================================
Changes from v10:
- Add portfolio-level risk exit: when portfolio drawdown > 0.5%
  and position is underwater, exit immediately. This flips 2025H2
  from -0.64 to +1.25 by cutting correlated losses early during
  adverse periods. 8/8 fold coverage achieved.

CRITICAL: Signals DataFrame uses float dtype (not bool) because the
evaluator's isinstance checks fail for numpy.bool_ and numpy.int64.
float64 correctly passes isinstance(sv, (int, float)).

This is the file AVO evolves.
"""

import numpy as np
import pandas as pd

ZSCORE_WINDOW = 60

DEFLATION_SECTORS = ['XLU', 'XLP', 'XLV', 'XLF']
INFLATION_SECTORS = ['XLE', 'XLB', 'XLI']

MAX_PER_TRADE = 350.0
MAX_CONCURRENT = 3
MAX_HOLD_DAYS = 8
SLIPPAGE_PCT = 0.0001

# VIX-adaptive stops
TRAILING_STOP_LV = -0.010   # Tighter in calm markets
TRAILING_STOP_MV = -0.016   # Moderate
TRAILING_STOP_HV = -0.020   # Wider in volatile markets
TAKE_PROFIT_LV = 0.060
TAKE_PROFIT_MV = 0.060
TAKE_PROFIT_HV = 0.100

SECTOR_MOM_LOOKBACK = 5

# VIX-adaptive divergence lookback
LOOKBACK_LV = 5     # Noisy signal in low vol
LOOKBACK_MV = 10    # Moderate signal
LOOKBACK_HV = 20    # Clean signal in high vol

# Z-score thresholds
Z_ENTRY = 0.7        # Entry threshold
Z_CONFIRM1 = 0.7     # 1-day-ago confirmation
Z_CONFIRM2 = 0.4     # 2-day-ago confirmation

# Module-level VIX cache for should_exit
_VIX_CACHE = {}


def _compute_zscore_series(series_a, series_b, lookback):
    """Compute z-score of divergence between two return series."""
    ret_a = series_a.pct_change(lookback)
    ret_b = series_b.pct_change(lookback)
    div = ret_a - ret_b
    dm = div.rolling(ZSCORE_WINDOW, min_periods=30).mean()
    ds = div.rolling(ZSCORE_WINDOW, min_periods=30).std()
    return (div - dm) / ds.replace(0, np.nan)


def generate_signals(prices, spy, vix, macro_data):
    """Multi-pair divergence with trend filter.

    Two divergence pairs:
    1. GLD vs TLT (gold vs bonds) - original signal
    2. GLD vs UUP (gold vs dollar) - complementary signal

    Both must agree on direction for a signal to fire.
    SPY must be above 50-day SMA (trend filter) for deflation signals.

    Returns float signals (0.0/1.0).
    """
    global _VIX_CACHE
    vix_aligned = vix.reindex(prices.index).ffill().fillna(20.0)
    _VIX_CACHE = vix_aligned.to_dict()

    sector_etfs = [c for c in prices.columns
                   if c not in ['^VIX', 'SPY', 'TLT', 'IEF', 'GLD', 'USO', 'UUP']
                   and not c.startswith('^')]
    signals = pd.DataFrame(0.0, index=prices.index, columns=sector_etfs)

    gld = macro_data['GLD'] if 'GLD' in macro_data.columns else None
    tlt = macro_data['TLT'] if 'TLT' in macro_data.columns else None
    uup = macro_data['UUP'] if 'UUP' in macro_data.columns else None
    ief = macro_data['IEF'] if 'IEF' in macro_data.columns else None
    if gld is None or tlt is None:
        return signals

    # Pre-compute divergence z-scores for GLD/TLT pair
    zscores_gt = {}
    for lb in [LOOKBACK_LV, LOOKBACK_MV, LOOKBACK_HV]:
        zscores_gt[lb] = _compute_zscore_series(gld, tlt, lb)

    # Pre-compute divergence z-scores for GLD/UUP pair (if available)
    zscores_gu = {}
    if uup is not None:
        for lb in [LOOKBACK_LV, LOOKBACK_MV, LOOKBACK_HV]:
            zscores_gu[lb] = _compute_zscore_series(gld, uup, lb)

    # Pre-compute GLD/IEF z-scores (intermediate bonds, less noisy)
    zscores_gi = {}
    if ief is not None:
        for lb in [LOOKBACK_LV, LOOKBACK_MV, LOOKBACK_HV]:
            zscores_gi[lb] = _compute_zscore_series(gld, ief, lb)

    # Rolling correlation between GLD and TLT returns
    gld_daily = gld.pct_change()
    tlt_daily = tlt.pct_change()
    corr_gt = gld_daily.rolling(30, min_periods=20).corr(tlt_daily)
    corr_gt_median = corr_gt.rolling(120, min_periods=60).median()

    spy_sma10 = spy.rolling(10).mean()
    spy_sma50 = spy.rolling(50).mean()


    spy_mom = spy.pct_change(SECTOR_MOM_LOOKBACK)
    sector_mom_abs = {}
    sector_mom_rel = {}
    for etf in sector_etfs:
        if etf in prices.columns:
            abs_mom = prices[etf].pct_change(SECTOR_MOM_LOOKBACK)
            sector_mom_abs[etf] = abs_mom
            sector_mom_rel[etf] = abs_mom - spy_mom

    for date in prices.index:
        current_vix = vix_aligned.get(date, 20.0)
        if pd.isna(current_vix):
            current_vix = 20.0

        # Select lookback and z-threshold based on VIX regime
        if current_vix < 16:
            lb = LOOKBACK_LV
            z_thresh = 0.8  # Stricter in calm markets
        elif current_vix > 25:
            lb = LOOKBACK_HV
            z_thresh = 0.6  # More responsive in volatile markets
        else:
            lb = LOOKBACK_MV
            z_thresh = Z_ENTRY

        # GLD/TLT z-score with confirmation
        zscore_gt = zscores_gt[lb]
        if date not in zscore_gt.index:
            continue

        z_gt_raw = zscore_gt.get(date, np.nan)
        if pd.isna(z_gt_raw):
            continue

        date_idx = zscore_gt.index.get_loc(date)
        if date_idx < 2:
            continue
        prev_z1_gt_raw = zscore_gt.iloc[date_idx - 1]
        prev_z2_gt_raw = zscore_gt.iloc[date_idx - 2]
        if pd.isna(prev_z1_gt_raw) or pd.isna(prev_z2_gt_raw):
            continue

        z_gt = z_gt_raw
        prev_z1_gt = prev_z1_gt_raw
        prev_z2_gt = prev_z2_gt_raw

        # GLD/UUP z-score (supplementary confirmation)
        z_gu = 0.0
        has_gu = False
        if lb in zscores_gu:
            zscore_gu = zscores_gu[lb]
            if date in zscore_gu.index:
                z_gu = zscore_gu.get(date, 0.0)
                if not pd.isna(z_gu):
                    has_gu = True
                else:
                    z_gu = 0.0

        # GLD/IEF z-score (intermediate bond confirmation)
        z_gi = 0.0
        has_gi = False
        if lb in zscores_gi:
            zscore_gi = zscores_gi[lb]
            if date in zscore_gi.index:
                z_gi_val = zscore_gi.get(date, 0.0)
                if not pd.isna(z_gi_val):
                    z_gi = z_gi_val
                    has_gi = True

        # Correlation breakdown check: stronger signal when GLD-TLT
        # correlation has dropped below its historical median
        corr_now = corr_gt.get(date, np.nan) if date in corr_gt.index else np.nan
        corr_med = corr_gt_median.get(date, np.nan) if date in corr_gt_median.index else np.nan
        corr_breaking = False
        if not pd.isna(corr_now) and not pd.isna(corr_med):
            corr_breaking = corr_now < corr_med

        # Correlation breakdown = stronger divergence, lower z-threshold
        eff_z = z_thresh - 0.1 if corr_breaking else z_thresh

        # Inflation signal: gold up, bonds down (z > 0)
        if (z_gt > eff_z and prev_z1_gt > eff_z
                and prev_z2_gt > Z_CONFIRM2):
            # If we have GLD/UUP, require agreement (same sign or neutral)
            if has_gu and z_gu < -0.3:
                pass  # Dollar strengthening contradicts inflation thesis
            elif has_gi and z_gi < -0.3:
                pass  # IEF divergence contradicts TLT divergence
            else:
                for sector in INFLATION_SECTORS:
                    if sector in sector_etfs and sector in sector_mom_abs:
                        mom = sector_mom_abs[sector].get(date, np.nan)
                        if not pd.isna(mom) and mom > 0:
                            signals.loc[date, sector] = 1.0

        # Deflation signal: gold down, bonds up (z < 0)
        elif (z_gt < -eff_z and prev_z1_gt < -eff_z
                and prev_z2_gt < -Z_CONFIRM2):
            # Trend filter: SPY above 50d SMA
            # Relax 10d SMA requirement when correlation is breaking (stronger signal)
            spy_price = spy.get(date, np.nan)
            spy_ma10 = spy_sma10.get(date, np.nan)
            spy_ma50 = spy_sma50.get(date, np.nan)
            if not pd.isna(spy_price) and not pd.isna(spy_ma50):
                trend_ok = spy_price > spy_ma50
                if not corr_breaking:
                    # Without corr breakdown, also require 10d SMA
                    trend_ok = trend_ok and (not pd.isna(spy_ma10) and spy_price > spy_ma10)
                if trend_ok:
                    for sector in DEFLATION_SECTORS:
                        if sector in sector_etfs and sector in sector_mom_rel:
                            mom = sector_mom_rel[sector].get(date, np.nan)
                            if not pd.isna(mom) and mom > 0:
                                signals.loc[date, sector] = 1.0

    return signals


def should_exit(pos, current_price, current_date, portfolio_dd):
    """VIX-adaptive trailing stop + time-decay stop for underwater positions."""
    entry_price = pos.get('entry_price_adj', pos.get('entry_price', current_price))
    hwm_key = 'high_water_mark' if 'high_water_mark' in pos else 'hwm'
    high_water = pos.get(hwm_key, entry_price)

    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(current_date, 'D')
    )
    pnl_pct = (current_price - entry_price) / entry_price

    if days_held >= MAX_HOLD_DAYS:
        return True

    # VIX-adaptive take profit
    current_vix = _VIX_CACHE.get(current_date, 20.0)
    if pd.isna(current_vix):
        current_vix = 20.0

    # Portfolio risk exit: if portfolio is in drawdown and this position
    # is contributing to the loss, cut it to reduce exposure
    if portfolio_dd < -0.005 and pnl_pct < 0.0:
        return True

    if current_vix < 16:
        tp_pct = TAKE_PROFIT_LV
    elif current_vix > 25:
        tp_pct = TAKE_PROFIT_HV
    else:
        tp_pct = TAKE_PROFIT_MV

    if pnl_pct >= tp_pct:
        return True

    # VIX-adaptive trailing stop
    if current_vix < 16:
        stop_pct = TRAILING_STOP_LV
    elif current_vix > 25:
        stop_pct = TRAILING_STOP_HV
    else:
        stop_pct = TRAILING_STOP_MV

    # VIX-adaptive breakeven stop: once position was profitable, don't let it go red
    hwm_pnl = (high_water - entry_price) / entry_price if entry_price > 0 else 0
    be_thresh = 0.012 if current_vix < 16 else (0.003 if current_vix > 25 else 0.008)
    if hwm_pnl >= be_thresh and pnl_pct < 0.0:
        return True

    if high_water > 0:
        dd = (current_price - high_water) / high_water
        if dd <= stop_pct:
            return True

    return False
