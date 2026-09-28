#!/usr/bin/env python3
"""
Insider Momentum Sector Rotation — v6 (Insider Gate)
=====================================================
Defensive sector dip-buy WITH insider confirmation gate.
Hypothesis: requiring positive insider z-score as GATE (not just boost)
asymmetrically helps regime balance because:
- High-vol: insider buying is rare & informative → keeps best trades
- Low-vol: insider buying is common & noisy → filters randomly → reduces Sharpe

Entry: Buy defensive sector ETF when:
  1. Sector dipped >= 1.8% in 5 days
  2. Not in major downtrend (30d return > -10%)
  3. Today's return not below -1.2% (dip deceleration)
  4. Sector insider buying z-score >= 0 (must have some insider conviction)

Exit: trail -2%, TP +3.5%, max hold 3 days, cut losers after 1 day
"""

import numpy as np
import pandas as pd

# -- Configuration -----------------------------------------------------------
TRADEABLE_SECTORS = ['XLU', 'XLP', 'XLV']
SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP',
               'XLU', 'XLRE', 'XLB']
BENCHMARK = 'SPY'

# Dip-buy parameters
DIP_LOOKBACK = 5
DIP_THRESHOLD = -0.018
TREND_LOOKBACK = 30
TREND_MIN = -0.10

# Insider signal parameters (GATE + BOOST)
INSIDER_LOOKBACK = 35          # tuned lookback for regime balance
INSIDER_GATE_THRESHOLD = 0.0   # must be non-negative
INSIDER_BOOST_THRESHOLD = 0.5  # conviction boost threshold

# Trade management
MAX_HOLD_DAYS = 3
MAX_PER_TRADE = 2000.0         # smaller positions for lower per-trade risk
MAX_CONCURRENT = 1
SLIPPAGE_PCT = 0.0001

# Exit parameters
TRAILING_STOP_PCT = -0.02      # wider trail for bigger high-vol moves
TAKE_PROFIT_PCT = 0.035        # wider TP
UNDERWATER_EXIT_DAYS = 1

# GICS sector mapping for insider aggregation
SECTOR_TICKERS = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM', 'ORCL', 'AMD', 'ADBE', 'CSCO', 'ACN',
            'INTC', 'IBM', 'INTU', 'NOW', 'QCOM', 'TXN', 'AMAT', 'MU', 'LRCX', 'KLAC'],
    'XLF': ['JPM', 'BRK-B', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SPGI', 'BLK',
            'AXP', 'C', 'SCHW', 'CB', 'MMC', 'PGR', 'ICE', 'AON', 'CME', 'USB'],
    'XLV': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT', 'DHR', 'BMY',
            'AMGN', 'ISRG', 'SYK', 'GILD', 'MDT', 'VRTX', 'CI', 'ELV', 'BSX', 'REGN'],
    'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'MPC', 'EOG', 'PSX', 'VLO', 'WMB', 'OKE',
            'HES', 'HAL', 'DVN', 'FANG', 'BKR', 'TRGP', 'OXY', 'KMI', 'CTRA', 'APA'],
    'XLI': ['GE', 'CAT', 'UNP', 'HON', 'RTX', 'BA', 'DE', 'LMT', 'UPS', 'ADP',
            'MMM', 'EMR', 'ITW', 'ETN', 'GD', 'TDG', 'NSC', 'WM', 'PH', 'CARR'],
    'XLC': ['META', 'GOOG', 'GOOGL', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'CHTR', 'TMUS'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG',
            'MAR', 'ORLY', 'GM', 'F', 'ROST', 'DHI', 'LEN', 'AZO', 'EBAY', 'YUM'],
    'XLP': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
            'GIS', 'SYY', 'HSY', 'K', 'KHC', 'STZ', 'MKC', 'CHD', 'CAG', 'CLX'],
    'XLU': ['NEE', 'SO', 'DUK', 'D', 'AEP', 'SRE', 'XEL', 'EXC', 'WEC', 'ED',
            'ES', 'AWK', 'DTE', 'ETR', 'PEG', 'FE', 'PPL', 'CMS', 'AES', 'ATO'],
    'XLRE': ['PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'DLR', 'O', 'WELL', 'SPG', 'VICI',
             'AVB', 'EQR', 'ARE', 'MAA', 'UDR', 'VTR', 'HST', 'KIM', 'REG', 'BXP'],
    'XLB': ['LIN', 'SHW', 'APD', 'FCX', 'ECL', 'NEM', 'DOW', 'NUE', 'VMC', 'MLM',
            'PPG', 'DD', 'ALB', 'CF', 'IFF', 'CTVA', 'EMN', 'CE', 'FMC', 'MOS'],
}


def generate_signals(prices, spy, vix, insider_data=None, cross_asset=None):
    """Generate defensive dip-buy signals with insider gate."""
    signals = pd.DataFrame(False, index=prices.index, columns=SECTOR_ETFS)

    # Compute returns
    dip_ret = {}
    trend_ret = {}
    daily_ret = {}
    for etf in TRADEABLE_SECTORS:
        if etf in prices.columns:
            dip_ret[etf] = prices[etf].pct_change(DIP_LOOKBACK)
            trend_ret[etf] = prices[etf].pct_change(TREND_LOOKBACK)
            daily_ret[etf] = prices[etf].pct_change(1)

    # Compute sector insider z-scores
    sector_insider_z = {}
    has_insider = insider_data is not None and not insider_data.empty
    if has_insider:
        sector_insider_z = _compute_sector_insider_z(insider_data, prices.index)

    for date in prices.index:
        candidates = []
        for etf in TRADEABLE_SECTORS:
            # Dip check
            if etf not in dip_ret or date not in dip_ret[etf].index:
                continue
            dv = dip_ret[etf].get(date, np.nan)
            if pd.isna(dv) or dv > DIP_THRESHOLD:
                continue

            # Trend guard
            if etf not in trend_ret or date not in trend_ret[etf].index:
                continue
            tv = trend_ret[etf].get(date, np.nan)
            if pd.isna(tv) or tv < TREND_MIN:
                continue

            # Dip deceleration: skip if today alone drops > 1.2%
            # (catching accelerating sell-off — wait for stabilization)
            if etf in daily_ret and date in daily_ret[etf].index:
                dr = daily_ret[etf].get(date, 0.0)
                if not pd.isna(dr) and dr < -0.012:
                    continue

            # Insider GATE: require non-negative z-score
            if has_insider:
                iz = sector_insider_z.get(etf, {}).get(date, -1.0)
                if iz < INSIDER_GATE_THRESHOLD:
                    continue
            else:
                iz = 0.0

            # Score = dip magnitude * trend strength, boosted by insider conviction
            trend_bonus = max(0.0, tv + 0.05) * 10.0  # positive trend = higher score
            score = abs(dv) * (1.0 + trend_bonus)
            if iz >= INSIDER_BOOST_THRESHOLD:
                score *= (1.0 + 2.0 * iz)  # aggressive boost for high conviction

            candidates.append((etf, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        for etf, _ in candidates[:MAX_CONCURRENT]:
            signals.loc[date, etf] = True

    return signals


def _compute_sector_insider_z(insider_data, dates):
    """Aggregate stock insider buys to sector level, compute rolling z-scores."""
    sector_insider_z = {}

    for etf, tickers in SECTOR_TICKERS.items():
        sector_df = insider_data[insider_data['ticker'].isin(tickers)].copy()
        if sector_df.empty:
            sector_insider_z[etf] = {}
            continue

        sector_df['date'] = pd.to_datetime(sector_df['date'])
        # Use net insider USD (buys - sells) for true conviction
        if 'net_insider_usd' in sector_df.columns:
            daily_buys = sector_df.groupby('date')['net_insider_usd'].sum()
        else:
            daily_buys = sector_df.groupby('date')['gross_buy_usd'].sum()
        daily_buys = daily_buys.reindex(dates, fill_value=0.0)

        rolling_sum = daily_buys.rolling(INSIDER_LOOKBACK, min_periods=10).sum()
        rolling_mean = rolling_sum.rolling(252, min_periods=60).mean()
        rolling_std = rolling_sum.rolling(252, min_periods=60).std()

        z_score = (rolling_sum - rolling_mean) / rolling_std.clip(lower=1e-6)
        z_score = z_score.fillna(0.0)

        sector_insider_z[etf] = z_score.to_dict()

    return sector_insider_z


def should_exit(pos, current_price, current_date, portfolio_dd):
    """Exit with winning continuation: cut underwater positions after 1 day."""
    days_held = np.busday_count(
        np.datetime64(pos['entry_date'], 'D'),
        np.datetime64(current_date, 'D'))

    entry_price = pos.get('entry_price_adj', pos.get('entry_price', current_price))
    pnl_pct = (current_price - entry_price) / entry_price

    hwm_key = 'hwm' if 'hwm' in pos else 'high_water'
    if current_price > pos.get(hwm_key, entry_price):
        pos[hwm_key] = current_price

    high_water = pos.get(hwm_key, entry_price)
    drawdown_from_high = (current_price - high_water) / high_water

    # Winning continuation: cut losers after 1 day
    if days_held >= UNDERWATER_EXIT_DAYS and pnl_pct < 0:
        return True

    if days_held >= MAX_HOLD_DAYS:
        return True
    if pnl_pct >= TAKE_PROFIT_PCT:
        return True
    if drawdown_from_high <= TRAILING_STOP_PCT:
        return True

    return False
