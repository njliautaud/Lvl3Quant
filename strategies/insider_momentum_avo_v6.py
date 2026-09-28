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
  1. Sector dipped >= 1.5% in 5 days
  2. Not in major downtrend (30d return > -10%)
  3. Sector insider buying z-score > 0 (must have some insider conviction)

Exit: trail -1.5%, TP +2.5%, max hold 3 days, winning continuation 1 day
"""

import os
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

# Google Trends attention filter
GTRENDS_DIR = '/home/jupiter/Lvl3Quant/data/feature_store/google_trends'
GTRENDS_ATTENTION_GATE = 2.0   # skip entry when sector attention z-score > this
# Trade management
MAX_HOLD_DAYS = 3              # max hold for non-winning trades
MAX_HOLD_WINNERS = 4           # let winners run one extra day
MAX_PER_TRADE = 2000.0         # smaller positions for lower per-trade risk
MAX_CONCURRENT = 1
SLIPPAGE_PCT = 0.0

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

    # Compute sector attention z-scores from Google Trends
    sector_attention = _compute_sector_attention(prices.index)

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

            # Attention GATE: skip when sector has abnormally high search
            # attention (news-driven dip likely to continue)
            if etf in sector_attention:
                att = sector_attention[etf].get(date, 0.0)
                if not pd.isna(att) and att > GTRENDS_ATTENTION_GATE:
                    continue

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

        # Penalize z-score when it's driven by a single large transaction
        # rather than broad-based buying across multiple insiders
        if 'n_buyers' in sector_df.columns:
            daily_n = sector_df.groupby('date')['n_buyers'].sum()
            daily_n = daily_n.reindex(dates, fill_value=0.0)
            rolling_n = daily_n.rolling(INSIDER_LOOKBACK, min_periods=5).sum()
            # Dampen z-score when fewer than 3 buyers in window
            breadth_factor = (rolling_n.clip(upper=5) / 5.0).clip(lower=0.3)
            z_score = z_score * breadth_factor

        sector_insider_z[etf] = z_score.to_dict()

    return sector_insider_z


def _compute_sector_attention(dates):
    """Aggregate Google Trends attention z-scores to sector level."""
    sector_attention = {}
    if not os.path.isdir(GTRENDS_DIR):
        return sector_attention

    for etf, tickers in SECTOR_TICKERS.items():
        all_z = []
        for ticker in tickers:
            fpath = os.path.join(GTRENDS_DIR, f'{ticker}.parquet')
            if not os.path.exists(fpath):
                continue
            try:
                df = pd.read_parquet(fpath)
                df['date'] = pd.to_datetime(df['date'])
                # Weekly data — forward-fill to daily
                z = df.set_index('date')['search_interest_z'].reindex(dates, method='ffill')
                all_z.append(z)
            except Exception:
                continue
        if all_z:
            # Sector attention = mean z-score across constituents
            sector_z = pd.concat(all_z, axis=1).mean(axis=1)
            sector_attention[etf] = sector_z.to_dict()
        else:
            sector_attention[etf] = {}

    return sector_attention


def should_exit(pos, current_price, current_date, portfolio_dd):
    """Exit with winning continuation: cut underwater positions after 1 day.
    Uses portfolio drawdown to tighten exits when strategy is drawing down."""
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

    # Asymmetric hold: only clear winners hold longer
    max_hold = MAX_HOLD_WINNERS if pnl_pct > 0.025 else MAX_HOLD_DAYS
    if days_held >= max_hold:
        return True
    # Wider TP for extended-hold winners: let them capture the full move
    tp = 1.0 if (days_held >= MAX_HOLD_DAYS and pnl_pct > 0.025) else TAKE_PROFIT_PCT
    if pnl_pct >= tp:
        return True
    if drawdown_from_high <= TRAILING_STOP_PCT:
        return True

    # Portfolio-level risk: exit breakeven positions when in drawdown
    if portfolio_dd is not None and portfolio_dd < -0.02 and days_held >= 2 and pnl_pct < 0.005:
        return True

    return False
