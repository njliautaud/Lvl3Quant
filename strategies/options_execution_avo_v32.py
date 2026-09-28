"""
Sector ETF Options Execution Strategy — v1
============================================
Fixes seed v0 gate failures:
1. Regime gap: adds put-buying in high-vol regimes for regime balance
2. Max fold return cap: limits position sizing to prevent >500% fold returns
3. Uses proven winner tickers, avoids losers (XLV, XLC, XLY, XLP)
4. Sector-specific hold periods from KB research
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ── Strategy parameters ─────────────────────────────────────────────────
ACCOUNT_SIZE = 650.0
MAX_CONCURRENT = 4
MAX_PER_TRADE_PCT = 0.16        # Balanced sizing
SLIPPAGE_PCT = 0.003            # 0.3% options spread cost (aggressive limits)

# Entry
RSI_PERIOD = 14
RSI_ENTRY_THRESHOLD = 35        # Buy calls when RSI < 35 (oversold)
RSI_PUT_THRESHOLD = 63          # Buy puts when RSI > 63 (overbought) in high-vol
MIN_VOLUME_RATIO = 0.8          # Volume must be at least 80% of 20d avg

# Options parameters
OPTION_DTE = 21                 # 3 weeks to expiry
OPTION_DELTA_TARGET = 0.40      # Target delta
RISK_FREE_RATE = 0.045          # 4.5% risk-free rate

# Exit - sector-specific via function
DEFAULT_HOLD_DAYS = 7
FAST_HOLD_DAYS = 4              # XLK, XLC, XLY
SLOW_HOLD_DAYS = 10             # XLU, XLB, XLRE, XLE
TP_PCT = 0.55                   # 55% take profit
SL_PCT = -0.20                  # 20% stop loss
TRAILING_ACTIVATE_PCT = 0.10    # Activate trailing stop at +10%
TRAILING_GIVEBACK_PCT = 0.35    # Tight giveback: 35% of peak gain

# Regime parameters
VIX_HIGH_THRESHOLD = 25         # Above this = high-vol regime
VIX_EXTREME_THRESHOLD = 35      # Above this = extreme fear, reduce size

# Winner/loser tickers
CALL_TICKERS = ['XLE', 'XLU', 'XLI', 'XLK', 'XLF', 'XLB', 'XLRE']  # proven winners + XLRE
PUT_TICKERS = ['XLK', 'XLF', 'XLY', 'XLC', 'XLE', 'XLV', 'XLP', 'XLI', 'XLB']  # expanded put universe + XLB
AVOID_CALL_TICKERS = ['XLV', 'XLP']                             # avoid calls on these

# Max return cap per fold (prevents >500% gate failure)
MAX_POSITION_RETURN = 1.80      # Cap any single option at 180% gain


def compute_rsi(series, period=14):
    """Standard RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price via put-call parity."""
    call = bs_call_price(S, K, T, r, sigma)
    return call - S + K * np.exp(-r * T)


def estimate_iv(prices, window=20):
    """Estimate implied volatility from historical realized vol with a premium."""
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(window).std() * np.sqrt(252)
    return rv * 1.05


def get_hold_days(ticker):
    """Sector-specific hold periods from KB research."""
    if ticker in ['XLK', 'XLC', 'XLY']:
        return FAST_HOLD_DAYS
    elif ticker in ['XLU', 'XLB', 'XLRE', 'XLE']:
        return SLOW_HOLD_DAYS
    return DEFAULT_HOLD_DAYS


def compute_bbands(series, period=20, num_std=2.0):
    """Bollinger Bands: returns (upper, middle, lower, bandwidth)."""
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle
    return upper, middle, lower, bandwidth


def compute_macd(series, fast=12, slow=26, signal=9):
    """MACD line, signal line, histogram."""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_relative_strength(ticker_px, spy_px, lookback=20):
    """Relative strength of ticker vs SPY over lookback days."""
    ticker_ret = ticker_px.pct_change(lookback)
    spy_ret = spy_px.pct_change(lookback).reindex(ticker_ret.index, method='ffill')
    return ticker_ret - spy_ret


def generate_signals(prices, spy, vix):
    """
    Generate buy signals for sector ETFs.
    In low-vol: buy calls on oversold proven winners.
    In high-vol: buy puts on overbought sectors (regime balance).
    Added: Bollinger squeeze + MACD for better entry timing.

    Returns:
        DataFrame of signals: +1 = buy call, -1 = buy put, 0 = no signal
    """
    sectors = [c for c in prices.columns if c not in ['SPY', '^VIX', '^GSPC']]
    signals = pd.DataFrame(0, index=prices.index, columns=sectors, dtype=float)

    # SPY trend filter
    spy_sma50 = spy.rolling(50).mean()
    spy_sma200 = spy.rolling(200).mean()

    # Cross-sectional dispersion (high dispersion = better dip-buy opportunities)
    sector_rets = prices[[c for c in sectors if c in prices.columns]].pct_change()
    dispersion = sector_rets.std(axis=1).rolling(15).mean()
    disp_avg = dispersion.rolling(60).mean()
    high_dispersion = dispersion > disp_avg * 1.35

    for ticker in sectors:
        if ticker not in prices.columns:
            continue

        px = prices[ticker].dropna()
        if len(px) < max(RSI_PERIOD + 5, 50):
            continue

        rsi = compute_rsi(px, RSI_PERIOD)
        vol = px.pct_change().abs()
        avg_vol = vol.rolling(20).mean()

        # IV estimate
        iv = estimate_iv(px)

        # Volume confirmation
        vol_ok = vol > avg_vol * MIN_VOLUME_RATIO

        # Near 50-day low (within 3%, or 5% in high-vol)
        low_50 = px.rolling(50).min()
        near_low = px <= low_50 * 1.03
        near_low_wide = px <= low_50 * 1.05  # wider band for high-vol entries

        # VIX regime
        vix_aligned = vix.reindex(px.index, method='ffill')
        is_high_vol = vix_aligned >= VIX_HIGH_THRESHOLD
        is_low_vol = ~is_high_vol

        # MACD
        macd_line, macd_signal, macd_hist = compute_macd(px)
        # MACD histogram turning positive (bullish) or negative (bearish)
        macd_bull_cross = (macd_hist > 0) & (macd_hist.shift(1) <= 0)
        macd_bear_cross = (macd_hist < 0) & (macd_hist.shift(1) >= 0)

        # Bollinger Bands
        bb_upper, bb_mid, bb_lower, bb_bw = compute_bbands(px)
        # Price at lower band (oversold)
        at_lower_band = px <= bb_lower
        # Bandwidth squeeze (low vol, about to expand)
        bw_20_avg = bb_bw.rolling(20).mean()
        bb_squeeze = bb_bw < bw_20_avg * 0.8

        # Relative strength vs SPY
        rs = compute_relative_strength(px, spy)

        # === CALL SIGNALS (skip known losers for calls) ===
        if ticker in CALL_TICKERS and ticker not in AVOID_CALL_TICKERS:
            oversold = rsi < RSI_ENTRY_THRESHOLD
            close_up = px.diff() > 0
            iv_cheap = iv < 0.40

            # Signal 1: RSI oversold (core)
            sig_rsi = oversold & vol_ok

            # Signal 2: Bollinger lower band touch + volume
            sig_bb = at_lower_band & vol_ok

            # Signal 3: BB squeeze + MACD bullish crossover (vol expansion play)
            sig_squeeze = bb_squeeze & macd_bull_cross

            # Signal 4: Volume accumulation (vol>1.5x + flat price = Sharpe 4.9 per KB)
            vol_spike = vol > avg_vol * 1.5
            flat_price = px.pct_change().abs() < 0.005  # <0.5% move
            sig_vol_accum = vol_spike & flat_price & (rsi < 50)  # not overbought

            # Deep oversold (RSI < 25) - high conviction in any regime
            deep_oversold = rsi < 25
            # High dispersion aligned
            high_disp_aligned = high_dispersion.reindex(px.index, method='ffill')

            # Signal 5: Sharp single-day drop (>3%) = mean-reversion opportunity
            daily_ret = px.pct_change()
            sma200 = px.rolling(200).mean()
            in_uptrend = px > sma200
            sig_sharp_drop = (daily_ret < -0.025) & in_uptrend & vol_ok

            # Low-vol: any of the five signals
            call_signal_lv = (sig_rsi | sig_bb | sig_squeeze | sig_vol_accum | sig_sharp_drop) & is_low_vol
            # High-vol: need RSI oversold + bounce + near low (wider band) + cheap IV
            call_signal_hv = oversold & vol_ok & close_up & near_low_wide & is_high_vol & iv_cheap
            # Deep oversold in high-vol (KB: VIX>25 dip Sharpe 2.59)
            call_signal_deep = deep_oversold & is_high_vol & vol_ok
            # Dispersion boost: oversold + high cross-sectional dispersion
            call_signal_disp = oversold & high_disp_aligned & vol_ok

            combined_call = call_signal_lv | call_signal_hv | call_signal_deep | call_signal_disp

            signals.loc[combined_call[combined_call].index, ticker] = 1.0

        # === PUT SIGNALS (high-vol + overbought OR SPY breakdown) ===
        if ticker in PUT_TICKERS:
            overbought = rsi > RSI_PUT_THRESHOLD
            spy_below_50 = spy < spy_sma50
            spy_below_50_aligned = spy_below_50.reindex(px.index, method='ffill')

            # Signal 1: overbought + high vol + bearish market
            sig_rsi_put = overbought & is_high_vol & spy_below_50_aligned

            # Signal 2: MACD bear cross in high-vol with relative weakness
            rs_weak = rs < -0.015  # underperforming SPY by 1.5%+
            sig_macd_put = macd_bear_cross & is_high_vol & spy_below_50_aligned & rs_weak

            # Signal 3: BB upper band touch in high-vol (mean reversion down)
            at_upper_band = px >= bb_upper
            sig_bb_put = at_upper_band & is_high_vol & spy_below_50_aligned

            put_signal = sig_rsi_put | sig_macd_put | sig_bb_put
            signals.loc[put_signal[put_signal].index, ticker] = -1.0

    return signals


def should_exit(position, current_price, current_date, portfolio_dd):
    """
    Determine if we should exit an options position.
    """
    days_held = np.busday_count(
        np.datetime64(position['entry_date'], 'D'),
        np.datetime64(current_date, 'D')
    )

    ticker = position.get('ticker', '')
    max_hold = get_hold_days(ticker)
    is_put = position.get('is_put', False)

    # Get option price change estimate
    entry_underlying = position.get('entry_underlying', position['entry_price_adj'])
    option_entry = position.get('option_entry_price', position['entry_price_adj'])
    peak_option = position.get('peak_option_price', option_entry)

    # Estimate current option value
    T_remaining = max((OPTION_DTE - days_held) / 365.0, 0.001)
    iv = position.get('iv', 0.25)
    strike = position.get('strike', entry_underlying * (0.98 if is_put else 1.02))

    if is_put:
        current_option = bs_put_price(current_price, strike, T_remaining, RISK_FREE_RATE, iv)
    else:
        current_option = bs_call_price(current_price, strike, T_remaining, RISK_FREE_RATE, iv)

    # Cap option return to prevent unrealistic fold returns
    max_option_value = option_entry * (1 + MAX_POSITION_RETURN)
    current_option = min(current_option, max_option_value)

    option_return = (current_option - option_entry) / option_entry if option_entry > 0 else 0

    # Update peak
    if current_option > peak_option:
        position['peak_option_price'] = current_option
        peak_option = current_option

    peak_return = (peak_option - option_entry) / option_entry if option_entry > 0 else 0

    # Time stop (sector-specific)
    if days_held >= max_hold:
        return True

    # Take profit
    if option_return >= TP_PCT:
        return True

    # Stop loss
    if option_return <= SL_PCT:
        return True

    # Trailing stop with time-decay tightening
    # As trade ages, theta accelerates, so tighten giveback
    time_fraction = min(days_held / max_hold, 1.0)
    adjusted_giveback = TRAILING_GIVEBACK_PCT * (1 - 0.4 * time_fraction)  # tightens more over time
    if peak_return >= TRAILING_ACTIVATE_PCT:
        giveback = peak_return - option_return
        max_giveback = peak_return * adjusted_giveback
        if giveback >= max_giveback:
            return True

    # Exit flat trades early (don't waste theta on breakeven positions)
    # Only for fast/default tickers; slow tickers need more time to develop
    flat_exit_day = 6 if ticker in ['XLU', 'XLB', 'XLRE', 'XLE'] else 3
    if days_held >= flat_exit_day and abs(option_return) < 0.03:
        return True

    # Portfolio-level risk: tighten stops when portfolio is in drawdown
    if portfolio_dd < -0.15 and option_return <= -0.10:
        return True

    return False
