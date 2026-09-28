"""
Alpaca Options Pricing Module
Real market quotes for options pricing, replacing BS synthetic pricing.
Uses alpaca-py SDK with paper trading credentials.

Usage:
    from alpaca_options_pricing import get_option_chain, get_option_quote, get_real_premium_or_fallback
"""

import os
import time
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest
from alpaca.trading.enums import ContractType

logger = logging.getLogger(__name__)

# --- Credentials ---
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

# --- Cache ---
_chain_cache: Dict[str, Tuple[float, pd.DataFrame]] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes

# --- Rate limiting ---
_request_times: List[float] = []
MAX_REQUESTS_PER_MINUTE = 200


def _get_client() -> OptionHistoricalDataClient:
    """Get or create the Alpaca option data client."""
    return OptionHistoricalDataClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
    )


def _rate_limit():
    """Enforce rate limit of 200 requests/minute."""
    global _request_times
    now = time.time()
    # Remove timestamps older than 60s
    _request_times = [t for t in _request_times if now - t < 60]
    if len(_request_times) >= MAX_REQUESTS_PER_MINUTE:
        sleep_time = 60 - (now - _request_times[0]) + 0.1
        if sleep_time > 0:
            logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
    _request_times.append(time.time())


def _is_market_hours() -> bool:
    """Check if we're within US market hours (9:30-16:00 ET)."""
    try:
        import pytz
        et = pytz.timezone('US/Eastern')
        now = datetime.now(et)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now <= market_close and now.weekday() < 5
    except ImportError:
        # Fallback: assume market is open
        return True


def _get_nearest_weekly_expiry(from_date: Optional[date] = None) -> date:
    """Get the nearest Friday expiry (weekly options expire on Fridays)."""
    d = from_date or date.today()
    days_ahead = 4 - d.weekday()  # Friday = 4
    if days_ahead <= 0:
        days_ahead += 7
    return d + timedelta(days=days_ahead)


def _format_occ_symbol(ticker: str, expiry: date, strike: float, option_type: str) -> str:
    """Format OCC option symbol: SPY230120P00400000"""
    otype = 'P' if option_type.lower().startswith('p') else 'C'
    strike_int = int(strike * 1000)
    return f"{ticker.upper():<6}{expiry.strftime('%y%m%d')}{otype}{strike_int:08d}"


def _parse_snapshot_to_row(symbol: str, snapshot) -> Dict[str, Any]:
    """Parse an OptionsSnapshot into a flat dict."""
    row = {'symbol': symbol}

    # Parse symbol for strike/expiry/type
    # OCC format: TICKER  YYMMDDTSSSSSSSS (padded to 6 chars)
    try:
        root = symbol.rstrip('0123456789PC').strip()
        remaining = symbol[len(root):]
        if len(remaining) >= 15:
            exp_str = remaining[:6]
            otype = remaining[6]
            strike_raw = remaining[7:]
            row['expiry'] = datetime.strptime(exp_str, '%y%m%d').date()
            row['option_type'] = 'put' if otype == 'P' else 'call'
            row['strike'] = int(strike_raw) / 1000.0
    except (ValueError, IndexError):
        pass

    # Quote data
    if snapshot.latest_quote:
        row['bid'] = snapshot.latest_quote.bid_price or 0
        row['ask'] = snapshot.latest_quote.ask_price or 0
        row['mid'] = (row['bid'] + row['ask']) / 2 if row['bid'] and row['ask'] else 0
        row['bid_size'] = getattr(snapshot.latest_quote, 'bid_size', 0) or 0
        row['ask_size'] = getattr(snapshot.latest_quote, 'ask_size', 0) or 0
    else:
        row['bid'] = row['ask'] = row['mid'] = 0
        row['bid_size'] = row['ask_size'] = 0

    # Trade data
    if snapshot.latest_trade:
        row['last'] = snapshot.latest_trade.price or 0
        row['volume'] = getattr(snapshot.latest_trade, 'size', 0) or 0
    else:
        row['last'] = 0
        row['volume'] = 0

    # IV
    row['iv'] = snapshot.implied_volatility or 0

    # Greeks
    if snapshot.greeks:
        row['delta'] = snapshot.greeks.delta or 0
        row['gamma'] = snapshot.greeks.gamma or 0
        row['theta'] = snapshot.greeks.theta or 0
        row['vega'] = snapshot.greeks.vega or 0
        row['rho'] = getattr(snapshot.greeks, 'rho', 0) or 0
    else:
        row['delta'] = row['gamma'] = row['theta'] = row['vega'] = row['rho'] = 0

    # Open interest not directly in snapshot, set 0
    row['open_interest'] = 0

    return row


def get_option_chain(
    ticker: str,
    expiry_date: Optional[date] = None,
    option_type: str = 'put'
) -> pd.DataFrame:
    """
    Get option chain for a ticker.

    Args:
        ticker: Underlying symbol (e.g., 'SPY')
        expiry_date: Expiration date. If None, uses nearest weekly expiry.
        option_type: 'put' or 'call'

    Returns:
        DataFrame with columns: strike, bid, ask, mid, last, volume, open_interest,
                                iv, delta, gamma, theta, vega
    """
    if expiry_date is None:
        expiry_date = _get_nearest_weekly_expiry()

    # Check cache
    cache_key = f"{ticker}_{expiry_date}_{option_type}"
    if cache_key in _chain_cache:
        cached_time, cached_df = _chain_cache[cache_key]
        if time.time() - cached_time < CACHE_TTL_SECONDS:
            return cached_df

    _rate_limit()
    client = _get_client()

    contract_type = ContractType.PUT if option_type.lower().startswith('p') else ContractType.CALL

    request = OptionChainRequest(
        underlying_symbol=ticker.upper(),
        expiration_date=expiry_date,
        type=contract_type,
    )

    try:
        snapshots = client.get_option_chain(request)
    except Exception as e:
        logger.error(f"Failed to get option chain for {ticker}: {e}")
        return pd.DataFrame()

    if not snapshots:
        logger.warning(f"No option chain data for {ticker} exp={expiry_date} type={option_type}")
        return pd.DataFrame()

    rows = []
    for symbol, snap in snapshots.items():
        row = _parse_snapshot_to_row(symbol, snap)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure required columns exist
    required_cols = ['strike', 'bid', 'ask', 'mid', 'last', 'volume',
                     'open_interest', 'iv', 'delta', 'gamma', 'theta', 'vega']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    df = df.sort_values('strike').reset_index(drop=True)

    # Cache it
    _chain_cache[cache_key] = (time.time(), df)
    return df[required_cols + ['symbol', 'expiry', 'option_type']]


def get_option_quote(
    ticker: str,
    strike: float,
    expiry: date,
    option_type: str
) -> Dict[str, Any]:
    """
    Get a single option quote.

    Returns:
        Dict with keys: bid, ask, mid, iv, delta, gamma, theta, vega, last
    """
    # Try to get from cached chain first
    chain = get_option_chain(ticker, expiry, option_type)
    if chain.empty:
        return {'bid': 0, 'ask': 0, 'mid': 0, 'iv': 0,
                'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'last': 0}

    # Find matching strike (within $0.01 tolerance)
    match = chain[abs(chain['strike'] - strike) < 0.01]
    if match.empty:
        # Try closest strike
        idx = (chain['strike'] - strike).abs().idxmin()
        match = chain.loc[[idx]]
        logger.info(f"Exact strike {strike} not found, using {chain.loc[idx, 'strike']}")

    row = match.iloc[0]
    return {
        'bid': row['bid'],
        'ask': row['ask'],
        'mid': row['mid'],
        'iv': row['iv'],
        'delta': row['delta'],
        'gamma': row['gamma'],
        'theta': row['theta'],
        'vega': row['vega'],
        'last': row['last'],
    }


def get_ic_quotes(
    ticker: str,
    put_strike: float,
    call_strike: float,
    wing_width: float,
    expiry: date
) -> Dict[str, Any]:
    """
    Get iron condor pricing.

    Iron condor = sell put spread + sell call spread:
      - Buy put at (put_strike - wing_width)
      - Sell put at put_strike
      - Sell call at call_strike
      - Buy call at (call_strike + wing_width)

    Returns:
        Dict with: net_credit_mid, net_credit_natural, net_credit_limit,
                   short_put, long_put, short_call, long_call (individual quotes)
    """
    put_chain = get_option_chain(ticker, expiry, 'put')
    call_chain = get_option_chain(ticker, expiry, 'call')

    def find_quote(chain: pd.DataFrame, strike: float) -> Dict[str, float]:
        if chain.empty:
            return {'bid': 0, 'ask': 0, 'mid': 0}
        match = chain[abs(chain['strike'] - strike) < 0.01]
        if match.empty:
            idx = (chain['strike'] - strike).abs().idxmin()
            match = chain.loc[[idx]]
        row = match.iloc[0]
        return {'bid': row['bid'], 'ask': row['ask'], 'mid': row['mid'],
                'strike': row['strike'], 'iv': row['iv'], 'delta': row['delta']}

    short_put = find_quote(put_chain, put_strike)
    long_put = find_quote(put_chain, put_strike - wing_width)
    short_call = find_quote(call_chain, call_strike)
    long_call = find_quote(call_chain, call_strike + wing_width)

    # Net credit calculations
    # Natural (worst fill): sell at bid, buy at ask
    net_credit_natural = (
        short_put['bid'] - long_put['ask'] +
        short_call['bid'] - long_call['ask']
    )

    # Mid (theoretical fair value)
    net_credit_mid = (
        short_put['mid'] - long_put['mid'] +
        short_call['mid'] - long_call['mid']
    )

    # Limit (between natural and mid, typical fill target)
    net_credit_limit = (net_credit_natural + net_credit_mid) / 2

    return {
        'net_credit_mid': round(net_credit_mid, 2),
        'net_credit_natural': round(net_credit_natural, 2),
        'net_credit_limit': round(net_credit_limit, 2),
        'short_put': short_put,
        'long_put': long_put,
        'short_call': short_call,
        'long_call': long_call,
        'max_loss': round(wing_width - net_credit_mid, 2),
        'max_profit': round(net_credit_mid, 2),
    }


def get_bps_quote(
    ticker: str,
    short_strike: float,
    long_strike: float,
    expiry: date
) -> Dict[str, Any]:
    """
    Get bull put spread pricing.

    Bull put spread = sell higher put, buy lower put.

    Returns:
        Dict with: net_credit_mid, net_credit_natural, net_credit_limit,
                   short_put, long_put quotes, max_loss, max_profit
    """
    put_chain = get_option_chain(ticker, expiry, 'put')

    def find_quote(chain: pd.DataFrame, strike: float) -> Dict[str, float]:
        if chain.empty:
            return {'bid': 0, 'ask': 0, 'mid': 0}
        match = chain[abs(chain['strike'] - strike) < 0.01]
        if match.empty:
            idx = (chain['strike'] - strike).abs().idxmin()
            match = chain.loc[[idx]]
        row = match.iloc[0]
        return {'bid': row['bid'], 'ask': row['ask'], 'mid': row['mid'],
                'strike': row['strike'], 'iv': row['iv'], 'delta': row['delta']}

    short_put = find_quote(put_chain, short_strike)
    long_put = find_quote(put_chain, long_strike)

    net_credit_natural = short_put['bid'] - long_put['ask']
    net_credit_mid = short_put['mid'] - long_put['mid']
    net_credit_limit = (net_credit_natural + net_credit_mid) / 2

    width = short_strike - long_strike

    return {
        'net_credit_mid': round(net_credit_mid, 2),
        'net_credit_natural': round(net_credit_natural, 2),
        'net_credit_limit': round(net_credit_limit, 2),
        'short_put': short_put,
        'long_put': long_put,
        'width': width,
        'max_loss': round(width - net_credit_mid, 2),
        'max_profit': round(net_credit_mid, 2),
    }


def get_straddle_quotes(
    ticker: str,
    dte_target: int = 14,
    strike: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get ATM straddle (call + put) pricing for a ticker.

    Finds the expiry closest to dte_target, then the ATM strike (delta ~0.50).
    Returns combined straddle price.

    Returns:
        Dict with: spot, strike, dte, call_price, put_price, straddle_price,
                   call_bid, call_ask, put_bid, put_ask, source
    """
    from datetime import date, timedelta
    today = date.today()
    target_expiry = today + timedelta(days=dte_target)

    # Find best expiry (try a few nearby dates)
    best_expiry = None
    for delta_days in range(-7, 14):
        candidate = target_expiry + timedelta(days=delta_days)
        # Options expire on Fridays typically
        if candidate.weekday() == 4:  # Friday
            best_expiry = candidate
            break
    if best_expiry is None:
        best_expiry = target_expiry

    try:
        call_chain = get_option_chain(ticker, best_expiry, 'call')
        put_chain  = get_option_chain(ticker, best_expiry, 'put')

        if call_chain.empty or put_chain.empty:
            return None

        # Find ATM call (delta closest to 0.50)
        if strike is None:
            atm_row = call_chain.iloc[(call_chain['delta'] - 0.50).abs().argsort()[:1]].iloc[0]
            strike = float(atm_row['strike'])

        # Get call at that strike
        call_match = call_chain[abs(call_chain['strike'] - strike) < 1.0]
        put_match  = put_chain[abs(put_chain['strike'] - strike) < 1.0]

        if call_match.empty:
            idx = (call_chain['strike'] - strike).abs().idxmin()
            call_match = call_chain.loc[[idx]]
        if put_match.empty:
            idx = (put_chain['strike'] - strike).abs().idxmin()
            put_match = put_chain.loc[[idx]]

        if call_match.empty or put_match.empty:
            return None

        c = call_match.iloc[0]
        p = put_match.iloc[0]

        # Use mid price for straddle
        call_price = float(c['mid'])
        put_price  = float(p['mid'])

        # Get spot (call_strike with delta ~0.5 ≈ spot)
        spot_row = call_chain.iloc[(call_chain['delta'] - 0.50).abs().argsort()[:1]].iloc[0]
        spot = float(spot_row['strike'])

        actual_dte = (best_expiry - today).days

        return {
            'spot': round(spot, 2),
            'strike': round(strike, 2),
            'expiry': str(best_expiry),
            'dte': actual_dte,
            'call_price': round(call_price, 2),
            'put_price': round(put_price, 2),
            'straddle_price': round(call_price + put_price, 2),
            'call_bid': float(c['bid']),
            'call_ask': float(c['ask']),
            'put_bid': float(p['bid']),
            'put_ask': float(p['ask']),
            'call_iv': float(c.get('iv', 0)),
            'put_iv': float(p.get('iv', 0)),
            'source': 'alpaca',
        }

    except Exception as e:
        logger.warning(f"get_straddle_quotes failed for {ticker}: {e}")
        return None


def price_comparison(
    ticker: str,
    strike: float,
    expiry: date,
    option_type: str,
    bs_price: float
) -> Dict[str, Any]:
    """
    Compare real Alpaca quote to Black-Scholes modeled price.

    Returns:
        Dict with: real_mid, bs_price, ratio (real/bs), diff_pct, bid, ask, iv
    """
    quote = get_option_quote(ticker, strike, expiry, option_type)
    real_mid = quote['mid']

    if bs_price > 0:
        ratio = real_mid / bs_price
        diff_pct = (real_mid - bs_price) / bs_price * 100
    else:
        ratio = float('inf') if real_mid > 0 else 0
        diff_pct = float('inf') if real_mid > 0 else 0

    return {
        'real_mid': real_mid,
        'bs_price': bs_price,
        'ratio': round(ratio, 4),
        'diff_pct': round(diff_pct, 2),
        'bid': quote['bid'],
        'ask': quote['ask'],
        'iv': quote['iv'],
        'real_delta': quote['delta'],
    }


def get_real_premium_or_fallback(
    ticker: str,
    strike: float,
    expiry: date,
    option_type: str,
    bs_fallback_fn=None,
) -> Dict[str, Any]:
    """
    Try to get real option premium from Alpaca. Fall back to BS if API fails or market closed.

    Args:
        ticker: Underlying symbol
        strike: Strike price
        expiry: Expiration date
        option_type: 'put' or 'call'
        bs_fallback_fn: Callable(ticker, strike, expiry, option_type) -> float
                        If None and fallback needed, returns 0.

    Returns:
        Dict with: premium (mid price), source ('alpaca' or 'bs_fallback'),
                   bid, ask, iv, delta
    """
    # Try Alpaca first
    if _is_market_hours():
        try:
            quote = get_option_quote(ticker, strike, expiry, option_type)
            if quote['mid'] > 0:
                return {
                    'premium': quote['mid'],
                    'source': 'alpaca',
                    'bid': quote['bid'],
                    'ask': quote['ask'],
                    'iv': quote['iv'],
                    'delta': quote['delta'],
                }
        except Exception as e:
            logger.warning(f"Alpaca quote failed for {ticker} {strike} {expiry}: {e}")

    # Fallback to BS
    if bs_fallback_fn:
        try:
            bs_price = bs_fallback_fn(ticker, strike, expiry, option_type)
            return {
                'premium': bs_price,
                'source': 'bs_fallback',
                'bid': 0,
                'ask': 0,
                'iv': 0,
                'delta': 0,
            }
        except Exception as e:
            logger.warning(f"BS fallback also failed: {e}")

    return {
        'premium': 0,
        'source': 'unavailable',
        'bid': 0,
        'ask': 0,
        'iv': 0,
        'delta': 0,
    }


def cache_daily_chains(
    tickers: List[str],
    output_dir: str = '/home/jupiter/Lvl3Quant/data/option_chains'
) -> Dict[str, str]:
    """
    Fetch and cache full option chains for a list of tickers.
    Saves to parquet for historical comparison.

    Call daily at market open.

    Returns:
        Dict mapping ticker -> saved file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    today = date.today()
    saved_files = {}

    for ticker in tickers:
        try:
            # Get puts and calls for nearest 2 expiries
            expiry1 = _get_nearest_weekly_expiry(today)
            expiry2 = _get_nearest_weekly_expiry(expiry1 + timedelta(days=1))

            frames = []
            for exp in [expiry1, expiry2]:
                for otype in ['put', 'call']:
                    chain = get_option_chain(ticker, exp, otype)
                    if not chain.empty:
                        chain['fetch_date'] = today
                        chain['fetch_time'] = datetime.now().isoformat()
                        frames.append(chain)
                    time.sleep(0.5)  # Be gentle with rate limits

            if frames:
                combined = pd.concat(frames, ignore_index=True)
                filename = f"{ticker}_{today.strftime('%Y%m%d')}_chains.parquet"
                filepath = output_path / filename
                combined.to_parquet(filepath, index=False)
                saved_files[ticker] = str(filepath)
                logger.info(f"Saved {len(combined)} contracts for {ticker} -> {filepath}")
            else:
                logger.warning(f"No chain data for {ticker}")

        except Exception as e:
            logger.error(f"Failed to cache chain for {ticker}: {e}")

    return saved_files


# --- Convenience / quick test ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Testing Alpaca Options Pricing Module")
    print("=" * 50)

    ticker = 'SPY'
    expiry = _get_nearest_weekly_expiry()
    print(f"\nFetching {ticker} put chain for {expiry}...")

    chain = get_option_chain(ticker, expiry, 'put')
    if not chain.empty:
        print(f"Got {len(chain)} contracts")
        print(chain[['strike', 'bid', 'ask', 'mid', 'iv', 'delta']].head(10).to_string())
    else:
        print("No data returned (market may be closed or subscription needed)")
