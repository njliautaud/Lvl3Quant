"""
Test script for Alpaca Options Pricing Module.
Pulls option chains, compares to BS prices, validates assumptions.

Usage:
    python scripts/test_alpaca_options.py
"""

import sys
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import norm

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'live_trading_linux'))

from alpaca_options_pricing import (
    get_option_chain,
    get_option_quote,
    get_ic_quotes,
    get_bps_quote,
    price_comparison,
    _get_nearest_weekly_expiry,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# --- Simple Black-Scholes for comparison ---
def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def test_chain_fetch():
    """Test fetching option chains for major tickers."""
    print("\n" + "=" * 70)
    print("TEST 1: Option Chain Fetch")
    print("=" * 70)

    tickers = ['SPY', 'AAPL', 'TSLA']
    expiry = _get_nearest_weekly_expiry()
    print(f"Target expiry: {expiry}")

    results = {}
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        chain = get_option_chain(ticker, expiry, 'put')
        if chain.empty:
            print(f"  NO DATA (market closed or subscription needed)")
            continue

        results[ticker] = chain
        print(f"  Contracts: {len(chain)}")
        print(f"  Strike range: ${chain['strike'].min():.0f} - ${chain['strike'].max():.0f}")
        print(f"  IV range: {chain['iv'].min():.3f} - {chain['iv'].max():.3f}")

        # Show ATM options
        if 'strike' in chain.columns and len(chain) > 0:
            # Estimate spot from put-call parity or just use middle strikes
            mid_idx = len(chain) // 2
            atm_slice = chain.iloc[max(0, mid_idx-3):min(len(chain), mid_idx+4)]
            print(f"\n  Near-ATM puts:")
            print(f"  {'Strike':>8} {'Bid':>7} {'Ask':>7} {'Mid':>7} {'IV':>6} {'Delta':>7}")
            for _, row in atm_slice.iterrows():
                print(f"  {row['strike']:>8.1f} {row['bid']:>7.2f} {row['ask']:>7.2f} "
                      f"{row['mid']:>7.2f} {row['iv']:>6.3f} {row['delta']:>7.3f}")

    return results


def test_bs_comparison(chains: dict):
    """Compare real quotes to BS model prices at various moneyness levels."""
    print("\n" + "=" * 70)
    print("TEST 2: Real Quote vs Black-Scholes Comparison")
    print("=" * 70)

    expiry = _get_nearest_weekly_expiry()
    T = max((expiry - date.today()).days / 365.0, 1/365)
    r = 0.05  # risk-free rate approximation

    for ticker, chain in chains.items():
        if chain.empty:
            continue

        print(f"\n--- {ticker} (T={T:.4f} yrs) ---")

        # Estimate spot price: use the strike where delta is closest to -0.50
        atm_candidates = chain[chain['delta'].between(-0.6, -0.4)]
        if atm_candidates.empty:
            atm_candidates = chain
        spot_est = atm_candidates['strike'].median()

        print(f"  Estimated spot: ${spot_est:.2f}")
        print(f"\n  {'Strike':>8} {'Money%':>7} {'Real Mid':>9} {'BS Price':>9} "
              f"{'Ratio':>7} {'Diff%':>7} {'IV Used':>7}")
        print(f"  {'-'*8} {'-'*7} {'-'*9} {'-'*9} {'-'*7} {'-'*7} {'-'*7}")

        # Sample across moneyness levels
        moneyness_targets = [0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04]
        for m in moneyness_targets:
            target_strike = round(spot_est * m, 0)
            match = chain[(chain['strike'] - target_strike).abs() < 2]
            if match.empty:
                continue
            row = match.iloc[0]
            actual_strike = row['strike']
            real_mid = row['mid']
            iv = row['iv'] if row['iv'] > 0 else 0.20  # fallback IV

            bs_price = bs_put_price(spot_est, actual_strike, T, r, iv)

            if bs_price > 0.01:
                ratio = real_mid / bs_price
                diff_pct = (real_mid - bs_price) / bs_price * 100
            else:
                ratio = 0
                diff_pct = 0

            moneyness_pct = actual_strike / spot_est * 100
            print(f"  {actual_strike:>8.0f} {moneyness_pct:>6.1f}% {real_mid:>9.2f} "
                  f"{bs_price:>9.2f} {ratio:>7.3f} {diff_pct:>6.1f}% {iv:>7.3f}")


def test_single_quote():
    """Test single option quote retrieval."""
    print("\n" + "=" * 70)
    print("TEST 3: Single Option Quote")
    print("=" * 70)

    expiry = _get_nearest_weekly_expiry()
    # SPY ~5% OTM put
    quote = get_option_quote('SPY', 540.0, expiry, 'put')
    print(f"\n  SPY 540 Put exp={expiry}:")
    for k, v in quote.items():
        print(f"    {k}: {v}")


def test_spread_pricing():
    """Test spread pricing functions."""
    print("\n" + "=" * 70)
    print("TEST 4: Spread Pricing")
    print("=" * 70)

    expiry = _get_nearest_weekly_expiry()

    # Bull put spread on SPY
    print(f"\n  SPY Bull Put Spread (530/525) exp={expiry}:")
    bps = get_bps_quote('SPY', 530.0, 525.0, expiry)
    if bps['net_credit_mid'] > 0:
        for k, v in bps.items():
            if isinstance(v, dict):
                print(f"    {k}: bid={v.get('bid',0):.2f} ask={v.get('ask',0):.2f} mid={v.get('mid',0):.2f}")
            else:
                print(f"    {k}: {v}")
    else:
        print("    No data (market closed?)")

    # Iron condor on SPY
    print(f"\n  SPY Iron Condor (530P/570C, 5-wide wings) exp={expiry}:")
    ic = get_ic_quotes('SPY', 530.0, 570.0, 5.0, expiry)
    if ic['net_credit_mid'] > 0:
        print(f"    Net credit (mid): ${ic['net_credit_mid']:.2f}")
        print(f"    Net credit (natural): ${ic['net_credit_natural']:.2f}")
        print(f"    Net credit (limit): ${ic['net_credit_limit']:.2f}")
        print(f"    Max loss: ${ic['max_loss']:.2f}")
        print(f"    Max profit: ${ic['max_profit']:.2f}")
    else:
        print("    No data (market closed?)")


def test_api_access():
    """Basic connectivity test - just checks if the API responds."""
    print("\n" + "=" * 70)
    print("TEST 0: API Connectivity Check")
    print("=" * 70)

    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca_options_pricing import ALPACA_API_KEY, ALPACA_SECRET_KEY

    try:
        client = OptionHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY,
        )
        # Try a minimal request
        from alpaca.data.requests import OptionChainRequest
        from alpaca.trading.enums import ContractType
        req = OptionChainRequest(
            underlying_symbol='SPY',
            expiration_date=_get_nearest_weekly_expiry(),
            type=ContractType.PUT,
            strike_price_gte=500,
            strike_price_lte=510,
        )
        result = client.get_option_chain(req)
        if result:
            print(f"  API connected. Got {len(result)} contracts in test query.")
            # Show one sample
            sample_sym = list(result.keys())[0]
            snap = result[sample_sym]
            print(f"  Sample: {sample_sym}")
            if snap.latest_quote:
                print(f"    Bid: {snap.latest_quote.bid_price}, Ask: {snap.latest_quote.ask_price}")
            if snap.greeks:
                print(f"    Delta: {snap.greeks.delta}, IV: {snap.implied_volatility}")
            return True
        else:
            print("  API responded but returned no data.")
            print("  This likely means market is closed or options subscription is needed.")
            return False
    except Exception as e:
        print(f"  API connection FAILED: {e}")
        print(f"  Error type: {type(e).__name__}")
        if 'forbidden' in str(e).lower() or '403' in str(e):
            print("  => Likely need Alpaca Options data subscription")
        elif 'unauthorized' in str(e).lower() or '401' in str(e):
            print("  => API key/secret may be wrong")
        return False


def main():
    print("Alpaca Options Pricing - Test Suite")
    print(f"Date: {date.today()}")
    print(f"Next weekly expiry: {_get_nearest_weekly_expiry()}")

    # Test 0: connectivity
    has_access = test_api_access()

    if not has_access:
        print("\n" + "=" * 70)
        print("STOPPING: Cannot access options data.")
        print("Possible causes:")
        print("  1. Market is closed (run during 9:30-16:00 ET Mon-Fri)")
        print("  2. Need Alpaca Options data subscription")
        print("  3. API credentials invalid")
        print("=" * 70)
        return

    # Test 1: chain fetch
    chains = test_chain_fetch()

    # Test 2: BS comparison
    if chains:
        test_bs_comparison(chains)

    # Test 3: single quote
    test_single_quote()

    # Test 4: spread pricing
    test_spread_pricing()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
