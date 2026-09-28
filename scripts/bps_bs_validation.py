#!/usr/bin/env python3
"""
BPS Black-Scholes Validation vs Live Option Chains
====================================================
HC #664 R4: All BPS backtests use BS-modeled IV (from realized vol).
This script validates BS modeled prices against real live option data
from yfinance to quantify any systematic bias.

Checks:
  1. BS mid-price vs yfinance mid-price at 25-30 delta puts
  2. BS implied vol (rv20 * 1.15) vs yfinance implied vol
  3. Assumed 5%/7% BA vs actual BA from yfinance
  4. BPS spread net credit: BS vs yfinance
  5. Impact on Sharpe estimate

Output: output/bps_bs_validation/
"""

import sys, json, time, math, warnings, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. pip install yfinance")
    sys.exit(1)

ROOT = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "bps_bs_validation"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# Ticker Universe (match BPS backtest)
# ═══════════════════════════════════════════════════════════════════

TIER1_TICKERS = [
    'AAPL', 'ABBV', 'AMD', 'AMZN', 'BA', 'BAC', 'BLK', 'C', 'CAT', 'COIN',
    'COST', 'CRM', 'CVX', 'DIS', 'GE', 'GM', 'GOOGL', 'GS', 'HD', 'INTC',
    'JNJ', 'JPM', 'KO', 'LLY', 'LOW', 'MA', 'MCD', 'META', 'MRNA', 'MS',
    'MSFT', 'NFLX', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PLTR', 'PYPL',
    'SBUX', 'SCHW', 'T', 'TGT', 'TSLA', 'UBER', 'UNH', 'V', 'WFC', 'WMT', 'XOM',
]

TIER2_TICKERS = [
    'BIIB', 'REGN', 'VRTX', 'GILD',  # Biotech
    'MRVL', 'ON', 'DVN', 'FANG', 'MPC',  # Energy/Semi
    'O', 'AMT', 'PLD',  # REITs
    'DG', 'TJX', 'BBY',  # Consumer
]

# Risk-free rate
RF_RATE = 0.0525

# Our backtest assumptions
ASSUMED_BA_TIER1 = 0.05  # 5%
ASSUMED_BA_TIER2 = 0.07  # 7%
IV_SCALE_FACTOR = 1.15   # rv20 * 1.15 = modeled IV


# ═══════════════════════════════════════════════════════════════════
# Black-Scholes Primitives (EXACT copy from backtest)
# ═══════════════════════════════════════════════════════════════════

def bs_price(S, K, T, sigma, r=0.0525, q=0.0, kind="put"):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)
    return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, sigma, r=0.0525, q=0.0, kind="put"):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if kind == "put":
        return math.exp(-q * T) * (norm.cdf(d1) - 1)
    return math.exp(-q * T) * norm.cdf(d1)


def bs_iv_from_price(S, K, T, price, r=0.0525, q=0.0, kind="put"):
    """Invert BS to get implied vol from market price."""
    if price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return np.nan
    intrinsic = max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    if price < intrinsic - 0.01:
        return np.nan
    try:
        iv = brentq(
            lambda sig: bs_price(S, K, T, sig, r, q, kind) - price,
            0.01, 5.0, xtol=1e-6, maxiter=100
        )
        return iv
    except (ValueError, RuntimeError):
        return np.nan


def strike_from_delta(S, T, sigma, target_delta, r=0.0525, q=0.0):
    """Find put strike for a given (negative) delta magnitude."""
    target = abs(target_delta)
    p = 1 - target  # For puts: delta = N(d1) - 1, so N(d1) = 1 - |delta|
    p = min(max(p, 1e-6), 1 - 1e-6)
    d1 = norm.ppf(p)
    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r - q + 0.5 * sigma**2) * T))
    return K


# ═══════════════════════════════════════════════════════════════════
# Data Fetching
# ═══════════════════════════════════════════════════════════════════

def get_realized_vol(ticker_obj, period="3mo"):
    """Get 20-day realized vol, matching backtest methodology."""
    try:
        hist = ticker_obj.history(period=period)
        if len(hist) < 25:
            return np.nan
        log_ret = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
        rv20 = log_ret.tail(20).std() * np.sqrt(252)
        return float(rv20)
    except Exception:
        return np.nan


def find_nearest_expiry(chain_dates, target_dte):
    """Find the option expiry closest to target DTE."""
    today = datetime.now().date()
    best = None
    best_diff = 999
    for d in chain_dates:
        if isinstance(d, str):
            exp_date = datetime.strptime(d, "%Y-%m-%d").date()
        else:
            exp_date = d
        dte = (exp_date - today).days
        if dte < 1:
            continue
        diff = abs(dte - target_dte)
        if diff < best_diff:
            best_diff = diff
            best = (exp_date, dte)
    return best


def get_put_chain_at_delta_range(ticker_obj, expiry_date, spot, rv20,
                                  delta_lo=0.15, delta_hi=0.35):
    """
    Get puts in the delta range from yfinance chain.
    Returns DataFrame with: strike, bid, ask, mid, iv_yf, delta_bs, bs_mid.
    """
    try:
        chain = ticker_obj.option_chain(str(expiry_date))
        puts = chain.puts.copy()
    except Exception as e:
        return pd.DataFrame()

    if puts.empty:
        return pd.DataFrame()

    today = datetime.now().date()
    T = max((expiry_date - today).days / 365.0, 1 / 365)

    # Model IV = rv20 * 1.15 (matching backtest)
    modeled_iv = rv20 * IV_SCALE_FACTOR if not np.isnan(rv20) else np.nan

    rows = []
    for _, row in puts.iterrows():
        K = float(row['strike'])
        bid = float(row.get('bid', 0))
        ask = float(row.get('ask', 0))
        yf_iv = float(row.get('impliedVolatility', 0))
        vol = int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0
        oi = int(row.get('openInterest', 0)) if pd.notna(row.get('openInterest')) else 0

        # Skip illiquid or zero-bid options
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        if bid < 0.05:
            continue

        mid = (bid + ask) / 2.0
        ba_spread = (ask - bid) / mid if mid > 0 else 999

        # Compute BS delta using yfinance IV
        delta_yf = abs(bs_delta(spot, K, T, yf_iv)) if yf_iv > 0.01 else np.nan

        # Compute BS delta using our modeled IV
        delta_mod = abs(bs_delta(spot, K, T, modeled_iv)) if not np.isnan(modeled_iv) else np.nan

        # BS price using our modeled IV
        bs_mid = bs_price(spot, K, T, modeled_iv) if not np.isnan(modeled_iv) else np.nan

        # BS price using yfinance IV
        bs_from_yf_iv = bs_price(spot, K, T, yf_iv) if yf_iv > 0.01 else np.nan

        # Implied vol from market mid-price
        iv_from_mid = bs_iv_from_price(spot, K, T, mid)

        # Filter to delta range (use yf IV delta for filtering since that's "truth")
        in_range = False
        if not np.isnan(delta_yf) and delta_lo <= delta_yf <= delta_hi:
            in_range = True
        elif not np.isnan(delta_mod) and delta_lo <= delta_mod <= delta_hi:
            in_range = True

        if not in_range:
            continue

        rows.append({
            'strike': K,
            'bid': bid,
            'ask': ask,
            'mid_market': mid,
            'ba_spread_pct': ba_spread,
            'iv_yfinance': yf_iv,
            'iv_from_mid': iv_from_mid,
            'iv_modeled': modeled_iv,
            'delta_yf': delta_yf,
            'delta_modeled': delta_mod,
            'bs_price_modeled_iv': bs_mid,
            'bs_price_yf_iv': bs_from_yf_iv,
            'volume': vol,
            'open_interest': oi,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Main Validation
# ═══════════════════════════════════════════════════════════════════

def validate_ticker(symbol, tier, target_dtes=(7, 14)):
    """Run full validation for a single ticker."""
    print(f"  {symbol} ({tier})...", end="", flush=True)
    try:
        tk = yf.Ticker(symbol)
        # Get spot price
        info = tk.fast_info
        spot = float(info.get('lastPrice', 0) or info.get('previousClose', 0))
        if spot <= 0:
            hist = tk.history(period="5d")
            if hist.empty:
                print(" NO DATA")
                return None
            spot = float(hist['Close'].iloc[-1])
    except Exception as e:
        print(f" ERROR getting price: {e}")
        return None

    # Get realized vol
    rv20 = get_realized_vol(tk)
    if np.isnan(rv20) or rv20 < 0.01:
        print(f" NO VOL DATA")
        return None

    modeled_iv = rv20 * IV_SCALE_FACTOR

    # Get available expiry dates
    try:
        exp_dates = tk.options
        if not exp_dates or len(exp_dates) == 0:
            print(" NO OPTIONS")
            return None
    except Exception:
        print(" NO OPTIONS")
        return None

    results_by_dte = {}
    for target_dte in target_dtes:
        nearest = find_nearest_expiry(exp_dates, target_dte)
        if nearest is None:
            continue
        exp_date, actual_dte = nearest

        # Get puts in delta range
        puts_df = get_put_chain_at_delta_range(tk, exp_date, spot, rv20,
                                                delta_lo=0.10, delta_hi=0.40)
        if puts_df.empty:
            continue

        results_by_dte[f"dte_{target_dte}"] = {
            'target_dte': target_dte,
            'actual_dte': actual_dte,
            'expiry': str(exp_date),
            'n_strikes': len(puts_df),
            'puts_data': puts_df.to_dict('records'),
        }

    if not results_by_dte:
        print(" NO VALID PUTS")
        return None

    assumed_ba = ASSUMED_BA_TIER1 if tier == "tier1" else ASSUMED_BA_TIER2
    print(f" OK (spot={spot:.1f}, rv20={rv20:.3f}, modeled_iv={modeled_iv:.3f})")

    return {
        'symbol': symbol,
        'tier': tier,
        'spot': spot,
        'rv20': rv20,
        'modeled_iv': modeled_iv,
        'assumed_ba': assumed_ba,
        'dte_results': results_by_dte,
    }


def compute_bps_spread_comparison(result):
    """
    For each ticker/expiry, find the ~30-delta and ~15-delta puts
    and compare BPS net credit: BS vs market.
    """
    spreads = []
    for dte_key, dte_data in result['dte_results'].items():
        puts = pd.DataFrame(dte_data['puts_data'])
        if len(puts) < 2:
            continue

        # Find closest to 30-delta (short leg) and 15-delta (long leg)
        # Use yfinance delta as truth
        delta_col = 'delta_yf' if puts['delta_yf'].notna().any() else 'delta_modeled'

        valid = puts[puts[delta_col].notna()].copy()
        if len(valid) < 2:
            continue

        # Short leg: closest to 0.30 delta
        valid['d30_dist'] = (valid[delta_col] - 0.30).abs()
        short_idx = valid['d30_dist'].idxmin()
        short_leg = valid.loc[short_idx]

        # Long leg: closest to 0.15 delta
        valid['d15_dist'] = (valid[delta_col] - 0.15).abs()
        long_idx = valid['d15_dist'].idxmin()
        long_leg = valid.loc[long_idx]

        if short_idx == long_idx:
            continue

        # Market net credit (sell short, buy long) using mids
        mkt_net_credit = short_leg['mid_market'] - long_leg['mid_market']

        # BS net credit using modeled IV
        bs_net_credit = short_leg['bs_price_modeled_iv'] - long_leg['bs_price_modeled_iv']

        # BS net credit using yfinance IV
        bs_yf_net_credit = short_leg['bs_price_yf_iv'] - long_leg['bs_price_yf_iv']

        # Spread width
        spread_width = short_leg['strike'] - long_leg['strike']

        # Actual BA for each leg
        short_ba = short_leg['ba_spread_pct']
        long_ba = long_leg['ba_spread_pct']
        avg_ba = (short_ba + long_ba) / 2

        spreads.append({
            'dte_key': dte_key,
            'actual_dte': dte_data['actual_dte'],
            'short_strike': short_leg['strike'],
            'long_strike': long_leg['strike'],
            'spread_width': spread_width,
            'short_delta': short_leg[delta_col],
            'long_delta': long_leg[delta_col],
            'mkt_net_credit': mkt_net_credit,
            'bs_net_credit_modeled': bs_net_credit,
            'bs_net_credit_yfiv': bs_yf_net_credit,
            'bs_vs_mkt_pct': ((bs_net_credit / mkt_net_credit - 1) * 100) if mkt_net_credit > 0 else np.nan,
            'short_iv_yf': short_leg['iv_yfinance'],
            'short_iv_mod': short_leg['iv_modeled'],
            'long_iv_yf': long_leg['iv_yfinance'],
            'long_iv_mod': long_leg['iv_modeled'],
            'short_ba_pct': short_ba,
            'long_ba_pct': long_ba,
            'avg_ba_pct': avg_ba,
            'assumed_ba': result['assumed_ba'],
        })

    return spreads


def aggregate_results(all_results):
    """Compute aggregate statistics across all tickers."""
    # Flatten all put-level data
    all_puts = []
    all_spreads = []

    for res in all_results:
        if res is None:
            continue

        tier = res['tier']
        assumed_ba = res['assumed_ba']

        for dte_key, dte_data in res['dte_results'].items():
            for p in dte_data['puts_data']:
                p['symbol'] = res['symbol']
                p['tier'] = tier
                p['assumed_ba'] = assumed_ba
                p['dte_key'] = dte_key
                p['actual_dte'] = dte_data['actual_dte']
                all_puts.append(p)

        # BPS spread comparison
        spreads = compute_bps_spread_comparison(res)
        for s in spreads:
            s['symbol'] = res['symbol']
            s['tier'] = tier
        all_spreads.extend(spreads)

    puts_df = pd.DataFrame(all_puts)
    spreads_df = pd.DataFrame(all_spreads)

    report = {}

    # ── 1. BS Price Accuracy ──
    if not puts_df.empty and 'bs_price_modeled_iv' in puts_df.columns:
        valid = puts_df.dropna(subset=['bs_price_modeled_iv', 'mid_market'])
        if not valid.empty:
            valid = valid.copy()
            valid['price_error_pct'] = (valid['bs_price_modeled_iv'] / valid['mid_market'] - 1) * 100
            valid['abs_error_pct'] = valid['price_error_pct'].abs()

            report['price_accuracy'] = {
                'n_options': len(valid),
                'mean_error_pct': round(float(valid['price_error_pct'].mean()), 2),
                'median_error_pct': round(float(valid['price_error_pct'].median()), 2),
                'std_error_pct': round(float(valid['price_error_pct'].std()), 2),
                'within_10pct': round(float((valid['abs_error_pct'] <= 10).mean() * 100), 1),
                'within_20pct': round(float((valid['abs_error_pct'] <= 20).mean() * 100), 1),
                'within_30pct': round(float((valid['abs_error_pct'] <= 30).mean() * 100), 1),
                'p25_error': round(float(valid['price_error_pct'].quantile(0.25)), 2),
                'p75_error': round(float(valid['price_error_pct'].quantile(0.75)), 2),
                'bs_systematically_low': float(valid['price_error_pct'].mean()) < -5,
                'bs_systematically_high': float(valid['price_error_pct'].mean()) > 5,
            }

            # By tier
            for t in ['tier1', 'tier2']:
                subset = valid[valid['tier'] == t]
                if not subset.empty:
                    report[f'price_accuracy_{t}'] = {
                        'n_options': len(subset),
                        'mean_error_pct': round(float(subset['price_error_pct'].mean()), 2),
                        'median_error_pct': round(float(subset['price_error_pct'].median()), 2),
                        'within_10pct': round(float((subset['abs_error_pct'] <= 10).mean() * 100), 1),
                        'within_20pct': round(float((subset['abs_error_pct'] <= 20).mean() * 100), 1),
                    }

    # ── 2. IV Comparison ──
    if not puts_df.empty:
        valid_iv = puts_df.dropna(subset=['iv_yfinance', 'iv_modeled'])
        valid_iv = valid_iv[valid_iv['iv_yfinance'] > 0.01].copy()
        if not valid_iv.empty:
            valid_iv['iv_error_pct'] = (valid_iv['iv_modeled'] / valid_iv['iv_yfinance'] - 1) * 100

            report['iv_comparison'] = {
                'n_options': len(valid_iv),
                'mean_iv_yfinance': round(float(valid_iv['iv_yfinance'].mean()), 4),
                'mean_iv_modeled': round(float(valid_iv['iv_modeled'].mean()), 4),
                'mean_iv_error_pct': round(float(valid_iv['iv_error_pct'].mean()), 2),
                'median_iv_error_pct': round(float(valid_iv['iv_error_pct'].median()), 2),
                'modeled_iv_too_low_frac': round(float((valid_iv['iv_error_pct'] < 0).mean() * 100), 1),
                'modeled_iv_too_high_frac': round(float((valid_iv['iv_error_pct'] > 0).mean() * 100), 1),
            }

            # By tier
            for t in ['tier1', 'tier2']:
                subset = valid_iv[valid_iv['tier'] == t]
                if not subset.empty:
                    report[f'iv_comparison_{t}'] = {
                        'n_options': len(subset),
                        'mean_iv_yf': round(float(subset['iv_yfinance'].mean()), 4),
                        'mean_iv_mod': round(float(subset['iv_modeled'].mean()), 4),
                        'mean_iv_error_pct': round(float(subset['iv_error_pct'].mean()), 2),
                    }

    # ── 3. Bid-Ask Spread Reality ──
    if not puts_df.empty and 'ba_spread_pct' in puts_df.columns:
        valid_ba = puts_df[puts_df['ba_spread_pct'] < 2.0].copy()  # filter outliers
        if not valid_ba.empty:
            report['bid_ask_reality'] = {
                'n_options': len(valid_ba),
                'mean_ba_pct': round(float(valid_ba['ba_spread_pct'].mean() * 100), 2),
                'median_ba_pct': round(float(valid_ba['ba_spread_pct'].median() * 100), 2),
                'p25_ba_pct': round(float(valid_ba['ba_spread_pct'].quantile(0.25) * 100), 2),
                'p75_ba_pct': round(float(valid_ba['ba_spread_pct'].quantile(0.75) * 100), 2),
                'p90_ba_pct': round(float(valid_ba['ba_spread_pct'].quantile(0.90) * 100), 2),
            }

            for t, assumed in [('tier1', ASSUMED_BA_TIER1), ('tier2', ASSUMED_BA_TIER2)]:
                subset = valid_ba[valid_ba['tier'] == t]
                if not subset.empty:
                    actual_mean = float(subset['ba_spread_pct'].mean())
                    report[f'bid_ask_{t}'] = {
                        'n_options': len(subset),
                        'assumed_ba_pct': assumed * 100,
                        'actual_mean_ba_pct': round(actual_mean * 100, 2),
                        'actual_median_ba_pct': round(float(subset['ba_spread_pct'].median() * 100), 2),
                        'actual_p75_ba_pct': round(float(subset['ba_spread_pct'].quantile(0.75) * 100), 2),
                        'assumption_vs_reality_pct': round(((assumed / actual_mean - 1) * 100) if actual_mean > 0 else 0, 1),
                        'assumption_is_optimistic': assumed < actual_mean,
                    }

    # ── 4. BPS Spread Net Credit Comparison ──
    if not spreads_df.empty:
        valid_sp = spreads_df.dropna(subset=['bs_vs_mkt_pct'])
        if not valid_sp.empty:
            report['bps_spread_credit'] = {
                'n_spreads': len(valid_sp),
                'mean_bs_vs_mkt_pct': round(float(valid_sp['bs_vs_mkt_pct'].mean()), 2),
                'median_bs_vs_mkt_pct': round(float(valid_sp['bs_vs_mkt_pct'].median()), 2),
                'std_bs_vs_mkt_pct': round(float(valid_sp['bs_vs_mkt_pct'].std()), 2),
                'bs_overestimates_credit_frac': round(float((valid_sp['bs_vs_mkt_pct'] > 0).mean() * 100), 1),
                'mean_mkt_credit': round(float(valid_sp['mkt_net_credit'].mean()), 4),
                'mean_bs_credit': round(float(valid_sp['bs_net_credit_modeled'].mean()), 4),
                'avg_actual_ba_pct': round(float(valid_sp['avg_ba_pct'].mean() * 100), 2),
                'avg_assumed_ba_pct': round(float(valid_sp['assumed_ba'].mean() * 100), 2),
            }

            # Per-ticker detail
            ticker_details = []
            for sym in valid_sp['symbol'].unique():
                sub = valid_sp[valid_sp['symbol'] == sym]
                ticker_details.append({
                    'symbol': sym,
                    'tier': sub['tier'].iloc[0],
                    'mean_bs_vs_mkt_pct': round(float(sub['bs_vs_mkt_pct'].mean()), 1),
                    'avg_actual_ba': round(float(sub['avg_ba_pct'].mean() * 100), 1),
                })
            report['bps_spread_by_ticker'] = sorted(ticker_details, key=lambda x: x['mean_bs_vs_mkt_pct'])

    # ── 5. Sharpe Impact Estimate ──
    if 'bps_spread_credit' in report:
        bs_vs_mkt = report['bps_spread_credit']['mean_bs_vs_mkt_pct']
        # If BS overestimates credit by X%, the backtest Sharpe is inflated by roughly:
        # Sharpe_real ~ Sharpe_bt * (1 + X/100) for credit-dominant strategies
        # But the relationship is nonlinear; credit scales return, not vol, so:
        # Sharpe_impact ~ X% applied to return numerator only
        # Conservative estimate: if BS is X% off, Sharpe drops by ~X% * 0.7
        # (0.7 because some trades are profitable regardless of credit level)

        report['sharpe_impact'] = {
            'bs_credit_bias_pct': round(bs_vs_mkt, 2),
            'direction': 'BS overestimates (backtest is optimistic)' if bs_vs_mkt > 0
                         else 'BS underestimates (backtest is conservative)',
            'estimated_sharpe_haircut_pct': round(abs(bs_vs_mkt) * 0.7, 1) if bs_vs_mkt > 0 else 0,
            'note': (
                'If BS credit is X% higher than reality, backtest return is inflated by ~X%. '
                'Sharpe impact is ~0.7*X% because vol contribution is largely unaffected. '
                'Additional BA friction compounds this further.'
            ),
        }

    # ── 6. BA Impact on Sharpe ──
    if 'bid_ask_reality' in report and 'bid_ask_tier1' in report:
        t1 = report.get('bid_ask_tier1', {})
        t2 = report.get('bid_ask_tier2', {})

        ba_underestimate_t1 = max(0, (t1.get('actual_mean_ba_pct', 5) - t1.get('assumed_ba_pct', 5)))
        ba_underestimate_t2 = max(0, (t2.get('actual_mean_ba_pct', 7) - t2.get('assumed_ba_pct', 7)))

        # Weighted average (70% tier1, 30% tier2 roughly)
        weighted_ba_error = ba_underestimate_t1 * 0.7 + ba_underestimate_t2 * 0.3

        report['ba_impact'] = {
            'tier1_ba_underestimate_pct_pts': round(ba_underestimate_t1, 2),
            'tier2_ba_underestimate_pct_pts': round(ba_underestimate_t2, 2),
            'weighted_ba_underestimate': round(weighted_ba_error, 2),
            'note': 'Each 1% of BA underestimate reduces net credit by ~2% (2 legs * open/close).',
            'estimated_additional_sharpe_drag_pct': round(weighted_ba_error * 2 * 0.7, 1),
        }

    return report, puts_df, spreads_df


def print_summary(report):
    """Print a human-readable summary."""
    print("\n" + "=" * 70)
    print("BS MODEL VALIDATION vs LIVE OPTION CHAINS — SUMMARY")
    print("=" * 70)

    if 'price_accuracy' in report:
        pa = report['price_accuracy']
        print(f"\n--- PRICE ACCURACY (BS modeled vs market mid) ---")
        print(f"  Options analyzed: {pa['n_options']}")
        print(f"  Mean error: {pa['mean_error_pct']:+.1f}%  (positive = BS overprices)")
        print(f"  Median error: {pa['median_error_pct']:+.1f}%")
        print(f"  Std of error: {pa['std_error_pct']:.1f}%")
        print(f"  Within 10%: {pa['within_10pct']:.0f}%")
        print(f"  Within 20%: {pa['within_20pct']:.0f}%")
        print(f"  Within 30%: {pa['within_30pct']:.0f}%")
        bias = "CONSERVATIVE (real > BS)" if pa['bs_systematically_low'] else \
               "OPTIMISTIC (BS > real)" if pa['bs_systematically_high'] else "NEUTRAL"
        print(f"  Systematic bias: {bias}")

    if 'iv_comparison' in report:
        iv = report['iv_comparison']
        print(f"\n--- IMPLIED VOL: Modeled (rv20*1.15) vs Market ---")
        print(f"  Options: {iv['n_options']}")
        print(f"  Mean IV (yfinance): {iv['mean_iv_yfinance']:.1%}")
        print(f"  Mean IV (modeled):  {iv['mean_iv_modeled']:.1%}")
        print(f"  Mean error: {iv['mean_iv_error_pct']:+.1f}%")
        print(f"  Modeled too low:  {iv['modeled_iv_too_low_frac']:.0f}% of options")
        print(f"  Modeled too high: {iv['modeled_iv_too_high_frac']:.0f}% of options")

    if 'bid_ask_reality' in report:
        ba = report['bid_ask_reality']
        print(f"\n--- BID-ASK SPREAD REALITY ---")
        print(f"  Options: {ba['n_options']}")
        print(f"  Mean BA:   {ba['mean_ba_pct']:.1f}%")
        print(f"  Median BA: {ba['median_ba_pct']:.1f}%")
        print(f"  P75 BA:    {ba['p75_ba_pct']:.1f}%")
        print(f"  P90 BA:    {ba['p90_ba_pct']:.1f}%")

    for t, label in [('tier1', 'Large-Cap'), ('tier2', 'Mid-Cap')]:
        key = f'bid_ask_{t}'
        if key in report:
            ba = report[key]
            print(f"\n  {label}:")
            print(f"    Assumed: {ba['assumed_ba_pct']:.0f}%  |  Actual mean: {ba['actual_mean_ba_pct']:.1f}%  |  Actual P75: {ba['actual_p75_ba_pct']:.1f}%")
            optimistic = "YES — assumption too tight" if ba['assumption_is_optimistic'] else "No — assumption is conservative"
            print(f"    Assumption optimistic? {optimistic}")

    if 'bps_spread_credit' in report:
        sp = report['bps_spread_credit']
        print(f"\n--- BPS SPREAD NET CREDIT: BS vs Market ---")
        print(f"  Spreads compared: {sp['n_spreads']}")
        print(f"  Mean BS vs Market: {sp['mean_bs_vs_mkt_pct']:+.1f}%")
        print(f"  Median: {sp['median_bs_vs_mkt_pct']:+.1f}%")
        print(f"  BS overestimates credit: {sp['bs_overestimates_credit_frac']:.0f}% of spreads")
        print(f"  Avg market credit: ${sp['mean_mkt_credit']:.3f}")
        print(f"  Avg BS credit:     ${sp['mean_bs_credit']:.3f}")

    if 'sharpe_impact' in report:
        si = report['sharpe_impact']
        print(f"\n--- SHARPE IMPACT ESTIMATE ---")
        print(f"  BS credit bias: {si['bs_credit_bias_pct']:+.1f}%")
        print(f"  Direction: {si['direction']}")
        if si['estimated_sharpe_haircut_pct'] > 0:
            print(f"  Estimated Sharpe haircut: ~{si['estimated_sharpe_haircut_pct']:.1f}%")
            print(f"  (e.g., backtest Sharpe 2.0 -> real ~{2.0 * (1 - si['estimated_sharpe_haircut_pct']/100):.2f})")

    if 'ba_impact' in report:
        bi = report['ba_impact']
        print(f"\n--- BA ASSUMPTION IMPACT ---")
        print(f"  Tier1 BA underestimate: {bi['tier1_ba_underestimate_pct_pts']:.1f} pct pts")
        print(f"  Tier2 BA underestimate: {bi['tier2_ba_underestimate_pct_pts']:.1f} pct pts")
        if bi['estimated_additional_sharpe_drag_pct'] > 0:
            print(f"  Additional Sharpe drag: ~{bi['estimated_additional_sharpe_drag_pct']:.1f}%")

    print(f"\n{'=' * 70}")

    # Bottom line
    total_haircut = 0
    if 'sharpe_impact' in report:
        total_haircut += report['sharpe_impact'].get('estimated_sharpe_haircut_pct', 0)
    if 'ba_impact' in report:
        total_haircut += report['ba_impact'].get('estimated_additional_sharpe_drag_pct', 0)

    if total_haircut > 0:
        print(f"\nBOTTOM LINE: Combined BS pricing + BA assumption error = ~{total_haircut:.1f}% Sharpe haircut")
        print(f"  Backtest Sharpe 2.0 -> adjusted ~{2.0 * (1 - total_haircut/100):.2f}")
        print(f"  Backtest Sharpe 3.7 -> adjusted ~{3.7 * (1 - total_haircut/100):.2f}")
    else:
        print(f"\nBOTTOM LINE: BS model appears conservative or neutral — no Sharpe inflation from pricing.")

    print()


def main():
    print("=" * 70)
    print("BPS Black-Scholes Validation vs Live Option Chains")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S ET')}")
    print("=" * 70)

    # Use a subset for speed — 25 tier1 + 15 tier2 = 40 tickers
    tickers_tier1 = TIER1_TICKERS[:25]
    tickers_tier2 = TIER2_TICKERS[:15]

    print(f"\nFetching live option chains for {len(tickers_tier1)} large-cap + {len(tickers_tier2)} mid-cap tickers...")
    print(f"Target DTEs: 7, 14")
    print(f"Delta range: 0.10 - 0.40 (captures both 15-delta and 30-delta legs)")
    print()

    all_results = []

    print("--- Tier 1 (Large-Cap) ---")
    for sym in tickers_tier1:
        res = validate_ticker(sym, "tier1")
        all_results.append(res)
        time.sleep(0.3)  # Rate limit

    print("\n--- Tier 2 (Mid-Cap) ---")
    for sym in tickers_tier2:
        res = validate_ticker(sym, "tier2")
        all_results.append(res)
        time.sleep(0.3)

    valid_results = [r for r in all_results if r is not None]
    print(f"\nSuccessfully fetched: {len(valid_results)} / {len(all_results)} tickers")

    if len(valid_results) < 5:
        print("ERROR: Too few tickers with valid data. Markets may be closed.")
        return

    # Run aggregation
    report, puts_df, spreads_df = aggregate_results(valid_results)

    # Print summary
    print_summary(report)

    # Save outputs
    with open(OUTPUT / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    if not puts_df.empty:
        puts_df.to_csv(OUTPUT / "all_puts_data.csv", index=False)

    if not spreads_df.empty:
        spreads_df.to_csv(OUTPUT / "bps_spreads_comparison.csv", index=False)

    # Save per-ticker results
    ticker_summary = []
    for res in valid_results:
        ticker_summary.append({
            'symbol': res['symbol'],
            'tier': res['tier'],
            'spot': res['spot'],
            'rv20': round(res['rv20'], 4),
            'modeled_iv': round(res['modeled_iv'], 4),
        })
    with open(OUTPUT / "ticker_summary.json", "w") as f:
        json.dump(ticker_summary, f, indent=2)

    print(f"Results saved to {OUTPUT}/")
    print("Done.")


if __name__ == "__main__":
    main()
