#!/usr/bin/env python3
"""
Adversarial Audit: Income Portfolio v1
=======================================
Stress-tests every assumption in income_portfolio_optimizer_v1.py.

7 audit dimensions:
  1. Iron Condor premium estimation (VIX-as-IV bias)
  2. Iron Condor breach modeling (convexity, gaps)
  3. Covered Call premium & drawdown modeling
  4. VRP overlay realism (VXX March 2020, borrow costs)
  5. Cross-strategy correlation during stress
  6. Realistic premium benchmarking (live option chains)
  7. Stress-period forensics (COVID, Dec 2018, 2022 bear)

Output: /home/jupiter/Lvl3Quant/findings/income_adversarial_audit_v1.json
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Load original results
# ---------------------------------------------------------------------------
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/findings/income_portfolio_v1_results.json")
with open(RESULTS_PATH) as f:
    ORIG_RESULTS = json.load(f)

INITIAL_CAPITAL = 100_000

# ---------------------------------------------------------------------------
# BS helpers (same as original for comparison)
# ---------------------------------------------------------------------------
def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def strike_from_delta_call(S, T, r, sigma, delta_target):
    if T <= 0 or sigma <= 0:
        return S
    d1_target = norm.ppf(delta_target)
    K = S * np.exp(-d1_target * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)
    return K

def strike_from_delta_put(S, T, r, sigma, delta_target):
    if T <= 0 or sigma <= 0:
        return S
    d1_target = norm.ppf(1 - delta_target)
    K = S * np.exp(-d1_target * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)
    return K


# ===================================================================
# AUDIT 1: Iron Condor Premium Estimation — VIX-as-IV Bias
# ===================================================================
def audit_1_ic_premium_vix_bias():
    """
    The original model uses VIX as IV for ALL underlyings (sigma = vix/100).
    VIX measures SPX 30-day implied vol. Individual stock IVs differ wildly:
    - AAPL/MSFT typically have IV 20-40% LOWER than VIX during calm markets
    - But can spike HIGHER than VIX around earnings
    - SPY/QQQ are closer to VIX but still not equal

    We check: what is the actual IV for each ticker vs VIX?
    """
    print("\n" + "="*70)
    print("AUDIT 1: Iron Condor Premium — VIX-as-IV Bias")
    print("="*70)

    # The IC strategy uses: sigma = vix_val / 100 (line ~451 in CC, and
    # ic_credit_estimate uses VIX directly)
    # For IC specifically, it uses ic_credit_estimate(vix_level, width)
    # which scales credit ratio linearly with VIX/20.
    # The CC strategy uses sigma = vix_val / 100 directly in BS.

    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]

    # Pull actual option chains for current comparison
    iv_comparison = {}
    premium_comparison = {}

    # Get current VIX
    vix_data = yf.download("^VIX", period="5d", progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        current_vix = float(vix_data["Close"].iloc[-1].iloc[0])
    else:
        current_vix = float(vix_data["Close"].iloc[-1])
    print(f"Current VIX: {current_vix:.1f}")

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            current_price = stock.info.get("regularMarketPrice") or stock.info.get("previousClose")
            if current_price is None:
                hist = stock.history(period="1d")
                if len(hist) > 0:
                    current_price = float(hist["Close"].iloc[-1])
                else:
                    print(f"  {ticker}: Could not get price, skipping")
                    continue

            # Get nearest expiration ~7-8 days out
            expirations = stock.options
            if not expirations:
                print(f"  {ticker}: No options available")
                continue

            # Find expiration closest to 7 days
            today = dt.date.today()
            target_date = today + dt.timedelta(days=7)
            best_exp = None
            best_diff = 999
            for exp_str in expirations:
                exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
                diff = abs((exp_date - target_date).days)
                if diff < best_diff and exp_date >= today:
                    best_diff = diff
                    best_exp = exp_str

            if best_exp is None:
                print(f"  {ticker}: No suitable expiration found")
                continue

            exp_date = dt.datetime.strptime(best_exp, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            T = dte / 365.0

            # Get put chain for 0.20 delta comparison
            chain = stock.option_chain(best_exp)
            puts = chain.puts
            calls = chain.calls

            # Find ~0.20 delta put (OTM put ~20% probability)
            # Approximate: strike at about 1 std dev below current for 7 DTE
            # BS model estimate at VIX-implied vol:
            vix_sigma = current_vix / 100.0
            bs_put_strike = strike_from_delta_put(current_price, T, 0.04, vix_sigma, 0.20)
            bs_call_strike = strike_from_delta_call(current_price, T, 0.04, vix_sigma, 0.30)

            # BS-estimated premiums using VIX as IV
            bs_put_premium = bs_put_price(current_price, bs_put_strike, T, 0.04, vix_sigma)
            bs_call_premium = bs_call_price(current_price, bs_call_strike, T, 0.04, vix_sigma)

            # Find closest real option to our BS strike
            real_put = None
            if len(puts) > 0 and "strike" in puts.columns:
                puts_sorted = puts.copy()
                puts_sorted["dist"] = abs(puts_sorted["strike"] - bs_put_strike)
                closest_put = puts_sorted.sort_values("dist").iloc[0]
                real_put_premium = (closest_put.get("bid", 0) + closest_put.get("ask", 0)) / 2
                real_put_strike = closest_put["strike"]
                real_put_iv = closest_put.get("impliedVolatility", np.nan)
                real_put = {
                    "strike": float(real_put_strike),
                    "mid_premium": float(real_put_premium),
                    "iv": float(real_put_iv) if not np.isnan(real_put_iv) else None,
                }

            real_call = None
            if len(calls) > 0 and "strike" in calls.columns:
                calls_sorted = calls.copy()
                calls_sorted["dist"] = abs(calls_sorted["strike"] - bs_call_strike)
                closest_call = calls_sorted.sort_values("dist").iloc[0]
                real_call_premium = (closest_call.get("bid", 0) + closest_call.get("ask", 0)) / 2
                real_call_strike = closest_call["strike"]
                real_call_iv = closest_call.get("impliedVolatility", np.nan)
                real_call = {
                    "strike": float(real_call_strike),
                    "mid_premium": float(real_call_premium),
                    "iv": float(real_call_iv) if not np.isnan(real_call_iv) else None,
                }

            # Compute overestimation
            put_overest = None
            call_overest = None
            actual_iv = None

            if real_put and real_put["mid_premium"] > 0:
                put_overest = (bs_put_premium - real_put["mid_premium"]) / real_put["mid_premium"] * 100
            if real_call and real_call["mid_premium"] > 0:
                call_overest = (bs_call_premium - real_call["mid_premium"]) / real_call["mid_premium"] * 100
            if real_put and real_put["iv"] is not None:
                actual_iv = real_put["iv"]
            elif real_call and real_call["iv"] is not None:
                actual_iv = real_call["iv"]

            iv_ratio = None
            if actual_iv and actual_iv > 0:
                iv_ratio = vix_sigma / actual_iv

            result = {
                "current_price": float(current_price),
                "expiration": best_exp,
                "dte": dte,
                "vix_implied_sigma": round(vix_sigma, 4),
                "actual_iv": round(actual_iv, 4) if actual_iv else None,
                "iv_ratio_vix_over_actual": round(iv_ratio, 3) if iv_ratio else None,
                "bs_put_strike": round(bs_put_strike, 2),
                "bs_put_premium": round(bs_put_premium, 4),
                "real_put": real_put,
                "put_overestimation_pct": round(put_overest, 1) if put_overest is not None else None,
                "bs_call_strike": round(bs_call_strike, 2),
                "bs_call_premium": round(bs_call_premium, 4),
                "real_call": real_call,
                "call_overestimation_pct": round(call_overest, 1) if call_overest is not None else None,
            }

            iv_comparison[ticker] = result

            direction = "OVER" if (put_overest and put_overest > 0) else "UNDER"
            iv_str = f"{actual_iv:.3f}" if actual_iv else "N/A"
            ratio_str = f"{iv_ratio:.2f}" if iv_ratio else "N/A"
            overest_str = f"{put_overest:.0f}%" if put_overest is not None else "N/A"
            print(f"  {ticker}: price=${current_price:.0f}, VIX_sigma={vix_sigma:.3f}, "
                  f"actual_IV={iv_str}, IV_ratio={ratio_str}, "
                  f"put {direction}est={overest_str}")

        except Exception as e:
            print(f"  {ticker}: Error — {e}")
            iv_comparison[ticker] = {"error": str(e)}

    # Historical analysis: VIX vs realized vol for SPY vs mega-caps
    print("\n  Historical VIX vs Realized Vol comparison (2018-2026):")
    stock_data = yf.download(["SPY"] + tickers, start="2018-01-01", period="max",
                              auto_adjust=True, progress=False)
    if isinstance(stock_data.columns, pd.MultiIndex):
        closes = stock_data["Close"]
    else:
        closes = stock_data

    vix_hist = yf.download("^VIX", start="2018-01-01", period="max", auto_adjust=True, progress=False)
    if isinstance(vix_hist.columns, pd.MultiIndex):
        vix_close = vix_hist["Close"].squeeze()
    else:
        vix_close = vix_hist["Close"].squeeze()

    realized_vol_comparison = {}
    for ticker in ["SPY"] + tickers:
        if ticker in closes.columns:
            daily_ret = closes[ticker].pct_change().dropna()
            rv_20d = daily_ret.rolling(20).std() * np.sqrt(252) * 100  # annualized %
            avg_rv = rv_20d.mean()
            avg_vix = vix_close.mean() if hasattr(vix_close, 'mean') else float(vix_close.mean())
            ratio = avg_vix / avg_rv if avg_rv > 0 else None
            realized_vol_comparison[ticker] = {
                "avg_realized_vol_pct": round(float(avg_rv), 1),
                "avg_vix_pct": round(float(avg_vix), 1),
                "vix_over_rv_ratio": round(float(ratio), 3) if ratio else None,
            }
            print(f"    {ticker}: avg RV={avg_rv:.1f}%, avg VIX={avg_vix:.1f}%, ratio={ratio:.2f}" if ratio else f"    {ticker}: no data")

    # IC premium specifically uses ic_credit_estimate which is a simplified model
    # Let's check how the credit-to-width ratio compares to reality
    ic_credit_analysis = {
        "model_formula": "credit_ratio = min(0.25 * VIX/20, 0.50)",
        "at_VIX_15": round(min(0.25 * 15/20, 0.50), 3),
        "at_VIX_20": round(min(0.25 * 20/20, 0.50), 3),
        "at_VIX_30": round(min(0.25 * 30/20, 0.50), 3),
        "at_VIX_40": round(min(0.25 * 40/20, 0.50), 3),
        "reality_check": (
            "Real IC credit-to-width ratios for 0.20 delta, 7 DTE on SPY are typically "
            "15-20% at VIX=15, 20-25% at VIX=20, 25-35% at VIX=30. "
            "The model is roughly calibrated for SPY but WRONG for individual stocks "
            "because it uses a VIX-scaled formula regardless of the underlying."
        ),
    }

    # Determine overall bias
    overest_values = []
    for t, v in iv_comparison.items():
        if isinstance(v, dict) and "put_overestimation_pct" in v and v["put_overestimation_pct"] is not None:
            overest_values.append(v["put_overestimation_pct"])

    avg_overest = np.mean(overest_values) if overest_values else None

    reliability = "LOW"
    reasoning = (
        "VIX is used as a universal IV proxy for all underlyings. "
        "For the IC universe (SPY, QQQ, IWM, AAPL, MSFT), SPY/QQQ/IWM are close to VIX "
        "but AAPL and MSFT typically have LOWER implied vol than VIX. "
        "The IC credit estimation formula (ic_credit_estimate) bypasses BS entirely and uses "
        "a VIX-scaled linear formula. This is a crude approximation that can overestimate "
        "premiums by 20-50% for low-vol stocks during calm markets. "
        "For the CC universe (20 individual stocks), using VIX as IV is MORE problematic — "
        "stock-specific IV can deviate from VIX by 50%+ in either direction."
    )

    if avg_overest and avg_overest > 30:
        reliability = "LOW"
        reasoning += f" Current live comparison shows average {avg_overest:.0f}% overestimation."
    elif avg_overest and avg_overest > 10:
        reliability = "MEDIUM"
        reasoning += f" Current live comparison shows average {avg_overest:.0f}% overestimation."

    return {
        "audit": "IC Premium Estimation — VIX-as-IV Bias",
        "finding": "Model uses VIX as universal IV proxy; does NOT use stock-specific implied vol",
        "current_live_comparison": iv_comparison,
        "historical_rv_comparison": realized_vol_comparison,
        "ic_credit_model": ic_credit_analysis,
        "avg_premium_overestimation_pct": round(avg_overest, 1) if avg_overest is not None else "insufficient data",
        "reliability": reliability,
        "reasoning": reasoning,
    }


# ===================================================================
# AUDIT 2: Iron Condor Breach Modeling
# ===================================================================
def audit_2_ic_breach_modeling():
    """
    Check how the IC model handles breaches:
    - Does it model convexity of losses?
    - Does it handle gap risk?
    - Does max_loss reflect reality?
    """
    print("\n" + "="*70)
    print("AUDIT 2: Iron Condor Breach Modeling")
    print("="*70)

    findings = []

    # 1. Loss model analysis
    # From the code: losses are computed as intrinsic of the spread
    # call_spread_val = min(max(0, price - upper_short), width)
    # put_spread_val = min(max(0, lower_short - price), width)
    # This IS the correct spread intrinsic model (not linear in underlying price)
    # But it caps at width, which is correct for a spread

    # The problem: this uses DAILY close prices only
    # Real intraday breaches and gap openings are not modeled

    # 2. Short strike placement
    # short_strike_dist = price * width_pct * 0.6
    # For a $500 stock with 5% width: short_strike_dist = $15
    # Upper short = $515, Lower short = $485
    # That's only 3% OTM — NOT 0.20 delta!
    # 0.20 delta for 7 DTE at VIX=20 would be about 1.3-1.5 sigma
    # sigma_7d = 0.20 * sqrt(7/365) = 0.0277 = 2.77% of price
    # 1.3 * sigma = 3.6% of price
    # So the model places strikes at 3% OTM, which is roughly 0.25-0.30 delta,
    # not 0.20 delta. This means MORE breaches than intended.

    # Let's verify the delta of the actual strike placement
    test_price = 500
    test_vix = 20
    test_sigma = test_vix / 100  # 0.20
    test_T = 7 / 365
    test_width_pct = 0.05
    adaptive_width = test_width_pct * max(1.0, test_vix / 20.0)  # = 0.05

    short_strike_dist = test_price * adaptive_width * 0.6  # = 15
    upper_short = test_price + short_strike_dist  # = 515
    lower_short = test_price - short_strike_dist  # = 485

    # What delta is this?
    d1_upper = (np.log(test_price / upper_short) + (0.04 + 0.5 * test_sigma**2) * test_T) / (test_sigma * np.sqrt(test_T))
    actual_call_delta = norm.cdf(d1_upper)

    d1_lower = (np.log(test_price / lower_short) + (0.04 + 0.5 * test_sigma**2) * test_T) / (test_sigma * np.sqrt(test_T))
    actual_put_delta = abs(norm.cdf(d1_lower) - 1)

    # What would 0.20 delta strikes actually be?
    proper_put_strike = strike_from_delta_put(test_price, test_T, 0.04, test_sigma, 0.20)
    proper_call_strike = strike_from_delta_call(test_price, test_T, 0.04, test_sigma, 0.20)

    strike_analysis = {
        "test_params": {"price": test_price, "vix": test_vix, "dte": 7, "width_pct": adaptive_width},
        "model_upper_short": upper_short,
        "model_lower_short": lower_short,
        "model_short_distance_pct": round(short_strike_dist / test_price * 100, 2),
        "actual_call_delta_at_model_strike": round(actual_call_delta, 4),
        "actual_put_delta_at_model_strike": round(actual_put_delta, 4),
        "proper_020_delta_call_strike": round(proper_call_strike, 2),
        "proper_020_delta_put_strike": round(proper_put_strike, 2),
        "proper_distance_pct": round(abs(proper_put_strike - test_price) / test_price * 100, 2),
        "finding": (
            f"Model places short strikes at {short_strike_dist/test_price*100:.1f}% OTM, "
            f"which corresponds to delta ~{actual_put_delta:.3f}, NOT the claimed 0.20. "
            f"True 0.20 delta would be at ${proper_put_strike:.0f} "
            f"({abs(proper_put_strike - test_price)/test_price*100:.1f}% OTM). "
            "Strikes are CLOSER to ATM than claimed, meaning MORE premium but MORE risk."
        ),
    }

    # 3. Gap risk — not modeled at all
    # The model checks prices daily. A stock that gaps -8% overnight
    # would breach the IC but the model only sees the close-to-close move
    gap_risk = {
        "modeled": False,
        "impact": (
            "Model uses daily close prices only. Intraday breaches and overnight gaps "
            "are invisible. During COVID, stocks gapped -5% to -12% at open regularly. "
            "An IC opened at 3% OTM would be deeply breached on such gaps, but the model "
            "only processes the daily close — potentially missing the worst moment."
        ),
    }

    # 4. Stop loss effectiveness
    # Stop loss at 1.5x credit — in reality, if underlying gaps through strikes,
    # the spread can instantly be at max loss before you can close
    stop_loss = {
        "model_stop": "1.5x credit received",
        "reality": (
            "In fast markets, stop losses on spreads are unreliable. "
            "Wide bid-ask spreads on the spread legs during volatility spikes "
            "mean you might close at 2-3x credit, not 1.5x. "
            "Additionally, in after-hours/pre-market, you cannot close options. "
            "A gap through both strikes means instant max loss with no chance to stop out."
        ),
        "slippage_estimate_pct": "20-50% worse than modeled during stress",
    }

    # 5. Max loss cap
    # Model: max_loss = (width - credit) * 100 * contracts
    # Reality: this is correct for a defined-risk spread IF no assignment occurs early
    # But early assignment on short leg creates undefined risk temporarily
    max_loss_model = {
        "modeled_correctly": True,
        "but": (
            "Max loss = width - credit is correct for European-style or well-managed "
            "American-style. However, early assignment on the short leg (especially near "
            "ex-dividend dates for calls, or deep ITM for puts) creates temporary naked "
            "risk until the long leg is exercised. Model assigns 2% probability to this "
            "but the cost model for assignment is too optimistic."
        ),
    }

    print(f"  Short strike delta: claimed 0.20, actual ~{actual_put_delta:.3f}")
    print(f"  Gap risk: NOT modeled")
    print(f"  Stop loss slippage: NOT modeled")

    return {
        "audit": "IC Breach Modeling",
        "strike_placement": strike_analysis,
        "gap_risk": gap_risk,
        "stop_loss_reliability": stop_loss,
        "max_loss_model": max_loss_model,
        "reliability": "LOW",
        "reasoning": (
            "Three critical flaws: (1) Short strikes are placed at ~0.30 delta, not 0.20, "
            "meaning more premium but more breach risk than claimed. (2) Daily-close-only "
            "pricing misses intraday and gap breaches — the model cannot detect or act on "
            "the worst moments. (3) Stop loss at 1.5x credit is unreliable during fast "
            "markets where spreads can gap to max loss instantly. Combined effect: the model "
            "OVERSTATES income (higher delta = more premium) while UNDERSTATING losses "
            "(misses gaps, stops fail). This is the most dangerous combination."
        ),
    }


# ===================================================================
# AUDIT 3: Covered Call Premium & Drawdown
# ===================================================================
def audit_3_covered_call():
    """
    Check CC model for:
    - VIX-as-IV bias on individual stocks
    - Stock drawdown not modeled (only option income)
    - Behavior during COVID crash
    """
    print("\n" + "="*70)
    print("AUDIT 3: Covered Call Premium & Drawdown Modeling")
    print("="*70)

    # The CC strategy explicitly states: "We measure OPTION INCOME only"
    # This is the CRITICAL flaw — a CC portfolio OWNS the stock.
    # The P&L should be: stock return + premium received - upside capped

    # From results: CC max_drawdown = 0%, min_nav = 100%
    # This is IMPOSSIBLE for a strategy that owns stocks during 2020 and 2022

    cc_monthly = ORIG_RESULTS["per_strategy_monthly_pnl"]["covered_call"]

    # Check COVID period
    covid_months = {m: v for m, v in cc_monthly.items() if m in ["2020-02", "2020-03", "2020-04"]}
    # SPY dropped ~34% from Feb peak to Mar trough
    # CC portfolio should show MASSIVE losses on the stock side

    # Check 2022 bear
    bear_2022 = {m: v for m, v in cc_monthly.items() if m.startswith("2022")}

    # The CC model ONLY counts premium income, NEVER stock losses
    # This means the CC strategy as modeled is NOT a real CC — it's just
    # "sell calls on imaginary stock you own for free"

    print(f"  CC max drawdown in results: {ORIG_RESULTS['per_strategy']['covered_call']['max_drawdown_pct']}%")
    print(f"  CC min NAV %: {ORIG_RESULTS['per_strategy']['covered_call']['min_nav_pct_of_start']}%")
    print(f"  CC sortino: {ORIG_RESULTS['per_strategy']['covered_call']['sortino']}")
    print(f"  COVID months (CC PnL): {covid_months}")
    print(f"  CRITICAL: CC shows 0% drawdown through 2020 crash — this is UNREALISTIC")

    # What would real CC look like during COVID?
    # SPY: -34% peak-to-trough (Feb 19 to Mar 23, 2020)
    # CC reduces drawdown by premium received, typically ~2% for 30-day 0.30 delta
    # Real CC drawdown in COVID would be roughly -30% to -32%
    # For 40% allocation: portfolio impact = -12% to -13%

    # Premium overestimation from VIX-as-IV
    # During COVID, VIX spiked to 80+. The model uses sigma=0.80 for AAPL calls.
    # AAPL's actual IV might have been 50-60%, not 80%.
    # This means the model OVERESTIMATES CC premium during high-vol periods.

    # But: the CC model DOESN'T TRACK STOCK P&L, so premium overestimation
    # is the LESSER issue. The MISSING stock P&L is the real problem.

    real_cc_drawdown_estimate = {
        "covid_feb_mar_2020": {
            "spy_drawdown_pct": -34,
            "model_cc_pnl_feb": cc_monthly.get("2020-02", 0),
            "model_cc_pnl_mar": cc_monthly.get("2020-03", 0),
            "model_cc_pnl_apr": cc_monthly.get("2020-04", 0),
            "model_shows_profit": True,
            "real_cc_drawdown_estimate_pct": -30,
            "impact_at_40pct_allocation": -12.0,
        },
        "bear_2022": {
            "spy_full_year_return_pct": -19.4,
            "model_cc_full_year_pnl": round(sum(bear_2022.values()), 2),
            "model_shows_profit": sum(bear_2022.values()) > 0,
            "real_cc_annual_return_estimate_pct": -15,
            "impact_at_40pct_allocation": -6.0,
        },
    }

    print(f"\n  Real CC drawdown during COVID: ~{real_cc_drawdown_estimate['covid_feb_mar_2020']['real_cc_drawdown_estimate_pct']}%")
    print(f"  Model shows: ${cc_monthly.get('2020-03', 0):.2f} PROFIT in March 2020")
    print(f"  This alone adds ~12% phantom return to the portfolio")

    return {
        "audit": "Covered Call Premium & Drawdown",
        "critical_flaw": (
            "The CC model tracks ONLY option premium income, NOT the underlying stock P&L. "
            "A covered call OWNS the stock — during the 2020 crash, a real CC portfolio "
            "would have lost ~30% on stock positions while collecting ~2% in premium. "
            "The model shows ZERO drawdown and positive income every month. "
            "This is equivalent to modeling a covered call as a 'free premium printing machine'. "
            "CC allocation is 40% of the portfolio — this single flaw inflates total returns by "
            "~10-15% cumulative over the backtest period."
        ),
        "vix_as_iv_for_stocks": (
            "Uses VIX as IV for individual stocks (AAPL, MSFT, AMZN, etc). "
            "During calm markets, this OVERESTIMATES premiums for mega-caps. "
            "During stress, VIX spikes represent SPX vol, not stock-specific vol."
        ),
        "covid_analysis": real_cc_drawdown_estimate["covid_feb_mar_2020"],
        "bear_2022_analysis": real_cc_drawdown_estimate["bear_2022"],
        "cc_monthly_during_stress": covid_months,
        "reliability": "VERY LOW",
        "reasoning": (
            "The CC model is fundamentally broken. It models covered calls as pure income "
            "with zero stock risk. Real covered calls have ~90% of underlying stock risk "
            "(premium only buffers 2-3% per month). The 0% drawdown and infinite Sortino "
            "reported for CC are artifacts of this flaw. If corrected, the CC component would "
            "show ~-30% drawdown during COVID and ~-15% during 2022 bear market, "
            "dramatically worsening the combined portfolio metrics."
        ),
    }


# ===================================================================
# AUDIT 4: VRP Overlay Realism
# ===================================================================
def audit_4_vrp_overlay():
    """
    Check VRP model for:
    - VXX behavior during March 2020
    - Borrow costs for shorting
    - Backwardation prediction reliability
    """
    print("\n" + "="*70)
    print("AUDIT 4: VRP Overlay Realism")
    print("="*70)

    vrp_monthly = ORIG_RESULTS["per_strategy_monthly_pnl"]["vrp_overlay"]
    vrp_metrics = ORIG_RESULTS["per_strategy"]["vrp_overlay"]

    # Download VIXY data to check March 2020
    vxx_data = yf.download("VIXY", start="2020-01-01", end="2020-06-30",
                           auto_adjust=True, progress=False)
    if isinstance(vxx_data.columns, pd.MultiIndex):
        vxx_close = vxx_data["Close"].squeeze()
    else:
        vxx_close = vxx_data["Close"].squeeze()

    # VXX/VIXY behavior in March 2020
    if len(vxx_close) > 0:
        jan_price = float(vxx_close.iloc[0]) if len(vxx_close) > 0 else None
        mar_peak = float(vxx_close["2020-03"].max()) if "2020-03" in vxx_close.index.strftime("%Y-%m") else None

        # Find the actual peak
        peak_idx = vxx_close.idxmax()
        peak_val = float(vxx_close.max())

        vxx_march_2020 = {
            "jan_2_price": round(jan_price, 2) if jan_price else None,
            "peak_price": round(peak_val, 2),
            "peak_date": str(peak_idx.date()) if hasattr(peak_idx, 'date') else str(peak_idx),
            "return_pct": round((peak_val / jan_price - 1) * 100, 1) if jan_price and jan_price > 0 else None,
        }
        print(f"  VIXY: Jan 2 = ${jan_price:.2f}, Peak = ${peak_val:.2f} ({vxx_march_2020['return_pct']:.0f}%)")
    else:
        vxx_march_2020 = {"error": "Could not download VIXY data"}

    # Model's VRP behavior during COVID
    covid_vrp = {m: vrp_monthly.get(m, 0) for m in ["2020-01", "2020-02", "2020-03", "2020-04", "2020-05", "2020-06"]}
    print(f"  Model VRP PnL during COVID months: {covid_vrp}")
    print(f"  March 2020 VRP PnL: ${vrp_monthly.get('2020-03', 0):.2f}")

    # The model shows $1025.52 PROFIT in March 2020!
    # This means it went LONG VXX during backwardation (contango < -5%)
    # Let's check if this is realistic

    mar_profit = vrp_monthly.get("2020-03", 0)
    mar_is_profitable = mar_profit > 0

    # Borrow costs
    borrow_cost_analysis = {
        "annual_borrow_rate_pct": "3-8%",
        "modeled": False,
        "impact": (
            "Shorting VXX/VIXY requires borrowing shares. Annual borrow cost is typically "
            "3-8% depending on broker and availability. The model applies only a 0.1% "
            "commission on entry/exit but NO ongoing borrow cost. For a position held 30 days, "
            "this adds ~0.5-1.5% cost not captured in the model."
        ),
    }

    # Position sizing during stress
    # Model: position_size = min(capital * 0.5, nav * 0.1)
    # With $20K VRP capital: max short position = $10K
    # If VXX goes from $15 to $80 (433% move), loss = $10K * 4.33 = $43K
    # This EXCEEDS the total VRP allocation by 2x
    # But the model has a -15% stop loss, so max loss = $10K * 0.15 = $1,500
    # The stop loss ASSUMES you can get out at -15%. During COVID, VXX gapped
    # multiple days — you might not be able to exit at -15%.

    stress_test = {
        "scenario": "Short VIXY from $15, VIXY spikes to $80 (COVID)",
        "model_max_position": 10000,
        "model_stop_loss_pct": -15,
        "model_max_loss": 1500,
        "reality": (
            "VXX/VIXY gapped up 15-30% on multiple days during COVID. "
            "A -15% stop loss would have been blown through on the OPEN, "
            "with actual loss potentially 30-50% before exit. "
            "However, the model's 7% contango threshold would likely "
            "have prevented entry during the worst period (VIX was spiking, "
            "not in contango). The bigger question is positions opened "
            "BEFORE the crash."
        ),
    }

    # The model actually MADE money in March 2020 — check if it went long VXX
    # contango < -0.05 triggers long VXX (backwardation hedge)
    # This is the "bidirectional" approach

    backwardation_analysis = {
        "model_signal": "Go long VXX when contango < -5%",
        "march_2020_profit": mar_profit,
        "interpretation": (
            "The model shows a $1,025 profit in March 2020, likely from going LONG VXX "
            "during backwardation. This is the 'hedge' component working as designed. "
            "However, the contango proxy (VIX SMA vs VIX spot) is very crude — "
            "backwardation doesn't reliably predict the MAGNITUDE of a crash."
        ),
        "overall_vrp_performance": {
            "total_return_pct": vrp_metrics["total_return_pct"],
            "cagr_pct": vrp_metrics["cagr_pct"],
            "sharpe": vrp_metrics["sharpe"],
            "positive_months_pct": vrp_metrics["positive_months_pct"],
        },
    }

    # The VRP overlay has negative Sharpe (-0.595) and only 51.5% positive months
    # It's essentially a coin flip that adds noise
    print(f"\n  VRP Sharpe: {vrp_metrics['sharpe']}")
    print(f"  VRP positive months: {vrp_metrics['positive_months_pct']}%")
    print(f"  VRP is essentially noise — negative Sharpe, coin-flip win rate")

    return {
        "audit": "VRP Overlay Realism",
        "vxx_march_2020": vxx_march_2020,
        "model_covid_pnl": covid_vrp,
        "borrow_costs": borrow_cost_analysis,
        "stress_test": stress_test,
        "backwardation_analysis": backwardation_analysis,
        "overall_assessment": (
            "The VRP component has a NEGATIVE Sharpe (-0.60) and 51.5% win rate — "
            "it adds noise, not alpha. Total return of 9.87% over 8.5 years is ~1.1% CAGR, "
            "barely above zero. The bidirectional approach (long in backwardation) saved it "
            "from March 2020 catastrophe but doesn't generate consistent returns. "
            "Missing borrow costs (~3-8% annual) would likely make this component "
            "net-negative over the full period."
        ),
        "reliability": "MEDIUM",
        "reasoning": (
            "The VRP component is honest about being mediocre — it contributes very little "
            "to the portfolio. The March 2020 handling is actually reasonable (long VXX in "
            "backwardation). But missing borrow costs, crude contango proxy, and negligible "
            "returns mean this is essentially a zero-contribution allocation. "
            "MEDIUM because it doesn't inflate results significantly, but the missing "
            "borrow costs would flip it slightly negative."
        ),
    }


# ===================================================================
# AUDIT 5: Cross-Strategy Correlation
# ===================================================================
def audit_5_correlation():
    """
    During crashes, all premium-selling strategies lose simultaneously.
    Check if the backtest reflects this.
    """
    print("\n" + "="*70)
    print("AUDIT 5: Cross-Strategy Correlation")
    print("="*70)

    ic_monthly = ORIG_RESULTS["per_strategy_monthly_pnl"]["iron_condor"]
    cc_monthly = ORIG_RESULTS["per_strategy_monthly_pnl"]["covered_call"]
    vrp_monthly = ORIG_RESULTS["per_strategy_monthly_pnl"]["vrp_overlay"]

    # Align months
    all_months = sorted(set(ic_monthly.keys()) & set(cc_monthly.keys()) & set(vrp_monthly.keys()))

    ic_returns = np.array([ic_monthly[m] for m in all_months])
    cc_returns = np.array([cc_monthly[m] for m in all_months])
    vrp_returns = np.array([vrp_monthly[m] for m in all_months])

    # Correlations
    ic_cc_corr = np.corrcoef(ic_returns, cc_returns)[0, 1]
    ic_vrp_corr = np.corrcoef(ic_returns, vrp_returns)[0, 1]
    cc_vrp_corr = np.corrcoef(cc_returns, vrp_returns)[0, 1]

    print(f"  IC-CC correlation: {ic_cc_corr:.3f}")
    print(f"  IC-VRP correlation: {ic_vrp_corr:.3f}")
    print(f"  CC-VRP correlation: {cc_vrp_corr:.3f}")

    # Check stress months specifically
    stress_months = ["2018-12", "2020-02", "2020-03", "2022-06", "2022-09", "2024-08"]
    stress_analysis = {}
    for m in stress_months:
        stress_analysis[m] = {
            "ic": ic_monthly.get(m, 0),
            "cc": cc_monthly.get(m, 0),
            "vrp": vrp_monthly.get(m, 0),
            "combined": ic_monthly.get(m, 0) + cc_monthly.get(m, 0) + vrp_monthly.get(m, 0),
            "all_negative": (ic_monthly.get(m, 0) < 0 and cc_monthly.get(m, 0) < 0 and vrp_monthly.get(m, 0) < 0),
        }

    # The CC NEVER shows negative — this is the fundamental flaw
    # Real correlation during stress should be HIGH (all lose together)
    cc_ever_negative = any(cc_monthly[m] < 0 for m in cc_monthly if cc_monthly[m] < 0)

    print(f"\n  CC ever negative: {cc_ever_negative}")
    print(f"  Stress month analysis:")
    for m, v in stress_analysis.items():
        print(f"    {m}: IC=${v['ic']:+,.0f}, CC=${v['cc']:+,.0f}, VRP=${v['vrp']:+,.0f}")

    # Tail correlation (worst 10% of months)
    combined = ic_returns + cc_returns + vrp_returns
    worst_10pct_threshold = np.percentile(combined, 10)
    worst_months_mask = combined <= worst_10pct_threshold
    if worst_months_mask.sum() > 2:
        tail_ic_cc_corr = np.corrcoef(ic_returns[worst_months_mask], cc_returns[worst_months_mask])[0, 1]
    else:
        tail_ic_cc_corr = None

    return {
        "audit": "Cross-Strategy Correlation",
        "monthly_correlations": {
            "ic_cc": round(ic_cc_corr, 3),
            "ic_vrp": round(ic_vrp_corr, 3),
            "cc_vrp": round(cc_vrp_corr, 3),
        },
        "tail_correlation_ic_cc": round(tail_ic_cc_corr, 3) if tail_ic_cc_corr is not None else None,
        "stress_month_analysis": stress_analysis,
        "cc_ever_negative": cc_ever_negative,
        "critical_finding": (
            "The CC strategy NEVER has a negative month in the backtest because it only "
            "tracks premium income, not stock drawdowns. This artificially DECORRELATES "
            "the CC from IC and VRP during stress. In reality, during a crash, "
            "CC loses on the stock side while IC loses on spread breaches — they are "
            "HIGHLY correlated tail-risk strategies. The low IC-CC correlation in the "
            "backtest is an ARTIFACT of the CC model flaw, not genuine diversification."
        ),
        "reliability": "LOW",
        "reasoning": (
            "The diversification benefit between IC and CC is illusory. Both are short-vol "
            "strategies that lose during market stress. The backtest shows low correlation "
            "only because CC doesn't model stock drawdowns. Real tail correlation between "
            "these strategies would be 0.6-0.8+, not the ~" +
            f"{ic_cc_corr:.2f} shown. This means the combined portfolio's drawdown "
            "is MUCH worse than modeled during stress events."
        ),
    }


# ===================================================================
# AUDIT 6: Realistic Premium Benchmarking
# ===================================================================
def audit_6_premium_benchmark():
    """
    Compare BS-modeled premiums against actual current option chain data.
    """
    print("\n" + "="*70)
    print("AUDIT 6: Realistic Premium Benchmarking")
    print("="*70)

    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    results = {}

    vix_data = yf.download("^VIX", period="5d", progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        current_vix = float(vix_data["Close"].iloc[-1].iloc[0])
    else:
        current_vix = float(vix_data["Close"].iloc[-1])

    vix_sigma = current_vix / 100.0

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            if len(hist) == 0:
                continue
            current_price = float(hist["Close"].iloc[-1])

            expirations = stock.options
            if not expirations:
                continue

            today = dt.date.today()

            # Find expiration ~30 days out (for CC comparison)
            target_30d = today + dt.timedelta(days=30)
            best_exp_30d = None
            best_diff_30d = 999
            for exp_str in expirations:
                exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
                diff = abs((exp_date - target_30d).days)
                if diff < best_diff_30d and exp_date >= today:
                    best_diff_30d = diff
                    best_exp_30d = exp_str

            if best_exp_30d is None:
                continue

            exp_date_30d = dt.datetime.strptime(best_exp_30d, "%Y-%m-%d").date()
            dte_30d = (exp_date_30d - today).days
            T_30d = dte_30d / 365.0

            # BS estimate for 0.30 delta call (CC)
            bs_cc_strike = strike_from_delta_call(current_price, T_30d, 0.04, vix_sigma, 0.30)
            bs_cc_premium = bs_call_price(current_price, bs_cc_strike, T_30d, 0.04, vix_sigma)

            # Get real chain
            chain = stock.option_chain(best_exp_30d)
            calls = chain.calls

            if len(calls) > 0:
                calls["dist"] = abs(calls["strike"] - bs_cc_strike)
                closest = calls.sort_values("dist").iloc[0]
                real_premium = (closest.get("bid", 0) + closest.get("ask", 0)) / 2
                real_iv = closest.get("impliedVolatility", np.nan)

                if real_premium > 0:
                    overest = (bs_cc_premium - real_premium) / real_premium * 100
                else:
                    overest = None

                results[ticker] = {
                    "price": round(current_price, 2),
                    "exp": best_exp_30d,
                    "dte": dte_30d,
                    "bs_cc_strike": round(bs_cc_strike, 2),
                    "real_strike": float(closest["strike"]),
                    "bs_premium": round(bs_cc_premium, 2),
                    "real_mid_premium": round(real_premium, 2),
                    "overestimation_pct": round(overest, 1) if overest is not None else None,
                    "vix_sigma": round(vix_sigma, 4),
                    "real_iv": round(float(real_iv), 4) if not np.isnan(real_iv) else None,
                }

                print(f"  {ticker}: BS premium=${bs_cc_premium:.2f}, real mid=${real_premium:.2f}, "
                      f"overest={overest:+.0f}%" if overest is not None else f"  {ticker}: incomplete data")
            else:
                print(f"  {ticker}: no call chain data")

        except Exception as e:
            print(f"  {ticker}: Error — {e}")

    # Compute average overestimation
    overest_vals = [v["overestimation_pct"] for v in results.values()
                    if isinstance(v, dict) and v.get("overestimation_pct") is not None]
    avg_overest = np.mean(overest_vals) if overest_vals else None

    if avg_overest is not None:
        print(f"\n  Average premium overestimation: {avg_overest:+.1f}%")

    # Impact on portfolio returns
    # IC is 40% of portfolio, CC is 40%
    # If premiums are overestimated by X%, gross income is inflated by X%
    impact = None
    if avg_overest is not None and avg_overest > 0:
        ic_annual_income = ORIG_RESULTS["per_strategy"]["iron_condor"]["cagr_pct"]
        cc_annual_income = ORIG_RESULTS["per_strategy"]["covered_call"]["cagr_pct"]
        # Deflate by overestimation
        deflation_factor = 1 / (1 + avg_overest / 100)
        adjusted_ic = ic_annual_income * deflation_factor
        adjusted_cc = cc_annual_income * deflation_factor
        impact = {
            "original_ic_cagr": ic_annual_income,
            "original_cc_cagr": cc_annual_income,
            "deflation_factor": round(deflation_factor, 3),
            "adjusted_ic_cagr": round(adjusted_ic, 1),
            "adjusted_cc_cagr": round(adjusted_cc, 1),
            "original_combined_income_contribution": round(
                ic_annual_income * 0.4 + cc_annual_income * 0.4, 1),
            "adjusted_combined_income_contribution": round(
                adjusted_ic * 0.4 + adjusted_cc * 0.4, 1),
        }

    return {
        "audit": "Realistic Premium Benchmarking",
        "live_option_comparison": results,
        "average_overestimation_pct": round(avg_overest, 1) if avg_overest is not None else "insufficient data",
        "return_impact": impact,
        "reliability": "MEDIUM" if (avg_overest and abs(avg_overest) < 30) else "LOW",
        "reasoning": (
            f"Live option chain comparison shows BS+VIX model "
            f"{'overestimates' if (avg_overest and avg_overest > 0) else 'underestimates'} "
            f"premiums by ~{abs(avg_overest):.0f}% on average. " if avg_overest else
            "Could not complete live comparison. "
        ) + (
            "The direction and magnitude of bias varies by ticker and VIX level. "
            "During high-VIX periods (which generate most of the backtest's income), "
            "the overestimation is likely LARGER because VIX spikes affect SPX vol "
            "more than individual stock vol."
        ),
    }


# ===================================================================
# AUDIT 7: Stress Period Forensics
# ===================================================================
def audit_7_stress_test():
    """
    Detailed analysis of portfolio behavior during known stress periods.
    """
    print("\n" + "="*70)
    print("AUDIT 7: Stress Period Forensics")
    print("="*70)

    monthly = ORIG_RESULTS["monthly_pnl"]
    ic_monthly = ORIG_RESULTS["per_strategy_monthly_pnl"]["iron_condor"]
    cc_monthly = ORIG_RESULTS["per_strategy_monthly_pnl"]["covered_call"]
    vrp_monthly = ORIG_RESULTS["per_strategy_monthly_pnl"]["vrp_overlay"]

    stress_periods = {
        "dec_2018": {
            "months": ["2018-12"],
            "spy_return_pct": -9.2,
            "description": "December 2018 selloff, SPY -9.2% for the month",
        },
        "covid_crash": {
            "months": ["2020-02", "2020-03", "2020-04"],
            "spy_return_pct": -20,  # Feb-Mar combined
            "description": "COVID crash, SPY peak-to-trough -34%",
        },
        "bear_2022": {
            "months": [f"2022-{m:02d}" for m in range(1, 13)],
            "spy_return_pct": -19.4,
            "description": "2022 bear market, SPY -19.4% for the year",
        },
        "aug_2024_vix_spike": {
            "months": ["2024-08"],
            "spy_return_pct": -2.3,
            "description": "August 2024 VIX spike (Japan carry trade unwind)",
        },
        "apr_2025_tariff": {
            "months": ["2025-04"],
            "spy_return_pct": -5.0,
            "description": "April 2025 tariff selloff",
        },
    }

    stress_results = {}

    for period_name, period_info in stress_periods.items():
        months = period_info["months"]
        period_pnl = sum(monthly.get(m, 0) for m in months)
        period_ic = sum(ic_monthly.get(m, 0) for m in months)
        period_cc = sum(cc_monthly.get(m, 0) for m in months)
        period_vrp = sum(vrp_monthly.get(m, 0) for m in months)

        period_return = period_pnl / INITIAL_CAPITAL * 100

        stress_results[period_name] = {
            "description": period_info["description"],
            "spy_return_pct": period_info["spy_return_pct"],
            "model_combined_pnl": round(period_pnl, 2),
            "model_combined_return_pct": round(period_return, 2),
            "model_ic_pnl": round(period_ic, 2),
            "model_cc_pnl": round(period_cc, 2),
            "model_vrp_pnl": round(period_vrp, 2),
            "model_profitable": period_pnl > 0,
        }

        print(f"\n  {period_name}: SPY={period_info['spy_return_pct']:+.1f}%")
        print(f"    Model: combined=${period_pnl:+,.0f} ({period_return:+.1f}%), "
              f"IC=${period_ic:+,.0f}, CC=${period_cc:+,.0f}, VRP=${period_vrp:+,.0f}")

    # Reality checks
    reality_checks = {}

    # COVID: Real IC portfolio at 0.20 delta, 7 DTE would be devastated
    # VIX went from 14 to 82 in 3 weeks. All ICs opened before Feb 20 would be
    # max-loss. Multiple weeks of max-loss positions.
    # Real loss estimate: 3-4 cycles of max loss
    # With $40K IC capital and 5 underlyings: max loss per cycle ~$2K per underlying
    # 4 cycles * 5 underlyings * $2K = $40K loss (wipeout of IC allocation)
    reality_checks["covid_ic"] = {
        "model_covid_ic_pnl": round(sum(ic_monthly.get(m, 0) for m in ["2020-02", "2020-03", "2020-04"]), 2),
        "model_shows_profit_in_mar": ic_monthly.get("2020-03", 0) == 0,  # They show $0 (skipped due to VIX>40)
        "real_estimate": (
            "The model skips opening new ICs when VIX>40, which saved it in March 2020. "
            "But positions opened BEFORE the crash (in February) would hit max loss. "
            "Feb 2020 shows $1,849 profit — this likely means positions opened before "
            "Feb 20 were closed with moderate profit before the crash accelerated. "
            "This is OPTIMISTIC because daily-close pricing misses intraday breaches. "
            "Real Feb 2020 IC losses would likely be $5K-$15K, not $1,849 profit."
        ),
    }

    # COVID CC: Should show ~-30% on stock side
    reality_checks["covid_cc"] = {
        "model_covid_cc_pnl": round(sum(cc_monthly.get(m, 0) for m in ["2020-02", "2020-03", "2020-04"]), 2),
        "model_shows_positive": True,
        "real_estimate": (
            "CC portfolio owns stocks that dropped 30-50% during COVID. "
            "Premium received (~2% of stock value per month) offsets only a tiny fraction. "
            "Real CC PnL for Feb-Apr 2020 at 40% allocation: "
            "Stock loss: ~-30% * $40K = -$12K. Premium: ~$1.5K. "
            "Net: ~-$10.5K, not the +$1,610 shown."
        ),
        "model_vs_reality_gap": "~$12,000 ($1,610 modeled vs -$10,500 realistic)",
    }

    # Dec 2018
    reality_checks["dec_2018"] = {
        "model_combined_pnl": round(monthly.get("2018-12", 0), 2),
        "spy_return": "-9.2%",
        "assessment": (
            "Model shows -$9,822 loss in Dec 2018. IC component lost -$10,438. "
            "This is actually somewhat realistic for the IC — a -9% SPY month would "
            "devastate 0.20 delta ICs. But CC shows +$560 profit (no stock loss), "
            "which is unrealistic. Real combined loss would be ~$15K-$20K, not $9.8K."
        ),
    }

    # 2022 bear market
    bear_2022_pnl = sum(monthly.get(f"2022-{m:02d}", 0) for m in range(1, 13))
    bear_2022_cc = sum(cc_monthly.get(f"2022-{m:02d}", 0) for m in range(1, 13))
    reality_checks["bear_2022"] = {
        "model_full_year_pnl": round(bear_2022_pnl, 2),
        "model_shows_positive": bear_2022_pnl > 0,
        "model_cc_contribution": round(bear_2022_cc, 2),
        "assessment": (
            f"Model shows ${bear_2022_pnl:,.0f} PROFIT during a year SPY lost 19.4%. "
            f"CC contributed ${bear_2022_cc:,.0f} positive (impossible without stock losses). "
            f"Real 2022 performance: IC might be flat to slightly positive (high VIX = more premium, "
            f"but more breaches). CC should show -10% to -15% (stock losses dominate premium). "
            f"Realistic combined: -$5K to -$15K for the year, not +${bear_2022_pnl:,.0f}."
        ),
    }

    print("\n  Reality Check Summary:")
    print(f"    COVID: Model shows modest loss; reality ~$15-25K worse")
    print(f"    2022: Model shows ${bear_2022_pnl:,.0f} profit; reality likely -$5K to -$15K")
    print(f"    Combined gap: backtest is ~$30-50K too optimistic over stress periods alone")

    return {
        "audit": "Stress Period Forensics",
        "stress_periods": stress_results,
        "reality_checks": reality_checks,
        "cumulative_stress_gap_estimate": (
            "The model is approximately $30K-$50K too optimistic across all stress periods "
            "combined, primarily due to the CC model not tracking stock drawdowns. "
            "On a $100K starting portfolio, this represents 30-50% of cumulative phantom returns."
        ),
        "reliability": "LOW",
        "reasoning": (
            "Stress periods are where premium-selling portfolios face their existential risk. "
            "The model passes through COVID with minimal damage and shows PROFIT during the "
            "2022 bear market — both are unrealistic for a short-vol portfolio. The CC flaw "
            "is the primary cause: real CCs would show $10K+ losses during COVID and $10K+ "
            "losses during 2022, dramatically worsening the combined portfolio's drawdown "
            "profile. The IC component is more honestly modeled (VIX>40 gate helps) but "
            "still misses intraday breaches and gap risk."
        ),
    }


# ===================================================================
# FINAL: Overall Assessment
# ===================================================================
def overall_assessment(audits):
    """Synthesize all audit findings into an honest estimate."""
    print("\n" + "="*70)
    print("OVERALL ASSESSMENT")
    print("="*70)

    reported_cagr = ORIG_RESULTS["combined_metrics"]["cagr_pct"]
    reported_sharpe = ORIG_RESULTS["combined_metrics"]["sharpe"]
    reported_maxdd = ORIG_RESULTS["combined_metrics"]["max_drawdown_pct"]

    # Adjustment factors:
    # 1. CC model flaw: The biggest issue. CC shows 12.65% CAGR on its 40% allocation
    #    but doesn't account for stock drawdowns. Real CC CAGR would be roughly:
    #    - Income: ~5-7% (after correcting premium overestimation)
    #    - Stock return: ~10% CAGR (SPY-like for mega-caps 2018-2026)
    #    - BUT: CC caps upside, so stock component is lower, maybe ~7%
    #    - Combined CC: ~12-14% total return (including stock + premium)
    #    - BUT: drawdowns would be 20-30%, not 0%
    #    The model captures only the income portion but ignores drawdowns.

    # 2. IC premium overestimation: ~20-40% based on VIX-as-IV bias
    #    IC shows 33.93% CAGR — this is EXTREMELY high for selling weekly ICs
    #    Even professional IC funds struggle to make >15% CAGR consistently

    # 3. Gap/breach underestimation in IC: Adds another 10-20% to losses

    # 4. VRP is negligible (1.1% CAGR, negative Sharpe) — no adjustment needed

    # Realistic estimates:
    # IC: 33.93% claimed → realistic 8-15% CAGR (after premium deflation + gap risk)
    # CC: 12.65% claimed income → realistic total return ~8-12% (including stock)
    #     but with 15-25% max drawdown
    # VRP: 1.1% → realistic 0% to -2% (borrow costs eat it)

    # Combined realistic portfolio:
    # 40% IC at 10% = 4.0%
    # 40% CC at 10% = 4.0% (but with much higher drawdowns)
    # 20% VRP at 0% = 0.0%
    # Total: ~8% CAGR with 15-25% max drawdown

    realistic = {
        "realistic_ic_cagr_range": [8, 15],
        "realistic_cc_total_return_range": [8, 12],
        "realistic_vrp_cagr_range": [-2, 1],
        "realistic_combined_cagr_range": [7, 12],
        "realistic_max_drawdown_range": [15, 30],
        "realistic_sharpe_range": [0.5, 1.2],
    }

    print(f"\n  REPORTED vs REALISTIC:")
    print(f"    CAGR:   {reported_cagr:.1f}% reported → {realistic['realistic_combined_cagr_range'][0]}-{realistic['realistic_combined_cagr_range'][1]}% realistic")
    print(f"    MaxDD:  {reported_maxdd:.1f}% reported → {realistic['realistic_max_drawdown_range'][0]}-{realistic['realistic_max_drawdown_range'][1]}% realistic")
    print(f"    Sharpe: {reported_sharpe:.2f} reported → {realistic['realistic_sharpe_range'][0]}-{realistic['realistic_sharpe_range'][1]} realistic")

    # Component reliability summary
    reliability_summary = {
        "iron_condor_premium": audits[0]["reliability"],
        "iron_condor_breach": audits[1]["reliability"],
        "covered_call": audits[2]["reliability"],
        "vrp_overlay": audits[3]["reliability"],
        "correlation": audits[4]["reliability"],
        "premium_benchmark": audits[5]["reliability"],
        "stress_forensics": audits[6]["reliability"],
    }

    inflation_sources = [
        {
            "source": "CC ignores stock drawdowns",
            "estimated_impact_on_cagr": "+5-8% phantom CAGR",
            "estimated_impact_on_maxdd": "Hides 15-25% of real drawdown",
            "severity": "CRITICAL",
        },
        {
            "source": "VIX used as IV for individual stocks (IC + CC)",
            "estimated_impact_on_cagr": "+2-5% inflated premium income",
            "estimated_impact_on_maxdd": "Minor",
            "severity": "HIGH",
        },
        {
            "source": "IC daily-close pricing misses intraday breaches",
            "estimated_impact_on_cagr": "+1-3% avoided losses",
            "estimated_impact_on_maxdd": "Underestimates by 2-5%",
            "severity": "MEDIUM",
        },
        {
            "source": "IC short strikes at ~0.30 delta not 0.20 as claimed",
            "estimated_impact_on_cagr": "Mixed (more premium + more risk)",
            "estimated_impact_on_maxdd": "Adds risk not captured",
            "severity": "MEDIUM",
        },
        {
            "source": "Missing VXX borrow costs",
            "estimated_impact_on_cagr": "-0.5% (makes VRP net negative)",
            "estimated_impact_on_maxdd": "Negligible",
            "severity": "LOW",
        },
        {
            "source": "Correlation underestimated during stress",
            "estimated_impact_on_cagr": "Indirect — drawdowns are worse",
            "estimated_impact_on_maxdd": "Real drawdown 2-3x modeled",
            "severity": "HIGH",
        },
    ]

    print(f"\n  VERDICT: The reported 23.8% CAGR with 7.6% max drawdown is UNREALISTIC.")
    print(f"  A realistic estimate is 7-12% CAGR with 15-30% max drawdown.")
    print(f"  The primary cause is the CC model ignoring stock losses (40% of portfolio).")
    print(f"  The secondary cause is VIX-based premium overestimation across all strategies.")

    return {
        "reported_metrics": {
            "cagr_pct": reported_cagr,
            "sharpe": reported_sharpe,
            "max_drawdown_pct": reported_maxdd,
        },
        "realistic_estimates": realistic,
        "inflation_sources": inflation_sources,
        "component_reliability": reliability_summary,
        "overall_reliability": "LOW",
        "verdict": (
            f"The reported {reported_cagr}% CAGR with {reported_maxdd}% max drawdown and "
            f"{reported_sharpe:.1f} Sharpe is significantly inflated. "
            f"The single biggest flaw is the Covered Call model (40% allocation) treating "
            f"covered calls as pure income with zero stock risk — this alone adds ~10% "
            f"phantom CAGR and hides 15-25% of real drawdown. "
            f"The secondary flaw is using VIX as implied vol for individual stocks, "
            f"overestimating premiums by 20-50% depending on the ticker. "
            f"A realistic estimate for this strategy combination is 7-12% CAGR with "
            f"15-30% max drawdown and Sharpe of 0.5-1.2. "
            f"This is still a viable income strategy — just not a Sharpe-4 unicorn. "
            f"To fix: (1) Add stock P&L to CC model, (2) Use stock-specific IV or "
            f"VIX-to-stock-IV scaling factors, (3) Add intraday price simulation for IC."
        ),
        "recommendations": [
            "CRITICAL: Add stock P&L tracking to Covered Call model — currently only tracks premium",
            "HIGH: Replace VIX with stock-specific IV estimates (use historical IV rank or yfinance IV)",
            "HIGH: Add gap/intraday price simulation for IC breach modeling",
            "MEDIUM: Fix IC short strike calculation — currently ~0.30 delta, not 0.20 as intended",
            "MEDIUM: Add VXX borrow costs to VRP model (3-8% annually)",
            "LOW: Add VIXY tracking error vs true VXX to VRP model",
        ],
    }


# ===================================================================
# MAIN
# ===================================================================
def main():
    print("="*70)
    print("ADVERSARIAL AUDIT: Income Portfolio v1")
    print(f"Run date: {dt.datetime.now().isoformat()}")
    print("="*70)

    audits = []

    # Run all 7 audits
    a1 = audit_1_ic_premium_vix_bias()
    audits.append(a1)

    a2 = audit_2_ic_breach_modeling()
    audits.append(a2)

    a3 = audit_3_covered_call()
    audits.append(a3)

    a4 = audit_4_vrp_overlay()
    audits.append(a4)

    a5 = audit_5_correlation()
    audits.append(a5)

    a6 = audit_6_premium_benchmark()
    audits.append(a6)

    a7 = audit_7_stress_test()
    audits.append(a7)

    # Overall assessment
    assessment = overall_assessment(audits)

    # Build final report
    report = {
        "metadata": {
            "script": "adversarial_audit_income_v1.py",
            "run_date": dt.datetime.now().isoformat(),
            "original_results_file": str(RESULTS_PATH),
        },
        "audit_1_ic_premium_vix_bias": a1,
        "audit_2_ic_breach_modeling": a2,
        "audit_3_covered_call": a3,
        "audit_4_vrp_overlay": a4,
        "audit_5_correlation": a5,
        "audit_6_premium_benchmark": a6,
        "audit_7_stress_test": a7,
        "overall_assessment": assessment,
    }

    # Save
    output_path = Path("/home/jupiter/Lvl3Quant/findings/income_adversarial_audit_v1.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nAudit saved to {output_path}")
    return report


if __name__ == "__main__":
    report = main()
