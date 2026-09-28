#!/usr/bin/env python3
"""
Income Research: Jade Lizard + Strangle Selling Backtests
==========================================================
Walk-forward backtests (60-month train, 1-month OOT, sliding window)
for two new income strategies, compared against V5 CSP baseline.

Strategy 1: JADE LIZARD
  - Sell OTM put (like CSP)
  - Sell OTM call spread (sell call + buy higher call for cap)
  - Extra premium from call side; undefined risk on downside, defined on upside
  - Best for: bullish-neutral stocks

Strategy 2: STRANGLE SELLING (VIX-sized)
  - Sell OTM put + sell OTM call (no wings)
  - Higher premium than CSP or IC, but unlimited risk both sides
  - VIX-based sizing: reduce size when VIX > 25
  - Stop-loss at 2x premium received

BS pricing (conservative — known to underprice by ~41%).
Universe: 70 tickers (same as V5 CSP).
Commission: $0 (Robinhood) + realistic BA spread 5-15% of premium.
"""

import sys, json, time, math, warnings, os
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats
from datetime import timedelta

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUTPUT = ROOT / "output" / "income_research"
OUTPUT.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100_000

# ═══════════════════════════════════════════════════════════════════
# Ticker Universe (same 70 as V5 CSP)
# ═══════════════════════════════════════════════════════════════════

TICKERS = [
    'AAPL','ABBV','ABNB','ADBE','AMD','AMZN','ARM','AXP','BA','BAC',
    'BLK','BRK-B','C','CAT','CL','COIN','COST','CRM','CRWD','CVX',
    'DDOG','DE','DIS','F','GE','GM','GOOGL','GS','HD','HOOD',
    'INTC','JNJ','JPM','KO','LLY','LOW','MA','MCD','META','MRNA',
    'MS','MSFT','NFLX','NOW','NVDA','ORCL','OXY','PANW','PEP','PFE',
    'PG','PLTR','PYPL','RTX','SBUX','SCHW','SHOP','SLB','SMCI','T',
    'TGT','TMUS','TSLA','UBER','UNH','V','VZ','WFC','WMT','XOM',
]

SECTORS = {
    'AAPL':'Tech','ABBV':'Health','ABNB':'ConsDisc','ADBE':'Tech','AMD':'Tech',
    'AMZN':'ConsDisc','ARM':'Tech','AXP':'Fin','BA':'Industrial','BAC':'Fin',
    'BLK':'Fin','BRK-B':'Fin','C':'Fin','CAT':'Industrial','CL':'ConsSt',
    'COIN':'Fin','COST':'ConsSt','CRM':'Tech','CRWD':'Tech','CVX':'Energy',
    'DDOG':'Tech','DE':'Industrial','DIS':'Comm','F':'ConsDisc','GE':'Industrial',
    'GM':'ConsDisc','GOOGL':'Comm','GS':'Fin','HD':'ConsDisc','HOOD':'Fin',
    'INTC':'Tech','JNJ':'Health','JPM':'Fin','KO':'ConsSt','LLY':'Health',
    'LOW':'ConsDisc','MA':'Fin','MCD':'ConsDisc','META':'Comm','MRNA':'Health',
    'MS':'Fin','MSFT':'Tech','NFLX':'Comm','NOW':'Tech','NVDA':'Tech',
    'ORCL':'Tech','OXY':'Energy','PANW':'Tech','PEP':'ConsSt','PFE':'Health',
    'PG':'ConsSt','PLTR':'Tech','PYPL':'Fin','RTX':'Industrial','SBUX':'ConsDisc',
    'SCHW':'Fin','SHOP':'Tech','SLB':'Energy','SMCI':'Tech','T':'Comm',
    'TGT':'ConsDisc','TMUS':'Comm','TSLA':'ConsDisc','UBER':'Tech',
    'UNH':'Health','V':'Fin','VZ':'Comm','WFC':'Fin','WMT':'ConsSt','XOM':'Energy',
}


# ═══════════════════════════════════════════════════════════════════
# Black-Scholes Primitives
# ═══════════════════════════════════════════════════════════════════

def _Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def _ndtri(p):
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

def bs_price(S, K, T, sigma, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if kind == "put":
        return K*math.exp(-r*T)*_Phi(-d2) - S*math.exp(-q*T)*_Phi(-d1)
    return S*math.exp(-q*T)*_Phi(d1) - K*math.exp(-r*T)*_Phi(d2)

def strike_from_delta(S, T, sigma, target_delta, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0:
        return S
    target = abs(target_delta)
    p = target if kind == "call" else (1 - target)
    p = min(max(p, 1e-6), 1 - 1e-6)
    d1 = _ndtri(p)
    K = S * math.exp(-(d1 * sigma * math.sqrt(T) - (r - q + 0.5*sigma**2)*T))
    return K


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def generate_iv_features(prices_df, tickers):
    frames = []
    for tk in tickers:
        tk_px = prices_df[prices_df["ticker"] == tk].sort_values("date").copy()
        if len(tk_px) < 60:
            continue
        tk_px["log_ret"] = np.log1p(tk_px["close"].pct_change())
        tk_px["rv_20"] = tk_px["log_ret"].rolling(20).std() * np.sqrt(252)
        tk_px["rv_60"] = tk_px["log_ret"].rolling(60).std() * np.sqrt(252)
        tk_px["sigma"] = tk_px["rv_20"] * 1.15
        tk_px["iv_high_252"] = tk_px["sigma"].rolling(252).max()
        tk_px["iv_low_252"] = tk_px["sigma"].rolling(252).min()
        iv_range = tk_px["iv_high_252"] - tk_px["iv_low_252"]
        tk_px["iv_rank"] = np.where(iv_range > 0.001,
            (tk_px["sigma"] - tk_px["iv_low_252"]) / iv_range, 0.5)
        tk_px["sigma_rv"] = tk_px["rv_20"]
        tk_px["iv_rv_ratio"] = np.where(tk_px["rv_20"] > 0.001, tk_px["sigma"] / tk_px["rv_20"], 1.15)
        tk_px["term_ratio"] = np.where(tk_px["rv_20"] > 0.001,
            tk_px["rv_60"].fillna(tk_px["rv_20"]) / tk_px["rv_20"], 1.0)
        tk_px["r_1m"] = tk_px["close"].pct_change(21)
        tk_px["pricing_source"] = "modeled_rv"
        tk_px["ticker"] = tk
        cols = ["date", "ticker", "sigma_rv", "iv_rv_ratio", "term_ratio",
                "sigma", "iv_rank", "r_1m", "pricing_source"]
        frame = tk_px.dropna(subset=["sigma"])[cols]
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def load_all_data():
    print("Loading data...")
    prices = pd.read_parquet(CACHE / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    for extra in ["prices_expanded.parquet", "prices_v3_expansion.parquet"]:
        try:
            pexp = pd.read_parquet(CACHE / extra)
            if "Open" in pexp.columns:
                pexp = pexp.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                             "Close": "close", "Volume": "volume"})
            pexp["date"] = pd.to_datetime(pexp["date"]).dt.tz_localize(None)
            new_tks = set(pexp["ticker"].unique()) - set(prices["ticker"].unique())
            if new_tks:
                pexp = pexp[pexp["ticker"].isin(new_tks)]
                prices = pd.concat([prices, pexp], ignore_index=True)
        except:
            pass

    prices = prices[prices["date"] >= "2019-01-01"].copy()
    prices = prices.drop_duplicates(subset=["ticker", "date"], keep="first")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    if "ret" not in prices.columns:
        prices["ret"] = prices.groupby("ticker")["close"].pct_change()
    if "log_ret" not in prices.columns:
        prices["log_ret"] = np.log1p(prices["ret"])
    if "rv_20" not in prices.columns:
        prices["rv_20"] = prices.groupby("ticker")["log_ret"].transform(
            lambda x: x.rolling(20).std() * np.sqrt(252))

    iv = pd.read_parquet(CACHE / "iv_features_modeled.parquet")
    iv["date"] = pd.to_datetime(iv["date"]).dt.tz_localize(None)
    iv = iv[iv["date"] >= "2019-01-01"]

    all_needed = set(TICKERS)
    iv_tickers = set(iv["ticker"].unique())
    need_iv = (all_needed & set(prices["ticker"].unique())) - iv_tickers
    if need_iv:
        print(f"  Generating modeled IV for {len(need_iv)} tickers...")
        iv_new = generate_iv_features(prices, sorted(need_iv))
        if not iv_new.empty:
            iv = pd.concat([iv, iv_new], ignore_index=True)

    macro = pd.read_parquet(CACHE / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
    macro = macro[macro["date"] >= "2019-01-01"]

    # VIX history for regime classification
    vix_df = pd.read_parquet(CACHE / "vix_history.parquet")
    vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.tz_localize(None)

    # SPY for regime classification
    spy = pd.read_parquet(CACHE / "spy_prices.parquet")
    spy["date"] = pd.to_datetime(spy["date"]).dt.tz_localize(None)
    spy = spy.sort_values("date")
    spy["spy_ret"] = spy["close"].pct_change()

    try:
        earnings = pd.read_parquet(CACHE / "earnings_dates.parquet")
        earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"]).dt.tz_localize(None)
    except:
        earnings = pd.DataFrame(columns=["ticker", "earnings_date"])

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique()) & all_needed
    avail_list = sorted([t for t in TICKERS if t in available])

    print(f"  Prices: {prices.shape[0]} rows, {prices['ticker'].nunique()} tickers")
    print(f"  IV: {iv.shape[0]} rows, {iv['ticker'].nunique()} tickers")
    print(f"  Available tickers: {len(avail_list)}")

    return prices, iv, macro, vix_df, spy, earnings, avail_list


def build_earnings_lookup(earnings_df):
    lookup = {}
    for ticker in earnings_df["ticker"].unique():
        dates = earnings_df[earnings_df["ticker"] == ticker]["earnings_date"].sort_values().values
        if len(dates) > 0:
            lookup[ticker] = dates
    return lookup


def has_earnings_within(ticker, open_date, dte_target, earnings_lookup, buffer_days=7):
    if ticker not in earnings_lookup:
        return False
    earn_dates = earnings_lookup[ticker]
    hold_start = np.datetime64(open_date) - np.timedelta64(buffer_days, 'D')
    hold_end = np.datetime64(open_date) + np.timedelta64(dte_target + buffer_days, 'D')
    mask = (earn_dates >= hold_start) & (earn_dates <= hold_end)
    return mask.any()


# ═══════════════════════════════════════════════════════════════════
# Cost Model (Robinhood: $0 commission, but BA spread matters)
# ═══════════════════════════════════════════════════════════════════

def ba_cost_per_leg(premium, ba_frac):
    """Cost of crossing the bid-ask spread for one option leg.
    Seller gets bid = mid * (1 - ba_frac/2), buyer pays ask = mid * (1 + ba_frac/2).
    Cost = ba_frac/2 * premium * 100 per contract.
    """
    return premium * (ba_frac / 2.0) * 100


# ═══════════════════════════════════════════════════════════════════
# VIX-based sizing
# ═══════════════════════════════════════════════════════════════════

def vix_size_multiplier(vix_val, strategy="strangle"):
    """Scale position size based on VIX. More aggressive reduction for strangles."""
    if np.isnan(vix_val):
        return 0.75  # conservative default
    if strategy == "strangle":
        if vix_val > 35:
            return 0.0   # no new strangles in extreme vol
        elif vix_val > 30:
            return 0.25
        elif vix_val > 25:
            return 0.50
        elif vix_val > 20:
            return 0.75
        else:
            return 1.0
    else:  # jade lizard
        if vix_val > 35:
            return 0.25
        elif vix_val > 30:
            return 0.50
        elif vix_val > 25:
            return 0.75
        else:
            return 1.0


# ═══════════════════════════════════════════════════════════════════
# Strategy: JADE LIZARD
# sell OTM put + sell OTM call + buy further OTM call
# Net credit structure. Undefined risk on downside, defined on upside.
# ═══════════════════════════════════════════════════════════════════

def run_jade_lizard(prices, iv, macro, vix_df, earnings_lookup, ticker_list,
                    ba_frac=0.10, dte_target=35, put_delta=-0.25, call_delta=0.25,
                    long_call_offset=0.05, profit_take=0.50, stop_loss_mult=2.0,
                    max_concurrent=25, per_name_pct=0.04, vix_hard_cutoff=35,
                    earnings_buffer=7):
    """
    Jade Lizard backtest.

    Legs:
    1. Sell 1 OTM put at put_delta
    2. Sell 1 OTM call at call_delta
    3. Buy 1 OTM call at call_delta + long_call_offset (cap upside risk)

    Net credit = put_premium + short_call_premium - long_call_premium
    Max loss on upside = (long_call_strike - short_call_strike)*100 - net_credit
    Max loss on downside = unlimited (like CSP)

    Ideally: net credit > (long_call_strike - short_call_strike)*100
    so there's NO risk on the upside at all (true Jade Lizard).
    """
    print(f"\n{'='*70}")
    print(f"JADE LIZARD BACKTEST (DTE={dte_target}, BA={ba_frac*100:.0f}%)")
    print(f"{'='*70}")

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique()) & set(ticker_list)
    prices_df = prices[prices["ticker"].isin(available)].copy()
    iv_df = iv[iv["ticker"].isin(available)].copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()

    macro_copy = macro.copy()
    macro_copy["date"] = pd.to_datetime(macro_copy["date"])
    macro_by_date = macro_copy.set_index("date").to_dict("index")

    all_dates = sorted(prices_df["date"].unique())

    cash = float(STARTING_CAPITAL)
    positions = {}
    equity_curve = []
    detailed_trades = []
    stats = {"n_opened": 0, "n_closed": 0, "n_profit_take": 0, "n_stop_loss": 0,
             "n_expiry_win": 0, "n_expiry_loss": 0, "n_assigned_put": 0,
             "n_earnings_blocked": 0, "n_vix_blocked": 0, "total_ba_cost": 0.0}

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        m_data = macro_by_date.get(dt, {})
        vix = m_data.get("vix", float("nan")) if isinstance(m_data, dict) else float("nan")
        try:
            vix_val = float(vix)
        except:
            vix_val = float("nan")

        # ── Mark-to-market and manage existing positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_cur = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            if T_days <= 0:
                # ── Expiry settlement ──
                put_itm = S < pos["put_strike"]
                call_itm = S > pos["short_call_strike"]

                pnl = pos["net_credit_dollar"]  # start with credit received

                if put_itm:
                    # Put assigned: loss = (strike - S) * 100 * contracts
                    assign_loss = (pos["put_strike"] - S) * 100 * pos["contracts"]
                    pnl -= assign_loss
                    stats["n_assigned_put"] += 1

                if call_itm:
                    # Short call assigned, long call exercised
                    # Loss capped at (long_call_strike - short_call_strike) * 100 * contracts
                    if S >= pos["long_call_strike"]:
                        call_loss = (pos["long_call_strike"] - pos["short_call_strike"]) * 100 * pos["contracts"]
                    else:
                        call_loss = (S - pos["short_call_strike"]) * 100 * pos["contracts"]
                    pnl -= call_loss

                if pnl >= 0:
                    stats["n_expiry_win"] += 1
                else:
                    stats["n_expiry_loss"] += 1

                cash += pnl
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": pnl, "close_reason": "expiry",
                    "put_itm": put_itm, "call_itm": call_itm
                })
                to_remove.append(tk)
                stats["n_closed"] += 1
                continue

            # ── Mid-life: check profit-take / stop-loss ──
            put_prem_now = bs_price(S, pos["put_strike"], T, sigma_cur, kind="put")
            short_call_prem_now = bs_price(S, pos["short_call_strike"], T, sigma_cur, kind="call")
            long_call_prem_now = bs_price(S, pos["long_call_strike"], T, sigma_cur, kind="call")
            current_cost_to_close = (put_prem_now + short_call_prem_now - long_call_prem_now) * 100 * pos["contracts"]
            # BA cost to close (3 legs)
            ba_close = (ba_cost_per_leg(put_prem_now, ba_frac) +
                        ba_cost_per_leg(short_call_prem_now, ba_frac) +
                        ba_cost_per_leg(long_call_prem_now, ba_frac)) * pos["contracts"]

            unrealized_pnl = pos["net_credit_dollar"] - current_cost_to_close - ba_close

            # Profit take: close when X% of max profit captured
            if unrealized_pnl >= profit_take * pos["net_credit_dollar"]:
                cash += unrealized_pnl
                stats["total_ba_cost"] += ba_close
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": unrealized_pnl, "close_reason": "profit_take"
                })
                to_remove.append(tk)
                stats["n_profit_take"] += 1
                stats["n_closed"] += 1
                continue

            # Stop loss: close when loss exceeds stop_loss_mult * credit received
            if unrealized_pnl < -(stop_loss_mult * pos["net_credit_dollar"]):
                cash += unrealized_pnl
                stats["total_ba_cost"] += ba_close
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": unrealized_pnl, "close_reason": "stop_loss"
                })
                to_remove.append(tk)
                stats["n_stop_loss"] += 1
                stats["n_closed"] += 1
                continue

        for tk in to_remove:
            if tk in positions:
                del positions[tk]

        # ── Open new positions ──
        if len(positions) < max_concurrent and not np.isnan(vix_val):
            if vix_val >= vix_hard_cutoff:
                stats["n_vix_blocked"] += 1
            else:
                size_mult = vix_size_multiplier(vix_val, "jade_lizard")
                if size_mult > 0:
                    candidates = []
                    for tk in ticker_list:
                        if tk in positions:
                            continue
                        S = date_px.get(tk)
                        sigma = date_sigma.get(tk)
                        if S is None or sigma is None or np.isnan(S) or np.isnan(sigma):
                            continue
                        if sigma < 0.10:
                            continue
                        if has_earnings_within(tk, dt, dte_target, earnings_lookup, earnings_buffer):
                            stats["n_earnings_blocked"] += 1
                            continue
                        candidates.append((tk, S, sigma))

                    # Sort by IV rank (highest IV = most premium)
                    candidates.sort(key=lambda x: x[2], reverse=True)

                    for tk, S, sigma in candidates:
                        if len(positions) >= max_concurrent:
                            break

                        T = dte_target / 365.0

                        # Calculate strikes
                        K_put = strike_from_delta(S, T, sigma, put_delta, kind="put")
                        K_short_call = strike_from_delta(S, T, sigma, call_delta, kind="call")
                        # Long call is further OTM by offset fraction of stock price
                        K_long_call = K_short_call * (1 + long_call_offset)

                        # Calculate premiums
                        put_prem = bs_price(S, K_put, T, sigma, kind="put")
                        short_call_prem = bs_price(S, K_short_call, T, sigma, kind="call")
                        long_call_prem = bs_price(S, K_long_call, T, sigma, kind="call")

                        net_credit_per_share = put_prem + short_call_prem - long_call_prem
                        if net_credit_per_share <= 0.10:
                            continue

                        # BA cost (3 legs: sell put, sell call, buy call)
                        ba_open = (ba_cost_per_leg(put_prem, ba_frac) +
                                   ba_cost_per_leg(short_call_prem, ba_frac) +
                                   ba_cost_per_leg(long_call_prem, ba_frac))

                        # Sizing: capital = max(put assignment risk, call spread width * 100)
                        put_margin = K_put * 100  # cash needed to cover put assignment
                        call_spread_width = (K_long_call - K_short_call) * 100
                        margin_needed = max(put_margin, call_spread_width)

                        sizing_base = min(cash, STARTING_CAPITAL * 1.5)
                        max_alloc = sizing_base * per_name_pct * size_mult
                        contracts = max(1, int(max_alloc / margin_needed))
                        if contracts * margin_needed > sizing_base * 0.15:
                            contracts = max(1, int(sizing_base * 0.15 / margin_needed))

                        net_credit_dollar = net_credit_per_share * 100 * contracts
                        total_ba_open = ba_open * contracts
                        net_credit_dollar -= total_ba_open
                        stats["total_ba_cost"] += total_ba_open

                        if net_credit_dollar <= 0:
                            continue

                        # Check jade lizard condition: credit > call spread width
                        # If true, NO risk on upside
                        call_spread_risk = call_spread_width * contracts
                        jade_condition = net_credit_dollar > call_spread_risk

                        expiry = dt + pd.Timedelta(days=dte_target)
                        positions[tk] = {
                            "open_date": dt,
                            "expiry": expiry,
                            "contracts": contracts,
                            "put_strike": K_put,
                            "short_call_strike": K_short_call,
                            "long_call_strike": K_long_call,
                            "net_credit_dollar": net_credit_dollar,
                            "open_sigma": sigma,
                            "jade_condition": jade_condition,
                            "open_price": S,
                        }
                        cash += net_credit_dollar  # credit received
                        stats["n_opened"] += 1

        # ── Equity curve ──
        mtm = 0
        for tk, pos in positions.items():
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_cur = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            put_prem = bs_price(S, pos["put_strike"], T, sigma_cur, kind="put")
            sc_prem = bs_price(S, pos["short_call_strike"], T, sigma_cur, kind="call")
            lc_prem = bs_price(S, pos["long_call_strike"], T, sigma_cur, kind="call")
            cost_to_close = (put_prem + sc_prem - lc_prem) * 100 * pos["contracts"]
            mtm -= cost_to_close  # negative because we'd have to buy back

        total_equity = cash + mtm
        equity_curve.append({"date": dt, "equity": total_equity, "cash": cash, "n_pos": len(positions)})

    return equity_curve, detailed_trades, stats


# ═══════════════════════════════════════════════════════════════════
# Strategy: STRANGLE SELLING (VIX-sized)
# sell OTM put + sell OTM call (naked, no wings)
# ═══════════════════════════════════════════════════════════════════

def run_strangle(prices, iv, macro, vix_df, earnings_lookup, ticker_list,
                 ba_frac=0.10, dte_target=35, put_delta=-0.20, call_delta=0.20,
                 profit_take=0.50, stop_loss_mult=2.0,
                 max_concurrent=20, per_name_pct=0.03,
                 vix_hard_cutoff=30, earnings_buffer=7):
    """
    Short Strangle backtest with VIX-based sizing.

    Legs:
    1. Sell 1 OTM put at put_delta
    2. Sell 1 OTM call at call_delta

    Net credit = put_premium + call_premium
    Max loss = unlimited both sides
    Risk managed via: VIX sizing, stop-loss at 2x credit, delta selection
    """
    print(f"\n{'='*70}")
    print(f"STRANGLE SELLING BACKTEST (DTE={dte_target}, BA={ba_frac*100:.0f}%)")
    print(f"{'='*70}")

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique()) & set(ticker_list)
    prices_df = prices[prices["ticker"].isin(available)].copy()
    iv_df = iv[iv["ticker"].isin(available)].copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()

    macro_copy = macro.copy()
    macro_copy["date"] = pd.to_datetime(macro_copy["date"])
    macro_by_date = macro_copy.set_index("date").to_dict("index")

    all_dates = sorted(prices_df["date"].unique())

    cash = float(STARTING_CAPITAL)
    positions = {}
    equity_curve = []
    detailed_trades = []
    stats = {"n_opened": 0, "n_closed": 0, "n_profit_take": 0, "n_stop_loss": 0,
             "n_expiry_win": 0, "n_expiry_loss": 0, "n_assigned_put": 0,
             "n_assigned_call": 0, "n_earnings_blocked": 0, "n_vix_blocked": 0,
             "total_ba_cost": 0.0}

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        m_data = macro_by_date.get(dt, {})
        vix = m_data.get("vix", float("nan")) if isinstance(m_data, dict) else float("nan")
        try:
            vix_val = float(vix)
        except:
            vix_val = float("nan")

        # ── Manage existing positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_cur = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            if T_days <= 0:
                # ── Expiry ──
                pnl = pos["net_credit_dollar"]
                put_itm = S < pos["put_strike"]
                call_itm = S > pos["call_strike"]

                if put_itm:
                    assign_loss = (pos["put_strike"] - S) * 100 * pos["contracts"]
                    pnl -= assign_loss
                    stats["n_assigned_put"] += 1

                if call_itm:
                    # Naked call assignment: loss = (S - call_strike) * 100 * contracts
                    assign_loss = (S - pos["call_strike"]) * 100 * pos["contracts"]
                    pnl -= assign_loss
                    stats["n_assigned_call"] += 1

                if pnl >= 0:
                    stats["n_expiry_win"] += 1
                else:
                    stats["n_expiry_loss"] += 1

                cash += pnl
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": pnl, "close_reason": "expiry",
                    "put_itm": put_itm, "call_itm": call_itm
                })
                to_remove.append(tk)
                stats["n_closed"] += 1
                continue

            # ── Mark-to-market for stop/profit ──
            put_prem_now = bs_price(S, pos["put_strike"], T, sigma_cur, kind="put")
            call_prem_now = bs_price(S, pos["call_strike"], T, sigma_cur, kind="call")
            current_cost_to_close = (put_prem_now + call_prem_now) * 100 * pos["contracts"]
            ba_close = (ba_cost_per_leg(put_prem_now, ba_frac) +
                        ba_cost_per_leg(call_prem_now, ba_frac)) * pos["contracts"]

            unrealized_pnl = pos["net_credit_dollar"] - current_cost_to_close - ba_close

            if unrealized_pnl >= profit_take * pos["net_credit_dollar"]:
                cash += unrealized_pnl
                stats["total_ba_cost"] += ba_close
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": unrealized_pnl, "close_reason": "profit_take"
                })
                to_remove.append(tk)
                stats["n_profit_take"] += 1
                stats["n_closed"] += 1
                continue

            if unrealized_pnl < -(stop_loss_mult * pos["net_credit_dollar"]):
                cash += unrealized_pnl
                stats["total_ba_cost"] += ba_close
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": unrealized_pnl, "close_reason": "stop_loss"
                })
                to_remove.append(tk)
                stats["n_stop_loss"] += 1
                stats["n_closed"] += 1
                continue

        for tk in to_remove:
            if tk in positions:
                del positions[tk]

        # ── Open new positions ──
        if len(positions) < max_concurrent and not np.isnan(vix_val):
            if vix_val >= vix_hard_cutoff:
                stats["n_vix_blocked"] += 1
            else:
                size_mult = vix_size_multiplier(vix_val, "strangle")
                if size_mult > 0:
                    candidates = []
                    for tk in ticker_list:
                        if tk in positions:
                            continue
                        S = date_px.get(tk)
                        sigma = date_sigma.get(tk)
                        if S is None or sigma is None or np.isnan(S) or np.isnan(sigma):
                            continue
                        if sigma < 0.10:
                            continue
                        if has_earnings_within(tk, dt, dte_target, earnings_lookup, earnings_buffer):
                            stats["n_earnings_blocked"] += 1
                            continue
                        candidates.append((tk, S, sigma))

                    candidates.sort(key=lambda x: x[2], reverse=True)

                    for tk, S, sigma in candidates:
                        if len(positions) >= max_concurrent:
                            break

                        T = dte_target / 365.0
                        K_put = strike_from_delta(S, T, sigma, put_delta, kind="put")
                        K_call = strike_from_delta(S, T, sigma, call_delta, kind="call")

                        put_prem = bs_price(S, K_put, T, sigma, kind="put")
                        call_prem = bs_price(S, K_call, T, sigma, kind="call")
                        net_credit_per_share = put_prem + call_prem

                        if net_credit_per_share <= 0.10:
                            continue

                        # BA cost (2 legs)
                        ba_open = (ba_cost_per_leg(put_prem, ba_frac) +
                                   ba_cost_per_leg(call_prem, ba_frac))

                        # Margin: naked strangle margin is substantial
                        # CBOE: max(put_side, call_side) + other_premium
                        # Put side: 20% of stock + put_prem - OTM amount
                        # Call side: 20% of stock + call_prem - OTM amount
                        put_otm = max(0, S - K_put)
                        call_otm = max(0, K_call - S)
                        put_margin_req = (0.20 * S + put_prem - put_otm) * 100
                        call_margin_req = (0.20 * S + call_prem - call_otm) * 100
                        margin_needed = max(put_margin_req, call_margin_req) + min(put_margin_req, call_margin_req) * 0.5

                        # Cap sizing base to prevent runaway compounding
                        sizing_base = min(cash, STARTING_CAPITAL * 1.5)
                        max_alloc = sizing_base * per_name_pct * size_mult
                        contracts = max(1, int(max_alloc / margin_needed))
                        if contracts * margin_needed > sizing_base * 0.12:
                            contracts = max(1, int(sizing_base * 0.12 / margin_needed))

                        net_credit_dollar = net_credit_per_share * 100 * contracts
                        total_ba_open = ba_open * contracts
                        net_credit_dollar -= total_ba_open
                        stats["total_ba_cost"] += total_ba_open

                        if net_credit_dollar <= 0:
                            continue

                        expiry = dt + pd.Timedelta(days=dte_target)
                        positions[tk] = {
                            "open_date": dt,
                            "expiry": expiry,
                            "contracts": contracts,
                            "put_strike": K_put,
                            "call_strike": K_call,
                            "net_credit_dollar": net_credit_dollar,
                            "open_sigma": sigma,
                            "open_price": S,
                        }
                        cash += net_credit_dollar
                        stats["n_opened"] += 1

        # ── Equity curve ──
        mtm = 0
        for tk, pos in positions.items():
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_cur = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            put_prem = bs_price(S, pos["put_strike"], T, sigma_cur, kind="put")
            call_prem = bs_price(S, pos["call_strike"], T, sigma_cur, kind="call")
            cost_to_close = (put_prem + call_prem) * 100 * pos["contracts"]
            mtm -= cost_to_close

        total_equity = cash + mtm
        equity_curve.append({"date": dt, "equity": total_equity, "cash": cash, "n_pos": len(positions)})

    return equity_curve, detailed_trades, stats


# ═══════════════════════════════════════════════════════════════════
# CSP BASELINE (simplified for comparison)
# ═══════════════════════════════════════════════════════════════════

def run_csp_baseline(prices, iv, macro, vix_df, earnings_lookup, ticker_list,
                     ba_frac=0.10, dte_target=35, put_delta=-0.25,
                     profit_take=0.65, stop_loss_mult=2.0,
                     max_concurrent=30, per_name_pct=0.03):
    """CSP baseline for fair comparison."""
    print(f"\n{'='*70}")
    print(f"CSP BASELINE (DTE={dte_target}, BA={ba_frac*100:.0f}%)")
    print(f"{'='*70}")

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique()) & set(ticker_list)
    prices_df = prices[prices["ticker"].isin(available)].copy()
    iv_df = iv[iv["ticker"].isin(available)].copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()

    macro_copy = macro.copy()
    macro_copy["date"] = pd.to_datetime(macro_copy["date"])
    macro_by_date = macro_copy.set_index("date").to_dict("index")

    all_dates = sorted(prices_df["date"].unique())

    cash = float(STARTING_CAPITAL)
    positions = {}
    equity_curve = []
    detailed_trades = []
    stats = {"n_opened": 0, "n_closed": 0, "n_profit_take": 0, "n_stop_loss": 0,
             "n_expiry_win": 0, "n_expiry_loss": 0, "n_assigned": 0,
             "total_ba_cost": 0.0}

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        m_data = macro_by_date.get(dt, {})
        vix = m_data.get("vix", float("nan")) if isinstance(m_data, dict) else float("nan")
        try:
            vix_val = float(vix)
        except:
            vix_val = float("nan")

        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_cur = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            if T_days <= 0:
                pnl = pos["net_credit_dollar"]
                if S < pos["put_strike"]:
                    assign_loss = (pos["put_strike"] - S) * 100 * pos["contracts"]
                    pnl -= assign_loss
                    stats["n_assigned"] += 1
                    stats["n_expiry_loss"] += 1
                else:
                    stats["n_expiry_win"] += 1

                cash += pnl
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": pnl, "close_reason": "expiry"
                })
                to_remove.append(tk)
                stats["n_closed"] += 1
                continue

            put_prem_now = bs_price(S, pos["put_strike"], T, sigma_cur, kind="put")
            cost_to_close = put_prem_now * 100 * pos["contracts"]
            ba_close = ba_cost_per_leg(put_prem_now, ba_frac) * pos["contracts"]
            unrealized_pnl = pos["net_credit_dollar"] - cost_to_close - ba_close

            if unrealized_pnl >= profit_take * pos["net_credit_dollar"]:
                cash += unrealized_pnl
                stats["total_ba_cost"] += ba_close
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": unrealized_pnl, "close_reason": "profit_take"
                })
                to_remove.append(tk)
                stats["n_profit_take"] += 1
                stats["n_closed"] += 1
                continue

            if unrealized_pnl < -(stop_loss_mult * pos["net_credit_dollar"]):
                cash += unrealized_pnl
                stats["total_ba_cost"] += ba_close
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": unrealized_pnl, "close_reason": "stop_loss"
                })
                to_remove.append(tk)
                stats["n_stop_loss"] += 1
                stats["n_closed"] += 1
                continue

        for tk in to_remove:
            if tk in positions:
                del positions[tk]

        if len(positions) < max_concurrent and not np.isnan(vix_val) and vix_val < 35:
            candidates = []
            for tk in ticker_list:
                if tk in positions:
                    continue
                S = date_px.get(tk)
                sigma = date_sigma.get(tk)
                if S is None or sigma is None or np.isnan(S) or np.isnan(sigma):
                    continue
                if sigma < 0.10:
                    continue
                if has_earnings_within(tk, dt, dte_target, earnings_lookup, 7):
                    continue
                candidates.append((tk, S, sigma))

            candidates.sort(key=lambda x: x[2], reverse=True)

            for tk, S, sigma in candidates:
                if len(positions) >= max_concurrent:
                    break
                T = dte_target / 365.0
                K_put = strike_from_delta(S, T, sigma, put_delta, kind="put")
                put_prem = bs_price(S, K_put, T, sigma, kind="put")
                if put_prem <= 0.05:
                    continue

                ba_open = ba_cost_per_leg(put_prem, ba_frac)
                margin_needed = K_put * 100
                sizing_base = min(cash, STARTING_CAPITAL * 1.5)
                max_alloc = sizing_base * per_name_pct
                contracts = max(1, int(max_alloc / margin_needed))
                if contracts * margin_needed > sizing_base * 0.15:
                    contracts = max(1, int(sizing_base * 0.15 / margin_needed))

                net_credit = put_prem * 100 * contracts - ba_open * contracts
                stats["total_ba_cost"] += ba_open * contracts
                if net_credit <= 0:
                    continue

                expiry = dt + pd.Timedelta(days=dte_target)
                positions[tk] = {
                    "open_date": dt, "expiry": expiry, "contracts": contracts,
                    "put_strike": K_put, "net_credit_dollar": net_credit,
                    "open_sigma": sigma, "open_price": S,
                }
                cash += net_credit
                stats["n_opened"] += 1

        mtm = 0
        for tk, pos in positions.items():
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_cur = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            put_prem = bs_price(S, pos["put_strike"], T, sigma_cur, kind="put")
            cost_to_close = put_prem * 100 * pos["contracts"]
            mtm -= cost_to_close

        total_equity = cash + mtm
        equity_curve.append({"date": dt, "equity": total_equity, "cash": cash, "n_pos": len(positions)})

    return equity_curve, detailed_trades, stats


# ═══════════════════════════════════════════════════════════════════
# Analysis & Regime Testing
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(equity_curve, label="Strategy"):
    """Compute Sharpe, Sortino, CAGR, max DD, PF, WR from equity curve."""
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["daily_ret"] = df["equity"].pct_change()
    df = df.dropna(subset=["daily_ret"])

    if len(df) < 30:
        return {"label": label, "error": "too few data points"}

    rets = df["daily_ret"].values
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1)

    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-6
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    total_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    years = total_days / 365.25
    total_return = df["equity"].iloc[-1] / df["equity"].iloc[0] - 1
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0

    cum_max = df["equity"].cummax()
    drawdown = (df["equity"] - cum_max) / cum_max
    max_dd = drawdown.min()

    # PF and WR from daily returns
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")
    wr = np.mean(rets > 0)

    return {
        "label": label,
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "cagr": round(cagr * 100, 1),
        "max_dd": round(max_dd * 100, 1),
        "pf": round(pf, 2),
        "wr": round(wr * 100, 1),
        "total_return": round(total_return * 100, 1),
        "n_days": len(df),
        "start": str(df["date"].iloc[0].date()),
        "end": str(df["date"].iloc[-1].date()),
    }


def regime_test(equity_curve, spy_df, label="Strategy"):
    """R1 regime-agnostic test: stratify returns by SPY regime."""
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["daily_ret"] = df["equity"].pct_change()
    df = df.dropna(subset=["daily_ret"])

    spy = spy_df.copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy["spy_ret"] = spy["close"].pct_change()
    spy = spy[["date", "spy_ret"]].dropna()

    merged = df.merge(spy, on="date", how="inner")
    if len(merged) < 60:
        return {"label": label, "error": "insufficient data for regime test"}

    # Classify: green (SPY > 0), red (SPY < 0), flat (|SPY| < 0.001)
    merged["regime"] = np.where(merged["spy_ret"] > 0.001, "green",
                       np.where(merged["spy_ret"] < -0.001, "red", "flat"))

    results = {}
    for regime in ["green", "red", "flat"]:
        subset = merged[merged["regime"] == regime]["daily_ret"]
        if len(subset) < 10:
            results[regime] = {"sharpe": 0, "n_days": len(subset)}
            continue
        mean_r = subset.mean()
        std_r = subset.std(ddof=1)
        sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0
        results[regime] = {"sharpe": round(sharpe, 2), "n_days": len(subset)}

    # R1 gap test
    s_green = results.get("green", {}).get("sharpe", 0)
    s_red = results.get("red", {}).get("sharpe", 0)
    max_abs = max(abs(s_green), abs(s_red))
    r1_gap = abs(s_green - s_red) / max_abs if max_abs > 0 else 0

    return {
        "label": label,
        "green_sharpe": results.get("green", {}).get("sharpe", 0),
        "red_sharpe": results.get("red", {}).get("sharpe", 0),
        "flat_sharpe": results.get("flat", {}).get("sharpe", 0),
        "green_days": results.get("green", {}).get("n_days", 0),
        "red_days": results.get("red", {}).get("n_days", 0),
        "flat_days": results.get("flat", {}).get("n_days", 0),
        "r1_gap": round(r1_gap, 3),
        "r1_pass": r1_gap < 0.50,
    }


def permutation_test_trades(trades, n_perms=2000, label="Strategy"):
    """Permutation test on TRADE PnLs (not equity returns).
    Null hypothesis: trade PnLs have zero mean (random entry timing).
    Shuffle the signs of trade PnLs and compare mean PnL."""
    if not trades or len(trades) < 10:
        return {"label": label, "error": "insufficient trades"}

    pnls = np.array([t["pnl"] for t in trades])
    actual_mean = np.mean(pnls)
    actual_t = actual_mean / (np.std(pnls, ddof=1) / np.sqrt(len(pnls)))

    perm_ts = []
    for _ in range(n_perms):
        # Random sign flip (Pitman permutation test)
        signs = np.random.choice([-1, 1], size=len(pnls))
        shuffled = pnls * signs
        perm_mean = np.mean(shuffled)
        perm_t = perm_mean / (np.std(shuffled, ddof=1) / np.sqrt(len(shuffled)))
        perm_ts.append(perm_t)

    p_value = np.mean(np.array(perm_ts) >= actual_t)

    return {
        "label": label,
        "actual_mean_pnl": round(actual_mean, 2),
        "actual_t_stat": round(actual_t, 3),
        "p_value": round(p_value, 4),
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
        "n_trades": len(pnls),
    }


def walk_forward_analysis(equity_curve, train_months=60, test_months=1, label="Strategy"):
    """Walk-forward OOT analysis: 60-month train, 1-month test, sliding."""
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["daily_ret"] = df["equity"].pct_change()
    df = df.dropna(subset=["daily_ret"])

    df["ym"] = df["date"].dt.to_period("M")
    months = sorted(df["ym"].unique())

    if len(months) < train_months + test_months:
        return {"label": label, "error": f"need {train_months+test_months} months, have {len(months)}"}

    oot_results = []
    for i in range(train_months, len(months)):
        test_month = months[i]
        train_start = months[max(0, i - train_months)]

        test_rets = df[df["ym"] == test_month]["daily_ret"].values
        if len(test_rets) < 5:
            continue

        monthly_ret = np.sum(test_rets)
        monthly_std = np.std(test_rets, ddof=1) if len(test_rets) > 1 else 0.01
        monthly_sharpe = (np.mean(test_rets) / monthly_std) * np.sqrt(252) if monthly_std > 0 else 0

        oot_results.append({
            "month": str(test_month),
            "return_pct": round(monthly_ret * 100, 2),
            "sharpe": round(monthly_sharpe, 2),
            "n_days": len(test_rets),
            "positive": monthly_ret > 0,
        })

    if not oot_results:
        return {"label": label, "error": "no OOT months"}

    oot_df = pd.DataFrame(oot_results)
    win_rate = oot_df["positive"].mean()
    avg_sharpe = oot_df["sharpe"].mean()
    avg_return = oot_df["return_pct"].mean()

    return {
        "label": label,
        "n_oot_months": len(oot_results),
        "monthly_win_rate": round(win_rate * 100, 1),
        "avg_monthly_sharpe": round(avg_sharpe, 2),
        "avg_monthly_return_pct": round(avg_return, 2),
        "worst_month_pct": round(oot_df["return_pct"].min(), 2),
        "best_month_pct": round(oot_df["return_pct"].max(), 2),
        "oot_months": oot_results,
    }


def trade_analysis(trades, label="Strategy"):
    """Analyze trade-level statistics."""
    if not trades:
        return {"label": label, "error": "no trades"}

    df = pd.DataFrame(trades)
    n_trades = len(df)
    n_winners = (df["pnl"] > 0).sum()
    n_losers = (df["pnl"] <= 0).sum()
    wr = n_winners / n_trades if n_trades > 0 else 0

    avg_win = df[df["pnl"] > 0]["pnl"].mean() if n_winners > 0 else 0
    avg_loss = df[df["pnl"] <= 0]["pnl"].mean() if n_losers > 0 else 0
    pf = abs(df[df["pnl"] > 0]["pnl"].sum() / df[df["pnl"] <= 0]["pnl"].sum()) if n_losers > 0 and df[df["pnl"] <= 0]["pnl"].sum() != 0 else float("inf")

    close_reasons = df["close_reason"].value_counts().to_dict()

    return {
        "label": label,
        "n_trades": n_trades,
        "win_rate": round(wr * 100, 1),
        "profit_factor": round(pf, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_pnl": round(df["pnl"].sum(), 2),
        "close_reasons": close_reasons,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    t0 = time.time()

    prices, iv, macro, vix_df, spy, earnings, avail_tickers = load_all_data()
    earnings_lookup = build_earnings_lookup(earnings)

    print(f"\nData loaded in {time.time()-t0:.1f}s")
    print(f"Running backtests for {len(avail_tickers)} tickers...")

    all_results = {}

    # ═══════════════════════════════════════════════════════════════
    # 1. JADE LIZARD — sweep parameters
    # ═══════════════════════════════════════════════════════════════
    jade_configs = [
        {"label": "JL_d25_35dte_ba10", "put_delta": -0.25, "call_delta": 0.25,
         "dte_target": 35, "ba_frac": 0.10, "long_call_offset": 0.05},
        {"label": "JL_d20_35dte_ba10", "put_delta": -0.20, "call_delta": 0.20,
         "dte_target": 35, "ba_frac": 0.10, "long_call_offset": 0.05},
        {"label": "JL_d30_35dte_ba10", "put_delta": -0.30, "call_delta": 0.30,
         "dte_target": 35, "ba_frac": 0.10, "long_call_offset": 0.05},
        {"label": "JL_d25_45dte_ba10", "put_delta": -0.25, "call_delta": 0.25,
         "dte_target": 45, "ba_frac": 0.10, "long_call_offset": 0.05},
        {"label": "JL_d25_35dte_ba15", "put_delta": -0.25, "call_delta": 0.25,
         "dte_target": 35, "ba_frac": 0.15, "long_call_offset": 0.05},
        {"label": "JL_d25_35dte_ba05", "put_delta": -0.25, "call_delta": 0.25,
         "dte_target": 35, "ba_frac": 0.05, "long_call_offset": 0.05},
    ]

    for cfg in jade_configs:
        label = cfg.pop("label")
        eq, trades, stats = run_jade_lizard(
            prices, iv, macro, vix_df, earnings_lookup, avail_tickers, **cfg)
        metrics = compute_metrics(eq, label)
        regime = regime_test(eq, spy, label)
        perm = permutation_test_trades(trades, n_perms=2000, label=label)
        wf = walk_forward_analysis(eq, label=label)
        trd = trade_analysis(trades, label)

        all_results[label] = {
            "metrics": metrics, "regime": regime, "permutation": perm,
            "walk_forward": wf, "trade_stats": trd, "engine_stats": stats,
        }
        print(f"\n  {label}: Sharpe={metrics.get('sharpe','?')}, Sortino={metrics.get('sortino','?')}, "
              f"CAGR={metrics.get('cagr','?')}%, MaxDD={metrics.get('max_dd','?')}%, "
              f"R1_gap={regime.get('r1_gap','?')}, R1_pass={regime.get('r1_pass','?')}")

    # ═══════════════════════════════════════════════════════════════
    # 2. STRANGLE SELLING — sweep parameters
    # ═══════════════════════════════════════════════════════════════
    strangle_configs = [
        {"label": "STR_d20_35dte_ba10", "put_delta": -0.20, "call_delta": 0.20,
         "dte_target": 35, "ba_frac": 0.10},
        {"label": "STR_d25_35dte_ba10", "put_delta": -0.25, "call_delta": 0.25,
         "dte_target": 35, "ba_frac": 0.10},
        {"label": "STR_d15_35dte_ba10", "put_delta": -0.15, "call_delta": 0.15,
         "dte_target": 35, "ba_frac": 0.10},
        {"label": "STR_d20_45dte_ba10", "put_delta": -0.20, "call_delta": 0.20,
         "dte_target": 45, "ba_frac": 0.10},
        {"label": "STR_d20_35dte_ba15", "put_delta": -0.20, "call_delta": 0.20,
         "dte_target": 35, "ba_frac": 0.15},
        {"label": "STR_d20_35dte_ba05", "put_delta": -0.20, "call_delta": 0.20,
         "dte_target": 35, "ba_frac": 0.05},
    ]

    for cfg in strangle_configs:
        label = cfg.pop("label")
        eq, trades, stats = run_strangle(
            prices, iv, macro, vix_df, earnings_lookup, avail_tickers, **cfg)
        metrics = compute_metrics(eq, label)
        regime = regime_test(eq, spy, label)
        perm = permutation_test_trades(trades, n_perms=2000, label=label)
        wf = walk_forward_analysis(eq, label=label)
        trd = trade_analysis(trades, label)

        all_results[label] = {
            "metrics": metrics, "regime": regime, "permutation": perm,
            "walk_forward": wf, "trade_stats": trd, "engine_stats": stats,
        }
        print(f"\n  {label}: Sharpe={metrics.get('sharpe','?')}, Sortino={metrics.get('sortino','?')}, "
              f"CAGR={metrics.get('cagr','?')}%, MaxDD={metrics.get('max_dd','?')}%, "
              f"R1_gap={regime.get('r1_gap','?')}, R1_pass={regime.get('r1_pass','?')}")

    # ═══════════════════════════════════════════════════════════════
    # 3. CSP BASELINE
    # ═══════════════════════════════════════════════════════════════
    for ba in [0.05, 0.10, 0.15]:
        label = f"CSP_d25_35dte_ba{int(ba*100):02d}"
        eq, trades, stats = run_csp_baseline(
            prices, iv, macro, vix_df, earnings_lookup, avail_tickers,
            ba_frac=ba, dte_target=35, put_delta=-0.25)
        metrics = compute_metrics(eq, label)
        regime = regime_test(eq, spy, label)
        perm = permutation_test_trades(trades, n_perms=2000, label=label)
        wf = walk_forward_analysis(eq, label=label)
        trd = trade_analysis(trades, label)

        all_results[label] = {
            "metrics": metrics, "regime": regime, "permutation": perm,
            "walk_forward": wf, "trade_stats": trd, "engine_stats": stats,
        }
        print(f"\n  {label}: Sharpe={metrics.get('sharpe','?')}, Sortino={metrics.get('sortino','?')}, "
              f"CAGR={metrics.get('cagr','?')}%, MaxDD={metrics.get('max_dd','?')}%, "
              f"R1_gap={regime.get('r1_gap','?')}, R1_pass={regime.get('r1_pass','?')}")

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print(f"\n\n{'='*80}")
    print("FINAL COMPARISON")
    print(f"{'='*80}")
    print(f"{'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>7} {'PF':>6} {'WR%':>6} {'R1gap':>7} {'R1':>5} {'p-val':>7}")
    print("-" * 100)

    for label in sorted(all_results.keys()):
        r = all_results[label]
        m = r["metrics"]
        rg = r["regime"]
        pm = r["permutation"]
        if "error" in m:
            print(f"  {label}: ERROR: {m['error']}")
            continue
        print(f"  {label:<28} {m['sharpe']:>7} {m['sortino']:>8} {m['cagr']:>6.1f}% {m['max_dd']:>6.1f}% "
              f"{m['pf']:>6.2f} {m['wr']:>5.1f}% {rg.get('r1_gap',0):>6.3f} "
              f"{'PASS' if rg.get('r1_pass') else 'FAIL':>5} {pm.get('p_value',1):>6.4f}")

    # Save results
    output_file = OUTPUT / "jade_lizard_strangle_results.json"

    # Convert any non-serializable types
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    def clean_dict(d):
        if isinstance(d, dict):
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [clean_dict(x) for x in d]
        else:
            return make_serializable(d)

    with open(output_file, "w") as f:
        json.dump(clean_dict(all_results), f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print(f"Total runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
