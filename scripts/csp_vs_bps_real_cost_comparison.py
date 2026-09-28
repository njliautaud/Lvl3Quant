#!/usr/bin/env python3
"""
CSP vs BPS Real-World Cost Comparison
========================================
Tests whether Cash-Secured Puts survive realistic bid-ask costs
where Bull Put Spreads failed definitively.

KEY HYPOTHESIS:
  - BPS crosses the spread TWICE (sell short put at bid, buy long put at ask)
  - CSP crosses the spread ONCE (sell one put at bid)
  - CSP should have ~half the cost drag per trade
  - BUT: CSP has unlimited downside (assignment risk) and needs more capital

COST MODEL:
  - BA cost = fraction of option mid-price lost to crossing the spread
  - BPS: ba_frac * 2 legs (buy + sell), applied to net spread value
  - CSP: ba_frac * 1 leg (sell only), applied to put premium
  - Commission: $0.65/contract/leg (BPS=2 legs, CSP=1 leg per side)

SCENARIOS:
  1-4: CSP at 7 DTE across BA = {5%, 10%, 15%, 20%}
  5-8: BPS at 7 DTE across same BA levels (for comparison)
  9-11: CSP at longer DTE {14, 30, 45} at 15% BA
  12: CSP break-even BA% (binary search for Sharpe=0)
  13: CSP assignment risk / crash analysis

Output: output/csp_vs_bps_real_costs/
"""

import sys, json, time, math, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUTPUT = ROOT / "output" / "csp_vs_bps_real_costs"
OUTPUT.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100_000

# ═══════════════════════════════════════════════════════════════════
# Ticker Universe (same as BPS full stack for fair comparison)
# ═══════════════════════════════════════════════════════════════════

TIER1_TICKERS = [
    'AAPL','ABBV','ABNB','ADBE','AMD','AMZN','ARM','AXP','BA','BAC',
    'BLK','BRK-B','C','CAT','CL','COIN','COST','CRM','CRWD','CVX',
    'DDOG','DE','DIS','F','GE','GM','GOOGL','GS','HD','HOOD',
    'INTC','JNJ','JPM','KO','LLY','LOW','MA','MCD','META','MRNA',
    'MS','MSFT','NFLX','NOW','NVDA','ORCL','OXY','PANW','PEP','PFE',
    'PG','PLTR','PYPL','RTX','SBUX','SCHW','SHOP','SLB','SMCI','T',
    'TGT','TMUS','TSLA','UBER','UNH','V','VZ','WFC','WMT','XOM',
]

TIER2_TICKERS = [
    'BIIB','REGN','VRTX','GILD',
    'MRVL','ON','SWKS','QRVO',
    'DVN','FANG','MPC','VLO','PSX',
    'O','AMT','PLD','EQIX','SPG',
    'GWW','EMR','ROK','ITW','ETN',
    'DG','DLTR','TJX','ROST','BBY',
]

TIER2_SECTORS = {
    'BIIB': 'Healthcare', 'REGN': 'Healthcare', 'VRTX': 'Healthcare', 'GILD': 'Healthcare',
    'MRVL': 'Technology', 'ON': 'Technology', 'SWKS': 'Technology', 'QRVO': 'Technology',
    'DVN': 'Energy', 'FANG': 'Energy', 'MPC': 'Energy', 'VLO': 'Energy', 'PSX': 'Energy',
    'O': 'Real Estate', 'AMT': 'Real Estate', 'PLD': 'Real Estate', 'EQIX': 'Real Estate', 'SPG': 'Real Estate',
    'GWW': 'Industrials', 'EMR': 'Industrials', 'ROK': 'Industrials', 'ITW': 'Industrials', 'ETN': 'Industrials',
    'DG': 'Consumer Defensive', 'DLTR': 'Consumer Defensive', 'TJX': 'Consumer Cyclical',
    'ROST': 'Consumer Cyclical', 'BBY': 'Consumer Cyclical',
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
# Cost Model
# ═══════════════════════════════════════════════════════════════════

COST_PER_CONTRACT = 0.65  # per contract per leg


def csp_trade_cost_open(put_premium, contracts, ba_frac):
    """
    CSP open: sell one put.
    - Commission: $0.65 * contracts (1 leg)
    - BA cost: seller gets bid = mid * (1 - ba_frac/2), so loses ba_frac/2 of premium
      Equivalently: ba_cost = put_premium * ba_frac/2 * 100 * contracts
    - Slippage already captured in BA model
    """
    comm = COST_PER_CONTRACT * contracts
    ba_cost = put_premium * (ba_frac / 2.0) * 100 * contracts
    return comm + ba_cost


def csp_trade_cost_close(put_premium, contracts, ba_frac):
    """
    CSP close: buy back the put.
    - Commission: $0.65 * contracts (1 leg)
    - BA cost: buyer pays ask = mid * (1 + ba_frac/2), so loses ba_frac/2 of premium
    """
    comm = COST_PER_CONTRACT * contracts
    ba_cost = put_premium * (ba_frac / 2.0) * 100 * contracts
    return comm + ba_cost


def bps_trade_cost_open(short_prem, long_prem, contracts, ba_frac):
    """
    BPS open: sell short put at bid, buy long put at ask.
    - Commission: $0.65 * contracts * 2 legs
    - BA on short leg: lose ba_frac/2 on sell (get bid)
    - BA on long leg: lose ba_frac/2 on buy (pay ask)
    """
    comm = COST_PER_CONTRACT * contracts * 2
    ba_short = short_prem * (ba_frac / 2.0) * 100 * contracts  # sell at bid
    ba_long = long_prem * (ba_frac / 2.0) * 100 * contracts    # buy at ask
    return comm + ba_short + ba_long


def bps_trade_cost_close(short_prem, long_prem, contracts, ba_frac):
    """
    BPS close: buy back short put at ask, sell long put at bid.
    - Commission: $0.65 * contracts * 2 legs
    - BA on both legs
    """
    comm = COST_PER_CONTRACT * contracts * 2
    ba_short = short_prem * (ba_frac / 2.0) * 100 * contracts  # buy at ask
    ba_long = long_prem * (ba_frac / 2.0) * 100 * contracts    # sell at bid
    return comm + ba_short + ba_long


# ═══════════════════════════════════════════════════════════════════
# IV Feature Generation
# ═══════════════════════════════════════════════════════════════════

def generate_iv_features(prices_df, tickers):
    """Generate modeled IV features from realized vol."""
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


# ═══════════════════════════════════════════════════════════════════
# Data Loading (same as BPS full stack)
# ═══════════════════════════════════════════════════════════════════

def load_all_data():
    """Load base + expanded prices, IV, macro, fundamentals, earnings."""
    print("Loading data...")

    prices = pd.read_parquet(CACHE / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    try:
        pexp = pd.read_parquet(CACHE / "prices_expanded.parquet")
        pexp = pexp.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                     "Close": "close", "Volume": "volume"})
        pexp["date"] = pd.to_datetime(pexp["date"]).dt.tz_localize(None)
        new_tks = set(pexp["ticker"].unique()) - set(prices["ticker"].unique())
        if new_tks:
            pexp = pexp[pexp["ticker"].isin(new_tks)]
            prices = pd.concat([prices, pexp], ignore_index=True)
    except Exception as e:
        print(f"  Warning: expanded prices: {e}")

    try:
        p3 = pd.read_parquet(CACHE / "prices_v3_expansion.parquet")
        p3["date"] = pd.to_datetime(p3["date"]).dt.tz_localize(None)
        new_tks = set(p3["ticker"].unique()) - set(prices["ticker"].unique())
        if new_tks:
            p3 = p3[p3["ticker"].isin(new_tks)]
            prices = pd.concat([prices, p3], ignore_index=True)
    except Exception as e:
        print(f"  Warning: V3 prices: {e}")

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

    all_needed = set(TIER1_TICKERS + TIER2_TICKERS)
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

    fund = pd.read_parquet(CACHE / "fundamentals.parquet")
    if "sector" not in fund.columns:
        fund["sector"] = "Unknown"

    existing_fund_tickers = set(fund["ticker"].unique())
    new_fund_rows = []
    for tk, sec in TIER2_SECTORS.items():
        if tk not in existing_fund_tickers:
            new_fund_rows.append({'ticker': tk, 'sector': sec, 'market_cap': 0, 'beta': 1.5})
    if new_fund_rows:
        fund = pd.concat([fund, pd.DataFrame(new_fund_rows)], ignore_index=True)

    universe = fund[["ticker", "sector"]].drop_duplicates("ticker")

    try:
        earnings = pd.read_parquet(CACHE / "earnings_dates.parquet")
        earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"]).dt.tz_localize(None)
    except:
        earnings = pd.DataFrame(columns=["ticker", "earnings_date"])

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique()) & all_needed
    t1_avail = [t for t in TIER1_TICKERS if t in available]
    t2_avail = [t for t in TIER2_TICKERS if t in available]

    print(f"  Prices: {prices.shape[0]} rows, {prices['ticker'].nunique()} tickers")
    print(f"  IV: {iv.shape[0]} rows, {iv['ticker'].nunique()} tickers")
    print(f"  Universe: Tier 1 = {len(t1_avail)}, Tier 2 = {len(t2_avail)}, Total = {len(t1_avail)+len(t2_avail)}")

    return prices, iv, macro, fund, universe, earnings


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
# Dynamic Delta (same as BPS)
# ═══════════════════════════════════════════════════════════════════

def dynamic_delta(vix):
    """d35 VIX<15, d30 VIX 15-22, d25 VIX>22."""
    if vix < 15:
        return 0.35
    elif vix < 22:
        return 0.30
    else:
        return 0.25


# ═══════════════════════════════════════════════════════════════════
# CSP Backtest Engine
# ═══════════════════════════════════════════════════════════════════

def run_csp(prices, iv, macro, earnings_lookup,
            ticker_list, ticker_tier_map, ba_frac,
            label="CSP", dte_target=7,
            profit_take=0.65, margin_cap=0.40,
            max_concurrent=30, per_name_pct=0.03,
            vix_scale=True, vix_base=15.0,
            vix_hard_cutoff=30,
            portfolio_cb_threshold=-0.02, portfolio_cb_freeze_days=1,
            earnings_filter=True, earnings_buffer_days=7):
    """
    Cash-Secured Put backtest engine.

    CSP = sell naked puts, cash-secured (need full assignment value in cash).
    If assigned, we immediately sell shares (no wheel, pure CSP).
    One option leg = one BA crossing.
    """
    print(f"  Running: {label} (DTE={dte_target}, BA={ba_frac*100:.0f}%)...")

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique()) & set(ticker_list)
    prices_df = prices[prices["ticker"].isin(available)].copy()
    iv_df = iv[iv["ticker"].isin(available)].copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    iv_rank_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()
        iv_rank_by_date[d] = g.set_index("ticker")["iv_rank"].to_dict()

    macro_copy = macro.copy()
    macro_copy["date"] = pd.to_datetime(macro_copy["date"])
    macro_by_date = macro_copy.set_index("date").to_dict("index")

    all_dates = sorted(prices_df["date"].unique())

    cash = float(STARTING_CAPITAL)
    positions = {}  # ticker -> pos dict
    equity_curve = []
    detailed_trades = []

    frozen_until = None
    cb_trigger_count = 0
    n_earnings_blocked = 0
    n_vix_blocked = 0
    n_cb_frozen = 0
    n_assignments = 0
    assignment_losses = []

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})
        m_data = macro_by_date.get(dt, {})
        vix = m_data.get("vix", float("nan")) if isinstance(m_data, dict) else float("nan")

        try:
            vix_val = float(vix)
        except (TypeError, ValueError):
            vix_val = float("nan")

        # ── Update/close existing positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            tier = ticker_tier_map.get(tk, "tier1")

            # ── Expiry ──
            if T_days <= 0:
                if S < pos["strike"]:
                    # ASSIGNED: stock below strike
                    # For pure CSP test: immediately sell the assigned shares at market
                    # Loss = (strike - stock_price) * 100 * contracts
                    # We already collected the premium at open
                    assignment_loss = (pos["strike"] - S) * 100 * pos["contracts"]
                    # Close commission on assignment
                    close_comm = COST_PER_CONTRACT * pos["contracts"]
                    realized = pos["net_credit"] - assignment_loss - close_comm
                    cash -= assignment_loss + close_comm
                    n_assignments += 1
                    assignment_losses.append({
                        "date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                        "ticker": tk,
                        "strike": pos["strike"],
                        "stock_price": S,
                        "loss_per_share": pos["strike"] - S,
                        "contracts": pos["contracts"],
                        "total_loss": assignment_loss,
                        "net_pnl_after_premium": realized,
                    })

                    detailed_trades.append({
                        "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                        "exit_type": "assignment",
                        "strike": pos["strike"], "open_price": pos["open_stock_price"],
                        "close_price": S, "net_credit": pos["net_credit"],
                        "realized_pnl": realized, "contracts": pos["contracts"],
                        "days_held": (dt - pos["open_date"]).days,
                        "put_delta_used": pos.get("put_delta_used", 0.30),
                        "open_vix": pos.get("open_vix", np.nan),
                        "tier": tier,
                    })
                else:
                    # Expire OTM = keep full premium (already in cash)
                    close_comm = COST_PER_CONTRACT * pos["contracts"]
                    realized = pos["net_credit"] - close_comm
                    cash -= close_comm

                    detailed_trades.append({
                        "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                        "exit_type": "expire_otm",
                        "strike": pos["strike"], "open_price": pos["open_stock_price"],
                        "close_price": S, "net_credit": pos["net_credit"],
                        "realized_pnl": realized, "contracts": pos["contracts"],
                        "days_held": (dt - pos["open_date"]).days,
                        "put_delta_used": pos.get("put_delta_used", 0.30),
                        "open_vix": pos.get("open_vix", np.nan),
                        "tier": tier,
                    })
                to_remove.append(tk)
                continue

            # ── Early close at 1 DTE ──
            if T_days == 1:
                cur_val = bs_price(S, pos["strike"], T, sigma_atm, kind="put")
                close_cost = csp_trade_cost_close(cur_val, pos["contracts"], ba_frac)
                buyback = cur_val * 100 * pos["contracts"]
                realized = pos["net_credit"] - buyback - close_cost
                cash -= buyback + close_cost

                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "exit_type": "early_close_1DTE",
                    "strike": pos["strike"], "open_price": pos["open_stock_price"],
                    "close_price": S, "net_credit": pos["net_credit"],
                    "realized_pnl": realized, "contracts": pos["contracts"],
                    "days_held": (dt - pos["open_date"]).days,
                    "put_delta_used": pos.get("put_delta_used", 0.30),
                    "open_vix": pos.get("open_vix", np.nan),
                    "tier": tier,
                })
                to_remove.append(tk)
                continue

            # ── Profit take ──
            cur_val = bs_price(S, pos["strike"], T, sigma_atm, kind="put")
            captured = (pos["open_put_price"] - cur_val) / max(pos["open_put_price"], 1e-6)
            if captured >= profit_take:
                close_cost = csp_trade_cost_close(cur_val, pos["contracts"], ba_frac)
                buyback = cur_val * 100 * pos["contracts"]
                realized = pos["net_credit"] - buyback - close_cost
                cash -= buyback + close_cost

                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "exit_type": "profit_take",
                    "strike": pos["strike"], "open_price": pos["open_stock_price"],
                    "close_price": S, "net_credit": pos["net_credit"],
                    "realized_pnl": realized, "contracts": pos["contracts"],
                    "days_held": (dt - pos["open_date"]).days,
                    "put_delta_used": pos.get("put_delta_used", 0.30),
                    "open_vix": pos.get("open_vix", np.nan),
                    "tier": tier,
                })
                to_remove.append(tk)

        for tk in to_remove:
            del positions[tk]

        # ── MTM equity ──
        equity = cash
        for tk, pos in positions.items():
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T = max((pos["expiry"] - dt).days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            cur_val = bs_price(S, pos["strike"], T, sigma_atm, kind="put")
            # MTM liability: we owe the current put value
            equity -= cur_val * 100 * pos["contracts"]

        equity_curve.append({"date": dt, "equity": equity})

        # ── Circuit Breaker ──
        if portfolio_cb_threshold is not None and len(equity_curve) >= 2:
            prev_eq = equity_curve[-2]["equity"]
            if prev_eq > 0:
                daily_ret = (equity - prev_eq) / prev_eq
                if daily_ret < portfolio_cb_threshold:
                    freeze_end = dt + pd.Timedelta(days=portfolio_cb_freeze_days)
                    if frozen_until is None or freeze_end > frozen_until:
                        frozen_until = freeze_end
                        cb_trigger_count += 1

        # ── VIX hard cutoff ──
        if vix_hard_cutoff is not None and not np.isnan(vix_val) and vix_val > vix_hard_cutoff:
            n_vix_blocked += 1
            continue

        # ── CB frozen? ──
        if frozen_until is not None and dt <= frozen_until:
            n_cb_frozen += 1
            continue

        # ── VIX sizing scalar ──
        if vix_scale and not np.isnan(vix_val):
            vix_scalar = max(0.0, 1.0 - (vix_val - vix_base) / 30.0) if vix_val > vix_base else 1.0
        else:
            vix_scalar = 1.0

        # ── Dynamic delta ──
        if np.isnan(vix_val):
            put_delta = 0.30
        else:
            put_delta = dynamic_delta(vix_val)

        # ── Open new CSP positions ──
        # Cash-secured: need strike * 100 * contracts in cash as collateral
        # Use margin_cap as fraction of total equity available for CSP collateral
        current_collateral = sum(
            p["strike"] * 100 * p["contracts"] for p in positions.values()
        )
        max_collateral = margin_cap * equity
        remaining_collateral = max_collateral - current_collateral
        slots = max_concurrent - len(positions)

        if slots <= 0 or remaining_collateral <= 0:
            continue

        candidates = []
        for tk in ticker_list:
            if tk not in available or tk in positions:
                continue
            S = date_px.get(tk)
            sigma = date_sigma.get(tk)
            iv_rk = date_iv_rank.get(tk, 0)
            if S is None or sigma is None or np.isnan(S) or np.isnan(sigma):
                continue
            if sigma < 0.05:
                continue
            if earnings_filter and has_earnings_within(tk, dt, dte_target, earnings_lookup, earnings_buffer_days):
                n_earnings_blocked += 1
                continue
            candidates.append((tk, S, sigma, iv_rk))

        candidates.sort(key=lambda x: -x[3])  # highest IV rank first

        for tk, S, sigma, iv_rk in candidates[:slots]:
            T = dte_target / 365.0
            K = strike_from_delta(S, T, sigma, put_delta, kind="put")

            if K <= 0:
                continue

            put_price = bs_price(S, K, T, sigma, kind="put")
            if put_price <= 0.05:
                continue

            # Cash-secured: collateral = strike * 100
            collateral_per_contract = K * 100
            max_alloc = per_name_pct * equity
            n_contracts = max(1, int(max_alloc // collateral_per_contract))

            # VIX sizing
            n_contracts = max(1, int(n_contracts * vix_scalar))

            if collateral_per_contract * n_contracts > remaining_collateral:
                n_contracts = max(1, int(remaining_collateral // collateral_per_contract))
            if n_contracts < 1:
                continue

            # Open cost: sell put (1 leg, 1 crossing)
            open_cost = csp_trade_cost_open(put_price, n_contracts, ba_frac)
            net_credit = put_price * 100 * n_contracts - open_cost

            if net_credit <= 0:
                continue

            cash += net_credit
            positions[tk] = {
                "strike": K,
                "contracts": n_contracts,
                "net_credit": net_credit,
                "open_put_price": put_price,
                "open_date": dt,
                "expiry": dt + pd.Timedelta(days=dte_target),
                "open_sigma": sigma,
                "open_stock_price": S,
                "put_delta_used": put_delta,
                "open_vix": vix_val,
                "vix_scalar": vix_scalar,
            }
            remaining_collateral -= collateral_per_contract * n_contracts

            if len(positions) >= max_concurrent:
                break

    # ── Results ──
    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty or len(eq_df) < 30:
        return {"label": label, "error": "insufficient data"}

    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.sort_values("date").reset_index(drop=True)
    trades_df = pd.DataFrame(detailed_trades)

    return {
        "label": label,
        "equity_df": eq_df,
        "trades_df": trades_df,
        "n_earnings_blocked": n_earnings_blocked,
        "n_vix_blocked": n_vix_blocked,
        "n_cb_frozen": n_cb_frozen,
        "cb_triggers": cb_trigger_count,
        "n_assignments": n_assignments,
        "assignment_losses": assignment_losses,
    }


# ═══════════════════════════════════════════════════════════════════
# BPS Backtest Engine (simplified, for fair comparison)
# ═══════════════════════════════════════════════════════════════════

def run_bps(prices, iv, macro, earnings_lookup,
            ticker_list, ticker_tier_map, ba_frac,
            label="BPS", dte_target=7, spread_width=15.0,
            profit_take=0.65, margin_cap=0.25,
            max_concurrent=40, per_name_pct=0.03,
            vix_scale=True, vix_base=15.0,
            vix_hard_cutoff=30,
            portfolio_cb_threshold=-0.02, portfolio_cb_freeze_days=1,
            earnings_filter=True, earnings_buffer_days=7):
    """
    Bull Put Spread backtest engine.
    Two legs = two BA crossings.
    """
    print(f"  Running: {label} (DTE={dte_target}, BA={ba_frac*100:.0f}%)...")

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique()) & set(ticker_list)
    prices_df = prices[prices["ticker"].isin(available)].copy()
    iv_df = iv[iv["ticker"].isin(available)].copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    iv_rank_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()
        iv_rank_by_date[d] = g.set_index("ticker")["iv_rank"].to_dict()

    macro_copy = macro.copy()
    macro_copy["date"] = pd.to_datetime(macro_copy["date"])
    macro_by_date = macro_copy.set_index("date").to_dict("index")

    all_dates = sorted(prices_df["date"].unique())

    cash = float(STARTING_CAPITAL)
    positions = {}
    equity_curve = []
    detailed_trades = []

    frozen_until = None
    cb_trigger_count = 0
    n_earnings_blocked = 0
    n_vix_blocked = 0
    n_cb_frozen = 0

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})
        m_data = macro_by_date.get(dt, {})
        vix = m_data.get("vix", float("nan")) if isinstance(m_data, dict) else float("nan")

        try:
            vix_val = float(vix)
        except (TypeError, ValueError):
            vix_val = float("nan")

        # ── Update/close positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            tier = ticker_tier_map.get(tk, "tier1")

            # Expiry
            if T_days <= 0:
                short_itm = S < pos["short_strike"]
                long_itm = S < pos["long_strike"]
                close_comm = COST_PER_CONTRACT * 2 * pos["contracts"]

                if not short_itm:
                    realized = pos["net_credit"] - close_comm
                    cash -= close_comm
                    status = "OTM_safe"
                elif short_itm and not long_itm:
                    loss = (pos["short_strike"] - S) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_comm
                    cash -= loss + close_comm
                    status = "partial_breach"
                else:
                    loss = (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_comm
                    cash -= loss + close_comm
                    status = "full_breach"

                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "exit_type": "expiry", "status": status,
                    "short_strike": pos["short_strike"], "long_strike": pos["long_strike"],
                    "open_price": pos.get("open_stock_price", 0), "close_price": S,
                    "net_credit": pos["net_credit"], "realized_pnl": realized,
                    "contracts": pos["contracts"],
                    "days_held": (dt - pos["open_date"]).days,
                    "tier": tier,
                })
                to_remove.append(tk)
                continue

            # Early close 1 DTE
            if T_days == 1:
                short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
                long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
                close_cost = bps_trade_cost_close(short_val, long_val, pos["contracts"], ba_frac)
                spread_cost = (short_val - long_val) * 100 * pos["contracts"]
                realized = pos["net_credit"] - spread_cost - close_cost
                cash -= spread_cost + close_cost

                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "exit_type": "early_close_1DTE", "status": "closed",
                    "short_strike": pos["short_strike"], "long_strike": pos["long_strike"],
                    "open_price": pos.get("open_stock_price", 0), "close_price": S,
                    "net_credit": pos["net_credit"], "realized_pnl": realized,
                    "contracts": pos["contracts"],
                    "days_held": (dt - pos["open_date"]).days,
                    "tier": tier,
                })
                to_remove.append(tk)
                continue

            # Profit take
            short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
            long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
            spread_cost = (short_val - long_val) * 100 * pos["contracts"]
            close_cost = bps_trade_cost_close(short_val, long_val, pos["contracts"], ba_frac)

            captured = (pos["net_credit"] - spread_cost - close_cost) / max(pos["net_credit"], 1e-6)
            if captured >= profit_take:
                realized = pos["net_credit"] - spread_cost - close_cost
                cash -= spread_cost + close_cost

                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "exit_type": "profit_take", "status": "closed",
                    "short_strike": pos["short_strike"], "long_strike": pos["long_strike"],
                    "open_price": pos.get("open_stock_price", 0), "close_price": S,
                    "net_credit": pos["net_credit"], "realized_pnl": realized,
                    "contracts": pos["contracts"],
                    "days_held": (dt - pos["open_date"]).days,
                    "tier": tier,
                })
                to_remove.append(tk)

        for tk in to_remove:
            del positions[tk]

        # ── MTM equity ──
        equity = cash
        for tk, pos in positions.items():
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T = max((pos["expiry"] - dt).days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
            long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
            equity -= (short_val - long_val) * 100 * pos["contracts"]

        equity_curve.append({"date": dt, "equity": equity})

        # Circuit Breaker
        if portfolio_cb_threshold is not None and len(equity_curve) >= 2:
            prev_eq = equity_curve[-2]["equity"]
            if prev_eq > 0:
                daily_ret = (equity - prev_eq) / prev_eq
                if daily_ret < portfolio_cb_threshold:
                    freeze_end = dt + pd.Timedelta(days=portfolio_cb_freeze_days)
                    if frozen_until is None or freeze_end > frozen_until:
                        frozen_until = freeze_end
                        cb_trigger_count += 1

        if vix_hard_cutoff is not None and not np.isnan(vix_val) and vix_val > vix_hard_cutoff:
            n_vix_blocked += 1
            continue

        if frozen_until is not None and dt <= frozen_until:
            n_cb_frozen += 1
            continue

        if vix_scale and not np.isnan(vix_val):
            vix_scalar = max(0.0, 1.0 - (vix_val - vix_base) / 30.0) if vix_val > vix_base else 1.0
        else:
            vix_scalar = 1.0

        if np.isnan(vix_val):
            put_delta = 0.30
        else:
            put_delta = dynamic_delta(vix_val)

        # Open new BPS positions
        current_margin = sum(
            (p["short_strike"] - p["long_strike"]) * 100 * p["contracts"]
            for p in positions.values()
        )
        max_margin = margin_cap * equity
        remaining_margin = max_margin - current_margin
        slots = max_concurrent - len(positions)

        if slots <= 0 or remaining_margin <= 0:
            continue

        candidates = []
        for tk in ticker_list:
            if tk not in available or tk in positions:
                continue
            S = date_px.get(tk)
            sigma = date_sigma.get(tk)
            iv_rk = date_iv_rank.get(tk, 0)
            if S is None or sigma is None or np.isnan(S) or np.isnan(sigma):
                continue
            if sigma < 0.05:
                continue
            if earnings_filter and has_earnings_within(tk, dt, dte_target, earnings_lookup, earnings_buffer_days):
                n_earnings_blocked += 1
                continue
            candidates.append((tk, S, sigma, iv_rk))

        candidates.sort(key=lambda x: -x[3])

        for tk, S, sigma, iv_rk in candidates[:slots]:
            T = dte_target / 365.0
            K_short = strike_from_delta(S, T, sigma, put_delta, kind="put")
            K_long = K_short - spread_width

            if K_long <= 0 or K_short <= 0:
                continue

            prem_short = bs_price(S, K_short, T, sigma, kind="put")
            prem_long = bs_price(S, K_long, T, sigma, kind="put")
            net_prem = prem_short - prem_long

            if net_prem <= 0.05:
                continue

            margin_per_contract = spread_width * 100
            max_alloc = per_name_pct * equity
            n_contracts = max(1, int(max_alloc // margin_per_contract))
            n_contracts = max(1, int(n_contracts * vix_scalar))

            if margin_per_contract * n_contracts > remaining_margin:
                n_contracts = max(1, int(remaining_margin // margin_per_contract))
            if n_contracts < 1:
                continue

            # BPS open: 2 legs, 2 crossings
            open_cost = bps_trade_cost_open(prem_short, prem_long, n_contracts, ba_frac)
            net_credit = net_prem * 100 * n_contracts - open_cost

            if net_credit <= 0:
                continue

            cash += net_credit
            positions[tk] = {
                "short_strike": K_short,
                "long_strike": K_long,
                "contracts": n_contracts,
                "net_credit": net_credit,
                "open_date": dt,
                "expiry": dt + pd.Timedelta(days=dte_target),
                "open_sigma": sigma,
                "open_stock_price": S,
                "open_short_prem": prem_short,
                "open_long_prem": prem_long,
                "put_delta_used": put_delta,
                "open_vix": vix_val,
                "vix_scalar": vix_scalar,
            }
            remaining_margin -= margin_per_contract * n_contracts

            if len(positions) >= max_concurrent:
                break

    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty or len(eq_df) < 30:
        return {"label": label, "error": "insufficient data"}

    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.sort_values("date").reset_index(drop=True)
    trades_df = pd.DataFrame(detailed_trades)

    return {
        "label": label,
        "equity_df": eq_df,
        "trades_df": trades_df,
        "n_earnings_blocked": n_earnings_blocked,
        "n_vix_blocked": n_vix_blocked,
        "n_cb_frozen": n_cb_frozen,
        "cb_triggers": cb_trigger_count,
    }


# ═══════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(eq_df, trades_df, label, starting_cap=STARTING_CAPITAL):
    """Comprehensive risk-adjusted metrics."""
    eq = eq_df.copy().sort_values("date").reset_index(drop=True)
    eq["ret"] = eq["equity"].pct_change()
    rets = eq["ret"].dropna()

    total_days = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days
    total_years = max(total_days / 365.25, 0.01)
    total_return = eq["equity"].iloc[-1] / starting_cap
    cagr = (total_return ** (1 / total_years)) - 1 if total_return > 0 else -1.0

    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    downside = rets[rets < 0]
    sortino = float(rets.mean() / downside.std() * np.sqrt(252)) if len(downside) > 5 and downside.std() > 0 else 0.0

    eq["peak"] = eq["equity"].cummax()
    eq["dd"] = (eq["equity"] - eq["peak"]) / eq["peak"]
    max_dd = float(eq["dd"].min())

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    daily_wr = float(len(rets[rets > 0]) / len(rets) * 100) if len(rets) > 0 else 0.0

    wins_sum = rets[rets > 0].sum()
    losses_sum = abs(rets[rets < 0].sum())
    pf = float(wins_sum / losses_sum) if losses_sum > 0 else float("inf")

    # Trade metrics
    n_trades = len(trades_df)
    if n_trades > 0:
        trade_wr = float((trades_df["realized_pnl"] > 0).sum() / n_trades * 100)
        avg_pnl = float(trades_df["realized_pnl"].mean())
        total_pnl = float(trades_df["realized_pnl"].sum())
        avg_credit = float(trades_df["net_credit"].mean())
    else:
        trade_wr = 0.0
        avg_pnl = 0.0
        total_pnl = 0.0
        avg_credit = 0.0

    # Per-year
    eq["year"] = eq["date"].dt.year
    per_year = {}
    for yr, grp in eq.groupby("year"):
        if len(grp) < 5:
            continue
        yr_ret = grp["equity"].iloc[-1] / grp["equity"].iloc[0] - 1
        yr_rets = grp["ret"].dropna()
        yr_sharpe = float(yr_rets.mean() / yr_rets.std() * np.sqrt(252)) if yr_rets.std() > 0 else 0.0
        yr_down = yr_rets[yr_rets < 0]
        yr_sortino = float(yr_rets.mean() / yr_down.std() * np.sqrt(252)) if len(yr_down) > 3 and yr_down.std() > 0 else 0.0
        yr_dd = float(((grp["equity"] / grp["equity"].cummax()) - 1).min())
        per_year[int(yr)] = {
            "return_pct": round(yr_ret * 100, 2),
            "sharpe": round(yr_sharpe, 2),
            "sortino": round(yr_sortino, 2),
            "max_dd_pct": round(yr_dd * 100, 2),
        }

    # Worst drawdown episode
    max_dd_idx = eq["dd"].idxmin()
    trough_eq = eq.loc[max_dd_idx, "equity"]
    pre_peak = eq.loc[max_dd_idx, "peak"]
    peak_dates = eq[eq["equity"] == pre_peak]
    peak_start = peak_dates.iloc[0]["date"] if len(peak_dates) > 0 else eq["date"].iloc[0]
    max_dd_date = eq.loc[max_dd_idx, "date"]

    post_trough = eq.loc[max_dd_idx:]
    recovered = post_trough[post_trough["equity"] >= pre_peak]
    recovery_days = (recovered.iloc[0]["date"] - max_dd_date).days if len(recovered) > 0 else None

    return {
        "label": label,
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "profit_factor": round(pf, 2),
        "daily_wr_pct": round(daily_wr, 1),
        "trade_wr_pct": round(trade_wr, 1),
        "n_trades": n_trades,
        "avg_trade_pnl": round(avg_pnl, 2),
        "avg_credit": round(avg_credit, 2),
        "total_trade_pnl": round(total_pnl, 2),
        "final_equity": round(float(eq["equity"].iloc[-1]), 2),
        "per_year": per_year,
        "worst_dd": {
            "peak_date": str(peak_start.date()) if hasattr(peak_start, 'date') else str(peak_start),
            "trough_date": str(max_dd_date.date()) if hasattr(max_dd_date, 'date') else str(max_dd_date),
            "depth_pct": round(max_dd * 100, 2),
            "recovery_days": recovery_days,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Permutation Test
# ═══════════════════════════════════════════════════════════════════

def run_permutation_test(trades_df, n_permutations=200):
    """Shuffle trade PnLs to test if results are statistically significant."""
    if trades_df.empty:
        return {"verdict": "NO TRADES", "p_value": 1.0}

    pnls = trades_df["realized_pnl"].values.copy()
    actual_total = float(pnls.sum())
    actual_sharpe = float(pnls.mean() / max(pnls.std(), 1e-9))

    random_totals = []
    random_sharpes = []
    rng = np.random.default_rng(42)

    for _ in range(n_permutations):
        shuffled = pnls.copy()
        # Randomly flip signs (destroys any timing edge)
        signs = rng.choice([-1, 1], size=len(shuffled))
        shuffled = shuffled * signs
        random_totals.append(float(shuffled.sum()))
        random_sharpes.append(float(shuffled.mean() / max(shuffled.std(), 1e-9)))

    p_value_pnl = float(np.mean(np.array(random_totals) >= actual_total))
    p_value_sharpe = float(np.mean(np.array(random_sharpes) >= actual_sharpe))

    if p_value_pnl < 0.05:
        verdict = "SIGNIFICANT (p < 0.05) -- real edge, not luck"
    elif p_value_pnl < 0.10:
        verdict = "MARGINAL (p < 0.10) -- weak evidence of edge"
    else:
        verdict = "NOT SIGNIFICANT (p >= 0.10) -- could be luck"

    return {
        "actual_total_pnl": actual_total,
        "actual_sharpe": actual_sharpe,
        "random_pnl_mean": float(np.mean(random_totals)),
        "random_pnl_std": float(np.std(random_totals)),
        "p_value_pnl": p_value_pnl,
        "p_value_sharpe": p_value_sharpe,
        "n_permutations": n_permutations,
        "verdict": verdict,
    }


# ═══════════════════════════════════════════════════════════════════
# Break-Even Analysis
# ═══════════════════════════════════════════════════════════════════

def find_csp_breakeven_ba(prices, iv, macro, earnings_lookup,
                          ticker_list, ticker_tier_map, dte_target=7):
    """Binary search for the BA% where CSP Sharpe crosses 0."""
    print("\n  Finding CSP break-even BA% (binary search)...")

    def sharpe_at_ba(ba_pct):
        ba_frac = ba_pct / 100.0
        result = run_csp(
            prices, iv, macro, earnings_lookup,
            ticker_list, ticker_tier_map, ba_frac,
            label=f"breakeven_{ba_pct:.1f}pct",
            dte_target=dte_target,
        )
        if "error" in result:
            return -1.0
        eq = result["equity_df"].copy().sort_values("date").reset_index(drop=True)
        rets = eq["equity"].pct_change().dropna()
        if rets.std() <= 0:
            return 0.0
        return float(rets.mean() / rets.std() * np.sqrt(252))

    s_low = sharpe_at_ba(1.0)
    s_high = sharpe_at_ba(60.0)
    print(f"    Sharpe at 1% BA: {s_low:.3f}")
    print(f"    Sharpe at 60% BA: {s_high:.3f}")

    if s_low <= 0:
        print("    CSP unprofitable even at 1% BA!")
        return 1.0, s_low
    if s_high > 0:
        print("    CSP still profitable at 60% BA! Break-even > 60%")
        return 60.0, s_high

    lo, hi = 1.0, 60.0
    for iteration in range(15):
        mid = (lo + hi) / 2.0
        s_mid = sharpe_at_ba(mid)
        print(f"    Iteration {iteration+1}: BA={mid:.2f}% -> Sharpe={s_mid:.3f}")
        if s_mid > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.2:
            break

    breakeven = (lo + hi) / 2.0
    s_be = sharpe_at_ba(breakeven)
    print(f"    Break-even BA: ~{breakeven:.1f}% (Sharpe={s_be:.3f})")
    return breakeven, s_be


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 120)
    print("CSP vs BPS REAL-WORLD COST COMPARISON")
    print("Hypothesis: CSP crosses spread ONCE (vs BPS twice), so it may survive higher BA costs")
    print("=" * 120)

    # Load data
    prices, iv, macro, fund, universe, earnings = load_all_data()
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    earnings_lookup = build_earnings_lookup(earnings)

    # Build tier maps
    ticker_tier = {}
    for tk in TIER1_TICKERS:
        ticker_tier[tk] = "tier1"
    for tk in TIER2_TICKERS:
        ticker_tier[tk] = "tier2"

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique())
    t1_avail = [t for t in TIER1_TICKERS if t in available]
    t2_avail = [t for t in TIER2_TICKERS if t in available]
    all_avail = t1_avail + t2_avail

    print(f"\n  Available: Tier 1 = {len(t1_avail)}, Tier 2 = {len(t2_avail)}, Total = {len(all_avail)}")

    # ═══════════════════════════════════════════════════════════════
    # PART 1: CSP vs BPS at varying BA costs (7 DTE)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("PART 1: CSP vs BPS HEAD-TO-HEAD AT VARYING BA COSTS (7 DTE)")
    print("=" * 120)

    ba_levels = [0.05, 0.10, 0.15, 0.20]
    all_results = {}

    for ba in ba_levels:
        # CSP
        csp_result = run_csp(
            prices, iv, macro, earnings_lookup,
            all_avail, ticker_tier, ba,
            label=f"CSP_7DTE_{int(ba*100)}pct_BA",
            dte_target=7,
        )
        if "error" not in csp_result:
            csp_metrics = compute_metrics(csp_result["equity_df"], csp_result["trades_df"],
                                          f"CSP_7DTE_{int(ba*100)}pct")
            csp_metrics["n_assignments"] = csp_result.get("n_assignments", 0)
            all_results[f"CSP_7DTE_{int(ba*100)}pct"] = {
                "metrics": csp_metrics, "result": csp_result, "ba": ba, "strategy": "CSP", "dte": 7
            }
        else:
            all_results[f"CSP_7DTE_{int(ba*100)}pct"] = {"error": csp_result.get("error"), "ba": ba}

        # BPS
        bps_result = run_bps(
            prices, iv, macro, earnings_lookup,
            all_avail, ticker_tier, ba,
            label=f"BPS_7DTE_{int(ba*100)}pct_BA",
            dte_target=7,
        )
        if "error" not in bps_result:
            bps_metrics = compute_metrics(bps_result["equity_df"], bps_result["trades_df"],
                                          f"BPS_7DTE_{int(ba*100)}pct")
            all_results[f"BPS_7DTE_{int(ba*100)}pct"] = {
                "metrics": bps_metrics, "result": bps_result, "ba": ba, "strategy": "BPS", "dte": 7
            }
        else:
            all_results[f"BPS_7DTE_{int(ba*100)}pct"] = {"error": bps_result.get("error"), "ba": ba}

    # Print head-to-head comparison
    print("\n" + "=" * 140)
    print("HEAD-TO-HEAD: CSP vs BPS at Each BA Level (7 DTE)")
    print("=" * 140)
    header = f"{'Strategy':<25} {'BA%':>5} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>7} {'PF':>6} {'TradeWR':>8} {'Trades':>7} {'AvgPnL':>9} {'Final$':>12}"
    print(header)
    print("-" * 140)

    for ba in ba_levels:
        for strat in ["CSP", "BPS"]:
            key = f"{strat}_7DTE_{int(ba*100)}pct"
            r = all_results.get(key, {})
            if "error" in r:
                print(f"  {key:<25}: ERROR - {r.get('error')}")
                continue
            m = r["metrics"]
            assign_str = f" [A={m.get('n_assignments', 0)}]" if strat == "CSP" else ""
            print(f"{key:<25} {ba*100:>4.0f}% {m['cagr_pct']:>6.1f}% {m['sharpe']:>7.2f} "
                  f"{m['sortino']:>8.2f} {m['max_dd_pct']:>7.1f}% {m['calmar']:>7.2f} "
                  f"{m['profit_factor']:>6.2f} {m['trade_wr_pct']:>7.1f}% {m['n_trades']:>7d} "
                  f"${m['avg_trade_pnl']:>8.2f} ${m['final_equity']:>11,.0f}{assign_str}")
        print()

    # ═══════════════════════════════════════════════════════════════
    # PART 2: CSP at Longer DTEs (14, 30, 45) at 15% BA
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("PART 2: CSP AT LONGER DTE (14, 30, 45 days) AT 15% BA")
    print("Hypothesis: longer DTE = bigger premiums = more BA tolerance")
    print("=" * 120)

    for dte in [14, 30, 45]:
        csp_result = run_csp(
            prices, iv, macro, earnings_lookup,
            all_avail, ticker_tier, 0.15,
            label=f"CSP_{dte}DTE_15pct_BA",
            dte_target=dte,
        )
        if "error" not in csp_result:
            csp_metrics = compute_metrics(csp_result["equity_df"], csp_result["trades_df"],
                                          f"CSP_{dte}DTE_15pct")
            csp_metrics["n_assignments"] = csp_result.get("n_assignments", 0)
            all_results[f"CSP_{dte}DTE_15pct"] = {
                "metrics": csp_metrics, "result": csp_result, "ba": 0.15, "strategy": "CSP", "dte": dte
            }
        else:
            all_results[f"CSP_{dte}DTE_15pct"] = {"error": csp_result.get("error"), "ba": 0.15}

    print("\n" + "=" * 140)
    print("CSP DTE COMPARISON at 15% BA")
    print("=" * 140)
    print(header)
    print("-" * 140)

    for dte in [7, 14, 30, 45]:
        key = f"CSP_{dte}DTE_15pct"
        r = all_results.get(key, {})
        if "error" in r:
            print(f"  {key}: ERROR - {r.get('error')}")
            continue
        m = r["metrics"]
        print(f"{key:<25} {15:>4.0f}% {m['cagr_pct']:>6.1f}% {m['sharpe']:>7.2f} "
              f"{m['sortino']:>8.2f} {m['max_dd_pct']:>7.1f}% {m['calmar']:>7.2f} "
              f"{m['profit_factor']:>6.2f} {m['trade_wr_pct']:>7.1f}% {m['n_trades']:>7d} "
              f"${m['avg_trade_pnl']:>8.2f} ${m['final_equity']:>11,.0f} [A={m.get('n_assignments', 0)}]")

    # ═══════════════════════════════════════════════════════════════
    # PART 3: CSP Break-Even BA%
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("PART 3: CSP BREAK-EVEN BA% (binary search for Sharpe=0)")
    print("=" * 120)

    csp_breakeven_ba, csp_breakeven_sharpe = find_csp_breakeven_ba(
        prices, iv, macro, earnings_lookup,
        all_avail, ticker_tier, dte_target=7,
    )

    # Also find BPS break-even for comparison
    print("\n  Finding BPS break-even BA% for comparison...")

    def bps_sharpe_at_ba(ba_pct):
        ba_frac = ba_pct / 100.0
        result = run_bps(
            prices, iv, macro, earnings_lookup,
            all_avail, ticker_tier, ba_frac,
            label=f"bps_be_{ba_pct:.1f}pct",
            dte_target=7,
        )
        if "error" in result:
            return -1.0
        eq = result["equity_df"].copy().sort_values("date").reset_index(drop=True)
        rets = eq["equity"].pct_change().dropna()
        if rets.std() <= 0:
            return 0.0
        return float(rets.mean() / rets.std() * np.sqrt(252))

    s_low = bps_sharpe_at_ba(1.0)
    s_high = bps_sharpe_at_ba(40.0)
    print(f"    BPS Sharpe at 1% BA: {s_low:.3f}")
    print(f"    BPS Sharpe at 40% BA: {s_high:.3f}")

    if s_low <= 0:
        bps_breakeven = 1.0
    elif s_high > 0:
        bps_breakeven = 40.0
    else:
        lo, hi = 1.0, 40.0
        for it in range(12):
            mid = (lo + hi) / 2.0
            s_mid = bps_sharpe_at_ba(mid)
            print(f"    Iteration {it+1}: BA={mid:.2f}% -> Sharpe={s_mid:.3f}")
            if s_mid > 0:
                lo = mid
            else:
                hi = mid
            if hi - lo < 0.2:
                break
        bps_breakeven = (lo + hi) / 2.0

    print(f"\n  CSP break-even BA: ~{csp_breakeven_ba:.1f}%")
    print(f"  BPS break-even BA: ~{bps_breakeven:.1f}%")
    print(f"  CSP advantage: {csp_breakeven_ba - bps_breakeven:+.1f} percentage points of BA tolerance")

    # ═══════════════════════════════════════════════════════════════
    # PART 4: Assignment Risk / Crash Analysis
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("PART 4: CSP ASSIGNMENT RISK ANALYSIS")
    print("CSP has unlimited downside on assignment -- how bad does it get?")
    print("=" * 120)

    # Get the 15% BA 7DTE result for detailed analysis
    csp_15_key = "CSP_7DTE_15pct"
    r15 = all_results.get(csp_15_key, {})
    bps_15_key = "BPS_7DTE_15pct"
    r15_bps = all_results.get(bps_15_key, {})

    if "error" not in r15:
        assign_data = r15["result"].get("assignment_losses", [])
        m15 = r15["metrics"]

        print(f"\n  Total trades: {m15['n_trades']}")
        print(f"  Total assignments: {m15.get('n_assignments', 0)}")
        if m15['n_trades'] > 0:
            assign_rate = m15.get('n_assignments', 0) / m15['n_trades'] * 100
            print(f"  Assignment rate: {assign_rate:.1f}%")

        if assign_data:
            losses = [a["total_loss"] for a in assign_data]
            net_pnls = [a["net_pnl_after_premium"] for a in assign_data]
            print(f"\n  Assignment statistics:")
            print(f"    Count:           {len(assign_data)}")
            print(f"    Avg loss:        ${np.mean(losses):>10,.2f}")
            print(f"    Max loss:        ${np.max(losses):>10,.2f}")
            print(f"    Median loss:     ${np.median(losses):>10,.2f}")
            print(f"    Total losses:    ${np.sum(losses):>10,.2f}")
            print(f"    Avg net PnL:     ${np.mean(net_pnls):>10,.2f} (after premium offset)")

            # Worst 5 assignments
            sorted_assigns = sorted(assign_data, key=lambda x: -x["total_loss"])[:5]
            print(f"\n  Worst 5 Assignments:")
            print(f"    {'Date':<12} {'Ticker':<8} {'Strike':>8} {'Stock':>8} {'Loss/sh':>8} {'Ctrs':>5} {'Total$':>12}")
            for a in sorted_assigns:
                print(f"    {a['date']:<12} {a['ticker']:<8} ${a['strike']:>7.2f} ${a['stock_price']:>7.2f} "
                      f"${a['loss_per_share']:>7.2f} {a['contracts']:>5d} ${a['total_loss']:>11,.2f}")

        # Compare MaxDD: CSP vs BPS at same BA
        if "error" not in r15_bps:
            m15_bps = r15_bps["metrics"]
            print(f"\n  Risk Comparison at 15% BA:")
            print(f"    {'Metric':<20} {'CSP':<15} {'BPS':<15} {'Delta':<15}")
            print(f"    {'-'*60}")
            for metric, fmt in [("max_dd_pct", ".1f"), ("sharpe", ".2f"), ("sortino", ".2f"),
                                ("cagr_pct", ".1f"), ("calmar", ".2f")]:
                vc = m15[metric]
                vb = m15_bps[metric]
                delta = vc - vb
                print(f"    {metric:<20} {format(vc, fmt):<15} {format(vb, fmt):<15} {format(delta, '+' + fmt)}")

    # ═══════════════════════════════════════════════════════════════
    # PART 5: Permutation Test on best CSP config
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("PART 5: PERMUTATION TEST ON BEST CSP CONFIG AT REALISTIC BA")
    print("=" * 120)

    # Find best CSP config by Sharpe
    best_csp_key = None
    best_csp_sharpe = -999
    for key, val in all_results.items():
        if key.startswith("CSP") and "error" not in val:
            s = val["metrics"]["sharpe"]
            if s > best_csp_sharpe:
                best_csp_sharpe = s
                best_csp_key = key

    perm_result = None
    if best_csp_key and "error" not in all_results[best_csp_key]:
        best_r = all_results[best_csp_key]
        trades_df = best_r["result"]["trades_df"]
        if not trades_df.empty:
            print(f"  Testing: {best_csp_key} (Sharpe={best_csp_sharpe:.2f})")
            perm_result = run_permutation_test(trades_df, n_permutations=200)
            print(f"  Actual PnL:       ${perm_result['actual_total_pnl']:>11,.0f}")
            print(f"  Actual Sharpe:    {perm_result['actual_sharpe']:>8.4f}")
            print(f"  Random PnL mean:  ${perm_result['random_pnl_mean']:>11,.0f}")
            print(f"  P-value (PnL):    {perm_result['p_value_pnl']:>8.4f}")
            print(f"  P-value (Sharpe): {perm_result['p_value_sharpe']:>8.4f}")
            print(f"  >>> VERDICT: {perm_result['verdict']}")

    # Also test CSP at 15% BA specifically
    if csp_15_key != best_csp_key and "error" not in r15:
        trades_df_15 = r15["result"]["trades_df"]
        if not trades_df_15.empty:
            print(f"\n  Also testing: {csp_15_key} (realistic cost scenario)")
            perm_15 = run_permutation_test(trades_df_15, n_permutations=200)
            print(f"  Actual PnL:       ${perm_15['actual_total_pnl']:>11,.0f}")
            print(f"  P-value (PnL):    {perm_15['p_value_pnl']:>8.4f}")
            print(f"  >>> VERDICT: {perm_15['verdict']}")

    # ═══════════════════════════════════════════════════════════════
    # EXECUTIVE SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("EXECUTIVE SUMMARY: CSP vs BPS REAL-WORLD COST COMPARISON")
    print("=" * 120)

    # Collect CSP and BPS Sharpes at each BA level
    print("\n  Sharpe Ratio Degradation by BA Cost:")
    print(f"  {'BA%':>5}  {'CSP Sharpe':>11}  {'BPS Sharpe':>11}  {'CSP Advantage':>14}")
    print(f"  {'-'*50}")
    for ba in ba_levels:
        csp_key = f"CSP_7DTE_{int(ba*100)}pct"
        bps_key = f"BPS_7DTE_{int(ba*100)}pct"
        csp_s = all_results[csp_key]["metrics"]["sharpe"] if "error" not in all_results.get(csp_key, {"error": True}) else float("nan")
        bps_s = all_results[bps_key]["metrics"]["sharpe"] if "error" not in all_results.get(bps_key, {"error": True}) else float("nan")
        adv = csp_s - bps_s if not (np.isnan(csp_s) or np.isnan(bps_s)) else float("nan")
        print(f"  {ba*100:>4.0f}%  {csp_s:>11.2f}  {bps_s:>11.2f}  {adv:>+13.2f}")

    print(f"\n  Break-Even BA%:")
    print(f"    CSP: ~{csp_breakeven_ba:.1f}%")
    print(f"    BPS: ~{bps_breakeven:.1f}%")
    print(f"    CSP extra tolerance: {csp_breakeven_ba - bps_breakeven:+.1f} percentage points")

    # Final verdict
    csp_15_sharpe = all_results.get("CSP_7DTE_15pct", {}).get("metrics", {}).get("sharpe", float("nan"))
    bps_15_sharpe = all_results.get("BPS_7DTE_15pct", {}).get("metrics", {}).get("sharpe", float("nan"))

    print(f"\n  AT REALISTIC 15% BA COST:")
    print(f"    CSP Sharpe: {csp_15_sharpe:.2f}")
    print(f"    BPS Sharpe: {bps_15_sharpe:.2f}")

    if csp_15_sharpe >= 1.0:
        print(f"\n  VERDICT: CSP SURVIVES realistic costs (Sharpe {csp_15_sharpe:.2f} >= 1.0).")
        print(f"  The single-leg cost advantage is REAL and makes CSP viable where BPS failed.")
    elif csp_15_sharpe >= 0.5:
        print(f"\n  VERDICT: CSP is MARGINAL at realistic costs (Sharpe {csp_15_sharpe:.2f}).")
        print(f"  Better than BPS but still needs excellent execution.")
    elif csp_15_sharpe > 0:
        print(f"\n  VERDICT: CSP is BARELY POSITIVE at realistic costs (Sharpe {csp_15_sharpe:.2f}).")
        print(f"  Marginally better than BPS, but not tradeable with confidence.")
    else:
        print(f"\n  VERDICT: CSP ALSO FAILS at realistic costs (Sharpe {csp_15_sharpe:.2f}).")
        print(f"  Even with one-leg advantage, the edge is too thin to survive real BA costs.")

    if not np.isnan(csp_15_sharpe) and not np.isnan(bps_15_sharpe):
        if csp_15_sharpe > bps_15_sharpe:
            print(f"\n  CSP has {csp_15_sharpe - bps_15_sharpe:.2f} Sharpe advantage over BPS at 15% BA.")
            print(f"  This confirms the single-leg cost model gives CSP a structural edge on costs.")
        else:
            print(f"\n  CSP has NO advantage over BPS despite fewer legs.")
            print(f"  Assignment risk likely offsets the cost savings.")

    # Assignment risk warning
    csp_15_data = all_results.get("CSP_7DTE_15pct", {})
    if "error" not in csp_15_data:
        csp_15_m = csp_15_data["metrics"]
        bps_15_data = all_results.get("BPS_7DTE_15pct", {})
        if "error" not in bps_15_data:
            bps_15_m = bps_15_data["metrics"]
            print(f"\n  RISK COMPARISON:")
            print(f"    CSP MaxDD: {csp_15_m['max_dd_pct']:.1f}% (unlimited downside)")
            print(f"    BPS MaxDD: {bps_15_m['max_dd_pct']:.1f}% (capped by spread width)")
            if abs(csp_15_m["max_dd_pct"]) > abs(bps_15_m["max_dd_pct"]) * 1.5:
                print(f"    WARNING: CSP drawdowns are significantly worse than BPS.")
                print(f"    The cost savings may not justify the extra tail risk.")

    # ═══════════════════════════════════════════════════════════════
    # SAVE ALL RESULTS
    # ═══════════════════════════════════════════════════════════════

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        elif isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    save_data = {
        "generated": pd.Timestamp.now().isoformat(),
        "purpose": "CSP vs BPS real-world cost comparison -- does single-leg CSP survive where two-leg BPS failed?",
        "starting_capital": STARTING_CAPITAL,
        "cost_model": {
            "CSP": "1 leg, 1 BA crossing: seller gets bid = mid * (1 - ba/2). Commission: $0.65/contract",
            "BPS": "2 legs, 2 BA crossings: sell short at bid, buy long at ask. Commission: $0.65/contract * 2",
        },
        "csp_breakeven_ba_pct": round(csp_breakeven_ba, 1),
        "bps_breakeven_ba_pct": round(bps_breakeven, 1),
        "scenarios": {},
    }

    for key, val in all_results.items():
        if "error" in val:
            save_data["scenarios"][key] = {"error": val.get("error", "unknown")}
        else:
            save_data["scenarios"][key] = {
                "ba_pct": val["ba"] * 100,
                "strategy": val.get("strategy", "unknown"),
                "dte": val.get("dte", 7),
                "metrics": val["metrics"],
            }

    if perm_result:
        save_data["permutation_test"] = {
            "config": best_csp_key,
            "results": convert(perm_result),
        }

    save_data = convert(save_data)

    with open(OUTPUT / "csp_vs_bps_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    # Save equity curves for key scenarios
    for key in ["CSP_7DTE_15pct", "BPS_7DTE_15pct", "CSP_7DTE_5pct", "CSP_30DTE_15pct", "CSP_45DTE_15pct"]:
        r = all_results.get(key, {})
        if "error" not in r:
            r["result"]["equity_df"][["date", "equity"]].to_parquet(
                OUTPUT / f"eq_{key}.parquet", index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*120}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Results saved to {OUTPUT}")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
