#!/usr/bin/env python3
"""
BPS Viable Universe Backtest — Realistic BA Filtering
=======================================================
Tests whether the BPS strategy remains viable when restricted to ONLY
tickers that survive real-world bid-ask spread filtering.

From fresh chain analysis (2026-07-09):
  - 61 tickers scanned
  - 13 GREEN (BA% <= 15% on short leg, net_credit > $1.00)
  - 40 YELLOW (marginal)
  - 8 RED (negative net_credit or extreme BA%)

Key question: Does a smaller universe of ~25 liquid names still produce
an edge, or does the strategy need 98 tickers for diversification?

Configs compared:
  A) Full 98 tickers at 5% BA (original, overly optimistic)
  B) Full 98 tickers at 15% BA (prior realistic test — failed)
  C) GREEN-only ~25 tickers at 12% BA (realistic — avg GREEN BA%)
  D) GREEN-only ~25 tickers at 15% BA (pessimistic stress test)

All configs use full risk stack: dynamic delta, VIX-scale, 2% CB,
VIX cutoff 30, earnings filter.

Output: output/bps_viable_universe/
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
OUTPUT = ROOT / "output" / "bps_viable_universe"
OUTPUT.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100_000
DTE_TARGET = 10

# ═══════════════════════════════════════════════════════════════════
# Ticker Universe Definitions
# ═══════════════════════════════════════════════════════════════════

# Full universe (98 tickers from original study)
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

# GREEN tickers — passed real-world BA filter (short_ba_pct <= ~15%, net_credit > $1.00)
# From fresh_chain_analysis.csv 2026-07-09
GREEN_CORE = [
    'AMD', 'AVGO', 'CAT', 'COIN', 'CRWD', 'DDOG', 'GS',
    'INTC', 'LLY', 'META', 'SNOW', 'TSLA', 'UNH',
]

# Extended viable: YELLOW tickers with short_ba_pct <= 15% AND net_credit > $0.50
# These are liquid enough to trade with reasonable fills
# From the chain data: AAPL(7.2%), AMZN(2.0%), NVDA(1.1%), GOOGL(8.1%),
# MSFT(12.1%), NFLX(2.4%), ORCL(6.3%), SHOP(9.8%), BABA(5.1%),
# MARA(5.7%), RIOT(6.6%), CRM(9.9%), UBER(14.3%)
GREEN_EXTENDED = [
    'AMD', 'AVGO', 'CAT', 'COIN', 'CRWD', 'DDOG', 'GS',
    'INTC', 'LLY', 'META', 'SNOW', 'TSLA', 'UNH',
    # Extended from YELLOW with tight BA on short leg
    'AAPL', 'AMZN', 'NVDA', 'GOOGL', 'MSFT', 'NFLX', 'ORCL',
    'SHOP', 'BABA', 'CRM', 'UBER',
    # Crypto/high-IV names (tight BA, decent premium)
    'MARA', 'RIOT',
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

COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03

def trade_cost(premium, contracts):
    slip = max(SLIPPAGE_MIN, SLIPPAGE_FRAC * premium) * 100 * contracts if premium > 0 else 0
    comm = COST_PER_CONTRACT * contracts
    return slip + comm


# ═══════════════════════════════════════════════════════════════════
# Data Loading
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
        tk_px["sigma_atm_30d"] = tk_px["rv_20"] * 1.10
        tk_px["iv_high_252"] = tk_px["sigma"].rolling(252).max()
        tk_px["iv_low_252"] = tk_px["sigma"].rolling(252).min()
        iv_range = tk_px["iv_high_252"] - tk_px["iv_low_252"]
        tk_px["iv_rank"] = np.where(iv_range > 0.001,
            (tk_px["sigma"] - tk_px["iv_low_252"]) / iv_range, 0.5)
        tk_px["sigma_rv"] = tk_px["rv_20"]
        tk_px["iv_rv_ratio"] = np.where(tk_px["rv_20"] > 0.001, tk_px["sigma"] / tk_px["rv_20"], 1.15)
        tk_px["term_ratio"] = np.where(tk_px["rv_20"] > 0.001,
            tk_px["rv_60"].fillna(tk_px["rv_20"]) / tk_px["rv_20"], 1.0)
        tk_px["term_proxy"] = tk_px["term_ratio"]
        tk_px["r_1m"] = tk_px["close"].pct_change(21)
        tk_px["pricing_source"] = "modeled_rv"
        tk_px["ticker"] = tk
        cols = ["date", "ticker", "sigma_rv", "iv_rv_ratio", "term_ratio",
                "term_proxy", "sigma", "sigma_atm_30d", "iv_rank", "r_1m", "pricing_source"]
        frame = tk_px.dropna(subset=["sigma"])[cols]
        frames.append(frame)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


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

    all_needed = set(TIER1_TICKERS + TIER2_TICKERS + GREEN_EXTENDED)
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
    print(f"  Prices: {prices.shape[0]} rows, {prices['ticker'].nunique()} tickers")
    print(f"  IV: {iv.shape[0]} rows, {iv['ticker'].nunique()} tickers")
    print(f"  Available from full universe: {len(available)}")

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
# Dynamic Delta
# ═══════════════════════════════════════════════════════════════════

def dynamic_aggressive(vix):
    if vix < 15:
        return 0.35, 15.0
    elif vix < 22:
        return 0.30, 15.0
    else:
        return 0.25, 15.0


def fixed_d30(vix):
    return 0.30, 15.0


# ═══════════════════════════════════════════════════════════════════
# BPS Backtest Engine
# ═══════════════════════════════════════════════════════════════════

def run_bps(prices, iv, macro, fund, universe, earnings_lookup,
            ticker_list, ba_flat_pct,
            delta_fn, label="test",
            dte_target=DTE_TARGET, profit_take=0.65,
            margin_cap=0.25, max_concurrent=40,
            per_name_pct=0.03,
            vix_scale=False, vix_base=15.0,
            vix_hard_cutoff=None,
            portfolio_cb_threshold=None, portfolio_cb_freeze_days=1,
            earnings_filter=False, earnings_buffer_days=7):
    """
    BPS backtest engine with FLAT BA cost applied uniformly.
    ba_flat_pct: e.g. 0.05 = 5%, 0.12 = 12%, 0.15 = 15%
    """
    print(f"  Running: {label} ({len(ticker_list)} tickers, {ba_flat_pct*100:.0f}% BA)...")

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

    macro_by_date = macro.set_index("date").to_dict("index")
    all_dates = sorted(prices_df["date"].unique())

    cash = STARTING_CAPITAL
    positions = {}
    equity_curve = []
    detailed_trades = []

    frozen_until = None
    cb_trigger_count = 0
    n_earnings_blocked = 0
    n_vix_blocked = 0
    n_cb_frozen = 0

    n_available_in_run = len(available)

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

            if S < pos.get("min_price", float("inf")):
                pos["min_price"] = S

            # Close 1 DTE
            if T_days == 1:
                short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
                long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
                cost_to_close = (short_val - long_val) * 100 * pos["contracts"]
                close_fees = trade_cost(short_val, pos["contracts"]) + trade_cost(long_val, pos["contracts"])
                ba_close = abs(short_val - long_val) * 100 * pos["contracts"] * ba_flat_pct * 2

                realized = pos["net_credit"] - cost_to_close - close_fees - ba_close
                cash -= cost_to_close + close_fees + ba_close

                distance_to_short = (S - pos["short_strike"]) / pos["short_strike"]
                status = "safe" if distance_to_short > 0.02 else ("pin_risk" if distance_to_short > -0.02 else "breached")

                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "exit_type": "early_close_1DTE",
                    "short_strike": pos["short_strike"], "long_strike": pos["long_strike"],
                    "open_price": pos.get("open_stock_price", 0), "close_price": S,
                    "net_credit": pos["net_credit"], "realized_pnl": realized,
                    "contracts": pos["contracts"], "status_at_close": status,
                    "spread_width": pos["short_strike"] - pos["long_strike"],
                    "max_loss": (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"],
                    "days_held": (dt - pos["open_date"]).days,
                    "put_delta_used": pos.get("put_delta_used", 0.30),
                    "open_vix": pos.get("open_vix", np.nan),
                    "vix_scalar": pos.get("vix_scalar", 1.0),
                })
                to_remove.append(tk)
                continue

            # Expiry
            if T_days <= 0:
                short_itm = S < pos["short_strike"]
                long_itm = S < pos["long_strike"]
                close_cost = COST_PER_CONTRACT * 2 * pos["contracts"]

                if not short_itm:
                    realized = pos["net_credit"] - close_cost
                    cash -= close_cost
                    status = "OTM_safe"
                elif short_itm and not long_itm:
                    loss = (pos["short_strike"] - S) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost
                    status = "partial_breach"
                else:
                    loss = (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost
                    status = "full_breach"

                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "exit_type": "expiry",
                    "short_strike": pos["short_strike"], "long_strike": pos["long_strike"],
                    "open_price": pos.get("open_stock_price", 0), "close_price": S,
                    "net_credit": pos["net_credit"], "realized_pnl": realized,
                    "contracts": pos["contracts"], "status_at_close": status,
                    "spread_width": pos["short_strike"] - pos["long_strike"],
                    "max_loss": (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"],
                    "days_held": (dt - pos["open_date"]).days,
                    "put_delta_used": pos.get("put_delta_used", 0.30),
                    "open_vix": pos.get("open_vix", np.nan),
                    "vix_scalar": pos.get("vix_scalar", 1.0),
                })
                to_remove.append(tk)
                continue

            # Profit take check
            short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
            long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
            spread_val = (short_val - long_val) * 100 * pos["contracts"]
            initial_credit = pos["net_credit"]
            cost_to_close = spread_val + trade_cost(short_val, pos["contracts"]) + trade_cost(long_val, pos["contracts"])
            ba_close = abs(short_val - long_val) * 100 * pos["contracts"] * ba_flat_pct * 2

            captured = (initial_credit - cost_to_close - ba_close) / max(initial_credit, 1e-6)
            if captured >= profit_take:
                realized = initial_credit - cost_to_close - ba_close
                cash -= cost_to_close + ba_close

                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "exit_type": "profit_take",
                    "short_strike": pos["short_strike"], "long_strike": pos["long_strike"],
                    "open_price": pos.get("open_stock_price", 0), "close_price": S,
                    "net_credit": pos["net_credit"], "realized_pnl": realized,
                    "contracts": pos["contracts"], "status_at_close": "profit_take",
                    "spread_width": pos["short_strike"] - pos["long_strike"],
                    "max_loss": (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"],
                    "days_held": (dt - pos["open_date"]).days,
                    "put_delta_used": pos.get("put_delta_used", 0.30),
                    "open_vix": pos.get("open_vix", np.nan),
                    "vix_scalar": pos.get("vix_scalar", 1.0),
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

        # ── Get delta ──
        if np.isnan(vix_val):
            put_delta, spread_width = 0.30, 15.0
        else:
            put_delta, spread_width = delta_fn(vix_val)

        # ── Open new positions ──
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
            net_prem_per_share = prem_short - prem_long

            if net_prem_per_share <= 0.05:
                continue

            margin_per_contract = spread_width * 100
            max_alloc = per_name_pct * equity
            n_contracts = max(1, int(max_alloc // margin_per_contract))
            n_contracts = max(1, int(n_contracts * vix_scalar))

            if margin_per_contract * n_contracts > remaining_margin:
                n_contracts = max(1, int(remaining_margin // margin_per_contract))
            if n_contracts < 1:
                continue

            net_credit = net_prem_per_share * 100 * n_contracts
            open_costs = trade_cost(prem_short, n_contracts) + trade_cost(prem_long, n_contracts)
            ba_open = net_prem_per_share * 100 * n_contracts * ba_flat_pct * 2

            net_credit -= open_costs + ba_open

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
                "min_price": S,
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
        "n_tickers_available": n_available_in_run,
        "n_earnings_blocked": n_earnings_blocked,
        "n_vix_blocked": n_vix_blocked,
        "n_cb_frozen": n_cb_frozen,
        "cb_triggers": cb_trigger_count,
    }


# ═══════════════════════════════════════════════════════════════════
# Metrics Computation
# ═══════════════════════════════════════════════════════════════════

def compute_full_metrics(eq_df, trades_df, label, starting_cap=STARTING_CAPITAL):
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
    max_dd_idx = eq["dd"].idxmin()
    max_dd_date = eq.loc[max_dd_idx, "date"]

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    wr = float(len(rets[rets > 0]) / len(rets) * 100) if len(rets) > 0 else 0.0
    wins = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = float(wins / losses) if losses > 0 else float("inf")

    daily_pnl = eq["equity"].diff().dropna()
    pnl_stats = {
        "mean": round(float(daily_pnl.mean()), 2),
        "std": round(float(daily_pnl.std()), 2),
        "skew": round(float(daily_pnl.skew()), 3),
        "kurtosis": round(float(daily_pnl.kurtosis()), 3),
        "median": round(float(daily_pnl.median()), 2),
        "p5": round(float(daily_pnl.quantile(0.05)), 2),
        "p95": round(float(daily_pnl.quantile(0.95)), 2),
    }

    # Worst drawdown
    trough_eq = eq.loc[max_dd_idx, "equity"]
    pre_peak = eq.loc[max_dd_idx, "peak"]
    peak_dates = eq[eq["equity"] == pre_peak]
    peak_start = peak_dates.iloc[0]["date"] if len(peak_dates) > 0 else eq["date"].iloc[0]
    post_trough = eq.loc[max_dd_idx:]
    recovered = post_trough[post_trough["equity"] >= pre_peak]
    if len(recovered) > 0:
        recovery_date = recovered.iloc[0]["date"]
        recovery_days = (recovery_date - max_dd_date).days
    else:
        recovery_date = None
        recovery_days = None

    worst_dd_episode = {
        "peak_date": str(peak_start.date()) if hasattr(peak_start, 'date') else str(peak_start),
        "trough_date": str(max_dd_date.date()) if hasattr(max_dd_date, 'date') else str(max_dd_date),
        "max_dd_pct": round(max_dd * 100, 2),
        "peak_equity": round(float(pre_peak), 2),
        "trough_equity": round(float(trough_eq), 2),
        "loss_dollars": round(float(pre_peak - trough_eq), 2),
        "recovery_date": str(recovery_date.date()) if recovery_date is not None and hasattr(recovery_date, 'date') else str(recovery_date),
        "recovery_days": recovery_days,
        "days_peak_to_trough": (max_dd_date - peak_start).days if hasattr(max_dd_date, 'date') else 0,
    }

    # Per-year
    eq["year"] = eq["date"].dt.year
    per_year = {}
    for yr, grp in eq.groupby("year"):
        if len(grp) < 5:
            continue
        yr_ret = (grp["equity"].iloc[-1] / grp["equity"].iloc[0] - 1)
        yr_rets = grp["ret"].dropna()
        yr_sharpe = float(yr_rets.mean() / yr_rets.std() * np.sqrt(252)) if yr_rets.std() > 0 else 0.0
        yr_down = yr_rets[yr_rets < 0]
        yr_sortino = float(yr_rets.mean() / yr_down.std() * np.sqrt(252)) if len(yr_down) > 3 and yr_down.std() > 0 else 0.0
        yr_dd = float(((grp["equity"] / grp["equity"].cummax()) - 1).min())
        yr_wr = float(len(yr_rets[yr_rets > 0]) / len(yr_rets) * 100) if len(yr_rets) > 0 else 0.0
        per_year[int(yr)] = {
            "return_pct": round(yr_ret * 100, 2),
            "sharpe": round(yr_sharpe, 2),
            "sortino": round(yr_sortino, 2),
            "max_dd_pct": round(yr_dd * 100, 2),
            "daily_wr_pct": round(yr_wr, 1),
        }

    # Monthly returns
    eq["month"] = eq["date"].dt.month
    monthly_returns = {}
    for (yr, mo), grp in eq.groupby(["year", "month"]):
        if len(grp) < 3:
            continue
        mo_ret = (grp["equity"].iloc[-1] / grp["equity"].iloc[0] - 1) * 100
        monthly_returns[f"{int(yr)}-{int(mo):02d}"] = round(mo_ret, 2)

    # Trade metrics
    n_trades = len(trades_df)
    if n_trades > 0:
        trade_wr = float((trades_df["realized_pnl"] > 0).sum() / n_trades * 100)
        avg_credit = float(trades_df["net_credit"].mean())
        breach_trades = trades_df[trades_df["status_at_close"].isin(
            ["breached", "partial_breach", "full_breach"])]
        breach_rate = float(len(breach_trades) / n_trades * 100)
        avg_pnl = float(trades_df["realized_pnl"].mean())
        total_pnl = float(trades_df["realized_pnl"].sum())
        trades_df_tmp = trades_df.copy()
        trades_df_tmp["year"] = pd.to_datetime(trades_df_tmp["close_date"]).dt.year
        trades_per_year = trades_df_tmp.groupby("year").size().to_dict()

        # Per-ticker breakdown
        ticker_stats = {}
        for tk, grp in trades_df.groupby("ticker"):
            tk_pnl = float(grp["realized_pnl"].sum())
            tk_wr = float((grp["realized_pnl"] > 0).sum() / len(grp) * 100)
            ticker_stats[tk] = {"trades": len(grp), "total_pnl": round(tk_pnl, 2), "wr_pct": round(tk_wr, 1)}
    else:
        trade_wr = 0.0
        avg_credit = 0.0
        breach_rate = 0.0
        avg_pnl = 0.0
        total_pnl = 0.0
        trades_per_year = {}
        ticker_stats = {}

    return {
        "label": label,
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 2),
        "daily_wr_pct": round(wr, 1),
        "profit_factor": round(pf, 2),
        "final_equity": round(float(eq["equity"].iloc[-1]), 2),
        "total_return_pct": round((total_return - 1) * 100, 2),
        "n_trades": n_trades,
        "trade_wr_pct": round(trade_wr, 1),
        "avg_premium_collected": round(avg_credit, 2),
        "avg_trade_pnl": round(avg_pnl, 2),
        "total_trade_pnl": round(total_pnl, 2),
        "breach_rate_pct": round(breach_rate, 2),
        "trades_per_year": {int(k): int(v) for k, v in trades_per_year.items()},
        "daily_pnl_distribution": pnl_stats,
        "worst_dd_episode": worst_dd_episode,
        "per_year": per_year,
        "monthly_returns": monthly_returns,
        "ticker_stats": ticker_stats,
    }


# ═══════════════════════════════════════════════════════════════════
# Permutation Test
# ═══════════════════════════════════════════════════════════════════

def run_permutation_test(trades_df, n_permutations=200):
    """Shuffle trade P&L signs. If random is also profitable, edge is suspect."""
    print(f"\n  Running permutation test ({n_permutations} shuffles)...")

    if trades_df.empty:
        return {"pass": False, "reason": "no trades"}

    actual_pnl = float(trades_df["realized_pnl"].sum())
    trades_tmp = trades_df.copy()
    trades_tmp["close_date"] = pd.to_datetime(trades_tmp["close_date"])
    daily_actual = trades_tmp.groupby("close_date")["realized_pnl"].sum()
    actual_sharpe = 0.0
    if len(daily_actual) > 10 and daily_actual.std() > 0:
        actual_sharpe = float(daily_actual.mean() / daily_actual.std() * np.sqrt(252))

    random_pnls = []
    random_sharpes = []
    rng = np.random.RandomState(42)

    for i in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(trades_df))
        shuffled_pnl = trades_df["realized_pnl"].values * signs
        random_pnls.append(float(shuffled_pnl.sum()))

        shuffled_df = trades_df.copy()
        shuffled_df["realized_pnl"] = shuffled_pnl
        shuffled_df["close_date"] = pd.to_datetime(shuffled_df["close_date"])
        daily_shuffled = shuffled_df.groupby("close_date")["realized_pnl"].sum()
        if len(daily_shuffled) > 10 and daily_shuffled.std() > 0:
            random_sharpes.append(float(daily_shuffled.mean() / daily_shuffled.std() * np.sqrt(252)))
        else:
            random_sharpes.append(0.0)

    random_pnls = np.array(random_pnls)
    random_sharpes = np.array(random_sharpes)

    p_value_pnl = float(np.mean(random_pnls >= actual_pnl))
    p_value_sharpe = float(np.mean(random_sharpes >= actual_sharpe))

    passed = p_value_pnl < 0.05 and p_value_sharpe < 0.10

    return {
        "pass": passed,
        "actual_total_pnl": round(actual_pnl, 2),
        "actual_sharpe": round(actual_sharpe, 2),
        "random_pnl_mean": round(float(random_pnls.mean()), 2),
        "random_pnl_std": round(float(random_pnls.std()), 2),
        "random_sharpe_mean": round(float(random_sharpes.mean()), 2),
        "random_sharpe_std": round(float(random_sharpes.std()), 2),
        "p_value_pnl": round(p_value_pnl, 4),
        "p_value_sharpe": round(p_value_sharpe, 4),
        "n_random_profitable": int(np.sum(random_pnls > 0)),
        "n_permutations": n_permutations,
        "verdict": "PASS — edge is REAL (random loses money)" if passed
                   else "FAIL — random also profitable, may be ARTIFACT",
    }


# ═══════════════════════════════════════════════════════════════════
# Concentration Risk Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_concentration(trades_df, label):
    """Analyze single-name and sector concentration risk."""
    if trades_df.empty:
        return {}

    # Single-name concentration
    tk_pnl = trades_df.groupby("ticker")["realized_pnl"].sum().sort_values(ascending=False)
    total_pnl = tk_pnl.sum()
    if total_pnl > 0:
        top1_pct = float(tk_pnl.iloc[0] / total_pnl * 100) if tk_pnl.iloc[0] > 0 else 0.0
        top3_pct = float(tk_pnl.iloc[:3].sum() / total_pnl * 100) if len(tk_pnl) >= 3 else top1_pct
        top5_pct = float(tk_pnl.iloc[:5].sum() / total_pnl * 100) if len(tk_pnl) >= 5 else top3_pct
    else:
        top1_pct = top3_pct = top5_pct = 0.0

    # Trades per ticker
    trades_per_tk = trades_df.groupby("ticker").size()
    hhi = float(((trades_per_tk / trades_per_tk.sum()) ** 2).sum())

    return {
        "n_unique_tickers_traded": int(trades_df["ticker"].nunique()),
        "top1_pnl_pct": round(top1_pct, 1),
        "top3_pnl_pct": round(top3_pct, 1),
        "top5_pnl_pct": round(top5_pct, 1),
        "top_ticker": str(tk_pnl.index[0]) if len(tk_pnl) > 0 else "none",
        "top_ticker_pnl": round(float(tk_pnl.iloc[0]), 2) if len(tk_pnl) > 0 else 0,
        "hhi_trades": round(hhi, 4),
        "hhi_interpretation": "high concentration" if hhi > 0.15 else ("moderate" if hhi > 0.06 else "well diversified"),
        "avg_trades_per_ticker": round(float(trades_per_tk.mean()), 1),
    }


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 90)
    print("BPS VIABLE UNIVERSE BACKTEST")
    print("Testing: does the strategy survive with ONLY BA-viable tickers?")
    print("=" * 90)

    prices, iv, macro, fund, universe, earnings = load_all_data()
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])
    earnings_lookup = build_earnings_lookup(earnings)

    available = set(prices["ticker"].unique()) & set(iv["ticker"].unique())
    all_98 = [t for t in (TIER1_TICKERS + TIER2_TICKERS) if t in available]
    green_ext = [t for t in GREEN_EXTENDED if t in available]
    green_core_only = [t for t in GREEN_CORE if t in available]

    print(f"\n  Full universe available:     {len(all_98)} tickers")
    print(f"  GREEN extended available:    {len(green_ext)} tickers ({', '.join(green_ext)})")
    print(f"  GREEN core only available:   {len(green_core_only)} tickers ({', '.join(green_core_only)})")

    # ═══════════════════════════════════════════════════════════
    # Common risk stack for ALL configs
    # ═══════════════════════════════════════════════════════════
    RISK_PARAMS = dict(
        delta_fn=dynamic_aggressive,
        profit_take=0.65, margin_cap=0.25,
        vix_scale=True, vix_base=15.0,
        vix_hard_cutoff=30,
        portfolio_cb_threshold=-0.02, portfolio_cb_freeze_days=1,
        earnings_filter=True, earnings_buffer_days=7,
    )

    configs = [
        # A: Full 98 at 5% BA (overly optimistic baseline)
        {"ticker_list": all_98, "ba_flat_pct": 0.05,
         "label": "A_full98_5pct_BA", **RISK_PARAMS},

        # B: Full 98 at 15% BA (realistic across all — expected to fail)
        {"ticker_list": all_98, "ba_flat_pct": 0.15,
         "label": "B_full98_15pct_BA", **RISK_PARAMS},

        # C: GREEN extended (~26 tickers) at 12% BA (realistic avg)
        {"ticker_list": green_ext, "ba_flat_pct": 0.12,
         "label": "C_green26_12pct_BA", **RISK_PARAMS},

        # D: GREEN extended (~26 tickers) at 15% BA (pessimistic stress)
        {"ticker_list": green_ext, "ba_flat_pct": 0.15,
         "label": "D_green26_15pct_BA", **RISK_PARAMS},

        # E: GREEN core only (13 tickers) at 12% BA — worst-case diversification
        {"ticker_list": green_core_only, "ba_flat_pct": 0.12,
         "label": "E_green13_12pct_BA", **RISK_PARAMS},
    ]

    all_results = {}

    for cfg in configs:
        print(f"\n{'='*90}")
        result = run_bps(prices, iv, macro, fund, universe, earnings_lookup, **cfg)
        name = cfg["label"]

        if "error" in result:
            print(f"  {name}: ERROR - {result['error']}")
            all_results[name] = {"error": result["error"]}
            continue

        metrics = compute_full_metrics(result["equity_df"], result["trades_df"], name)
        concentration = analyze_concentration(result["trades_df"], name)
        metrics["concentration"] = concentration
        metrics["n_tickers_available"] = result.get("n_tickers_available", 0)
        metrics["n_earnings_blocked"] = result.get("n_earnings_blocked", 0)
        metrics["n_vix_blocked_days"] = result.get("n_vix_blocked", 0)
        metrics["n_cb_frozen_days"] = result.get("n_cb_frozen", 0)
        metrics["cb_triggers"] = result.get("cb_triggers", 0)

        all_results[name] = {
            "metrics": metrics,
            "equity_df": result["equity_df"],
            "trades_df": result["trades_df"],
        }

        result["equity_df"][["date", "equity"]].to_parquet(OUTPUT / f"eq_{name}.parquet", index=False)
        if not result["trades_df"].empty:
            result["trades_df"].to_parquet(OUTPUT / f"trades_{name}.parquet", index=False)

    # ═══════════════════════════════════════════════════════════
    # Comparison Table
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 140)
    print("COMPARISON TABLE — VIABLE UNIVERSE BACKTEST")
    print("=" * 140)
    header = (f"{'Config':<25} {'Tickers':>7} {'BA%':>5} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} "
              f"{'MaxDD':>8} {'Calmar':>7} {'PF':>6} {'TradeWR':>8} {'Trades':>7} {'Final$':>12}")
    print(header)
    print("-" * 140)

    for name in [c["label"] for c in configs]:
        if name not in all_results or "error" in all_results[name]:
            print(f"  {name}: ERROR")
            continue
        m = all_results[name]["metrics"]
        n_tk = m.get("n_tickers_available", 0)
        ba_str = name.split("_")[-1].replace("BA", "")
        print(f"{name:<25} {n_tk:>7} {ba_str:>5} {m['cagr_pct']:>6.1f}% {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
              f"{m['max_dd_pct']:>7.1f}% {m['calmar']:>7.2f} {m['profit_factor']:>6.2f} "
              f"{m['trade_wr_pct']:>7.1f}% {m['n_trades']:>7d} ${m['final_equity']:>11,.0f}")

    # ═══════════════════════════════════════════════════════════
    # Detailed Report for GREEN configs
    # ═══════════════════════════════════════════════════════════
    for target in ["C_green26_12pct_BA", "D_green26_15pct_BA", "E_green13_12pct_BA"]:
        if target not in all_results or "error" in all_results[target]:
            continue
        m = all_results[target]["metrics"]
        trades_df = all_results[target]["trades_df"]

        print(f"\n{'='*90}")
        print(f"DETAILED REPORT: {target}")
        print(f"{'='*90}")

        print(f"\n  -- Core Metrics --")
        print(f"    CAGR:              {m['cagr_pct']:>8.2f}%")
        print(f"    Sharpe:            {m['sharpe']:>8.2f}")
        print(f"    Sortino:           {m['sortino']:>8.2f}")
        print(f"    Calmar:            {m['calmar']:>8.2f}")
        print(f"    Max Drawdown:      {m['max_dd_pct']:>8.2f}%")
        print(f"    Daily Win Rate:    {m['daily_wr_pct']:>8.1f}%")
        print(f"    Profit Factor:     {m['profit_factor']:>8.2f}")
        print(f"    Final Equity:      ${m['final_equity']:>11,.0f}")

        print(f"\n  -- Trade Stats --")
        print(f"    Total Trades:       {m['n_trades']:>7d}")
        print(f"    Trade Win Rate:     {m['trade_wr_pct']:>7.1f}%")
        print(f"    Avg Premium:        ${m['avg_premium_collected']:>8.2f}")
        print(f"    Avg Trade P&L:      ${m['avg_trade_pnl']:>8.2f}")
        print(f"    Breach Rate:        {m['breach_rate_pct']:>7.2f}%")

        print(f"\n  -- Concentration Risk --")
        conc = m.get("concentration", {})
        print(f"    Unique tickers traded: {conc.get('n_unique_tickers_traded', 0)}")
        print(f"    Top ticker:            {conc.get('top_ticker', 'N/A')} (${conc.get('top_ticker_pnl', 0):,.0f})")
        print(f"    Top 1 P&L share:       {conc.get('top1_pnl_pct', 0):.1f}%")
        print(f"    Top 3 P&L share:       {conc.get('top3_pnl_pct', 0):.1f}%")
        print(f"    Top 5 P&L share:       {conc.get('top5_pnl_pct', 0):.1f}%")
        print(f"    HHI (trades):          {conc.get('hhi_trades', 0):.4f} ({conc.get('hhi_interpretation', '')})")

        print(f"\n  -- Per-Year Breakdown --")
        print(f"    {'Year':>6} {'Return':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'DailyWR':>8}")
        print(f"    {'-'*48}")
        for yr in sorted(m["per_year"].keys()):
            y = m["per_year"][yr]
            print(f"    {yr:>6} {y['return_pct']:>7.1f}% {y['sharpe']:>8.2f} {y['sortino']:>8.2f} "
                  f"{y['max_dd_pct']:>7.1f}% {y['daily_wr_pct']:>7.1f}%")

        # Per-ticker P&L table
        if m.get("ticker_stats"):
            print(f"\n  -- Per-Ticker P&L --")
            sorted_tks = sorted(m["ticker_stats"].items(), key=lambda x: -x[1]["total_pnl"])
            print(f"    {'Ticker':<8} {'Trades':>7} {'Total P&L':>12} {'WR%':>6}")
            print(f"    {'-'*35}")
            for tk, st in sorted_tks:
                print(f"    {tk:<8} {st['trades']:>7} ${st['total_pnl']:>10,.0f} {st['wr_pct']:>5.1f}%")

    # ═══════════════════════════════════════════════════════════
    # Permutation Tests on GREEN configs
    # ═══════════════════════════════════════════════════════════
    perm_results = {}
    for target in ["C_green26_12pct_BA", "D_green26_15pct_BA"]:
        if target not in all_results or "error" in all_results[target]:
            continue
        trades_df = all_results[target]["trades_df"]
        if trades_df.empty:
            continue

        print(f"\n{'='*90}")
        print(f"PERMUTATION TEST: {target}")
        print(f"{'='*90}")

        perm = run_permutation_test(trades_df, n_permutations=200)
        perm_results[target] = perm

        print(f"\n  Actual total P&L:    ${perm['actual_total_pnl']:>11,.0f}")
        print(f"  Actual Sharpe:       {perm['actual_sharpe']:>8.2f}")
        print(f"  Random P&L mean:     ${perm['random_pnl_mean']:>11,.0f}")
        print(f"  Random P&L std:      ${perm['random_pnl_std']:>11,.0f}")
        print(f"  Random Sharpe mean:  {perm['random_sharpe_mean']:>8.2f}")
        print(f"  P-value (PnL):       {perm['p_value_pnl']:>8.4f}")
        print(f"  P-value (Sharpe):    {perm['p_value_sharpe']:>8.4f}")
        print(f"  Random profitable:   {perm['n_random_profitable']}/{perm['n_permutations']}")
        print(f"\n  >>> VERDICT: {perm['verdict']}")

    # ═══════════════════════════════════════════════════════════
    # Executive Summary
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("EXECUTIVE SUMMARY")
    print(f"{'='*90}")

    # Compare key configs
    if "C_green26_12pct_BA" in all_results and "error" not in all_results["C_green26_12pct_BA"]:
        mc = all_results["C_green26_12pct_BA"]["metrics"]
        print(f"\n  KEY QUESTION: Is BPS viable with only ~26 BA-viable tickers?")
        print(f"\n  ANSWER:")
        if mc["sharpe"] >= 0.5:
            print(f"    YES — Sharpe {mc['sharpe']:.2f} with realistic 12% BA on liquid names only.")
            print(f"    The strategy does NOT need 98 tickers to work.")
        elif mc["sharpe"] >= 0.0:
            print(f"    MARGINAL — Sharpe {mc['sharpe']:.2f} is positive but weak.")
            print(f"    The smaller universe degrades returns significantly.")
        else:
            print(f"    NO — Sharpe {mc['sharpe']:.2f} is negative. Strategy fails with realistic costs.")

    if "A_full98_5pct_BA" in all_results and "C_green26_12pct_BA" in all_results:
        if "error" not in all_results["A_full98_5pct_BA"] and "error" not in all_results["C_green26_12pct_BA"]:
            ma = all_results["A_full98_5pct_BA"]["metrics"]
            mc = all_results["C_green26_12pct_BA"]["metrics"]
            sharpe_drop = mc["sharpe"] - ma["sharpe"]
            print(f"\n  SHARPE DEGRADATION:")
            print(f"    Optimistic (98 tickers, 5% BA):   {ma['sharpe']:.2f}")
            print(f"    Realistic (26 tickers, 12% BA):   {mc['sharpe']:.2f}")
            print(f"    Drop:                             {sharpe_drop:+.2f}")

    if "D_green26_15pct_BA" in all_results and "error" not in all_results["D_green26_15pct_BA"]:
        md = all_results["D_green26_15pct_BA"]["metrics"]
        print(f"\n  STRESS TEST (15% BA pessimistic):")
        print(f"    Sharpe: {md['sharpe']:.2f}, CAGR: {md['cagr_pct']:.1f}%, MaxDD: {md['max_dd_pct']:.1f}%")
        if md["sharpe"] > 0:
            print(f"    Strategy SURVIVES even at pessimistic 15% BA.")
        else:
            print(f"    Strategy BREAKS at 15% BA. 12% is the boundary.")

    # ═══════════════════════════════════════════════════════════
    # Save All Results
    # ═══════════════════════════════════════════════════════════
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
        "starting_capital": STARTING_CAPITAL,
        "question": "Does BPS survive with only BA-viable tickers?",
        "green_core_tickers": GREEN_CORE,
        "green_extended_tickers": GREEN_EXTENDED,
        "methodology": {
            "spread_width": "$15",
            "margin_cap": "25%",
            "profit_take": "65%",
            "dte_target": DTE_TARGET,
            "dynamic_delta": "d35 VIX<15, d30 VIX 15-22, d25 VIX>22",
            "vix_scale": "linear scale down from VIX=15",
            "circuit_breaker": "2% daily loss -> 1 day freeze",
            "vix_hard_cutoff": 30,
            "earnings_filter": "skip if earnings within 7 days",
        },
        "configs": {},
    }

    for name in [c["label"] for c in configs]:
        if name not in all_results or "error" in all_results[name]:
            save_data["configs"][name] = {"error": all_results.get(name, {}).get("error", "unknown")}
        else:
            save_data["configs"][name] = all_results[name]["metrics"]

    save_data["permutation_tests"] = convert(perm_results)

    save_data = convert(save_data)

    with open(OUTPUT / "viable_universe_results.json", "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n{'='*90}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
