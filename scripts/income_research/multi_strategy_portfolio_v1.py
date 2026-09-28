#!/usr/bin/env python3
"""
Multi-Strategy Portfolio Backtest V1
====================================
Hypothesis: combining long-biased growth strategies with premium-selling
income strategies creates a portfolio that passes R1 regime-agnostic
validation in both bull and bear regimes.

Strategies & Allocations:
  1. UPRO Momentum (200-SMA)       — 20%  (equity growth, leveraged)
  2. Vol Compression Long-Only     — 15%  (equity growth, mean-reversion)
  3. Jade Lizard Income (7d DTE)   — 40%  (options income, short vol)
  4. Cash-Secured Put Income (35d) — 25%  (options income, short vol)

Each strategy runs on $100K notional independently, then combined with
allocation weights via daily equity curve blending.
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

TICKERS = [
    'AAPL','ABBV','ABNB','ADBE','AMD','AMZN','ARM','AXP','BA','BAC',
    'BLK','BRK-B','C','CAT','CL','COIN','COST','CRM','CRWD','CVX',
    'DDOG','DE','DIS','F','GE','GM','GOOGL','GS','HD','HOOD',
    'INTC','JNJ','JPM','KO','LLY','LOW','MA','MCD','META','MRNA',
    'MS','MSFT','NFLX','NOW','NVDA','ORCL','OXY','PANW','PEP','PFE',
    'PG','PLTR','PYPL','RTX','SBUX','SCHW','SHOP','SLB','SMCI','T',
    'TGT','TMUS','TSLA','UBER','UNH','V','VZ','WFC','WMT','XOM',
]

# Allocation weights
ALLOC_UPRO = 0.20
ALLOC_VOLCOMP = 0.15
ALLOC_JADE = 0.40
ALLOC_CSP = 0.25


# ===================================================================
# Black-Scholes Primitives (from jade_lizard_v2)
# ===================================================================

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


# ===================================================================
# Data Loading (from jade_lizard_v2)
# ===================================================================

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

    vix_df = pd.read_parquet(CACHE / "vix_history.parquet")
    vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.tz_localize(None)

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


def ba_cost_per_leg(premium, ba_frac):
    return premium * (ba_frac / 2.0) * 100


def vix_size_multiplier(vix_val, strategy="jade_lizard"):
    if np.isnan(vix_val):
        return 0.75
    if strategy == "jade_lizard":
        if vix_val > 35: return 0.25
        elif vix_val > 30: return 0.50
        elif vix_val > 25: return 0.75
        else: return 1.0
    return 1.0


# ===================================================================
# STRATEGY 1: UPRO Momentum (200-SMA)
# ===================================================================

def run_upro_momentum(spy_df, capital=STARTING_CAPITAL):
    """
    Buy UPRO (3x leveraged SPY) when SPY > 200-day SMA, else hold cash.
    Simulate 3x daily returns of SPY when signal is on.
    """
    print(f"\n{'='*70}")
    print("STRATEGY 1: UPRO MOMENTUM (200-SMA)")
    print(f"{'='*70}")

    spy = spy_df.copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").reset_index(drop=True)
    spy = spy[spy["date"] >= "2019-01-01"].copy()
    spy["spy_ret"] = spy["close"].pct_change()
    spy["sma200"] = spy["close"].rolling(200).mean()
    spy = spy.dropna(subset=["sma200", "spy_ret"]).reset_index(drop=True)

    equity = float(capital)
    equity_curve = []

    for i, row in spy.iterrows():
        signal = row["close"] > row["sma200"]
        if signal:
            # UPRO = 3x daily return of SPY, minus ~0.95% annual expense ratio
            daily_expense = 0.0095 / 252
            daily_ret = 3.0 * row["spy_ret"] - daily_expense
        else:
            # Cash earns ~0% (SHV-like)
            daily_ret = 0.0

        equity *= (1 + daily_ret)
        equity_curve.append({"date": row["date"], "equity": equity})

    print(f"  {len(equity_curve)} trading days, final equity: ${equity:,.0f}")
    return equity_curve


# ===================================================================
# STRATEGY 2: Vol Compression Long-Only (10d hold)
# ===================================================================

def run_vol_compression(prices_df, capital=STARTING_CAPITAL):
    """
    Buy S&P 500 stocks showing vol compression breakouts to the upside.
    Detection: price breaks above upper Bollinger band (2 sigma) after
    vol compression (current 20d vol < 50th percentile of trailing 252 days).
    Hold for 10 trading days.
    """
    print(f"\n{'='*70}")
    print("STRATEGY 2: VOL COMPRESSION LONG-ONLY (10d hold)")
    print(f"{'='*70}")

    HOLD_DAYS = 10
    MAX_POSITIONS = 20  # max concurrent positions
    PER_POSITION_PCT = 0.05  # 5% of capital per position

    # Use all available tickers in prices
    all_tickers = sorted(prices_df["ticker"].unique())

    # Pre-compute features per ticker
    ticker_data = {}
    for tk in all_tickers:
        df = prices_df[prices_df["ticker"] == tk].sort_values("date").copy()
        if len(df) < 260:
            continue
        df["log_ret"] = np.log1p(df["close"].pct_change())
        df["rv_20"] = df["log_ret"].rolling(20).std() * np.sqrt(252)
        df["rv_median_252"] = df["rv_20"].rolling(252).median()
        df["sma_20"] = df["close"].rolling(20).mean()
        df["std_20"] = df["close"].rolling(20).std()
        df["upper_bb"] = df["sma_20"] + 2.0 * df["std_20"]
        df = df.dropna(subset=["rv_20", "rv_median_252", "upper_bb"])
        if len(df) < 30:
            continue
        ticker_data[tk] = df.set_index("date")

    # Get all unique dates
    all_dates_set = set()
    for tk, df in ticker_data.items():
        all_dates_set.update(df.index.tolist())
    all_dates = sorted(all_dates_set)
    all_dates = [d for d in all_dates if d >= pd.Timestamp("2019-01-01")]

    equity = float(capital)
    positions = {}  # ticker -> {"entry_date": date, "entry_price": price, "shares": n, "exit_idx": int}
    equity_curve = []
    trades = []

    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for di, dt in enumerate(all_dates):
        # Close expired positions
        to_close = []
        for tk, pos in list(positions.items()):
            if di >= pos["exit_idx"]:
                # Get exit price
                if tk in ticker_data and dt in ticker_data[tk].index:
                    exit_price = ticker_data[tk].loc[dt, "close"]
                    if isinstance(exit_price, pd.Series):
                        exit_price = exit_price.iloc[0]
                else:
                    exit_price = pos["entry_price"]  # fallback

                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                equity += pnl + pos["entry_price"] * pos["shares"]  # return capital + pnl
                trades.append({
                    "open_date": pos["entry_date"], "close_date": dt,
                    "ticker": tk, "pnl": pnl, "close_reason": "time_exit",
                    "hold_days": HOLD_DAYS,
                    "ret_pct": (exit_price / pos["entry_price"] - 1) * 100,
                })
                to_close.append(tk)

        for tk in to_close:
            del positions[tk]

        # Open new positions
        if len(positions) < MAX_POSITIONS:
            candidates = []
            for tk, df in ticker_data.items():
                if tk in positions:
                    continue
                if dt not in df.index:
                    continue
                row = df.loc[dt]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                # Vol compression: current vol < median of trailing 252 days
                vol_compressed = row["rv_20"] < row["rv_median_252"]
                # Breakout: close above upper Bollinger band
                breakout = row["close"] > row["upper_bb"]

                if vol_compressed and breakout:
                    candidates.append((tk, row["close"], row["rv_20"]))

            # Sort by lowest vol (most compressed) first
            candidates.sort(key=lambda x: x[2])

            for tk, price, vol in candidates:
                if len(positions) >= MAX_POSITIONS:
                    break
                alloc = equity * PER_POSITION_PCT
                shares = int(alloc / price)
                if shares < 1:
                    continue
                cost = shares * price
                if cost > equity * 0.9:  # don't over-allocate
                    continue
                equity -= cost  # deduct capital
                positions[tk] = {
                    "entry_date": dt, "entry_price": price,
                    "shares": shares, "exit_idx": di + HOLD_DAYS,
                }

        # MTM positions
        pos_value = 0
        for tk, pos in positions.items():
            if tk in ticker_data and dt in ticker_data[tk].index:
                cur_price = ticker_data[tk].loc[dt, "close"]
                if isinstance(cur_price, pd.Series):
                    cur_price = cur_price.iloc[0]
            else:
                cur_price = pos["entry_price"]
            pos_value += cur_price * pos["shares"]

        total_equity = equity + pos_value
        equity_curve.append({"date": dt, "equity": total_equity})

    print(f"  {len(equity_curve)} trading days, {len(trades)} trades, "
          f"final equity: ${equity_curve[-1]['equity'] if equity_curve else 0:,.0f}")
    return equity_curve, trades


# ===================================================================
# STRATEGY 3: Jade Lizard Income (7d DTE)
# ===================================================================

def run_jade_lizard_v2(prices, iv, macro, vix_df, earnings_lookup, ticker_list,
                       ba_frac=0.10, dte_target=7, put_delta=-0.25, call_delta=0.25,
                       long_call_offset=0.05, profit_take=0.50, stop_loss_mult=2.0,
                       max_concurrent=25, per_name_pct=0.04, vix_hard_cutoff=35,
                       earnings_buffer=7, premium_mult=2.2,
                       allow_reentry=True, reentry_cooldown=0,
                       capital=STARTING_CAPITAL):
    """Jade Lizard V2 with premium multiplier."""
    print(f"\n{'='*70}")
    print(f"STRATEGY 3: JADE LIZARD (DTE={dte_target}, prem_mult={premium_mult}x, max_pos={max_concurrent})")
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

    cash = float(capital)
    positions = {}
    equity_curve = []
    detailed_trades = []
    last_close = {}
    pos_counter = 0
    stats = {"n_opened": 0, "n_closed": 0, "n_profit_take": 0, "n_stop_loss": 0,
             "n_expiry_win": 0, "n_expiry_loss": 0, "n_assigned_put": 0,
             "n_earnings_blocked": 0, "n_vix_blocked": 0, "total_ba_cost": 0.0,
             "total_premium_collected": 0.0, "total_premium_paid": 0.0}

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
        for pos_id, pos in list(positions.items()):
            tk = pos["ticker"]
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_cur = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            if T_days <= 0:
                put_itm = S < pos["put_strike"]
                call_itm = S > pos["short_call_strike"]
                pnl = pos["net_credit_dollar"]
                if put_itm:
                    assign_loss = (pos["put_strike"] - S) * 100 * pos["contracts"]
                    pnl -= assign_loss
                    stats["n_assigned_put"] += 1
                if call_itm:
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
                    "put_itm": put_itm, "call_itm": call_itm,
                    "hold_days": (dt - pos["open_date"]).days,
                })
                to_remove.append(pos_id)
                last_close[tk] = dt
                stats["n_closed"] += 1
                continue

            put_prem_now = bs_price(S, pos["put_strike"], T, sigma_cur, kind="put") * premium_mult
            short_call_prem_now = bs_price(S, pos["short_call_strike"], T, sigma_cur, kind="call") * premium_mult
            long_call_prem_now = bs_price(S, pos["long_call_strike"], T, sigma_cur, kind="call") * premium_mult
            current_cost_to_close = (put_prem_now + short_call_prem_now - long_call_prem_now) * 100 * pos["contracts"]
            ba_close = (ba_cost_per_leg(put_prem_now, ba_frac) +
                        ba_cost_per_leg(short_call_prem_now, ba_frac) +
                        ba_cost_per_leg(long_call_prem_now, ba_frac)) * pos["contracts"]
            unrealized_pnl = pos["net_credit_dollar"] - current_cost_to_close - ba_close

            if unrealized_pnl >= profit_take * pos["net_credit_dollar"]:
                cash += unrealized_pnl
                stats["total_ba_cost"] += ba_close
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": unrealized_pnl, "close_reason": "profit_take",
                    "hold_days": (dt - pos["open_date"]).days,
                })
                to_remove.append(pos_id)
                last_close[tk] = dt
                stats["n_profit_take"] += 1
                stats["n_closed"] += 1
                continue

            if unrealized_pnl < -(stop_loss_mult * pos["net_credit_dollar"]):
                cash += unrealized_pnl
                stats["total_ba_cost"] += ba_close
                detailed_trades.append({
                    "open_date": pos["open_date"], "close_date": dt, "ticker": tk,
                    "pnl": unrealized_pnl, "close_reason": "stop_loss",
                    "hold_days": (dt - pos["open_date"]).days,
                })
                to_remove.append(pos_id)
                last_close[tk] = dt
                stats["n_stop_loss"] += 1
                stats["n_closed"] += 1
                continue

        for pid in to_remove:
            if pid in positions:
                del positions[pid]

        if len(positions) < max_concurrent and not np.isnan(vix_val):
            if vix_val >= vix_hard_cutoff:
                stats["n_vix_blocked"] += 1
            else:
                size_mult = vix_size_multiplier(vix_val, "jade_lizard")
                if size_mult > 0:
                    ticker_pos_count = defaultdict(int)
                    for pid, pos in positions.items():
                        ticker_pos_count[pos["ticker"]] += 1
                    candidates = []
                    for tk in ticker_list:
                        if tk not in available:
                            continue
                        if ticker_pos_count.get(tk, 0) >= 1:
                            continue
                        if reentry_cooldown > 0 and tk in last_close:
                            days_since = (dt - last_close[tk]).days
                            if days_since < reentry_cooldown:
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
                        K_short_call = strike_from_delta(S, T, sigma, call_delta, kind="call")
                        K_long_call = K_short_call * (1 + long_call_offset)
                        put_prem = bs_price(S, K_put, T, sigma, kind="put") * premium_mult
                        short_call_prem = bs_price(S, K_short_call, T, sigma, kind="call") * premium_mult
                        long_call_prem = bs_price(S, K_long_call, T, sigma, kind="call") * premium_mult
                        net_credit_per_share = put_prem + short_call_prem - long_call_prem
                        if net_credit_per_share <= 0.10:
                            continue
                        ba_open = (ba_cost_per_leg(put_prem, ba_frac) +
                                   ba_cost_per_leg(short_call_prem, ba_frac) +
                                   ba_cost_per_leg(long_call_prem, ba_frac))
                        put_margin = K_put * 100
                        call_spread_width = (K_long_call - K_short_call) * 100
                        margin_needed = max(put_margin, call_spread_width)
                        sizing_base = min(cash, capital * 1.5)
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
                        expiry = dt + pd.Timedelta(days=dte_target)
                        pos_counter += 1
                        pos_id = f"{tk}_{pos_counter}"
                        positions[pos_id] = {
                            "ticker": tk, "open_date": dt, "expiry": expiry,
                            "contracts": contracts, "put_strike": K_put,
                            "short_call_strike": K_short_call, "long_call_strike": K_long_call,
                            "net_credit_dollar": net_credit_dollar, "open_sigma": sigma,
                            "open_price": S,
                        }
                        cash += net_credit_dollar
                        stats["n_opened"] += 1
                        stats["total_premium_collected"] += net_credit_dollar

        mtm = 0
        for pos_id, pos in positions.items():
            tk = pos["ticker"]
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_cur = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20
            put_prem = bs_price(S, pos["put_strike"], T, sigma_cur, kind="put") * premium_mult
            sc_prem = bs_price(S, pos["short_call_strike"], T, sigma_cur, kind="call") * premium_mult
            lc_prem = bs_price(S, pos["long_call_strike"], T, sigma_cur, kind="call") * premium_mult
            cost_to_close = (put_prem + sc_prem - lc_prem) * 100 * pos["contracts"]
            mtm -= cost_to_close

        total_equity = cash + mtm
        equity_curve.append({"date": dt, "equity": total_equity, "cash": cash, "n_pos": len(positions)})

    print(f"  {len(equity_curve)} trading days, {stats['n_opened']} trades opened, "
          f"final equity: ${equity_curve[-1]['equity'] if equity_curve else 0:,.0f}")
    return equity_curve, detailed_trades, stats


# ===================================================================
# STRATEGY 4: Cash-Secured Put Income (35d DTE)
# ===================================================================

def run_csp_baseline(prices, iv, macro, vix_df, earnings_lookup, ticker_list,
                     ba_frac=0.10, dte_target=35, put_delta=-0.25,
                     profit_take=0.65, stop_loss_mult=2.0,
                     max_concurrent=30, per_name_pct=0.03,
                     capital=STARTING_CAPITAL):
    """CSP baseline strategy."""
    print(f"\n{'='*70}")
    print(f"STRATEGY 4: CASH-SECURED PUT (DTE={dte_target}, max_pos={max_concurrent})")
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

    cash = float(capital)
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
                    "pnl": pnl, "close_reason": "expiry",
                    "hold_days": (dt - pos["open_date"]).days,
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
                    "pnl": unrealized_pnl, "close_reason": "profit_take",
                    "hold_days": (dt - pos["open_date"]).days,
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
                    "pnl": unrealized_pnl, "close_reason": "stop_loss",
                    "hold_days": (dt - pos["open_date"]).days,
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
                sizing_base = min(cash, capital * 1.5)
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

    print(f"  {len(equity_curve)} trading days, {stats['n_opened']} trades opened, "
          f"final equity: ${equity_curve[-1]['equity'] if equity_curve else 0:,.0f}")
    return equity_curve, detailed_trades, stats


# ===================================================================
# Analysis Functions
# ===================================================================

def compute_metrics(equity_curve, label="Strategy"):
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

    gains = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")
    wr = np.mean(rets > 0)

    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["daily_ret"].sum()
    avg_monthly_ret = monthly.mean() * 100
    monthly_wr = (monthly > 0).mean() * 100

    return {
        "label": label,
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "cagr_pct": round(cagr * 100, 1),
        "max_dd_pct": round(max_dd * 100, 1),
        "pf": round(pf, 2),
        "daily_wr_pct": round(wr * 100, 1),
        "total_return_pct": round(total_return * 100, 1),
        "avg_monthly_ret_pct": round(avg_monthly_ret, 2),
        "monthly_wr_pct": round(monthly_wr, 1),
        "n_days": len(df),
        "start": str(df["date"].iloc[0].date()),
        "end": str(df["date"].iloc[-1].date()),
        "final_equity": round(df["equity"].iloc[-1], 2),
    }


def regime_test(equity_curve, spy_df, label="Strategy"):
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
        return {"label": label, "error": "insufficient data"}

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

    s_green = results.get("green", {}).get("sharpe", 0)
    s_red = results.get("red", {}).get("sharpe", 0)
    max_abs = max(abs(s_green), abs(s_red))
    r1_gap = abs(s_green - s_red) / max_abs if max_abs > 0 else 0

    return {
        "label": label,
        "green_sharpe": s_green,
        "green_days": results.get("green", {}).get("n_days", 0),
        "red_sharpe": s_red,
        "red_days": results.get("red", {}).get("n_days", 0),
        "flat_sharpe": results.get("flat", {}).get("sharpe", 0),
        "flat_days": results.get("flat", {}).get("n_days", 0),
        "r1_gap": round(r1_gap, 3),
        "r1_pass": r1_gap < 0.50,
    }


def permutation_test_equity(equity_curve, n_perms=2000, label="Strategy"):
    """Permutation test on daily returns of equity curve."""
    df = pd.DataFrame(equity_curve)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["daily_ret"] = df["equity"].pct_change()
    df = df.dropna(subset=["daily_ret"])

    rets = df["daily_ret"].values
    if len(rets) < 30:
        return {"label": label, "error": "insufficient data"}

    actual_sharpe = (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(252) if np.std(rets, ddof=1) > 0 else 0
    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = np.random.permutation(rets)
        s = (np.mean(shuffled) / np.std(shuffled, ddof=1)) * np.sqrt(252) if np.std(shuffled, ddof=1) > 0 else 0
        perm_sharpes.append(s)
    p_value = np.mean(np.array(perm_sharpes) >= actual_sharpe)
    return {
        "label": label,
        "actual_sharpe": round(actual_sharpe, 3),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "n_days": len(rets),
    }


def permutation_test_trades(trades, n_perms=2000, label="Strategy"):
    """Permutation test on trade PnLs."""
    if not trades or len(trades) < 10:
        return {"label": label, "error": "insufficient trades"}
    pnls = np.array([t["pnl"] for t in trades])
    actual_mean = np.mean(pnls)
    actual_t = actual_mean / (np.std(pnls, ddof=1) / np.sqrt(len(pnls))) if np.std(pnls, ddof=1) > 0 else 0
    perm_ts = []
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(pnls))
        shuffled = pnls * signs
        perm_mean = np.mean(shuffled)
        perm_std = np.std(shuffled, ddof=1)
        perm_t = perm_mean / (perm_std / np.sqrt(len(shuffled))) if perm_std > 0 else 0
        perm_ts.append(perm_t)
    p_value = np.mean(np.array(perm_ts) >= actual_t)
    return {
        "label": label,
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
        "n_trades": len(pnls),
        "avg_pnl": round(actual_mean, 2),
    }


# ===================================================================
# PORTFOLIO COMBINER
# ===================================================================

def combine_portfolios(curves_dict, weights, label="Combined Portfolio"):
    """
    Combine multiple strategy equity curves into a single portfolio.
    Each strategy runs on its own $100K. The combined portfolio is:
      combined_equity = sum(weight_i * equity_i)
    normalized so the starting capital = $100K.
    """
    # Convert all to DataFrames indexed by date
    dfs = {}
    for name, curve in curves_dict.items():
        df = pd.DataFrame(curve)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first")
        df = df.set_index("date")[["equity"]]
        df.columns = [name]
        dfs[name] = df

    # Merge on date (inner join — only dates where ALL strategies have data)
    combined = None
    for name, df in dfs.items():
        if combined is None:
            combined = df
        else:
            combined = combined.join(df, how="inner")

    if combined is None or len(combined) < 30:
        return []

    # Normalize each to return series (growth of $1)
    for name in curves_dict.keys():
        first_val = combined[name].iloc[0]
        if first_val > 0:
            combined[f"{name}_growth"] = combined[name] / first_val
        else:
            combined[f"{name}_growth"] = 1.0

    # Combined portfolio: weighted sum of growth
    combined["portfolio_growth"] = 0.0
    for name, weight in weights.items():
        combined["portfolio_growth"] += weight * combined[f"{name}_growth"]

    # Convert back to equity ($100K base)
    combined["equity"] = combined["portfolio_growth"] * STARTING_CAPITAL

    equity_curve = []
    for date, row in combined.iterrows():
        equity_curve.append({"date": date, "equity": row["equity"]})

    return equity_curve


# ===================================================================
# MAIN
# ===================================================================

def main():
    np.random.seed(42)
    t0 = time.time()

    prices, iv, macro, vix_df, spy, earnings, avail_tickers = load_all_data()
    earnings_lookup = build_earnings_lookup(earnings)

    print(f"\nData loaded in {time.time()-t0:.1f}s")
    print(f"Running multi-strategy portfolio backtest...")
    print(f"Period: 2019-01-01 to present")
    print(f"Starting capital per strategy: ${STARTING_CAPITAL:,}")

    # ── Run each strategy ──
    t1 = time.time()

    # Strategy 1: UPRO Momentum
    upro_curve = run_upro_momentum(spy, capital=STARTING_CAPITAL)
    print(f"  UPRO done in {time.time()-t1:.1f}s")

    # Strategy 2: Vol Compression
    t2 = time.time()
    volcomp_curve, volcomp_trades = run_vol_compression(prices, capital=STARTING_CAPITAL)
    print(f"  VolComp done in {time.time()-t2:.1f}s")

    # Strategy 3: Jade Lizard (7d DTE, 2.2x premium)
    t3 = time.time()
    jade_curve, jade_trades, jade_stats = run_jade_lizard_v2(
        prices, iv, macro, vix_df, earnings_lookup, avail_tickers,
        dte_target=7, premium_mult=2.2, max_concurrent=25,
        capital=STARTING_CAPITAL,
    )
    print(f"  Jade Lizard done in {time.time()-t3:.1f}s")

    # Strategy 4: CSP (35d DTE)
    t4 = time.time()
    csp_curve, csp_trades, csp_stats = run_csp_baseline(
        prices, iv, macro, vix_df, earnings_lookup, avail_tickers,
        dte_target=35, max_concurrent=30,
        capital=STARTING_CAPITAL,
    )
    print(f"  CSP done in {time.time()-t4:.1f}s")

    # ── Combine ──
    curves = {
        "UPRO_Momentum": upro_curve,
        "Vol_Compression": volcomp_curve,
        "Jade_Lizard": jade_curve,
        "CSP_Income": csp_curve,
    }
    weights = {
        "UPRO_Momentum": ALLOC_UPRO,
        "Vol_Compression": ALLOC_VOLCOMP,
        "Jade_Lizard": ALLOC_JADE,
        "CSP_Income": ALLOC_CSP,
    }
    combined_curve = combine_portfolios(curves, weights, "Combined Portfolio")

    # ── Compute metrics for each + combined ──
    all_strategies = {
        "UPRO Momentum (20%)": upro_curve,
        "Vol Compression (15%)": volcomp_curve,
        "Jade Lizard 7d (40%)": jade_curve,
        "CSP 35d (25%)": csp_curve,
        "COMBINED PORTFOLIO": combined_curve,
    }

    print(f"\n\n{'='*120}")
    print("MULTI-STRATEGY PORTFOLIO BACKTEST RESULTS")
    print(f"{'='*120}")
    print(f"{'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MaxDD%':>7} "
          f"{'PF':>6} {'DailyWR%':>9} {'TotalRet%':>10} {'Final$':>12}")
    print("-" * 120)

    all_metrics = {}
    for name, curve in all_strategies.items():
        m = compute_metrics(curve, name)
        all_metrics[name] = m
        if "error" in m:
            print(f"  {name:<28} ERROR: {m['error']}")
            continue
        is_combined = "COMBINED" in name
        prefix = ">>>" if is_combined else "   "
        print(f"{prefix}{name:<28} {m['sharpe']:>7} {m['sortino']:>8} {m['cagr_pct']:>6.1f}% "
              f"{m['max_dd_pct']:>6.1f}% {m['pf']:>6.2f} {m['daily_wr_pct']:>8.1f}% "
              f"{m['total_return_pct']:>9.1f}% ${m['final_equity']:>10,.0f}")

    # ── Regime Test (R1) for each + combined ──
    print(f"\n\n{'='*100}")
    print("REGIME TEST (R1) — Regime-Agnostic Validation")
    print(f"{'='*100}")
    print(f"{'Strategy':<30} {'Green Sharpe':>13} {'Green Days':>11} {'Red Sharpe':>11} "
          f"{'Red Days':>9} {'Flat Sharpe':>12} {'R1 Gap':>7} {'Result':>8}")
    print("-" * 100)

    all_regime = {}
    for name, curve in all_strategies.items():
        r = regime_test(curve, spy, name)
        all_regime[name] = r
        if "error" in r:
            print(f"  {name:<28} ERROR: {r['error']}")
            continue
        is_combined = "COMBINED" in name
        prefix = ">>>" if is_combined else "   "
        result_str = "PASS" if r["r1_pass"] else "FAIL"
        print(f"{prefix}{name:<28} {r['green_sharpe']:>13} {r['green_days']:>11} "
              f"{r['red_sharpe']:>11} {r['red_days']:>9} {r['flat_sharpe']:>12} "
              f"{r['r1_gap']:>7.3f} {result_str:>8}")

    # ── Permutation test on combined portfolio ──
    print(f"\n\n{'='*80}")
    print("PERMUTATION TEST (2000 permutations)")
    print(f"{'='*80}")

    for name, curve in all_strategies.items():
        perm = permutation_test_equity(curve, n_perms=2000, label=name)
        is_combined = "COMBINED" in name
        prefix = ">>>" if is_combined else "   "
        if "error" in perm:
            print(f"{prefix}{name:<28} ERROR: {perm['error']}")
        else:
            sig_str = "SIGNIFICANT" if perm["significant"] else "NOT SIG"
            print(f"{prefix}{name:<28} Sharpe={perm['actual_sharpe']:.3f}  "
                  f"p-value={perm['p_value']:.4f}  {sig_str}")

    # ── Trade-level stats for trade-based strategies ──
    print(f"\n\n{'='*100}")
    print("TRADE-LEVEL STATISTICS")
    print(f"{'='*100}")

    trade_sets = {
        "Vol Compression": volcomp_trades,
        "Jade Lizard 7d": jade_trades,
        "CSP 35d": csp_trades,
    }

    for name, trades in trade_sets.items():
        if not trades:
            print(f"\n  {name}: No trades")
            continue
        tdf = pd.DataFrame(trades)
        n = len(tdf)
        wins = (tdf["pnl"] > 0).sum()
        avg_win = tdf[tdf["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
        avg_loss = tdf[tdf["pnl"] <= 0]["pnl"].mean() if (n - wins) > 0 else 0
        total_pnl = tdf["pnl"].sum()
        avg_hold = tdf["hold_days"].mean() if "hold_days" in tdf.columns else 0
        years_span = max((pd.to_datetime(tdf["close_date"]).max() - pd.to_datetime(tdf["open_date"]).min()).days / 365.25, 0.1)
        tpy = n / years_span

        print(f"\n  {name}:")
        print(f"    Trades: {n} ({tpy:.0f}/yr)  WR: {wins/n*100:.1f}%  Avg hold: {avg_hold:.1f}d")
        print(f"    Avg win: ${avg_win:,.2f}  Avg loss: ${avg_loss:,.2f}  Total PnL: ${total_pnl:,.2f}")
        if "close_reason" in tdf.columns:
            reasons = tdf["close_reason"].value_counts().to_dict()
            print(f"    Close reasons: {reasons}")

        # Permutation test on trades
        perm = permutation_test_trades(trades, n_perms=2000, label=name)
        if "error" not in perm:
            print(f"    Perm test: p={perm['p_value']:.4f} {'SIGNIFICANT' if perm['significant'] else 'NOT SIG'}")

    # ── Options engine stats ──
    print(f"\n\n{'='*80}")
    print("OPTIONS ENGINE STATS")
    print(f"{'='*80}")
    for name, stats in [("Jade Lizard", jade_stats), ("CSP", csp_stats)]:
        print(f"\n  {name}:")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"    {k}: ${v:,.2f}" if "cost" in k or "premium" in k else f"    {k}: {v:.2f}")
            else:
                print(f"    {k}: {v}")

    # ── Annual breakdown of combined portfolio ──
    print(f"\n\n{'='*80}")
    print("ANNUAL BREAKDOWN — COMBINED PORTFOLIO")
    print(f"{'='*80}")

    cdf = pd.DataFrame(combined_curve)
    cdf["date"] = pd.to_datetime(cdf["date"])
    cdf = cdf.sort_values("date").reset_index(drop=True)
    cdf["daily_ret"] = cdf["equity"].pct_change()
    cdf["year"] = cdf["date"].dt.year

    print(f"  {'Year':>6} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'Days':>6}")
    print("  " + "-" * 40)
    for year, grp in cdf.groupby("year"):
        if len(grp) < 20:
            continue
        rets = grp["daily_ret"].dropna().values
        if len(rets) < 10:
            continue
        yr_ret = (grp["equity"].iloc[-1] / grp["equity"].iloc[0] - 1) * 100
        yr_sharpe = (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(252) if np.std(rets, ddof=1) > 0 else 0
        cum_max = grp["equity"].cummax()
        yr_dd = ((grp["equity"] - cum_max) / cum_max).min() * 100
        print(f"  {year:>6} {yr_ret:>8.1f}% {yr_sharpe:>8.2f} {yr_dd:>7.1f}% {len(grp):>6}")

    # ── Correlation matrix of daily returns ──
    print(f"\n\n{'='*80}")
    print("DAILY RETURN CORRELATIONS")
    print(f"{'='*80}")

    ret_dfs = {}
    for name, curve in curves.items():
        df = pd.DataFrame(curve)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date", keep="first").set_index("date")
        df["ret"] = df["equity"].pct_change()
        ret_dfs[name] = df["ret"]

    corr_df = pd.DataFrame(ret_dfs)
    corr_df = corr_df.dropna()
    corr_matrix = corr_df.corr()
    print(f"\n  {'':>20}", end="")
    for col in corr_matrix.columns:
        print(f"{col[:15]:>17}", end="")
    print()
    for idx in corr_matrix.index:
        print(f"  {idx[:20]:>20}", end="")
        for col in corr_matrix.columns:
            print(f"{corr_matrix.loc[idx, col]:>17.3f}", end="")
        print()

    # ── Income projection ──
    print(f"\n\n{'='*80}")
    print("INCOME PROJECTION (on $100K combined portfolio)")
    print(f"{'='*80}")
    cm = all_metrics.get("COMBINED PORTFOLIO", {})
    if "error" not in cm:
        cagr = cm["cagr_pct"]
        monthly = cm["avg_monthly_ret_pct"]
        print(f"  CAGR: {cagr:.1f}%")
        print(f"  Monthly avg return: {monthly:.2f}% (${monthly/100*100000:,.0f}/mo)")
        print(f"  Annual income: ${cagr/100*100000:,.0f}")
        print(f"  Max drawdown: {cm['max_dd_pct']:.1f}%")
        print(f"  Sharpe: {cm['sharpe']}  Sortino: {cm['sortino']}")

    # ── Hypothesis verdict ──
    print(f"\n\n{'='*80}")
    print("HYPOTHESIS VERDICT")
    print(f"{'='*80}")

    r1_combined = all_regime.get("COMBINED PORTFOLIO", {})
    r1_upro = all_regime.get("UPRO Momentum (20%)", {})
    r1_jade = all_regime.get("Jade Lizard 7d (40%)", {})

    print(f"\n  Question: Does combining equity-long + premium-selling strategies")
    print(f"           produce a regime-agnostic portfolio?")
    print()

    if "error" not in r1_combined:
        if r1_combined.get("r1_pass"):
            print(f"  ANSWER: YES — Combined portfolio PASSES R1")
            print(f"    Green-day Sharpe: {r1_combined['green_sharpe']}")
            print(f"    Red-day Sharpe:   {r1_combined['red_sharpe']}")
            print(f"    R1 gap:           {r1_combined['r1_gap']:.3f} (< 0.50 threshold)")
        else:
            print(f"  ANSWER: NO — Combined portfolio FAILS R1")
            print(f"    Green-day Sharpe: {r1_combined['green_sharpe']}")
            print(f"    Red-day Sharpe:   {r1_combined['red_sharpe']}")
            print(f"    R1 gap:           {r1_combined['r1_gap']:.3f} (>= 0.50 threshold)")

        # Show individual strategy R1 for comparison
        print(f"\n  Individual strategy R1 for context:")
        for name in all_regime:
            r = all_regime[name]
            if "error" not in r:
                status = "PASS" if r["r1_pass"] else "FAIL"
                print(f"    {name:<30} gap={r['r1_gap']:.3f} ({status})")

    # Save results
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        return obj

    def clean_dict(d):
        if isinstance(d, dict): return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(d, list): return [clean_dict(x) for x in d]
        else: return make_serializable(d)

    output_data = {
        "metrics": clean_dict(all_metrics),
        "regime": clean_dict(all_regime),
        "allocations": {"UPRO": ALLOC_UPRO, "VolComp": ALLOC_VOLCOMP,
                        "Jade": ALLOC_JADE, "CSP": ALLOC_CSP},
    }
    output_file = OUTPUT / "multi_strategy_portfolio_v1.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n\nTotal runtime: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
