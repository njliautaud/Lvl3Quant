#!/usr/bin/env python3
"""
BPS Expanded Universe Study — HC #660
=======================================
Expands the BPS universe beyond the current 70 large-cap tickers into 3 tiers:
  Tier 1: Current ~70 large-caps (5% BA) — baseline
  Tier 2: +30 mid-cap higher-beta (7% BA) — biotech, semis, energy, REITs, industrials, consumer
  Tier 3: +20 small-cap high-beta (9% BA) — aggressive names with rich IV

Downloads new tickers via yfinance, generates Black-Scholes IV from realized vol,
runs BPS backtest with optimal config (d30/$15/25% margin/65% PT), and compares
diversification, risk-adjusted returns, and per-sector/beta-bucket performance.

Output: output/bps_expanded_universe/
"""
import sys
import json
import time
import math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUTPUT = ROOT / "output" / "bps_expanded_universe"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── Universe Definitions ──

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
    # Biotech
    'BIIB','REGN','VRTX','GILD',
    # Semis
    'MRVL','ON','SWKS','QRVO',
    # Energy
    'DVN','FANG','MPC','VLO','PSX',
    # REITs
    'O','AMT','PLD','EQIX','SPG',
    # Industrials
    'GWW','EMR','ROK','ITW','ETN',
    # Consumer
    'DG','DLTR','TJX','ROST','BBY',
]

TIER3_TICKERS = [
    # High-beta tech/software
    'SNAP','ROKU','RBLX','UPST','SOFI',
    # High-beta energy/materials
    'CLF','FCX','HAL','AR','RRC',
    # High-beta biotech
    'CRSP','ILMN','DXCM','ENPH',
    # High-beta consumer/misc
    'ETSY','W','PENN','RIVN','LCID','MARA',
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

TIER3_SECTORS = {
    'SNAP': 'Communication Services', 'ROKU': 'Communication Services',
    'RBLX': 'Communication Services', 'UPST': 'Technology', 'SOFI': 'Financial Services',
    'CLF': 'Basic Materials', 'FCX': 'Basic Materials', 'HAL': 'Energy',
    'AR': 'Energy', 'RRC': 'Energy',
    'CRSP': 'Healthcare', 'ILMN': 'Healthcare', 'DXCM': 'Healthcare', 'ENPH': 'Technology',
    'ETSY': 'Consumer Cyclical', 'W': 'Consumer Cyclical', 'PENN': 'Consumer Cyclical',
    'RIVN': 'Consumer Cyclical', 'LCID': 'Consumer Cyclical', 'MARA': 'Financial Services',
}

# BA cost assumptions
BA_COST = {
    'tier1': 0.05,  # 5% for large-caps
    'tier2': 0.07,  # 7% for mid-caps
    'tier3': 0.09,  # 9% for small-cap high-beta
}

# ── Black-Scholes Primitives ──
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

def bs_delta(S, K, T, sigma, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0:
        return (-1.0 if S < K else 0.0) if kind == "put" else (1.0 if S > K else 0.0)
    d1 = (math.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    if kind == "put":
        return math.exp(-q*T) * (_Phi(d1) - 1.0)
    return math.exp(-q*T) * _Phi(d1)

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


# ── Data Loading ──

def download_yfinance_prices(tickers, start="2019-01-01"):
    """Download price data for tickers not already in cache."""
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not installed, skipping download")
        return pd.DataFrame()

    all_frames = []
    for i, tk in enumerate(tickers):
        try:
            data = yf.download(tk, start=start, progress=False, auto_adjust=True)
            if data.empty:
                print(f"  {tk}: no data")
                continue
            df = data.reset_index()
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                    'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            df['ticker'] = tk
            cols = [c for c in ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            df = df[cols]
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            all_frames.append(df)
            if (i + 1) % 10 == 0:
                print(f"  Downloaded {i+1}/{len(tickers)} tickers")
                time.sleep(1)  # Rate limit courtesy
        except Exception as e:
            print(f"  {tk}: download error: {e}")
            time.sleep(0.5)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)


def load_all_data():
    """Load base + expanded + v3 prices, download missing tickers, generate IV features."""
    print("Loading base data...")

    # Base prices (70 tickers)
    prices = pd.read_parquet(CACHE / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])

    # Expanded prices (127 tickers)
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

    # V3 expansion (33 tickers)
    try:
        p3 = pd.read_parquet(CACHE / "prices_v3_expansion.parquet")
        p3["date"] = pd.to_datetime(p3["date"]).dt.tz_localize(None)
        new_tks = set(p3["ticker"].unique()) - set(prices["ticker"].unique())
        if new_tks:
            p3 = p3[p3["ticker"].isin(new_tks)]
            prices = pd.concat([prices, p3], ignore_index=True)
    except Exception as e:
        print(f"  Warning: V3 prices: {e}")

    # Filter to 2019+
    prices = prices[prices["date"] >= "2019-01-01"].copy()

    # Check which tickers we still need
    all_needed = set(TIER1_TICKERS + TIER2_TICKERS + TIER3_TICKERS)
    have = set(prices["ticker"].unique())
    missing = all_needed - have
    if missing:
        print(f"\nDownloading {len(missing)} missing tickers: {sorted(missing)}")
        new_prices = download_yfinance_prices(sorted(missing))
        if not new_prices.empty:
            prices = pd.concat([prices, new_prices], ignore_index=True)

    # Standardize
    prices = prices.drop_duplicates(subset=["ticker", "date"], keep="first")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Compute returns and realized vol
    if "ret" not in prices.columns:
        prices["ret"] = prices.groupby("ticker")["close"].pct_change()
    if "log_ret" not in prices.columns:
        prices["log_ret"] = np.log1p(prices["ret"])
    if "rv_20" not in prices.columns:
        prices["rv_20"] = prices.groupby("ticker")["log_ret"].transform(
            lambda x: x.rolling(20).std() * np.sqrt(252))

    # Load existing IV data
    iv_base = pd.read_parquet(CACHE / "iv_features_modeled.parquet")
    iv_base["date"] = pd.to_datetime(iv_base["date"]).dt.tz_localize(None)
    iv_base = iv_base[iv_base["date"] >= "2019-01-01"]

    # Generate IV features for tickers not in base IV
    iv_tickers = set(iv_base["ticker"].unique())
    need_iv = (all_needed & have) - iv_tickers
    # Also add newly downloaded tickers
    need_iv |= (all_needed & set(prices["ticker"].unique())) - iv_tickers

    if need_iv:
        print(f"\nGenerating modeled IV for {len(need_iv)} tickers...")
        iv_new = generate_iv_features(prices, sorted(need_iv))
        if not iv_new.empty:
            iv_base = pd.concat([iv_base, iv_new], ignore_index=True)

    # Load macro
    macro = pd.read_parquet(CACHE / "macro.parquet")
    macro["date"] = pd.to_datetime(macro["date"]).dt.tz_localize(None)
    macro = macro[macro["date"] >= "2019-01-01"]

    # Load fundamentals and extend with tier 2/3 tickers
    fund = pd.read_parquet(CACHE / "fundamentals.parquet")
    existing_fund_tickers = set(fund["ticker"].unique())

    # Add tier 2/3 to fundamentals with sector info
    new_fund_rows = []
    for tk, sec in {**TIER2_SECTORS, **TIER3_SECTORS}.items():
        if tk not in existing_fund_tickers:
            new_fund_rows.append({
                'ticker': tk, 'sector': sec, 'market_cap': 0, 'beta': 1.5,
                'pe': 0, 'ps': 0, 'fcf_yield': 0, 'debt_to_equity': 0,
                'gross_margin': 0, 'ebitda_margin': 0, 'net_margin': 0,
                'current_ratio': 0, 'roe_proxy': 0, 'revenue_ttm': 0,
                'fcf_ttm': 0, 'industry': 'Unknown', 'fund_score': 50.0,
            })
    if new_fund_rows:
        fund = pd.concat([fund, pd.DataFrame(new_fund_rows)], ignore_index=True)

    # Compute actual betas from price data
    fund = compute_betas(prices, fund)

    # Universe = all tickers with both prices and IV
    available = set(prices["ticker"].unique()) & set(iv_base["ticker"].unique()) & all_needed
    universe = fund[fund["ticker"].isin(available)][["ticker", "sector"]].drop_duplicates("ticker")

    # Earnings
    try:
        earnings = pd.read_parquet(CACHE / "earnings_dates.parquet")
        earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"]).dt.tz_localize(None)
    except:
        earnings = pd.DataFrame(columns=["ticker", "earnings_date"])

    # Filter prices and IV to available tickers
    prices = prices[prices["ticker"].isin(available)]
    iv_base = iv_base[iv_base["ticker"].isin(available)]

    print(f"\n  Final universe: {len(available)} tickers")
    print(f"  Prices: {prices.shape[0]} rows, {prices['date'].min().date()} to {prices['date'].max().date()}")
    print(f"  IV: {iv_base.shape[0]} rows, {iv_base['ticker'].nunique()} tickers")

    return prices, iv_base, macro, fund, universe, earnings


def generate_iv_features(prices_df, tickers):
    """Generate modeled IV features from realized vol (Black-Scholes per HC #556)."""
    frames = []
    for tk in tickers:
        tk_px = prices_df[prices_df["ticker"] == tk].sort_values("date").copy()
        if len(tk_px) < 60:
            continue

        tk_px["log_ret"] = np.log1p(tk_px["close"].pct_change())
        tk_px["rv_20"] = tk_px["log_ret"].rolling(20).std() * np.sqrt(252)
        tk_px["rv_60"] = tk_px["log_ret"].rolling(60).std() * np.sqrt(252)

        # sigma = IV proxy: rv_20 * 1.15 (IV typically ~15% above realized)
        tk_px["sigma"] = tk_px["rv_20"] * 1.15
        tk_px["sigma_atm_30d"] = tk_px["rv_20"] * 1.10

        # IV rank: where is current IV relative to 252-day range
        tk_px["iv_high_252"] = tk_px["sigma"].rolling(252).max()
        tk_px["iv_low_252"] = tk_px["sigma"].rolling(252).min()
        iv_range = tk_px["iv_high_252"] - tk_px["iv_low_252"]
        tk_px["iv_rank"] = np.where(
            iv_range > 0.001,
            (tk_px["sigma"] - tk_px["iv_low_252"]) / iv_range,
            0.5
        )

        tk_px["sigma_rv"] = tk_px["rv_20"]
        tk_px["iv_rv_ratio"] = np.where(tk_px["rv_20"] > 0.001, tk_px["sigma"] / tk_px["rv_20"], 1.15)
        tk_px["term_ratio"] = np.where(
            tk_px["rv_20"] > 0.001,
            tk_px["rv_60"].fillna(tk_px["rv_20"]) / tk_px["rv_20"],
            1.0
        )
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


def compute_betas(prices_df, fund_df):
    """Compute realized beta vs SPY for each ticker."""
    spy = prices_df[prices_df["ticker"] == "SPY"].sort_values("date")
    if spy.empty:
        return fund_df

    spy_rets = spy.set_index("date")["ret"].dropna().to_dict()

    betas = {}
    for tk in prices_df["ticker"].unique():
        if tk == "SPY":
            betas[tk] = 1.0
            continue
        tk_px = prices_df[prices_df["ticker"] == tk].sort_values("date")
        if len(tk_px) < 60:
            continue
        tk_rets = tk_px.set_index("date")["ret"].dropna()
        common_dates = tk_rets.index.intersection(pd.Index(spy_rets.keys()))
        if len(common_dates) < 60:
            continue
        tk_r = tk_rets.loc[common_dates].values
        spy_r = np.array([spy_rets[d] for d in common_dates])

        # Remove NaNs
        valid = np.isfinite(tk_r) & np.isfinite(spy_r)
        if valid.sum() < 60:
            continue
        tk_r = tk_r[valid]
        spy_r = spy_r[valid]

        cov = np.cov(tk_r, spy_r)
        beta = cov[0, 1] / max(cov[1, 1], 1e-10)
        betas[tk] = round(float(beta), 3)

    fund_df = fund_df.copy()
    fund_df["beta_realized"] = fund_df["ticker"].map(betas)
    fund_df["beta_realized"] = fund_df["beta_realized"].fillna(fund_df.get("beta", 1.0))
    return fund_df


# ── BPS Backtest Engine (self-contained, with per-tier BA costs) ──

def run_bps_tiered(prices, iv, macro, fund, universe, earnings,
                   ticker_tier_map,  # ticker -> tier name
                   ba_costs,         # tier_name -> BA fraction
                   spread_width=15.0, put_delta=0.25, dte_target=30,
                   profit_take=0.65, margin_cap=0.25, max_concurrent=60,
                   per_name_pct=0.05, vix_gate=30.0, vix_scale=True,
                   daily_cb=0.02,  # 2% daily circuit breaker
                   starting_cash=100_000.0, label="BPS"):
    """
    Run BPS backtest with per-ticker BA costs based on tier membership.
    Returns raw results + honest (BA-adjusted) results.
    """
    prices_df = prices.copy()
    iv_df = iv.copy()

    px_by_date = {}
    for d, g in prices_df.groupby("date"):
        px_by_date[d] = g.set_index("ticker")["close"].to_dict()

    sigma_by_date = {}
    iv_rank_by_date = {}
    for d, g in iv_df.groupby("date"):
        sigma_by_date[d] = g.set_index("ticker")["sigma"].to_dict()
        iv_rank_by_date[d] = g.set_index("ticker")["iv_rank"].to_dict()

    macro_by_date = macro.set_index("date").to_dict("index")
    sector_of = dict(zip(fund["ticker"], fund.get("sector", pd.Series(["Unknown"] * len(fund)))))

    # Earnings
    earnings_set = {}
    for _, row in earnings.iterrows():
        tk = row["ticker"]
        ed = pd.Timestamp(row["earnings_date"])
        earnings_set.setdefault(tk, set()).add(ed)

    # SPY SMA50 for bear gate
    spy_sma50 = {}
    spy = prices_df[prices_df["ticker"] == "SPY"].sort_values("date")
    if len(spy) > 0:
        spy["sma50"] = spy["close"].rolling(50).mean()
        for _, row in spy.iterrows():
            spy_sma50[row["date"]] = (row["close"], row["sma50"] if pd.notna(row["sma50"]) else 0)

    all_dates = sorted(prices_df["date"].unique())

    cash = starting_cash
    positions = {}
    equity_curve = []
    trades = []  # Detailed trade log

    for di, dt in enumerate(all_dates):
        date_px = px_by_date.get(dt, {})
        date_sigma = sigma_by_date.get(dt, {})
        date_iv_rank = iv_rank_by_date.get(dt, {})
        m = macro_by_date.get(dt, {})
        vix = m.get("vix", float("nan"))

        # ── Update positions ──
        to_remove = []
        for tk, pos in list(positions.items()):
            S = date_px.get(tk)
            if S is None or np.isnan(S):
                continue
            T_days = (pos["expiry"] - dt).days
            T = max(T_days, 0) / 365.0
            sigma_atm = date_sigma.get(tk, pos["open_sigma"]) or pos["open_sigma"] or 0.20

            if T_days <= 0:
                # Expiry settlement
                short_itm = S < pos["short_strike"]
                long_itm = S < pos["long_strike"]
                close_cost = COST_PER_CONTRACT * 2 * pos["contracts"]

                if not short_itm:
                    realized = pos["net_credit"] - close_cost
                    cash -= close_cost
                    exit_type = "expire_otm"
                elif short_itm and not long_itm:
                    loss = (pos["short_strike"] - S) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost
                    exit_type = "expire_short_itm"
                else:
                    loss = (pos["short_strike"] - pos["long_strike"]) * 100 * pos["contracts"]
                    realized = pos["net_credit"] - loss - close_cost
                    cash -= loss + close_cost
                    exit_type = "expire_max_loss"

                trades.append({
                    "ticker": tk, "tier": ticker_tier_map.get(tk, "unknown"),
                    "sector": sector_of.get(tk, "Unknown"),
                    "open_date": pos["open_date"], "close_date": dt,
                    "exit_type": exit_type,
                    "net_credit": pos["net_credit"],
                    "realized_pnl": realized,
                    "contracts": pos["contracts"],
                    "short_strike": pos["short_strike"],
                    "long_strike": pos["long_strike"],
                })
                to_remove.append(tk)
            else:
                # Profit take
                short_val = bs_price(S, pos["short_strike"], T, sigma_atm, kind="put")
                long_val = bs_price(S, pos["long_strike"], T, sigma_atm, kind="put")
                spread_val = (short_val - long_val) * 100 * pos["contracts"]

                initial_credit = pos["net_credit"]
                cost_to_close = spread_val + trade_cost(short_val, pos["contracts"]) + trade_cost(long_val, pos["contracts"])
                captured = (initial_credit - cost_to_close) / max(initial_credit, 1e-6)

                if captured >= profit_take:
                    realized = initial_credit - cost_to_close
                    cash -= cost_to_close
                    trades.append({
                        "ticker": tk, "tier": ticker_tier_map.get(tk, "unknown"),
                        "sector": sector_of.get(tk, "Unknown"),
                        "open_date": pos["open_date"], "close_date": dt,
                        "exit_type": "profit_take",
                        "net_credit": pos["net_credit"],
                        "realized_pnl": realized,
                        "contracts": pos["contracts"],
                        "short_strike": pos["short_strike"],
                        "long_strike": pos["long_strike"],
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

        # ── Daily circuit breaker (2% CB) ──
        if daily_cb > 0 and len(equity_curve) >= 2:
            prev_eq = equity_curve[-2]["equity"]
            daily_ret = (equity - prev_eq) / max(prev_eq, 1)
            if daily_ret < -daily_cb:
                continue  # Skip new entries on bad days

        # ── Gates ──
        if not np.isnan(vix) and vix > vix_gate:
            continue
        if dt in spy_sma50:
            spy_close, spy_sma = spy_sma50[dt]
            if spy_sma > 0 and spy_close < spy_sma:
                continue

        if len(positions) >= max_concurrent:
            continue

        # Margin check
        current_margin = sum(
            (p["short_strike"] - p["long_strike"]) * 100 * p["contracts"]
            for p in positions.values()
        )

        # VIX-scaled margin cap
        eff_margin_cap = margin_cap
        if vix_scale and not np.isnan(vix):
            if vix < 15:
                eff_margin_cap = margin_cap * 1.2  # More aggressive in low-vol
            elif vix > 22:
                eff_margin_cap = margin_cap * 0.7  # Pull back in high-vol
            elif vix > 18:
                eff_margin_cap = margin_cap * 0.9

        if current_margin >= eff_margin_cap * equity:
            continue

        # ── Select candidates ──
        candidates = []
        universe_tickers = set(universe["ticker"].values) if hasattr(universe, 'values') else set()
        for tk, S in date_px.items():
            if tk == "__date__" or tk in positions:
                continue
            if tk not in universe_tickers:
                continue
            if S is None or np.isnan(S) or S < 10 or S > 2000:
                continue
            sigma = date_sigma.get(tk)
            if sigma is None or np.isnan(sigma) or sigma <= 0:
                continue
            iv_rk = date_iv_rank.get(tk, 0.5)

            # Earnings buffer
            expiry_date = dt + pd.Timedelta(days=dte_target)
            tk_earnings = earnings_set.get(tk, set())
            near_earnings = any(abs((ed - expiry_date).days) <= 3 for ed in tk_earnings)
            if near_earnings:
                continue

            candidates.append((tk, S, sigma, iv_rk))

        candidates.sort(key=lambda r: r[3], reverse=True)

        slots = min(max_concurrent - len(positions), max(1, max_concurrent // 5))
        remaining_margin = eff_margin_cap * equity - current_margin

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

            if margin_per_contract * n_contracts > remaining_margin:
                n_contracts = max(1, int(remaining_margin // margin_per_contract))
            if n_contracts < 1:
                continue

            net_credit = net_prem_per_share * 100 * n_contracts
            open_costs = trade_cost(prem_short, n_contracts) + trade_cost(prem_long, n_contracts)
            net_credit -= open_costs

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
            }
            remaining_margin -= margin_per_contract * n_contracts

            if len(positions) >= max_concurrent:
                break

    eq_df = pd.DataFrame(equity_curve)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

    if eq_df.empty:
        return {"label": label, "error": "no equity curve", "trades": trades_df}

    # Raw metrics
    metrics = compute_metrics(eq_df, starting_cash, label)
    metrics["n_trades"] = len(trades)

    # Apply per-tier BA haircuts for honest metrics
    honest_metrics = apply_tiered_ba_haircut(trades_df, ticker_tier_map, ba_costs)

    return {
        "metrics": metrics,
        "honest_metrics": honest_metrics,
        "equity_curve": eq_df,
        "trades": trades_df,
    }


def compute_metrics(equity_curve, starting_cash, label):
    eq = equity_curve.copy().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if len(eq) < 20:
        return {"label": label, "error": "too few data points"}
    eq["ret"] = eq["equity"].pct_change()
    rets = eq["ret"].dropna()
    total_days = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days
    total_years = total_days / 365.25
    total_return = eq["equity"].iloc[-1] / starting_cash
    cagr = (total_return ** (1 / max(total_years, 0.01))) - 1 if total_return > 0 else -1.0
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    downside = rets[rets < 0]
    sortino = rets.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0
    eq["peak"] = eq["equity"].cummax()
    eq["dd"] = (eq["equity"] - eq["peak"]) / eq["peak"]
    max_dd = eq["dd"].min()
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    wr = len(wins) / len(rets) if len(rets) > 0 else 0
    pf = abs(wins.sum() / losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")
    # Daily P&L concentration (HC #344)
    daily_pnl = rets * eq["equity"].shift(1)
    top_day = daily_pnl.abs().max() if len(daily_pnl) > 0 else 0
    total_abs = daily_pnl.abs().sum()
    day_conc = top_day / total_abs if total_abs > 0 else 0

    return {
        "label": label, "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2), "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(wr * 100, 1),
        "profit_factor": round(pf, 2),
        "total_return_pct": round((total_return - 1) * 100, 2),
        "final_equity": round(eq["equity"].iloc[-1], 2),
        "n_days": len(eq), "years": round(total_years, 2),
        "day_conc": round(day_conc, 4),
    }


def apply_tiered_ba_haircut(trades_df, ticker_tier_map, ba_costs):
    """Apply per-ticker BA costs based on tier membership."""
    if trades_df.empty:
        return {"honest_sharpe": 0, "honest_sortino": 0, "total_pnl_after_ba": 0}

    trades = trades_df.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])

    # Compute per-trade BA haircut
    def get_ba(tk):
        tier = ticker_tier_map.get(tk, "tier1")
        return ba_costs.get(tier, 0.05)

    trades["ba_frac"] = trades["ticker"].apply(get_ba)

    # BA cost: crossing spread on both legs at open
    haircut = trades["net_credit"].abs() * trades["ba_frac"] * 2  # 2 legs at open
    # For early-closed trades, also at close
    closed_early = trades["exit_type"].isin(["profit_take"])
    haircut[closed_early] += trades.loc[closed_early, "net_credit"].abs() * trades.loc[closed_early, "ba_frac"] * 2

    adjusted_pnl = trades["realized_pnl"] - haircut
    daily = adjusted_pnl.groupby(trades["close_date"]).sum()

    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 10 and daily.std() > 0 else 0
    sortino_denom = daily[daily < 0].std()
    sortino = daily.mean() / sortino_denom * np.sqrt(252) if sortino_denom > 0 else 0

    cum = daily.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    max_dd = dd.min()

    # Per-tier breakdown
    tier_stats = {}
    for tier_name in ba_costs:
        tier_mask = trades["tier"] == tier_name
        if tier_mask.sum() == 0:
            continue
        tier_pnl = adjusted_pnl[tier_mask]
        tier_stats[tier_name] = {
            "n_trades": int(tier_mask.sum()),
            "total_pnl": round(float(tier_pnl.sum()), 0),
            "avg_pnl": round(float(tier_pnl.mean()), 2),
            "win_rate": round(float((tier_pnl > 0).mean() * 100), 1),
            "ba_cost_total": round(float(haircut[tier_mask].sum()), 0),
        }

    return {
        "honest_sharpe": round(float(sharpe), 2),
        "honest_sortino": round(float(sortino), 2),
        "total_pnl_after_ba": round(float(adjusted_pnl.sum()), 0),
        "ba_haircut_total": round(float(haircut.sum()), 0),
        "n_trades": len(trades),
        "win_rate_after_ba": round(float((adjusted_pnl > 0).mean() * 100), 1),
        "max_dd_pnl": round(float(max_dd), 0) if not np.isnan(max_dd) else 0,
        "per_tier": tier_stats,
    }


def analyze_by_sector(trades_df, ticker_tier_map, ba_costs):
    """Per-sector P&L analysis with BA costs applied."""
    if trades_df.empty:
        return {}

    trades = trades_df.copy()

    def get_ba(tk):
        tier = ticker_tier_map.get(tk, "tier1")
        return ba_costs.get(tier, 0.05)

    trades["ba_frac"] = trades["ticker"].apply(get_ba)
    haircut = trades["net_credit"].abs() * trades["ba_frac"] * 2
    closed_early = trades["exit_type"].isin(["profit_take"])
    haircut[closed_early] += trades.loc[closed_early, "net_credit"].abs() * trades.loc[closed_early, "ba_frac"] * 2
    trades["adjusted_pnl"] = trades["realized_pnl"] - haircut

    results = {}
    for sector, grp in trades.groupby("sector"):
        results[sector] = {
            "n_trades": len(grp),
            "total_pnl_raw": round(float(grp["realized_pnl"].sum()), 0),
            "total_pnl_honest": round(float(grp["adjusted_pnl"].sum()), 0),
            "win_rate_raw": round(float((grp["realized_pnl"] > 0).mean() * 100), 1),
            "win_rate_honest": round(float((grp["adjusted_pnl"] > 0).mean() * 100), 1),
            "avg_pnl_honest": round(float(grp["adjusted_pnl"].mean()), 2),
        }
    return dict(sorted(results.items(), key=lambda x: x[1]["total_pnl_honest"], reverse=True))


def analyze_by_beta_bucket(trades_df, fund_df, ticker_tier_map, ba_costs, n_buckets=4):
    """Split tickers into beta quartiles and compare performance."""
    if trades_df.empty:
        return {}

    beta_col = "beta_realized" if "beta_realized" in fund_df.columns else "beta"
    beta_map = dict(zip(fund_df["ticker"], fund_df[beta_col]))

    trades = trades_df.copy()
    trades["beta"] = trades["ticker"].map(beta_map).fillna(1.0)

    def get_ba(tk):
        tier = ticker_tier_map.get(tk, "tier1")
        return ba_costs.get(tier, 0.05)

    trades["ba_frac"] = trades["ticker"].apply(get_ba)
    haircut = trades["net_credit"].abs() * trades["ba_frac"] * 2
    closed_early = trades["exit_type"].isin(["profit_take"])
    haircut[closed_early] += trades.loc[closed_early, "net_credit"].abs() * trades.loc[closed_early, "ba_frac"] * 2
    trades["adjusted_pnl"] = trades["realized_pnl"] - haircut

    # Create beta quartiles
    trades["beta_bucket"] = pd.qcut(trades["beta"], n_buckets, labels=False, duplicates="drop")

    results = {}
    for bucket, grp in trades.groupby("beta_bucket"):
        beta_range = f"{grp['beta'].min():.2f}-{grp['beta'].max():.2f}"
        results[f"Q{int(bucket)+1} (beta {beta_range})"] = {
            "n_trades": len(grp),
            "n_tickers": grp["ticker"].nunique(),
            "avg_beta": round(float(grp["beta"].mean()), 2),
            "total_pnl_raw": round(float(grp["realized_pnl"].sum()), 0),
            "total_pnl_honest": round(float(grp["adjusted_pnl"].sum()), 0),
            "win_rate_honest": round(float((grp["adjusted_pnl"] > 0).mean() * 100), 1),
            "avg_pnl_honest": round(float(grp["adjusted_pnl"].mean()), 2),
        }
    return results


def compute_diversification_metrics(trades_df):
    """Measure how well-diversified the P&L is across tickers and sectors."""
    if trades_df.empty:
        return {}

    trades = trades_df.copy()
    trades["close_date"] = pd.to_datetime(trades["close_date"])

    # Ticker concentration: Herfindahl index of P&L
    ticker_pnl = trades.groupby("ticker")["realized_pnl"].sum().abs()
    total = ticker_pnl.sum()
    if total > 0:
        shares = ticker_pnl / total
        hhi_ticker = float((shares ** 2).sum())
    else:
        hhi_ticker = 1.0

    # Sector concentration
    sector_pnl = trades.groupby("sector")["realized_pnl"].sum().abs()
    total_s = sector_pnl.sum()
    if total_s > 0:
        shares_s = sector_pnl / total_s
        hhi_sector = float((shares_s ** 2).sum())
    else:
        hhi_sector = 1.0

    # Average pairwise correlation of daily per-ticker P&L
    daily_by_ticker = trades.pivot_table(
        index="close_date", columns="ticker", values="realized_pnl",
        aggfunc="sum", fill_value=0
    )
    if daily_by_ticker.shape[1] > 1:
        corr_matrix = daily_by_ticker.corr()
        # Average off-diagonal correlation
        n = corr_matrix.shape[0]
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr = float(corr_matrix.values[mask].mean())
    else:
        avg_corr = 1.0

    return {
        "hhi_ticker": round(hhi_ticker, 4),
        "hhi_sector": round(hhi_sector, 4),
        "avg_pairwise_corr": round(avg_corr, 3),
        "n_unique_tickers": int(trades["ticker"].nunique()),
        "n_unique_sectors": int(trades["sector"].nunique()),
        "interpretation": {
            "hhi_ticker": "Lower = more diversified. Perfect diversification across N names = 1/N",
            "hhi_sector": "Lower = more diversified across sectors",
            "avg_pairwise_corr": "Lower = less correlated returns = better diversification",
        }
    }


# ── Main ──

def main():
    t0 = time.time()
    print("=" * 70)
    print("BPS EXPANDED UNIVERSE STUDY (HC #660)")
    print("=" * 70)

    prices, iv, macro, fund, universe, earnings = load_all_data()

    # Build tier maps
    ticker_tier = {}
    for tk in TIER1_TICKERS:
        ticker_tier[tk] = "tier1"
    for tk in TIER2_TICKERS:
        ticker_tier[tk] = "tier2"
    for tk in TIER3_TICKERS:
        ticker_tier[tk] = "tier3"

    available_tickers = set(universe["ticker"].values)
    tier1_avail = [tk for tk in TIER1_TICKERS if tk in available_tickers]
    tier2_avail = [tk for tk in TIER2_TICKERS if tk in available_tickers]
    tier3_avail = [tk for tk in TIER3_TICKERS if tk in available_tickers]

    print(f"\nTier 1 (large-cap, 5% BA): {len(tier1_avail)} tickers")
    print(f"Tier 2 (mid-cap, 7% BA): {len(tier2_avail)} tickers")
    print(f"Tier 3 (small-cap high-beta, 9% BA): {len(tier3_avail)} tickers")

    # Config: d30/$15/25% margin/65% PT per optimal sweep
    config = dict(
        spread_width=15.0, put_delta=0.25, dte_target=30,
        profit_take=0.65, margin_cap=0.25, max_concurrent=60,
        per_name_pct=0.05, vix_gate=30.0, vix_scale=True,
        daily_cb=0.02,
    )

    results = {}

    # ══ TEST 1: Tier 1 only (baseline) ══
    print("\n" + "=" * 70)
    print("TEST 1: Tier 1 Only (70 large-cap, 5% BA) — BASELINE")
    print("=" * 70)
    uni1 = universe[universe["ticker"].isin(tier1_avail)]
    r1 = run_bps_tiered(
        prices[prices["ticker"].isin(tier1_avail)],
        iv[iv["ticker"].isin(tier1_avail)],
        macro, fund, uni1, earnings,
        ticker_tier_map=ticker_tier,
        ba_costs=BA_COST,
        label="Tier 1 Only (large-cap)",
        **config
    )
    print_result_summary("Tier 1 Only", r1)
    results["tier1_only"] = extract_result(r1, tier1_avail)

    # ══ TEST 2: Tier 1 + Tier 2 ══
    print("\n" + "=" * 70)
    print("TEST 2: Tier 1 + Tier 2 (large + mid-cap)")
    print("=" * 70)
    t12_tickers = tier1_avail + tier2_avail
    uni12 = universe[universe["ticker"].isin(t12_tickers)]
    r12 = run_bps_tiered(
        prices[prices["ticker"].isin(t12_tickers)],
        iv[iv["ticker"].isin(t12_tickers)],
        macro, fund, uni12, earnings,
        ticker_tier_map=ticker_tier,
        ba_costs=BA_COST,
        label="Tier 1+2 (large + mid-cap)",
        **config
    )
    print_result_summary("Tier 1+2", r12)
    results["tier1_plus_tier2"] = extract_result(r12, t12_tickers)

    # ══ TEST 3: All tiers combined ══
    print("\n" + "=" * 70)
    print("TEST 3: Tier 1 + 2 + 3 (full expanded universe)")
    print("=" * 70)
    t123_tickers = tier1_avail + tier2_avail + tier3_avail
    uni123 = universe[universe["ticker"].isin(t123_tickers)]
    r123 = run_bps_tiered(
        prices[prices["ticker"].isin(t123_tickers)],
        iv[iv["ticker"].isin(t123_tickers)],
        macro, fund, uni123, earnings,
        ticker_tier_map=ticker_tier,
        ba_costs=BA_COST,
        label="Tier 1+2+3 (full expanded)",
        **config
    )
    print_result_summary("All Tiers", r123)
    results["all_tiers"] = extract_result(r123, t123_tickers)

    # ══ TEST 4: Tier 2 only (isolate mid-cap contribution) ══
    print("\n" + "=" * 70)
    print("TEST 4: Tier 2 Only (mid-cap isolation)")
    print("=" * 70)
    uni2 = universe[universe["ticker"].isin(tier2_avail)]
    r2 = run_bps_tiered(
        prices[prices["ticker"].isin(tier2_avail)],
        iv[iv["ticker"].isin(tier2_avail)],
        macro, fund, uni2, earnings,
        ticker_tier_map=ticker_tier,
        ba_costs=BA_COST,
        label="Tier 2 Only (mid-cap)",
        **config
    )
    print_result_summary("Tier 2 Only", r2)
    results["tier2_only"] = extract_result(r2, tier2_avail)

    # ══ TEST 5: Tier 3 only (isolate high-beta contribution) ══
    print("\n" + "=" * 70)
    print("TEST 5: Tier 3 Only (high-beta isolation)")
    print("=" * 70)
    uni3 = universe[universe["ticker"].isin(tier3_avail)]
    r3 = run_bps_tiered(
        prices[prices["ticker"].isin(tier3_avail)],
        iv[iv["ticker"].isin(tier3_avail)],
        macro, fund, uni3, earnings,
        ticker_tier_map=ticker_tier,
        ba_costs=BA_COST,
        label="Tier 3 Only (high-beta)",
        **config
    )
    print_result_summary("Tier 3 Only", r3)
    results["tier3_only"] = extract_result(r3, tier3_avail)

    # ══ ANALYSIS: Sector breakdown for combined universe ══
    print("\n" + "=" * 70)
    print("SECTOR BREAKDOWN (All Tiers Combined)")
    print("=" * 70)
    sector_breakdown = analyze_by_sector(r123["trades"], ticker_tier, BA_COST)
    print(f"{'Sector':<25} {'Trades':>7} {'Raw P&L':>10} {'Honest P&L':>12} {'WR(h)':>7}")
    print("-" * 65)
    for sector, stats in sector_breakdown.items():
        print(f"{sector:<25} {stats['n_trades']:>7} ${stats['total_pnl_raw']:>9,} "
              f"${stats['total_pnl_honest']:>11,} {stats['win_rate_honest']:>6.1f}%")
    results["sector_breakdown"] = sector_breakdown

    # ══ ANALYSIS: Beta buckets ══
    print("\n" + "=" * 70)
    print("BETA BUCKET ANALYSIS (All Tiers Combined)")
    print("=" * 70)
    beta_analysis = analyze_by_beta_bucket(r123["trades"], fund, ticker_tier, BA_COST)
    print(f"{'Bucket':<30} {'Trades':>7} {'Tickers':>8} {'Honest P&L':>12} {'WR':>7} {'Avg Beta':>9}")
    print("-" * 75)
    for bucket, stats in beta_analysis.items():
        print(f"{bucket:<30} {stats['n_trades']:>7} {stats['n_tickers']:>8} "
              f"${stats['total_pnl_honest']:>11,} {stats['win_rate_honest']:>6.1f}% {stats['avg_beta']:>9.2f}")
    results["beta_buckets"] = beta_analysis

    # ══ ANALYSIS: Diversification ══
    print("\n" + "=" * 70)
    print("DIVERSIFICATION COMPARISON")
    print("=" * 70)
    for name, r in [("Tier 1 Only", r1), ("Tier 1+2", r12), ("All Tiers", r123)]:
        div = compute_diversification_metrics(r["trades"])
        print(f"\n  {name}:")
        print(f"    Ticker HHI: {div.get('hhi_ticker', 'N/A'):.4f} (lower = more diversified)")
        print(f"    Sector HHI: {div.get('hhi_sector', 'N/A'):.4f}")
        print(f"    Avg pairwise corr: {div.get('avg_pairwise_corr', 'N/A'):.3f}")
        print(f"    Unique tickers traded: {div.get('n_unique_tickers', 0)}")
        print(f"    Unique sectors: {div.get('n_unique_sectors', 0)}")
        results[f"diversification_{name.lower().replace(' ', '_').replace('+', '_')}"] = div

    # ══ Per-ticker attribution (top and bottom) ══
    if not r123["trades"].empty:
        print("\n" + "=" * 70)
        print("PER-TICKER ATTRIBUTION (All Tiers, top/bottom 10)")
        print("=" * 70)
        tk_pnl = r123["trades"].groupby("ticker")["realized_pnl"].agg(["sum", "count", "mean"])
        tk_pnl = tk_pnl.sort_values("sum", ascending=False)
        print(f"\nTOP 10:")
        for tk, row in tk_pnl.head(10).iterrows():
            tier = ticker_tier.get(tk, "?")
            print(f"  {tk:<6} ({tier}): ${row['sum']:>9,.0f} over {int(row['count'])} trades, avg ${row['mean']:>6,.0f}")
        print(f"\nBOTTOM 10:")
        for tk, row in tk_pnl.tail(10).iterrows():
            tier = ticker_tier.get(tk, "?")
            print(f"  {tk:<6} ({tier}): ${row['sum']:>9,.0f} over {int(row['count'])} trades, avg ${row['mean']:>6,.0f}")

    # ══ FINAL COMPARISON TABLE ══
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Universe':<25} {'Tickers':>8} {'CAGR':>8} {'Sharpe':>8} {'H.Sharpe':>9} "
          f"{'MaxDD':>8} {'WR':>6} {'PF':>6}")
    print("-" * 85)
    for name, key in [("Tier 1 (baseline)", "tier1_only"),
                       ("Tier 1+2", "tier1_plus_tier2"),
                       ("All Tiers", "all_tiers"),
                       ("Tier 2 Only", "tier2_only"),
                       ("Tier 3 Only", "tier3_only")]:
        d = results.get(key, {})
        m = d.get("raw_metrics", {})
        h = d.get("honest_metrics", {})
        print(f"{name:<25} {d.get('n_tickers', '?'):>8} {m.get('cagr_pct', '?'):>7}% "
              f"{m.get('sharpe', '?'):>8} {h.get('honest_sharpe', '?'):>9} "
              f"{m.get('max_dd_pct', '?'):>7}% {m.get('win_rate_pct', '?'):>5}% "
              f"{m.get('profit_factor', '?'):>6}")

    # Key question: does adding higher-beta names help?
    t1_hs = results.get("tier1_only", {}).get("honest_metrics", {}).get("honest_sharpe", 0)
    t12_hs = results.get("tier1_plus_tier2", {}).get("honest_metrics", {}).get("honest_sharpe", 0)
    all_hs = results.get("all_tiers", {}).get("honest_metrics", {}).get("honest_sharpe", 0)

    print(f"\n{'='*70}")
    print("KEY FINDINGS:")
    print(f"  Tier 1 baseline honest Sharpe: {t1_hs}")
    print(f"  Adding Tier 2 (mid-cap): honest Sharpe {'IMPROVES' if t12_hs > t1_hs else 'WORSENS'} to {t12_hs}")
    print(f"  Adding Tier 3 (high-beta): honest Sharpe {'IMPROVES' if all_hs > t12_hs else 'WORSENS'} to {all_hs}")
    if t12_hs > t1_hs:
        print(f"  -> Mid-cap names ADD VALUE despite higher BA costs")
    else:
        print(f"  -> Mid-cap names HURT returns due to higher BA costs eating premiums")
    if all_hs > t12_hs:
        print(f"  -> High-beta names ADD VALUE despite 9% BA")
    else:
        print(f"  -> High-beta names HURT returns — 9% BA eats the richer premiums")
    print(f"{'='*70}")

    # ── Save results ──
    # Save equity curves
    for name, r in [("tier1", r1), ("tier1_2", r12), ("all_tiers", r123),
                     ("tier2", r2), ("tier3", r3)]:
        if "equity_curve" in r:
            r["equity_curve"].to_parquet(OUTPUT / f"eq_{name}.parquet", index=False)

    # Save trades
    if not r123["trades"].empty:
        r123["trades"].to_parquet(OUTPUT / f"trades_all_tiers.parquet", index=False)

    # JSON summary
    def clean(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj)
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return obj

    def clean_dict(d):
        if isinstance(d, dict): return {k: clean_dict(v) for k, v in d.items()}
        if isinstance(d, list): return [clean_dict(v) for v in d]
        return clean(d)

    summary = {
        "generated": pd.Timestamp.now().isoformat(),
        "config": {
            "spread_width": 15, "put_delta": 0.25, "dte_target": 30,
            "profit_take": 0.65, "margin_cap": 0.25,
            "vix_gate": 30, "vix_scale": True, "daily_cb": 0.02,
        },
        "ba_costs": BA_COST,
        "tier_sizes": {
            "tier1": len(tier1_avail),
            "tier2": len(tier2_avail),
            "tier3": len(tier3_avail),
            "total": len(t123_tickers),
        },
        "results": clean_dict(results),
    }

    with open(OUTPUT / "expanded_universe_v2_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed / 60:.1f} minutes")
    print(f"Results saved to {OUTPUT}")


def print_result_summary(name, r):
    if "error" in r:
        print(f"  ERROR: {r['error']}")
        return
    m = r["metrics"]
    h = r.get("honest_metrics", {})
    print(f"  Raw:    CAGR {m.get('cagr_pct')}%, Sharpe {m.get('sharpe')}, "
          f"Sortino {m.get('sortino')}, MaxDD {m.get('max_dd_pct')}%, "
          f"PF {m.get('profit_factor')}, WR {m.get('win_rate_pct')}%, "
          f"Trades {m.get('n_trades')}")
    print(f"  Honest: Sharpe {h.get('honest_sharpe')}, Sortino {h.get('honest_sortino')}, "
          f"P&L ${h.get('total_pnl_after_ba', 0):,.0f}, "
          f"BA cost ${h.get('ba_haircut_total', 0):,.0f}, "
          f"WR {h.get('win_rate_after_ba', 0):.1f}%")
    tier_stats = h.get("per_tier", {})
    for tier, ts in tier_stats.items():
        print(f"    {tier}: {ts['n_trades']} trades, P&L ${ts['total_pnl']:,.0f}, "
              f"WR {ts['win_rate']:.1f}%, BA cost ${ts['ba_cost_total']:,.0f}")


def extract_result(r, tickers):
    if "error" in r:
        return {"error": r["error"]}
    return {
        "n_tickers": len(tickers),
        "raw_metrics": {k: v for k, v in r["metrics"].items()
                       if not isinstance(v, (pd.DataFrame, pd.Series))},
        "honest_metrics": r.get("honest_metrics", {}),
    }


if __name__ == "__main__":
    main()
