#!/usr/bin/env python3
"""
wheel_universe_v3_expand.py — HC #660: Expand wheel universe with high-beta,
mid-cap, multi-industry names. Downloads data via yfinance and caches it.

Phase 1: Download + cache price data for ~100 additional tickers
Phase 2: Run v2 margin portfolio engine on full expanded universe
Phase 3: Report by sector + beta bucket
"""
import sys
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "wheel_universe_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [WHEEL-V3] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger('WHEEL-V3')

# ========================= NEW HIGH-BETA / MID-CAP TICKERS ====================
# HC #660: higher-beta, different industries, mid-cap growth, biotech, semis,
# energy, REITs, small-cap growth — anything with enough option liquidity.
# These are all optionable with decent volume.

NEW_TICKERS = {
    # Biotech / Pharma (high beta, high vol = rich premiums)
    'MRNA': 'Biotech', 'CRSP': 'Biotech', 'EXAS': 'Biotech',
    'RARE': 'Biotech', 'BMRN': 'Biotech', 'SGEN': 'Biotech',
    'ARWR': 'Biotech', 'IONS': 'Biotech', 'ALNY': 'Biotech',
    'REGN': 'Biotech', 'NBIX': 'Biotech', 'PCVX': 'Biotech',

    # Semiconductors (cyclical, high beta)
    'MRVL': 'Semiconductors', 'KLAC': 'Semiconductors', 'LRCX': 'Semiconductors',
    'NXPI': 'Semiconductors', 'MCHP': 'Semiconductors', 'TER': 'Semiconductors',
    'ONTO': 'Semiconductors', 'MPWR': 'Semiconductors', 'WOLF': 'Semiconductors',

    # Energy (cyclical, commodity-linked)
    'DVN': 'Energy', 'FANG': 'Energy', 'MPC': 'Energy',
    'PSX': 'Energy', 'EOG': 'Energy', 'PXD': 'Energy',
    'HAL': 'Energy', 'SLB': 'Energy', 'OXY': 'Energy',
    'AR': 'Energy', 'RRC': 'Energy',

    # REITs (higher yield, different risk profile)
    'O': 'REIT', 'PLD': 'REIT', 'EQIX': 'REIT',
    'AVB': 'REIT', 'ESS': 'REIT', 'MAA': 'REIT',
    'MPW': 'REIT', 'PEAK': 'REIT', 'KIM': 'REIT',

    # Consumer / Retail (mid-cap, volatile)
    'DASH': 'Consumer', 'ETSY': 'Consumer', 'RVLV': 'Consumer',
    'DKS': 'Consumer', 'GPS': 'Consumer', 'LULU': 'Consumer',
    'CROX': 'Consumer', 'EL': 'Consumer',

    # Industrials / Transports (cyclical beta)
    'GNRC': 'Industrials', 'XPO': 'Industrials', 'DAL': 'Industrials',
    'LUV': 'Industrials', 'BLDR': 'Industrials', 'URI': 'Industrials',

    # Fintech / Growth (high vol, high beta)
    'COIN': 'Fintech', 'HOOD': 'Fintech', 'SOFI': 'Fintech',
    'NU': 'Fintech', 'MARA': 'Fintech',

    # Software / Cloud (growth, volatile)
    'SNOW': 'Software', 'DDOG': 'Software', 'NET': 'Software',
    'CRWD': 'Software', 'MDB': 'Software', 'OKTA': 'Software',
    'ZM': 'Software', 'DOCU': 'Software',

    # Mining / Materials (commodity, cyclical)
    'FCX': 'Materials', 'NEM': 'Materials', 'CLF': 'Materials',
    'X': 'Materials', 'AA': 'Materials',

    # Cannabis / High-vol speculative (if optionable)
    'TLRY': 'Cannabis',

    # Misc high-beta
    'RIVN': 'EV', 'LCID': 'EV', 'NIO': 'EV',
    'PLUG': 'CleanEnergy', 'FSLR': 'CleanEnergy', 'ENPH': 'CleanEnergy',
}

def download_new_data():
    """Download price data for new tickers via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    # Load existing tickers to avoid re-downloading
    existing = set()
    p1_file = CACHE / "prices.parquet"
    p2_file = CACHE / "prices_expanded.parquet"
    if p1_file.exists():
        existing |= set(pd.read_parquet(p1_file)["ticker"].unique())
    if p2_file.exists():
        existing |= set(pd.read_parquet(p2_file)["ticker"].unique())

    new_only = {t: s for t, s in NEW_TICKERS.items() if t not in existing}
    # Also include existing ones that might be in NEW_TICKERS for sector tagging
    already_have = {t: s for t, s in NEW_TICKERS.items() if t in existing}

    log.info(f"Already have {len(already_have)} of {len(NEW_TICKERS)} new tickers in cache")
    log.info(f"Need to download: {len(new_only)} tickers")

    if not new_only:
        log.info("All tickers already cached. Skipping download.")
        return

    tickers_list = sorted(new_only.keys())
    log.info(f"Downloading {len(tickers_list)} tickers: {tickers_list}")

    all_frames = []
    batch_size = 20
    for i in range(0, len(tickers_list), batch_size):
        batch = tickers_list[i:i+batch_size]
        log.info(f"  Batch {i//batch_size + 1}: {batch}")
        try:
            data = yf.download(batch, start="2015-01-01", auto_adjust=True,
                             threads=True, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    if t in data["Close"].columns:
                        df = data["Close"][[t]].dropna().reset_index()
                        df.columns = ["date", "close"]
                        df["ticker"] = t
                        all_frames.append(df)
            else:
                # Single ticker
                if len(batch) == 1:
                    df = data[["Close"]].dropna().reset_index()
                    df.columns = ["date", "close"]
                    df["ticker"] = batch[0]
                    all_frames.append(df)
        except Exception as e:
            log.warning(f"  Failed batch: {e}")
        time.sleep(1)  # Rate limit courtesy

    if not all_frames:
        log.warning("No data downloaded!")
        return

    new_prices = pd.concat(all_frames, ignore_index=True)
    new_prices["date"] = pd.to_datetime(new_prices["date"])
    if new_prices["date"].dt.tz is not None:
        new_prices["date"] = new_prices["date"].dt.tz_localize(None)

    # Save as prices_v3_expansion.parquet
    out_file = CACHE / "prices_v3_expansion.parquet"
    new_prices.to_parquet(out_file, index=False)
    log.info(f"Saved {len(new_prices)} rows for {new_prices['ticker'].nunique()} tickers to {out_file}")

    return new_prices


def load_all_data(start_date="2019-01-01"):
    """Load all price data: original 70 + expanded 127 + v3 expansion."""
    import math

    frames = []

    # Original prices
    p1_file = CACHE / "prices.parquet"
    if p1_file.exists():
        p1 = pd.read_parquet(p1_file)[["ticker", "date", "close"]].copy()
        p1["date"] = pd.to_datetime(p1["date"], utc=False)
        if p1["date"].dt.tz is not None:
            p1["date"] = p1["date"].dt.tz_localize(None)
        frames.append(p1)

    # Expanded prices
    p2_file = CACHE / "prices_expanded.parquet"
    if p2_file.exists():
        p2 = pd.read_parquet(p2_file)
        if "Close" in p2.columns:
            p2 = p2.rename(columns={"Close": "close"})
        p2 = p2[["ticker", "date", "close"]].copy()
        p2["date"] = pd.to_datetime(p2["date"], utc=False)
        if p2["date"].dt.tz is not None:
            p2["date"] = p2["date"].dt.tz_localize(None)
        frames.append(p2)

    # V3 expansion
    p3_file = CACHE / "prices_v3_expansion.parquet"
    if p3_file.exists():
        p3 = pd.read_parquet(p3_file)[["ticker", "date", "close"]].copy()
        p3["date"] = pd.to_datetime(p3["date"], utc=False)
        if p3["date"].dt.tz is not None:
            p3["date"] = p3["date"].dt.tz_localize(None)
        frames.append(p3)

    if not frames:
        log.error("No price data found!")
        sys.exit(1)

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"]).reset_index(drop=True)
    prices = prices.dropna(subset=["close"])
    prices = prices[prices["close"] > 0]
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices[prices["date"] >= pd.Timestamp(start_date)]

    # Compute 20-day realized vol
    prices["log_ret"] = prices.groupby("ticker")["close"].transform(lambda x: np.log(x / x.shift(1)))
    prices["sigma"] = prices.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(20, min_periods=15).std() * np.sqrt(252)
    )
    prices["sigma"] = prices["sigma"].clip(lower=0.05, upper=2.0)

    # VIX
    macro_file = CACHE / "macro.parquet"
    if macro_file.exists():
        macro = pd.read_parquet(macro_file)[["date", "vix"]].copy()
        macro["date"] = pd.to_datetime(macro["date"], utc=False)
        if macro["date"].dt.tz is not None:
            macro["date"] = macro["date"].dt.tz_localize(None)
        prices = prices.merge(macro, on="date", how="left")
        prices["vix"] = prices["vix"].ffill().fillna(20.0)
    else:
        prices["vix"] = 20.0

    # Beta computation (vs SPY)
    spy = prices[prices["ticker"] == "SPY"][["date", "close"]].copy()
    spy = spy.sort_values("date").drop_duplicates("date")
    spy["spy_ret"] = np.log(spy["close"] / spy["close"].shift(1))
    spy_ret_map = dict(zip(spy["date"], spy["spy_ret"]))

    prices["spy_ret"] = prices["date"].map(spy_ret_map)

    # Rolling 60-day beta
    def calc_beta(group):
        g = group[["log_ret", "spy_ret"]].dropna()
        if len(g) < 40:
            return pd.Series(np.nan, index=group.index, name="beta")
        beta = g["log_ret"].rolling(60, min_periods=40).corr(g["spy_ret"]) * \
               (g["log_ret"].rolling(60, min_periods=40).std() /
                g["spy_ret"].rolling(60, min_periods=40).std().clip(lower=1e-6))
        return beta.reindex(group.index)

    prices["beta"] = prices.groupby("ticker", group_keys=False).apply(calc_beta)
    prices["beta"] = prices["beta"].clip(lower=-2, upper=5)

    # SPY regime
    spy_file = CACHE / "spy_prices.parquet"
    if spy_file.exists():
        spy_df = pd.read_parquet(spy_file)[["date", "close"]].copy()
        spy_df["date"] = pd.to_datetime(spy_df["date"], utc=False)
        if spy_df["date"].dt.tz is not None:
            spy_df["date"] = spy_df["date"].dt.tz_localize(None)
    else:
        spy_df = prices[prices["ticker"] == "SPY"][["date", "close"]].copy()

    spy_df = spy_df.sort_values("date").drop_duplicates("date")
    spy_df["spy_sma"] = spy_df["close"].rolling(50, min_periods=50).mean()
    spy_df["bear"] = (spy_df["close"] < spy_df["spy_sma"]).astype(int)
    spy_regime = spy_df[["date", "bear"]].copy()

    # Build sector map (combine old basket + new)
    ALL_SECTORS = {}
    # Load from existing wheel scripts
    try:
        from wheel_elite40_v2_margin import BASKET as OLD_BASKET
        ALL_SECTORS.update(OLD_BASKET)
    except:
        pass
    ALL_SECTORS.update(NEW_TICKERS)

    n_tickers = prices["ticker"].nunique()
    log.info(f"Total universe: {n_tickers} tickers, date range: {prices['date'].min()} to {prices['date'].max()}")

    return prices, spy_regime, ALL_SECTORS


def run_portfolio_v3(prices_df, spy_regime, sector_map,
                     starting_cash=100_000, margin_cap=0.40, per_name_pct=0.03,
                     put_delta=0.25, dte_target=14, profit_take=0.65,
                     bear_mode="liq_csp_only", max_assignments_5d=3,
                     max_share_positions=5, loss_cut_pct=-0.15,
                     min_price=10.0, max_price=500.0,
                     earnings_lookup=None, earnings_buffer_days=2):
    """
    Portfolio-level wheel with proper margin accounting.
    Adapted from elite40_v2 but for arbitrary universe size.

    Args:
        earnings_lookup: Optional dict {ticker: np.array of earnings dates}.
            If provided, skip CSP entries where ticker has earnings within
            [today - buffer, today + DTE + buffer].
        earnings_buffer_days: Buffer days around earnings dates to avoid.
    """
    import math
    from collections import deque
    from dataclasses import dataclass

    MARGIN_REQ_PCT = 0.20
    COST_PER_CONTRACT = 0.65
    SLIPPAGE_FRAC = 0.025
    SLIPPAGE_MIN = 0.03
    VIX_MAX = 35.0
    RISK_FREE = 0.04
    CALL_DELTA = 0.30

    def _Phi(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if kind == "put":
            return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
        return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)

    def find_strike(S, sigma, T, delta_target, kind="put"):
        if T <= 0 or sigma <= 0:
            return S
        if kind == "put":
            lo, hi = S * 0.3, S * 1.0
        else:
            lo, hi = S * 1.0, S * 2.0
        for _ in range(60):
            K = (lo + hi) / 2
            d1 = (math.log(S / K) + (RISK_FREE + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T) + 1e-9)
            if kind == "put":
                delta_abs = _Phi(-d1)
            else:
                delta_abs = _Phi(d1)
            if delta_abs > delta_target:
                if kind == "put":
                    hi = K
                else:
                    lo = K
            else:
                if kind == "put":
                    lo = K
                else:
                    hi = K
        return round(K * 2) / 2

    # Filter to stocks in price range (no penny stocks, no $1000+ stocks that eat margin)
    valid_tickers = set()
    for t, grp in prices_df.groupby("ticker"):
        median_price = grp["close"].median()
        if min_price <= median_price <= max_price:
            valid_tickers.add(t)

    prices_df = prices_df[prices_df["ticker"].isin(valid_tickers)].copy()
    tickers_available = sorted(prices_df["ticker"].unique())
    log.info(f"Portfolio universe: {len(tickers_available)} tickers (price filter: ${min_price}-${max_price})")

    # Build date-indexed lookups
    ticker_data = {}
    for t in tickers_available:
        tdf = prices_df[prices_df["ticker"] == t].set_index("date").sort_index()
        ticker_data[t] = tdf

    spy_map = {}
    if not spy_regime.empty:
        for _, row in spy_regime.iterrows():
            spy_map[pd.Timestamp(row["date"])] = int(row["bear"])

    all_dates = sorted(pd.Timestamp(d) for d in prices_df["date"].unique())

    # State
    cash = float(starting_cash)
    csp_positions = {}
    share_positions = {}
    assignment_dates = deque()

    trades = []
    daily_equity = []

    @dataclass
    class CSPPos:
        ticker: str
        strike: float
        premium: float
        entry_date: object
        expiry_date: object
        margin_held: float

    @dataclass
    class SharePos:
        ticker: str
        shares: int
        cost_basis: float
        entry_date: object
        cc_strike: float = 0.0
        cc_premium: float = 0.0
        cc_expiry: object = None
        has_cc: bool = False

    import random

    for di, date in enumerate(all_dates):
        is_bear = spy_map.get(date, 0) == 1

        # Mark to market
        nav = cash
        for pos in csp_positions.values():
            nav += pos.margin_held
        for t, pos in share_positions.items():
            if t in ticker_data and date in ticker_data[t].index:
                px = ticker_data[t].loc[date]["close"]
                nav += pos.shares * px
            else:
                nav += pos.shares * pos.cost_basis

        daily_equity.append({"date": date, "equity": nav})

        # Process CSP expirations
        expired_csps = [t for t, pos in csp_positions.items() if date >= pos.expiry_date]
        for t in expired_csps:
            pos = csp_positions[t]
            if t not in ticker_data or date not in ticker_data[t].index:
                cash += pos.margin_held
                trades.append({"date": date, "ticker": t, "action": "csp_expired_otm",
                              "pnl": pos.premium * 100})
                del csp_positions[t]
                continue

            px = ticker_data[t].loc[date]["close"]

            if px <= pos.strike:
                # Assignment check
                while assignment_dates and (date - assignment_dates[0]).days > 5:
                    assignment_dates.popleft()

                if (len(assignment_dates) >= max_assignments_5d or
                    len(share_positions) >= max_share_positions):
                    intrinsic = (pos.strike - px) * 100
                    loss = intrinsic - pos.premium * 100 + COST_PER_CONTRACT
                    cash += pos.margin_held
                    cash -= loss
                    trades.append({"date": date, "ticker": t, "action": "assignment_refused",
                                  "pnl": -(loss / 100)})
                    del csp_positions[t]
                    continue

                share_cost = pos.strike * 100 + COST_PER_CONTRACT
                cash += pos.margin_held
                cash -= share_cost
                share_positions[t] = SharePos(
                    ticker=t, shares=100,
                    cost_basis=pos.strike - pos.premium,
                    entry_date=date,
                )
                assignment_dates.append(date)
                trades.append({"date": date, "ticker": t, "action": "assigned"})
                del csp_positions[t]
            else:
                cash += pos.margin_held
                trades.append({"date": date, "ticker": t, "action": "csp_expired_otm",
                              "pnl": pos.premium})
                del csp_positions[t]

        # Process CC expirations
        for t in list(share_positions.keys()):
            pos = share_positions[t]
            if not pos.has_cc or pos.cc_expiry is None or date < pos.cc_expiry:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                pos.has_cc = False
                continue
            px = ticker_data[t].loc[date]["close"]
            if px >= pos.cc_strike:
                proceeds = pos.cc_strike * 100 - COST_PER_CONTRACT
                cash += proceeds
                pnl = (pos.cc_strike - pos.cost_basis) * 100 + pos.cc_premium * 100
                trades.append({"date": date, "ticker": t, "action": "called_away", "pnl": pnl / 100})
                del share_positions[t]
            else:
                pos.cost_basis -= pos.cc_premium
                pos.has_cc = False
                trades.append({"date": date, "ticker": t, "action": "cc_expired_otm"})

        # Loss-cut on shares
        for t in list(share_positions.keys()):
            pos = share_positions[t]
            if pos.has_cc:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                continue
            px = ticker_data[t].loc[date]["close"]
            pnl_pct = (px - pos.cost_basis) / pos.cost_basis if pos.cost_basis > 0 else 0
            if pnl_pct <= loss_cut_pct:
                proceeds = px * 100 - COST_PER_CONTRACT
                cash += proceeds
                realized = (px - pos.cost_basis) * 100
                trades.append({"date": date, "ticker": t, "action": "loss_cut", "pnl": realized / 100})
                del share_positions[t]

        # Profit-take on CSPs
        for t in list(csp_positions.keys()):
            pos = csp_positions[t]
            if t not in ticker_data or date not in ticker_data[t].index:
                continue
            row = ticker_data[t].loc[date]
            px = row["close"]
            sigma = row.get("sigma", 0.3) if isinstance(row, pd.Series) else 0.3
            dte_remain = (pos.expiry_date - date).days
            if dte_remain <= 0:
                continue
            current_val = bs_price(px, pos.strike, dte_remain / 365, sigma)
            if current_val <= pos.premium * (1 - profit_take):
                buyback = current_val * 100 + COST_PER_CONTRACT
                cash += pos.margin_held
                cash -= buyback
                profit = (pos.premium - current_val) * 100 - 2 * COST_PER_CONTRACT
                trades.append({"date": date, "ticker": t, "action": "profit_take", "pnl": profit / 100})
                del csp_positions[t]

        # Bear protection
        if is_bear and bear_mode == "liq_csp_only":
            for t in list(csp_positions.keys()):
                pos = csp_positions[t]
                if t not in ticker_data or date not in ticker_data[t].index:
                    continue
                row = ticker_data[t].loc[date]
                px = row["close"]
                sigma = row.get("sigma", 0.3) if isinstance(row, pd.Series) else 0.3
                dte_remain = (pos.expiry_date - date).days
                if dte_remain <= 0:
                    continue
                current_val = bs_price(px, pos.strike, dte_remain / 365, sigma)
                buyback = current_val * 100 + COST_PER_CONTRACT
                cash += pos.margin_held
                cash -= buyback
                pnl = (pos.premium - current_val) * 100 - 2 * COST_PER_CONTRACT
                trades.append({"date": date, "ticker": t, "action": "bear_close", "pnl": pnl / 100})
                del csp_positions[t]

        # Write CCs on shares
        for t in list(share_positions.keys()):
            pos = share_positions[t]
            if pos.has_cc:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                continue
            row = ticker_data[t].loc[date]
            px = row["close"]
            sigma = row.get("sigma", 0.3) if isinstance(row, pd.Series) else 0.3
            if pd.isna(sigma) or sigma < 0.05:
                continue
            T = dte_target / 365
            K = find_strike(px, sigma, T, CALL_DELTA, kind="call")
            premium = bs_price(px, K, T, sigma, kind="call")
            premium = max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)
            if premium < 0.10:
                continue
            expiry = date + pd.Timedelta(days=dte_target)
            pos.has_cc = True
            pos.cc_strike = K
            pos.cc_premium = premium
            pos.cc_expiry = expiry
            cash += premium * 100 - COST_PER_CONTRACT
            trades.append({"date": date, "ticker": t, "action": "sell_cc", "premium": premium})

        # Open new CSPs
        if is_bear and bear_mode != "none":
            continue

        current_margin_used = sum(p.margin_held for p in csp_positions.values())
        available_margin = nav * margin_cap - current_margin_used
        per_name_limit = nav * per_name_pct

        rng = random.Random(di)
        candidates = list(tickers_available)
        rng.shuffle(candidates)

        for t in candidates:
            if t in csp_positions or t in share_positions:
                continue
            if t not in ticker_data or date not in ticker_data[t].index:
                continue

            # Earnings avoidance: skip if ticker has earnings within window
            if earnings_lookup is not None and t in earnings_lookup:
                import numpy as _np
                _ed = earnings_lookup[t]
                _ws = _np.datetime64(date) - _np.timedelta64(earnings_buffer_days, 'D')
                _we = _np.datetime64(date) + _np.timedelta64(dte_target + earnings_buffer_days, 'D')
                _idx_s = _np.searchsorted(_ed, _ws, side='left')
                _idx_e = _np.searchsorted(_ed, _we, side='right')
                if _idx_e > _idx_s:
                    continue

            row = ticker_data[t].loc[date]
            px = row["close"]
            sigma = row.get("sigma", 0.3) if isinstance(row, pd.Series) else 0.3
            vix = row.get("vix", 20.0) if isinstance(row, pd.Series) else 20.0

            if vix > VIX_MAX:
                continue
            if pd.isna(sigma) or sigma < 0.05:
                continue

            notional = px * 100
            margin_req = notional * MARGIN_REQ_PCT

            if margin_req > per_name_limit:
                continue
            if margin_req > available_margin:
                continue
            if cash < margin_req:
                continue

            T = dte_target / 365
            K = find_strike(px, sigma, T, put_delta, kind="put")
            premium = bs_price(px, K, T, sigma)
            premium = max(premium * (1 - SLIPPAGE_FRAC), premium - SLIPPAGE_MIN)

            if premium < 0.10:
                continue

            expiry = date + pd.Timedelta(days=dte_target)
            cash -= margin_req
            cash += premium * 100 - COST_PER_CONTRACT

            csp_positions[t] = CSPPos(
                ticker=t, strike=K, premium=premium,
                entry_date=date, expiry_date=expiry,
                margin_held=margin_req,
            )
            available_margin -= margin_req
            trades.append({"date": date, "ticker": t, "action": "sell_csp",
                          "strike": K, "premium": premium, "margin": margin_req})

    return daily_equity, trades


def compute_metrics(daily_equity, starting_cash):
    """Compute risk-adjusted metrics from daily equity curve."""
    eq = pd.DataFrame(daily_equity)
    if len(eq) < 50:
        return {}

    returns = eq["equity"].pct_change().dropna()
    total_ret = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) - 1
    years = len(eq) / 252
    cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1

    mu = returns.mean() * 252
    std = returns.std() * np.sqrt(252)
    sharpe = mu / std if std > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252) if (returns < 0).any() else 1e-6
    sortino = mu / downside if downside > 0 else 0

    # Max drawdown
    eq_arr = eq["equity"].values
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / peak
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd < 0 else 0

    # Win rate from trades would require more data; compute from daily returns
    n_up = (returns > 0).sum()
    n_total = len(returns)
    daily_wr = n_up / n_total if n_total > 0 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        "total_return_pct": round(total_ret * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
        "daily_wr": round(daily_wr, 3),
        "profit_factor": round(pf, 2),
        "final_equity": round(eq["equity"].iloc[-1], 2),
        "starting_equity": starting_cash,
        "years": round(years, 2),
        "n_days": len(eq),
    }


def regime_analysis(daily_equity, spy_regime):
    """HC #428 R1: regime-stratified analysis."""
    eq = pd.DataFrame(daily_equity)
    eq["return"] = eq["equity"].pct_change()

    spy_map = dict(zip(spy_regime["date"], spy_regime["bear"]))
    eq["bear"] = eq["date"].map(spy_map).fillna(0).astype(int)

    bull = eq[eq["bear"] == 0]["return"].dropna()
    bear = eq[eq["bear"] == 1]["return"].dropna()

    bull_sharpe = (bull.mean() * 252) / (bull.std() * np.sqrt(252)) if len(bull) > 20 and bull.std() > 0 else 0
    bear_sharpe = (bear.mean() * 252) / (bear.std() * np.sqrt(252)) if len(bear) > 20 and bear.std() > 0 else 0

    max_s = max(abs(bull_sharpe), abs(bear_sharpe))
    gap = abs(bull_sharpe - bear_sharpe) / max_s if max_s > 0 else 0

    return {
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(gap, 3),
        "hc428_pass": gap <= 0.50,
        "n_bull_days": len(bull),
        "n_bear_days": len(bear),
    }


def main():
    import json

    log.info("=" * 60)
    log.info("HC #660: WHEEL UNIVERSE V3 EXPANSION")
    log.info("=" * 60)

    # Phase 1: Download new data
    log.info("Phase 1: Downloading new ticker data...")
    download_new_data()

    # Phase 2: Load all data
    log.info("Phase 2: Loading all data...")
    prices, spy_regime, sector_map = load_all_data()

    # Phase 3: Run backtests with multiple configs
    configs = [
        {"name": "base_bear_gate", "margin_cap": 0.40, "per_name_pct": 0.03,
         "profit_take": 0.65, "bear_mode": "liq_csp_only", "put_delta": 0.25},
        {"name": "no_bear_gate", "margin_cap": 0.40, "per_name_pct": 0.03,
         "profit_take": 0.65, "bear_mode": "none", "put_delta": 0.25},
        {"name": "aggressive_margin", "margin_cap": 0.60, "per_name_pct": 0.04,
         "profit_take": 0.50, "bear_mode": "liq_csp_only", "put_delta": 0.25},
        {"name": "higher_delta", "margin_cap": 0.40, "per_name_pct": 0.03,
         "profit_take": 0.65, "bear_mode": "liq_csp_only", "put_delta": 0.30},
    ]

    results = {}
    for cfg in configs:
        name = cfg.pop("name")
        log.info(f"\n--- Running config: {name} ---")
        t0 = time.time()

        daily_eq, trades_list = run_portfolio_v3(
            prices, spy_regime, sector_map,
            starting_cash=100_000,
            **cfg,
        )

        elapsed = time.time() - t0
        log.info(f"  Completed in {elapsed:.1f}s, {len(trades_list)} trades")

        metrics = compute_metrics(daily_eq, 100_000)
        regime = regime_analysis(daily_eq, spy_regime)

        # Trade action breakdown
        trade_df = pd.DataFrame(trades_list)
        action_counts = {}
        if not trade_df.empty and "action" in trade_df.columns:
            action_counts = trade_df["action"].value_counts().to_dict()

        # Per-sector breakdown (from trades)
        sector_stats = {}
        if not trade_df.empty and "ticker" in trade_df.columns:
            for t in trade_df["ticker"].unique():
                sector = sector_map.get(t, "Unknown")
                if sector not in sector_stats:
                    sector_stats[sector] = {"n_tickers": 0, "n_trades": 0}
                sector_stats[sector]["n_tickers"] += 1  # will over-count, fix below
                sector_stats[sector]["n_trades"] += len(trade_df[trade_df["ticker"] == t])
            # Fix ticker count
            for s in sector_stats:
                sector_stats[s]["n_tickers"] = len(
                    trade_df[trade_df["ticker"].map(lambda t: sector_map.get(t, "Unknown")) == s]["ticker"].unique()
                )

        results[name] = {
            "metrics": metrics,
            "regime": regime,
            "n_trades": len(trades_list),
            "action_counts": action_counts,
            "sector_breakdown": sector_stats,
        }

        log.info(f"  CAGR: {metrics.get('cagr_pct')}% | Sharpe: {metrics.get('sharpe')} | "
                f"MaxDD: {metrics.get('max_dd_pct')}% | Calmar: {metrics.get('calmar')}")
        log.info(f"  Regime: bull={regime.get('bull_sharpe')} bear={regime.get('bear_sharpe')} "
                f"gap={regime.get('regime_gap')} pass={regime.get('hc428_pass')}")

    # Save results
    out_file = OUT_DIR / "v3_expanded_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "generated": datetime.now().isoformat(),
            "universe_size": prices["ticker"].nunique(),
            "date_range": f"{prices['date'].min()} to {prices['date'].max()}",
            "results": results,
        }, f, indent=2, default=str)

    log.info(f"\nResults saved to {out_file}")
    log.info("=" * 60)
    log.info("DONE")


if __name__ == "__main__":
    from datetime import datetime
    main()
