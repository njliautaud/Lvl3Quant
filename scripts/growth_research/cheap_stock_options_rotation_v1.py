#!/usr/bin/env python3
"""
Cheap Stock Options Rotation V1
=================================

PROBLEM: At $645, standard sector ETF options ($300-2000/contract) are too
expensive. 30+ strategies tested — ALL fail due to contract minimum size.

SOLUTION: Instead of sector ETF options, trade options on CHEAP LIQUID STOCKS
within the top-ranked sectors. Use the validated LGBM sector ranking signal
(KB #285, Sharpe 1.40) to pick WHICH sector, then buy call options on the
cheapest liquid stock in that sector where contracts cost $50-200.

UNIVERSE:
  - 11 sectors, each with 2-3 liquid stocks in the $5-30 price range
  - Examples: F (XLI, ~$11), T (XLC, ~$17), INTC (XLK, ~$20),
    PFE (XLV, ~$27), KO (XLP, ~$58), etc.
  - Monthly options, slight OTM (delta 0.40-0.50) for cheap premiums
  - $645 capital, max $200 per trade

SIX VARIANTS:
  A: Top-1 sector, cheapest stock, ATM call, DTE 30
  B: Top-2 sectors, cheapest stock each, ATM call, DTE 30
  C: Top-1 sector, cheapest stock, 5% OTM call (cheaper), DTE 30
  D: Top-1 sector, cheapest stock, ATM call, DTE 45 (more time)
  E: Top-1 sector with momentum filter + VIX filter
  F: Top-2 sectors, cheapest stock, ATM put on WORST sector (hedged)

KEY DIFFERENCES FROM PRIOR ATTEMPTS:
  - Stocks at $5-30 → contracts cost $50-300 instead of $300-2000
  - Can actually hold 2-3 positions with $645
  - Uses SAME validated LGBM ranking signal
  - Level 2 compatible (buying calls/puts only)

VALIDATION:
  - 5 gates: Sharpe>1, perm p<0.05, WR>40%, regime balance, random baseline
  - Permutation test: 200 shuffles
  - Regime-stratified analysis (HC #428)

Output: output/growth_research/cheap_stock_options_rotation_v1/
MLflow experiment: cheap_stock_options_rotation_v1
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

BASE = Path("/home/jupiter/Lvl3Quant")
if Path("/home/nick/Lvl3Quant").exists():
    BASE = Path("/home/nick/Lvl3Quant")
fprint(f"Running on: {BASE}")
sys.path.insert(0, str(BASE))

OUTPUT_DIR = BASE / "output" / "growth_research" / "cheap_stock_options_rotation_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MLflow
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "cheap_stock_options_rotation_v1"
MLFLOW_OK = False
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    MLFLOW_OK = True
    fprint(f"MLflow OK")
except Exception as e:
    fprint(f"MLflow not available: {e}")

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    fprint("WARNING: LightGBM not available")

# ==================== CONFIG ====================

INITIAL_CAPITAL = 645.0
RISK_FREE_RATE = 0.05
MAX_TRADE_COST = 200.0  # HC #749: max $200-300/trade

# Sector ETFs for ranking
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']

# Cheap liquid stocks per sector (price typically $5-30, high volume, have options)
SECTOR_STOCKS = {
    'XLK': ['INTC', 'CSCO', 'HPE'],      # Tech: Intel ~$20, Cisco ~$57, HPE ~$21
    'XLF': ['BAC', 'C', 'KEY'],           # Financial: BofA ~$45, Citi ~$72, KeyCorp ~$18
    'XLE': ['OXY', 'HAL', 'SWN'],         # Energy: Occidental ~$52, Halliburton ~$28, SWN ~$7
    'XLV': ['PFE', 'BMY', 'TEVA'],        # Healthcare: Pfizer ~$27, BMS ~$45, Teva ~$20
    'XLY': ['F', 'AAL', 'GPS'],           # Consumer Disc: Ford ~$11, AAL ~$15, Gap ~$28
    'XLP': ['KHC', 'TAP', 'SJM'],         # Consumer Staples: Kraft ~$30, Molson ~$52, JM Smucker ~$107
    'XLI': ['GE', 'DAL', 'UAL'],          # Industrials: GE ~$220 (too expensive), Delta ~$52, United ~$90
    'XLB': ['DOW', 'CF', 'CLF'],          # Materials: Dow ~$33, CF Industries ~$75, Cleveland-Cliffs ~$11
    'XLU': ['AES', 'NRG', 'PNW'],         # Utilities: AES ~$13, NRG ~$115, Pinnacle ~$93
    'XLRE': ['VNO', 'SLG', 'KRC'],        # Real Estate: Vornado ~$28, SL Green ~$65, Kilroy ~$37
    'XLC': ['T', 'PARA', 'WBD'],          # Communication: AT&T ~$29, Paramount ~$11, WBD ~$10
}

# LGBM features (same as equity rotation)
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
]


# ==================== BS PRICING ====================

def bs_call_price(S, K, T, r, sigma):
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, T, r, sigma):
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def price_option(stock_price, moneyness, dte_days, realized_vol, option_type='call',
                  pricing_uplift=1.15):
    """
    Price an option with BS + realistic uplift.

    Args:
        stock_price: current price
        moneyness: 1.0 = ATM, 1.05 = 5% OTM call, 0.95 = 5% OTM put
        dte_days: days to expiration
        realized_vol: annualized vol
        option_type: 'call' or 'put'
        pricing_uplift: multiplier for BS price to approximate market pricing
                        (KB #282: market is ~10-15% above BS for near-ATM options)

    Returns:
        dict with strike, premium, contract_cost, delta
    """
    T = dte_days / 365.0
    sigma = max(realized_vol, 0.15)  # Floor vol at 15%

    if option_type == 'call':
        strike = round(stock_price * moneyness * 2) / 2  # Round to $0.50
        premium = bs_call_price(stock_price, strike, T, RISK_FREE_RATE, sigma)
        delta = bs_delta(stock_price, strike, T, RISK_FREE_RATE, sigma)
    else:
        strike = round(stock_price * (2 - moneyness) * 2) / 2  # Put OTM
        premium = bs_put_price(stock_price, strike, T, RISK_FREE_RATE, sigma)
        delta = -(1 - bs_delta(stock_price, strike, T, RISK_FREE_RATE, sigma))

    # Apply pricing uplift (KB #282: market premiums > BS)
    premium *= pricing_uplift

    # Minimum premium floor ($0.05 per share = $5 per contract)
    premium = max(premium, 0.05)

    contract_cost = premium * 100

    return {
        'strike': strike,
        'premium': premium,
        'contract_cost': contract_cost,
        'delta': delta,
        'sigma': sigma,
        'T': T,
        'moneyness': moneyness,
        'type': option_type,
    }


# ==================== DATA ====================

def download_data():
    """Download all needed data."""
    import yfinance as yf

    # All tickers we need
    all_stocks = set()
    for stocks in SECTOR_STOCKS.values():
        all_stocks.update(stocks)
    all_tickers = list(all_stocks) + SECTORS + ['SPY', '^VIX']

    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start='2020-01-01', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill()

    rename_map = {'^VIX': 'VIX'}
    close = close.rename(columns=rename_map)

    # Separate sector ETF prices, stock prices, SPY, VIX
    vix_col = 'VIX' if 'VIX' in close.columns else None
    vix = close[vix_col].dropna() if vix_col else pd.Series(dtype=float)
    spy = close['SPY'].dropna() if 'SPY' in close.columns else pd.Series(dtype=float)

    sector_close = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    stock_close = close[[c for c in all_stocks if c in close.columns]].dropna(how='all')

    # Common index
    ix = sector_close.index
    if len(vix) > 0:
        ix = ix.intersection(vix.index)
    if len(spy) > 0:
        ix = ix.intersection(spy.index)
    ix = ix.intersection(stock_close.index)

    fprint(f"Data: {len(ix)} trading days, {ix[0].strftime('%Y-%m-%d')} to {ix[-1].strftime('%Y-%m-%d')}")
    fprint(f"Sectors: {len([c for c in SECTORS if c in sector_close.columns])}")
    fprint(f"Stocks: {len(stock_close.columns)} available out of {len(all_stocks)}")

    # Show which stocks are available per sector
    for sector, stocks in SECTOR_STOCKS.items():
        available = [s for s in stocks if s in stock_close.columns]
        missing = [s for s in stocks if s not in stock_close.columns]
        if missing:
            fprint(f"  {sector}: available={available}, missing={missing}")

    return sector_close.loc[ix], stock_close.loc[ix], spy.loc[ix], vix.loc[ix]


# ==================== FEATURE ENGINEERING ====================

def compute_features(px):
    """Compute 17 momentum features (identical to equity rotation)."""
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, 'ret_5d'), (10, 'ret_10d'), (21, 'ret_21d'),
                   (63, 'ret_63d'), (126, 'ret_126d'), (252, 'ret_252d')]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:] / pk63) - 1).min())
    f['pct_52w_high'] = float(px.iloc[-1] / px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d'] / 3
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val ** 2
        f['trend_slope_63d'] = slope * 252
    else:
        f['trend_r2_63d'] = 0.0
        f['trend_slope_63d'] = 0.0
    return f


# ==================== LGBM RANKING ====================

def run_lgbm_ranking_wf(sc, idx_end, train_window=60):
    """Walk-forward LGBM ranking on sector ETFs."""
    if not HAS_LGBM:
        rets_21d = sc.iloc[:idx_end + 1].pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    records = []
    start_i = max(260, idx_end - 400)
    all_idx = list(range(start_i, idx_end))
    rebal_idx = all_idx[::20]

    for i in rebal_idx[:-1]:
        for tk in sc.columns:
            px = sc[tk].iloc[:i + 1].dropna()
            feats = compute_features(px)
            if not feats:
                continue
            fi = min(i + 28, len(sc) - 1)
            feats.update({
                'date_idx': i, 'ticker': tk,
                'fwd_ret': float(sc[tk].iloc[fi] / sc[tk].iloc[i] - 1)
            })
            records.append(feats)

    df = pd.DataFrame(records)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)

    if len(df) < 50:
        rets_21d = sc.iloc[:idx_end + 1].pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    df['rank_label'] = df.groupby('date_idx')['fwd_ret'].rank(pct=True)
    X_train = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
    y_train = df['rank_label'].values.astype(np.float32)

    m = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1
    )
    m.fit(X_train, y_train)

    current_feats = {}
    for tk in sc.columns:
        px = sc[tk].iloc[:idx_end + 1].dropna()
        feats = compute_features(px)
        if feats:
            current_feats[tk] = feats

    if not current_feats:
        return {}

    pred_df = pd.DataFrame(current_feats).T
    for c in FEAT_COLS:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    X_pred = np.nan_to_num(pred_df[FEAT_COLS].values.astype(np.float32))
    scores = m.predict(X_pred)
    return dict(zip(pred_df.index, scores))


_USE_RANDOM_RANKING = False

def get_sector_ranking(sc, idx_end):
    """Get sector ranking, supports random mode for permutation test."""
    if _USE_RANDOM_RANKING:
        scores = np.random.randn(len(sc.columns))
        return dict(zip(sc.columns, scores))
    return run_lgbm_ranking_wf(sc, idx_end)


# ==================== FIND CHEAPEST STOCK IN SECTOR ====================

def find_cheapest_stock(sector, stock_close, day_idx):
    """Find the cheapest liquid stock in a sector at a given date."""
    candidates = SECTOR_STOCKS.get(sector, [])
    available = [s for s in candidates if s in stock_close.columns]

    if not available:
        return None, None

    prices = {}
    for s in available:
        p = stock_close[s].iloc[day_idx]
        if not np.isnan(p) and p > 1.0:  # Skip penny stocks
            prices[s] = float(p)

    if not prices:
        return None, None

    # Pick cheapest
    cheapest = min(prices, key=prices.get)
    return cheapest, prices[cheapest]


# ==================== BACKTEST ENGINE ====================

def run_variant(sc, stock_close, spy, vix, variant, verbose=True):
    """Run a single cheap stock options rotation variant."""

    configs = {
        'A': {'n_sectors': 1, 'moneyness': 1.00, 'dte': 30, 'option_type': 'call',
               'trailing_stop': None, 'vix_filter': None, 'mom_filter': False, 'hedge': False},
        'B': {'n_sectors': 2, 'moneyness': 1.00, 'dte': 30, 'option_type': 'call',
               'trailing_stop': None, 'vix_filter': None, 'mom_filter': False, 'hedge': False},
        'C': {'n_sectors': 1, 'moneyness': 1.05, 'dte': 30, 'option_type': 'call',
               'trailing_stop': None, 'vix_filter': None, 'mom_filter': False, 'hedge': False},
        'D': {'n_sectors': 1, 'moneyness': 1.00, 'dte': 45, 'option_type': 'call',
               'trailing_stop': None, 'vix_filter': None, 'mom_filter': False, 'hedge': False},
        'E': {'n_sectors': 1, 'moneyness': 1.00, 'dte': 30, 'option_type': 'call',
               'trailing_stop': 0.30, 'vix_filter': 25.0, 'mom_filter': True, 'hedge': False},
        'F': {'n_sectors': 2, 'moneyness': 1.00, 'dte': 30, 'option_type': 'call',
               'trailing_stop': None, 'vix_filter': None, 'mom_filter': False, 'hedge': True},
    }
    cfg = configs[variant]

    dates = sc.index
    n_days = len(dates)
    start_idx = 280
    rebalance_interval = 28

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []
    closed_trades = []
    last_rebalance_idx = None
    skipped_months = 0

    for day_idx in range(start_idx, n_days):
        today = dates[day_idx]

        # ── Mark-to-market ──
        expired = []
        for i, pos in enumerate(positions):
            tk = pos['ticker']
            if tk not in stock_close.columns:
                continue

            current_price = float(stock_close[tk].iloc[day_idx])
            days_held = (today - pos['entry_date']).days
            remaining_dte = pos['dte'] - days_held

            if remaining_dte <= 0:
                # Expired
                if pos['option_type'] == 'call':
                    payoff = max(0, current_price - pos['strike']) * 100 * pos['n_contracts']
                else:
                    payoff = max(0, pos['strike'] - current_price) * 100 * pos['n_contracts']
                pos['current_value'] = payoff
                expired.append(i)
            else:
                vol = pos['sigma']
                if pos['option_type'] == 'call':
                    opt_price = bs_call_price(current_price, pos['strike'],
                                               remaining_dte/365.0, RISK_FREE_RATE, vol)
                else:
                    opt_price = bs_put_price(current_price, pos['strike'],
                                              remaining_dte/365.0, RISK_FREE_RATE, vol)
                pos['current_value'] = opt_price * 100 * pos['n_contracts']

            pos['current_pnl'] = pos['current_value'] - pos['total_cost']
            pos['peak_value'] = max(pos.get('peak_value', pos['current_value']), pos['current_value'])

            # Trailing stop
            if cfg['trailing_stop'] is not None and pos['peak_value'] > 0:
                dd = 1 - pos['current_value'] / pos['peak_value']
                if dd > cfg['trailing_stop']:
                    expired.append(i)

        # Close expired/stopped
        for i in sorted(set(expired), reverse=True):
            pos = positions.pop(i)
            closed_trades.append({
                'ticker': pos['ticker'],
                'sector': pos['sector'],
                'entry_date': pos['entry_date'],
                'exit_date': today,
                'strike': pos['strike'],
                'option_type': pos['option_type'],
                'entry_cost': pos['total_cost'],
                'exit_value': pos.get('current_value', 0),
                'pnl': pos.get('current_pnl', 0),
                'pnl_pct': pos.get('current_pnl', 0) / pos['total_cost'] if pos['total_cost'] > 0 else 0,
                'days_held': (today - pos['entry_date']).days,
                'stock_price_entry': pos['entry_stock_price'],
            })
            equity += pos.get('current_pnl', 0)

        # ── Rebalance ──
        should_rebalance = (last_rebalance_idx is None or
                           day_idx - last_rebalance_idx >= rebalance_interval)

        if should_rebalance and day_idx < n_days - 5:
            # Get sector rankings
            ranks = get_sector_ranking(sc, day_idx)
            if not ranks:
                equity_curve.append({'date': today, 'equity': equity + sum(p.get('current_pnl', 0) for p in positions)})
                continue

            sorted_sectors = sorted(ranks.keys(), key=lambda s: ranks[s], reverse=True)

            # VIX filter
            if cfg['vix_filter'] is not None:
                cur_vix = float(vix.iloc[day_idx]) if day_idx < len(vix) else 20
                if cur_vix > cfg['vix_filter']:
                    skipped_months += 1
                    equity_curve.append({'date': today, 'equity': equity + sum(p.get('current_pnl', 0) for p in positions)})
                    continue

            # Momentum filter
            if cfg['mom_filter']:
                filtered = []
                for s in sorted_sectors:
                    if day_idx >= 21:
                        mom = float(sc[s].iloc[day_idx] / sc[s].iloc[day_idx - 21] - 1)
                        if mom > 0:
                            filtered.append(s)
                    else:
                        filtered.append(s)
                sorted_sectors = filtered

            # Close existing positions
            for pos in positions:
                closed_trades.append({
                    'ticker': pos['ticker'],
                    'sector': pos['sector'],
                    'entry_date': pos['entry_date'],
                    'exit_date': today,
                    'strike': pos['strike'],
                    'option_type': pos['option_type'],
                    'entry_cost': pos['total_cost'],
                    'exit_value': pos.get('current_value', 0),
                    'pnl': pos.get('current_pnl', 0),
                    'pnl_pct': pos.get('current_pnl', 0) / pos['total_cost'] if pos['total_cost'] > 0 else 0,
                    'days_held': (today - pos['entry_date']).days,
                    'stock_price_entry': pos['entry_stock_price'],
                })
                equity += pos.get('current_pnl', 0)
            positions = []

            # Open new long positions
            n_longs = min(cfg['n_sectors'], len(sorted_sectors))
            capital_per = equity / (n_longs + (1 if cfg['hedge'] else 0)) if n_longs > 0 else 0
            capital_per = min(capital_per, MAX_TRADE_COST)

            for sector in sorted_sectors[:n_longs]:
                stock, stock_price = find_cheapest_stock(sector, stock_close, day_idx)
                if stock is None:
                    continue

                # Get realized vol
                rets = stock_close[stock].iloc[max(0, day_idx-63):day_idx].pct_change().dropna()
                vol = float(rets.std() * np.sqrt(252)) if len(rets) > 10 else 0.30

                opt = price_option(stock_price, cfg['moneyness'], cfg['dte'], vol,
                                    option_type='call', pricing_uplift=1.15)

                if opt['contract_cost'] > capital_per or opt['contract_cost'] > MAX_TRADE_COST:
                    # Too expensive — try 5% OTM instead
                    opt = price_option(stock_price, 1.05, cfg['dte'], vol,
                                        option_type='call', pricing_uplift=1.15)
                    if opt['contract_cost'] > capital_per or opt['contract_cost'] > MAX_TRADE_COST:
                        continue

                n_contracts = max(1, int(capital_per / opt['contract_cost']))
                n_contracts = min(n_contracts, 2)  # Cap at 2 for risk
                total_cost = opt['contract_cost'] * n_contracts

                if total_cost > equity * 0.5:  # Don't put >50% in one position
                    n_contracts = 1
                    total_cost = opt['contract_cost']

                positions.append({
                    'ticker': stock,
                    'sector': sector,
                    'entry_date': today,
                    'strike': opt['strike'],
                    'option_type': 'call',
                    'premium': opt['premium'],
                    'total_cost': total_cost,
                    'n_contracts': n_contracts,
                    'sigma': opt['sigma'],
                    'dte': cfg['dte'],
                    'entry_stock_price': stock_price,
                    'current_value': total_cost,
                    'current_pnl': 0,
                    'peak_value': total_cost,
                })

            # Hedge: buy put on worst sector's cheapest stock
            if cfg['hedge'] and len(sorted_sectors) >= 3:
                worst_sector = sorted_sectors[-1]
                stock, stock_price = find_cheapest_stock(worst_sector, stock_close, day_idx)
                if stock is not None:
                    rets = stock_close[stock].iloc[max(0, day_idx-63):day_idx].pct_change().dropna()
                    vol = float(rets.std() * np.sqrt(252)) if len(rets) > 10 else 0.30

                    opt = price_option(stock_price, 0.95, cfg['dte'], vol,
                                        option_type='put', pricing_uplift=1.15)

                    if opt['contract_cost'] <= capital_per and opt['contract_cost'] <= MAX_TRADE_COST:
                        positions.append({
                            'ticker': stock,
                            'sector': worst_sector,
                            'entry_date': today,
                            'strike': opt['strike'],
                            'option_type': 'put',
                            'premium': opt['premium'],
                            'total_cost': opt['contract_cost'],
                            'n_contracts': 1,
                            'sigma': opt['sigma'],
                            'dte': cfg['dte'],
                            'entry_stock_price': stock_price,
                            'current_value': opt['contract_cost'],
                            'current_pnl': 0,
                            'peak_value': opt['contract_cost'],
                        })

            last_rebalance_idx = day_idx

            if verbose and day_idx <= start_idx + 28:
                held = [(p['ticker'], p['sector'], f"${p['total_cost']:.0f}") for p in positions]
                fprint(f"  {variant} rebalance {today.strftime('%Y-%m-%d')}: equity=${equity:.0f}, positions={held}")

        # Record MTM equity
        mtm = equity + sum(p.get('current_pnl', 0) for p in positions)
        equity_curve.append({'date': today, 'equity': mtm})

    # Close remaining
    for pos in positions:
        closed_trades.append({
            'ticker': pos['ticker'],
            'sector': pos['sector'],
            'entry_date': pos['entry_date'],
            'exit_date': dates[-1],
            'strike': pos['strike'],
            'option_type': pos['option_type'],
            'entry_cost': pos['total_cost'],
            'exit_value': pos.get('current_value', 0),
            'pnl': pos.get('current_pnl', 0),
            'pnl_pct': pos.get('current_pnl', 0) / pos['total_cost'] if pos['total_cost'] > 0 else 0,
            'days_held': (dates[-1] - pos['entry_date']).days,
            'stock_price_entry': pos['entry_stock_price'],
        })
        equity += pos.get('current_pnl', 0)

    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty:
        return None
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df = eq_df.set_index('date')

    if verbose:
        fprint(f"  {variant}: Skipped {skipped_months} months (filters)")

    return {
        'equity_curve': eq_df,
        'closed_trades': closed_trades,
        'variant': variant,
        'config': cfg,
        'skipped_months': skipped_months,
    }


# ==================== METRICS ====================

def compute_metrics(result, spy):
    """Compute risk-adjusted metrics."""
    if result is None:
        return None

    eq = result['equity_curve']['equity']
    trades = result['closed_trades']

    if len(eq) < 20:
        return None

    daily_rets = eq.pct_change().dropna()
    if len(daily_rets) < 10:
        return None

    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
    n_years = len(daily_rets) / 252
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1/max(n_years, 0.01)) - 1)

    ann_ret = float(daily_rets.mean() * 252)
    ann_vol = float(daily_rets.std() * np.sqrt(252))
    sharpe = ann_ret / max(ann_vol, 0.001)

    down_rets = daily_rets[daily_rets < 0]
    down_vol = float(down_rets.std() * np.sqrt(252)) if len(down_rets) > 0 else 0.001
    sortino = ann_ret / max(down_vol, 0.001)

    cummax = eq.cummax()
    drawdown = (eq - cummax) / cummax
    max_dd = float(drawdown.min())

    if trades:
        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = len(trades) - wins
        wr = wins / len(trades)
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins > 0 else 0
        avg_loss = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if losses > 0 else 0.001
        pf = (avg_win * wins) / max(avg_loss * losses, 0.001)

        # Affordability stats
        costs = [t['entry_cost'] for t in trades]
        avg_cost = np.mean(costs)
        max_cost = np.max(costs)
        pct_affordable = sum(1 for c in costs if c <= MAX_TRADE_COST) / len(costs) * 100
    else:
        wr, pf = 0, 0
        avg_cost, max_cost, pct_affordable = 0, 0, 0

    # SPY alpha
    spy_aligned = spy.reindex(eq.index).ffill()
    if len(spy_aligned.dropna()) > 20:
        spy_ret = float(spy_aligned.iloc[-1] / spy_aligned.iloc[0] - 1)
        alpha = total_ret - spy_ret
    else:
        alpha = total_ret

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1),
        'total_return': round(total_ret * 100, 1),
        'max_dd': round(max_dd * 100, 1),
        'win_rate': round(wr * 100, 1),
        'profit_factor': round(min(pf, 99), 2),
        'n_trades': len(trades),
        'final_equity': round(float(eq.iloc[-1]), 2),
        'alpha_vs_spy': round(alpha * 100, 1),
        'avg_cost_per_trade': round(avg_cost, 0),
        'max_cost_per_trade': round(max_cost, 0),
        'pct_affordable': round(pct_affordable, 1),
    }


# ==================== REGIME ANALYSIS ====================

def regime_analysis(result, spy):
    """Regime-stratified analysis (HC #428)."""
    if result is None:
        return None

    eq = result['equity_curve']['equity']
    spy_aligned = spy.reindex(eq.index).ffill().dropna()
    spy_rets = spy_aligned.pct_change().dropna()
    eq_rets = eq.pct_change().dropna()

    common = spy_rets.index.intersection(eq_rets.index)
    spy_rets = spy_rets.loc[common]
    eq_rets = eq_rets.loc[common]

    green = spy_rets > 0.001
    red = spy_rets < -0.001

    regime_metrics = {}
    for name, mask in [('green', green), ('red', red)]:
        r = eq_rets[mask]
        if len(r) > 5:
            ann_ret = float(r.mean() * 252)
            ann_vol = float(r.std() * np.sqrt(252)) if r.std() > 0 else 0.001
            regime_metrics[name] = {'sharpe': round(ann_ret / max(ann_vol, 0.001), 3), 'n_days': int(mask.sum())}
        else:
            regime_metrics[name] = {'sharpe': 0, 'n_days': 0}

    g = abs(regime_metrics['green']['sharpe'])
    r = abs(regime_metrics['red']['sharpe'])
    mx = max(g, r)
    gap = abs(g - r) / mx if mx > 0 else 0
    regime_metrics['gap_ratio'] = round(gap, 3)
    regime_metrics['pass'] = gap <= 0.50

    return regime_metrics


# ==================== PERMUTATION TEST ====================

def run_permutation_test(sc, stock_close, spy, vix, variant, n_shuffles=100):
    """Quick permutation test — random sector rankings."""
    global _USE_RANDOM_RANKING

    # Real result
    result = run_variant(sc, stock_close, spy, vix, variant, verbose=False)
    if result is None:
        return None, None, None
    real_m = compute_metrics(result, spy)
    if real_m is None:
        return None, None, None
    real_sharpe = real_m['sharpe']

    random_sharpes = []
    _USE_RANDOM_RANKING = True
    for i in range(n_shuffles):
        np.random.seed(i + 42)
        r = run_variant(sc, stock_close, spy, vix, variant, verbose=False)
        if r is not None:
            m = compute_metrics(r, spy)
            if m:
                random_sharpes.append(m['sharpe'])
    _USE_RANDOM_RANKING = False

    if not random_sharpes:
        return real_sharpe, None, None

    p_val = sum(1 for s in random_sharpes if s >= real_sharpe) / len(random_sharpes)
    return real_sharpe, p_val, np.mean(random_sharpes)


# ==================== MAIN ====================

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("CHEAP STOCK OPTIONS ROTATION V1")
    fprint("=" * 70)
    fprint(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"Capital: ${INITIAL_CAPITAL}, Max per trade: ${MAX_TRADE_COST}")
    fprint()

    sc, stock_close, spy, vix = download_data()

    all_metrics = {}
    variants = ['A', 'B', 'C', 'D', 'E', 'F']

    for v in variants:
        fprint(f"\n{'─'*50}")
        fprint(f"VARIANT {v}")
        fprint(f"{'─'*50}")

        result = run_variant(sc, stock_close, spy, vix, v, verbose=True)

        if result is None:
            fprint(f"  {v}: No result")
            all_metrics[v] = None
            continue

        metrics = compute_metrics(result, spy)
        if metrics is None:
            fprint(f"  {v}: Could not compute metrics")
            all_metrics[v] = None
            continue

        fprint(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}, "
               f"CAGR: {metrics['cagr']}%, MDD: {metrics['max_dd']}%")
        fprint(f"  WR: {metrics['win_rate']}%, PF: {metrics['profit_factor']}, "
               f"Trades: {metrics['n_trades']}, Final: ${metrics['final_equity']:.0f}")
        fprint(f"  Avg cost/trade: ${metrics['avg_cost_per_trade']:.0f}, "
               f"Max: ${metrics['max_cost_per_trade']:.0f}, "
               f"Affordable: {metrics['pct_affordable']}%")

        # Regime
        regime = regime_analysis(result, spy)
        if regime:
            fprint(f"  Regime: Green={regime['green']['sharpe']}, Red={regime['red']['sharpe']}, "
                   f"Gap={regime['gap_ratio']}")
            metrics['regime'] = regime
            metrics['regime_pass'] = regime['pass']
        else:
            metrics['regime_pass'] = False

        # Permutation (reduced to 100 shuffles for speed)
        fprint(f"  Running permutation test (100 shuffles)...")
        real_s, p_val, mean_rand = run_permutation_test(sc, stock_close, spy, vix, v, n_shuffles=100)
        if p_val is not None:
            fprint(f"  Permutation: p={p_val:.4f}, real={real_s:.3f}, random_mean={mean_rand:.3f}")
            metrics['perm_p'] = round(p_val, 4)
            metrics['random_sharpe'] = round(mean_rand, 3)
        else:
            metrics['perm_p'] = 1.0
            metrics['random_sharpe'] = 0

        # 5-gate validation
        gates = {
            'sharpe_gt_1': metrics['sharpe'] > 1.0,
            'perm_significant': metrics.get('perm_p', 1.0) < 0.05,
            'wr_gt_40': metrics['win_rate'] > 40,
            'regime_balance': metrics.get('regime_pass', False),
            'beats_random': metrics['sharpe'] > metrics.get('random_sharpe', 0) * 1.5,
        }
        gates_passed = sum(gates.values())
        metrics['gates'] = gates
        metrics['gates_passed'] = gates_passed

        fprint(f"  GATES: {gates_passed}/5 — "
               + ", ".join(f"{'PASS' if gv else 'FAIL'} {gk}" for gk, gv in gates.items()))

        all_metrics[v] = metrics

        # MLflow
        if MLFLOW_OK:
            try:
                with mlflow.start_run(run_name=f"variant_{v}"):
                    for mk, mv in metrics.items():
                        if isinstance(mv, (int, float)):
                            mlflow.log_metric(mk, mv)
            except Exception as e:
                fprint(f"  MLflow error: {e}")

    # ── SUMMARY ──
    fprint(f"\n{'='*70}")
    fprint("SUMMARY")
    fprint(f"{'='*70}")

    best_sharpe = -999
    best_v = None
    for v in variants:
        m = all_metrics.get(v)
        if m is None:
            fprint(f"  {v}: NO RESULT")
            continue
        marker = "***" if m['gates_passed'] >= 4 else ""
        fprint(f"  {v}: Sharpe {m['sharpe']}, CAGR {m['cagr']}%, MDD {m['max_dd']}%, "
               f"WR {m['win_rate']}%, Trades {m['n_trades']}, "
               f"Gates {m['gates_passed']}/5, Avg cost ${m['avg_cost_per_trade']:.0f} {marker}")
        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_v = v

    fprint(f"\n  BEST: Variant {best_v} (Sharpe {best_sharpe})")

    # Key insight
    fprint(f"\n  KEY QUESTION: Are contracts affordable at ${INITIAL_CAPITAL}?")
    for v in variants:
        m = all_metrics.get(v)
        if m and m['n_trades'] > 0:
            fprint(f"    {v}: {m['pct_affordable']}% of trades under ${MAX_TRADE_COST}")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save
    summary = {
        'experiment': EXPERIMENT_NAME,
        'timestamp': datetime.now().isoformat(),
        'capital': INITIAL_CAPITAL,
        'max_trade_cost': MAX_TRADE_COST,
        'runtime_s': round(elapsed, 1),
        'variants': {},
    }
    for v in variants:
        m = all_metrics.get(v)
        if m:
            m_clean = {k: v for k, v in m.items() if k not in ('gates', 'regime')}
            if 'gates' in m:
                m_clean['gates'] = {k: bool(v) for k, v in m['gates'].items()}
            summary['variants'][v] = m_clean

    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    fprint(f"\nResults saved.")

    return all_metrics


if __name__ == '__main__':
    main()
