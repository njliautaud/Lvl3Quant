#!/usr/bin/env python3
"""
R4B: GROWTH PUSH — Higher Return Strategies
HC #697: No crypto | HC #0: Sliding window only | HC #694: Commission-free
HC #428: Regime test (regime_gap < 0.50)

6 strategies targeting 30%+ CAGR:
1. Leveraged Risk Parity with Predictive Tilt
2. Systematic LEAPS Buying (Call Buying)
3. Concentrated Best-Ideas (Top 5 Stocks)
4. Trend Following on Leveraged Products
5. Options Momentum (Selling Puts on Uptrending Stocks)
6. Pairs/Relative Value with Leverage
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from scipy.stats import spearmanr, norm
import json
import os
from datetime import datetime

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r4'
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DAYS = 252
TEST_DAYS = 21
START = '2012-01-01'
END = '2026-07-14'


# ─── Shared Utilities ───────────────────────────────────────────────────────

def calc_metrics(returns, name=''):
    rets = returns.dropna()
    if len(rets) < 20:
        return {'name': name, 'sharpe': 0, 'sortino': 0, 'cagr': 0,
                'max_dd': 0, 'win_rate': 0, 'pf': 0, 'n_days': len(rets)}
    total_ret = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = rets.mean() * 252 / downside if downside > 0 else 0
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    # MaxDD sanity: long-only can't lose more than 100%
    if max_dd < -1.0:
        max_dd = -1.0
    wr = (rets > 0).mean()
    gp = rets[rets > 0].sum()
    gl = abs(rets[rets < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    return {'name': name, 'sharpe': float(sharpe), 'sortino': float(sortino),
            'cagr': float(cagr), 'max_dd': float(max_dd), 'win_rate': float(wr),
            'pf': float(pf), 'n_days': int(len(rets)),
            'total_return_pct': float(total_ret * 100),
            'cagr_pct': float(cagr * 100)}


def classify_regime(spy_ret, threshold=0.003):
    regimes = pd.Series('flat', index=spy_ret.index)
    regimes[spy_ret > threshold] = 'green'
    regimes[spy_ret < -threshold] = 'red'
    return regimes


def regime_test(strat_rets, spy_rets, name=''):
    regimes = classify_regime(spy_rets.reindex(strat_rets.index))
    results = {}
    for r in ['green', 'red', 'flat']:
        mask = regimes == r
        m = calc_metrics(strat_rets[mask], f'{name} ({r})')
        results[r] = m

    sg = results['green']['sharpe']
    sr = results['red']['sharpe']
    max_s = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / max_s if max_s > 0 else float('inf')

    return {
        'per_regime': results,
        'sharpe_green': float(sg),
        'sharpe_red': float(sr),
        'sharpe_flat': float(results['flat']['sharpe']),
        'regime_gap': float(gap),
        'regime_pass': gap < 0.50,
        'distribution': {
            'green': int((regimes == 'green').sum()),
            'red': int((regimes == 'red').sum()),
            'flat': int((regimes == 'flat').sum()),
        }
    }


def download_data(tickers, start=START, end=END):
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[t] = df
                print(f"  {t}: {len(df)} days")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")
    return data


def rank_ic(signal, actual):
    mask = ~(np.isnan(signal) | np.isnan(actual))
    if mask.sum() < 20:
        return float('nan')
    ic, _ = spearmanr(signal[mask], actual[mask])
    return float(ic)


def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ─── STRATEGY 1: LEVERAGED RISK PARITY WITH PREDICTIVE TILT ─────────────────

def strategy_leveraged_risk_parity(data):
    """
    Base: risk parity across SPY/TLT/GLD/DBC with 2x leverage.
    Tilt: use momentum+carry+value signals to overweight/underweight.
    Walk-forward: does the tilt beat static risk parity?
    """
    print("\n" + "=" * 70)
    print("STRATEGY 1: LEVERAGED RISK PARITY WITH PREDICTIVE TILT")
    print("=" * 70)

    assets = ['SPY', 'TLT', 'GLD', 'DBC']
    available = [a for a in assets if a in data]
    if len(available) < 3:
        return {'error': f'Need >= 3 assets, got {len(available)}'}

    spy = data['SPY']
    common_idx = spy.index

    rets = pd.DataFrame()
    for a in available:
        rets[a] = data[a]['Close'].reindex(common_idx).pct_change()

    # Risk parity weights: inverse vol
    def risk_parity_weights(returns_window, leverage=2.0):
        vols = returns_window.std()
        vols = vols.replace(0, np.nan).dropna()
        if len(vols) == 0:
            return pd.Series(0, index=returns_window.columns)
        inv_vol = 1.0 / vols
        w = inv_vol / inv_vol.sum() * leverage
        return w

    # Predictive signals for tilt
    prices = pd.DataFrame()
    for a in available:
        prices[a] = data[a]['Close'].reindex(common_idx)

    # Momentum signal: 12-1 month
    mom = prices.shift(21) / prices.shift(252) - 1

    # Carry proxy: trailing 63d return
    carry = prices.pct_change(63)

    # Value proxy: distance from 3-year average (mean reversion at long horizon)
    value = prices.rolling(756).mean() / prices - 1  # positive = undervalued

    # Walk-forward: predict next-month asset return rank using momentum+carry+value
    strat_rets_list = []
    static_rets_list = []

    rebal_dates = list(range(TRAIN_DAYS + 252, len(common_idx) - TEST_DAYS, TEST_DAYS))

    for i in rebal_dates:
        dt = common_idx[i]

        # Static risk parity weights
        lookback = rets.iloc[max(0, i-63):i]
        static_w = risk_parity_weights(lookback, leverage=2.0)

        # Tilt: adjust risk parity by momentum+carry+value composite
        tilt_scores = pd.Series(0.0, index=available)
        for a in available:
            m = mom[a].iloc[i] if i < len(mom) else np.nan
            c = carry[a].iloc[i] if i < len(carry) else np.nan
            v = value[a].iloc[i] if i < len(value) else np.nan
            score = 0.0
            n = 0
            if not np.isnan(m):
                score += np.sign(m) * min(abs(m), 0.5)
                n += 1
            if not np.isnan(c):
                score += np.sign(c) * min(abs(c), 0.5)
                n += 1
            if not np.isnan(v):
                score += np.sign(v) * min(abs(v), 0.5)
                n += 1
            tilt_scores[a] = score / max(n, 1)

        # Tilt: rank assets, overweight top, underweight bottom
        tilt_rank = tilt_scores.rank()
        tilt_factor = 1.0 + (tilt_rank - tilt_rank.mean()) / tilt_rank.std() * 0.3
        tilt_factor = tilt_factor.clip(0.5, 1.5)

        tilted_w = static_w * tilt_factor
        # Re-normalize to 2x leverage
        tilted_w = tilted_w / tilted_w.sum() * 2.0

        # Next period returns
        for j in range(i, min(i + TEST_DAYS, len(common_idx) - 1)):
            next_ret = rets.iloc[j + 1]
            strat_r = (tilted_w * next_ret).sum()
            static_r = (static_w * next_ret).sum()
            strat_rets_list.append({'date': common_idx[j], 'ret': strat_r})
            static_rets_list.append({'date': common_idx[j], 'ret': static_r})

    strat_s = pd.DataFrame(strat_rets_list).set_index('date')['ret']
    strat_s = strat_s[~strat_s.index.duplicated(keep='first')].dropna()
    static_s = pd.DataFrame(static_rets_list).set_index('date')['ret']
    static_s = static_s[~static_s.index.duplicated(keep='first')].dropna()

    spy_ret = rets['SPY'].reindex(strat_s.index).dropna()

    strat_m = calc_metrics(strat_s, 'LevRP + Tilt (2x)')
    static_m = calc_metrics(static_s, 'Static RP (2x)')
    bh_m = calc_metrics(spy_ret, 'SPY B&H')
    rt = regime_test(strat_s, spy_ret, 'LevRP')

    # Tilt IC: does tilt prediction help?
    tilt_adds = strat_m['sharpe'] > static_m['sharpe']

    print(f"\n  Tilted RP (2x): Sharpe={strat_m['sharpe']:.3f}, Sortino={strat_m['sortino']:.3f}, "
          f"CAGR={strat_m['cagr']:.1%}, MaxDD={strat_m['max_dd']:.1%}")
    print(f"  Static RP (2x): Sharpe={static_m['sharpe']:.3f}, CAGR={static_m['cagr']:.1%}")
    print(f"  SPY B&H:        Sharpe={bh_m['sharpe']:.3f}, CAGR={bh_m['cagr']:.1%}")
    print(f"  Tilt adds value: {'YES' if tilt_adds else 'NO'}")
    print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
    print(f"  Sharpe green={rt['sharpe_green']:.3f}, red={rt['sharpe_red']:.3f}, flat={rt['sharpe_flat']:.3f}")

    return {
        'strategy_metrics': strat_m,
        'static_rp_metrics': static_m,
        'benchmark': bh_m,
        'tilt_adds_value': tilt_adds,
        'regime_test': rt,
        'leverage': 2.0,
        'notes': 'Risk parity 2x leverage across SPY/TLT/GLD/DBC with momentum+carry+value tilt',
    }


# ─── STRATEGY 2: SYSTEMATIC LEAPS BUYING ────────────────────────────────────

def strategy_leaps_buying(data):
    """
    Buy long-dated calls on stocks/ETFs with strong predicted momentum.
    Simulate using Black-Scholes + historical vol.
    Walk-forward: momentum+earnings revision signals to select which to buy.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 2: SYSTEMATIC LEAPS BUYING (CALL BUYING)")
    print("=" * 70)

    # Use sector ETFs as universe (avoid survivorship bias)
    universe = ['XLK', 'XLC', 'XLY', 'XLI', 'XLF', 'XLE', 'XLV', 'QQQ', 'SMH', 'IGV']
    available = [t for t in universe if t in data]

    if len(available) < 5:
        return {'error': f'Need >= 5 tickers, got {len(available)}'}

    spy = data['SPY']
    common_idx = spy.index
    spy_ret = spy['Close'].reindex(common_idx).pct_change()

    rets = pd.DataFrame()
    prices = pd.DataFrame()
    for t in available:
        prices[t] = data[t]['Close'].reindex(common_idx)
        rets[t] = prices[t].pct_change()

    def bs_call_delta(S, K, T, r, sigma):
        """Black-Scholes call price as fraction of spot (simplified LEAPS sim)."""
        if T <= 0 or sigma <= 0:
            return max(S - K, 0) / S if S > 0 else 0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return call_price / S  # as fraction of spot

    def leaps_return(spot_entry, spot_exit, vol, T_entry=1.0, T_exit=0.75, r=0.04):
        """Simulate LEAPS call return (ATM, 1-year expiry, hold 3 months)."""
        K = spot_entry  # ATM
        entry_price = bs_call_delta(spot_entry, K, T_entry, r, vol) * spot_entry
        exit_price = bs_call_delta(spot_exit, K, T_exit, r, vol * 0.95) * spot_exit  # slight vol decay
        if entry_price <= 0:
            return 0.0
        return (exit_price - entry_price) / entry_price

    # Walk-forward: predict 63d return, buy LEAPS on top predicted
    # Features: momentum, vol, relative strength
    all_strat_rets = []

    rebal_dates = list(range(TRAIN_DAYS + 252, len(common_idx) - 63, 63))  # quarterly rebalance

    for i in rebal_dates:
        dt = common_idx[i]

        # Build features for each asset
        asset_scores = {}
        for t in available:
            if i >= len(prices) or i - 252 < 0:
                continue

            p = prices[t].iloc[:i+1]
            r = rets[t].iloc[:i+1]

            if len(p.dropna()) < 252:
                continue

            mom_12_1 = p.iloc[-21] / p.iloc[-252] - 1 if len(p) >= 252 else np.nan
            mom_6 = p.iloc[-1] / p.iloc[-126] - 1 if len(p) >= 126 else np.nan
            mom_3 = p.iloc[-1] / p.iloc[-63] - 1 if len(p) >= 63 else np.nan
            vol_60 = r.iloc[-60:].std() * np.sqrt(252) if len(r) >= 60 else np.nan
            rel_str = (r.iloc[-63:].sum() - spy_ret.iloc[i-63:i].sum()) if i >= 63 else np.nan

            # Composite score: momentum + relative strength
            score = 0
            n = 0
            for s in [mom_12_1, mom_6, mom_3, rel_str]:
                if not np.isnan(s):
                    score += s
                    n += 1
            if n > 0:
                asset_scores[t] = {'score': score / n, 'vol': vol_60 if not np.isnan(vol_60) else 0.2}

        if len(asset_scores) < 3:
            continue

        # Pick top 3 momentum assets for LEAPS
        scores_s = pd.Series({t: v['score'] for t, v in asset_scores.items()})
        top3 = scores_s.nlargest(3).index.tolist()

        # Simulate LEAPS return over next 63 days (quarterly hold)
        for t in top3:
            if i + 63 < len(prices):
                spot_entry = prices[t].iloc[i]
                spot_exit = prices[t].iloc[i + 63]
                vol = asset_scores[t]['vol']

                if spot_entry > 0 and not np.isnan(spot_exit):
                    lr = leaps_return(spot_entry, spot_exit, vol)
                    # Each LEAPS gets 1/3 of portfolio allocation
                    # But LEAPS itself has ~3-5x leverage built in
                    # Allocate 20% of portfolio to LEAPS (conservative sizing)
                    portfolio_alloc = 0.20 / 3  # 1/3 of 20% per position
                    # Remaining 80% in cash/SPY
                    cash_return = spy_ret.iloc[i:i+63].sum() * 0.80

                    # Distribute LEAPS return over 63 days
                    daily_leaps_ret = lr / 63 * portfolio_alloc
                    daily_cash_ret = cash_return / 63

                    for d in range(63):
                        if i + d < len(common_idx):
                            all_strat_rets.append({
                                'date': common_idx[i + d],
                                'ret': daily_leaps_ret + daily_cash_ret / 3  # 1/3 per asset
                            })

    if not all_strat_rets:
        return {'error': 'No LEAPS trades generated'}

    strat_df = pd.DataFrame(all_strat_rets)
    strat_s = strat_df.groupby('date')['ret'].sum()
    strat_s = strat_s[~strat_s.index.duplicated(keep='first')].dropna()

    bh_rets = spy_ret.reindex(strat_s.index).dropna()

    strat_m = calc_metrics(strat_s, 'LEAPS Momentum')
    bh_m = calc_metrics(bh_rets, 'SPY B&H')
    rt = regime_test(strat_s, bh_rets, 'LEAPS')

    print(f"\n  Strategy: Sharpe={strat_m['sharpe']:.3f}, Sortino={strat_m['sortino']:.3f}, "
          f"CAGR={strat_m['cagr']:.1%}, MaxDD={strat_m['max_dd']:.1%}")
    print(f"  SPY B&H: Sharpe={bh_m['sharpe']:.3f}, CAGR={bh_m['cagr']:.1%}")
    print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
    print(f"  Sharpe green={rt['sharpe_green']:.3f}, red={rt['sharpe_red']:.3f}, flat={rt['sharpe_flat']:.3f}")
    print(f"  CAVEAT: Simulated LEAPS via BS model, real bid-ask spreads on options are wide")
    print(f"  CAVEAT: Vol surface assumptions simplified, real skew matters")

    return {
        'strategy_metrics': strat_m,
        'benchmark': bh_m,
        'regime_test': rt,
        'caveats': ['BS model approximation', 'Real option spreads much wider',
                    'Vol surface/skew not modeled', 'Liquidity not modeled'],
        'notes': 'Buy ATM LEAPS on top-3 momentum ETFs, 20% alloc, quarterly rebal',
    }


# ─── STRATEGY 3: CONCENTRATED BEST-IDEAS (TOP 5 STOCKS) ─────────────────────

def strategy_concentrated_top5(data):
    """
    Multi-factor model: momentum + quality + earnings revision + relative strength.
    Hold TOP 5 from a large-cap universe, equal weight, monthly rebalance.
    Uses sector ETFs to avoid survivorship bias (proxy for stock picking).
    """
    print("\n" + "=" * 70)
    print("STRATEGY 3: CONCENTRATED BEST-IDEAS (TOP 5)")
    print("NOTE: Using sector ETFs as proxy to avoid survivorship bias")
    print("=" * 70)

    # Using sector ETFs + thematic ETFs as stock-picking proxy
    universe = ['XLK', 'XLC', 'XLY', 'XLI', 'XLF', 'XLE', 'XLV', 'XLU', 'XLP', 'XLB',
                'QQQ', 'SMH', 'IGV', 'IWF', 'IWD', 'VNQ', 'DVY', 'IWM']
    available = [t for t in universe if t in data]

    if len(available) < 10:
        return {'error': f'Need >= 10 tickers, got {len(available)}'}

    spy = data['SPY']
    common_idx = spy.index
    spy_ret = spy['Close'].reindex(common_idx).pct_change()

    rets = pd.DataFrame()
    prices = pd.DataFrame()
    for t in available:
        prices[t] = data[t]['Close'].reindex(common_idx)
        rets[t] = prices[t].pct_change()

    # Multi-factor scoring per asset
    strat_rets_list = []
    rebal_dates = list(range(TRAIN_DAYS, len(common_idx) - TEST_DAYS, TEST_DAYS))

    for i in rebal_dates:
        scores = {}
        for t in available:
            p = prices[t].iloc[:i+1].dropna()
            r = rets[t].iloc[:i+1].dropna()
            if len(p) < 252:
                continue

            # Factor 1: 12-1 month momentum
            mom = p.iloc[-21] / p.iloc[-252] - 1

            # Factor 2: Quality proxy (low vol = higher quality for ETFs)
            vol = r.iloc[-60:].std()
            quality = -vol  # lower vol = higher quality

            # Factor 3: Relative strength (vs SPY)
            spy_r_63 = spy_ret.iloc[max(0, i-63):i].sum()
            asset_r_63 = r.iloc[-63:].sum()
            rel_str = asset_r_63 - spy_r_63

            # Factor 4: Short-term momentum (reversal filter)
            mom_1m = p.iloc[-1] / p.iloc[-21] - 1

            # Composite z-score
            scores[t] = {
                'momentum': mom,
                'quality': quality,
                'rel_strength': rel_str,
                'short_mom': mom_1m,
            }

        if len(scores) < 5:
            continue

        # Z-score normalize each factor across assets
        score_df = pd.DataFrame(scores).T
        for col in score_df.columns:
            s = score_df[col]
            if s.std() > 0:
                score_df[col] = (s - s.mean()) / s.std()

        # Composite: equal weight factors
        score_df['composite'] = score_df.mean(axis=1)
        top5 = score_df['composite'].nlargest(5).index.tolist()

        # Equal weight top 5
        for j in range(i, min(i + TEST_DAYS, len(common_idx) - 1)):
            day_ret = rets.iloc[j + 1][top5].mean()
            strat_rets_list.append({'date': common_idx[j], 'ret': day_ret})

    strat_s = pd.DataFrame(strat_rets_list).set_index('date')['ret']
    strat_s = strat_s[~strat_s.index.duplicated(keep='first')].dropna()
    bh_rets = spy_ret.reindex(strat_s.index).dropna()

    strat_m = calc_metrics(strat_s, 'Top 5 Concentrated')
    bh_m = calc_metrics(bh_rets, 'SPY B&H')
    rt = regime_test(strat_s, bh_rets, 'Top5')

    # Walk-forward IC: does the composite score predict next-month return rank?
    # Quick IC calc
    ics = []
    for i in rebal_dates:
        scores_dict = {}
        for t in available:
            p = prices[t].iloc[:i+1].dropna()
            if len(p) < 252:
                continue
            mom = p.iloc[-21] / p.iloc[-252] - 1
            scores_dict[t] = mom
        if len(scores_dict) < 5 and i + 21 < len(common_idx):
            continue
        # Actual next-month return
        fwd = {}
        for t in scores_dict:
            if i + 21 < len(rets):
                fwd[t] = rets[t].iloc[i:i+21].sum()
        if len(fwd) > 5:
            s = pd.Series(scores_dict)
            a = pd.Series(fwd)
            common_t = s.index.intersection(a.index)
            if len(common_t) > 5:
                ic, _ = spearmanr(s[common_t], a[common_t])
                ics.append(ic)

    avg_ic = np.mean(ics) if ics else 0

    print(f"\n  Walk-forward IC (momentum -> next month): {avg_ic:.4f}")
    print(f"  Strategy: Sharpe={strat_m['sharpe']:.3f}, Sortino={strat_m['sortino']:.3f}, "
          f"CAGR={strat_m['cagr']:.1%}, MaxDD={strat_m['max_dd']:.1%}")
    print(f"  SPY B&H: Sharpe={bh_m['sharpe']:.3f}, CAGR={bh_m['cagr']:.1%}")
    print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
    print(f"  Sharpe green={rt['sharpe_green']:.3f}, red={rt['sharpe_red']:.3f}, flat={rt['sharpe_flat']:.3f}")
    print(f"  CAVEAT: Using sector ETFs as proxy for stock picking — real stock picking")
    print(f"           could have higher returns but also survivorship bias risk")

    return {
        'strategy_metrics': strat_m,
        'benchmark': bh_m,
        'walk_forward_ic': float(avg_ic),
        'regime_test': rt,
        'caveats': ['Sector ETFs as proxy — less concentrated than real stock picking',
                    'Real S&P500 stock picking needs point-in-time constituents',
                    'ETF diversification dampens both alpha and risk'],
        'notes': 'Top 5 ETFs by composite factor (momentum+quality+relstr+shortmom), monthly rebal',
    }


# ─── STRATEGY 4: TREND FOLLOWING ON LEVERAGED PRODUCTS ──────────────────────

def strategy_trend_leveraged(data):
    """
    Apply trend signals to TQQQ, UPRO, SOXL.
    Key: leveraged ETFs have strong trends due to vol drag compounding.
    Walk-forward: does the trend signal predict when to be in vs out?
    """
    print("\n" + "=" * 70)
    print("STRATEGY 4: TREND FOLLOWING ON LEVERAGED PRODUCTS")
    print("=" * 70)

    # Leveraged ETFs + their unleveraged counterparts
    lev_map = {
        'TQQQ': 'QQQ',
        'UPRO': 'SPY',
        'SOXL': 'SMH',
    }

    # Check which leveraged ETFs are available
    available_lev = {k: v for k, v in lev_map.items() if k in data and v in data}
    if len(available_lev) < 1:
        # Try downloading leveraged ETFs
        lev_tickers = list(lev_map.keys())
        for t in lev_tickers:
            if t not in data:
                try:
                    df = yf.download(t, start='2012-01-01', end=END, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    if len(df) > 100:
                        data[t] = df
                        available_lev[t] = lev_map[t]
                        print(f"  Downloaded {t}: {len(df)} days")
                except:
                    pass

    if len(available_lev) < 1:
        return {'error': 'No leveraged ETFs available'}

    spy = data['SPY']
    common_idx = spy.index
    spy_ret = spy['Close'].reindex(common_idx).pct_change()

    all_strat_rets = []

    for lev_ticker, base_ticker in available_lev.items():
        lev_close = data[lev_ticker]['Close'].reindex(common_idx).dropna()
        base_close = data[base_ticker]['Close'].reindex(common_idx).dropna()

        # Use the base (unleveraged) for signal, trade the leveraged
        base_ret = base_close.pct_change()
        lev_ret = lev_close.pct_change()

        # Trend signals on base
        ema_20 = base_close.ewm(span=20).mean()
        ema_50 = base_close.ewm(span=50).mean()
        ema_200 = base_close.ewm(span=200).mean()
        sma_200 = base_close.rolling(200).mean()

        # Features for ML trend prediction
        features_df = pd.DataFrame(index=base_close.index)
        features_df['price_vs_ema20'] = base_close / ema_20 - 1
        features_df['price_vs_ema50'] = base_close / ema_50 - 1
        features_df['price_vs_ema200'] = base_close / ema_200 - 1
        features_df['ema20_vs_ema50'] = ema_20 / ema_50 - 1
        features_df['ema50_vs_ema200'] = ema_50 / ema_200 - 1
        features_df['mom_20d'] = base_close.pct_change(20)
        features_df['mom_60d'] = base_close.pct_change(60)
        features_df['mom_120d'] = base_close.pct_change(120)
        features_df['vol_20d'] = base_ret.rolling(20).std()
        features_df['vol_60d'] = base_ret.rolling(60).std()
        features_df['vol_ratio'] = features_df['vol_20d'] / features_df['vol_60d']
        features_df['rsi_14'] = compute_rsi(base_close, 14)
        features_df['adx_proxy'] = abs(features_df['mom_20d']) / features_df['vol_20d']

        if '^VIX' in data:
            vix = data['^VIX']['Close'].reindex(base_close.index).ffill()
            features_df['vix'] = vix
            features_df['vix_pctile'] = vix.rolling(252).rank(pct=True)

        # Target: next 21d return of LEVERAGED product (sign only for trend)
        features_df['target'] = lev_ret.rolling(21).sum().shift(-21)
        features_df['target_sign'] = (features_df['target'] > 0).astype(int)

        feat_cols = [c for c in features_df.columns if c not in ['target', 'target_sign']]
        df = features_df.dropna().copy()

        if len(df) < TRAIN_DAYS + 100:
            print(f"  {lev_ticker}: insufficient data ({len(df)} rows)")
            continue

        X = df[feat_cols].values
        y = df['target_sign'].values
        dates = df.index

        # Walk-forward
        preds = []
        for ii in range(TRAIN_DAYS, len(X) - TEST_DAYS, TEST_DAYS):
            ts = max(0, ii - TRAIN_DAYS)
            train_ds = lgb.Dataset(X[ts:ii], label=y[ts:ii])
            model = lgb.train(
                {'objective': 'binary', 'metric': 'auc', 'num_leaves': 10,
                 'learning_rate': 0.03, 'verbose': -1, 'seed': 42,
                 'min_child_samples': 20, 'subsample': 0.8},
                train_ds, num_boost_round=100
            )
            for j in range(ii, min(ii + TEST_DAYS, len(X))):
                p = model.predict(X[j:j+1])[0]
                preds.append({'date': dates[j], 'pred': p, 'actual': y[j]})

        pred_df = pd.DataFrame(preds).set_index('date')
        acc = ((pred_df['pred'] > 0.5).astype(int) == pred_df['actual']).mean()
        ic = rank_ic(pred_df['pred'].values, pred_df['actual'].values.astype(float))

        # Strategy: hold leveraged when trend predicted up, cash when down
        # Also: simple dual-MA crossover as baseline comparison
        positions = (pred_df['pred'] > 0.5).astype(float)

        lev_daily = lev_ret.reindex(pred_df.index)
        strat_rets_ticker = lev_daily * positions
        strat_rets_ticker = strat_rets_ticker.dropna()

        m = calc_metrics(strat_rets_ticker, f'Trend-{lev_ticker}')
        pct_in = positions.mean() * 100

        print(f"\n  {lev_ticker}:")
        print(f"    Trend accuracy: {acc:.1%}, IC: {ic:.4f}")
        print(f"    Time invested: {pct_in:.1f}%")
        print(f"    Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
              f"CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}")

        for d in strat_rets_ticker.index:
            all_strat_rets.append({'date': d, 'ret': strat_rets_ticker.loc[d] / len(available_lev)})

    if not all_strat_rets:
        return {'error': 'No trend trades generated'}

    # Combine all leveraged trend strategies (equal weight)
    combined = pd.DataFrame(all_strat_rets)
    combined_s = combined.groupby('date')['ret'].sum()
    combined_s = combined_s[~combined_s.index.duplicated(keep='first')].dropna()

    bh_rets = spy_ret.reindex(combined_s.index).dropna()

    strat_m = calc_metrics(combined_s, 'Trend Leveraged Combo')
    bh_m = calc_metrics(bh_rets, 'SPY B&H')
    rt = regime_test(combined_s, bh_rets, 'TrendLev')

    # Compare: timed leveraged vs buy-and-hold unleveraged
    print(f"\n  COMBINED (equal weight all leveraged trends):")
    print(f"  Strategy: Sharpe={strat_m['sharpe']:.3f}, Sortino={strat_m['sortino']:.3f}, "
          f"CAGR={strat_m['cagr']:.1%}, MaxDD={strat_m['max_dd']:.1%}")
    print(f"  SPY B&H: Sharpe={bh_m['sharpe']:.3f}, CAGR={bh_m['cagr']:.1%}")
    print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
    print(f"  Sharpe green={rt['sharpe_green']:.3f}, red={rt['sharpe_red']:.3f}, flat={rt['sharpe_flat']:.3f}")

    return {
        'strategy_metrics': strat_m,
        'benchmark': bh_m,
        'regime_test': rt,
        'notes': 'ML trend signal on base ETF, trade leveraged 3x ETF, equal weight combo',
    }


# ─── STRATEGY 5: OPTIONS MOMENTUM (SELLING PUTS ON UPTRENDING STOCKS) ───────

def strategy_options_momentum(data):
    """
    Hybrid income/growth: sell ATM puts on ETFs with positive momentum.
    If assigned, hold the ETF (it's trending up). If not, collect premium.
    Simulated via BS model.
    """
    print("\n" + "=" * 70)
    print("STRATEGY 5: OPTIONS MOMENTUM (SELLING PUTS ON UPTRENDING)")
    print("=" * 70)

    universe = ['XLK', 'XLC', 'XLY', 'XLI', 'XLF', 'XLE', 'XLV', 'QQQ', 'SMH', 'SPY']
    available = [t for t in universe if t in data]

    if len(available) < 5:
        return {'error': f'Need >= 5 tickers, got {len(available)}'}

    spy = data['SPY']
    common_idx = spy.index
    spy_ret = spy['Close'].reindex(common_idx).pct_change()

    rets = pd.DataFrame()
    prices = pd.DataFrame()
    for t in available:
        prices[t] = data[t]['Close'].reindex(common_idx)
        rets[t] = prices[t].pct_change()

    strat_rets_list = []

    # Monthly put selling cycle (30 DTE)
    rebal_dates = list(range(TRAIN_DAYS, len(common_idx) - 21, 21))

    for i in rebal_dates:
        dt = common_idx[i]

        # Select ETFs with positive momentum (trend filter)
        mom_scores = {}
        for t in available:
            p = prices[t].iloc[:i+1].dropna()
            r = rets[t].iloc[:i+1].dropna()
            if len(p) < 252:
                continue

            mom = p.iloc[-21] / p.iloc[-252] - 1
            above_200ma = p.iloc[-1] > p.rolling(200).mean().iloc[-1]
            vol = r.iloc[-21:].std() * np.sqrt(252)

            if mom > 0 and above_200ma:
                mom_scores[t] = {'mom': mom, 'vol': vol}

        if len(mom_scores) < 1:
            # No momentum — go to cash
            for j in range(i, min(i + 21, len(common_idx))):
                strat_rets_list.append({'date': common_idx[j], 'ret': 0.0})
            continue

        # Pick top 3 momentum
        scores_s = pd.Series({t: v['mom'] for t, v in mom_scores.items()})
        top = scores_s.nlargest(min(3, len(scores_s))).index.tolist()

        for t in top:
            vol = mom_scores[t]['vol']
            spot_entry = prices[t].iloc[i]

            # Simulate ATM put sale
            # Premium received ≈ BS put price at ATM, 30 DTE
            T = 30 / 365
            K = spot_entry
            d1 = (np.log(1) + (0.04 + 0.5 * vol**2) * T) / (vol * np.sqrt(T)) if vol > 0 else 0
            d2 = d1 - vol * np.sqrt(T) if vol > 0 else 0
            put_price = K * np.exp(-0.04 * T) * norm.cdf(-d2) - spot_entry * norm.cdf(-d1)
            premium_pct = put_price / spot_entry if spot_entry > 0 else 0

            # After 21 days, check outcome
            if i + 21 < len(prices):
                spot_exit = prices[t].iloc[i + 21]
                ret_underlying = (spot_exit - spot_entry) / spot_entry

                if spot_exit >= K:
                    # Put expired worthless — keep premium
                    put_ret = premium_pct
                else:
                    # Assigned — bought stock at K, now worth spot_exit
                    # Loss = (K - spot_exit)/K, offset by premium
                    put_ret = premium_pct + ret_underlying

                # Allocate proportionally
                alloc = 1.0 / len(top)
                for j in range(i, min(i + 21, len(common_idx))):
                    daily_ret = put_ret / 21 * alloc
                    strat_rets_list.append({'date': common_idx[j], 'ret': daily_ret})

    if not strat_rets_list:
        return {'error': 'No trades generated'}

    strat_df = pd.DataFrame(strat_rets_list)
    strat_s = strat_df.groupby('date')['ret'].sum()
    strat_s = strat_s[~strat_s.index.duplicated(keep='first')].dropna()

    bh_rets = spy_ret.reindex(strat_s.index).dropna()

    strat_m = calc_metrics(strat_s, 'Put Selling Momentum')
    bh_m = calc_metrics(bh_rets, 'SPY B&H')
    rt = regime_test(strat_s, bh_rets, 'PutMom')

    print(f"\n  Strategy: Sharpe={strat_m['sharpe']:.3f}, Sortino={strat_m['sortino']:.3f}, "
          f"CAGR={strat_m['cagr']:.1%}, MaxDD={strat_m['max_dd']:.1%}")
    print(f"  SPY B&H: Sharpe={bh_m['sharpe']:.3f}, CAGR={bh_m['cagr']:.1%}")
    print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
    print(f"  Sharpe green={rt['sharpe_green']:.3f}, red={rt['sharpe_red']:.3f}, flat={rt['sharpe_flat']:.3f}")
    print(f"  CAVEAT: BS model approximation, real spreads/assignment mechanics differ")

    return {
        'strategy_metrics': strat_m,
        'benchmark': bh_m,
        'regime_test': rt,
        'caveats': ['BS model approximation', 'Real spreads/slippage not modeled',
                    'Assignment mechanics simplified', 'Early assignment risk ignored'],
        'notes': 'Sell ATM puts on top-3 momentum ETFs (above 200MA), 21d cycle',
    }


# ─── STRATEGY 6: PAIRS/RELATIVE VALUE WITH LEVERAGE ─────────────────────────

def strategy_pairs_rv(data):
    """
    Long strong sector, short weak sector (or long outperformer, short underperformer).
    Market-neutral = regime-agnostic by construction.
    Walk-forward: does relative strength predict pair divergence?
    """
    print("\n" + "=" * 70)
    print("STRATEGY 6: PAIRS/RELATIVE VALUE WITH LEVERAGE")
    print("Market-neutral by construction → should be regime-agnostic")
    print("=" * 70)

    sectors = ['XLK', 'XLC', 'XLY', 'XLI', 'XLF', 'XLE', 'XLV', 'XLU', 'XLP', 'XLB']
    available = [t for t in sectors if t in data]

    if len(available) < 6:
        return {'error': f'Need >= 6 sectors, got {len(available)}'}

    spy = data['SPY']
    common_idx = spy.index
    spy_ret = spy['Close'].reindex(common_idx).pct_change()

    rets = pd.DataFrame()
    prices = pd.DataFrame()
    for t in available:
        prices[t] = data[t]['Close'].reindex(common_idx)
        rets[t] = prices[t].pct_change()

    # Walk-forward: predict relative return using momentum + mean-reversion features
    strat_rets_list = []
    ics = []

    rebal_dates = list(range(TRAIN_DAYS, len(common_idx) - TEST_DAYS, TEST_DAYS))

    for i in rebal_dates:
        # Score each sector: momentum + relative value
        scores = {}
        for t in available:
            p = prices[t].iloc[:i+1].dropna()
            r = rets[t].iloc[:i+1].dropna()
            if len(p) < 252:
                continue

            # Momentum: 12-1 month
            mom_12_1 = p.iloc[-21] / p.iloc[-252] - 1
            # Short-term momentum: 1-month
            mom_1m = p.iloc[-1] / p.iloc[-21] - 1
            # Relative strength vs SPY
            spy_r_63 = spy_ret.iloc[max(0, i-63):i].sum()
            asset_r_63 = r.iloc[-63:].sum()
            rel_str = asset_r_63 - spy_r_63

            scores[t] = mom_12_1 * 0.5 + rel_str * 0.3 + mom_1m * 0.2

        if len(scores) < 4:
            continue

        scores_s = pd.Series(scores)

        # Track IC
        if i + 21 < len(rets):
            fwd = rets.iloc[i:i+21].sum()
            common_t = scores_s.index.intersection(fwd.index)
            if len(common_t) > 5:
                ic, _ = spearmanr(scores_s[common_t], fwd[common_t])
                ics.append(ic)

        # Long top 3, short bottom 3 (2x leverage = 1x long, 1x short)
        top3 = scores_s.nlargest(3).index.tolist()
        bot3 = scores_s.nsmallest(3).index.tolist()

        leverage = 2.0  # 1x each side

        for j in range(i, min(i + TEST_DAYS, len(common_idx) - 1)):
            long_ret = rets.iloc[j + 1][top3].mean()
            short_ret = rets.iloc[j + 1][bot3].mean()
            ls_ret = (long_ret - short_ret) * leverage / 2  # each side gets half the leverage
            strat_rets_list.append({'date': common_idx[j], 'ret': ls_ret})

    strat_s = pd.DataFrame(strat_rets_list).set_index('date')['ret']
    strat_s = strat_s[~strat_s.index.duplicated(keep='first')].dropna()
    bh_rets = spy_ret.reindex(strat_s.index).dropna()

    avg_ic = np.mean(ics) if ics else 0

    strat_m = calc_metrics(strat_s, 'Pairs L/S 2x')
    bh_m = calc_metrics(bh_rets, 'SPY B&H')
    rt = regime_test(strat_s, bh_rets, 'PairsRV')

    print(f"\n  Walk-forward IC: {avg_ic:.4f}")
    print(f"  Strategy (L/S 2x): Sharpe={strat_m['sharpe']:.3f}, Sortino={strat_m['sortino']:.3f}, "
          f"CAGR={strat_m['cagr']:.1%}, MaxDD={strat_m['max_dd']:.1%}")
    print(f"  SPY B&H:          Sharpe={bh_m['sharpe']:.3f}, CAGR={bh_m['cagr']:.1%}")
    print(f"  Regime gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
    print(f"  Sharpe green={rt['sharpe_green']:.3f}, red={rt['sharpe_red']:.3f}, flat={rt['sharpe_flat']:.3f}")
    print(f"  Note: Market neutral — beta ~0 by construction")

    return {
        'strategy_metrics': strat_m,
        'benchmark': bh_m,
        'walk_forward_ic': float(avg_ic),
        'regime_test': rt,
        'market_neutral': True,
        'leverage': 2.0,
        'notes': 'Long top-3 / short bottom-3 sectors by momentum+relstr, 2x leverage, monthly rebal',
    }


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R4B: GROWTH PUSH — Higher Return Strategies")
    print("Targeting 30%+ CAGR with genuine walk-forward edge")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Download all data upfront
    all_tickers = [
        'SPY', 'QQQ', 'IWM', 'IWF', 'IWD', 'EFA', '^VIX',
        'TLT', 'GLD', 'DBC', 'HYG', 'LQD', 'VNQ', 'DVY',
        'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLU', 'XLP', 'XLY', 'XLB', 'XLC',
        'SMH', 'IGV',
        'TQQQ', 'UPRO', 'SOXL',
    ]

    print("\n--- Downloading Data ---")
    data = download_data(all_tickers)

    if 'SPY' not in data:
        print("FATAL: Cannot download SPY data")
        return

    results = {}

    print("\n" + "=" * 70)
    print("RUNNING 6 GROWTH STRATEGIES")
    print("=" * 70)

    results['1_leveraged_risk_parity'] = strategy_leveraged_risk_parity(data)
    results['2_leaps_buying'] = strategy_leaps_buying(data)
    results['3_concentrated_top5'] = strategy_concentrated_top5(data)
    results['4_trend_leveraged'] = strategy_trend_leveraged(data)
    results['5_options_momentum'] = strategy_options_momentum(data)
    results['6_pairs_rv'] = strategy_pairs_rv(data)

    # ─── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("R4B SUMMARY — GROWTH PUSH STRATEGIES")
    print(f"{'='*70}")

    summary = []
    for key, res in results.items():
        if 'error' in res:
            print(f"  {key}: ERROR - {res['error']}")
            summary.append({'strategy': key, 'verdict': 'ERROR', 'error': res['error']})
            continue

        sm = res['strategy_metrics']
        bm = res.get('benchmark', {})
        rt = res.get('regime_test', {})

        regime_pass = rt.get('regime_pass', False)
        beats_spy = sm.get('sharpe', 0) > bm.get('sharpe', 0)
        high_cagr = sm.get('cagr', 0) > 0.30  # 30%+ target

        row = {
            'strategy': sm['name'],
            'sharpe': sm['sharpe'],
            'sortino': sm['sortino'],
            'cagr_pct': sm.get('cagr', 0) * 100,
            'max_dd_pct': sm.get('max_dd', 0) * 100,
            'spy_sharpe': bm.get('sharpe', 0),
            'spy_cagr_pct': bm.get('cagr', 0) * 100,
            'regime_gap': rt.get('regime_gap', 'N/A'),
            'regime_pass': regime_pass,
            'beats_spy': beats_spy,
            'hits_30pct_cagr': high_cagr,
            'ic': res.get('walk_forward_ic', res.get('ic', 'N/A')),
            'market_neutral': res.get('market_neutral', False),
            'caveats': res.get('caveats', []),
            'notes': res.get('notes', ''),
        }
        summary.append(row)

        regime_str = f"gap={rt.get('regime_gap','?'):.3f}" if isinstance(rt.get('regime_gap'), float) else 'N/A'
        verdict = []
        if regime_pass:
            verdict.append('REGIME-PASS')
        if beats_spy:
            verdict.append('BEATS-SPY')
        if high_cagr:
            verdict.append('30%+CAGR')

        verdict_str = ', '.join(verdict) if verdict else 'NONE'

        print(f"\n  [{verdict_str}] {sm['name']}")
        print(f"    Sharpe={sm['sharpe']:.3f}, Sortino={sm['sortino']:.3f}, "
              f"CAGR={sm.get('cagr',0)*100:.1f}%, MaxDD={sm.get('max_dd',0)*100:.1f}%")
        print(f"    SPY B&H: Sharpe={bm.get('sharpe',0):.3f}, CAGR={bm.get('cagr',0)*100:.1f}%")
        print(f"    Regime: {regime_str}")

    # ─── Final Verdict ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FINAL VERDICT")
    print(f"{'='*70}")

    passed_regime = [s for s in summary if s.get('regime_pass')]
    high_cagr_strats = [s for s in summary if s.get('hits_30pct_cagr')]
    beats_spy_strats = [s for s in summary if s.get('beats_spy')]

    print(f"\n  Regime test passed:  {len(passed_regime)}/{len(summary)}")
    print(f"  Beats SPY (Sharpe): {len(beats_spy_strats)}/{len(summary)}")
    print(f"  Hits 30%+ CAGR:    {len(high_cagr_strats)}/{len(summary)}")

    if passed_regime:
        print(f"\n  REGIME-AGNOSTIC strategies:")
        for p in passed_regime:
            print(f"    + {p['strategy']}: Sharpe={p['sharpe']:.3f}, CAGR={p['cagr_pct']:.1f}%, "
                  f"MaxDD={p['max_dd_pct']:.1f}%, regime_gap={p['regime_gap']:.3f}")

    if high_cagr_strats:
        print(f"\n  30%+ CAGR strategies:")
        for h in high_cagr_strats:
            rp = 'REGIME-PASS' if h.get('regime_pass') else 'REGIME-FAIL'
            print(f"    + {h['strategy']}: CAGR={h['cagr_pct']:.1f}%, Sharpe={h['sharpe']:.3f} [{rp}]")

    # Save results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, float)):
            if np.isnan(obj) or np.isinf(obj):
                return str(obj)
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, bool):
            return obj
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return obj

    output = {
        'run_date': datetime.now().isoformat(),
        'methodology': 'Sliding walk-forward (252d train, 21d test), regime-agnostic (HC #428)',
        'target': '30%+ CAGR growth strategies',
        'detailed_results': results,
        'summary': summary,
        'n_regime_pass': len(passed_regime),
        'n_beats_spy': len(beats_spy_strats),
        'n_high_cagr': len(high_cagr_strats),
    }

    with open(os.path.join(OUT_DIR, 'growth_push_r4b_results.json'), 'w') as f:
        json.dump(make_serializable(output), f, indent=2, default=str)

    print(f"\nResults saved.")
    return output


if __name__ == '__main__':
    main()
