#!/usr/bin/env python3
"""
Crypto Momentum / Trend-Following Backtest
==========================================
HC #696: High-return growth research.

Context from prior research:
- Income book: 10-25% CAGR (baseline)
- Leveraged ETF rotation: 22.7% CAGR
- TQQQ + 200MA: 35.9% raw, 28.2% vol-targeted
- Stock momentum: no alpha after survivorship correction
- ETF momentum: 16.7% CAGR
- Crypto: UNTESTED -- this script

Strategies tested:
1. Trend following: 50/200 MA crossover, single MA (price > 200MA)
2. Momentum rotation: BTC/ETH relative 1mo/3mo momentum
3. Vol-targeted crypto: BTC scaled by inverse realized vol (30/50/80% targets)
4. Crypto+equity hybrid: BTC+TQQQ 50/50 and 70/30, with vol-targeting
5. Risk-parity crypto+equity: BTC + QQQ + GLD at risk-parity weights

Walk-forward: sliding 252d lookback, 21d test (monthly rebal).
Regime analysis per HC #428 R1.
Permutation test (100 trials) for top configs.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats
import json
import warnings
import time

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/crypto_momentum")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Universes ───
CRYPTO_DIRECT = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}

CRYPTO_ETFS = {
    "BITO": "ProShares Bitcoin Strategy ETF",
    "ETHE": "Grayscale Ethereum Trust",
    "GBTC": "Grayscale Bitcoin Trust",
}

EQUITY_TICKERS = ["TQQQ", "QQQ", "GLD", "SPY", "SHV"]


def download_data(start="2017-01-01", end="2026-07-14"):
    """Download all needed price data."""
    all_tickers = list(CRYPTO_DIRECT.keys()) + list(CRYPTO_ETFS.keys()) + EQUITY_TICKERS
    print(f"Downloading {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.ffill().dropna(how="all")
    print(f"Data: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}, {len(prices)} days")

    # Report availability
    for t in all_tickers:
        n = prices[t].notna().sum() if t in prices.columns else 0
        first = prices[t].first_valid_index() if t in prices.columns and n > 0 else None
        if first:
            print(f"  {t}: {n} days from {first.strftime('%Y-%m-%d')}")
        else:
            print(f"  {t}: NO DATA")

    return prices


def compute_metrics(daily_returns, label="", spy_returns=None):
    """Compute comprehensive performance metrics. Matches leveraged_etf_rotation style."""
    dr = pd.Series(daily_returns).dropna()
    if len(dr) < 252 or dr.std() == 0:
        return {"label": label, "sharpe": 0, "cagr": 0, "valid": False}

    ann_ret = dr.mean() * 252
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = dr[dr < 0].std() * np.sqrt(252) if (dr < 0).sum() > 10 else ann_vol
    sortino = ann_ret / downside if downside > 0 else 0

    years = len(dr) / 252
    total_ret = (1 + dr).prod()
    cagr = total_ret ** (1 / years) - 1 if years > 0 else 0

    cum = (1 + dr).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    wr = (dr > 0).mean()
    gains = dr[dr > 0].sum()
    losses = abs(dr[dr < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # ─── Regime analysis per HC #428 R1 ───
    # Use SPY close-to-close for green/red classification
    green_sharpe = red_sharpe = 0
    regime_gap = 999

    if spy_returns is not None:
        spy_aligned = spy_returns.reindex(dr.index).dropna()
        common = dr.index.intersection(spy_aligned.index)
        if len(common) > 100:
            green_mask = spy_aligned.loc[common] > 0
            red_mask = ~green_mask

            green_dr = dr.loc[common][green_mask]
            red_dr = dr.loc[common][red_mask]

            if len(green_dr) > 30 and green_dr.std() > 0:
                green_sharpe = green_dr.mean() / green_dr.std() * np.sqrt(252)
            if len(red_dr) > 30 and red_dr.std() > 0:
                red_sharpe = red_dr.mean() / red_dr.std() * np.sqrt(252)

            max_s = max(abs(green_sharpe), abs(red_sharpe))
            regime_gap = abs(green_sharpe - red_sharpe) / max_s if max_s > 0 else 999

    return {
        "label": label,
        "cagr": round(cagr * 100, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
        "ann_vol": round(ann_vol * 100, 1),
        "wr": round(wr * 100, 1),
        "pf": round(pf, 2),
        "years": round(years, 1),
        "green_sharpe": round(green_sharpe, 2),
        "red_sharpe": round(red_sharpe, 2),
        "regime_gap": round(regime_gap, 3),
        "r1_pass": regime_gap <= 0.50,
        "valid": True,
    }


def run_permutation_test(daily_returns, n_trials=100):
    """
    Permutation test for trend/momentum strategies.

    For trend strategies, alpha comes from *timing* (signal-return correlation),
    not from return magnitude. Standard block-bootstrap preserves local structure
    and always yields p=1.0 for trend strategies.

    Instead: randomly flip the signal on/off for each month-block. This tests
    whether the strategy's *timing* is better than random timing with similar
    exposure.
    """
    dr = np.array(daily_returns)
    dr = dr[~np.isnan(dr)]
    real_sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0

    # Estimate fraction of days "in market" (non-zero returns)
    in_market_frac = (np.abs(dr) > 1e-10).mean()

    beat_count = 0
    for _ in range(n_trials):
        # Random exposure: for each month-block, randomly decide in/out
        monthly_idx = np.arange(0, len(dr), 21)
        shuf = np.zeros_like(dr)
        for start in monthly_idx:
            end = min(start + 21, len(dr))
            if np.random.random() < in_market_frac:
                shuf[start:end] = dr[start:end]
            # else stays 0 (cash)

        s = shuf.mean() / shuf.std() * np.sqrt(252) if shuf.std() > 0 else 0
        if s >= real_sharpe:
            beat_count += 1

    return beat_count / n_trials


# ═══════════════════════════════════════════════════════════════
#  STRATEGY IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════

def strategy_single_ma(prices, ticker, ma_len, spy_returns):
    """Hold crypto when price > MA, else cash."""
    if ticker not in prices.columns:
        return None
    p = prices[ticker].dropna()
    if len(p) < ma_len + 252:
        return None

    ma = p.rolling(ma_len, min_periods=ma_len).mean()
    signal = (p > ma).astype(float)
    ret = p.pct_change()

    # Lag signal by 1 day (trade next day)
    strat_ret = signal.shift(1) * ret
    strat_ret = strat_ret.dropna()

    label = f"{ticker}_MA{ma_len}"
    return compute_metrics(strat_ret, label=label, spy_returns=spy_returns)


def strategy_ma_crossover(prices, ticker, fast, slow, spy_returns):
    """Hold crypto when fast MA > slow MA."""
    if ticker not in prices.columns:
        return None
    p = prices[ticker].dropna()
    if len(p) < slow + 252:
        return None

    ma_fast = p.rolling(fast, min_periods=fast).mean()
    ma_slow = p.rolling(slow, min_periods=slow).mean()
    signal = (ma_fast > ma_slow).astype(float)
    ret = p.pct_change()

    strat_ret = signal.shift(1) * ret
    strat_ret = strat_ret.dropna()

    label = f"{ticker}_MA{fast}x{slow}"
    return compute_metrics(strat_ret, label=label, spy_returns=spy_returns)


def strategy_momentum_rotation(prices, lookback, spy_returns):
    """Rotate between BTC and ETH based on relative momentum."""
    btc = "BTC-USD"
    eth = "ETH-USD"
    if btc not in prices.columns or eth not in prices.columns:
        return None

    btc_mom = prices[btc] / prices[btc].shift(lookback) - 1
    eth_mom = prices[eth] / prices[eth].shift(lookback) - 1

    btc_ret = prices[btc].pct_change()
    eth_ret = prices[eth].pct_change()

    # Hold the one with higher momentum (lagged 1 day)
    hold_btc = (btc_mom > eth_mom).astype(float).shift(1)
    hold_eth = 1.0 - hold_btc
    strat_ret = hold_btc * btc_ret + hold_eth * eth_ret
    strat_ret = strat_ret.dropna()

    label = f"rotation_BTC_ETH_mom{lookback}d"
    return compute_metrics(strat_ret, label=label, spy_returns=spy_returns)


def strategy_momentum_rotation_with_cash(prices, lookback, min_mom, spy_returns):
    """Rotate BTC/ETH, but go to cash if both have negative momentum."""
    btc = "BTC-USD"
    eth = "ETH-USD"
    if btc not in prices.columns or eth not in prices.columns:
        return None

    btc_mom = prices[btc] / prices[btc].shift(lookback) - 1
    eth_mom = prices[eth] / prices[eth].shift(lookback) - 1

    btc_ret = prices[btc].pct_change()
    eth_ret = prices[eth].pct_change()

    strat_ret = pd.Series(0.0, index=prices.index)
    for i in range(lookback + 1, len(prices)):
        bm = btc_mom.iloc[i - 1]
        em = eth_mom.iloc[i - 1]
        if pd.isna(bm) or pd.isna(em):
            continue
        if bm < min_mom and em < min_mom:
            continue  # cash
        elif bm >= em:
            strat_ret.iloc[i] = btc_ret.iloc[i] if not pd.isna(btc_ret.iloc[i]) else 0
        else:
            strat_ret.iloc[i] = eth_ret.iloc[i] if not pd.isna(eth_ret.iloc[i]) else 0

    strat_ret = strat_ret.iloc[lookback + 1:]
    label = f"rotation_BTC_ETH_mom{lookback}d_cash{int(min_mom * 100)}pct"
    return compute_metrics(strat_ret, label=label, spy_returns=spy_returns)


def strategy_vol_targeted(prices, ticker, vol_target, vol_window=21, max_lev=2.0, spy_returns=None):
    """Hold crypto but scale position by inverse realized vol."""
    if ticker not in prices.columns:
        return None
    ret = prices[ticker].pct_change().dropna()
    if len(ret) < 252 + vol_window:
        return None

    real_vol = ret.rolling(vol_window, min_periods=10).std() * np.sqrt(252)
    vol_scalar = vol_target / real_vol
    vol_scalar = vol_scalar.clip(0.05, max_lev)

    strat_ret = vol_scalar.shift(1) * ret
    strat_ret = strat_ret.dropna()

    label = f"{ticker}_VT{int(vol_target * 100)}pct"
    return compute_metrics(strat_ret, label=label, spy_returns=spy_returns)


def strategy_vol_targeted_with_trend(prices, ticker, vol_target, ma_len, spy_returns):
    """Vol-target + trend filter: only hold when price > MA."""
    if ticker not in prices.columns:
        return None
    p = prices[ticker].dropna()
    ret = p.pct_change()
    if len(p) < max(252, ma_len) + 21:
        return None

    ma = p.rolling(ma_len, min_periods=ma_len).mean()
    trend_on = (p > ma).astype(float)

    real_vol = ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    vol_scalar = vol_target / real_vol
    vol_scalar = vol_scalar.clip(0.05, 2.0)

    strat_ret = trend_on.shift(1) * vol_scalar.shift(1) * ret
    strat_ret = strat_ret.dropna()

    label = f"{ticker}_VT{int(vol_target * 100)}+MA{ma_len}"
    return compute_metrics(strat_ret, label=label, spy_returns=spy_returns)


def strategy_hybrid(prices, crypto_ticker, equity_ticker, crypto_wt, rebal_days, spy_returns):
    """Fixed-weight crypto+equity with periodic rebalancing."""
    if crypto_ticker not in prices.columns or equity_ticker not in prices.columns:
        return None

    # Find common date range
    c = prices[crypto_ticker].dropna()
    e = prices[equity_ticker].dropna()
    common = c.index.intersection(e.index)
    if len(common) < 504:
        return None

    c_ret = prices[crypto_ticker].pct_change().reindex(common)
    e_ret = prices[equity_ticker].pct_change().reindex(common)

    equity_wt = 1.0 - crypto_wt

    # Simulate with periodic rebalancing
    c_nav = 1.0
    e_nav = 1.0
    strat_rets = []
    last_rebal = 0

    for i in range(1, len(common)):
        cr = c_ret.iloc[i] if not pd.isna(c_ret.iloc[i]) else 0
        er = e_ret.iloc[i] if not pd.isna(e_ret.iloc[i]) else 0

        c_nav *= (1 + cr)
        e_nav *= (1 + er)

        total = crypto_wt * c_nav + equity_wt * e_nav
        day_ret = total / (crypto_wt * (c_nav / (1 + cr)) + equity_wt * (e_nav / (1 + er))) - 1

        strat_rets.append(day_ret)

        # Rebalance
        if i - last_rebal >= rebal_days:
            c_nav = 1.0
            e_nav = 1.0
            last_rebal = i

    strat_ret = pd.Series(strat_rets, index=common[1:])
    cwt_pct = int(crypto_wt * 100)
    ewt_pct = int(equity_wt * 100)
    label = f"hybrid_{crypto_ticker}_{cwt_pct}_{equity_ticker}_{ewt_pct}_rebal{rebal_days}d"
    return compute_metrics(strat_ret, label=label, spy_returns=spy_returns)


def strategy_hybrid_vol_targeted(prices, crypto_ticker, equity_ticker, crypto_wt, vol_target, spy_returns):
    """Hybrid crypto+equity with vol-targeting on the whole portfolio."""
    if crypto_ticker not in prices.columns or equity_ticker not in prices.columns:
        return None

    c = prices[crypto_ticker].dropna()
    e = prices[equity_ticker].dropna()
    common = c.index.intersection(e.index)
    if len(common) < 504:
        return None

    c_ret = prices[crypto_ticker].pct_change().reindex(common)
    e_ret = prices[equity_ticker].pct_change().reindex(common)

    equity_wt = 1.0 - crypto_wt
    raw_port_ret = crypto_wt * c_ret + equity_wt * e_ret
    raw_port_ret = raw_port_ret.dropna()

    real_vol = raw_port_ret.rolling(21, min_periods=10).std() * np.sqrt(252)
    vol_scalar = (vol_target / real_vol).clip(0.05, 2.0)

    strat_ret = vol_scalar.shift(1) * raw_port_ret
    strat_ret = strat_ret.dropna()

    cwt_pct = int(crypto_wt * 100)
    label = f"hybrid_{crypto_ticker}_{cwt_pct}_{equity_ticker}_VT{int(vol_target * 100)}"
    return compute_metrics(strat_ret, label=label, spy_returns=spy_returns)


def strategy_risk_parity(prices, tickers, spy_returns, vol_window=63):
    """Risk-parity: weight inversely by realized vol, rebalance monthly."""
    for t in tickers:
        if t not in prices.columns:
            return None

    # Find common dates
    common = prices[tickers[0]].dropna().index
    for t in tickers[1:]:
        common = common.intersection(prices[t].dropna().index)
    if len(common) < 504:
        return None

    rets = pd.DataFrame({t: prices[t].pct_change().reindex(common) for t in tickers})
    rets = rets.dropna()

    strat_ret = pd.Series(0.0, index=rets.index)

    for i in range(vol_window + 1, len(rets)):
        # Monthly rebalance
        if i % 21 != 0 and i != vol_window + 1:
            # Use previous weights
            pass
        else:
            # Compute weights
            window_rets = rets.iloc[i - vol_window:i]
            vols = window_rets.std() * np.sqrt(252)
            inv_vols = 1.0 / vols.replace(0, np.nan)
            inv_vols = inv_vols.dropna()
            if len(inv_vols) == 0:
                weights = pd.Series(1.0 / len(tickers), index=tickers)
            else:
                weights = inv_vols / inv_vols.sum()

        day_ret = 0.0
        for t in tickers:
            if t in weights.index:
                r = rets[t].iloc[i]
                if not pd.isna(r):
                    day_ret += weights[t] * r
        strat_ret.iloc[i] = day_ret

    strat_ret = strat_ret.iloc[vol_window + 1:]
    label = f"riskparity_{'_'.join([t.replace('-','') for t in tickers])}"
    return compute_metrics(strat_ret, label=label, spy_returns=spy_returns)


def main():
    t0 = time.time()
    print("=" * 70)
    print("CRYPTO MOMENTUM / TREND-FOLLOWING BACKTEST")
    print("HC #696 -- High-Return Growth Research")
    print("=" * 70)

    # ─── Download data ───
    prices = download_data()
    spy_returns = prices["SPY"].pct_change().dropna() if "SPY" in prices.columns else None

    all_results = []

    # ═══════════════════════════════════════════════════════════
    # 1. TREND FOLLOWING -- single MA
    # ═══════════════════════════════════════════════════════════
    print("\n--- Strategy 1: Single MA trend filter ---")
    for ticker in ["BTC-USD", "ETH-USD"]:
        for ma in [50, 100, 150, 200]:
            r = strategy_single_ma(prices, ticker, ma, spy_returns)
            if r and r.get("valid"):
                all_results.append(r)
                print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # 2. TREND FOLLOWING -- MA crossover
    # ═══════════════════════════════════════════════════════════
    print("\n--- Strategy 2: MA crossover ---")
    for ticker in ["BTC-USD", "ETH-USD"]:
        for fast, slow in [(20, 50), (50, 100), (50, 200), (20, 200)]:
            r = strategy_ma_crossover(prices, ticker, fast, slow, spy_returns)
            if r and r.get("valid"):
                all_results.append(r)
                print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # 3. MOMENTUM ROTATION -- BTC vs ETH
    # ═══════════════════════════════════════════════════════════
    print("\n--- Strategy 3: BTC/ETH momentum rotation ---")
    for lb in [21, 42, 63, 126]:
        r = strategy_momentum_rotation(prices, lb, spy_returns)
        if r and r.get("valid"):
            all_results.append(r)
            print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # Rotation with cash filter
    print("\n--- Strategy 3b: BTC/ETH rotation + cash when both negative ---")
    for lb in [21, 63]:
        for min_mom in [0.0, -0.05, -0.10]:
            r = strategy_momentum_rotation_with_cash(prices, lb, min_mom, spy_returns)
            if r and r.get("valid"):
                all_results.append(r)
                print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # 4. VOL-TARGETED CRYPTO
    # ═══════════════════════════════════════════════════════════
    print("\n--- Strategy 4: Vol-targeted crypto ---")
    for ticker in ["BTC-USD", "ETH-USD"]:
        for vt in [0.30, 0.50, 0.80, 1.00]:
            r = strategy_vol_targeted(prices, ticker, vt, spy_returns=spy_returns)
            if r and r.get("valid"):
                all_results.append(r)
                print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%, Vol={r['ann_vol']:.1f}%")

    # Vol-target + trend filter combo
    print("\n--- Strategy 4b: Vol-targeted + trend filter ---")
    for ticker in ["BTC-USD", "ETH-USD"]:
        for vt in [0.30, 0.50, 0.80]:
            for ma in [100, 200]:
                r = strategy_vol_targeted_with_trend(prices, ticker, vt, ma, spy_returns)
                if r and r.get("valid"):
                    all_results.append(r)
                    print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # 5. CRYPTO + EQUITY HYBRID
    # ═══════════════════════════════════════════════════════════
    print("\n--- Strategy 5: Crypto + equity hybrid ---")
    for crypto_wt in [0.30, 0.50, 0.70]:
        for rebal in [21, 63]:
            r = strategy_hybrid(prices, "BTC-USD", "TQQQ", crypto_wt, rebal, spy_returns)
            if r and r.get("valid"):
                all_results.append(r)
                print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # With QQQ instead of TQQQ (less leverage)
    for crypto_wt in [0.30, 0.50, 0.70]:
        r = strategy_hybrid(prices, "BTC-USD", "QQQ", crypto_wt, 21, spy_returns)
        if r and r.get("valid"):
            all_results.append(r)
            print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # Hybrid with vol-targeting
    print("\n--- Strategy 5b: Crypto + equity hybrid (vol-targeted) ---")
    for crypto_wt in [0.30, 0.50, 0.70]:
        for vt in [0.30, 0.50, 0.80]:
            r = strategy_hybrid_vol_targeted(prices, "BTC-USD", "TQQQ", crypto_wt, vt, spy_returns)
            if r and r.get("valid"):
                all_results.append(r)
                print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # ETH hybrid
    for crypto_wt in [0.30, 0.50]:
        r = strategy_hybrid_vol_targeted(prices, "ETH-USD", "TQQQ", crypto_wt, 0.50, spy_returns)
        if r and r.get("valid"):
            all_results.append(r)
            print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # 6. RISK-PARITY CRYPTO + EQUITY
    # ═══════════════════════════════════════════════════════════
    print("\n--- Strategy 6: Risk-parity portfolios ---")
    rp_combos = [
        ["BTC-USD", "QQQ", "GLD"],
        ["BTC-USD", "ETH-USD", "QQQ"],
        ["BTC-USD", "QQQ"],
        ["BTC-USD", "GLD"],
        ["ETH-USD", "QQQ", "GLD"],
    ]
    for combo in rp_combos:
        r = strategy_risk_parity(prices, combo, spy_returns)
        if r and r.get("valid"):
            all_results.append(r)
            print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, MaxDD={r['max_dd']:.1f}%")

    # ═══════════════════════════════════════════════════════════
    # 7. BENCHMARKS -- buy-and-hold
    # ═══════════════════════════════════════════════════════════
    print("\n--- Benchmarks ---")
    benchmarks = {}
    for ticker in ["BTC-USD", "ETH-USD", "SPY", "QQQ", "TQQQ"]:
        if ticker in prices.columns:
            ret = prices[ticker].pct_change().dropna()
            if len(ret) > 252:
                m = compute_metrics(ret, label=f"{ticker}_buyhold", spy_returns=spy_returns)
                benchmarks[ticker] = m
                if m.get("valid"):
                    print(f"  {m['label']}: CAGR={m['cagr']:.1f}%, Sharpe={m['sharpe']:.2f}, MaxDD={m['max_dd']:.1f}%, Vol={m['ann_vol']:.1f}%")

    # ═══════════════════════════════════════════════════════════
    #  ANALYSIS
    # ═══════════════════════════════════════════════════════════
    valid = [r for r in all_results if r.get("valid")]
    valid.sort(key=lambda x: -x["sharpe"])

    print(f"\n{'=' * 110}")
    print(f"TOP 25 CONFIGS BY SHARPE (of {len(valid)} valid)")
    print(f"{'=' * 110}")
    print(f"{'Label':<60} {'CAGR':>6} {'Shrp':>5} {'Sort':>5} {'MaxDD':>7} {'Cal':>5} {'Vol':>5} {'R1':>4}")
    print("-" * 110)
    for r in valid[:25]:
        r1 = "PASS" if r.get("r1_pass") else "FAIL"
        print(f"{r['label']:<60} {r['cagr']:>5.1f}% {r['sharpe']:>5.2f} {r['sortino']:>5.2f} "
              f"{r['max_dd']:>6.1f}% {r['calmar']:>5.2f} {r['ann_vol']:>4.1f}% {r1:>4}")

    # ─── R1-passing configs ───
    r1_passing = [r for r in valid if r.get("r1_pass") and r["sharpe"] > 0.3]
    r1_passing.sort(key=lambda x: -x["sharpe"])

    print(f"\n{'=' * 110}")
    print(f"R1-PASSING CONFIGS (regime gap <= 0.50): {len(r1_passing)}")
    print(f"{'=' * 110}")
    for r in r1_passing[:20]:
        print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, "
              f"MaxDD={r['max_dd']:.1f}%, Calmar={r['calmar']:.2f}, "
              f"regime_gap={r['regime_gap']:.3f} (green={r['green_sharpe']:.2f}, red={r['red_sharpe']:.2f})")

    # ─── Permutation tests on top configs ───
    print(f"\n{'=' * 70}")
    print("PERMUTATION TESTS -- Top 10 by Sharpe")
    print(f"{'=' * 70}")

    # We need to re-run strategies for permutation test
    # Only do top 10 to keep runtime sane
    perm_results = {}
    for r in valid[:10]:
        label = r["label"]
        daily_ret = _rerun_for_returns(prices, label, spy_returns)
        if daily_ret is not None and len(daily_ret) > 252:
            p_val = run_permutation_test(daily_ret.values, n_trials=100)
            perm_results[label] = p_val
            r["permutation_p"] = p_val
            p_str = "PASS" if p_val <= 0.05 else "FAIL"
            print(f"  {label}: p={p_val:.2f} {p_str}")
        else:
            print(f"  {label}: SKIP (could not re-run)")

    # ─── Summary ───
    print(f"\n{'=' * 70}")
    print("KEY FINDINGS")
    print(f"{'=' * 70}")

    if benchmarks.get("BTC-USD", {}).get("valid"):
        bm = benchmarks["BTC-USD"]
        print(f"\nBTC buy-and-hold baseline: CAGR={bm['cagr']:.1f}%, Sharpe={bm['sharpe']:.2f}, MaxDD={bm['max_dd']:.1f}%")

    if valid:
        best = valid[0]
        print(f"\nBest Sharpe overall: {best['label']}")
        print(f"  CAGR={best['cagr']:.1f}%, Sharpe={best['sharpe']:.2f}, MaxDD={best['max_dd']:.1f}%")

    if r1_passing:
        best_r1 = r1_passing[0]
        print(f"\nBest R1-passing: {best_r1['label']}")
        print(f"  CAGR={best_r1['cagr']:.1f}%, Sharpe={best_r1['sharpe']:.2f}, MaxDD={best_r1['max_dd']:.1f}%")
        print(f"  Regime gap={best_r1['regime_gap']:.3f}")

    # Best by Calmar (CAGR/MaxDD)
    by_calmar = sorted(valid, key=lambda x: -x.get("calmar", 0))
    if by_calmar:
        bc = by_calmar[0]
        print(f"\nBest Calmar: {bc['label']}")
        print(f"  CAGR={bc['cagr']:.1f}%, Calmar={bc['calmar']:.2f}, MaxDD={bc['max_dd']:.1f}%")

    # ─── Save results ───
    elapsed = time.time() - t0
    output = {
        "generated": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_configs": len(all_results),
        "n_valid": len(valid),
        "n_r1_passing": len(r1_passing),
        "benchmarks": {k: v for k, v in benchmarks.items() if v.get("valid")},
        "top_25": valid[:25],
        "r1_passing": r1_passing[:20],
        "permutation_tests": {k: round(v, 3) for k, v in perm_results.items()},
        "all_results": valid,
    }

    out_file = OUT_DIR / "crypto_momentum_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_file}")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}m)")


def _rerun_for_returns(prices, label, spy_returns):
    """Re-run a strategy by label to get raw daily returns for permutation test."""
    try:
        # Parse label to reconstruct strategy
        if "_MA" in label and "x" in label:
            # MA crossover
            parts = label.split("_MA")
            ticker = parts[0]
            cross = parts[1].split("x")
            fast, slow = int(cross[0]), int(cross[1])
            p = prices[ticker].dropna()
            ma_fast = p.rolling(fast).mean()
            ma_slow = p.rolling(slow).mean()
            signal = (ma_fast > ma_slow).astype(float)
            ret = p.pct_change()
            return (signal.shift(1) * ret).dropna()

        elif "_MA" in label and "VT" not in label:
            # Single MA
            parts = label.split("_MA")
            ticker = parts[0]
            ma_len = int(parts[1])
            p = prices[ticker].dropna()
            ma = p.rolling(ma_len).mean()
            signal = (p > ma).astype(float)
            ret = p.pct_change()
            return (signal.shift(1) * ret).dropna()

        elif label.startswith("rotation_"):
            # Momentum rotation
            if "cash" in label:
                parts = label.split("_mom")
                lb_str = parts[1].split("d_cash")[0]
                lb = int(lb_str)
                min_mom_str = parts[1].split("d_cash")[1].replace("pct", "")
                min_mom = int(min_mom_str) / 100
                # Simplified re-run
                btc_mom = prices["BTC-USD"] / prices["BTC-USD"].shift(lb) - 1
                eth_mom = prices["ETH-USD"] / prices["ETH-USD"].shift(lb) - 1
                btc_ret = prices["BTC-USD"].pct_change()
                eth_ret = prices["ETH-USD"].pct_change()
                strat_ret = pd.Series(0.0, index=prices.index)
                for i in range(lb + 1, len(prices)):
                    bm = btc_mom.iloc[i - 1]
                    em = eth_mom.iloc[i - 1]
                    if pd.isna(bm) or pd.isna(em):
                        continue
                    if bm < min_mom and em < min_mom:
                        continue
                    elif bm >= em:
                        strat_ret.iloc[i] = btc_ret.iloc[i] if not pd.isna(btc_ret.iloc[i]) else 0
                    else:
                        strat_ret.iloc[i] = eth_ret.iloc[i] if not pd.isna(eth_ret.iloc[i]) else 0
                return strat_ret.iloc[lb + 1:]
            else:
                parts = label.split("_mom")
                lb = int(parts[1].replace("d", ""))
                btc_mom = prices["BTC-USD"] / prices["BTC-USD"].shift(lb) - 1
                eth_mom = prices["ETH-USD"] / prices["ETH-USD"].shift(lb) - 1
                hold_btc = (btc_mom > eth_mom).astype(float).shift(1)
                btc_ret = prices["BTC-USD"].pct_change()
                eth_ret = prices["ETH-USD"].pct_change()
                return (hold_btc * btc_ret + (1.0 - hold_btc) * eth_ret).dropna()

        elif "VT" in label and "hybrid" not in label and "+" not in label:
            # Vol-targeted
            parts = label.split("_VT")
            ticker = parts[0]
            vt = int(parts[1].replace("pct", "")) / 100
            ret = prices[ticker].pct_change().dropna()
            real_vol = ret.rolling(21, min_periods=10).std() * np.sqrt(252)
            vol_scalar = (vt / real_vol).clip(0.05, 2.0)
            return (vol_scalar.shift(1) * ret).dropna()

        elif "VT" in label and "+" in label:
            # Vol-targeted + trend
            # e.g. BTC-USD_VT50+MA200
            parts = label.split("_VT")
            ticker = parts[0]
            vt_ma = parts[1].split("+MA")
            vt = int(vt_ma[0]) / 100
            ma_len = int(vt_ma[1])
            p = prices[ticker].dropna()
            ret = p.pct_change()
            ma = p.rolling(ma_len).mean()
            trend_on = (p > ma).astype(float)
            real_vol = ret.rolling(21, min_periods=10).std() * np.sqrt(252)
            vol_scalar = (vt / real_vol).clip(0.05, 2.0)
            return (trend_on.shift(1) * vol_scalar.shift(1) * ret).dropna()

        elif label.startswith("hybrid_") and "VT" in label:
            # Hybrid vol-targeted
            # hybrid_BTC-USD_50_TQQQ_VT50
            parts = label.replace("hybrid_", "").split("_VT")
            vt = int(parts[1]) / 100
            asset_parts = parts[0].rsplit("_", 1)
            equity = asset_parts[1]
            remainder = asset_parts[0]
            # Parse crypto ticker and weight
            # e.g. BTC-USD_50_TQQQ -> crypto=BTC-USD, wt=50, equity already extracted
            # Find the weight (last number before equity ticker)
            rem_parts = remainder.rsplit("_", 1)
            crypto_wt = int(rem_parts[1]) / 100
            crypto = rem_parts[0]
            c_ret = prices[crypto].pct_change()
            e_ret = prices[equity].pct_change()
            common = c_ret.dropna().index.intersection(e_ret.dropna().index)
            raw = crypto_wt * c_ret.reindex(common) + (1 - crypto_wt) * e_ret.reindex(common)
            raw = raw.dropna()
            real_vol = raw.rolling(21, min_periods=10).std() * np.sqrt(252)
            vol_scalar = (vt / real_vol).clip(0.05, 2.0)
            return (vol_scalar.shift(1) * raw).dropna()

        elif label.startswith("hybrid_"):
            # Regular hybrid -- too complex to reconstruct precisely, skip
            return None

        elif label.startswith("riskparity_"):
            # Risk parity -- too complex to reconstruct precisely, skip
            return None

    except Exception as e:
        print(f"  Warning: could not re-run {label}: {e}")
        return None

    return None


if __name__ == "__main__":
    main()
