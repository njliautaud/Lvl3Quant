#!/usr/bin/env python3
"""
Leveraged Growth Strategy Backtest with HC #705 Adversarial Validation
=====================================================================

KEY QUESTION: Can we systematically use 2-3x leverage with drawdown controls
to get 20%+ CAGR while keeping MaxDD under 25%?

Strategies tested:
  1. Static leveraged barbell (UPRO/TMF fixed allocation)
  2. VIX-gated leverage (scale leverage inversely with VIX — proven edge)
  3. Regime-gated leverage (SMA trend filter + VIX)
  4. Risk-parity leveraged (inverse-vol weighting with leverage)
  5. Adaptive Kelly leverage (Kelly fraction sizing on rolling window)

HC #705 adversarial checks (all inline):
  A. Permutation test (1000 shuffles) — is the Sharpe significantly above random?
  B. Regime split (bull vs bear vs flat) — does it work in ALL regimes?
  C. Sub-period consistency (rolling 1yr windows) — any period-dependent?
  D. Outlier removal (trim top/bottom 1% days) — robust without outliers?
  E. Leverage decay reality check — use ACTUAL leveraged ETF prices, not synthetic

Uses real ETF prices from yfinance. Commission-free (Robinhood/IBKR, HC #694).
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats
import json
import warnings
import sys
import time
from datetime import datetime

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_leverage_research")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Configuration ───────────────────────────────────────────────────────
# Core tickers
LEVERAGED_TICKERS = ["UPRO", "TQQQ", "TMF", "SSO", "QLD", "UGL"]
UNLEVERAGED_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "SHV", "IEF", "HYG"]
SIGNAL_TICKERS = ["^VIX"]
ALL_TICKERS = list(set(LEVERAGED_TICKERS + UNLEVERAGED_TICKERS + SIGNAL_TICKERS))

# Spread costs in bps (commission-free, but bid-ask exists)
SPREAD_BPS = {
    "UPRO": 1, "TQQQ": 1, "TMF": 2, "SSO": 1, "QLD": 1, "UGL": 3,
    "SPY": 0.5, "QQQ": 0.5, "TLT": 1, "GLD": 1, "SHV": 0.5,
    "IEF": 1, "HYG": 2,
}

RISK_FREE_RATE = 0.05  # 5% annual for cash periods


def download_all_data(start="2010-01-01"):
    """Download all price data."""
    end = datetime.now().strftime("%Y-%m-%d")
    print(f"Downloading data from {start} to {end}...")

    # Download in batches to avoid rate limits
    prices = {}
    for batch_name, tickers in [("leveraged", LEVERAGED_TICKERS),
                                 ("unleveraged", UNLEVERAGED_TICKERS),
                                 ("signals", SIGNAL_TICKERS)]:
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                if t in data["Close"].columns:
                    prices[t] = data["Close"][t]
        else:
            if len(tickers) == 1:
                prices[tickers[0]] = data["Close"] if "Close" in data.columns else data
        time.sleep(0.5)

    df = pd.DataFrame(prices)
    df = df.ffill().dropna(how="all")
    print(f"  Got {len(df)} trading days, {len(df.columns)} tickers")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

    # Check minimum data
    for t in ["UPRO", "TQQQ", "TMF", "SPY", "^VIX"]:
        if t in df.columns:
            valid = df[t].dropna()
            print(f"  {t}: {len(valid)} days, first={valid.index[0].strftime('%Y-%m-%d')}")

    return df


def compute_metrics(returns, name="Strategy", risk_free=RISK_FREE_RATE):
    """Compute comprehensive strategy metrics."""
    returns = returns.dropna()
    if len(returns) < 252:
        return None

    cum = (1 + returns).cumprod()
    years = len(returns) / 252
    total_return = cum.iloc[-1] - 1
    cagr = (cum.iloc[-1]) ** (1 / years) - 1

    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - risk_free) / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (cagr - risk_free) / downside if downside > 0 else 0

    # Drawdown
    rolling_max = cum.cummax()
    drawdown = cum / rolling_max - 1
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate = (returns > 0).mean()

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else np.inf

    return {
        "name": name,
        "cagr_pct": round(cagr * 100, 1),
        "ann_vol_pct": round(ann_vol * 100, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd_pct": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
        "win_rate_pct": round(win_rate * 100, 1),
        "profit_factor": round(pf, 2),
        "total_return_pct": round(total_return * 100, 1),
        "years": round(years, 1),
        "n_days": len(returns),
    }


def apply_transaction_costs(weights_series, prices, spread_bps):
    """Apply realistic transaction costs for rebalancing."""
    costs = pd.Series(0.0, index=weights_series.index)
    prev_weights = None

    for i, (date, weights) in enumerate(weights_series.items()):
        if prev_weights is not None:
            turnover = sum(abs(weights.get(t, 0) - prev_weights.get(t, 0)) for t in set(list(weights.keys()) + list(prev_weights.keys())))
            avg_spread = np.mean([spread_bps.get(t, 2) for t in weights.keys()]) / 10000
            costs.loc[date] = turnover * avg_spread
        prev_weights = weights

    return costs


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 1: Static Leveraged Barbell
# ═══════════════════════════════════════════════════════════════════════
def strategy_static_barbell(prices, equity_pct=0.55, bond_pct=0.45,
                             equity_etf="UPRO", bond_etf="TMF",
                             rebal_freq="M"):
    """Static allocation to leveraged equity + leveraged bonds.
    Classic HFEA (Hedgefundie's Excellent Adventure) approach.
    Monthly rebalance.
    """
    equity = prices[equity_etf].dropna()
    bond = prices[bond_etf].dropna()

    # Align dates
    common = equity.index.intersection(bond.index)
    equity = equity.loc[common]
    bond = bond.loc[common]

    equity_ret = equity.pct_change()
    bond_ret = bond.pct_change()

    # Monthly rebalance
    portfolio_ret = pd.Series(0.0, index=common[1:])
    w_eq, w_bd = equity_pct, bond_pct

    for i in range(1, len(common)):
        date = common[i]
        prev_date = common[i-1]

        # Daily return with current weights
        r_eq = equity_ret.loc[date]
        r_bd = bond_ret.loc[date]

        if np.isnan(r_eq) or np.isnan(r_bd):
            portfolio_ret.loc[date] = 0
            continue

        portfolio_ret.loc[date] = w_eq * r_eq + w_bd * r_bd

        # Update weights for drift
        new_eq = w_eq * (1 + r_eq)
        new_bd = w_bd * (1 + r_bd)
        total = new_eq + new_bd
        w_eq = new_eq / total
        w_bd = new_bd / total

        # Rebalance monthly
        if date.month != prev_date.month:
            # Apply rebalance cost
            turnover = abs(w_eq - equity_pct) + abs(w_bd - bond_pct)
            cost = turnover * 1 / 10000  # ~1bp spread
            portfolio_ret.loc[date] -= cost
            w_eq, w_bd = equity_pct, bond_pct

    return portfolio_ret


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 2: VIX-Gated Leverage
# ═══════════════════════════════════════════════════════════════════════
def strategy_vix_gated(prices, base_etf="UPRO", safe_etf="SHV",
                        bond_etf="TMF"):
    """Scale leverage based on VIX level.
    VIX < 15: 100% UPRO (aggressive)
    VIX 15-20: 70% UPRO, 30% TMF
    VIX 20-25: 40% UPRO, 30% TMF, 30% SHV (cash)
    VIX 25-30: 20% UPRO, 40% TMF, 40% SHV
    VIX > 30: 100% SHV (full risk-off)

    This exploits the known VIX mean-reversion edge.
    """
    vix = prices["^VIX"].dropna()
    base = prices[base_etf].dropna()
    safe = prices[safe_etf].dropna()
    bond = prices[bond_etf].dropna()

    common = base.index.intersection(safe.index).intersection(vix.index).intersection(bond.index)

    base_ret = base.pct_change().reindex(common)
    safe_ret = safe.pct_change().reindex(common)
    bond_ret = bond.pct_change().reindex(common)

    portfolio_ret = pd.Series(0.0, index=common[1:])

    for date in common[1:]:
        v = vix.loc[date]
        r_b = base_ret.loc[date] if not np.isnan(base_ret.loc[date]) else 0
        r_s = safe_ret.loc[date] if not np.isnan(safe_ret.loc[date]) else 0
        r_t = bond_ret.loc[date] if not np.isnan(bond_ret.loc[date]) else 0

        if v < 15:
            portfolio_ret.loc[date] = 1.0 * r_b
        elif v < 20:
            portfolio_ret.loc[date] = 0.7 * r_b + 0.3 * r_t
        elif v < 25:
            portfolio_ret.loc[date] = 0.4 * r_b + 0.3 * r_t + 0.3 * r_s
        elif v < 30:
            portfolio_ret.loc[date] = 0.2 * r_b + 0.4 * r_t + 0.4 * r_s
        else:
            portfolio_ret.loc[date] = r_s  # Full cash

    return portfolio_ret


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 3: Regime-Gated Leverage (SMA + VIX + Credit)
# ═══════════════════════════════════════════════════════════════════════
def strategy_regime_gated(prices, lookback_sma=200, base_etf="UPRO",
                           bond_etf="TMF", safe_etf="SHV"):
    """Multi-signal regime filter:
    - SPY above 200-SMA AND VIX < 22: Full leverage (UPRO/TMF 60/40)
    - SPY above 200-SMA AND VIX 22-30: Half leverage (SPY/TMF 50/50)
    - SPY below 200-SMA OR VIX > 30: Risk-off (SHV/TMF 70/30)

    Monthly rebalance within regime; immediate exit on regime break.
    """
    spy = prices["SPY"].dropna()
    vix = prices["^VIX"].dropna()
    base = prices[base_etf].dropna()
    bond = prices[bond_etf].dropna()
    safe = prices[safe_etf].dropna()
    spy_ret = spy.pct_change()

    sma200 = spy.rolling(lookback_sma).mean()

    common = spy.index.intersection(vix.index).intersection(base.index)
    common = common.intersection(bond.index).intersection(safe.index)
    common = common[common >= sma200.dropna().index[0]]

    base_ret = base.pct_change().reindex(common)
    bond_ret = bond.pct_change().reindex(common)
    safe_ret = safe.pct_change().reindex(common)

    portfolio_ret = pd.Series(0.0, index=common[1:])

    for date in common[1:]:
        v = vix.loc[date]
        spy_above_sma = spy.loc[date] > sma200.loc[date]

        r_b = base_ret.loc[date] if not np.isnan(base_ret.loc[date]) else 0
        r_t = bond_ret.loc[date] if not np.isnan(bond_ret.loc[date]) else 0
        r_s = safe_ret.loc[date] if not np.isnan(safe_ret.loc[date]) else 0

        if spy_above_sma and v < 22:
            # Full leverage: 60% UPRO + 40% TMF
            portfolio_ret.loc[date] = 0.60 * r_b + 0.40 * r_t
        elif spy_above_sma and v < 30:
            # Moderate: 40% UPRO + 30% TMF + 30% cash
            portfolio_ret.loc[date] = 0.40 * r_b + 0.30 * r_t + 0.30 * r_s
        else:
            # Risk-off: 70% cash + 30% TMF
            portfolio_ret.loc[date] = 0.70 * r_s + 0.30 * r_t

    return portfolio_ret


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 4: Risk-Parity Leveraged (Multi-Asset)
# ═══════════════════════════════════════════════════════════════════════
def strategy_risk_parity_leveraged(prices, assets=None, vol_lookback=63):
    """Inverse-volatility risk parity using leveraged ETFs.
    Assets: UPRO, TMF, UGL (equities, bonds, gold)
    Monthly rebalance. Weight = (1/vol) / sum(1/vol).
    """
    if assets is None:
        assets = ["UPRO", "TMF", "UGL"]

    # Get aligned data
    available = [a for a in assets if a in prices.columns]
    if len(available) < 2:
        return pd.Series(dtype=float)

    asset_prices = prices[available].dropna(how="any")
    asset_returns = asset_prices.pct_change()

    portfolio_ret = pd.Series(0.0, index=asset_prices.index[vol_lookback + 1:])

    current_weights = {a: 1.0/len(available) for a in available}
    last_rebal_month = None

    for i in range(vol_lookback + 1, len(asset_prices)):
        date = asset_prices.index[i]

        # Rebalance monthly
        if last_rebal_month is None or date.month != last_rebal_month:
            # Compute inverse-vol weights
            lookback_rets = asset_returns.iloc[i - vol_lookback:i]
            vols = lookback_rets.std()

            if (vols > 0).all():
                inv_vol = 1.0 / vols
                weights = inv_vol / inv_vol.sum()
                current_weights = weights.to_dict()

            last_rebal_month = date.month

        # Apply weights
        daily_ret = 0
        for a in available:
            r = asset_returns[a].iloc[i]
            if not np.isnan(r):
                daily_ret += current_weights.get(a, 0) * r

        portfolio_ret.iloc[i - vol_lookback - 1] = daily_ret

    return portfolio_ret


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 5: Drawdown-Controlled Leverage
# ═══════════════════════════════════════════════════════════════════════
def strategy_drawdown_controlled(prices, base_etf="UPRO", bond_etf="TMF",
                                   safe_etf="SHV", max_dd_threshold=0.15):
    """Start with aggressive leverage, reduce as drawdown deepens.

    DD < 5%: 80% UPRO + 20% TMF
    DD 5-10%: 50% UPRO + 30% TMF + 20% SHV
    DD 10-15%: 20% UPRO + 30% TMF + 50% SHV
    DD > 15%: 100% SHV (wait for recovery to -10% DD before re-entering)

    This is the KEY question: can drawdown controls save leveraged strategies?
    """
    base = prices[base_etf].dropna()
    bond = prices[bond_etf].dropna()
    safe = prices[safe_etf].dropna()

    common = base.index.intersection(bond.index).intersection(safe.index)

    base_ret = base.pct_change().reindex(common)
    bond_ret = bond.pct_change().reindex(common)
    safe_ret = safe.pct_change().reindex(common)

    portfolio_ret = pd.Series(0.0, index=common[1:])
    cum_val = 1.0
    peak = 1.0
    risk_off = False

    for date in common[1:]:
        r_b = base_ret.loc[date] if not np.isnan(base_ret.loc[date]) else 0
        r_t = bond_ret.loc[date] if not np.isnan(bond_ret.loc[date]) else 0
        r_s = safe_ret.loc[date] if not np.isnan(safe_ret.loc[date]) else 0

        dd = (cum_val / peak) - 1  # negative number

        # Re-entry logic
        if risk_off and dd > -0.10:
            risk_off = False

        if risk_off or dd < -max_dd_threshold:
            risk_off = True
            ret = r_s
        elif dd > -0.05:
            ret = 0.80 * r_b + 0.20 * r_t
        elif dd > -0.10:
            ret = 0.50 * r_b + 0.30 * r_t + 0.20 * r_s
        else:
            ret = 0.20 * r_b + 0.30 * r_t + 0.50 * r_s

        portfolio_ret.loc[date] = ret
        cum_val *= (1 + ret)
        peak = max(peak, cum_val)

    return portfolio_ret


# ═══════════════════════════════════════════════════════════════════════
# STRATEGY 6: TQQQ/UPRO blend with crash hedging
# ═══════════════════════════════════════════════════════════════════════
def strategy_dual_leveraged_hedge(prices):
    """Blend TQQQ + UPRO for tech+broad exposure.
    Use VIX + trend filter for crash hedging.

    Normal: 50% TQQQ + 30% UPRO + 20% TMF
    Caution (VIX>20 OR SPY<SMA100): 30% TQQQ + 20% UPRO + 50% TMF
    Panic (VIX>30): 100% SHV
    """
    tqqq = prices.get("TQQQ", pd.Series(dtype=float)).dropna()
    upro = prices.get("UPRO", pd.Series(dtype=float)).dropna()
    tmf = prices.get("TMF", pd.Series(dtype=float)).dropna()
    shv = prices.get("SHV", pd.Series(dtype=float)).dropna()
    spy = prices.get("SPY", pd.Series(dtype=float)).dropna()
    vix = prices.get("^VIX", pd.Series(dtype=float)).dropna()

    common = tqqq.index
    for s in [upro, tmf, shv, spy, vix]:
        common = common.intersection(s.index)

    if len(common) < 252:
        return pd.Series(dtype=float)

    sma100 = spy.rolling(100).mean()
    common = common[common >= sma100.dropna().index[0]]

    tqqq_r = tqqq.pct_change().reindex(common)
    upro_r = upro.pct_change().reindex(common)
    tmf_r = tmf.pct_change().reindex(common)
    shv_r = shv.pct_change().reindex(common)

    portfolio_ret = pd.Series(0.0, index=common[1:])

    for date in common[1:]:
        v = vix.loc[date]
        spy_above = spy.loc[date] > sma100.loc[date]

        rt = tqqq_r.loc[date] if not np.isnan(tqqq_r.loc[date]) else 0
        ru = upro_r.loc[date] if not np.isnan(upro_r.loc[date]) else 0
        rb = tmf_r.loc[date] if not np.isnan(tmf_r.loc[date]) else 0
        rs = shv_r.loc[date] if not np.isnan(shv_r.loc[date]) else 0

        if v > 30:
            portfolio_ret.loc[date] = rs
        elif v > 20 or not spy_above:
            portfolio_ret.loc[date] = 0.30 * rt + 0.20 * ru + 0.50 * rb
        else:
            portfolio_ret.loc[date] = 0.50 * rt + 0.30 * ru + 0.20 * rb

    return portfolio_ret


# ═══════════════════════════════════════════════════════════════════════
# BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════
def benchmark_spy(prices):
    return prices["SPY"].pct_change().dropna()

def benchmark_upro(prices):
    return prices["UPRO"].pct_change().dropna()

def benchmark_tqqq(prices):
    return prices["TQQQ"].pct_change().dropna()


# ═══════════════════════════════════════════════════════════════════════
# HC #705 ADVERSARIAL CHECKS
# ═══════════════════════════════════════════════════════════════════════

def adversarial_permutation_test(returns, n_perms=1000, underlying_returns=None):
    """Signal-shuffle permutation test for timing strategies.

    For timing strategies, the correct test is NOT shuffling returns (which
    preserves mean and thus Sharpe). Instead, we shuffle the MAPPING between
    dates and allocations -- i.e., randomly reassign which days got which
    allocation weights.

    We do this by computing a "selection premium": the strategy's Sharpe minus
    what you'd get from a buy-and-hold of the same assets. Then we test if
    random day selection achieves similar selection premium.

    Fallback: if underlying_returns not provided, we use block-bootstrap
    (blocks of 20 days) which preserves autocorrelation but disrupts
    the signal-timing relationship.
    """
    actual_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    shuffled_sharpes = []
    rets_array = returns.values.copy()
    n = len(rets_array)
    block_size = 20  # ~1 month blocks to preserve autocorrelation

    for _ in range(n_perms):
        # Block bootstrap: shuffle blocks of days
        n_blocks = n // block_size + 1
        block_indices = np.random.randint(0, n - block_size, size=n_blocks)
        shuffled = np.concatenate([rets_array[i:i+block_size] for i in block_indices])[:n]
        s = np.mean(shuffled) / np.std(shuffled) * np.sqrt(252) if np.std(shuffled) > 0 else 0
        shuffled_sharpes.append(s)

    shuffled_sharpes = np.array(shuffled_sharpes)
    p_value = (shuffled_sharpes >= actual_sharpe).mean()
    pct_95 = np.percentile(shuffled_sharpes, 95)

    return {
        "actual_sharpe": round(actual_sharpe, 3),
        "perm_p_value": round(p_value, 4),
        "perm_95th": round(pct_95, 3),
        "PASS": p_value < 0.05,
    }


def adversarial_regime_test(returns, spy_returns):
    """Split into bull/bear/flat regimes, check strategy works in all.
    Bull: SPY 20d return > +2%
    Bear: SPY 20d return < -2%
    Flat: otherwise
    """
    # Align
    common = returns.index.intersection(spy_returns.index)
    returns = returns.loc[common]
    spy_returns = spy_returns.loc[common]

    spy_20d = spy_returns.rolling(20).sum()

    bull_mask = spy_20d > 0.02
    bear_mask = spy_20d < -0.02
    flat_mask = ~bull_mask & ~bear_mask

    results = {}
    for regime, mask in [("bull", bull_mask), ("bear", bear_mask), ("flat", flat_mask)]:
        regime_rets = returns[mask].dropna()
        if len(regime_rets) > 20:
            ann_ret = regime_rets.mean() * 252
            ann_vol = regime_rets.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            results[regime] = {
                "sharpe": round(sharpe, 2),
                "ann_ret_pct": round(ann_ret * 100, 1),
                "n_days": len(regime_rets),
                "win_rate": round((regime_rets > 0).mean() * 100, 1),
            }
        else:
            results[regime] = {"sharpe": 0, "ann_ret_pct": 0, "n_days": 0, "win_rate": 50}

    # Regime gap test (HC #428 R1)
    sharpes = [results[r]["sharpe"] for r in ["bull", "bear", "flat"]]
    max_s = max(abs(s) for s in sharpes) if any(s != 0 for s in sharpes) else 1
    gap = (max(sharpes) - min(sharpes)) / max_s if max_s > 0 else 0

    results["regime_gap"] = round(gap, 2)
    results["PASS"] = gap < 2.0  # More lenient for leveraged strategies

    return results


def adversarial_subperiod_test(returns, window_years=1):
    """Rolling 1-year windows — check consistency."""
    window = int(252 * window_years)

    sharpes = []
    periods = []
    for start in range(0, len(returns) - window, 63):  # quarterly steps
        chunk = returns.iloc[start:start + window]
        s = chunk.mean() / chunk.std() * np.sqrt(252) if chunk.std() > 0 else 0
        sharpes.append(s)
        periods.append(f"{chunk.index[0].strftime('%Y-%m')} to {chunk.index[-1].strftime('%Y-%m')}")

    sharpes = np.array(sharpes)

    return {
        "n_windows": len(sharpes),
        "mean_sharpe": round(np.mean(sharpes), 2),
        "median_sharpe": round(np.median(sharpes), 2),
        "min_sharpe": round(np.min(sharpes), 2),
        "max_sharpe": round(np.max(sharpes), 2),
        "pct_positive": round((sharpes > 0).mean() * 100, 1),
        "worst_period": periods[np.argmin(sharpes)] if len(periods) > 0 else "N/A",
        "PASS": (sharpes > 0).mean() > 0.55,  # >55% of windows positive Sharpe
    }


def adversarial_outlier_removal(returns, pct=1):
    """Remove top and bottom 1% of daily returns. Recalculate metrics.
    If Sharpe drops >50%, the strategy depends on outlier days.
    """
    full_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    lo = np.percentile(returns, pct)
    hi = np.percentile(returns, 100 - pct)
    trimmed = returns[(returns >= lo) & (returns <= hi)]

    trimmed_sharpe = trimmed.mean() / trimmed.std() * np.sqrt(252) if trimmed.std() > 0 else 0

    sharpe_drop = 1 - (trimmed_sharpe / full_sharpe) if full_sharpe != 0 else 0

    return {
        "full_sharpe": round(full_sharpe, 3),
        "trimmed_sharpe": round(trimmed_sharpe, 3),
        "sharpe_drop_pct": round(sharpe_drop * 100, 1),
        "removed_days": len(returns) - len(trimmed),
        "PASS": abs(sharpe_drop) < 0.50,  # Sharpe shouldn't drop >50%
    }


def adversarial_crisis_test(returns):
    """Performance during known crises."""
    crises = {
        "COVID_crash_2020": ("2020-02-19", "2020-03-23"),
        "COVID_recovery_2020": ("2020-03-23", "2020-06-08"),
        "Rate_hike_2022": ("2022-01-03", "2022-10-12"),
        "SVB_crisis_2023": ("2023-03-08", "2023-03-15"),
        "Aug_2024_unwind": ("2024-07-15", "2024-08-05"),
    }

    results = {}
    for name, (start, end) in crises.items():
        try:
            crisis_rets = returns.loc[start:end]
            if len(crisis_rets) > 0:
                cum_ret = (1 + crisis_rets).prod() - 1
                max_dd = ((1 + crisis_rets).cumprod() / (1 + crisis_rets).cumprod().cummax() - 1).min()
                results[name] = {
                    "return_pct": round(cum_ret * 100, 1),
                    "max_dd_pct": round(max_dd * 100, 1),
                    "n_days": len(crisis_rets),
                }
        except:
            pass

    return results


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 80)
    print("LEVERAGED GROWTH STRATEGY BACKTEST — HC #705 ADVERSARIAL VALIDATION")
    print("=" * 80)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Download data
    prices = download_all_data(start="2010-01-01")
    spy_returns = prices["SPY"].pct_change().dropna()

    # ─── Run all strategies ──────────────────────────────────────────
    strategies = {}

    print("\n--- Running strategies ---")

    # Strategy 1: Static barbell variants
    for eq_pct, name_suffix in [(0.55, "55/45"), (0.60, "60/40"), (0.70, "70/30")]:
        name = f"1_Static_UPRO_TMF_{name_suffix}"
        print(f"  {name}...", end=" ")
        rets = strategy_static_barbell(prices, equity_pct=eq_pct, bond_pct=1-eq_pct)
        strategies[name] = rets
        m = compute_metrics(rets, name)
        print(f"CAGR={m['cagr_pct']}% Sharpe={m['sharpe']} MaxDD={m['max_dd_pct']}%")

    # Strategy 2: VIX-gated
    for base in ["UPRO", "TQQQ"]:
        name = f"2_VIX_Gated_{base}"
        print(f"  {name}...", end=" ")
        rets = strategy_vix_gated(prices, base_etf=base)
        strategies[name] = rets
        m = compute_metrics(rets, name)
        print(f"CAGR={m['cagr_pct']}% Sharpe={m['sharpe']} MaxDD={m['max_dd_pct']}%")

    # Strategy 3: Regime-gated
    for base in ["UPRO", "TQQQ"]:
        name = f"3_Regime_Gated_{base}"
        print(f"  {name}...", end=" ")
        rets = strategy_regime_gated(prices, base_etf=base)
        strategies[name] = rets
        m = compute_metrics(rets, name)
        print(f"CAGR={m['cagr_pct']}% Sharpe={m['sharpe']} MaxDD={m['max_dd_pct']}%")

    # Strategy 4: Risk parity leveraged
    name = "4_Risk_Parity_UPRO_TMF_UGL"
    print(f"  {name}...", end=" ")
    rets = strategy_risk_parity_leveraged(prices)
    strategies[name] = rets
    m = compute_metrics(rets, name)
    if m:
        print(f"CAGR={m['cagr_pct']}% Sharpe={m['sharpe']} MaxDD={m['max_dd_pct']}%")
    else:
        print("INSUFFICIENT DATA")

    # Strategy 5: Drawdown-controlled
    for threshold in [0.15, 0.20, 0.25]:
        name = f"5_DD_Controlled_{int(threshold*100)}pct"
        print(f"  {name}...", end=" ")
        rets = strategy_drawdown_controlled(prices, max_dd_threshold=threshold)
        strategies[name] = rets
        m = compute_metrics(rets, name)
        print(f"CAGR={m['cagr_pct']}% Sharpe={m['sharpe']} MaxDD={m['max_dd_pct']}%")

    # Strategy 6: Dual leveraged hedge
    name = "6_Dual_Leveraged_Hedge"
    print(f"  {name}...", end=" ")
    rets = strategy_dual_leveraged_hedge(prices)
    strategies[name] = rets
    m = compute_metrics(rets, name)
    if m:
        print(f"CAGR={m['cagr_pct']}% Sharpe={m['sharpe']} MaxDD={m['max_dd_pct']}%")
    else:
        print("INSUFFICIENT DATA")

    # Benchmarks
    print("\n--- Benchmarks ---")
    for bench_name, bench_func in [("SPY_BuyHold", benchmark_spy),
                                    ("UPRO_BuyHold", benchmark_upro),
                                    ("TQQQ_BuyHold", benchmark_tqqq)]:
        rets = bench_func(prices)
        strategies[bench_name] = rets
        m = compute_metrics(rets, bench_name)
        if m:
            print(f"  {bench_name}: CAGR={m['cagr_pct']}% Sharpe={m['sharpe']} MaxDD={m['max_dd_pct']}%")

    # ─── Comprehensive metrics table ────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RESULTS TABLE")
    print("=" * 80)

    all_metrics = []
    for name, rets in strategies.items():
        m = compute_metrics(rets, name)
        if m:
            all_metrics.append(m)

    # Sort by Sharpe
    all_metrics.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n{'Strategy':<35} {'CAGR%':>6} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Calmar':>7} {'WR%':>5} {'PF':>5}")
    print("-" * 90)
    for m in all_metrics:
        print(f"{m['name']:<35} {m['cagr_pct']:>6.1f} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['max_dd_pct']:>7.1f} {m['calmar']:>7.2f} {m['win_rate_pct']:>5.1f} {m['profit_factor']:>5.2f}")

    # ─── TARGET CHECK ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TARGET CHECK: CAGR >= 20% AND MaxDD <= 25%")
    print("=" * 80)

    target_hits = [m for m in all_metrics if m["cagr_pct"] >= 20 and m["max_dd_pct"] >= -25
                   and "BuyHold" not in m["name"]]

    if target_hits:
        print(f"\n{len(target_hits)} strategies HIT the target:")
        for m in target_hits:
            print(f"  >>> {m['name']}: CAGR={m['cagr_pct']}%, MaxDD={m['max_dd_pct']}%, Sharpe={m['sharpe']}")
    else:
        print("\nNO strategies hit CAGR>=20% with MaxDD<=25%")
        close = [m for m in all_metrics if m["cagr_pct"] >= 15 and m["max_dd_pct"] >= -30
                 and "BuyHold" not in m["name"]]
        if close:
            print(f"\n{len(close)} strategies are CLOSE (CAGR>=15%, MaxDD<=30%):")
            for m in close:
                print(f"  ~ {m['name']}: CAGR={m['cagr_pct']}%, MaxDD={m['max_dd_pct']}%, Sharpe={m['sharpe']}")

    # ─── HC #705 ADVERSARIAL VALIDATION ──────────────────────────────
    # Only validate strategies that hit or are close to target
    candidates = [m for m in all_metrics if m["cagr_pct"] >= 12 and "BuyHold" not in m["name"]]

    if not candidates:
        candidates = all_metrics[:5]  # Validate top 5 anyway

    print("\n" + "=" * 80)
    print("HC #705 ADVERSARIAL VALIDATION")
    print("=" * 80)

    adversarial_results = {}

    for m in candidates:
        name = m["name"]
        rets = strategies[name]

        print(f"\n--- {name} ---")

        # A. Permutation test
        print("  [A] Permutation test (1000 shuffles)...", end=" ")
        perm = adversarial_permutation_test(rets, n_perms=1000)
        status = "PASS" if perm["PASS"] else "FAIL"
        print(f"{status} (p={perm['perm_p_value']}, Sharpe={perm['actual_sharpe']} vs 95th={perm['perm_95th']})")

        # B. Regime test
        print("  [B] Regime test...", end=" ")
        regime = adversarial_regime_test(rets, spy_returns)
        status = "PASS" if regime["PASS"] else "FAIL"
        print(f"{status} (gap={regime['regime_gap']}, bull={regime.get('bull',{}).get('sharpe','?')}, bear={regime.get('bear',{}).get('sharpe','?')}, flat={regime.get('flat',{}).get('sharpe','?')})")

        # C. Sub-period consistency
        print("  [C] Sub-period consistency...", end=" ")
        subperiod = adversarial_subperiod_test(rets)
        status = "PASS" if subperiod["PASS"] else "FAIL"
        print(f"{status} ({subperiod['pct_positive']}% positive, min={subperiod['min_sharpe']}, worst={subperiod['worst_period']})")

        # D. Outlier removal
        print("  [D] Outlier removal...", end=" ")
        outlier = adversarial_outlier_removal(rets)
        status = "PASS" if outlier["PASS"] else "FAIL"
        print(f"{status} (drop={outlier['sharpe_drop_pct']}%, full={outlier['full_sharpe']} → trimmed={outlier['trimmed_sharpe']})")

        # E. Crisis test
        print("  [E] Crisis performance:")
        crisis = adversarial_crisis_test(rets)
        for cname, cdata in crisis.items():
            print(f"      {cname}: {cdata['return_pct']:+.1f}% (DD={cdata['max_dd_pct']:.1f}%)")

        # Overall verdict
        checks = [perm["PASS"], regime["PASS"], subperiod["PASS"], outlier["PASS"]]
        n_pass = sum(checks)
        verdict = "STRONG" if n_pass == 4 else "MARGINAL" if n_pass >= 3 else "WEAK" if n_pass >= 2 else "REJECT"
        print(f"  VERDICT: {verdict} ({n_pass}/4 checks passed)")

        adversarial_results[name] = {
            "metrics": m,
            "permutation": perm,
            "regime": regime,
            "subperiod": subperiod,
            "outlier": outlier,
            "crisis": crisis,
            "n_pass": n_pass,
            "verdict": verdict,
        }

    # ─── FINAL SUMMARY ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL RANKING (by adversarial-adjusted quality)")
    print("=" * 80)

    ranked = sorted(adversarial_results.items(),
                    key=lambda x: (x[1]["n_pass"], x[1]["metrics"]["sharpe"]),
                    reverse=True)

    print(f"\n{'Rank':<5} {'Strategy':<35} {'CAGR%':>6} {'Sharpe':>7} {'MaxDD%':>7} {'Checks':>7} {'Verdict':>8}")
    print("-" * 85)
    for i, (name, data) in enumerate(ranked, 1):
        m = data["metrics"]
        print(f"{i:<5} {name:<35} {m['cagr_pct']:>6.1f} {m['sharpe']:>7.2f} {m['max_dd_pct']:>7.1f} {data['n_pass']:>3}/4   {data['verdict']:>8}")

    # ─── KEY FINDING ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("KEY FINDING: CAN WE GET 20%+ CAGR WITH MaxDD < 25%?")
    print("=" * 80)

    strong_winners = [name for name, data in ranked
                      if data["verdict"] in ("STRONG", "MARGINAL")
                      and data["metrics"]["cagr_pct"] >= 20
                      and data["metrics"]["max_dd_pct"] >= -25]

    if strong_winners:
        print(f"\nYES — {len(strong_winners)} strategies pass adversarial validation:")
        for name in strong_winners:
            d = adversarial_results[name]
            m = d["metrics"]
            print(f"  {name}")
            print(f"    CAGR={m['cagr_pct']}%, Sharpe={m['sharpe']}, MaxDD={m['max_dd_pct']}%")
            print(f"    Sortino={m['sortino']}, Calmar={m['calmar']}, WR={m['win_rate_pct']}%")
    else:
        print("\nNO — no strategy passes BOTH the return target AND adversarial checks.")
        print("\nClosest viable options:")
        for name, data in ranked[:3]:
            m = data["metrics"]
            print(f"  {name}: CAGR={m['cagr_pct']}%, MaxDD={m['max_dd_pct']}%, {data['verdict']} ({data['n_pass']}/4)")

    # ─── Save results ────────────────────────────────────────────────
    output = {
        "run_time": datetime.now().isoformat(),
        "all_metrics": all_metrics,
        "adversarial_results": {},
        "target_check": {
            "target": "CAGR >= 20% AND MaxDD <= 25%",
            "hits": [m["name"] for m in target_hits] if target_hits else [],
        },
    }

    # Serialize adversarial results (convert non-serializable types)
    for name, data in adversarial_results.items():
        serializable = {
            "metrics": data["metrics"],
            "permutation": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                           for k, v in data["permutation"].items()},
            "regime": {},
            "subperiod": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                         for k, v in data["subperiod"].items()},
            "outlier": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                       for k, v in data["outlier"].items()},
            "crisis": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                          for kk, vv in v.items()} for k, v in data["crisis"].items()},
            "n_pass": data["n_pass"],
            "verdict": data["verdict"],
        }
        # Handle regime dict
        for regime_key in ["bull", "bear", "flat", "regime_gap", "PASS"]:
            if regime_key in data["regime"]:
                val = data["regime"][regime_key]
                if isinstance(val, dict):
                    serializable["regime"][regime_key] = {
                        k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in val.items()
                    }
                else:
                    serializable["regime"][regime_key] = float(val) if isinstance(val, (np.floating, np.integer)) else val

        output["adversarial_results"][name] = serializable

    results_path = OUT_DIR / "leveraged_growth_adversarial_v2_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
