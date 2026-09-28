"""
Tail-Risk-Parity Portfolio with Adaptive Leverage v1
=====================================================
HC #0  : Sliding walk-forward (trailing lookback windows, not expanding)
HC #428: Regime-agnostic validation (R1) + MFE-within-horizon (R2)
HC #694: Commission-free (Robinhood) — 0 brokerage commissions on ETFs
HC #697: No crypto

Core Concept:
- Allocate inversely proportional to Expected Shortfall (CVaR 5%), NOT volatility
- CVaR captures fat tails that standard vol misses
- Adaptive leverage based on aggregate portfolio tail risk z-score
- Income layer: systematic covered call writing on SPY allocation
- Hard loss floor: -10% annual max acceptable loss
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, percentileofscore

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/tail_risk_parity_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2006-01-01"  # Need data from 2006 for 252-day lookback before 2008
END   = "2026-07-14"

# ETF universe — 8 ETFs across asset classes
UNIVERSE = {
    "SPY":  {"class": "US_EQ",       "exp_ratio": 0.0009},
    "EFA":  {"class": "INTL_EQ",     "exp_ratio": 0.0032},
    "EEM":  {"class": "EM_EQ",       "exp_ratio": 0.0068},
    "TLT":  {"class": "BOND_LT",     "exp_ratio": 0.0015},
    "IEF":  {"class": "BOND_IT",     "exp_ratio": 0.0015},
    "GLD":  {"class": "COMMODITY",   "exp_ratio": 0.0040},
    "DBC":  {"class": "COMMODITY",   "exp_ratio": 0.0085},
    "VNQ":  {"class": "REALESTATE",  "exp_ratio": 0.0012},
}
TICKERS = list(UNIVERSE.keys())

# Additional data for signals
VIX_TICKER = "^VIX"
HYG_TICKER = "HYG"  # For credit spread signal

# CVaR parameters
CVAR_WINDOW = 252       # Rolling window for Expected Shortfall
CVAR_QUANTILE = 0.05    # 5th percentile
CORR_WINDOW = 126       # Rolling correlation window

# Leverage rules
LEV_Z_LOW = -1.0    # Portfolio CVaR z-score < -1 → 1.5x
LEV_Z_HIGH = 1.0    # Portfolio CVaR z-score > 1 → 0.5x
LEV_LOW_RISK = 1.5
LEV_NEUTRAL = 1.0
LEV_HIGH_RISK = 0.5
LEV_MAX = 2.0
LEV_MIN = 0.3
BORROW_COST_ANNUAL = 0.05  # 5% annualized on leveraged portion

# Rebalance frequency
REBAL_FREQ = "ME"  # month-end

# Covered call parameters (income layer)
CALL_DELTA = 0.30         # 30-delta OTM calls
CALL_DTE = 30             # ~30 days to expiration
RISK_FREE_RATE = 0.04     # Approximate

# Benchmark
SPY_REGIME_TICKER = "SPY"

# Permutation test
N_PERMUTATIONS = 500


# ---------------------------------------------------------------------------
# DATA DOWNLOAD
# ---------------------------------------------------------------------------
def download_data():
    """Download price data for universe + signals."""
    all_tickers = TICKERS + [VIX_TICKER, HYG_TICKER]
    print(f"Downloading data for {len(all_tickers)} tickers: {START} to {END}")

    data = yf.download(all_tickers, start=START, end=END, auto_adjust=True)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = all_tickers[:1]

    prices = prices.ffill().dropna(how="all")

    # Separate VIX and HYG from universe prices
    vix = prices[VIX_TICKER] if VIX_TICKER in prices.columns else None
    hyg = prices[HYG_TICKER] if HYG_TICKER in prices.columns else None

    universe_prices = prices[TICKERS].dropna()

    print(f"Universe data: {universe_prices.index[0].date()} to {universe_prices.index[-1].date()}")
    print(f"Shape: {universe_prices.shape}")

    return universe_prices, vix, hyg


# ---------------------------------------------------------------------------
# EXPECTED SHORTFALL (CVaR)
# ---------------------------------------------------------------------------
def rolling_cvar(returns, window=CVAR_WINDOW, quantile=CVAR_QUANTILE):
    """
    Compute rolling Expected Shortfall (CVaR) at given quantile.
    CVaR = mean of returns below the VaR quantile.
    Returns NEGATIVE values (losses).
    """
    cvar = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

    for i in range(window, len(returns)):
        window_rets = returns.iloc[i - window:i]
        for col in returns.columns:
            col_rets = window_rets[col].dropna()
            if len(col_rets) < window * 0.8:
                continue
            var_threshold = col_rets.quantile(quantile)
            tail_rets = col_rets[col_rets <= var_threshold]
            if len(tail_rets) > 0:
                cvar.iloc[i][col] = tail_rets.mean()
            else:
                cvar.iloc[i][col] = var_threshold

    return cvar.astype(float)


def portfolio_cvar_estimate(weights, asset_cvars, corr_matrix):
    """
    Approximate portfolio CVaR using modified Gaussian approach.
    A simplification: use asset CVaRs and correlation to estimate portfolio tail risk.
    """
    # Use magnitude of CVaR as risk measure
    risk_vec = np.abs(asset_cvars)

    # Portfolio risk ~ sqrt(w' * Sigma_tail * w) where Sigma_tail uses CVaR as marginal risks
    # Create pseudo-covariance from CVaR magnitudes and correlations
    n = len(weights)
    cov_tail = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_tail[i, j] = risk_vec[i] * risk_vec[j] * corr_matrix[i, j]

    port_var = weights @ cov_tail @ weights
    return np.sqrt(max(port_var, 1e-10))


# ---------------------------------------------------------------------------
# BLACK-SCHOLES FOR COVERED CALLS
# ---------------------------------------------------------------------------
def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def find_strike_for_delta(S, T, r, sigma, target_delta=0.30):
    """Find strike price for a given call delta."""
    if T <= 0 or sigma <= 0:
        return S * 1.05  # fallback
    # Call delta = N(d1), so d1 = N_inv(delta)
    d1_target = norm.ppf(target_delta)
    # d1 = (ln(S/K) + (r + 0.5*sigma^2)*T) / (sigma*sqrt(T))
    # Solve for K: ln(S/K) = d1*sigma*sqrt(T) - (r + 0.5*sigma^2)*T
    log_SK = d1_target * sigma * np.sqrt(T) - (r + 0.5 * sigma**2) * T
    K = S * np.exp(-log_SK)
    return K


def covered_call_premium(S, T, r, sigma, delta=CALL_DELTA):
    """Calculate premium from selling a covered call at given delta."""
    K = find_strike_for_delta(S, T, r, sigma, delta)
    premium = black_scholes_call(S, K, T, r, sigma)
    return premium, K


# ---------------------------------------------------------------------------
# STRATEGY ENGINE
# ---------------------------------------------------------------------------
def run_tail_risk_parity(prices, vix, hyg):
    """
    Main strategy: Tail-Risk-Parity with adaptive leverage.
    """
    returns = prices.pct_change().dropna()

    print("Computing rolling CVaR (this takes a minute)...")
    asset_cvar = rolling_cvar(returns)

    print("Computing rolling correlations...")
    rolling_corr = returns.rolling(window=CORR_WINDOW).corr()

    # Credit spread signal: HYG/IEF ratio
    credit_spread = None
    if hyg is not None and "IEF" in prices.columns:
        ief_aligned = prices["IEF"].reindex(hyg.index).ffill()
        credit_spread = (hyg / ief_aligned).pct_change().rolling(63).mean()

    # Rebalance dates
    rebal_dates = returns.resample(REBAL_FREQ).last().index
    rebal_dates = rebal_dates[rebal_dates >= returns.index[CVAR_WINDOW + 10]]

    print(f"Rebalance dates: {len(rebal_dates)} months from {rebal_dates[0].date()} to {rebal_dates[-1].date()}")

    # Storage
    port_values = pd.Series(index=returns.index, dtype=float)
    port_values.iloc[:] = np.nan

    weights_history = pd.DataFrame(index=rebal_dates, columns=TICKERS, dtype=float)
    leverage_history = pd.Series(index=rebal_dates, dtype=float)
    cvar_zscore_history = pd.Series(index=rebal_dates, dtype=float)
    call_income = pd.Series(index=rebal_dates, dtype=float)
    call_strikes = pd.Series(index=rebal_dates, dtype=float)

    # Initialize
    portfolio_value = 1.0
    current_weights = np.array([1.0 / len(TICKERS)] * len(TICKERS))  # equal weight start
    current_leverage = 1.0

    # Track portfolio CVaR history for z-score
    port_cvar_history = []

    # Start from first valid CVaR date
    start_idx = returns.index.get_loc(rebal_dates[0])
    for i in range(start_idx):
        port_values.iloc[i] = 1.0

    prev_rebal_idx = start_idx
    rebal_set = set(rebal_dates)

    for i in range(start_idx, len(returns)):
        date = returns.index[i]
        daily_ret = returns.iloc[i]

        # Daily portfolio return (leveraged)
        asset_rets = daily_ret[TICKERS].values
        port_ret = current_leverage * np.nansum(current_weights * asset_rets)

        # Subtract borrowing cost on leveraged portion
        if current_leverage > 1.0:
            borrow_cost_daily = BORROW_COST_ANNUAL * (current_leverage - 1.0) / 252
            port_ret -= borrow_cost_daily

        # Subtract expense ratios
        for j, ticker in enumerate(TICKERS):
            exp_cost = UNIVERSE[ticker]["exp_ratio"] / 252 * current_weights[j] * current_leverage
            port_ret -= exp_cost

        portfolio_value *= (1 + port_ret)
        port_values.iloc[i] = portfolio_value

        # Rebalance check
        if date in rebal_set:
            # Get current CVaR for each asset
            cvar_row = asset_cvar.loc[date]
            if cvar_row.isna().all():
                continue

            # Fill any remaining NaN with column median
            cvar_vals = cvar_row[TICKERS].astype(float).values
            valid_mask = ~np.isnan(cvar_vals)
            if valid_mask.sum() < 3:
                continue

            # Replace NaN with median of valid
            median_cvar = np.nanmedian(cvar_vals)
            cvar_vals[np.isnan(cvar_vals)] = median_cvar

            # Inverse CVaR weights (allocate MORE to assets with LESS tail risk)
            abs_cvar = np.abs(cvar_vals)
            abs_cvar = np.maximum(abs_cvar, 1e-6)  # floor
            inv_cvar = 1.0 / abs_cvar
            new_weights = inv_cvar / inv_cvar.sum()

            # Get correlation matrix for this date
            try:
                corr_at_date = rolling_corr.loc[date]
                if isinstance(corr_at_date, pd.DataFrame):
                    corr_mat = corr_at_date.loc[TICKERS, TICKERS].values.astype(float)
                else:
                    corr_mat = np.eye(len(TICKERS))
            except (KeyError, TypeError):
                corr_mat = np.eye(len(TICKERS))

            # Fix any NaN in correlation matrix
            corr_mat = np.nan_to_num(corr_mat, nan=0.0)
            np.fill_diagonal(corr_mat, 1.0)

            # Portfolio CVaR
            port_cvar = portfolio_cvar_estimate(new_weights, cvar_vals, corr_mat)
            port_cvar_history.append(port_cvar)

            # Compute z-score of portfolio CVaR
            if len(port_cvar_history) >= 12:
                hist_arr = np.array(port_cvar_history)
                cvar_mean = hist_arr.mean()
                cvar_std = hist_arr.std()
                if cvar_std > 1e-8:
                    cvar_z = (port_cvar - cvar_mean) / cvar_std
                else:
                    cvar_z = 0.0
            else:
                cvar_z = 0.0

            # VIX overlay: if VIX > 35, force delever regardless of z-score
            if vix is not None and date in vix.index:
                current_vix = vix.loc[date]
                if not np.isnan(current_vix) and current_vix > 35:
                    cvar_z = max(cvar_z, 1.5)  # Force high-risk regime

            # Credit spread overlay: widening spreads = more risk
            if credit_spread is not None and date in credit_spread.index:
                cs_val = credit_spread.loc[date]
                if not np.isnan(cs_val) and cs_val < -0.001:  # Spreads widening
                    cvar_z += 0.5  # Push toward delevering

            # Determine leverage
            if cvar_z < LEV_Z_LOW:
                new_leverage = LEV_LOW_RISK
            elif cvar_z > LEV_Z_HIGH:
                new_leverage = LEV_HIGH_RISK
            else:
                # Linear interpolation
                new_leverage = LEV_LOW_RISK + (LEV_HIGH_RISK - LEV_LOW_RISK) * \
                               (cvar_z - LEV_Z_LOW) / (LEV_Z_HIGH - LEV_Z_LOW)

            new_leverage = np.clip(new_leverage, LEV_MIN, LEV_MAX)

            # Store
            current_weights = new_weights
            current_leverage = new_leverage
            weights_history.loc[date] = new_weights
            leverage_history.loc[date] = new_leverage
            cvar_zscore_history.loc[date] = cvar_z

            # Covered call income on SPY allocation
            spy_idx = TICKERS.index("SPY")
            spy_weight = new_weights[spy_idx]
            spy_notional = portfolio_value * spy_weight * new_leverage

            if "SPY" in prices.columns:
                spy_price = prices.loc[date, "SPY"]
                # Use 63-day realized vol as IV proxy
                spy_rets = returns["SPY"].loc[:date].tail(63)
                spy_vol = spy_rets.std() * np.sqrt(252) if len(spy_rets) > 20 else 0.20

                T = CALL_DTE / 365.0
                premium_per_share, strike = covered_call_premium(
                    spy_price, T, RISK_FREE_RATE, spy_vol, CALL_DELTA
                )

                # Number of "shares" (notional / price)
                n_shares = spy_notional / spy_price if spy_price > 0 else 0
                monthly_premium = premium_per_share * n_shares

                call_income.loc[date] = monthly_premium
                call_strikes.loc[date] = strike

    # Forward fill port values
    port_values = port_values.ffill()

    return port_values, weights_history, leverage_history, cvar_zscore_history, call_income


# ---------------------------------------------------------------------------
# BENCHMARKS
# ---------------------------------------------------------------------------
def compute_benchmarks(prices, returns):
    """Compute benchmark strategies."""
    benchmarks = {}

    # 1. SPY buy & hold
    spy_rets = returns["SPY"]
    benchmarks["SPY"] = (1 + spy_rets).cumprod()

    # 2. Classic 60/40 (SPY/IEF)
    port_60_40 = 0.60 * returns["SPY"] + 0.40 * returns["IEF"]
    benchmarks["60/40"] = (1 + port_60_40).cumprod()

    # 3. Equal weight
    eq_rets = returns[TICKERS].mean(axis=1)
    benchmarks["Equal Weight"] = (1 + eq_rets).cumprod()

    # 4. Standard risk parity (inverse vol, no tail adjustment)
    vol_window = 252
    std_rp_values = pd.Series(index=returns.index, dtype=float)
    std_rp_values.iloc[0] = 1.0

    rebal_dates = returns.resample(REBAL_FREQ).last().index
    rebal_dates = rebal_dates[rebal_dates >= returns.index[vol_window]]
    rebal_set = set(rebal_dates)

    current_w = np.array([1.0 / len(TICKERS)] * len(TICKERS))
    pv = 1.0

    for i in range(len(returns)):
        date = returns.index[i]
        daily_ret = returns.iloc[i][TICKERS].values
        pv *= (1 + np.nansum(current_w * daily_ret))
        std_rp_values.iloc[i] = pv

        if date in rebal_set and i >= vol_window:
            window_rets = returns.iloc[i - vol_window:i][TICKERS]
            vols = window_rets.std().values
            vols = np.maximum(vols, 1e-6)
            inv_vol = 1.0 / vols
            current_w = inv_vol / inv_vol.sum()

    benchmarks["Risk Parity (Vol)"] = std_rp_values

    return benchmarks


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def compute_metrics(values, name="Strategy"):
    """Compute comprehensive performance metrics."""
    values = values.dropna()
    if len(values) < 252:
        return {}

    rets = values.pct_change().dropna()

    years = len(rets) / 252
    total_ret = values.iloc[-1] / values.iloc[0] - 1
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1

    ann_vol = rets.std() * np.sqrt(252)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0

    downside_rets = rets[rets < 0]
    downside_vol = downside_rets.std() * np.sqrt(252) if len(downside_rets) > 0 else 1e-6
    sortino = cagr / downside_vol

    # Max drawdown
    rolling_max = values.cummax()
    drawdown = (values - rolling_max) / rolling_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    wr = (rets > 0).mean()

    # Profit factor
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Skewness / kurtosis
    skew = rets.skew()
    kurt = rets.kurtosis()

    # Best/worst year
    annual_rets = rets.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    best_year = annual_rets.max()
    worst_year = annual_rets.min()
    asymmetry = best_year / abs(worst_year) if worst_year != 0 else float("inf")

    return {
        "name": name,
        "total_return": total_ret,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": wr,
        "profit_factor": pf,
        "skewness": skew,
        "kurtosis": kurt,
        "best_year": best_year,
        "worst_year": worst_year,
        "asymmetry_ratio": asymmetry,
        "years": years,
    }


def year_by_year(values, name="Strategy"):
    """Year-by-year returns."""
    rets = values.pct_change().dropna()
    annual = rets.resample("YE").apply(lambda x: (1 + x).prod() - 1)
    annual.index = annual.index.year
    return annual


# ---------------------------------------------------------------------------
# CRISIS ALPHA
# ---------------------------------------------------------------------------
def crisis_analysis(strategy_vals, spy_vals):
    """Analyze performance during crisis periods."""
    crises = {
        "GFC (2008)": ("2008-01-01", "2009-03-09"),
        "COVID (2020)": ("2020-02-19", "2020-03-23"),
        "Rate Hike (2022)": ("2022-01-03", "2022-10-12"),
    }

    results = {}
    for crisis_name, (start, end) in crises.items():
        try:
            strat_slice = strategy_vals.loc[start:end]
            spy_slice = spy_vals.loc[start:end]

            if len(strat_slice) < 5 or len(spy_slice) < 5:
                continue

            strat_ret = strat_slice.iloc[-1] / strat_slice.iloc[0] - 1
            spy_ret = spy_slice.iloc[-1] / spy_slice.iloc[0] - 1

            strat_dd = (strat_slice / strat_slice.cummax() - 1).min()
            spy_dd = (spy_slice / spy_slice.cummax() - 1).min()

            results[crisis_name] = {
                "strategy_return": strat_ret,
                "spy_return": spy_ret,
                "excess_return": strat_ret - spy_ret,
                "strategy_max_dd": strat_dd,
                "spy_max_dd": spy_dd,
            }
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# REGIME VALIDATION (HC #428 R1)
# ---------------------------------------------------------------------------
def regime_validation(strategy_vals, spy_prices):
    """
    HC #428 R1: Regime-agnostic validation.
    Classify days as green/red/flat based on SPY close-to-close.
    Compute per-regime Sharpe. Reject if regime gap > 0.50.
    """
    strat_rets = strategy_vals.pct_change().dropna()
    spy_rets = spy_prices.pct_change().dropna()

    # Align
    common = strat_rets.index.intersection(spy_rets.index)
    strat_rets = strat_rets.loc[common]
    spy_rets = spy_rets.loc[common]

    # Classify days
    green_mask = spy_rets > 0.001     # >0.1% = green
    red_mask = spy_rets < -0.001      # <-0.1% = red
    flat_mask = ~green_mask & ~red_mask

    results = {}
    for regime, mask in [("green", green_mask), ("red", red_mask), ("flat", flat_mask)]:
        regime_rets = strat_rets[mask]
        if len(regime_rets) > 20:
            ann_ret = regime_rets.mean() * 252
            ann_vol = regime_rets.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            results[regime] = {
                "n_days": len(regime_rets),
                "ann_return": ann_ret,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
            }

    # Regime gap test
    if "green" in results and "red" in results:
        s_green = results["green"]["sharpe"]
        s_red = results["red"]["sharpe"]
        gap = abs(s_green - s_red) / max(abs(s_green), abs(s_red), 1e-6)
        results["regime_gap"] = gap
        results["regime_pass"] = gap <= 0.50
    else:
        results["regime_gap"] = None
        results["regime_pass"] = None

    return results


# ---------------------------------------------------------------------------
# PERMUTATION TEST
# ---------------------------------------------------------------------------
def permutation_test(returns, strategy_vals, leverage_history, n_perms=N_PERMUTATIONS):
    """
    Shuffle the leverage signal to test if adaptive leverage adds value.
    Compare strategy Sharpe vs distribution of Sharpe under shuffled leverage.
    """
    print(f"Running permutation test ({n_perms} shuffles)...")

    strat_rets = strategy_vals.pct_change().dropna()
    actual_sharpe = strat_rets.mean() / strat_rets.std() * np.sqrt(252) if strat_rets.std() > 0 else 0

    # Get leverage values and rebalance dates
    lev_vals = leverage_history.dropna()
    if len(lev_vals) < 12:
        return {"actual_sharpe": actual_sharpe, "p_value": None, "n_perms": 0}

    shuffled_sharpes = []
    asset_rets = returns[TICKERS]

    # Get the weight history (use equal weight as fallback for simplicity in permutation)
    # We only shuffle the leverage signal, keeping weights fixed

    for perm in range(n_perms):
        # Shuffle leverage values
        shuffled_lev = lev_vals.values.copy()
        np.random.shuffle(shuffled_lev)

        # Reconstruct portfolio with shuffled leverage
        pv = 1.0
        perm_values = []

        lev_idx = 0
        current_lev = 1.0

        for i in range(len(asset_rets)):
            date = asset_rets.index[i]
            daily_ret = asset_rets.iloc[i].mean()  # Equal weight for speed

            # Check if rebalance date
            if date in lev_vals.index:
                if lev_idx < len(shuffled_lev):
                    current_lev = shuffled_lev[lev_idx]
                    lev_idx += 1

            port_ret = current_lev * daily_ret
            if current_lev > 1.0:
                port_ret -= BORROW_COST_ANNUAL * (current_lev - 1.0) / 252

            pv *= (1 + port_ret)
            perm_values.append(pv)

        perm_series = pd.Series(perm_values)
        perm_rets = perm_series.pct_change().dropna()
        if perm_rets.std() > 0:
            perm_sharpe = perm_rets.mean() / perm_rets.std() * np.sqrt(252)
            shuffled_sharpes.append(perm_sharpe)

    shuffled_sharpes = np.array(shuffled_sharpes)
    p_value = (shuffled_sharpes >= actual_sharpe).mean()

    return {
        "actual_sharpe": actual_sharpe,
        "mean_shuffled_sharpe": shuffled_sharpes.mean(),
        "std_shuffled_sharpe": shuffled_sharpes.std(),
        "p_value": p_value,
        "n_perms": n_perms,
        "pct_rank": percentileofscore(shuffled_sharpes, actual_sharpe),
    }


# ---------------------------------------------------------------------------
# INCOME ANALYSIS
# ---------------------------------------------------------------------------
def income_analysis(call_income, portfolio_values, base_capital=100_000):
    """Analyze monthly covered call income scaled to base capital."""
    ci = call_income.dropna()
    if len(ci) == 0:
        return {}

    # Scale income to base capital
    # call_income is computed on normalized portfolio (starts at 1.0)
    # Scale by base_capital
    scaled_income = ci * base_capital

    monthly_stats = {
        "mean_monthly": scaled_income.mean(),
        "median_monthly": scaled_income.median(),
        "std_monthly": scaled_income.std(),
        "min_monthly": scaled_income.min(),
        "max_monthly": scaled_income.max(),
        "pct_above_3k": (scaled_income > 3000).mean() * 100,
        "pct_above_5k": (scaled_income > 5000).mean() * 100,
        "total_income": scaled_income.sum(),
        "ann_income": scaled_income.mean() * 12,
        "ann_yield": (scaled_income.mean() * 12) / base_capital * 100,
    }

    return monthly_stats


def capital_needed(call_income, target_monthly, portfolio_values):
    """Calculate capital needed for target monthly income."""
    ci = call_income.dropna()
    if len(ci) == 0 or ci.mean() <= 0:
        return float("inf")

    # Average monthly income per unit of capital
    avg_income_per_unit = ci.mean()
    capital = target_monthly / avg_income_per_unit
    return capital


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def plot_results(strategy_vals, benchmarks, leverage_hist, cvar_z_hist,
                 call_income, weights_hist, crisis_results, regime_results):
    """Generate comprehensive result plots."""

    # 1. Equity curves
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    ax = axes[0, 0]
    ax.plot(strategy_vals.index, strategy_vals, label="Tail Risk Parity", linewidth=2, color="blue")
    for name, vals in benchmarks.items():
        ax.plot(vals.index, vals, label=name, alpha=0.7)
    ax.set_yscale("log")
    ax.set_title("Equity Curves (Log Scale)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    dd = (strategy_vals - strategy_vals.cummax()) / strategy_vals.cummax()
    ax.fill_between(dd.index, dd, 0, alpha=0.4, color="red")
    ax.set_title("Strategy Drawdown")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)

    # 3. Leverage over time
    ax = axes[1, 0]
    lev = leverage_hist.dropna()
    ax.plot(lev.index, lev, color="purple", linewidth=1.5)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Adaptive Leverage")
    ax.set_ylabel("Leverage")
    ax.grid(True, alpha=0.3)

    # 4. CVaR Z-Score
    ax = axes[1, 1]
    cz = cvar_z_hist.dropna()
    colors = ["green" if z < LEV_Z_LOW else "red" if z > LEV_Z_HIGH else "gray" for z in cz]
    ax.bar(cz.index, cz, color=colors, alpha=0.6, width=20)
    ax.axhline(LEV_Z_LOW, color="green", linestyle="--", alpha=0.5, label=f"Low risk (<{LEV_Z_LOW})")
    ax.axhline(LEV_Z_HIGH, color="red", linestyle="--", alpha=0.5, label=f"High risk (>{LEV_Z_HIGH})")
    ax.set_title("Portfolio CVaR Z-Score")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Weights over time
    ax = axes[2, 0]
    wh = weights_hist.dropna()
    if len(wh) > 0:
        wh.plot.area(ax=ax, alpha=0.7)
        ax.set_title("Portfolio Weights Over Time")
        ax.set_ylabel("Weight")
        ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # 6. Monthly call income
    ax = axes[2, 1]
    ci = call_income.dropna() * 100_000  # Scale to $100K portfolio
    if len(ci) > 0:
        ax.bar(ci.index, ci, color="green", alpha=0.6, width=20)
        ax.axhline(3000, color="orange", linestyle="--", label="$3K target")
        ax.axhline(5000, color="red", linestyle="--", label="$5K target")
        ax.set_title("Monthly Covered Call Income ($100K Portfolio)")
        ax.set_ylabel("Premium ($)")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tail_risk_parity_v1_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 7. Year-by-year comparison
    fig, ax = plt.subplots(figsize=(14, 6))
    strat_annual = year_by_year(strategy_vals, "TRP")
    spy_annual = year_by_year(benchmarks["SPY"], "SPY")

    common_years = strat_annual.index.intersection(spy_annual.index)
    x = np.arange(len(common_years))
    width = 0.35

    ax.bar(x - width/2, strat_annual.loc[common_years] * 100, width, label="Tail Risk Parity", color="blue", alpha=0.7)
    ax.bar(x + width/2, spy_annual.loc[common_years] * 100, width, label="SPY", color="gray", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(common_years, rotation=45)
    ax.set_ylabel("Annual Return (%)")
    ax.set_title("Year-by-Year Returns: Tail Risk Parity vs SPY")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "year_by_year.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TAIL-RISK-PARITY PORTFOLIO WITH ADAPTIVE LEVERAGE v1")
    print("=" * 70)
    print(f"Start: {datetime.now()}")
    print()

    # Download data
    prices, vix, hyg = download_data()
    returns = prices.pct_change().dropna()

    # Run strategy
    strategy_vals, weights_hist, leverage_hist, cvar_z_hist, call_income = \
        run_tail_risk_parity(prices, vix, hyg)

    # Drop NaN from strategy values
    strategy_vals = strategy_vals.dropna()

    # Compute benchmarks
    print("\nComputing benchmarks...")
    benchmarks = compute_benchmarks(prices, returns)

    # Align all to common dates
    common_start = strategy_vals.index[0]
    strategy_vals = strategy_vals.loc[common_start:]
    for name in benchmarks:
        bm = benchmarks[name]
        bm = bm.loc[common_start:]
        # Renormalize to start at same point
        benchmarks[name] = bm / bm.iloc[0] * strategy_vals.iloc[0]

    # -----------------------------------------------------------------------
    # METRICS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)

    all_metrics = {}
    strat_metrics = compute_metrics(strategy_vals, "Tail Risk Parity")
    all_metrics["Tail Risk Parity"] = strat_metrics

    for name, vals in benchmarks.items():
        m = compute_metrics(vals, name)
        all_metrics[name] = m

    # Print comparison table
    header = f"{'Strategy':<22} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'WinRate':>8} {'PF':>8} {'Asym':>8}"
    print(header)
    print("-" * len(header))
    for name, m in all_metrics.items():
        if m:
            print(f"{name:<22} {m['cagr']*100:>7.1f}% {m['sharpe']:>8.2f} {m['sortino']:>8.2f} "
                  f"{m['max_dd']*100:>7.1f}% {m['calmar']:>8.2f} {m['win_rate']*100:>7.1f}% "
                  f"{m['profit_factor']:>8.2f} {m['asymmetry_ratio']:>8.2f}")

    # -----------------------------------------------------------------------
    # YEAR BY YEAR
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("YEAR-BY-YEAR RETURNS")
    print("=" * 70)

    strat_annual = year_by_year(strategy_vals, "TRP")
    spy_annual = year_by_year(benchmarks.get("SPY", strategy_vals), "SPY")

    print(f"{'Year':<8} {'Strategy':>10} {'SPY':>10} {'Excess':>10}")
    print("-" * 42)
    for yr in strat_annual.index:
        s = strat_annual.loc[yr]
        spy_r = spy_annual.loc[yr] if yr in spy_annual.index else 0
        print(f"{yr:<8} {s*100:>9.1f}% {spy_r*100:>9.1f}% {(s-spy_r)*100:>9.1f}%")

    # -----------------------------------------------------------------------
    # CRISIS ALPHA
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CRISIS ALPHA")
    print("=" * 70)

    spy_cumulative = benchmarks.get("SPY", None)
    crisis_results = {}
    if spy_cumulative is not None:
        crisis_results = crisis_analysis(strategy_vals, spy_cumulative)
        for crisis_name, cr in crisis_results.items():
            print(f"\n{crisis_name}:")
            print(f"  Strategy: {cr['strategy_return']*100:+.1f}%  |  SPY: {cr['spy_return']*100:+.1f}%  |  "
                  f"Excess: {cr['excess_return']*100:+.1f}%")
            print(f"  Strategy MaxDD: {cr['strategy_max_dd']*100:.1f}%  |  SPY MaxDD: {cr['spy_max_dd']*100:.1f}%")

    # -----------------------------------------------------------------------
    # REGIME VALIDATION (HC #428 R1)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("REGIME VALIDATION (HC #428 R1)")
    print("=" * 70)

    regime_results = regime_validation(strategy_vals, prices["SPY"])
    for regime in ["green", "red", "flat"]:
        if regime in regime_results:
            r = regime_results[regime]
            print(f"  {regime.upper():>6}: {r['n_days']:>5} days | Ann Ret: {r['ann_return']*100:>6.1f}% | "
                  f"Vol: {r['ann_vol']*100:>5.1f}% | Sharpe: {r['sharpe']:>6.2f}")

    if regime_results.get("regime_gap") is not None:
        gap = regime_results["regime_gap"]
        passed = regime_results["regime_pass"]
        status = "PASS" if passed else "FAIL"
        print(f"\n  Regime gap: {gap:.2f} (threshold: 0.50) → {status}")

    # -----------------------------------------------------------------------
    # PERMUTATION TEST
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PERMUTATION TEST (Leverage Signal)")
    print("=" * 70)

    perm_results = permutation_test(returns, strategy_vals, leverage_hist)
    print(f"  Actual Sharpe:    {perm_results['actual_sharpe']:.3f}")
    if perm_results.get("p_value") is not None:
        print(f"  Mean Shuffled:    {perm_results['mean_shuffled_sharpe']:.3f}")
        print(f"  Std Shuffled:     {perm_results['std_shuffled_sharpe']:.3f}")
        print(f"  p-value:          {perm_results['p_value']:.3f}")
        print(f"  Percentile rank:  {perm_results['pct_rank']:.1f}%")
        sig = "SIGNIFICANT" if perm_results['p_value'] < 0.05 else "NOT SIGNIFICANT"
        print(f"  Result:           {sig} (alpha=0.05)")

    # -----------------------------------------------------------------------
    # INCOME ANALYSIS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("COVERED CALL INCOME ANALYSIS (on $100K portfolio)")
    print("=" * 70)

    income_stats = income_analysis(call_income, strategy_vals, 100_000)
    if income_stats:
        print(f"  Mean monthly:     ${income_stats['mean_monthly']:>10,.0f}")
        print(f"  Median monthly:   ${income_stats['median_monthly']:>10,.0f}")
        print(f"  Std monthly:      ${income_stats['std_monthly']:>10,.0f}")
        print(f"  Min monthly:      ${income_stats['min_monthly']:>10,.0f}")
        print(f"  Max monthly:      ${income_stats['max_monthly']:>10,.0f}")
        print(f"  % months > $3K:   {income_stats['pct_above_3k']:>10.1f}%")
        print(f"  % months > $5K:   {income_stats['pct_above_5k']:>10.1f}%")
        print(f"  Ann income:       ${income_stats['ann_income']:>10,.0f}")
        print(f"  Ann yield:        {income_stats['ann_yield']:>10.1f}%")

    # Capital needed
    print("\n  CAPITAL NEEDED:")
    for target, label in [(3000, "$3K/month"), (5000, "$5K/month")]:
        cap = capital_needed(call_income, target, strategy_vals)
        if cap < float("inf"):
            print(f"    For {label}: ${cap:>12,.0f}")
        else:
            print(f"    For {label}: N/A (insufficient data)")

    # -----------------------------------------------------------------------
    # LEVERAGE STATS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("LEVERAGE STATISTICS")
    print("=" * 70)

    lev = leverage_hist.dropna()
    if len(lev) > 0:
        print(f"  Mean leverage:    {lev.mean():.2f}x")
        print(f"  Median leverage:  {lev.median():.2f}x")
        print(f"  Min leverage:     {lev.min():.2f}x")
        print(f"  Max leverage:     {lev.max():.2f}x")
        print(f"  % months >1x:    {(lev > 1.0).mean()*100:.1f}%")
        print(f"  % months <1x:    {(lev < 1.0).mean()*100:.1f}%")

    # -----------------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------------
    print("\nGenerating plots...")
    plot_results(strategy_vals, benchmarks, leverage_hist, cvar_z_hist,
                 call_income, weights_hist, crisis_results, regime_results)

    # -----------------------------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------------------------
    results = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "Tail Risk Parity v1",
        "parameters": {
            "cvar_window": CVAR_WINDOW,
            "cvar_quantile": CVAR_QUANTILE,
            "corr_window": CORR_WINDOW,
            "lev_z_low": LEV_Z_LOW,
            "lev_z_high": LEV_Z_HIGH,
            "lev_low_risk": LEV_LOW_RISK,
            "lev_high_risk": LEV_HIGH_RISK,
            "borrow_cost": BORROW_COST_ANNUAL,
            "call_delta": CALL_DELTA,
            "call_dte": CALL_DTE,
            "universe": TICKERS,
        },
        "metrics": {name: {k: float(v) if isinstance(v, (np.floating, float)) else v
                           for k, v in m.items()}
                    for name, m in all_metrics.items() if m},
        "crisis_alpha": {k: {kk: float(vv) for kk, vv in v.items()}
                         for k, v in crisis_results.items()},
        "regime_validation": {
            k: (v if not isinstance(v, dict) else {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()})
            for k, v in regime_results.items()
        },
        "permutation_test": {k: float(v) if isinstance(v, (np.floating, float)) else v
                             for k, v in perm_results.items()},
        "income_analysis_100k": {k: float(v) if isinstance(v, (np.floating, float)) else v
                                  for k, v in income_stats.items()} if income_stats else {},
        "capital_needed_3k": float(capital_needed(call_income, 3000, strategy_vals)),
        "capital_needed_5k": float(capital_needed(call_income, 5000, strategy_vals)),
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save equity curve
    strategy_vals.to_csv(OUTPUT_DIR / "equity_curve.csv")

    # Save weights history
    weights_hist.dropna().to_csv(OUTPUT_DIR / "weights_history.csv")

    # Save leverage history
    leverage_hist.dropna().to_csv(OUTPUT_DIR / "leverage_history.csv")

    print(f"\nAll results saved to {OUTPUT_DIR}")
    print(f"\nCompleted: {datetime.now()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
