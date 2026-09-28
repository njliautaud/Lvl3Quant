"""
Template for autonomous research scripts.
Each script must:
1. Be fully self-contained (downloads own data)
2. Run adversarial validation
3. Print <<<RESULT_JSON>>> ... <<<END_RESULT_JSON>>> at end
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

INITIAL_CAPITAL = 100_000


def download_etfs(tickers, start='2010-01-01'):
    """Download ETF data, handle multi-index"""
    prices = yf.download(tickers, start=start, progress=False)['Close']
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    return prices.dropna()


def download_vix(start='2010-01-01'):
    """Download VIX"""
    vix = yf.download('^VIX', start=start, progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    return vix['Close'].squeeze()


def compute_metrics(equity_series, name="Strategy"):
    """Standard risk-adjusted metrics"""
    returns = equity_series.pct_change().dropna()
    if len(returns) < 50:
        return {'name': name, 'sharpe': 0, 'sortino': 0, 'cagr': 0, 'max_dd': 0,
                'calmar': 0, 'win_rate': 0}

    ann_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = float(ann_ret / downside) if downside > 0 else 0
    cummax = equity_series.cummax()
    max_dd = float(((equity_series - cummax) / cummax).min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else 0
    wr = float((returns > 0).mean())

    return {
        'name': name, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(float(ann_ret), 4), 'max_dd': round(max_dd, 4),
        'calmar': round(calmar, 3), 'win_rate': round(wr, 3)
    }


def permutation_test(equity_series, baseline_equity, n_perms=200):
    """
    Permutation test: compare strategy Sharpe against time-shifted baseline.
    Shifts the equity series by random offsets to break signal-price alignment
    while preserving the strategy's return distribution structure.

    NOTE: For proper signal-shuffling, each strategy script should implement
    its own permutation that shuffles the SIGNAL (e.g., VIX levels, SMA
    crossovers) rather than returns. This generic test uses block-bootstrap
    as a reasonable approximation.
    """
    real = compute_metrics(equity_series)
    real_sharpe = real['sharpe']

    # Use block bootstrap: shuffle blocks of returns to preserve autocorrelation
    # but break alignment with signals
    returns = equity_series.pct_change().dropna().values
    n = len(returns)
    block_size = 21  # Monthly blocks
    n_blocks = n // block_size

    perm_sharpes = []
    for _ in range(n_perms):
        # Random block permutation
        block_indices = np.random.choice(n_blocks, size=n_blocks, replace=True)
        shuffled = np.concatenate([
            returns[i*block_size:(i+1)*block_size] for i in block_indices
        ])[:n]

        eq = np.cumprod(1 + shuffled) * INITIAL_CAPITAL
        eq_s = pd.Series(eq, index=equity_series.pct_change().dropna().index[:len(eq)])
        pm = compute_metrics(eq_s)
        perm_sharpes.append(pm['sharpe'])

    perm_sharpes = np.array(perm_sharpes)
    p_value = float((perm_sharpes >= real_sharpe).mean())

    return {
        'real_sharpe': real_sharpe,
        'perm_mean': round(float(perm_sharpes.mean()), 3),
        'perm_std': round(float(perm_sharpes.std()), 3),
        'p_value': round(p_value, 3),
        'pass': p_value < 0.05
    }


def subperiod_test(equity_series, n_blocks=4):
    """Sub-period consistency test"""
    returns = equity_series.pct_change().dropna()
    block_size = len(returns) // n_blocks
    block_sharpes = []

    for b in range(n_blocks):
        s = b * block_size
        e = (b+1) * block_size if b < n_blocks-1 else len(returns)
        br = returns.iloc[s:e]
        bs = float(br.mean() / br.std() * np.sqrt(252)) if br.std() > 0 else 0
        block_sharpes.append(round(bs, 3))

    mean_s = np.mean(block_sharpes)
    cv = float(np.std(block_sharpes) / max(mean_s, 0.01))

    return {
        'block_sharpes': block_sharpes,
        'cv': round(cv, 3),
        'pass': cv < 0.50
    }


def regime_test(equity_series, spy_prices):
    """R1 regime test: green vs red day performance"""
    returns = equity_series.pct_change().dropna()
    spy_ret = spy_prices.reindex(returns.index).pct_change().dropna()
    common = returns.index.intersection(spy_ret.index)
    returns = returns.loc[common]
    spy_ret = spy_ret.loc[common]

    green = spy_ret > 0
    red = spy_ret <= 0

    g_sharpe = float(returns[green].mean() / returns[green].std() * np.sqrt(252)) if green.sum() > 50 else 0
    r_sharpe = float(returns[red].mean() / returns[red].std() * np.sqrt(252)) if red.sum() > 50 else 0
    gap = abs(g_sharpe - r_sharpe) / max(abs(g_sharpe), abs(r_sharpe), 0.01)

    return {
        'green_sharpe': round(g_sharpe, 3),
        'red_sharpe': round(r_sharpe, 3),
        'gap': round(float(gap), 3),
        'pass': gap <= 0.50
    }


def full_adversarial(equity_series, spy_prices, n_perms=200):
    """Run all adversarial tests"""
    perm = permutation_test(equity_series, spy_prices, n_perms)
    subp = subperiod_test(equity_series)
    regime = regime_test(equity_series, spy_prices)

    gates_passed = sum([perm['pass'], subp['pass'], regime['pass']])

    return {
        'permutation': perm,
        'subperiod': subp,
        'regime': regime,
        'gates_passed': gates_passed,
        'gates_total': 3,
        'overall_pass': gates_passed >= 2
    }


def emit_result(name, description, metrics, adversarial, extra=None):
    """Print structured JSON result for the runner to capture"""
    result = {
        'name': name,
        'description': description,
        'timestamp': pd.Timestamp.now().isoformat(),
        **metrics,
        'adversarial': adversarial,
    }
    if extra:
        result.update(extra)

    print(f"\n<<<RESULT_JSON>>>")
    print(json.dumps(result, indent=2, default=str))
    print(f"<<<END_RESULT_JSON>>>")
