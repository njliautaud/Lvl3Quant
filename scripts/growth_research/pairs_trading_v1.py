"""
Pairs Trading Research v1 — Statistical Arbitrage on S&P 500 Top 100
HC #0: Sliding walk-forward ONLY (252d train, 63d test, 21d slide)
HC #428 R1: Regime-agnostic validation
HC #694: Zero commission (Robinhood)
2015-2026 backtest, MLflow logging
"""

import os
import sys
import time
import warnings
import itertools
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/pairs_trading_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# Walk-forward params (HC #0: sliding only)
TRAIN_DAYS    = 252   # 1 year training window
TEST_DAYS     = 63    # 3 months test
SLIDE_DAYS    = 21    # 1 month slide

# Signal thresholds
Z_ENTRY       = 2.0
Z_EXIT        = 0.0
Z_STOP        = 3.5
MAX_HOLD_DAYS = 60

# Cointegration significance
COINT_PVAL    = 0.05

# Walk-forward data range
DATA_START    = "2014-01-01"   # extra year for first training window
DATA_END      = "2026-07-01"
BACKTEST_START = "2015-01-01"

# Permutation test iterations
N_PERMS       = 100

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "pairs_trading_v1"

# ─────────────────────────────────────────────────────────────────────────────
# S&P 500 TOP 100 BY MARKET CAP  (static list for reproducibility)
# Sector mapping included
# ─────────────────────────────────────────────────────────────────────────────
UNIVERSE = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "GOOG": "Technology", "META": "Technology",
    "AVGO": "Technology", "ORCL": "Technology", "CRM": "Technology",
    "AMD": "Technology", "INTC": "Technology", "QCOM": "Technology",
    "TXN": "Technology", "AMAT": "Technology", "MU": "Technology",
    "NOW": "Technology", "LRCX": "Technology", "KLAC": "Technology",
    "ADI": "Technology", "MRVL": "Technology",
    # Communication Services
    "AMZN": "Communication", "NFLX": "Communication", "DIS": "Communication",
    "CMCSA": "Communication", "T": "Communication", "VZ": "Communication",
    "TMUS": "Communication", "CHTR": "Communication",
    # Financials
    "BRK-B": "Financials", "JPM": "Financials", "V": "Financials",
    "MA": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "BLK": "Financials",
    "SCHW": "Financials", "AXP": "Financials", "C": "Financials",
    "USB": "Financials", "PNC": "Financials", "COF": "Financials",
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare", "TMO": "Healthcare",
    "ABT": "Healthcare", "DHR": "Healthcare", "PFE": "Healthcare",
    "AMGN": "Healthcare", "ISRG": "Healthcare", "VRTX": "Healthcare",
    "REGN": "Healthcare", "BSX": "Healthcare", "MDT": "Healthcare",
    # Consumer Discretionary
    "TSLA": "Consumer_Disc", "HD": "Consumer_Disc", "MCD": "Consumer_Disc",
    "NKE": "Consumer_Disc", "LOW": "Consumer_Disc", "SBUX": "Consumer_Disc",
    "TJX": "Consumer_Disc", "BKNG": "Consumer_Disc", "GM": "Consumer_Disc",
    "F": "Consumer_Disc",
    # Consumer Staples
    "WMT": "Consumer_Staples", "PG": "Consumer_Staples", "KO": "Consumer_Staples",
    "PEP": "Consumer_Staples", "COST": "Consumer_Staples", "PM": "Consumer_Staples",
    "MO": "Consumer_Staples", "CL": "Consumer_Staples",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "EOG": "Energy", "SLB": "Energy", "OXY": "Energy",
    # Industrials
    "GE": "Industrials", "CAT": "Industrials", "HON": "Industrials",
    "UPS": "Industrials", "BA": "Industrials", "RTX": "Industrials",
    "LMT": "Industrials", "DE": "Industrials",
    # Materials / Utilities / REIT
    "LIN": "Materials", "APD": "Materials",
    "NEE": "Utilities", "DUK": "Utilities",
    "AMT": "REIT", "PLD": "REIT",
}

TICKERS = list(UNIVERSE.keys())

# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def download_prices(tickers, start, end):
    log.info(f"Downloading {len(tickers)} tickers {start}→{end} ...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(axis=1, how="all")
    # drop tickers with >5% missing
    thresh = int(len(prices) * 0.95)
    prices = prices.dropna(axis=1, thresh=thresh)
    prices = prices.ffill().bfill()
    log.info(f"Prices shape after cleaning: {prices.shape}")
    return prices

# ─────────────────────────────────────────────────────────────────────────────
# COINTEGRATION
# ─────────────────────────────────────────────────────────────────────────────
def find_cointegrated_pairs(prices, sector_map, pval_thresh=COINT_PVAL):
    """
    For all same-sector pairs compute Engle-Granger cointegration.
    Returns list of (t1, t2, pval, hedge_ratio).
    """
    tickers = list(prices.columns)
    pairs = []
    checked = 0
    for t1, t2 in itertools.combinations(tickers, 2):
        if sector_map.get(t1) != sector_map.get(t2):
            continue
        checked += 1
        try:
            s1 = np.log(prices[t1].values)
            s2 = np.log(prices[t2].values)
            score, pval, _ = coint(s1, s2)
            if pval < pval_thresh:
                # OLS hedge ratio
                from numpy.linalg import lstsq
                A = np.vstack([s2, np.ones(len(s2))]).T
                hr, _ = lstsq(A, s1, rcond=None)[:2]
                pairs.append((t1, t2, pval, hr[0]))
        except Exception:
            pass
    return pairs, checked

# ─────────────────────────────────────────────────────────────────────────────
# SPREAD Z-SCORE
# ─────────────────────────────────────────────────────────────────────────────
def compute_zscore(s1, s2, hedge_ratio):
    """Spread = log(s1) - hedge_ratio * log(s2)"""
    spread = np.log(s1) - hedge_ratio * np.log(s2)
    mu = spread.mean()
    sigma = spread.std()
    if sigma < 1e-10:
        return pd.Series(np.zeros(len(spread)), index=s1.index)
    return (spread - mu) / sigma

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ONE PAIR ON TEST WINDOW
# ─────────────────────────────────────────────────────────────────────────────
def backtest_pair(test_prices, t1, t2, hedge_ratio, train_spread_mu, train_spread_sigma):
    """
    Simulate trading the pair on test_prices using train window stats.
    Returns daily PnL series (in % return units, equal-weight legs).
    """
    s1 = test_prices[t1].values
    s2 = test_prices[t2].values
    dates = test_prices.index

    spread = np.log(s1) - hedge_ratio * np.log(s2)
    if train_spread_sigma < 1e-10:
        return pd.Series(0.0, index=dates)
    z = (spread - train_spread_mu) / train_spread_sigma

    position = 0   # +1 = long spread, -1 = short spread
    entry_day = None
    entry_s1 = entry_s2 = 0.0
    pnl = np.zeros(len(dates))

    for i in range(1, len(dates)):
        if position == 0:
            # Entry signals
            if z[i] > Z_ENTRY:
                position = -1   # short spread: short t1, long t2
                entry_day = i
                entry_s1, entry_s2 = s1[i], s2[i]
            elif z[i] < -Z_ENTRY:
                position = +1   # long spread: long t1, short t2
                entry_day = i
                entry_s1, entry_s2 = s1[i], s2[i]
        else:
            # Daily P&L while in position
            # long-spread: +ret_t1 - hedge * ret_t2 (normalized to unit dollar)
            ret_t1 = (s1[i] - s1[i-1]) / s1[i-1]
            ret_t2 = (s2[i] - s2[i-1]) / s2[i-1]
            # Dollar-neutral: $1 in t1, $hedge_ratio in t2 (normalized to 0.5+0.5 weight)
            w = 0.5
            if position == +1:
                pnl[i] = w * ret_t1 - w * ret_t2
            else:
                pnl[i] = -w * ret_t1 + w * ret_t2

            # Exit conditions
            exit_now = False
            if position == +1 and z[i] >= Z_EXIT:
                exit_now = True
            if position == -1 and z[i] <= Z_EXIT:
                exit_now = True
            if abs(z[i]) > Z_STOP:
                exit_now = True
            if (i - entry_day) >= MAX_HOLD_DAYS:
                exit_now = True

            if exit_now:
                position = 0
                entry_day = None

    return pd.Series(pnl, index=dates)

# ─────────────────────────────────────────────────────────────────────────────
# REGIME CLASSIFICATION (market up/down/flat day)
# ─────────────────────────────────────────────────────────────────────────────
def classify_regime(spy_returns):
    """Green/red/flat based on SPY daily return."""
    spy_returns = spy_returns.squeeze()  # ensure 1D
    regimes = pd.Series("flat", index=spy_returns.index, dtype=str)
    regimes = regimes.where(~(spy_returns > 0.003), "green")
    regimes = regimes.where(~(spy_returns < -0.003), "red")
    return regimes

# ─────────────────────────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def calc_metrics(returns, label=""):
    r = np.array(returns)
    if len(r) == 0 or r.std() == 0:
        return {}
    ann = 252
    sharpe    = (r.mean() / r.std()) * np.sqrt(ann)
    down_std  = r[r < 0].std() if (r < 0).any() else 1e-10
    sortino   = (r.mean() / down_std) * np.sqrt(ann)
    total_ret = (1 + r).prod() - 1
    cum       = (1 + r).cumprod()
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    max_dd    = drawdowns.min()
    wins      = (r > 0).sum()
    wr        = wins / len(r)
    gross_profit = r[r > 0].sum() if (r > 0).any() else 0
    gross_loss   = abs(r[r < 0].sum()) if (r < 0).any() else 1e-10
    pf        = gross_profit / gross_loss
    return dict(
        label=label, sharpe=sharpe, sortino=sortino,
        total_ret=total_ret, max_dd=max_dd,
        wr=wr, pf=pf, n_days=len(r),
    )

def regime_gap(returns_series, regimes, label=""):
    """Check HC #428 R1: regime-stratified Sharpe gap."""
    green_r = returns_series[regimes == "green"].values
    red_r   = returns_series[regimes == "red"].values
    flat_r  = returns_series[regimes == "flat"].values

    def ann_sharpe(r):
        if len(r) < 5 or r.std() < 1e-10:
            return 0.0
        return (r.mean() / r.std()) * np.sqrt(252)

    sg = ann_sharpe(green_r)
    sr = ann_sharpe(red_r)
    sf = ann_sharpe(flat_r)
    denom = max(abs(sg), abs(sr), 1e-10)
    gap = abs(sg - sr) / denom
    log.info(f"[Regime {label}] Sharpe_green={sg:.2f} Sharpe_red={sr:.2f} Sharpe_flat={sf:.2f} gap={gap:.2f}")
    return dict(sharpe_green=sg, sharpe_red=sr, sharpe_flat=sf, regime_gap=gap)

# ─────────────────────────────────────────────────────────────────────────────
# PERMUTATION TEST
# ─────────────────────────────────────────────────────────────────────────────
def permutation_test(all_pair_returns_df, n_perms=N_PERMS):
    """
    Shuffle which pairs are active on each test window.
    Compute Sharpe distribution of shuffled portfolios.
    Returns p-value = P(permuted Sharpe >= observed Sharpe).
    """
    if all_pair_returns_df.empty:
        return 1.0, []

    port = all_pair_returns_df.mean(axis=1)
    obs_sharpe = calc_metrics(port.values)["sharpe"]

    perm_sharpes = []
    cols = list(all_pair_returns_df.columns)
    rng = np.random.default_rng(42)
    for _ in range(n_perms):
        shuffled_cols = rng.permutation(cols)
        # reassign columns (shuffle pair identity, keep timing same)
        shuf_df = all_pair_returns_df.copy()
        shuf_df.columns = shuffled_cols
        shuf_port = shuf_df.mean(axis=1)
        s = calc_metrics(shuf_port.values)
        perm_sharpes.append(s.get("sharpe", 0.0))

    pval = np.mean(np.array(perm_sharpes) >= obs_sharpe)
    log.info(f"Permutation test: obs_sharpe={obs_sharpe:.3f} | perm_mean={np.mean(perm_sharpes):.3f} | p-value={pval:.3f}")
    return pval, perm_sharpes

# ─────────────────────────────────────────────────────────────────────────────
# MAIN WALK-FORWARD LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_walkforward(prices):
    dates = prices.index
    backtest_start_idx = dates.searchsorted(pd.Timestamp(BACKTEST_START))

    all_portfolio_returns = []
    all_pair_returns_dict = {}  # pair_key -> pd.Series of daily returns
    wf_results = []
    total_pairs_found = 0
    window_num = 0

    train_start_idx = backtest_start_idx - TRAIN_DAYS
    if train_start_idx < 0:
        train_start_idx = 0

    while True:
        train_end_idx  = train_start_idx + TRAIN_DAYS
        test_start_idx = train_end_idx
        test_end_idx   = test_start_idx + TEST_DAYS

        if test_end_idx > len(dates):
            break

        train_dates = dates[train_start_idx:train_end_idx]
        test_dates  = dates[test_start_idx:test_end_idx]
        window_num += 1

        log.info(f"\n── Window {window_num}: train {train_dates[0].date()}→{train_dates[-1].date()} "
                 f"| test {test_dates[0].date()}→{test_dates[-1].date()}")

        train_prices = prices.loc[train_dates]
        test_prices  = prices.loc[test_dates]

        # Build sector map for available tickers
        avail = list(prices.columns)
        sector_map = {t: UNIVERSE.get(t, "Unknown") for t in avail}

        # Find cointegrated pairs in training window
        pairs, n_checked = find_cointegrated_pairs(train_prices, sector_map)
        log.info(f"  Checked {n_checked} same-sector pairs → {len(pairs)} cointegrated (p<{COINT_PVAL})")
        total_pairs_found += len(pairs)

        if len(pairs) == 0:
            train_start_idx += SLIDE_DAYS
            continue

        # Backtest each pair on test window
        window_pair_returns = []
        for t1, t2, pval, hr in pairs:
            if t1 not in test_prices.columns or t2 not in test_prices.columns:
                continue

            # Compute spread stats from training window
            train_spread = np.log(train_prices[t1]) - hr * np.log(train_prices[t2])
            mu_train = train_spread.mean()
            sigma_train = train_spread.std()

            pair_ret = backtest_pair(test_prices, t1, t2, hr, mu_train, sigma_train)
            window_pair_returns.append(pair_ret)

            pair_key = f"{t1}_{t2}_w{window_num}"
            if pair_key not in all_pair_returns_dict:
                all_pair_returns_dict[pair_key] = pair_ret
            else:
                all_pair_returns_dict[pair_key] = pd.concat([all_pair_returns_dict[pair_key], pair_ret])

        if window_pair_returns:
            # Equal-weight portfolio across all active pairs in window
            window_df = pd.concat(window_pair_returns, axis=1)
            window_portfolio = window_df.mean(axis=1)
            all_portfolio_returns.append(window_portfolio)

            w_metrics = calc_metrics(window_portfolio.values, label=f"W{window_num}")
            wf_results.append({
                "window": window_num,
                "train_start": train_dates[0].date(),
                "train_end": train_dates[-1].date(),
                "test_start": test_dates[0].date(),
                "test_end": test_dates[-1].date(),
                "n_pairs": len(pairs),
                **{k: v for k, v in w_metrics.items() if k != "label"},
            })
            log.info(f"  Window Sharpe={w_metrics['sharpe']:.2f} WR={w_metrics['wr']:.1%} PF={w_metrics['pf']:.2f}")

        train_start_idx += SLIDE_DAYS

    return all_portfolio_returns, all_pair_returns_dict, wf_results, total_pairs_found

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("PAIRS TRADING V1 — Statistical Arbitrage Research")
    log.info("=" * 70)

    # MLflow setup
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Download data
    prices = download_prices(TICKERS, DATA_START, DATA_END)
    spy_raw = yf.download("SPY", start=DATA_START, end=DATA_END, auto_adjust=True, progress=False)
    # Flatten multi-level columns if present
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = spy_raw.columns.get_level_values(0)
    spy_prices = spy_raw["Close"]
    if isinstance(spy_prices, pd.DataFrame):
        spy_prices = spy_prices.iloc[:, 0]
    spy_returns = spy_prices.pct_change().dropna()

    with mlflow.start_run(run_name=f"pairs_v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        mlflow.log_params({
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "slide_days": SLIDE_DAYS,
            "z_entry": Z_ENTRY,
            "z_exit": Z_EXIT,
            "z_stop": Z_STOP,
            "max_hold_days": MAX_HOLD_DAYS,
            "coint_pval": COINT_PVAL,
            "n_tickers": len(prices.columns),
            "data_start": DATA_START,
            "data_end": DATA_END,
            "commission": 0.0,
        })

        # Walk-forward
        all_port_returns, all_pair_returns, wf_results, total_pairs = run_walkforward(prices)

        if not all_port_returns:
            log.error("No portfolio returns generated — check data/pair finding.")
            return

        # Concatenate all test period returns
        full_returns = pd.concat(all_port_returns).sort_index()

        # Overall metrics
        overall = calc_metrics(full_returns.values, label="OVERALL")
        log.info("\n" + "=" * 50)
        log.info("OVERALL BACKTEST RESULTS")
        log.info("=" * 50)
        for k, v in overall.items():
            if isinstance(v, float):
                log.info(f"  {k:15s}: {v:.4f}")

        # Regime analysis (HC #428 R1)
        aligned_regimes = spy_returns.reindex(full_returns.index).fillna(0)
        regimes = classify_regime(aligned_regimes)
        reg_stats = regime_gap(full_returns, regimes, label="OVERALL")
        log.info(f"\nRegime gap (reject if >0.50): {reg_stats['regime_gap']:.3f}")
        regime_pass = reg_stats["regime_gap"] <= 0.50
        log.info(f"HC #428 R1 regime test: {'PASS' if regime_pass else 'FAIL (gap too large)'}")

        # Permutation test
        log.info("\nRunning permutation test ...")
        # Build a common-index DataFrame of all pair returns
        all_series = list(all_pair_returns.values())
        if len(all_series) > 1:
            pair_df = pd.concat(all_series, axis=1).fillna(0)
            perm_pval, perm_dist = permutation_test(pair_df, N_PERMS)
        else:
            perm_pval = 1.0
            perm_dist = []

        # Per-window summary
        wf_df = pd.DataFrame(wf_results)
        wf_csv = OUTPUT_DIR / "walkforward_results.csv"
        wf_df.to_csv(wf_csv, index=False)
        log.info(f"\nWalk-forward results saved to {wf_csv}")

        # Full returns series
        ret_csv = OUTPUT_DIR / "portfolio_returns.csv"
        full_returns.to_csv(ret_csv)

        # Log to MLflow
        mlflow.log_metrics({
            "sharpe":          overall["sharpe"],
            "sortino":         overall["sortino"],
            "total_return":    overall["total_ret"],
            "max_drawdown":    overall["max_dd"],
            "win_rate":        overall["wr"],
            "profit_factor":   overall["pf"],
            "n_days_tested":   overall["n_days"],
            "total_pairs_found": float(total_pairs),
            "n_wf_windows":    float(len(wf_results)),
            "regime_gap":      reg_stats["regime_gap"],
            "sharpe_green":    reg_stats["sharpe_green"],
            "sharpe_red":      reg_stats["sharpe_red"],
            "perm_pval":       perm_pval,
        })
        mlflow.log_artifact(str(wf_csv))
        mlflow.log_artifact(str(ret_csv))
        mlflow.log_artifact(str(LOG_FILE))

        # Final report
        elapsed = (time.time() - t0) / 60
        log.info("\n" + "=" * 70)
        log.info("FINAL SUMMARY")
        log.info("=" * 70)
        log.info(f"  Walk-forward windows:    {len(wf_results)}")
        log.info(f"  Total pairs found:       {total_pairs}")
        log.info(f"  Days in backtest:        {overall['n_days']}")
        log.info(f"  Sharpe:                  {overall['sharpe']:.3f}")
        log.info(f"  Sortino:                 {overall['sortino']:.3f}")
        log.info(f"  Win Rate:                {overall['wr']:.1%}")
        log.info(f"  Profit Factor:           {overall['pf']:.3f}")
        log.info(f"  Max Drawdown:            {overall['max_dd']:.1%}")
        log.info(f"  Total Return:            {overall['total_ret']:.1%}")
        log.info(f"  Regime gap:              {reg_stats['regime_gap']:.3f} ({'PASS' if regime_pass else 'FAIL'})")
        log.info(f"  Permutation p-value:     {perm_pval:.3f}")
        log.info(f"  Elapsed:                 {elapsed:.1f} min")
        log.info("=" * 70)

        # Window-level distribution
        if not wf_df.empty:
            log.info(f"\nPer-window Sharpe distribution:")
            log.info(f"  mean={wf_df['sharpe'].mean():.2f}  median={wf_df['sharpe'].median():.2f}  "
                     f"std={wf_df['sharpe'].std():.2f}  min={wf_df['sharpe'].min():.2f}  max={wf_df['sharpe'].max():.2f}")

    log.info("Done. Results in /home/nick/Lvl3Quant/output/pairs_trading_v1/")


if __name__ == "__main__":
    main()
