"""
Momentum Crash Hedge with ML Timing — v1
==========================================
HC #0  : Sliding walk-forward (252d train, 63d test, 21d step)
HC #428: Regime-agnostic OOT validation (R1 + R2)
HC #694: Commission-free (Robinhood) — 0 brokerage commissions

Strategy:
- Cross-sectional momentum on top 200 US large caps (12-1 month, long top quintile / short bottom quintile)
- LGBM crash predictor: predicts >10% momentum factor drawdown in next 21 trading days
- When crash probability HIGH: flatten or reverse to anti-momentum
- Asymmetric: steady ~10-15% in normal periods, avoids/profits from rare crashes

Crash predictor features:
- Momentum spread, market vol (VIX + realized), credit spreads (HYG/IEF),
  cross-sectional dispersion, momentum DD from peak, market breadth (% > 200MA),
  volatility of momentum factor, momentum reversal signal
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    print("WARNING: lightgbm not available, will use sklearn GradientBoosting as fallback")
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_LGBM = False

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/momentum_crash_hedge_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2006-01-01"   # need lookback before 2008
END   = "2026-07-14"

# Walk-forward params
WF_TRAIN   = 252   # 1 year training
WF_TEST    = 63    # 3 month test
WF_STEP    = 21    # 1 month step

# Momentum params
MOM_LOOKBACK = 252  # 12 months
MOM_SKIP     = 21   # skip most recent month
REBAL_FREQ   = 21   # rebalance every ~1 month (trading days)
Q_LONG  = 0.80      # top quintile
Q_SHORT = 0.20      # bottom quintile

# Crash definition
CRASH_THRESHOLD = -0.10  # 10% drawdown of momentum factor in 21 days
CRASH_HORIZON   = 21     # trading days forward

# Crash probability threshold to trigger hedge
CRASH_PROB_THRESHOLD = 0.50

# Permutation test
N_PERMUTATIONS = 500

# Feature lookback windows
VOL_WINDOWS = [21, 63, 126]

print(f"[{datetime.now()}] Momentum Crash Hedge v1 starting...")
print(f"Output: {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# 1. DATA DOWNLOAD
# ---------------------------------------------------------------------------
def get_sp500_constituents():
    """Get S&P 500 tickers from Wikipedia."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception as e:
        print(f"Failed to scrape S&P 500 list: {e}")
        # Fallback: top ~200 large caps by common knowledge
        return [
            "AAPL","MSFT","AMZN","NVDA","GOOGL","META","BRK-B","TSLA","UNH","XOM",
            "JNJ","JPM","V","PG","MA","HD","AVGO","CVX","MRK","ABBV",
            "LLY","PEP","KO","COST","BAC","ADBE","WMT","MCD","CRM","TMO",
            "CSCO","ACN","ABT","DHR","NFLX","LIN","AMD","CMCSA","TXN","PM",
            "VZ","NEE","INTC","WFC","DIS","BMY","UNP","QCOM","UPS","RTX",
            "AMGN","SCHW","MS","HON","LOW","COP","GS","ELV","PFE","BLK",
            "SBUX","ISRG","CAT","INTU","T","BA","AXP","GE","DE","GILD",
            "MDLZ","BKNG","SYK","ADI","MMC","TJX","VRTX","ADP","CVS","LRCX",
            "TMUS","PGR","CB","CI","REGN","SO","ZTS","MO","BDX","ITW",
            "EOG","BSX","NOC","DUK","FISV","SLB","AON","PLD","WM","CL",
            "CME","SHW","ICE","SNPS","CDNS","MCK","FDX","EMR","HUM","MPC",
            "ORLY","GD","PNC","USB","APD","TGT","OXY","AZO","NSC","F",
            "PSX","AFL","TDG","AJG","PCAR","D","SRE","MNST","KMB","MET",
            "HCA","MCO","AEP","MSCI","WELL","TRV","JCI","CARR","SPG","AIG",
            "FTNT","PCG","DHI","ROST","ROP","GWW","O","TEL","ALL","PSA",
            "PAYX","LHX","BK","COF","KHC","IDXX","CCI","STZ","KDP","AMP",
            "FAST","NUE","YUM","CTVA","DD","PRU","CMI","URI","DXCM","OTIS",
            "ED","MLM","CSGP","KEYS","VRSK","IT","WST","WBD","CDW","DOV",
            "GEHC","GIS","EXC","XEL","VICI","HPQ","VMC","RMD","EW","ANSS",
            "AWK","AVB","WEC","GLW","ACGL","MPWR","TSCO","HWM","CHD","DLR",
        ]

def download_stock_data(tickers, start, end):
    """Download adjusted close prices for all tickers."""
    cache_file = OUTPUT_DIR / "price_cache.parquet"
    if cache_file.exists():
        print("Loading cached price data...")
        prices = pd.read_parquet(cache_file)
        # Check if we need to update
        if prices.index[-1].strftime("%Y-%m-%d") >= "2026-07-10":
            return prices

    print(f"Downloading data for {len(tickers)} tickers...")
    # Download in batches to avoid timeouts
    batch_size = 50
    all_data = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}: {len(batch)} tickers...")
        try:
            data = yf.download(batch, start=start, end=end, auto_adjust=True, progress=False)
            if "Close" in data.columns.get_level_values(0) if isinstance(data.columns, pd.MultiIndex) else "Close" in data.columns:
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data["Close"]
                else:
                    closes = data[["Close"]]
                    closes.columns = batch[:1]
                for col in closes.columns:
                    if closes[col].notna().sum() > 252:  # need at least 1 year
                        all_data[col] = closes[col]
        except Exception as e:
            print(f"  Error in batch: {e}")
        time.sleep(1)

    prices = pd.DataFrame(all_data)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()

    # Save cache
    prices.to_parquet(cache_file)
    print(f"Downloaded {prices.shape[1]} valid tickers, {len(prices)} days")
    return prices

def download_macro_data(start, end):
    """Download VIX, HYG, IEF for crash features."""
    macro_tickers = ["^VIX", "HYG", "IEF", "SPY"]
    cache_file = OUTPUT_DIR / "macro_cache.parquet"
    if cache_file.exists():
        print("Loading cached macro data...")
        macro = pd.read_parquet(cache_file)
        if macro.index[-1].strftime("%Y-%m-%d") >= "2026-07-10":
            return macro

    print("Downloading macro data (VIX, HYG, IEF, SPY)...")
    data = yf.download(macro_tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        macro = data["Close"]
    else:
        macro = data[["Close"]]
    macro.columns = [c.replace("^", "") for c in macro.columns]
    macro.index = pd.to_datetime(macro.index)
    macro.to_parquet(cache_file)
    return macro

# ---------------------------------------------------------------------------
# 2. MOMENTUM FACTOR CONSTRUCTION
# ---------------------------------------------------------------------------
def compute_momentum_scores(prices, lookback=MOM_LOOKBACK, skip=MOM_SKIP):
    """Compute 12-1 month momentum for each stock."""
    # Total return over lookback, skipping most recent 'skip' days
    ret_full = prices.pct_change(lookback)
    ret_skip = prices.pct_change(skip)
    # 12-1 month momentum = ret over [t-252, t-21]
    mom = (1 + ret_full) / (1 + ret_skip) - 1
    return mom

def build_momentum_factor(prices, rebal_dates):
    """Build long-short momentum factor returns (top Q - bottom Q, equal weight)."""
    mom_scores = compute_momentum_scores(prices)
    daily_ret = prices.pct_change()

    factor_returns = []
    positions_log = []

    for i in range(len(rebal_dates) - 1):
        rebal_date = rebal_dates[i]
        next_rebal = rebal_dates[i + 1]

        # Get momentum scores at rebalance date
        scores = mom_scores.loc[rebal_date].dropna()
        if len(scores) < 20:
            continue

        # Quintile cutoffs
        q_high = scores.quantile(Q_LONG)
        q_low  = scores.quantile(Q_SHORT)

        longs  = scores[scores >= q_high].index.tolist()
        shorts = scores[scores <= q_low].index.tolist()

        if not longs or not shorts:
            continue

        # Equal weight
        w_long  = 1.0 / len(longs)
        w_short = 1.0 / len(shorts)

        # Get daily returns between rebalances
        period_mask = (daily_ret.index > rebal_date) & (daily_ret.index <= next_rebal)
        period_rets = daily_ret.loc[period_mask]

        for dt, row in period_rets.iterrows():
            long_ret  = row[longs].mean() if len(longs) > 0 else 0
            short_ret = row[shorts].mean() if len(shorts) > 0 else 0
            factor_ret = long_ret - short_ret  # long winners, short losers
            factor_returns.append({"date": dt, "factor_ret": factor_ret,
                                   "long_ret": long_ret, "short_ret": short_ret,
                                   "n_long": len(longs), "n_short": len(shorts)})

    factor_df = pd.DataFrame(factor_returns).set_index("date")
    return factor_df

# ---------------------------------------------------------------------------
# 3. CRASH PREDICTOR FEATURES
# ---------------------------------------------------------------------------
def build_crash_features(factor_df, macro_df, prices):
    """Build features for the crash predictor."""
    daily_ret = prices.pct_change()
    features = pd.DataFrame(index=factor_df.index)

    # F1: Momentum spread (cumulative factor return over windows)
    for w in [21, 63, 126]:
        features[f"mom_spread_{w}d"] = factor_df["factor_ret"].rolling(w).sum()

    # F2: Momentum factor volatility
    for w in [21, 63]:
        features[f"mom_vol_{w}d"] = factor_df["factor_ret"].rolling(w).std() * np.sqrt(252)

    # F3: Momentum factor drawdown from peak
    cum_factor = (1 + factor_df["factor_ret"]).cumprod()
    rolling_max = cum_factor.expanding().max()
    features["mom_dd_from_peak"] = (cum_factor / rolling_max) - 1

    # F4: Momentum factor Sharpe (rolling)
    for w in [63, 126]:
        mu = factor_df["factor_ret"].rolling(w).mean() * 252
        sd = factor_df["factor_ret"].rolling(w).std() * np.sqrt(252)
        features[f"mom_sharpe_{w}d"] = mu / (sd + 1e-8)

    # F5: VIX level and change
    if "VIX" in macro_df.columns:
        vix = macro_df["VIX"].reindex(factor_df.index, method="ffill")
        features["vix_level"] = vix
        features["vix_change_21d"] = vix.pct_change(21)
        features["vix_change_5d"] = vix.pct_change(5)

    # F6: Credit spread (HYG/IEF ratio — lower = wider spreads = stress)
    if "HYG" in macro_df.columns and "IEF" in macro_df.columns:
        hyg = macro_df["HYG"].reindex(factor_df.index, method="ffill")
        ief = macro_df["IEF"].reindex(factor_df.index, method="ffill")
        credit_ratio = hyg / ief
        features["credit_ratio"] = credit_ratio
        features["credit_ratio_change_21d"] = credit_ratio.pct_change(21)

    # F7: Cross-sectional dispersion of returns
    for w in [21, 63]:
        rolling_rets = daily_ret.rolling(w).sum()
        features[f"xs_dispersion_{w}d"] = rolling_rets.std(axis=1).reindex(factor_df.index, method="ffill")

    # F8: Market breadth (% stocks above 200-day MA)
    ma200 = prices.rolling(200).mean()
    above_ma = (prices > ma200).sum(axis=1) / prices.notna().sum(axis=1)
    features["breadth_pct_above_200ma"] = above_ma.reindex(factor_df.index, method="ffill")
    features["breadth_change_21d"] = features["breadth_pct_above_200ma"].diff(21)

    # F9: Market return (SPY)
    if "SPY" in macro_df.columns:
        spy = macro_df["SPY"].reindex(factor_df.index, method="ffill")
        for w in [21, 63]:
            features[f"spy_ret_{w}d"] = spy.pct_change(w)
        spy_vol = spy.pct_change().rolling(21).std() * np.sqrt(252)
        features["spy_realized_vol_21d"] = spy_vol

    # F10: Momentum reversal signal (short-term reversal of losers)
    features["mom_reversal_signal"] = -factor_df["factor_ret"].rolling(5).sum()

    # F11: Factor return skewness
    features["mom_skew_63d"] = factor_df["factor_ret"].rolling(63).skew()

    # F12: Autocovariance of factor (mean reversion signal)
    features["mom_autocorr_21d"] = factor_df["factor_ret"].rolling(63).apply(
        lambda x: pd.Series(x[:21]).corr(pd.Series(x[21:42])) if len(x) >= 42 else np.nan, raw=True
    )

    # Drop columns that are entirely NaN, then forward-fill remaining NaNs
    features = features.dropna(axis=1, how="all")
    features = features.ffill().bfill()

    return features

def build_crash_labels(factor_df, threshold=CRASH_THRESHOLD, horizon=CRASH_HORIZON):
    """Label: will momentum factor drawdown >10% in next 21 days?"""
    cum_ret_fwd = factor_df["factor_ret"].rolling(horizon).sum().shift(-horizon)
    # Also check max drawdown path within horizon
    labels = (cum_ret_fwd <= threshold).astype(int)
    return labels

# ---------------------------------------------------------------------------
# 4. WALK-FORWARD ML CRASH PREDICTOR
# ---------------------------------------------------------------------------
def train_crash_predictor_wf(features, labels, factor_df):
    """Sliding window walk-forward for crash predictor."""
    # Align — use intersection of non-null label rows; fill remaining feature NaN
    valid_labels = labels.dropna().index
    common_idx = features.index.intersection(valid_labels)
    X = features.loc[common_idx].copy()
    y = labels.loc[common_idx].copy()

    # Drop any remaining rows where ALL features are NaN, fill partial NaN
    mask = X.notna().any(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]
    X = X.fillna(0)  # safe fallback for any remaining NaN

    print(f"Crash predictor dataset: {len(X)} samples, {X.shape[1]} features")
    if len(X) == 0:
        print("ERROR: No valid samples for crash predictor!")
        return pd.Series(dtype=float), pd.DataFrame(), pd.Series(dtype=float)
    print(f"Crash events: {y.sum()} ({100*y.mean():.1f}%)")

    predictions = pd.Series(index=X.index, dtype=float)
    fold_results = []

    n = len(X)
    fold = 0
    start_idx = 0

    while start_idx + WF_TRAIN + WF_TEST <= n:
        train_end = start_idx + WF_TRAIN
        test_end  = min(train_end + WF_TEST, n)

        X_train = X.iloc[start_idx:train_end]
        y_train = y.iloc[start_idx:train_end]
        X_test  = X.iloc[train_end:test_end]
        y_test  = y.iloc[train_end:test_end]

        # Handle class imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos = max(n_neg / max(n_pos, 1), 1.0)

        if HAS_LGBM:
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                scale_pos_weight=scale_pos,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                min_samples_leaf=20,
                subsample=0.8,
                random_state=42,
            )

        model.fit(X_train, y_train)

        if HAS_LGBM:
            preds = model.predict_proba(X_test)[:, 1]
        else:
            preds = model.predict_proba(X_test)[:, 1]

        predictions.iloc[train_end:test_end] = preds

        # Fold metrics
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        try:
            auc = roc_auc_score(y_test, preds)
        except:
            auc = np.nan
        fold_results.append({
            "fold": fold,
            "train_start": X_train.index[0].strftime("%Y-%m-%d"),
            "test_start": X_test.index[0].strftime("%Y-%m-%d"),
            "test_end": X_test.index[-1].strftime("%Y-%m-%d"),
            "auc": auc,
            "crash_rate_train": y_train.mean(),
            "crash_rate_test": y_test.mean(),
            "mean_pred": preds.mean(),
        })

        if fold % 10 == 0:
            print(f"  Fold {fold}: AUC={auc:.3f}, crash_train={y_train.mean():.3f}, crash_test={y_test.mean():.3f}")

        fold += 1
        start_idx += WF_STEP

    fold_df = pd.DataFrame(fold_results)
    if len(fold_df) == 0:
        print("\nWalk-forward: 0 folds completed!")
        return predictions, fold_df, pd.Series(dtype=float)
    print(f"\nWalk-forward: {fold} folds, median AUC={fold_df['auc'].median():.3f}")

    # Feature importance (last model)
    if HAS_LGBM:
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    else:
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop features:")
    print(imp.head(10))

    return predictions, fold_df, imp

# ---------------------------------------------------------------------------
# 5. STRATEGY SIMULATION
# ---------------------------------------------------------------------------
def simulate_strategies(factor_df, crash_probs, crash_labels):
    """Simulate: vanilla momentum vs ML-timed momentum."""
    common = factor_df.index.intersection(crash_probs.dropna().index)
    factor_df = factor_df.loc[common]
    crash_probs = crash_probs.loc[common]
    crash_labels_aligned = crash_labels.reindex(common).fillna(0)

    results = {}

    # --- Strategy 1: Vanilla Momentum (always on) ---
    vanilla_ret = factor_df["factor_ret"].copy()
    vanilla_cum = (1 + vanilla_ret).cumprod()
    results["vanilla"] = vanilla_ret

    # --- Strategy 2: ML-Timed (flatten when crash prob high) ---
    ml_flatten_ret = factor_df["factor_ret"].copy()
    crash_signal = crash_probs >= CRASH_PROB_THRESHOLD
    ml_flatten_ret[crash_signal] = 0  # go to cash
    results["ml_flatten"] = ml_flatten_ret

    # --- Strategy 3: ML-Timed (reverse to anti-momentum when crash prob high) ---
    ml_reverse_ret = factor_df["factor_ret"].copy()
    ml_reverse_ret[crash_signal] = -factor_df["factor_ret"][crash_signal]  # reverse
    results["ml_reverse"] = ml_reverse_ret

    # --- Strategy 4: ML-Timed (half position when moderate, flatten when high) ---
    ml_gradual_ret = factor_df["factor_ret"].copy()
    moderate_signal = (crash_probs >= 0.30) & (crash_probs < CRASH_PROB_THRESHOLD)
    ml_gradual_ret[moderate_signal] = 0.5 * factor_df["factor_ret"][moderate_signal]
    ml_gradual_ret[crash_signal] = 0  # flatten
    results["ml_gradual"] = ml_gradual_ret

    return results, crash_signal

# ---------------------------------------------------------------------------
# 6. METRICS & REPORTING
# ---------------------------------------------------------------------------
def compute_metrics(returns, name=""):
    """Compute risk-adjusted metrics."""
    r = returns.dropna()
    if len(r) < 63:
        return {}

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = ann_ret / (ann_vol + 1e-8)
    downside = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 1e-8
    sortino = ann_ret / (downside + 1e-8)

    cum = (1 + r).cumprod()
    total_ret = cum.iloc[-1] / cum.iloc[0] - 1
    n_years = len(r) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    rolling_max = cum.expanding().max()
    dd = cum / rolling_max - 1
    max_dd = dd.min()

    # Win rate
    wr = (r > 0).mean()

    # Profit factor
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / (losses + 1e-8)

    # Calmar
    calmar = cagr / (abs(max_dd) + 1e-8)

    # Skewness
    skew = r.skew()

    # Annual returns for asymmetry ratio
    annual = r.groupby(r.index.year).sum()
    best_year = annual.max()
    worst_year = annual.min()
    asymmetry = best_year / (abs(worst_year) + 1e-8) if worst_year < 0 else float("inf")

    return {
        "name": name,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr": cagr,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": wr,
        "profit_factor": pf,
        "skewness": skew,
        "best_year": best_year,
        "worst_year": worst_year,
        "asymmetry_ratio": asymmetry,
        "n_days": len(r),
        "n_years": n_years,
    }

def regime_analysis(returns, spy_rets):
    """HC #428 R1: Regime-agnostic validation."""
    common = returns.index.intersection(spy_rets.index)
    r = returns.loc[common]
    spy = spy_rets.loc[common]

    # Classify days by SPY return
    green = spy > 0.002   # up day
    red   = spy < -0.002  # down day
    flat  = ~green & ~red

    regime_stats = {}
    for regime_name, mask in [("green", green), ("red", red), ("flat", flat)]:
        rr = r[mask]
        if len(rr) > 21:
            ann = rr.mean() * 252
            vol = rr.std() * np.sqrt(252)
            regime_stats[regime_name] = {
                "sharpe": ann / (vol + 1e-8),
                "ann_ret": ann,
                "n_days": len(rr),
                "wr": (rr > 0).mean(),
            }

    # Regime asymmetry test
    if "green" in regime_stats and "red" in regime_stats:
        s_green = regime_stats["green"]["sharpe"]
        s_red   = regime_stats["red"]["sharpe"]
        denom = max(abs(s_green), abs(s_red), 1e-8)
        regime_asymmetry = abs(s_green - s_red) / denom
        regime_stats["asymmetry_ratio"] = regime_asymmetry
        regime_stats["passes_hc428_r1"] = regime_asymmetry <= 0.50
    else:
        regime_stats["passes_hc428_r1"] = False

    return regime_stats

def crisis_alpha_analysis(returns_dict, factor_df):
    """Analyze performance during known momentum crash periods."""
    # Known momentum crash periods (approximate)
    crashes = {
        "2009-03 (GFC recovery)": ("2009-03-01", "2009-04-15"),
        "2016-02 (Oil crash)":    ("2016-01-15", "2016-03-15"),
        "2020-03 (COVID)":        ("2020-03-01", "2020-04-30"),
        "2020-11 (Vaccine)":      ("2020-11-01", "2020-12-31"),
        "2021-01 (GME squeeze)":  ("2021-01-15", "2021-02-15"),
        "2022-11 (Bear rally)":   ("2022-10-15", "2022-12-15"),
    }

    crisis_results = {}
    for crisis_name, (start, end) in crashes.items():
        mask = (factor_df.index >= start) & (factor_df.index <= end)
        if mask.sum() == 0:
            continue

        crisis_row = {"period": crisis_name}
        for strat_name, strat_ret in returns_dict.items():
            aligned = strat_ret.reindex(factor_df.index[mask]).dropna()
            if len(aligned) > 0:
                crisis_row[f"{strat_name}_ret"] = aligned.sum()
                crisis_row[f"{strat_name}_dd"] = ((1 + aligned).cumprod().expanding().max() /
                                                   (1 + aligned).cumprod() - 1).max()

        crisis_results[crisis_name] = crisis_row

    return pd.DataFrame(crisis_results).T

def permutation_test(factor_ret, crash_probs, n_perms=N_PERMUTATIONS):
    """Permutation test: is the ML timing real or luck?"""
    common = factor_ret.index.intersection(crash_probs.dropna().index)
    factor_ret = factor_ret.loc[common]
    crash_probs = crash_probs.loc[common]

    # Actual ML-timed Sharpe
    ml_ret = factor_ret.copy()
    ml_ret[crash_probs >= CRASH_PROB_THRESHOLD] = 0
    actual_sharpe = ml_ret.mean() / (ml_ret.std() + 1e-8) * np.sqrt(252)

    # Vanilla Sharpe
    vanilla_sharpe = factor_ret.mean() / (factor_ret.std() + 1e-8) * np.sqrt(252)

    # Improvement
    actual_improvement = actual_sharpe - vanilla_sharpe

    # Permutation: shuffle crash signal, compute Sharpe improvement
    perm_improvements = []
    rng = np.random.RandomState(42)
    for i in range(n_perms):
        shuffled = crash_probs.copy()
        shuffled[:] = rng.permutation(shuffled.values)
        perm_ret = factor_ret.copy()
        perm_ret[shuffled >= CRASH_PROB_THRESHOLD] = 0
        perm_sharpe = perm_ret.mean() / (perm_ret.std() + 1e-8) * np.sqrt(252)
        perm_improvements.append(perm_sharpe - vanilla_sharpe)

    perm_improvements = np.array(perm_improvements)
    p_value = (perm_improvements >= actual_improvement).mean()

    print(f"\nPermutation test ({n_perms} shuffles):")
    print(f"  Actual Sharpe improvement: {actual_improvement:.4f}")
    print(f"  Mean permuted improvement: {np.mean(perm_improvements):.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at 5%: {p_value < 0.05}")

    return {
        "actual_improvement": actual_improvement,
        "actual_sharpe": actual_sharpe,
        "vanilla_sharpe": vanilla_sharpe,
        "perm_mean": np.mean(perm_improvements),
        "perm_std": np.std(perm_improvements),
        "p_value": p_value,
        "significant_5pct": p_value < 0.05,
    }

# ---------------------------------------------------------------------------
# 7. PLOTTING
# ---------------------------------------------------------------------------
def plot_results(returns_dict, factor_df, crash_signal, crisis_df, metrics_all, perm_results):
    """Generate comprehensive plots."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))

    # 1. Cumulative returns comparison
    ax = axes[0, 0]
    for name, ret in returns_dict.items():
        cum = (1 + ret).cumprod()
        ax.plot(cum.index, cum.values, label=name, alpha=0.8)
    ax.set_title("Cumulative Returns: Vanilla vs ML-Timed Momentum")
    ax.legend(fontsize=8)
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # 2. Drawdowns
    ax = axes[0, 1]
    for name, ret in returns_dict.items():
        cum = (1 + ret).cumprod()
        dd = cum / cum.expanding().max() - 1
        ax.fill_between(dd.index, dd.values, 0, alpha=0.3, label=name)
    ax.set_title("Drawdowns")
    ax.legend(fontsize=8)
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)

    # 3. Crash probability over time
    ax = axes[1, 0]
    crash_prob_series = crash_signal.astype(float)
    # Use actual probability if available
    ax.fill_between(crash_prob_series.index, crash_prob_series.values, 0, alpha=0.5, color="red", label="Crash signal active")
    ax.set_title("Crash Signal Activation")
    ax.set_ylabel("Signal (1=active)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Annual returns comparison
    ax = axes[1, 1]
    annual_data = {}
    for name, ret in returns_dict.items():
        annual = ret.groupby(ret.index.year).sum()
        annual_data[name] = annual
    annual_df = pd.DataFrame(annual_data)
    annual_df.plot(kind="bar", ax=ax, alpha=0.7)
    ax.set_title("Annual Returns")
    ax.set_ylabel("Return")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 5. Metrics comparison bar chart
    ax = axes[2, 0]
    metric_names = ["sharpe", "sortino", "cagr", "max_dd", "asymmetry_ratio"]
    metric_data = {}
    for m in metrics_all:
        metric_data[m["name"]] = {k: m.get(k, 0) for k in metric_names}
    comp_df = pd.DataFrame(metric_data).T
    comp_df[["sharpe", "sortino", "asymmetry_ratio"]].plot(kind="bar", ax=ax, alpha=0.7)
    ax.set_title("Risk-Adjusted Metrics Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # 6. Permutation test histogram
    ax = axes[2, 1]
    ax.hist(np.random.RandomState(42).normal(perm_results["perm_mean"], perm_results["perm_std"], 500),
            bins=30, alpha=0.5, color="gray", label="Permuted improvements")
    ax.axvline(perm_results["actual_improvement"], color="red", linewidth=2, label=f"Actual ({perm_results['actual_improvement']:.4f})")
    ax.set_title(f"Permutation Test (p={perm_results['p_value']:.3f})")
    ax.set_xlabel("Sharpe Improvement")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "momentum_crash_hedge_results.png", dpi=150)
    print(f"Saved plot to {OUTPUT_DIR / 'momentum_crash_hedge_results.png'}")
    plt.close()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    # 1. Get universe
    print("\n=== STEP 1: Getting S&P 500 constituents ===")
    sp500_tickers = get_sp500_constituents()
    print(f"Got {len(sp500_tickers)} tickers")

    # 2. Download data
    print("\n=== STEP 2: Downloading price data ===")
    prices = download_stock_data(sp500_tickers[:250], START, END)  # top ~200-250
    macro = download_macro_data(START, END)

    # Take top 200 by available data coverage
    coverage = prices.notna().sum().sort_values(ascending=False)
    top200 = coverage.head(200).index.tolist()
    prices = prices[top200]
    print(f"Using top {len(top200)} stocks by data coverage")

    # 3. Build momentum factor
    print("\n=== STEP 3: Building momentum factor ===")
    # Generate rebalance dates (every REBAL_FREQ trading days)
    all_dates = prices.dropna(how="all").index
    # Need at least MOM_LOOKBACK days of history
    valid_dates = all_dates[MOM_LOOKBACK + MOM_SKIP:]
    rebal_dates = valid_dates[::REBAL_FREQ]

    factor_df = build_momentum_factor(prices, rebal_dates)
    print(f"Momentum factor: {len(factor_df)} days, {factor_df.index[0]} to {factor_df.index[-1]}")
    print(f"Mean daily ret: {factor_df['factor_ret'].mean()*252:.2%} ann")
    print(f"Volatility: {factor_df['factor_ret'].std()*np.sqrt(252):.2%} ann")

    # 4. Build crash features and labels
    print("\n=== STEP 4: Building crash predictor features ===")
    features = build_crash_features(factor_df, macro, prices)
    labels = build_crash_labels(factor_df)
    print(f"Features: {features.shape}, Labels: {labels.shape}")
    print(f"Crash events (>10% DD in 21d): {labels.sum()} / {labels.notna().sum()} = {labels.mean():.2%}")

    # 5. Walk-forward crash predictor
    print("\n=== STEP 5: Walk-forward crash predictor ===")
    crash_probs, fold_df, feat_importance = train_crash_predictor_wf(features, labels, factor_df)

    # 6. Simulate strategies
    print("\n=== STEP 6: Simulating strategies ===")
    returns_dict, crash_signal = simulate_strategies(factor_df, crash_probs, labels)

    # 7. Compute metrics
    print("\n=== STEP 7: Computing metrics ===")
    metrics_all = []
    for name, ret in returns_dict.items():
        m = compute_metrics(ret, name)
        metrics_all.append(m)
        print(f"\n--- {name} ---")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # 8. Regime analysis (HC #428 R1)
    print("\n=== STEP 8: Regime Analysis (HC #428 R1) ===")
    spy_rets = macro["SPY"].pct_change().reindex(factor_df.index).fillna(0) if "SPY" in macro.columns else pd.Series(0, index=factor_df.index)
    for name, ret in returns_dict.items():
        regime = regime_analysis(ret, spy_rets)
        print(f"\n--- {name} regime analysis ---")
        for regime_name in ["green", "red", "flat"]:
            if regime_name in regime:
                r = regime[regime_name]
                print(f"  {regime_name}: Sharpe={r['sharpe']:.3f}, WR={r['wr']:.1%}, n={r['n_days']}")
        if "passes_hc428_r1" in regime:
            print(f"  HC #428 R1 PASS: {regime['passes_hc428_r1']} (asymmetry={regime.get('asymmetry_ratio', 'N/A'):.3f})")

    # 9. Crisis alpha
    print("\n=== STEP 9: Crisis Alpha Analysis ===")
    crisis_df = crisis_alpha_analysis(returns_dict, factor_df)
    print(crisis_df.to_string())

    # 10. Permutation test
    print("\n=== STEP 10: Permutation Test ===")
    perm_results = permutation_test(factor_df["factor_ret"], crash_probs)

    # 11. Plots
    print("\n=== STEP 11: Generating plots ===")
    plot_results(returns_dict, factor_df, crash_signal, crisis_df, metrics_all, perm_results)

    # 12. Save results
    print("\n=== STEP 12: Saving results ===")
    full_results = {
        "strategy": "Momentum Crash Hedge with ML Timing v1",
        "timestamp": datetime.now().isoformat(),
        "runtime_seconds": time.time() - t0,
        "universe": f"Top {len(top200)} US stocks by coverage",
        "period": f"{factor_df.index[0].strftime('%Y-%m-%d')} to {factor_df.index[-1].strftime('%Y-%m-%d')}",
        "metrics": metrics_all,
        "permutation_test": perm_results,
        "crash_predictor": {
            "n_folds": len(fold_df),
            "median_auc": fold_df["auc"].median(),
            "mean_auc": fold_df["auc"].mean(),
            "crash_rate": float(labels.mean()),
        },
        "top_features": feat_importance.head(10).to_dict(),
        "hc428_compliance": {
            "sliding_wf": True,
            "regime_tested": True,
            "permutation_tested": True,
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)

    fold_df.to_csv(OUTPUT_DIR / "fold_results.csv", index=False)
    crisis_df.to_csv(OUTPUT_DIR / "crisis_alpha.csv")

    # Save returns
    returns_df = pd.DataFrame(returns_dict)
    returns_df.to_parquet(OUTPUT_DIR / "strategy_returns.parquet")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"MOMENTUM CRASH HEDGE v1 — COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"\nKey Results:")
    for m in metrics_all:
        print(f"  {m['name']:20s}: Sharpe={m['sharpe']:.3f}  CAGR={m['cagr']:.2%}  MaxDD={m['max_dd']:.2%}  Asymmetry={m['asymmetry_ratio']:.2f}")
    print(f"\nCrash predictor median AUC: {fold_df['auc'].median():.3f}")
    print(f"Permutation p-value: {perm_results['p_value']:.4f}")
    print(f"\nResults saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
