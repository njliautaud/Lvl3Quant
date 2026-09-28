#!/usr/bin/env python3
"""
ML Covered Call Timing — Vol-Prediction-Gated Income Strategy
==============================================================
HC #0 (sliding walk-forward), HC #713 (fixed $100K, no DCA, BS 25% haircut),
HC #428 R1 (regime-agnostic OOT validation), HC #705 (adversarial validation).

Strategy:
  - Hold SPY continuously (long position)
  - Each week, GBM predicts next-5d realized vol (proven model: R²=0.65, corr=0.81)
  - Low-vol regime  (< median hist vol)  → sell ATM weekly covered call
  - Mid-vol regime  (median – 75th pct)  → sell 2% OTM weekly covered call
  - High-vol regime (> 75th pct)         → no call, hold SPY naked

Three strategies compared:
  (a) SPY buy-and-hold
  (b) Always-sell ATM covered call (BuyWrite proxy)
  (c) ML-timed covered call (this model)

Features match the proven vol model:
  spy_rvol_10d, spy_rvol_20d, vix_level, vix_rv_ratio, vol_of_vol_20d,
  uup_mom_20d, gld_vol_20d, xlf_vol_20d, spy_drawdown, spy_kurt_20d,
  spy_skew_20d, spy_tlt_corr_20d, spy_abs_ret_5d_avg

Walk-forward: 252d sliding train window, predict weekly (every Friday close).
Adversarial: permutation (shuffle ML signal, keep market returns in order),
             sub-period, outlier removal, R1 regime-stratified.

Output: /home/jupiter/Lvl3Quant/output/ml_covered_call_timing/results.json
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/ml_covered_call_timing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL   = 100_000          # HC #713: fixed, no DCA
RISK_FREE_ANNUAL  = 0.045            # ~current T-bill yield
BS_HAIRCUT        = 0.25             # HC #713 R2
ANNUALIZE         = np.sqrt(252)
TRAIN_DAYS        = 252              # HC #0: sliding 252d
CALL_DTE_DAYS     = 5               # weekly call (5 trading days ≈ 1 week)
CALL_DTE_YEARS    = CALL_DTE_DAYS / 252
OTM_MID_PCT       = 0.02            # 2% OTM for mid-vol regime
N_PERM            = 500             # permutation test iterations

# Vol regime thresholds (computed each WF step from training set)
LOW_VOL_PCTILE    = 50              # below median → sell ATM
HIGH_VOL_PCTILE   = 75             # above 75th    → no call

TICKERS = ["SPY", "^VIX", "UUP", "GLD", "XLF", "TLT"]
START   = "2010-01-01"
END     = "2026-07-18"


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────

def download_data() -> pd.DataFrame:
    """Download daily OHLCV for all tickers, cache to parquet."""
    cache = OUTPUT_DIR / "raw_data.parquet"
    if cache.exists():
        age_h = (pd.Timestamp.now().timestamp() - cache.stat().st_mtime) / 3600
        if age_h < 24:
            print(f"  Loading cached data…")
            return pd.read_parquet(cache)

    print(f"  Downloading {TICKERS} from {START} to {END}…")
    frames: dict[str, pd.DataFrame] = {}
    for tkr in TICKERS:
        label = tkr.replace("^", "")
        try:
            df = yf.download(tkr, start=START, end=END, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                print(f"    WARNING: no data for {tkr}")
                continue
            frames[label] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            print(f"    {label}: {len(df)} rows")
        except Exception as e:
            print(f"    ERROR {tkr}: {e}")

    if "SPY" not in frames:
        raise RuntimeError("SPY download failed")

    # Merge all tickers on SPY index
    base = frames["SPY"].copy()
    base.columns = [f"SPY_{c}" for c in base.columns]
    for label, df in frames.items():
        if label == "SPY":
            continue
        base[f"{label}_Close"] = df["Close"].reindex(base.index)
        if "Volume" in df.columns:
            base[f"{label}_Volume"] = df["Volume"].reindex(base.index)

    base.index = pd.to_datetime(base.index)
    if base.index.tz is not None:
        base.index = base.index.tz_localize(None)
    base = base.ffill(limit=5).dropna(subset=["SPY_Close"])
    base.to_parquet(cache)
    print(f"  Downloaded {len(base)} rows, {base.index[0].date()} – {base.index[-1].date()}")
    return base


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the proven vol-prediction feature set (GBM R²=0.65, corr=0.81)."""
    feat = pd.DataFrame(index=raw.index)

    spy  = raw["SPY_Close"].astype(float)
    lr   = np.log(spy / spy.shift(1))

    # ── Realized vol features ──────────────────────────────────────────────
    feat["spy_rvol_10d"] = lr.rolling(10).std() * ANNUALIZE
    feat["spy_rvol_20d"] = lr.rolling(20).std() * ANNUALIZE

    # ── VIX level ─────────────────────────────────────────────────────────
    vix = raw.get("VIX_Close", pd.Series(dtype=float)).reindex(raw.index).ffill()
    feat["vix_level"] = vix / 100.0

    # ── VIX / RV ratio (implied-vs-realized premium) ──────────────────────
    feat["vix_rv_ratio"] = (vix / 100.0) / (feat["spy_rvol_20d"] + 1e-8)

    # ── Vol-of-vol (20d rolling std of daily rvol) ────────────────────────
    daily_rvol = lr.abs()   # proxy: daily |ret|
    feat["vol_of_vol_20d"] = daily_rvol.rolling(20).std() * ANNUALIZE

    # ── UUP 20d momentum ──────────────────────────────────────────────────
    uup = raw.get("UUP_Close", pd.Series(dtype=float)).reindex(raw.index).ffill()
    feat["uup_mom_20d"] = uup.pct_change(20)

    # ── GLD 20d realized vol ──────────────────────────────────────────────
    gld = raw.get("GLD_Close", pd.Series(dtype=float)).reindex(raw.index).ffill()
    gld_lr = np.log(gld / gld.shift(1))
    feat["gld_vol_20d"] = gld_lr.rolling(20).std() * ANNUALIZE

    # ── XLF 20d realized vol ──────────────────────────────────────────────
    xlf = raw.get("XLF_Close", pd.Series(dtype=float)).reindex(raw.index).ffill()
    xlf_lr = np.log(xlf / xlf.shift(1))
    feat["xlf_vol_20d"] = xlf_lr.rolling(20).std() * ANNUALIZE

    # ── SPY drawdown from rolling 252d high ───────────────────────────────
    roll_high = spy.rolling(252, min_periods=20).max()
    feat["spy_drawdown"] = (spy - roll_high) / roll_high

    # ── SPY 20d kurtosis (fat-tail environment) ───────────────────────────
    feat["spy_kurt_20d"] = lr.rolling(20).kurt()

    # ── SPY 20d skewness (left-skew = bearish) ────────────────────────────
    feat["spy_skew_20d"] = lr.rolling(20).skew()

    # ── SPY-TLT 20d rolling correlation ───────────────────────────────────
    tlt = raw.get("TLT_Close", pd.Series(dtype=float)).reindex(raw.index).ffill()
    tlt_lr = np.log(tlt / tlt.shift(1))
    feat["spy_tlt_corr_20d"] = lr.rolling(20).corr(tlt_lr)

    # ── Average absolute 5d return (realized short-term activity) ─────────
    feat["spy_abs_ret_5d_avg"] = lr.abs().rolling(5).mean()

    feat = feat.replace([np.inf, -np.inf], np.nan).ffill(limit=5)
    return feat


def build_target(raw: pd.DataFrame, horizon: int = 5) -> pd.Series:
    """Next-N-day forward realized vol (annualized) — what the GBM predicts."""
    spy = raw["SPY_Close"].astype(float)
    lr  = np.log(spy / spy.shift(1))
    fwd_rv = (
        lr.shift(-horizon)
          .rolling(horizon)
          .std()
        * ANNUALIZE
    )
    # shift back so row t has the RV realized over [t+1 … t+horizon]
    fwd_rv = lr.rolling(horizon).std().shift(-horizon) * ANNUALIZE
    return fwd_rv.rename("fwd_rv_5d")


# ─────────────────────────────────────────────────────────────────────────────
# 3. BLACK-SCHOLES HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Standard Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def premium_with_haircut(S: float, K: float, T: float, r: float,
                          sigma_iv: float) -> float:
    """BS call price × (1 − BS_HAIRCUT) for realism (HC #713 R2)."""
    raw_p = bs_call_price(S, K, T, r, sigma_iv)
    return raw_p * (1.0 - BS_HAIRCUT)


# ─────────────────────────────────────────────────────────────────────────────
# 4. WEEKLY INDEX EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def get_weekly_fridays(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last trading day of each ISO week (i.e., 'Friday closes')."""
    df = pd.DataFrame({"date": idx})
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.isocalendar().year.astype(int)
    fridays = df.groupby(["year", "week"])["date"].max()
    return pd.DatetimeIndex(sorted(fridays.values))


# ─────────────────────────────────────────────────────────────────────────────
# 5. GBM WALK-FORWARD SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def run_wf_gbm(feat: pd.DataFrame, target: pd.Series,
               fridays: pd.DatetimeIndex) -> pd.Series:
    """
    Sliding 252d walk-forward GBM.
    At each Friday, train on the past 252 days and predict next-5d vol.
    Returns a Series of predicted vol aligned to each Friday.
    """
    try:
        import lightgbm as lgb
        LGB_AVAILABLE = True
    except ImportError:
        LGB_AVAILABLE = False

    if not LGB_AVAILABLE:
        try:
            from sklearn.ensemble import GradientBoostingRegressor as GBR
        except ImportError:
            raise RuntimeError("Neither lightgbm nor sklearn available")

    all_dates = feat.index
    predictions: dict[pd.Timestamp, float] = {}

    print(f"  Running {len(fridays)} weekly WF GBM predictions…")
    for i, friday in enumerate(fridays):
        # Training window: last TRAIN_DAYS days BEFORE this Friday
        mask = all_dates < friday
        hist = all_dates[mask]
        if len(hist) < TRAIN_DAYS + 20:
            continue   # not enough history yet
        train_end_iloc   = len(hist)
        train_start_iloc = max(0, train_end_iloc - TRAIN_DAYS)
        train_dates = all_dates[train_start_iloc:train_end_iloc]

        X_tr = feat.loc[train_dates].values
        y_tr = target.loc[train_dates].values

        valid_mask = ~(np.isnan(X_tr).any(axis=1) | np.isnan(y_tr))
        X_tr, y_tr = X_tr[valid_mask], y_tr[valid_mask]
        if len(X_tr) < 50:
            continue

        # Feature vector for prediction: the Friday itself
        if friday not in feat.index:
            # Find nearest date
            nearest = all_dates[all_dates <= friday]
            if len(nearest) == 0:
                continue
            friday_feat_date = nearest[-1]
        else:
            friday_feat_date = friday

        X_pred = feat.loc[[friday_feat_date]].values
        if np.isnan(X_pred).any():
            continue

        try:
            if LGB_AVAILABLE:
                model = lgb.LGBMRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=10,
                    random_state=42,
                    verbose=-1,
                    n_jobs=4,
                )
            else:
                from sklearn.ensemble import GradientBoostingRegressor as GBR
                model = GBR(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    random_state=42,
                )
            model.fit(X_tr, y_tr)
            pred = float(model.predict(X_pred)[0])
            predictions[friday] = max(pred, 0.01)   # clip at 1% floor
        except Exception as e:
            print(f"    WF error at {friday.date()}: {e}")
            continue

        if (i + 1) % 50 == 0:
            print(f"    … {i+1}/{len(fridays)} done")

    return pd.Series(predictions, name="ml_pred_vol")


# ─────────────────────────────────────────────────────────────────────────────
# 6. REGIME CLASSIFICATION (per WF training distribution)
# ─────────────────────────────────────────────────────────────────────────────

def classify_regime(pred_vol: pd.Series, target: pd.Series,
                    fridays: pd.DatetimeIndex) -> pd.Series:
    """
    For each Friday, classify the ML prediction as LOW / MID / HIGH vol regime
    using the 50th and 75th percentiles of the TRAINING distribution
    (computed from the same rolling 252d window → no lookahead).

    Returns a Series with values 'LOW', 'MID', 'HIGH' indexed by friday dates.
    """
    all_dates = target.index
    regimes: dict[pd.Timestamp, str] = {}

    for friday in pred_vol.index:
        if friday not in pred_vol.index:
            continue
        ml_vol = pred_vol[friday]

        # Training distribution of realized vol up to this Friday
        hist_mask = all_dates < friday
        hist_dates = all_dates[hist_mask]
        if len(hist_dates) < TRAIN_DAYS:
            train_dates = hist_dates
        else:
            train_dates = hist_dates[-TRAIN_DAYS:]

        train_rv = target.loc[train_dates].dropna()
        if len(train_rv) < 30:
            regimes[friday] = "MID"
            continue

        p50 = train_rv.quantile(LOW_VOL_PCTILE / 100)
        p75 = train_rv.quantile(HIGH_VOL_PCTILE / 100)

        if ml_vol < p50:
            regimes[friday] = "LOW"
        elif ml_vol > p75:
            regimes[friday] = "HIGH"
        else:
            regimes[friday] = "MID"

    return pd.Series(regimes, name="regime")


# ─────────────────────────────────────────────────────────────────────────────
# 7. COVERED CALL BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def backtest_strategy(raw: pd.DataFrame, regimes: pd.Series,
                       mode: str = "ML") -> pd.DataFrame:
    """
    Run covered call strategy week-by-week.

    mode:
      'ML'       — use regime signal (LOW/MID/HIGH) to decide whether to sell calls
      'ALWAYS'   — always sell ATM calls (BuyWrite proxy)
      'HOLD'     — never sell calls (pure SPY buy-and-hold)

    Returns a DataFrame of weekly portfolio values and P&L breakdown.
    """
    spy_px = raw["SPY_Close"].astype(float)
    vix    = raw.get("VIX_Close", pd.Series(dtype=float)).reindex(raw.index).ffill() / 100.0

    fridays = get_weekly_fridays(spy_px.index)
    # Restrict to fridays where we have both SPY and (for ML mode) a regime signal
    if mode == "ML":
        valid_fridays = [f for f in fridays if f in regimes.index]
    else:
        valid_fridays = list(fridays)

    # Need at least one year of history before trading starts
    # Start from the first friday where we have ML signal
    start_idx = 0
    while start_idx < len(valid_fridays):
        if spy_px.index[0] <= valid_fridays[start_idx] <= spy_px.index[-1]:
            break
        start_idx += 1

    capital = float(INITIAL_CAPITAL)
    n_shares = capital / spy_px.loc[spy_px.index[spy_px.index <= valid_fridays[start_idx]][-1]]
    records = []

    for i, friday in enumerate(valid_fridays[start_idx:], start=start_idx):
        # Price at this Friday's close (entry point)
        if friday not in spy_px.index:
            nearest = spy_px.index[spy_px.index <= friday]
            if len(nearest) == 0:
                continue
            entry_date = nearest[-1]
        else:
            entry_date = friday

        entry_price = float(spy_px[entry_date])
        entry_vix   = float(vix[entry_date]) if entry_date in vix.index else 0.20

        # Next Friday = expiry
        if i + 1 >= len(valid_fridays):
            break
        next_friday = valid_fridays[i + 1]
        if next_friday not in spy_px.index:
            nearest = spy_px.index[spy_px.index <= next_friday]
            if len(nearest) == 0:
                continue
            exit_date = nearest[-1]
        else:
            exit_date = next_friday
        exit_price = float(spy_px[exit_date])

        spy_return = (exit_price / entry_price) - 1.0

        # Determine action
        if mode == "HOLD":
            action = "HOLD"
        elif mode == "ALWAYS":
            action = "ATM"
        else:  # ML
            regime = regimes.get(friday, "MID")
            if regime == "LOW":
                action = "ATM"
            elif regime == "HIGH":
                action = "HOLD"
            else:
                action = "OTM"

        # ── Position value at entry ────────────────────────────────────────
        portfolio_value_entry = n_shares * entry_price

        # ── Option premium collected ───────────────────────────────────────
        premium_collected = 0.0
        strike = 0.0
        if action in ("ATM", "OTM"):
            if action == "ATM":
                strike = entry_price          # ATM
            else:
                strike = entry_price * (1.0 + OTM_MID_PCT)  # 2% OTM

            sigma_iv = max(entry_vix, 0.05)   # VIX as proxy for implied vol
            T        = CALL_DTE_YEARS
            r        = RISK_FREE_ANNUAL

            premium_per_share = premium_with_haircut(
                S=entry_price, K=strike, T=T, r=r, sigma_iv=sigma_iv
            )
            # Number of contracts: 1 contract = 100 shares, but we simulate
            # per-share economics for simplicity (fractional contracts allowed)
            premium_collected = premium_per_share * n_shares

        # ── Determine whether called away ──────────────────────────────────
        called_away = False
        if action in ("ATM", "OTM") and exit_price > strike:
            called_away = True

        # ── P&L this week ─────────────────────────────────────────────────
        if called_away:
            # Gain: (strike - entry_price) × shares + premium
            equity_gain   = (strike - entry_price) * n_shares
            week_pnl      = equity_gain + premium_collected
            # Must "rebuy" SPY at exit price next week
            new_portfolio = portfolio_value_entry + week_pnl
            n_shares      = new_portfolio / exit_price  # buy at new price
        else:
            # SPY return unrestricted + premium
            equity_gain   = spy_return * portfolio_value_entry
            week_pnl      = equity_gain + premium_collected
            new_portfolio = portfolio_value_entry + week_pnl
            n_shares      = new_portfolio / exit_price

        capital = new_portfolio

        records.append({
            "date":              entry_date,
            "exit_date":         exit_date,
            "entry_price":       entry_price,
            "exit_price":        exit_price,
            "strike":            strike,
            "action":            action,
            "called_away":       called_away,
            "spy_return":        spy_return,
            "equity_gain":       equity_gain,
            "premium_collected": premium_collected,
            "week_pnl":          week_pnl,
            "portfolio_value":   new_portfolio,
            "n_shares":          n_shares,
        })

    df = pd.DataFrame(records).set_index("date")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 8. PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(bt: pd.DataFrame, label: str) -> dict:
    """Compute full risk-adjusted metrics from backtest DataFrame."""
    pv = bt["portfolio_value"]
    wr = bt["week_pnl"].apply(lambda x: 1 if x > 0 else 0)

    weekly_ret = pv.pct_change().dropna()
    annual_factor = 52.0   # weekly → annual

    if len(weekly_ret) < 4:
        return {"label": label, "error": "insufficient data"}

    # CAGR
    total_ret = pv.iloc[-1] / INITIAL_CAPITAL
    n_years   = len(weekly_ret) / 52.0
    cagr      = total_ret ** (1.0 / max(n_years, 0.1)) - 1.0

    # Sharpe
    sharpe = (weekly_ret.mean() / (weekly_ret.std() + 1e-9)) * np.sqrt(annual_factor)

    # Sortino
    neg = weekly_ret[weekly_ret < 0]
    downside_std = neg.std() if len(neg) > 1 else 1e-9
    sortino = (weekly_ret.mean() / (downside_std + 1e-9)) * np.sqrt(annual_factor)

    # Max drawdown
    roll_max = pv.cummax()
    dd = (pv - roll_max) / roll_max
    max_dd = float(dd.min())

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # Win rate & profit factor
    wins  = weekly_ret[weekly_ret > 0]
    losses = weekly_ret[weekly_ret <= 0]
    wr_pct = len(wins) / len(weekly_ret)
    pf = (wins.sum() / (-losses.sum() + 1e-9)) if len(losses) > 0 else np.nan

    # Annual income yield (premium collected / average portfolio value)
    total_premium  = bt["premium_collected"].sum()
    avg_portfolio  = pv.mean()
    n_years_actual = len(bt) / 52.0
    annual_income_yield = (total_premium / avg_portfolio) / max(n_years_actual, 0.1)

    # Action breakdown (ML / ALWAYS strategies)
    action_counts = bt["action"].value_counts().to_dict() if "action" in bt.columns else {}

    return {
        "label":               label,
        "final_value":         round(float(pv.iloc[-1]), 2),
        "total_return_pct":    round((total_ret - 1) * 100, 2),
        "cagr_pct":            round(cagr * 100, 2),
        "sharpe":              round(sharpe, 3),
        "sortino":             round(sortino, 3),
        "max_drawdown_pct":    round(max_dd * 100, 2),
        "calmar":              round(calmar, 3) if not np.isnan(calmar) else None,
        "win_rate_pct":        round(wr_pct * 100, 2),
        "profit_factor":       round(pf, 3) if not np.isnan(pf) else None,
        "annual_income_yield_pct": round(annual_income_yield * 100, 2),
        "n_weeks":             len(bt),
        "n_years":             round(n_years_actual, 1),
        "action_counts":       action_counts,
    }


def regime_stratified_sharpe(bt: pd.DataFrame, raw: pd.DataFrame) -> dict:
    """
    R1 regime-agnostic validation: compute Sharpe per market regime.
    Regime = sign of SPY 21d close-to-close return at each weekly bar.
    """
    spy_close = raw["SPY_Close"].astype(float)
    spy_21d   = spy_close.pct_change(21)

    bt_idx = bt.index
    ret = bt["week_pnl"] / (bt["portfolio_value"].shift(1).fillna(INITIAL_CAPITAL))

    results = {}
    for regime_label, mask_fn in [
        ("green", lambda r: r > 0.01),
        ("red",   lambda r: r < -0.01),
        ("flat",  lambda r: r.between(-0.01, 0.01)),
    ]:
        regime_dates = spy_21d[mask_fn(spy_21d)].index
        sub = ret[ret.index.isin(regime_dates)]
        if len(sub) < 5:
            results[f"sharpe_{regime_label}"] = None
            continue
        s = (sub.mean() / (sub.std() + 1e-9)) * np.sqrt(52)
        results[f"sharpe_{regime_label}"] = round(s, 3)

    # R1 rejection test: |Sharpe_green − Sharpe_red| / max(|g|,|r|) > 0.50
    sg = results.get("sharpe_green")
    sr = results.get("sharpe_red")
    if sg is not None and sr is not None and max(abs(sg), abs(sr)) > 0:
        ratio = abs(sg - sr) / max(abs(sg), abs(sr))
        results["r1_regime_divergence"] = round(ratio, 3)
        results["r1_pass"] = bool(ratio <= 0.50)
    else:
        results["r1_regime_divergence"] = None
        results["r1_pass"] = None

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 9. ADVERSARIAL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def permutation_test(raw: pd.DataFrame, regimes: pd.Series,
                     true_sharpe: float, n_perm: int = N_PERM) -> dict:
    """
    Shuffle the ML SIGNAL (which weeks we sell calls) while keeping
    market returns in their original time order.  Measures whether the
    ML timing genuinely adds value vs random call-selling timing.
    """
    print(f"  Running {n_perm}-iteration permutation test…")
    perm_sharpes = []

    regime_values = regimes.values.copy()
    regime_index  = regimes.index

    rng = np.random.default_rng(seed=42)
    for it in range(n_perm):
        shuffled_vals  = rng.permutation(regime_values)
        shuffled_reg   = pd.Series(shuffled_vals, index=regime_index, name="regime")
        bt_perm = backtest_strategy(raw, shuffled_reg, mode="ML")
        if len(bt_perm) < 10:
            continue
        m = compute_metrics(bt_perm, "perm")
        perm_sharpes.append(m["sharpe"])

    if not perm_sharpes:
        return {"error": "permutation test produced no results"}

    perm_arr = np.array(perm_sharpes)
    p_value  = float((perm_arr >= true_sharpe).mean())
    return {
        "true_sharpe":         round(true_sharpe, 3),
        "perm_mean_sharpe":    round(perm_arr.mean(), 3),
        "perm_p95_sharpe":     round(np.percentile(perm_arr, 95), 3),
        "p_value":             round(p_value, 4),
        "n_permutations":      len(perm_sharpes),
        "significant_p05":     bool(p_value < 0.05),
        "significant_p10":     bool(p_value < 0.10),
    }


def sub_period_test(raw: pd.DataFrame, regimes: pd.Series) -> dict:
    """Split history into halves; compute Sharpe for ML strategy in each."""
    fridays = get_weekly_fridays(raw["SPY_Close"].index)
    mid     = fridays[len(fridays) // 2]

    results = {}
    for period_label, date_mask in [
        ("first_half",  regimes.index < mid),
        ("second_half", regimes.index >= mid),
    ]:
        sub_reg = regimes[date_mask]
        if len(sub_reg) < 20:
            results[period_label] = None
            continue
        raw_sub = raw[raw.index <= sub_reg.index.max()]
        bt_sub  = backtest_strategy(raw_sub, sub_reg, mode="ML")
        if len(bt_sub) < 10:
            results[period_label] = None
            continue
        m = compute_metrics(bt_sub, period_label)
        results[period_label] = {
            "sharpe":  m["sharpe"],
            "cagr_pct": m["cagr_pct"],
            "max_dd_pct": m["max_drawdown_pct"],
        }

    return results


def outlier_test(raw: pd.DataFrame, regimes: pd.Series) -> dict:
    """Remove top 5% absolute weekly moves (crisis/meme events); rerun backtest."""
    spy_px     = raw["SPY_Close"].astype(float)
    spy_weekly = spy_px.resample("W").last().pct_change().dropna()
    p95        = spy_weekly.abs().quantile(0.95)
    normal_weeks = spy_weekly[spy_weekly.abs() <= p95].index

    # Map weekly dates to business-day fridays
    norm_reg = regimes[regimes.index.map(
        lambda d: any(abs((d - w).days) <= 4 for w in normal_weeks)
    )]
    if len(norm_reg) < 20:
        return {"error": "too few non-outlier weeks"}

    bt_no_outlier = backtest_strategy(raw, norm_reg, mode="ML")
    if len(bt_no_outlier) < 10:
        return {"error": "insufficient post-outlier-removal data"}

    m = compute_metrics(bt_no_outlier, "no_outlier")
    return {
        "sharpe":          m["sharpe"],
        "cagr_pct":        m["cagr_pct"],
        "max_dd_pct":      m["max_drawdown_pct"],
        "n_weeks_removed": int(len(regimes) - len(norm_reg)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. MAIN ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n=== ML Covered Call Timing — Full Run ===\n")

    # ── 1. Data ───────────────────────────────────────────────────────────
    print("[1/7] Downloading data…")
    raw = download_data()

    # ── 2. Features & target ─────────────────────────────────────────────
    print("[2/7] Building features…")
    feat   = build_features(raw)
    target = build_target(raw, horizon=5)

    # Align
    common = feat.index.intersection(target.index).intersection(raw.index)
    feat   = feat.loc[common]
    target = target.loc[common]
    raw_   = raw.loc[common]

    # ── 3. WF GBM signal ─────────────────────────────────────────────────
    print("[3/7] Walk-forward GBM vol prediction (252d sliding)…")
    fridays    = get_weekly_fridays(raw_["SPY_Close"].index)
    # Only use fridays after enough history is available
    min_date   = common[TRAIN_DAYS + 20] if len(common) > TRAIN_DAYS + 20 else common[-1]
    fridays_wf = pd.DatetimeIndex([f for f in fridays if f >= min_date])

    ml_pred = run_wf_gbm(feat, target, fridays_wf)
    print(f"  Generated {len(ml_pred)} weekly ML vol predictions")
    print(f"  Pred range: {ml_pred.min():.3f} – {ml_pred.max():.3f} annualized vol")

    # ── 4. Regime classification ─────────────────────────────────────────
    print("[4/7] Classifying vol regimes…")
    regimes = classify_regime(ml_pred, target, fridays_wf)
    regime_counts = regimes.value_counts().to_dict()
    print(f"  Regime distribution: {regime_counts}")

    # ── 5. Backtests ─────────────────────────────────────────────────────
    print("[5/7] Running three backtests…")

    # For HOLD and ALWAYS strategies, restrict to same time window as ML
    ml_start = ml_pred.index[0]
    raw_trim  = raw_[raw_.index >= ml_start]

    # ML strategy uses regimes starting from first WF prediction
    bt_ml    = backtest_strategy(raw_trim, regimes, mode="ML")
    bt_always = backtest_strategy(raw_trim, regimes, mode="ALWAYS")
    bt_hold  = backtest_strategy(raw_trim, regimes, mode="HOLD")

    print(f"  ML strategy: {len(bt_ml)} weeks, final ${bt_ml['portfolio_value'].iloc[-1]:,.0f}")
    print(f"  Always-sell: {len(bt_always)} weeks, final ${bt_always['portfolio_value'].iloc[-1]:,.0f}")
    print(f"  Hold (B&H):  {len(bt_hold)} weeks, final ${bt_hold['portfolio_value'].iloc[-1]:,.0f}")

    # ── 6. Metrics ───────────────────────────────────────────────────────
    print("[6/7] Computing performance metrics…")
    m_ml    = compute_metrics(bt_ml,    "ML-Timed Covered Call")
    m_always = compute_metrics(bt_always, "Always-Sell ATM (BuyWrite)")
    m_hold  = compute_metrics(bt_hold,  "SPY Buy-and-Hold")

    regime_ml   = regime_stratified_sharpe(bt_ml,    raw_trim)
    regime_bwri = regime_stratified_sharpe(bt_always, raw_trim)
    regime_hold = regime_stratified_sharpe(bt_hold,   raw_trim)

    # ── 7. Adversarial validation ─────────────────────────────────────────
    print("[7/7] Adversarial validation…")

    perm_result   = permutation_test(raw_trim, regimes, m_ml["sharpe"])
    subperiod_res = sub_period_test(raw_trim, regimes)
    outlier_res   = outlier_test(raw_trim, regimes)

    # ── Assemble results ─────────────────────────────────────────────────
    results = {
        "run_date": pd.Timestamp.now().isoformat(),
        "data_range": {
            "start": str(raw_trim.index[0].date()),
            "end":   str(raw_trim.index[-1].date()),
        },
        "model_description": {
            "type":          "LightGBM GBM (252d sliding walk-forward)",
            "target":        "next-5d realized vol (annualized)",
            "reported_r2":   0.65,
            "reported_corr": 0.81,
            "features":      list(feat.columns),
            "n_features":    len(feat.columns),
            "n_weekly_preds": len(ml_pred),
        },
        "regime_distribution": regime_counts,
        "vol_regimes": {
            "low_vol_threshold":  "below median of 252d training RV",
            "high_vol_threshold": "above 75th pctile of 252d training RV",
            "low_vol_action":     "sell ATM weekly covered call",
            "mid_vol_action":     "sell 2% OTM weekly covered call",
            "high_vol_action":    "hold SPY naked",
        },
        "performance": {
            "ml_timed":   m_ml,
            "always_sell": m_always,
            "buy_and_hold": m_hold,
        },
        "regime_stratified": {
            "ml_timed":    regime_ml,
            "always_sell": regime_bwri,
            "buy_and_hold": regime_hold,
        },
        "adversarial": {
            "permutation_test": perm_result,
            "sub_period":       subperiod_res,
            "outlier_removal":  outlier_res,
        },
        "constants": {
            "initial_capital":  INITIAL_CAPITAL,
            "bs_haircut":       BS_HAIRCUT,
            "call_dte_days":    CALL_DTE_DAYS,
            "otm_pct":          OTM_MID_PCT,
            "risk_free_annual": RISK_FREE_ANNUAL,
            "train_window_days": TRAIN_DAYS,
        },
    }

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    header = f"{'Metric':<28} {'ML-Timed':>12} {'Always ATM':>12} {'B&H SPY':>10}"
    print(header)
    print("-" * 65)
    rows = [
        ("CAGR %",              "cagr_pct"),
        ("Sharpe",              "sharpe"),
        ("Sortino",             "sortino"),
        ("Max DD %",            "max_drawdown_pct"),
        ("Calmar",              "calmar"),
        ("Win Rate %",          "win_rate_pct"),
        ("Profit Factor",       "profit_factor"),
        ("Annual Income Yield %","annual_income_yield_pct"),
        ("Final Value $",       "final_value"),
    ]
    for row_label, key in rows:
        v_ml   = m_ml.get(key, "–")
        v_alw  = m_always.get(key, "–")
        v_hold = m_hold.get(key, "–")
        fmt = lambda v: f"{v:>12.2f}" if isinstance(v, (int, float)) and v is not None else f"{'–':>12}"
        print(f"{row_label:<28}{fmt(v_ml)}{fmt(v_alw)}{fmt(v_hold)}")

    print("\nRegime-Stratified Sharpe (ML-Timed):")
    for k, v in regime_ml.items():
        if k.startswith("sharpe"):
            regime = k.replace("sharpe_", "").upper()
            print(f"  {regime:<10} {v}")
    r1_pass = regime_ml.get("r1_pass")
    r1_div  = regime_ml.get("r1_regime_divergence")
    print(f"  R1 regime divergence: {r1_div}  →  {'PASS' if r1_pass else 'FAIL (regime-biased!)'}")

    print("\nPermutation Test (signal shuffle):")
    for k, v in perm_result.items():
        print(f"  {k}: {v}")

    print(f"\nAll results → {out_path}")
    return results


if __name__ == "__main__":
    main()
