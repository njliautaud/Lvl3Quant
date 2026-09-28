#!/usr/bin/env python3
"""
Stock Predictor v4 — Insider Buying + Relative Return Fusion (HC #698)
========================================================================
v2 passed R1 regime test (regime-neutral, gap 0.101) but signal too weak:
  - Only 50 high-confidence signals in 6 years
  - Long-short Sharpe only 0.13 (2.4% CAGR)

v4 adds SEC Form 4 insider buying as a differentiated alpha signal:
  - Insider cluster buying (2+ insiders in 30d) is a documented edge (6%/yr alpha)
  - C-suite purchases >$100K are the strongest signal
  - Contrarian flag: insider buys when stock is below 200MA = highest conviction

v4 APPROACH:
  1. Use v2 relative return model as base signal
  2. Add insider buying features to the feature set
  3. Test insider-only model for comparison
  4. Test fusion: "only trade v2 signals when insider signal also present"

OUTCOME: 3 models compared:
  A. v2 base (relative return, LGBM)
  B. Insider-only (cluster/C-suite buy in last 20-30 days)
  C. Fusion: v2 signal × insider confirmation (BOTH required)

All tested with:
  - Walk-forward (252d train, 21d step, 60d embargo)
  - HC #428 R1: regime-agnostic gate (gap ≤ 0.50)
  - 100-trial permutation test
  - Real SEC EDGAR Form 4 data (cached)

Output: /home/nick/Lvl3Quant/output/growth_research/stock_prediction/v4_insider_fusion/
"""

import json
import os
import re
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Ensure dependencies ---
for pkg in ["yfinance", "lightgbm", "sklearn", "requests"]:
    try:
        __import__(pkg)
    except ImportError:
        os.system(f"{sys.executable} -m pip install {pkg} -q")

import lightgbm as lgb
import requests
import yfinance as yf
from sklearn.metrics import precision_score, accuracy_score

# --- Paths (Neptune paths) ---
BASE = Path("/home/nick/Lvl3Quant")
OUTPUT_DIR = BASE / "output/growth_research/stock_prediction/v4_insider_fusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR  = BASE / "output/growth_research/stock_prediction/cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
INSIDER_CACHE = BASE / "output/growth_research/insider_alpha/cache"
INSIDER_CACHE.mkdir(parents=True, exist_ok=True)

# Jupiter shared cache path (may be available via NFS or copy)
JUPITER_INSIDER_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/insider_alpha")

# --- SEC EDGAR ---
SEC_HEADERS = {"User-Agent": "ResearchBot research@example.com", "Accept": "application/json"}
SEC_RATE = 0.12  # ~8 req/sec

# --- Universe (~150 diversified) ---
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "META": "Tech", "NVDA": "Tech",
    "AMD": "Tech", "AVGO": "Tech", "MU": "Tech", "QCOM": "Tech", "INTC": "Tech",
    "ORCL": "Tech", "ADBE": "Tech", "CRM": "Tech", "NOW": "Tech", "INTU": "Tech",
    "CRWD": "Cyber", "ZS": "Cyber", "PANW": "Cyber", "FTNT": "Cyber",
    "AMZN": "Consume", "TSLA": "Consume", "NFLX": "Consume", "HD": "Consume",
    "LOW": "Consume", "NKE": "Consume", "SBUX": "Consume", "MCD": "Consume",
    "JPM": "Fin", "BAC": "Fin", "GS": "Fin", "MS": "Fin", "V": "Fin", "MA": "Fin",
    "BRK-B": "Fin", "AXP": "Fin", "SCHW": "Fin", "COF": "Fin",
    "JNJ": "Health", "PFE": "Health", "ABBV": "Health", "MRK": "Health",
    "LLY": "Health", "UNH": "Health", "TMO": "Health", "AMGN": "Health",
    "GILD": "Health", "REGN": "Health", "VRTX": "Health", "ISRG": "Health",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy", "SLB": "Energy",
    "HON": "Indust", "CAT": "Indust", "DE": "Indust", "UNP": "Indust", "UPS": "Indust",
    "RTX": "Indust", "LMT": "Indust", "GD": "Indust", "BA": "Indust", "MMM": "Indust",
    "PG": "Staple", "KO": "Staple", "PEP": "Staple", "WMT": "Staple", "COST": "Staple",
    "NEE": "Utility", "DUK": "Utility", "SO": "Utility",
    "DIS": "Media", "CMCSA": "Media", "T": "Telecom", "VZ": "Telecom",
    "LIN": "Matl", "APD": "Matl", "NEM": "Matl", "FCX": "Matl",
    "PLTR": "Other", "UBER": "Other", "COIN": "Other",
}
UNIVERSE = list(SECTOR_MAP.keys())

HOLD_DAYS = 60
EXCESS_THRESHOLD = 0.03  # 3% outperformance target (stock vs SPY in 60d)
INSIDER_LOOKBACK = 30    # days to look back for insider signal


# ============================================================
#  SEC EDGAR INSIDER DATA
# ============================================================

def get_cik_map():
    cache = INSIDER_CACHE / "cik_map.json"
    if cache.exists():
        return json.loads(cache.read_text())
    resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=30)
    data = resp.json()
    cik_map = {v["ticker"].upper(): str(v["cik_str"]) for v in data.values() if "ticker" in v}
    cache.write_text(json.dumps(cik_map))
    return cik_map


def fetch_insider_transactions(ticker, cik, start_date, end_date):
    """Fetch Form 4 transactions from EDGAR for a given ticker/CIK."""
    cache_file = INSIDER_CACHE / f"{ticker}_{start_date[:7]}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    cik_padded = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    try:
        time.sleep(SEC_RATE)
        resp = requests.get(url, headers=SEC_HEADERS, timeout=20)
        if resp.status_code != 200:
            return []
        sub_data = resp.json()
    except Exception:
        return []

    filings = sub_data.get("filings", {}).get("recent", {})
    forms   = filings.get("form", [])
    dates   = filings.get("filingDate", [])
    accnos  = filings.get("accessionNumber", [])

    transactions = []
    for form, date_str, accno in zip(forms, dates, accnos):
        if form not in ("4", "4/A"):
            continue
        if date_str < start_date or date_str > end_date:
            continue
        acc_clean = accno.replace("-", "")
        for fname in ("form4.xml", "doc4.xml"):
            xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{fname}"
            try:
                time.sleep(SEC_RATE)
                xresp = requests.get(xml_url, headers=SEC_HEADERS, timeout=15)
                if xresp.status_code == 200 and "<ownershipDocument" in xresp.text:
                    txns = parse_form4(xresp.text, date_str, ticker)
                    transactions.extend(txns)
                    break
            except Exception:
                pass

    cache_file.write_text(json.dumps(transactions))
    return transactions


def parse_form4(xml_text, filing_date, ticker):
    """Parse buy transactions from Form 4 XML."""
    name_m  = re.search(r"<rptOwnerName>([^<]+)</rptOwnerName>", xml_text)
    title_m = re.search(r"<officerTitle>([^<]+)</officerTitle>", xml_text)
    reporter = name_m.group(1).strip() if name_m else "Unknown"
    title    = title_m.group(1).strip() if title_m else ""

    C_SUITE = ["ceo", "cfo", "coo", "cto", "president", "chairman",
               "chief executive", "chief financial", "chief operating"]
    is_csuite = any(kw in title.lower() for kw in C_SUITE)

    blocks = re.findall(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>",
                        xml_text, re.DOTALL)
    txns = []
    for blk in blocks:
        date_m   = re.search(r"<transactionDate>\s*<value>(\d{4}-\d{2}-\d{2})</value>", blk)
        code_m   = re.search(r"<transactionCode>(\w)</transactionCode>", blk)
        shares_m = re.search(r"<transactionShares>\s*<value>([\d.]+)</value>", blk)
        price_m  = re.search(r"<transactionPricePerShare>\s*<value>([\d.]+)</value>", blk)
        if not (date_m and code_m and shares_m):
            continue
        if code_m.group(1) != "P":  # P = open-market purchase
            continue
        shares = float(shares_m.group(1))
        price  = float(price_m.group(1)) if price_m else 0.0
        value  = shares * price
        txns.append({
            "ticker": ticker,
            "date": date_m.group(1),
            "reporter": reporter,
            "title": title,
            "is_csuite": is_csuite,
            "shares": shares,
            "price": price,
            "value": value,
            "filing_date": filing_date,
        })
    return txns


def build_insider_panel(universe, start_date="2019-01-01", end_date="2026-06-30"):
    """Build daily insider signal panel for entire universe."""
    panel_cache = OUTPUT_DIR / "insider_panel.parquet"
    if panel_cache.exists():
        print("Loading cached insider panel...")
        return pd.read_parquet(panel_cache)

    # Check if Jupiter has it
    jupiter_panel = JUPITER_INSIDER_DIR / "insider_transactions_all.parquet"
    if jupiter_panel.exists():
        print("Loading insider panel from Jupiter...")
        df = pd.read_parquet(jupiter_panel)
        df.to_parquet(panel_cache)
        return df

    print(f"Building insider panel for {len(universe)} stocks...")
    cik_map = get_cik_map()
    all_txns = []
    for i, ticker in enumerate(universe):
        cik = cik_map.get(ticker.upper())
        if not cik:
            continue
        txns = fetch_insider_transactions(ticker, cik, start_date, end_date)
        all_txns.extend(txns)
        if (i + 1) % 20 == 0:
            print(f"  Fetched {i+1}/{len(universe)} tickers, {len(all_txns)} transactions so far")

    if not all_txns:
        print("WARNING: No insider transactions found")
        return pd.DataFrame()

    df = pd.DataFrame(all_txns)
    df["date"] = pd.to_datetime(df["date"])
    df.to_parquet(panel_cache)
    print(f"Insider panel built: {len(df)} transactions")
    return df


def build_insider_features(insider_df, tickers, dates):
    """
    For each (ticker, date), compute rolling 30d insider features:
      - ins_buy_count: number of distinct insider buyers in last 30d
      - ins_buy_value: total dollar value of insider purchases
      - ins_csuite_buy: 1 if C-suite purchase >$100K in last 30d
      - ins_cluster: 1 if ≥2 distinct buyers in last 30d
      - ins_signal: composite signal (cluster OR csuite > $100K)
    """
    if insider_df is None or insider_df.empty:
        return pd.DataFrame()

    insider_df = insider_df.copy()
    insider_df["date"] = pd.to_datetime(insider_df["date"])

    rows = []
    for ticker in tickers:
        tkr_df = insider_df[insider_df["ticker"] == ticker].sort_values("date")
        for dt in dates:
            cutoff = pd.Timestamp(dt) - pd.Timedelta(days=INSIDER_LOOKBACK)
            window = tkr_df[(tkr_df["date"] >= cutoff) & (tkr_df["date"] <= pd.Timestamp(dt))]
            buy_count  = window["reporter"].nunique() if not window.empty else 0
            buy_value  = window["value"].sum() if not window.empty else 0
            csuite_buy = int((window["is_csuite"] & (window["value"] >= 100_000)).any()) if not window.empty else 0
            cluster    = int(buy_count >= 2)
            signal     = int(cluster or csuite_buy)
            rows.append({
                "ticker": ticker,
                "date": dt,
                "ins_buy_count": buy_count,
                "ins_buy_value": buy_value,
                "ins_csuite_buy": csuite_buy,
                "ins_cluster": cluster,
                "ins_signal": signal,
            })

    return pd.DataFrame(rows)


# ============================================================
#  PRICE DATA + FEATURES
# ============================================================

def fetch_prices(tickers, start="2018-01-01", end="2026-07-01"):
    cache = CACHE_DIR / f"prices_v4_{len(tickers)}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    print(f"Fetching price data for {len(tickers)} tickers...")
    all_tickers = list(set(tickers + ["SPY"]))
    df = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    df.to_parquet(cache)
    return df


def build_feature_panel(prices, spy_prices, insider_features=None):
    """Build ML feature panel with returns, momentum, relative strength, insider."""
    dates = prices.index
    tickers = [c for c in prices.columns if c != "SPY"]

    rows = []
    spy = spy_prices.reindex(dates).ffill()

    print(f"Building feature panel for {len(tickers)} tickers × {len(dates)} dates...")
    for ticker in tickers:
        px = prices[ticker].reindex(dates).ffill().dropna()
        if len(px) < 252:
            continue

        sector = SECTOR_MAP.get(ticker, "Other")
        ticker_dates = px.index

        # Returns at various horizons
        ret_5d  = px.pct_change(5)
        ret_10d = px.pct_change(10)
        ret_21d = px.pct_change(21)
        ret_63d = px.pct_change(63)
        ret_126d = px.pct_change(126)
        ret_252d = px.pct_change(252)

        spy_r5   = spy.pct_change(5)
        spy_r21  = spy.pct_change(21)
        spy_r63  = spy.pct_change(63)
        spy_r126 = spy.pct_change(126)

        # Relative to SPY
        rel_5  = ret_5d  - spy_r5
        rel_21 = ret_21d - spy_r21
        rel_63 = ret_63d - spy_r63

        # Volatility
        vol_21 = ret_5d.rolling(21).std() * np.sqrt(52)
        vol_63 = ret_5d.rolling(63).std() * np.sqrt(52)
        vol_ratio = vol_21 / vol_63.replace(0, np.nan)

        # Z-score (52-week range)
        roll_max = px.rolling(252).max()
        roll_min = px.rolling(252).min()
        z_52w = (px - roll_min) / (roll_max - roll_min).replace(0, np.nan)

        # RSI
        delta = px.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        dn = (-delta.clip(upper=0)).rolling(14).mean()
        rsi = 100 - 100 / (1 + up / dn.replace(0, np.nan))

        # Spy vs 200MA (regime)
        spy_200ma = spy.rolling(200).mean()
        spy_regime = (spy > spy_200ma).astype(int)

        # Forward target: stock excess return vs SPY in 60d
        fwd_stock = px.pct_change(HOLD_DAYS).shift(-HOLD_DAYS)
        fwd_spy   = spy.pct_change(HOLD_DAYS).shift(-HOLD_DAYS)
        target    = ((fwd_stock - fwd_spy) >= EXCESS_THRESHOLD).astype(int)

        for dt in ticker_dates:
            if dt not in ret_21d.index:
                continue
            try:
                row = {
                    "ticker": ticker,
                    "sector": sector,
                    "date": dt,
                    "ret_5d": ret_5d.get(dt, np.nan),
                    "ret_10d": ret_10d.get(dt, np.nan),
                    "ret_21d": ret_21d.get(dt, np.nan),
                    "ret_63d": ret_63d.get(dt, np.nan),
                    "ret_126d": ret_126d.get(dt, np.nan),
                    "ret_252d": ret_252d.get(dt, np.nan),
                    "rel_5d": rel_5.get(dt, np.nan),
                    "rel_21d": rel_21.get(dt, np.nan),
                    "rel_63d": rel_63.get(dt, np.nan),
                    "vol_21d": vol_21.get(dt, np.nan),
                    "vol_ratio": vol_ratio.get(dt, np.nan),
                    "z_52w": z_52w.get(dt, np.nan),
                    "rsi": rsi.get(dt, np.nan),
                    "spy_ret_21d": spy_r21.get(dt, np.nan),
                    "spy_ret_63d": spy_r63.get(dt, np.nan),
                    "spy_regime": spy_regime.get(dt, 0),
                    "target": target.get(dt, np.nan),
                }
                rows.append(row)
            except Exception:
                pass

    panel = pd.DataFrame(rows)
    panel = panel.dropna(subset=["target"])
    panel["date"] = pd.to_datetime(panel["date"])

    # Merge insider features
    if insider_features is not None and not insider_features.empty:
        insider_features["date"] = pd.to_datetime(insider_features["date"])
        panel = panel.merge(insider_features, on=["ticker", "date"], how="left")
        for col in ["ins_buy_count", "ins_buy_value", "ins_csuite_buy", "ins_cluster", "ins_signal"]:
            if col not in panel.columns:
                panel[col] = 0
        panel[["ins_buy_count", "ins_buy_value", "ins_csuite_buy", "ins_cluster", "ins_signal"]] = \
            panel[["ins_buy_count", "ins_buy_value", "ins_csuite_buy", "ins_cluster", "ins_signal"]].fillna(0)
    else:
        for col in ["ins_buy_count", "ins_buy_value", "ins_csuite_buy", "ins_cluster", "ins_signal"]:
            panel[col] = 0

    return panel


# ============================================================
#  WALK-FORWARD VALIDATION
# ============================================================

FEATURE_COLS_BASE = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "rel_5d", "rel_21d", "rel_63d",
    "vol_21d", "vol_ratio", "z_52w", "rsi",
    "spy_ret_21d", "spy_ret_63d", "spy_regime",
]
FEATURE_COLS_WITH_INSIDER = FEATURE_COLS_BASE + [
    "ins_buy_count", "ins_buy_value", "ins_csuite_buy", "ins_cluster", "ins_signal"
]

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 20,
    "class_weight": "balanced",
    "n_jobs": 4,
    "verbose": -1,
    "random_state": 42,
}

TRAIN_DAYS = 252
STEP_DAYS  = 21
EMBARGO    = 60  # trading days between train end and test start


def walk_forward_eval(panel, feature_cols, model_name="base"):
    """Run walk-forward evaluation with sliding window."""
    panel = panel.sort_values("date").reset_index(drop=True)
    dates = sorted(panel["date"].unique())

    folds = []
    oof_rows = []
    step = 0
    while True:
        train_end_idx = TRAIN_DAYS + step * STEP_DAYS
        if train_end_idx >= len(dates):
            break
        test_start_idx = train_end_idx + EMBARGO
        test_end_idx   = test_start_idx + STEP_DAYS
        if test_end_idx > len(dates):
            break

        train_dates = dates[:train_end_idx]
        test_dates  = dates[test_start_idx:test_end_idx]

        train = panel[panel["date"].isin(train_dates)].dropna(subset=feature_cols + ["target"])
        test  = panel[panel["date"].isin(test_dates)].dropna(subset=feature_cols + ["target"])

        if len(train) < 100 or len(test) < 10:
            step += 1
            continue

        X_tr, y_tr = train[feature_cols].values, train["target"].values
        X_te, y_te = test[feature_cols].values,  test["target"].values

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]

        test_out = test[["ticker", "date", "sector", "target"]].copy()
        test_out["prob"] = probs
        test_out["fold"] = step
        oof_rows.append(test_out)

        base_rate = y_te.mean()
        at_55 = probs >= 0.55
        prec_55 = precision_score(y_te[at_55], (probs >= 0.55)[at_55], zero_division=0) if at_55.sum() > 0 else 0

        folds.append({
            "fold": step,
            "train_end": str(dates[train_end_idx - 1].date()),
            "test_start": str(dates[test_start_idx].date()),
            "n_train": len(train),
            "n_test": len(test),
            "base_rate": round(base_rate, 4),
            "prec_55": round(prec_55, 4),
        })
        step += 1

        if step % 20 == 0:
            print(f"  [{model_name}] Fold {step}: test={dates[test_start_idx].date()} n_test={len(test)}")

    oof = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    return oof, folds


def evaluate_oof(oof, model_name="base"):
    """Compute threshold analysis, R1 regime test, permutation test."""
    if oof.empty:
        return {}

    results = {"model": model_name, "n_total": len(oof), "base_rate": round(oof["target"].mean(), 4)}

    # Threshold analysis
    threshold_res = []
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = oof["prob"] >= thresh
        if mask.sum() < 5:
            continue
        sub = oof[mask]
        prec = sub["target"].mean()
        lift = prec / results["base_rate"] if results["base_rate"] > 0 else 0
        threshold_res.append({
            "threshold": thresh,
            "n_signals": int(mask.sum()),
            "precision": round(prec, 4),
            "lift": round(lift, 4),
        })
    results["threshold_analysis"] = threshold_res

    # R1 Regime test using spy_regime column
    if "spy_regime" in oof.columns:
        regime_res = {}
        for regime_val, regime_name in [(1, "bull"), (0, "bear")]:
            sub = oof[oof["spy_regime"] == regime_val]
            if len(sub) < 20:
                continue
            prec_at_60 = sub[sub["prob"] >= 0.60]["target"].mean() if (sub["prob"] >= 0.60).sum() > 0 else 0
            regime_res[regime_name] = {
                "n": len(sub),
                "precision_60": round(prec_at_60, 4),
                "base_rate": round(sub["target"].mean(), 4),
            }
        if regime_res:
            sharpes = [v["precision_60"] for v in regime_res.values() if v["precision_60"] > 0]
            if len(sharpes) >= 2:
                gap = abs(max(sharpes) - min(sharpes)) / max(abs(max(sharpes)), abs(min(sharpes)))
            else:
                gap = None
            results["regime"] = {
                "per_regime": regime_res,
                "gap": round(gap, 3) if gap is not None else None,
                "r1_pass": bool(gap is not None and gap <= 0.50),
            }

    # Permutation test (shuffle labels 100 times)
    rng = np.random.default_rng(42)
    real_lift = 0
    mask_60 = oof["prob"] >= 0.60
    if mask_60.sum() > 10:
        real_lift = oof[mask_60]["target"].mean() / results["base_rate"]
    null_lifts = []
    for _ in range(100):
        shuffled = rng.permutation(oof["target"].values)
        tmp = oof.copy()
        tmp["target"] = shuffled
        sub = tmp[mask_60]
        null_lift = sub["target"].mean() / results["base_rate"] if len(sub) > 0 and results["base_rate"] > 0 else 1
        null_lifts.append(null_lift)
    p_val = np.mean(np.array(null_lifts) >= real_lift)
    results["permutation"] = {
        "real_lift_60": round(real_lift, 4),
        "null_mean": round(float(np.mean(null_lifts)), 4),
        "p_value": round(float(p_val), 4),
        "significant": bool(p_val < 0.05),
    }

    print(f"\n  [{model_name}] Results:")
    print(f"    Base rate: {results['base_rate']:.1%}")
    for t in threshold_res:
        print(f"    @{t['threshold']:.0%}: prec={t['precision']:.1%} lift={t['lift']:.2f}x n={t['n_signals']}")
    if "regime" in results:
        rg = results["regime"]
        print(f"    Regime gap: {rg.get('gap')} R1={'PASS' if rg.get('r1_pass') else 'FAIL'}")
        for rn, rv in rg.get("per_regime", {}).items():
            print(f"      {rn}: n={rv['n']} prec@60={rv['precision_60']:.1%}")
    print(f"    Permutation: p={results['permutation']['p_value']:.3f} "
          f"({'PASS' if results['permutation']['significant'] else 'FAIL'})")

    return results


# ============================================================
#  FUSION ANALYSIS
# ============================================================

def fusion_analysis(oof_base, oof_with_insider, insider_panel):
    """
    Test: does requiring insider signal when using base model signal improve results?
    Fusion strategy: trade only when base model prob ≥ 0.60 AND ins_signal = 1.
    """
    if oof_base.empty or oof_with_insider.empty:
        return {}

    # Base model @ 60% threshold
    base_60 = oof_base[oof_base["prob"] >= 0.60].copy()
    # With insider signal
    base_60_ins = oof_with_insider[(oof_with_insider["prob"] >= 0.60) &
                                   (oof_with_insider.get("ins_signal", pd.Series(0)) >= 1)].copy() \
        if "ins_signal" in oof_with_insider.columns else pd.DataFrame()

    results = {
        "base_60": {
            "n": len(base_60),
            "precision": round(base_60["target"].mean(), 4) if len(base_60) > 0 else 0,
        },
    }

    if not base_60_ins.empty:
        results["fusion_60_ins"] = {
            "n": len(base_60_ins),
            "precision": round(base_60_ins["target"].mean(), 4),
            "lift_vs_base": round(base_60_ins["target"].mean() / base_60["target"].mean(), 3)
                if len(base_60) > 0 and base_60["target"].mean() > 0 else None,
        }
        print(f"\n  Fusion: base@60 prec={results['base_60']['precision']:.1%} n={results['base_60']['n']}")
        print(f"  Fusion+Insider: prec={results['fusion_60_ins']['precision']:.1%} "
              f"n={results['fusion_60_ins']['n']} "
              f"lift={results['fusion_60_ins']['lift_vs_base']:.2f}x")

    # Insider-only: just use ins_signal = 1 as buy signal
    if "ins_signal" in oof_base.columns:
        ins_only = oof_base[oof_base["ins_signal"] == 1]
        results["insider_only"] = {
            "n": len(ins_only),
            "precision": round(ins_only["target"].mean(), 4) if len(ins_only) > 0 else 0,
            "lift": round(ins_only["target"].mean() / oof_base["target"].mean(), 3)
                if len(ins_only) > 0 and oof_base["target"].mean() > 0 else None,
        }
        print(f"  Insider-only: prec={results['insider_only']['precision']:.1%} "
              f"n={results['insider_only']['n']} lift={results['insider_only']['lift']}")

    return results


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  STOCK PREDICTOR v4 — Insider Buying Fusion (HC #698)")
    print("=" * 70)

    # 1. Fetch prices
    prices = fetch_prices(UNIVERSE)
    if "SPY" not in prices.columns:
        print("ERROR: SPY prices not available")
        sys.exit(1)
    spy_prices = prices["SPY"]
    stock_prices = prices[[t for t in UNIVERSE if t in prices.columns]]
    print(f"Prices: {stock_prices.shape[0]} dates × {stock_prices.shape[1]} stocks")

    # 2. Load/build insider data
    print("\nLoading insider transaction data...")
    insider_df = build_insider_panel(UNIVERSE)

    # 3. Build insider features (sparse — only load unique dates monthly for speed)
    insider_feat = None
    if not insider_df.empty:
        print("Building insider feature panel...")
        # Sample dates monthly to reduce computation (insider signals change slowly)
        all_dates = sorted(stock_prices.index)
        monthly_dates = [d for i, d in enumerate(all_dates) if i % 21 == 0]  # every 21 trading days
        insider_feat = build_insider_features(insider_df, list(stock_prices.columns), monthly_dates)
        print(f"Insider features: {len(insider_feat)} rows")

    # 4. Build feature panel
    print("\nBuilding feature panel...")
    panel = build_feature_panel(stock_prices, spy_prices, insider_feat)
    panel.to_parquet(OUTPUT_DIR / "panel_v4.parquet")
    print(f"Panel: {len(panel)} rows, {panel['ticker'].nunique()} stocks, "
          f"dates: {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print(f"Base rate: {panel['target'].mean():.1%}")
    insider_coverage = panel["ins_signal"].mean() if "ins_signal" in panel.columns else 0
    print(f"Insider signal coverage: {insider_coverage:.1%}")

    # 5. Walk-forward: base model (no insider)
    print("\n--- Model A: Base (relative return features, no insider) ---")
    oof_base, folds_base = walk_forward_eval(panel, FEATURE_COLS_BASE, "base")
    results_base = evaluate_oof(oof_base, "base")

    # 6. Walk-forward: with insider features
    print("\n--- Model B: Fusion (relative return + insider features) ---")
    available_ins_features = [f for f in FEATURE_COLS_WITH_INSIDER if f in panel.columns]
    oof_fused, folds_fused = walk_forward_eval(panel, available_ins_features, "fusion")
    results_fused = evaluate_oof(oof_fused, "fusion")

    # 7. Fusion analysis
    print("\n--- Fusion Analysis ---")
    fusion_res = fusion_analysis(oof_base, oof_fused, insider_df)

    # 8. Save everything
    all_results = {
        "model_a_base": results_base,
        "model_b_fusion": results_fused,
        "fusion_analysis": fusion_res,
        "data_info": {
            "n_stocks": panel["ticker"].nunique(),
            "n_rows": len(panel),
            "date_range": f"{panel['date'].min().date()} -> {panel['date'].max().date()}",
            "base_rate": round(panel["target"].mean(), 4),
            "insider_coverage_pct": round(insider_coverage * 100, 2) if "ins_signal" in panel.columns else 0,
            "insider_transactions": len(insider_df) if not insider_df.empty else 0,
        }
    }

    out_path = OUTPUT_DIR / "results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults saved to {out_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for model_name, res in [("Model A (Base)", results_base), ("Model B (Fusion)", results_fused)]:
        if not res:
            continue
        best = max(res.get("threshold_analysis", [{}]), key=lambda x: x.get("lift", 0), default={})
        print(f"{model_name}:")
        print(f"  Best threshold: {best.get('threshold', '?')} | "
              f"precision={best.get('precision', 0):.1%} | lift={best.get('lift', 0):.2f}x | "
              f"n_signals={best.get('n_signals', 0)}")
        if "regime" in res:
            print(f"  Regime R1: {'PASS' if res['regime'].get('r1_pass') else 'FAIL'} "
                  f"gap={res['regime'].get('gap')}")
        print(f"  Permutation: {'PASS' if res.get('permutation', {}).get('significant') else 'FAIL'} "
              f"p={res.get('permutation', {}).get('p_value', '?')}")

    if fusion_res:
        print(f"\nFusion (Base@60 + Insider required):")
        if "fusion_60_ins" in fusion_res:
            fi = fusion_res["fusion_60_ins"]
            print(f"  precision={fi['precision']:.1%} n={fi['n']} lift={fi.get('lift_vs_base', '?')}x vs base")

    return all_results


if __name__ == "__main__":
    main()
