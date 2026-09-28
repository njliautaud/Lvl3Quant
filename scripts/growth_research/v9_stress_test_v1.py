#!/usr/bin/env python3
"""
V9 Stress Test — How Robust Is V9 Under Adverse Conditions?
=============================================================

Tests V9 core (adaptive width + cost/width filter + real pricing) under:
  A: V9 core normal (baseline)
  B: 3x commission ($7.80 instead of $2.60)
  C: 25% additional haircut on entry cost
  D: Max position reduced to 20% of equity (from 40%)
  E: Remove cost/width filter (just adaptive width alone)
  F: Remove top 3 most profitable tickers from universe
  G: Monte Carlo bootstrap (1000 resamples for CI)

Goal: V9 must survive stress — Sharpe > 1.5 under 3x commission to be truly robust.

Output: output/growth_research/v9_stress_test_v1/
MLflow experiment: v9_stress_test_v1
"""

import json
import sys
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

sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread, price_bear_put_spread, COMMISSION_RT_SPREAD,
)
from research.tools.adversarial_validator import validate_trades

BASE = Path("/home/jupiter/Lvl3Quant")
CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
OUTPUT_DIR = BASE / "output" / "growth_research" / "v9_stress_test_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
TOP_K = 3
REGIME_BULL_THRESHOLD = 0.4
DTE = 14
OTM_PCT = 0.02
REBAL_FREQ = "W-FRI"
WF_TRAIN_PERIODS = 12
COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03
MIN_BID = 0.05
MIN_SPREAD_WIDTH = 3.0

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v9_stress_test_v1"

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
except Exception:
    pass

V6_FEATURES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel",
    "pct_pos_months_12m", "sortino_63d", "calmar_1y", "up_capture",
    "trend_r2_63d", "trend_slope_63d",
    "sector_spy_beta_63d", "cross_sector_dispersion",
]


# ═══ Reused functions (compact) ═══

def load_all_chains():
    chains = {}
    for tk in SECTORS:
        path = CHAINS_DIR / f"{tk}.parquet"
        if not path.exists(): continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        for c in ["strike", "bid", "ask", "mid", "vol", "delta"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df["dte"] = (df["expiration"] - df["date"]).dt.days
        chains[tk] = df
    return chains

def find_chain_spread_price(chain_df, trade_date, direction, K1, K2, dte_target):
    if chain_df is None: return None
    day_mask = chain_df["date"] == pd.Timestamp(trade_date)
    chain_day = chain_df[day_mask]
    if chain_day.empty:
        nearby = chain_df[(chain_df["date"] >= pd.Timestamp(trade_date) - pd.Timedelta(days=2)) &
                          (chain_df["date"] <= pd.Timestamp(trade_date) + pd.Timedelta(days=2))]
        if nearby.empty: return None
        nearest_date = min(nearby["date"].unique(), key=lambda x: abs((x - pd.Timestamp(trade_date)).days))
        chain_day = chain_df[chain_df["date"] == nearest_date]
    exps = chain_day[["expiration", "dte"]].drop_duplicates()
    exps["dte_dist"] = (exps["dte"] - dte_target).abs()
    valid_exps = exps[exps["dte_dist"] <= DTE_TOLERANCE]
    if valid_exps.empty: return None
    chain_exp = chain_day[chain_day["expiration"] == valid_exps.loc[valid_exps["dte_dist"].idxmin()]["expiration"]]
    opt_type = "c" if direction == "bull" else "p"
    near_target = K1 if direction == "bull" else K2
    far_target = K2 if direction == "bull" else K1
    near_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    if near_opts.empty: return None
    near_opts["dist"] = (near_opts["strike"] - near_target).abs()
    near_leg = near_opts.sort_values("dist").iloc[0]
    if near_leg["dist"] / max(near_target, 1) > STRIKE_TOLERANCE: return None
    far_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    far_opts["dist"] = (far_opts["strike"] - far_target).abs()
    far_leg = far_opts.sort_values("dist").iloc[0]
    if far_leg["dist"] / max(far_target, 1) > STRIKE_TOLERANCE: return None
    near_mid = float(near_leg["mid"]) if not pd.isna(near_leg["mid"]) else (float(near_leg["bid"]) + float(near_leg["ask"])) / 2
    far_mid = float(far_leg["mid"]) if not pd.isna(far_leg["mid"]) else (float(far_leg["bid"]) + float(far_leg["ask"])) / 2
    return {"found": True, "spread_cost_mid": abs(near_mid - far_mid),
            "near_strike": float(near_leg["strike"]), "far_strike": float(far_leg["strike"])}

def download_data():
    import yfinance as yf
    raw = yf.download(SECTORS + EXTRA_TICKERS, start="2008-01-01", progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw[["Close"]]
    high = raw["High"] if mi else raw[["High"]]
    low = raw["Low"] if mi else raw[["Low"]]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)
    close = close.ffill(); high = high.ffill(); low = low.ffill()
    rename = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    return close.rename(columns=rename), high.rename(columns=rename), low.rename(columns=rename)

def load_regime():
    if not REGIME_FILE.exists(): return None
    data = np.load(REGIME_FILE, allow_pickle=True)
    s = pd.Series(data["regime_scores"], index=pd.to_datetime(data["dates"]))
    return s[~s.index.duplicated(keep="last")]

def get_regime(rs, dt):
    if rs is None: return 0.5
    if dt in rs.index: return float(rs.loc[dt])
    nearest = rs.index[rs.index.get_indexer([dt], method="ffill")]
    return float(rs.loc[nearest[0]]) if len(nearest) > 0 else 0.5

def compute_features(px, spy_slice):
    if len(px) < 260: return None
    f = {}
    for lb, nm in [(5,"ret_5d"),(10,"ret_10d"),(21,"ret_21d"),(63,"ret_63d"),(126,"ret_126d"),(252,"ret_252d")]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f["sharpe_63d"] = float(rets.iloc[-63:].mean() / (rets.iloc[-63:].std() + 1e-10) * np.sqrt(252)) if len(rets) > 63 else 0.0
    f["pct_52w_high"] = float(px.iloc[-1] / px.iloc[-252:].max())
    f["mom_accel"] = f["ret_21d"] - f["ret_63d"] / 3
    monthly = rets.resample("ME").sum()
    f["pct_pos_months_12m"] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = rets.iloc[-63:][rets.iloc[-63:] < 0]
    f["sortino_63d"] = float(rets.iloc[-63:].mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f["calmar_1y"] = cagr / (abs(mdd) + 1e-10)
    up_days = rets[rets > 0]
    f["up_capture"] = float(up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)) if len(up_days) > 10 else 1.0
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10); x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f["trend_r2_63d"] = r_val ** 2; f["trend_slope_63d"] = slope * 252
    else:
        f["trend_r2_63d"] = 0.0; f["trend_slope_63d"] = 0.0
    return f

def compute_cross_features(tk, idx, close):
    f = {}
    spy = close["SPY"].iloc[:idx+1].dropna()
    sec_px = close[tk].iloc[:idx+1].dropna() if tk in close.columns else None
    if spy is None or len(spy) < 63:
        return {"sector_spy_beta_63d": 1.0, "cross_sector_dispersion": 0.01}
    spy_ret = spy.pct_change().dropna()
    if sec_px is not None and len(sec_px) > 63:
        sec_ret = sec_px.pct_change().dropna()
        common = spy_ret.index.intersection(sec_ret.index)
        if len(common) > 63:
            cov = np.cov(sec_ret.loc[common].iloc[-63:].values, spy_ret.loc[common].iloc[-63:].values)
            f["sector_spy_beta_63d"] = float(cov[0,1] / (cov[1,1] + 1e-10))
        else: f["sector_spy_beta_63d"] = 1.0
    else: f["sector_spy_beta_63d"] = 1.0
    sc = [c for c in SECTORS if c in close.columns]
    if len(sc) > 3:
        disp = close[sc].iloc[:idx+1].pct_change().std(axis=1)
        f["cross_sector_dispersion"] = float(disp.rolling(21).mean().iloc[-1]) if len(disp)>21 else 0.01
    else: f["cross_sector_dispersion"] = 0.01
    return f

def compute_atr_series(high, low, close, period=14):
    atr = {}
    for tk in SECTORS:
        if tk in high.columns and tk in low.columns and tk in close.columns:
            h, l, c = high[tk].dropna(), low[tk].dropna(), close[tk].dropna()
            com = h.index.intersection(l.index).intersection(c.index)
            if len(com) > period:
                tr = pd.concat([h.loc[com]-l.loc[com], (h.loc[com]-c.loc[com].shift(1)).abs(),
                                (l.loc[com]-c.loc[com].shift(1)).abs()], axis=1).max(axis=1)
                atr[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr

def build_records(close, rebal_dates, regime):
    records = []
    spy = close["SPY"]
    for dt in rebal_dates:
        idx = close.index.get_indexer([dt], method="ffill")[0]
        if idx < 260: continue
        if get_regime(regime, dt) <= REGIME_BULL_THRESHOLD: continue
        for tk in [c for c in SECTORS if c in close.columns]:
            px = close[tk].iloc[:idx+1].dropna()
            legacy = compute_features(px, spy.iloc[:idx+1])
            if not legacy: continue
            cross = compute_cross_features(tk, idx, close)
            fi = min(idx + DTE, len(close) - 1)
            if fi <= idx: continue
            fwd = float(close[tk].iloc[fi] / close[tk].iloc[idx] - 1)
            records.append({**legacy, **cross, "date": dt, "ticker": tk, "fwd_ret": fwd})
    df = pd.DataFrame(records)
    for c in V6_FEATURES:
        if c not in df.columns: df[c] = 0.0
    df[V6_FEATURES] = df[V6_FEATURES].fillna(0.0)
    return df

def wf_lgbm_rank(df):
    import lightgbm as lgb
    if len(df) < 100: return {}
    df["rank_label"] = df.groupby("date")["fwd_ret"].rank(pct=True)
    dates = sorted(df["date"].unique())
    rankings = {}
    for i in range(WF_TRAIN_PERIODS, len(dates)):
        td = dates[max(0, i-WF_TRAIN_PERIODS):i]
        test_date = dates[i]
        train = df[df["date"].isin(td)]; test = df[df["date"]==test_date].copy()
        if len(test) < 3 or len(train) < 50: continue
        try:
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
            m.fit(np.nan_to_num(train[V6_FEATURES].values.astype(np.float32)),
                  train["rank_label"].values.astype(np.float32))
            test["score"] = m.predict(np.nan_to_num(test[V6_FEATURES].values.astype(np.float32)))
            rankings[test_date] = dict(zip(test["ticker"], test["score"]))
        except: continue
    return rankings

def compute_strikes(S, direction):
    if direction == "bull":
        K1 = round(S * (1 + OTM_PCT), 2)
        K2 = round(K1 + max(MIN_SPREAD_WIDTH, K1 * 0.03), 2)
    else:
        K2 = round(S * (1 - OTM_PCT), 2)
        K1 = round(K2 - max(MIN_SPREAD_WIDTH, K2 * 0.03), 2)
    if K2 <= K1: K2 = K1 + 0.50
    return K1, K2


def execute_trade(tk, dt, direction, close, atr_dict, vix, equity, chains,
                  max_pos, commission, haircut_mult, max_pos_pct, cost_filter):
    if tk not in close.columns or tk not in atr_dict: return None
    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di: return None
    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    K1, K2 = compute_strikes(S, direction)

    used_real = False; entry_cost_ps = None
    chain_df = chains.get(tk)
    if chain_df is not None:
        r = find_chain_spread_price(chain_df, dt, direction, K1, K2, DTE)
        if r and r["found"]:
            entry_cost_ps = r["spread_cost_mid"] * haircut_mult
            used_real = True
            if direction == "bull": K1, K2 = r["near_strike"], r["far_strike"]
            else: K1, K2 = r["far_strike"], r["near_strike"]

    if entry_cost_ps is None:
        try:
            if direction == "bull":
                entry_cost_ps, _ = price_bull_call_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix)
            else:
                entry_cost_ps, _ = price_bear_put_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix)
        except: return None

    if entry_cost_ps is None or entry_cost_ps <= 0: return None

    sw = abs(K2 - K1)
    if cost_filter and entry_cost_ps / max(sw, 0.01) > COST_WIDTH_MAX: return None

    total_cost = entry_cost_ps * 100 + commission
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * max_pos_pct: return None

    Se = float(close[tk].iloc[ei])
    if direction == "bull": intrinsic = max(Se-K1,0) - max(Se-K2,0)
    else: intrinsic = max(K2-Se,0) - max(K1-Se,0)
    pnl = (intrinsic - entry_cost_ps) * 100 - commission

    return {"pnl": round(pnl,2), "entry_cost_ps": round(entry_cost_ps,4),
            "used_real_pricing": used_real, "K1": K1, "K2": K2,
            "spread_width": round(sw,2)}


def simulate(rankings, close, atr_dict, chains, commission, haircut_mult,
             max_pos_pct, cost_filter, sector_universe=None):
    spy = close["SPY"]; vix = close.get("VIX")
    equity = CAP; trades = []; real_c = 0; bs_c = 0

    for dt in sorted(rankings.keys()):
        if dt not in spy.index: continue
        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0
        scores = {k: v for k, v in rankings[dt].items()
                  if sector_universe is None or k in sector_universe}
        if not scores: continue

        trade_mode = "pairs" if cv < 20.0 else "bull_only"
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bulls = [t for t,_ in ranked[:TOP_K]]
        bears = [t for t,_ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        mp = min(100, equity/6) if trade_mode == "pairs" else min(200, equity/3)
        if mp < 30: continue

        for d, picks in [("bull", bulls), ("bear", bears)]:
            for tk in picks:
                r = execute_trade(tk, dt, d, close, atr_dict, cv, equity, chains,
                                  mp, commission, haircut_mult, max_pos_pct, cost_filter)
                if r:
                    equity += r["pnl"]
                    real_c += 1 if r["used_real_pricing"] else 0
                    bs_c += 0 if r["used_real_pricing"] else 1
                    di = close.index.get_loc(dt); ei = min(di+DTE, len(close)-1)
                    sv = float(spy.loc[dt]); se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    trades.append({**r, "entry_date": str(dt.date()),
                                   "exit_date": str(close.index[ei].date()),
                                   "ticker": tk, "regime": "bull" if se>=sv else "bear",
                                   "direction": d, "win": r["pnl"]>0})
    return trades, equity, real_c, bs_c


def chain_only(trades):
    ct = [t for t in trades if t["entry_date"] >= "2019"]
    if not ct: return None
    pnls = [t["pnl"] for t in ct]
    return {"trades": len(ct), "wr": sum(1 for p in pnls if p>0)/len(pnls),
            "sharpe": float(np.mean(pnls)/(np.std(pnls)+1e-10)*np.sqrt(52)),
            "total_pnl": sum(pnls)}


def monte_carlo_ci(trades, n_boot=1000):
    """Bootstrap confidence interval on Sharpe."""
    pnls = np.array([t["pnl"] for t in trades])
    sharpes = []
    np.random.seed(42)
    for _ in range(n_boot):
        sample = np.random.choice(pnls, size=len(pnls), replace=True)
        sh = float(np.mean(sample) / (np.std(sample) + 1e-10) * np.sqrt(52))
        sharpes.append(sh)
    return {
        "mean": float(np.mean(sharpes)),
        "std": float(np.std(sharpes)),
        "ci_95": [float(np.percentile(sharpes, 2.5)), float(np.percentile(sharpes, 97.5))],
        "ci_99": [float(np.percentile(sharpes, 0.5)), float(np.percentile(sharpes, 99.5))],
        "p_positive": float(np.mean(np.array(sharpes) > 0)),
    }


def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"V9 STRESS TEST v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)

    chains = load_all_chains()
    fprint(f"Chains: {len(chains)} sectors")
    close, high, low = download_data()
    fprint(f"Data: {len(close)} days")
    regime = load_regime()
    atr_dict = compute_atr_series(high, low, close)

    rebal = pd.DatetimeIndex(close.index.to_series().resample(REBAL_FREQ).last().dropna().values)
    fprint(f"Rebalance dates: {len(rebal)}")
    records = build_records(close, rebal, regime)
    rankings = wf_lgbm_rank(records)
    fprint(f"Rankings: {len(rankings)} dates")

    if not rankings:
        fprint("ERROR: No rankings"); return

    spy_close = close["SPY"]

    # ═══ STRESS VARIANTS ═══
    STRESS = {
        "A_v9_normal": {
            "commission": COMMISSION_RT_SPREAD, "haircut": 1.0,
            "max_pos_pct": 0.40, "cost_filter": True, "universe": None,
            "desc": "V9 core normal (baseline)",
        },
        "B_3x_commission": {
            "commission": COMMISSION_RT_SPREAD * 3, "haircut": 1.0,
            "max_pos_pct": 0.40, "cost_filter": True, "universe": None,
            "desc": "3x commission ($7.80 RT)",
        },
        "C_25pct_haircut": {
            "commission": COMMISSION_RT_SPREAD, "haircut": 1.25,
            "max_pos_pct": 0.40, "cost_filter": True, "universe": None,
            "desc": "25% additional haircut on entry cost",
        },
        "D_20pct_max_pos": {
            "commission": COMMISSION_RT_SPREAD, "haircut": 1.0,
            "max_pos_pct": 0.20, "cost_filter": True, "universe": None,
            "desc": "Max position 20% of equity (from 40%)",
        },
        "E_no_cost_filter": {
            "commission": COMMISSION_RT_SPREAD, "haircut": 1.0,
            "max_pos_pct": 0.40, "cost_filter": False, "universe": None,
            "desc": "No cost/width filter (adaptive width alone)",
        },
        "F_remove_top3": {
            "commission": COMMISSION_RT_SPREAD, "haircut": 1.0,
            "max_pos_pct": 0.40, "cost_filter": True,
            "universe": set(SECTORS) - {"XLK", "XLY", "XLE"},  # remove most profitable
            "desc": "Remove top 3 tickers (XLK, XLY, XLE)",
        },
    }

    all_results = {}
    all_trades = {}

    for vname, vcfg in STRESS.items():
        fprint(f"\n{'~' * 80}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"{'~' * 80}")

        trades, final_eq, real_c, bs_c = simulate(
            rankings, close, atr_dict, chains,
            vcfg["commission"], vcfg["haircut"], vcfg["max_pos_pct"],
            vcfg["cost_filter"], vcfg["universe"]
        )
        if not trades or len(trades) < 10:
            fprint(f"  Only {len(trades) if trades else 0} trades"); continue

        all_trades[vname] = trades
        total = real_c + bs_c
        fprint(f"  Trades: {len(trades)} | Real: {real_c} ({real_c/total*100:.0f}%) | Final: ${final_eq:,.0f}")

        result = validate_trades(trades, initial_capital=CAP, spy_prices=spy_close, strategy_name=vname)
        result.print_summary()

        co = chain_only(trades)
        rd = result.to_dict()
        all_results[vname] = {**rd, "chain_only": co, "real_pct": round(real_c/max(total,1)*100, 1)}

    # ═══ MONTE CARLO ═══
    fprint(f"\n{'=' * 80}")
    fprint("MONTE CARLO BOOTSTRAP (1000 resamples)")
    fprint(f"{'=' * 80}")

    normal_trades = all_trades.get("A_v9_normal", [])
    if normal_trades:
        mc_full = monte_carlo_ci(normal_trades, n_boot=1000)
        fprint(f"\n  FULL PERIOD:")
        fprint(f"    Mean Sharpe: {mc_full['mean']:.2f} ± {mc_full['std']:.2f}")
        fprint(f"    95% CI: [{mc_full['ci_95'][0]:.2f}, {mc_full['ci_95'][1]:.2f}]")
        fprint(f"    99% CI: [{mc_full['ci_99'][0]:.2f}, {mc_full['ci_99'][1]:.2f}]")
        fprint(f"    P(Sharpe > 0): {mc_full['p_positive']:.1%}")

        chain_trades = [t for t in normal_trades if t["entry_date"] >= "2019"]
        if chain_trades:
            mc_chain = monte_carlo_ci(chain_trades, n_boot=1000)
            fprint(f"\n  CHAIN-ONLY 2019+:")
            fprint(f"    Mean Sharpe: {mc_chain['mean']:.2f} ± {mc_chain['std']:.2f}")
            fprint(f"    95% CI: [{mc_chain['ci_95'][0]:.2f}, {mc_chain['ci_95'][1]:.2f}]")
            fprint(f"    P(Sharpe > 0): {mc_chain['p_positive']:.1%}")
        else:
            mc_chain = None

        all_results["monte_carlo_full"] = mc_full
        if mc_chain:
            all_results["monte_carlo_chain"] = mc_chain

    # ═══ COMPARISON ═══
    fprint(f"\n{'=' * 120}")
    fprint("STRESS TEST COMPARISON")
    fprint(f"{'=' * 120}")
    fprint(f"  {'Variant':<25} {'N':>5} {'Sharpe':>7} {'Sort':>7} {'WR':>6} "
           f"{'PF':>6} {'MDD':>7} {'Gate':>5} {'Final$':>9}")
    fprint(f"  {'-' * 85}")
    for vn in STRESS:
        r = all_results.get(vn)
        if not r: continue
        fprint(f"  {vn:<25} {r['n_trades']:>5} {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
               f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['max_dd']*100:>6.1f}% {r['gates_passed']}/{r['gates_total']} "
               f"${r['final_equity']:>8,.0f}")

    fprint(f"\n{'=' * 80}")
    fprint("CHAIN-ONLY 2019+")
    fprint(f"{'=' * 80}")
    fprint(f"  {'Variant':<25} {'N':>5} {'Sharpe':>8} {'WR':>6} {'PnL':>9}")
    fprint(f"  {'-' * 55}")
    for vn in STRESS:
        r = all_results.get(vn)
        if not r or not r.get("chain_only"): continue
        co = r["chain_only"]
        fprint(f"  {vn:<25} {co['trades']:>5} {co['sharpe']:>8.2f} {co['wr']*100:>5.1f}% ${co['total_pnl']:>8.0f}")

    # ═══ VERDICT ═══
    fprint(f"\n{'=' * 100}")
    fprint("STRESS TEST VERDICT")
    fprint(f"{'=' * 100}")

    baseline = all_results.get("A_v9_normal", {})
    three_x = all_results.get("B_3x_commission", {})
    if baseline and three_x:
        base_sh = baseline.get("sharpe", 0)
        stress_sh = three_x.get("sharpe", 0)
        fprint(f"\n  Normal Sharpe:      {base_sh:.2f}")
        fprint(f"  3x Commission:      {stress_sh:.2f}")
        fprint(f"  Survival ratio:     {stress_sh/max(base_sh,0.01):.1%}")
        if stress_sh >= 1.5:
            fprint(f"  ✅ ROBUST: Survives 3x commission (Sharpe {stress_sh:.2f} >= 1.5)")
        elif stress_sh >= 1.0:
            fprint(f"  ⚠️ MARGINAL: Sharpe {stress_sh:.2f} under 3x commission")
        else:
            fprint(f"  ❌ FRAGILE: Sharpe {stress_sh:.2f} under 3x commission")

    # All variants above 1.0?
    all_above_1 = all(all_results.get(vn, {}).get("sharpe", 0) >= 1.0 for vn in STRESS if vn in all_results)
    fprint(f"\n  All variants Sharpe > 1.0: {'✅ YES' if all_above_1 else '❌ NO'}")

    # ═══ Save ═══
    results_file = OUTPUT_DIR / "results.json"
    with open(results_file, "w") as f:
        ser = {}
        for k, v in all_results.items():
            sv = {}
            for k2, v2 in v.items():
                if isinstance(v2, (np.floating, np.integer)): sv[k2] = float(v2)
                elif isinstance(v2, (dict, list)): sv[k2] = v2
                else: sv[k2] = v2
            ser[k] = sv
        json.dump(ser, f, indent=2, default=str)

    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name=f"stress_{t0.strftime('%Y%m%d_%H%M')}"):
                for vn, r in all_results.items():
                    if isinstance(r, dict) and "sharpe" in r:
                        mlflow.log_metric(f"{vn}_sharpe", r["sharpe"])
                        mlflow.log_metric(f"{vn}_wr", r.get("win_rate", 0))
                if mc_full:
                    mlflow.log_metric("mc_mean_sharpe", mc_full["mean"])
                    mlflow.log_metric("mc_ci95_low", mc_full["ci_95"][0])
                    mlflow.log_metric("mc_ci95_high", mc_full["ci_95"][1])
                mlflow.log_artifact(str(results_file))
        except Exception as e:
            fprint(f"MLflow: {e}")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    fprint(f"\n{'=' * 100}")
    fprint("DONE — V9 Stress Test v1")
    fprint(f"{'=' * 100}")


if __name__ == "__main__":
    main()
