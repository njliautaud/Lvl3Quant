#!/usr/bin/env python3
"""
Yield Curve Carry / ETF Switching Strategy
==========================================
Signal: Treasury 10Y-2Y (or 10Y-3M) yield spread as a regime indicator.
  - Steep curve (spread > threshold)  → risk-on  (UPRO / QQQ)
  - Flat / inverted (spread < threshold) → risk-off (TLT / GLD)

Optional momentum overlay: 20-day change in spread (steepening vs flattening).

Assets tested: UPRO, SPY, TLT, GLD, SHY

Validation gates (all must pass):
  1. Permutation test: 1000 shuffles, p < 0.05
  2. Sub-period consistency: 3 blocks, CV < 0.50
  3. Outlier robustness: remove top 5 pct returns, degradation < 30 pct
  4. Walk-forward: annual windows, count years beating SPY
  5. R1 regime symmetry: |Sharpe_green - Sharpe_red| / max < 0.50
"""

import json
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/yield_curve_signal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/output/growth_research/yield_curve_signal_results.json")

TICKERS_ETF  = ["UPRO", "SPY", "TLT", "GLD", "SHY", "QQQ"]
TICKERS_YIELD = ["^TNX", "^IRX", "^FVX"]   # 10Y, 13-week, 5Y


# ─── Data ────────────────────────────────────────────────────────────────────

def download_data(start="2010-01-01"):
    print("Downloading ETF prices ...")
    etf_raw  = yf.download(TICKERS_ETF,  start=start, auto_adjust=True, progress=False)
    print("Downloading yield tickers ...")
    yld_raw  = yf.download(TICKERS_YIELD, start=start, auto_adjust=True, progress=False)

    def extract_close(raw, tickers):
        if isinstance(raw.columns, pd.MultiIndex):
            c = raw["Close"]
        else:
            c = raw[["Close"]] if "Close" in raw.columns else raw
        c.columns = [str(x).strip() for x in c.columns]
        return c

    etf  = extract_close(etf_raw,  TICKERS_ETF)
    ylds = extract_close(yld_raw, TICKERS_YIELD)

    # Merge on common dates
    df = etf.join(ylds, how="left")
    df = df.ffill().dropna(subset=["SPY","UPRO","TLT","GLD","SHY"])
    print(f"  Data: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} rows")
    return df


def build_yield_spread(df):
    """
    Primary: ^TNX (10Y) minus ^IRX (13-week, already annualised %).
    Secondary proxy: TLT/SHY price momentum ratio as robustness check.
    """
    df = df.copy()

    # Raw yield spread (in %)
    if "^TNX" in df.columns and "^IRX" in df.columns:
        df["spread_raw"] = df["^TNX"] - df["^IRX"]
    else:
        raise ValueError("Missing yield columns ^TNX / ^IRX")

    # 20-day momentum of spread (steepening = positive, flattening = negative)
    df["spread_mom20"] = df["spread_raw"].diff(20)

    # TLT/SHY log-ratio as a market-implied proxy (robustness)
    df["tlt_shy_ratio"] = np.log(df["TLT"] / df["SHY"])

    return df


# ─── Strategy ────────────────────────────────────────────────────────────────

def compute_signal(df, threshold, use_momentum, freq):
    """
    Returns a Series of daily position weights for risk-on asset (1) or
    risk-off asset (-1-equivalent, handled at allocation level).
    """
    raw   = df["spread_raw"]
    mom   = df["spread_mom20"]

    # Base signal: 1 = steep (risk-on), 0 = flat/inverted (risk-off)
    base = (raw > threshold).astype(float)

    if use_momentum:
        # Confirm only: if steep but flattening rapidly → hold risk-off
        # if flat but steepening rapidly → early risk-on
        flattening = mom < -0.10   # spread falling >10bp/20d
        steepening = mom >  0.10
        # Override: steep+flattening → risk-off
        base = np.where(base == 1,
                        np.where(flattening, 0.0, 1.0),
                        np.where(steepening, 1.0, 0.0))
        base = pd.Series(base, index=df.index)
    else:
        base = base.astype(float)

    # Resample to desired frequency (carry signal is slow-moving)
    if freq == "weekly":
        # Signal only changes on Mondays
        base = base.resample("W-MON").last().reindex(df.index, method="ffill")
    elif freq == "monthly":
        base = base.resample("MS").last().reindex(df.index, method="ffill")

    return base.shift(1)   # No look-ahead: signal from yesterday drives today


def run_strategy(df, signal, risk_on_asset, risk_off_asset):
    """
    Returns daily strategy returns.
    signal = 1 → hold risk_on_asset
    signal = 0 → hold risk_off_asset
    """
    ron = df[risk_on_asset].pct_change()
    rof = df[risk_off_asset].pct_change()
    strat = signal * ron + (1 - signal) * rof
    return strat.dropna()


# ─── Metrics ─────────────────────────────────────────────────────────────────

TRADING_DAYS = 252

def sharpe(rets, rf=0.0):
    rets = rets.dropna()
    if len(rets) < 20 or rets.std() == 0:
        return np.nan
    return (rets.mean() - rf / TRADING_DAYS) / rets.std() * np.sqrt(TRADING_DAYS)

def sortino(rets, rf=0.0):
    rets = rets.dropna()
    down = rets[rets < 0]
    if len(down) < 5 or down.std() == 0:
        return np.nan
    return (rets.mean() - rf / TRADING_DAYS) / down.std() * np.sqrt(TRADING_DAYS)

def max_drawdown(rets):
    rets = rets.dropna()
    cum = (1 + rets).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())

def cagr(rets):
    rets = rets.dropna()
    if len(rets) < 2:
        return np.nan
    total = (1 + rets).prod()
    years = len(rets) / TRADING_DAYS
    return float(total ** (1 / years) - 1)

def calmar(rets):
    c = cagr(rets)
    mdd = max_drawdown(rets)
    if mdd == 0:
        return np.nan
    return c / abs(mdd)

def profit_factor(rets):
    rets = rets.dropna()
    wins   = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    if losses == 0:
        return np.nan
    return float(wins / losses)

def win_rate(rets):
    rets = rets.dropna()
    return float((rets > 0).mean())

def day_concentration(rets):
    """Fraction of total cumulative PnL from the single best day."""
    rets = rets.dropna()
    total = rets.sum()
    if total <= 0:
        return np.nan
    return float(rets.max() / total)

def compute_all_metrics(rets):
    sh = sharpe(rets)
    so = sortino(rets)
    mdd = max_drawdown(rets)
    cg = cagr(rets)
    cl = calmar(rets)
    pf = profit_factor(rets)
    wr = win_rate(rets)
    dc = day_concentration(rets)
    return dict(sharpe=sh, sortino=so, max_dd=mdd, cagr=cg,
                calmar=cl, pf=pf, win_rate=wr, day_conc=dc,
                n_days=int(len(rets)))


# ─── Regime Split (R1) ────────────────────────────────────────────────────────

def regime_split(rets, spy_rets):
    """
    Classify each day by SPY close-to-close direction.
    Returns Sharpe per regime and symmetry verdict.
    """
    aligned = rets.reindex(spy_rets.index).dropna()
    spy_aligned = spy_rets.reindex(aligned.index).dropna()
    common = aligned.index.intersection(spy_aligned.index)
    aligned = aligned.loc[common]
    spy_al  = spy_aligned.loc[common]

    green = aligned[spy_al >  0.0]
    red   = aligned[spy_al <  0.0]
    flat  = aligned[spy_al == 0.0]

    sh_g = sharpe(green)
    sh_r = sharpe(red)
    sh_f = sharpe(flat)

    denom = max(abs(sh_g) if not np.isnan(sh_g) else 0,
                abs(sh_r) if not np.isnan(sh_r) else 0)
    skew = abs(sh_g - sh_r) / denom if denom > 0 else np.nan
    r1_pass = bool(skew <= 0.50) if not np.isnan(skew) else False

    return dict(
        sharpe_green=sh_g, sharpe_red=sh_r, sharpe_flat=sh_f,
        regime_skew=skew, r1_pass=r1_pass,
        n_green=int(len(green)), n_red=int(len(red)), n_flat=int(len(flat))
    )


# ─── Validation Gates ─────────────────────────────────────────────────────────

def permutation_test(rets, n_perms=1000):
    obs_sh = sharpe(rets)
    arr = rets.values.copy()
    beat = 0
    for _ in range(n_perms):
        np.random.shuffle(arr)
        sh_p = sharpe(pd.Series(arr))
        if not np.isnan(sh_p) and sh_p >= obs_sh:
            beat += 1
    p_val = (beat + 1) / (n_perms + 1)
    return dict(obs_sharpe=obs_sh, p_value=p_val, pass_gate=bool(p_val < 0.05))


def sub_period_consistency(rets, n_blocks=3):
    blocks = np.array_split(rets.dropna(), n_blocks)
    sharpes = [sharpe(pd.Series(b)) for b in blocks]
    sharpes = [s for s in sharpes if not np.isnan(s)]
    if len(sharpes) < 2:
        return dict(block_sharpes=sharpes, cv=np.nan, pass_gate=False)
    mean_s = np.mean(sharpes)
    std_s  = np.std(sharpes, ddof=1)
    cv = abs(std_s / mean_s) if mean_s != 0 else np.nan
    return dict(block_sharpes=[round(s,3) for s in sharpes],
                cv=round(float(cv), 3) if not np.isnan(cv) else None,
                pass_gate=bool(not np.isnan(cv) and cv < 0.50 and mean_s > 0))


def outlier_robustness(rets, top_pct=0.05):
    full_sh = sharpe(rets)
    cutoff  = rets.quantile(1 - top_pct)
    trimmed = rets[rets <= cutoff]
    trim_sh = sharpe(trimmed)
    if np.isnan(full_sh) or full_sh == 0:
        return dict(full_sharpe=full_sh, trimmed_sharpe=trim_sh,
                    degradation=np.nan, pass_gate=False)
    deg = (full_sh - trim_sh) / abs(full_sh)
    return dict(full_sharpe=round(full_sh,3), trimmed_sharpe=round(trim_sh,3),
                degradation=round(float(deg),3),
                pass_gate=bool(deg < 0.30))


def walk_forward_annual(rets, spy_rets):
    years = sorted(set(rets.index.year))
    results = []
    years_beat = 0
    for yr in years:
        yr_strat = rets[rets.index.year == yr]
        yr_spy   = spy_rets.reindex(yr_strat.index).dropna()
        yr_strat = yr_strat.reindex(yr_spy.index).dropna()
        if len(yr_strat) < 20:
            continue
        sh_s = sharpe(yr_strat)
        sh_spy = sharpe(yr_spy)
        beat = bool(sh_s > sh_spy)
        if beat:
            years_beat += 1
        results.append(dict(year=yr, strat_sharpe=round(sh_s,3),
                            spy_sharpe=round(sh_spy,3), beat_spy=beat))
    n_years = len(results)
    return dict(annual_results=results, years_beat=years_beat,
                total_years=n_years,
                win_pct=round(years_beat/n_years,3) if n_years else 0,
                pass_gate=bool(years_beat >= max(1, int(n_years*0.60))))


# ─── Config Grid ─────────────────────────────────────────────────────────────

THRESHOLDS     = [0.0, 0.5, 1.0]          # spread threshold in %
FREQS          = ["daily", "weekly", "monthly"]
MOMENTUM_FLAGS = [False, True]
ALLOCATIONS    = [
    ("UPRO", "TLT"),   # levered equity vs long bond
    ("UPRO", "GLD"),   # levered equity vs gold
    ("SPY",  "TLT"),   # vanilla equity vs long bond
    ("QQQ",  "GLD"),   # tech vs gold
]


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_all():
    np.random.seed(42)

    df = download_data(start="2010-01-01")
    df = build_yield_spread(df)

    spy_rets = df["SPY"].pct_change().dropna()

    all_results = []
    best_sharpe = -999
    best_config = None

    total = len(THRESHOLDS) * len(FREQS) * len(MOMENTUM_FLAGS) * len(ALLOCATIONS)
    done  = 0

    for (thresh, freq, use_mom, (ron, rof)) in product(
            THRESHOLDS, FREQS, MOMENTUM_FLAGS, ALLOCATIONS):

        done += 1
        print(f"[{done}/{total}] thresh={thresh} freq={freq} mom={use_mom} {ron}/{rof}")

        try:
            sig  = compute_signal(df, thresh, use_mom, freq)
            rets = run_strategy(df, sig, ron, rof)

            # Align to shared index
            common = rets.index.intersection(spy_rets.index)
            rets_c = rets.loc[common].dropna()
            spy_c  = spy_rets.loc[common].dropna()
            if len(rets_c) < 200:
                continue

            metrics = compute_all_metrics(rets_c)
            reg     = regime_split(rets_c, spy_c)
            perm    = permutation_test(rets_c, n_perms=1000)
            subp    = sub_period_consistency(rets_c)
            outlier = outlier_robustness(rets_c)
            wf      = walk_forward_annual(rets_c, spy_c)

            gates_pass = all([
                perm["pass_gate"],
                subp["pass_gate"],
                outlier["pass_gate"],
                wf["pass_gate"],
                reg["r1_pass"],
            ])

            rec = dict(
                config=dict(threshold=thresh, freq=freq,
                            use_momentum=use_mom,
                            risk_on=ron, risk_off=rof),
                metrics=metrics,
                regime=reg,
                gates=dict(
                    permutation=perm,
                    sub_period=subp,
                    outlier_robust=outlier,
                    walk_forward=wf,
                    r1_regime=dict(r1_pass=reg["r1_pass"],
                                   skew=reg["regime_skew"]),
                    all_pass=gates_pass,
                ),
            )
            all_results.append(rec)

            if not np.isnan(metrics["sharpe"]) and metrics["sharpe"] > best_sharpe:
                best_sharpe = metrics["sharpe"]
                best_config = rec

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Sort by Sharpe descending
    all_results.sort(key=lambda r: r["metrics"].get("sharpe") or -999, reverse=True)

    # Save full results
    def safe_json(o):
        if isinstance(o, float):
            return None if (o != o or o == float("inf") or o == float("-inf")) else round(o, 4)
        if isinstance(o, np.floating):
            v = float(o)
            return None if (v != v or v == float("inf") or v == float("-inf")) else round(v, 4)
        if isinstance(o, (np.integer, np.bool_)):
            return o.item()
        return o

    def recursive_clean(obj):
        if isinstance(obj, dict):
            return {k: recursive_clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [recursive_clean(x) for x in obj]
        return safe_json(obj)

    output = dict(
        summary=dict(
            total_configs=len(all_results),
            configs_all_gates_pass=sum(1 for r in all_results if r["gates"]["all_pass"]),
            best_sharpe=round(best_sharpe, 4) if best_sharpe > -999 else None,
            best_config=recursive_clean(best_config),
        ),
        all_results=recursive_clean(all_results),
    )

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ── Print Summary ──────────────────────────────────────────────────────
    passing = [r for r in all_results if r["gates"]["all_pass"]]
    print(f"\n{'='*70}")
    print(f"YIELD CURVE SIGNAL — RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Configs tested:  {len(all_results)}")
    print(f"All-gates pass:  {len(passing)}")

    if passing:
        print(f"\nTOP 5 CONFIGS (all gates pass, ranked by Sharpe):")
        for i, r in enumerate(passing[:5]):
            c = r["config"]
            m = r["metrics"]
            g = r["regime"]
            print(f"\n  #{i+1}: thresh={c['threshold']} | freq={c['freq']} | "
                  f"mom={c['use_momentum']} | {c['risk_on']}/{c['risk_off']}")
            print(f"       Sharpe {m['sharpe']:.2f}  Sortino {m['sortino']:.2f}  "
                  f"Calmar {m['calmar']:.2f}  MaxDD {m['max_dd']:.1%}")
            print(f"       WR {m['win_rate']:.1%}  PF {m['pf']:.2f}  "
                  f"CAGR {m['cagr']:.1%}  DayConc {m['day_conc']:.1%}")
            print(f"       Regime: G={g['sharpe_green']:.2f} R={g['sharpe_red']:.2f} "
                  f"skew={g['regime_skew']:.2f} R1={'PASS' if g['r1_pass'] else 'FAIL'}")
    else:
        print("\nNo config passed ALL gates. Top 5 by Sharpe (for analysis):")
        for i, r in enumerate(all_results[:5]):
            c = r["config"]
            m = r["metrics"]
            g = r["regime"]
            gates = r["gates"]
            failed = []
            if not gates["permutation"]["pass_gate"]:   failed.append("perm")
            if not gates["sub_period"]["pass_gate"]:     failed.append("subperiod")
            if not gates["outlier_robust"]["pass_gate"]: failed.append("outlier")
            if not gates["walk_forward"]["pass_gate"]:   failed.append("walkfwd")
            if not gates["r1_regime"]["r1_pass"]:        failed.append("R1")
            print(f"\n  #{i+1}: thresh={c['threshold']} | freq={c['freq']} | "
                  f"mom={c['use_momentum']} | {c['risk_on']}/{c['risk_off']}")
            print(f"       Sharpe {m['sharpe']:.2f}  Sortino {m['sortino']:.2f}  "
                  f"MaxDD {m['max_dd']:.1%}  WR {m['win_rate']:.1%}")
            print(f"       Failed gates: {', '.join(failed)}")

    if best_config:
        print(f"\n{'='*70}")
        print(f"BEST CONFIG: {best_config['config']}")
        m = best_config["metrics"]
        print(f"  Sharpe {m['sharpe']:.2f}  Sortino {m['sortino']:.2f}  "
              f"Calmar {m['calmar']:.2f}")
        print(f"  MaxDD {m['max_dd']:.1%}  CAGR {m['cagr']:.1%}")
        print(f"  WR {m['win_rate']:.1%}  PF {m['pf']:.2f}  DayConc {m['day_conc']:.1%}")

    return output


if __name__ == "__main__":
    run_all()
