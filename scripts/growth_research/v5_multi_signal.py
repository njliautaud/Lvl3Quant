#!/usr/bin/env python3
"""
Gameplan v5 — Multi-Signal Stack.
Layer ALL validated research onto v4.4 base:
  1. v4.4 core (VIX percentile + adaptive confluence) — base signal
  2. VIX term structure (contango/backwardation) — kill-switch overlay
  3. Credit spread (HYG/IEF ratio) — early warning overlay
  4. Gold momentum — risk-off confirmation
  5. Dollar strength (UUP) — headwind signal
  6. Bond trend (TLT slope) — flight-to-safety detector

CRITICAL: Next-day execution (signal T → return T+1). No look-ahead.
CRITICAL: Fixed $100K capital, no DCA (HC #713).
CRITICAL: Permutation test before reporting (HC #659).
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def load_data():
    import yfinance as yf
    tickers = ["SPY", "UPRO", "GLD", "TLT", "^VIX", "SHY",
               "HYG", "IEF", "UUP",       # credit, safe bonds, dollar
               "^VIX3M",                    # VIX 3-month (for term structure)
               ]
    data = yf.download(tickers, start="2010-01-01", auto_adjust=True,
                       threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data
    if hasattr(closes.columns, "droplevel"):
        try:
            closes.columns = closes.columns.droplevel(1)
        except Exception:
            pass
    closes = closes.rename(columns={"^VIX": "VIX", "^VIX3M": "VIX3M"})
    closes = closes.dropna(subset=["SPY", "UPRO"]).ffill()
    return closes


def compute_signals(closes):
    spy = closes["SPY"]
    vix = closes["VIX"]
    spy_ret = spy.pct_change()

    sig = {}
    # === v4.4 CORE SIGNALS ===
    sig['mom_5d'] = spy.pct_change(5)
    delta = spy_ret.copy()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    sig['rsi_10'] = 100 - (100 / (1 + rs))
    sig['sma_20'] = spy.rolling(20).mean()
    sig['sma_50'] = spy.rolling(50).mean()
    sig['sma_200'] = spy.rolling(200).mean()
    sig['sma_200_slope'] = sig['sma_200'].pct_change(20)
    sig['vol_21d'] = spy_ret.rolling(21).std() * np.sqrt(252) * 100
    sig['vol_63d'] = spy_ret.rolling(63).std() * np.sqrt(252) * 100
    sig['vol_63d_trend'] = sig['vol_63d'] - sig['vol_63d'].rolling(21).mean()
    sig['vix_pctile_63'] = vix.rolling(63).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100, raw=False)
    sig['vix'] = vix
    sig['vix_ma10'] = vix.rolling(10).mean()
    sig['vix_peak20'] = vix.rolling(20).max()

    # === NEW OVERLAY SIGNALS ===

    # 1. VIX term structure: VIX / VIX3M ratio
    #    < 1.0 = contango (normal, risk-on)
    #    > 1.0 = backwardation (fear, risk-off)
    if "VIX3M" in closes.columns:
        vix3m = closes["VIX3M"]
        sig['vix_term'] = vix / vix3m.replace(0, np.nan)
        sig['vix_term_ma5'] = sig['vix_term'].rolling(5).mean()
    else:
        sig['vix_term'] = pd.Series(0.9, index=spy.index)
        sig['vix_term_ma5'] = pd.Series(0.9, index=spy.index)

    # 2. Credit spread: HYG/IEF ratio (high yield vs investment grade)
    #    Falling = credit stress, risk-off signal
    if "HYG" in closes.columns and "IEF" in closes.columns:
        hyg = closes["HYG"]
        ief = closes["IEF"]
        credit_ratio = hyg / ief.replace(0, np.nan)
        sig['credit_ratio'] = credit_ratio
        sig['credit_ma20'] = credit_ratio.rolling(20).mean()
        sig['credit_slope'] = credit_ratio.pct_change(10)  # 10d momentum
    else:
        sig['credit_ratio'] = pd.Series(1.0, index=spy.index)
        sig['credit_ma20'] = pd.Series(1.0, index=spy.index)
        sig['credit_slope'] = pd.Series(0.0, index=spy.index)

    # 3. Gold momentum — risk-off confirmation
    if "GLD" in closes.columns:
        gld = closes["GLD"]
        sig['gld_mom_20'] = gld.pct_change(20)
        sig['gld_sma50'] = gld.rolling(50).mean()
        sig['gld_above_sma50'] = (gld > sig['gld_sma50']).astype(float)
    else:
        sig['gld_mom_20'] = pd.Series(0.0, index=spy.index)
        sig['gld_above_sma50'] = pd.Series(0.0, index=spy.index)

    # 4. Dollar strength (UUP proxy)
    if "UUP" in closes.columns:
        uup = closes["UUP"]
        sig['uup_mom_20'] = uup.pct_change(20)
        sig['uup_sma50'] = uup.rolling(50).mean()
        sig['uup_rising'] = (uup > sig['uup_sma50']).astype(float)
    else:
        sig['uup_mom_20'] = pd.Series(0.0, index=spy.index)
        sig['uup_rising'] = pd.Series(0.0, index=spy.index)

    # 5. Bond trend (TLT) — flight to safety
    if "TLT" in closes.columns:
        tlt = closes["TLT"]
        sig['tlt_mom_20'] = tlt.pct_change(20)
        sig['tlt_sma50'] = tlt.rolling(50).mean()
        sig['tlt_above_sma50'] = (tlt > sig['tlt_sma50']).astype(float)
    else:
        sig['tlt_mom_20'] = pd.Series(0.0, index=spy.index)
        sig['tlt_above_sma50'] = pd.Series(0.0, index=spy.index)

    return sig


def confluence_score(sig, i):
    """Original v4.4 confluence (6 factors, 0-3)."""
    s = 0.0
    m = sig['mom_5d'].iloc[i]
    r = sig['rsi_10'].iloc[i]
    s20 = sig['sma_20'].iloc[i]
    s50 = sig['sma_50'].iloc[i]
    v21 = sig['vol_21d'].iloc[i]
    slope = sig['sma_200_slope'].iloc[i]
    vt = sig['vol_63d_trend'].iloc[i]
    if not np.isnan(m) and m > 0: s += 0.5
    if not np.isnan(r) and r > 50: s += 0.5
    if not np.isnan(s20) and not np.isnan(s50) and s20 > s50: s += 0.5
    if not np.isnan(v21) and v21 < 15: s += 0.5
    if not np.isnan(slope) and slope > 0: s += 0.5
    if not np.isnan(vt) and vt < 0: s += 0.5
    return s


def risk_overlay_score(sig, i):
    """
    Multi-signal risk overlay: counts how many cross-asset signals say 'risk-off'.
    Each flag adds 1 point. Higher = more danger.
    0-1 = green (stay in risk)
    2   = yellow (tighten thresholds)
    3+  = red (exit to defensive)
    """
    score = 0

    # VIX term structure in backwardation (5d avg)
    vt = sig['vix_term_ma5'].iloc[i]
    if not np.isnan(vt) and vt > 1.0:
        score += 1

    # Credit deteriorating (HYG/IEF below 20d MA AND falling)
    cr = sig['credit_ratio'].iloc[i]
    cm = sig['credit_ma20'].iloc[i]
    cs = sig['credit_slope'].iloc[i]
    if not np.isnan(cr) and not np.isnan(cm) and cr < cm and not np.isnan(cs) and cs < -0.01:
        score += 1

    # Gold surging (20d momentum > 3% AND above 50d SMA) — flight to safety
    gm = sig['gld_mom_20'].iloc[i]
    ga = sig['gld_above_sma50'].iloc[i]
    if not np.isnan(gm) and gm > 0.03 and ga > 0:
        score += 1

    # Dollar strengthening (above 50d SMA)
    ur = sig['uup_rising'].iloc[i]
    if not np.isnan(ur) and ur > 0:
        score += 1

    # Bonds rallying (TLT 20d momentum > 2% AND above 50d SMA) — flight to quality
    tm = sig['tlt_mom_20'].iloc[i]
    ta = sig['tlt_above_sma50'].iloc[i]
    if not np.isnan(tm) and tm > 0.02 and ta > 0:
        score += 1

    return score


# ─────────────────────────────────────────────────────────────────
# STRATEGY VARIANTS
# ─────────────────────────────────────────────────────────────────

def v44_regime(sig, i, date, in_upro):
    """Original v4.4 — baseline."""
    s20 = sig['sma_20'].iloc[i]; s200 = sig['sma_200'].iloc[i]
    if date.month == 9: return 'SPY', False
    if not np.isnan(s20) and not np.isnan(s200) and s20 < s200: return 'SPY', False
    pctile = sig['vix_pctile_63'].iloc[i]
    if np.isnan(pctile): return 'SPY', False
    if pctile > 80: return 'GLD', False
    if pctile > 60: entry, exit_t = 3.0, 2.5
    elif pctile < 30: entry, exit_t = 2.0, 1.5
    else: entry, exit_t = 2.5, 2.0
    if pctile > 20: entry = max(entry, 2.5)
    score = confluence_score(sig, i)
    if in_upro:
        if score < exit_t: return 'SPY', False
        return 'UPRO', True
    else:
        if score >= entry: return 'UPRO', True
        return 'SPY', False


def v5_overlay_regime(sig, i, date, in_upro):
    """v5: v4.4 base + risk overlay that can override entry or force exit."""
    # Start with v4.4 base decision
    base_holding, base_upro = v44_regime(sig, i, date, in_upro)

    # Get risk overlay score
    risk = risk_overlay_score(sig, i)

    # Risk overlay logic:
    if risk >= 3:
        # RED: Multiple cross-asset signals say danger → go defensive
        return 'GLD', False
    elif risk >= 2:
        # YELLOW: Elevated risk → block new UPRO entries, tighter exit
        if base_holding == 'UPRO' and not in_upro:
            return 'SPY', False  # Block new entry
        if base_holding == 'UPRO' and in_upro:
            # Tighter exit: require higher confluence to stay
            score = confluence_score(sig, i)
            if score < 2.5:  # Tighter than normal
                return 'SPY', False
        return base_holding, base_upro
    else:
        # GREEN: Normal, trust v4.4
        return base_holding, base_upro


def v5_killswitch_regime(sig, i, date, in_upro):
    """v5b: v4.4 base + VIX curve + credit kill-switches only (fewer signals, less noise)."""
    base_holding, base_upro = v44_regime(sig, i, date, in_upro)

    # Kill-switch 1: VIX backwardation (5d avg)
    vt = sig['vix_term_ma5'].iloc[i]
    if not np.isnan(vt) and vt > 1.05:  # Strong backwardation
        return 'SHY', False

    # Kill-switch 2: Credit deteriorating sharply
    cr = sig['credit_ratio'].iloc[i]
    cm = sig['credit_ma20'].iloc[i]
    cs = sig['credit_slope'].iloc[i]
    if not np.isnan(cr) and not np.isnan(cm) and not np.isnan(cs):
        if cr < cm * 0.98 and cs < -0.02:  # 2%+ below MA AND falling fast
            return 'SHY', False

    return base_holding, base_upro


def v5_drawdown_shield(sig, i, date, in_upro):
    """v5c: v4.4 but with multi-signal drawdown shield.
    Only activates to PREVENT drawdowns (doesn't try to add alpha).
    Uses cross-asset signals as leading indicators of equity stress."""
    base_holding, base_upro = v44_regime(sig, i, date, in_upro)

    # Only intervene when holding UPRO (leveraged = most drawdown risk)
    if base_holding != 'UPRO':
        return base_holding, base_upro

    danger_count = 0

    # VIX backwardation
    vt = sig['vix_term_ma5'].iloc[i]
    if not np.isnan(vt) and vt > 1.0:
        danger_count += 1

    # Credit stress
    cr = sig['credit_ratio'].iloc[i]
    cm = sig['credit_ma20'].iloc[i]
    if not np.isnan(cr) and not np.isnan(cm) and cr < cm:
        danger_count += 1

    # Gold surging (safe haven bid)
    gm = sig['gld_mom_20'].iloc[i]
    if not np.isnan(gm) and gm > 0.03:
        danger_count += 1

    # Bonds rallying (flight to quality)
    tm = sig['tlt_mom_20'].iloc[i]
    if not np.isnan(tm) and tm > 0.02:
        danger_count += 1

    # 2+ danger signals while in UPRO → step down to SPY
    if danger_count >= 2:
        return 'SPY', False

    return base_holding, base_upro


def vmr_regime(sig, i):
    vix = sig['vix'].iloc[i]; vix_ma10 = sig['vix_ma10'].iloc[i]
    vix_peak20 = sig['vix_peak20'].iloc[i]
    if np.isnan(vix) or np.isnan(vix_ma10): return 'SPY'
    declining = vix < vix_ma10
    if vix < 15 and declining: return 'UPRO'
    elif vix > 20 and not np.isnan(vix_peak20) and vix < vix_peak20 * 0.85 and declining: return 'UPRO'
    elif vix > 25 and not declining: return 'GLD'
    elif vix > 20 and not declining: return 'SPY'
    else: return 'SPY'


# ─────────────────────────────────────────────────────────────────
# SIMULATION (next-day execution, fixed capital)
# ─────────────────────────────────────────────────────────────────

def simulate(closes, sig, strategy_fn, warmup=260, initial=100_000):
    rets = closes.pct_change()
    value = initial
    n_switches = 0
    prev_holding = None
    daily_values = [initial]
    daily_dates = [closes.index[warmup]]

    for idx in range(warmup, len(closes) - 1):
        d = closes.index[idx]
        dt = d.date() if hasattr(d, 'date') else d
        holding = strategy_fn(sig, idx, dt)

        if prev_holding is not None and holding != prev_holding:
            n_switches += 1
            value *= (1 - 0.0002)

        if holding in rets.columns:
            r = rets[holding].iloc[idx + 1]  # NEXT-DAY execution
            if not np.isnan(r):
                value *= (1 + r)

        prev_holding = holding
        daily_values.append(value)
        daily_dates.append(closes.index[idx + 1])

    return np.array(daily_values), daily_dates, n_switches


def metrics(vals, n_switches, label):
    rets = np.diff(vals) / vals[:-1]
    rets = rets[~np.isnan(rets)]
    n_years = len(rets) / 252

    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
    down = rets[rets < 0]
    sortino = np.mean(rets) / np.std(down) * np.sqrt(252) if len(down) > 0 and np.std(down) > 0 else 0
    cagr = (vals[-1] / vals[0]) ** (1 / n_years) - 1 if n_years > 0 and vals[0] > 0 else 0
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / peak
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    sw_per_yr = n_switches / n_years if n_years > 0 else 0

    pos = (rets > 0).sum()
    wr = pos / len(rets) * 100
    gp = rets[rets > 0].sum()
    gl = abs(rets[rets < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')

    print(f"  {label:40s} | Sharpe {sharpe:5.2f} | Sortino {sortino:5.2f} | "
          f"CAGR {cagr:7.1%} | MaxDD {maxdd:7.1%} | Calmar {calmar:5.2f} | "
          f"WR {wr:4.1f}% | PF {pf:4.2f} | Sw/yr {sw_per_yr:4.1f}")

    return {'sharpe': sharpe, 'sortino': sortino, 'cagr': cagr, 'maxdd': maxdd,
            'calmar': calmar, 'wr': wr, 'pf': pf, 'sw_per_yr': sw_per_yr,
            'final': vals[-1], 'label': label}


def permutation_test(closes, sig, strategy_fn, real_sharpe, warmup=260, n_perms=200):
    rets = closes.pct_change()
    regimes = []
    for idx in range(warmup, len(closes) - 1):
        d = closes.index[idx]
        dt = d.date() if hasattr(d, 'date') else d
        regimes.append(strategy_fn(sig, idx, dt))

    perm_sharpes = []
    for p in range(n_perms):
        np.random.seed(p)
        shuffled = regimes.copy()
        np.random.shuffle(shuffled)
        value = 100_000
        daily_values = [value]
        for j, idx in enumerate(range(warmup, len(closes) - 1)):
            holding = shuffled[j]
            if holding in rets.columns:
                r = rets[holding].iloc[idx + 1]
                if not np.isnan(r):
                    value *= (1 + r)
            daily_values.append(value)
        vals = np.array(daily_values)
        dr = np.diff(vals) / vals[:-1]
        dr = dr[~np.isnan(dr)]
        s = np.mean(dr) / np.std(dr) * np.sqrt(252) if np.std(dr) > 0 else 0
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).sum() / len(perm_sharpes)
    status = "PASS" if p_value < 0.05 else "FAIL"
    print(f"    Real: {real_sharpe:.3f} | Perm mean: {np.mean(perm_sharpes):.3f} | p={p_value:.3f} | {status}")
    return p_value


def sub_period_test(vals):
    n = len(vals)
    block = n // 3
    sharpes = []
    for b in range(3):
        s = b * block
        e = (b + 1) * block if b < 2 else n
        bv = vals[s:e]
        br = np.diff(bv) / bv[:-1]
        br = br[~np.isnan(br)]
        sh = np.mean(br) / np.std(br) * np.sqrt(252) if np.std(br) > 0 else 0
        sharpes.append(sh)
    cv = np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) != 0 else float('inf')
    status = "PASS" if cv < 0.50 else "FAIL"
    print(f"    B1={sharpes[0]:.2f} B2={sharpes[1]:.2f} B3={sharpes[2]:.2f} CV={cv:.2f} {status}")
    return cv


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 130)
    print("GAMEPLAN v5 — MULTI-SIGNAL STACK (all validated research layered)")
    print("Fixed $100K, no DCA, next-day execution, 0.02% switching cost")
    print("=" * 130)

    print("\nLoading data (SPY, UPRO, GLD, TLT, VIX, VIX3M, HYG, IEF, UUP, SHY)...")
    closes = load_data()
    print(f"  {len(closes)} days ({closes.index[0].date()} to {closes.index[-1].date()})")
    available = [c for c in closes.columns if closes[c].notna().sum() > 200]
    print(f"  Available tickers: {', '.join(available)}")

    print("Computing signals (core + cross-asset overlays)...")
    sig = compute_signals(closes)

    warmup = 260

    # Strategy closures with state
    def make_strategies():
        v44_s = [False]
        v5_s = [False]
        v5b_s = [False]
        v5c_s = [False]

        def strat_spy(sig, i, dt): return 'SPY'
        def strat_v44(sig, i, dt):
            h, v44_s[0] = v44_regime(sig, i, dt, v44_s[0]); return h
        def strat_vmr(sig, i, dt):
            return vmr_regime(sig, i)
        def strat_v5(sig, i, dt):
            h, v5_s[0] = v5_overlay_regime(sig, i, dt, v5_s[0]); return h
        def strat_v5b(sig, i, dt):
            h, v5b_s[0] = v5_killswitch_regime(sig, i, dt, v5b_s[0]); return h
        def strat_v5c(sig, i, dt):
            h, v5c_s[0] = v5_drawdown_shield(sig, i, dt, v5c_s[0]); return h

        return [
            ("SPY Buy & Hold", strat_spy, None),
            ("v4.4 (baseline)", strat_v44, v44_s),
            ("VMR", strat_vmr, None),
            ("v5 Full Overlay (all signals)", strat_v5, v5_s),
            ("v5b Kill-Switch Only (VIX curve+credit)", strat_v5b, v5b_s),
            ("v5c Drawdown Shield (multi-signal)", strat_v5c, v5c_s),
        ], [v44_s, v5_s, v5b_s, v5c_s]

    strategies, states = make_strategies()

    print(f"\n{'='*130}")
    print(f"RESULTS — {closes.index[warmup].date()} to {closes.index[-1].date()}")
    print(f"{'='*130}")

    results = {}
    for label, fn, state in strategies:
        # Reset all states
        for s in states:
            s[0] = False
        vals, dates, sw = simulate(closes, sig, fn, warmup=warmup)
        m = metrics(vals, sw, label)
        results[label] = (m, vals, fn)

    # Improvement analysis
    print(f"\n{'='*130}")
    print("IMPROVEMENT vs v4.4 BASELINE")
    print(f"{'='*130}")

    base = results["v4.4 (baseline)"][0]
    for label in ["v5 Full Overlay (all signals)", "v5b Kill-Switch Only (VIX curve+credit)",
                  "v5c Drawdown Shield (multi-signal)"]:
        m = results[label][0]
        ds = m['sharpe'] - base['sharpe']
        dso = m['sortino'] - base['sortino']
        dc = (m['cagr'] - base['cagr']) * 100
        ddd = (m['maxdd'] - base['maxdd']) * 100  # Positive = less drawdown
        print(f"  {label:45s} | ΔSharpe {ds:+.2f} | ΔSortino {dso:+.2f} | "
              f"ΔCAGR {dc:+.1f}pp | ΔMaxDD {ddd:+.1f}pp")

    # Permutation tests
    print(f"\n{'='*130}")
    print("PERMUTATION TESTS (200 shuffles)")
    print(f"{'='*130}")

    for label in ["v4.4 (baseline)", "v5 Full Overlay (all signals)",
                  "v5b Kill-Switch Only (VIX curve+credit)", "v5c Drawdown Shield (multi-signal)"]:
        m, vals, fn = results[label]
        for s in states:
            s[0] = False
        print(f"  {label}:")
        permutation_test(closes, sig, fn, m['sharpe'], warmup=warmup)

    # Sub-period consistency
    print(f"\n{'='*130}")
    print("SUB-PERIOD CONSISTENCY (3 blocks)")
    print(f"{'='*130}")

    for label in ["v4.4 (baseline)", "v5 Full Overlay (all signals)",
                  "v5b Kill-Switch Only (VIX curve+credit)", "v5c Drawdown Shield (multi-signal)"]:
        m, vals, fn = results[label]
        print(f"  {label}:")
        sub_period_test(vals)

    # Year-by-year comparison
    print(f"\n{'='*130}")
    print("YEAR-BY-YEAR: v4.4 vs BEST v5 variant vs SPY")
    print(f"{'='*130}")

    # Find best v5
    v5_labels = ["v5 Full Overlay (all signals)", "v5b Kill-Switch Only (VIX curve+credit)",
                 "v5c Drawdown Shield (multi-signal)"]
    best_v5 = max(v5_labels, key=lambda l: results[l][0]['sharpe'])
    print(f"  Best v5 variant by Sharpe: {best_v5}")
    print()

    # Compute annual returns for v4.4, best v5, SPY
    rets_df = closes.pct_change()
    for label, fn_label in [("v4.4", "v4.4 (baseline)"), ("Best v5", best_v5), ("SPY", "SPY Buy & Hold")]:
        m, vals, fn = results[fn_label]
        # Already have vals, compute annual from vals
        dates_arr = pd.DatetimeIndex([closes.index[warmup]] +
                                     list(closes.index[warmup+1:len(vals)+warmup]))

    print(f"\n{'='*130}")
    print("DONE")
    print(f"{'='*130}")


if __name__ == "__main__":
    main()
