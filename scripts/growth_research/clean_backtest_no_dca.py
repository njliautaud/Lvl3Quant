#!/usr/bin/env python3
"""
HC #713: Clean backtests — FIXED CAPITAL, NO DCA.
Pure strategy performance on $100K starting capital.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

def load_data():
    import yfinance as yf
    tickers = ["SPY", "UPRO", "GLD", "TLT", "^VIX", "SHY"]
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
    closes = closes.rename(columns={"^VIX": "VIX"})
    return closes.dropna(subset=["SPY", "UPRO"]).ffill()


def compute_signals(closes):
    spy = closes["SPY"]
    vix = closes["VIX"]
    spy_ret = spy.pct_change()
    sig = {}
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
    return sig


def confluence_score(sig, i):
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


def gp3_regime(sig, i, date, in_upro):
    s20 = sig['sma_20'].iloc[i]; s200 = sig['sma_200'].iloc[i]
    vol = sig['vol_21d'].iloc[i]
    if np.isnan(vol): vol = 15.0
    if date.month == 9: return 'SPY', False
    if not np.isnan(s20) and not np.isnan(s200) and s20 < s200: return 'SPY', False
    if vol > 30: return 'GLD', False
    if vol > 15: return 'SPY', False
    score = confluence_score(sig, i)
    if in_upro:
        if score < 2.0: return 'SPY', False
        return 'UPRO', True
    else:
        if score >= 2.5: return 'UPRO', True
        return 'SPY', False


def v44_regime(sig, i, date, in_upro):
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


def simulate_fixed_capital(closes, sig, strategy_fn, warmup=260, initial=100_000):
    """FIXED CAPITAL — NO DCA. Pure strategy performance.

    CRITICAL FIX (adversarial audit): Signal on day T → execute on day T+1.
    No same-bar look-ahead. Signal uses close[T], trade earns return[T+1].
    """
    rets = closes.pct_change()
    value = initial
    n_switches = 0
    prev_holding = None
    daily_values = [initial]  # Start from initial capital exactly
    daily_dates = [closes.index[warmup]]

    for idx in range(warmup, len(closes) - 1):  # -1: signal on T, return on T+1
        d = closes.index[idx]
        dt = d.date() if hasattr(d, 'date') else d
        holding = strategy_fn(sig, idx, dt)  # Signal computed from data up to day T

        if prev_holding is not None and holding != prev_holding:
            n_switches += 1
            value *= (1 - 0.0002)  # switching cost

        # Execute NEXT day's return (T+1) — the fix for look-ahead bias
        if holding in rets.columns:
            r = rets[holding].iloc[idx + 1]
            if not np.isnan(r):
                value *= (1 + r)

        prev_holding = holding
        daily_values.append(value)
        daily_dates.append(closes.index[idx + 1])

    return {
        'values': np.array(daily_values),
        'dates': daily_dates,
        'switches': n_switches,
        'final': value,
    }


def compute_metrics(result, label):
    vals = result['values']
    rets = np.diff(vals) / vals[:-1]
    rets = rets[~np.isnan(rets)]
    n_years = len(rets) / 252

    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
    downside = rets[rets < 0]
    sortino = np.mean(rets) / np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) > 0 else 0
    cagr = (vals[-1] / vals[0]) ** (1 / n_years) - 1 if n_years > 0 and vals[0] > 0 else 0

    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / peak
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    sw_per_yr = result['switches'] / n_years if n_years > 0 else 0

    # Win rate (positive days)
    pos_days = (rets > 0).sum()
    wr = pos_days / len(rets) * 100

    # Profit factor
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    print(f"  {label:40s} | Sharpe {sharpe:5.2f} | Sortino {sortino:5.2f} | "
          f"CAGR {cagr:7.1%} | MaxDD {maxdd:7.1%} | Calmar {calmar:5.2f} | "
          f"WR {wr:4.1f}% | PF {pf:4.2f} | Sw/yr {sw_per_yr:4.1f} | "
          f"Final ${vals[-1]:>12,.0f}")

    return {
        'label': label, 'sharpe': sharpe, 'sortino': sortino,
        'cagr': cagr, 'maxdd': maxdd, 'calmar': calmar,
        'wr': wr, 'pf': pf, 'sw_per_yr': sw_per_yr, 'final': vals[-1],
    }


def year_by_year(closes, sig, strategy_fn, label, warmup=260):
    """Annual returns for year-by-year breakdown."""
    rets = closes.pct_change()
    prev_holding = None
    in_upro = [False]

    # Collect daily strategy returns
    daily_rets = []
    daily_dates = []

    for idx in range(warmup, len(closes) - 1):  # Signal T, return T+1
        d = closes.index[idx]
        dt = d.date() if hasattr(d, 'date') else d
        holding = strategy_fn(sig, idx, dt)

        r = 0.0
        if holding in rets.columns:
            r = rets[holding].iloc[idx + 1]  # Next-day return
            if np.isnan(r): r = 0.0

        if prev_holding is not None and holding != prev_holding:
            r -= 0.0002  # switching cost

        prev_holding = holding
        daily_rets.append(r)
        daily_dates.append(closes.index[idx + 1])

    df = pd.DataFrame({'date': daily_dates, 'ret': daily_rets})
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    annual = df.groupby('year')['ret'].apply(lambda x: (1 + x).prod() - 1)
    return annual


def permutation_test(closes, sig, strategy_fn, real_sharpe, warmup=260, n_perms=200):
    """Permutation test on fixed capital."""
    print(f"\n  Permutation test ({n_perms} shuffles, fixed capital)...")
    rets = closes.pct_change()
    perm_sharpes = []

    # Get all regimes first (signal at T)
    regimes = []
    for idx in range(warmup, len(closes) - 1):  # Signal T, return T+1
        d = closes.index[idx]
        dt = d.date() if hasattr(d, 'date') else d
        regimes.append(strategy_fn(sig, idx, dt))

    for p in range(n_perms):
        np.random.seed(p)
        shuffled = regimes.copy()
        np.random.shuffle(shuffled)

        value = 100_000
        daily_values = [value]
        for j, idx in enumerate(range(warmup, len(closes) - 1)):
            holding = shuffled[j]
            if holding in rets.columns:
                r = rets[holding].iloc[idx + 1]  # Next-day return
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
    print(f"  Real: {real_sharpe:.3f} | Perm mean: {np.mean(perm_sharpes):.3f} | "
          f"p={p_value:.3f} | {'PASS' if p_value < 0.05 else 'FAIL'}")
    return p_value


def regime_test(closes, sig, strategy_fn, warmup=260):
    """R1 regime-agnostic test: green vs red day performance."""
    rets = closes.pct_change()
    spy_rets = rets['SPY']

    strat_rets_green = []
    strat_rets_red = []

    prev_holding = None
    for idx in range(warmup, len(closes) - 1):  # Signal T, return T+1
        d = closes.index[idx]
        dt = d.date() if hasattr(d, 'date') else d
        holding = strategy_fn(sig, idx, dt)

        r = 0.0
        if holding in rets.columns:
            r = rets[holding].iloc[idx + 1]  # Next-day return
            if np.isnan(r): r = 0.0

        spy_r = spy_rets.iloc[idx + 1]  # Classify by next day's SPY return
        if np.isnan(spy_r): continue

        if spy_r >= 0:
            strat_rets_green.append(r)
        else:
            strat_rets_red.append(r)

        prev_holding = holding

    green = np.array(strat_rets_green)
    red = np.array(strat_rets_red)

    sharpe_g = np.mean(green) / np.std(green) * np.sqrt(252) if np.std(green) > 0 else 0
    sharpe_r = np.mean(red) / np.std(red) * np.sqrt(252) if np.std(red) > 0 else 0

    gap = abs(sharpe_g - sharpe_r) / max(abs(sharpe_g), abs(sharpe_r)) if max(abs(sharpe_g), abs(sharpe_r)) > 0 else 0

    print(f"  R1: Green Sharpe {sharpe_g:.2f} | Red Sharpe {sharpe_r:.2f} | "
          f"Gap {gap:.2f} | {'PASS' if gap <= 0.50 else 'FAIL (expected for growth)'}")
    return gap


def main():
    print("=" * 130)
    print("HC #713: CLEAN BACKTESTS — FIXED $100K CAPITAL, NO DCA, PURE PERFORMANCE")
    print("=" * 130)

    print("\nLoading data...")
    closes = load_data()
    print(f"  {len(closes)} days ({closes.index[0].date()} to {closes.index[-1].date()})")

    print("Computing signals...")
    sig = compute_signals(closes)

    warmup = 260

    # Strategy functions
    gp3_state = [False]
    def strat_gp3(sig, i, dt):
        h, gp3_state[0] = gp3_regime(sig, i, dt, gp3_state[0])
        return h

    v44_state = [False]
    def strat_v44(sig, i, dt):
        h, v44_state[0] = v44_regime(sig, i, dt, v44_state[0])
        return h

    def strat_vmr(sig, i, dt):
        return vmr_regime(sig, i)

    gp3_cons = [False]
    def strat_consensus(sig, i, dt):
        vmr_h = vmr_regime(sig, i)
        gp3_h, gp3_cons[0] = gp3_regime(sig, i, dt, gp3_cons[0])
        if vmr_h == 'UPRO' and gp3_h == 'UPRO': return 'UPRO'
        if vmr_h == 'GLD': return 'GLD'
        return 'SPY'

    gp3_port = [False]
    def strat_portfolio(sig, i, dt):
        vmr_h = vmr_regime(sig, i)
        gp3_h, gp3_port[0] = gp3_regime(sig, i, dt, gp3_port[0])
        vmr_upro = vmr_h == 'UPRO'
        cons_upro = vmr_upro and gp3_h == 'UPRO'
        if vmr_upro and cons_upro: return 'UPRO'
        if vmr_h == 'GLD': return 'GLD'
        return 'SPY'

    def strat_spy(sig, i, dt):
        return 'SPY'

    def strat_upro(sig, i, dt):
        return 'UPRO'

    strategies = [
        ("SPY Buy & Hold", strat_spy),
        ("UPRO Buy & Hold (no timing)", strat_upro),
        ("Gameplan v3 (GP3)", strat_gp3),
        ("Gameplan v4.4 Full Adaptive", strat_v44),
        ("Vol Mean Reversion (VMR)", strat_vmr),
        ("GP3+VMR Consensus", strat_consensus),
        ("Portfolio: 50/50 VMR+Consensus", strat_portfolio),
    ]

    print(f"\n{'='*130}")
    print(f"RESULTS — Fixed $100K, no DCA, 0.02% switching cost, {closes.index[warmup].date()} to {closes.index[-1].date()}")
    print(f"{'='*130}")

    results = {}
    for label, fn in strategies:
        # Reset states
        gp3_state[0] = False; v44_state[0] = False
        gp3_cons[0] = False; gp3_port[0] = False

        res = simulate_fixed_capital(closes, sig, fn, warmup=warmup)
        metrics = compute_metrics(res, label)
        results[label] = metrics

    # Permutation tests
    print(f"\n{'='*130}")
    print("PERMUTATION TESTS (200 shuffles, fixed capital)")
    print(f"{'='*130}")

    for label, fn in [("Gameplan v4.4", strat_v44), ("VMR", strat_vmr),
                       ("GP3", strat_gp3), ("Consensus", strat_consensus)]:
        v44_state[0] = False; gp3_state[0] = False; gp3_cons[0] = False
        permutation_test(closes, sig, fn, results.get(f"Gameplan v4.4 Full Adaptive" if "v4.4" in label else
                                                       f"Vol Mean Reversion (VMR)" if "VMR" in label else
                                                       f"Gameplan v3 (GP3)" if "GP3" == label else
                                                       f"GP3+VMR Consensus", {}).get('sharpe', 0),
                         warmup=warmup)

    # Regime tests
    print(f"\n{'='*130}")
    print("REGIME TESTS (R1 — green vs red day Sharpe)")
    print(f"{'='*130}")

    for label, fn in strategies[2:]:  # skip benchmarks
        gp3_state[0] = False; v44_state[0] = False
        gp3_cons[0] = False; gp3_port[0] = False
        print(f"  {label}:")
        regime_test(closes, sig, fn, warmup=warmup)

    # Year-by-year for top strategies
    print(f"\n{'='*130}")
    print("YEAR-BY-YEAR RETURNS — v4.4 vs VMR vs SPY")
    print(f"{'='*130}")

    v44_state[0] = False
    annual_v44 = year_by_year(closes, sig, strat_v44, "v4.4", warmup=warmup)
    annual_vmr = year_by_year(closes, sig, strat_vmr, "VMR", warmup=warmup)
    annual_spy = year_by_year(closes, sig, strat_spy, "SPY", warmup=warmup)

    print(f"  {'Year':>6} | {'v4.4':>8} | {'VMR':>8} | {'SPY':>8} | v4.4 vs SPY")
    print(f"  {'-'*55}")
    for year in sorted(set(annual_v44.index) & set(annual_spy.index)):
        v = annual_v44.get(year, 0)
        m = annual_vmr.get(year, 0)
        s = annual_spy.get(year, 0)
        better = "WIN" if v > s else "LOSE"
        print(f"  {year:>6} | {v:>7.1%} | {m:>7.1%} | {s:>7.1%} | {better}")

    v44_wins = sum(1 for y in annual_v44.index if y in annual_spy.index and annual_v44[y] > annual_spy[y])
    total = len(set(annual_v44.index) & set(annual_spy.index))
    print(f"\n  v4.4 beats SPY: {v44_wins}/{total} years")

    # Sub-period consistency: 3 blocks
    print(f"\n{'='*130}")
    print("SUB-PERIOD CONSISTENCY — 3 equal blocks")
    print(f"{'='*130}")

    for label, fn in [("v4.4", strat_v44), ("VMR", strat_vmr), ("Consensus", strat_consensus)]:
        gp3_state[0] = False; v44_state[0] = False
        gp3_cons[0] = False; gp3_port[0] = False

        res = simulate_fixed_capital(closes, sig, fn, warmup=warmup)
        vals = res['values']
        n = len(vals)
        block_size = n // 3

        block_sharpes = []
        for b in range(3):
            start = b * block_size
            end = (b + 1) * block_size if b < 2 else n
            block_vals = vals[start:end]
            block_rets = np.diff(block_vals) / block_vals[:-1]
            block_rets = block_rets[~np.isnan(block_rets)]
            s = np.mean(block_rets) / np.std(block_rets) * np.sqrt(252) if np.std(block_rets) > 0 else 0
            block_sharpes.append(s)

        cv = np.std(block_sharpes) / np.mean(block_sharpes) if np.mean(block_sharpes) != 0 else float('inf')
        status = "PASS" if cv < 0.50 else "FAIL"
        print(f"  {label:20s} | B1={block_sharpes[0]:.2f} | B2={block_sharpes[1]:.2f} | B3={block_sharpes[2]:.2f} | CV={cv:.2f} | {status}")

    print(f"\n{'='*130}")
    print("DONE — These are the REAL numbers. Fixed capital. No DCA inflation. Next-day execution.")
    print(f"{'='*130}")


if __name__ == "__main__":
    main()
