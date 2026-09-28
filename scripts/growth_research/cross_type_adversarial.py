#!/usr/bin/env python3
"""
Cross-Type Confluence Adversarial Validation
Strategy E: Fear + Micro + Dip (Type 2 + Type 5 + Type 1)
Strategy F: Perfect Storm (all 5 types)
6-test adversarial battery for each.
Optimized: pre-compute all features into aligned numpy arrays.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product
import warnings
import sys
import time

warnings.filterwarnings('ignore')
np.random.seed(42)

# Unbuffered output
def log(msg):
    print(msg, flush=True)

# ─── CONFIG ───
UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'UNH', 'LLY', 'AVGO', 'AMD']
START = '2019-01-01'
END = '2026-07-01'
TRADE_START = '2020-01-01'
POSITION_SIZE = 300
MAX_CONCURRENT = 2


def download_data():
    log("Downloading data...")
    tickers = UNIVERSE + ['^VIX', '^TNX']
    all_data = {}
    for t in tickers:
        for attempt in range(3):
            try:
                df = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
                if len(df) > 100:
                    all_data[t] = df
                    break
            except Exception:
                time.sleep(1)
        if t not in all_data:
            log(f"  WARNING: Failed {t}")
    log(f"  Got {len(all_data)} tickers")
    return all_data


def build_aligned_features(all_data):
    """Build a single DataFrame with all features aligned to a common date index."""
    log("Building aligned features...")

    # Use VIX index as reference (most trading days)
    ref = all_data['^VIX']
    idx = ref.index

    feat = pd.DataFrame(index=idx)

    # VIX
    vix_close = all_data['^VIX']['Close'].squeeze().reindex(idx)
    feat['vix'] = vix_close
    feat['vix_30ma'] = vix_close.rolling(30, min_periods=20).mean()
    feat['vix_45ma'] = vix_close.rolling(45, min_periods=30).mean()
    feat['vix_60ma'] = vix_close.rolling(60, min_periods=40).mean()
    feat['vix_75ma'] = vix_close.rolling(75, min_periods=50).mean()
    feat['vix_90ma'] = vix_close.rolling(90, min_periods=60).mean()

    # TNX
    if '^TNX' in all_data:
        tnx = all_data['^TNX']['Close'].squeeze().reindex(idx)
        feat['tnx'] = tnx
        feat['tnx_5d_chg'] = tnx.diff(5)

    # Stock features
    for sym in UNIVERSE:
        if sym not in all_data:
            continue
        df = all_data[sym].reindex(idx)
        close = df['Close'].squeeze()
        high = df['High'].squeeze()
        low = df['Low'].squeeze()

        feat[f'{sym}_close'] = close

        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        feat[f'{sym}_rsi'] = 100 - (100 / (1 + rs))

        # HL spread
        hl = (high - low) / close.replace(0, np.nan)
        feat[f'{sym}_hl'] = hl
        for lb in [30, 45, 60, 75, 90]:
            feat[f'{sym}_hl_{lb}ma'] = hl.rolling(lb, min_periods=lb//2).mean()

        # SMA20
        feat[f'{sym}_sma20'] = close.rolling(20).mean()

        # For RSI divergence: shifted values
        feat[f'{sym}_close_10ago'] = close.shift(10)
        feat[f'{sym}_rsi_10ago'] = feat[f'{sym}_rsi'].shift(10)

    # Trim to trade period
    feat = feat.loc[TRADE_START:]
    log(f"  Features ready: {len(feat)} days, {len(feat.columns)} columns")
    return feat


def backtest_strategy(feat, signal_mask_func, exit_params=None):
    """
    Fast backtest using pre-computed feature DataFrame.
    signal_mask_func(feat, date_idx, sym) -> bool
    """
    if exit_params is None:
        exit_params = {'max_hold': 21, 'tp_pct': 10, 'sl_pct': 15}

    max_hold = exit_params.get('max_hold', 21)
    tp = exit_params.get('tp_pct', 10) / 100
    sl = exit_params.get('sl_pct', 15) / 100

    dates = feat.index
    n_dates = len(dates)

    # Pre-extract arrays for speed
    vix_arr = feat['vix'].values
    vix_60ma_arr = feat['vix_60ma'].values

    close_arrays = {}
    for sym in UNIVERSE:
        col = f'{sym}_close'
        if col in feat.columns:
            close_arrays[sym] = feat[col].values

    trades = []
    active = []  # (sym, entry_idx, entry_price)

    for i in range(n_dates):
        # Exits
        new_active = []
        for sym, ei, ep in active:
            if sym not in close_arrays:
                new_active.append((sym, ei, ep))
                continue
            cp = close_arrays[sym][i]
            if np.isnan(cp):
                new_active.append((sym, ei, ep))
                continue

            days_held = (dates[i] - dates[ei]).days
            ret = (cp - ep) / ep
            reason = None

            if ret >= tp:
                reason = 'TP'
            elif ret <= -sl:
                reason = 'SL'
            elif days_held >= max_hold:
                reason = 'MAX_HOLD'
            else:
                v = vix_arr[i]
                vm = vix_60ma_arr[i]
                if not np.isnan(v) and not np.isnan(vm) and v < vm:
                    reason = 'VIX_NORM'

            if reason:
                trades.append({
                    'sym': sym, 'entry_date': dates[ei], 'exit_date': dates[i],
                    'entry_price': ep, 'exit_price': cp,
                    'return': ret, 'exit_reason': reason, 'days_held': days_held
                })
            else:
                new_active.append((sym, ei, ep))
        active = new_active

        # Entries
        if len(active) >= MAX_CONCURRENT:
            continue

        for sym in UNIVERSE:
            if len(active) >= MAX_CONCURRENT:
                break
            if any(s == sym for s, _, _ in active):
                continue
            if sym not in close_arrays:
                continue

            if signal_mask_func(feat, i, sym):
                cp = close_arrays[sym][i]
                if not np.isnan(cp) and cp > 0:
                    active.append((sym, i, cp))

    # Close remaining
    if active and n_dates > 0:
        last_i = n_dates - 1
        for sym, ei, ep in active:
            if sym in close_arrays:
                cp = close_arrays[sym][last_i]
                if not np.isnan(cp):
                    ret = (cp - ep) / ep
                    trades.append({
                        'sym': sym, 'entry_date': dates[ei], 'exit_date': dates[last_i],
                        'entry_price': ep, 'exit_price': cp,
                        'return': ret, 'exit_reason': 'EOD',
                        'days_held': (dates[last_i] - dates[ei]).days
                    })

    return trades


# ─── SIGNAL FUNCTIONS (work on pre-computed arrays via feat DataFrame) ───

def make_signal_e(params=None):
    if params is None:
        params = {'vix_mult': 1.15, 'rsi_thresh': 40, 'spread_lb': 60}

    vm = params['vix_mult']
    rst = params['rsi_thresh']
    slb = params['spread_lb']
    vix_ma_col = f'vix_{slb}ma' if f'vix_{slb}ma' in ['vix_30ma','vix_45ma','vix_60ma','vix_75ma','vix_90ma'] else 'vix_60ma'
    hl_ma_col_template = '{sym}_hl_{lb}ma'

    def signal(feat, i, sym):
        # VIX elevated
        v = feat['vix'].iat[i]
        vma_col = vix_ma_col
        if vma_col not in feat.columns:
            vma_col = 'vix_60ma'
        vma = feat[vma_col].iat[i]
        if np.isnan(v) or np.isnan(vma) or v <= vm * vma:
            return False

        # HL spread < avg
        hl_col = f'{sym}_hl'
        hl_ma = f'{sym}_hl_{slb}ma'
        if hl_col not in feat.columns:
            return False
        if hl_ma not in feat.columns:
            hl_ma = f'{sym}_hl_60ma'
        hlv = feat[hl_col].iat[i]
        hlma = feat[hl_ma].iat[i]
        if np.isnan(hlv) or np.isnan(hlma) or hlv >= hlma:
            return False

        # RSI < threshold
        rsi_col = f'{sym}_rsi'
        if rsi_col not in feat.columns:
            return False
        rsiv = feat[rsi_col].iat[i]
        if np.isnan(rsiv) or rsiv >= rst:
            return False

        return True

    return signal


def make_signal_e_inverse(params=None):
    if params is None:
        params = {'vix_mult': 1.15, 'rsi_thresh': 40, 'spread_lb': 60}

    vm = params['vix_mult']
    rst = params['rsi_thresh']
    slb = params['spread_lb']

    def signal(feat, i, sym):
        # VIX LOW (not elevated)
        v = feat['vix'].iat[i]
        vma_col = f'vix_{slb}ma' if f'vix_{slb}ma' in feat.columns else 'vix_60ma'
        vma = feat[vma_col].iat[i]
        if np.isnan(v) or np.isnan(vma) or v >= vm * vma:
            return False

        # HL spread WIDE
        hl_col = f'{sym}_hl'
        hl_ma = f'{sym}_hl_{slb}ma' if f'{sym}_hl_{slb}ma' in feat.columns else f'{sym}_hl_60ma'
        if hl_col not in feat.columns:
            return False
        hlv = feat[hl_col].iat[i]
        hlma = feat[hl_ma].iat[i]
        if np.isnan(hlv) or np.isnan(hlma) or hlv <= hlma:
            return False

        # RSI HIGH
        rsi_col = f'{sym}_rsi'
        if rsi_col not in feat.columns:
            return False
        rsiv = feat[rsi_col].iat[i]
        if np.isnan(rsiv) or rsiv <= (100 - rst):
            return False

        return True

    return signal


def make_signal_f(params=None):
    if params is None:
        params = {'vix_mult': 1.15, 'tnx_drop': 0.05, 'spread_lb': 60, 'dip_pct': 5}

    vm = params['vix_mult']
    td = params['tnx_drop']
    slb = params['spread_lb']
    dp = params['dip_pct']

    def signal(feat, i, sym):
        # 1. VIX elevated
        v = feat['vix'].iat[i]
        vma = feat['vix_60ma'].iat[i]
        if np.isnan(v) or np.isnan(vma) or v <= vm * vma:
            return False

        # 2. 10Y yield dropped
        if 'tnx_5d_chg' not in feat.columns:
            return False
        tc = feat['tnx_5d_chg'].iat[i]
        if np.isnan(tc) or tc >= -td:
            return False

        # 3. RSI divergence: price lower low, RSI higher low
        pc = f'{sym}_close'
        rc = f'{sym}_rsi'
        p10 = f'{sym}_close_10ago'
        r10 = f'{sym}_rsi_10ago'
        for c in [pc, rc, p10, r10]:
            if c not in feat.columns:
                return False
        cprice = feat[pc].iat[i]
        pprice = feat[p10].iat[i]
        crsi = feat[rc].iat[i]
        prsi = feat[r10].iat[i]
        if np.isnan(cprice) or np.isnan(pprice) or np.isnan(crsi) or np.isnan(prsi):
            return False
        if not (cprice < pprice and crsi > prsi):
            return False

        # 4. HL spread < avg
        hl_col = f'{sym}_hl'
        hl_ma = f'{sym}_hl_{slb}ma' if f'{sym}_hl_{slb}ma' in feat.columns else f'{sym}_hl_60ma'
        if hl_col not in feat.columns:
            return False
        hlv = feat[hl_col].iat[i]
        hlma = feat[hl_ma].iat[i]
        if np.isnan(hlv) or np.isnan(hlma) or hlv >= hlma:
            return False

        # 5. Stock > dip_pct% below 20d SMA
        sma_col = f'{sym}_sma20'
        if sma_col not in feat.columns:
            return False
        sma = feat[sma_col].iat[i]
        if np.isnan(sma) or cprice >= sma * (1 - dp / 100):
            return False

        return True

    return signal


def make_signal_f_inverse(params=None):
    if params is None:
        params = {'vix_mult': 1.15, 'tnx_drop': 0.05, 'spread_lb': 60, 'dip_pct': 5}

    vm = params['vix_mult']
    td = params['tnx_drop']
    slb = params['spread_lb']
    dp = params['dip_pct']

    def signal(feat, i, sym):
        # 1. VIX LOW
        v = feat['vix'].iat[i]
        vma = feat['vix_60ma'].iat[i]
        if np.isnan(v) or np.isnan(vma) or v >= vm * vma:
            return False

        # 2. Yield RISING
        if 'tnx_5d_chg' not in feat.columns:
            return False
        tc = feat['tnx_5d_chg'].iat[i]
        if np.isnan(tc) or tc <= td:
            return False

        # 3. Bearish divergence: price higher high, RSI lower high
        pc = f'{sym}_close'
        rc = f'{sym}_rsi'
        p10 = f'{sym}_close_10ago'
        r10 = f'{sym}_rsi_10ago'
        for c in [pc, rc, p10, r10]:
            if c not in feat.columns:
                return False
        cprice = feat[pc].iat[i]
        pprice = feat[p10].iat[i]
        crsi = feat[rc].iat[i]
        prsi = feat[r10].iat[i]
        if np.isnan(cprice) or np.isnan(pprice) or np.isnan(crsi) or np.isnan(prsi):
            return False
        if not (cprice > pprice and crsi < prsi):
            return False

        # 4. HL spread WIDE
        hl_col = f'{sym}_hl'
        hl_ma = f'{sym}_hl_{slb}ma' if f'{sym}_hl_{slb}ma' in feat.columns else f'{sym}_hl_60ma'
        if hl_col not in feat.columns:
            return False
        hlv = feat[hl_col].iat[i]
        hlma = feat[hl_ma].iat[i]
        if np.isnan(hlv) or np.isnan(hlma) or hlv <= hlma:
            return False

        # 5. Stock ABOVE 20d SMA
        sma_col = f'{sym}_sma20'
        if sma_col not in feat.columns:
            return False
        sma = feat[sma_col].iat[i]
        if np.isnan(sma) or cprice <= sma * (1 + dp / 100):
            return False

        return True

    return signal


def compute_metrics(trades):
    if not trades:
        return {'sharpe': 0.0, 'wr': 0.0, 'pf': 0.0, 'n_trades': 0, 'avg_ret': 0.0, 'total_ret': 0.0}

    rets = np.array([t['return'] for t in trades])
    days = np.array([max(t['days_held'], 1) for t in trades])

    n = len(rets)
    wr = np.mean(rets > 0) * 100
    gw = rets[rets > 0].sum() if np.any(rets > 0) else 0
    gl = abs(rets[rets < 0].sum()) if np.any(rets < 0) else 0.001
    pf = gw / gl if gl > 0 else 999

    avg = np.mean(rets)
    std = np.std(rets)
    avg_days = np.mean(days)
    sharpe = (avg / std) * np.sqrt(252 / avg_days) if std > 0 else 0

    return {
        'sharpe': round(sharpe, 3), 'wr': round(wr, 1), 'pf': round(pf, 2),
        'n_trades': n, 'avg_ret': round(avg * 100, 2), 'total_ret': round(rets.sum() * 100, 2)
    }


def sub_period_split(trades, n_periods=4):
    if not trades:
        return [[] for _ in range(n_periods)]
    dates = sorted([t['entry_date'] for t in trades])
    mn, mx = dates[0], dates[-1]
    td = (mx - mn).days
    if td == 0:
        return [trades] + [[] for _ in range(n_periods - 1)]
    pd_len = td / n_periods
    periods = [[] for _ in range(n_periods)]
    for t in trades:
        idx = min(int((t['entry_date'] - mn).days / pd_len), n_periods - 1)
        periods[idx].append(t)
    return periods


def per_stock_contribution(trades):
    contribs = {}
    for t in trades:
        contribs[t['sym']] = contribs.get(t['sym'], 0) + t['return']
    return contribs


def random_timing_test(feat, n_target, n_perms):
    dates = feat.index.tolist()
    n_dates = len(dates)

    close_arrays = {}
    for sym in UNIVERSE:
        col = f'{sym}_close'
        if col in feat.columns:
            close_arrays[sym] = feat[col].values

    syms = [s for s in UNIVERSE if s in close_arrays]
    random_sharpes = []

    for _ in range(n_perms):
        idxs = np.random.choice(n_dates, size=min(n_target * 2, n_dates), replace=False)
        chosen_syms = np.random.choice(syms, size=len(idxs))

        fake_trades = []
        for j, di in enumerate(idxs[:n_target]):
            sym = chosen_syms[j]
            ep = close_arrays[sym][di]
            if np.isnan(ep) or ep <= 0:
                continue
            hold = np.random.randint(5, 21)
            ei = min(di + hold, n_dates - 1)
            xp = close_arrays[sym][ei]
            if np.isnan(xp):
                continue
            ret = (xp - ep) / ep
            fake_trades.append({
                'return': ret, 'days_held': (dates[ei] - dates[di]).days,
                'sym': sym, 'entry_date': dates[di], 'exit_date': dates[ei],
                'entry_price': ep, 'exit_price': xp
            })

        m = compute_metrics(fake_trades)
        random_sharpes.append(m['sharpe'])

    return random_sharpes


def param_sensitivity_e(feat, n_combos):
    log(f"    Running {n_combos} param combos...")
    vix_mults = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
    rsi_thresholds = [30, 35, 40, 45]
    spread_lbs = [30, 45, 60, 75, 90]
    max_holds = [10, 15, 21, 25, 30]
    tps = [5, 7.5, 10, 12.5, 15]
    sls = [10, 12.5, 15, 17.5, 20]

    all_combos = list(product(vix_mults, rsi_thresholds, spread_lbs, max_holds, tps, sls))
    np.random.shuffle(all_combos)
    combos = all_combos[:n_combos]

    results = []
    for j, (vm, rst, slb, mh, tp, sl) in enumerate(combos):
        if (j + 1) % 50 == 0:
            log(f"      {j+1}/{n_combos} done...")
        sig = make_signal_e({'vix_mult': vm, 'rsi_thresh': rst, 'spread_lb': slb})
        trades = backtest_strategy(feat, sig, {'max_hold': mh, 'tp_pct': tp, 'sl_pct': sl})
        results.append(compute_metrics(trades))

    return results


def param_sensitivity_f(feat, n_combos):
    log(f"    Running {n_combos} param combos...")
    vix_mults = [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
    tnx_drops = [0.02, 0.04, 0.06, 0.08, 0.10]
    dip_pcts = [3, 4.5, 5, 6.5, 8]
    spread_lbs = [30, 45, 60, 75, 90]
    max_holds = [10, 15, 21, 25, 30]
    tps = [5, 7.5, 10, 12.5, 15]
    sls = [10, 12.5, 15, 17.5, 20]

    all_combos = list(product(vix_mults, tnx_drops, dip_pcts, spread_lbs, max_holds, tps, sls))
    np.random.shuffle(all_combos)
    combos = all_combos[:n_combos]

    results = []
    for j, (vm, td, dp, slb, mh, tp, sl) in enumerate(combos):
        if (j + 1) % 50 == 0:
            log(f"      {j+1}/{n_combos} done...")
        sig = make_signal_f({'vix_mult': vm, 'tnx_drop': td, 'spread_lb': slb, 'dip_pct': dp})
        trades = backtest_strategy(feat, sig, {'max_hold': mh, 'tp_pct': tp, 'sl_pct': sl})
        results.append(compute_metrics(trades))

    return results


def run_6_tests(feat, strategy_name, signal_func, inverse_func, baseline,
                n_perms, n_param_combos, param_sens_func):
    log(f"\n{'='*70}")
    log(f"  STRATEGY {strategy_name} — ADVERSARIAL VALIDATION")
    log(f"{'='*70}")

    # Test 1: Re-implementation
    log(f"\n[Test 1] Re-implementation...")
    trades = backtest_strategy(feat, signal_func)
    m = compute_metrics(trades)
    thresh1 = round(baseline['sharpe'] * 0.70, 3)
    p1 = m['sharpe'] > thresh1
    log(f"  Sharpe={m['sharpe']}, WR={m['wr']}%, PF={m['pf']}, Trades={m['n_trades']}, AvgRet={m['avg_ret']}%")
    log(f"  Threshold: > {thresh1} | {'PASS' if p1 else 'FAIL'}")

    # Test 2: Inverse
    log(f"\n[Test 2] Inverse signals...")
    inv_trades = backtest_strategy(feat, inverse_func)
    im = compute_metrics(inv_trades)
    thresh2 = round(baseline['sharpe'] * 0.50, 3)
    p2 = im['sharpe'] < thresh2
    log(f"  Sharpe={im['sharpe']}, WR={im['wr']}%, Trades={im['n_trades']}")
    log(f"  Threshold: < {thresh2} | {'PASS' if p2 else 'FAIL'}")

    # Test 3: Random timing
    log(f"\n[Test 3] Random timing ({n_perms} perms)...")
    rsharpes = random_timing_test(feat, m['n_trades'], n_perms)
    perm_p = np.mean([rs >= m['sharpe'] for rs in rsharpes])
    p3 = perm_p < 0.05
    log(f"  Actual Sharpe: {m['sharpe']}, Random median: {np.median(rsharpes):.3f}, p95: {np.percentile(rsharpes,95):.3f}")
    log(f"  p-value: {perm_p:.4f} | {'PASS' if p3 else 'FAIL'}")

    # Test 4: Sub-period
    log(f"\n[Test 4] Sub-period stability...")
    periods = sub_period_split(trades)
    psharpes = []
    n_pos = 0
    any_bad = False
    for i, pt in enumerate(periods):
        pm = compute_metrics(pt)
        psharpes.append(pm['sharpe'])
        if pm['sharpe'] > 0: n_pos += 1
        if pm['sharpe'] < -0.50: any_bad = True
        log(f"  P{i+1}: Sharpe={pm['sharpe']}, N={pm['n_trades']}, WR={pm['wr']}%")
    p4 = n_pos >= 3 and not any_bad
    log(f"  Positive: {n_pos}/4, Any<-0.50: {any_bad} | {'PASS' if p4 else 'FAIL'}")

    # Test 5: Top-3 removal
    log(f"\n[Test 5] Top-3 stock removal...")
    contribs = per_stock_contribution(trades)
    top3 = [s for s, _ in sorted(contribs.items(), key=lambda x: x[1], reverse=True)[:3]]
    red_trades = [t for t in trades if t['sym'] not in top3]
    rm = compute_metrics(red_trades)
    thresh5 = round(baseline['sharpe'] * 0.50, 3)
    p5 = rm['sharpe'] > thresh5
    log(f"  Removed: {top3}")
    log(f"  Remaining: Sharpe={rm['sharpe']}, N={rm['n_trades']}")
    log(f"  Threshold: > {thresh5} | {'PASS' if p5 else 'FAIL'}")

    # Test 6: Param sensitivity
    log(f"\n[Test 6] Parameter sensitivity ({n_param_combos} combos)...")
    sens = param_sens_func(feat, n_param_combos)
    sharpes = [r['sharpe'] for r in sens]
    n_above = sum(1 for s in sharpes if s > 0.30)
    pct = n_above / max(len(sens), 1) * 100
    p6 = pct > 50
    log(f"  Sharpe: min={min(sharpes):.3f}, med={np.median(sharpes):.3f}, max={max(sharpes):.3f}")
    log(f"  >0.30: {n_above}/{len(sens)} ({pct:.1f}%) | {'PASS' if p6 else 'FAIL'}")

    # Summary
    results = [p1, p2, p3, p4, p5, p6]
    labels = ['Re-impl', 'Inverse', 'Perm', 'SubPeriod', 'Top3Remove', 'ParamSens']
    log(f"\n{'─'*50}")
    log(f"  STRATEGY {strategy_name}: {sum(results)}/6 PASSED")
    for l, r in zip(labels, results):
        log(f"    {l}: {'PASS' if r else 'FAIL'}")
    log(f"{'─'*50}")

    return {
        'strategy': strategy_name, 'reimpl': m, 'inverse': im,
        'perm_p': perm_p, 'psharpes': psharpes, 'reduced': rm,
        'param_pct': pct, 'n_pass': sum(results), 'results': results
    }


def main():
    all_data = download_data()
    if len(all_data) < 5:
        log("ERROR: insufficient data")
        sys.exit(1)

    feat = build_aligned_features(all_data)

    # Strategy E
    baseline_e = {'sharpe': 1.922}
    re = run_6_tests(
        feat, 'E', make_signal_e(), make_signal_e_inverse(),
        baseline_e, 300, 150, param_sensitivity_e
    )

    # Strategy F
    baseline_f = {'sharpe': 3.139}
    rf = run_6_tests(
        feat, 'F', make_signal_f(), make_signal_f_inverse(),
        baseline_f, 500, 200, param_sensitivity_f
    )

    # Final
    log(f"\n{'='*70}")
    log(f"  FINAL SUMMARY")
    log(f"{'='*70}")
    for r in [re, rf]:
        m = r['reimpl']
        log(f"\n  Strategy {r['strategy']}:")
        log(f"    Sharpe={m['sharpe']}, WR={m['wr']}%, PF={m['pf']}, N={m['n_trades']}")
        log(f"    Tests: {r['n_pass']}/6")
        labels = ['Re-impl', 'Inverse', 'Perm', 'SubPeriod', 'Top3Remove', 'ParamSens']
        for l, p in zip(labels, r['results']):
            log(f"      {l}: {'PASS' if p else 'FAIL'}")
    log(f"\n  COMBINED: {re['n_pass'] + rf['n_pass']}/12 passed")
    log(f"{'='*70}")


if __name__ == '__main__':
    main()
