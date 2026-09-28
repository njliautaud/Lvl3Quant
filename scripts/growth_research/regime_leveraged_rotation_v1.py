#!/usr/bin/env python3
"""
Regime-Switching Leveraged ETF Strategy v1 — HIGH GROWTH TRACK
================================================================
HYPOTHESIS: Leveraged ETFs produce 30%+ CAGR but MDD of -82% destroys accounts.
Solution: Use a regime detector to AVOID bear markets.

INSIGHT FROM PREVIOUS RUN:
- Leveraged ETF LGBM rotation: CAGR 35.6% but MDD -82%, Sharpe 0.78
- The problem is NOT which leveraged ETF to pick (LGBM adds minimal edge)
- The problem is WHETHER to hold leveraged ETFs at all
- Bull Sharpe=2.4, Bear Sharpe=-1.9 → MASSIVE regime dependency

APPROACH:
- Layer 1: Trend/Regime detector (not LGBM, simpler and faster)
  - SPY above/below 50d/200d MA crossover
  - Market breadth (% above 50d MA)
  - VIX level and trend
  - Momentum (SPY 21d return)
- Layer 2: IF regime = bullish → hold top leveraged ETF by momentum
         IF regime = neutral → hold cash or 1x ETF (SPY)
         IF regime = bearish → hold inverse leveraged ETF or cash

VARIANTS:
  A: Simple MA200 (SPY > 200d MA → TQQQ, else cash)
  B: Golden/Death Cross (50/200 MA crossover → TQQQ/cash)
  C: Multi-signal regime (MA + VIX + momentum → bull/bear/neutral)
  D: Risk-parity regime (leverage when vol low, delever when vol high)
  E: Adaptive leverage (1x/2x/3x based on regime score)
  F: Best-of-breed rotation (regime filter + pick best momentum leveraged ETF)
  G: Bear profiter (inverse ETFs in bear regime, bull ETFs in bull)
  H: Concentrated with stop (all-in TQQQ when bullish, tight trailing stop)
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = '.'

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'regime_leveraged_rotation_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

STARTING_CAPITAL = 645.0
START_DATE = '2015-01-01'
END_DATE = '2026-07-28'
N_PERMUTATIONS = 150

# Universe
BULL_ETFS = ['TQQQ', 'SOXL', 'UPRO', 'TECL', 'SPXL', 'TNA']
BEAR_ETFS = ['SQQQ', 'SOXS', 'SPXU', 'TECS', 'SPXS', 'TZA']
NEUTRAL_ETFS = ['SPY', 'QQQ']  # 1x ETFs for neutral regime

print(f"Running on: {LVL3_ROOT}", flush=True)

# ============================================================
# DATA
# ============================================================

def load_data():
    import yfinance as yf
    all_tickers = list(set(BULL_ETFS + BEAR_ETFS + NEUTRAL_ETFS + ['^VIX']))
    print(f"\nDownloading {len(all_tickers)} tickers...", flush=True)

    frames = {}
    for t in sorted(all_tickers):
        try:
            d = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if len(d) < 100:
                print(f"  SKIP {t}: {len(d)} rows", flush=True)
                continue
            d.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in d.columns]
            frames[t] = d
        except Exception as e:
            print(f"  ERR {t}: {e}", flush=True)

    avail_bull = [t for t in BULL_ETFS if t in frames]
    avail_bear = [t for t in BEAR_ETFS if t in frames]
    print(f"Bull: {avail_bull}", flush=True)
    print(f"Bear: {avail_bear}", flush=True)
    return frames, avail_bull, avail_bear


# ============================================================
# REGIME DETECTION
# ============================================================

def detect_regime_simple_ma200(spy, date):
    """SPY above 200d MA = bull, below = bear."""
    try:
        idx = spy.index.get_loc(date)
    except KeyError:
        return 'neutral'
    if idx < 200:
        return 'neutral'
    close = spy['close'].iloc[:idx+1]
    ma200 = close.rolling(200).mean().iloc[-1]
    return 'bull' if close.iloc[-1] > ma200 else 'bear'


def detect_regime_golden_cross(spy, date):
    """50/200 MA crossover. Golden cross = bull, death cross = bear."""
    try:
        idx = spy.index.get_loc(date)
    except KeyError:
        return 'neutral'
    if idx < 200:
        return 'neutral'
    close = spy['close'].iloc[:idx+1]
    ma50 = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1]
    return 'bull' if ma50 > ma200 else 'bear'


def detect_regime_multi_signal(spy, vix_data, date):
    """
    Multi-signal regime: combine MA, VIX, momentum.
    Score: -3 to +3. >= 2 = bull, <= -2 = bear, else neutral.
    """
    try:
        spy_idx = spy.index.get_loc(date)
    except KeyError:
        return 'neutral', 0
    if spy_idx < 200:
        return 'neutral', 0

    close = spy['close'].iloc[:spy_idx+1]
    score = 0

    # Signal 1: SPY above 200d MA
    ma200 = close.rolling(200).mean().iloc[-1]
    if close.iloc[-1] > ma200:
        score += 1
    else:
        score -= 1

    # Signal 2: SPY above 50d MA
    ma50 = close.rolling(50).mean().iloc[-1]
    if close.iloc[-1] > ma50:
        score += 1
    else:
        score -= 1

    # Signal 3: SPY 21d momentum > 0
    if spy_idx >= 21:
        mom_21 = (close.iloc[-1] / close.iloc[-22]) - 1.0
        if mom_21 > 0.02:
            score += 1
        elif mom_21 < -0.02:
            score -= 1

    # Signal 4: VIX < 25
    if vix_data is not None:
        try:
            vix_idx = vix_data.index.get_loc(date)
            vix_level = vix_data['close'].iloc[vix_idx]
            if vix_level < 20:
                score += 1
            elif vix_level > 30:
                score -= 1
        except (KeyError, IndexError):
            pass

    if score >= 2:
        regime = 'bull'
    elif score <= -2:
        regime = 'bear'
    else:
        regime = 'neutral'

    return regime, score


def compute_vol_regime(spy, date, window=21):
    """Realized vol for risk-parity variant."""
    try:
        idx = spy.index.get_loc(date)
    except KeyError:
        return 0.20
    if idx < window + 1:
        return 0.20
    close = spy['close'].iloc[max(0, idx-window):idx+1]
    rets = np.diff(np.log(close.values))
    return np.std(rets) * np.sqrt(252)


def pick_best_momentum(frames, etf_list, date, lookback=63):
    """Pick ETF with best recent momentum."""
    best_ticker = None
    best_ret = -999

    for etf in etf_list:
        if etf not in frames:
            continue
        if date not in frames[etf].index:
            continue
        try:
            idx = frames[etf].index.get_loc(date)
            if idx < lookback:
                continue
            ret = (frames[etf]['close'].iloc[idx] / frames[etf]['close'].iloc[idx - lookback]) - 1.0
            if ret > best_ret:
                best_ret = ret
                best_ticker = etf
        except (KeyError, IndexError):
            continue

    return best_ticker


# ============================================================
# BACKTEST ENGINE
# ============================================================

def backtest_variant(key, cfg, frames, avail_bull, avail_bear):
    """Run a single variant."""
    spy = frames.get('SPY')
    vix = frames.get('^VIX')
    if spy is None:
        return None

    dates = spy.index[250:]  # skip warmup
    equity = STARTING_CAPITAL
    eq_curve = [equity]
    eq_dates = [dates[0]]
    trades = []
    current_holding = None  # (ticker, n_shares, entry_price, entry_date)
    trailing_peak = 0

    rebal_freq = cfg.get('rebal_freq', 21)

    for i, date in enumerate(dates):
        # Determine regime
        regime_name = cfg['regime_func'].__name__ if hasattr(cfg['regime_func'], '__name__') else 'unknown'

        if cfg.get('multi_signal'):
            regime, score = cfg['regime_func'](spy, vix, date)
        else:
            regime = cfg['regime_func'](spy, date)
            score = 0

        # Should we rebalance?
        should_rebal = (i % rebal_freq == 0)

        # Trailing stop check
        if current_holding and cfg.get('trailing_stop_pct'):
            ticker, n_shares, entry_price, entry_date = current_holding
            if ticker in frames and date in frames[ticker].index:
                current_price = frames[ticker].loc[date, 'close']
                if current_price > trailing_peak:
                    trailing_peak = current_price
                if trailing_peak > 0:
                    dd = (trailing_peak - current_price) / trailing_peak
                    if dd > cfg['trailing_stop_pct']:
                        # Trigger stop
                        pnl = n_shares * (current_price - entry_price)
                        equity += pnl
                        trades.append({
                            'ticker': ticker, 'entry': str(entry_date)[:10],
                            'exit': str(date)[:10], 'pnl': round(pnl, 2),
                            'reason': 'trailing_stop', 'regime': regime
                        })
                        current_holding = None
                        trailing_peak = 0
                        should_rebal = True

        # Determine target holding
        if should_rebal or current_holding is None:
            target = None

            if cfg.get('adaptive_leverage'):
                # Adaptive: choose 1x/3x based on regime
                vol = compute_vol_regime(spy, date)
                if regime == 'bull' and vol < 0.20:
                    target = pick_best_momentum(frames, avail_bull, date) or 'TQQQ'
                elif regime == 'bull':
                    target = 'SPY'  # 1x in high-vol bull
                else:
                    target = None  # cash

            elif cfg.get('vol_scaling'):
                # Risk parity: scale by inverse vol
                vol = compute_vol_regime(spy, date)
                target_vol = 0.15
                if vol > 0.05:
                    leverage_ratio = min(target_vol / vol, 3.0)
                    if leverage_ratio > 2.0:
                        target = pick_best_momentum(frames, avail_bull, date) or 'TQQQ'
                    elif leverage_ratio > 1.0:
                        target = 'SPY'
                    else:
                        target = None  # cash

            elif cfg.get('bear_profit'):
                # Go inverse in bear regime
                if regime == 'bull':
                    target = cfg.get('bull_default', 'TQQQ')
                    if cfg.get('pick_best'):
                        target = pick_best_momentum(frames, avail_bull, date) or target
                elif regime == 'bear':
                    target = pick_best_momentum(frames, avail_bear, date) or 'SQQQ'
                else:
                    target = None  # cash

            elif cfg.get('pick_best'):
                # Pick best momentum leveraged ETF, with regime filter
                if regime == 'bull':
                    target = pick_best_momentum(frames, avail_bull, date) or cfg.get('bull_default', 'TQQQ')
                elif regime == 'neutral' and cfg.get('neutral_hold'):
                    target = 'SPY'
                else:
                    target = None  # cash

            else:
                # Simple: fixed ETF based on regime
                if regime == 'bull':
                    target = cfg.get('bull_default', 'TQQQ')
                else:
                    target = None  # cash

            # Execute rebalance if target changed
            current_ticker = current_holding[0] if current_holding else None
            if target != current_ticker:
                # Sell current
                if current_holding:
                    ticker, n_shares, entry_price, entry_date = current_holding
                    if ticker in frames and date in frames[ticker].index:
                        current_price = frames[ticker].loc[date, 'close']
                        pnl = n_shares * (current_price - entry_price)
                        equity += pnl
                        trades.append({
                            'ticker': ticker, 'entry': str(entry_date)[:10],
                            'exit': str(date)[:10], 'pnl': round(pnl, 2),
                            'reason': 'rebalance', 'regime': regime
                        })
                    current_holding = None
                    trailing_peak = 0

                # Buy new
                if target and target in frames and date in frames[target].index:
                    price = frames[target].loc[date, 'close']
                    if price > 0 and equity > 1:
                        n_shares = equity / price  # fractional shares on RH
                        current_holding = (target, n_shares, price, date)
                        trailing_peak = price

        # Mark to market
        if current_holding:
            ticker, n_shares, entry_price, _ = current_holding
            if ticker in frames and date in frames[ticker].index:
                current_price = frames[ticker].loc[date, 'close']
                current_equity = equity + n_shares * (current_price - entry_price)
            else:
                current_equity = equity
        else:
            current_equity = equity

        eq_curve.append(current_equity)
        eq_dates.append(date)

    # Close final position
    if current_holding:
        ticker, n_shares, entry_price, entry_date = current_holding
        final_date = dates[-1]
        if ticker in frames and final_date in frames[ticker].index:
            current_price = frames[ticker].loc[final_date, 'close']
            pnl = n_shares * (current_price - entry_price)
            equity += pnl
            trades.append({
                'ticker': ticker, 'entry': str(entry_date)[:10],
                'exit': str(final_date)[:10], 'pnl': round(pnl, 2),
                'reason': 'final', 'regime': 'final'
            })

    return {
        'equity_curve': eq_curve,
        'equity_dates': [str(d)[:10] for d in eq_dates],
        'trades': trades,
        'final_equity': eq_curve[-1] if eq_curve else STARTING_CAPITAL,
    }


# ============================================================
# METRICS & VALIDATION
# ============================================================

def compute_metrics(result, key, cfg):
    if not result or not result['equity_curve'] or len(result['equity_curve']) < 20:
        return {'variant': key, 'name': cfg.get('name', ''), 'sharpe': 0, 'total_trades': 0}

    eq = np.array(result['equity_curve'])
    rets = np.diff(eq) / np.maximum(eq[:-1], 1e-8)
    rets = rets[np.isfinite(rets)]
    trades = result['trades']

    mean_r = np.mean(rets)
    std_r = np.std(rets)
    sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0

    down_r = rets[rets < 0]
    down_std = np.std(down_r) if len(down_r) > 0 else std_r
    sortino = (mean_r / down_std) * np.sqrt(252) if down_std > 0 else 0

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-8)
    mdd = np.min(dd) * 100

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) * 100 if pnls else 0
    gw = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0
    pf = gw / gl if gl > 0 else (999 if gw > 0 else 0)

    n_years = len(rets) / 252
    total_ret = eq[-1] / max(eq[0], 1e-8)
    cagr = (total_ret ** (1 / max(n_years, 0.1)) - 1) * 100 if total_ret > 0 else 0

    # Time in market
    n_invested = sum(1 for r in rets if abs(r) > 1e-8)
    time_in_mkt = n_invested / len(rets) * 100 if len(rets) > 0 else 0

    return {
        'variant': key, 'name': cfg.get('name', ''),
        'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'profit_factor': round(pf, 2), 'win_rate': round(wr, 1),
        'max_drawdown_pct': round(mdd, 2), 'cagr_pct': round(cagr, 1),
        'total_return_pct': round((total_ret - 1) * 100, 1),
        'final_equity': round(eq[-1], 2), 'total_trades': len(trades),
        'time_in_market_pct': round(time_in_mkt, 1),
    }


def permutation_test(result, frames, avail_bull, avail_bear, cfg, n_perms=N_PERMUTATIONS):
    """Shuffle regime assignments and re-run."""
    actual_m = compute_metrics(result, 'actual', cfg)
    actual_sharpe = actual_m['sharpe']
    print(f"    Running {n_perms} permutation tests (actual Sharpe={actual_sharpe:.3f})...", flush=True)

    better = 0
    rand_sharpes = []

    for _ in range(n_perms):
        # Create a randomized regime function that randomly picks bull/bear/neutral
        class RandomRegime:
            def __init__(self):
                self.cache = {}
            def __call__(self, spy, date, *args):
                if date not in self.cache:
                    self.cache[date] = np.random.choice(['bull', 'bear', 'neutral'], p=[0.6, 0.2, 0.2])
                return self.cache[date]

        rand_cfg = dict(cfg)
        rand_regime = RandomRegime()
        rand_cfg['regime_func'] = rand_regime
        rand_cfg['multi_signal'] = False

        rand_result = backtest_variant('rand', rand_cfg, frames, avail_bull, avail_bear)
        if rand_result:
            rm = compute_metrics(rand_result, 'rand', cfg)
            rand_sharpes.append(rm['sharpe'])
            if rm['sharpe'] >= actual_sharpe:
                better += 1

    p = better / max(n_perms, 1)
    rand_mean = np.mean(rand_sharpes) if rand_sharpes else 0
    return p, actual_sharpe, rand_mean


def regime_balance(result, spy):
    """Sharpe in bull vs bear SPY regimes."""
    if not result or len(result['equity_curve']) < 50:
        return 1.0

    eq = np.array(result['equity_curve'])
    dates = pd.to_datetime(result['equity_dates'])

    close = spy['close']
    ma50 = close.rolling(50).mean()

    bull_r, bear_r = [], []
    for i in range(1, min(len(eq), len(dates))):
        r = (eq[i] - eq[i-1]) / max(eq[i-1], 1e-8)
        d = dates[i]
        if d in close.index and d in ma50.index:
            if close.loc[d] > ma50.loc[d]:
                bull_r.append(r)
            else:
                bear_r.append(r)

    if len(bull_r) < 20 or len(bear_r) < 20:
        return 0.3

    bs = (np.mean(bull_r) / np.std(bull_r)) * np.sqrt(252) if np.std(bull_r) > 0 else 0
    brs = (np.mean(bear_r) / np.std(bear_r)) * np.sqrt(252) if np.std(bear_r) > 0 else 0

    mx = max(abs(bs), abs(brs))
    gap = abs(bs - brs) / mx if mx > 0 else 0
    print(f"    Regime: Bull Sharpe={bs:.3f}, Bear Sharpe={brs:.3f}, Gap={gap:.3f}", flush=True)
    return gap


# ============================================================
# VARIANTS
# ============================================================

VARIANTS = {
    'A': {
        'name': 'Simple MA200 → TQQQ/Cash',
        'regime_func': detect_regime_simple_ma200,
        'bull_default': 'TQQQ',
        'rebal_freq': 1,  # daily check
    },
    'B': {
        'name': 'Golden Cross → TQQQ/Cash',
        'regime_func': detect_regime_golden_cross,
        'bull_default': 'TQQQ',
        'rebal_freq': 1,
    },
    'C': {
        'name': 'Multi-Signal → TQQQ/Cash',
        'regime_func': detect_regime_multi_signal,
        'multi_signal': True,
        'bull_default': 'TQQQ',
        'rebal_freq': 5,
    },
    'D': {
        'name': 'Vol-Scaled Leverage',
        'regime_func': detect_regime_multi_signal,
        'multi_signal': True,
        'vol_scaling': True,
        'rebal_freq': 5,
    },
    'E': {
        'name': 'Adaptive 1x/3x',
        'regime_func': detect_regime_multi_signal,
        'multi_signal': True,
        'adaptive_leverage': True,
        'rebal_freq': 5,
    },
    'F': {
        'name': 'Best Momentum + Regime',
        'regime_func': detect_regime_multi_signal,
        'multi_signal': True,
        'pick_best': True,
        'bull_default': 'TQQQ',
        'rebal_freq': 21,
    },
    'G': {
        'name': 'Bull/Bear Profiter',
        'regime_func': detect_regime_multi_signal,
        'multi_signal': True,
        'bear_profit': True,
        'pick_best': True,
        'bull_default': 'TQQQ',
        'rebal_freq': 5,
    },
    'H': {
        'name': 'TQQQ + 12% Trail Stop',
        'regime_func': detect_regime_golden_cross,
        'bull_default': 'TQQQ',
        'trailing_stop_pct': 0.12,
        'rebal_freq': 1,
    },
}


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = datetime.now()

    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://jupiter:5000')
            mlflow.set_experiment('regime_leveraged_rotation_v1')
            print("MLflow OK", flush=True)
        except Exception as e:
            print(f"MLflow: {e}", flush=True)

    print("=" * 70, flush=True)
    print("  REGIME-SWITCHING LEVERAGED ETF STRATEGY V1", flush=True)
    print("  Key insight: 30%+ CAGR with leveraged ETFs is real,", flush=True)
    print("  but -82% MDD in bear markets destroys accounts.", flush=True)
    print("  Solution: regime filter to avoid bear markets.", flush=True)
    print("=" * 70, flush=True)

    frames, avail_bull, avail_bear = load_data()
    spy = frames.get('SPY')
    if spy is None:
        print("ERROR: no SPY data")
        return

    # Date ranges
    for etf in ['TQQQ', 'SOXL', 'UPRO', 'SPY']:
        if etf in frames:
            d = frames[etf]
            print(f"  {etf}: {d.index[0].date()} to {d.index[-1].date()}, {len(d)} days", flush=True)

    all_metrics = {}

    for vkey in sorted(VARIANTS.keys()):
        vcfg = VARIANTS[vkey]
        print(f"\n{'='*60}", flush=True)
        print(f"  VARIANT {vkey}: {vcfg['name']}", flush=True)
        print(f"{'='*60}", flush=True)

        result = backtest_variant(vkey, vcfg, frames, avail_bull, avail_bear)
        if not result:
            print("  No result", flush=True)
            continue

        m = compute_metrics(result, vkey, vcfg)
        print(f"  Trades: {m['total_trades']} | Sharpe: {m['sharpe']} | Sortino: {m['sortino']} | "
              f"PF: {m['profit_factor']} | WR: {m['win_rate']}% | MDD: {m['max_drawdown_pct']}%", flush=True)
        print(f"  CAGR: {m['cagr_pct']}% | $645→${m['final_equity']} | "
              f"Time in market: {m.get('time_in_market_pct', 0)}%", flush=True)

        # 5-Gate validation
        print(f"  Running 5-gate validation...", flush=True)

        g1 = m['sharpe'] > 1.0
        p_val, act_s, rand_s = permutation_test(result, frames, avail_bull, avail_bear, vcfg)
        g2 = p_val < 0.05
        g3 = m['win_rate'] > 40.0
        gap = regime_balance(result, spy)
        g4 = gap < 0.50
        g5 = act_s > rand_s

        gates = sum([g1, g2, g3, g4, g5])
        print(f"  5-Gate: {gates}/5 PASS", flush=True)
        print(f"    sharpe_gt_1: {'PASS' if g1 else 'FAIL'} ({m['sharpe']})", flush=True)
        print(f"    perm_p: {'PASS' if g2 else 'FAIL'} (p={p_val:.3f})", flush=True)
        print(f"    wr_gt_40: {'PASS' if g3 else 'FAIL'} ({m['win_rate']}%)", flush=True)
        print(f"    regime_bal: {'PASS' if g4 else 'FAIL'} (gap={gap:.3f})", flush=True)
        print(f"    beats_random: {'PASS' if g5 else 'FAIL'} (act={act_s:.3f} vs rand={rand_s:.3f})", flush=True)

        m['gates_passed'] = gates
        m['perm_p'] = round(p_val, 4)
        m['regime_gap'] = round(gap, 3)
        m['random_sharpe'] = round(rand_s, 3)
        all_metrics[vkey] = m

    # Summary
    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n{'='*70}", flush=True)
    print(f"  SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    best_v = None
    best_s = -999

    for vk in sorted(all_metrics.keys()):
        m = all_metrics[vk]
        g = m.get('gates_passed', 0)
        mk = "✅" if g >= 4 else "❌"
        print(f"  {mk} {vk}: {m['name']:30s} Sharpe={m['sharpe']:6.3f}  "
              f"${645}→${m['final_equity']:>10.2f}  CAGR={m['cagr_pct']:5.1f}%  "
              f"MDD={m['max_drawdown_pct']:6.1f}%  Gates={g}/5  p={m.get('perm_p', 1):.3f}",
              flush=True)
        if m['sharpe'] > best_s and g >= 3:
            best_s = m['sharpe']
            best_v = vk

    if best_v:
        bm = all_metrics[best_v]
        print(f"\n  BEST: {best_v} ({bm['name']}) — Sharpe {bm['sharpe']}, "
              f"CAGR {bm['cagr_pct']}%, MDD {bm['max_drawdown_pct']}%", flush=True)
    else:
        print(f"\n  VERDICT: No variant achieved 3+ gates", flush=True)

    print(f"\nRuntime: {elapsed:.0f}s", flush=True)

    # Save
    rpath = os.path.join(OUTPUT_DIR, 'results.json')
    with open(rpath, 'w') as f:
        json.dump({'metrics': all_metrics, 'runtime': elapsed}, f, indent=2, default=str)

    # MLflow
    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name=f"regime_lev_{datetime.now():%Y%m%d_%H%M}"):
                for vk, m in all_metrics.items():
                    for mk, mv in m.items():
                        if isinstance(mv, (int, float)):
                            mlflow.log_metric(f"{vk}_{mk}", mv)
                if best_v:
                    mlflow.log_metric("best_sharpe", best_s)
                    mlflow.log_param("best_variant", best_v)
                mlflow.log_artifact(rpath)
            print("MLflow logged", flush=True)
        except Exception as e:
            print(f"MLflow error: {e}", flush=True)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
