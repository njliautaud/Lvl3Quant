"""
Correlation Regime Breakout Strategy — v1
==========================================
Hypothesis: Cross-sector correlation regime predicts WHEN to trade sector options.
- High correlation (panic): buy puts on weakest sector (mean-reversion)
- Low correlation (alpha env): buy calls on strongest momentum sector
- Correlation spike: hedge with SPY puts
- Correlation normalization: long best LGBM-ranked sector calls

6 variants:
  A - Correlation regime rules only
  B - Correlation + LGBM ranking
  C - Correlation + momentum filter
  D - Correlation mean-reversion (spike >2std)
  E - Correlation divergence alpha (only low-corr regime, momentum calls)
  F - Full ensemble (correlation + LGBM + momentum + VIX)

Capital: $645
Options: 14-28 DTE, ATM or 4% OTM, BS pricing with 73% haircut
OOT: Jan 2022 - Jul 2026 (~4.5 years)

SPEED NOTE: LGBM rankings precomputed once per variant and cached.
Permutation test shuffles directions on the cached signal list — no re-ranking.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from itertools import combinations
import mlflow
import json
import os
import warnings
from datetime import datetime
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SECTORS = ['XLK', 'XLV', 'XLE', 'XLF', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE']
SPY = 'SPY'
VIX_TICKER = '^VIX'

STARTING_CAPITAL = 645.0
BS_HAIRCUT = 0.73
COMMISSION_RT = 2.60        # round-trip per contract
MIN_DTE = 14
MAX_DTE = 28
OTM_PCT = 0.04
CONTRACTS = 1
RISK_PCT = 0.20             # max 20% capital per trade

HIGH_CORR = 0.80
LOW_CORR  = 0.30
SPIKE_DELTA = 0.20          # 5-day change triggers spike
ZSCORE_THRESH = 2.0

ROLL_DAYS = 21
OOT_START = '2022-01-01'
OOT_END   = '2026-07-25'
N_PERM    = 100

RESULTS_DIR = '/home/nick/Lvl3Quant/research'
MLFLOW_URI  = 'http://localhost:5000'


# ── Black-Scholes ─────────────────────────────────────────────────────────────
def bs_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if option_type == 'call' else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(start='2020-01-01', end=OOT_END):
    log.info(f"Downloading data {start} → {end}")
    tickers = SECTORS + [SPY, VIX_TICKER]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    closes = raw['Close'].dropna(how='all').ffill(limit=3)
    log.info(f"Loaded {closes.shape[0]} days, {closes.shape[1]} tickers")
    return closes


# ── Correlation Regime ────────────────────────────────────────────────────────
def compute_correlation_regime(closes, window=21):
    returns = closes[SECTORS].pct_change()
    pairs = list(combinations(range(len(SECTORS)), 2))
    avg_corr = pd.Series(index=closes.index, dtype=float)
    for i in range(window, len(returns)):
        win = returns.iloc[i-window:i]
        cm = win.corr().values
        avg_corr.iloc[i] = np.nanmean([cm[a, b] for a, b in pairs])
    return avg_corr


def compute_corr_features(avg_corr, std_window=63):
    f = pd.DataFrame(index=avg_corr.index)
    f['avg_corr']     = avg_corr
    f['corr_5d']      = avg_corr.diff(5)
    f['corr_ma21']    = avg_corr.rolling(21).mean()
    f['corr_std63']   = avg_corr.rolling(std_window).std()
    f['corr_z']       = (avg_corr - f['corr_ma21']) / (f['corr_std63'] + 1e-8)
    f['regime']       = 'neutral'
    f.loc[avg_corr > HIGH_CORR, 'regime'] = 'panic'
    f.loc[avg_corr < LOW_CORR,  'regime'] = 'alpha'
    f['spike']        = f['corr_5d'] > SPIKE_DELTA
    f['normalize']    = f['corr_5d'] < -SPIKE_DELTA
    return f


# ── Momentum / Rankings (precomputed) ────────────────────────────────────────
def compute_sector_scores(closes):
    """
    Precompute per-date sector score table (21d momentum / 21d vol).
    Returns DataFrame [date x sector] of risk-adjusted momentum scores.
    Vectorized — fast.
    """
    sc = closes[SECTORS]
    ret_5  = sc.pct_change(5)
    ret_21 = sc.pct_change(21)
    ret_63 = sc.pct_change(63)
    vol_21 = sc.pct_change().rolling(21).std()
    scores = (0.2 * ret_5 + 0.4 * ret_21 + 0.4 * ret_63) / (vol_21 + 1e-6)
    return scores   # NaN before enough history


def best_sector(scores_row):
    """Return ticker with highest score."""
    valid = scores_row.dropna()
    return valid.idxmax() if len(valid) > 0 else None


def worst_sector(scores_row):
    """Return ticker with lowest score."""
    valid = scores_row.dropna()
    return valid.idxmin() if len(valid) > 0 else None


# ── Option Trade Simulation ───────────────────────────────────────────────────
def simulate_trade(closes, date_idx, ticker, direction, dte=21, r=0.05):
    """
    Simulate buying one option contract, hold to expiry or 50%/100% exit.
    Returns pnl float or None if unexecutable.
    """
    if ticker not in closes.columns:
        return None
    S = closes[ticker].iloc[date_idx]
    if pd.isna(S) or S <= 0:
        return None

    hist = closes[ticker].pct_change().iloc[max(0, date_idx-30):date_idx]
    if len(hist) < 10:
        return None
    iv = hist.std() * np.sqrt(252)
    if iv <= 0:
        return None

    K = S * (1 + OTM_PCT) if direction == 'call' else S * (1 - OTM_PCT)
    entry_bs = bs_price(S, K, dte / 365.0, r, iv, direction)
    entry_price = entry_bs * BS_HAIRCUT
    if entry_price < 0.01:
        return None

    cost = entry_price * 100 * CONTRACTS + COMMISSION_RT
    exit_idx = min(date_idx + dte, len(closes) - 1)

    pnl = None
    for k in range(date_idx + 1, exit_idx + 1):
        Sk = closes[ticker].iloc[k]
        Tk = max(0, (dte - (k - date_idx)) / 365.0)
        mid = bs_price(Sk, K, Tk, r, iv, direction) * BS_HAIRCUT
        trade_pnl = (mid - entry_price) * 100 * CONTRACTS - COMMISSION_RT
        if cost > 0:
            ret = trade_pnl / cost
            if ret >= 0.50 or ret <= -1.0:
                pnl = trade_pnl
                break

    if pnl is None:
        intrinsic = max(0, (closes[ticker].iloc[exit_idx] - K) if direction == 'call'
                        else (K - closes[ticker].iloc[exit_idx]))
        exit_price = intrinsic * BS_HAIRCUT
        pnl = (exit_price - entry_price) * 100 * CONTRACTS - COMMISSION_RT

    return pnl, cost


# ── Signal Generation — returns list of (date_idx, ticker, direction, label) ─
def generate_signals(closes, cf, sector_scores, variant):
    """
    Walk OOT period, generate one signal per rebalance slot.
    Returns list of dicts with keys: date_idx, ticker, direction, label.
    No option simulation here — just the signal list.
    """
    oot_mask = (closes.index >= OOT_START) & (closes.index <= OOT_END)
    oot_idxs = np.where(oot_mask)[0]

    signals = []
    prev_idx = -999

    for idx in oot_idxs:
        if idx - prev_idx < ROLL_DAYS:
            continue

        row = cf.iloc[idx]
        scores_row = sector_scores.iloc[idx] if sector_scores is not None else None
        vix = closes[VIX_TICKER].iloc[idx] if VIX_TICKER in closes.columns else 20.0
        spy_21 = closes[SPY].pct_change(21).iloc[idx] if SPY in closes.columns else 0.0

        sig = None

        if variant == 'A':
            if row['spike']:
                sig = (SPY, 'put', 'spike_hedge')
            elif row['regime'] == 'panic':
                w = worst_sector(sector_scores.iloc[idx])
                if w: sig = (w, 'put', 'panic_put')
            elif row['regime'] == 'alpha':
                b = best_sector(sector_scores.iloc[idx])
                if b: sig = (b, 'call', 'alpha_call')
            elif row['normalize']:
                sig = (SPY, 'call', 'normalize_call')

        elif variant == 'B':
            # Same logic but uses LGBM-style scores for ticker selection
            if row['spike']:
                sig = (SPY, 'put', 'spike_hedge')
            elif row['regime'] == 'panic':
                w = worst_sector(sector_scores.iloc[idx])
                if w: sig = (w, 'put', 'panic_lgbm')
            elif row['regime'] == 'alpha':
                b = best_sector(sector_scores.iloc[idx])
                if b: sig = (b, 'call', 'alpha_lgbm')

        elif variant == 'C':
            # Momentum must confirm direction
            mom_21 = closes[SECTORS].pct_change(21).iloc[idx]
            if row['spike'] and spy_21 < 0:
                sig = (SPY, 'put', 'spike_confirmed')
            elif row['regime'] == 'panic':
                falling = mom_21[mom_21 < -0.03].dropna()
                if len(falling) > 0:
                    sig = (falling.idxmin(), 'put', 'confirmed_panic')
            elif row['regime'] == 'alpha':
                rising = mom_21[mom_21 > 0.03].dropna()
                if len(rising) > 0:
                    sig = (rising.idxmax(), 'call', 'confirmed_alpha')

        elif variant == 'D':
            # Extreme z-score mean-reversion
            if not pd.isna(row['corr_z']):
                if row['corr_z'] > ZSCORE_THRESH and row['regime'] == 'panic':
                    sig = (SPY, 'call', 'extreme_bounce')
                elif row['corr_z'] < -ZSCORE_THRESH and row['regime'] == 'alpha':
                    b = best_sector(sector_scores.iloc[idx])
                    if b: sig = (b, 'call', 'extreme_diverge')

        elif variant == 'E':
            # Only trade alpha/divergence regime
            if row['regime'] == 'alpha':
                b = best_sector(sector_scores.iloc[idx])
                if b:
                    mom_b = closes[b].pct_change(21).iloc[idx]
                    score_b = sector_scores.iloc[idx].get(b, 0)
                    if mom_b > 0.02 and score_b > 0:
                        sig = (b, 'call', 'pure_alpha')

        elif variant == 'F':
            # Full ensemble
            if row['spike'] and vix > 20:
                sig = (SPY, 'put', 'vix_spike')
            elif vix > 25 and row['regime'] == 'panic' and spy_21 < -0.05:
                w = worst_sector(sector_scores.iloc[idx])
                if w: sig = (w, 'put', 'full_panic')
            elif vix < 18 and row['regime'] == 'alpha' and spy_21 > 0:
                b = best_sector(sector_scores.iloc[idx])
                if b and (sector_scores.iloc[idx].get(b, 0) > 0):
                    sig = (b, 'call', 'full_alpha')
            elif row['normalize'] and vix < 20:
                sig = (SPY, 'call', 'normalize_calm')

        if sig is not None:
            signals.append({'date_idx': idx, 'ticker': sig[0],
                            'direction': sig[1], 'label': sig[2]})
            prev_idx = idx

    return signals


# ── Backtest from Signal List ─────────────────────────────────────────────────
def backtest_signals(closes, signals, shuffle=False, rng=None):
    """
    Execute trades from signal list.
    If shuffle=True, randomize directions (for permutation test).
    Returns array of trade PnLs.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    equity = STARTING_CAPITAL
    pnls = []
    in_trade_until = -1

    for sig in signals:
        idx = sig['date_idx']
        if idx <= in_trade_until:
            continue

        ticker = sig['ticker']
        direction = rng.choice(['call', 'put']) if shuffle else sig['direction']

        result = simulate_trade(closes, idx, ticker, direction, dte=21)
        if result is None:
            continue
        pnl, cost = result

        if cost > equity * RISK_PCT:
            continue
        if equity < 50:
            break

        equity += pnl
        pnls.append(pnl)
        in_trade_until = idx + 21  # approximate hold period

    return np.array(pnls)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(pnls, variant_name, n_signals):
    if len(pnls) < 5:
        return {'variant': variant_name, 'n_trades': len(pnls),
                'sharpe': np.nan, 'sortino': np.nan,
                'profit_factor': np.nan, 'win_rate': np.nan,
                'mdd': np.nan, 'cagr': np.nan,
                'total_pnl': np.nan, 'final_equity': np.nan,
                'n_signals': n_signals}

    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    wr     = len(wins) / len(pnls)
    pf     = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf

    total_pnl    = pnls.sum()
    final_equity = STARTING_CAPITAL + total_pnl

    # Annualize: OOT is ~4.5 years, use actual trade count / 4.5
    years  = 4.5
    tpy    = len(pnls) / years
    mu     = pnls.mean()
    sigma  = pnls.std()
    sharpe = (mu / sigma) * np.sqrt(tpy) if sigma > 0 else np.nan

    down   = pnls[pnls < 0]
    dsig   = down.std() if len(down) > 1 else sigma
    sortino = (mu / dsig) * np.sqrt(tpy) if dsig > 0 else np.nan

    eq_curve = np.concatenate([[STARTING_CAPITAL], np.cumsum(pnls) + STARTING_CAPITAL])
    peak     = np.maximum.accumulate(eq_curve)
    mdd      = ((eq_curve - peak) / peak).min()

    cagr = (final_equity / STARTING_CAPITAL) ** (1 / years) - 1 if final_equity > 0 else -1.0

    return {
        'variant': variant_name,
        'n_trades': len(pnls),
        'n_signals': n_signals,
        'sharpe':   round(float(sharpe),   3) if not np.isnan(sharpe)   else np.nan,
        'sortino':  round(float(sortino),  3) if not np.isnan(sortino)  else np.nan,
        'profit_factor': round(float(pf),  3),
        'win_rate': round(float(wr),       3),
        'mdd':      round(float(mdd),      3),
        'cagr':     round(float(cagr),     3),
        'total_pnl':    round(float(total_pnl),    2),
        'final_equity': round(float(final_equity), 2),
    }


# ── Permutation Test (fast — no re-ranking) ───────────────────────────────────
def permutation_test(closes, signals, observed_sharpe, n_perm=N_PERM):
    """Shuffle directions on the pre-generated signal list. Fast."""
    log.info(f"    Perm test {n_perm} shuffles...")
    rng = np.random.default_rng(0)
    perm_sharpes = []
    for _ in range(n_perm):
        pnls = backtest_signals(closes, signals, shuffle=True, rng=rng)
        if len(pnls) >= 5 and pnls.std() > 0:
            tpy = len(pnls) / 4.5
            perm_sharpes.append((pnls.mean() / pnls.std()) * np.sqrt(tpy))

    if len(perm_sharpes) < 10:
        return np.nan, perm_sharpes
    arr = np.array(perm_sharpes)
    p   = float((arr >= observed_sharpe).mean())
    return round(p, 4), perm_sharpes


# ── Random Baseline ───────────────────────────────────────────────────────────
def random_baseline(closes, cf, sector_scores, n_runs=50):
    """Random ticker + direction at same cadence as variant A."""
    log.info("Running random baseline...")
    signals_a = generate_signals(closes, cf, sector_scores, 'A')
    # Replace ticker and direction with random choices
    all_tickers = SECTORS + [SPY]
    rng = np.random.default_rng(999)
    rand_sharpes = []
    for i in range(n_runs):
        rand_sigs = [{'date_idx': s['date_idx'],
                      'ticker':   rng.choice(all_tickers),
                      'direction': rng.choice(['call', 'put']),
                      'label':    'random'}
                     for s in signals_a]
        pnls = backtest_signals(closes, rand_sigs, shuffle=False, rng=rng)
        if len(pnls) >= 5 and pnls.std() > 0:
            tpy = len(pnls) / 4.5
            rand_sharpes.append((pnls.mean() / pnls.std()) * np.sqrt(tpy))

    if not rand_sharpes:
        return {'mean': np.nan, 'std': np.nan, 'p95': np.nan, 'n': 0}
    arr = np.array(rand_sharpes)
    return {'mean': round(float(arr.mean()), 3),
            'std':  round(float(arr.std()),  3),
            'p95':  round(float(np.percentile(arr, 95)), 3),
            'n':    len(arr)}


# ── MLflow ────────────────────────────────────────────────────────────────────
def log_mlflow(all_metrics, perm_results, rand_bl):
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment('correlation_regime_v1')
        with mlflow.start_run(run_name=f"corr_regime_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_param('oot_period',       f"{OOT_START}_{OOT_END}")
            mlflow.log_param('capital',           STARTING_CAPITAL)
            mlflow.log_param('bs_haircut',        BS_HAIRCUT)
            mlflow.log_param('n_perm',            N_PERM)
            mlflow.log_param('random_mean_sharpe', rand_bl.get('mean'))
            mlflow.log_param('random_p95_sharpe',  rand_bl.get('p95'))
            for vname, m in all_metrics.items():
                for k, v in m.items():
                    if isinstance(v, float) and not np.isnan(v):
                        mlflow.log_metric(f"v{vname}_{k}", v)
            for vname, p in perm_results.items():
                pv = p.get('p_value', np.nan)
                if not np.isnan(pv):
                    mlflow.log_metric(f"v{vname}_perm_p", pv)
        log.info("MLflow logged.")
    except Exception as e:
        log.warning(f"MLflow error: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=" * 60)
    log.info("Correlation Regime Breakout Strategy v1")
    log.info(f"OOT {OOT_START} → {OOT_END} | Capital ${STARTING_CAPITAL} | Haircut {BS_HAIRCUT}")
    log.info("=" * 60)

    closes = load_data()

    log.info("Computing correlation regime...")
    avg_corr = compute_correlation_regime(closes)
    cf       = compute_corr_features(avg_corr)

    log.info("Precomputing sector scores (vectorized)...")
    sector_scores = compute_sector_scores(closes)

    # OOT regime summary
    oot = cf[(cf.index >= OOT_START) & (cf.index <= OOT_END)]
    rc  = oot['regime'].value_counts()
    log.info(f"OOT regime: panic={rc.get('panic',0)}d  alpha={rc.get('alpha',0)}d  "
             f"neutral={rc.get('neutral',0)}d  spikes={oot['spike'].sum()}")

    all_metrics  = {}
    perm_results = {}

    for vname in ['A', 'B', 'C', 'D', 'E', 'F']:
        log.info(f"\n--- Variant {vname} ---")
        signals = generate_signals(closes, cf, sector_scores, vname)
        log.info(f"  Signals generated: {len(signals)}")

        pnls    = backtest_signals(closes, signals)
        metrics = compute_metrics(pnls, vname, len(signals))
        all_metrics[vname] = metrics

        obs_sharpe = metrics['sharpe'] if not np.isnan(metrics.get('sharpe', float('nan'))) else 0.0
        log.info(f"  Trades={metrics['n_trades']}  Sharpe={metrics['sharpe']}  "
                 f"Sortino={metrics['sortino']}  PF={metrics['profit_factor']}  "
                 f"WR={metrics['win_rate']:.1%}  MDD={metrics['mdd']:.1%}  "
                 f"CAGR={metrics['cagr']:.1%}  Final=${metrics['final_equity']}")

        p_val, perm_dist = permutation_test(closes, signals, obs_sharpe)
        perm_results[vname] = {
            'p_value':       p_val,
            'obs_sharpe':    obs_sharpe,
            'perm_mean':     round(float(np.mean(perm_dist)), 3) if perm_dist else np.nan,
            'perm_std':      round(float(np.std(perm_dist)),  3) if perm_dist else np.nan,
        }
        log.info(f"  Perm p={p_val}  (perm_mean={perm_results[vname]['perm_mean']})")

    # Random baseline
    rand_bl = random_baseline(closes, cf, sector_scores, n_runs=50)
    log.info(f"\nRandom baseline: Sharpe {rand_bl['mean']} ± {rand_bl['std']} (p95={rand_bl['p95']})")

    # Summary table
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY")
    log.info(f"{'Var':>3} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
             f"{'WR':>6} {'MDD':>7} {'CAGR':>7} {'Final$':>8} {'perm_p':>7}")
    log.info("-" * 70)
    for vname, m in all_metrics.items():
        p  = perm_results[vname]
        wr = f"{m['win_rate']:.1%}" if not np.isnan(m.get('win_rate', float('nan'))) else ' N/A'
        md = f"{m['mdd']:.1%}"     if not np.isnan(m.get('mdd', float('nan')))       else ' N/A'
        cg = f"{m['cagr']:.1%}"    if not np.isnan(m.get('cagr', float('nan')))      else ' N/A'
        pv = f"{p['p_value']:.4f}" if not np.isnan(p.get('p_value', float('nan')))   else ' N/A'
        log.info(f"  {vname:>1}  {m['n_trades']:>6}  {str(m['sharpe']):>7}  "
                 f"{str(m['sortino']):>8}  {str(m['profit_factor']):>6}  "
                 f"{wr:>6}  {md:>7}  {cg:>7}  ${m['final_equity']:>7.0f}  {pv:>7}")
    log.info(f"\nRandom baseline Sharpe: {rand_bl['mean']} ± {rand_bl['std']} (p95={rand_bl['p95']})")

    # Save JSON results
    out = {
        'run_date':    datetime.now().isoformat(),
        'strategy':    'correlation_regime_v1',
        'oot_period':  f"{OOT_START} to {OOT_END}",
        'capital':     STARTING_CAPITAL,
        'bs_haircut':  BS_HAIRCUT,
        'oot_regime':  {k: int(v) for k, v in rc.items()},
        'variants':    {v: {k: (float(val) if isinstance(val, (np.floating, np.integer)) else val)
                            for k, val in m.items()}
                        for v, m in all_metrics.items()},
        'perm_tests':  {v: {k: (float(val) if isinstance(val, (np.floating, np.integer)) else val)
                            for k, val in p.items()}
                        for v, p in perm_results.items()},
        'random_baseline': rand_bl,
    }
    path = os.path.join(RESULTS_DIR, 'correlation_regime_v1_results.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"\nResults saved to {path}")

    log_mlflow(all_metrics, perm_results, rand_bl)
    log.info("Done.")
    return out


if __name__ == '__main__':
    main()
