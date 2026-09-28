#!/usr/bin/env python3
"""
ML Sector Momentum v1 — Cross-Sector Rotation with GBM Timing
================================================================
HYPOTHESIS: ML can predict which sectors will outperform over the next 21 days
using cross-sector relative strength, macro features, and sector-specific
momentum/mean-reversion signals.

Unlike broad factor timing (FAIL, entry 498), sector rotation has:
  - More granular opportunity set (11 sectors vs 4-5 factors)
  - Clearer economic driver relationships (rates → utilities/financials, oil → energy)
  - Higher dispersion between winners/losers (more alpha potential)

Strategy:
  1. Universe: 11 SPDR sector ETFs (XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, XLB, XLRE, XLC)
  2. ML predicts: which sectors will be in top-3 over next 21 days
  3. Features: relative momentum, vol, VIX interaction, cross-sector flow, mean reversion
  4. Position: equal-weight top-3 predicted sectors, vol-targeted
  5. Walk-forward: 504d sliding, retrain monthly

Fixed $100K, NO DCA (HC #713). Full adversarial.
"""

import json
import time
import warnings
from pathlib import Path
from functools import partial

print = partial(print, flush=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 100_000
TRAIN_WINDOW     = 504
FWD_HORIZON      = 21           # predict 21-day forward
TOP_K            = 3            # hold top-K predicted sectors
TARGET_VOL       = 0.12
N_PERMUTATIONS   = 50

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_sector_momentum"
OUTPUT.mkdir(parents=True, exist_ok=True)

SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
SECTOR_NAMES = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy',
    'XLV': 'Healthcare', 'XLI': 'Industrials', 'XLY': 'Consumer Disc',
    'XLP': 'Consumer Staples', 'XLU': 'Utilities', 'XLB': 'Materials',
    'XLRE': 'Real Estate', 'XLC': 'Communications'
}


def download_data():
    cache = BASE / "data" / "cache" / "sector_momentum_data.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and (time.time() - cache.stat().st_mtime) < 3600:
        print(f"  Using cached data")
        return pd.read_parquet(cache)

    print("  Downloading sector ETFs + macro data...")
    tickers = SECTORS + ['SPY', '^VIX', 'TLT', 'HYG', 'GLD', 'UUP', 'IEF']
    data = yf.download(tickers, start='2007-01-01', end='2026-07-18',
                       auto_adjust=True, progress=False)
    closes = data['Close'].copy()
    closes.columns = [c.replace('^', '') for c in closes.columns]
    closes = closes.ffill().dropna(how='all')
    closes.to_parquet(cache)
    print(f"  Data: {len(closes)} days, {closes.shape[1]} assets")
    return closes


def build_sector_features(closes):
    """Build per-sector and cross-sector features."""
    available_sectors = [s for s in SECTORS if s in closes.columns]
    returns = closes[available_sectors].pct_change()

    all_features = {}

    for sector in available_sectors:
        feats = pd.DataFrame(index=closes.index)

        # Momentum at multiple horizons
        feats['mom5'] = closes[sector].pct_change(5)
        feats['mom20'] = closes[sector].pct_change(20)
        feats['mom60'] = closes[sector].pct_change(60)
        feats['mom120'] = closes[sector].pct_change(120)

        # Relative momentum (vs SPY)
        if 'SPY' in closes.columns:
            feats['rel_mom20'] = closes[sector].pct_change(20) - closes['SPY'].pct_change(20)
            feats['rel_mom60'] = closes[sector].pct_change(60) - closes['SPY'].pct_change(60)

        # Mean reversion
        feats['zscore_20'] = (closes[sector] - closes[sector].rolling(20).mean()) / \
                             (closes[sector].rolling(20).std() + 1e-8)
        feats['zscore_60'] = (closes[sector] - closes[sector].rolling(60).mean()) / \
                             (closes[sector].rolling(60).std() + 1e-8)

        # Volatility
        feats['vol20'] = returns[sector].rolling(20).std()
        feats['vol60'] = returns[sector].rolling(60).std()
        feats['vol_ratio'] = feats['vol20'] / (feats['vol60'] + 1e-8)

        # Sector rank (relative strength)
        # Rank this sector vs others on 20d momentum
        mom20_all = returns[available_sectors].rolling(20).sum()
        rank = mom20_all.rank(axis=1, pct=True)
        feats['rank_20d'] = rank[sector] if sector in rank.columns else 0.5

        # Macro interactions
        if 'VIX' in closes.columns:
            feats['vix_level'] = closes['VIX']
            feats['vix_x_mom'] = closes['VIX'] * feats['mom20']  # interaction

        if 'TLT' in closes.columns:
            feats['tlt_mom20'] = closes['TLT'].pct_change(20)  # rate sensitivity

        if 'HYG' in closes.columns:
            feats['hyg_mom20'] = closes['HYG'].pct_change(20)  # credit conditions

        # Cross-sector: sector's correlation to SPY (beta proxy)
        if 'SPY' in closes.columns:
            rolling_corr = returns[sector].rolling(60).corr(closes['SPY'].pct_change())
            feats['spy_corr'] = rolling_corr

        all_features[sector] = feats

    return all_features, available_sectors


def run_sector_momentum(closes, all_features, available_sectors):
    """Walk-forward ML sector rotation."""
    returns = closes[available_sectors].pct_change()

    # Compute forward returns for labeling
    fwd_returns = {}
    for s in available_sectors:
        fwd_returns[s] = returns[s].rolling(FWD_HORIZON).sum().shift(-FWD_HORIZON)
    fwd_df = pd.DataFrame(fwd_returns)

    # Label: is this sector in top-K over next 21 days?
    fwd_rank = fwd_df.rank(axis=1, ascending=False)  # rank 1 = best
    labels = {}
    for s in available_sectors:
        labels[s] = (fwd_rank[s] <= TOP_K).astype(int)
    labels_df = pd.DataFrame(labels)

    # Get common valid index
    feat_idx = all_features[available_sectors[0]].dropna().index
    for s in available_sectors[1:]:
        feat_idx = feat_idx.intersection(all_features[s].dropna().index)
    feat_idx = feat_idx.intersection(labels_df.dropna().index)

    portfolio_ret = pd.Series(0.0, index=feat_idx)
    selections = pd.DataFrame(0, index=feat_idx, columns=available_sectors)

    model = None
    last_retrain = 0
    feature_cols = None
    prev_selected = set()  # HC #718 R3: track previous selections for turnover cost
    COST_BPS = 5  # HC #718 R3: 5 bps per leg

    start_idx = TRAIN_WINDOW

    print(f"  Walk-forward over {len(feat_idx) - start_idx} days...")

    for i in range(start_idx, len(feat_idx)):
        # Monthly retrain
        if model is None or (i - last_retrain) >= 21:
            # Build training data: stack all sectors' features + labels
            train_start = max(0, i - TRAIN_WINDOW)
            train_end = i - FWD_HORIZON  # avoid lookahead

            if train_end <= train_start + 60:
                continue

            X_parts, y_parts = [], []
            for s in available_sectors:
                feat_s = all_features[s].loc[feat_idx[train_start:train_end]].copy()
                label_s = labels_df[s].loc[feat_idx[train_start:train_end]]
                valid = feat_s.notna().all(axis=1) & label_s.notna()
                X_parts.append(feat_s[valid])
                y_parts.append(label_s[valid])

            X_train = pd.concat(X_parts, ignore_index=True)
            y_train = pd.concat(y_parts, ignore_index=True)

            if len(X_train) < 100:
                continue

            feature_cols = X_train.columns.tolist()

            model = GradientBoostingClassifier(
                n_estimators=80, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=15, random_state=42
            )
            model.fit(X_train.fillna(0), y_train.astype(int))
            last_retrain = i

        if model is None or feature_cols is None:
            continue

        # Predict for each sector
        probs = {}
        for s in available_sectors:
            X_pred = all_features[s].loc[[feat_idx[i]]][feature_cols].fillna(0)
            try:
                probs[s] = model.predict_proba(X_pred)[0][1]
            except:
                probs[s] = 0.5

        # Select top-K predicted sectors
        sorted_sectors = sorted(probs.keys(), key=lambda s: probs[s], reverse=True)
        selected = sorted_sectors[:TOP_K]

        # Equal weight, vol-targeted
        for s in selected:
            selections.loc[feat_idx[i], s] = 1

        # Portfolio return
        sel_returns = returns[selected].loc[feat_idx[i]]
        if not sel_returns.isna().all():
            avg_ret = sel_returns.mean()
            # Vol targeting
            recent_vol = returns[selected].iloc[max(0,i-20):i].std().mean()
            vol_scalar = min(2.0, TARGET_VOL / np.sqrt(252) / (recent_vol + 1e-8))
            day_ret = avg_ret * vol_scalar
            # HC #718 R3: transaction costs on turnover (changed positions)
            current_set = set(selected)
            if prev_selected:
                changed = len(prev_selected.symmetric_difference(current_set))
                if changed > 0:
                    turnover_frac = changed / max(len(prev_selected), len(current_set))
                    day_ret -= turnover_frac * COST_BPS / 10000 * vol_scalar
            else:
                day_ret -= COST_BPS / 10000 * vol_scalar  # initial buy
            prev_selected = current_set
            portfolio_ret.iloc[i] = day_ret

    return portfolio_ret, selections


# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL
# ─────────────────────────────────────────────────────────────────────────────

def compute_sharpe(r):
    r = r.dropna()
    return r.mean() / r.std() * np.sqrt(252) if len(r) > 0 and r.std() > 0 else 0

def compute_metrics(returns, name=""):
    r = returns.dropna()
    if len(r) == 0 or r.std() == 0:
        return {'name': name, 'sharpe': 0}
    sharpe = compute_sharpe(r)
    down = r[r < 0].std()
    sortino = r.mean() / down * np.sqrt(252) if down > 0 else 0
    cum = (1 + r).cumprod()
    years = len(r) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100 if years > 0 and cum.iloc[-1] > 0 else 0
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    pf_pos = r[r > 0].sum()
    pf_neg = abs(r[r < 0].sum())
    pf = pf_pos / pf_neg if pf_neg > 0 else 99
    wr = (r > 0).mean() * 100
    return {'name': name, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
            'cagr': round(cagr, 1), 'max_dd': round(max_dd, 1), 'calmar': round(calmar, 3),
            'profit_factor': round(pf, 3), 'win_rate': round(wr, 1)}


def permutation_test(closes, all_features, available_sectors, real_sharpe, n_perms=50):
    """Random sector selection (uniform) as baseline."""
    print(f"  Running {n_perms} permutations (random top-{TOP_K} selection)...")
    returns = closes[available_sectors].pct_change()

    feat_idx = all_features[available_sectors[0]].dropna().index
    for s in available_sectors[1:]:
        feat_idx = feat_idx.intersection(all_features[s].dropna().index)

    perm_sharpes = []
    for p in range(n_perms):
        if (p+1) % 10 == 0:
            print(f"    Perm {p+1}/{n_perms}...")

        port_ret = pd.Series(0.0, index=feat_idx)
        for i in range(TRAIN_WINDOW, len(feat_idx)):
            # Random top-K selection
            selected = np.random.choice(available_sectors, TOP_K, replace=False)
            sel_ret = returns[list(selected)].loc[feat_idx[i]]
            if not sel_ret.isna().all():
                recent_vol = returns[list(selected)].iloc[max(0,i-20):i].std().mean()
                vol_scalar = min(2.0, TARGET_VOL / np.sqrt(252) / (recent_vol + 1e-8))
                port_ret.iloc[i] = sel_ret.mean() * vol_scalar

        perm_sharpes.append(compute_sharpe(port_ret))

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()
    print(f"  Real: {real_sharpe:.3f}, Random: {np.mean(perm_sharpes):.3f} ± {np.std(perm_sharpes):.3f}")
    print(f"  P-value: {p_value:.4f}")
    return {'p_value': float(p_value), 'perm_mean': float(np.mean(perm_sharpes)),
            'perm_std': float(np.std(perm_sharpes)), 'pass': p_value < 0.05}


def sub_period_test(returns, n=4):
    bs = len(returns) // n
    sharpes = [compute_sharpe(returns.iloc[b*bs:(b+1)*bs]) for b in range(n)]
    cv = np.std(sharpes) / (abs(np.mean(sharpes)) + 1e-8)
    print(f"  Blocks: {[f'{s:.2f}' for s in sharpes]}, CV={cv:.3f}")
    return {'sharpes': [float(s) for s in sharpes], 'cv': float(cv), 'pass': cv < 0.50}


def r1_test(returns, closes):
    spy = closes['SPY'].pct_change()
    common = returns.index.intersection(spy.index)
    ret = returns.loc[common]
    spy_m = spy.loc[common].rolling(21).sum()
    gs = compute_sharpe(ret[spy_m > 0].dropna())
    rs = compute_sharpe(ret[spy_m <= 0].dropna())
    gap = abs(gs - rs) / max(abs(gs), abs(rs), 0.01)
    print(f"  Green: {gs:.3f}, Red: {rs:.3f}, Gap: {gap:.3f}")
    return {'green': float(gs), 'red': float(rs), 'gap': float(gap), 'pass': gap < 0.50}


def outlier_test(returns):
    full = compute_sharpe(returns)
    lo, hi = np.percentile(returns.dropna(), [5, 95])
    trimmed = returns[(returns >= lo) & (returns <= hi)]
    ts = compute_sharpe(trimmed)
    deg = (ts - full) / (abs(full) + 1e-8)
    print(f"  Full: {full:.3f}, Trimmed: {ts:.3f}, Deg: {deg:.1%}")
    return {'full': float(full), 'trimmed': float(ts), 'deg': float(deg), 'pass': deg > -0.40}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 80)
    print("ML SECTOR MOMENTUM v1 — Cross-Sector Rotation")
    print("=" * 80)

    # 1. Data
    print("\n[1/5] DATA...")
    closes = download_data()

    # 2. Features
    print("\n[2/5] BUILDING SECTOR FEATURES...")
    all_features, available_sectors = build_sector_features(closes)
    print(f"  Available sectors: {len(available_sectors)} ({', '.join(available_sectors)})")
    n_feats = all_features[available_sectors[0]].shape[1]
    print(f"  Features per sector: {n_feats}")

    # 3. Strategy
    print("\n[3/5] WALK-FORWARD ML SECTOR ROTATION...")
    port_ret, selections = run_sector_momentum(closes, all_features, available_sectors)
    active = port_ret[port_ret != 0].dropna()
    metrics = compute_metrics(active, "ML Sector Momentum")

    # Benchmark: equal-weight all sectors
    returns = closes[available_sectors].pct_change()
    ew_ret = returns.mean(axis=1).dropna()
    ew_m = compute_metrics(ew_ret, "Equal-Weight Sectors")

    # SPY benchmark
    spy_ret = closes['SPY'].pct_change().dropna()
    spy_m = compute_metrics(spy_ret, "SPY")

    # Simple momentum benchmark (buy top-3 by 20d momentum, no ML)
    mom20 = returns.rolling(20).sum()
    simple_port = pd.Series(0.0, index=returns.index)
    for i in range(60, len(returns)):
        ranks = mom20.iloc[i-1].rank(ascending=False)
        top3 = ranks[ranks <= TOP_K].index.tolist()
        if top3:
            simple_port.iloc[i] = returns[top3].iloc[i].mean()
    simple_m = compute_metrics(simple_port[simple_port != 0].dropna(), "Simple Mom Top-3")

    print(f"\n  RESULTS:")
    print(f"    ML Sector Mom:    Sharpe={metrics['sharpe']}, CAGR={metrics['cagr']}%, MaxDD={metrics['max_dd']}%")
    print(f"    Simple Mom Top-3: Sharpe={simple_m['sharpe']}, CAGR={simple_m['cagr']}%")
    print(f"    EW All Sectors:   Sharpe={ew_m['sharpe']}, CAGR={ew_m['cagr']}%")
    print(f"    SPY B&H:          Sharpe={spy_m['sharpe']}, CAGR={spy_m['cagr']}%")

    # Sector frequency analysis
    print(f"\n  Sector Selection Frequency:")
    sel_freq = selections[selections.index.isin(active.index)].sum() / len(active) * 100
    for s in sel_freq.sort_values(ascending=False).index:
        pct = sel_freq[s]
        if pct > 5:
            print(f"    {SECTOR_NAMES.get(s, s):20s}: {pct:.1f}%")

    # 4. Adversarial
    print("\n[4/5] ADVERSARIAL VALIDATION...")

    print("\n  --- Permutation (random selection) ---")
    perm = permutation_test(closes, all_features, available_sectors, metrics['sharpe'], N_PERMUTATIONS)

    print("\n  --- Sub-Period ---")
    subp = sub_period_test(active)

    print("\n  --- R1 Regime ---")
    r1 = r1_test(active, closes)

    print("\n  --- Outlier ---")
    outlier = outlier_test(active)

    # 5. Summary
    gates = sum([perm['pass'], subp['pass'], r1['pass'], outlier['pass']])

    print("\n" + "=" * 80)
    print(f"FINAL: {gates}/4 gates | Sharpe {metrics['sharpe']}")
    verdict = "PASS" if gates >= 3 and metrics['sharpe'] > 1.5 else \
              "MARGINAL" if gates >= 2 and metrics['sharpe'] > 1.0 else "FAIL"
    print(f"  Verdict: {verdict}")
    print(f"  ML vs Simple Mom: {metrics['sharpe'] - simple_m['sharpe']:+.3f} Sharpe improvement")
    print("=" * 80)

    # Save
    results = {
        'strategy': 'ML Sector Momentum v1',
        'metrics': metrics,
        'benchmarks': {'simple_momentum': simple_m, 'equal_weight': ew_m, 'spy': spy_m},
        'adversarial': {
            'permutation': perm, 'sub_period': subp, 'r1_regime': r1, 'outlier': outlier,
            'gates_passed': gates, 'total': 4, 'verdict': verdict
        },
        'parameters': {
            'train_window': TRAIN_WINDOW, 'fwd_horizon': FWD_HORIZON,
            'top_k': TOP_K, 'target_vol': TARGET_VOL, 'sectors': available_sectors
        },
        'sector_frequency': {s: round(float(v), 1) for s, v in sel_freq.items()},
        'runtime_seconds': round(time.time() - t0, 1)
    }

    with open(OUTPUT / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    cum_ml = (1 + port_ret).cumprod()
    cum_simple = (1 + simple_port).cumprod()
    cum_ew = (1 + ew_ret).cumprod()
    cum_spy = (1 + spy_ret).cumprod()

    ax = axes[0]
    ax.plot(cum_ml, label=f'ML Sector Mom (Sharpe {metrics["sharpe"]})', linewidth=2)
    ax.plot(cum_simple, label=f'Simple Mom (Sharpe {simple_m["sharpe"]})', alpha=0.7)
    ax.plot(cum_ew, label=f'EW Sectors (Sharpe {ew_m["sharpe"]})', alpha=0.5)
    ax.plot(cum_spy, label=f'SPY (Sharpe {spy_m["sharpe"]})', alpha=0.4, color='gray')
    ax.set_yscale('log')
    ax.set_title('ML Sector Momentum v1')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sector heatmap (selection frequency by year)
    ax = axes[1]
    sel_annual = selections.resample('YE').sum()
    if len(sel_annual) > 0:
        sel_pct = sel_annual.div(sel_annual.sum(axis=1), axis=0) * 100
        sel_pct = sel_pct[[s for s in available_sectors if s in sel_pct.columns]]
        sel_pct.plot.bar(stacked=True, ax=ax, legend=False)
        ax.set_title('Sector Selection by Year (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'equity_curve.png', dpi=100, bbox_inches='tight')
    plt.close()

    print(f"\n  Runtime: {time.time()-t0:.0f}s")
    return results


if __name__ == '__main__':
    main()
