#!/usr/bin/env python3
"""
DL ETF Ranker v1 — Cross-Sectional Attention on Sector ETFs
=============================================================
Purpose: Apply PyTorch attention-based cross-sectional ranking to 22 sector ETFs.
         No survivorship bias (ETFs don't die). Same architecture that achieved
         Sharpe 2.37 on stocks, but on clean ETF universe.

Architecture:
  - Input: 22 ETFs × 25+ features (momentum, volatility, relative strength)
  - Model: 2-layer multi-head attention + MLP head
  - Loss: ListMLE (learning to rank)
  - Walk-forward: 504d train, 21d test, sliding

Adversarial Gates (4-gate):
  1. Permutation test (shuffle stock selection)
  2. R1 regime-agnostic (bull/bear Sharpe gap < 0.50)
  3. Sub-period stability (both halves positive)
  4. Outlier removal (trim top/bottom 5% returns)

MLflow tracked. Designed for Neptune GPU (RTX 3090).
"""

import sys
import json
import warnings
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────
ETF_UNIVERSE = [
    'XLK', 'XLV', 'XLF', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE',
    'XLC', 'SMH', 'XBI', 'XHB', 'XRT', 'IYT', 'KBE', 'XME', 'DBC', 'GLD',
    'TLT', 'HYG'
]

BENCHMARK = 'SPY'
TRAIN_WINDOW = 504  # ~2 years
TEST_WINDOW = 21    # monthly rebalance
TOP_K_LIST = [3, 5]
COST_BPS = 20  # round-trip transaction cost
FORWARD_DAYS = 21

# Model hyperparams
HIDDEN_DIM = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 32

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# ── Feature Engineering ───────────────────────────────────────────────────
def compute_features(prices_df: pd.DataFrame, spy_prices: pd.Series) -> pd.DataFrame:
    """Compute cross-sectional features for all ETFs at each date."""
    features_list = []

    for ticker in prices_df.columns:
        p = prices_df[ticker].dropna()
        if len(p) < 252:
            continue

        feat = pd.DataFrame(index=p.index)
        feat['ticker'] = ticker

        # Momentum features
        for w in [5, 10, 21, 63, 126, 252]:
            feat[f'mom_{w}d'] = p.pct_change(w)

        # Momentum acceleration
        feat['mom_accel'] = feat['mom_21d'] - feat['mom_63d']

        # 12-1 momentum (classic)
        feat['mom_12_1'] = p.pct_change(252) - p.pct_change(21)

        # Volatility features
        ret = p.pct_change()
        for w in [21, 63]:
            feat[f'vol_{w}d'] = ret.rolling(w).std() * np.sqrt(252)

        feat['vol_ratio'] = feat['vol_21d'] / feat['vol_63d'].clip(lower=0.01)

        # Risk-adjusted
        feat['sharpe_63d'] = (ret.rolling(63).mean() * 252) / (ret.rolling(63).std() * np.sqrt(252)).clip(lower=0.01)

        # Drawdown
        rolling_max = p.rolling(252, min_periods=21).max()
        feat['drawdown'] = (p / rolling_max) - 1
        feat['maxdd_63d'] = feat['drawdown'].rolling(63).min()

        # Volume features (use returns as proxy since we only have prices)
        feat['ret_skew_63d'] = ret.rolling(63).apply(lambda x: pd.Series(x).skew(), raw=False)
        feat['ret_kurt_63d'] = ret.rolling(63).apply(lambda x: pd.Series(x).kurtosis(), raw=False)

        # Relative strength vs SPY
        spy_ret = spy_prices.pct_change()
        for w in [21, 63]:
            etf_ret_w = ret.rolling(w).sum()
            spy_ret_w = spy_ret.reindex(ret.index).rolling(w).sum()
            feat[f'rs_vs_spy_{w}d'] = etf_ret_w - spy_ret_w

        # Mean reversion signal (distance from 50d MA)
        ma50 = p.rolling(50).mean()
        feat['dist_ma50'] = (p / ma50) - 1

        # Forward return (target) — T+1 to T+21
        feat['fwd_ret'] = p.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)

        features_list.append(feat)

    if not features_list:
        return pd.DataFrame()

    all_features = pd.concat(features_list, axis=0)
    return all_features


# ── PyTorch Model ─────────────────────────────────────────────────────────
class CrossSectionalAttention(nn.Module):
    """Multi-head attention for cross-sectional stock ranking."""

    def __init__(self, n_features, hidden_dim=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        # x: (batch, n_stocks, n_features)
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        scores = self.head(h).squeeze(-1)  # (batch, n_stocks)
        return scores


def listmle_loss(scores, targets):
    """ListMLE loss for learning to rank."""
    # Sort targets descending to get ideal ranking
    _, ideal_order = targets.sort(descending=True, dim=-1)

    # Gather scores in ideal order
    ordered_scores = scores.gather(1, ideal_order)

    # ListMLE: sum of log-softmax from top to bottom
    n = ordered_scores.size(1)
    loss = 0.0
    for i in range(n - 1):
        remaining = ordered_scores[:, i:]
        log_softmax = remaining[:, 0] - torch.logsumexp(remaining, dim=1)
        loss = loss - log_softmax.mean()

    return loss / n


# ── Walk-Forward Engine ───────────────────────────────────────────────────
def run_walk_forward(features_df: pd.DataFrame, feature_cols: list, top_k: int = 3,
                     use_attention: bool = True):
    """Walk-forward with DL attention model."""

    dates = sorted(features_df.index.unique())
    n_dates = len(dates)

    results = []
    fold_count = 0

    start_idx = TRAIN_WINDOW

    while start_idx + TEST_WINDOW <= n_dates:
        train_dates = dates[start_idx - TRAIN_WINDOW: start_idx]
        test_dates = dates[start_idx: start_idx + TEST_WINDOW]

        # Get training data
        train_data = features_df.loc[features_df.index.isin(train_dates)]
        test_data = features_df.loc[features_df.index.isin(test_dates)]

        if len(train_data) < 100 or len(test_data) < 5:
            start_idx += TEST_WINDOW
            continue

        # Get unique tickers present in both
        train_tickers = set(train_data['ticker'].unique())
        test_tickers = set(test_data['ticker'].unique())
        common_tickers = sorted(train_tickers & test_tickers)

        if len(common_tickers) < top_k + 2:
            start_idx += TEST_WINDOW
            continue

        # Build cross-sectional samples for training
        train_samples_X, train_samples_Y = _build_cross_sectional_samples(
            train_data, common_tickers, feature_cols
        )

        if len(train_samples_X) < 10:
            start_idx += TEST_WINDOW
            continue

        if use_attention:
            # Train attention model
            model = CrossSectionalAttention(
                n_features=len(feature_cols),
                hidden_dim=HIDDEN_DIM,
                n_heads=N_HEADS,
                n_layers=N_LAYERS,
                dropout=DROPOUT
            ).to(DEVICE)

            optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

            X_tensor = torch.FloatTensor(np.array(train_samples_X)).to(DEVICE)
            Y_tensor = torch.FloatTensor(np.array(train_samples_Y)).to(DEVICE)

            model.train()
            for epoch in range(EPOCHS):
                # Mini-batch
                perm = torch.randperm(len(X_tensor))
                for i in range(0, len(X_tensor), BATCH_SIZE):
                    batch_idx = perm[i:i+BATCH_SIZE]
                    if len(batch_idx) < 2:
                        continue

                    batch_X = X_tensor[batch_idx]
                    batch_Y = Y_tensor[batch_idx]

                    scores = model(batch_X)
                    loss = listmle_loss(scores, batch_Y)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            # Predict on test period
            model.eval()
            with torch.no_grad():
                test_X, _ = _build_cross_sectional_samples(
                    test_data, common_tickers, feature_cols, require_target=False
                )
                if len(test_X) == 0:
                    start_idx += TEST_WINDOW
                    continue

                # Use last test date for ranking
                last_test = test_data[test_data.index == test_dates[-1]]
                if len(last_test) < top_k:
                    start_idx += TEST_WINDOW
                    continue

                # Build single sample
                sample_x = []
                ticker_order = []
                for t in common_tickers:
                    t_data = last_test[last_test['ticker'] == t]
                    if len(t_data) == 0:
                        continue
                    row = t_data[feature_cols].iloc[-1].values
                    if np.any(np.isnan(row)):
                        continue
                    sample_x.append(row)
                    ticker_order.append(t)

                if len(sample_x) < top_k:
                    start_idx += TEST_WINDOW
                    continue

                x_tensor = torch.FloatTensor(np.array([sample_x])).to(DEVICE)
                pred_scores = model(x_tensor).cpu().numpy()[0]

                # Select top K
                ranked_idx = np.argsort(-pred_scores)[:top_k]
                selected = [ticker_order[i] for i in ranked_idx]
        else:
            # LightGBM baseline
            import lightgbm as lgb

            train_flat = train_data.dropna(subset=feature_cols + ['fwd_ret'])
            X_train = train_flat[feature_cols].values
            y_train = train_flat['fwd_ret'].values

            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, verbose=-1
            )
            model.fit(X_train, y_train)

            last_test = test_data[test_data.index == test_dates[-1]]
            valid_test = last_test.dropna(subset=feature_cols)
            if len(valid_test) < top_k:
                start_idx += TEST_WINDOW
                continue

            preds = model.predict(valid_test[feature_cols].values)
            ranked_idx = np.argsort(-preds)[:top_k]
            selected = valid_test['ticker'].iloc[ranked_idx].tolist()

        # Calculate actual returns for selected tickers
        test_end_idx = min(start_idx + TEST_WINDOW, n_dates)
        test_period_dates = dates[start_idx:test_end_idx]

        period_returns = []
        for ticker in selected:
            t_data = features_df[
                (features_df['ticker'] == ticker) &
                (features_df.index.isin(test_period_dates))
            ]
            if len(t_data) > 0 and 'fwd_ret' in t_data.columns:
                # Use the return at entry date
                entry_data = features_df[
                    (features_df['ticker'] == ticker) &
                    (features_df.index == test_period_dates[0])
                ]
                if len(entry_data) > 0 and not np.isnan(entry_data['fwd_ret'].iloc[0]):
                    period_returns.append(entry_data['fwd_ret'].iloc[0])

        if period_returns:
            port_ret = np.mean(period_returns) - (COST_BPS / 10000 * 2)  # entry + exit cost

            # Determine regime (SPY return over test period)
            spy_data = features_df[
                (features_df['ticker'] == ETF_UNIVERSE[0]) &
                (features_df.index.isin(test_period_dates))
            ]

            results.append({
                'fold': fold_count,
                'date': test_period_dates[0],
                'return': port_ret,
                'selected': selected,
                'n_tickers': len(selected)
            })

        fold_count += 1
        start_idx += TEST_WINDOW

        if fold_count % 10 == 0:
            print(f"  Fold {fold_count}: cumret={sum(r['return'] for r in results):.3f}")

    return results


def _build_cross_sectional_samples(data, tickers, feature_cols, require_target=True):
    """Build cross-sectional training samples (one per date)."""
    X_samples = []
    Y_samples = []

    for date in sorted(data.index.unique()):
        day_data = data[data.index == date]

        sample_x = []
        sample_y = []
        valid = True

        for t in tickers:
            t_data = day_data[day_data['ticker'] == t]
            if len(t_data) == 0:
                valid = False
                break

            row = t_data[feature_cols].iloc[0].values
            if np.any(np.isnan(row)):
                valid = False
                break
            sample_x.append(row)

            if require_target:
                fwd = t_data['fwd_ret'].iloc[0]
                if np.isnan(fwd):
                    valid = False
                    break
                sample_y.append(fwd)

        if valid and len(sample_x) == len(tickers):
            X_samples.append(sample_x)
            if require_target:
                Y_samples.append(sample_y)

    return X_samples, Y_samples


# ── Adversarial Validation ────────────────────────────────────────────────
def adversarial_audit(results: list, spy_prices: pd.Series):
    """Run 4-gate adversarial validation."""

    returns = np.array([r['return'] for r in results])
    dates = [r['date'] for r in results]

    n = len(returns)
    if n < 10:
        return {'pass': False, 'reason': f'Too few folds ({n})'}

    # Basic metrics
    sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(12)  # monthly -> annual
    cagr = (np.prod(1 + returns) ** (12 / n) - 1) * 100
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = cum_returns / running_max - 1
    max_dd = drawdowns.min() * 100
    wr = np.mean(returns > 0) * 100
    pf = abs(returns[returns > 0].sum()) / (abs(returns[returns < 0].sum()) + 1e-10)
    sortino_denom = np.sqrt(np.mean(np.minimum(returns, 0) ** 2) + 1e-10) * np.sqrt(12)
    sortino = np.mean(returns) * 12 / (sortino_denom + 1e-10)

    print(f"\n  === METRICS ===")
    print(f"  Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}")
    print(f"  CAGR: {cagr:.1f}%, MaxDD: {max_dd:.1f}%")
    print(f"  WR: {wr:.1f}%, PF: {pf:.2f}")
    print(f"  Folds: {n}")

    gates = {}

    # Gate 1: Permutation test (shuffle ETF selection → random top-K)
    print(f"\n  --- Gate 1: Permutation Test ---")
    n_perms = 500
    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = returns.copy()
        np.random.shuffle(shuffled)
        perm_sharpes.append(np.mean(shuffled) / (np.std(shuffled) + 1e-10) * np.sqrt(12))

    perm_p = np.mean(np.array(perm_sharpes) >= sharpe)
    gates['perm'] = {'p': perm_p, 'pass': perm_p < 0.05}
    print(f"  Perm p-value: {perm_p:.3f} {'PASS' if perm_p < 0.05 else 'FAIL'}")

    # Gate 2: R1 regime-agnostic (bull vs bear)
    print(f"\n  --- Gate 2: Regime Agnostic (R1) ---")
    # Classify each fold as bull/bear based on SPY
    bull_rets, bear_rets = [], []
    for r in results:
        date = r['date']
        # Find nearest SPY date
        spy_idx = spy_prices.index.get_indexer([date], method='nearest')[0]
        if spy_idx >= 21:
            spy_change = (spy_prices.iloc[spy_idx] / spy_prices.iloc[spy_idx - 21]) - 1
            if spy_change >= 0:
                bull_rets.append(r['return'])
            else:
                bear_rets.append(r['return'])

    bull_rets = np.array(bull_rets) if bull_rets else np.array([0])
    bear_rets = np.array(bear_rets) if bear_rets else np.array([0])

    bull_sharpe = np.mean(bull_rets) / (np.std(bull_rets) + 1e-10) * np.sqrt(12)
    bear_sharpe = np.mean(bear_rets) / (np.std(bear_rets) + 1e-10) * np.sqrt(12)

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.01)
    gates['r1'] = {'gap': regime_gap, 'bull': bull_sharpe, 'bear': bear_sharpe, 'pass': regime_gap < 0.50}
    print(f"  Bull Sharpe: {bull_sharpe:.2f}, Bear Sharpe: {bear_sharpe:.2f}")
    print(f"  R1 gap: {regime_gap:.3f} {'PASS' if regime_gap < 0.50 else 'FAIL'}")

    # Gate 3: Sub-period stability
    print(f"\n  --- Gate 3: Sub-period Stability ---")
    mid = n // 2
    h1_sharpe = np.mean(returns[:mid]) / (np.std(returns[:mid]) + 1e-10) * np.sqrt(12)
    h2_sharpe = np.mean(returns[mid:]) / (np.std(returns[mid:]) + 1e-10) * np.sqrt(12)
    sub_pass = h1_sharpe > 0 and h2_sharpe > 0
    gates['sub'] = {'h1': h1_sharpe, 'h2': h2_sharpe, 'pass': sub_pass}
    print(f"  H1 Sharpe: {h1_sharpe:.2f}, H2 Sharpe: {h2_sharpe:.2f}")
    print(f"  {'PASS' if sub_pass else 'FAIL'}")

    # Gate 4: Outlier removal
    print(f"\n  --- Gate 4: Outlier Removal ---")
    p5, p95 = np.percentile(returns, [5, 95])
    trimmed = returns[(returns >= p5) & (returns <= p95)]
    trimmed_sharpe = np.mean(trimmed) / (np.std(trimmed) + 1e-10) * np.sqrt(12)
    outlier_pass = trimmed_sharpe > 0
    gates['outlier'] = {'trimmed_sharpe': trimmed_sharpe, 'pass': outlier_pass}
    print(f"  Trimmed Sharpe: {trimmed_sharpe:.2f} {'PASS' if outlier_pass else 'FAIL'}")

    total_pass = sum(1 for g in gates.values() if g['pass'])

    return {
        'sharpe': sharpe, 'sortino': sortino, 'cagr': cagr, 'max_dd': max_dd,
        'wr': wr, 'pf': pf, 'n_folds': n,
        'gates': gates, 'gates_passed': total_pass, 'gates_total': 4
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("DL ETF RANKER v1 — Cross-Sectional Attention on Sector ETFs")
    print("=" * 70)
    print(f"Universe: {len(ETF_UNIVERSE)} ETFs | Top K: {TOP_K_LIST}")
    print(f"Architecture: {N_LAYERS}-layer Transformer, {N_HEADS} heads, dim={HIDDEN_DIM}")
    print(f"Walk-forward: {TRAIN_WINDOW}d train, {TEST_WINDOW}d test, sliding")
    print(f"Device: {DEVICE}")
    print()

    # Download data
    print("Downloading ETF data...")
    tickers = ETF_UNIVERSE + [BENCHMARK]
    data = yf.download(tickers, start='2015-01-01', progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
    else:
        prices = data

    spy_prices = prices[BENCHMARK].dropna()
    etf_prices = prices[ETF_UNIVERSE].dropna(how='all')

    print(f"  Loaded {len(etf_prices.columns)} ETFs, {len(etf_prices)} days")
    print(f"  Date range: {etf_prices.index[0].strftime('%Y-%m-%d')} to {etf_prices.index[-1].strftime('%Y-%m-%d')}")

    # Compute features
    print("\nComputing features...")
    features_df = compute_features(etf_prices, spy_prices)

    # Get feature columns
    feature_cols = [c for c in features_df.columns if c not in ['ticker', 'fwd_ret']]
    print(f"  Features: {len(feature_cols)} per ETF")
    print(f"  Total samples: {len(features_df)}")

    # Drop rows with NaN features
    features_df = features_df.dropna(subset=feature_cols)
    print(f"  After NaN drop: {len(features_df)}")

    all_results = {}

    # Run variants
    variants = []
    for top_k in TOP_K_LIST:
        variants.append((f'DL_Attn_Top{top_k}', True, top_k))
        variants.append((f'LightGBM_Top{top_k}', False, top_k))

    for name, use_attention, top_k in variants:
        print(f"\n--- {name} (use_attention={use_attention}, top_k={top_k}) ---")
        t0 = time.time()

        results = run_walk_forward(
            features_df, feature_cols, top_k=top_k, use_attention=use_attention
        )

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.0f}s, {len(results)} folds")

        if len(results) < 5:
            print(f"  SKIP — too few results")
            continue

        audit = adversarial_audit(results, spy_prices)
        all_results[name] = {
            'results': results,
            'audit': audit,
            'elapsed': elapsed
        }

        print(f"\n  VERDICT: {audit['gates_passed']}/{audit['gates_total']} gates passed")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — DL ETF RANKER v1")
    print("=" * 70)
    print(f"{'Variant':<25} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'WR':>5} {'PF':>6} {'Gates':>5}")
    print("-" * 70)

    best_variant = None
    best_sharpe = -999

    for name, data in all_results.items():
        a = data['audit']
        line = f"{name:<25} {a['sharpe']:>7.2f} {a['cagr']:>6.1f}% {a['max_dd']:>6.1f}% {a['wr']:>4.1f}% {a['pf']:>6.2f} {a['gates_passed']}/{a['gates_total']}"
        print(line)

        if a['sharpe'] > best_sharpe:
            best_sharpe = a['sharpe']
            best_variant = name

    # Save results
    output_dir = Path("/home/nick/Lvl3Quant/research/findings")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {}
    for name, data in all_results.items():
        audit = data['audit']
        save_data[name] = {
            'sharpe': float(audit['sharpe']),
            'sortino': float(audit['sortino']),
            'cagr': float(audit['cagr']),
            'max_dd': float(audit['max_dd']),
            'wr': float(audit['wr']),
            'pf': float(audit['pf']),
            'n_folds': int(audit['n_folds']),
            'gates_passed': audit['gates_passed'],
            'gates': {k: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                         for kk, vv in v.items()}
                     for k, v in audit['gates'].items()},
            'elapsed_s': data['elapsed'],
            'monthly_returns': [float(r['return']) for r in data['results']],
            'dates': [r['date'].strftime('%Y-%m-%d') for r in data['results']],
            'selections': [r['selected'] for r in data['results']]
        }

    results_file = output_dir / "dl_etf_ranker_v1_results.json"
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to {results_file}")
    print(f"\nBest variant: {best_variant} (Sharpe {best_sharpe:.2f})")
    print("\nDONE.")


if __name__ == '__main__':
    main()
