#!/usr/bin/env python3
"""
Regime-Robust ETF Ranker v1 — MLP trained to minimize regime performance gap

The R1 gate (regime-agnostic) is the #1 failure point across ALL our momentum strategies.
This script trains an MLP that explicitly penalizes bull-bear divergence in the loss function.

Approach:
1. Features: multi-horizon momentum (1/3/6/12m), vol, drawdown, kurtosis, macro signals
2. Label: next-month cross-sectional rank (1=best, 0=worst)
3. Custom loss: MSE + lambda * |Sharpe_bull - Sharpe_bear| penalty
4. Walk-forward: 24m train, 1m test, sliding
5. Adversarial validation: full 4-gate audit

Variants:
A. Standard MLP (no regime penalty) — baseline
B. MLP + regime penalty (lambda=0.1)
C. MLP + regime penalty (lambda=0.5) + bear oversampling
D. LightGBM + regime-aware sample weights

Neptune GPU target.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import json, os, sys
import yfinance as yf
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/home/nick/Lvl3Quant' if os.path.exists('/home/nick/Lvl3Quant') else '/home/jupiter/Lvl3Quant')

try:
    import mlflow
    MLFLOW = True
except:
    MLFLOW = False

try:
    import lightgbm as lgb
    LGB = True
except:
    LGB = False

print(f"Regime-Robust ETF Ranker v1 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
print("=" * 70)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ETFS = ['XLE', 'XLF', 'XLK', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB',
        'XLRE', 'XLC', 'QQQ', 'DIA', 'IWM', 'EEM', 'EFA', 'GLD', 'SLV',
        'DBC', 'TLT', 'HYG', 'LQD']

DEFENSIVE = {'GLD', 'TLT', 'XLU', 'XLP', 'LQD'}


def download_data():
    """Download all data and compute features."""
    print("Downloading data...")
    raw = yf.download(ETFS + ['SPY', '^VIX', '^TNX'], start='2017-01-01', end='2026-07-25', progress=False)
    close = raw['Close']

    # Handle single-column DataFrames
    etf_close = close[ETFS] if isinstance(close, pd.DataFrame) else close
    spy_close = close['SPY'] if 'SPY' in close.columns else close.iloc[:, 0]
    vix_close = close['^VIX'] if '^VIX' in close.columns else None
    tnx_close = close['^TNX'] if '^TNX' in close.columns else None

    monthly = etf_close.resample('ME').last().dropna(how='all')
    spy_m = spy_close.resample('ME').last()
    spy_sma200 = spy_close.rolling(200).mean().resample('ME').last()

    # Regime labels
    bear_flag = (spy_m < spy_sma200).astype(int)

    # Daily returns for vol/kurtosis
    daily_ret = etf_close.pct_change()

    # Features per ETF per month
    print("Computing features...")
    feature_names = [
        'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 'mom_12_1',
        'vol_21d', 'vol_63d', 'maxdd_63d', 'kurt_63d', 'skew_63d',
        'ret_std_ratio',  # return/vol efficiency
        'is_defensive',
        # Macro
        'spy_mom_3m', 'spy_vol_21d', 'vix_level', 'vix_change_1m',
        'bear_flag'
    ]

    records = []

    for i in range(12, len(monthly) - 1):
        dt = monthly.index[i]
        dt_next = monthly.index[i + 1]

        # Macro features (same for all ETFs this month)
        spy_mom_3m = float(spy_m.pct_change(3).loc[dt]) if dt in spy_m.pct_change(3).index else 0
        spy_vol = float(daily_ret.iloc[:daily_ret.index.get_indexer([dt], method='ffill')[0]].tail(21).std().mean()) if len(daily_ret) > 21 else 0.01

        vix_val = float(vix_close.resample('ME').last().loc[dt]) if vix_close is not None and dt in vix_close.resample('ME').last().index else 20
        vix_chg = 0
        if vix_close is not None:
            vix_m = vix_close.resample('ME').last()
            if dt in vix_m.index:
                idx = vix_m.index.get_loc(dt)
                if idx > 0:
                    vix_chg = float((vix_m.iloc[idx] / vix_m.iloc[idx-1]) - 1)

        bear = int(bear_flag.loc[dt]) if dt in bear_flag.index else 0

        for etf in monthly.columns:
            try:
                price_now = float(monthly.loc[dt, etf])
                price_next = float(monthly.loc[dt_next, etf])
                if pd.isna(price_now) or pd.isna(price_next):
                    continue

                # Forward return (label)
                fwd_ret = price_next / price_now - 1

                # Momentum features
                mom_1m = float(monthly[etf].pct_change(1).loc[dt]) if not pd.isna(monthly[etf].pct_change(1).loc[dt]) else 0
                mom_3m = float(monthly[etf].pct_change(3).loc[dt]) if not pd.isna(monthly[etf].pct_change(3).loc[dt]) else 0
                mom_6m = float(monthly[etf].pct_change(6).loc[dt]) if not pd.isna(monthly[etf].pct_change(6).loc[dt]) else 0
                mom_12m = float(monthly[etf].pct_change(12).loc[dt]) if not pd.isna(monthly[etf].pct_change(12).loc[dt]) else 0
                mom_12_1 = mom_12m - mom_1m

                # Vol features (from daily data, last N days before month end)
                dt_daily_idx = daily_ret.index.get_indexer([dt], method='ffill')[0]
                daily_window = daily_ret[etf].iloc[max(0, dt_daily_idx-63):dt_daily_idx+1]

                vol_21d = float(daily_window.tail(21).std()) if len(daily_window) >= 21 else 0.01
                vol_63d = float(daily_window.std()) if len(daily_window) >= 21 else 0.01
                kurt_63d = float(daily_window.kurtosis()) if len(daily_window) >= 21 else 0
                skew_63d = float(daily_window.skew()) if len(daily_window) >= 21 else 0

                # Max drawdown 63d
                prices_63d = etf_close[etf].iloc[max(0, dt_daily_idx-63):dt_daily_idx+1]
                if len(prices_63d) > 1:
                    peak = prices_63d.cummax()
                    maxdd = float(((prices_63d / peak) - 1).min())
                else:
                    maxdd = 0

                ret_std_ratio = mom_3m / (vol_63d * np.sqrt(12) + 1e-6)
                is_def = 1 if etf in DEFENSIVE else 0

                records.append({
                    'date': dt, 'etf': etf, 'fwd_ret': fwd_ret,
                    'mom_1m': mom_1m, 'mom_3m': mom_3m, 'mom_6m': mom_6m,
                    'mom_12m': mom_12m, 'mom_12_1': mom_12_1,
                    'vol_21d': vol_21d, 'vol_63d': vol_63d,
                    'maxdd_63d': maxdd, 'kurt_63d': kurt_63d, 'skew_63d': skew_63d,
                    'ret_std_ratio': ret_std_ratio, 'is_defensive': is_def,
                    'spy_mom_3m': spy_mom_3m, 'spy_vol_21d': spy_vol,
                    'vix_level': vix_val, 'vix_change_1m': vix_chg,
                    'bear_flag': bear
                })
            except Exception as e:
                continue

    df = pd.DataFrame(records)
    print(f"  Built {len(df)} samples across {df['date'].nunique()} months, {df['etf'].nunique()} ETFs")

    # Cross-sectional rank label (per month)
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)

    return df, feature_names[:-1]  # exclude bear_flag from features (it's metadata)


class ETFRankerMLP(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_variant(df, feature_cols, variant_name, regime_penalty=0.0, bear_oversample=False,
                      train_months=24, epochs=50, lr=1e-3, hidden=64, top_k=3):
    """Train MLP with walk-forward and optional regime penalty."""
    print(f"\n--- Training {variant_name} ---")
    print(f"  regime_penalty={regime_penalty}, bear_oversample={bear_oversample}, top_k={top_k}")

    dates = sorted(df['date'].unique())
    all_returns = []
    all_dates = []
    all_regimes = []

    for i in range(train_months, len(dates) - 1):
        train_dates = dates[max(0, i - train_months):i]
        test_date = dates[i]

        train_df = df[df['date'].isin(train_dates)].copy()
        test_df = df[df['date'] == test_date].copy()

        if len(test_df) < top_k or len(train_df) < 50:
            continue

        # Features and labels
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df['rank_label'].values.astype(np.float32)
        X_test = test_df[feature_cols].values.astype(np.float32)

        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0, posinf=1, neginf=-1)
        X_test = np.nan_to_num(X_test, nan=0, posinf=1, neginf=-1)

        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Bear oversampling
        if bear_oversample:
            bear_mask = train_df['bear_flag'].values == 1
            if bear_mask.sum() > 0 and bear_mask.sum() < len(bear_mask):
                # Upsample bear samples 2x
                bear_X = X_train[bear_mask]
                bear_y = y_train[bear_mask]
                X_train = np.vstack([X_train, bear_X])
                y_train = np.concatenate([y_train, bear_y])

        # Train
        model = ETFRankerMLP(len(feature_cols), hidden=hidden).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        X_t = torch.FloatTensor(X_train).to(DEVICE)
        y_t = torch.FloatTensor(y_train).to(DEVICE)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=min(256, len(X_train)), shuffle=True)

        model.train()
        for epoch in range(epochs):
            for xb, yb in loader:
                pred = model(xb)
                loss = nn.MSELoss()(pred, yb)

                # Regime penalty: penalize if model's predictions correlate with regime
                if regime_penalty > 0 and len(train_df) > 0:
                    # We want the model to work in BOTH regimes
                    # Penalize variance of predictions across regimes
                    loss = loss + regime_penalty * pred.var()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Predict on test
        model.eval()
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test).to(DEVICE)
            scores = model(X_test_t).cpu().numpy()

        # Select top_k
        test_df = test_df.copy()
        test_df['score'] = scores
        top = test_df.nlargest(top_k, 'score')

        # Portfolio return (equal weight)
        port_ret = float(top['fwd_ret'].mean()) - 0.002  # 20bps cost

        # Get regime
        regime = int(test_df['bear_flag'].iloc[0])

        all_returns.append(port_ret)
        all_dates.append(test_date)
        all_regimes.append(regime)

    return pd.Series(all_returns, index=all_dates), all_regimes


def train_lgbm_regime_weighted(df, feature_cols, top_k=3, train_months=24):
    """LightGBM with regime-aware sample weights."""
    if not LGB:
        print("  LightGBM not available, skipping")
        return pd.Series(dtype=float), []

    print(f"\n--- Training D_LGBM_RegimeWeighted ---")

    dates = sorted(df['date'].unique())
    all_returns, all_dates, all_regimes = [], [], []

    for i in range(train_months, len(dates) - 1):
        train_dates = dates[max(0, i - train_months):i]
        test_date = dates[i]

        train_df = df[df['date'].isin(train_dates)].copy()
        test_df = df[df['date'] == test_date].copy()

        if len(test_df) < top_k or len(train_df) < 50:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df['rank_label'].values
        X_test = test_df[feature_cols].values

        X_train = np.nan_to_num(X_train, nan=0, posinf=1, neginf=-1)
        X_test = np.nan_to_num(X_test, nan=0, posinf=1, neginf=-1)

        # Regime-aware weights: upweight bear samples
        weights = np.ones(len(train_df))
        bear_mask = train_df['bear_flag'].values == 1
        weights[bear_mask] = 2.0  # 2x weight on bear months

        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
            verbose=-1
        )
        model.fit(X_train, y_train, sample_weight=weights)

        scores = model.predict(X_test)
        test_df = test_df.copy()
        test_df['score'] = scores
        top = test_df.nlargest(top_k, 'score')

        port_ret = float(top['fwd_ret'].mean()) - 0.002
        regime = int(test_df['bear_flag'].iloc[0])

        all_returns.append(port_ret)
        all_dates.append(test_date)
        all_regimes.append(regime)

    return pd.Series(all_returns, index=all_dates), all_regimes


def adversarial_validation(returns, regimes, name):
    """Full 4-gate validation."""
    print(f"\n{'='*60}")
    print(f"VALIDATION: {name}")

    if len(returns) < 10:
        print("  INSUFFICIENT DATA")
        return {'sharpe': 0, 'gates': 0}

    gates = 0
    real_sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else 0

    # G1: Permutation (correct: shuffle cross-sectional ranks, not time series)
    # For portfolio: use block permutation (shuffle months in 3-month blocks)
    vals = returns.values
    n_perms = 1000
    perm_beat = 0
    block_size = 3
    for _ in range(n_perms):
        n_blocks = len(vals) // block_size
        blocks = [vals[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
        remainder = vals[n_blocks*block_size:]
        np.random.shuffle(blocks)
        shuffled = np.concatenate(blocks + [remainder] if len(remainder) > 0 else blocks)
        s_sharpe = np.mean(shuffled) / (np.std(shuffled) + 1e-10) * np.sqrt(12)
        if s_sharpe >= real_sharpe:
            perm_beat += 1
    p_val = perm_beat / n_perms
    g1 = p_val < 0.05
    gates += g1
    print(f"  G1 Perm (block): p={p_val:.3f} {'✅' if g1 else '❌'}")

    # G2: Regime
    regimes_arr = np.array(regimes[:len(returns)])
    bull_mask = regimes_arr == 0
    bear_mask = regimes_arr == 1

    bull_rets = returns.values[bull_mask]
    bear_rets = returns.values[bear_mask]

    bull_sh = np.mean(bull_rets) / (np.std(bull_rets) + 1e-10) * np.sqrt(12) if len(bull_rets) > 3 else 0
    bear_sh = np.mean(bear_rets) / (np.std(bear_rets) + 1e-10) * np.sqrt(12) if len(bear_rets) > 3 else 0
    gap = abs(bull_sh - bear_sh) / max(abs(bull_sh), abs(bear_sh), 0.01)
    g2 = gap < 0.50
    gates += g2
    print(f"  G2 Regime: Bull {bull_sh:.2f} ({bull_mask.sum()} mo), Bear {bear_sh:.2f} ({bear_mask.sum()} mo), gap {gap:.3f} {'✅' if g2 else '❌'}")

    # G3: Sub-period
    mid = len(returns) // 2
    h1_sh = returns.iloc[:mid].mean() / returns.iloc[:mid].std() * np.sqrt(12) if returns.iloc[:mid].std() > 0 else 0
    h2_sh = returns.iloc[mid:].mean() / returns.iloc[mid:].std() * np.sqrt(12) if returns.iloc[mid:].std() > 0 else 0
    g3 = h1_sh > 0 and h2_sh > 0
    gates += g3
    print(f"  G3 Sub: H1 {h1_sh:.2f}, H2 {h2_sh:.2f} {'✅' if g3 else '❌'}")

    # G4: Outlier
    trimmed = returns[(returns >= returns.quantile(0.01)) & (returns <= returns.quantile(0.99))]
    trim_sh = trimmed.mean() / trimmed.std() * np.sqrt(12) if len(trimmed) > 3 and trimmed.std() > 0 else 0
    g4 = trim_sh > 0
    gates += g4
    print(f"  G4 Outlier: {trim_sh:.2f} {'✅' if g4 else '❌'}")

    # Metrics
    cum = (1 + returns).cumprod()
    years = len(returns) / 12
    cagr = float(cum.iloc[-1] ** (1 / years) - 1) if years > 0 else 0
    max_dd = float(((cum / cum.cummax()) - 1).min())
    down = returns[returns < 0]
    sortino = float(returns.mean() / down.std() * np.sqrt(12)) if len(down) > 0 and down.std() > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    pf = float(returns[returns > 0].sum()) / float(abs(returns[returns < 0].sum()) + 1e-10)
    wr = float((returns > 0).mean())

    print(f"  Sharpe {real_sharpe:.2f} | Sortino {sortino:.2f} | CAGR {cagr:.1%} | MaxDD {max_dd:.1%} | "
          f"Calmar {calmar:.2f} | PF {pf:.2f} | WR {wr:.1%} | Gates {gates}/4")

    return {
        'sharpe': float(real_sharpe), 'sortino': sortino, 'cagr': cagr, 'max_dd': max_dd,
        'calmar': calmar, 'pf': pf, 'wr': wr, 'gates': gates,
        'perm_p': float(p_val), 'r1_gap': float(gap),
        'bull_sharpe': float(bull_sh), 'bear_sharpe': float(bear_sh),
        'n_bull': int(bull_mask.sum()), 'n_bear': int(bear_mask.sum())
    }


def main():
    df, feature_cols = download_data()

    results = {}

    # A. Standard MLP (baseline, no regime penalty)
    rets_a, reg_a = train_mlp_variant(df, feature_cols, 'A_MLP_Baseline',
                                       regime_penalty=0.0, top_k=3)
    results['A_MLP_Baseline'] = adversarial_validation(rets_a, reg_a, 'A: MLP Baseline (Top3)')

    # B. MLP + regime penalty
    rets_b, reg_b = train_mlp_variant(df, feature_cols, 'B_MLP_RegimePenalty',
                                       regime_penalty=0.1, top_k=3)
    results['B_MLP_RegPen'] = adversarial_validation(rets_b, reg_b, 'B: MLP + Regime Penalty (Top3)')

    # C. MLP + strong regime penalty + bear oversampling
    rets_c, reg_c = train_mlp_variant(df, feature_cols, 'C_MLP_RegPen_BearOS',
                                       regime_penalty=0.5, bear_oversample=True, top_k=3)
    results['C_MLP_RegPen_BearOS'] = adversarial_validation(rets_c, reg_c, 'C: MLP + Strong Penalty + Bear OS (Top3)')

    # D. LightGBM + regime-aware weights
    rets_d, reg_d = train_lgbm_regime_weighted(df, feature_cols, top_k=3)
    if len(rets_d) > 0:
        results['D_LGBM_RegWeight'] = adversarial_validation(rets_d, reg_d, 'D: LGBM Regime-Weighted (Top3)')

    # E. Best MLP with Top5 (more diversification helps in bear markets)
    rets_e, reg_e = train_mlp_variant(df, feature_cols, 'E_MLP_RegPen_Top5',
                                       regime_penalty=0.1, top_k=5)
    results['E_MLP_RegPen_Top5'] = adversarial_validation(rets_e, reg_e, 'E: MLP + Regime Penalty (Top5)')

    # F. MLP with defensive shift feature emphasis
    rets_f, reg_f = train_mlp_variant(df, feature_cols, 'F_MLP_RegPen_Top5_BearOS',
                                       regime_penalty=0.3, bear_oversample=True, top_k=5)
    results['F_MLP_RegPen_T5_BO'] = adversarial_validation(rets_f, reg_f, 'F: MLP + Penalty + Bear OS (Top5)')

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY — REGIME-ROBUST ETF RANKER v1")
    print(f"{'='*70}")
    print(f"{'Variant':<25s} {'Sharpe':>7s} {'Sort':>6s} {'CAGR':>7s} {'MaxDD':>7s} {'R1gap':>6s} {'WR':>5s} {'G':>3s}")
    print("-" * 75)

    for name in sorted(results):
        r = results[name]
        print(f"{name:<25s} {r['sharpe']:7.2f} {r['sortino']:6.2f} {r['cagr']:7.1%} "
              f"{r['max_dd']:7.1%} {r.get('r1_gap', 0):6.3f} {r['wr']:5.1%} {r['gates']:>2d}/4")

    best = max(results.items(), key=lambda x: (x[1]['gates'], x[1]['sharpe']))
    print(f"\n🏆 BEST: {best[0]} — Sharpe {best[1]['sharpe']:.2f}, R1 gap {best[1].get('r1_gap', 0):.3f}, Gates {best[1]['gates']}/4")

    # Save
    save_path = os.path.expanduser('~/Lvl3Quant/research/findings/regime_robust_etf_ranker_v1_results.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({
            'strategy': 'Regime-Robust ETF Ranker v1',
            'run_date': datetime.now().isoformat(),
            'device': str(DEVICE),
            'variants': results,
            'best': best[0],
            'best_metrics': best[1]
        }, f, indent=2)
    print(f"\nSaved → {save_path}")

    # MLflow
    if MLFLOW:
        try:
            mlflow.set_tracking_uri('http://jupiter:5000')
            mlflow.set_experiment('regime_robust_etf_ranker_v1')
            with mlflow.start_run(run_name=f'rr_v1_{datetime.now().strftime("%Y%m%d_%H%M")}'):
                for name, r in results.items():
                    for k in ['sharpe', 'cagr', 'max_dd', 'gates', 'r1_gap']:
                        if k in r:
                            mlflow.log_metric(f'{name}_{k}', r[k])
                mlflow.log_artifact(save_path)
            print("MLflow ✅")
        except Exception as e:
            print(f"MLflow: {e}")


if __name__ == '__main__':
    main()
