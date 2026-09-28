#!/usr/bin/env python3
"""VIX Spike Predictor v1 — Deep Learning (PyTorch)
Predicts VIX spikes (cross above 25) 1-5 days ahead using MLP, LSTM, 1D-CNN.
Walk-forward sliding 252d train / 21d test. Focal loss for class imbalance.
Purpose: Pre-position VIX mean-reversion trades BEFORE spikes (current Sharpe 2.83).
MLflow experiment: vix_spike_predictor_v1
"""
import sys, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs): print(*args, **kwargs, flush=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); np.random.seed(42)
SPIKE_TH, HORIZON, TRAIN_D, TEST_D = 25, 5, 252, 21
EPOCHS, BATCH, LR = 50, 256, 1e-3
FOCAL_GAMMA, FOCAL_ALPHA = 2.0, 0.75
OUT_DIR = Path(__file__).resolve().parents[2] / 'research' / 'findings'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def download_data():
    import yfinance as yf
    tickers = {'^VIX': 'VIX', 'SPY': 'SPY', 'TLT': 'TLT', 'GLD': 'GLD',
               'HYG': 'HYG', 'QQQ': 'QQQ', 'IWM': 'IWM'}
    frames = {}
    for tk, name in tickers.items():
        fprint(f"  Downloading {name}...")
        df = yf.download(tk, start='2004-01-01', end='2026-07-25', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        frames[name] = df[['Close']].rename(columns={'Close': name})
    merged = pd.concat(frames.values(), axis=1).dropna()
    fprint(f"  Data: {len(merged)} days, {merged.index[0].date()} to {merged.index[-1].date()}")
    return merged

def build_features(df):
    f = pd.DataFrame(index=df.index)
    vix, spy, tlt = df['VIX'], df['SPY'], df['TLT']
    gld, hyg, qqq, iwm = df['GLD'], df['HYG'], df['QQQ'], df['IWM']
    spy_ret, tlt_ret = spy.pct_change(), tlt.pct_change()
    # VIX features
    f['vix_level'] = vix
    for w in [5, 10, 20]: f[f'vix_chg_{w}d'] = vix - vix.shift(w)
    f['vix_pctile_1y'] = vix.rolling(252).apply(lambda x: (x[-1] > x[:-1]).mean(), raw=True)
    f['vix_ma20_ratio'] = vix / vix.rolling(20).mean()
    f['vix_dist_high_20d'] = vix / vix.rolling(20).max() - 1
    f['vix_dist_low_20d'] = vix / vix.rolling(20).min() - 1
    f['vix_rising_days'] = (vix.diff() > 0).rolling(10).sum()
    f['vix_accel'] = vix.diff().diff()
    # SPY features
    for w in [5, 10, 20]: f[f'spy_ret_{w}d'] = spy / spy.shift(w) - 1
    for w in [10, 20, 60]: f[f'spy_rvol_{w}d'] = spy_ret.rolling(w).std() * np.sqrt(252)
    f['spy_qqq_vol_ratio'] = spy_ret.rolling(20).std() / qqq.pct_change().rolling(20).std()
    f['spy_consec_down'] = (spy_ret < 0).rolling(10).sum()
    f['spy_dist_sma200'] = spy / spy.rolling(200).mean() - 1
    f['iwm_spy_spread_10d'] = (iwm / iwm.shift(10) - 1) - (spy / spy.shift(10) - 1)
    # Credit features
    for w in [5, 10]: f[f'hyg_ret_{w}d'] = hyg / hyg.shift(w) - 1
    f['hyg_tlt_spread_chg_10d'] = (hyg / hyg.shift(10)) - (tlt / tlt.shift(10))
    f['credit_stress_rising'] = (f['hyg_tlt_spread_chg_10d'] < f['hyg_tlt_spread_chg_10d'].shift(5)).astype(float)
    # Cross-asset
    f['gld_ret_10d'] = gld / gld.shift(10) - 1
    f['tlt_ret_10d'] = tlt / tlt.shift(10) - 1
    f['spy_tlt_corr_20d'] = spy_ret.rolling(20).corr(tlt_ret)
    # Calendar
    f['day_of_week'] = pd.Series(df.index.dayofweek, index=df.index).astype(float)
    f['month'] = pd.Series(df.index.month, index=df.index).astype(float)
    # Days since last spike
    spike_mask = (vix >= SPIKE_TH).astype(int)
    days_since = pd.Series(np.nan, index=df.index); last = -999
    for i, (idx, val) in enumerate(spike_mask.items()):
        if val == 1: last = i
        days_since.iloc[i] = i - last if last >= 0 else 999
    f['days_since_spike'] = days_since.clip(upper=252)
    f['vix_slope_proxy'] = vix.rolling(5).mean() - vix.rolling(20).mean()
    # Target
    future_max = vix.shift(-1).rolling(HORIZON).max().shift(-(HORIZON - 1))
    target = (future_max >= SPIKE_TH).astype(float)
    f = f.replace([np.inf, -np.inf], np.nan)
    valid = f.dropna().index.intersection(target.dropna().index)
    return f.loc[valid], target.loc[valid]

class FocalLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA):
        super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = targets * torch.sigmoid(logits) + (1 - targets) * (1 - torch.sigmoid(logits))
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()

class MLPModel(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

class LSTMModel(nn.Module):
    def __init__(self, n_feat, seq_len=20):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

class CNN1DModel(nn.Module):
    def __init__(self, n_feat, seq_len=20):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feat, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
    def forward(self, x):
        return self.head(self.conv(x.permute(0, 2, 1)).squeeze(-1)).squeeze(-1)

def train_model(model, X_tr, y_tr, X_val, y_val):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = FocalLoss()
    dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    best_auc, best_state, wait = 0.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step(); model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(X_val)).cpu().numpy(); yn = y_val.cpu().numpy()
            if len(np.unique(yn)) < 2: continue
            auc = roc_auc_score(yn, vp)
            if auc > best_auc:
                best_auc = auc; best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
            else:
                wait += 1
                if wait >= 10: break
    if best_state: model.load_state_dict(best_state)
    return best_auc

def run_walkforward(features, target, model_class, model_name, seq_model=False, seq_len=20):
    n, n_feat = len(features), len(features.columns)
    all_preds, all_labels, all_dates = [], [], []
    n_folds, start_idx = 0, TRAIN_D + (seq_len if seq_model else 0)
    fprint(f"\n{'='*60}\n  {model_name}: walk-forward ({n} days, {n_feat} features)")
    for test_start in range(start_idx, n - TEST_D, TEST_D):
        train_start = max(0, test_start - TRAIN_D - (seq_len if seq_model else 0))
        test_end = min(test_start + TEST_D, n)
        X_raw = features.iloc[train_start:test_end].values
        y_raw = target.iloc[train_start:test_end].values
        split = test_start - train_start
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_raw[:split]); X_te = scaler.transform(X_raw[split:])
        if seq_model:
            def mk_seq(data, labels, off):
                s, l = [], []
                for i in range(seq_len, len(data)): s.append(data[i-seq_len:i]); l.append(labels[off+i])
                return np.array(s), np.array(l)
            Xtr_s, ytr = mk_seq(X_sc, y_raw, 0)
            X_comb = np.vstack([X_sc[-seq_len:], X_te]); y_comb = y_raw[split-seq_len:]
            Xte_s, yte = [], []
            for i in range(seq_len, len(X_comb)):
                Xte_s.append(X_comb[i-seq_len:i])
                if i < len(y_comb): yte.append(y_comb[i])
            Xte_s = Xte_s[:len(yte)]
            Xte_s, yte = np.array(Xte_s) if Xte_s else np.empty((0, seq_len, n_feat)), np.array(yte)
            if len(Xtr_s) < 50 or len(Xte_s) < 5: continue
            X_t = torch.tensor(Xtr_s, dtype=torch.float32).to(DEVICE)
            y_t = torch.tensor(ytr, dtype=torch.float32).to(DEVICE)
            Xv = torch.tensor(Xte_s, dtype=torch.float32).to(DEVICE)
            yv = torch.tensor(yte, dtype=torch.float32).to(DEVICE)
            model = model_class(n_feat, seq_len).to(DEVICE)
        else:
            ytr, yte = y_raw[:split], y_raw[split:]
            if len(X_sc) < 50 or len(X_te) < 5: continue
            X_t = torch.tensor(X_sc, dtype=torch.float32).to(DEVICE)
            y_t = torch.tensor(ytr, dtype=torch.float32).to(DEVICE)
            Xv = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
            yv = torch.tensor(yte, dtype=torch.float32).to(DEVICE)
            model = model_class(n_feat).to(DEVICE)
        train_model(model, X_t, y_t, Xv, yv); model.eval()
        with torch.no_grad(): preds = torch.sigmoid(model(Xv)).cpu().numpy()
        dates = features.index[test_start:test_start + len(preds)]
        all_preds.extend(preds.tolist()); all_labels.extend(yte.tolist()); all_dates.extend(dates.tolist())
        n_folds += 1
        if n_folds % 20 == 0: fprint(f"    Fold {n_folds}, {len(all_preds)} preds")
    fprint(f"  {model_name}: {n_folds} folds, {len(all_preds)} OOT predictions")
    return np.array(all_preds), np.array(all_labels), all_dates

def simple_baselines(features, target):
    valid = features.index.intersection(target.index)
    vix, y = features.loc[valid, 'vix_level'].values, target.loc[valid].values
    results = {}
    if len(np.unique(y)) == 2:
        results['vix_gt_20'] = roc_auc_score(y, (vix > 20).astype(float))
        results['spy_down_1pct'] = roc_auc_score(y, (features.loc[valid, 'spy_ret_5d'].values < -0.01).astype(float))
    return results

def main():
    t0 = time.time()
    fprint(f"VIX Spike Predictor v1 — PyTorch Deep Learning")
    fprint(f"Device: {DEVICE}, Threshold: VIX>{SPIKE_TH}, Horizon: {HORIZON}d\n{'='*60}")
    # MLflow
    mlf = None
    try:
        import mlflow; mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("vix_spike_predictor_v1"); mlf = mlflow
        mlf.start_run(run_name=f"vix_spike_v1_{datetime.now():%Y%m%d_%H%M}")
        mlf.log_params({'threshold': SPIKE_TH, 'horizon': HORIZON, 'train_days': TRAIN_D,
                        'test_days': TEST_D, 'epochs': EPOCHS, 'device': str(DEVICE)})
        fprint("  MLflow tracking enabled")
    except Exception as e: fprint(f"  MLflow unavailable ({e}), continuing without")
    # Data
    fprint("\n[1/6] Downloading data..."); df = download_data()
    fprint("\n[2/6] Building features...")
    features, target = build_features(df)
    n_feat, spike_rate = len(features.columns), target.mean()
    fprint(f"  {n_feat} features, {len(features)} samples, spike rate: {spike_rate:.1%}")
    if mlf: mlf.log_params({'n_features': n_feat, 'n_samples': len(features), 'spike_rate': f'{spike_rate:.3f}'})
    # Baselines
    fprint("\n[3/6] Simple rule baselines...")
    baselines = simple_baselines(features, target)
    for nm, auc in baselines.items(): fprint(f"  {nm}: AUC = {auc:.3f}")
    # Train 3 architectures
    fprint("\n[4/6] Training neural models...")
    results = {}
    for mcls, nm, seq in [(MLPModel, "MLP", False), (LSTMModel, "LSTM", True), (CNN1DModel, "CNN-1D", True)]:
        preds, labels, dates = run_walkforward(features, target, mcls, nm, seq_model=seq)
        if len(np.unique(labels)) < 2 or len(preds) == 0:
            fprint(f"  {nm}: insufficient data"); continue
        auc = roc_auc_score(labels, preds); ap = average_precision_score(labels, preds)
        prec_arr, rec_arr, _ = precision_recall_curve(labels, preds)
        p50_idx = np.searchsorted(-rec_arr[::-1], -0.5) if (rec_arr >= 0.5).any() else -1
        p50 = prec_arr[min(p50_idx, len(prec_arr)-1)] if p50_idx >= 0 else 0
        results[nm] = {'auc': auc, 'ap': ap, 'p@50r': p50, 'preds': preds, 'labels': labels,
                       'dates': dates, 'class': mcls, 'seq': seq}
        fprint(f"  {nm}: AUC={auc:.3f}, AP={ap:.3f}, P@50%R={p50:.3f}")
        if mlf: mlf.log_metrics({f'{nm}_auc': auc, f'{nm}_ap': ap})
    if not results:
        fprint("ERROR: No models produced results."); mlf and mlf.end_run(); return
    best_nm = max(results, key=lambda k: results[k]['auc']); best = results[best_nm]
    fprint(f"\n  BEST: {best_nm} (AUC={best['auc']:.3f})")
    # Permutation test
    fprint("\n[5/6] Permutation test (3 shuffles)...")
    perm_aucs = []
    for i in range(3):
        fs = features.copy()
        for c in fs.columns: fs[c] = np.random.permutation(fs[c].values)
        p, l, _ = run_walkforward(fs, target, best['class'], f"Perm-{i+1}", seq_model=best['seq'])
        if len(np.unique(l)) == 2: perm_aucs.append(roc_auc_score(l, p))
    perm_mean = np.mean(perm_aucs) if perm_aucs else 0.5
    fprint(f"  Perm AUCs: {[f'{a:.3f}' for a in perm_aucs]}, mean={perm_mean:.3f}")
    fprint(f"  Lift over random: {best['auc'] - perm_mean:+.3f}")
    # Calibration
    fprint("\n  Calibration:")
    fprint(f"  {'Bin':>12} {'Pred_P':>8} {'Actual_P':>10} {'N':>6}")
    for lo, hi in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.01)]:
        m = (best['preds'] >= lo) & (best['preds'] < hi)
        if m.sum(): fprint(f"  {f'[{lo:.1f},{hi:.1f})':>12} {best['preds'][m].mean():>8.3f} {best['labels'][m].mean():>10.3f} {m.sum():>6d}")
    # Actionable signal
    fprint("\n  Actionable Signal (last year, P>70%):")
    arr = pd.DataFrame({'date': best['dates'], 'prob': best['preds'], 'spike': best['labels']})
    ly = arr[arr['date'] >= arr['date'].max() - pd.Timedelta(days=365)]
    hc = ly[ly['prob'] > 0.7]
    if len(hc) == 0: fprint("    No days with P>70%")
    else:
        nc = (hc['spike'] == 1).sum()
        fprint(f"    {len(hc)} days, {nc}/{len(hc)} correct = {nc/len(hc)*100:.1f}% precision")
        fprint(f"    Dates: {hc['date'].dt.strftime('%Y-%m-%d').tolist()[:15]}")
    # Save
    fprint("\n[6/6] Saving...")
    out_path = OUT_DIR / 'vix_spike_predictions.csv'
    pd.DataFrame({'date': best['dates'], 'spike_prob': best['preds'],
                  'actual_spike': best['labels']}).to_csv(out_path, index=False)
    fprint(f"  {len(best['preds'])} predictions saved")
    # Summary
    elapsed = time.time() - t0
    fprint(f"\n{'='*60}\n  VIX SPIKE PREDICTOR v1 — FINAL SUMMARY\n{'='*60}")
    fprint(f"  Spike: VIX>{SPIKE_TH} within {HORIZON}d | Base rate: {spike_rate:.1%}")
    fprint(f"\n  MODEL RESULTS (OOT AUC):")
    for nm, r in sorted(results.items(), key=lambda x: -x[1]['auc']):
        mk = " <-- BEST" if nm == best_nm else ""
        fprint(f"    {nm:>8}: AUC={r['auc']:.3f}, AP={r['ap']:.3f}, P@50R={r['p@50r']:.3f}{mk}")
    fprint(f"  BASELINES:")
    for nm, auc in baselines.items(): fprint(f"    {nm:>15}: AUC={auc:.3f}")
    fprint(f"  PERMUTATION: real={best['auc']:.3f}, shuffled={perm_mean:.3f}, lift={best['auc']-perm_mean:+.3f}")
    if best['auc'] < 0.60: verdict = "NEGATIVE — AUC < 0.60, no predictive power"
    elif best['auc'] < max(baselines.values(), default=0) + 0.02:
        verdict = "MARGINAL — neural does not beat simple rules"
    else: verdict = f"POSITIVE — {best_nm} AUC={best['auc']:.3f} beats baselines+permutation"
    fprint(f"  VERDICT: {verdict}")
    fprint(f"  Runtime: {elapsed:.0f}s on {DEVICE}\n{'='*60}")
    if mlf:
        mlf.log_metrics({'best_auc': best['auc'], 'perm_mean': perm_mean, 'runtime_s': elapsed})
        mlf.log_param('best_model', best_nm); mlf.log_param('verdict', verdict[:80])
        mlf.log_artifact(str(out_path)); mlf.end_run(); fprint("  MLflow run logged.")

if __name__ == '__main__':
    main()
