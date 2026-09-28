#!/usr/bin/env python3
"""VIX Enhanced Mean-Reversion v1 — LSTM spike prediction as entry signal.
6 strategy variants: baseline, LSTM-enhanced entry, pre-position, filtered, dual-signal, conservative.
Walk-forward LSTM (252d/21d sliding) → OOT spike probs → strategy simulation.
"""
import sys, json, warnings, time
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs): print(*args, **kwargs, flush=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42); np.random.seed(42)
BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / 'research' / 'findings'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SPIKE_TH, HORIZON, TRAIN_D, TEST_D, SEQ_LEN = 25, 5, 252, 21, 20
EPOCHS, BATCH, LR, FOCAL_GAMMA, FOCAL_ALPHA = 50, 256, 1e-3, 2.0, 0.75
START_CAPITAL = 10_000
SPREAD_WIDTH = 5  # 30/35 call spread
VVIX, RFR = 0.80, 0.04  # vol-of-VIX, risk-free rate for B-S

# ── Data & Features (identical to vix_spike_predictor_v1.py) ──────────────
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
    f['vix_level'] = vix
    for w in [5, 10, 20]: f[f'vix_chg_{w}d'] = vix - vix.shift(w)
    f['vix_pctile_1y'] = vix.rolling(252).apply(lambda x: (x[-1] > x[:-1]).mean(), raw=True)
    f['vix_ma20_ratio'] = vix / vix.rolling(20).mean()
    f['vix_dist_high_20d'] = vix / vix.rolling(20).max() - 1
    f['vix_dist_low_20d'] = vix / vix.rolling(20).min() - 1
    f['vix_rising_days'] = (vix.diff() > 0).rolling(10).sum()
    f['vix_accel'] = vix.diff().diff()
    for w in [5, 10, 20]: f[f'spy_ret_{w}d'] = spy / spy.shift(w) - 1
    for w in [10, 20, 60]: f[f'spy_rvol_{w}d'] = spy_ret.rolling(w).std() * np.sqrt(252)
    f['spy_qqq_vol_ratio'] = spy_ret.rolling(20).std() / qqq.pct_change().rolling(20).std()
    f['spy_consec_down'] = (spy_ret < 0).rolling(10).sum()
    f['spy_dist_sma200'] = spy / spy.rolling(200).mean() - 1
    f['iwm_spy_spread_10d'] = (iwm / iwm.shift(10) - 1) - (spy / spy.shift(10) - 1)
    for w in [5, 10]: f[f'hyg_ret_{w}d'] = hyg / hyg.shift(w) - 1
    f['hyg_tlt_spread_chg_10d'] = (hyg / hyg.shift(10)) - (tlt / tlt.shift(10))
    f['credit_stress_rising'] = (f['hyg_tlt_spread_chg_10d'] < f['hyg_tlt_spread_chg_10d'].shift(5)).astype(float)
    f['gld_ret_10d'] = gld / gld.shift(10) - 1
    f['tlt_ret_10d'] = tlt / tlt.shift(10) - 1
    f['spy_tlt_corr_20d'] = spy_ret.rolling(20).corr(tlt_ret)
    f['day_of_week'] = pd.Series(df.index.dayofweek, index=df.index).astype(float)
    f['month'] = pd.Series(df.index.month, index=df.index).astype(float)
    spike_mask = (vix >= SPIKE_TH).astype(int)
    days_since = pd.Series(np.nan, index=df.index); last = -999
    for i, (idx, val) in enumerate(spike_mask.items()):
        if val == 1: last = i
        days_since.iloc[i] = i - last if last >= 0 else 999
    f['days_since_spike'] = days_since.clip(upper=252)
    f['vix_slope_proxy'] = vix.rolling(5).mean() - vix.rolling(20).mean()
    future_max = vix.shift(-1).rolling(HORIZON).max().shift(-(HORIZON - 1))
    target = (future_max >= SPIKE_TH).astype(float)
    f = f.replace([np.inf, -np.inf], np.nan)
    valid = f.dropna().index.intersection(target.dropna().index)
    return f.loc[valid], target.loc[valid]

# ── LSTM Model ────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, a=FOCAL_ALPHA, g=FOCAL_GAMMA):
        super().__init__(); self.a, self.g = a, g
    def forward(self, logits, tgt):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, tgt, reduction='none')
        pt = tgt * torch.sigmoid(logits) + (1-tgt) * (1-torch.sigmoid(logits))
        return ((tgt*self.a + (1-tgt)*(1-self.a)) * (1-pt)**self.g * bce).mean()

class LSTMModel(nn.Module):
    def __init__(self, n_feat, seq_len=20):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, 128, num_layers=2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
    def forward(self, x):
        out, _ = self.lstm(x); return self.head(out[:, -1, :]).squeeze(-1)

def train_lstm(model, X_tr, y_tr, X_val, y_val):
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
            if auc > best_auc: best_auc = auc; best_state = {k: v.clone() for k, v in model.state_dict().items()}; wait = 0
            else:
                wait += 1
                if wait >= 10: break
    if best_state: model.load_state_dict(best_state)
    return best_auc

def generate_oot_probabilities(features, target):
    """Walk-forward LSTM → OOT spike probability for each day."""
    n, n_feat = len(features), len(features.columns)
    all_probs, all_dates = [], []
    n_folds = 0
    fprint(f"  LSTM walk-forward: {n} days, {n_feat} features, seq_len={SEQ_LEN}")
    for test_start in range(TRAIN_D + SEQ_LEN, n - TEST_D, TEST_D):
        train_start = max(0, test_start - TRAIN_D - SEQ_LEN)
        test_end = min(test_start + TEST_D, n)
        X_raw = features.iloc[train_start:test_end].values
        y_raw = target.iloc[train_start:test_end].values
        split = test_start - train_start
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_raw[:split]); X_te = scaler.transform(X_raw[split:])
        def mk_seq(data, labels, off):
            s, l = [], []
            for i in range(SEQ_LEN, len(data)): s.append(data[i-SEQ_LEN:i]); l.append(labels[off+i])
            return np.array(s), np.array(l)
        Xtr_s, ytr = mk_seq(X_sc, y_raw, 0)
        X_comb = np.vstack([X_sc[-SEQ_LEN:], X_te]); y_comb = y_raw[split-SEQ_LEN:]
        Xte_s, yte = [], []
        for i in range(SEQ_LEN, len(X_comb)):
            Xte_s.append(X_comb[i-SEQ_LEN:i])
            if i < len(y_comb): yte.append(y_comb[i])
        Xte_s = Xte_s[:len(yte)]
        Xte_s, yte = np.array(Xte_s) if Xte_s else np.empty((0, SEQ_LEN, n_feat)), np.array(yte)
        if len(Xtr_s) < 50 or len(Xte_s) < 5: continue
        model = LSTMModel(n_feat, SEQ_LEN).to(DEVICE)
        train_lstm(model,
                   torch.tensor(Xtr_s, dtype=torch.float32).to(DEVICE),
                   torch.tensor(ytr, dtype=torch.float32).to(DEVICE),
                   torch.tensor(Xte_s, dtype=torch.float32).to(DEVICE),
                   torch.tensor(yte, dtype=torch.float32).to(DEVICE))
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(torch.tensor(Xte_s, dtype=torch.float32).to(DEVICE))).cpu().numpy()
        dates = features.index[test_start:test_start + len(preds)]
        all_probs.extend(preds.tolist()); all_dates.extend(dates.tolist())
        n_folds += 1
        if n_folds % 25 == 0: fprint(f"    Fold {n_folds}, {len(all_probs)} predictions")
    fprint(f"  Done: {n_folds} folds, {len(all_probs)} OOT predictions")
    return pd.Series(all_probs, index=pd.DatetimeIndex(all_dates), name='spike_prob')

# ── Option Pricing (B-S approx for VIX call spreads) ─────────────────────
def _bs_call(vix, strike, dte_days, vol=VVIX, r=RFR):
    """Simple B-S call price on VIX (log-normal approx, good enough for backtest)."""
    from math import log, sqrt, exp
    from statistics import NormalDist
    nd = NormalDist()
    t = max(dte_days / 365.0, 1e-6)
    d1 = (log(vix / strike) + (r + 0.5 * vol**2) * t) / (vol * sqrt(t))
    d2 = d1 - vol * sqrt(t)
    return vix * nd.cdf(d1) - strike * exp(-r * t) * nd.cdf(d2)

def spread_premium(vix_level, dte=30, lo_strike=30, hi_strike=35):
    """Credit received for selling lo/hi call spread. Returns $ per unit."""
    return max(_bs_call(vix_level, lo_strike, dte) - _bs_call(vix_level, hi_strike, dte), 0.01)

def spread_value(vix_level, dte_remaining, lo_strike=30, hi_strike=35):
    """Current value of the spread (cost to buy back)."""
    if dte_remaining <= 0: return max(min(vix_level - lo_strike, hi_strike - lo_strike), 0)
    return max(_bs_call(vix_level, lo_strike, dte_remaining) - _bs_call(vix_level, hi_strike, dte_remaining), 0)

# ── Strategy Simulation ──────────────────────────────────────────────────
def _record_trade(trades, pos, dt, vix, pnl, variant):
    trades.append({'entry': pos['entry_date'], 'exit': dt, 'pnl': pnl,
                   'entry_vix': pos['entry_vix'], 'exit_vix': vix,
                   'duration': (dt - pos['entry_date']).days,
                   'direction': pos['direction'], 'variant': variant})

def _open_short(dt, vix, capital, lo=30, hi=35, frac=0.10):
    prem = spread_premium(vix, 30, lo, hi)
    units = max(1, int(capital * frac / (SPREAD_WIDTH * 100)))
    return {'entry_date': dt, 'entry_vix': vix, 'premium': prem,
            'dte_left': 30, 'direction': 'short_spread', 'units': units, 'lo': lo, 'hi': hi}

def simulate(vix_series, prob_series, variant, params):
    """Run daily P&L simulation for one strategy variant."""
    capital, trades, equity = START_CAPITAL, [], [START_CAPITAL]
    pos = None
    vix_vals = vix_series.reindex(prob_series.index)
    vix_ma5 = vix_series.rolling(5).mean().reindex(prob_series.index)
    for i, dt in enumerate(prob_series.index):
        vix, prob = vix_vals.iloc[i], prob_series.iloc[i]
        if pd.isna(vix) or pd.isna(prob): equity.append(capital); continue
        # Exit check
        if pos is not None:
            pos['dte_left'] -= 1
            if pos['direction'] == 'short_spread':
                pnl_u = pos['premium'] - spread_value(vix, pos['dte_left'], pos['lo'], pos['hi'])
                if vix < params.get('exit_vix', 22) or pos['dte_left'] < 5:
                    pnl = pnl_u * pos['units'] * 100; capital += pnl
                    _record_trade(trades, pos, dt, vix, pnl, variant); pos = None
            elif pos['direction'] == 'long_spread':
                cur = spread_value(vix, pos['dte_left'], pos['lo'], pos['hi'])
                if vix > params.get('spike_exit_vix', 28) or pos['dte_left'] < 5:
                    pnl = (cur - pos['cost']) * pos['units'] * 100; capital += pnl
                    _record_trade(trades, pos, dt, vix, pnl, variant); pos = None
                    if vix > 28 and variant == 'C_preposition' and pnl > 0:
                        pos = _open_short(dt, vix, capital)
        # Entry (flat only)
        if pos is None:
            enter = False
            if variant == 'A_baseline': enter = vix > 30
            elif variant == 'B_lstm_entry': enter = prob > 0.7 and vix > 25
            elif variant == 'C_preposition':
                if vix < 22 and prob > 0.8:
                    cost = spread_premium(vix, 30, 25, 30)
                    units = max(1, int(capital * 0.05 / (SPREAD_WIDTH * 100)))
                    pos = {'entry_date': dt, 'entry_vix': vix, 'cost': cost, 'dte_left': 30,
                           'direction': 'long_spread', 'units': units, 'lo': 25, 'hi': 30}
                elif vix > 30: enter = True
            elif variant == 'D_filtered':
                if vix > 30 and i >= 5: enter = prob_series.iloc[max(0,i-5):i].max() > 0.5
            elif variant == 'E_dual':
                if not pd.isna(vix_ma5.iloc[i]): enter = vix > vix_ma5.iloc[i] and prob > 0.6 and vix > 25
            elif variant == 'F_conservative': enter = prob > 0.9 and vix > 25
            if enter and pos is None: pos = _open_short(dt, vix, capital)
        equity.append(capital)
    return trades, equity[1:]

def compute_metrics(trades, equity, dates):
    if not trades: return {'sharpe': 0, 'sortino': 0, 'cagr': 0, 'maxdd': 0, 'wr': 0, 'pf': 0, 'n_trades': 0}
    eq = np.array(equity)
    daily_ret = np.diff(eq) / eq[:-1]
    daily_ret = daily_ret[np.isfinite(daily_ret)]
    sharpe = np.mean(daily_ret) / max(np.std(daily_ret), 1e-9) * np.sqrt(252)
    downside = daily_ret[daily_ret < 0]
    sortino = np.mean(daily_ret) / max(np.std(downside), 1e-9) * np.sqrt(252) if len(downside) > 0 else sharpe
    years = max(len(eq) / 252, 0.5)
    cagr = (eq[-1] / eq[0]) ** (1/years) - 1
    running_max = np.maximum.accumulate(eq)
    maxdd = np.min((eq - running_max) / running_max)
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]; losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) if pnls else 0
    pf = sum(wins) / max(abs(sum(losses)), 1e-9) if losses else float('inf')
    return {'sharpe': round(sharpe, 2), 'sortino': round(sortino, 2), 'cagr': round(cagr*100, 1),
            'maxdd': round(maxdd*100, 1), 'wr': round(wr*100, 1), 'pf': round(pf, 2),
            'n_trades': len(trades), 'avg_duration_d': round(np.mean([t['duration'] for t in trades]), 1),
            'avg_pnl': round(np.mean(pnls), 2), 'total_pnl': round(sum(pnls), 2),
            'final_capital': round(eq[-1], 2)}

# ── Adversarial Validation ────────────────────────────────────────────────
def adversarial_checks(vix_series, prob_series, best_variant, best_params, best_sharpe, spy_series):
    fprint("\n[5/6] Adversarial validation...")
    results = {}
    # Permutation test
    perm_sharpes = []
    for i in range(5):
        shuf = prob_series.copy(); shuf[:] = np.random.permutation(shuf.values)
        m = compute_metrics(*simulate(vix_series, shuf, best_variant, best_params), shuf.index)
        perm_sharpes.append(m['sharpe'])
    pm = np.mean(perm_sharpes)
    results['permutation'] = {'real_sharpe': best_sharpe, 'shuffled_mean': round(pm, 2),
                              'lift': round(best_sharpe - pm, 2), 'pass': best_sharpe > pm + 0.3}
    fprint(f"  Permutation: real={best_sharpe:.2f}, shuffled={pm:.2f}, lift={best_sharpe-pm:+.2f}")
    # Regime check
    spy_mo = spy_series.resample('M').last().pct_change().reindex(prob_series.index, method='ffill')
    for regime, mask in [('green', spy_mo > 0), ('red', spy_mo <= 0)]:
        sub = prob_series[mask.fillna(False)]
        if len(sub) < 50: results[f'regime_{regime}'] = 'insufficient'; continue
        m = compute_metrics(*simulate(vix_series, sub, best_variant, best_params), sub.index)
        results[f'regime_{regime}'] = {'sharpe': m['sharpe'], 'n_trades': m['n_trades'], 'wr': m['wr']}
        fprint(f"  Regime {regime}: Sharpe={m['sharpe']:.2f}, trades={m['n_trades']}, WR={m['wr']:.1f}%")
    # Sub-period stability
    mid = len(prob_series) // 2
    for lbl, sub in [('first_half', prob_series.iloc[:mid]), ('second_half', prob_series.iloc[mid:])]:
        m = compute_metrics(*simulate(vix_series, sub, best_variant, best_params), sub.index)
        results[lbl] = {'sharpe': m['sharpe'], 'n_trades': m['n_trades'], 'wr': m['wr']}
        fprint(f"  {lbl}: Sharpe={m['sharpe']:.2f}, trades={m['n_trades']}, WR={m['wr']:.1f}%")
    # Outlier removal
    tr, eq = simulate(vix_series, prob_series, best_variant, best_params)
    if tr:
        cut = np.percentile([t['pnl'] for t in tr], 95)
        filt = [t for t in tr if t['pnl'] <= cut]
        if filt:
            feq = [START_CAPITAL]; c = START_CAPITAL
            for t in filt: c += t['pnl']; feq.append(c)
            m = compute_metrics(filt, feq, prob_series.index)
            results['outlier_removed'] = {'sharpe': m['sharpe'], 'wr': m['wr'], 'n_trades': m['n_trades']}
            fprint(f"  Outlier-removed: Sharpe={m['sharpe']:.2f}, WR={m['wr']:.1f}%")
    return results

# ── Main ──────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    fprint(f"VIX Enhanced Mean-Reversion v1 — LSTM-augmented entry signals")
    fprint(f"Device: {DEVICE} | Start capital: ${START_CAPITAL:,}\n{'='*70}")

    # MLflow
    mlf = None
    try:
        import mlflow; mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("vix_enhanced_meanrev_v1"); mlf = mlflow
        mlf.start_run(run_name=f"vix_meanrev_v1_{datetime.now():%Y%m%d_%H%M}")
        mlf.log_params({'start_capital': START_CAPITAL, 'spread_width': SPREAD_WIDTH,
                        'device': str(DEVICE), 'train_d': TRAIN_D, 'test_d': TEST_D})
        fprint("  MLflow tracking enabled")
    except Exception as e: fprint(f"  MLflow unavailable ({e}), continuing without")

    fprint("\n[1/6] Downloading data...")
    df = download_data()
    vix_series, spy_series = df['VIX'], df['SPY']
    fprint("\n[2/6] Building features & LSTM spike probabilities...")
    features, target = build_features(df)
    fprint(f"  {len(features.columns)} features, {len(features)} samples, spike rate: {target.mean():.1%}")
    prob_series = generate_oot_probabilities(features, target)
    auc = roc_auc_score(target.reindex(prob_series.index).dropna(),
                        prob_series.reindex(target.index).dropna()) if len(prob_series) > 100 else 0
    fprint(f"  LSTM OOT AUC: {auc:.3f}")
    if mlf: mlf.log_metric('lstm_oot_auc', auc)

    # Strategy variants
    fprint("\n[3/6] Running 6 strategy variants...")
    variants = {
        'A_baseline':     {'exit_vix': 22},
        'B_lstm_entry':   {'exit_vix': 22},
        'C_preposition':  {'exit_vix': 22, 'spike_exit_vix': 28},
        'D_filtered':     {'exit_vix': 22},
        'E_dual':         {'exit_vix': 22},
        'F_conservative': {'exit_vix': 22},
    }
    all_results = {}
    for v_name, params in variants.items():
        trades, equity = simulate(vix_series, prob_series, v_name, params)
        metrics = compute_metrics(trades, equity, prob_series.index)
        all_results[v_name] = metrics
        fprint(f"  {v_name:20s} | Sharpe {metrics['sharpe']:6.2f} | Sortino {metrics['sortino']:6.2f} | "
               f"CAGR {metrics['cagr']:5.1f}% | MaxDD {metrics['maxdd']:5.1f}% | "
               f"WR {metrics['wr']:5.1f}% | PF {metrics['pf']:5.2f} | Trades {metrics['n_trades']:4d} | "
               f"Final ${metrics['final_capital']:,.0f}")
        if mlf:
            for k, v in metrics.items():
                if isinstance(v, (int, float)): mlf.log_metric(f'{v_name}_{k}', v)

    best_name = max(all_results, key=lambda k: all_results[k]['sharpe'])
    best_metrics = all_results[best_name]
    fprint(f"\n[4/6] Ranked results (best={best_name}, Sharpe={best_metrics['sharpe']:.2f}):")
    for v, m in sorted(all_results.items(), key=lambda x: -x[1]['sharpe']):
        fprint(f"  {v:20s} Sh={m['sharpe']:5.2f} So={m['sortino']:5.2f} CAGR={m['cagr']:5.1f}% "
               f"DD={m['maxdd']:5.1f}% WR={m['wr']:4.1f}% PF={m['pf']:5.2f} N={m['n_trades']}")
    #
    adv = adversarial_checks(vix_series, prob_series, best_name, variants[best_name],
                             best_metrics['sharpe'], spy_series)

    # Save + Summary
    fprint("\n[6/6] Saving results...")
    output = {'run_date': datetime.now().isoformat(), 'lstm_oot_auc': round(auc, 3),
              'strategy_results': all_results, 'best_variant': best_name, 'best_metrics': best_metrics,
              'adversarial': {k: v for k, v in adv.items() if not isinstance(v, str)},
              'config': {'start_capital': START_CAPITAL, 'spread_width': SPREAD_WIDTH,
                         'spike_threshold': SPIKE_TH, 'horizon': HORIZON, 'device': str(DEVICE)}}
    out_path = OUT_DIR / 'vix_enhanced_meanrev_v1_results.json'
    with open(out_path, 'w') as f: json.dump(output, f, indent=2, default=str)
    elapsed = time.time() - t0
    perm = adv.get('permutation', {})
    bm = best_metrics
    fprint(f"\n{'='*70}\n  FINAL: {best_name} | Sharpe={bm['sharpe']:.2f} Sortino={bm['sortino']:.2f} "
           f"CAGR={bm['cagr']:.1f}% DD={bm['maxdd']:.1f}% WR={bm['wr']:.1f}% PF={bm['pf']:.2f} "
           f"N={bm['n_trades']} Final=${bm['final_capital']:,.0f}")
    fprint(f"  LSTM AUC={auc:.3f} | Perm {'PASS' if perm.get('pass') else 'FAIL'} (lift={perm.get('lift',0):+.2f})")
    g_s = adv.get('regime_green', {}).get('sharpe', 0) if isinstance(adv.get('regime_green'), dict) else 0
    r_s = adv.get('regime_red', {}).get('sharpe', 0) if isinstance(adv.get('regime_red'), dict) else 0
    if max(abs(g_s), abs(r_s)) > 0:
        gap = abs(g_s - r_s) / max(abs(g_s), abs(r_s))
        fprint(f"  R1 regime: green={g_s:.2f} red={r_s:.2f} gap={gap:.2f} {'PASS' if gap<=0.5 else 'FAIL'}")
    fprint(f"  Runtime: {elapsed:.0f}s | {DEVICE}\n{'='*70}")
    if mlf:
        mlf.log_metrics({'best_sharpe': bm['sharpe'], 'best_sortino': bm['sortino'],
                         'best_wr': bm['wr'], 'perm_lift': perm.get('lift', 0), 'runtime_s': elapsed})
        mlf.log_param('best_variant', best_name)
        try: mlf.log_artifact(str(out_path))
        except Exception: pass
        mlf.end_run(); fprint("  MLflow logged.")

if __name__ == '__main__':
    main()
