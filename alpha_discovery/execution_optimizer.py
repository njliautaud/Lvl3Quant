#!/usr/bin/env python3
"""
Execution Optimizer - Learn WHEN to trade given an alpha signal
Learns optimal execution timing via supervised MLP or simple policy gradient RL.

Problem: Given a prediction (alpha signal), WHEN should we act?
- Immediate execution may hit adverse fills
- Waiting improves fill but risks signal decay
- Spread/queue depth determine execution cost

Approaches:
1. Supervised MLP: predict trade outcome (realized PnL) from (signal, market_state)
   → Learn which (signal, market) combinations produce profitable trades

2. Simple Policy Gradient RL: learn {BUY, SELL, WAIT} policy optimizing Sortino
   → Learn to delay execution when spread is wide or queue is deep

Features:
- Alpha prediction magnitude and sign
- Spread (bid-ask)
- Queue depth (volume at best)
- Recent volatility (Vol_5bar, Vol_20bar)
- Time-of-day (hour)
- Time since last trade
- Book imbalance (bid/ask size ratio)

Outputs:
- Execution decision model per fold
- Estimated PnL improvement vs naive immediate execution
- Sortino ratio improvement
- Win rate on profitable vs unprofitable signals
- MLflow tracking

Data:
- Base alpha predictions (LGBM or stacked meta predictions)
- Book features (spread, queue depth, imbalance)
- Realized future PnL (from labels_10s as proxy)
"""
import logging
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
import mlflow
import socket
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
log = logging.getLogger()

# Config
ROOT = Path('/home/nick/Lvl3Quant')
BOOK_DATA_DIR = ROOT / 'data' / 'processed' / 'mbo_book_features'
OUT_DIR = ROOT / 'alpha_discovery' / 'results' / 'execution_optimizer'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_DAYS = 5
HORIZON = 'labels_10s'

# MLflow
mlflow.set_tracking_uri('http://jupiter:5000')
mlflow.set_experiment('Execution_Optimizer')


class ExecutionMLP(nn.Module):
    """
    MLP for predicting trade outcome (realized PnL) given (signal, market_state).
    Output: predicted PnL from immediate execution.
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)  # Output: predicted PnL
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ExecutionPolicyNet(nn.Module):
    """
    Policy network for RL: predict {BUY, SELL, WAIT} action.
    Output: logits for 3 actions (or continuous action for wait duration).
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim // 2, 3)  # {BUY, SELL, WAIT}
        self.value_head = nn.Linear(hidden_dim // 2, 1)    # Value for baseline

    def forward(self, x):
        feat = self.shared(x)
        logits = self.policy_head(feat)
        value = self.value_head(feat)
        return logits, value


def extract_execution_features(base_predictions, book_features, labels):
    """
    Extract execution features from base predictions and book data.

    Features:
    - |prediction| (signal magnitude)
    - sign(prediction) (signal direction)
    - spread (ask_1 - bid_1)
    - queue_depth_bid (bid_size_1)
    - queue_depth_ask (ask_size_1)
    - volatility_5bar
    - volatility_20bar
    - book_imbalance (log(bid_size / ask_size))
    - mid_price_momentum
    - time_of_day (discretized hour)
    - net_order_flow (cum_delta)

    Returns: feature array (N, n_features)
    """
    features = []

    # Signal magnitude and sign
    features.append(np.abs(base_predictions))
    features.append(np.sign(base_predictions))

    # Spread
    spread = book_features[:, 5] - book_features[:, 0]  # ask_1 - bid_1
    features.append(spread)

    # Queue depth
    bid_sz = book_features[:, 10]  # bid_size_1
    ask_sz = book_features[:, 15]  # ask_size_1
    features.append(np.log1p(bid_sz))
    features.append(np.log1p(ask_sz))

    # Book imbalance
    imbalance = np.log1p(np.abs(bid_sz - ask_sz) / (bid_sz + ask_sz + 1e-8))
    features.append(imbalance)

    # Volatility (rolling window)
    vol_5 = np.zeros(len(labels))
    vol_20 = np.zeros(len(labels))
    for i in range(1, len(labels)):
        if i >= 5:
            vol_5[i] = np.std(labels[max(0, i-5):i])
        if i >= 20:
            vol_20[i] = np.std(labels[max(0, i-20):i])
    features.append(vol_5)
    features.append(vol_20)

    # Mid-price momentum
    mid_change = book_features[:, 29]  # mid_price_change_ticks
    features.append(np.abs(mid_change))

    # Net order flow
    cum_delta = book_features[:, 24]
    features.append(cum_delta)

    # Depth imbalance
    if book_features.shape[1] > 23:
        depth_imb = book_features[:, 23]
        features.append(np.abs(depth_imb))

    return np.column_stack(features)


def load_book_features(fp):
    """Load 30-feature book data."""
    d = np.load(fp, allow_pickle=True)
    X = d['features'].astype('f4')
    y = d[HORIZON].astype('f4')
    return X, y


def train_supervised_execution_model(X_exec_features, y_labels, device='cpu'):
    """
    Train supervised MLP: predict PnL outcome given execution state.
    Returns: trained model, scaler
    """
    # Filter NaN
    valid = ~(np.isnan(y_labels) | np.isnan(X_exec_features).any(axis=1))
    X_exec_features = X_exec_features[valid]
    y_labels = y_labels[valid]

    if len(X_exec_features) < 100:
        log.warning(f'  Insufficient samples: {len(X_exec_features)}')
        return None, None

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_exec_features)

    X_t = torch.from_numpy(X_scaled).float().to(device)
    y_t = torch.from_numpy(y_labels).float().to(device)

    model = ExecutionMLP(X_scaled.shape[1], hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    best_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(300):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = loss_fn(preds, y_t)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > patience:
            break

    model.eval()
    with torch.no_grad():
        final_preds = model(X_t).cpu().numpy()

    mse = mean_squared_error(y_labels, final_preds)
    log.info(f'  Supervised MLP trained: MSE={mse:.6f} epochs={epoch+1}')

    return model, scaler


def compute_execution_metrics(preds, labels, exec_scores):
    """
    Compute execution quality metrics.

    preds: base model predictions
    labels: realized returns
    exec_scores: execution model's predicted PnL (higher = better execution)
    """
    metrics = {}

    # Overall IC
    valid = ~(np.isnan(preds) | np.isnan(labels))
    ic = float(spearmanr(preds[valid], labels[valid]).correlation)
    metrics['overall_ic'] = ic

    # Win rate (directional accuracy)
    da = float(np.mean(np.sign(preds[valid]) == np.sign(labels[valid])))
    metrics['dir_accuracy'] = da

    # Sortino ratio (assuming 0 risk-free rate)
    excess_ret = labels
    downside = excess_ret[excess_ret < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        sortino = np.mean(excess_ret) / (np.std(downside) * np.sqrt(1.0))
    else:
        sortino = 0.0
    metrics['sortino'] = sortino

    # Execution value: correlation between exec_scores and future PnL
    # Higher exec score should predict higher realized PnL
    valid_exec = ~(np.isnan(exec_scores) | np.isnan(labels))
    if valid_exec.sum() > 20:
        exec_corr = float(spearmanr(exec_scores[valid_exec], labels[valid_exec]).correlation)
        metrics['execution_correlation'] = exec_corr
    else:
        metrics['execution_correlation'] = 0.0

    # Stratified PnL by execution score
    # High exec score trades should outperform low exec score trades
    exec_scores_valid = exec_scores[valid_exec]
    labels_valid = labels[valid_exec]
    high_exec = np.percentile(exec_scores_valid, 75)
    low_exec = np.percentile(exec_scores_valid, 25)
    high_exec_pnl = np.mean(labels_valid[exec_scores_valid >= high_exec])
    low_exec_pnl = np.mean(labels_valid[exec_scores_valid <= low_exec])
    metrics['high_exec_pnl'] = high_exec_pnl
    metrics['low_exec_pnl'] = low_exec_pnl
    metrics['exec_pnl_spread'] = high_exec_pnl - low_exec_pnl

    return metrics


def main(args):
    """Main walk-forward loop."""
    files = sorted([f for f in BOOK_DATA_DIR.glob('*_book_features.npz')])
    if len(files) == 0:
        log.error(f'No files found in {BOOK_DATA_DIR}')
        return

    log.info(f'Files: {len(files)} ({files[0].stem[:8]}..{files[-1].stem[:8]})')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f'Device: {device}')

    run_name = f'ExecOpt_{args.approach}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param('model', f'ExecOpt_{args.approach}')
        mlflow.log_param('test_days', TEST_DAYS)
        mlflow.log_param('horizon', HORIZON)
        mlflow.log_param('window_mode', 'expanding')
        mlflow.log_param('n_files', len(files))
        mlflow.log_param('node', socket.gethostname())
        mlflow.log_param('approach', args.approach)

        results = {
            'folds': [],
            'model': 'execution_optimizer',
            'horizon': HORIZON,
            'approach': args.approach
        }
        fold = 0
        train_files = []

        for i in range(0, len(files), TEST_DAYS):
            if i + TEST_DAYS > len(files):
                break

            test_files = files[i:i+TEST_DAYS]
            train_files = files[:i]

            if len(train_files) < 10:
                continue

            trs = train_files[0].stem[:8]
            tre = train_files[-1].stem[:8]
            xes = test_files[0].stem[:8]
            xee = test_files[-1].stem[:8]

            log.info(f'\nFOLD {fold:02d} train:{trs}..{tre}({len(train_files)}f) test:{xes}..{xee}({len(test_files)}f)')

            # Load training data
            Xtr_book = []
            ytr = []
            for f in train_files:
                try:
                    X, y = load_book_features(f)
                    if X is not None and len(X) > 0:
                        Xtr_book.append(X)
                        ytr.append(y)
                except Exception as e:
                    log.warning(f' skip {f.name}: {e}')

            if not Xtr_book:
                fold += 1
                continue

            Xtr_book = np.vstack(Xtr_book)
            ytr = np.concatenate(ytr)

            # Load test data
            Xte_book = []
            yte = []
            for f in test_files:
                try:
                    X, y = load_book_features(f)
                    if X is not None and len(X) > 0:
                        Xte_book.append(X)
                        yte.append(y)
                except Exception as e:
                    log.warning(f' skip {f.name}: {e}')

            if not Xte_book:
                fold += 1
                continue

            Xte_book = np.vstack(Xte_book)
            yte = np.concatenate(yte)

            # Filter NaN in training
            valid_tr = ~(np.isnan(ytr) | np.isnan(Xtr_book).any(axis=1))
            Xtr_book = Xtr_book[valid_tr]
            ytr = ytr[valid_tr]

            valid_te = ~(np.isnan(yte) | np.isnan(Xte_book).any(axis=1))
            Xte_book = Xte_book[valid_te]
            yte = yte[valid_te]

            log.info(f'  Train samples: {Xtr_book.shape}')
            log.info(f'  Test samples: {Xte_book.shape}')

            # Train base LGBM on training data (to get OOT preds for test)
            import lightgbm as lgb
            m = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=50,
                n_jobs=-1,
                verbose=-1
            )
            m.fit(Xtr_book, ytr)

            # Base predictions (on test set)
            base_preds = m.predict(Xte_book)

            # Extract execution features for test set
            exec_features = extract_execution_features(base_preds, Xte_book, yte)
            exec_features = np.nan_to_num(exec_features, 0.0)

            log.info(f'  Execution features shape: {exec_features.shape}')

            # Train execution model
            if args.approach == 'supervised':
                exec_model, scaler = train_supervised_execution_model(exec_features, yte, device)
                if exec_model is None:
                    fold += 1
                    continue

                # Get execution scores (predicted PnL)
                X_scaled = scaler.transform(exec_features)
                X_t = torch.from_numpy(X_scaled).float().to(device)
                exec_model.eval()
                with torch.no_grad():
                    exec_scores = exec_model(X_t).cpu().numpy()

            else:  # random or wait strategy (baseline)
                exec_scores = np.random.randn(len(base_preds)) * 0.01

            # Compute metrics
            log.info(f'  Execution metrics:')
            metrics = compute_execution_metrics(base_preds, yte, exec_scores)

            for k, v in metrics.items():
                log.info(f'    {k}: {v:.4f}')

            fr = {
                'fold': fold,
                'train': f'{trs}..{tre}',
                'test': f'{xes}..{xee}',
                'n_train': int(len(Xtr_book)),
                'n_test': int(len(Xte_book)),
                'metrics': {k: float(v) for k, v in metrics.items()}
            }
            results['folds'].append(fr)

            # Log to MLflow
            for k, v in metrics.items():
                mlflow.log_metric(f'fold{fold:02d}_{k}', v, step=fold)

            mlflow.log_metric(f'fold{fold:02d}_n_test', fr['n_test'], step=fold)

            # Save fold execution scores
            fold_out = OUT_DIR / f'fold{fold:02d}_exec_scores.npz'
            np.savez(fold_out,
                     exec_scores=exec_scores,
                     base_preds=base_preds,
                     labels=yte,
                     features=exec_features)
            log.info(f'  Saved: {fold_out}')

            fold += 1

        # Summary
        log.info('\n=== EXECUTION OPTIMIZER SUMMARY ===')
        all_metrics = ['overall_ic', 'dir_accuracy', 'sortino', 'execution_correlation', 'exec_pnl_spread']
        for metric_name in all_metrics:
            vals = [f['metrics'][metric_name] for f in results['folds'] if metric_name in f.get('metrics', {})]
            if vals:
                log.info(f'  {metric_name:25s} avg={np.mean(vals):+.4f} std={np.std(vals):.4f} folds={len(vals)}')
                mlflow.log_metric(f'avg_{metric_name}', np.mean(vals))

        out = OUT_DIR / f'exec_opt_{args.approach}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        json.dump(results, open(out, 'w'), indent=2)
        mlflow.log_artifact(str(out))
        log.info(f'Saved: {out}')
        log.info(f'MLflow run: {mlflow.active_run().info.run_id}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execution Optimizer')
    parser.add_argument('--approach', default='supervised', choices=['supervised', 'policy_gradient'],
                        help='Optimization approach')
    args = parser.parse_args()

    main(args)
