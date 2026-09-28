# v3.3 META-ENSEMBLE — HC #363 deliverable 4

Source: `fold_00_predictions.npz` (FIFO-fillable 59,024 samples, chronological 60/20/20 split).
Target: passive SHORT FIFO net P&L (target_fifo_tp4sl3_net × -1 − 0.376 commission).
Train block z-score + sign-fit per head (correlation-with-target on train).

## Ridge baseline

| Block | Band | n_picked | Sharpe | Net t/fill |
|---|---|---|---|---|
| Val Top1% | 1% | 118 | 0.079 | 0.271 |
| Val Top5% | 5% | 590 | 0.080 | 0.274 |
| Test Top1% | 1% | 118 | -0.265 | -0.928 |
| Test Top5% | 5% | 590 | -0.075 | -0.261 |
| Test Top10% | 10% | 1180 | -0.021 | -0.078 |

## Meta-MLP (32→32→16→1, MLPRegressor)

| Block | Band | n_picked | Sharpe | Net t/fill |
|---|---|---|---|---|
| Val Top1% | 1% | 118 | 0.059 | 0.208 |
| Val Top5% | 5% | 590 | 0.059 | 0.207 |
| Test Top1% | 1% | 118 | -0.247 | -0.839 |
| Test Top5% | 5% | 590 | -0.113 | -0.392 |
| Test Top10% | 10% | 1180 | -0.057 | -0.195 |

## Solo-best head reference (test block)

- Best solo head on test block (n>=10, top1%): **pred_realized_vol_30s_ticks** — Sharpe 1.013, net 2.229 ticks, n=118

## LinUCB bandit (top-8 arms by |train corr|)

- Arms: p_up_60s, fifo_tp4sl3_hit_tp, pred_mae_60s_ticks, fifo_tp4sl3_net, fifo_tp8sl5_net, fifo_tp8sl5_hit_tp, log_ret_60s_q50, log_ret_30s_q90
- Test events: 11,806  Fired (traded): 2  Fire rate: 0.02%
- **Test Sharpe: 0.000  Net t/fill: -4.000**
- Train arm-pull distribution:
  - p_up_60s: 4037
  - fifo_tp4sl3_hit_tp: 4154
  - pred_mae_60s_ticks: 5097
  - fifo_tp4sl3_net: 3768
  - fifo_tp8sl5_net: 3829
  - fifo_tp8sl5_hit_tp: 4413
  - log_ret_60s_q50: 4970
  - log_ret_30s_q90: 5146

## Per-head sign + train correlation (z-scored signed feature vs SHORT target)

| Head | Sign | |Train corr| |
|---|---|---|
| p_up_60s | − | 0.0323 |
| fifo_tp4sl3_hit_tp | + | 0.0321 |
| pred_mae_60s_ticks | + | 0.0310 |
| fifo_tp4sl3_net | − | 0.0296 |
| fifo_tp8sl5_net | − | 0.0271 |
| fifo_tp8sl5_hit_tp | + | 0.0269 |
| log_ret_60s_q50 | − | 0.0238 |
| log_ret_30s_q90 | + | 0.0236 |
| p_reversal_30s | − | 0.0233 |
| p_reversal_15s | − | 0.0231 |
| log_ret_10s_q90 | + | 0.0228 |
| pred_time_to_mfe_secs | − | 0.0215 |
| pred_mae_30s_ticks | − | 0.0210 |
| pred_mfe_30s_ticks | + | 0.0203 |
| pred_realized_vol_30s_ticks | + | 0.0202 |
| p_up_30s | + | 0.0190 |
| log_ret_60s_q90 | − | 0.0189 |
| log_ret_30s_q10 | − | 0.0181 |
| log_ret_10s_q10 | − | 0.0176 |
| log_ret_60s | − | 0.0172 |
