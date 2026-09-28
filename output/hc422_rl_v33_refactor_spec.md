# HC #422 Rule 8 — Refactor RL_v3_3_smart_exec to consume ALL heads

Found candidates:
  - output/rl_v3_3_smart_exec_v3
  - output/rl_v3_3_smart_exec
  - scripts/rl_v3_3_smart_exec

Required input features (per Rule 8):
  - pred_log_ret_1s,5s,10s,30s
  - MFE/MAE
  - Vol, spread forecast, confidence
  - book imbalance, depth, microprice, queue position