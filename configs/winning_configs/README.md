# Winning Configs Registry

Per user directive (2026-04-27): "Ensure we saving keys!" — every execution config that
qualifies as a candidate for production gets persisted here with full metadata.

## Format
Each config is a JSON file: `{model}_{strategy}_{date_range}.json`

## Required fields
- `model`: e.g. "cnn_mamba_v2"
- `strategy_name`: e.g. "tp8_z2.3_chase2_5repr_sl15_volexit5_5b_60shold_prime"
- `validation_dates`: list of dates tested
- `n_dates`: count
- `n_trades`: total
- `n_signals`, `n_filled`: order book stats
- `total_pnl_dollars`
- `win_rate`, `profit_factor`, `pnl_per_trade`, `sharpe_per_trade`
- `n_pos_days`, `n_neg_days`
- `daily_pnl`: list of per-day PnL
- `daily_sharpe`: PnL_mean / PnL_std (daily)
- `cli_args`: full fill_sim_cli arg list to reproduce
- `pred_cache_dir`: path
- `notes`: caveats, regime context, news events during window
- `qualifies_for_paper_trading`: bool with reasoning

## Status meanings
- `candidate`: passed initial validation, more testing needed
- `paper_trading`: actively running in paper mode
- `production`: live trading
- `retired`: lost in subsequent validation
