#!/usr/bin/env python3
"""
Adversarial Realism Check for Weekly Theta Harvest
===================================================
The base backtest showed Sharpe 2.18-2.69 with 85-88% WR.
Before declaring victory, test under harsher assumptions:

1. Option premium haircut: BS*0.7 (bid-ask spread eats 30% of theoretical premium)
2. Wider slippage: $0.02/share slippage on entry/exit
3. Stricter margin: 30% of notional (not 20%)
4. Max 2 concurrent positions (not 3)
5. Assignment cost: $50 per assignment event (realistic for small accounts)
6. Exclude 2022 (pure bear market — check if strategy is just selling puts into a crash)

This is the "does it still work when we stop being nice?" test.
"""

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Import from main backtest
sys.path.insert(0, str(Path(__file__).parent))
from vol_options_backtest import (
    bs_price, compute_historical_iv, compute_iv_rank,
    download_ohlcv, compute_metrics, permutation_test,
    regime_analysis, validate_5gate, random_entry_benchmark,
    OptionTrade, COMMISSION_PER_LEG, WEEKLY_UNIVERSE,
    OOT_START, OOT_END, DATA_START, DATA_END,
)

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/vol_options")
CACHE_DIR = OUTPUT_DIR / "cache"


@dataclass
class AdversarialConfig:
    name: str = "adv_base"
    option_type: str = "put"
    otm_offset_pct: float = 0.05
    target_dte: int = 5
    iv_rank_threshold: float = 0.50
    min_premium: float = 0.05
    max_trade_loss: float = 200.0
    max_concurrent: int = 2           # Stricter: 2 not 3
    r: float = 0.05
    directional_filter: bool = False
    # Adversarial params
    premium_haircut: float = 0.70     # Only collect 70% of BS premium (bid-ask)
    slippage_per_share: float = 0.02  # $0.02 slippage
    margin_pct: float = 0.30          # 30% margin requirement
    assignment_cost: float = 50.0     # $50 assignment fee
    starting_capital: float = 645.0
    exclude_2022: bool = False        # Test without 2022 bear


def run_adversarial_theta(ohlcv_data: dict, config: AdversarialConfig) -> list:
    """Weekly theta with adversarial assumptions."""
    all_trades = []
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)

    if config.exclude_2022:
        oot_start = pd.Timestamp("2023-01-01")

    for ticker, ohlcv in ohlcv_data.items():
        if ticker not in WEEKLY_UNIVERSE:
            continue

        close = ohlcv["Close"]
        iv_series = compute_historical_iv(close, window=20, markup=1.3)
        iv_rank = compute_iv_rank(iv_series, lookback=252)
        sma20 = close.rolling(20).mean()

        for i in range(252, len(ohlcv)):
            date = ohlcv.index[i]
            if date < oot_start or date > oot_end:
                continue
            if date.dayofweek > 1:
                continue

            S = close.iloc[i]
            if S < 3.0 or S > 20.0:
                continue

            current_iv = iv_series.iloc[i]
            current_iv_rank = iv_rank.iloc[i]
            if pd.isna(current_iv_rank) or current_iv_rank < config.iv_rank_threshold:
                continue

            if config.directional_filter:
                above_sma = S > sma20.iloc[i] if not pd.isna(sma20.iloc[i]) else True
                opt_type = "put" if above_sma else "call"
            else:
                opt_type = config.option_type

            if opt_type == "put":
                K = round(S * (1 - config.otm_offset_pct), 2)
                K = max(K, 0.50)
            else:
                K = round(S * (1 + config.otm_offset_pct), 2)

            T = config.target_dte / 252.0
            theoretical_premium = bs_price(S, K, T, config.r, current_iv, opt_type)

            # ADVERSARIAL: haircut premium for bid-ask
            premium = theoretical_premium * config.premium_haircut

            if premium < config.min_premium:
                continue

            # ADVERSARIAL: slippage
            premium -= config.slippage_per_share

            if premium <= 0:
                continue

            n_contracts = 1
            commission = n_contracts * 2 * COMMISSION_PER_LEG

            # Simulate to expiry
            exit_idx = min(i + config.target_dte, len(ohlcv) - 1)
            max_loss_hit = False
            actual_exit_idx = exit_idx

            for day_i in range(i + 1, exit_idx + 1):
                if day_i >= len(ohlcv):
                    break
                day_price = close.iloc[day_i]
                days_left = max(1, exit_idx - day_i)
                T_rem = days_left / 252.0

                if opt_type == "put" and day_price < K * 0.93:
                    day_opt_price = bs_price(day_price, K, T_rem, config.r, current_iv * 0.9, opt_type)
                    unrealized = (day_opt_price - premium) * 100 * n_contracts
                    if unrealized > config.max_trade_loss:
                        max_loss_hit = True
                        actual_exit_idx = day_i
                        break
                elif opt_type == "call" and day_price > K * 1.07:
                    day_opt_price = bs_price(day_price, K, T_rem, config.r, current_iv * 0.9, opt_type)
                    unrealized = (day_opt_price - premium) * 100 * n_contracts
                    if unrealized > config.max_trade_loss:
                        max_loss_hit = True
                        actual_exit_idx = day_i
                        break

            exit_date = ohlcv.index[min(actual_exit_idx, len(ohlcv) - 1)]
            exit_price = close.iloc[min(actual_exit_idx, len(ohlcv) - 1)]

            if max_loss_hit:
                T_exit = max(1, exit_idx - actual_exit_idx) / 252.0
                # ADVERSARIAL: buy back at theoretical (no haircut on closing)
                exit_opt = bs_price(exit_price, K, T_exit, config.r, current_iv * 0.95, opt_type)
                exit_opt += config.slippage_per_share  # Slippage on close too
            else:
                if opt_type == "put":
                    exit_opt = max(0, K - exit_price)
                else:
                    exit_opt = max(0, exit_price - K)

            pnl = (premium - exit_opt) * 100 * n_contracts - commission

            assigned = False
            if not max_loss_hit:
                if opt_type == "put" and exit_price < K:
                    assigned = True
                    assignment_loss = (K - exit_price) * 100 * n_contracts
                    pnl = premium * 100 * n_contracts - assignment_loss - commission
                    pnl -= config.assignment_cost  # ADVERSARIAL: assignment cost
                elif opt_type == "call" and exit_price > K:
                    assigned = True
                    assignment_loss = (exit_price - K) * 100 * n_contracts
                    pnl = premium * 100 * n_contracts - assignment_loss - commission
                    pnl -= config.assignment_cost

            if pnl < -config.max_trade_loss:
                pnl = -config.max_trade_loss

            exit_reason = "max_loss" if max_loss_hit else ("assigned" if assigned else "expired_otm")

            trade = OptionTrade(
                ticker=ticker,
                entry_date=date,
                exit_date=exit_date,
                structure=f"{opt_type}_sell",
                entry_stock_price=S,
                put_strike=K if opt_type == "put" else 0,
                call_strike=K if opt_type == "call" else 0,
                entry_iv=current_iv,
                exit_iv=current_iv * 0.9,
                entry_premium=premium * 100 * n_contracts,
                exit_cost=exit_opt * 100 * n_contracts,
                dte_at_entry=config.target_dte,
                pnl=pnl,
                commission=commission,
                max_loss_hit=max_loss_hit,
                exit_reason=exit_reason,
                contracts=n_contracts,
            )
            all_trades.append(trade)

    return all_trades


def simulate_portfolio_strict(trades: list, starting_capital: float = 645.0,
                               max_concurrent: int = 2) -> dict:
    """Portfolio sim with stricter capital management."""
    if not trades:
        return {"equity_curve": [], "trades_taken": 0, "trades_skipped": 0}

    trades_sorted = sorted(trades, key=lambda t: t.entry_date)
    capital = starting_capital
    equity_curve = [(trades_sorted[0].entry_date, capital)]
    active_trades = []
    executed = []
    skipped = 0

    for trade in trades_sorted:
        new_active = []
        for at in active_trades:
            if at.exit_date <= trade.entry_date:
                capital += at.pnl
                equity_curve.append((at.exit_date, capital))
            else:
                new_active.append(at)
        active_trades = new_active

        if len(active_trades) >= max_concurrent:
            skipped += 1
            continue

        # Stricter margin check: need 30% of notional
        notional = trade.entry_stock_price * 100
        margin_needed = notional * 0.30
        if capital < margin_needed:
            skipped += 1
            continue

        # Don't trade if capital is dangerously low
        if capital < 150:
            skipped += 1
            continue

        active_trades.append(trade)
        executed.append(trade)

    for at in active_trades:
        capital += at.pnl
        equity_curve.append((at.exit_date if at.exit_date else at.entry_date + timedelta(days=7), capital))

    equity_curve.sort(key=lambda x: x[0])

    return {
        "equity_curve": equity_curve,
        "final_capital": capital,
        "trades_taken": len(executed),
        "trades_skipped": skipped,
        "executed_trades": executed,
    }


def main():
    log.info("=" * 70)
    log.info("ADVERSARIAL REALISM CHECK — WEEKLY THETA HARVEST")
    log.info("=" * 70)

    all_tickers = list(set(WEEKLY_UNIVERSE + ["SPY"]))
    ohlcv_data = download_ohlcv(all_tickers, DATA_START, DATA_END)
    log.info(f"Got data for {len(ohlcv_data)} tickers")

    spy_data = ohlcv_data["SPY"]

    configs = [
        # Best variants from base test, now with adversarial assumptions
        AdversarialConfig(name="adv_put_8pct", option_type="put", otm_offset_pct=0.08,
                          iv_rank_threshold=0.50, premium_haircut=0.70),
        AdversarialConfig(name="adv_call_5pct", option_type="call", otm_offset_pct=0.05,
                          iv_rank_threshold=0.50, premium_haircut=0.70),
        AdversarialConfig(name="adv_directional", directional_filter=True, otm_offset_pct=0.05,
                          iv_rank_threshold=0.50, premium_haircut=0.70),
        AdversarialConfig(name="adv_put_8pct_harsh", option_type="put", otm_offset_pct=0.08,
                          iv_rank_threshold=0.50, premium_haircut=0.60, slippage_per_share=0.03),
        # Exclude 2022 bear to check regime dependence
        AdversarialConfig(name="adv_put_8pct_no2022", option_type="put", otm_offset_pct=0.08,
                          iv_rank_threshold=0.50, premium_haircut=0.70, exclude_2022=True),
        AdversarialConfig(name="adv_call_5pct_no2022", option_type="call", otm_offset_pct=0.05,
                          iv_rank_threshold=0.50, premium_haircut=0.70, exclude_2022=True),
    ]

    mlflow.set_tracking_uri("sqlite:////home/jupiter/teleclaude-main/mlflow.db")
    mlflow.set_experiment("vol_options_adversarial")

    results = {}

    for cfg in configs:
        log.info(f"\n--- {cfg.name} (haircut={cfg.premium_haircut}, slip={cfg.slippage_per_share}) ---")

        with mlflow.start_run(run_name=cfg.name):
            mlflow.log_params({
                "variant": cfg.name,
                "option_type": cfg.option_type,
                "otm_offset": cfg.otm_offset_pct,
                "premium_haircut": cfg.premium_haircut,
                "slippage": cfg.slippage_per_share,
                "margin_pct": cfg.margin_pct,
                "assignment_cost": cfg.assignment_cost,
                "exclude_2022": cfg.exclude_2022,
                "max_concurrent": cfg.max_concurrent,
            })

            trades = run_adversarial_theta(ohlcv_data, cfg)
            log.info(f"  Raw trades: {len(trades)}")

            portfolio = simulate_portfolio_strict(trades, cfg.starting_capital, cfg.max_concurrent)
            executed = portfolio.get("executed_trades", [])

            if not executed:
                log.warning(f"  No trades for {cfg.name}")
                mlflow.log_metric("n_trades", 0)
                continue

            metrics = compute_metrics(portfolio["equity_curve"], executed, cfg.starting_capital)
            perm_p = permutation_test(executed, n_perms=3000)
            regime = regime_analysis(executed, spy_data)
            random_sharpe = random_entry_benchmark(ohlcv_data, n_sims=100, hold_days=5)
            gates = validate_5gate(metrics, perm_p, regime, random_sharpe)

            # Log metrics
            mlflow.log_metrics({
                "sharpe": metrics["sharpe"],
                "sortino": metrics["sortino"],
                "profit_factor": metrics["profit_factor"],
                "win_rate": metrics["win_rate"],
                "max_dd": metrics["max_dd"],
                "total_return": metrics["total_return"],
                "n_trades": metrics["n_trades"],
                "final_capital": metrics["final_capital"],
                "perm_p_value": perm_p,
                "regime_gap": regime["regime_gap"],
                "gates_passed": gates["passed"],
            })

            # Trade breakdown
            assigned_trades = [t for t in executed if t.exit_reason == "assigned"]
            expired_otm = [t for t in executed if t.exit_reason == "expired_otm"]
            stopped = [t for t in executed if t.exit_reason == "max_loss"]

            results[cfg.name] = {
                "metrics": metrics,
                "perm_p": perm_p,
                "regime": regime,
                "gates": gates,
            }

            log.info(f"  Trades: {metrics['n_trades']} (OTM: {len(expired_otm)}, Assigned: {len(assigned_trades)}, Stopped: {len(stopped)})")
            log.info(f"  WR: {metrics['win_rate']:.1%} | Avg PnL: ${metrics['avg_pnl']:.2f}")
            log.info(f"  Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f}")
            log.info(f"  PF: {metrics['profit_factor']:.2f} | MDD: {metrics['max_dd']:.1%}")
            log.info(f"  ${cfg.starting_capital:.0f} → ${metrics['final_capital']:.0f} ({metrics['total_return']:.1%})")
            log.info(f"  Perm p: {perm_p:.4f} | Regime gap: {regime['regime_gap']:.2f}")
            log.info(f"  Bull Sharpe: {regime['bull_sharpe']:.2f} ({regime['bull_trades']} trades) | Bear Sharpe: {regime['bear_sharpe']:.2f} ({regime['bear_trades']} trades)")
            log.info(f"  Gates: {gates['passed']}/{gates['total']} {'PASS' if gates['all_passed'] else 'FAIL'}")

            if metrics.get("yearly"):
                for yr, yd in sorted(metrics["yearly"].items()):
                    log.info(f"    {yr}: Sharpe={yd['sharpe']:.2f}, Return={yd['return']:.1%}")

    # Summary
    log.info("\n" + "=" * 70)
    log.info("ADVERSARIAL SUMMARY")
    log.info("=" * 70)

    summary_rows = []
    for name, res in sorted(results.items()):
        m = res["metrics"]
        g = res["gates"]
        summary_rows.append({
            "Variant": name,
            "Trades": m["n_trades"],
            "WR": f"{m['win_rate']:.1%}",
            "Sharpe": f"{m['sharpe']:.2f}",
            "Sortino": f"{m['sortino']:.2f}",
            "PF": f"{m['profit_factor']:.2f}",
            "MDD": f"{m['max_dd']:.1%}",
            "Final$": f"${m['final_capital']:.0f}",
            "RegGap": f"{res['regime']['regime_gap']:.2f}",
            "Gates": f"{g['passed']}/{g['total']}",
            "Pass": "YES" if g["all_passed"] else "no",
        })

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print("\n" + df.to_string(index=False))

    # Save
    with open(OUTPUT_DIR / "adversarial_results.json", "w") as f:
        serializable = {}
        for k, v in results.items():
            serializable[k] = {
                "metrics": v["metrics"],
                "perm_p": v["perm_p"],
                "regime": v["regime"],
                "gates": {gk: (bool(gv) if isinstance(gv, (bool, np.bool_)) else gv)
                          for gk, gv in v["gates"].items()},
            }
        json.dump(serializable, f, indent=2, default=str)

    log.info(f"\nSaved to {OUTPUT_DIR / 'adversarial_results.json'}")


if __name__ == "__main__":
    main()
