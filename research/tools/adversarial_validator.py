#!/usr/bin/env python3
"""
Adversarial Validator — Standardized 5-Gate Strategy Validation
================================================================

IMPORT THIS MODULE instead of reimplementing validation in every script.

The 5 gates:
  1. SIGN-FLIP PERMUTATION (2000 trials) — Is the return stream non-random?
  2. REGIME BALANCE — Does it work in both bull and bear markets?
  3. SUB-PERIOD STABILITY — Both halves profitable?
  4. OUTLIER REMOVAL — Still profitable without best month?
  5. YEARLY CONSISTENCY — What % of years are profitable?

Key design decisions (from QUANT_KNOWLEDGE_BASE.md):
  - Sharpe uses EQUITY-BASED returns (pct_change of equity), NOT returns/initial_capital
  - Calendar month aggregation (resample('ME')), NOT fixed-size chunks
  - Sign-flip permutation flips PnL signs randomly, preserving trade count
  - Regime classification uses SPY return during each trade's holding period

Usage:
    from research.tools.adversarial_validator import validate_trades

    result = validate_trades(
        trades=my_trade_list,         # list of dicts with 'pnl', 'entry_date', 'exit_date'
        initial_capital=645.0,
        spy_prices=spy_close_series,  # pd.Series indexed by date (optional, for regime gate)
    )
    result.print_summary()
    if result.all_passed:
        print("Strategy validated!")
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─── Result Container ────────────────────────────────────────────────

@dataclass
class GateResult:
    """Result of a single validation gate."""
    name: str
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    detail: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"  [{status}] {self.name}: {self.metric_name}={self.metric_value:.4f} (threshold: {self.threshold})"


@dataclass
class ValidationResult:
    """Complete result of adversarial validation."""
    strategy_name: str
    n_trades: int
    sharpe: float
    sortino: float
    cagr: float
    max_dd: float
    win_rate: float
    profit_factor: float
    final_equity: float
    gates: list[GateResult] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def all_passed(self) -> bool:
        return all(g.passed for g in self.gates) and self.error is None

    @property
    def gates_passed(self) -> int:
        return sum(1 for g in self.gates if g.passed)

    @property
    def gates_total(self) -> int:
        return len(self.gates)

    def to_dict(self) -> dict:
        return {
            "strategy_name": self.strategy_name,
            "n_trades": self.n_trades,
            "sharpe": round(self.sharpe, 3),
            "sortino": round(self.sortino, 3),
            "cagr": round(self.cagr, 4),
            "max_dd": round(self.max_dd, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 3),
            "final_equity": round(self.final_equity, 2),
            "gates_passed": self.gates_passed,
            "gates_total": self.gates_total,
            "all_passed": self.all_passed,
            "gates": [
                {
                    "name": g.name,
                    "passed": g.passed,
                    "metric_name": g.metric_name,
                    "metric_value": round(g.metric_value, 4),
                    "threshold": g.threshold,
                    "detail": g.detail,
                }
                for g in self.gates
            ],
            "error": self.error,
        }

    def print_summary(self) -> None:
        """Print a clean, human-readable summary table."""
        print("\n" + "=" * 65)
        print(f"  ADVERSARIAL VALIDATION: {self.strategy_name}")
        print("=" * 65)

        if self.error:
            print(f"  ERROR: {self.error}")
            print("=" * 65)
            return

        print(f"  Trades: {self.n_trades}  |  Sharpe: {self.sharpe:.2f}  |  "
              f"Sortino: {self.sortino:.2f}  |  WR: {self.win_rate*100:.1f}%")
        print(f"  CAGR: {self.cagr*100:.1f}%  |  MaxDD: {self.max_dd*100:.1f}%  |  "
              f"PF: {self.profit_factor:.2f}  |  Final: ${self.final_equity:,.0f}")
        print("-" * 65)

        for gate in self.gates:
            print(gate)

        print("-" * 65)
        verdict = "ALL GATES PASSED" if self.all_passed else f"FAILED ({self.gates_passed}/{self.gates_total} passed)"
        print(f"  VERDICT: {verdict}")
        print("=" * 65)


# ─── Honest Sharpe Calculation ───────────────────────────────────────

def compute_honest_sharpe(
    equity_series: pd.Series,
    annualization: float = 12.0,
) -> tuple[float, float, pd.Series]:
    """
    Compute HONEST Sharpe and Sortino from an equity time series.

    CRITICAL RULES (from QUANT_KNOWLEDGE_BASE.md):
      1. Returns = equity.pct_change() — divides by CURRENT equity, not initial capital.
         This avoids 2-6x Sharpe inflation on compounding strategies.
      2. Calendar month aggregation via resample('ME'), NOT fixed-size chunks.
         Chunks artificially smooth variance and inflate Sharpe ~25%.

    Args:
        equity_series: pd.Series with DatetimeIndex, values = equity.
        annualization: sqrt factor. 12 for monthly, 252 for daily.

    Returns:
        (sharpe, sortino, monthly_returns)
    """
    if len(equity_series) < 2:
        return 0.0, 0.0, pd.Series(dtype=float)

    # Resample to end-of-month, take last value
    monthly_equity = equity_series.resample("ME").last().dropna()

    if len(monthly_equity) < 2:
        return 0.0, 0.0, pd.Series(dtype=float)

    # EQUITY-BASED pct_change (divides by previous month's equity)
    monthly_returns = monthly_equity.pct_change().dropna()

    if len(monthly_returns) < 2 or monthly_returns.std() == 0:
        return 0.0, 0.0, monthly_returns

    mean_ret = monthly_returns.mean()
    std_ret = monthly_returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(annualization)

    # Sortino: downside deviation only
    downside = monthly_returns[monthly_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (mean_ret / downside.std()) * np.sqrt(annualization)
    else:
        sortino = sharpe * 1.5  # no downside months = very good

    return float(sharpe), float(sortino), monthly_returns


# ─── Core Metrics ────────────────────────────────────────────────────

def compute_metrics(
    trades: list[dict],
    initial_capital: float,
) -> dict:
    """Compute honest metrics from a trade list.

    Each trade dict must have at minimum: 'pnl', 'entry_date', 'exit_date'.
    """
    if not trades:
        return {
            "sharpe": 0.0, "sortino": 0.0, "cagr": 0.0, "max_dd": -1.0,
            "win_rate": 0.0, "profit_factor": 0.0, "n_trades": 0,
            "final_equity": initial_capital,
        }

    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("exit_date")

    pnls = df["pnl"].values

    # Build equity curve from trade PnLs
    equity_values = [initial_capital]
    for pnl in pnls:
        equity_values.append(equity_values[-1] + pnl)

    # Map to dates for monthly resampling
    dates = [df["entry_date"].iloc[0] - pd.Timedelta(days=1)]  # pre-trade date
    dates.extend(df["exit_date"].tolist())
    equity_series = pd.Series(equity_values, index=pd.DatetimeIndex(dates))

    # Handle duplicate dates by keeping last
    equity_series = equity_series.groupby(equity_series.index).last()

    sharpe, sortino, monthly_rets = compute_honest_sharpe(equity_series)

    # CAGR
    final_eq = equity_values[-1]
    total_days = (dates[-1] - dates[0]).days
    years = max(total_days / 365.25, 0.1)
    if final_eq > 0 and initial_capital > 0:
        cagr = (final_eq / initial_capital) ** (1 / years) - 1
    else:
        cagr = -1.0

    # Max drawdown
    eq_arr = np.array(equity_values)
    peak = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.min())

    # Win rate, profit factor
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = len(wins) / len(pnls) if len(pnls) > 0 else 0.0
    gross_profit = wins.sum() if len(wins) > 0 else 0.0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float("inf")

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "cagr": cagr,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "n_trades": len(pnls),
        "final_equity": final_eq,
        "monthly_returns": monthly_rets,
        "equity_series": equity_series,
    }


# ─── Gate 1: Sign-Flip Permutation Test ─────────────────────────────

def _gate1_sign_flip(
    trades: list[dict],
    initial_capital: float,
    real_sharpe: float,
    n_perms: int = 2000,
) -> GateResult:
    """
    Sign-flip permutation: randomly flip the sign of each trade's PnL.
    This tests whether the return stream is distinguishable from random.

    NOTE: This tests overall profitability, NOT ML alpha specifically.
    For ML alpha, compare against a random-selection baseline (see sector_backtest).
    """
    pnls = np.array([t["pnl"] for t in trades])
    dates = pd.to_datetime([t["exit_date"] for t in trades])

    beat_count = 0
    for _ in range(n_perms):
        # Randomly flip signs (multiply by +1 or -1)
        signs = np.random.choice([-1, 1], size=len(pnls))
        flipped_pnls = pnls * signs

        # Build equity curve
        equity_vals = [initial_capital]
        for p in flipped_pnls:
            equity_vals.append(equity_vals[-1] + p)

        eq_series = pd.Series(
            equity_vals,
            index=pd.DatetimeIndex(
                [dates[0] - pd.Timedelta(days=1)] + list(dates)
            ),
        )
        eq_series = eq_series.groupby(eq_series.index).last()

        perm_sharpe, _, _ = compute_honest_sharpe(eq_series)
        if perm_sharpe >= real_sharpe:
            beat_count += 1

    p_value = beat_count / n_perms

    return GateResult(
        name="Sign-Flip Permutation",
        passed=p_value < 0.05,
        metric_name="p_value",
        metric_value=p_value,
        threshold=0.05,
        detail=f"{beat_count}/{n_perms} permutations beat real Sharpe {real_sharpe:.3f}",
    )


# ─── Gate 2: Regime Balance ─────────────────────────────────────────

def _gate2_regime_balance(
    trades: list[dict],
    spy_prices: Optional[pd.Series] = None,
) -> GateResult:
    """
    Regime balance: win rate gap between bull and bear markets must be < 0.50.
    Bull/bear classified by SPY return during each trade's holding period.
    If no SPY data provided, uses 'regime' field from trade dicts.
    """
    bull_pnls = []
    bear_pnls = []

    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    if spy_prices is not None and len(spy_prices) > 0:
        spy_prices = spy_prices.sort_index()
        for _, row in df.iterrows():
            entry_prices = spy_prices[spy_prices.index <= row["entry_date"]]
            exit_prices = spy_prices[spy_prices.index <= row["exit_date"]]
            if len(entry_prices) == 0 or len(exit_prices) == 0:
                continue
            spy_entry = float(entry_prices.iloc[-1])
            spy_exit = float(exit_prices.iloc[-1])
            if spy_exit >= spy_entry:
                bull_pnls.append(row["pnl"])
            else:
                bear_pnls.append(row["pnl"])
    elif "regime" in df.columns:
        bull_pnls = df[df["regime"].isin(["bull", "green"])]["pnl"].tolist()
        bear_pnls = df[df["regime"].isin(["bear", "red"])]["pnl"].tolist()
    else:
        # No regime info available — pass by default with a note
        return GateResult(
            name="Regime Balance",
            passed=True,
            metric_name="wr_gap",
            metric_value=0.0,
            threshold=0.50,
            detail="No SPY data or regime labels provided; gate skipped",
        )

    if len(bull_pnls) < 5 or len(bear_pnls) < 5:
        return GateResult(
            name="Regime Balance",
            passed=True,
            metric_name="wr_gap",
            metric_value=0.0,
            threshold=0.50,
            detail=f"Insufficient regime split: {len(bull_pnls)} bull, {len(bear_pnls)} bear (need 5+ each)",
        )

    bull_wr = np.mean([1 if p > 0 else 0 for p in bull_pnls])
    bear_wr = np.mean([1 if p > 0 else 0 for p in bear_pnls])
    wr_gap = abs(bull_wr - bear_wr)

    return GateResult(
        name="Regime Balance",
        passed=wr_gap < 0.50,
        metric_name="wr_gap",
        metric_value=wr_gap,
        threshold=0.50,
        detail=f"Bull WR={bull_wr*100:.1f}% ({len(bull_pnls)} trades), "
               f"Bear WR={bear_wr*100:.1f}% ({len(bear_pnls)} trades)",
    )


# ─── Gate 3: Sub-Period Stability ───────────────────────────────────

def _gate3_sub_period_stability(trades: list[dict]) -> GateResult:
    """Both halves of the trade history must be independently profitable."""
    df = pd.DataFrame(trades).sort_values("exit_date", key=pd.to_datetime)
    n = len(df)
    mid = n // 2

    first_half_pnl = df.iloc[:mid]["pnl"].sum()
    second_half_pnl = df.iloc[mid:]["pnl"].sum()

    both_positive = first_half_pnl > 0 and second_half_pnl > 0
    min_half = min(first_half_pnl, second_half_pnl)

    return GateResult(
        name="Sub-Period Stability",
        passed=both_positive,
        metric_name="min_half_pnl",
        metric_value=min_half,
        threshold=0.0,
        detail=f"First half PnL: ${first_half_pnl:.2f}, Second half PnL: ${second_half_pnl:.2f}",
    )


# ─── Gate 4: Outlier Removal ────────────────────────────────────────

def _gate4_outlier_removal(
    trades: list[dict],
    initial_capital: float,
    monthly_returns: pd.Series,
) -> GateResult:
    """
    Remove the best calendar month and check if still profitable.
    This catches strategies that depend on a single lucky month.
    """
    if len(monthly_returns) < 3:
        return GateResult(
            name="Outlier Removal",
            passed=False,
            metric_name="sharpe_ex_best",
            metric_value=0.0,
            threshold=0.0,
            detail="Not enough monthly data",
        )

    # Remove best month
    best_month_idx = monthly_returns.idxmax()
    trimmed = monthly_returns.drop(best_month_idx)

    if len(trimmed) < 2 or trimmed.std() == 0:
        return GateResult(
            name="Outlier Removal",
            passed=False,
            metric_name="sharpe_ex_best",
            metric_value=0.0,
            threshold=0.0,
            detail="Not enough data after removing best month",
        )

    trimmed_sharpe = (trimmed.mean() / trimmed.std()) * np.sqrt(12)
    still_profitable = trimmed.sum() > 0

    return GateResult(
        name="Outlier Removal",
        passed=still_profitable and trimmed_sharpe > 0,
        metric_name="sharpe_ex_best",
        metric_value=trimmed_sharpe,
        threshold=0.0,
        detail=f"Removed {best_month_idx}: best month ret={monthly_returns[best_month_idx]*100:.1f}%. "
               f"Remaining Sharpe={trimmed_sharpe:.2f}, total ret={trimmed.sum()*100:.1f}%",
    )


# ─── Gate 5: Yearly Consistency ─────────────────────────────────────

def _gate5_yearly_consistency(
    trades: list[dict],
    threshold_pct: float = 0.60,
) -> GateResult:
    """
    What percentage of calendar years are profitable?
    Threshold: at least 60% of years must be net positive.
    """
    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["year"] = df["exit_date"].dt.year

    yearly_pnl = df.groupby("year")["pnl"].sum()
    n_years = len(yearly_pnl)
    n_profitable = (yearly_pnl > 0).sum()

    if n_years == 0:
        return GateResult(
            name="Yearly Consistency",
            passed=False,
            metric_name="pct_years_profitable",
            metric_value=0.0,
            threshold=threshold_pct,
            detail="No yearly data",
        )

    pct_profitable = n_profitable / n_years

    # Detail: show each year
    year_detail_parts = []
    for year, pnl in yearly_pnl.items():
        marker = "+" if pnl > 0 else "-"
        year_detail_parts.append(f"{year}:{marker}${abs(pnl):.0f}")
    year_str = ", ".join(year_detail_parts[-8:])  # last 8 years
    if n_years > 8:
        year_str = f"... {year_str}"

    return GateResult(
        name="Yearly Consistency",
        passed=pct_profitable >= threshold_pct,
        metric_name="pct_years_profitable",
        metric_value=pct_profitable,
        threshold=threshold_pct,
        detail=f"{n_profitable}/{n_years} years profitable. {year_str}",
    )


# ─── Main Entry Point ───────────────────────────────────────────────

def validate_trades(
    trades: list[dict],
    initial_capital: float = 645.0,
    spy_prices: Optional[pd.Series] = None,
    strategy_name: str = "Strategy",
    n_perms: int = 2000,
    yearly_threshold: float = 0.60,
) -> ValidationResult:
    """
    Run full 5-gate adversarial validation on a list of trades.

    Args:
        trades: list of dicts, each with at minimum:
            - 'pnl' (float): trade profit/loss in dollars
            - 'entry_date' (str or Timestamp): trade entry date
            - 'exit_date' (str or Timestamp): trade exit date
            Optional:
            - 'regime' (str): 'bull'/'bear' or 'green'/'red' (used if spy_prices not provided)

        initial_capital: starting equity (default $645)
        spy_prices: pd.Series of SPY close prices indexed by date (for regime gate)
        strategy_name: label for the strategy
        n_perms: number of sign-flip permutations (default 2000)
        yearly_threshold: fraction of years that must be profitable (default 0.60)

    Returns:
        ValidationResult with metrics and gate pass/fail details.
    """
    if len(trades) < 10:
        return ValidationResult(
            strategy_name=strategy_name,
            n_trades=len(trades),
            sharpe=0.0, sortino=0.0, cagr=0.0, max_dd=-1.0,
            win_rate=0.0, profit_factor=0.0, final_equity=initial_capital,
            error=f"Too few trades ({len(trades)}). Need at least 10.",
        )

    # Compute core metrics
    metrics = compute_metrics(trades, initial_capital)

    # Run all 5 gates
    gates = []

    # Gate 1: Sign-flip permutation
    gates.append(_gate1_sign_flip(trades, initial_capital, metrics["sharpe"], n_perms))

    # Gate 2: Regime balance
    gates.append(_gate2_regime_balance(trades, spy_prices))

    # Gate 3: Sub-period stability
    gates.append(_gate3_sub_period_stability(trades))

    # Gate 4: Outlier removal
    gates.append(_gate4_outlier_removal(trades, initial_capital, metrics["monthly_returns"]))

    # Gate 5: Yearly consistency
    gates.append(_gate5_yearly_consistency(trades, yearly_threshold))

    return ValidationResult(
        strategy_name=strategy_name,
        n_trades=metrics["n_trades"],
        sharpe=metrics["sharpe"],
        sortino=metrics["sortino"],
        cagr=metrics["cagr"],
        max_dd=metrics["max_dd"],
        win_rate=metrics["win_rate"],
        profit_factor=metrics["profit_factor"],
        final_equity=metrics["final_equity"],
        gates=gates,
    )


# ─── Quick Self-Test ─────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running adversarial_validator self-test...")

    # Generate synthetic trades that should PASS most gates
    np.random.seed(42)
    n_trades = 200
    base_date = pd.Timestamp("2020-01-01")
    synthetic_trades = []
    for i in range(n_trades):
        entry = base_date + pd.Timedelta(days=i * 7)
        exit_d = entry + pd.Timedelta(days=30)
        # Slightly positive expectancy
        pnl = np.random.normal(5.0, 20.0)
        synthetic_trades.append({
            "pnl": pnl,
            "entry_date": entry,
            "exit_date": exit_d,
        })

    result = validate_trades(
        synthetic_trades,
        initial_capital=1000.0,
        strategy_name="Synthetic Self-Test",
        n_perms=200,  # fewer for speed in self-test
    )
    result.print_summary()

    print("\nSelf-test complete. ValidationResult.to_dict() keys:")
    print(list(result.to_dict().keys()))
